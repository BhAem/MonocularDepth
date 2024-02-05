import math

import numpy as np

import paddle
paddle.disable_static()
import paddle.nn as nn
import paddle.vision.models as models
from paddle.utils.download import get_weights_path_from_url
from utils import load_weight_file
import paddle.nn.functional as F


class ASPP_module(nn.Layer):
    def __init__(self, inplanes, planes, os):
        super(ASPP_module, self).__init__()
        if os == 16:
            dilations = [1, 2, 4, 6] # 1, 2, 4, 6
        elif os == 8:
            dilations = [1, 12, 24, 36]
        # 空洞率为1、6、12、18,padding为1、6、12、18的空洞卷积
        # 四组aspp卷积块特征图输出大小相等
        self.aspp1 = nn.Sequential(nn.Conv2D(inplanes, planes, kernel_size=1, stride=1,
                                             padding=0, dilation=dilations[0], bias_attr=False),
                                   nn.BatchNorm2D(planes),
                                   nn.ReLU())
        self.aspp2 = nn.Sequential(nn.Conv2D(inplanes, planes, kernel_size=3, stride=1,
                                             padding=dilations[1], dilation=dilations[1], bias_attr=False),
                                   nn.BatchNorm2D(planes),
                                   nn.ReLU())
        self.aspp3 = nn.Sequential(nn.Conv2D(inplanes, planes, kernel_size=3, stride=1,
                                             padding=dilations[2], dilation=dilations[2], bias_attr=False),
                                   nn.BatchNorm2D(planes),
                                   nn.ReLU())
        self.aspp4 = nn.Sequential(nn.Conv2D(inplanes, planes, kernel_size=3, stride=1,
                                             padding=dilations[3], dilation=dilations[3], bias_attr=False),
                                   nn.BatchNorm2D(planes),
                                   nn.ReLU())
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2D((1, 1)),
                                             nn.Conv2D(512, 256, 1, stride=1, bias_attr=False),
                                             nn.BatchNorm2D(256),
                                             nn.ReLU())
        # self._init_weight()

    # 对ASPP模块中的卷积与BatchNorm使用如下方式初始化
    def _init_weight(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2D):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        x1 = self.aspp1(x)  # 1,256,x,x
        x2 = self.aspp2(x)  # 1,256,x,x
        x3 = self.aspp3(x)  # 1,256,x,x
        x4 = self.aspp4(x)  # 1,256,x,x
        x5 = self.global_avg_pool(x)  # 1,256,1,1
        # 双线性插值扩大x5尺寸至x4同样大小, 1,256,x,x
        x5 = F.interpolate(x5, size=x4.shape[2:], mode='bilinear', align_corners=True)
        x = paddle.concat((x1, x2, x3, x4, x5), axis=1)  # 1,1280,x,x
        return x


class ResNetMultiImageInput(models.ResNet):
    """
    Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/vision/models/resnet.py
    """
    def __init__(self, block, depth, num_classes=1000, with_pool=True, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, depth, num_classes, with_pool)
        self.conv1 = nn.Conv2D(num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, weight_attr=nn.initializer.KaimingUniform(), bias_attr=False) # replace the first conv


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """
    Constructs a ResNet model with multiple input images.
    Args:
        num_layers (int): Number of resnet layers. Must be 18, 34 50, 101, 152
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 34, 50, 101, 152], "Can only run with 18, 34 50, 101, 152 layer resnet"
    block_type = models.resnet.BasicBlock if num_layers <= 34 else models.resnet.Bottleneck
    model = ResNetMultiImageInput(block_type, num_layers, num_input_images=num_input_images)

    if pretrained is True:
        loaded = paddle.load(get_weights_path_from_url(*models.resnet.model_urls['resnet{}'.format(num_layers)]))
        loaded['conv1.weight'] = paddle.concat([loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_dict(loaded)
    elif isinstance(pretrained, str):
        loaded = load_weight_file(pretrained)
        loaded['conv1.weight'] = paddle.concat([loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_dict(loaded)    

    return model


class ResnetEncoder(nn.Layer):
    """
    Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, use_aspp, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.use_aspp = use_aspp
        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if pretrained == 'paddle_pretrained':
            if num_input_images > 1:
                self.encoder = resnet_multiimage_input(num_layers, True, num_input_images)
            else:
                self.encoder = resnets[num_layers](True)
        elif pretrained == 'scratch':
            if num_input_images > 1:
                self.encoder = resnet_multiimage_input(num_layers, False, num_input_images)
            else:
                self.encoder = resnets[num_layers](False)
        else:
            if num_input_images > 1:
                self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
            else:
                self.encoder = resnets[num_layers](False)   
                # self.encoder.load_dict(load_weight_file(pretrained))

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        self.aspp = ASPP_module(512, 256, 16)
        self.conv2 = nn.Conv2D(1280, 512, 1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(512)

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225 # normalization?
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))

        if not self.use_aspp:
            self.features.append(self.encoder.layer4(self.features[-1]))
        else:
            x = self.encoder.layer4(self.features[-1])
            x = self.aspp(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.encoder.relu(x)
            self.features.append(x)

        return self.features


if __name__=='__main__':
    model = ResnetEncoder(18, False, True)
    image = paddle.randn(shape=(2, 3, 192, 640))
    output = model(image)
    print(output.size())