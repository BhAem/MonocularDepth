import math

import numpy as np

import paddle
paddle.disable_static()
import paddle.nn as nn
import paddle.vision.models as models
from paddle.utils.download import get_weights_path_from_url
from utils import load_weight_file
import paddle.nn.functional as F
from networks.CBAM import ChannelAttention, SpatialAttention


class ASPP_module(nn.Layer):
    def __init__(self, inplanes, planes, os):
        super(ASPP_module, self).__init__()
        if os == 16:
            dilations = [1, 2, 4, 6] # 1,6,12,18
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
                                             nn.Conv2D(inplanes, planes, 1, stride=1, bias_attr=False),
                                             nn.BatchNorm2D(planes),
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


__all__ = []

model_urls = {
    'resnet18': ('https://paddle-hapi.bj.bcebos.com/models/resnet18.pdparams',
                 'cf548f46534aa3560945be4b95cd11c4'),
    'resnet34': ('https://paddle-hapi.bj.bcebos.com/models/resnet34.pdparams',
                 '8d2275cf8706028345f78ac0e1d31969'),
    'resnet50': ('https://paddle-hapi.bj.bcebos.com/models/resnet50.pdparams',
                 'ca6f485ee1ab0492d38f323885b0ad80'),
    'resnet101': ('https://paddle-hapi.bj.bcebos.com/models/resnet101.pdparams',
                  '02f35f034ca3858e1e54d4036443c92d'),
    'resnet152': ('https://paddle-hapi.bj.bcebos.com/models/resnet152.pdparams',
                  '7ad16a2f1e7333859ff986138630fd7a'),
}


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D

        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")

        self.conv1 = nn.Conv2D(
            inplanes, planes, 3, padding=1, stride=stride, bias_attr=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(planes, planes, 3, padding=1, bias_attr=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResnetEncoder(nn.Layer):
    def __init__(self, num_layers, pretrained, num_input_images=1, block=BasicBlock, depth=18, num_classes=1000, with_pool=True):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        layer_cfg = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]
        }
        layers = layer_cfg[depth]
        self.num_classes = num_classes
        self.with_pool = with_pool
        self._norm_layer = nn.BatchNorm2D

        self.inplanes = 64
        self.dilation = 1

        self.conv1 = nn.Conv2D(
            3,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias_attr=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if with_pool:
            self.avgpool = nn.AdaptiveAvgPool2D((1, 1))

        if num_classes > 0:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # self.ca0 = ChannelAttention(64)
        # self.sa0 = SpatialAttention()
        # self.ca1 = ChannelAttention(64)
        # self.sa1 = SpatialAttention()
        # self.ca2 = ChannelAttention(128)
        # self.sa2 = SpatialAttention()
        # self.ca3 = ChannelAttention(256)
        # self.sa3 = SpatialAttention()
        # self.ca4 = ChannelAttention(512)
        # self.sa4 = SpatialAttention()

        # if pretrained == 'paddle_pretrained':
        # self.load_dict(load_weight_file(pretrained))

        self.aspp = ASPP_module(512, 256, 16)
        # self.aspp = ASPP_module(256, 256, 16)
        self.conv2 = nn.Conv2D(1280, 512, 1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(512)
        # self.bn2 = nn.BatchNorm2D(256)

        # self.aspp = ASPP_module(256, 256, 16)
        # self.conv2 = nn.Conv2D(1280, 256, 1, bias_attr=False)
        # self.bn2 = nn.BatchNorm2D(256)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(
                    self.inplanes,
                    planes * block.expansion,
                    1,
                    stride=stride,
                    bias_attr=False),
                norm_layer(planes * block.expansion), )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, 1, 64,
                  previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, input_image):

        self.features = []
        x = (input_image - 0.45) / 0.225 # normalization?
        x = self.conv1(x)
        x = self.bn1(x)
        self.features.append(self.relu(x))
        self.features.append(self.layer1(self.maxpool(self.features[-1])))
        self.features.append(self.layer2(self.features[-1]))
        self.features.append(self.layer3(self.features[-1]))
        x = self.layer4(self.features[-1])

        """"
            attention
        """
        # x = self.relu(x)
        # t0 = x
        # t0 = self.ca0(t0) * t0
        # t0 = self.sa0(t0) * t0
        # self.features.append(t0)
        #
        # x = self.layer1(self.maxpool(x))
        # t1 = x
        # t1 = self.ca1(t1) * t1
        # t1 = self.sa1(t1) * t1
        # self.features.append(t1)
        #
        # x = self.layer2(x)
        # t2 = x
        # t2 = self.ca2(t2) * t2
        # t2 = self.sa2(t2) * t2
        # self.features.append(t2)
        #
        # x = self.layer3(x)
        # t3 = x
        # t3 = self.ca3(t3) * t3
        # t3 = self.sa3(t3) * t3
        # self.features.append(t3)
        #
        # x = self.layer4(x)
        # t4 = x
        # t4 = self.ca4(t4) * t4
        # t4 = self.sa4(t4) * t4
        # self.features.append(t4)

        """
            ASPP
        """
        x = self.aspp(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        self.features.append(x)

        return self.features


if __name__=='__main__':
    model = ResnetEncoder(18, False)
    image = paddle.randn(shape=(4, 3, 192, 640))
    output = model(image)
    print(output.size())