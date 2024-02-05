import paddle
import paddle.nn as nn
import math
import paddle.nn.functional as F


class CoordAtt(nn.Layer):
    def __init__(self, in_ch, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2D((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2D((1, None))

        mip = max(8, in_ch // reduction)

        self.conv1 = nn.Conv2D(in_ch, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2D(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2D(mip, in_ch, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2D(mip, in_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).transpose([0, 1, 3, 2])

        y = paddle.concat([x_h, x_w], axis=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = paddle.split(y, [h, w], axis=2)
        x_w = x_w.transpose([0, 1, 3, 2])

        x_h = F.sigmoid(self.conv_h(x_h))
        x_w = F.sigmoid(self.conv_w(x_w))

        out = identity * x_w * x_h

        return out