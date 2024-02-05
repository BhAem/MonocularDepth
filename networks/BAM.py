import paddle
import math
import paddle.nn as nn
import paddle.nn.functional as F


class ChannelAttention(nn.Layer):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        mid_channel = channel // reduction
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            nn.ReLU(),
            nn.Linear(in_features=mid_channel, out_features=channel)
        )
    def forward(self, x):
        avg = self.avg_pool(x).flatten(x.size(0), -1)
        out = self.shared_MLP(avg).unsqueeze(2).unsqueeze(3).expand_as(x)
        return out


class SpatialAttention(nn.Layer):
    def __init__(self, channel, reduction=16, dilation_conv_num=2, dilation_rate=4):
        super(SpatialAttention, self).__init__()
        mid_channel = channel // reduction
        self.reduce_conv = nn.Sequential(
            nn.Conv2D(channel, mid_channel, kernel_size=1),
            nn.BatchNorm2D(mid_channel),
            nn.ReLU()
        )
        dilation_convs_list = []
        for i in range(dilation_conv_num):
            dilation_convs_list.append(nn.Conv2D(mid_channel, mid_channel, kernel_size=3, padding=dilation_rate, dilation=dilation_rate))
            dilation_convs_list.append(nn.BatchNorm2D(mid_channel))
            dilation_convs_list.append(nn.ReLU())
        self.dilation_convs = nn.Sequential(*dilation_convs_list)
        self.final_conv = nn.Conv2D(mid_channel, channel, kernel_size=1)
    def forward(self, x):
        y = self.reduce_conv(x)
        x = self.dilation_convs(y)
        out = self.final_conv(y)#.expand_as(x)
        return out


class BAM(nn.Layer):
    """
        BAM: Bottleneck Attention Module
        https://arxiv.org/pdf/1807.06514.pdf
    """
    def __init__(self, channel):
        super(BAM, self).__init__()
        self.channel_attention = ChannelAttention(channel)
        self.spatial_attention = SpatialAttention(channel)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        att = 1 + self.sigmoid(self.channel_attention(x) * self.spatial_attention(x))
        return att * x