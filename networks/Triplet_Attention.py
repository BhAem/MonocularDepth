import paddle
import paddle.nn as nn


class BasicConv(nn.Layer):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias_attr=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2D(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias_attr=bias_attr,
        )
        self.bn = (
            nn.BatchNorm2D(out_planes, epsilon=1e-5, momentum=0.01)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Layer):
    def forward(self, x):
        #print(x.shape)#[4, 16, 512, 16][512, 1, 16][4, 1, 512, 16]

        #print(paddle.max(x, 1).unsqueeze(1).shape)
        #print(paddle.mean(x, 1).unsqueeze(1).shape)
        return paddle.concat(
                            (paddle.max(x, 1).unsqueeze(1),
                            paddle.mean(x, 1).unsqueeze(1))
                            ,axis=1)


class AttentionGate(nn.Layer):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = paddle.nn.functional.sigmoid(x_out)
        return x * scale


class TripletAttention(nn.Layer):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.transpose([0, 2, 1, 3])
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.transpose([0, 2, 1, 3])
        x_perm2 = x.transpose([0, 3, 2, 1])
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.transpose([0, 3, 2, 1])
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out


