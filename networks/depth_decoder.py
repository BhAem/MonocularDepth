import numpy as np
import paddle
import paddle.nn as nn
from collections import OrderedDict
from layers import *
from networks.BAM import BAM
from networks.CBAM import ChannelAttention, SpatialAttention
from networks.Coordinate_Attention import CoordAtt
from networks.SimAM import simam_module
from networks.Triplet_Attention import TripletAttention
from networks.scSE import scSE


class DepthDecoder(nn.Layer):
    def __init__(self, num_ch_enc, att_type=None, use_frelu=None, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.att_type = att_type
        self.num_ch_enc = num_ch_enc # [64, 64, 128, 256, 512]
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0                                                            # 4   3   2   1  0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1] # 512 256 128 64 32
            num_ch_out = self.num_ch_dec[i]                                       # 256 128 64  32 16
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out, use_frelu)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:            # 4   3   2  1
                num_ch_in += self.num_ch_enc[i - 1] # 256 128 64 64
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out, use_frelu)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.LayerList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        if self.att_type == 'CBAM':
            self.ca0 = ChannelAttention(512)
            self.sa0 = SpatialAttention()
            self.ca1 = ChannelAttention(256)
            self.sa1 = SpatialAttention()
            self.ca2 = ChannelAttention(128)
            self.sa2 = SpatialAttention()
            self.ca3 = ChannelAttention(64)
            self.sa3 = SpatialAttention()
            self.ca4 = ChannelAttention(32)
            self.sa4 = SpatialAttention()
        if self.att_type == 'BAM':
            self.bam0 = BAM(512)
            self.bam1 = BAM(256)
            self.bam2 = BAM(128)
            self.bam3 = BAM(64)
            self.bam4 = BAM(32)
        if self.att_type == 'scSE':
            self.scse0 = scSE(512)
            self.scse1 = scSE(256)
            self.scse2 = scSE(128)
            self.scse3 = scSE(64)
            self.scse4 = scSE(32)
        if self.att_type == 'TA':
            self.triplet_attention0 = TripletAttention()
            self.triplet_attention1 = TripletAttention()
            self.triplet_attention2 = TripletAttention()
            self.triplet_attention3 = TripletAttention()
            self.triplet_attention4 = TripletAttention()
        if self.att_type == "simam":
            self.simam0 = simam_module(512)
            self.simam1 = simam_module(256)
            self.simam2 = simam_module(128)
            self.simam3 = simam_module(64)
            self.simam4 = simam_module(32)
        if self.att_type == 'CA':
            self.coa0 = CoordAtt(512, 512)
            self.coa1 = CoordAtt(256, 256)
            self.coa2 = CoordAtt(128, 128)
            self.coa3 = CoordAtt(64, 64)
            self.coa4 = CoordAtt(32, 32)

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):

            if self.att_type == 'CBAM':
                if i == 4:
                    x = self.ca0(x) * x
                    x = self.sa0(x) * x
                elif i == 3:
                    x = self.ca1(x) * x
                    x = self.sa1(x) * x
                elif i == 2:
                    x = self.ca2(x) * x
                    x = self.sa2(x) * x
                elif i == 1:
                    x = self.ca3(x) * x
                    x = self.sa3(x) * x
                elif i == 0:
                    x = self.ca4(x) * x
                    x = self.sa4(x) * x

            if self.att_type == 'BAM':
                if i == 4:
                    x = self.bam0(x) * x
                elif i == 3:
                    x = self.bam1(x) * x
                elif i == 2:
                    x = self.bam2(x) * x
                elif i == 1:
                    x = self.bam3(x) * x
                elif i == 0:
                    x = self.bam4(x) * x

            if self.att_type == 'scSE':
                if i == 4:
                    x = self.scse0(x)
                elif i == 3:
                    x = self.scse1(x)
                elif i == 2:
                    x = self.scse2(x)
                elif i == 1:
                    x = self.scse3(x)
                elif i == 0:
                    x = self.scse4(x)

            if self.att_type == 'TA':
                if i == 4:
                    x = self.triplet_attention0(x)
                elif i == 3:
                    x = self.triplet_attention1(x)
                elif i == 2:
                    x = self.triplet_attention2(x)
                elif i == 1:
                    x = self.triplet_attention3(x)
                elif i == 0:
                    x = self.triplet_attention4(x)

            if self.att_type == 'simam':
                if i == 4:
                    x = self.simam0(x)
                elif i == 3:
                    x = self.simam1(x)
                elif i == 2:
                    x = self.simam2(x)
                elif i == 1:
                    x = self.simam3(x)
                elif i == 0:
                    x = self.simam4(x)

            if self.att_type == 'CA':
                if i == 4:
                    x = self.coa0(x)
                elif i == 3:
                    x = self.coa1(x)
                elif i == 2:
                    x = self.coa2(x)
                elif i == 1:
                    x = self.coa3(x)
                elif i == 0:
                    x = self.coa4(x)

            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = paddle.concat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs
