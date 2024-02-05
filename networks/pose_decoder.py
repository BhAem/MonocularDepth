import paddle
import paddle.nn as nn
from collections import OrderedDict

from networks.BAM import BAM
from networks.CBAM import ChannelAttention, SpatialAttention
from networks.Coordinate_Attention import CoordAtt
from networks.SimAM import simam_module
from networks.Triplet_Attention import TripletAttention
from networks.scSE import scSE


class PoseDecoder(nn.Layer):
    def __init__(self, num_ch_enc, att_type, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features
        self.att_type = att_type
        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2D(self.num_ch_enc[-1], 256, 1, weight_attr=nn.initializer.KaimingUniform())
        self.convs[("pose", 0)] = nn.Conv2D(num_input_features * 256, 256, 3, stride, 1, weight_attr=nn.initializer.KaimingUniform())
        self.convs[("pose", 1)] = nn.Conv2D(256, 256, 3, stride, 1, weight_attr=nn.initializer.KaimingUniform())
        self.convs[("pose", 2)] = nn.Conv2D(256, 6 * num_frames_to_predict_for, 1, weight_attr=nn.initializer.KaimingUniform())

        self.relu = nn.ReLU()

        if self.att_type == 'CBAM':
            self.ca0 = ChannelAttention(256)
            self.sa0 = SpatialAttention()
        if self.att_type == 'BAM':
            self.bam0 = BAM(256)
        if self.att_type == 'scSE':
            self.scse0 = scSE(256)
        if self.att_type == 'TA':
            self.triplet_attention = TripletAttention()
        if self.att_type == "simam":
            self.simam = simam_module(256)
        if self.att_type == 'CA':
            self.ca = CoordAtt(256, 256)

        self.net = nn.LayerList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = paddle.concat(cat_features, axis=1)

        out = cat_features
        # for i in range(3):
        #     out = self.convs[("pose", i)](out)
        #     if i != 2:
        #         out = self.relu(out)

        """
            attention
        """
        out = self.convs[("pose", 0)](out)
        out = self.relu(out)
        out = self.convs[("pose", 1)](out)
        out = self.relu(out)
        if self.att_type == 'CBAM':
            out = self.ca0(out) * out
            out = self.sa0(out) * out
        if self.att_type == 'BAM':
            out = self.bam0(out) * out
        if self.att_type == 'scSE':
            out = self.scse0(out)
        if self.att_type == 'TA':
            out = self.triplet_attention(out)
        if self.att_type == "simam":
            out = self.simam(out)
        if self.att_type == 'CA':
            out = self.ca(out)
        out = self.convs[("pose", 2)](out)

        out = out.mean(3).mean(2)

        out = 0.01 * out.reshape((-1, self.num_frames_to_predict_for, 1, 6))

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation
