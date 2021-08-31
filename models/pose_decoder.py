# Obtained from monodepth2
# https://github.com/nianticlabs/monodepth2/blob/master/networks/pose_decoder.py
#
# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import torch
import torch.nn as nn


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1, norm=None, scale=0.01):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features
        self.scale = scale
        print(f"Pose scale {scale}")

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()

        def _wrap_norm(conv, ch):
            if norm is None:
                return conv
            elif norm == "batch":
                return nn.Sequential(conv, nn.BatchNorm2d(ch))
            elif norm == "group":
                return nn.Sequential(conv, nn.GroupNorm(16, ch))
            else:
                raise NotImplementedError(norm)

        self.convs[("squeeze")] = _wrap_norm(nn.Conv2d(self.num_ch_enc[-1], 256, 1), 256)
        self.convs[("pose", 0)] = _wrap_norm(nn.Conv2d(num_input_features * 256, 256, 3, stride, 1), 256)
        self.convs[("pose", 1)] = _wrap_norm(nn.Conv2d(256, 256, 3, stride, 1), 256)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        out = self.scale * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation
