# Adapted from monodepth2
# https://github.com/nianticlabs/monodepth2/blob/master/networks/depth_decoder.py
#
# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import numpy as np
import torch
from torch import nn

from models.model_parts import ASPP
from models.monodepth_layers import ConvBlock, Conv3x3, upsample


class DepthDecoder(nn.Module):
    first_iter = True

    def __init__(self, num_ch_enc, scales, max_scale_size, num_output_channels=1, use_skips=True,
                 intermediate_aspp=False, aspp_rates=[6, 12, 18], num_ch_dec=[16, 32, 64, 128, 256],
                 n_upconv=4, batch_norm=False, dropout=0.0, n_project_skip_ch=-1,
                 aspp_pooling=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.enable_disparity = True
        self.max_scale_size = np.asarray(max_scale_size)

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array(num_ch_dec)
        self.n_upconv = n_upconv

        # decoder
        self.convs = OrderedDict()
        for i in range(self.n_upconv, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == self.n_upconv else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            if i == self.n_upconv and intermediate_aspp:
                self.convs[("upconv", i, 0)] = ASPP(num_ch_in, aspp_rates, aspp_pooling, num_ch_out)
            else:
                self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out, bn=batch_norm, dropout=dropout)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                if n_project_skip_ch == -1:
                    num_ch_in += self.num_ch_enc[i - 1]
                    self.convs[("skip_proj", i)] = nn.Identity()
                else:
                    num_ch_in += n_project_skip_ch
                    self.convs[("skip_proj", i)] = nn.Sequential(
                        nn.Conv2d(self.num_ch_enc[i - 1], n_project_skip_ch, kernel_size=1),
                        nn.BatchNorm2d(n_project_skip_ch),
                        nn.ReLU(inplace=True)
                    )
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out, bn=batch_norm, dropout=dropout)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features, x=None, exec_layer=None):
        self.outputs = {}

        # decoder
        if x is None:
            x = input_features[-1]
        if exec_layer is None:
            exec_layer = "all"
        if DepthDecoder.first_iter:
            print(f"bottleneck shape {x.shape}")
        for i in range(self.n_upconv, -1, -1):
            if exec_layer != "all" and i not in exec_layer:
                continue
            x = self.convs[("upconv", i, 0)](x)
            # if i == self.n_upconv:
            #     self.outputs["aspp"] = x
            if DepthDecoder.first_iter:
                print(f"upconv{i}-0 shape: {x.shape}")
            if x.shape[-1] < input_features[i - 1].shape[-1] or i == 0:
                x = [upsample(x)]
            else:
                x = [x]
            if self.use_skips and i > 0:
                projected_features = self.convs[("skip_proj", i)](input_features[i - 1])
                x += [projected_features]
            x = torch.cat(x, 1)
            if DepthDecoder.first_iter:
                print(f"concatenated features shape: {x.shape}")
            x = self.convs[("upconv", i, 1)](x)
            self.outputs[("upconv", i)] = x
            if DepthDecoder.first_iter:
                print(f"upconv{i}-1 shape: {x.shape}")
            if i in self.scales and self.enable_disparity:
                size = self.max_scale_size // (2 ** i)
                disp_out = self.sigmoid(self.convs[("dispconv", i)](x))
                if DepthDecoder.first_iter:
                    print(f"disp{i} shape: {disp_out.shape}, expected {size}")
                self.outputs[("disp", i)] = disp_out
            if i == 0:
                DepthDecoder.first_iter = False

        return self.outputs
