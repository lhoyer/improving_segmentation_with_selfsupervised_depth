import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model_parts import SelfAttention
from .utils import _get_layer
from .utils import get_depth_decoder


class JointSegDepthDecoder(nn.Module):
    first_iter = True

    def __init__(self, num_ch_enc, num_ch_dec, num_classes, layers=None,
                 head_inter_channels=64, weights='none',
                 head_dropout=0.1, layer_dropout=0, output_stride=1, layer_out_channels=64,
                 depth_args=None, head_inter=True):
        super(JointSegDepthDecoder, self).__init__()
        if layers is None:
            layers = [9]
        self.output_stride = output_stride
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = num_ch_dec
        self.num_classes = num_classes
        self.layers = layers
        assert len(num_ch_enc) == 5
        assert len(num_ch_dec) == 5
        self.last_layer = len(num_ch_enc) + len(num_ch_dec) - 1
        self.unet_dec = get_depth_decoder(weights, num_ch_enc, **depth_args)
        accumulated_ch = 0
        self.project = {}
        for layer in layers:
            ch = num_ch_enc[layer] if layer <= 4 else num_ch_dec[self.last_layer - layer]
            accumulated_ch += layer_out_channels
            self.project[f"seg{layer}"] = nn.Sequential(
                nn.Conv2d(ch, layer_out_channels, 1, bias=False),
            )
        self.project = nn.ModuleDict(self.project)

        if head_inter:
            head_conv = [
                nn.Conv2d(accumulated_ch, head_inter_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(head_inter_channels),
                nn.ReLU(),
                nn.Dropout(head_dropout),
            ]
        else:
            head_conv = [nn.Identity()]
        self.head = nn.Sequential(
            nn.Dropout(layer_dropout) if layer_dropout > 0 else nn.Identity(),
            *head_conv,
            nn.Conv2d(head_inter_channels, self.num_classes, 1)
        )

    def forward(self, encoder_features):
        seg_features = self.unet_dec(encoder_features)
        segmentation_size = _get_layer(encoder_features, seg_features, self.last_layer).shape[2:]
        last_layer_size = tuple(np.array(segmentation_size) // self.output_stride)
        stacked_features = []
        for layer in self.layers:
            if f"seg{layer}" in self.project:
                proj_own = self.project[f"seg{layer}"](_get_layer(encoder_features, seg_features, layer))
                if JointSegDepthDecoder.first_iter: print(f'proj_seg{layer}', proj_own.shape)
                proj_own = F.interpolate(proj_own, size=last_layer_size,
                                         mode='bilinear', align_corners=False)
                if JointSegDepthDecoder.first_iter: print(f'proj_seg{layer} upsampled', proj_own.shape)
                stacked_features.append(proj_own)
        stacked_features = torch.cat(stacked_features, dim=1)
        if JointSegDepthDecoder.first_iter: print(f'stacked', stacked_features.shape)
        score = self.head(stacked_features)
        if JointSegDepthDecoder.first_iter: print(f'score before interpolate', score.shape)
        if last_layer_size != segmentation_size:
            score = F.interpolate(score, size=segmentation_size, mode='bilinear', align_corners=False)
        JointSegDepthDecoder.first_iter = False
        return score


class PAD(nn.Module):
    first_iter = True

    def __init__(self, num_ch_enc, num_ch_dec, num_classes, final_layer=9,
                 weights=None, output_stride=1, depth_args=None, distillation_layer=7, side_output=True):
        super(PAD, self).__init__()
        self.output_stride = output_stride
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = num_ch_dec
        self.num_classes = num_classes
        self.side_output = side_output
        assert len(num_ch_enc) == 5
        assert len(num_ch_dec) == 5
        self.final_layer = final_layer
        self.last_layer = len(num_ch_enc) + len(num_ch_dec) - 1
        self.distillation_layer = distillation_layer
        self.dec_n_upconv = depth_args.get("n_upconv", 4)
        distillation_ch = self.layer_channels(self.distillation_layer)
        final_ch = self.layer_channels(self.final_layer)

        num_scales = 4
        self.depth_dec = get_depth_decoder(weights, num_ch_enc, range(num_scales), **depth_args)
        self.seg_dec = get_depth_decoder(weights, num_ch_enc, range(num_scales), **depth_args)
        self.seg_dec.enable_disparity = False
        for s in range(num_scales):
            self.seg_dec.convs[("dispconv", s)] = nn.Identity()

        self.sa_depth = SelfAttention(distillation_ch, distillation_ch)
        self.sa_seg = SelfAttention(distillation_ch, distillation_ch)
        if self.side_output:
            self.seg_intermediate_head = nn.Sequential(
                nn.Conv2d(distillation_ch, self.num_classes, 1)
            )
        self.seg_final_head = nn.Sequential(
            nn.Conv2d(final_ch, self.num_classes, 1)
        )

    def layer_channels(self, layer):
        return self.num_ch_enc[layer] if layer <= 4 else self.num_ch_dec[self.last_layer - layer]

    def depth_params(self):
        return [
            *self.depth_dec.parameters(),
            *self.sa_seg.parameters(),
        ]

    def segmentation_params(self):
        params = [
            *self.seg_dec.parameters(),
            *self.sa_depth.parameters(),
            *self.seg_final_head.parameters(),
        ]
        if self.side_output:
            params.extend(self.seg_intermediate_head.parameters())
        return params

    def forward(self, encoder_features):
        segmentation_size = encoder_features[0].shape[2:]
        last_layer_size = tuple(np.array(segmentation_size) // self.output_stride)

        dec_distill_i = self.last_layer - self.distillation_layer
        intermediate_layer_name = ("upconv", dec_distill_i)
        first_exec_layers = list(range(self.dec_n_upconv, dec_distill_i - 1, -1))
        second_exec_layers = list(range(dec_distill_i - 1, -1, -1))

        # Initial predictions
        if PAD.first_iter:
            print(f"PAD run first half of decoder ({first_exec_layers}).")
        depth_features = self.depth_dec(encoder_features, exec_layer=first_exec_layers)
        seg_features = self.seg_dec(encoder_features, exec_layer=first_exec_layers)
        if self.side_output:
            intermediate_seg = self.seg_intermediate_head(seg_features[intermediate_layer_name])

        # Self-attention
        features_sa_depth = self.sa_depth(depth_features[intermediate_layer_name])
        features_sa_seg = self.sa_seg(seg_features[intermediate_layer_name])
        if PAD.first_iter:
            print(f"PAD self attention for {intermediate_layer_name} with shape {features_sa_depth.shape}.")

        # Feature stack
        merged_for_seg = seg_features[intermediate_layer_name] + features_sa_depth
        merged_for_depth = depth_features[intermediate_layer_name] + features_sa_seg

        if PAD.first_iter:
            print(f"PAD run second half of decoder ({second_exec_layers}).")
        depth_features.update(self.depth_dec(encoder_features, x=merged_for_depth, exec_layer=second_exec_layers))
        seg_features = self.seg_dec(encoder_features, x=merged_for_seg, exec_layer=second_exec_layers)
        final_seg = self.seg_final_head(_get_layer(features_sa_depth, seg_features, self.final_layer))

        if self.side_output and last_layer_size != segmentation_size:
            intermediate_seg = F.interpolate(
                intermediate_seg, size=segmentation_size, mode='bilinear', align_corners=False
            )
        if last_layer_size != segmentation_size:
            final_seg = F.interpolate(
                final_seg, size=segmentation_size, mode='bilinear', align_corners=False
            )

        PAD.first_iter = False

        out = {
            **depth_features,
            "semantics": final_seg,
        }
        if self.side_output:
            out["intermediate_semantics"] = intermediate_seg
        return out
