import torch
from torch import nn

from .joint_segmentation_depth_decoder import JointSegDepthDecoder, PAD
from .monodepth_layers import transformation_from_parameters
from .utils import get_depth_decoder, get_posenet
from .utils import get_resnet_backbone


class JointSegmentationMonodepth(nn.Module):
    def __init__(self, models, frame_ids, use_pose_net, num_pose_frames, provide_uncropped_for_pose):
        super(JointSegmentationMonodepth, self).__init__()
        self.frame_ids = frame_ids
        self.use_pose_net = use_pose_net
        self.num_pose_frames = num_pose_frames
        self.provide_uncropped_for_pose = provide_uncropped_for_pose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = nn.ModuleDict(models)

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.provide_uncropped_for_pose:
                pose_feats = {f_i: inputs["color_full_aug", f_i, 0] for f_i in self.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.frame_ids}

            for f_i in self.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.provide_uncropped_for_pose:
                pose_inputs = torch.cat(
                    [inputs[("color_full_aug", i, 0)] for i in self.frame_ids if i != "s"], 1)
            else:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.frame_ids if i != "s"], 1)
            pose_inputs = [self.models["pose_encoder"](pose_inputs)]
            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def predict_test_disp(self, x):
        input_color = x[("color", 0, 0)]
        output = self.models["depth"](self.models["encoder"](input_color))
        return output

    def forward(self, x):
        outputs = {}
        inputs = x

        features = self.models["encoder"](inputs["color_aug", 0, 0])
        outputs["bottleneck"] = features[-1]
        if "mtl_decoder" in self.models:
            outputs.update(self.models["mtl_decoder"](features))
        else:
            if "depth" in self.models:
                outputs.update(self.models["depth"](features))
            if "segmentation" in self.models:
                outputs["semantics"] = self.models["segmentation"](features)

        if "imnet_encoder" in self.models:
            outputs["encoder_features"] = features[-1]
            self.models["imnet_encoder"].eval()
            with torch.no_grad():
                outputs["imnet_features"] = self.models["imnet_encoder"](inputs["color_aug", 0, 0])[-1].detach()

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        return outputs


def get_segmentation_network(segmentation_name, num_ch_enc, segmentation_size, num_classes, segmentation_args,
                             depth_args):
    model_map = {
        'joint_seg_depth_dec': JointSegDepthDecoder,
        'mtl_pad': PAD,
    }
    num_ch_dec = depth_args.get("num_ch_dec", [16, 32, 64, 128, 256])
    segmentation_net = model_map[segmentation_name](num_ch_enc, num_ch_dec,
                                                    num_classes, **segmentation_args, depth_args=depth_args)

    return segmentation_net


def joint_segmentation_depth(name, backbone_name, segmentation_name, segmentation_args,
                             num_classes, backbone_pretraining,
                             depth_pretraining, pose_pretraining, freeze_backbone,
                             freeze_segmentation,
                             freeze_depth, freeze_pose, replace_stride_with_dilation,
                             frame_ids, num_scales, pose_model_input, provide_uncropped_for_pose,
                             height, width, depth_args, disable_monodepth, enable_imnet_encoder,
                             disable_pose, imnet_encoder_dilation=True, **kwargs):
    num_input_frames = len(frame_ids)
    num_pose_frames = 2 if pose_model_input == "pairs" else num_input_frames
    assert frame_ids[0] == 0
    use_pose_net = not (frame_ids == (0, "s")) and not disable_pose

    models = {}
    models["encoder"] = get_resnet_backbone(
        backbone_name, backbone_pretraining,
        replace_stride_with_dilation, use_intermediate_layer_getter=False
    )
    num_ch_enc = models["encoder"].num_ch_enc

    if enable_imnet_encoder:
        models["imnet_encoder"] = get_resnet_backbone(
            backbone_name, 'imnet',
            replace_stride_with_dilation=replace_stride_with_dilation if imnet_encoder_dilation else None,
            use_intermediate_layer_getter=False
        )
        for param in models["imnet_encoder"].parameters():
            param.requires_grad = False

    if use_pose_net and not disable_monodepth:
        models.update(get_posenet("resnet18", backbone_pretraining, pose_pretraining, num_pose_frames))

    if segmentation_name in ["mtl_pad"]:
        models["mtl_decoder"] = get_segmentation_network(segmentation_name, num_ch_enc, (height, width),
                                                         num_classes, segmentation_args, depth_args)
    else:
        if not disable_monodepth:
            models["depth"] = get_depth_decoder(depth_pretraining, num_ch_enc, range(num_scales), **depth_args)
        if segmentation_name is not None:
            models["segmentation"] = get_segmentation_network(segmentation_name, num_ch_enc, (height, width),
                                                              num_classes, segmentation_args, depth_args)

    if freeze_backbone:
        print('Freeze backbone weights')
        for param in models["encoder"].parameters():
            param.requires_grad = False

    if not disable_monodepth and freeze_depth:
        print('Freeze depth decoder weights')
        for param in models["depth"].parameters():
            param.requires_grad = False

    if not disable_monodepth and freeze_pose:
        print('Freeze pose decoder weights')
        if "pose_encoder" in models:
            for param in models["pose_encoder"].parameters():
                param.requires_grad = False
        for param in models["pose"].parameters():
            param.requires_grad = False

    if "segmentation" in models and freeze_segmentation:
        print('Freeze segmentation decoder weights')
        for param in models["segmentation"].parameters():
            param.requires_grad = False

    model = JointSegmentationMonodepth(models, frame_ids, use_pose_net, num_pose_frames,
                                       provide_uncropped_for_pose)
    return model
