# Adapted from monodepth2
# https://github.com/nianticlabs/monodepth2/blob/master/trainer.py
#
# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import torch
import torch.nn.functional as F

from models.monodepth_layers import disp_to_depth, get_smooth_loss, SSIM, BackprojectDepth, Project3D


class MonodepthLoss:
    def __init__(self, num_scales, frame_ids, height, width, batch_size, min_depth, max_depth,
                 test_min_depth, test_max_depth, disparity_smoothness,
                 no_ssim, avg_reprojection, disable_automasking, crop_h=None, crop_w=None, is_train=True):
        self.num_scales = num_scales
        self.scales = list(range(self.num_scales))
        self.height = height if crop_h is None or not is_train else crop_h
        self.width = width if crop_w is None or not is_train else crop_w
        self.batch_size = batch_size
        self.frame_ids = frame_ids
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.test_min_depth = test_min_depth
        self.test_max_depth = test_max_depth
        self.disparity_smoothness = disparity_smoothness
        self.no_ssim = no_ssim
        self.avg_reprojection = avg_reprojection
        self.disable_automasking = disable_automasking
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.depth_metric_names = [
            "abs_rel", "sq_rel", "rms", "log_rms", "a1", "a2", "a3"]

        if not self.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.scales:
            h = self.height // (2 ** scale)
            w = self.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.batch_size, h, w)
            self.project_3d[scale].to(self.device)

    def generate_depth_test_pred(self, outputs):
        assert outputs[("disp", 0)].shape[-2:] == (self.height, self.width), outputs[("disp", 0)].shape[-2:]
        for scale in self.scales:
            disp = outputs[("disp", scale)]
            disp = F.interpolate(
                disp, [self.height, self.width], mode="bilinear", align_corners=False)

            _, depth = disp_to_depth(disp, self.test_min_depth, self.test_max_depth)
            outputs[("depth", 0, scale)] = depth

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        assert outputs[("disp", 0)].shape[-2:] == (
            self.height, self.width), f'{outputs[("disp", 0)].shape[-2:]} should be {(self.height, self.width)} '
        for scale in self.scales:
            disp = outputs[("disp", scale)]
            disp = F.interpolate(
                disp, [self.height, self.width], mode="bilinear", align_corners=False)
            source_scale = 0

            _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border",
                    align_corners=True)

                if not self.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.scales:
            loss = 0
            reprojection_losses = []

            source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            if self.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.disable_automasking:
                # add random numbers to break ties
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).to(device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                        idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()
            # outputs["to_optimise/{}".format(scale)] = to_optimise

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

