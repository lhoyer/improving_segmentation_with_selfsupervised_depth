import os
import random

import torch
from matplotlib import pyplot as plt
from torch.utils import data

from loader.transformsgpu import strongTransformOneMix
from utils.visualization import subplotimg


def debug_depthmix(source_img, source_depth, target_img, target_depth):
    depthcomp_margin = 0.05
    depthcomp_margin_ie_random_flip = False

    own_disp = source_depth
    other_disp = target_depth
    local_depthcomp_margin = depthcomp_margin
    if depthcomp_margin_ie_random_flip and random.random() > 0.5:
        local_depthcomp_margin *= -1
    foreground_mask = torch.ge(own_disp, other_disp - local_depthcomp_margin).long()
    MixMask = foreground_mask.unsqueeze(0)
    strong_parameters = {
        "Mix": MixMask,
        "ColorJitter": 0,
        "GaussianBlur": 0,
    }
    inputs_u_s, _ = strongTransformOneMix(
        strong_parameters, onemix=True,
        data=torch.cat((source_img.unsqueeze(0), target_img.unsqueeze(0))),
        target=None,
    )
    return inputs_u_s

class MatchingGeometryDataset(data.Dataset):
    def __init__(self, source_dataset, target_dataset, opts, debug=False):
        self.lambda_depthdiff = opts["lambda_depthdiff"]
        self.n_candidates = opts["n_candidates"]
        self.diff_bound = opts["diff_bound"]
        self.diff_crop = opts["diff_crop"]
        self.disp_scale = opts["disp_scale"]
        self.bidirectional = opts["bidirectional"]
        self.debug = debug

        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

    def __len__(self):
        if self.bidirectional:
            return 2 * max(len(self.target_dataset), len(self.source_dataset))
        else:
            return len(self.target_dataset)

    def __getitem__(self, index):
        if not self.bidirectional:
            data_i = index
            base_domain = "target"
            base_dataset = self.target_dataset
            cand_dataset = self.source_dataset
        elif (index % 2) == 0:
            data_i = (index // 2) % len(self.target_dataset)
            base_domain = "target"
            base_dataset = self.target_dataset
            cand_dataset = self.source_dataset
        else:
            data_i = (index // 2) % len(self.source_dataset)
            base_domain = "source"
            base_dataset = self.source_dataset
            cand_dataset = self.target_dataset

        def scale_disp(disp):
            if self.disp_scale == "synthia":
                return 655.36 / (disp * 255. + 1.) - 0.01
            elif self.disp_scale == "inv":
                return 1 / (disp + 0.1)
            elif self.disp_scale == "loginv":
                return torch.log(1 / (disp + 0.01))
            elif self.disp_scale == "log01":
                return torch.log(disp + 0.1)
            elif self.disp_scale == "log1":
                return torch.log(disp + 1)
            elif self.disp_scale == "id":
                return disp
            else:
                raise NotImplementedError(self.disp_scale)

        base_inp = base_dataset[data_i]
        base_disp = base_inp["pseudo_depth"]
        base_disp = scale_disp(base_disp)
        # print("disp", torch.min(base_disp), torch.max(base_disp), "depth", torch.min(base_depth), torch.max(base_depth))

        if self.debug:
            rows, cols = 1, 4
            tile_size = 2.5
            fig, axs = plt.subplots(
                rows, cols, figsize=(tile_size * cols, tile_size * rows + 0.4),
                gridspec_kw={'hspace': 0, 'wspace': 0.02, 'top': 0.95, 'bottom': 0, 'right': 1, 'left': 0},
            )
            subplotimg(axs[0], base_inp[("color", 0, 0)], "Target Image")
            subplotimg(axs[1], torch.log(0.1 + base_disp), "Target Depth", cmap="plasma_r" if "inv" in self.disp_scale else "plasma")
            # subplotimg(axs[2], base_inp["lbl"], "Target Seg GT", cmap="cityscapes")
            for ax in axs.flat:
                ax.axis("off")
            os.makedirs("plots/matching_geom_vis", exist_ok=True)
            plt.savefig(f"plots/matching_geom_vis/base_{data_i}.jpg", dpi=400)
            plt.show()

        min_criterion = 1e15
        candidates = []
        for _ in range(self.n_candidates):
            # Important: Use random.choice instead of np.choice here as otherwise the seed is not setup properly for the
            # DataLoader worker: https://discuss.pytorch.org/t/is-there-a-way-to-fix-the-random-seed-of-every-workers-in-dataloader/21687/9
            cand_i = random.choice(list(range(len(cand_dataset))))
            cand_inp = cand_dataset[cand_i]
            cand_disp = cand_inp["pseudo_depth"]
            cand_disp = scale_disp(cand_disp)
            criterion = 0
            if self.lambda_depthdiff > 0:
                geom_diff = torch.clamp(torch.abs(base_disp - cand_disp), 0, self.diff_bound)
                if self.diff_crop:
                    # Remove ego car
                    geom_diff[-100:, :] = 0
                    # Remove sky artifacts
                    geom_diff[:80, :] = 0
                criterion += self.lambda_depthdiff * torch.mean(geom_diff)
            else:
                raise ValueError
            if criterion < min_criterion:
                min_criterion = criterion
                matching_inp = cand_inp
            if self.debug:
                candidates.append({"data_i": cand_i, "criterion": criterion, "img": cand_inp[("color", 0, 0)],
                                   "lbl": cand_inp["lbl"], "depth": cand_disp,
                                   "diff": geom_diff})

        if self.debug:
            candidates = sorted(candidates, key=lambda k: k['criterion'])
            for c in [*candidates[:4], *candidates[-4:]]:
                # print(c["criterion"])
                rows, cols = 1, 4
                fig, axs = plt.subplots(
                    rows, cols, figsize=(tile_size * cols, tile_size * rows + 0.4),
                    gridspec_kw={'hspace': 0, 'wspace': 0.02, 'top': 0.95, 'bottom': 0, 'right': 1, 'left': 0},
                )
                mixed_img = debug_depthmix(c["img"], c["depth"], base_inp[("color", 0, 0)], base_disp)
                subplotimg(axs[0], c["img"], "Source Candidate Image")
                subplotimg(axs[1], torch.log(0.1 + c["depth"]), "Source Candidate Depth", cmap="plasma_r" if "inv" in self.disp_scale else "plasma")
                if c["diff"] is not None:
                    subplotimg(axs[2], c["diff"], "Geometric Difference", cmap="viridis", vmin=0, vmax=1)
                subplotimg(axs[3], mixed_img[0], "DepthMix Image")
                # subplotimg(axs[3], c["lbl"], "Source Candidate Seg GT", cmap="cityscapes")
                for ax in axs.flat:
                    ax.axis("off")
                plt.savefig(f"plots/matching_geom_vis/base_{data_i}_cand_{c['data_i']}.jpg", dpi=400)
                plt.show()

        if not self.bidirectional or base_domain == "target":
            return matching_inp, base_inp
        else:
            return base_inp, matching_inp
