import os
from copy import deepcopy

import torch
from torch.utils import data
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from experiments import decoder_variant
from loader import build_loader
from loss import get_monodepth_loss
from models import get_model


class DepthEstimator:
    def __init__(self, cfg):
        cfg = deepcopy(cfg)
        cfg["model"]["arch"] = "joint_segmentation_depth"
        cfg["model"]["segmentation_name"] = None
        cfg["model"]["disable_monodepth"] = False
        cfg["model"]["disable_pose"] = True
        cfg['data']['augmentations'] = {}
        cfg['data'].pop('crop_h', None)
        cfg['data'].pop('crop_w', None)
        assert not (cfg["data"].get("depth_teacher") is not None and
                    cfg['model'].get("depth_estimator_weights") is not None)
        if cfg["data"].get("depth_teacher") is not None:
            cfg['model']['backbone_name'] = "resnet101"
            cfg, load_backbone = decoder_variant(cfg, 6, (512, 512))
            cfg['model']['depth_pretraining'] = cfg["data"]["depth_teacher"]
            cfg['model']['backbone_pretraining'] = cfg["data"]["depth_teacher"]
        if cfg['model'].get("depth_estimator_weights") is not None:
            cfg['model']['backbone_pretraining'] = cfg['model']['depth_estimator_weights']
            cfg['model']['depth_pretraining'] = cfg['model']['depth_estimator_weights']

        self.cfg = cfg
        assert cfg['model']['depth_pretraining'] == cfg['model']['backbone_pretraining']
        self.depth_dir = cfg["data"]["generated_depth_dir"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.monodepth_loss_calculator = get_monodepth_loss(self.cfg, is_train=False)

        unrestricted_cfg = deepcopy(self.cfg["data"])
        unrestricted_cfg.update({"restrict_to_subset": None, "generated_depth_dir": None})
        self.train_loader = build_loader(unrestricted_cfg, "train", load_labels=False, load_sequence=False)
        self.val_loader = build_loader(unrestricted_cfg, "val", load_labels=False, load_sequence=False)
        self.loader = data.ConcatDataset([self.train_loader, self.val_loader])
        self.n_classes = self.train_loader.n_classes

        batch_size = 4
        self.data_loader = data.DataLoader(
            self.loader,
            batch_size=batch_size,
            num_workers=self.cfg["training"]["n_workers"],
            pin_memory=True,
        )

        self.model = get_model(cfg["model"], self.n_classes).to(self.device)

    def build_filename(self, subname):
        return self.depth_dir + subname.replace('.jpg', '.png')

    def prepare_depth_estimates(self):
        self.model.eval()
        with torch.no_grad():
            for inputs_val in tqdm(self.data_loader,
                                   total=len(self.data_loader)):
                batch_exists = True
                for f in inputs_val["filename"]:
                    filename = self.build_filename(f)
                    if not os.path.isfile(filename):
                        batch_exists = False
                if batch_exists:
                    continue

                for k, v in inputs_val.items():
                    if torch.is_tensor(v) and k == ("color", 0, 0):
                        inputs_val[k] = v.to(self.device)

                mono_outputs = self.model.predict_test_disp(inputs_val)
                self.monodepth_loss_calculator.generate_depth_test_pred(mono_outputs)
                # depths = mono_outputs[("depth", 0, 0)].cpu()
                depths = mono_outputs[("disp", 0)].cpu()

                for subname, depth in zip(inputs_val["filename"], depths):
                    filename = self.build_filename(subname)
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    dmin = torch.min(depth)
                    dmax = torch.max(depth)
                    depth = torch.clamp(depth, dmin, dmax)
                    depth = (depth - dmin) / (dmax - dmin)
                    img = ToPILImage()(depth.squeeze_(0))
                    if not os.path.isfile(filename):
                        img.save(filename)
