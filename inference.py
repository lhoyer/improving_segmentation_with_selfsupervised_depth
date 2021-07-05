import argparse
import os
from datetime import datetime

import torch
import yaml
from torch.cuda.amp import autocast
from torch.utils import data
from torchvision.utils import save_image
from tqdm import tqdm

from configs.machine_config import MachineConfig
from loader import build_loader
from loader.depth_estimator import DepthEstimator
from loss import get_monodepth_loss
from models import get_model
from train import setup_seeds, expand_cfg_vars


class Inference():
    def __init__(self, cfg, logdir, run_id):
        # Copy shared config fields
        if "monodepth_options" in cfg:
            cfg["data"].update(cfg["monodepth_options"])
            cfg["model"].update(cfg["monodepth_options"])
            cfg["training"]["monodepth_loss"].update(cfg["monodepth_options"])
            cfg['model']['depth_args']['max_scale_size'] = (cfg["monodepth_options"]["crop_h"], cfg["monodepth_options"]["crop_w"])

        # Setup seeds
        setup_seeds(cfg.get("seed", 1337))
        if cfg["data"]["dataset_seed"] == "same":
            cfg["data"]["dataset_seed"] = cfg["seed"]

        # Setup device
        torch.backends.cudnn.benchmark = cfg["training"].get("benchmark", True)
        self.cfg = cfg
        self.logdir = logdir
        self.run_id = run_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Prepare depth estimates
        do_precalculate_depth = False
        if do_precalculate_depth:
            print("Prepare depth estimates")
            depth_estimator = DepthEstimator(cfg)
            depth_estimator.prepare_depth_estimates()
            del depth_estimator
            torch.cuda.empty_cache()
        else:
            self.cfg["data"]["generated_depth_dir"] = None

        # Setup Dataloader
        self.val_loader = build_loader(self.cfg["data"], "val", load_labels=False, load_sequence=False)
        self.n_classes = self.val_loader.n_classes

        self.val_batch_size = self.cfg["training"]["val_batch_size"]
        self.val_data_loader = data.DataLoader(
            self.val_loader,
            batch_size=self.val_batch_size,
            num_workers=self.cfg["training"]["n_workers"],
            pin_memory=True,
            # If using a dataset with odd number of samples (CamVid), the memory consumption suddenly increases for the
            # last batch. This can be circumvented by dropping the last batch. Only do that if it is necessary for your
            # system as it will result in an incomplete validation set.
            # drop_last=True,
        )

        # Setup Model
        self.model = get_model(cfg["model"], self.n_classes).to(self.device)
        # print(self.model)

        self.monodepth_loss_calculator_val = get_monodepth_loss(self.cfg, is_train=False, batch_size=self.val_batch_size)

        if self.cfg["training"]["resume"] is not None:
            self.load_resume(strict=False)

    def load_resume(self, strict=True):
        if os.path.isfile(self.cfg["training"]["resume"]):
            checkpoint = torch.load(self.cfg["training"]["resume"])
            self.model.load_state_dict(checkpoint["model_state"], strict=strict)
        else:
            print(f"WARNING: load_resume - {self.cfg['training']['resume']} not found")

    def run(self):
        print(f"Validate {self.cfg['name']}")
        self.model.eval()
        with torch.no_grad():
            for inputs_val in tqdm(self.val_data_loader,
                                   total=len(self.val_data_loader),
                                   disable=False):
                for k, v in inputs_val.items():
                    if torch.is_tensor(v):
                        inputs_val[k] = v.to(self.device, non_blocking=True)
                images_val = inputs_val[("color_aug", 0, 0)]
                with autocast(enabled=self.cfg["training"]["amp"]):
                    outputs = self.model(inputs_val)

                if self.cfg["training"]["segmentation_lambda"] > 0:
                    semantics = outputs["semantics"]
                    pred = semantics.data.max(1)[1].cpu().numpy()
                else:
                    pred = [None] * images_val.shape[0]

                if not self.cfg["model"]["disable_monodepth"]:
                    self.monodepth_loss_calculator_val.generate_depth_test_pred(outputs)
                else:
                    outputs[("disp", 0)] = [None] * images_val.shape[0]

                for filename, img, seg, depth in zip(inputs_val["filename"], images_val, pred, outputs[("disp", 0)]):
                    fn = f"{self.logdir}/{filename}"
                    os.makedirs(os.path.dirname(fn), exist_ok=True)
                    save_image(img, fn)
                    if depth is not None:
                        save_image(depth, fn.replace(".jpg", "_depth.png"))
                    ps_lab_col = torch.tensor(self.val_loader.decode_segmap_tocolor(seg)).permute(2, 0, 1)
                    save_image(ps_lab_col, fn.replace(".jpg", "_label.png"))


def inference_main(cfg):
    MachineConfig(cfg["machine"])
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")

    cfg["name"] = "inference" + run_id
    cfg['training']['log_path'] = os.path.join(cfg["training"]["log_path"], cfg["name"]) + "/"
    name = cfg['name']
    print('Start', name)

    expand_cfg_vars(cfg)

    logdir = cfg['training']['log_path']

    print("RUNDIR: {}".format(logdir))
    os.makedirs(logdir, exist_ok=True)
    with open(logdir + "/cfg.yml", 'w') as fp:
        yaml.dump(cfg, fp)

    inference = Inference(cfg, logdir, os.path.join(name, str(run_id)))
    inference.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--model",
        type=str,
        default="models/cityscapes_sel_rev2_ds_us_pad_transfer_dcompgt0030_D372fixed_S7_sgdLr1E-021E-031E-061E-03stepx_clip10False_m1s1_crop512x512bs2_flip_dec6_lr5_fd2_crop512x512bs4_l9i7Trueos1_Unlab1.0depthcompFPLFalsejitblur/",
        help="Path to model directory containing the model checkpoint pkl file and the cfg.yml"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="MachineConfig.CITYSCAPES_DIR/leftImg8bit_small/val/"
    )
    parser.add_argument(
        "--machine",
        type=str,
        default="",
        choices=[*MachineConfig.AVAIL_MACHINES, ""]
    )
    args = parser.parse_args()
    checkpoint_file = os.path.join(args.model, "best_model_without_opt.pkl")
    cfg_file = os.path.join(args.model, "cfg.yml")
    with open(cfg_file) as fp:
        cfg = yaml.safe_load(fp)

    cfg["machine"] = args.machine
    cfg['data']['dataset'] = "inference"
    cfg['data']['path'] = args.data
    cfg['model']['disable_pose'] = True
    cfg['training']['log_path'] = "MachineConfig.LOG_DIR"
    cfg["training"]["resume"] = checkpoint_file

    inference_main(cfg)
