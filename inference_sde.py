import argparse
import os
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm

import train
from configs.machine_config import MachineConfig
from loader import build_loader
from models import get_model
from models.monodepth_layers import disp_to_depth


class SDEInferenceModel(nn.Module):

    def __init__(self, cfg, n_classes):
        super().__init__()
        self.log_path = cfg["training"]["log_path"]
        self.n_classes = n_classes

        self.source_model = get_model(cfg["model"], n_classes)
        for param in self.source_model.parameters():
            param.requires_grad = False
        if "sde_ckpt" in cfg["model"]:
            checkpoint = torch.load(cfg["model"]["sde_ckpt"])
            self.source_model.load_state_dict(checkpoint["model_state"])

    def forward(self, input):
        self.source_model.eval()
        with torch.no_grad():
            out = self.source_model.predict_test_disp(input)
            disp = out[("disp", 0)]

        return disp

def save_img(args):
    out_filename, out = args
    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    img = Image.fromarray(out, mode="I;16")
    img.save(out_filename)

class SDEInference:
    def __init__(self, cfg, save_mode):
        cfg = train.prepare_config_logic(cfg)
        MachineConfig.GLOBAL_TRAIN_CFG = cfg
        train.setup_seeds(cfg.get("seed", 1337))
        self.cfg = cfg
        self.save_mode = save_mode

        # Setup Dataloader
        self.train_loader = build_loader(self.cfg["data"], "train", load_labels=False, load_sequence=False)
        self.val_loader = build_loader(self.cfg["data"], "val", load_labels=False, load_sequence=False)
        self.loader = data.ConcatDataset([self.train_loader, self.val_loader])
        self.train_data_loader = DataLoader(
            self.train_loader,
            batch_size=self.cfg["training"]["batch_size"],
            num_workers=self.cfg["training"]["n_workers"],
            pin_memory=True,
        )
        self.data_loader = data.DataLoader(
            self.loader,
            batch_size=self.cfg["training"]["batch_size"],
            num_workers=self.cfg["training"]["n_workers"],
            pin_memory=True,
        )

        self.model = SDEInferenceModel(self.cfg, self.train_loader.n_classes).cuda()

    def predict(self):
        self.pool = Pool(self.cfg["training"]["n_workers"])
        for batch in tqdm(self.data_loader):
            batch[("color", 0, 0)] = batch[("color", 0, 0)].cuda()
            preds = self.model(batch)

            if self.save_mode == "raw":
                outs = preds
            elif self.save_mode == "normdisp":
                raise NotImplementedError
            elif self.save_mode == "normdepth":
                scdisp, scdepth = disp_to_depth(preds, self.cfg["training"]["monodepth_loss"]["min_depth"],
                                                self.cfg["training"]["monodepth_loss"]["max_depth"])
                out_max_depth = 10.0
                scdepth = torch.clamp(scdepth, self.cfg["training"]["monodepth_loss"]["min_depth"], out_max_depth)
                outs = scdepth / out_max_depth
            else:
                raise NotImplementedError(self.save_mode)

            outs *= (2 ** 16 - 1)
            outs.squeeze_(1)
            outs = outs.cpu().numpy().astype(np.uint16)

            out_list = []
            name_list = []
            for i in range(outs.shape[0]):
                out_list.append(outs[i])
                name = os.path.join(self.cfg["training"]["log_path"], batch["filename"][i])
                name = name.replace("images_small", "images")
                name = name.replace("RGB_small", "RGB")
                name = name.replace(".jpg", ".png")
                name_list.append(name)

            self.pool.map_async(save_img, zip(name_list, out_list))
        self.pool.close()
        self.pool.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--model",
        nargs="?",
        type=str,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--machine",
        type=str,
        default="",
        choices=[*MachineConfig.AVAIL_MACHINES, ""]
    )
    parser.add_argument(
        "--save",
        type=str,
        default="raw",
        choices=["raw", "normdisp", "normdepth"]
    )

    args = parser.parse_args()
    cfg_path = os.path.join(args.model.rsplit("/", 1)[0], "cfg.yml")
    with open(cfg_path) as fp:
        cfg = yaml.safe_load(fp)
    if args.machine != "":
        cfg["machine"] = args.machine

    # Override inference specific fields
    cfg['exp_name'] = "inference_sde"
    cfg['run_name'] = f"infsde{args.save}"
    cfg['training']['log_path'] = "MachineConfig.LOG_DIR/"
    cfg['training']['n_workers'] = 6
    cfg['training']['batch_size'] = 16
    if cfg['data']['dataset'] == "cityscapes":
        cfg['data']['path'] = "MachineConfig.CITYSCAPES_DIR"
    elif cfg['data']['dataset'] in ["gta", "gtaseq"]:
        cfg['data']['dataset'] = "gtaseg"
        cfg['data']['path'] = "MachineConfig.GTASEG_DIR"
    elif cfg['data']['dataset'] == "synthiaseq":
        cfg['data']['dataset'] = "synthiaseg"
        cfg['data']['path'] = "MachineConfig.SYNTHIA_DIR"
    else:
        raise NotImplementedError(cfg['data']['dataset'])
    cfg['data']['num_val_samples'] = None
    cfg['data']['augmentations'] = None
    cfg['data']['only_sequences_with_segmentation'] = True
    cfg['monodepth_options']['crop_h'] = None
    cfg['monodepth_options']['crop_w'] = None
    cfg['model']['sde_ckpt'] = args.model

    cfg_str = f'{args.model.rsplit("/", 2)[-2]}'
    cfg["run_name"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + cfg["run_name"] + '___' + cfg_str
    cfg['training']['log_path'] = os.path.join(cfg['training']['log_path'], cfg["exp_name"], cfg["run_name"])

    MachineConfig(cfg["machine"])
    train.expand_cfg_vars(cfg)
    os.makedirs(cfg["training"]["log_path"])
    with open(cfg['training']['log_path'] + "/cfg.yml", 'w') as fp:
        yaml.dump(cfg, fp)

    print('Start', cfg["run_name"])
    print('Save logs under', cfg["training"]["log_path"])

    inference = SDEInference(cfg, args.save)
    inference.predict()
