import argparse
import os
import random
import time
from copy import deepcopy
from datetime import datetime

import numpy as np
import psutil
import torch
import torch.nn.functional as F
import yaml
from matplotlib import pyplot as plt
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter  # causes omp infos in contrast to tensorboardX
from tqdm import tqdm

import loader.depth_estimator_v2 as dev2
from configs.machine_config import MachineConfig
from evaluation.metrics import runningScore, AverageMeter, AverageMeterDict
from loader import build_loader, transformsgpu, transformmasks
from loader.depth_estimator import DepthEstimator
from loader.matching_geometry_loader import MatchingGeometryDataset
from loader.transformmasks import get_class_mask
from loader.transformsgpu import strongTransformOneMix
from loss import get_segmentation_loss_function, get_monodepth_loss
from loss.loss import cross_entropy2d, berhu
from models import get_model
from models.joint_segmentation_depth_decoder import PAD
from utils.early_stopping import EarlyStopping
from utils.optimizers import get_optimizer
from utils.schedulers import get_scheduler
from utils.utils import get_logger, gen_code_archive
from utils.visualization import subplotimg


def get_lr(optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        lrs.append(param_group['lr'])
    return lrs


def extract_param_dict(model):
    param_dict = {}
    has_pad = False
    for k, v in model.models.items():
        if isinstance(v, PAD):
            param_dict["segmentation"] = v.segmentation_params()
            param_dict["depth"] = v.depth_params()
            has_pad = True
        elif not (has_pad and k in ["depth", "segmentation"]):
            param_dict[k] = v.parameters()
    return param_dict


def get_params(model, submodules):
    all_params = extract_param_dict(model)
    requested_params = []
    for sm in submodules:
        assert sm in all_params.keys(), f"{sm} not in {all_params.keys()}"
    for k, v in all_params.items():
        if k in submodules:
            requested_params.extend(v)
    return requested_params


def get_train_params(model, cfg):
    train_params = []
    remaining_params = extract_param_dict(model)
    if "backbone_lr" in cfg["training"]["optimizer"]:
        train_params.append(
            {'params': remaining_params["encoder"], 'lr': cfg["training"]["optimizer"]["backbone_lr"]}
        )
        remaining_params.pop("encoder")
    if "pose_lr" in cfg["training"]["optimizer"] and "pose_encoder" in model.models:
        train_params.append(
            {'params': [*remaining_params["pose_encoder"], *remaining_params["pose"]],
             'lr': cfg["training"]["optimizer"]["pose_lr"]}
        )
        remaining_params.pop("pose_encoder")
        remaining_params.pop("pose")
    if "depth_lr" in cfg["training"]["optimizer"]:
        train_params.append(
            {'params': remaining_params["depth"], 'lr': cfg["training"]["optimizer"]["depth_lr"]}
        )
        remaining_params.pop("depth")
    if "segmentation_lr" in cfg["training"]["optimizer"]:
        train_params.append(
            {'params': remaining_params["segmentation"], 'lr': cfg["training"]["optimizer"]["segmentation_lr"]}
        )
        remaining_params.pop("segmentation")
    if len(train_params) >= 1:
        p = []
        for v in remaining_params.values():
            p.extend(v)
        train_params.append(
            {'params': p}
        )
    else:
        train_params = model.parameters()
    return train_params


def setup_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def infinite_iterator(generator):
    while True:
        for data in generator:
            yield data


def current_val_interval(cfg, step):
    v_intervals = [(int(k), int(v)) for k, v in cfg["training"]["val_interval"].items()]
    for k, v in sorted(v_intervals, reverse=True):
        if step > k:
            return v


def extract_ema_params(model, ema_model, model_names):
    relevant_params = []
    relevant_ema_params = []
    for k, v in model.models.items():
        if k in model_names:
            relevant_params.extend(v.parameters())
    for k, v in ema_model.models.items():
        if k in model_names:
            relevant_ema_params.extend(v.parameters())

    return relevant_params, relevant_ema_params


def _colorize(img, cmap, mask_zero=False, max_percentile=80):
    img = img.detach().cpu().numpy()
    # img = torch.log(img.to(torch.float32) + 1).detach().cpu().numpy()
    vmin = np.min(img)
    if max_percentile == 100:
        vmax = np.max(img)
    else:
        vmax = np.percentile(img, max_percentile)
    mask = (img <= 0).squeeze()
    cm = plt.get_cmap(cmap)
    colored_image = cm(np.clip(img.squeeze(), vmin, vmax) / vmax)[:, :, :3]
    # Use white if no depth is available (<= 0)
    if mask_zero:
        colored_image[mask, :] = [1, 1, 1]
    return colored_image


def prepare_config_logic(cfg):
    # Copy shared config fields
    if "monodepth_options" in cfg:
        cfg["data"].update(cfg["monodepth_options"])
        cfg["model"].update(cfg["monodepth_options"])
        cfg["training"]["monodepth_loss"].update(cfg["monodepth_options"])
        if "source_data" in cfg:
            cfg["source_data"].update(cfg["monodepth_options"])
    if "generated_depth_dir" in cfg["data"]:
        dataset_name = f"{cfg['data']['dataset']}_" \
                       f"{cfg['data']['width']}x{cfg['data']['height']}"
        depth_teacher = cfg["data"].get("depth_teacher", None)
        depth_estimator_weights = cfg["model"].get("depth_estimator_weights", None)
        generated_depth_defined = False
        if depth_teacher is not None:
            cfg["data"]["generated_depth_dir"] += dataset_name + "/" + depth_teacher + "/"
            generated_depth_defined = True
        if depth_estimator_weights is not None:
            assert not generated_depth_defined
            cfg["data"]["generated_depth_dir"] += dataset_name + "/" + cfg['model']['depth_estimator_weights'] + "/"
            generated_depth_defined = True
        if cfg["data"].get("sde_ckpt") is not None:
            assert not generated_depth_defined
            cfg["data"]["generated_depth_dir"] += f'{cfg["data"]["dataset"]}_' \
                                                  f'{dev2.DepthEstimatorV2.sde_name(cfg["data"]["sde_ckpt"])}/'
            generated_depth_defined = True
        if not generated_depth_defined:
            cfg["data"]["generated_depth_dir"] = None
        # Source depth
        if "source_data" in cfg and cfg["source_data"].get("sde_ckpt") is not None:
            cfg["source_data"]["generated_depth_dir"] += f'{cfg["source_data"]["dataset"]}_' \
                                                         f'{dev2.DepthEstimatorV2.sde_name(cfg["source_data"]["sde_ckpt"])}/'
        else:
            cfg["source_data"]["generated_depth_dir"] = None
    if cfg["data"]["dataset_seed"] == "same":
        cfg["data"]["dataset_seed"] = cfg["seed"]
    if cfg["training"].get("lr_schedule") is not None and cfg["training"]["lr_schedule"]["name"] == "poly_lr_2" and \
            cfg["training"]["lr_schedule"].get("max_iter") is None:
        cfg["training"]["lr_schedule"]["max_iter"] = cfg["training"]["train_iters"]

    return cfg

class Trainer():
    def __init__(self, cfg, writer, img_writer, logger, run_id):
        cfg = prepare_config_logic(cfg)
        setup_seeds(cfg.get("seed", 1337))

        # Setup device
        torch.backends.cudnn.benchmark = cfg["training"].get("benchmark", True)
        torch.backends.cudnn.deterministic = cfg["training"].get("deterministic", False)
        self.cfg = cfg
        MachineConfig.GLOBAL_TRAIN_CFG = self.cfg
        self.writer = writer
        self.img_writer = img_writer
        self.logger = logger
        self.run_id = run_id
        self.mIoU = 0
        self.fwAcc = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.setup_segmentation_unlabeled()
        self.use_source_data = self.cfg.get("source_data") is not None
        self.use_matching_geom = self.cfg.get("matching_geometry", {}).get("enabled", False)

        self.unlabeled_require_depth = (self.cfg["training"]["unlabeled_segmentation"] is not None and
                                        (self.cfg["training"]["unlabeled_segmentation"]["mix_mask"] == "depth" or
                                         self.cfg["training"]["unlabeled_segmentation"]["mix_mask"] == "depthcomp"))

        # Prepare depth estimates
        do_precalculate_depth = self.cfg["training"]["segmentation_lambda"] != 0 and self.unlabeled_require_depth or \
                                self.cfg["training"]["pseudo_depth_lambda"] or \
                                self.use_matching_geom

        use_depth_teacher = cfg["data"].get("depth_teacher", None) is not None
        if do_precalculate_depth or use_depth_teacher:
            assert not (do_precalculate_depth and use_depth_teacher)
            if not self.cfg["training"].get("disable_depth_estimator", False):
                for dataset_key in ["data", "source_data"]:
                    if dataset_key in cfg and cfg[dataset_key].get("sde_ckpt") is not None:
                        print(f"Prepare depth estimates: {dataset_key} sde_ckpt")
                        dev2.DepthEstimatorV2(cfg, dataset_key)
                if cfg["model"].get("depth_estimator_weights") is not None:
                    print("Prepare depth estimates: data depth_estimator_weights")
                    depth_estimator = DepthEstimator(cfg)
                    depth_estimator.prepare_depth_estimates()
                    del depth_estimator
                torch.cuda.empty_cache()
        else:
            self.cfg["data"]["generated_depth_dir"] = None
            if self.use_source_data:
                self.cfg["source_data"]["generated_depth_dir"] = None

        # Setup Dataloader
        load_labels, load_sequence = True, True
        if self.cfg["training"]["monodepth_lambda"] == 0:
            load_sequence = False
        if self.cfg["training"]["segmentation_lambda"] == 0:
            load_labels = False
        train_data_cfg = deepcopy(self.cfg["data"])
        if not do_precalculate_depth and not use_depth_teacher:
            train_data_cfg["generated_depth_dir"] = None
        self.train_loader = build_loader(train_data_cfg, "train", load_labels=load_labels, load_sequence=load_sequence)
        if self.use_source_data:
            source_data_cfg = deepcopy(self.cfg["source_data"])
            # source_data_cfg["load_onehot"] = True
            if not do_precalculate_depth and not use_depth_teacher:
                source_data_cfg["generated_depth_dir"] = None
            # Don't load sequences as there are non for gta
            self.source_loader = build_loader(source_data_cfg, "train", load_labels=load_labels,
                                              load_sequence=False)
        else:
            self.source_loader = None
        if self.cfg["training"].get("minimize_entropy_unlabeled", False) or self.enable_unlabled_segmentation:
            unlabeled_segmentation_cfg = deepcopy(self.cfg["data"])
            if not self.only_unlabeled and self.mix_use_gt:
                unlabeled_segmentation_cfg["load_onehot"] = True
            if self.only_unlabeled:
                unlabeled_segmentation_cfg.update({"load_unlabeled": True, "load_labeled": False})
            elif self.only_labeled:
                unlabeled_segmentation_cfg.update({"load_unlabeled": False, "load_labeled": True})
            else:
                unlabeled_segmentation_cfg.update({"load_unlabeled": True, "load_labeled": True})
            if self.mix_video:
                assert not self.mix_use_gt and not self.only_labeled and not self.only_unlabeled, \
                    "Video sample indices are not compatible with non-video indices."
                unlabeled_segmentation_cfg.update({"only_sequences_with_segmentation": not self.mix_video,
                                                   "restrict_to_subset": None})
            self.unlabeled_loader = build_loader(unlabeled_segmentation_cfg, "train",
                                                 load_labels=load_labels if not self.mix_video else False,
                                                 load_sequence=load_sequence)
        else:
            self.unlabeled_loader = None
        if self.use_matching_geom:
            assert self.enable_unlabled_segmentation and self.use_source_data
            self.mg_loader = MatchingGeometryDataset(self.source_loader, self.unlabeled_loader,
                                                     self.cfg["matching_geometry"])
        else:
            self.mg_loader = None
        self.val_loader = build_loader(self.cfg["data"], "val", load_labels=load_labels,
                                       load_sequence=load_sequence)
        self.n_classes = self.train_loader.n_classes

        # monodepth dataloader settings uses drop_last=True and shuffle=True even for val
        self.train_data_loader, self.unlabeled_data_loader, self.source_data_loader = None, None, None
        if self.train_loader is not None and len(self.train_loader) > 0:
            self.train_data_loader = infinite_iterator(data.DataLoader(
                self.train_loader,
                batch_size=self.cfg["training"]["batch_size"],
                num_workers=self.cfg["training"]["n_workers"],
                shuffle=self.cfg["data"]["shuffle_trainset"],
                pin_memory=True,
                # Setting to false will cause crash at the end of epoch
                drop_last=True,
            ))
        if self.mg_loader is not None:
            self.mg_data_loader = infinite_iterator(data.DataLoader(
                self.mg_loader,
                batch_size=self.cfg["training"]["batch_size"],
                num_workers=self.cfg["training"]["n_workers"],
                shuffle=self.cfg["data"]["shuffle_trainset"],
                pin_memory=True,
                # Setting to false will cause crash at the end of epoch
                drop_last=True,
            ))
        else:
            if self.source_loader is not None:
                self.source_data_loader = infinite_iterator(data.DataLoader(
                    self.source_loader,
                    batch_size=self.cfg["training"]["batch_size"],
                    num_workers=self.cfg["training"]["n_workers"],
                    shuffle=self.cfg["data"]["shuffle_trainset"],
                    pin_memory=True,
                    # Setting to false will cause crash at the end of epoch
                    drop_last=True,
                ))
            if self.unlabeled_loader is not None:
                self.unlabeled_data_loader = infinite_iterator(data.DataLoader(
                    self.unlabeled_loader,
                    batch_size=self.cfg["training"]["batch_size"],
                    num_workers=self.cfg["training"]["n_workers"],
                    shuffle=self.cfg["data"]["shuffle_trainset"],
                    pin_memory=True,
                    # Setting to false will cause crash at the end of epoch
                    drop_last=True,
                ))

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
        assert not (self.enable_unlabled_segmentation and self.cfg["training"]["save_monodepth_ema"])
        if self.enable_unlabled_segmentation and not self.only_labeled:
            print("Create segmentation ema model.")
            self.ema_model = self.create_ema_model(self.model).to(self.device)
        elif self.cfg["training"]["save_monodepth_ema"]:
            print("Create depth ema model.")
            # TODO: Try to remove unnecessary components and fit into gpu for better performance
            self.ema_model = self.create_ema_model(self.model)  # .to(self.device)
        else:
            self.ema_model = None

        # Setup optimizer, lr_scheduler and loss function
        optimizer_cls = get_optimizer(cfg)
        optimizer_params = {k: v for k, v in cfg["training"]["optimizer"].items() if
                            k not in ["name", "backbone_lr", "pose_lr", "depth_lr", "segmentation_lr"]}
        train_params = get_train_params(self.model, self.cfg)
        self.optimizer = optimizer_cls(train_params, **optimizer_params)

        self.scheduler = get_scheduler(self.optimizer, self.cfg["training"]["lr_schedule"])

        # Creates a GradScaler once at the beginning of training.
        self.scaler = GradScaler(enabled=self.cfg["training"]["amp"])

        self.loss_fn = get_segmentation_loss_function(self.cfg)
        self.monodepth_loss_calculator_train = get_monodepth_loss(self.cfg, is_train=True)
        self.monodepth_loss_calculator_val = get_monodepth_loss(self.cfg, is_train=False, batch_size=self.val_batch_size)

        if cfg["training"]["early_stopping"] is None:
            logger.info("Using No Early Stopping")
            self.earlyStopping = None
        else:
            self.earlyStopping = EarlyStopping(
                patience=round(cfg["training"]["early_stopping"]["patience"] / cfg["training"]["val_interval"]),
                min_delta=cfg["training"]["early_stopping"]["min_delta"],
                cumulative_delta=cfg["training"]["early_stopping"]["cum_delta"],
                logger=logger
            )

    def extract_monodepth_ema_params(self, model, ema_model):
        model_names = ["depth"]
        if not self.cfg["model"]["freeze_backbone"]:
            model_names.append("encoder")
        elif "unfreeze_backbone_iter" in self.cfg["model"]:
            raise NotImplementedError

        return extract_ema_params(model, ema_model, model_names)

    def extract_pad_ema_params(self, model, ema_model):
        model_names = ["depth", "encoder", "mtl_decoder"]
        return extract_ema_params(model, ema_model, model_names)

    def create_ema_model(self, model):
        ema_cfg = deepcopy(self.cfg["model"])
        ema_cfg["disable_pose"] = True
        ema_model = get_model(ema_cfg, self.n_classes)
        for param in ema_model.parameters():
            param.detach_()
        if self.cfg["training"]["save_monodepth_ema"]:
            mp, mcp = self.extract_monodepth_ema_params(model, ema_model)
        elif self.cfg['model']['segmentation_name'] == 'mtl_pad':
            mp, mcp = self.extract_pad_ema_params(model, ema_model)
        else:
            mp, mcp = list(model.parameters()), list(ema_model.parameters())
        assert len(mp) == len(mcp), f"len(mp)={len(mp)}; len(mcp)={len(mcp)}"
        n = len(mp)
        for i in range(0, n):
            mcp[i].data[:] = mp[i].to(mcp[i].device, non_blocking=True).data[:].clone()
        return ema_model

    def update_ema_variables(self, ema_model, model, alpha_teacher, iteration):
        if self.cfg["training"]["save_monodepth_ema"]:
            model_params, ema_params = self.extract_monodepth_ema_params(model, ema_model)
        elif self.cfg['model']['segmentation_name'] == 'mtl_pad':
            model_params, ema_params = self.extract_pad_ema_params(model, ema_model)
        else:
            model_params, ema_params = model.parameters(), ema_model.parameters()
        # Use the "true" average until the exponential average is more correct
        alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
        for ema_param, param in zip(ema_params, model_params):
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + \
                                (1 - alpha_teacher) * param.to(ema_param.device, non_blocking=True)[:].data[:]
        return ema_model

    def save_resume(self, step):
        if self.ema_model is not None:
            raise NotImplementedError("ema model not supported")
        state = {
            "epoch": step + 1,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_iou": self.best_iou,
        }
        save_path = os.path.join(
            self.writer.file_writer.get_logdir(),
            "best_model.pkl"
        )
        torch.save(state, save_path)
        return save_path

    def save_monodepth_models(self):
        if self.cfg["training"]["save_monodepth_ema"]:
            print("Save ema monodepth models.")
            assert self.ema_model is not None
            model_to_save = self.ema_model
        else:
            model_to_save = self.model
        models = ["depth", "pose_encoder", "pose"]
        if not self.cfg["model"]["freeze_backbone"] or "unfreeze_backbone_iter" in self.cfg["model"]:
            models.append("encoder")
        for model_name in models:
            save_path = os.path.join(self.writer.file_writer.get_logdir(), "{}.pth".format(model_name))
            to_save = model_to_save.models[model_name].state_dict()
            torch.save(to_save, save_path)

    def load_resume(self, strict=True, load_model_only=False):
        if os.path.isfile(self.cfg["training"]["resume"]):
            self.logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(self.cfg["training"]["resume"])
            )
            checkpoint = torch.load(self.cfg["training"]["resume"])
            self.model.load_state_dict(checkpoint["model_state"], strict=strict)
            if not load_model_only:
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            self.start_iter = checkpoint["epoch"]
            self.best_iou = checkpoint["best_iou"]
            self.logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    self.cfg["training"]["resume"], checkpoint["epoch"]
                )
            )
        else:
            self.logger.info("No checkpoint found at '{}'".format(self.cfg["training"]["resume"]))

    def tensorboard_training_images(self):
        num_saved = 0
        if self.cfg["training"]["n_tensorboard_trainimgs"] == 0:
            return
        for inputs in self.train_data_loader:
            images = inputs[("color_aug", 0, 0)]
            labels = inputs["lbl"]
            for img, label in zip(images.numpy(), labels.numpy()):
                if num_saved < self.cfg["training"]["n_tensorboard_trainimgs"]:
                    num_saved += 1
                    self.img_writer.add_image(
                        "trainset_{}/{}_0image".format(self.run_id.replace('/', '_'), num_saved), img,
                        global_step=0)
                    colored_image = self.val_loader.decode_segmap_tocolor(label)
                    self.img_writer.add_image(
                        "trainset_{}/{}_1ground_truth".format(self.run_id.replace('/', '_'), num_saved),
                        colored_image,
                        global_step=0, dataformats="HWC")
            if num_saved >= self.cfg["training"]["n_tensorboard_trainimgs"]:
                break

    def _train_model_part(self, model_parts, train_grad=None, train_bn=None):
        for model_part in model_parts:
            if self.cfg["model"]["disable_monodepth"] and model_part in ["depth", "pose", "pose_encoder"] or \
                    self.cfg["model"]["disable_pose"] and model_part in ["pose", "pose_encoder"]:
                continue
            if train_grad is not None:
                for param in self.model.models[model_part].parameters():
                    param.requires_grad = train_grad
            if train_bn is not None:
                for m in self.model.models[model_part].modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.train(train_bn)

    def train_step_segmentation(self, inputs, outputs, retain_graph=False):
        # Train segmentation
        segmentation_loss = torch.tensor(0)
        if self.cfg["training"]["segmentation_lambda"] > 0:
            with autocast(enabled=self.cfg["training"]["amp"]):
                segmentation_loss = self.loss_fn(input=outputs["semantics"], target=inputs["lbl"])
                if "intermediate_semantics" in outputs:
                    segmentation_loss += self.loss_fn(input=outputs["intermediate_semantics"],
                                                      target=inputs["lbl"])
                    segmentation_loss /= 2
                segmentation_loss *= self.cfg["training"]["segmentation_lambda"]
        return segmentation_loss

    def train_step_monodepth(self, inputs, outputs, retain_graph=False):
        mono_loss = torch.tensor(0)
        feat_dist_loss = torch.tensor(0)
        if self.cfg["training"]["monodepth_lambda"] > 0:
            for k, v in outputs.items():
                if "depth" in k or "cam_T_cam" in k:
                    outputs[k] = v.to(torch.float32)
            self.monodepth_loss_calculator_train.generate_images_pred(inputs, outputs)
            mono_losses = self.monodepth_loss_calculator_train.compute_losses(inputs, outputs)
            mono_lambda = self.cfg["training"]["monodepth_lambda"]
            mono_loss = mono_lambda * mono_losses["loss"]
            feat_dist_lambda = self.cfg["training"]["feat_dist_lambda"]
            if feat_dist_lambda > 0:
                feat_dist = torch.dist(outputs["encoder_features"], outputs["imnet_features"], p=2)
                feat_dist_loss = feat_dist_lambda * feat_dist
        return mono_loss, feat_dist_loss

    def train_step_pseudo_depth(self, inputs, outputs, is_source, retain_graph=False):
        pseudo_depth_loss = torch.tensor(0)
        if self.cfg["training"].get("pseudo_depth_lambda", 0) > 0:
            # Crop away bottom of image with own car
            disp_key = "src_disp" if is_source else "disp"
            inter_disp_key = "src_inter_disp" if is_source else "inter_disp"
            with torch.no_grad():
                depth_loss_mask = torch.ones(outputs[disp_key, 0].shape, device=self.device)
                depth_loss_mask[:, :, int(outputs[disp_key, 0].shape[2] * 0.9):, :] = 0

            pseudo_depth_loss = berhu(outputs[disp_key, 0], inputs["pseudo_depth"], depth_loss_mask)
            if (inter_disp_key, 0) in outputs:
                pseudo_depth_loss += berhu(outputs[inter_disp_key, 0], inputs["pseudo_depth"], depth_loss_mask)
                pseudo_depth_loss /= 2
            pseudo_depth_loss *= self.cfg["training"]["pseudo_depth_lambda"]
        return pseudo_depth_loss

    def train_step(self, step):
        self.model.train()
        if self.ema_model is not None:
            self.ema_model.train()

        inputs, source_inputs, unlabeled_inputs = None, None, None
        if self.train_data_loader is not None:
            inputs = self.train_data_loader.__next__()
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(self.device, non_blocking=True)

        if self.mg_loader is not None:
            self.debug("Use matching geometry loader.")
            source_inputs, unlabeled_inputs = self.mg_data_loader.__next__()
        else:
            if self.use_source_data:
                source_inputs = self.source_data_loader.__next__()
            if self.enable_unlabled_segmentation:
                unlabeled_inputs = self.unlabeled_data_loader.__next__()

        if self.use_source_data:
            for k, v in source_inputs.items():
                if torch.is_tensor(v):
                    source_inputs[k] = v.to(self.device, non_blocking=True)

        if self.enable_unlabled_segmentation:
            for k in unlabeled_inputs.keys():
                if "color_aug" in k or "K" in k or "inv_K" in k or "color" in k or k in ["onehot_lbl", "pseudo_depth"]:
                    # print(f"Move {k} to gpu.")
                    unlabeled_inputs[k] = unlabeled_inputs[k].to(self.device, non_blocking=True)

        self.optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=self.device)

        if self.cfg["model"].get("freeze_backbone_bn", False):
            self._train_model_part(["encoder"], train_bn=False)
        if "unfreeze_backbone_iter" in self.cfg["model"] and \
                step == self.cfg["model"]["unfreeze_backbone_iter"]:
            print("Unfreeze encoder.")
            self._train_model_part(["encoder"], train_grad=True)
        if self.cfg["model"].get("freeze_pose_bn", False):
            self._train_model_part(["pose_encoder","pose"], train_bn=False)

        if inputs is not None:
            self.debug("Train on labeled target inputs")
            with autocast(enabled=self.cfg["training"]["amp"]):
                outputs = self.model(inputs)

            mono_loss, feat_dist_loss = self.train_step_monodepth(inputs, outputs, retain_graph=True)
            pseudo_depth_loss = self.train_step_pseudo_depth(inputs, outputs, retain_graph=False, is_source=False)
            segmentation_loss = self.train_step_segmentation(inputs, outputs, retain_graph=False)
            train_total_loss = mono_loss + feat_dist_loss + pseudo_depth_loss + segmentation_loss
            if train_total_loss != 0:
                self.scaler.scale(train_total_loss).backward()
            total_loss += train_total_loss.detach()
            del inputs, outputs  # Free significant GPU memory
        else:
            mono_loss, feat_dist_loss = torch.tensor(0), torch.tensor(0)
            pseudo_depth_loss, segmentation_loss = torch.tensor(0), torch.tensor(0)

        if self.use_source_data:
            self.debug("Train on source data")
            with autocast(enabled=self.cfg["training"]["amp"]):
                source_outputs = self.model(source_inputs)
            source_mono_loss, source_feat_dist_loss = torch.tensor(0), torch.tensor(0)
            source_pseudo_depth_loss = self.train_step_pseudo_depth(
                source_inputs, source_outputs, retain_graph=True, is_source=True)
            source_segmentation_loss = self.train_step_segmentation(
                source_inputs, source_outputs, retain_graph=False)
            self.scaler.scale(source_mono_loss + source_feat_dist_loss + source_pseudo_depth_loss +
                              source_segmentation_loss).backward()
            total_loss += (source_mono_loss + source_feat_dist_loss + source_pseudo_depth_loss +
                           source_segmentation_loss).detach()
        else:
            source_mono_loss, source_feat_dist_loss = torch.tensor(0), torch.tensor(0)
            source_pseudo_depth_loss, source_segmentation_loss = torch.tensor(0), torch.tensor(0)

        unlabeled_loss = torch.tensor(0.0, device=self.device)
        if self.cfg["training"]["segmentation_lambda"] > 0 and self.enable_unlabled_segmentation:
            self.debug("Train on unlabeled data")
            if self.inter_domain_mix:
                self.debug("Inter domain mix")
                unlabeled_loss += self.train_step_domain_mixing(source_inputs=source_inputs,
                                                                target_inputs=unlabeled_inputs, step=step)
            if self.intra_domain_mix:
                self.debug("Intra domain mix")
                intra_mix_loss = self.train_step_segmentation_unlabeled(unlabeled_inputs, step)
                unlabeled_loss += intra_mix_loss[0]
                assert intra_mix_loss[1] == 0
            total_loss += unlabeled_loss.detach()

        if self.cfg["training"].get("clip_grad_norm") is not None:
            self.debug("Clip gradient norm")
            # Unscales the gradients of optimizer's assigned params in-place
            self.scaler.unscale_(self.optimizer)
            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            if self.cfg["training"].get("disable_depth_grad_clip", False):
                torch.nn.utils.clip_grad_norm_(get_params(self.model, ["encoder", "segmentation"]),
                                               self.cfg["training"]["clip_grad_norm"])
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["training"]["clip_grad_norm"])
        # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
        # although it still skips optimizer.step() if the gradients contain infs or NaNs.
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(metrics=self.mIoU)
        else:
            self.scheduler.step()

        # update Mean teacher network
        if self.ema_model is not None:
            self.ema_model = self.update_ema_variables(ema_model=self.ema_model, model=self.model,
                                                       alpha_teacher=0.99, iteration=step)

        return {
            'segmentation_loss': segmentation_loss.detach(),
            'mono_loss': mono_loss.detach(),
            'pseudo_depth_loss': pseudo_depth_loss.detach(),
            'feat_dist_loss': feat_dist_loss.detach(),
            'source_segmentation_loss': source_segmentation_loss.detach(),
            'source_mono_loss': source_mono_loss.detach(),
            'source_pseudo_depth_loss': source_pseudo_depth_loss.detach(),
            'source_feat_dist_loss': source_feat_dist_loss.detach(),
            'unlabeled_loss': unlabeled_loss.detach(),
            'total_loss': total_loss.detach(),
        }

    def setup_segmentation_unlabeled(self):
        if self.cfg["training"].get("unlabeled_segmentation", None) is None:
            self.enable_unlabled_segmentation = False
            return
        unlabeled_cfg = self.cfg["training"]["unlabeled_segmentation"]
        self.enable_unlabled_segmentation = True
        self.intra_domain_mix = unlabeled_cfg.get("intra_domain_mix", True)
        self.inter_domain_mix = unlabeled_cfg.get("inter_domain_mix", False)
        self.consistency_weight = unlabeled_cfg.get("consistency_weight", 1.0)
        self.mix_mask = unlabeled_cfg.get("mix_mask", None)
        self.unlabeled_color_jitter = unlabeled_cfg.get("color_jitter", True)
        self.unlabeled_blur = unlabeled_cfg.get("blur", True)
        self.only_unlabeled = unlabeled_cfg.get("only_unlabeled", True)
        self.only_labeled = unlabeled_cfg.get("only_labeled", False)
        self.mix_video = unlabeled_cfg.get("mix_video", False)
        assert not (self.only_unlabeled and self.only_labeled)
        self.mix_use_gt = unlabeled_cfg.get("mix_use_gt", False)
        self.unlabeled_debug_imgs = unlabeled_cfg.get("debug_images", False)
        self.depthcomp_margin = unlabeled_cfg["depthcomp_margin"]
        self.depthcomp_margin_ie_random_flip = unlabeled_cfg.get("depthcomp_margin_ie_random_flip", True)
        self.depthcomp_foreground_threshold = unlabeled_cfg.get("depthcomp_foreground_threshold", 0)
        self.unlabeled_backward_first_pseudo_label = unlabeled_cfg.get("backward_first_pseudo_label", False)
        self.depthmix_online_depth = unlabeled_cfg.get("depthmix_online_depth", False)

    def train_step_domain_mixing(self, source_inputs, target_inputs, step):
        source_imgs = source_inputs[("color_aug", 0, 0)]
        target_imgs = target_inputs[("color_aug", 0, 0)]

        # First Step: Run teacher to generate pseudo labels
        self.ema_model.use_pose_net = False
        logits_u_w = self.ema_model(target_inputs)["semantics"]
        max_probs, targets_u_w = torch.max(torch.softmax(logits_u_w.detach(), dim=1), dim=1)
        if self.mix_use_gt:
            with torch.no_grad():
                for i in range(target_imgs.shape[0]):
                    # .data is necessary to access truth value of tensor
                    if target_inputs["is_labeled"][i].data:
                        targets_u_w[i] = target_inputs["lbl"][i]
                        max_probs[i] = torch.ones_like(max_probs[i])

        # Third Step: Run Mix
        MixMask = []
        if self.mix_mask == "class":
            MixMask.append(get_class_mask(source_inputs["lbl"][0]))
            MixMask.append(get_class_mask(source_inputs["lbl"][1]))
        elif self.mix_mask == "depthcomp":
            assert self.cfg["training"]["batch_size"] == 2
            for image_i in range(2):
                own_disp = source_inputs["pseudo_depth"][image_i]
                other_disp = target_inputs["pseudo_depth"][image_i]
                local_depthcomp_margin = self.depthcomp_margin
                if self.depthcomp_margin_ie_random_flip and random.random() > 0.5:
                    local_depthcomp_margin *= -1
                # Margin avoids too much of mixing road with same depth
                foreground_mask = torch.ge(own_disp, other_disp - local_depthcomp_margin).long()
                # Avoid hiding the real background of the other image with own a bit closer background
                if isinstance(self.depthcomp_foreground_threshold, tuple) or isinstance(
                        self.depthcomp_foreground_threshold, list):
                    ft_l, ft_u = self.depthcomp_foreground_threshold
                    assert ft_u > ft_l
                    ft = torch.rand(1, device=own_disp.device) * (ft_u - ft_l) + ft_l
                else:
                    ft = self.depthcomp_foreground_threshold
                foreground_mask *= torch.ge(own_disp, ft).long()
                MixMask.append(foreground_mask.unsqueeze(0))
        else:
            raise NotImplementedError(self.mix_mask)

        inputs_u_s = [None, None]
        targets_u = [None, None]
        strong_parameters = {
            "ColorJitter": random.uniform(0, 1) if self.unlabeled_color_jitter else 0,
            "GaussianBlur": random.uniform(0, 1) if self.unlabeled_blur else 0,
        }
        for i in range(2):
            strong_parameters["Mix"] = MixMask[i]
            inputs_u_s[i], targets_u[i] = strongTransformOneMix(
                strong_parameters, onemix=True,
                data=torch.cat((source_imgs[i].unsqueeze(0), target_imgs[i].unsqueeze(0))),
                target=torch.cat((source_inputs["lbl"][i].unsqueeze(0), targets_u_w[i].unsqueeze(0)))
            )
        inputs_u_s = torch.cat(inputs_u_s)
        targets_u = torch.cat(targets_u)
        outputs = self.model({("color_aug", 0, 0): inputs_u_s})
        logits_u_s = outputs["semantics"]

        unlabeled_weight = torch.sum(max_probs.ge(0.968).long() == 1).item() / np.size(np.array(targets_u.cpu()))
        pixelWiseWeight = unlabeled_weight * torch.ones(max_probs.shape).cuda()

        onesWeights = torch.ones((pixelWiseWeight.shape)).cuda()
        for i in range(2):
            strong_parameters["Mix"] = MixMask[i]
            _, pixelWiseWeight[i] = strongTransformOneMix(strong_parameters, onemix=True, target=torch.cat(
                (onesWeights[i].unsqueeze(0), pixelWiseWeight[i].unsqueeze(0))))

        L_2 = self.consistency_weight * cross_entropy2d(input=logits_u_s, target=targets_u,
                                                        pixel_weights=pixelWiseWeight)

        self.scaler.scale(L_2).backward()

        if (step + 1) % 1000 == 0:
            out_dir = os.path.join(self.cfg['training']['log_path'], "inter_class_mix_debug")
            os.makedirs(out_dir, exist_ok=True)
            for j in range(2):
                rows, cols = 2, 5
                fig, axs = plt.subplots(
                    rows, cols, figsize=(3 * cols, 3 * rows),
                    gridspec_kw={'hspace': 0.1, 'wspace': 0, 'top': 0.95, 'bottom': 0, 'right': 1, 'left': 0},
                )
                _, pred_u_s = torch.max(logits_u_s, dim=1)
                subplotimg(axs[0][0], source_imgs[j], "Source Image")
                subplotimg(axs[1][0], target_imgs[j], "Target Image")
                subplotimg(axs[0][1], source_inputs["lbl"][j], "Source Seg GT", cmap="cityscapes")
                subplotimg(axs[1][1], targets_u_w[j], "Target Seg (Pseudo) GT", cmap="cityscapes")
                subplotimg(axs[0][2], inputs_u_s[j], "Mixed Image")
                subplotimg(axs[1][2], MixMask[j], "Domain Mask", cmap="gray")
                subplotimg(axs[0][3], pred_u_s[j], "Seg Pred", cmap="cityscapes")
                subplotimg(axs[1][3], targets_u[j], "Seg Targ", cmap="cityscapes")
                for ax in axs.flat:
                    ax.axis("off")
                plt.savefig(os.path.join(out_dir, f'{(step + 1):06d}_{j}.png'))
                plt.close()

        return L_2

    def generate_mix_mask(self, mode, argmax_u_w, unlabeled_imgs, depths):
        if mode == "class":
            for image_i in range(self.cfg["training"]["batch_size"]):
                classes = torch.unique(argmax_u_w[image_i])
                classes = classes[classes != 250]
                nclasses = classes.shape[0]
                classes = (classes[torch.Tensor(
                    np.random.choice(nclasses, int((nclasses - nclasses % 2) / 2), replace=False)).long()]).cuda()
                if image_i == 0:
                    MixMask = transformmasks.generate_class_mask(argmax_u_w[image_i], classes).unsqueeze(0).cuda()
                else:
                    MixMask = torch.cat(
                        (MixMask, transformmasks.generate_class_mask(argmax_u_w[image_i], classes).unsqueeze(0).cuda()))
        elif self.mix_mask == "depthcomp":
            assert self.cfg["training"]["batch_size"] == 2
            for image_i, other_image_i in [(0, 1), (1, 0)]:
                own_disp = depths[image_i]
                other_disp = depths[other_image_i]
                # Margin avoids too much of mixing road with same depth
                foreground_mask = torch.ge(own_disp, other_disp - self.depthcomp_margin).long()
                # Avoid hiding the real background of the other image with own a bit closer background
                if isinstance(self.depthcomp_foreground_threshold, tuple) or isinstance(
                        self.depthcomp_foreground_threshold, list):
                    ft_l, ft_u = self.depthcomp_foreground_threshold
                    assert ft_u > ft_l
                    ft = torch.rand(1, device=own_disp.device) * (ft_u - ft_l) + ft_l
                else:
                    ft = self.depthcomp_foreground_threshold
                foreground_mask *= torch.ge(own_disp, ft).long()
                if image_i == 0:
                    MixMask = foreground_mask.unsqueeze(0)
                else:
                    MixMask = torch.cat((MixMask, foreground_mask.unsqueeze(0)))
        elif mode is None:
            MixMask = torch.ones((unlabeled_imgs.shape[0], *unlabeled_imgs.shape[2:]), device=self.device)
        else:
            raise NotImplementedError(f"Unknown mix_mask {self.mix_mask}")

        return MixMask

    def calc_pseudo_label_loss(self, teacher_softmax, student_logits):
        max_probs, pseudo_label = torch.max(teacher_softmax, dim=1)
        pseudo_label[max_probs == 0] = self.unlabeled_loader.ignore_index
        unlabeled_weight = torch.sum(max_probs.ge(0.968).long() == 1).item() / np.prod(pseudo_label.shape)
        pixelWiseWeight = unlabeled_weight * torch.ones(max_probs.shape, device=self.device)
        L_u = self.consistency_weight * cross_entropy2d(input=student_logits, target=pseudo_label,
                                                        pixel_weights=pixelWiseWeight)
        return L_u, pseudo_label

    def train_step_segmentation_unlabeled(self, unlabeled_inputs, step):
        def strongTransform(parameters, data=None, target=None):
            assert ((data is not None) or (target is not None))
            data, target = transformsgpu.mix(mask=parameters["Mix"], data=data, target=target)
            data, target = transformsgpu.color_jitter(jitter=parameters["ColorJitter"], data=data, target=target)
            data, target = transformsgpu.gaussian_blur(blur=parameters["GaussianBlur"], data=data, target=None)
            return data, target

        unlabeled_imgs = unlabeled_inputs[("color_aug", 0, 0)]

        # First Step: Run teacher to generate pseudo labels
        self.ema_model.use_pose_net = False
        logits_u_w = self.ema_model(unlabeled_inputs)["semantics"]
        softmax_u_w = torch.softmax(logits_u_w.detach(), dim=1)
        if self.mix_use_gt:
            with torch.no_grad():
                for i in range(unlabeled_imgs.shape[0]):
                    # .data is necessary to access truth value of tensor
                    if unlabeled_inputs["is_labeled"][i].data:
                        softmax_u_w[i] = unlabeled_inputs["onehot_lbl"][i]
        _, argmax_u_w = torch.max(softmax_u_w, dim=1)

        # Second Step: Run student network on unaugmented data to generate depth for DepthMix, calculate monodepth loss,
        # and unaugmented segmentation pseudo label loss
        mono_loss = 0
        L_1 = 0
        if self.depthmix_online_depth:
            outputs_1 = self.model(unlabeled_inputs)
            if self.cfg["training"]["monodepth_lambda"] > 0:
                self.monodepth_loss_calculator_train.generate_images_pred(unlabeled_inputs, outputs_1)
                mono_losses = self.monodepth_loss_calculator_train.compute_losses(unlabeled_inputs, outputs_1)
                mono_lambda = self.cfg["training"]["monodepth_lambda"]
                mono_loss = mono_lambda * mono_losses["loss"]
                self.scaler.scale(mono_loss).backward(retain_graph=self.unlabeled_backward_first_pseudo_label)
                depths = outputs_1[("disp", 0)].detach()
                for j in range(depths.shape[0]):
                    dmin = torch.min(depths[j])
                    dmax = torch.max(depths[j])
                    depths[j] = torch.clamp(depths[j], dmin, dmax)
                    depths[j] = (depths[j] - dmin) / (dmax - dmin)
            else:
                depths = unlabeled_inputs["pseudo_depth"]
            if self.unlabeled_backward_first_pseudo_label:
                logits_1 = outputs_1["semantics"]
                L_1, _ = self.calc_pseudo_label_loss(teacher_softmax=softmax_u_w, student_logits=logits_1)
                self.scaler.scale(L_1).backward()
        elif "pseudo_depth" in unlabeled_inputs:
            depths = unlabeled_inputs["pseudo_depth"]
        else:
            depths = [None] * unlabeled_imgs.shape[0]

        # Third Step: Run Mix
        MixMask = self.generate_mix_mask(self.mix_mask, argmax_u_w, unlabeled_imgs, depths)

        strong_parameters = {
            "Mix": MixMask,
            "ColorJitter": random.uniform(0, 1) if self.unlabeled_color_jitter else 0,
            "GaussianBlur": random.uniform(0, 1) if self.unlabeled_blur else 0,
        }

        inputs_u_s, _ = strongTransform(strong_parameters, data=unlabeled_imgs)
        unlabeled_inputs[("color_aug", 0, 0)] = inputs_u_s
        outputs = self.model(unlabeled_inputs)
        logits_u_s = outputs["semantics"]

        softmax_u_w_mixed, _ = strongTransform(strong_parameters, data=softmax_u_w)
        L_2, pseudo_label = self.calc_pseudo_label_loss(teacher_softmax=softmax_u_w_mixed, student_logits=logits_u_s)
        self.scaler.scale(L_2).backward()

        for j, (f, img, ps_lab, mask, d) in enumerate(
                zip(unlabeled_inputs["filename"], inputs_u_s, pseudo_label, MixMask, depths)):
            if (step + 1) % 1000 != 0:
                continue
            fn = f"{self.cfg['training']['log_path']}/intra_class_mix_debug/{step}_{j}_img.jpg"
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            rows, cols = 2, 2
            fig, axs = plt.subplots(rows, cols, sharex='col', sharey='row',
                                    gridspec_kw={'hspace': 0, 'wspace': 0},
                                    figsize=(4 * cols, 4 * rows))
            axs[0][0].imshow(img.permute(1, 2, 0).cpu().numpy())
            axs[0][1].imshow(mask.float().cpu().numpy(), cmap="gray")
            if d is not None:
                axs[1][1].imshow(d.cpu().numpy(), cmap="plasma")
            axs[1][0].imshow(self.val_loader.decode_segmap_tocolor(ps_lab.cpu().numpy()))
            for ax in axs.flat:
                ax.axis("off")
            plt.savefig(fn)
            plt.close()

        return L_2 + L_1, mono_loss

    def train(self):
        self.start_iter = 0
        self.best_iou = -100.0
        if self.cfg["training"]["resume"] is not None:
            self.load_resume()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.cfg["training"]["optimizer"]["lr"]

        train_loss_meter = AverageMeterDict()
        time_meter = AverageMeter()

        step = self.start_iter
        flag = True

        self.tensorboard_training_images()

        start_ts = time.time()
        while step <= self.cfg["training"]["train_iters"] and flag:
            step += 1
            self.step = step
            losses = self.train_step(step)

            time_meter.update(time.time() - start_ts)
            train_loss_meter.update(losses)

            if (step + 1) % self.cfg["training"]["print_interval"] == 0:
                fmt_str = "Iter [{}/{}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(
                    step + 1,
                    self.cfg["training"]["train_iters"],
                    train_loss_meter.avgs["total_loss"],
                    time_meter.avg / self.cfg["training"]["batch_size"],
                )

                self.logger.info(print_str)
                log_scalars = {}
                for k, v in train_loss_meter.avgs.items():
                    log_scalars["training/" + k] = v
                learning_rates = get_lr(self.optimizer)
                for lr_i, lr in enumerate(learning_rates):
                    log_scalars[f"training/learning_rate_{lr_i}"] = lr
                log_scalars["training/time_per_image"] = time_meter.avg / self.cfg["training"]["batch_size"]
                log_scalars["training/amp_scale"] = self.scaler.get_scale()
                log_scalars["training/memory"] = psutil.virtual_memory().used / 1e9
                for k, v in log_scalars.items():
                    self.writer.add_scalar(k, v, step + 1)
                time_meter.reset()
                train_loss_meter.reset()

            if (step + 1) % current_val_interval(self.cfg, step + 1) == 0 or (step + 1) == self.cfg["training"][
                "train_iters"
            ]:
                self.validate(step)

                if self.mIoU >= self.best_iou:
                    self.best_iou = self.mIoU
                    if self.cfg["training"]["save_model"]:
                        self.save_resume(step)

                if self.cfg["training"]["save_separate_monodepth_models"]:
                    self.save_monodepth_models()

                if self.earlyStopping is not None:
                    if not self.earlyStopping.step(self.mIoU):
                        flag = False
                        break

            if (step + 1) == self.cfg["training"]["train_iters"]:
                flag = False
                break

            start_ts = time.time()

        return step

    def validate(self, step):
        print(f"Validate {self.cfg['name']}")
        self.model.eval()
        val_loss_meter = AverageMeterDict()
        running_metrics_val = runningScore(self.n_classes)
        imgs_to_save = []
        with torch.no_grad():
            for inputs_val in tqdm(self.val_data_loader,
                                   total=len(self.val_data_loader)):
                if self.cfg["model"]["disable_monodepth"]:
                    required_inputs = [("color_aug", 0, 0), "lbl"]
                else:
                    required_inputs = inputs_val.keys()
                for k, v in inputs_val.items():
                    if torch.is_tensor(v) and k in required_inputs:
                        inputs_val[k] = v.to(self.device, non_blocking=True)
                images_val = inputs_val[("color_aug", 0, 0)]
                with autocast(enabled=self.cfg["training"]["amp"]):
                    outputs = self.model(inputs_val)

                if self.cfg["training"]["segmentation_lambda"] > 0:
                    labels_val = inputs_val["lbl"]
                    semantics = outputs["semantics"]
                    val_segmentation_loss = self.loss_fn(input=semantics, target=labels_val)
                    # Handle inconsistent size between input and target
                    n, c, h, w = semantics.size()
                    nt, ht, wt = labels_val.size()
                    if h != ht and w != wt:  # upsample labels
                        semantics = F.interpolate(
                            semantics, size=(ht, wt),
                            mode="bilinear", align_corners=True
                        )
                    pred = semantics.data.max(1)[1].cpu().numpy()
                    gt = labels_val.data.cpu().numpy()

                    running_metrics_val.update(gt, pred)
                else:
                    pred = [None] * images_val.shape[0]
                    gt = [None] * images_val.shape[0]
                    val_segmentation_loss = torch.tensor(0)

                last_scale = self.cfg["monodepth_options"]["scales"][0]
                if not self.cfg["model"]["disable_monodepth"]:
                    if not self.cfg["model"]["disable_pose"]:
                        for k, v in outputs.items():
                            if "depth" in k or "cam_T_cam" in k:
                                outputs[k] = v.to(torch.float32)
                        self.monodepth_loss_calculator_val.generate_images_pred(inputs_val, outputs)
                        mono_losses = self.monodepth_loss_calculator_val.compute_losses(inputs_val, outputs)
                        val_mono_loss = mono_losses["loss"]
                    else:
                        outputs.update(self.model.predict_test_disp(inputs_val))
                        self.monodepth_loss_calculator_val.generate_depth_test_pred(outputs)
                        val_mono_loss = torch.tensor(0)
                        outputs[("color", -1, last_scale)] = [None] * images_val.shape[0]
                else:
                    outputs[("color", -1, last_scale)] = [None] * images_val.shape[0]
                    outputs[f"to_optimise/{last_scale}"] = [None] * images_val.shape[0]
                    val_mono_loss = torch.tensor(0)

                if self.cfg["data"].get("depth_teacher", None) is not None:
                    # Crop away bottom of image with own car
                    with torch.no_grad():
                        depth_loss_mask = torch.ones(outputs["disp", 0].shape, device=self.device)
                        depth_loss_mask[:, :, int(outputs["disp", 0].shape[2] * 0.9):, :] = 0
                    val_pseudo_depth_loss = berhu(outputs["disp", 0], inputs_val["pseudo_depth"], depth_loss_mask,
                                              apply_log=self.cfg["training"].get("pseudo_depth_loss_log", False))
                else:
                    val_pseudo_depth_loss = torch.tensor(0)

                val_loss_meter.update({
                    "segmentation_loss": val_segmentation_loss.detach(),
                    "monodepth_loss": val_mono_loss.detach(),
                    "pseudo_depth_loss": val_pseudo_depth_loss.detach()
                })

                if ("disp", 0) not in outputs:
                    outputs[("disp", 0)] = [None] * images_val.shape[0]
                for i in range(images_val.shape[0]):
                    if len(imgs_to_save) < self.cfg["training"]["n_tensorboard_imgs"]:
                        imgs_to_save.append([
                            images_val[i], gt[i], pred[i],
                            outputs[("disp", 0)][i],
                        ])

        log_metrics = {}
        for k, v in val_loss_meter.avgs.items():
            log_metrics["validation/" + k] = v
        if self.cfg["training"]["segmentation_lambda"] > 0:
            score, class_iou = running_metrics_val.get_scores()
            for k, v in score.items():
                print(k, v)
                log_metrics["val_metrics/{}".format(k)] = v
            for k, v in class_iou.items():
                log_metrics[f"val_metrics/cls_{k}_{self.val_data_loader.dataset.class_names[k + 1]}"] = v
            self.mIoU = score["Mean IoU : \t"]
            self.fwAcc = score["FreqW Acc : \t"]
        if self.cfg["training"]["monodepth_lambda"] > 0:
            print(f"Monodepth Loss: {val_loss_meter.avgs['monodepth_loss']}")
        for k, v in log_metrics.items():
            self.writer.add_scalar(k, v, step + 1)

        for j, imgs in enumerate(imgs_to_save):
            # Only log the first image as they won't change -> save memory
            if (step + 1) // current_val_interval(self.cfg, step + 1) == 1:
                self.img_writer.add_image(
                    "{}/{}_0image".format(self.run_id.replace('/', '_'), j), imgs[0], global_step=step + 1)
                if imgs[1] is not None:
                    colored_image = self.val_loader.decode_segmap_tocolor(imgs[1])
                    self.img_writer.add_image(
                        "{}/{}_1ground_truth".format(self.run_id.replace('/', '_'), j), colored_image,
                        global_step=step + 1, dataformats="HWC")
            if imgs[2] is not None:
                colored_image = self.val_loader.decode_segmap_tocolor(imgs[2])
                self.img_writer.add_image(
                    "{}/{}_2prediction".format(self.run_id.replace('/', '_'), j), colored_image, global_step=step + 1,
                    dataformats="HWC")
            if imgs[3] is not None:
                colored_image = _colorize(imgs[3], "plasma", max_percentile=100)
                self.img_writer.add_image(
                    "{}/{}_3depth".format(self.run_id.replace('/', '_'), j), colored_image, global_step=step + 1,
                    dataformats="HWC")

    def debug(self, str):
        if self.step == 2:
            print(str)


def expand_cfg_vars(cfg):
    for k, v in cfg.items():
        if isinstance(v, dict):
            expand_cfg_vars(cfg[k])
        elif isinstance(v, str):
            if "MachineConfig." in cfg[k]:
                var_name = cfg[k].replace("MachineConfig.", "").split("/")[0]
                cfg[k] = cfg[k].replace(cfg[k].split("/")[0], getattr(MachineConfig, var_name))
            cfg[k] = os.path.expandvars(cfg[k])
    return True


def save_training_meta_data(cfg):
    with open(cfg['training']['log_path'] + "/cfg.yml", 'w') as fp:
        yaml.dump(cfg, fp)
    gen_code_archive(cfg['training']['log_path'], 0)

def train_main(cfg):
    MachineConfig(cfg["machine"])
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")

    if "name" not in cfg:
        cfg["name"] = "test" + run_id
    cfg['training']['log_path'] += cfg["name"]
    name = cfg['name']
    print('Start', name)

    expand_cfg_vars(cfg)

    logdir = cfg['training']['log_path']
    writer = SummaryWriter(log_dir=logdir, filename_suffix='.metrics')
    img_writer = SummaryWriter(log_dir=logdir, filename_suffix='.tensorboardimgs')

    print("RUNDIR: {}".format(logdir))
    save_training_meta_data(cfg)

    logger = get_logger(logdir)
    logger.info("Let the games begin")

    trainer = Trainer(cfg, writer, img_writer, logger, os.path.join(name, str(run_id)))
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/ssda.yml",
        help="Configuration file to use",
    )
    parser.add_argument(
        "--machine",
        type=str,
        default="",
        choices=[*MachineConfig.AVAIL_MACHINES, ""]
    )
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)
    if args.machine != "":
        cfg["machine"] = args.machine
    train_main(cfg)
