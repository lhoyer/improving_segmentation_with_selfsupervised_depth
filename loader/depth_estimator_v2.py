import os
import shutil
from copy import deepcopy

import yaml

import train
from configs.machine_config import MachineConfig
from inference_sde import SDEInference
from models.utils import download_model_if_doesnt_exist


class DepthEstimatorV2:
    def __init__(self, host_cfg, dataset_key):
        host_cfg = deepcopy(host_cfg)
        model_checkpoint = host_cfg[dataset_key]["sde_ckpt"]
        save_dir = host_cfg[dataset_key]["generated_depth_dir"]
        assert save_dir is not None

        if DepthEstimatorV2.load_archive_if_exists(save_dir):
            return

        if not os.path.isfile(model_checkpoint):
            download_model_if_doesnt_exist(model_checkpoint)
            model_checkpoint = os.path.join(
                MachineConfig.DOWNLOAD_MODEL_DIR, model_checkpoint,
                "best_model_without_opt.pkl")
            assert os.path.isfile(model_checkpoint), f"{model_checkpoint}"

        cfg_path = os.path.join(model_checkpoint.rsplit("/", 1)[0], "cfg.yml")
        with open(cfg_path) as fp:
            cfg = yaml.safe_load(fp)
        cfg['machine'] = host_cfg["machine"]

        cfg['training']['n_workers'] = 6
        cfg['training']['batch_size'] = 16
        cfg['data']['num_val_samples'] = None
        cfg['data']['augmentations'] = None
        cfg['data']['only_sequences_with_segmentation'] = True
        cfg['data']['restrict_to_subset'] = None
        cfg['data']['load_preprocessed'] = cfg['data'].get('load_preprocessed', True)
        cfg['monodepth_options']['crop_h'] = None
        cfg['monodepth_options']['crop_w'] = None
        cfg['model']['sde_ckpt'] = model_checkpoint
        cfg['training']['log_path'] = save_dir

        # Update data paths
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
        train.expand_cfg_vars(cfg)

        inference = SDEInference(cfg, "raw")
        inference.predict()

        DepthEstimatorV2.make_archive(save_dir)

    @staticmethod
    def load_archive_if_exists(save_dir):
        if MachineConfig.GENERATED_DEPTH_ARCHIVE_DIR is None:
            return False
        archive_file = DepthEstimatorV2.archive_name(save_dir)
        print("Check", archive_file)
        if not os.path.isfile(archive_file):
            return False
        print("Load", archive_file)
        if not os.path.isdir(save_dir):
            shutil.unpack_archive(archive_file, save_dir)
        return True

    @staticmethod
    def make_archive(save_dir):
        if MachineConfig.GENERATED_DEPTH_ARCHIVE_DIR is None:
            return False
        archive_file = DepthEstimatorV2.archive_name(save_dir)
        print("Save", archive_file)
        if os.path.isfile(archive_file):
            return False
        shutil.make_archive(archive_file.replace(".tar", ""), format="tar", root_dir=save_dir)

    @staticmethod
    def sde_name(checkpoint_path):
        if "/" not in checkpoint_path:
            return checkpoint_path
        return f'{checkpoint_path.rsplit("/", 2)[-2].replace("=", "-")}'

    @staticmethod
    def archive_name(save_dir):
        return save_dir.replace(MachineConfig.GENERATED_DEPTH_DIR, MachineConfig.GENERATED_DEPTH_ARCHIVE_DIR).rstrip("/") + ".tar"
