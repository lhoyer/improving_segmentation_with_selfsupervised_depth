import hashlib
import os
import re
import urllib
import zipfile

import torch
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from configs.machine_config import MachineConfig
from models.depth_decoder import DepthDecoder
from models.pose_decoder import PoseDecoder
from models.resnet_encoder import ResnetEncoder
from utils.google_drive_downloader import GoogleDriveDownloader


def get_resnet_backbone(backbone_name, backbone_pretraining="none", replace_stride_with_dilation=None,
                        use_intermediate_layer_getter=False, num_input_images=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if backbone_name in ["resnet18", "resnet50", "resnet101"]:
        match = re.match(r"([a-z]+)([0-9]+)", backbone_name, re.I)
        if match:
            n_res = int(match.groups()[-1])
        else:
            raise ValueError
        if backbone_pretraining == "none":
            backbone = ResnetEncoder(n_res, False, num_input_images=num_input_images,
                                     replace_stride_with_dilation=replace_stride_with_dilation)
        elif backbone_pretraining == "imnet":
            backbone = ResnetEncoder(n_res, True, num_input_images=num_input_images,
                                     replace_stride_with_dilation=replace_stride_with_dilation)
        elif "mono" in backbone_pretraining:
            backbone = ResnetEncoder(n_res, False, num_input_images=num_input_images,
                                     replace_stride_with_dilation=replace_stride_with_dilation)
            print('Load ' + backbone_pretraining + 'weights')
            download_model_if_doesnt_exist(backbone_pretraining)
            encoder_path = os.path.join(MachineConfig.DOWNLOAD_MODEL_DIR, backbone_pretraining, "encoder.pth")
            loaded_dict_enc = torch.load(encoder_path, map_location=torch.device(device))
            filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in backbone.state_dict()}
            backbone.load_state_dict(filtered_dict_enc, strict=False)
        else:
            raise NotImplementedError

        backbone.encoder.avgpool = nn.Identity()
        backbone.encoder.fc = nn.Identity()

        if use_intermediate_layer_getter:
            return_layers = {'layer4': 'out', 'layer3': 'layer3', 'layer2': 'layer2'}
            backbone = IntermediateLayerGetter(backbone.encoder, return_layers=return_layers)
    else:
        raise NotImplementedError

    return backbone


def get_depth_decoder(depth_pretraining, num_ch_enc, scales=range(4), **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    depth_decoder = DepthDecoder(num_ch_enc, scales, **kwargs)
    depth_decoder.to(device)

    if depth_pretraining != 'none':
        print('Load ' + depth_pretraining + 'depth weights')
        download_model_if_doesnt_exist(depth_pretraining)
        model_path = os.path.join(MachineConfig.DOWNLOAD_MODEL_DIR, depth_pretraining, "depth.pth")
        loaded_dict = torch.load(model_path, map_location=torch.device(device))
        filtered_dict = loaded_dict
        # filtered_dict = {k: v for k, v in loaded_dict.items() if k in depth_decoder.state_dict()}
        depth_decoder.load_state_dict(filtered_dict)

    return depth_decoder


def get_posenet(backbone_name, backbone_pretraining, pose_pretraining, num_pose_frames):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    models["pose_encoder"] = get_resnet_backbone(backbone_name=backbone_name,
                                                 backbone_pretraining="imnet" if backbone_pretraining == "imnet" else "none",
                                                 num_input_images=num_pose_frames)
    models["pose"] = PoseDecoder(
        models["pose_encoder"].num_ch_enc,
        num_input_features=1,
        num_frames_to_predict_for=2)

    if "mono" in pose_pretraining:
        for mn in ["pose_encoder", "pose"]:
            if mn not in models:
                continue
            download_model_if_doesnt_exist(pose_pretraining)
            path = os.path.join(MachineConfig.DOWNLOAD_MODEL_DIR, pose_pretraining, "{}.pth".format(mn))
            loaded_dict = torch.load(path, map_location=torch.device(device))
            filtered_dict = {k: v for k, v in loaded_dict.items() if k in models[mn].state_dict()}
            models[mn].load_state_dict(filtered_dict)

    return models


def _get_layer(encoder, decoder, layer):
    if layer <= 4:
        x = encoder[layer]
    else:
        x = decoder[("upconv", 9 - layer)]
    return x


def download_model_if_doesnt_exist(model_name, download_dir=None):
    """If pretrained kitti model doesn't exist, download and unzip it
    """
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "mono_cityscapes_1024x512_r101dil_aspp_dec5":
            ("gdrive_id=1VF86Wqv9x7afLt_B8t2OaWtb-lG0vwyN",
             ""),
        "mono_cityscapes_1024x512_r101dil_aspp_dec6_lr5_fd2":
            ("gdrive_id=1Kki3vwDxCeSdLQI5LLJVwk7erTk6EVkB",
             ""),
        "mono_cityscapes_1024x512_r101dil_aspp_dec6_lr5":
            ("gdrive_id=19rJIafDLyAW348bYE3M_EoQcIK0OIj0V",
             ""),
        "mono_cityscapes_1024x512_r101dil_aspp_dec5_posepretrain_crop512x512bs4":
            ("gdrive_id=1V3qzmCIfErOhLILnwCCchYMkaKLtUA7c",
             ""),
        "mono_cityscapes_1024x512_r101dil_aspp_dec6_lr5_fd2_crop512x512bs4":
            ("gdrive_id=1woRzEPVuhaafrS_2_GlsJuVRyxWaGO4O",
             ""),
        "mono_cityscapes_1024x512_r101dil_aspp_dec6_lr5_fd0_crop512x512bs4":
            ("gdrive_id=1G7bDZ-0PsHeMSHK59EqJn5ncqMzWB1Js",
             ""),
        "mono_cityscapes_1024x512_r101dil_aspp_dec6_lr5_fd2_crop512x512bs2":
            ("gdrive_id=1bHlAYHKSv6sVbQBMlQ-D7kkUcAMb8-Jq",
             ""),
    }
    if download_dir is None:
        download_dir = MachineConfig.DOWNLOAD_MODEL_DIR
        download_dir = os.path.expandvars(download_dir)
        download_dir = download_dir.replace('$SLURM_JOB_ID/', '')
    os.makedirs(download_dir, exist_ok=True)
    model_path = os.path.join(download_dir, model_name)

    def check_file_matches_md5(checksum, fpath):
        if not os.path.exists(fpath):
            return False
        with open(fpath, 'rb') as f:
            current_md5checksum = hashlib.md5(f.read()).hexdigest()
            print('Monodepth2 model download checksum', current_md5checksum)
        return True
        # return current_md5checksum == checksum

    # see if we have the model already downloaded...
    if not os.path.exists(os.path.join(model_path, "depth.pth")):

        model_url, required_md5checksum = download_paths[model_name]

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("-> Downloading pretrained model to {}".format(model_path + ".zip"))
            if "https://" in model_url:
                urllib.request.urlretrieve(model_url, model_path + ".zip")
            else:
                model_url = model_url.replace("gdrive_id=", "")
                GoogleDriveDownloader.download_file_from_google_drive(model_url, model_path + ".zip")

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("   Failed to download a file which matches the checksum - quitting")
            quit()

        print("   Unzipping model...")
        with zipfile.ZipFile(model_path + ".zip", 'r') as f:
            f.extractall(model_path)

        print("   Model unzipped to {}".format(model_path))