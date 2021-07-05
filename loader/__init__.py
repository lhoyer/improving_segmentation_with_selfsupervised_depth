from loader.camvid_loader import CamvidLoader
from loader.cityscapes_loader import CityscapesLoader
from loader.mapillary_vistas_loader import MapillaryVistasLoader
from loader.inference_loader import InferenceLoader


def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        "cityscapes": CityscapesLoader,
        "camvid": CamvidLoader,
        "mapillary": MapillaryVistasLoader,
        "inference": InferenceLoader,
    }[name]

def build_loader(cfg, split='train', load_labels=True, load_sequence=True):
    data_loader = get_loader(cfg["dataset"])

    if split == "train":
        loader = data_loader(
            root=cfg["path"],
            downsample_gt=True,
            split=split,
            img_size=(cfg["height"], cfg["width"]),
            crop_h=cfg.get("crop_h", cfg["height"]),
            crop_w=cfg.get("crop_w", cfg["width"]),
            color_full_scale=cfg["color_full_scale"],
            frame_idxs=cfg["frame_ids"],
            num_scales=cfg["num_scales"],
            augmentations=cfg["augmentations"],
            dataset_seed=cfg["dataset_seed"],
            restrict_dict=cfg["restrict_to_subset"],
            load_labeled=cfg.get("load_labeled", True),
            load_unlabeled=cfg.get("load_unlabeled", False),
            generated_depth_dir=cfg.get("generated_depth_dir", None),
            load_onehot=cfg.get("load_onehot", False),
            only_sequences_with_segmentation=cfg["only_sequences_with_segmentation"],
            load_labels=load_labels,
            load_sequence=load_sequence
        )
    elif split == "val":
        loader = data_loader(
            root=cfg["path"],
            downsample_gt=cfg["val_downsample_gt"],
            split=cfg.get("val_split", "val"),
            img_size=(cfg["height"], cfg["width"]),
            crop_h=cfg.get("crop_h", cfg["height"]),
            crop_w=cfg.get("crop_w", cfg["width"]),
            color_full_scale=cfg["color_full_scale"],
            frame_idxs=cfg["frame_ids"],
            num_scales=cfg["num_scales"],
            augmentations={},
            generated_depth_dir=cfg.get("generated_depth_dir", None),
            load_onehot=cfg.get("load_onehot", False),
            num_val_samples=cfg.get("num_val_samples", None),
            only_sequences_with_segmentation=cfg.get("val_only_sequences_with_segmentation", True),
            load_labels=load_labels,
            load_sequence=load_sequence
        )
    else:
        raise NotImplementedError(cfg["dataset"])

    return loader
    
