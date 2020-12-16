import os
import random
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms

from loader.loader_utils import pil_loader, restrict_to_subset


class SequenceSegmentationLoader(data.Dataset):
    def __init__(
            self,
            root,
            split="train",
            img_size=(512, 1024),
            crop_h=None,
            crop_w=None,
            augmentations=None,
            downsample_gt=True,
            frame_idxs=None,
            num_scales=None,
            color_full_scale=0,
            restrict_dict=None,
            dataset_seed=42,
            load_labeled=True,
            load_unlabeled=False,
            generated_depth_dir=None,
            load_onehot=False,
            num_val_samples=None,
            only_sequences_with_segmentation=True,
            load_labels=True,
            load_sequence=True,
    ):
        super(SequenceSegmentationLoader, self).__init__()
        self.n_classes = None
        self.ignore_index = None
        self.label_colors = None
        self.class_map = None
        self.void_classes = None
        self.valid_classes = None
        self.full_res_shape = None
        self.fy = None
        self.fx = None
        self.u0 = None
        self.v0 = None
        self.images_base = None
        self.sequence_base = None
        self.annotations_base = None
        if augmentations is None:
            augmentations = {}
        self.root = root
        self.split = split
        self.is_train = (split == "train")
        self.augmentations = augmentations
        self.downsample_gt = downsample_gt
        self.seed = dataset_seed
        self.restrict_dict = restrict_dict
        self.load_labeled = load_labeled
        self.load_unlabeled = load_unlabeled
        self.generated_depth_dir = generated_depth_dir
        self.load_onehot = load_onehot
        self.num_val_samples = num_val_samples
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.height = self.img_size[0]
        self.width = self.img_size[1]
        self.num_scales = num_scales
        self.frame_idxs = frame_idxs
        assert self.width >= self.height
        self.only_sequences_with_segmentation = only_sequences_with_segmentation
        self.load_labels = load_labels
        self.load_sequence = load_sequence

        if not self.load_sequence:
            self.frame_idxs = [0]
            self.num_scales = 1

        if crop_h is None or crop_w is None or not self.is_train:
            self.crop_h = self.height
            self.crop_w = self.width
        else:
            self.crop_h = crop_h
            self.crop_w = crop_w
        assert self.crop_w >= self.crop_h

        self.enable_color_aug = self.augmentations.get("color_aug", False)
        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
        self.saturation = (0.8, 1.2)
        self.hue = (-0.1, 0.1)

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.crop_h // s, self.crop_w // s),
                                               interpolation=Image.ANTIALIAS)
        s = 2 ** color_full_scale
        self.resize_full = transforms.Resize((self.height // s, self.width // s),
                                             interpolation=Image.ANTIALIAS)
        self.to_tensor = transforms.ToTensor()

        self._prepare_filenames()

        for i in range(len(self.files)):
            self.files[i] = {
                "idx": i,
                "name": self.files[i],
                "labeled": True
            }
        if len(self.files) == 0:
            raise RuntimeError(f"Found no segmentation files in {self.images_base}")

        self._filter_available_files()

        if self.split == "train" and self.restrict_dict is not None:
            self.files = restrict_to_subset(self.files, **self.restrict_dict, seed=self.seed,
                                            load_labeled=self.load_labeled, load_unlabeled=self.load_unlabeled)
        if self.split != "train" and self.num_val_samples is not None:
            self.files = self.files[:self.num_val_samples]
        if not self.files or len(self.files) == 0:
            raise Exception(f"No files for split={self.split} found in {self.images_base}")

        print(f"Found {len(self.files)} {self.split} images")
        sys.stdout.flush()

    def _filter_available_files(self):
        """ Filter file list, so that all frame_idxs are available
        """
        filtered_files = []
        for idx in range(len(self.files)):
            available = True
            for j in self.frame_idxs:
                if not os.path.isfile(self.get_image_path(idx, j)):
                    available = False
                    break
            if available:
                filtered_files.append(self.files[idx])
            if self.only_sequences_with_segmentation:
                assert available
        self.files = filtered_files

    def get_color(self, index, offset, do_flip):
        img_path = self.get_image_path(index, offset)
        img = pil_loader(img_path, self.width, self.height)
        if do_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def get_segmentation(self, index, do_flip):
        lbl_path = self.get_segmentation_path(index)
        if self.downsample_gt:
            lbl = pil_loader(lbl_path, self.width, self.height, is_segmentation=True)
        else:
            lbl = pil_loader(lbl_path, -1, -1, is_segmentation=True)
        if do_flip:
            lbl = lbl.transpose(Image.FLIP_LEFT_RIGHT)

        return lbl

    def get_teacher_depth(self, index, do_flip):
        img_path = self.get_image_path(index, offset=0)
        subname = os.path.join(*img_path.split(os.sep)[-3:])

        if self.generated_depth_dir:
            depth_path = os.path.join(
                self.generated_depth_dir,
                subname.replace(".jpg", ".png")
            )
            depth = pil_loader(depth_path, -1, -1, is_segmentation=True, lru_cache=True)
            if do_flip:
                depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            depth = None

        return depth

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index'.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        is_labeled = self.files[index]["labeled"]
        inputs = {
            "idx": self.files[index]["idx"],
            "filename": os.path.join(*self.get_image_path(index).split(os.sep)[-3:]),
            "is_labeled": is_labeled,
        }

        do_color_aug = self.is_train and random.random() > 0.5 and self.enable_color_aug
        do_flip = self.is_train and "random_horizontal_flip" in self.augmentations and \
                  random.random() < self.augmentations["random_horizontal_flip"]

        for i in self.frame_idxs:
            inputs[("color", i, -1)] = self.get_color(index, i, do_flip)

        if self.load_labels:
            inputs["lbl"] = self.get_segmentation(index, do_flip)

        if self.generated_depth_dir is not None:
            inputs["pseudo_depth"] = self.get_teacher_depth(index, do_flip)

        inputs = self.random_crop(inputs, do_flip)

        self.preprocess(inputs, do_color_aug)

        if self.load_labels:
            inputs["lbl"] = inputs["lbl"] if is_labeled else (
                        self.ignore_index * torch.ones(inputs["lbl"].shape)).long()

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            # del inputs[("color_full", i, -1)]
            # del inputs[("color_full", i, 0)]

        if self.load_labels:
            if self.load_onehot and is_labeled:
                dense_lbl = inputs["lbl"].clone()
                dense_lbl[dense_lbl == self.ignore_index] = self.n_classes
                onehot_lbl = torch.nn.functional.one_hot(dense_lbl, self.n_classes + 2)
                onehot_lbl = onehot_lbl[..., :self.n_classes]
                onehot_lbl = onehot_lbl.permute(2, 0, 1)
            elif self.load_onehot and not is_labeled:
                onehot_lbl = torch.zeros((self.n_classes, *inputs["lbl"].shape)).long()
            else:
                onehot_lbl = None
            if self.load_onehot:
                inputs["onehot_lbl"] = onehot_lbl

        return inputs

    def random_crop(self, inputs, do_flip):
        w, h = inputs[("color", 0, -1)].size
        th, tw = self.crop_h, self.crop_w
        assert h <= w and th <= tw
        if w < tw or h < th:
            raise NotImplementedError

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        crop_region = (x1, y1, x1 + tw, y1 + th)

        for i in self.frame_idxs:
            img = inputs[("color", i, -1)]
            # inputs[("color_full", i, -1)] = img
            if w != tw or h != th:
                inputs[("color", i, -1)] = img.crop(crop_region)

        if (w != tw or h != th) and "lbl" in inputs:
            inputs["lbl"] = inputs["lbl"].crop(crop_region)

        if (w != tw or h != th) and "pseudo_depth" in inputs:
            inputs["pseudo_depth"] = inputs["pseudo_depth"].crop(crop_region)

        # adjusting intrinsics to match each scale in the pyramid
        if self.load_sequence:
            for scale in range(self.num_scales):
                K = self.get_K(x1, y1, do_flip)

                K[0, :] /= (2 ** scale)
                K[1, :] /= (2 ** scale)

                inv_K = np.linalg.pinv(K)

                inputs[("K", scale)] = torch.from_numpy(K)
                inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        return inputs

    def preprocess(self, inputs, do_color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        for k in list(inputs):
            if len(k) != 3:
                continue
            n, im, i = k
            if n == "color":
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
            if n == "color_full":
                inputs[(n, im, 0)] = self.resize_full(inputs[(n, im, -1)])

        for k in list(inputs):
            f = inputs[k]
            if len(k) != 3:
                continue
            n, im, i = k
            if "color" in n:
                inputs[(n, im, i)] = self.to_tensor(f)
                if i == 0:
                    processed_f = self.to_tensor(color_aug(f))
                    inputs[(n + "_aug", im, i)] = processed_f

        if "lbl" in inputs:
            lbl = np.asarray(inputs["lbl"])
            lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
            inputs["lbl"] = torch.from_numpy(lbl).long()

        if "pseudo_depth" in inputs:
            inputs["pseudo_depth"] = self.to_tensor(inputs["pseudo_depth"])

    def get_K(self, u_offset, v_offset, do_flip):
        u0 = self.u0
        v0 = self.v0
        if do_flip:
            u0 = self.full_res_shape[0] - u0
            v0 = self.full_res_shape[1] - v0

        return np.array([[self.fx, 0, u0 - u_offset, 0],
                         [0, self.fy, v0 - v_offset, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=np.float32)

    def _prepare_filenames(self):
        raise NotImplementedError

    def decode_segmap_tocolor(self, temp):
        raise NotImplementedError

    def encode_segmap(self, mask):
        raise NotImplementedError

    def get_image_path(self, index, offset=0):
        raise NotImplementedError

    def get_segmentation_path(self, index):
        raise NotImplementedError
