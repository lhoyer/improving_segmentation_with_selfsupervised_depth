import os

import numpy as np

from loader import CityscapesLoader
from loader.cityscapes_loader import Cityscapes
from loader.sequence_segmentation_loader import SequenceSegmentationLoader
from utils.utils import recursive_glob, np_local_seed


class GTASegLoader(SequenceSegmentationLoader):
    def __init__(self, **kwargs):
        super(GTASegLoader, self).__init__(**kwargs)

        assert not kwargs["load_sequence"]
        self.n_classes = Cityscapes.n_classes
        self.ignore_index = Cityscapes.ignore_index
        self.class_names = Cityscapes.class_names
        self.label_colors = Cityscapes.label_colours

        self.pil_convert_segmentation = False
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

    def _prepare_filenames(self):
        if self.img_size == (1052, 1914) or self.img_size == (1024, 2048) or not self.load_preprocessed:
            suffix = ""
        elif self.img_size == (512, 1024):
            suffix = "_small"
        else:
            raise NotImplementedError(f"Unexpected image size {self.img_size}")
        self.images_base = os.path.join(self.root, "images" + suffix)
        self.annotations_base = os.path.join(self.root, "labels")

        self.files = sorted(recursive_glob(rootdir=self.images_base))
        n_val = 500 if self.num_val_samples is None else self.num_val_samples
        if self.split == "train":
            self.files = self.files[:-n_val]
        elif self.split == "val":
            self.files = self.files[-n_val:]
        else:
            raise NotImplementedError(self.split)
        # Shuffle to avoid too similar images close together in visualizations
        with np_local_seed(0):
            self.files = np.random.permutation(self.files).tolist()

    def get_image_path(self, index, offset=0):
        assert offset == 0
        img_path = self.files[index]["name"].rstrip()
        return img_path

    def get_segmentation_path(self, index):
        img_path = self.files[index]["name"].rstrip()
        segmentation_path = img_path.replace(self.images_base, self.annotations_base).replace('.jpg', '.png')
        return segmentation_path

    def encode_segmap(self, mask):
        label_copy = self.ignore_index * np.ones(mask.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[mask == k] = v
        return label_copy

    @staticmethod
    def decode_segmap_tocolor(temp):
        return CityscapesLoader.decode_segmap_tocolor(temp)

