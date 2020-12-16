# Based on https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loader/mapillary_vistas_loader.py

import json
import os

import numpy as np

from loader.sequence_segmentation_loader import SequenceSegmentationLoader
from utils.utils import recursive_glob


class MapillaryVistasLoader(SequenceSegmentationLoader):
    n_classes = 65
    ignore_index = 250

    def __init__(self, **kwargs):
        super(MapillaryVistasLoader, self).__init__(**kwargs)

        self.ignore_index = MapillaryVistasLoader.ignore_index
        self.n_classes = MapillaryVistasLoader.n_classes
        self.class_ids, self.class_names, self.class_colors = self.parse_config()

    def parse_config(self):
        with open(os.path.join(self.root, "config.json")) as config_file:
            config = json.load(config_file)

        labels = config["labels"]

        class_names = []
        class_ids = []
        class_colors = []
        print("There are {} labels in the config file".format(len(labels)))
        for label_id, label in enumerate(labels):
            class_names.append(label["readable"])
            class_ids.append(label_id)
            class_colors.append(label["color"])

        return class_names, class_ids, class_colors

    def _prepare_filenames(self):
        self.images_base = os.path.join(self.root, self.split, "images")
        self.annotations_base = os.path.join(self.root, self.split, "labels")
        self.files = sorted(recursive_glob(rootdir=self.images_base, suffix=".jpg"))

    def get_image_path(self, index, offset=0):
        assert offset == 0
        img_path = self.files[index]["name"].rstrip()
        return img_path

    def get_segmentation_path(self, index):
        img_path = self.files[index]["name"].rstrip()
        segmentation_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-1].replace(".jpg", ".png")
        )
        return segmentation_path

    def encode_segmap(self, mask):
        id_mask = np.zeros(mask.shape[:-1])
        r, g, b = mask[:, :, 0], mask[:, :, 1], mask[:, :, 2]
        for l in range(0, self.n_classes+1):
            cmask = (r == self.class_colors[l][0]) & (g == self.class_colors[l][1]) & (b == self.class_colors[l][2])
            id_mask[cmask] = l
        # Replace Mapillary unlabelled with default ignore index of our framework
        id_mask[id_mask == 65] = self.ignore_index
        return id_mask

    def decode_segmap_tocolor(self, temp):
        r = np.zeros((temp.shape[0], temp.shape[1]))
        g = np.zeros((temp.shape[0], temp.shape[1]))
        b = np.zeros((temp.shape[0], temp.shape[1]))
        for l in range(0, self.n_classes):
            r[temp == l] = self.class_colors[l][0]
            g[temp == l] = self.class_colors[l][1]
            b[temp == l] = self.class_colors[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb
