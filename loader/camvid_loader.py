# Based on https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loader/camvid_loader.py

import os

import numpy as np

from loader.sequence_segmentation_loader import SequenceSegmentationLoader
from utils.utils import recursive_glob


class CamvidLoader(SequenceSegmentationLoader):
    valid_classes = list(range(11))
    class_names = [
        "sky",
        "building",
        "pole",
        "road",
        "pavement",
        "tree",
        "signsymbol",
        "fence",
        "car",
        "pedestrian",
        "bicyclist",
        "unlabeled"
    ]
    n_classes = 12
    ignore_index = 250

    def __init__(self, **kwargs):
        super(CamvidLoader, self).__init__(**kwargs)

        self.n_classes = CamvidLoader.n_classes
        self.ignore_index = CamvidLoader.ignore_index

        self.full_res_shape = (480, 360)

    def _prepare_filenames(self):
        self.images_base = os.path.join(self.root, self.split)
        self.annotations_base = os.path.join(self.root, self.split + "annot")
        self.files = sorted(recursive_glob(rootdir=self.images_base))

    def get_image_path(self, index, offset=0):
        assert offset == 0
        img_path = self.files[index]["name"].rstrip()
        return img_path

    def get_segmentation_path(self, index):
        img_path = self.files[index]["name"].rstrip()
        segmentation_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-1]
        )
        return segmentation_path

    def encode_segmap(self, mask):
        # Replace CamVid unlabelled with default ignore index of our framework
        mask[mask == 11] = self.ignore_index
        return mask

    def decode_segmap_tocolor(self, temp):
        Sky = [128, 128, 128]
        Building = [128, 0, 0]
        Pole = [192, 192, 128]
        Road = [128, 64, 128]
        Pavement = [60, 40, 222]
        Tree = [128, 128, 0]
        SignSymbol = [192, 128, 128]
        Fence = [64, 64, 128]
        Car = [64, 0, 128]
        Pedestrian = [64, 64, 0]
        Bicyclist = [0, 128, 192]
        Unlabelled = [0, 0, 0]

        label_colours = np.array(
            [
                Sky,
                Building,
                Pole,
                Road,
                Pavement,
                Tree,
                SignSymbol,
                Fence,
                Car,
                Pedestrian,
                Bicyclist,
                Unlabelled,
            ]
        )
        r = np.zeros((temp.shape[0], temp.shape[1]))
        g = np.zeros((temp.shape[0], temp.shape[1]))
        b = np.zeros((temp.shape[0], temp.shape[1]))
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb