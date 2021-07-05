import os

import numpy as np

from loader.sequence_segmentation_loader import SequenceSegmentationLoader
from loader.cityscapes_loader import Cityscapes
from utils.utils import recursive_glob

class InferenceLoader(SequenceSegmentationLoader):
    def __init__(self, **kwargs):
        super(InferenceLoader, self).__init__(**kwargs)

        self.n_classes = Cityscapes.n_classes
        self.ignore_index = Cityscapes.ignore_index
        self.void_classes = Cityscapes.void_classes
        self.valid_classes = Cityscapes.valid_classes
        self.label_colors = Cityscapes.label_colours
        self.class_names = Cityscapes.class_names
        self.class_map = Cityscapes.class_map
        self.decode_class_map = Cityscapes.decode_class_map

        self.full_res_shape = (2048, 1024)
        # See https://www.cityscapes-dataset.com/file-handling/?packageID=8
        self.fx = 2262.52
        self.fy = 2265.3017905988554
        self.u0 = 1096.98
        self.v0 = 513.137

    def _prepare_filenames(self):
        self.images_base = self.root
        self.sequence_base = None
        self.annotations_base = None

        self.files = sorted(recursive_glob(rootdir=self.images_base))

    def get_image_path(self, index, offset=0):
        img_path = self.files[index]["name"].rstrip()
        assert offset == 0
        return img_path

    def get_segmentation_path(self, index):
        return None

    def decode_segmap_tocolor(self, temp):
        return Cityscapes.decode_segmap_tocolor(temp)

    def encode_segmap(self, mask):
        return Cityscapes.encode_segmap(mask)
