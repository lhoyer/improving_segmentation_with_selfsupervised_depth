import os

import numpy as np

from loader import CityscapesLoader
from loader.cityscapes_loader import Cityscapes
from loader.sequence_segmentation_loader import SequenceSegmentationLoader
from utils.utils import recursive_glob, np_local_seed


class GTASeqLoader(SequenceSegmentationLoader):
    def __init__(self, **kwargs):
        super(GTASeqLoader, self).__init__(**kwargs)

        # The sequence dataset from https://playing-for-benchmarks.org/download/ uses a different class map than
        # cityscapes. The mapping can be obtained from https://github.com/srrichter/viper/blob/master/classes.csv.
        # Note that some classes are missing (e.g. train) and some are in other categories (e.g. pole in infrastructure)
        self.n_classes = Cityscapes.n_classes
        self.ignore_index = Cityscapes.ignore_index
        self.class_names = Cityscapes.class_names
        self.label_colors = Cityscapes.label_colours
        self.encode_map = [
            *Cityscapes.label_colours.items(),
            (8, [87, 182, 35]),  # tree to vegetation
            (8, [35, 142, 35]),  # gta specific vegetation color to vegetation
        ]

        self.full_res_shape = (1920, 1080)
        # See https://playing-for-benchmarks.org/download/ camera.zip
        self.fx = 1.206285
        self.fy = 2.144507
        self.u0 = 0.5
        self.v0 = 0.5

    def _prepare_filenames(self):
        def _extract_file_id(f):
            return int(f[:-4].rsplit("_", 1)[1])

        self.images_base = os.path.join(self.root, self.split, "img")
        self.annotations_base = os.path.join(self.root, self.split, "cls")

        self.files = sorted(recursive_glob(rootdir=self.images_base))
        mid_sequence_files = []
        files_with_neighbors = []
        for i, f in enumerate(self.files):
            if i == 0 or i == len(self.files) - 1:
                continue
            f_id = _extract_file_id(f)
            f_prev_id = _extract_file_id(self.files[i - 1])
            f_next_id = _extract_file_id(self.files[i + 1])

            if f_prev_id == f_id - 1 and f_next_id == f_id + 1:
                files_with_neighbors.append(f)
                if f_id % 10 == 0:
                    mid_sequence_files.append(f)

        if self.only_sequences_with_segmentation:
            self.files = mid_sequence_files
        else:
            self.files = files_with_neighbors
        # Shuffle to avoid too similar images close together in visualizations
        with np_local_seed(0):
            self.files = np.random.permutation(self.files).tolist()

    def get_image_path(self, index, offset=0):
        img_path = self.files[index]["name"].rstrip()
        if offset != 0:
            prefix, frame_number = img_path[:-4].rsplit("_", 1)
            suffix = img_path[-4:]
            frame_number = int(frame_number)
            img_path = f"{prefix}_{frame_number + offset:05d}{suffix}"
        return img_path

    def get_segmentation_path(self, index):
        img_path = self.files[index]["name"].rstrip()
        segmentation_path = img_path.replace(self.images_base, self.annotations_base).replace('.jpg', '.png')
        return segmentation_path

    def encode_segmap(self, mask):
        label_copy = self.ignore_index * np.ones(mask.shape[:2], dtype=np.float32)
        for i, c in self.encode_map:
            label_copy[np.where(np.all(mask == c, axis=-1))] = i
        return label_copy

    @staticmethod
    def decode_segmap_tocolor(temp):
        return CityscapesLoader.decode_segmap_tocolor(temp)