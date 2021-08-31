import os

import numpy as np

from loader import CityscapesLoader
from loader.cityscapes_loader import Cityscapes
from loader.sequence_segmentation_loader import SequenceSegmentationLoader
from utils.utils import recursive_glob, np_local_seed


class SynthiaSeqLoader(SequenceSegmentationLoader):
    def __init__(self, **kwargs):
        print(kwargs)
        super(SynthiaSeqLoader, self).__init__(**kwargs)

        self.n_classes = Cityscapes.n_classes
        self.ignore_index = Cityscapes.ignore_index
        self.class_names = Cityscapes.class_names
        self.label_colors = Cityscapes.label_colours
        self.encode_map = Cityscapes.label_colours.items()

        self.full_res_shape = (1280, 760)
        self.fx = 532.740352 / self.full_res_shape[0]
        self.fy = 532.740352 / self.full_res_shape[1]
        self.u0 = 0.5
        self.v0 = 0.5

    def _prepare_filenames(self):
        def _extract_file_id(f):
            return int(f[:-4].rsplit("/", 1)[1])

        if self.split == "train":
            self.images_base = os.path.join(self.root, "video_small")
        elif self.split == "val":
            self.images_base = os.path.join(self.root, "RGB_small")
        else:
            raise NotImplementedError(self.split)
        self.annotations_base = None

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
            prefix, frame_number = img_path[:-4].rsplit("/", 1)
            suffix = img_path[-4:]
            frame_number = int(frame_number)
            if self.split == "train":
                img_path = f"{prefix}/{frame_number + offset:06d}{suffix}"
            else:
                img_path = f"{prefix}/{frame_number + offset:07d}{suffix}"
        return img_path

    def encode_segmap(self, mask):
        label_copy = self.ignore_index * np.ones(mask.shape[:2], dtype=np.float32)
        for i, c in self.encode_map:
            label_copy[np.where(np.all(mask == c, axis=-1))] = i
        return label_copy

    @staticmethod
    def decode_segmap_tocolor(temp):
        return CityscapesLoader.decode_segmap_tocolor(temp)
