from copy import deepcopy
from functools import lru_cache

import numpy as np
from PIL import Image

from utils.utils import np_local_seed


def _build_size(orig_img, width, height):
    size = [width, height]
    if size[0] == -1: size[0] = orig_img.width
    if size[1] == -1: size[1] = orig_img.height
    return size


# 3Gb / 300k = 10000 (per worker)
@lru_cache(maxsize=5000)
def _load_lru_cache(*args, **kwargs):
    return _load(*args, **kwargs)


def _load(_path, is_segmentation, resize, width, height):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(_path, 'rb') as f:
        with Image.open(f) as _img:
            if is_segmentation:
                _img = _img.convert()
                if resize: _img = _img.resize(_build_size(_img, width, height), Image.NEAREST)
            else:
                _img = _img.convert('RGB')
                if resize: _img = _img.resize(_build_size(_img, width, height), Image.ANTIALIAS)
    # print(np.asarray(_img).nbytes/1e6)
    return _img


def pil_loader(path, std_width, std_height, is_segmentation=False, lru_cache=False):
    if lru_cache:
        load_fn = _load_lru_cache
    else:
        load_fn = _load
    return load_fn(path, is_segmentation, True, std_width, std_height)


def restrict_to_subset(files, mode, n_subset, seed, load_labeled, load_unlabeled, subset=None):
    assert mode == "fixed" or subset is None
    print(f'Restrict subset from {len(files)} to {n_subset} images ...')

    if mode == "random":
        with np_local_seed(seed):
            p = np.random.permutation(len(files))
        p = p[:n_subset]
    elif mode == "fixed":
        assert subset is not None
        assert len(subset) == n_subset
        p = subset
    else:
        raise NotImplementedError(mode)

    p = sorted(p)
    print('Use image subset {} with class frequencies:'.format(p))

    labeled_files = [f for f in files if f["idx"] in p]
    assert len(labeled_files) == n_subset
    unlabeled_files = [f for f in files if f["idx"] not in p]
    for i in range(len(unlabeled_files)):
        unlabeled_files[i]["labeled"] = False
    assert len(unlabeled_files) == len(files) - n_subset

    if load_labeled and load_unlabeled:
        concat_files = deepcopy(labeled_files)
        concat_files.extend(unlabeled_files)
        files = concat_files
    elif load_labeled:
        files = labeled_files
    elif load_unlabeled:
        files = unlabeled_files
    else:
        raise ValueError("Neither unlabeled or labeled data is specified to be loaded.")
    print("Keep %d images" % (len(files)))

    return files
