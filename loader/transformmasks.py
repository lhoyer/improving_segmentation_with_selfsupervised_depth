# Adapted from ClassMix
# https://github.com/WilhelmT/ClassMix/blob/master/utils/transformmasks.py

import numpy as np
import torch


def generate_cutout_mask(img_size, seed=None):
    np.random.seed(seed)

    cutout_area = img_size[0] * img_size[1] / 2

    w = np.random.randint(img_size[1] / 2, img_size[1] + 1)
    h = np.round(cutout_area / w)

    x_start = np.random.randint(0, img_size[1] - w + 1)
    y_start = np.random.randint(0, img_size[0] - h + 1)

    x_end = int(x_start + w)
    y_end = int(y_start + h)

    mask = np.ones(img_size)
    mask[y_start:y_end, x_start:x_end] = 0
    return mask.astype(float)


def generate_class_mask(pred, classes):
    pred, classes = torch.broadcast_tensors(pred.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))
    N = pred.eq(classes).sum(0)
    return N


def generate_depth_mask(depth, threshold):
    if threshold.shape[0] == 1:
        return depth.ge(threshold).float()
    elif threshold.shape[0] == 2:
        t1 = torch.min(threshold)
        t2 = torch.max(threshold)
        return depth.ge(t1).le(t2).float()
    else:
        raise NotImplementedError

