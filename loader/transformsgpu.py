# Adapted from ClassMix
# https://github.com/WilhelmT/ClassMix/blob/master/utils/transformsgpu.py

import kornia
import numpy as np
import torch
import torch.nn as nn

from configs.machine_config import MachineConfig


def strongTransformOneMix(parameters, data=None, target=None, onemix=True):
    assert ((data is not None) or (target is not None))
    if onemix:
        data, target = oneMix(mask=parameters["Mix"], data=data, target=target)
    else:
        data, target = mix(mask=parameters["Mix"], data=data, target=target)
    data, target = color_jitter(jitter=parameters["ColorJitter"], data=data, target=target)
    data, target = gaussian_blur(blur=parameters["GaussianBlur"], data=data, target=target)
    return data, target


def color_jitter(jitter, data=None, target=None, s=0.25):
    # s is the strength of color jitter
    if not (data is None):
        if data.shape[1] == 3:
            if jitter > 0.2:
                seq = nn.Sequential(kornia.augmentation.ColorJitter(brightness=s, contrast=s, saturation=s, hue=s))
                assert data.shape[1] == 3
                data = seq(data).float()
    return data, target


def gaussian_blur(blur, data=None, target=None):
    if not (data is None):
        if data.shape[1] == 3:
            if blur > 0.5:
                sigma = np.random.uniform(0.15, 1.15)
                kernel_size_y = int(np.floor(np.ceil(0.1 * data.shape[2]) - 0.5 + np.ceil(0.1 * data.shape[2]) % 2))
                kernel_size_x = int(np.floor(np.ceil(0.1 * data.shape[3]) - 0.5 + np.ceil(0.1 * data.shape[3]) % 2))
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(kornia.filters.GaussianBlur2d(kernel_size=kernel_size, sigma=(sigma, sigma)))
                data = seq(data)
    return data, target


def oneMix(mask, data = None, target = None):
    #Mix
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0*data[0]+(1-stackedMask0)*data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0*target[0]+(1-stackedMask0)*target[1]).unsqueeze(0)
    return data, target


def mix(mask, data=None, target=None):
    if not (data is None):
        if mask.shape[0] == data.shape[0]:
            data = torch.cat([(mask[i] * data[i] + (1 - mask[i]) * data[(i + 1) % data.shape[0]]).unsqueeze(0) for i in
                              range(data.shape[0])])
        elif mask.shape[0] == data.shape[0] / 2:
            data = torch.cat((torch.cat([(mask[i] * data[2 * i] + (1 - mask[i]) * data[2 * i + 1]).unsqueeze(0) for i in
                                         range(int(data.shape[0] / 2))]),
                              torch.cat([((1 - mask[i]) * data[2 * i] + mask[i] * data[2 * i + 1]).unsqueeze(0) for i in
                                         range(int(data.shape[0] / 2))])))
    if not (target is None):
        target = torch.cat(
            [(mask[i] * target[i] + (1 - mask[i]) * target[(i + 1) % target.shape[0]]).unsqueeze(0) for i in
             range(target.shape[0])])
    return data, target

