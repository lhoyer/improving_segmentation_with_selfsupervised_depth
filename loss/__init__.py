import functools
import logging

from loss.loss import (
    cross_entropy2d,
)
from loss.monodepth_loss import MonodepthLoss

logger = logging.getLogger("segsde")

key2loss = {
    "cross_entropy": cross_entropy2d,
}


def get_segmentation_loss_function(cfg):
    if cfg["training"]["segmentation_loss"] is None:
        logger.info("Using default cross entropy loss")
        return cross_entropy2d

    else:
        loss_dict = cfg["training"]["segmentation_loss"]
        loss_name = loss_dict["name"]
        loss_params = {k: v for k, v in loss_dict.items() if k != "name"}

        if loss_name not in key2loss:
            raise NotImplementedError("Loss {} not implemented".format(loss_name))

        logger.info("Using {} with {} params".format(loss_name, loss_params))
        return functools.partial(key2loss[loss_name], **loss_params)

def get_monodepth_loss(cfg, is_train, batch_size=None):
    if batch_size is None:
        batch_size = cfg["training"]["batch_size"]
    return MonodepthLoss(**cfg["training"]["monodepth_loss"],
                         batch_size=batch_size,
                         is_train=is_train)
