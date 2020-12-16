# Adapated from https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/schedulers/schedulers.py

import logging

from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, CosineAnnealingLR, StepLR, ReduceLROnPlateau, LambdaLR
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger("segsde")


def get_scheduler(optimizer, scheduler_dict):
    key2scheduler = {
        "constant_lr": ConstantLR,
        "poly_lr": PolynomialLR,
        "step_lr": StepLR,
        "reduce_lr_on_plateau": ReduceLROnPlateau,
        "multi_step": MultiStepLR,
        "cosine_annealing": CosineAnnealingLR,
        "exp_lr": ExponentialLR,
    }

    if scheduler_dict is None:
        logger.info("Using No LR Scheduling")
        return ConstantLR(optimizer)

    s_type = scheduler_dict["name"]
    scheduler_dict.pop("name")

    logging.info("Using {} scheduler with {} params".format(s_type, scheduler_dict))

    warmup_dict = {}
    if "warmup_iters" in scheduler_dict:
        # This can be done in a more pythonic way...
        warmup_dict["warmup_iters"] = scheduler_dict.get("warmup_iters", 100)
        warmup_dict["mode"] = scheduler_dict.get("warmup_mode", "linear")
        warmup_dict["gamma"] = scheduler_dict.get("warmup_factor", 0.2)

        logger.info(
            "Using Warmup with {} iters {} gamma and {} mode".format(
                warmup_dict["warmup_iters"], warmup_dict["gamma"], warmup_dict["mode"]
            )
        )

        scheduler_dict.pop("warmup_iters", None)
        scheduler_dict.pop("warmup_mode", None)
        scheduler_dict.pop("warmup_factor", None)

        base_scheduler = key2scheduler[s_type](optimizer, **scheduler_dict)
        return WarmUpLR(optimizer, base_scheduler, **warmup_dict)

    if s_type == "poly_lr_2":
        return LambdaLR(
            optimizer,
            lambda ep: max(1e-6, (1 - ep / scheduler_dict["max_iter"]) ** scheduler_dict["power"]),
        )
    else:
        return key2scheduler[s_type](optimizer, **scheduler_dict)


class ConstantLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr for base_lr in self.base_lrs]


class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, max_iter, decay_iter=1, gamma=0.9, last_epoch=-1):
        self.decay_iter = decay_iter
        self.max_iter = max_iter
        self.gamma = gamma
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch % self.decay_iter or self.last_epoch % self.max_iter:
            return [base_lr for base_lr in self.base_lrs]
        else:
            factor = (1 - self.last_epoch / float(self.max_iter)) ** self.gamma
            return [base_lr * factor for base_lr in self.base_lrs]


class WarmUpLR(_LRScheduler):
    def __init__(
        self, optimizer, scheduler, mode="linear", warmup_iters=100, gamma=0.2, last_epoch=-1
    ):
        self.mode = mode
        self.scheduler = scheduler
        self.warmup_iters = warmup_iters
        self.gamma = gamma
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        cold_lrs = self.scheduler.get_lr()

        if self.last_epoch < self.warmup_iters:
            if self.mode == "linear":
                alpha = self.last_epoch / float(self.warmup_iters)
                factor = self.gamma * (1 - alpha) + alpha

            elif self.mode == "constant":
                factor = self.gamma
            else:
                raise KeyError("WarmUp type {} not implemented".format(self.mode))

            return [factor * base_lr for base_lr in cold_lrs]

        return cold_lrs
