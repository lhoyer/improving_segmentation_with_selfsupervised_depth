import numpy
import torch
import torch.nn.functional as F

def berhu(input, target, mask, apply_log=False):
    threshold = 0.2
    if apply_log:
        input = torch.log(1 + input)
        target = torch.log(1 + target)
    absdiff = torch.abs(target - input) * mask
    C = threshold * torch.max(absdiff).item()
    loss = torch.mean(torch.where(absdiff <= C,
                                  absdiff,
                                  (absdiff * absdiff + C * C) / (2 * C)))
    return loss

def cross_entropy2d(input, target, class_weight=None, pixel_weights=None):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=class_weight, reduction="mean" if pixel_weights is None else "none", ignore_index=250
    )
    if pixel_weights is not None:
        if torch.any(torch.isnan(pixel_weights)):
            print("WARN cross_entropy2d pixel_weights contains NaN. Skip weighting.")
        else:
            pixel_weights = pixel_weights.view(-1)
            loss = pixel_weights.detach() * loss
        loss = torch.mean(loss)
    return loss


def pixel_wise_entropy(logits, normalize=False):
    assert logits.dim() == 4
    p = F.softmax(logits, dim=1)
    N, C, H, W = p.shape
    pw_entropy = -torch.sum(p * torch.log2(p + 1e-30), dim=1) / numpy.log2(C)
    if normalize:
        pw_entropy = (pw_entropy - torch.min(pw_entropy)) / (torch.max(pw_entropy) - torch.min(pw_entropy))
    return pw_entropy
