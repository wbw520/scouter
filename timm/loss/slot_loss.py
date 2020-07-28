import torch.nn as nn
import torch.nn.functional as F


class SoltLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, target):
        loss = F.nll_loss(x[0], target) + x[1]
        return loss