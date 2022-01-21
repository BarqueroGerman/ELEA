import torch.nn.functional as F
import torch
import numpy as np


def mse_loss(output, target):
    return F.mse_loss(output, target, reduction="mean")