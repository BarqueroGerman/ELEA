import torch.nn.functional as F
import torch
import numpy as np


def mse_loss(output, target):
    return F.mse_loss(output, target, reduction="mean")


def mse_loss_O(output, target):
    return F.mse_loss(output[:,0], target[:,0], reduction="mean")


def mse_loss_C(output, target):
    return F.mse_loss(output[:,1], target[:,1], reduction="mean")


def mse_loss_E(output, target):
    return F.mse_loss(output[:,2], target[:,2], reduction="mean")


def mse_loss_A(output, target):
    return F.mse_loss(output[:,3], target[:,3], reduction="mean")


def mse_loss_N(output, target):
    return F.mse_loss(output[:,4], target[:,4], reduction="mean")