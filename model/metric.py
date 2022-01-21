import torch
import numpy as np



def mse(output, target):
    with torch.no_grad():
        return torch.nn.functional.mse_loss(output, target, reduction="mean")

def mse_O(output, target):
    with torch.no_grad():
        return torch.nn.functional.mse_loss(output[:, 0], target[:, 0], reduction="mean")

def mse_C(output, target):
    with torch.no_grad():
        return torch.nn.functional.mse_loss(output[:, 1], target[:, 1], reduction="mean")

def mse_E(output, target):
    with torch.no_grad():
        return torch.nn.functional.mse_loss(output[:, 2], target[:, 2], reduction="mean")

def mse_A(output, target):
    with torch.no_grad():
        return torch.nn.functional.mse_loss(output[:, 3], target[:, 3], reduction="mean")

def mse_N(output, target):
    with torch.no_grad():
        return torch.nn.functional.mse_loss(output[:, 4], target[:, 4], reduction="mean")
