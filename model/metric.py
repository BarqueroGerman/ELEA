import torch
import numpy as np



def mse_loss(output, target):
    with torch.no_grad():
        return torch.nn.functional.mse_loss(output, target, reduction="mean")
