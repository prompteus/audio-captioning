import torch
import numpy as np


def mixup(size, alpha):
    rn_indices = torch.randperm(size)
    lambd = np.random.beta(alpha, alpha, size).astype(np.float32)
    lambd = np.concatenate([lambd[:, None], 1 - lambd[:, None]], 1).max(1)
    lam = torch.FloatTensor(lambd)
    return rn_indices, lam
