import torch
import numpy as np


def mixup(size, alpha):
    # Mixup is a very common data augmentation and improves generalization for a lot of audio tasks
    # https://arxiv.org/abs/1710.09412
    rn_indices = torch.randperm(size)  # randomly shuffled batch indices
    lambd = np.random.beta(alpha, alpha, size).astype(np.float32)  # choose mixing coefficients from Beta distribution
    lambd = np.concatenate([lambd[:, None], 1 - lambd[:, None]], 1).max(1)  # choose lambda closer to 1
    lam = torch.FloatTensor(lambd)  # convert to pytorch float tensor
    return rn_indices, lam
