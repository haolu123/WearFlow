import os
import random
import numpy as np
import torch


def setup(seed, device, deterministic):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; CPU fallback is disabled')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    return torch.device(device)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
