"""Small-data full-batch tensors; observations and pairs are separate populations."""
import numpy as np
import torch
from .preprocessing import transform, forecasting_pairs


def tensors(table, scalers, device):
    return {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in transform(table, scalers).items()}


def observations(table, cases, labeled=True):
    mask = np.isin(table['case'], cases)
    if labeled:
        mask &= np.isfinite(table['VB'])
    return np.flatnonzero(mask)


def pair_tensors(table, cases, horizons, device):
    return torch.tensor(forecasting_pairs(table, cases, horizons), dtype=torch.long, device=device)
