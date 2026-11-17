"""Diagnostics only: the monitor time partial is not a measured physical derivative."""
import numpy as np
import pandas as pd
import torch
from ..data.datasets import observations


def evaluate_derivative(model, batch, table, scalers, cases):
    idx = observations(table, cases)
    time = batch['time'][idx].detach().clone().requires_grad_(True)
    model.eval()
    estimate = model.estimate(batch, idx, time=time)
    partial = torch.autograd.grad(estimate.sum(), time)[0]
    with torch.no_grad():
        rate = model.dynamics.rate(estimate, batch['condition'][idx])
    factor = scalers['wear'][1] / scalers['time_scale']
    partial = partial.detach().cpu().numpy().reshape(-1)
    rate = rate.cpu().numpy().reshape(-1)
    frame = pd.DataFrame({'case': table['case'][idx], 'run': table['run'][idx],
                          'monitor_time_partial_mm_s': partial*factor,
                          'ode_rate_mm_s': rate*factor,
                          'residual_normalized': partial-rate})
    return {'normalized_residual_rmse': float(np.sqrt(np.mean((partial-rate)**2))),
            'physical_residual_rmse_mm_s': float(np.sqrt(np.mean((partial-rate)**2))*factor),
            'negative_time_partial_fraction': float(np.mean(partial < 0))}, frame
