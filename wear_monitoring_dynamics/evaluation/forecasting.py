import numpy as np
import pandas as pd
import torch
from ..data.datasets import pair_tensors


def evaluate_forecast(model, batch, table, scalers, cases, horizons):
    pairs = pair_tensors(table, cases, horizons, batch['time'].device)
    model.eval()
    with torch.no_grad():
        prediction = model.forecast(batch, pairs).cpu().numpy().reshape(-1)
    raw = pairs.cpu().numpy()
    mean, std = scalers['wear']
    frame = pd.DataFrame({'case': table['case'][raw[:, 0]], 'origin_run': table['run'][raw[:, 0]],
                          'target_run': table['run'][raw[:, 1]], 'horizon': raw[:, 2],
                          'target_vb_mm': table['VB'][raw[:, 1]], 'predicted_vb_mm': prediction * std + mean})
    if not np.isfinite(frame.predicted_vb_mm).all():
        raise ValueError('Nonfinite forecast')
    return {str(h): {'n': len(g), 'mae': float(abs(g.predicted_vb_mm-g.target_vb_mm).mean())}
            for h, g in frame.groupby('horizon')}, frame
