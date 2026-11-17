import numpy as np
import pandas as pd
import torch
from ..data.datasets import observations


def regression_metrics(frame):
    if frame.empty:
        raise ValueError('No labeled evaluation observations')
    error = frame['predicted_vb_mm'] - frame['vb_mm']
    per_case = frame.assign(abs_error=abs(error), signed_error=error, squared_error=error**2).groupby('case').agg(
        n=('abs_error', 'size'), mae=('abs_error', 'mean'), bias=('signed_error', 'mean'), mse=('squared_error', 'mean')).reset_index()
    per_case['rmse'] = np.sqrt(per_case.pop('mse'))
    return {'n': len(frame), 'n_cases': len(per_case), 'case_macro_mae': float(per_case.mae.mean()),
            'mae': float(abs(error).mean()), 'rmse': float(np.sqrt((error**2).mean())),
            'bias': float(error.mean()), 'negative_prediction_fraction': float((frame.predicted_vb_mm < 0).mean())}, per_case


def evaluate_monitor(model, batch, table, scalers, cases):
    idx = observations(table, cases)
    model.eval()
    with torch.no_grad():
        prediction = model.estimate(batch, idx).cpu().numpy().reshape(-1)
    mean, std = scalers['wear']
    frame = pd.DataFrame({k: table[k][idx] for k in ('case', 'run', 'time', 'DOC', 'feed', 'material')})
    frame['vb_mm'] = table['VB'][idx]
    frame['predicted_vb_mm'] = prediction * std + mean
    if not np.isfinite(frame.predicted_vb_mm).all():
        raise ValueError('Nonfinite monitoring predictions')
    frame['wear_regime'] = pd.cut(frame.vb_mm, [-np.inf, .2, .5, .74, np.inf],
                                  labels=['low', 'moderate', 'high', 'above_0.74']).astype(str)
    metrics, per_case = regression_metrics(frame)
    metrics['regimes'] = {name: {'n': len(g), 'mae': float(abs(g.predicted_vb_mm-g.vb_mm).mean())}
                          for name, g in frame.groupby('wear_regime')}
    return metrics, frame, per_case
