"""Observed-trajectory diagnostics in physical mm; no prediction correction."""
import numpy as np
import pandas as pd

NUMERICAL_TOLERANCE_MM = 1e-8


def per_case_monotonicity(frame):
    rows = []
    for case, values in frame.groupby('case'):
        difference = np.diff(values.sort_values('run').predicted_vb_mm.to_numpy())
        if not len(difference):
            continue
        drops = np.maximum(-difference, 0)
        rows.append(dict(case=case, pairs=len(difference), decreases=int((difference < -NUMERICAL_TOLERANCE_MM).sum()),
                         violation_rate=float((difference < -NUMERICAL_TOLERANCE_MM).mean()),
                         mean_drop_mm=float(drops.mean()), max_drop_mm=float(drops.max())))
    return pd.DataFrame(rows)


def epoch_monotonicity(path):
    """Vectorized post-hoc statistics, same computation for old and new runs."""
    frame = pd.read_csv(path, usecols=['epoch', 'split', 'task', 'case', 'origin_run', 'predicted_vb_mm'])
    frame = frame[frame.task == 'monitoring'].sort_values(['epoch', 'split', 'case', 'origin_run'])
    delta = frame.groupby(['epoch', 'split', 'case']).predicted_vb_mm.diff()
    frame = frame.loc[delta.notna()].copy()
    delta = delta.loc[delta.notna()]
    frame['decreases'] = (delta < -NUMERICAL_TOLERANCE_MM).astype(int)
    frame['drop_mm'] = (-delta).clip(lower=0)
    result = frame.groupby(['epoch', 'split']).agg(pairs=('drop_mm', 'size'), decreases=('decreases', 'sum'),
                                                mean_drop_mm=('drop_mm', 'mean'), max_drop_mm=('drop_mm', 'max')).reset_index()
    result['violation_rate'] = result.decreases/result.pairs
    return result
