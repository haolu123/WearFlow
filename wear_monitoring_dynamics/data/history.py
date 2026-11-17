"""Case-local windows ending at the current record, including unlabeled records."""
import numpy as np
import torch


def add_history(batch, table, length, time_scale, device):
    if int(length) != length or length < 1:
        raise ValueError('Window length must be a positive integer')
    n = len(table['case'])
    indices = np.zeros((n, length), dtype=np.int64)
    mask = np.zeros((n, length), dtype=bool)
    delta = np.zeros((n, length, 1), dtype=np.float32)
    valid = table.get('sensor_valid', np.ones(n,dtype=bool))
    for case in np.unique(table['case']):
        rows = np.flatnonzero(table['case'] == case)
        if np.any(np.diff(table['time'][rows]) < 0):
            raise ValueError('Case records must be in chronological order')
        usable = rows[valid[rows]]
        gaps = np.r_[0., np.diff(table['time'][usable])] / time_scale
        for rank, current in enumerate(rows):
            end = np.searchsorted(usable,current,side='right')
            start = max(0,end-length)
            past = usable[start:end]
            indices[current, :] = current
            if not len(past):continue
            indices[current, -len(past):] = past
            mask[current, -len(past):] = True
            delta[current, -len(past):, 0] = gaps[start:end]
    batch['history_indices'] = torch.tensor(indices, device=device)
    batch['history_mask'] = torch.tensor(mask, device=device)
    batch['history_delta'] = torch.tensor(delta, device=device)
    batch['sensor_valid'] = torch.tensor(valid, device=device)
    return batch
