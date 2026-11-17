"""Train-only scaling. z=(VB-mean)/std, tau=time/time_scale."""
import numpy as np


def forecasting_pairs(table, cases, horizons):
    rows = []
    for case in cases:
        idx = np.flatnonzero((table['case'] == case) & np.isfinite(table['VB']))
        for rank, origin in enumerate(idx):
            for horizon in horizons:
                if rank + horizon >= len(idx):
                    continue
                target = idx[rank + horizon]
                if table['time'][target] <= table['time'][origin]:
                    continue
                if any(table[k][origin] != table[k][target] for k in ('DOC', 'feed', 'material')):
                    raise ValueError('First version requires constant conditions within a forecast interval')
                rows.append((origin, target, horizon))
    return np.asarray(rows, dtype=np.int64).reshape(-1, 3)


def fit_scalers(table, train_cases, sensor_preprocessing=None):
    mask = np.isin(table['case'], train_cases)
    valid = mask & np.isfinite(table['VB'])
    if not valid.any():
        raise ValueError('No training wear labels')
    pairs = forecasting_pairs(table, train_cases, [3])
    if not len(pairs):
        raise ValueError('No positive h=3 training intervals for time scaling')
    scale = float(np.median(table['time'][pairs[:, 1]] - table['time'][pairs[:, 0]]))
    def stats(x):
        return np.mean(x, axis=0).tolist(), np.maximum(np.std(x, axis=0), 1e-8).tolist()
    sensor_mask = mask & table.get('sensor_valid', np.ones(len(mask),dtype=bool))
    if not sensor_mask.any(): raise ValueError('No valid training sensor records')
    result = {'wear': stats(table['VB'][valid]), 'sensor': stats(table['sensor'][sensor_mask]),
            'condition': stats(np.column_stack([table['DOC'], table['feed']])[mask]),
            'time_scale': scale, 'train_cases': sorted(train_cases)}
    if sensor_preprocessing:
        from .sensor_transforms import fit_sensor_transform
        result['sensor_transform'],result['sensor']=fit_sensor_transform(table['sensor'][sensor_mask],sensor_preprocessing)
    return result


def transform(table, scalers):
    def standardize(x, key):
        mean, std = scalers[key]
        return (x - np.asarray(mean)) / np.asarray(std)
    continuous = standardize(np.column_stack([table['DOC'], table['feed']]), 'condition')
    onehot = np.column_stack([table['material'] == 1, table['material'] == 2])
    from .sensor_transforms import apply_sensor_transform
    sensor=apply_sensor_transform(table['sensor'],scalers.get('sensor_transform'))
    return {'sensor': standardize(sensor, 'sensor'),
            'condition': np.column_stack([continuous, onehot]),
            'time': (table['time'] / scalers['time_scale'])[:, None],
            'target': standardize(table['VB'], 'wear')[:, None]}
