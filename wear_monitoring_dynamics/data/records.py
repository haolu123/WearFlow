"""Read trusted local NASA NPY and cache raw, unfitted features by content hash."""
from pathlib import Path
import numpy as np
from ..config import digest, file_hash, write_json
from .features import sensor_features, SENSORS, STATISTICS


def load_table(path, cache, raw_sensor_abs_limit=1e6):
    path, cache = Path(path), Path(cache)
    signature = digest({'data': file_hash(path), 'features': file_hash(Path(__file__).with_name('features.py')),
                        'reader': file_hash(__file__), 'raw_sensor_abs_limit': raw_sensor_abs_limit})
    target = cache / f'features_{signature[:16]}.npz'
    if target.exists():
        with np.load(target, allow_pickle=False) as archive:
            return {k: archive[k] for k in archive.files}, signature
    records = sorted(np.load(path, allow_pickle=True).tolist(), key=lambda r: (int(r['case']), int(r['run'])))
    # Label-independent numerical-integrity quarantine, shared by ALL ablations.
    # Never silently clip an extreme run into a plausible sensor observation.
    original_count = len(records)
    retained, excluded = [], []
    for record in records:
        peak = max(float(np.max(np.abs(np.asarray(record[k], dtype=np.float64)))) for k in SENSORS)
        if peak > raw_sensor_abs_limit:
            excluded.append({'case': int(record['case']), 'run': int(record['run']),
                             'peak_abs_sensor': peak, 'reason': 'raw_sensor_abs_limit_exceeded'})
        else:
            retained.append(record)
    records = retained
    table = {k: np.asarray([r[k] for r in records], dtype=np.float64)
             for k in ('case', 'run', 'time', 'VB', 'DOC', 'feed', 'material')}
    table['sensor'] = np.stack([sensor_features(r) for r in records])
    if len(set(zip(table['case'], table['run']))) != len(records):
        raise ValueError('Duplicate case/run observations')
    for k in ('case', 'run', 'time', 'DOC', 'feed', 'material'):
        if not np.isfinite(table[k]).all():
            raise ValueError(f'Nonfinite metadata: {k}')
    if not np.isin(table['material'], [1, 2]).all():
        raise ValueError('Unknown material')
    if np.isinf(table['VB']).any() or (table['VB'][np.isfinite(table['VB'])] < 0).any():
        raise ValueError('Invalid wear label')
    for case in np.unique(table['case']):
        if (np.diff(table['time'][table['case'] == case]) < 0).any():
            raise ValueError(f'Time decreases in case {case}')
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **table)
    write_json(target.with_suffix('.json'), {'signature': signature, 'records': len(records),
               'original_records': original_count, 'excluded_records': excluded,
               'raw_sensor_abs_limit': raw_sensor_abs_limit,
               'valid_VB': int(np.isfinite(table['VB']).sum()), 'sensor_fields': SENSORS,
               'statistics': STATISTICS, 'fitted_preprocessing': False})
    return table, signature
