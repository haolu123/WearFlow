"""Deterministic run-level summaries; definition matches the research release."""
import numpy as np

SENSORS = ('smcAC', 'smcDC', 'vib_table', 'vib_spindle', 'AE_table', 'AE_spindle')
STATISTICS = ('mean', 'std', 'rms', 'min', 'max', 'peak_to_peak', 'skewness', 'kurtosis', 'energy')


def sensor_features(record):
    result = []
    for name in SENSORS:
        x = np.asarray(record[name], dtype=np.float64).reshape(-1)
        if not len(x) or not np.isfinite(x).all():
            raise ValueError(f'Invalid {name} in case {record["case"]}, run {record["run"]}')
        mean, std = x.mean(), x.std()
        z = (x - mean) / std if std > 1e-12 else np.zeros_like(x)
        result.extend([mean, std, np.sqrt(np.mean(x*x)), x.min(), x.max(), np.ptp(x),
                       np.mean(z**3), np.mean(z**4), np.mean(x*x)])
    return np.asarray(result)
