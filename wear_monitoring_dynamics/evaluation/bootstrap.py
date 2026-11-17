import numpy as np


def paired_case_bootstrap(left, right, repeats=10000, seed=0):
    """Inputs are one MAE per held-out case after within-case averaging over seeds."""
    a, b = left.align(right, join='inner')
    if len(a) != len(left) or len(b) != len(right) or len(a) < 2:
        raise ValueError('Paired comparison requires identical case sets, at least two cases')
    delta = (a-b).to_numpy()
    means = np.random.default_rng(seed).choice(delta, size=(repeats, len(delta)), replace=True).mean(axis=1)
    return {'n_cases': len(delta), 'delta_mae': float(delta.mean()),
            'ci_low': float(np.quantile(means, .025)), 'ci_high': float(np.quantile(means, .975))}
