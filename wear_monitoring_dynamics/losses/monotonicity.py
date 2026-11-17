"""Soft nondecreasing current-wear estimates along observed training trajectories."""


def monotonicity_loss(model, batch, pairs, tolerance_z=0.0):
    """Adjacent same-case observations, standardized VB; no label direction used.

    Pair construction is split-safe and chronological in data.preprocessing.
    Sensor/condition inputs vary with the observation, unlike a partial-t loss.
    """
    if not len(pairs):
        raise ValueError('Monotonicity loss needs at least one training pair')
    earlier = model.estimate(batch, pairs[:, 0])
    later = model.estimate(batch, pairs[:, 1])
    return (earlier - later - tolerance_z).relu().square().mean()
