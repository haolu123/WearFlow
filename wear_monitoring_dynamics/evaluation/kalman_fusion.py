"""Scalar EKF update in physical mm, with optional monotone mean projection."""
import numpy as np


def update(prior, variance, observation, noise, previous, valid=True, monotone=False):
    if variance < 0 or noise <= 0:
        raise ValueError('Invalid covariance')
    gain = variance / (variance + noise) if valid else 0.
    posterior = prior + gain * (observation - prior) if valid else prior
    # Joseph form remains nonnegative, including very small observation noise.
    covariance = (1-gain)**2 * variance + gain**2 * noise
    clipped = bool(monotone and posterior < previous)
    return max(previous, posterior) if monotone else posterior, covariance, gain, clipped

