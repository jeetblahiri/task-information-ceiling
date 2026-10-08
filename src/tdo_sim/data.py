"""Balanced simulated epochs and shared label-independent acquisition mixtures."""

import numpy as np
from scipy.linalg import cholesky


def sample_gaussian_mixture(plus_means: np.ndarray, minus_means: np.ndarray,
                           probabilities: np.ndarray, covariance: np.ndarray,
                           n_samples: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return x,y,a; y has exactly equal ±1 counts, and a is drawn independently.

    Finite empirical acquisition counts fluctuate by label as expected from
    independent draws. Odd sample counts are rejected rather than unbalancing.
    """
    from .information import MixtureChannel
    if n_samples < 2 or n_samples % 2:
        raise ValueError("Balanced simulation requires a positive even sample count")
    channel = MixtureChannel(plus_means, minus_means, probabilities, covariance)
    rng = np.random.default_rng(seed)
    y = np.repeat([-1, 1], n_samples // 2)
    rng.shuffle(y)
    acquisition = rng.choice(len(channel.probabilities), n_samples, p=channel.probabilities)
    means = np.where(y[:, None] == 1, channel.original_plus[acquisition], channel.original_minus[acquisition])
    noise = rng.normal(size=(n_samples, means.shape[1])) @ cholesky(channel.covariance, lower=True).T
    return means + noise, y, acquisition
