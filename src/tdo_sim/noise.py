"""Covariances declared by the simulator, not estimated from recordings."""

import numpy as np
from scipy.spatial.distance import cdist


def sensor_covariance(positions_m: np.ndarray, standard_deviation_v: float = 1e-6,
                      correlation_fraction: float = 0.25,
                      length_scale_m: float = 0.04) -> np.ndarray:
    """SPD raw-electrode covariance: independent noise + smooth spatial kernel."""
    if standard_deviation_v <= 0 or not 0 <= correlation_fraction < 1 or length_scale_m <= 0:
        raise ValueError("Require positive noise/length scales and 0≤fraction<1")
    distance = cdist(positions_m, positions_m)
    kernel = np.exp(-distance ** 2 / (2 * length_scale_m ** 2))
    return standard_deviation_v ** 2 * ((1 - correlation_fraction) * np.eye(len(distance))
                                      + correlation_fraction * kernel)


def temporal_covariance(n_times: int, ar1: float = 0.0) -> np.ndarray:
    """Dimensionless AR(1) correlation; Cs⊗Ct carries units V²."""
    if n_times < 1 or abs(ar1) >= 1:
        raise ValueError("Require n_times≥1 and |ar1|<1")
    index = np.arange(n_times)
    return ar1 ** np.abs(index[:, None] - index[None, :])
