"""Full-rank reference coordinates and declared covariance transformations."""

import numpy as np
from scipy.linalg import cholesky, solve_triangular


def car_basis(n_channels: int) -> np.ndarray:
    """Helmert basis U: U.T U=I and U.T 1=0; CAR data are U.T V.

    Returning n by n-1 coordinates avoids inverting singular CAR covariances.
    """
    if n_channels < 2:
        raise ValueError("At least two electrodes are needed for reference-free differences")
    u = np.zeros((n_channels, n_channels - 1))
    for j in range(n_channels - 1):
        scale = np.sqrt((j + 1) * (j + 2))
        u[:j + 1, j] = 1 / scale
        u[j + 1, j] = -(j + 1) / scale
    return u


def covariance_whitener(covariance: np.ndarray) -> np.ndarray:
    """Return W with W Sigma W.T=I via Cholesky, never a singular inverse.

    W need not be the symmetric square root; its Euclidean/spectral norms are
    identical to symmetric whitening up to an orthogonal coordinate rotation.
    """
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or not np.all(np.isfinite(cov)):
        raise ValueError("Covariance must be a finite square matrix")
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=np.finfo(float).eps * max(np.max(np.abs(cov)), 1e-30)):
        raise ValueError("Covariance must be symmetric")
    lower = cholesky(cov, lower=True, check_finite=True)
    return solve_triangular(lower, np.eye(len(cov)), lower=True, check_finite=False)


def observed_forward(raw_gain: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Restrict a fixed montage before applying its orthonormal CAR coordinates."""
    index = np.asarray(indices, dtype=int)
    if len(np.unique(index)) != len(index):
        raise ValueError("Electrode indices must be unique")
    return car_basis(len(index)).T @ np.asarray(raw_gain)[index]


def observed_covariance(raw_covariance: np.ndarray, indices: np.ndarray) -> np.ndarray:
    index = np.asarray(indices, dtype=int)
    u = car_basis(len(index))
    return u.T @ np.asarray(raw_covariance)[np.ix_(index, index)] @ u


def separation(delta: np.ndarray, covariance: np.ndarray) -> float:
    """Mahalanobis mean separation d, whose oracle MI is F(d)."""
    return float(np.linalg.norm(covariance_whitener(covariance) @ np.asarray(delta, dtype=float)))


def separable_whiten_epoch(epoch: np.ndarray, spatial_covariance: np.ndarray,
                          temporal_covariance: np.ndarray) -> np.ndarray:
    """Whiten channel×time arrays under Cs ⊗ Ct with C-order flattening."""
    ws = covariance_whitener(spatial_covariance)
    wt = covariance_whitener(temporal_covariance)
    return np.einsum("ij,...jk,lk->...il", ws, np.asarray(epoch), wt)
