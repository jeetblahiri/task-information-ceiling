"""Probe-span audit and target-specific spectral calibration radii."""

import numpy as np
from scipy.special import ndtr
from .coordinates import covariance_whitener
from .information import gaussian_information


def probe_error_score(operator_error: np.ndarray, probes: np.ndarray,
                      covariance: np.ndarray) -> float:
    """ε=||W E P||op: largest singular value, not eigenvalue spectral radius."""
    return float(np.linalg.norm(covariance_whitener(covariance) @ operator_error @ probes, ord=2))


def _probe_coefficients(delta: np.ndarray, probes: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    delta = np.asarray(delta, dtype=float)
    probes = np.asarray(probes, dtype=float)
    if probes.ndim != 2 or delta.shape != (probes.shape[0],):
        raise ValueError("Probe matrix and contrast source dimension mismatch")
    coefficients = np.linalg.pinv(probes) @ delta
    remainder = delta - probes @ coefficients
    scale = max(np.linalg.norm(delta), np.linalg.norm(probes) * np.linalg.norm(coefficients), 1e-300)
    tolerance = max(probes.shape) * np.finfo(float).eps * scale * 20
    in_span = bool(np.linalg.norm(remainder) <= tolerance)
    return coefficients, remainder, in_span


def probe_information_bounds(nominal_gain: np.ndarray, delta: np.ndarray,
                             probes: np.ndarray, covariance: np.ndarray, epsilon: float,
                             *, midpoint_known: bool = True, common_covariance: bool = True,
                             acquisition_independent: bool = True,
                             temporal_factor: float = 1.0) -> dict:
    """I8 for centered Gaussian tasks, with unrestricted linear probe uncertainty.

    Temporal factor is ||Wt g|| for a separable contrast delta·g.T. This exact
    matched temporal projection scales both d0 and the target radius. It does
    not turn a general nonseparable matrix target into an exact uncertainty ball.
    """
    if epsilon < 0 or not np.isfinite(epsilon) or temporal_factor < 0 or not np.isfinite(temporal_factor):
        raise ValueError("Finite nonnegative epsilon and temporal factor required")
    coefficients, remainder, in_span = _probe_coefficients(delta, probes)
    nominal = covariance_whitener(covariance) @ np.asarray(nominal_gain) @ delta
    d0 = float(np.linalg.norm(nominal) * temporal_factor)
    amplification = float(np.linalg.norm(coefficients)) if in_span else None
    radius = float(epsilon * amplification * temporal_factor) if in_span else None
    reasons = []
    if not in_span:
        reasons.append("target outside probe span; no declared remainder restriction")
    if not midpoint_known:
        reasons.append("midpoint unknown or acquisition dependent")
    if not common_covariance:
        reasons.append("covariance differs across acquisitions or labels")
    if not acquisition_independent:
        reasons.append("acquisition law is not label independent")
    applicable = not reasons
    if applicable:
        lower_d, upper_d = max(0, d0 - radius), d0 + radius
        lower, upper = gaussian_information(lower_d), gaussian_information(upper_d)
        minimax_error = float(ndtr(-lower_d / 2))
    else:
        lower, upper, minimax_error = 0.0, 1.0, None
        # Unknown midpoint alone invalidates the floor, but a contrast ceiling
        # remains valid under common covariance and independent acquisition.
        if in_span and common_covariance and acquisition_independent:
            upper = gaussian_information(d0 + radius)
    return {
        "equation": "I8", "applicable": applicable, "reasons": reasons,
        "nominal_separation": d0, "epsilon": float(epsilon),
        "in_probe_span": in_span, "probe_amplification": amplification,
        "target_radius": radius, "source_remainder_norm": float(np.linalg.norm(remainder)),
        "lower_bits": float(lower), "upper_bits": float(upper),
        "centered_minimax_error": minimax_error, "temporal_factor": float(temporal_factor),
        "norm_model": "spatial ||Ws E P||op≤epsilon, fixed separable waveform; radius=epsilon*kappa*||Wt g||",
        "sharp_for": "unrestricted calibrated linear uncertainty; possibly conservative for physical sphere family",
    }


def endpoint_operator(nominal_gain: np.ndarray, delta: np.ndarray, probes: np.ndarray,
                      covariance: np.ndarray, epsilon: float, endpoint: str) -> np.ndarray:
    """Construct an admitted E attaining a nearest/farthest whitened contrast.

    This is an algebraic witness for the unrestricted uncertainty set; it is
    generally not a member of the physical spherical-head family.
    """
    if endpoint not in ("lower", "upper") or epsilon < 0:
        raise ValueError("Endpoint must be lower/upper, with nonnegative epsilon")
    w = covariance_whitener(covariance)
    a, _, in_span = _probe_coefficients(delta, probes)
    if not in_span:
        raise ValueError("Endpoint construction requires an in-span target")
    u0 = w @ nominal_gain @ delta
    d0 = np.linalg.norm(u0)
    kappa = np.linalg.norm(a)
    if kappa == 0:
        return np.zeros_like(nominal_gain)
    direction = u0 / d0 if d0 > 0 else np.eye(len(u0))[0]
    radius = epsilon * kappa
    z = direction * (radius if endpoint == "upper" else -min(radius, d0))
    b = np.outer(z, a) / (a @ a)
    return np.linalg.solve(w, b @ np.linalg.pinv(probes))


def interval_from_separation_radius(d0: float, radius: float, *, midpoint_known: bool = True,
                                    common_covariance: bool = True,
                                    acquisition_independent: bool = True) -> dict:
    """Generic Gaussian contrast-ball interval; caller declares its norm model.

    For a full source-time identity probe matrix use its actual epsilon and
    coefficient norm to construct radius *before* calling this helper. No
    temporal factor is silently applied here.
    """
    if d0 < 0 or radius < 0 or not np.isfinite(d0) or not np.isfinite(radius):
        raise ValueError("Finite nonnegative d0 and radius required")
    valid = midpoint_known and common_covariance and acquisition_independent
    upper = gaussian_information(d0 + radius) if common_covariance and acquisition_independent else 1.0
    return {"nominal_separation": float(d0), "target_radius": float(radius),
            "applicable": bool(valid),
            "lower_bits": gaussian_information(max(0, d0 - radius)) if valid else 0.0,
            "upper_bits": float(upper),
            "centered_minimax_error": float(ndtr(-max(0, d0 - radius) / 2)) if valid else None}
