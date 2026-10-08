"""New, isolated numerical utilities for the frozen full simulation study.

Existing core modules and the pilot are deliberately untouched. These helpers
implement bounded finite-mixture integration, exact head-count intervals, and
the audited tolerance/partial-content contract. They claim no new theorem.
"""
from __future__ import annotations
from dataclasses import replace
import math
import numpy as np
from scipy.special import expit, logsumexp, ndtr
from scipy.stats import beta as beta_distribution
from .forward import HeadParameters, make_spherical_forward
from .information import binary_entropy, gaussian_information


def dct_basis(n, modes=None):
    j = np.arange(n) + .5
    k = np.arange(n if modes is None else modes)
    output = np.sqrt(2 / n) * np.cos(np.pi * k[:, None] * j[None, :] / n)
    output[0] /= np.sqrt(2)
    return output


def head_seed(bank_seed, severity_index, head_index):
    """Nonoverlapping severity blocks for the frozen maximum327 heads/bank."""
    if not 0 <= head_index < 1000 or not 0 <= severity_index < 1000:
        raise ValueError("Seed block bounds exceeded")
    return int(bank_seed + severity_index * 1000000 + (head_index + 1) * 1000)


def make_targets(geometry, config):
    output = {}
    positions = geometry.source_positions_m
    for name, definition in config["target_definitions"].items():
        width = definition["width_m"]
        patches = [np.exp(-np.sum((positions - np.asarray(c)) ** 2, axis=1) / (2 * width ** 2))
                   for c in definition["centres_m"]]
        vector = patches[0] - patches[1] if len(patches) == 2 else patches[0]
        source = config["source_contrast_l2_am"] * vector / np.linalg.norm(vector)
        waveform = dct_basis(config["n_times"])[definition["dct_index"]]
        output[name] = (source, waveform)
    return output


def generate_head(geometry, perturbations, severity, seed):
    """Independent synthetic draw with contacts on the actual outer conductor."""
    rng = np.random.default_rng(seed)
    center = rng.uniform(-1, 1, 3) * perturbations["center_uniform_halfwidth_m"] * severity
    radius = .09 * (1 + rng.uniform(-1, 1) * perturbations["radius_uniform_fraction"] * severity)
    brain = 1 + rng.uniform(-1, 1) * perturbations["brain_conductivity_uniform_fraction"] * severity
    skull = 1 + rng.uniform(-1, 1) * perturbations["skull_conductivity_uniform_fraction"] * severity
    angle = rng.uniform(-1, 1) * perturbations["sensor_rotation_uniform_halfwidth_degrees"] * severity
    radians = np.deg2rad(angle)
    rotation = np.array([[np.cos(radians), -np.sin(radians), 0],
                         [np.sin(radians), np.cos(radians), 0], [0, 0, 1]])
    sensors = geometry.sensor_positions_m @ rotation.T
    direction = sensors / np.linalg.norm(sensors, axis=1)[:, None]
    jitter = rng.uniform(-1, 1, sensors.shape) * perturbations["sensor_tangential_uniform_halfwidth_m"] * severity
    tangential = jitter - np.sum(jitter * direction, axis=1)[:, None] * direction
    sensors += tangential
    sensors *= (radius / np.linalg.norm(sensors, axis=1))[:, None]
    sensors += center
    perturbed_geometry = replace(geometry, sensor_positions_m=sensors)
    parameters = HeadParameters(head_radius_m=radius, center_m=tuple(center),
                                conductivities_s_m=(.33 * brain, 1., .004 * skull, .33))
    model = make_spherical_forward(perturbed_geometry, parameters)
    metadata = {"seed": seed, "severity": severity, "radius_m": radius,
                "center_m": center.tolist(), "brain_conductivity_multiplier": brain,
                "skull_conductivity_multiplier": skull, "cap_rotation_degrees": angle,
                "maximum_tangential_jitter_m": float(np.max(np.linalg.norm(tangential, axis=1))),
                "maximum_contact_surface_error_m": float(np.max(np.abs(np.linalg.norm(sensors - center, axis=1) - radius)))}
    return model.gain_v_per_am, metadata


def centered_mixture_information(contrasts, probabilities=None, n_samples=4096,
                                 seed=0, batch_size=256, confidence=.95):
    """Exact finite-mixture density, bounded marginal posterior-entropy MC.

    Input contrasts are already in white coordinates. Full spatial dimensions
    are retained; no SVD truncation occurs. A dot-product identity avoids the
    N×K×q distance array. The output CI is conditional on the supplied finite
    bank and covers integration uncertainty only.
    """
    contrasts = np.asarray(contrasts, dtype=float)
    if contrasts.ndim != 2 or not np.all(np.isfinite(contrasts)) or n_samples < 2:
        raise ValueError("Require finite K×q contrasts and at least two draws")
    k, dimension = contrasts.shape
    probabilities = np.full(k, 1 / k) if probabilities is None else np.asarray(probabilities, dtype=float)
    if probabilities.shape != (k,) or np.any(probabilities <= 0) or not np.isclose(probabilities.sum(), 1):
        raise ValueError("Positive normalized K probabilities required")
    if not 0 < confidence < 1 or batch_size < 1:
        raise ValueError("Invalid integration controls")
    rng = np.random.default_rng(seed)
    acquisition = rng.choice(k, n_samples, p=probabilities)
    labels = rng.integers(0, 2, n_samples) * 2 - 1
    information = np.empty(n_samples)
    risk = np.empty(n_samples)
    log_weights = np.log(probabilities) - np.sum(contrasts ** 2, axis=1) / 8
    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        samples = labels[start:stop, None] * contrasts[acquisition[start:stop]] / 2
        samples += rng.standard_normal(samples.shape)
        dot = samples @ contrasts.T / 2
        positive = logsumexp(log_weights[None, :] + dot, axis=1)
        negative = logsumexp(log_weights[None, :] - dot, axis=1)
        posterior = expit(positive - negative)
        information[start:stop] = 1 - binary_entropy(posterior)
        risk[start:stop] = np.minimum(posterior, 1 - posterior)
    value = float(information.mean())
    error = float(risk.mean())
    halfwidth = math.sqrt(math.log(2 / (1 - confidence)) / (2 * n_samples))
    return {"bits": value, "standard_error": float(information.std(ddof=1) / np.sqrt(n_samples)),
            "hoeffding_lower_bits": max(0., value - halfwidth), "hoeffding_upper_bits": min(1., value + halfwidth),
            "bayes_error": error, "bayes_error_standard_error": float(risk.std(ddof=1) / np.sqrt(n_samples)),
            "bayes_error_hoeffding_lower": max(0., error - halfwidth / 2),
            "bayes_error_hoeffding_upper": min(.5, error + halfwidth / 2),
            "n_samples": n_samples, "seed": seed, "components": k, "white_dimension": dimension,
            "method": "exact centered finite-mixture likelihood; bounded posterior-entropy Monte Carlo; no spatial rank truncation"}


def partial_content_bounds(d0, radius, content):
    """Audited established robust-testing/Fano/indicator bounds, in bits."""
    if d0 < 0 or radius < 0 or not 0 <= content <= 1:
        raise ValueError("Invalid separation, radius, or covered content")
    minus, plus = max(0., d0 - radius), d0 + radius
    lower, upper = float(gaussian_information(minus)), float(gaussian_information(plus))
    miss = 1 - content
    p = float(ndtr(-minus / 2))
    error_cap = min(.5, content * p + miss)
    indicator_floor = max(0., content * lower - float(binary_entropy(miss)))
    fano_floor = max(0., 1 - float(binary_entropy(error_cap)))
    return {"conditional_lower_bits": lower, "conditional_upper_bits": upper,
            "conditional_decoder_error_cap": p, "content": content, "miss_fraction": miss,
            "partial_indicator_floor_bits": indicator_floor, "partial_fano_floor_bits": fano_floor,
            "partial_lower_bits": max(indicator_floor, fano_floor),
            "partial_upper_bits": content * upper + miss,
            "partial_decoder_error_cap": error_cap}


def maximum_content_confidence(n, content=.95):
    if n < 1 or not 0 < content < 1:
        raise ValueError("Positive sample size and interior content required")
    return 1 - content ** n


def clopper_pearson(successes, trials, confidence=.95):
    if not 0 <= successes <= trials or trials < 1:
        raise ValueError("Invalid binomial counts")
    alpha = 1 - confidence
    lower = 0. if successes == 0 else float(beta_distribution.ppf(alpha / 2, successes, trials - successes + 1))
    upper = 1. if successes == trials else float(beta_distribution.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return lower, upper


def sample_feature_oracle(feature_contrast, full_d, n_samples, seed):
    """Joint epoch draws for fixed projected features and full known-head oracle."""
    rng = np.random.default_rng(seed)
    feature_contrast = np.asarray(feature_contrast)
    labels = np.r_[np.ones(n_samples // 2), -np.ones(n_samples - n_samples // 2)]
    rng.shuffle(labels)
    noise = rng.standard_normal((n_samples, len(feature_contrast)))
    correlation = feature_contrast / full_d if full_d else np.zeros_like(feature_contrast)
    remainder = max(0., 1 - float(correlation @ correlation))
    oracle_noise = noise @ correlation + np.sqrt(remainder) * rng.standard_normal(n_samples)
    features = labels[:, None] * feature_contrast[None, :] / 2 + noise
    oracle_statistic = labels * full_d / 2 + oracle_noise
    return features, labels, oracle_statistic
