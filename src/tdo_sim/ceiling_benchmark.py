"""Physical preprocessing and information references for Gaussian task epochs.

Epochs have shape N x C x T; flattened coordinates use sensor-major C order.
The base covariance is Cs (x) Ct. A fixed raw spatial/time map is applied
BEFORE whitening, with the complete pushed-forward covariance. The true-bank
affine-mean/background span below is a likelihood-only numerical reduction,
never a fitted encoder or a feature supplied to an operational learner.

Targets are balanced and independent of the supplied finite latent-state law.
State is fixed across a subject's epochs. Optional rank-one Gaussian background
has a fresh independent amplitude each epoch; static background belongs in the
means. There is zero subject calibration and no revealed latent-state identity.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import ndtr

from .coordinates import car_basis
from .decoders import train_linear, train_mlp
from .information import binary_entropy, gaussian_information
from .insight import GaussianHeadChannel


def _copy(array):
    value = np.asarray(array, dtype=float).copy()
    value.setflags(write=False)
    return value


def _positive_integer(value, name, minimum=1):
    if not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _covariance(value, dimension, name):
    array = np.asarray(value, float)
    if array.shape != (dimension, dimension) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} covariance must be finite {dimension} x {dimension}")
    # Relative symmetry check remains meaningful for volt-scale covariances.
    scale = max(float(np.max(np.abs(array))), np.finfo(float).tiny)
    if np.max(np.abs(array - array.T)) > 1e-12 * scale:
        raise ValueError(f"{name} covariance must be symmetric")
    array = (array + array.T) / 2
    try:
        chol = np.linalg.cholesky(array)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} covariance must be positive definite") from exc
    return _copy(array), _copy(chol)


def _encoder(value, input_dimension, name):
    array = np.asarray(value, float)
    if array.ndim != 2 or array.shape[1] != input_dimension or not 0 < array.shape[0] <= input_dimension or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have finite nonempty independent rows")
    # Row rescaling is invertible. Avoid calling a small but independent row
    # information-free merely because its physical units have a small scale.
    scale = np.max(np.abs(array), axis=1)
    if np.any(scale == 0):
        raise ValueError(f"{name} contains a zero row")
    normalized = array / scale[:, None]
    singular = np.linalg.svd(normalized, compute_uv=False)
    tolerance = np.finfo(float).eps * max(normalized.shape) * singular[0]
    if singular[-1] <= tolerance:
        raise ValueError(f"{name} rows are dependent or numerically ambiguous; supply a stable full-rank rowspace basis, without silently truncating nonzero directions")
    return _copy(array), {"rows": len(array), "rank": len(array),
                          "row_normalized_minimum_singular_value": float(singular[-1]),
                          "rank_validation_tolerance": float(tolerance),
                          "rank_policy": "reject dependencies/ambiguity; no encoder singular values dropped"}


def _apply(epochs, spatial, temporal):
    """R_s X T_b^T without a materialized Kronecker matrix."""
    return np.einsum("ac,nct,bt->nab", spatial, epochs, temporal, optimize=True)


def _whiten(epochs, spatial_cholesky, temporal_cholesky):
    array = np.asarray(epochs, float)
    if array.ndim != 3 or array.shape[1:] != (len(spatial_cholesky), len(temporal_cholesky)) or not np.all(np.isfinite(array)):
        raise ValueError("Epochs must be finite N x C x T in declared observation coordinates")
    n, c, t = array.shape
    spatial = solve_triangular(spatial_cholesky, array.transpose(1, 0, 2).reshape(c, n * t), lower=True)
    spatial = spatial.reshape(c, n, t).transpose(1, 0, 2)
    return solve_triangular(temporal_cholesky, spatial.reshape(n * c, t).T, lower=True).T.reshape(n, c, t)


def block_average_matrix(n_times, block_size):
    """Contiguous nonoverlapping arithmetic means, retaining a final short block.

    Output block i covers [i*block_size, min((i+1)*block_size,n_times)).
    Weights are 1/(actual block length), with no signal/noise renormalization.
    """
    n_times = _positive_integer(n_times, "n_times")
    block_size = _positive_integer(block_size, "block_size")
    result = np.zeros((math.ceil(n_times / block_size), n_times))
    for i, start in enumerate(range(0, n_times, block_size)):
        stop = min(n_times, start + block_size)
        result[i, start:stop] = 1 / (stop - start)
    return result


def channel_average_reference_matrix(n_channels, groups):
    """Map full CAR coordinates to averages followed by output CAR.

    Each group is a nonempty sequence of distinct raw sensor indices. Groups
    can overlap, but the resulting map must have independent rows. For an
    ordinary channel subset, use singleton groups. With A the group-average
    matrix, returns Q_k.T @ A @ Q_m and A for an independent physical audit.
    """
    n_channels = _positive_integer(n_channels, "n_channels", 2)
    if len(groups) < 2:
        raise ValueError("Output CAR requires at least two channel groups")
    average = np.zeros((len(groups), n_channels))
    for row, group in enumerate(groups):
        indices = np.asarray(group)
        if indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer) or len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= n_channels):
            raise ValueError("Channel groups require nonempty distinct valid integer indices")
        average[row, indices] = 1 / len(indices)
    result = car_basis(len(groups)).T @ average @ car_basis(n_channels)
    _encoder(result, n_channels - 1, "channel average/reference")
    return result, average


def summarize_bounded(values, value_range, confidence=.95, multiplicity=1):
    """Two separate valid bounded-draw intervals and an empirical SE.

    Empirical Bernstein uses Maurer-Pontil's sample-variance inequality on
    both tails: sqrt(2*s2*log(4M/alpha)/N)+7*w*log(4M/alpha)/(3*(N-1)).
    Each interval separately spends alpha/M. Their intersection is NOT
    advertised at the same confidence. Draws must be independent outer units.
    """
    array = np.asarray(values, float)
    if array.ndim != 1 or len(array) < 2 or not np.all(np.isfinite(array)) or not 0 < confidence < 1:
        raise ValueError("At least two finite independent outer draws and valid confidence required")
    multiplicity = _positive_integer(multiplicity, "multiplicity")
    low, high = map(float, value_range)
    width = high - low
    if not np.isfinite(width) or width <= 0 or np.any(array < low - 1e-12 * max(1., abs(low))) or np.any(array > high + 1e-12 * max(1., abs(high))):
        raise ValueError("Values must lie in declared nonempty finite range")
    mean, variance = float(array.mean()), float(array.var(ddof=1))
    alpha = (1 - confidence) / multiplicity
    hoeffding = width * math.sqrt(math.log(2 / alpha) / (2 * len(array)))
    bernstein = math.sqrt(2 * variance * math.log(4 / alpha) / len(array)) + 7 * width * math.log(4 / alpha) / (3 * (len(array) - 1))
    return {"estimate": mean, "standard_error": math.sqrt(variance / len(array)),
            "confidence_lower": max(low, mean - hoeffding), "confidence_upper": min(high, mean + hoeffding),
            "empirical_bernstein_lower": max(low, mean - bernstein), "empirical_bernstein_upper": min(high, mean + bernstein),
            "hoeffding_halfwidth": hoeffding, "empirical_bernstein_halfwidth": bernstein,
            "confidence": confidence, "multiplicity": multiplicity,
            "n_outer_draws": len(array), "value_range": [low, high],
            "interval_scope": "conditional on declared finite law; Hoeffding and empirical Bernstein are separate bounded-draw intervals"}


def summarize_information(information, risk, confidence=.95):
    info = summarize_bounded(information, (0., 1.), confidence)
    error = summarize_bounded(risk, (0., .5), confidence)
    return {"bits": info["estimate"], "standard_error": info["standard_error"],
            "confidence_lower_bits": info["confidence_lower"], "confidence_upper_bits": info["confidence_upper"],
            "empirical_bernstein_lower_bits": info["empirical_bernstein_lower"], "empirical_bernstein_upper_bits": info["empirical_bernstein_upper"],
            "bayes_error": error["estimate"], "bayes_error_standard_error": error["standard_error"],
            "bayes_error_confidence_lower": error["confidence_lower"], "bayes_error_confidence_upper": error["confidence_upper"],
            "bayes_error_empirical_bernstein_lower": error["empirical_bernstein_lower"], "bayes_error_empirical_bernstein_upper": error["empirical_bernstein_upper"],
            "n_samples": len(information), "confidence": confidence,
            "interval_scope": info["interval_scope"]}


class PreparedGaussianExperiment:
    """Separable raw Gaussian epoch law and exact finite-state likelihood.

    Uses only two small Cholesky solves. The likelihood span includes ALL
    whitened affine class-mean and rank-one background directions. This span
    can use the true law because it is never an operational feature design.
    """

    def __init__(self, plus, minus, spatial_covariance, temporal_covariance, *,
                 probabilities=None, background=None, background_sd=1., head_ids=None):
        plus, minus = np.asarray(plus, float), np.asarray(minus, float)
        if plus.shape != minus.shape or plus.ndim != 3 or not all(plus.shape) or not np.all(np.isfinite(plus)) or not np.all(np.isfinite(minus)):
            raise ValueError("Matching finite nonempty K x C x T class means required")
        self.plus, self.minus = _copy(plus), _copy(minus)
        self.n_heads, self.n_channels, self.n_times = plus.shape
        self.dimension = self.n_channels * self.n_times
        self.spatial_covariance, self.spatial_cholesky = _covariance(spatial_covariance, self.n_channels, "Spatial")
        self.temporal_covariance, self.temporal_cholesky = _covariance(temporal_covariance, self.n_times, "Temporal")
        weights = np.full(self.n_heads, 1 / self.n_heads) if probabilities is None else np.asarray(probabilities, float)
        if weights.shape != (self.n_heads,) or not np.all(np.isfinite(weights)) or np.any(weights < 0) or not np.isclose(weights.sum(), 1, rtol=1e-12, atol=1e-12):
            raise ValueError("Finite nonnegative normalized head probabilities required")
        self.probabilities = _copy(weights / weights.sum())
        self.head_ids = None if head_ids is None else tuple(head_ids)
        if self.head_ids is not None and (len(self.head_ids) != self.n_heads or len(set(self.head_ids)) != self.n_heads):
            raise ValueError("Unique head IDs matching the means required")
        if background is None:
            self.background = None
            self.background_sd = _copy(np.zeros(self.n_heads))
        else:
            background = np.asarray(background, float)
            if background.shape == plus.shape[1:]:
                background = np.broadcast_to(background, plus.shape)
            try:
                sd = np.broadcast_to(np.asarray(background_sd, float), (self.n_heads,))
            except ValueError as exc:
                raise ValueError("Background SD must be scalar or one per head") from exc
            if background.shape != plus.shape or not np.all(np.isfinite(background)) or not np.all(np.isfinite(sd)) or np.any(sd < 0):
                raise ValueError("Background must be finite C x T or K x C x T, with nonnegative SD")
            self.background, self.background_sd = _copy(background), _copy(sd)
        self.white_plus = _copy(self.whiten(self.plus))
        self.white_minus = _copy(self.whiten(self.minus))
        self.white_background = None if self.background is None else _copy(self.whiten(self.background) * self.background_sd[:, None, None])
        self.white_center = _copy(self.white_plus[0])
        mean_rows = np.concatenate((self.white_plus - self.white_center, self.white_minus - self.white_center)).reshape(2 * self.n_heads, -1)
        rows = mean_rows if self.white_background is None else np.vstack((mean_rows, self.white_background.reshape(self.n_heads, -1)))
        _, singular, vt = np.linalg.svd(rows, full_matrices=False)
        threshold = np.finfo(float).eps * max(rows.shape) * (singular[0] if len(singular) else 0.)
        rank = int(np.count_nonzero(singular > threshold))
        self.likelihood_basis = _copy(vt[:rank])
        self.likelihood_rank = rank
        residual = rows - (rows @ self.likelihood_basis.T) @ self.likelihood_basis
        reconstruction = float(np.linalg.norm(residual))
        scale = float(np.linalg.norm(rows))
        means = mean_rows @ self.likelihood_basis.T
        if rank:
            vectors = None if self.white_background is None else self.white_background.reshape(self.n_heads, -1) @ self.likelihood_basis.T
            self.channel = GaussianHeadChannel(means[:self.n_heads], means[self.n_heads:], self.probabilities,
                background_vectors=vectors, background_sd=1., head_ids=self.head_ids)
        else:
            self.channel = None
        self.metadata = {"observation_dimension": self.dimension, "observation_shape": [self.n_channels, self.n_times],
            "vectorization": "sensor-major C order; covariance Cs kron Ct",
            "likelihood_dimension": rank, "likelihood_rank_tolerance": float(threshold),
            "affine_mean_background_span_reconstruction_error": reconstruction,
            "affine_mean_background_span_relative_error": reconstruction / scale if scale else 0.,
            "likelihood_reduction": "true-law affine class-mean AND background span; numerical-zero SVD only; not an encoder",
            "background_contract": "fresh independent Gaussian rank-one amplitude each epoch; static background included in state means",
            "side_information": "declared task/noise/montage; zero subject calibration; hidden fixed state; Bayes knows finite state law",
            "state_law_scope": "supplied finite bank, not continuous acquisition/source generator"}
        self.transform_metadata = None

    def whiten(self, epochs):
        return _whiten(epochs, self.spatial_cholesky, self.temporal_cholesky)

    def color(self, white_epochs):
        return _apply(np.asarray(white_epochs, float), self.spatial_cholesky, self.temporal_cholesky)

    def likelihood_coordinates(self, white_epochs):
        epochs = np.asarray(white_epochs, float)
        if epochs.ndim != 3 or epochs.shape[1:] != (self.n_channels, self.n_times) or not np.all(np.isfinite(epochs)):
            raise ValueError("White epochs must match finite N x C x T")
        return (epochs - self.white_center).reshape(len(epochs), -1) @ self.likelihood_basis.T

    def posterior_white(self, white_epochs):
        coordinates = self.likelihood_coordinates(white_epochs)
        return np.full(len(coordinates), .5) if self.channel is None else self.channel.posterior(coordinates)

    def posterior_raw(self, epochs):
        return self.posterior_white(self.whiten(epochs))

    def known_head_posterior_raw(self, epochs, head_indices):
        return self.known_head_posterior_white(self.whiten(epochs), head_indices)

    def known_head_posterior_white(self, white_epochs, head_indices):
        coordinates = self.likelihood_coordinates(white_epochs)
        heads = _heads(head_indices, len(coordinates), self.n_heads)
        return np.full(len(heads), .5) if self.channel is None else self.channel.known_head_posterior(coordinates, heads)

    def oracle_summary(self):
        # Direct Sherman-Morrison preserves the raw full epoch oracle and does
        # not rely on numerical removal of any affine-span singular direction.
        delta = (self.white_plus - self.white_minus).reshape(self.n_heads, -1)
        square = np.sum(delta * delta, axis=1)
        if self.white_background is not None:
            vectors = self.white_background.reshape(self.n_heads, -1)
            square -= np.sum(delta * vectors, axis=1) ** 2 / (1 + np.sum(vectors * vectors, axis=1))
        distances = np.sqrt(np.maximum(0., square))
        info, errors = gaussian_information(distances), ndtr(-distances / 2)
        return {"separations": distances.tolist(), "information_bits": np.asarray(info).tolist(),
                "bayes_errors": errors.tolist(), "average_information_bits": float(self.probabilities @ info),
                "average_bayes_error": float(self.probabilities @ errors),
                "scope": "entire latent state observed; arbitrary within-state midpoint; exact rank-one covariance update"}


def prepared_experiment(plus, minus, spatial_covariance, temporal_covariance, *,
                        probabilities=None, background=None, background_sd=1., head_ids=None):
    return PreparedGaussianExperiment(plus, minus, spatial_covariance, temporal_covariance,
        probabilities=probabilities, background=background, background_sd=background_sd, head_ids=head_ids)


def prepare_transform_experiment(experiment, R_s, T_b, *, name="transform"):
    """Push forward raw means, separable covariance, and background together."""
    spatial, spatial_audit = _encoder(R_s, experiment.n_channels, "R_s")
    temporal, temporal_audit = _encoder(T_b, experiment.n_times, "T_b")
    transformed = prepared_experiment(_apply(experiment.plus, spatial, temporal), _apply(experiment.minus, spatial, temporal),
        spatial @ experiment.spatial_covariance @ spatial.T, temporal @ experiment.temporal_covariance @ temporal.T,
        probabilities=experiment.probabilities,
        background=None if experiment.background is None else _apply(experiment.background, spatial, temporal),
        background_sd=experiment.background_sd, head_ids=experiment.head_ids)
    transformed.R_s, transformed.T_b = spatial, temporal
    transformed.white_spatial_map = _copy(solve_triangular(transformed.spatial_cholesky, spatial @ experiment.spatial_cholesky, lower=True))
    transformed.white_temporal_map = _copy(solve_triangular(transformed.temporal_cholesky, temporal @ experiment.temporal_cholesky, lower=True))
    transformed.transform_metadata = {"name": str(name), "spatial": spatial_audit, "temporal": temporal_audit,
        "output_features": len(spatial) * len(temporal),
        "operation": "fixed raw R_s X T_b.T, then covariance-propagating likelihood whitening",
        "spatial_whitening_identity_error": float(np.linalg.norm(transformed.white_spatial_map @ transformed.white_spatial_map.T - np.eye(len(spatial)))),
        "temporal_whitening_identity_error": float(np.linalg.norm(transformed.white_temporal_map @ transformed.white_temporal_map.T - np.eye(len(temporal))))}
    transformed.metadata["raw_transform"] = transformed.transform_metadata
    return transformed


def _heads(value, n_samples, n_heads):
    array = np.asarray(value)
    if array.shape != (n_samples,) or not np.issubdtype(array.dtype, np.integer) or np.any(array < 0) or np.any(array >= n_heads):
        raise ValueError("One valid integer head index per epoch required")
    return array.astype(int, copy=False)


@dataclass(frozen=True)
class EpochDraw:
    epochs: np.ndarray
    labels: np.ndarray
    head_indices: np.ndarray
    subject_indices: np.ndarray | None = None


def draw_raw_epochs(experiment, n_samples, seed=0, *, head_indices=None, labels=None, subject_indices=None):
    """Raw epochs with state fixed by each supplied head index.

    Hidden head/subject indices are audit fields, NEVER predictor features.
    Supplying repeated head indices generates independent epoch noise and
    background with the same fixed mean/source/static-background state.
    """
    n_samples = _positive_integer(n_samples, "n_samples")
    streams = [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(4)]
    heads = streams[0].choice(experiment.n_heads, n_samples, p=experiment.probabilities) if head_indices is None else _heads(head_indices, n_samples, experiment.n_heads)
    labels = streams[1].integers(0, 2, n_samples) * 2 - 1 if labels is None else np.asarray(labels)
    if labels.shape != (n_samples,) or not np.all(np.isin(labels, [-1, 1])):
        raise ValueError("One +/-1 label per epoch required")
    white_noise = streams[2].standard_normal((n_samples, experiment.n_channels, experiment.n_times))
    epochs = np.where(labels[:, None, None] > 0, experiment.plus[heads], experiment.minus[heads]).copy()
    epochs += experiment.color(white_noise)
    if experiment.background is not None:
        amplitudes = streams[3].standard_normal(n_samples) * experiment.background_sd[heads]
        epochs += amplitudes[:, None, None] * experiment.background[heads]
    if subject_indices is not None:
        subject_indices = np.asarray(subject_indices)
        if subject_indices.shape != (n_samples,) or not np.issubdtype(subject_indices.dtype, np.integer):
            raise ValueError("Subject audit indices must be one integer per epoch")
        # A subject is one persistent complete state, not a head redrawn each
        # epoch. This catches accidental per-epoch state resampling.
        for subject in np.unique(subject_indices):
            if len(np.unique(heads[subject_indices == subject])) != 1:
                raise ValueError("Each subject must retain one fixed head/source/background state")
    return EpochDraw(epochs, labels.astype(int), heads.copy(), None if subject_indices is None else subject_indices.copy())


def paired_task_information(experiment, transforms=None, *, n_samples=8192, seed=0,
                            batch_size=256, confidence=.95, return_sample_arrays=False,
                            include_pairwise=False):
    """True finite-bank full/encoded MI and Bayes risks on identical draws.

    Empty transforms use direct exact-sufficient-coordinate draws, avoiding
    native C*T noise arrays. With encoders, bounded white-epoch batches retain
    the same physical noise realization across ALL raw transforms. Draw RNG
    streams are separate, making samples reproducible across batch sizes.
    """
    n_samples = _positive_integer(n_samples, "n_samples", 2)
    batch_size = _positive_integer(batch_size, "batch_size")
    representations = {name: prepare_transform_experiment(experiment, *transform, name=name) for name, transform in (transforms or {}).items()}
    if any(not isinstance(name, str) or name in ("full", "known_head") for name in representations):
        raise ValueError("Representation names must be strings distinct from full/known_head")
    streams = [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(4)]
    heads = streams[0].choice(experiment.n_heads, n_samples, p=experiment.probabilities)
    labels = streams[1].integers(0, 2, n_samples) * 2 - 1
    information = {name: np.empty(n_samples) for name in ("full", "known_head", *representations)}
    risks = {name: np.empty(n_samples) for name in information}
    for start in range(0, n_samples, batch_size):
        stop = min(n_samples, start + batch_size)
        h, y, n = heads[start:stop], labels[start:stop], stop - start
        if not representations:
            if experiment.channel is None:
                probabilities = {"full": np.full(n, .5), "known_head": np.full(n, .5)}
            else:
                channel = experiment.channel
                # GaussianHeadChannel.draw interleaves its random streams;
                # draw here to preserve batching-invariant Gaussian samples.
                coordinates = np.where(y[:, None] > 0, channel.plus[h], channel.minus[h]).copy()
                coordinates += streams[2].standard_normal(coordinates.shape)
                if channel.background_vectors is not None:
                    coordinates += streams[3].standard_normal(n)[:, None] * channel.background_vectors[h] * channel.background_sd[h, None]
                probabilities = {"full": channel.posterior(coordinates), "known_head": channel.known_head_posterior(coordinates, h)}
        else:
            white = np.where(y[:, None, None] > 0, experiment.white_plus[h], experiment.white_minus[h]).copy()
            white += streams[2].standard_normal(white.shape)
            if experiment.white_background is not None:
                white += streams[3].standard_normal(n)[:, None, None] * experiment.white_background[h]
            probabilities = {"full": experiment.posterior_white(white), "known_head": experiment.known_head_posterior_white(white, h)}
            for name, channel in representations.items():
                compressed = _apply(white, channel.white_spatial_map, channel.white_temporal_map)
                probabilities[name] = channel.posterior_white(compressed)
        for name, posterior in probabilities.items():
            information[name][start:stop] = 1 - binary_entropy(posterior)
            risks[name][start:stop] = np.minimum(posterior, 1 - posterior)
    losses, summaries = {}, {}
    for name, channel in representations.items():
        losses[name] = {"information_loss": summarize_bounded(information["full"] - information[name], (-1., 1.), confidence),
                        "bayes_risk_increase": summarize_bounded(risks[name] - risks["full"], (-.5, .5), confidence)}
        summaries[name] = {**summarize_information(information[name], risks[name], confidence),
                           "paired_information_loss": losses[name]["information_loss"],
                           "paired_bayes_risk_increase": losses[name]["bayes_risk_increase"],
                           "known_head_exact": channel.oracle_summary(), "metadata": channel.metadata}
    pairwise = {}
    for left, right in (combinations(representations, 2) if include_pairwise else ()):
        pairwise[left + "__minus__" + right] = {"left": left, "right": right,
            "information_difference": summarize_bounded(information[left] - information[right], (-1., 1.), confidence),
            "bayes_risk_difference": summarize_bounded(risks[left] - risks[right], (-.5, .5), confidence)}
    result = {"full": summarize_information(information["full"], risks["full"], confidence),
              "known_head": summarize_information(information["known_head"], risks["known_head"], confidence),
              "known_head_exact": experiment.oracle_summary(), "representations": summaries,
              "losses": losses, "pairwise": pairwise,
              "acquisition_information_loss": summarize_bounded(information["known_head"] - information["full"], (-1., 1.), confidence),
              "acquisition_bayes_risk_increase": summarize_bounded(risks["full"] - risks["known_head"], (-.5, .5), confidence),
              "metadata": experiment.metadata, "seed": seed,
              "sampling": "direct likelihood-sufficient coordinates" if not representations else "bounded common-white epoch batches; all raw transforms paired",
              "pairing": "same heads, labels, Gaussian noise and epoch-background amplitude",
              "estimand": "balanced target MI/risk under supplied finite hidden-state law; zero calibration"}
    if return_sample_arrays:
        result["sample_arrays"] = {"labels": labels, "head_indices": heads,
            "full": {"information": information["full"], "risk": risks["full"]},
            "known_head": {"information": information["known_head"], "risk": risks["known_head"]},
            "representations": {name: {"information": information[name], "risk": risks[name]} for name in representations}}
    return result


def _prediction_metrics(draw, probabilities, *, confidence=.95, probability_clip=1e-6):
    labels, p = np.asarray(draw.labels), np.asarray(probabilities, float)
    if p.shape != labels.shape or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)) or not 0 < probability_clip < .5:
        raise ValueError("Finite prediction probabilities in [0,1] and clip in (0,.5) required")
    error = (np.where(p >= .5, 1, -1) != labels).astype(float)
    clipped = np.clip(p, probability_clip, 1 - probability_clip)
    log_loss_bits = -np.where(labels > 0, np.log2(clipped), np.log2(1 - clipped))
    subject = np.arange(len(labels)) if draw.subject_indices is None else np.asarray(draw.subject_indices)
    units = np.unique(subject)
    errors = np.array([error[subject == unit].mean() for unit in units])
    losses = np.array([log_loss_bits[subject == unit].mean() for unit in units])
    if len(units) < 2:
        raise ValueError("At least two independent evaluation subject units required")
    e = summarize_bounded(errors, (0., 1.), confidence)
    loss = summarize_bounded(losses, (0., -math.log2(probability_clip)), confidence)
    # Each confidence construction is separate. These are predictive lower
    # bounds and can be negative; a fitted posterior entropy is never called MI.
    error_interval = (e["confidence_lower"], e["confidence_upper"])
    maximum_entropy = float(binary_entropy(.5 if error_interval[0] <= .5 <= error_interval[1] else min(error_interval, key=lambda x: abs(x - .5))))
    return {"error": e, "log_loss_bits": loss,
            "empirical_balanced_error": float(np.mean([error[labels == label].mean() for label in [-1, 1]])) if set(np.unique(labels)) == {-1, 1} else None,
            "epoch_error": float(error.mean()), "epoch_error_standard_error_conditional": float(error.std(ddof=1) / np.sqrt(len(error))),
            "predictor_information_lower_score_bits": 1 - loss["estimate"],
            "predictor_information_lower_bound_hoeffding_bits": 1 - loss["confidence_upper"],
            "fano_information_lower_bound_hoeffding_bits": 1 - maximum_entropy,
            "information_scope": "cross-entropy/Fano LOWER bounds for fitted predictor; not exact MI",
            "uncertainty_units": "independent subjects (equal subject weights); epoch SE is a separate conditional diagnostic",
            "probability_clip": probability_clip, "n_epochs": len(labels), "n_subjects": len(units),
            "subject_ids": units, "per_subject_error": errors, "per_subject_log_loss_bits": losses,
            "predicted_probabilities": p, "labels": labels}


def matched_decoder_benchmark(training, validation, evaluation, *, transform=None,
                              evaluation_experiment=None, ridges=(.01, .1, 1.),
                              mlp_l2=(.001, .01), mlp_kwargs=None, seed=0,
                              confidence=.95, probability_clip=1e-6):
    """Fit LDA/MLP on matching RAW features, select using validation only.

    All three inputs are EpochDraw. Identity/state metadata is audit-only. An
    optional true evaluation-law Bayes risk sees identical transformed test
    epochs and zero calibration, but has a known-law parameter advantage.
    MLP epochs use validation log loss; outer grids use balanced validation
    error. No evaluation labels/outcomes enter normalization or model choice.
    Returns the actual model objects so callers can persist fitted parameters.
    """
    draws = [training, validation, evaluation]
    if any(not isinstance(draw, EpochDraw) for draw in draws):
        raise ValueError("Training, validation and evaluation must be EpochDraw")
    shape = training.epochs.shape[1:]
    if len(shape) != 2 or any(draw.epochs.shape[1:] != shape for draw in draws):
        raise ValueError("All raw epoch observation shapes must match")
    if any(draw.subject_indices is not None and len(draw.subject_indices) != len(draw.labels) for draw in draws):
        raise ValueError("Subject indices must match epoch labels")
    if transform is None:
        spatial, temporal = np.eye(shape[0]), np.eye(shape[1])
    else:
        spatial, _ = _encoder(transform[0], shape[0], "decoder R_s")
        temporal, _ = _encoder(transform[1], shape[1], "decoder T_b")
    features = [_apply(draw.epochs, spatial, temporal).reshape(len(draw.labels), -1) for draw in draws]
    def balanced(labels, probabilities):
        if set(np.unique(labels)) != {-1, 1}:
            raise ValueError("Validation requires both classes")
        error = np.where(probabilities >= .5, 1, -1) != labels
        return float(np.mean([error[labels == label].mean() for label in [-1, 1]]))
    if not len(ridges) or not len(mlp_l2):
        raise ValueError("Nonempty predeclared hyperparameter grids required")
    linear_models = [train_linear(features[0], training.labels, ridge=float(ridge)) for ridge in ridges]
    linear_errors = [balanced(validation.labels, model.predict_proba(features[1])) for model in linear_models]
    linear_index = int(np.argmin(linear_errors))
    settings = dict(mlp_kwargs or {})
    forbidden = {"x_train", "y_train", "x_validation", "y_validation", "l2", "seed"} & settings.keys()
    if forbidden:
        raise ValueError("MLP kwargs cannot override data, l2 or seed")
    mlp_models = [train_mlp(features[0], training.labels, features[1], validation.labels,
                           l2=float(l2), seed=seed + i, **settings) for i, l2 in enumerate(mlp_l2)]
    mlp_errors = [balanced(validation.labels, model.predict_proba(features[1])) for model in mlp_models]
    mlp_index = int(np.argmin(mlp_errors))
    selected = {"linear": linear_models[linear_index], "mlp": mlp_models[mlp_index]}
    result = {name: _prediction_metrics(evaluation, model.predict_proba(features[2]), confidence=confidence, probability_clip=probability_clip) for name, model in selected.items()}
    result.update({"models": selected, "selection": {"linear_ridges": list(ridges), "linear_validation_balanced_errors": linear_errors,
                   "selected_ridge": float(ridges[linear_index]), "mlp_l2": list(mlp_l2), "mlp_validation_balanced_errors": mlp_errors,
                   "selected_mlp_l2": float(mlp_l2[mlp_index]), "epoch_selection": "MLP validation log loss with early stopping; outer grids validation balanced error; first grid entry wins exact ties"},
                   "metadata": {"feature_dimension": features[0].shape[1], "latent_indices_used_as_features": False,
                   "subject_calibration_epochs": 0, "bayes_advantage": "known true evaluation distribution versus finite development training; state identity hidden for both",
                   "operation": "same fixed raw transform; training-only fitted normalization"}})
    if evaluation_experiment is not None:
        if evaluation_experiment.plus.shape[1:] != shape:
            raise ValueError("Bayes evaluation law must match raw epoch shape")
        law = prepare_transform_experiment(evaluation_experiment, spatial, temporal, name="decoder representation")
        probabilities = law.posterior_raw(_apply(evaluation.epochs, spatial, temporal))
        result["bayes_reference_predictions"] = _prediction_metrics(evaluation, probabilities, confidence=confidence, probability_clip=probability_clip)
        result["bayes_reference_predictions"]["information_scope"] = "realized Bayes decisions/log loss on exactly matched test epochs; separate integration computes exact-law MI"
    return result
