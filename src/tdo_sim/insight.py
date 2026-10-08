"""Numerical diagnostics for declared finite Gaussian acquisition experiments.

This module claims no new theorem. Calibration keeps one latent head fixed
across known-label epochs and one test epoch. Rank-one background denotes a
fresh independent Gaussian amplitude EACH epoch; a subject-static background
must instead be included in the latent state/class means. Numerical intervals
cover integration draws conditional on the declared finite channel.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import brentq
from scipy.special import expit, logsumexp, ndtr
from .information import binary_entropy, gaussian_information


def _weights(probabilities, k):
    p = np.full(k, 1 / k) if probabilities is None else np.asarray(probabilities, float)
    if p.shape != (k,) or not np.all(np.isfinite(p)) or np.any(p < 0) or not np.isclose(p.sum(), 1, rtol=1e-12, atol=1e-12):
        raise ValueError("Require finite nonnegative normalized head probabilities")
    return p / p.sum()


def _covariance(covariance, dimension):
    c = np.asarray(covariance, float)
    if c.shape[-2:] != (dimension, dimension) or not np.all(np.isfinite(c)):
        raise ValueError("Covariance dimension/values invalid")
    if not np.allclose(c, np.swapaxes(c, -1, -2), rtol=1e-12, atol=1e-14):
        raise ValueError("Covariance must be symmetric")
    c = (c + np.swapaxes(c, -1, -2)) / 2
    np.linalg.cholesky(c)
    return c


def _summarize(values, value_range, confidence=.95):
    values = np.asarray(values, float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)) or not 0 < confidence < 1:
        raise ValueError("At least two finite independent outer draws required")
    lower, upper = value_range
    mean = float(values.mean())
    half = (upper - lower) * math.sqrt(math.log(2 / (1 - confidence)) / (2 * len(values)))
    return {"estimate": mean, "standard_error": float(values.std(ddof=1) / np.sqrt(len(values))),
            "confidence_lower": max(lower, mean - half), "confidence_upper": min(upper, mean + half),
            "confidence": confidence, "n_outer_draws": len(values),
            "interval": "pointwise Hoeffding conditional on declared model; empirical SE separate"}


def _information_summary(information, risk, confidence=.95):
    info = _summarize(information, (0., 1.), confidence)
    error = _summarize(risk, (0., .5), confidence)
    return {"bits": info["estimate"], "standard_error": info["standard_error"],
            "confidence_lower_bits": info["confidence_lower"], "confidence_upper_bits": info["confidence_upper"],
            "bayes_error": error["estimate"], "bayes_error_standard_error": error["standard_error"],
            "bayes_error_confidence_lower": error["confidence_lower"], "bayes_error_confidence_upper": error["confidence_upper"],
            "n_samples": info["n_outer_draws"], "confidence": confidence, "interval_scope": info["interval"]}


def rank_one_covariance(direction, standard_deviation, base=None):
    """Covariance of epochwise independent Gaussian background plus base noise."""
    b = np.asarray(direction, float)
    if b.ndim not in (1, 2) or not np.all(np.isfinite(b)):
        raise ValueError("Direction must be q or K×q")
    sd = np.asarray(standard_deviation, float)
    if np.any(sd < 0) or not np.all(np.isfinite(sd)):
        raise ValueError("Background SD must be finite and nonnegative")
    q = b.shape[-1]
    base = np.eye(q) if base is None else _covariance(base, q)
    return base + (sd ** 2)[..., None, None] * b[..., :, None] * b[..., None, :]


class GaussianHeadChannel:
    """Balanced binary Gaussian experiment with one shared finite head prior.

    Covariance can be common q×q or per-head K×q×q, and is common to the two
    labels within a head. Optional rank-one background requires a common base
    covariance and is independent each epoch. For that model, exact common-base
    whitening keeps the span of ALL mean differences AND background directions.
    Arbitrary varying covariances retain all observation coordinates.
    """

    def __init__(self, plus_means, minus_means, probabilities=None, covariance=None,
                 background_vectors=None, background_sd=0., head_ids=None):
        plus = np.asarray(plus_means, float)
        minus = np.asarray(minus_means, float)
        if plus.ndim == 1:
            plus = plus[:, None]
        if minus.ndim == 1:
            minus = minus[:, None]
        if plus.shape != minus.shape or plus.ndim != 2 or not plus.shape[0] or not plus.shape[1] or not np.all(np.isfinite(plus)) or not np.all(np.isfinite(minus)):
            raise ValueError("Require matching finite nonempty K×q class means")
        self.plus, self.minus = plus.copy(), minus.copy()
        self.n_heads, self.dimension = plus.shape
        self.probabilities = _weights(probabilities, self.n_heads)
        self.head_ids = None if head_ids is None else tuple(head_ids)
        if self.head_ids is not None and (len(self.head_ids) != self.n_heads or len(set(self.head_ids)) != self.n_heads):
            raise ValueError("Require unique head IDs matching means")
        covariance = np.eye(self.dimension) if covariance is None else covariance
        c = _covariance(covariance, self.dimension)
        if c.ndim not in (2, 3) or (c.ndim == 3 and len(c) != self.n_heads):
            raise ValueError("Covariance must be q×q or K×q×q")
        if c.ndim == 3 and np.array_equal(c, np.broadcast_to(c[0], c.shape)):
            c = c[0].copy()
        self.covariance = c
        self.background_vectors = None
        self.background_sd = None
        self._rank_one_vectors = None
        if background_vectors is not None:
            if c.ndim != 2:
                raise ValueError("Rank-one background requires common base covariance")
            b = np.asarray(background_vectors, float)
            if b.shape == (self.dimension,):
                b = np.broadcast_to(b, plus.shape).copy()
            sd = np.broadcast_to(np.asarray(background_sd, float), (self.n_heads,)).copy()
            if b.shape != plus.shape or not np.all(np.isfinite(b)) or np.any(sd < 0) or not np.all(np.isfinite(sd)):
                raise ValueError("Background dimensions/SD invalid")
            self.background_vectors, self.background_sd = b.copy(), sd
        self._draw_cholesky = np.linalg.cholesky(c)
        self.center = plus[0].copy() if c.ndim == 2 else np.zeros(self.dimension)
        if c.ndim == 2:
            base_whitener = solve_triangular(self._draw_cholesky, np.eye(self.dimension), lower=True)
            mean_white = (np.vstack((plus, minus)) - self.center) @ base_whitener.T
            rows = mean_white
            if self.background_vectors is not None:
                background_white = (self.background_vectors * self.background_sd[:, None]) @ base_whitener.T
                rows = np.vstack((rows, background_white))
            _, singular, vt = np.linalg.svd(rows, full_matrices=False)
            threshold = np.finfo(float).eps * max(rows.shape) * (singular[0] if len(singular) else 0.)
            rank = int(np.count_nonzero(singular > threshold))
            self.transform_matrix = vt[:rank] @ base_whitener
            self.rank = rank
            reconstructed = (rows @ vt[:rank].T) @ vt[:rank]
            self.reconstruction_error = float(np.linalg.norm(rows - reconstructed))
            means = (np.vstack((plus, minus)) - self.center) @ self.transform_matrix.T
            self._plus = means[:self.n_heads]
            self._minus = means[self.n_heads:]
            if self.background_vectors is None:
                density_cov = np.eye(rank)
            else:
                b_reduced = self.background_vectors @ self.transform_matrix.T
                density_cov = rank_one_covariance(b_reduced, self.background_sd)
                self._rank_one_vectors = b_reduced * self.background_sd[:, None]
        else:
            self.rank = self.dimension
            self.transform_matrix = np.eye(self.dimension)
            self.reconstruction_error = 0.
            self._plus, self._minus = plus, minus
            density_cov = c
        self._density_common = density_cov.ndim == 2
        if self.rank:
            chol = np.linalg.cholesky(density_cov)
            self._logdet = 2 * np.log(np.diagonal(chol, axis1=-2, axis2=-1)).sum(axis=-1)
            self._precision = np.linalg.inv(density_cov)
        else:
            self._logdet = 0.
            self._precision = np.empty((0, 0))
        self.metadata = {"original_dimension": self.dimension, "likelihood_dimension": self.rank,
                         "mean_background_span_reconstruction_error": self.reconstruction_error,
                         "reduction": "common-base exact mean/background span; numerical-zero singular values only" if c.ndim == 2 else "varying covariances; full coordinates retained",
                         "background_contract": "independent Gaussian amplitude each epoch; no shared unmodelled amplitude"}

    def _log_heads(self, samples, positive):
        x = (np.asarray(samples, float) - self.center) @ self.transform_matrix.T
        means = self._plus if positive else self._minus
        if x.ndim != 2 or x.shape[1] != self.rank:
            raise ValueError("Samples require N×q coordinates")
        if self.rank == 0:
            return np.zeros((len(x), self.n_heads))
        if self._rank_one_vectors is not None:
            # Sherman–Morrison: exact reduced covariance I+b_h b_h^T.
            # All work is N×K dot products; no N×K×q distance tensor.
            b = self._rank_one_vectors
            square = np.sum(x * x, axis=1)[:, None] - 2 * x @ means.T + np.sum(means * means, axis=1)[None, :]
            dot = x @ b.T - np.sum(means * b, axis=1)[None, :]
            denominator = 1 + np.sum(b * b, axis=1)
            return -.5 * (square - dot * dot / denominator[None, :] + np.log(denominator)[None, :])
        if self._density_common:
            precision_mean = means @ self._precision
            quadratic_x = np.sum((x @ self._precision) * x, axis=1)
            quadratic_mean = np.sum(precision_mean * means, axis=1)
            return -.5 * (quadratic_x[:, None] - 2 * x @ precision_mean.T + quadratic_mean[None, :] + self._logdet)
        output = np.empty((len(x), self.n_heads))
        for h in range(self.n_heads):
            diff = x - means[h]
            output[:, h] = -.5 * (np.sum((diff @ self._precision[h]) * diff, axis=1) + self._logdet[h])
        return output

    def posterior(self, samples, head_weights=None):
        samples = np.asarray(samples, float)
        if samples.ndim != 2 or samples.shape[1] != self.dimension or not np.all(np.isfinite(samples)):
            raise ValueError("Samples must be finite N×q")
        weights = self.probabilities if head_weights is None else np.asarray(head_weights, float)
        if weights.shape not in ((self.n_heads,), (len(samples), self.n_heads)) or np.any(weights < 0) or not np.all(np.isfinite(weights)) or not np.allclose(weights.sum(axis=-1), 1, rtol=1e-12, atol=1e-12):
            raise ValueError("Head weights must be normalized K or N×K")
        with np.errstate(divide="ignore"):
            logs = np.log(weights)
        positive = logsumexp(self._log_heads(samples, True) + logs, axis=1)
        negative = logsumexp(self._log_heads(samples, False) + logs, axis=1)
        return expit(positive - negative)

    def known_head_posterior(self, samples, head_indices):
        head_indices = np.asarray(head_indices, int)
        if head_indices.shape != (len(samples),) or np.any(head_indices < 0) or np.any(head_indices >= self.n_heads):
            raise ValueError("One head index per sample required")
        x = (np.asarray(samples) - self.center) @ self.transform_matrix.T
        difference = self._plus[head_indices] - self._minus[head_indices]
        residual = x - (self._plus[head_indices] + self._minus[head_indices]) / 2
        if self._rank_one_vectors is not None:
            b = self._rank_one_vectors[head_indices]
            ratio = np.sum(residual * difference, axis=1) - np.sum(residual * b, axis=1) * np.sum(difference * b, axis=1) / (1 + np.sum(b * b, axis=1))
        elif self._density_common:
            ratio = np.sum((residual @ self._precision) * difference, axis=1)
        else:
            ratio = np.empty(len(x))
            for h in np.unique(head_indices):
                selected = head_indices == h
                ratio[selected] = np.sum((residual[selected] @ self._precision[h]) * difference[selected], axis=1)
        return expit(ratio)

    def posterior_heads(self, calibration_samples, calibration_labels, prior=None):
        """Product epoch likelihood for ONE fixed head; labels are observed."""
        x, labels = np.asarray(calibration_samples, float), np.asarray(calibration_labels)
        single = x.ndim == 2
        if single:
            x, labels = x[None], labels[None]
        if x.ndim != 3 or x.shape[2] != self.dimension or labels.shape != x.shape[:2] or not np.all(np.isin(labels, [-1, 1])) or not np.all(np.isfinite(x)):
            raise ValueError("Require N×m×q calibration epochs and N×m labels ±1")
        p = self.probabilities if prior is None else _weights(prior, self.n_heads)
        with np.errstate(divide="ignore"):
            log_prior = np.log(p)
        if x.shape[1] == 0:
            result = np.broadcast_to(p, (len(x), self.n_heads)).copy()
        else:
            result = np.empty((len(x), self.n_heads))
            for start in range(0, len(x), 128):
                stop = min(start + 128, len(x))
                flat = x[start:stop].reshape(-1, self.dimension)
                plus = self._log_heads(flat, True).reshape(stop-start, x.shape[1], self.n_heads)
                minus = self._log_heads(flat, False).reshape(plus.shape)
                log_likelihood = np.where(labels[start:stop, :, None] > 0, plus, minus).sum(axis=1) + log_prior
                result[start:stop] = np.exp(log_likelihood - logsumexp(log_likelihood, axis=1, keepdims=True))
        return result[0] if single else result

    def draw(self, labels, head_indices, rng):
        labels, heads = np.asarray(labels), np.asarray(head_indices, int)
        if labels.ndim != 1 or heads.shape != labels.shape or not np.all(np.isin(labels, [-1, 1])) or np.any(heads < 0) or np.any(heads >= self.n_heads):
            raise ValueError("Require ±1 labels and valid head indices")
        means = np.where(labels[:, None] > 0, self.plus[heads], self.minus[heads])
        noise = rng.standard_normal(means.shape)
        if self.covariance.ndim == 2:
            noise = noise @ self._draw_cholesky.T
        else:
            noise = np.einsum("nj,nqj->nq", noise, self._draw_cholesky[heads])
        if self.background_vectors is not None:
            noise += rng.standard_normal((len(labels), 1)) * self.background_sd[heads, None] * self.background_vectors[heads]
        return means + noise

    def oracle_summary(self):
        contrast = self._plus - self._minus
        if self.rank == 0:
            d = np.zeros(self.n_heads)
        elif self._density_common:
            d = np.sqrt(np.maximum(0., np.sum((contrast @ self._precision) * contrast, axis=1)))
        else:
            d = np.sqrt(np.maximum(0., np.einsum("ki,kij,kj->k", contrast, self._precision, contrast)))
        information, risk = gaussian_information(d), ndtr(-d / 2)
        return {"separations": d.tolist(), "information_bits": information.tolist(),
                "bayes_errors": risk.tolist(),
                "average_information_bits": float(self.probabilities @ information),
                "average_bayes_error": float(self.probabilities @ risk),
                "scope": "head observed; arbitrary common midpoint allowed; covariance known within head"}

    def project(self, projection):
        """Push forward the actual channel, including all background covariance."""
        p = np.asarray(projection, float)
        if p.ndim != 2 or p.shape[1] != self.dimension or not p.shape[0] or np.linalg.matrix_rank(p) != p.shape[0]:
            raise ValueError("Projection must have independent rows and matching dimension")
        covariance = p @ self.covariance @ p.T
        background = None if self.background_vectors is None else self.background_vectors @ p.T
        return GaussianHeadChannel(self.plus @ p.T, self.minus @ p.T, self.probabilities, covariance,
                                   background, 0. if background is None else self.background_sd, self.head_ids)


def _batched_posterior(channel, samples, weights=None, batch_size=512):
    output = np.empty(len(samples))
    for start in range(0, len(samples), batch_size):
        stop = min(start + batch_size, len(samples))
        selected = weights[start:stop] if weights is not None and np.ndim(weights) == 2 else weights
        output[start:stop] = channel.posterior(samples[start:stop], selected)
    return output


def _paired_result(channel, samples, heads, projections, weights, confidence, batch_size=512):
    full_p = _batched_posterior(channel, samples, weights, batch_size)
    known_p = np.empty(len(samples))
    for start in range(0, len(samples), batch_size):
        stop = min(start + batch_size, len(samples))
        known_p[start:stop] = channel.known_head_posterior(samples[start:stop], heads[start:stop])
    full_i, known_i = 1 - binary_entropy(full_p), 1 - binary_entropy(known_p)
    full_r, known_r = np.minimum(full_p, 1 - full_p), np.minimum(known_p, 1 - known_p)
    output = {"full": _information_summary(full_i, full_r, confidence),
              "known_head_paired": _information_summary(known_i, known_r, confidence),
              "known_head_exact": channel.oracle_summary(), "projections": {}, "projection_pairwise": {},
              "acquisition_information_loss": _summarize(known_i - full_i, (-1., 1.), confidence),
              "acquisition_bayes_risk_increase": _summarize(full_r - known_r, (-.5, .5), confidence)}
    projected_values = {}
    for name, p in (projections or {}).items():
        projected = channel.project(p)
        posterior = _batched_posterior(projected, samples @ np.asarray(p).T, weights, batch_size)
        info, risk = 1 - binary_entropy(posterior), np.minimum(posterior, 1 - posterior)
        projected_values[name] = (info, risk)
        output["projections"][name] = {**_information_summary(info, risk, confidence),
                                      "paired_information_loss": _summarize(full_i - info, (-1., 1.), confidence),
                                      "paired_bayes_risk_increase": _summarize(risk - full_r, (-.5, .5), confidence),
                                      "metadata": projected.metadata}
    names = list(projected_values)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            li, lr = projected_values[left]
            ri, rr = projected_values[right]
            output["projection_pairwise"][left + "__minus__" + right] = {
                "left": left, "right": right,
                "information_difference": _summarize(li - ri, (-1., 1.), confidence),
                "bayes_risk_difference": _summarize(lr - rr, (-.5, .5), confidence)}
    output["metadata"] = channel.metadata
    output["pairing"] = "same heads, labels and observations for oracle/full/projected estimators"
    return output


def paired_information(channel, projections=None, n_samples=8192, seed=0, confidence=.95, batch_size=512):
    """Actual finite-prior MI/risk and paired acquisition/representation losses."""
    if n_samples < 2 or batch_size < 1:
        raise ValueError("Require at least two samples and positive batch size")
    rng = np.random.default_rng(seed)
    heads = rng.choice(channel.n_heads, n_samples, p=channel.probabilities)
    labels = rng.integers(0, 2, n_samples) * 2 - 1
    samples = channel.draw(labels, heads, rng)
    # Modest protocol dimensions permit one array; density work uses only exact
    # mean/background sufficient coordinates and N×K dot products.
    result = _paired_result(channel, samples, heads, projections, None, confidence, batch_size)
    result.update({"seed": seed, "estimand": "I(Y;X) under supplied finite head prior; head identity hidden"})
    return result


def mixture_information(channel, head_weights=None, n_samples=8192, seed=0, confidence=.95):
    """Actual information for a supplied fixed finite head law."""
    if head_weights is None:
        return paired_information(channel, n_samples=n_samples, seed=seed, confidence=confidence)["full"]
    weights = _weights(head_weights, channel.n_heads)
    copied = GaussianHeadChannel(channel.plus, channel.minus, weights, channel.covariance,
                                 channel.background_vectors, 0. if channel.background_sd is None else channel.background_sd,
                                 channel.head_ids)
    return paired_information(copied, n_samples=n_samples, seed=seed, confidence=confidence)["full"]


def calibration_information(channel, n_calibration, projections=None, n_samples=8192, seed=0, confidence=.95):
    """Exact-prior I(Ytest;Xtest | known-label calibration from the SAME head).

    Every independent outer draw resets the head. Calibration and test share it;
    the Gaussian epoch noises/background amplitudes are independent. Projections
    affect the test observation only: all methods receive the same full
    calibration record. Calibration information is not a learned-prior score.
    """
    if n_calibration < 0 or int(n_calibration) != n_calibration or n_samples < 2:
        raise ValueError("Nonnegative integer calibration count and >=2 draws required")
    n_calibration = int(n_calibration)
    rng = np.random.default_rng(seed)
    heads = rng.choice(channel.n_heads, n_samples, p=channel.probabilities)
    calibration_labels = rng.integers(0, 2, (n_samples, n_calibration)) * 2 - 1
    calibration_samples = channel.draw(calibration_labels.ravel(), np.repeat(heads, n_calibration), rng).reshape(n_samples, n_calibration, channel.dimension)
    weights = channel.posterior_heads(calibration_samples, calibration_labels)
    labels = rng.integers(0, 2, n_samples) * 2 - 1
    samples = channel.draw(labels, heads, rng)
    result = _paired_result(channel, samples, heads, projections, weights, confidence)
    zero_p = _batched_posterior(channel, samples)
    conditional_i = 1 - binary_entropy(_batched_posterior(channel, samples, weights))
    zero_i = 1 - binary_entropy(zero_p)
    result["paired_calibration_information_gain"] = _summarize(conditional_i - zero_i, (-1., 1.), confidence)
    result["head_posterior_entropy_bits"] = float(np.mean(-np.sum(np.where(weights > 0, weights * np.log2(np.maximum(weights, 1e-300)), 0), axis=1)))
    result.update({"n_calibration": n_calibration, "seed": seed,
                   "estimand": "exact supplied finite-prior I(Ytest;Xtest|C,Lcal); same head across epochs",
                   "projection_side_information": "full calibration record retained for every test projection"})
    return result


def evaluate_calibrated_prior(prior_channel, evaluation_channel, n_calibration,
                              n_subjects=200, test_epochs=32, seed=0, confidence=.95, probability_clip=1e-12):
    """Actual predictions from a development prior on a separate evaluation law.

    These are predictive risk/log loss, NOT true MI from plug-in posterior
    entropy. Subject means are the independent units for SE/CI; test epochs
    within a subject share one head and calibration record.
    """
    if prior_channel.dimension != evaluation_channel.dimension or n_calibration < 0 or int(n_calibration) != n_calibration or n_subjects < 2 or test_epochs < 1 or not 0 < probability_clip < .5:
        raise ValueError("Invalid held-out evaluation controls")
    verified = prior_channel.head_ids is not None and evaluation_channel.head_ids is not None
    if verified and set(prior_channel.head_ids).intersection(evaluation_channel.head_ids):
        raise ValueError("Development and evaluation head IDs overlap")
    rng = np.random.default_rng(seed)
    heads = rng.choice(evaluation_channel.n_heads, n_subjects, p=evaluation_channel.probabilities)
    n_calibration = int(n_calibration)
    cal_labels = rng.integers(0, 2, (n_subjects, n_calibration)) * 2 - 1
    cal = evaluation_channel.draw(cal_labels.ravel(), np.repeat(heads, n_calibration), rng).reshape(n_subjects, n_calibration, evaluation_channel.dimension)
    weights = prior_channel.posterior_heads(cal, cal_labels)
    labels = rng.integers(0, 2, (n_subjects, test_epochs)) * 2 - 1
    samples = evaluation_channel.draw(labels.ravel(), np.repeat(heads, test_epochs), rng)
    predicted = _batched_posterior(prior_channel, samples, np.repeat(weights, test_epochs, axis=0)).reshape(labels.shape)
    clipped = np.clip(predicted, probability_clip, 1 - probability_clip)
    true_prob = np.where(labels > 0, clipped, 1 - clipped)
    errors = ((predicted >= .5) != (labels > 0)).mean(axis=1)
    losses = -np.log2(true_prob).mean(axis=1)
    error_summary = _summarize(errors, (0., 1.), confidence)
    loss_summary = _summarize(losses, (0., -math.log2(probability_clip)), confidence)
    return {"error": error_summary, "log_loss_bits": loss_summary,
            "empirical_information_lower_score_bits": 1 - loss_summary["estimate"],
            "information_lower_score_scope": "expected 1-crossentropy is a lower bound for the evaluation experiment; empirical score is not exact MI",
            "n_subjects": n_subjects, "test_epochs": test_epochs, "n_calibration": n_calibration, "seed": seed,
            "independent_unit": "subject draw including its fixed head/calibration and test epochs",
            "disjoint_head_ids_verified": verified, "estimand": "held-out predictive risk/log loss of supplied development prior",
            "head_indices": heads.tolist(), "labels": labels.tolist(), "predicted_probabilities": predicted.tolist(),
            "log_loss_probability_clip": probability_clip,
            "posterior_head_weights": weights.tolist()}


def fit_projection(method, dimension, signal_samples, covariance=None, target_contrasts=None):
    """Fit a matched-dimension linear projection from declared design data only.

    'pca' centers design rows; 'svd' preserves their origin; 'target' first
    includes the contrast span and fills remaining dimensions with design SVD
    directions. All bases are orthonormal in base-noise white coordinates.
    """
    signal = np.asarray(signal_samples, float)
    if signal.ndim != 2 or not np.all(np.isfinite(signal)) or not 1 <= dimension <= signal.shape[1]:
        raise ValueError("Require finite design rows and valid feature dimension")
    if method not in ("pca", "svd", "target"):
        raise ValueError("Unknown projection method")
    q = signal.shape[1]
    covariance = np.eye(q) if covariance is None else _covariance(covariance, q)
    if covariance.ndim != 2:
        raise ValueError("Projection design requires one declared base covariance")
    whitener = solve_triangular(np.linalg.cholesky(covariance), np.eye(q), lower=True)
    white = signal @ whitener.T
    fit_rows = white - white.mean(axis=0) if method == "pca" else white
    _, _, design_vt = np.linalg.svd(fit_rows, full_matrices=True)
    if method != "target":
        return design_vt[:dimension] @ whitener
    targets = np.asarray(target_contrasts, float)
    if targets.ndim == 1:
        targets = targets[None]
    if targets.ndim != 2 or targets.shape[1] != q or not np.all(np.isfinite(targets)):
        raise ValueError("Target-aware projection requires declared finite contrasts")
    target_white = targets @ whitener.T
    _, singular, target_vt = np.linalg.svd(target_white, full_matrices=False)
    threshold = np.finfo(float).eps * max(target_white.shape) * (singular[0] if len(singular) else 0)
    rank = int(np.sum(singular > threshold))
    # A target span larger than the budget is approximated by its leading SVD;
    # this is an explicit representation loss, not an exact sufficient statistic.
    selected = [row.copy() for row in target_vt[:min(rank, dimension)]]
    for row in design_vt:
        if len(selected) == dimension:
            break
        residual = row.copy()
        for old in selected:
            residual -= old * (old @ residual)
        for old in selected:
            residual -= old * (old @ residual)
        norm = np.linalg.norm(residual)
        if norm > 1e-10:
            selected.append(residual / norm)
    return np.stack(selected) @ whitener


def structured_score(errors, shape, ball_radius):
    """Guaranteed ellipsoid+ball decomposition gauge (conservative, not exact).

    e=B(B†e)+e_perp and s=max(||B†e||,||e_perp||/tau) imply
    e in s*(B unitball + tau unitball). Shape is development-fitted and fixed.
    """
    e, b = np.asarray(errors, float), np.asarray(shape, float)
    if e.ndim == 1:
        e = e[None]
    if b.ndim != 2 or e.ndim != 2 or e.shape[1] != b.shape[0] or ball_radius < 0 or not np.all(np.isfinite(e)) or not np.all(np.isfinite(b)):
        raise ValueError("Invalid shape/errors/ball radius")
    coefficients = e @ np.linalg.pinv(b).T
    residual = e - coefficients @ b.T
    perpendicular = np.linalg.norm(residual, axis=1)
    if ball_radius == 0:
        tail = np.where(perpendicular <= 1e-12 * (1 + np.linalg.norm(e, axis=1)), 0., np.inf)
    else:
        tail = perpendicular / ball_radius
    return np.maximum(np.linalg.norm(coefficients, axis=1), tail)


def structured_information_bounds(center, shape, ball_radius, quantile, certificate_tolerance=1e-9):
    """Convex nearest-ellipsoid plus ball calculation with support certificate.

    Never treats a primal approximate minimum as an information floor.
    Any unit separator gives the stated support lower bound; the feasible
    primal point gives an upper bound on minimum distance and their gap audits
    numerical optimality. The maximum-distance bound is conservative.
    """
    c, b = np.asarray(center, float), np.asarray(shape, float)
    if c.ndim != 1 or b.ndim != 2 or b.shape[0] != len(c) or not np.all(np.isfinite(c)) or not np.all(np.isfinite(b)) or ball_radius < 0 or quantile < 0:
        raise ValueError("Invalid ellipsoid geometry")
    a = quantile * b
    u, singular, vt = np.linalg.svd(a, full_matrices=False)
    threshold = np.finfo(float).eps * max(a.shape) * (singular[0] if len(singular) else 0.)
    good = singular > threshold
    coefficients = u[:, good].T @ c
    s = singular[good]
    unconstrained = -coefficients / s if len(s) else np.empty(0)
    if np.linalg.norm(unconstrained) <= 1:
        reduced_z = unconstrained
        multiplier = 0.
    else:
        def equation(multiplier):
            return float(np.sum((s * coefficients / (s * s + multiplier)) ** 2) - 1)
        upper = max(1., float(np.linalg.norm(s * coefficients)))
        while equation(upper) > 0:
            upper *= 2
        multiplier = float(brentq(equation, 0., upper, xtol=1e-14, rtol=1e-14))
        reduced_z = -s * coefficients / (s * s + multiplier)
    z = vt[good].T @ reduced_z if len(s) else np.zeros(b.shape[1])
    # Protect the feasible witness from secular-root rounding.
    z /= max(1., float(np.linalg.norm(z)))
    ellipsoid_point = c + a @ z
    ellipsoid_distance = float(np.linalg.norm(ellipsoid_point))
    rho = quantile * ball_radius
    if rho and ellipsoid_distance:
        v = -ellipsoid_point / max(rho, ellipsoid_distance)
    else:
        v = np.zeros_like(c)
    primal_point = ellipsoid_point + rho * v
    primal_distance = float(np.linalg.norm(primal_point))
    h = ellipsoid_point / ellipsoid_distance if ellipsoid_distance else np.zeros_like(c)
    support_value = float(h @ c - quantile * (np.linalg.norm(b.T @ h) + ball_radius))
    support_lower = max(0., support_value)
    gap = max(0., primal_distance - support_lower)
    max_distance = float(np.linalg.norm(c) + quantile * (np.linalg.norm(b, ord=2) + ball_radius))
    feasibility = bool(np.linalg.norm(z) <= 1 + 1e-12 and np.linalg.norm(v) <= 1 + 1e-12)
    return {"minimum_distance_lower_certificate": support_lower,
            "minimum_distance_primal_upper": primal_distance, "optimizer_gap": gap,
            "optimizer_certificate_passed": bool(feasibility and gap <= certificate_tolerance * (1 + np.linalg.norm(c))),
            "primal_feasible": feasibility, "ellipsoid_coefficient": z.tolist(), "ball_coefficient": v.tolist(),
            "separator": h.tolist(), "support_value": support_value, "secular_multiplier": multiplier,
            "maximum_distance_upper": max_distance,
            "lower_bits": float(gaussian_information(support_lower)),
            "upper_bits": float(gaussian_information(max_distance)),
            "fixed_separator_error_upper": float(ndtr(-support_lower / 2)),
            "decoder": "unit separator at zero threshold" if support_value > 0 else "constant balanced-class prediction",
            "scope": "centered fixed-common-covariance contrast set; support certificate in floating-point arithmetic; physical sharpness not asserted"}
