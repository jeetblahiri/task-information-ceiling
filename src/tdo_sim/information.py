"""Oracle Gaussian information and actual hidden-acquisition mixture information."""

from dataclasses import dataclass, asdict
from functools import lru_cache
from itertools import product
import math
import numpy as np
from scipy.special import expit, logsumexp, ndtr, roots_hermitenorm, xlogy
from .coordinates import covariance_whitener


@lru_cache(maxsize=16)
def _normal_nodes(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = roots_hermitenorm(order)
    return nodes, weights / np.sqrt(2 * np.pi)


def gaussian_information(d: float | np.ndarray, order: int = 256) -> float | np.ndarray:
    """F(d) bits for balanced N(±d/2,1), using stable Gauss–Hermite softplus.

    These are oracle information values: acquisition is known. They must not be
    relabelled as actual information when acquisition is unobserved.
    """
    values = np.asarray(d, dtype=float)
    if np.any(values < 0) or np.any(~np.isfinite(values)) or order < 8:
        raise ValueError("Finite nonnegative separation and order≥8 required")
    nodes, weights = _normal_nodes(order)
    flat = values.reshape(-1)
    integrand = np.logaddexp(0, -flat[:, None] ** 2 / 2 - flat[:, None] * nodes)
    answer = np.clip(1 - integrand @ weights / np.log(2), 0, 1).reshape(values.shape)
    answer = np.where(values == 0, 0.0, answer)
    return float(answer) if answer.ndim == 0 else answer


def binary_entropy(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 0, 1)
    return -(xlogy(p, p) + xlogy(1 - p, 1 - p)) / np.log(2)


class MixtureChannel:
    """Exact density model for finite shared-nuisance Gaussian mixtures.

    A common positive definite covariance applies to every label/component.
    After whitening, projection keeps the full affine span of all class means.
    Only numerical-zero singular values are removed; there is no task-dependent
    dimension truncation. Discarded coordinates are independent label-free
    noise, so this sufficient statistic preserves actual MI and Bayes risk.
    """

    def __init__(self, plus_means: np.ndarray, minus_means: np.ndarray,
                 probabilities: np.ndarray, covariance: np.ndarray):
        plus = np.asarray(plus_means, dtype=float)
        minus = np.asarray(minus_means, dtype=float)
        if plus.ndim == 1:
            plus = plus[:, None]
        if minus.ndim == 1:
            minus = minus[:, None]
        prob = np.asarray(probabilities, dtype=float)
        if plus.shape != minus.shape or plus.ndim != 2 or len(plus) != len(prob):
            raise ValueError("Class means require matching K×q arrays and K probabilities")
        if not np.all(np.isfinite(plus)) or not np.all(np.isfinite(minus)):
            raise ValueError("All class means must be finite")
        if np.any(prob <= 0) or not np.isclose(prob.sum(), 1, atol=1e-12, rtol=1e-12):
            raise ValueError("Probabilities must be positive and sum to one")
        cov = np.asarray(covariance, dtype=float)
        if cov.shape != (plus.shape[1], plus.shape[1]):
            raise ValueError("Covariance shape does not match mean dimension")
        self.whitener = covariance_whitener(cov)
        self.center = plus[0].copy()
        all_means = np.vstack((plus, minus))
        whitened = (all_means - self.center) @ self.whitener.T
        _, singular, vt = np.linalg.svd(whitened, full_matrices=False)
        threshold = max(whitened.shape) * np.finfo(float).eps * (singular[0] if len(singular) else 0)
        self.rank = int(np.count_nonzero(singular > threshold))
        self.basis = vt[:self.rank].T
        reduced = whitened @ self.basis
        self.plus = reduced[:len(plus)]
        self.minus = reduced[len(plus):]
        self.probabilities = prob / prob.sum()
        self.log_probabilities = np.log(self.probabilities)
        self.original_plus = plus
        self.original_minus = minus
        self.covariance = cov
        self.reconstruction_error = float(np.linalg.norm(whitened - reduced @ self.basis.T))

    def transform(self, samples: np.ndarray) -> np.ndarray:
        return ((np.asarray(samples) - self.center) @ self.whitener.T) @ self.basis

    def log_density_reduced(self, samples: np.ndarray, positive: bool) -> np.ndarray:
        means = self.plus if positive else self.minus
        samples = np.asarray(samples, dtype=float)
        # Common Gaussian normalization cancels from posterior likelihood ratios.
        distances = np.sum((samples[:, None] - means[None]) ** 2, axis=-1)
        return logsumexp(self.log_probabilities[None] - distances / 2, axis=1)

    def posterior_reduced(self, samples: np.ndarray) -> np.ndarray:
        return expit(self.log_density_reduced(samples, True)
                     - self.log_density_reduced(samples, False))

    def predict_proba(self, samples: np.ndarray) -> np.ndarray:
        return self.posterior_reduced(self.transform(samples))

    def predict(self, samples: np.ndarray) -> np.ndarray:
        return np.where(self.predict_proba(samples) >= 0.5, 1, -1)


@dataclass(frozen=True)
class MixtureInformationEstimate:
    bits: float
    standard_error: float | None
    bayes_error: float
    bayes_error_standard_error: float | None
    method: str
    reduced_rank: int
    n_samples: int
    seed: int | None
    quadrature_order: int | None
    quadrature_change_bits: float | None
    confidence_lower_bits: float | None
    confidence_upper_bits: float | None
    reconstruction_error: float

    def to_dict(self) -> dict:
        return asdict(self)


def _quadrature(channel: MixtureChannel, order: int) -> tuple[float, float]:
    if channel.rank == 0:
        return 0.0, 0.5
    nodes, weights = _normal_nodes(order)
    mesh = list(product(range(order), repeat=channel.rank))
    index = np.asarray(mesh, dtype=int)
    noise = nodes[index]
    joint_weights = np.prod(weights[index], axis=1)
    information, risk = 0.0, 0.0
    for means in (channel.plus, channel.minus):
        for probability, mean in zip(channel.probabilities, means):
            posterior = channel.posterior_reduced(noise + mean)
            information += probability * float(joint_weights @ (1 - binary_entropy(posterior))) / 2
            risk += probability * float(joint_weights @ np.minimum(posterior, 1 - posterior)) / 2
    return float(np.clip(information, 0, 1)), float(risk)


def finite_mixture_information(plus_means: np.ndarray, minus_means: np.ndarray,
                               probabilities: np.ndarray, covariance: np.ndarray,
                               n_samples: int = 60000, seed: int = 0,
                               method: str = "auto", quadrature_order: int = 32,
                               quadrature_max_rank: int = 2,
                               confidence: float = 0.95) -> MixtureInformationEstimate:
    """Actual I(Y;X) for hidden acquisition, with exact finite-mixture densities.

    Auto uses tensor Gauss–Hermite quadrature for rank≤2, reproducible MC above.
    Quadrature order convergence is a diagnostic, not a rigorous error bound.
    MC integrates bounded 1−h₂(P(Y=+|X)) under marginal X. The reported SE is
    empirical; the confidence interval uses distribution-free Hoeffding bounds
    conditional on the completely specified synthetic channel.
    """
    channel = MixtureChannel(plus_means, minus_means, probabilities, covariance)
    if method not in ("auto", "quadrature", "monte_carlo"):
        raise ValueError("Unknown integration method")
    use_quadrature = method == "quadrature" or (method == "auto" and channel.rank <= quadrature_max_rank)
    if channel.rank == 0:
        return MixtureInformationEstimate(0, 0, 0.5, 0, "exact identical class means", 0,
                                          0, None, None, 0, 0, 0, channel.reconstruction_error)
    if use_quadrature:
        if channel.rank > 3 or quadrature_order < 8:
            raise ValueError("Tensor quadrature supports rank≤3 and order≥8; use Monte Carlo otherwise")
        coarse, _ = _quadrature(channel, quadrature_order)
        fine_order = 2 * quadrature_order
        bits, risk = _quadrature(channel, fine_order)
        return MixtureInformationEstimate(bits, None, risk, None,
            "Gauss–Hermite exact-density quadrature; order comparison diagnostic",
            channel.rank, 0, None, fine_order, abs(bits - coarse), None, None,
            channel.reconstruction_error)
    if n_samples < 2 or not 0 < confidence < 1:
        raise ValueError("Require n_samples≥2 and confidence∈(0,1)")
    rng = np.random.default_rng(seed)
    # Samples from the marginal include both labels and the independent nuisance.
    labels = rng.integers(0, 2, size=n_samples)
    acquisition = rng.choice(len(channel.probabilities), size=n_samples, p=channel.probabilities)
    means = np.where(labels[:, None] == 1, channel.plus[acquisition], channel.minus[acquisition])
    information = np.empty(n_samples)
    risks = np.empty(n_samples)
    # Batching bounds temporary K×rank work arrays for large component counts.
    for start in range(0, n_samples, 8192):
        stop = min(start + 8192, n_samples)
        samples = means[start:stop] + rng.normal(size=(stop - start, channel.rank))
        posterior = channel.posterior_reduced(samples)
        information[start:stop] = 1 - binary_entropy(posterior)
        risks[start:stop] = np.minimum(posterior, 1 - posterior)
    bits = float(information.mean())
    halfwidth = math.sqrt(math.log(2 / (1 - confidence)) / (2 * n_samples))
    return MixtureInformationEstimate(bits, float(information.std(ddof=1) / np.sqrt(n_samples)),
        float(risks.mean()), float(risks.std(ddof=1) / np.sqrt(n_samples)),
        "Monte Carlo of bounded posterior entropy with exact finite-mixture density",
        channel.rank, n_samples, seed, None, None, max(0, bits - halfwidth), min(1, bits + halfwidth),
        channel.reconstruction_error)


def oracle_summary(plus_means: np.ndarray, minus_means: np.ndarray,
                   probabilities: np.ndarray, covariance: np.ndarray) -> dict:
    channel = MixtureChannel(plus_means, minus_means, probabilities, covariance)
    contrasts = (channel.original_plus - channel.original_minus) @ channel.whitener.T
    d = np.linalg.norm(contrasts, axis=1)
    information = gaussian_information(d)
    risks = ndtr(-d / 2)
    return {"separations": d.tolist(), "information_bits": information.tolist(),
            "bayes_errors": risks.tolist(),
            "average_information_bits": float(channel.probabilities @ information),
            "average_bayes_error": float(channel.probabilities @ risks),
            "meaning": "operator observed; different oracle decoder permitted for every component"}
