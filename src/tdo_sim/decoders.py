"""Small fitted decoders, using only supplied training/validation examples."""

from dataclasses import dataclass
import numpy as np
from scipy.special import expit


def _check_training(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(x, dtype=float), np.asarray(y)
    if x.ndim != 2 or y.shape != (len(x),) or not np.all(np.isfinite(x)):
        raise ValueError("Training data require finite N×p features and N labels")
    if set(np.unique(y)) != {-1, 1}:
        raise ValueError("Both classes labelled −1 and +1 are required")
    return x, y


def _standardization(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > np.finfo(float).eps * max(1, np.max(scale)), scale, 1.0)
    return (x - center) / scale, center, scale


@dataclass(frozen=True)
class LinearDecoder:
    center: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: float
    metadata: dict

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return expit(((np.asarray(x) - self.center) / self.scale) @ self.weights + self.intercept)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.where(self.predict_proba(x) >= 0.5, 1, -1)


def train_linear(x_train: np.ndarray, y_train: np.ndarray, ridge: float = 1e-3) -> LinearDecoder:
    """Pooled-covariance ridge LDA; covariance and scaling use training rows only."""
    x, y = _check_training(x_train, y_train)
    if ridge <= 0:
        raise ValueError("Positive ridge required")
    z, center, scale = _standardization(x)
    plus, minus = z[y == 1], z[y == -1]
    mp, mm = plus.mean(axis=0), minus.mean(axis=0)
    residual = z - np.where(y[:, None] == 1, mp, mm)
    covariance = residual.T @ residual / max(len(z) - 2, 1)
    regularizer = ridge * max(float(np.trace(covariance) / len(mp)), 1e-8)
    weights = np.linalg.solve(covariance + regularizer * np.eye(len(mp)), mp - mm)
    intercept = float(-((mp + mm) / 2) @ weights + np.log(len(plus) / len(minus)))
    return LinearDecoder(center, scale, weights, intercept,
                         {"kind": "pooled covariance ridge LDA", "training_samples": len(x),
                          "ridge": ridge, "features": x.shape[1], "training_only_scaling": True})


@dataclass(frozen=True)
class MLPDecoder:
    center: np.ndarray
    scale: np.ndarray
    input_weights: np.ndarray
    hidden_bias: np.ndarray
    output_weights: np.ndarray
    output_bias: np.ndarray
    metadata: dict

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=float) - self.center) / self.scale
        hidden = np.tanh(z @ self.input_weights + self.hidden_bias)
        return expit(hidden @ self.output_weights + self.output_bias).reshape(-1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.where(self.predict_proba(x) >= 0.5, 1, -1)


def train_mlp(x_train: np.ndarray, y_train: np.ndarray,
              x_validation: np.ndarray | None = None, y_validation: np.ndarray | None = None,
              hidden_units: int = 24, epochs: int = 120, batch_size: int = 128,
              learning_rate: float = 0.01, l2: float = 1e-3, seed: int = 0,
              patience: int = 20) -> MLPDecoder:
    """One tanh hidden layer with Adam and optional validation early stopping.

    Test examples are never accepted by this fit API. Any validation set must be
    independently generated from the intended training experiment. The fit is
    a numerical baseline; its failed predictions cannot establish an MI ceiling.
    """
    x, labels = _check_training(x_train, y_train)
    if hidden_units < 1 or epochs < 1 or batch_size < 1 or learning_rate <= 0 or l2 < 0:
        raise ValueError("Invalid MLP training parameters")
    z, center, scale = _standardization(x)
    target = (labels + 1) / 2
    if (x_validation is None) != (y_validation is None):
        raise ValueError("Validation features and labels must be provided together")
    if x_validation is not None:
        xv, yv = _check_training(x_validation, y_validation)
        if xv.shape[1] != x.shape[1]:
            raise ValueError("Validation feature dimension differs from training")
        zv, tv = (xv - center) / scale, (yv + 1) / 2
    else:
        zv, tv = z, target
    rng = np.random.default_rng(seed)
    parameters = [rng.normal(size=(z.shape[1], hidden_units)) / np.sqrt(z.shape[1]),
                  np.zeros(hidden_units), rng.normal(size=hidden_units) / np.sqrt(hidden_units),
                  np.zeros(1)]
    first_moment = [np.zeros_like(p) for p in parameters]
    second_moment = [np.zeros_like(p) for p in parameters]
    best = [p.copy() for p in parameters]
    best_loss, best_epoch, stale, step = float("inf"), 0, 0, 0
    history = []
    for epoch in range(epochs):
        permutation = rng.permutation(len(z))
        for start in range(0, len(z), batch_size):
            batch = permutation[start:start + batch_size]
            xb, yb = z[batch], target[batch]
            w1, b1, w2, b2 = parameters
            hidden = np.tanh(xb @ w1 + b1)
            logits = hidden @ w2 + b2
            residual = (expit(logits) - yb) / len(batch)
            hidden_gradient = residual[:, None] * w2 * (1 - hidden ** 2)
            gradients = [xb.T @ hidden_gradient + l2 * w1,
                         hidden_gradient.sum(axis=0),
                         hidden.T @ residual + l2 * w2,
                         np.asarray([residual.sum()])]
            step += 1
            for i, gradient in enumerate(gradients):
                first_moment[i] = 0.9 * first_moment[i] + 0.1 * gradient
                second_moment[i] = 0.999 * second_moment[i] + 0.001 * gradient ** 2
                parameters[i] -= learning_rate * (first_moment[i] / (1 - 0.9 ** step)) / (
                    np.sqrt(second_moment[i] / (1 - 0.999 ** step)) + 1e-8)
        w1, b1, w2, b2 = parameters
        validation_logits = np.tanh(zv @ w1 + b1) @ w2 + b2
        loss = float(np.mean(np.logaddexp(0, validation_logits) - tv * validation_logits))
        history.append(loss)
        if loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = loss, epoch + 1, 0
            best = [p.copy() for p in parameters]
        else:
            stale += 1
        if x_validation is not None and stale >= patience:
            break
    return MLPDecoder(center, scale, *best,
        {"kind": "NumPy one-hidden-layer tanh MLP with Adam", "training_samples": len(z),
         "validation_samples": 0 if x_validation is None else len(zv),
         "hidden_units": hidden_units, "epochs_run": len(history), "selected_epoch": best_epoch,
         "validation_or_training_log_loss": best_loss, "learning_rate": learning_rate,
         "l2": l2, "seed": seed, "training_only_scaling": True,
         "selection": "validation loss" if x_validation is not None else "training loss"})


def evaluate_predictions(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    """Held-out error, approximate Wilson diagnostic, log loss, Brier score.

    Exactly balanced label strata need not share error probabilities. Hoeffding
    also bounds the mean independent test-row error without an iid-binomial
    assumption, conditional on the fitted decoder and synthetic channel.
    """
    y, p = np.asarray(y_true), np.asarray(probabilities, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or len(y) == 0 or not np.all(np.isfinite(p)):
        raise ValueError("Need matching nonempty labels and finite probabilities")
    if np.any((p < 0) | (p > 1)) or not set(np.unique(y)).issubset({-1, 1}):
        raise ValueError("Probabilities must be in [0,1], labels ±1")
    n = len(y)
    error = float(np.mean(np.where(p >= 0.5, 1, -1) != y))
    z = 1.959963984540054
    center = (error + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    halfwidth = z * np.sqrt(error * (1 - error) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    target = (y + 1) / 2
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    hoeffding_halfwidth = np.sqrt(np.log(40) / (2 * n))
    return {"error": error, "accuracy": 1 - error, "error_standard_error": float(np.sqrt(error * (1 - error) / n)),
            "error_ci95": [float(max(0, center - halfwidth)), float(min(1, center + halfwidth))],
            "log_loss_nats": float(-np.mean(target * np.log(clipped) + (1 - target) * np.log1p(-clipped))),
            "brier_score": float(np.mean((p - target) ** 2)), "n_test": n,
            "error_hoeffding_ci95": [float(max(0, error - hoeffding_halfwidth)),
                                      float(min(1, error + hoeffding_halfwidth))],
            "error_interval_method": "Wilson approximate binomial diagnostic; Hoeffding independent-row bound also reported"}
