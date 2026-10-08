"""Synthetic experiments only; no EEG recordings or empirical calibration claims."""

from .calibration import probe_information_bounds, probe_error_score, endpoint_operator
from .coordinates import car_basis, covariance_whitener, observed_forward, separation
from .data import sample_gaussian_mixture
from .decoders import train_linear, train_mlp, evaluate_predictions
from .forward import HeadParameters, ForwardModel, make_spherical_forward, make_abstract_forward
from .geometry import Geometry, make_nested_geometry
from .information import gaussian_information, finite_mixture_information, oracle_summary, MixtureChannel

__all__ = [
    "probe_information_bounds", "probe_error_score", "endpoint_operator",
    "car_basis", "covariance_whitener", "observed_forward", "separation",
    "sample_gaussian_mixture", "train_linear", "train_mlp", "evaluate_predictions",
    "HeadParameters", "ForwardModel", "make_spherical_forward", "make_abstract_forward",
    "Geometry", "make_nested_geometry", "gaussian_information",
    "finite_mixture_information", "oracle_summary", "MixtureChannel",
]
