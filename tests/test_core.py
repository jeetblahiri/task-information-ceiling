"""Meaningful channel, reference, calibration, decoder, and physics checks.

Run: PYTHONPATH=src <python> -m unittest discover -s tests -v
Physical checks skip when MNE is unavailable; they never fetch datasets.
"""

from dataclasses import replace
import importlib.util
import unittest
import numpy as np
from scipy.linalg import qr
from scipy.special import ndtr
from scipy.integrate import quad

from tdo_sim import (
    car_basis, covariance_whitener, separation, make_nested_geometry,
    make_spherical_forward, HeadParameters, gaussian_information,
    finite_mixture_information, MixtureChannel, oracle_summary,
    probe_information_bounds, probe_error_score, endpoint_operator,
    sample_gaussian_mixture, train_linear, train_mlp, evaluate_predictions,
)
from tdo_sim.coordinates import observed_forward, observed_covariance, separable_whiten_epoch
from tdo_sim.noise import sensor_covariance, temporal_covariance


class InformationTests(unittest.TestCase):
    def test_gaussian_information_and_oracle_risk(self):
        self.assertEqual(gaussian_information(0), 0)
        d = np.array([0, .1, .4, 1, 2, 4, 6])
        f = gaussian_information(d)
        self.assertTrue(np.all(np.diff(f) > 0))
        for value in (.3, .5, 2, 6):
            integral = quad(lambda n: np.exp(-n * n / 2) / np.sqrt(2 * np.pi)
                            * np.logaddexp(0, -value * value / 2 - value * n) / np.log(2),
                            -12, 12, epsabs=1e-12)[0]
            self.assertAlmostEqual(gaussian_information(value, order=256), 1 - integral, places=9)
        exact = finite_mixture_information([[1]], [[-1]], [1], np.eye(1), quadrature_order=64)
        self.assertAlmostEqual(exact.bits, gaussian_information(2), places=11)
        oracle = oracle_summary([[1]], [[-1]], [1], np.eye(1))
        self.assertAlmostEqual(oracle["average_bayes_error"], ndtr(-1), places=14)
        self.assertLess(abs(exact.bayes_error - ndtr(-1)), 0.002)

    def test_hidden_sign_swap_erases_all_actual_information(self):
        estimate = finite_mixture_information([[3], [-3]], [[-3], [3]], [.5, .5], np.eye(1))
        self.assertEqual(estimate.bits, 0)
        self.assertEqual(estimate.bayes_error, .5)
        oracle = oracle_summary([[3], [-3]], [[-3], [3]], [.5, .5], np.eye(1))
        self.assertGreater(oracle["average_information_bits"], .99)
        channel = MixtureChannel([[3], [-3]], [[-3], [3]], [.5, .5], np.eye(1))
        np.testing.assert_allclose(channel.predict_proba(np.array([[-8], [0], [4]])), .5)

    def test_affine_sufficient_reduction_keeps_actual_channel(self):
        rng = np.random.default_rng(12)
        q, _ = qr(rng.normal(size=(8, 8)))
        plus = np.array([[.2, -.8], [.2, .8], [1, .2]])
        minus = -plus
        probabilities = [.45, .45, .1]
        baseline = finite_mixture_information(plus, minus, probabilities, np.eye(2), quadrature_order=64)
        embedding = q[:, :2]
        offset = rng.normal(size=8)
        embedded = finite_mixture_information(plus @ embedding.T + offset,
            minus @ embedding.T + offset, probabilities, np.eye(8), quadrature_order=64)
        self.assertEqual(embedded.reduced_rank, 2)
        self.assertLess(embedded.reconstruction_error, 1e-13)
        self.assertAlmostEqual(embedded.bits, baseline.bits, places=11)
        self.assertGreaterEqual(embedded.bits + 1e-10, gaussian_information(.4))

    def test_mc_numerical_uncertainty_and_reproducibility(self):
        plus = np.eye(4) * 1.3
        minus = -plus
        first = finite_mixture_information(plus, minus, np.ones(4) / 4, np.eye(4),
                                           n_samples=12000, seed=31)
        second = finite_mixture_information(plus, minus, np.ones(4) / 4, np.eye(4),
                                            n_samples=12000, seed=31)
        self.assertEqual(first.bits, second.bits)
        self.assertEqual(first.reduced_rank, 4)
        self.assertGreater(first.standard_error, 0)
        self.assertLess(first.confidence_lower_bits, first.bits)
        self.assertGreater(first.confidence_upper_bits, first.bits)
        self.assertLess(first.bits, oracle_summary(plus, minus, np.ones(4) / 4, np.eye(4))["average_information_bits"])


class CoordinatesTests(unittest.TestCase):
    def test_car_reference_invariance_and_nested_data_processing(self):
        geometry = make_nested_geometry()
        raw_cov = sensor_covariance(geometry.sensor_positions_m)
        rng = np.random.default_rng(90)
        gain = rng.normal(size=(256, 4)) * 1e-6
        contrast = rng.normal(size=4)
        separations = []
        for count in (19, 64, 256):
            index = geometry.montages[count]
            u = car_basis(count)
            np.testing.assert_allclose(u.T @ u, np.eye(count - 1), atol=1e-14)
            np.testing.assert_allclose(u.T @ np.ones(count), 0, atol=1e-14)
            g = observed_forward(gain, index)
            cov = observed_covariance(raw_cov, index)
            separations.append(separation(g @ contrast, cov))
            transform, _ = qr(rng.normal(size=(count - 1, count - 1)))
            self.assertAlmostEqual(separation(transform @ g @ contrast, transform @ cov @ transform.T),
                                   separations[-1], places=10)
        self.assertTrue(np.all(np.diff(separations) >= -1e-11))
        # Retaining covariance under an invertible unit change cancels the gain.
        delta = observed_forward(gain, geometry.montages[19]) @ contrast
        cov = observed_covariance(raw_cov, geometry.montages[19])
        self.assertAlmostEqual(separation(delta * 1e6, cov * 1e12), separations[0], places=10)

    def test_separable_epoch_whitening_equals_full_kronecker(self):
        spatial = np.array([[2, .3], [.3, 1]])
        temporal = temporal_covariance(3, .4)
        epoch = np.arange(6).reshape(2, 3)
        transformed = separable_whiten_epoch(epoch, spatial, temporal).ravel()
        expected = covariance_whitener(np.kron(spatial, temporal)) @ epoch.ravel()
        np.testing.assert_allclose(transformed, expected, atol=1e-14)


class CalibrationTests(unittest.TestCase):
    def test_endpoints_are_attained_with_spectral_probe_error(self):
        gain = np.array([[1.2, .1, 0], [.3, .5, 0]])
        probes = np.array([[2, 0], [0, .25], [0, 0]])
        delta = np.array([.4, .2, 0])
        covariance = np.array([[2, .3], [.3, 1]])
        for epsilon in (.1, 4):
            interval = probe_information_bounds(gain, delta, probes, covariance, epsilon)
            self.assertTrue(interval["applicable"])
            self.assertAlmostEqual(interval["probe_amplification"], np.sqrt(.2 ** 2 + .8 ** 2))
            for side in ("lower", "upper"):
                error = endpoint_operator(gain, delta, probes, covariance, epsilon, side)
                self.assertLessEqual(probe_error_score(error, probes, covariance), epsilon + 1e-12)
                actual = gaussian_information(separation((gain + error) @ delta, covariance))
                self.assertAlmostEqual(actual, interval[side + "_bits"], places=11)

    def test_out_of_span_is_vacuous_even_with_perfect_probes(self):
        gain = np.array([[1, 2, 3], [0, 1, 2]], dtype=float)
        probes = np.array([[1], [0], [0]], dtype=float)
        delta = np.array([0, 0, 1], dtype=float)
        result = probe_information_bounds(gain, delta, probes, np.eye(2), 0)
        self.assertFalse(result["applicable"])
        self.assertEqual((result["lower_bits"], result["upper_bits"]), (0, 1))
        self.assertIsNone(result["target_radius"])
        error = -np.outer(gain @ delta, delta)
        np.testing.assert_allclose(error @ probes, 0)
        np.testing.assert_allclose((gain + error) @ delta, 0)

    def test_midpoint_and_covariance_flags_disable_centered_floor(self):
        arguments = (np.eye(2), np.array([1, 0]), np.eye(2), np.eye(2), .1)
        midpoint = probe_information_bounds(*arguments, midpoint_known=False)
        self.assertFalse(midpoint["applicable"])
        self.assertEqual(midpoint["lower_bits"], 0)
        self.assertLess(midpoint["upper_bits"], 1)
        covariance = probe_information_bounds(*arguments, common_covariance=False)
        self.assertEqual((covariance["lower_bits"], covariance["upper_bits"]), (0, 1))


class DecoderTests(unittest.TestCase):
    def test_actual_held_out_linear_and_nonlinear_predictions(self):
        # Opposite-axis mixtures are linearly inseparable, yet their exact
        # likelihood and a trained tanh MLP discriminate them.
        plus = np.array([[2, 0], [-2, 0]])
        minus = np.array([[0, 2], [0, -2]])
        covariance = np.eye(2) * .2
        train, yt, _ = sample_gaussian_mixture(plus, minus, [.5, .5], covariance, 1200, 1)
        validation, yv, _ = sample_gaussian_mixture(plus, minus, [.5, .5], covariance, 400, 2)
        test, ytest, _ = sample_gaussian_mixture(plus, minus, [.5, .5], covariance, 2000, 3)
        self.assertEqual(np.sum(yt == 1), 600)
        linear = train_linear(train, yt)
        nonlinear = train_mlp(train, yt, validation, yv, epochs=100, seed=4)
        linear_metrics = evaluate_predictions(ytest, linear.predict_proba(test))
        nonlinear_metrics = evaluate_predictions(ytest, nonlinear.predict_proba(test))
        self.assertGreater(linear_metrics["error"], .35)
        self.assertLess(nonlinear_metrics["error"], .08)
        self.assertLess(nonlinear_metrics["log_loss_nats"], linear_metrics["log_loss_nats"])
        np.testing.assert_allclose(nonlinear.center, train.mean(axis=0))


@unittest.skipUnless(importlib.util.find_spec("mne"), "Optional MNE not installed")
class SyntheticPhysicsTests(unittest.TestCase):
    def test_four_layer_sphere_units_fixed_orientation_and_head_variation(self):
        geometry = make_nested_geometry(n_sources=24)
        model = make_spherical_forward(geometry)
        reverse = make_spherical_forward(replace(geometry, source_orientations=-geometry.source_orientations))
        shifted_geometry = replace(geometry, sensor_positions_m=geometry.sensor_positions_m + [.001, -.001, .0005])
        shifted = make_spherical_forward(shifted_geometry, HeadParameters(center_m=(.001, -.001, .0005),
            conductivities_s_m=(.33, 1, .005, .33)))
        self.assertEqual(model.gain_v_per_am.shape, (256, 24))
        self.assertTrue(np.all(np.isfinite(model.gain_v_per_am)))
        np.testing.assert_allclose(reverse.gain_v_per_am, -model.gain_v_per_am, atol=1e-5)
        self.assertGreater(np.linalg.norm(shifted.gain_v_per_am - model.gain_v_per_am), 0)
        self.assertEqual(model.metadata["gain_units"], "V/(A m)")
        self.assertTrue(model.metadata["no_recordings_or_downloads"])
        # A 20 nA·m dipole yields microvolt scale, before reference/noise.
        voltage = np.max(np.abs(model.gain_v_per_am * 20e-9))
        self.assertGreater(voltage, 1e-7)
        self.assertLess(voltage, 1e-3)
        self.assertTrue(set(geometry.montages[19]).issubset(geometry.montages[64]))
        self.assertTrue(set(geometry.montages[64]).issubset(geometry.montages[256]))


if __name__ == "__main__":
    unittest.main()
