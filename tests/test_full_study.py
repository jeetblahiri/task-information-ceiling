"""Independent numerical checks for new full-study utilities; no pilot changes."""
import unittest
import json
from pathlib import Path
import numpy as np
from tdo_sim.full_study import (centered_mixture_information, partial_content_bounds,
                               maximum_content_confidence, clopper_pearson,
                               sample_feature_oracle, dct_basis, head_seed)
from tdo_sim.information import finite_mixture_information, gaussian_information


class FullStudyTests(unittest.TestCase):
    def test_all_frozen_head_seed_blocks_are_globally_unique(self):
        config = json.loads((Path(__file__).resolve().parents[1] / "config" / "full_study.json").read_text())
        n = (max(config["calibration"]["sample_sizes"]) + config["calibration"]["check_heads_per_replicate_severity"]
             + config["decoder"]["independent_train_heads"] + config["decoder"]["independent_validation_heads"])
        seeds = [head_seed(seed, severity, head) for seed in config["replicate_seeds"]
                 for severity in range(len(config["severity_multipliers"])) for head in range(n)]
        self.assertEqual(len(seeds), 4905)
        self.assertEqual(len(seeds), len(set(seeds)))
    def test_exact_likelihood_mc_matches_independent_low_rank_quadrature(self):
        contrasts = np.array([[1.0, .3], [2.0, -.4], [1.4, .7]])
        p = np.array([.2, .3, .5])
        reference = finite_mixture_information(contrasts / 2, -contrasts / 2, p, np.eye(2),
                                              method="quadrature", quadrature_order=64)
        estimate = centered_mixture_information(contrasts, p, n_samples=50000, seed=11)
        self.assertLess(abs(estimate["bits"] - reference.bits), 5 * estimate["standard_error"] + 1e-5)
        self.assertLess(abs(estimate["bayes_error"] - reference.bayes_error), 5 * estimate["bayes_error_standard_error"] + 1e-4)

    def test_hidden_sign_cancels_full_density(self):
        estimate = centered_mixture_information(np.array([[6.], [-6.]]), n_samples=1000, seed=1)
        self.assertEqual(estimate["bits"], 0.)
        self.assertEqual(estimate["bayes_error"], .5)

    def test_zero_mixture_has_no_information(self):
        estimate = centered_mixture_information(np.zeros((3, 7)), n_samples=100, seed=2)
        self.assertEqual(estimate["bits"], 0.)
        self.assertEqual(estimate["bayes_error"], .5)

    def test_monte_carlo_reproducibility_and_batch_contract(self):
        contrasts = np.array([[1., .2], [1.1, -.1]])
        a = centered_mixture_information(contrasts, n_samples=500, seed=3, batch_size=256)
        b = centered_mixture_information(contrasts, n_samples=500, seed=3, batch_size=256)
        self.assertEqual(a, b)
        self.assertTrue(a["hoeffding_lower_bits"] <= a["bits"] <= a["hoeffding_upper_bits"])

    def test_tolerance_probability_is_order_statistic_not_test_fraction(self):
        self.assertAlmostEqual(maximum_content_confidence(59), 1 - .95 ** 59)
        self.assertGreaterEqual(1 - 15 * .95 ** 119, .95)
        self.assertLess(maximum_content_confidence(2), .1)

    def test_full_content_reduces_to_conditioned_ball(self):
        value = partial_content_bounds(2., .4, 1.)
        self.assertAlmostEqual(value["partial_lower_bits"], gaussian_information(1.6))
        self.assertAlmostEqual(value["partial_upper_bits"], gaussian_information(2.4))

    def test_partial_mass_bound_against_actual_hidden_law(self):
        contrasts = np.array([[2.], [2.2], [-2.]])
        p = np.array([.4, .55, .05])
        actual = finite_mixture_information(contrasts / 2, -contrasts / 2, p, np.eye(1), method="quadrature", quadrature_order=128)
        value = partial_content_bounds(2., .2, .95)
        self.assertLessEqual(value["partial_lower_bits"], actual.bits + 1e-7)
        self.assertGreaterEqual(value["partial_upper_bits"], actual.bits - 1e-7)

    def test_exact_binomial_interval_extremes(self):
        a, b = clopper_pearson(0, 200)
        c, d = clopper_pearson(200, 200)
        self.assertEqual(a, 0.)
        self.assertEqual(d, 1.)
        self.assertGreater(b, 0.)
        self.assertLess(c, 1.)

    def test_dct_orthonormal_and_joint_sampling_covariance(self):
        basis = dct_basis(16)
        np.testing.assert_allclose(basis @ basis.T, np.eye(16), atol=1e-14)
        x, y, oracle = sample_feature_oracle(np.array([1., .5]), 2., 60000, 8)
        noise = x - y[:, None] * np.array([1., .5]) / 2
        scalar_noise = oracle - y
        observed = noise.T @ scalar_noise / len(y)
        np.testing.assert_allclose(observed, np.array([.5, .25]), atol=.012)


if __name__ == "__main__":
    unittest.main()
