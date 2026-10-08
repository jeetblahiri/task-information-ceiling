"""Independent likelihood, persistent-subject, projection and geometry checks."""
import unittest
import numpy as np
from scipy.special import expit, logsumexp
from scipy.stats import multivariate_normal
from tdo_sim.information import finite_mixture_information, gaussian_information
from tdo_sim.insight import (GaussianHeadChannel, paired_information,
    calibration_information, evaluate_calibrated_prior, fit_projection,
    rank_one_covariance, structured_score, structured_information_bounds)


class InsightTests(unittest.TestCase):
    def test_arbitrary_midpoint_mixture_against_independent_quadrature(self):
        plus = np.array([[2., .3], [0., 1.4], [1., -.4]])
        minus = np.array([[-.8, .3], [-1.7, -.2], [0., -.8]])
        weights = np.array([.2, .35, .45])
        covariance = np.array([[1.2, .25], [.25, .7]])
        channel = GaussianHeadChannel(plus, minus, weights, covariance)
        reference = finite_mixture_information(plus, minus, weights, covariance,
            method='quadrature', quadrature_order=64)
        estimate = paired_information(channel, n_samples=50000, seed=781)
        self.assertLess(abs(estimate['full']['bits'] - reference.bits), 5 * estimate['full']['standard_error'] + 1e-5)
        self.assertLess(abs(estimate['full']['bayes_error'] - reference.bayes_error), 5 * estimate['full']['bayes_error_standard_error'] + 1e-4)
        self.assertGreater(estimate['known_head_exact']['average_information_bits'], reference.bits)

    def test_rank_one_density_covariance_and_projection_against_direct_gaussians(self):
        rng = np.random.default_rng(81)
        plus, minus = rng.normal(size=(3, 5)), rng.normal(size=(3, 5))
        b = rng.normal(size=(3, 5))
        sd = np.array([.3, .8, 1.2])
        covariance = np.diag([1., 2., .8, 1.4, .7])
        weights = np.array([.2, .3, .5])
        channel = GaussianHeadChannel(plus, minus, weights, covariance, b, sd)
        samples = rng.normal(size=(25, 5))
        covariances = rank_one_covariance(b, sd, covariance)
        lp = np.column_stack([multivariate_normal.logpdf(samples, plus[h], covariances[h]) for h in range(3)])
        lm = np.column_stack([multivariate_normal.logpdf(samples, minus[h], covariances[h]) for h in range(3)])
        expected = expit(logsumexp(lp + np.log(weights), axis=1) - logsumexp(lm + np.log(weights), axis=1))
        np.testing.assert_allclose(channel.posterior(samples), expected, atol=2e-14)
        calibration = samples[:4]
        calibration_labels = np.array([1, -1, 1, -1])
        log_heads = np.where(calibration_labels[:, None] > 0, lp[:4], lm[:4]).sum(axis=0) + np.log(weights)
        exact_head_posterior = np.exp(log_heads - logsumexp(log_heads))
        np.testing.assert_allclose(channel.posterior_heads(calibration, calibration_labels), exact_head_posterior, atol=1e-14)
        oracle = channel.oracle_summary()
        d = np.array([np.sqrt((plus[h]-minus[h]) @ np.linalg.solve(covariances[h], plus[h]-minus[h])) for h in range(3)])
        np.testing.assert_allclose(oracle['separations'], d, atol=1e-13)
        projection = np.array([[1., .2, 0., 0., 0.], [0., 0., .4, 0., 1.]])
        projected = channel.project(projection)
        direct = GaussianHeadChannel(plus @ projection.T, minus @ projection.T, weights,
            projection @ covariances @ projection.T)
        np.testing.assert_allclose(projected.posterior(samples @ projection.T), direct.posterior(samples @ projection.T), atol=1e-13)

    def test_covariance_nuisance_directions_are_not_silently_removed(self):
        # The second coordinate has no class mean but identifies a hidden sign.
        covariance = np.array([np.diag([1., .04]), np.diag([1., 25.])])
        channel = GaussianHeadChannel(np.array([[2., 0.], [-2., 0.]]),
            np.array([[-2., 0.], [2., 0.]]), covariance=covariance)
        self.assertEqual(channel.rank, 2)
        value = paired_information(channel, {'mean_only': np.array([[1., 0.]])}, n_samples=16000, seed=19)
        self.assertGreater(value['full']['bits'], .35)
        self.assertAlmostEqual(value['projections']['mean_only']['bits'], 0., places=13)

    def test_common_noise_reduction_and_paired_invertible_loss(self):
        channel = GaussianHeadChannel(np.array([[1., 0., 0., 0.], [2., 0., 0., 0.]]),
            np.array([[-1., 0., 0., 0.], [-2., 0., 0., 0.]]))
        self.assertEqual(channel.rank, 1)
        transform = np.array([[2., .1, 0., 0.], [0., .5, 0., 0.], [0., 0., 1.3, .2], [0., 0., 0., .7]])
        value = paired_information(channel, {'invertible': transform, 'matched': np.array([[1., 0., 0., 0.]])}, n_samples=1000, seed=11)
        self.assertLess(abs(value['projections']['invertible']['paired_information_loss']['estimate']), 1e-14)
        pair = value['projection_pairwise']['invertible__minus__matched']
        self.assertLess(abs(pair['information_difference']['estimate']), 1e-14)
        np.testing.assert_allclose(value['known_head_exact']['information_bits'], gaussian_information(np.array([2., 4.])), atol=1e-13)

    def test_head_persists_across_calibration_and_test(self):
        channel = GaussianHeadChannel(np.array([[3.], [-3.]]), np.array([[-3.], [3.]]))
        x, labels = np.array([[2.], [1.]]), np.array([1, -1])
        posterior = channel.posterior_heads(x, labels)
        self.assertAlmostEqual(posterior[0], expit(6 * float(labels @ x[:, 0])))
        zero = calibration_information(channel, 0, n_samples=6000, seed=27)
        prior = paired_information(channel, n_samples=6000, seed=27)
        self.assertEqual(zero['full'], prior['full'])
        self.assertEqual(zero['full']['bits'], 0.)
        calibrated = calibration_information(channel, 2, n_samples=6000, seed=27)
        self.assertGreater(calibrated['full']['bits'], .95)
        self.assertGreater(calibrated['paired_calibration_information_gain']['estimate'], .95)
        self.assertLess(calibrated['full']['bayes_error'], .005)

    def test_development_prior_predictions_use_separate_subject_units(self):
        prior = GaussianHeadChannel(np.array([[3.], [-3.]]), np.array([[-3.], [3.]]), head_ids=['dev0', 'dev1'])
        check = GaussianHeadChannel(np.array([[2.8], [-2.8]]), np.array([[-2.8], [2.8]]), head_ids=['check0', 'check1'])
        zero = evaluate_calibrated_prior(prior, check, 0, n_subjects=200, test_epochs=32, seed=37)
        calibrated = evaluate_calibrated_prior(prior, check, 2, n_subjects=200, test_epochs=32, seed=37)
        self.assertGreater(zero['error']['estimate'], .45)
        self.assertLess(calibrated['error']['estimate'], .02)
        self.assertTrue(calibrated['disjoint_head_ids_verified'])
        self.assertEqual(calibrated['error']['n_outer_draws'], 200)
        self.assertEqual(len(calibrated['predicted_probabilities']), 200)
        self.assertIn('not exact MI', calibrated['information_lower_score_scope'])
        with self.assertRaises(ValueError):
            evaluate_calibrated_prior(prior, prior, 2)

    def test_matched_budget_pca_and_target_aware_projection(self):
        design = np.array([[100., 0.], [-100., 0.], [0., 1.], [0., -1.]])
        target = np.array([0., 2.])
        pca = fit_projection('pca', 1, design)
        aware = fit_projection('target', 1, design, target_contrasts=target)
        channel = GaussianHeadChannel(target[None] / 2, -target[None] / 2)
        self.assertAlmostEqual(channel.project(pca).oracle_summary()['average_information_bits'], 0.)
        self.assertAlmostEqual(channel.project(aware).oracle_summary()['average_information_bits'], gaussian_information(2.))
        colored = np.array([[1., .2], [.2, 2.]])
        projection = fit_projection('target', 1, design, covariance=colored, target_contrasts=target)
        np.testing.assert_allclose(projection @ colored @ projection.T, np.eye(1), atol=1e-14)

    def test_structured_score_and_checked_nearest_point_certificates(self):
        b = np.array([[2.], [0.]])
        error = np.array([[3., 4.]])
        score = structured_score(error, b, 1.)[0]
        self.assertEqual(score, 4.)
        coefficient = np.linalg.pinv(b) @ error[0] / score
        residual = (error[0] - b @ (np.linalg.pinv(b) @ error[0])) / score
        self.assertLessEqual(np.linalg.norm(coefficient), 1.)
        self.assertLessEqual(np.linalg.norm(residual), 1.)
        value = structured_information_bounds(np.array([3., 0.]), np.diag([.5, 2.]), .2, 1.)
        self.assertAlmostEqual(value['minimum_distance_lower_certificate'], 2.3, places=12)
        self.assertAlmostEqual(value['minimum_distance_primal_upper'], 2.3, places=12)
        self.assertTrue(value['optimizer_certificate_passed'])
        orthogonal = structured_information_bounds(np.array([0., 2.]), np.array([[1.], [0.]]), .5, 1.)
        self.assertAlmostEqual(orthogonal['minimum_distance_lower_certificate'], 1.5)
        rng = np.random.default_rng(812)
        center, shape, tau, q = np.array([1.2, -.7, .4]), rng.normal(size=(3, 2)), .2, .6
        cert = structured_information_bounds(center, shape, tau, q)
        self.assertTrue(cert['optimizer_certificate_passed'])
        z, v = rng.normal(size=(200, 2)), rng.normal(size=(200, 3))
        z /= np.maximum(1., np.linalg.norm(z, axis=1))[:, None]
        v /= np.maximum(1., np.linalg.norm(v, axis=1))[:, None]
        norms = np.linalg.norm(center + q * (z @ shape.T + tau * v), axis=1)
        self.assertTrue(np.all(norms >= cert['minimum_distance_lower_certificate'] - 1e-12))
        self.assertTrue(np.all(norms <= cert['maximum_distance_upper'] + 1e-12))


if __name__ == '__main__':
    unittest.main()
