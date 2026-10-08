"""Independent physical-transform, mixture-likelihood and prediction checks."""
import unittest
import numpy as np
from scipy.special import expit, logsumexp, ndtr
from scipy.stats import multivariate_normal

from tdo_sim.coordinates import car_basis
from tdo_sim.information import finite_mixture_information, gaussian_information
from tdo_sim.ceiling_benchmark import (EpochDraw, block_average_matrix,
    channel_average_reference_matrix, prepared_experiment,
    prepare_transform_experiment, draw_raw_epochs, paired_task_information,
    summarize_bounded, matched_decoder_benchmark)


class CeilingBenchmarkTests(unittest.TestCase):
    def test_physical_channel_average_and_reference_from_full_car(self):
        rng = np.random.default_rng(118)
        raw = rng.normal(size=(6, 8))
        covariance = np.diag(np.arange(1., 7.)) + .15 * np.ones((6, 6))
        groups = [[0, 1], [2, 3], [4, 5]]
        spatial, average = channel_average_reference_matrix(6, groups)
        q_full, q_out = car_basis(6), car_basis(3)
        np.testing.assert_allclose(spatial @ q_full.T @ raw, q_out.T @ average @ raw, atol=1e-14)
        np.testing.assert_allclose(spatial @ q_full.T @ covariance @ q_full @ spatial.T,
                                   q_out.T @ average @ covariance @ average.T @ q_out, atol=1e-14)
        np.testing.assert_allclose(q_out.T @ average @ np.ones(6), 0., atol=1e-15)
        self.assertEqual(np.linalg.matrix_rank(spatial), 2)

    def test_raw_block_average_color_variance_and_covariance(self):
        temporal = .65 ** np.abs(np.arange(9)[:, None] - np.arange(9)[None, :])
        block = block_average_matrix(9, 3)
        expected = np.array([[temporal[3*i:3*i+3, 3*j:3*j+3].sum() / 9
                              for j in range(3)] for i in range(3)])
        np.testing.assert_allclose(block @ temporal @ block.T, expected, atol=1e-15)
        self.assertAlmostEqual(expected[0, 0], .7161111111111111)
        self.assertGreater(expected[0, 1], .30)
        np.testing.assert_allclose(block @ np.eye(9) @ block.T, np.eye(3) / 3)
        np.testing.assert_allclose(block @ np.arange(9), [1., 4., 7.])
        np.testing.assert_allclose(block_average_matrix(8, 3) @ np.arange(8), [1., 4., 6.5])

    def test_separable_covariance_rank_one_density_and_oracle_independent(self):
        rng = np.random.default_rng(7813)
        plus, minus, background = rng.normal(size=(3, 2, 3)), rng.normal(size=(3, 2, 3)), rng.normal(size=(3, 2, 3))
        cs = np.array([[1.2, .25], [.25, .8]])
        ct = .4 ** np.abs(np.arange(3)[:, None] - np.arange(3)[None, :])
        sd, weights = np.array([.3, .8, 1.1]), np.array([.2, .3, .5])
        law = prepared_experiment(plus, minus, cs, ct, background=background,
                                  background_sd=sd, probabilities=weights)
        samples = rng.normal(size=(30, 2, 3))
        covs = [np.kron(cs, ct) + sd[h] ** 2 * np.outer(background[h].ravel(), background[h].ravel()) for h in range(3)]
        lp = np.column_stack([multivariate_normal.logpdf(samples.reshape(30, 6), plus[h].ravel(), covs[h]) for h in range(3)])
        lm = np.column_stack([multivariate_normal.logpdf(samples.reshape(30, 6), minus[h].ravel(), covs[h]) for h in range(3)])
        expected = expit(logsumexp(lp + np.log(weights), axis=1) - logsumexp(lm + np.log(weights), axis=1))
        np.testing.assert_allclose(law.posterior_raw(samples), expected, atol=3e-14)
        heads = np.arange(30) % 3
        np.testing.assert_allclose(law.known_head_posterior_raw(samples, heads), expit(lp[np.arange(30), heads] - lm[np.arange(30), heads]), atol=2e-14)
        distances = [np.sqrt((plus[h] - minus[h]).ravel() @ np.linalg.solve(covs[h], (plus[h] - minus[h]).ravel())) for h in range(3)]
        np.testing.assert_allclose(law.oracle_summary()['separations'], distances, atol=2e-14)
        np.testing.assert_allclose(law.oracle_summary()['information_bits'], gaussian_information(np.array(distances)), atol=2e-14)
        np.testing.assert_allclose(law.color(law.whiten(samples)), samples, atol=2e-14)

    def test_transform_is_raw_and_pushes_full_covariance_not_white_averaging(self):
        rng = np.random.default_rng(87)
        plus, minus = rng.normal(size=(2, 3, 6)), rng.normal(size=(2, 3, 6))
        cs = np.array([[2., .4, .1], [.4, 1., .2], [.1, .2, .8]])
        ct = .75 ** np.abs(np.arange(6)[:, None] - np.arange(6)[None, :])
        law = prepared_experiment(plus, minus, cs, ct)
        spatial = np.array([[.5, .5, 0.], [0., 0., 1.]])
        temporal = block_average_matrix(6, 2)
        transformed = prepare_transform_experiment(law, spatial, temporal)
        raw_encoder = np.kron(spatial, temporal)
        np.testing.assert_allclose(np.kron(transformed.spatial_covariance, transformed.temporal_covariance),
                                   raw_encoder @ np.kron(cs, ct) @ raw_encoder.T, atol=2e-14)
        epochs = rng.normal(size=(20, 3, 6))
        expected_epochs = np.array([spatial @ epoch @ temporal.T for epoch in epochs])
        np.testing.assert_allclose(transformed.plus.reshape(2, -1), plus.reshape(2, -1) @ raw_encoder.T, atol=1e-14)
        mapped_white = np.array([transformed.white_spatial_map @ epoch @ transformed.white_temporal_map.T for epoch in law.whiten(epochs)])
        np.testing.assert_allclose(mapped_white, transformed.whiten(expected_epochs), atol=2e-14)
        self.assertGreater(np.linalg.norm(mapped_white - np.array([spatial @ epoch @ temporal.T for epoch in law.whiten(epochs)])), 1.)
        self.assertTrue(np.all(np.asarray(transformed.oracle_summary()['separations']) <= np.asarray(law.oracle_summary()['separations']) + 1e-13))

    def test_affine_midpoint_cues_not_contrast_only_and_quadrature(self):
        plus = np.array([[1., 2.], [-1., -2.]])[:, :, None]
        minus = np.array([[-1., 2.], [1., -2.]])[:, :, None]
        law = prepared_experiment(plus, minus, np.eye(2), np.eye(1))
        self.assertEqual(law.likelihood_rank, 2)
        result = paired_task_information(law, {'contrast_only': (np.array([[1., 0.]]), np.eye(1))},
            n_samples=24000, seed=781, return_sample_arrays=True)
        reference = finite_mixture_information(plus[:, :, 0], minus[:, :, 0], np.array([.5, .5]), np.eye(2),
            method='quadrature', quadrature_order=64)
        self.assertLess(abs(result['full']['bits'] - reference.bits), 5 * result['full']['standard_error'])
        self.assertGreater(result['full']['bits'], .40)
        self.assertAlmostEqual(result['representations']['contrast_only']['bits'], 0., places=14)
        self.assertGreater(result['losses']['contrast_only']['information_loss']['estimate'], .40)
        self.assertEqual(result['sample_arrays']['full']['information'].shape, (24000,))
        self.assertAlmostEqual(result['known_head_exact']['average_information_bits'], gaussian_information(2.))

    def test_background_directions_can_reveal_hidden_state(self):
        plus = np.array([[2., 0.], [-2., 0.]])[:, :, None]
        minus = -plus
        background = np.array([[0., 0.], [0., 8.]])[:, :, None]
        law = prepared_experiment(plus, minus, np.eye(2), np.eye(1), background=background)
        self.assertEqual(law.likelihood_rank, 2)
        result = paired_task_information(law, {'means_only': (np.array([[1., 0.]]), np.eye(1))}, n_samples=16000, seed=791)
        self.assertGreater(result['full']['bits'], .30)
        self.assertAlmostEqual(result['representations']['means_only']['bits'], 0., places=14)

    def test_tiny_full_rank_rows_preserve_information_and_dependencies_raise(self):
        plus = np.array([[[.4, 1.], [1.2, -.5]]])
        law = prepared_experiment(plus, -.6 * plus, np.array([[1., .3], [.3, 2.]]), np.array([[1., .5], [.5, 1.]]))
        tiny = np.diag([1., 1e-20])
        result = paired_task_information(law, {'tiny_units': (tiny, np.eye(2)), 'identity': (np.eye(2), np.eye(2))},
                                         n_samples=1000, seed=18, include_pairwise=True)
        self.assertLess(abs(result['losses']['tiny_units']['information_loss']['estimate']), 2e-14)
        self.assertLess(abs(result['pairwise']['tiny_units__minus__identity']['information_difference']['estimate']), 2e-14)
        with self.assertRaises(ValueError):
            prepare_transform_experiment(law, np.array([[1., 0.], [2., 0.]]), np.eye(2))
        with self.assertRaises(ValueError):
            prepared_experiment(plus, plus, np.array([[1., 0.], [0., 0.]]), np.eye(2))

    def test_identical_class_zero_rank_and_native_direct_span_draws(self):
        zero = prepared_experiment(np.ones((2, 2, 3)), np.ones((2, 2, 3)), np.eye(2), np.eye(3))
        self.assertEqual(zero.likelihood_rank, 0)
        result = paired_task_information(zero, n_samples=100, seed=82)
        self.assertEqual(result['full']['bits'], 0.)
        self.assertEqual(result['full']['bayes_error'], .5)
        self.assertEqual(result['known_head_exact']['average_information_bits'], 0.)
        spatial = np.linspace(-1., 1., 255)
        temporal = np.sin(np.linspace(0., 2., 64))
        plus = np.array([np.outer(spatial, temporal), 1.1 * np.outer(spatial, temporal)])
        law = prepared_experiment(plus, -.8 * plus, np.eye(255), np.eye(64))
        self.assertEqual(law.dimension, 16320)
        self.assertEqual(law.likelihood_rank, 1)
        def fail_raw_draw(*args):
            raise AssertionError('Native MI must not draw full raw epochs')
        law.posterior_white = fail_raw_draw
        native = paired_task_information(law, n_samples=1000, seed=812)
        self.assertEqual(native['sampling'], 'direct likelihood-sufficient coordinates')
        self.assertGreater(native['full']['bits'], .99)

    def test_paired_draws_reproducible_across_batch_sizes_and_covariance_sampling(self):
        plus = np.array([[[.8, .4], [0., .2]], [[1.2, -.4], [.6, -.3]]])
        background = np.array([[[.1, .3], [.4, 0.]], [[.2, .1], [0., .3]]])
        cs, ct = np.array([[1., .2], [.2, 1.6]]), np.array([[1., .6], [.6, 1.]])
        law = prepared_experiment(plus, -.6 * plus, cs, ct, background=background)
        transforms = {'average': (np.array([[.5, .5]]), block_average_matrix(2, 2))}
        a = paired_task_information(law, transforms, n_samples=777, seed=18, batch_size=100, return_sample_arrays=True)
        b = paired_task_information(law, transforms, n_samples=777, seed=18, batch_size=512, return_sample_arrays=True)
        for name in ['full', 'known_head']:
            np.testing.assert_allclose(a['sample_arrays'][name]['information'], b['sample_arrays'][name]['information'], atol=2e-14)
        np.testing.assert_allclose(a['sample_arrays']['representations']['average']['information'], b['sample_arrays']['representations']['average']['information'], atol=2e-14)
        direct_a = paired_task_information(law, n_samples=777, seed=18, batch_size=100, return_sample_arrays=True)
        direct_b = paired_task_information(law, n_samples=777, seed=18, batch_size=512, return_sample_arrays=True)
        np.testing.assert_allclose(direct_a['sample_arrays']['full']['information'], direct_b['sample_arrays']['full']['information'], atol=2e-14)
        heads = np.zeros(24000, int)
        draw = draw_raw_epochs(law, len(heads), seed=77, head_indices=heads, labels=np.ones(len(heads)))
        residual = draw.epochs - plus[0]
        expected = np.kron(cs, ct) + np.outer(background[0].ravel(), background[0].ravel())
        np.testing.assert_allclose(np.cov(residual.reshape(len(heads), -1).T), expected, atol=.065)

    def test_fixed_subject_state_and_epoch_noise_are_distinct(self):
        law = prepared_experiment(np.array([[[2.]], [[-2.]]]), np.array([[[-2.]], [[2.]]]), np.eye(1), np.eye(1))
        heads = np.repeat([0, 1], 30)
        subjects = np.repeat([10, 11], 30)
        draw = draw_raw_epochs(law, 60, 91, head_indices=heads, labels=np.ones(60), subject_indices=subjects)
        np.testing.assert_array_equal(draw.head_indices, heads)
        self.assertGreater(draw.epochs[:30].std(), .5)
        self.assertGreater(draw.epochs[:30].mean(), 1.)
        self.assertLess(draw.epochs[30:].mean(), -1.)
        with self.assertRaises(ValueError):
            draw_raw_epochs(law, 60, 91, head_indices=heads, subject_indices=np.zeros(60, int))

    def test_bounded_intervals_and_signed_paired_differences(self):
        values = np.array([-.08, .01, -.02, -.01])
        result = summarize_bounded(values, (-1., 1.))
        self.assertLess(result['estimate'], 0.)
        self.assertLess(result['confidence_lower'], 0.)
        repeated = summarize_bounded(np.repeat(.5, 20000), (0., 1.))
        self.assertEqual(repeated['standard_error'], 0.)
        self.assertLess(repeated['empirical_bernstein_halfwidth'], repeated['hoeffding_halfwidth'])
        multiple = summarize_bounded(values, (-1., 1.), multiplicity=100)
        self.assertGreater(multiple['empirical_bernstein_halfwidth'], result['empirical_bernstein_halfwidth'])
        with self.assertRaises(ValueError):
            summarize_bounded(np.array([0., 2.]), (0., 1.))

    def test_fitted_models_match_raw_reference_and_never_select_on_test_labels(self):
        law = prepared_experiment(np.array([[[1.2]]]), np.array([[[-1.2]]]), np.eye(1), np.eye(1))
        train = draw_raw_epochs(law, 1000, 271)
        validation = draw_raw_epochs(law, 400, 272)
        test = draw_raw_epochs(law, 1000, 273, subject_indices=np.repeat(np.arange(100), 10))
        settings = dict(ridges=(.01, .1), mlp_l2=(.001,), mlp_kwargs={'hidden_units': 8, 'epochs': 30, 'batch_size': 128, 'patience': 8})
        a = matched_decoder_benchmark(train, validation, test, evaluation_experiment=law, seed=89, **settings)
        flipped = EpochDraw(test.epochs, -test.labels, test.head_indices, test.subject_indices)
        b = matched_decoder_benchmark(train, validation, flipped, seed=89, **settings)
        np.testing.assert_array_equal(a['models']['linear'].weights, b['models']['linear'].weights)
        np.testing.assert_array_equal(a['models']['mlp'].input_weights, b['models']['mlp'].input_weights)
        self.assertEqual(a['selection'], b['selection'])
        for model in ['linear', 'mlp']:
            self.assertLess(a[model]['error']['estimate'], .2)
            self.assertEqual(a[model]['error']['n_outer_draws'], 100)
            self.assertEqual(a[model]['n_epochs'], 1000)
            self.assertIn('not exact MI', a[model]['information_scope'])
        np.testing.assert_allclose(a['models']['linear'].center, train.epochs.reshape(1000, 1).mean(axis=0))
        np.testing.assert_allclose(a['bayes_reference_predictions']['predicted_probabilities'], expit(2.4 * test.epochs.ravel()), atol=1e-14)
        self.assertFalse(a['metadata']['latent_indices_used_as_features'])


if __name__ == '__main__':
    unittest.main()
