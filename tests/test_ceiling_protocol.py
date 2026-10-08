"""Prospective runner checks using synthetic fixtures only, never check assets."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch as mock_patch

import numpy as np
from scipy.linalg import eigh

from tdo_sim.ceiling_benchmark import prepared_experiment, block_average_matrix


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('ceiling_protocol_runner', ROOT / 'scripts/run_ceiling_benchmark.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
CONFIG = json.loads((ROOT / 'config/ceiling_benchmark.json').read_text())


def synthetic_bank():
    """An identity forward makes the source amplitude contract observable."""
    rng = np.random.default_rng(617)
    positions = rng.uniform(-.045, .045, (32, 3))
    sensors = rng.normal(size=(32, 3))
    sensors *= .09 / np.linalg.norm(sensors, axis=1)[:, None]
    parameters = []
    for i in range(36):
        split = 'development' if i < 8 else 'validation' if i < 12 else 'check'
        parameters.append({'head_id': f'fixture_{i:02d}', 'split': split})
    gains = np.broadcast_to(np.eye(32), (36, 32, 32)).copy()
    return {'name': 'independent_unit_fixture', 'parameters': parameters,
            'source_positions_m': positions, 'sensor_positions_m': sensors,
            'gains_v_per_am': gains, 'montage_order': np.arange(32)}


def law_fixture(seed=611, c=4, t=5, offset=None):
    rng = np.random.default_rng(seed)
    plus, minus = rng.normal(size=(3, c, t)), rng.normal(size=(3, c, t))
    if offset is not None:
        plus += offset
        minus += offset
    a = rng.normal(size=(c, c))
    cs = a @ a.T + np.eye(c)
    ct = .6 ** np.abs(np.arange(t)[:, None] - np.arange(t)[None, :])
    b = rng.normal(size=(3, c, t))
    return prepared_experiment(plus, minus, cs, ct, background=b, background_sd=np.array([.3, .7, 1.1]))


def independent_dense_covariance(experiments):
    """Equal-weight tasks, uniform states and balanced labels; C-order vec."""
    means = np.concatenate([np.concatenate((e.plus, e.minus)).reshape(2 * e.n_heads, -1) for e in experiments])
    centred = means - means.mean(axis=0)
    covariance = centred.T @ centred / len(means)
    base = sum(np.kron(e.spatial_covariance, e.temporal_covariance) for e in experiments) / len(experiments)
    background = np.zeros_like(base)
    for e in experiments:
        if e.background is not None:
            for h in range(e.n_heads):
                vector = e.background[h].ravel() * e.background_sd[h]
                background += np.outer(vector, vector) / (len(experiments) * e.n_heads)
    return covariance + base + background


def spatial_marginal(covariance, c, t):
    return np.array([[sum(covariance[i*t+k, j*t+k] for k in range(t)) / t
                      for j in range(c)] for i in range(c)])


class CeilingProtocolTests(unittest.TestCase):
    def test_source_coefficient_peak_and_persistent_static_background(self):
        bank = synthetic_bank()
        for task in ['spatial_lateral', 'temporal_fine_latency', 'temporal_rhythm']:
            state = runner.source_means(bank, task, 'development', CONFIG)
            repeated = runner.source_means(bank, task, 'development', CONFIG)
            sensitivity = runner.source_means(bank, task, 'development', CONFIG, CONFIG['sensitivity_signal_peak_am'])
            self.assertEqual(state['parameters'], repeated['parameters'])
            np.testing.assert_array_equal(state['plus'], repeated['plus'])
            np.testing.assert_array_equal(state['background'], sensitivity['background'])
            t = np.arange(CONFIG['n_times']) / CONFIG['sampling_frequency_hz']
            static_wave = np.exp(-((t - .12) / .06) ** 2 / 2)
            static_wave /= static_wave.max()
            background_patch = runner.patch(bank['source_positions_m'], CONFIG['background_centre_m'], CONFIG['background_width_m'])
            for h, parameters in enumerate(state['parameters']):
                common = CONFIG['static_background_peak_am'] * parameters['static_background_coefficient'] * background_patch[:, None] * static_wave
                expected_peak = CONFIG['signal_peak_am'] * parameters['source_amplitude_factor']
                for label in ['plus', 'minus']:
                    component = state[label][h] - common
                    self.assertAlmostEqual(float(np.linalg.norm(component, axis=0).max()) / expected_peak, 1., places=13)
                    np.testing.assert_allclose(sensitivity[label][h] - common,
                        (CONFIG['sensitivity_signal_peak_am'] / CONFIG['signal_peak_am']) * component, rtol=1e-13, atol=1e-24)
                self.assertTrue(parameters['fixed_across_observations'])

    def test_all_declared_source_and_head_seed_blocks_are_disjoint(self):
        source_seeds, head_seeds = [], []
        for name, definition in CONFIG['geometry_banks'].items():
            head_seeds += [definition['seed_base'] + i * CONFIG['head_seed_stride'] for i in range(36)]
            for task in CONFIG['tasks']:
                for domain, indices in [('development', range(8)), ('validation', range(8, 12)),
                                        ('check_id', range(12, 36)), ('check_shift', range(12, 36))]:
                    source_seeds += [runner.seed(CONFIG, 'source', name, task, domain, i) for i in indices]
        self.assertEqual(len(source_seeds), 1440)
        self.assertEqual(len(source_seeds), len(set(source_seeds)))
        self.assertEqual(len(head_seeds), len(set(head_seeds)))
        self.assertFalse(set(source_seeds) & set(head_seeds))
        numerical_seeds = []
        for name in CONFIG['geometry_banks']:
            for task in CONFIG['tasks']:
                for noise in CONFIG['representation_noise_regimes']:
                    key = f'{name}__{task}__{noise}'
                    numerical_seeds.append(runner.seed(CONFIG, 'development_mi', key))
                    for domain in ['check_id', 'check_shift']:
                        numerical_seeds.append(runner.seed(CONFIG, 'check_mi', key + '__' + domain))
        self.assertEqual(len(numerical_seeds), len(set(numerical_seeds)))
        self.assertFalse(set(numerical_seeds) & set(source_seeds + head_seeds))

    def test_moment_and_pooled_multitask_pca_match_independent_dense_covariance(self):
        first, second = law_fixture(), law_fixture(612, offset=np.array([8., 0., 0., 0.])[:, None])
        for law in [first, second]:
            dense = independent_dense_covariance([law])
            expected = spatial_marginal(dense, law.n_channels, law.n_times)
            np.testing.assert_allclose(runner.moment(law), expected, atol=2e-14)
        expected = spatial_marginal(independent_dense_covariance([first, second]), first.n_channels, first.n_times)
        _, vectors = np.linalg.eigh(expected)
        bases = runner.spatial_designs(first, [first, second])
        for rank in range(1, 5):
            actual = bases['generic_pca'][:rank]
            reference = vectors[:, -rank:]
            np.testing.assert_allclose(actual.T @ actual, reference @ reference.T, atol=3e-14)
        _, matched = np.linalg.eigh(runner.moment(first))
        np.testing.assert_allclose(np.outer(bases['task_pca'][0], bases['task_pca'][0]),
                                   np.outer(matched[:, -1], matched[:, -1]), atol=2e-14)

    def test_fisher_surrogate_whitens_background_time_and_solves_generalized_problem(self):
        law = law_fixture(714)
        temporal_precision = np.linalg.inv(law.temporal_covariance)
        effective = law.spatial_covariance.copy()
        signal = np.zeros_like(effective)
        for h in range(law.n_heads):
            vector = law.background[h] * law.background_sd[h]
            effective += vector @ temporal_precision @ vector.T / (law.n_heads * law.n_times)
            delta = law.plus[h] - law.minus[h]
            signal += delta @ temporal_precision @ delta.T / law.n_heads
        _, vectors = eigh(signal, effective)
        reference = vectors[:, ::-1].T
        actual = runner.spatial_designs(law, [law])['task_fisher']
        np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1., atol=2e-14)
        noise_metric = actual @ effective @ actual.T
        np.testing.assert_allclose(noise_metric - np.diag(np.diag(noise_metric)), 0., atol=4e-14)
        rayleigh = np.diag(actual @ signal @ actual.T) / np.diag(noise_metric)
        self.assertTrue(np.all(np.diff(rayleigh) <= 1e-12))
        for rank in range(1, 5):
            q1 = np.linalg.qr(actual[:rank].T)[0]
            q2 = np.linalg.qr(reference[:rank].T)[0]
            np.testing.assert_allclose(q1 @ q1.T, q2 @ q2.T, atol=3e-14)

    def test_variance_retention_is_physical_rowspace_trace_and_scale_invariant(self):
        law = law_fixture(661)
        spatial = np.array([[1., .2, 0., 0.], [0., 0., .3, 1.]])
        temporal = block_average_matrix(5, 2)
        dense = independent_dense_covariance([law])
        qs = np.linalg.qr(spatial.T)[0]
        qt = np.linalg.qr(temporal.T)[0]
        projector = np.kron(qs @ qs.T, qt @ qt.T)
        expected = np.trace(projector @ dense) / np.trace(dense)
        self.assertAlmostEqual(runner.variance_retention(law, spatial, temporal), expected, places=14)
        rescaled = np.array([[2., .4], [.2, .7]]) @ spatial
        time_recoded = np.array([[1., .2, 0.], [0., 2., .4], [.1, 0., .3]]) @ temporal
        self.assertAlmostEqual(runner.variance_retention(law, rescaled, time_recoded), expected, places=14)

    def test_development_design_is_unchanged_by_check_operator_perturbation(self):
        bank = synthetic_bank()
        changed = copy.deepcopy(bank)
        changed['gains_v_per_am'][12:] = np.random.default_rng(291).normal(size=(24, 32, 32)) * 1000
        a = runner.source_means(bank, 'spatial_lateral', 'development', CONFIG)
        b = runner.source_means(changed, 'spatial_lateral', 'development', CONFIG)
        np.testing.assert_array_equal(a['plus'], b['plus'])
        law_a = runner.experiment(bank, a, CONFIG['noise_regimes']['joint'], 19, CONFIG)
        law_b = runner.experiment(changed, b, CONFIG['noise_regimes']['joint'], 19, CONFIG)
        transforms_a, specs_a = runner.candidates(law_a, [law_a], CONFIG)
        transforms_b, specs_b = runner.candidates(law_b, [law_b], CONFIG)
        self.assertEqual(specs_a, specs_b)
        self.assertEqual(len(transforms_a), 36)
        for name in transforms_a:
            rs, tb = transforms_a[name]
            np.testing.assert_array_equal(rs, transforms_b[name][0])
            np.testing.assert_array_equal(tb, transforms_b[name][1])
            self.assertEqual(rs.shape[0] * tb.shape[0], specs_a[name]['budget'])
            self.assertEqual(np.linalg.matrix_rank(rs), rs.shape[0])
            self.assertEqual(np.linalg.matrix_rank(tb), tb.shape[0])
            self.assertTrue(np.all(tb >= 0))
            np.testing.assert_allclose(tb.sum(axis=1), 1.)
            for row in tb:
                support = np.flatnonzero(row)
                np.testing.assert_array_equal(support, np.arange(support[0], support[-1] + 1))
                np.testing.assert_allclose(row[support], 1 / len(support))
        check_a = runner.source_means(bank, 'spatial_lateral', 'check_id', CONFIG)
        check_b = runner.source_means(changed, 'spatial_lateral', 'check_id', CONFIG)
        self.assertGreater(np.linalg.norm(check_a['plus'] - check_b['plus']), 1e-6)

    def test_prospective_policies_and_frozen_roundoff_ties(self):
        law = law_fixture(c=18, t=64)
        transforms, specs = runner.candidates(law, [law], CONFIG)
        predicted = {name: .1 for name in transforms}
        prototype = dict(predicted)
        variance = dict(predicted)
        predicted['task_fisher_b64_s1_t64'] = .95
        predicted['task_pca_b64_s2_t32'] = .8
        predicted['generic_pca_b64_s4_t16'] = .9
        prototype['generic_pca_b64_s8_t8'] = .7
        variance['task_pca_b64_s4_t16'] = .7
        result = runner.decide(transforms, specs, predicted, prototype, variance, CONFIG)
        selected = {row['policy']: row['candidate'] for row in result if row['budget'] == 64}
        self.assertEqual(selected['development_information'], 'task_fisher_b64_s1_t64')
        self.assertEqual(selected['development_task_pca'], 'task_pca_b64_s2_t32')
        self.assertEqual(selected['mean_prototype_information'], 'generic_pca_b64_s8_t8')
        self.assertEqual(selected['variance_retention'], 'task_pca_b64_s4_t16')
        self.assertEqual(selected['fixed_generic_pca'], 'generic_pca_b64_s4_t16')
        self.assertEqual(runner.best(['first', 'second'], {'first': .7, 'second': .7 + 5e-13}), 'first')
        self.assertEqual(runner.best(['first', 'second'], {'first': .7, 'second': .7 + 5e-11}), 'second')

    def test_transfer_evaluates_saved_matrix_without_refitting_and_preserves_choice(self):
        # Deliberately synthetic law and fake posterior integrands: tests
        # orchestration/selection, not any physical-bank scientific endpoint.
        law = prepared_experiment(np.ones((2, 2, 4)), np.zeros((2, 2, 4)), np.eye(2), np.eye(4))
        transforms = {'first': (np.array([[1., 0.]]), block_average_matrix(4, 2)),
                      'second': (np.array([[0., 1.]]), block_average_matrix(4, 2))}
        design = {'transforms': transforms, 'specs': {name: {'budget': 64, 'method': 'fixture'} for name in transforms},
                  'predicted_full_bits': .9, 'predicted': {'first': .7, 'second': .6},
                  'selections': [{'budget': 64, 'policy': 'development_information', 'candidate': 'first'}]}
        snapshot = {name: (rs.copy(), tb.copy()) for name, (rs, tb) in transforms.items()}
        def fake_information(experiment, encoder, **kwargs):
            self.assertIs(experiment, law)
            self.assertIs(encoder, transforms)
            values = {'first': np.full(64, .60), 'second': np.full(64, .65)}
            return {'full': {'bits': .8}, 'representations': {name: {'bits': float(value.mean())} for name, value in values.items()},
                    'losses': {name: {} for name in values}, 'metadata': {},
                    'sample_arrays': {'labels': np.ones(64), 'head_indices': np.zeros(64, int),
                        'full': {'information': np.full(64, .8)}, 'known_head': {'information': np.full(64, .9)},
                        'representations': {name: {'information': value} for name, value in values.items()}}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'inputs').mkdir()
            (output / 'integration').mkdir()
            with mock_patch.object(runner, 'OUT', output), \
                 mock_patch.object(runner, 'source_means', return_value={'parameters': []}), \
                 mock_patch.object(runner, 'experiment', return_value=law), \
                 mock_patch.object(runner, 'paired_task_information', side_effect=fake_information), \
                 mock_patch.object(runner, 'spatial_designs', side_effect=AssertionError('No held-out design refit allowed')):
                rows, decisions, metadata = runner.check_family({'name': 'heldout_anatomy_fixture'}, 'toy', 'check_shift', 'joint', design, CONFIG,
                                                                key_override='transfer_fixture')
        self.assertEqual(decisions[0]['candidate'], 'first')
        self.assertEqual(decisions[0]['best_candidate'], 'second')
        self.assertAlmostEqual(decisions[0]['point_regret_bits'], .05)
        self.assertEqual(decisions[0]['regret_numerical_summary']['multiplicity'], 4)
        self.assertEqual(metadata['bank'], 'heldout_anatomy_fixture')
        self.assertTrue(rows[0]['retention_fraction_eligible'])
        self.assertIsNotNone(rows[0]['retention_fraction_joint_numerical_ci95'])
        for name, (rs, tb) in transforms.items():
            np.testing.assert_array_equal(rs, snapshot[name][0])
            np.testing.assert_array_equal(tb, snapshot[name][1])

    def test_fixed_strata_predictor_score_uses_valid_hoeffding_and_declared_law_advantage(self):
        labels = np.tile(np.r_[np.ones(4), -np.ones(4)], 3)
        heads = np.repeat(np.arange(3), 8)
        posterior = np.tile(np.r_[np.full(4, .75), np.full(4, .25)], 3)
        result = runner.prediction_metrics(labels, posterior, posterior, heads)
        self.assertAlmostEqual(result['log_loss_bits'], -np.log2(.75))
        self.assertAlmostEqual(result['accessible_information_lower_bound_estimate_bits'], 1 + np.log2(.75))
        self.assertEqual(result['paired_error_excess'], 0.)
        self.assertNotIn('empirical_bernstein_lower', result['accessible_information_numerical_summary'])
        self.assertIn('independent stratified epoch', result['accessible_information_numerical_summary']['interval_scope'])
        self.assertIn('point estimate is not exact MI', result['lower_bound_scope'])
        self.assertIn('zero subject calibration', CONFIG['side_information'])
        self.assertIn('Oracle knows true finite state bank', CONFIG['side_information'])
        self.assertIn('fitted learners receive only development examples', CONFIG['side_information'])


if __name__ == '__main__':
    unittest.main()
