#!/usr/bin/env python3
"""Independent read-only audit; writes only evidence/ceiling_*integrity*.json.

Uses NumPy/SciPy directly, never imports the scientific runner or its utility.
No model is refitted and no Monte Carlo integration is rerun. First64 actual
epochs from eight prespecified cases are replayed for independent likelihood
checks. Earlier studies and all scientific results remain immutable.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import zlib

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '1')
import numpy as np
from scipy.linalg import helmert, solve_triangular, cho_factor, cho_solve
from scipy.special import entr, expit, logsumexp, ndtr, roots_hermitenorm
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/ceiling_benchmark_v1'
ASSETS = ROOT / 'resources/ceiling_banks_v1'
EVIDENCE = ROOT / 'evidence/ceiling_benchmark_integrity_checks.json'
PRIOR_SHA = {
    'results/pilot_v1/run_manifest.json': '1da7073c054376cf35a37c45d05839f5a16fe815e4c4524d4d78cc2aabf41ec9',
    'results/full_study_v1/run_manifest.json': '9778bb0d56df4d08bed784e056206a878dcbbaab99eb7e274ae137c09c7f14d6',
    'results/insight_study_v1/run_manifest.json': 'ef81722276e2d75d6b50c935c655a9bda25a0262081d8f73be7d12edb1e761bd'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {name: f[name] for name in f.files}


def entropy(p):
    return (entr(p) + entr(1 - p)) / np.log(2)


def bounded(values, low, high, multiplicity=1, confidence=.95):
    values = np.asarray(values, float)
    n, width = len(values), high - low
    mean, variance = float(values.mean()), float(values.var(ddof=1))
    alpha = (1 - confidence) / multiplicity
    h = width * np.sqrt(np.log(2 / alpha) / (2 * n))
    b = np.sqrt(2 * variance * np.log(4 / alpha) / n) + 7 * width * np.log(4 / alpha) / (3 * (n - 1))
    return {'estimate': mean, 'standard_error': np.sqrt(variance / n),
        'confidence_lower': max(low, mean - h), 'confidence_upper': min(high, mean + h),
        'empirical_bernstein_lower': max(low, mean - b), 'empirical_bernstein_upper': min(high, mean + b),
        'hoeffding_halfwidth': float(h), 'empirical_bernstein_halfwidth': float(b)}


def bounded_saved(mean, standard_error, n, low, high, multiplicity):
    """Independent intervals from saved mean and sample-SE, without new draws."""
    variance = standard_error ** 2 * n
    width = high - low; alpha = .05 / multiplicity
    h = width * np.sqrt(np.log(2 / alpha) / (2 * n))
    b = np.sqrt(2 * variance * np.log(4 / alpha) / n) + 7 * width * np.log(4 / alpha) / (3 * (n - 1))
    return {'Hoeffding': (max(low, mean-h), min(high, mean+h)),
            'empirical_Bernstein': (max(low, mean-b), min(high, mean+b))}


class Audit:
    def __init__(self):
        self.failures, self.flags, self.hashes, self.maximum_residual = [], [], {}, {}
        self.counts = Counter()

    def require(self, condition, label, detail=None):
        if not condition:
            self.failures.append({'check': label, 'detail': detail})

    def close(self, label, observed, expected, *, atol=2e-11, rtol=2e-11):
        observed, expected = np.asarray(observed, float), np.asarray(expected, float)
        if observed.shape != expected.shape:
            self.require(False, label, {'observed_shape': list(observed.shape), 'expected_shape': list(expected.shape)})
            return
        difference = float(np.max(np.abs(observed - expected))) if observed.size else 0.
        self.maximum_residual[label.split(':')[0]] = max(self.maximum_residual.get(label.split(':')[0], 0.), difference)
        self.require(np.all(np.isfinite(observed)) and np.all(np.isfinite(expected)) and np.allclose(observed, expected, atol=atol, rtol=rtol),
                     label, {'maximum_absolute_difference': difference})

    def mapping(self, label, mapping, base=ROOT):
        failures = [name for name, digest in mapping.items() if not (base / name).is_file() or sha(base / name) != digest]
        self.hashes[label] = {'entries': len(mapping), 'failures': failures, 'passed': not failures}
        self.require(not failures, label, failures)


def summary_check(audit, name, summary, information, risk):
    i, r = bounded(information, 0., 1.), bounded(risk, 0., .5)
    pairs = {'bits': i['estimate'], 'standard_error': i['standard_error'],
        'confidence_lower_bits': i['confidence_lower'], 'confidence_upper_bits': i['confidence_upper'],
        'empirical_bernstein_lower_bits': i['empirical_bernstein_lower'], 'empirical_bernstein_upper_bits': i['empirical_bernstein_upper'],
        'bayes_error': r['estimate'], 'bayes_error_standard_error': r['standard_error'],
        'bayes_error_confidence_lower': r['confidence_lower'], 'bayes_error_confidence_upper': r['confidence_upper'],
        'bayes_error_empirical_bernstein_lower': r['empirical_bernstein_lower'], 'bayes_error_empirical_bernstein_upper': r['empirical_bernstein_upper']}
    for field, value in pairs.items():
        if field in summary:
            audit.close('integration_summary:' + name + ':' + field, summary[field], value)
    audit.close('entropy_risk_identity:' + name, information, 1 - entropy(risk), atol=2e-13)
    audit.require(np.all((risk >= 0) & (risk <= .5)), 'risk_range:' + name)
    audit.require(np.all((information >= 0) & (information <= 1)), 'information_range:' + name)
    audit.counts['integration_endpoint_summaries'] += 1


def bounded_check(audit, name, summary, values, low, high, multiplicity=1):
    expected = bounded(values, low, high, multiplicity)
    for field, value in expected.items():
        if field in summary:
            audit.close('bounded_summary:' + name + ':' + field, summary[field], value)


def white(array, cs, ct):
    # Independently organized solves per epoch, rather than the implementation's
    # combined C x (N*T) reshape; no full epoch covariance is materialized.
    ls, lt = np.linalg.cholesky(cs), np.linalg.cholesky(ct)
    return np.stack([solve_triangular(lt, solve_triangular(ls, epoch, lower=True).T, lower=True).T for epoch in array])


GH_X, GH_W = roots_hermitenorm(256)
GH_W /= np.sqrt(2 * np.pi)


def information_gaussian(d):
    d = np.atleast_1d(d)
    result = 1 - np.logaddexp(0., -d[:, None] * (d[:, None] / 2 + GH_X)) @ GH_W / np.log(2)
    return np.clip(result, 0., 1.)


def oracle(plus, minus, background, sd, cs, ct):
    delta = white(plus - minus, cs, ct).reshape(len(plus), -1)
    b = white(background, cs, ct).reshape(len(plus), -1) * np.asarray(sd)[:, None]
    separation2 = np.sum(delta ** 2, axis=1) - np.sum(delta * b, axis=1) ** 2 / (1 + np.sum(b ** 2, axis=1))
    d = np.sqrt(np.maximum(0., separation2))
    return {'separations': d, 'information_bits': information_gaussian(d), 'bayes_errors': ndtr(-d / 2)}


def oracle_check(audit, name, expected, saved):
    for field in ['separations', 'information_bits', 'bayes_errors']:
        audit.close('oracle_' + field + ':' + name, saved[field], expected[field], atol=2e-10)
    audit.close('oracle_mean_information:' + name, saved['average_information_bits'], expected['information_bits'].mean())
    audit.close('oracle_mean_risk:' + name, saved['average_bayes_error'], expected['bayes_errors'].mean())
    audit.counts['known_state_oracles'] += 1


def transform(law, rs, tb):
    mapped = lambda array: np.stack([rs @ epoch @ tb.T for epoch in array])
    return {'plus': mapped(law['plus']), 'minus': mapped(law['minus']), 'background': mapped(law['background']),
        'background_sd': law['background_sd'], 'spatial_covariance': rs @ law['spatial_covariance'] @ rs.T,
        'temporal_covariance': tb @ law['temporal_covariance'] @ tb.T}


def oracle_law(law):
    return oracle(law['plus'], law['minus'], law['background'], law['background_sd'], law['spatial_covariance'], law['temporal_covariance'])


def spatial_moment(law):
    means = np.concatenate([law['plus'], law['minus']])
    residual = means - means.mean(axis=0)
    k, c, t = law['plus'].shape
    covariance = sum(epoch @ epoch.T for epoch in residual) / (2 * k * t)
    covariance += law['spatial_covariance'] * np.trace(law['temporal_covariance']) / t
    covariance += sum((b * sd) @ (b * sd).T for b, sd in zip(law['background'], law['background_sd'])) / (k * t)
    return covariance


def variance_fraction(law, rs, tb):
    qs, qt = np.linalg.qr(rs.T)[0], np.linalg.qr(tb.T)[0]
    means = np.concatenate([law['plus'], law['minus']]); means -= means.mean(axis=0)
    b = law['background'] * law['background_sd'][:, None, None]
    represented = sum(np.sum((qs.T @ x @ qt) ** 2) for x in means) / len(means)
    represented += np.trace(qs.T @ law['spatial_covariance'] @ qs) * np.trace(qt.T @ law['temporal_covariance'] @ qt)
    represented += sum(np.sum((qs.T @ x @ qt) ** 2) for x in b) / len(b)
    total = np.mean(np.sum(means ** 2, axis=(1, 2))) + np.trace(law['spatial_covariance']) * np.trace(law['temporal_covariance'])
    total += np.mean(np.sum(b ** 2, axis=(1, 2)))
    return represented / total


def seed(config, *parts):
    return int(np.random.SeedSequence([config['seed']] + [zlib.crc32(str(p).encode()) for p in parts]).generate_state(1)[0])


def choose(names, scores):
    maximum = max(scores[n] for n in names)
    return next(n for n in names if maximum - scores[n] <= 1e-12)


def reconstruct_source(bank, task, parameters, config, amplitude=None):
    t = np.arange(config['n_times']) / config['sampling_frequency_hz']
    position = bank['source_positions_m']
    def patch(centre, width):
        d = np.sum((position - np.asarray(centre)) ** 2, axis=1)
        v = np.exp(-(d - d.min()) / (2 * width ** 2))
        return v / np.linalg.norm(v)
    bp = patch(config['background_centre_m'], config['background_width_m'])
    bw = np.cos(2 * np.pi * 8 * (t - .12)) * np.exp(-((t - .12) / .08) ** 2 / 2); bw /= np.max(np.abs(bw))
    sw = np.exp(-((t - .12) / .06) ** 2 / 2); sw /= sw.max()
    classes, backgrounds = [], []
    definition = config['tasks'][task]
    for p in parameters:
        gain = bank['gains_v_per_am'][p['head_index']]
        b = gain @ bp
        common = config['static_background_peak_am'] * p['static_background_coefficient'] * b[:, None] * sw
        means = []
        for y in range(2):
            spatial = patch(np.asarray(definition['centres_m'][y]) + p['common_source_jitter_m'], definition['patch_width_m'])
            centre = definition['wave_centres_s'][y] + p['common_latency_jitter_s']
            width = definition['wave_width_s'] * p['wave_width_factor']
            waveform = np.exp(-((t - centre) / width) ** 2 / 2)
            if 'frequencies_hz' in definition:
                waveform *= .5 * (1 + np.cos(2 * np.pi * definition['frequencies_hz'][y] * (t - centre)))
            waveform /= waveform.max()
            peak = config['signal_peak_am'] if amplitude is None else amplitude
            means.append(peak * p['source_amplitude_factor'] * (gain @ spatial)[:, None] * waveform + common)
        classes.append(means)
        backgrounds.append(config['epoch_background_peak_am'] * b[:, None] * bw)
    return {'minus': np.stack([x[0] for x in classes]), 'plus': np.stack([x[1] for x in classes]),
            'background': np.stack(backgrounds), 'background_sd': np.ones(len(parameters))}


def physical_law(bank, source, channels, regime, config):
    indices = bank['montage_order'][:channels]
    q = helmert(channels)
    positions = bank['sensor_positions_m']
    raw_cov = config['sensor_noise_sd_v'] ** 2 * ((1 - regime['spatial_correlation']) * np.eye(len(positions))
        + regime['spatial_correlation'] * np.exp(-cdist(positions, positions) ** 2 / (2 * .04 ** 2)))
    result = {field: np.stack([q @ epoch[indices] for epoch in source[field]]) for field in ['plus', 'minus', 'background']}
    result.update(background_sd=source['background_sd'], spatial_covariance=q @ raw_cov[np.ix_(indices, indices)] @ q.T,
                  temporal_covariance=regime['temporal_ar1'] ** np.abs(np.arange(config['n_times'])[:, None] - np.arange(config['n_times'])[None, :]))
    return result


def direct_rank_one_posterior(samples, plus, minus, b):
    denominator = 1 + np.sum(b * b, axis=1)
    def log_density(means):
        diff = samples[:, None, :] - means[None, :, :]
        return -.5 * (np.sum(diff * diff, axis=2) - np.einsum('nkd,kd->nk', diff, b) ** 2 / denominator + np.log(denominator))
    return expit(logsumexp(log_density(plus), axis=1) - logsumexp(log_density(minus), axis=1))


def dense_posterior(samples, law):
    base = np.kron(law['spatial_covariance'], law['temporal_covariance'])
    lp, lm = [], []
    for h in range(len(law['plus'])):
        b = law['background'][h].ravel() * law['background_sd'][h]
        covariance = base + np.outer(b, b)
        chol = cho_factor(covariance, lower=True)
        logdet = 2 * np.log(np.diag(chol[0])).sum()
        for means, out in [(law['plus'], lp), (law['minus'], lm)]:
            diff = samples - means[h].ravel()
            out.append(-.5 * (np.sum(diff * cho_solve(chol, diff.T).T, axis=1) + logdet))
    return expit(logsumexp(np.array(lp), axis=0) - logsumexp(np.array(lm), axis=0))


def main():
    if not (OUT / 'run_manifest.json').exists():
        raise RuntimeError('Scientific run has not completed; no results inspected or audit written')
    started = time.perf_counter(); audit = Audit()
    config, manifest = read(ROOT / 'config/ceiling_benchmark.json'), read(OUT / 'run_manifest.json')
    for label, mapping in [('sources', manifest['source_sha256_after']), ('outputs', manifest['output_sha256'])]:
        audit.mapping(label, mapping)
    audit.require(manifest['source_sha256_before'] == manifest['source_sha256_after'], 'source_stability')
    freeze = read(OUT / 'pre_outcome_freeze.json')
    audit.require(freeze['source_sha256'] == manifest['source_sha256_before'], 'initial_source_freeze')
    audit.require(freeze['config_sha256'] == sha(ROOT / 'config/ceiling_benchmark.json') == manifest['config_sha256'], 'scientific_config_freeze')
    asset_manifest = read(ASSETS / 'manifest.json')
    audit.mapping('physical_outputs', asset_manifest['output_sha256'])
    audit.mapping('physical_inputs', asset_manifest['input_sha256'])
    audit.require(sha(ASSETS / 'manifest.json') == manifest['physical_assets_manifest_sha256'], 'physical_manifest_identity')
    audit.require(asset_manifest['config_sha256'] == manifest['config_sha256'], 'physical_config_identity')
    audit.require(asset_manifest['script_sha256'] == sha(ROOT / 'scripts/prepare_ceiling_assets.py'), 'physical_preparation_source')
    for path, digest in PRIOR_SHA.items():
        audit.require(sha(ROOT / path) == digest, 'prior_manifest_preserved:' + path)
        old = read(ROOT / path)
        audit.mapping('prior_outputs:' + path, old['output_sha256'])
        source_key = 'source_sha256_after_run' if 'source_sha256_after_run' in old else 'source_sha256_after'
        audit.mapping('prior_sources:' + path, old[source_key])
        if 'input_sha256_after' in old:
            audit.mapping('prior_inputs:' + path, old['input_sha256_after'])
    tables = {name: read(OUT / (name + '.json')) for name in ['development_predictions', 'prospective_selections', 'development_state_records',
        'representation_conditions', 'decisions', 'case_metadata', 'anatomy_transfer_conditions', 'anatomy_transfer_decisions',
        'sensor_atlas', 'decoder_conditions', 'precision_conditions', 'summary']}
    expected_counts = {'representation_rows': 3456, 'decision_rows': 2016, 'anatomy_transfer_rows': 144,
        'anatomy_transfer_decisions': 84, 'sensor_atlas_rows': 480, 'decoder_rows': 128, 'precision_rows': 96}
    audit.require(manifest['counts'] == expected_counts, 'frozen_table_counts', manifest['counts'])
    for key, table in [('representation_rows', 'representation_conditions'), ('decision_rows', 'decisions'), ('anatomy_transfer_rows', 'anatomy_transfer_conditions'),
                       ('anatomy_transfer_decisions', 'anatomy_transfer_decisions'), ('sensor_atlas_rows', 'sensor_atlas'), ('decoder_rows', 'decoder_conditions'), ('precision_rows', 'precision_conditions')]:
        audit.require(len(tables[table]) == expected_counts[key], 'table_count:' + table, len(tables[table]))
    audit.require(len(tables['development_predictions']) == 1728 and len(tables['prospective_selections']) == 48, 'development_counts')
    audit.require(len(tables['case_metadata']) == 108, 'case_metadata_count')
    banks, head_seeds, head_ids = {}, [], []
    for name, definition in config['geometry_banks'].items():
        bank = npz(ASSETS / (name + '.npz'))
        params = read(ASSETS / (name + '_parameters.json')); bank['parameters'] = params
        banks[name] = bank
        audit.require(Counter(p['split'] for p in params) == {'development': 8, 'validation': 4, 'check': 24}, 'physical_split:' + name)
        audit.require(bank['gains_v_per_am'].shape == (36, 256 if definition['kind'] == 'sphere' else 64, 32), 'gain_shape:' + name)
        audit.require(np.all(np.isfinite(bank['gains_v_per_am'])), 'gain_finiteness:' + name)
        audit.close('source_orientation_norm:' + name, np.linalg.norm(bank['source_orientations'], axis=1), np.ones(32), atol=1e-6)
        audit.require(len(np.unique(bank['montage_order'])) == bank['gains_v_per_am'].shape[1], 'montage_permutation:' + name)
        for index, p in enumerate(params):
            audit.require(p['seed'] == definition['seed_base'] + index * config['head_seed_stride'], 'physical_seed:' + p['head_id'])
            head_seeds.append(p['seed']); head_ids.append(p['head_id'])
    audit.require(len(head_seeds) == len(set(head_seeds)) == 144, 'all_physical_seeds_unique')
    audit.require(len(head_ids) == len(set(head_ids)), 'all_physical_ids_unique')
    audit.require(np.array_equal(banks['bem_fsaverage']['montage_order'], banks['bem_sample']['montage_order']), 'matched_anatomy_contact_order')
    dev_state = {(r['bank'], r['task']): r['states'] for r in tables['development_state_records']}
    check_state = {(r['bank'], r['task'], r['domain']): r['states'] for r in tables['case_metadata']}
    planned_source_seeds = []
    for name in banks:
        for task in config['tasks']:
            for domain, indices in [('development', range(8)), ('validation', range(8, 12)), ('check_id', range(12, 36)), ('check_shift', range(12, 36))]:
                planned_source_seeds += [seed(config, 'source', name, task, domain, i) for i in indices]
    audit.require(len(planned_source_seeds) == len(set(planned_source_seeds)) == 1440, 'all_planned_source_seeds_unique')
    audit.require(not set(planned_source_seeds) & set(head_seeds), 'head_source_seed_blocks_disjoint')
    used_states = {(n, t, 'development'): p for (n, t), p in dev_state.items()}
    used_states.update(check_state)
    for (name, task, domain), parameters in used_states.items():
        for p in parameters:
            audit.require(p['state_seed'] == seed(config, 'source', name, task, domain, p['head_index']), 'state_seed:' + name + task + domain)
            audit.require(p['fixed_across_observations'], 'persistent_source_state:' + p['head_id'])
    design_manifest = read(OUT / 'development_stage_manifest.json')
    audit.mapping('prospective_designs', design_manifest['design_sha256'])
    audit.require(design_manifest['before_check_outcomes'] and design_manifest['completed_at_utc'] <= manifest['completed_at_utc'], 'development_stage_precedes_completion')
    audit.require(design_manifest['predictions_sha256'] == sha(OUT / 'development_predictions.json') and
                  design_manifest['selections_sha256'] == sha(OUT / 'prospective_selections.json'), 'prospective_choices_unchanged')
    designs = {key: npz(OUT / 'designs' / (key + '.npz')) for key in tables['prospective_selections']}
    development = {key: npz(OUT / 'inputs' / (key + '__development.npz')) for key in designs}
    by_design = {}
    for row in tables['development_predictions']:
        by_design.setdefault(row['design_key'], []).append(row)
    audited_forecast = {}
    for key, rows in by_design.items():
        law = development[key]; design = designs[key]
        bank_name, task, noise = key.split('__')
        raw = reconstruct_source(banks[bank_name], task, dev_state[bank_name, task], config)
        physical = physical_law(banks[bank_name], raw, 19, config['noise_regimes'][noise], config)
        for field in ['plus', 'minus', 'background', 'spatial_covariance', 'temporal_covariance']:
            audit.close('physical_development_' + field + ':' + key, law[field], physical[field], atol=1e-17, rtol=1e-10)
        arrays = npz(OUT / 'integration' / (key + '__development.npz'))
        generic_laws = [development[bank_name + '__' + other + '__' + noise] for other in config['tasks']]
        centres = [np.concatenate([e['plus'], e['minus']]).mean(axis=0) for e in generic_laws]
        overall = np.mean(centres, axis=0)
        generic = np.mean([spatial_moment(e) for e in generic_laws], axis=0) + np.mean([(c - overall) @ (c - overall).T / config['n_times'] for c in centres], axis=0)
        matched = spatial_moment(law)
        inverse_time = np.linalg.inv(law['temporal_covariance']); k = len(law['plus'])
        effective = law['spatial_covariance'].copy(); signal = np.zeros_like(effective)
        for h in range(k):
            b = law['background'][h] * law['background_sd'][h]
            effective += b @ inverse_time @ b.T / (k * config['n_times'])
            d = law['plus'][h] - law['minus'][h]
            signal += d @ inverse_time @ d.T / k
        prediction, prototype, variance = {}, {}, {}
        for row in rows:
            name = row['candidate']; rs, tb = design[name + '__spatial'], design[name + '__temporal']
            s, t = row['spatial_features'], row['time_bins']
            audit.require(rs.shape == (s, 18) and tb.shape == (t, 64) and s * t == row['budget'], 'physical_feature_budget:' + key + name)
            expected_tb = np.zeros_like(tb)
            for j in range(t): expected_tb[j, j * (64 // t):(j + 1) * (64 // t)] = t / 64
            audit.close('raw_temporal_weights:' + key + name, tb, expected_tb, atol=0., rtol=0.)
            audit.require(np.linalg.matrix_rank(rs) == s, 'spatial_encoder_rank:' + key + name)
            if row['method'] in ('generic_pca', 'task_pca'):
                covariance = generic if row['method'] == 'generic_pca' else matched
                audit.close('pca_orthonormality:' + key + name, rs @ rs.T, np.eye(s), atol=2e-10)
                ratio = np.trace(rs @ covariance @ rs.T) / np.sum(np.linalg.eigvalsh(covariance)[-s:])
                audit.close('pca_optimal_variance:' + key + name, ratio, 1., atol=2e-9)
            else:
                metric = rs @ effective @ rs.T; scale = np.sqrt(np.diag(metric))
                audit.close('fisher_generalized_orthogonality:' + key + name, metric / np.outer(scale, scale), np.eye(s), atol=2e-8)
                eigenvalue = np.diag(rs @ signal @ rs.T) / np.diag(metric)
                audit.require(np.all(np.diff(eigenvalue) <= 1e-8), 'fisher_generalized_order:' + key + name)
            transformed = transform(law, rs, tb)
            oracle_check(audit, key + name, oracle_law(transformed), row['information']['known_head_exact'])
            i, r = arrays[name + '__information'], arrays[name + '__risk']
            summary_check(audit, key + name, row['information'], i, r)
            audit.close('development_full_point:' + key + name, row['development_full_bits'], arrays['full__information'].mean())
            audit.close('development_prediction_point:' + key + name, row['predicted_bits'], i.mean())
            bounded_check(audit, key + name, row['paired_loss']['information_loss'], arrays['full__information'] - i, -1., 1.)
            audit.close('development_predicted_loss:' + key + name, row['predicted_loss_bits'], arrays['full__information'].mean() - i.mean())
            prediction[name] = float(i.mean())
            proto = {field: transformed[field].mean(axis=0, keepdims=True) for field in ['plus', 'minus', 'background']}
            proto.update(background_sd=np.ones(1), spatial_covariance=transformed['spatial_covariance'], temporal_covariance=transformed['temporal_covariance'])
            prototype[name] = float(oracle_law(proto)['information_bits'][0])
            variance[name] = variance_fraction(law, rs, tb)
            audit.close('development_prototype_prediction:' + key + name, row['mean_prototype_bits'], prototype[name])
            audit.close('development_variance_prediction:' + key + name, row['variance_retention_fraction'], variance[name])
        audited_forecast[key] = (float(arrays['full__information'].mean()), prediction)
        for choice in tables['prospective_selections'][key]:
            names = [r['candidate'] for r in rows if r['budget'] == choice['budget']]
            policy = choice['policy']
            if policy == 'fixed_generic_pca':
                budget = choice['budget']; time_bins = {32: 8, 64: 16, 128: 32}[budget]
                expected = f'generic_pca_b{budget}_s4_t{time_bins}'
            else:
                scores = prototype if policy == 'mean_prototype_information' else variance if policy == 'variance_retention' else prediction
                if policy.startswith('development_') and policy != 'development_information':
                    names = [n for n in names if n.startswith(policy[len('development_'):] + '_')]
                expected = choose(names, scores)
            audit.require(choice['candidate'] == expected, 'prospective_policy:' + key + policy)
            audit.counts['prospective_policy_choices'] += 1
    print('Hashes, physical laws and all development predictions checked', flush=True)
    cases = {}
    for table_name in ['representation_conditions', 'anatomy_transfer_conditions', 'precision_conditions']:
        for row in tables[table_name]: cases.setdefault((table_name, row['case_key']), []).append(row)
    hidden_atlas_count = sum('hidden_state_information' in r for r in tables['sensor_atlas'])
    global_multiplicity = 2 * (sum(len(v) for v in cases.values()) + len(cases)) + 3 * hidden_atlas_count
    oracle_inequalities = []
    for (table_name, key), rows in cases.items():
        precision = table_name == 'precision_conditions'
        first = rows[0]; bank_name, task, noise, domain = [first[f] for f in ['bank', 'task', 'noise', 'domain']]
        law = npz(OUT / 'inputs' / ((f'{bank_name}__{task}__{noise}__{domain}' if precision else key) + '.npz'))
        arrays = npz(OUT / ('precision' if precision else 'integration') / (key + '.npz'))
        expected_n = config['precision_audit']['draws'] if precision else config['check_information_draws']
        audit.require(len(arrays['labels']) == expected_n and np.all(np.isin(arrays['labels'], [-1, 1])), 'integration_labels:' + key)
        audit.require(np.all((arrays['heads'] >= 0) & (arrays['heads'] < 24)), 'integration_heads:' + key)
        design_key = ('bem_fsaverage' if table_name == 'anatomy_transfer_conditions' else bank_name) + '__' + task + '__' + noise
        design = designs[design_key]
        raw = reconstruct_source(banks[bank_name], task, check_state[bank_name, task, domain], config)
        physical = physical_law(banks[bank_name], raw, 19, config['noise_regimes'][noise], config)
        for field in ['plus', 'minus', 'background', 'spatial_covariance', 'temporal_covariance']:
            audit.close('physical_check_' + field + ':' + key, law[field], physical[field], atol=1e-17, rtol=1e-10)
        full_reference = oracle_law(law)
        summary_check(audit, key, first['sensor_information'], arrays['full__information'], arrays['full__risk'])
        for row in rows:
            name = row['candidate']; rs, tb = design[name + '__spatial'], design[name + '__temporal']
            transformed = transform(law, rs, tb); represented_oracle = oracle_law(transformed)
            oracle_check(audit, key + name, represented_oracle, row['representation_information']['known_head_exact'])
            audit.require(np.all(represented_oracle['separations'] <= full_reference['separations'] + 2e-10), 'exact_known_state_DPI:' + key + name)
            oracle_check(audit, key, full_reference, row['known_state_reference'])
            info, risk = arrays[name + '__information'], arrays[name + '__risk']
            summary_check(audit, key + name, row['representation_information'], info, risk)
            audit.close('represented_point_bits:' + key + name, row['representation_bits'], info.mean())
            audit.close('represented_point_loss:' + key + name, row['loss_bits'], arrays['full__information'].mean() - info.mean())
            forecast_full, forecast = audited_forecast[design_key]
            expected_forecast_loss = forecast_full - forecast[name]
            audit.close('heldout_frozen_forecast:' + key + name, row['predicted_loss_bits'], expected_forecast_loss)
            audit.close('heldout_forecast_absolute_error:' + key + name, row['absolute_prediction_error_bits'],
                abs(expected_forecast_loss - (arrays['full__information'].mean()-info.mean())))
            bounded_check(audit, key + name, row['paired_loss']['information_loss'], arrays['full__information'] - info, -1., 1.)
            bounded_check(audit, key + name, row['paired_loss']['bayes_risk_increase'], risk - arrays['full__risk'], -.5, .5)
            fj, rj = bounded(arrays['full__information'], 0., 1., 2), bounded(info, 0., 1., 2)
            eligible = fj['confidence_lower'] >= config['retention_fraction_min_full_information_bits']
            audit.require(row['retention_fraction_eligible'] == eligible, 'retention_eligibility:' + key + name)
            if eligible:
                interval = [max(0., rj['confidence_lower'] / fj['confidence_upper']), min(1., rj['confidence_upper'] / fj['confidence_lower'])]
                audit.close('retention_ratio_interval:' + key + name, row['retention_fraction_joint_numerical_ci95'], interval)
                audit.close('retention_ratio:' + key + name, row['sensor_relative_retention'], info.mean() / arrays['full__information'].mean())
            else:
                audit.require(row['sensor_relative_retention'] is None and row['retention_fraction_joint_numerical_ci95'] is None, 'retention_unresolved:' + key + name)
            for endpoint, values, lo, hi in [('MI_loss', arrays['full__information'] - info, -1., 1.), ('risk_increase', risk - arrays['full__risk'], -.5, .5)]:
                s = bounded(values, lo, hi, global_multiplicity)
                for method, upper in [('Hoeffding', s['confidence_upper']), ('empirical_Bernstein', s['empirical_bernstein_upper'])]:
                    if upper < -1e-12: audit.flags.append({'case': key, 'candidate': name, 'endpoint': endpoint, 'interval': method, 'upper': upper})
            audit.counts['represented_conditions'] += 1
        for endpoint, values, lo, hi in [('known_minus_hidden_MI', arrays['known_head__information'] - arrays['full__information'], -1., 1.),
                                        ('hidden_minus_known_risk', arrays['full__risk'] - arrays['known_head__risk'], -.5, .5)]:
            interval = bounded(values, lo, hi, global_multiplicity)
            if interval['confidence_upper'] < -1e-12 or interval['empirical_bernstein_upper'] < -1e-12:
                audit.flags.append({'case': key, 'endpoint': endpoint, 'intervals': interval})
        summary_check(audit, key + ':known', {}, arrays['known_head__information'], arrays['known_head__risk'])
        oracle_inequalities.append(float(np.max(np.asarray(full_reference['information_bits']))))
    for table_name in ['decisions', 'anatomy_transfer_decisions']:
        last_key, arrays = None, None
        for row in tables[table_name]:
            key = row['case_key']
            if key != last_key:
                arrays = npz(OUT / 'integration' / (key + '.npz'))
                last_key = key
            source = 'anatomy_transfer_conditions' if table_name == 'anatomy_transfer_decisions' else 'representation_conditions'
            names = [r['candidate'] for r in cases[source, key] if r['budget'] == row['budget']]
            bank = 'bem_fsaverage' if table_name == 'anatomy_transfer_decisions' else row['bank']
            dkey = bank + '__' + row['task'] + '__' + row['noise']
            frozen = next(s for s in tables['prospective_selections'][dkey] if s['budget'] == row['budget'] and s['policy'] == row['policy'])
            audit.require(row['candidate'] == frozen['candidate'], 'unchanged_heldout_selection:' + key + row['policy'])
            actual = {n: float(arrays[n + '__information'].mean()) for n in names}
            expected_best = choose(names, actual); chosen = row['candidate']
            audit.require(row['best_candidate'] == expected_best, 'retrospective_pool_choice:' + key + row['policy'])
            audit.close('selection_regret_point:' + key + row['policy'], row['point_regret_bits'], actual[expected_best] - actual[chosen])
            intervals = [bounded(arrays[n + '__information'] - arrays[chosen + '__information'], -1., 1., len(names)**2) for n in names]
            for field in ['confidence_lower', 'confidence_upper', 'empirical_bernstein_lower', 'empirical_bernstein_upper']:
                audit.close('selection_regret_interval:' + key + row['policy'] + field, row['regret_numerical_summary'][field], max(0., max(v[field] for v in intervals)))
            audit.counts['heldout_policy_choices'] += 1
    print('All saved integration endpoints, paired losses and held-out policies checked', flush=True)
    replay = []
    for bank in banks:
        for task in config['decoder_tasks']:
            key = bank + '__' + task + '__joint__check_id'
            law, arrays = npz(OUT / 'inputs' / (key + '.npz')), npz(OUT / 'integration' / (key + '.npz'))
            streams = [np.random.default_rng(child) for child in np.random.SeedSequence(seed(config, 'check_mi', key)).spawn(4)]
            heads = streams[0].choice(24, 64, p=np.full(24, 1/24)); labels = streams[1].integers(0, 2, 64) * 2 - 1
            audit.require(np.array_equal(heads, arrays['heads'][:64]) and np.array_equal(labels, arrays['labels'][:64]), 'replayed_draw_identifiers:' + key)
            plus, minus = white(law['plus'], law['spatial_covariance'], law['temporal_covariance']), white(law['minus'], law['spatial_covariance'], law['temporal_covariance'])
            b = white(law['background'], law['spatial_covariance'], law['temporal_covariance']) * law['background_sd'][:, None, None]
            samples = np.where(labels[:, None, None] > 0, plus[heads], minus[heads]) + streams[2].standard_normal((64, 18, 64))
            samples += streams[3].standard_normal(64)[:, None, None] * b[heads]
            p = direct_rank_one_posterior(samples.reshape(64, -1), plus.reshape(24, -1), minus.reshape(24, -1), b.reshape(24, -1))
            audit.close('physical_likelihood_replay_full_MI:' + key, arrays['full__information'][:64], 1 - entropy(p), atol=3e-11)
            audit.close('physical_likelihood_replay_full_risk:' + key, arrays['full__risk'][:64], np.minimum(p, 1-p), atol=3e-11)
            ls, lt = np.linalg.cholesky(law['spatial_covariance']), np.linalg.cholesky(law['temporal_covariance'])
            raw = np.stack([ls @ x @ lt.T for x in samples])
            dkey = bank + '__' + task + '__joint'
            candidates = []
            for method in ['task_pca', 'task_fisher']:
                selected = next(x['candidate'] for x in tables['prospective_selections'][dkey] if x['budget'] == 64 and x['policy'] == 'development_' + method)
                rs, tb = designs[dkey][selected + '__spatial'], designs[dkey][selected + '__temporal']
                projected_law = transform(law, rs, tb)
                projected_raw = np.stack([rs @ x @ tb.T for x in raw]).reshape(64, 64)
                pp = dense_posterior(projected_raw, projected_law)
                audit.close('physical_likelihood_replay_dense_MI:' + key + selected, arrays[selected + '__information'][:64], 1-entropy(pp), atol=3e-11)
                audit.close('physical_likelihood_replay_dense_risk:' + key + selected, arrays[selected + '__risk'][:64], np.minimum(pp, 1-pp), atol=3e-11)
                candidates.append(selected)
            replay.append({'case': key, 'epochs': 64, 'full_dimensions': 1152, 'dense_representation_dimensions': 64, 'candidates': candidates})
    atlas_groups = {}
    source_cache = {}
    for row in tables['sensor_atlas']:
        bank, task, amp = row['bank'], row['task'], row['source_peak_am']
        cache = (bank, task, amp)
        if cache not in source_cache: source_cache[cache] = reconstruct_source(banks[bank], task, check_state[bank, task, 'check_id'], config, amp)
        law = physical_law(banks[bank], source_cache[cache], row['channels'], config['noise_regimes'][row['noise']], config)
        expected = oracle_law(law); oracle_check(audit, str(cache) + str(row['channels']) + row['noise'], expected, row['known_state_reference'])
        audit.close('atlas_reported_information:' + str(cache) + row['noise'] + str(row['channels']), row['known_state_bits'], expected['information_bits'].mean())
        audit.close('atlas_reported_risk:' + str(cache) + row['noise'] + str(row['channels']), row['known_state_bayes_error'], expected['bayes_errors'].mean())
        atlas_groups.setdefault((bank, task, amp, row['noise']), []).append(row)
        if 'hidden_state_information' in row:
            i = row['hidden_state_information']; oracle_average = expected['information_bits'].mean()
            intervals = bounded_saved(i['bits'], i['standard_error'], i['n_samples'], 0., 1., global_multiplicity)
            for method, (low, high) in intervals.items():
                if low > oracle_average + 2e-10:
                    audit.flags.append({'case': str(cache), 'channels': row['channels'], 'noise': row['noise'],
                        'endpoint': 'atlas_hidden_MI_exceeds_known_state_oracle', 'interval': method, 'lower': low, 'oracle': float(oracle_average)})
    for key, rows in atlas_groups.items():
        rows.sort(key=lambda r: r['channels'])
        for small, large in zip(rows, rows[1:]):
            audit.require(np.all(np.asarray(small['known_state_reference']['separations']) <= np.asarray(large['known_state_reference']['separations']) + 2e-10), 'nested_montage_known_DPI:' + str(key))
            if 'hidden_state_information' in small and 'hidden_state_information' in large:
                si, li = small['hidden_state_information'], large['hidden_state_information']
                for endpoint, mean_key, se_key, maximum in [('MI', 'bits', 'standard_error', 1.), ('risk', 'bayes_error', 'bayes_error_standard_error', .5)]:
                    siv = bounded_saved(si[mean_key], si[se_key], si['n_samples'], 0., maximum, global_multiplicity)
                    liv = bounded_saved(li[mean_key], li[se_key], li['n_samples'], 0., maximum, global_multiplicity)
                    for method in siv:
                        violation = siv[method][0] > liv[method][1] + 1e-12 if endpoint == 'MI' else liv[method][0] > siv[method][1] + 1e-12
                        if violation:
                            audit.flags.append({'case': str(key), 'endpoint': 'nested_hidden_' + endpoint, 'interval': method,
                                'small_channels': small['channels'], 'large_channels': large['channels'], 'small_interval': list(siv[method]), 'large_interval': list(liv[method])})
            audit.counts['nested_montage_pairs'] += 1
    for row in tables['decoder_conditions']:
        key = '__'.join([row['bank'], row['task'], row['features'], row['decoder']])
        saved = npz(OUT / 'models' / (key + '__' + row['domain'] + '__predictions.npz'))
        labels, p, bp, heads = saved['labels'], np.clip(saved['probabilities'], 1e-6, 1-1e-6), saved['bayes_probabilities'], saved['heads']
        error, optimal = (np.where(p >= .5, 1, -1) != labels).astype(float), (np.where(bp >= .5, 1, -1) != labels).astype(float)
        loss = -np.where(labels > 0, np.log2(p), np.log2(1-p))
        for field, expected in [('error', error.mean()), ('bayes_classifier_realized_error', optimal.mean()), ('paired_error_excess', (error-optimal).mean()),
                                ('log_loss_bits', loss.mean()), ('accessible_information_lower_bound_estimate_bits', 1-loss.mean())]:
            audit.close('decoder_endpoint:' + key + row['domain'] + field, row[field], expected)
        bounded_check(audit, key, row['accessible_information_numerical_summary'], 1-loss, 1-math.log2(1e6), 1.)
        audit.require('empirical_bernstein_lower' not in row['accessible_information_numerical_summary'], 'decoder_non_iid_interval_contract:' + key)
        audit.require(len(labels) == 24 * config['decoder_check_epochs_per_head'], 'decoder_epoch_count:' + key)
        for h in np.unique(heads):
            selected = heads == h; detail = next(x for x in row['per_head'] if x['head_index'] == int(h))
            audit.require(np.sum(labels[selected] > 0) == np.sum(labels[selected] < 0) == 64, 'decoder_balanced_head_strata:' + key)
            for field, expected in [('error', error[selected].mean()), ('bayes_classifier_realized_error', optimal[selected].mean()),
                                    ('paired_error_excess', (error[selected]-optimal[selected]).mean()), ('log_loss_bits', loss[selected].mean())]:
                audit.close('decoder_head_endpoint:' + key + str(h) + field, detail[field], expected)
        unit = np.array([x['paired_error_excess'] for x in row['per_head']]); half = 1.96 * unit.std(ddof=1) / np.sqrt(len(unit))
        audit.close('decoder_head_diagnostic_interval:' + key, row['approximate_head_unit_ci95_error_excess'], [unit.mean()-half, unit.mean()+half])
        models = npz(OUT / 'models' / (key + '.npz'))
        audit.require(all(np.all(np.isfinite(v)) for v in models.values()), 'saved_decoder_parameters_finite:' + key)
        audit.require(row['fit']['training_samples'] == 8*config['decoder_training_epochs_per_head'], 'decoder_training_count:' + key)
        if row['decoder'] == 'linear': audit.require(row['fit']['ridge'] in config['decoder_ridges'], 'decoder_ridge_grid:' + key)
        else:
            audit.require(row['fit']['l2'] in config['decoder_mlp_l2'] and row['fit']['validation_samples'] == 4*config['decoder_validation_epochs_per_head'], 'decoder_mlp_selection_contract:' + key)
            audit.require(row['fit']['selection'] == 'validation loss' and row['fit']['epochs_run'] <= config['decoder_mlp_max_epochs'], 'decoder_mlp_epoch_selection:' + key)
        audit.counts['decoder_conditions'] += 1
    summary = tables['summary']
    for row in summary['prediction']:
        cases_ = [r for r in tables['representation_conditions'] if r['bank'] == row['bank'] and r['domain'] == row['domain'] and r['budget'] == 64]
        errors = np.array([abs(r['predicted_loss_bits'] - r['loss_bits']) for r in cases_])
        for field, expected in [('mean_absolute_error_bits', errors.mean()), ('median_absolute_error_bits', np.median(errors)), ('p90_absolute_error_bits', np.quantile(errors, .9))]:
            audit.close('forecast_summary:' + row['bank'] + row['domain'] + field, row[field], expected)
        audit.require(row['criterion_met'] == bool(errors.mean() <= config['primary_forecast_mae_max_bits']), 'forecast_frozen_gate:' + row['bank'] + row['domain'])
    for family, table in [('selection', 'decisions'), ('transfer', 'anatomy_transfer_decisions')]:
        for row in summary[family]:
            chosen = [r for r in tables[table] if r['domain'] == row['domain'] and r['policy'] == row['policy'] and r['budget'] == 64 and (family == 'transfer' or r['bank'] == row['bank'])]
            audit.close('selection_summary:' + str(row), row['mean_regret_bits'], np.mean([r['point_regret_bits'] for r in chosen]))
            audit.close('selection_summary_bits:' + str(row), row['mean_selected_bits'], np.mean([r['selected_bits'] for r in chosen]))
            if 'criterion_met' in row:
                audit.require(row['criterion_met'] == bool(np.mean([r['point_regret_bits'] for r in chosen]) <= config['primary_selection_mean_regret_max_bits']), 'selection_frozen_gate:' + str(row))
    for row in summary['sensor_atlas']:
        chosen = [r for r in tables['sensor_atlas'] if r['bank'] == row['bank'] and r['noise'] == row['noise'] and r['channels'] == 19 and r['source_peak_am'] == config['signal_peak_am']]
        audit.close('sensor_atlas_summary:' + str(row), row['mean_known_state_bits'], np.mean([r['known_state_bits'] for r in chosen]))
    for row in summary['learning']:
        chosen = [r for r in tables['decoder_conditions'] if r['features'] == row['features'] and r['decoder'] == row['decoder']]
        audit.close('learning_summary_error:' + str(row), row['mean_error'], np.mean([r['error'] for r in chosen]))
        audit.close('learning_summary_excess:' + str(row), row['mean_paired_error_excess'], np.mean([r['paired_error_excess'] for r in chosen]))
    for row in summary['precision']:
        coarse = next(r for r in tables['representation_conditions'] if r['case_key'] == row['case_key'] and r['candidate'] == row['candidate'])
        precise = next(r for r in tables['precision_conditions'] if r['case_key'] == row['case_key'] and r['candidate'] == row['candidate'])
        audit.close('precision_summary:' + row['case_key'] + row['candidate'], row['absolute_difference_bits'], abs(coarse['representation_bits'] - precise['representation_bits']))
    print('Likelihood replay, sensor atlas, decoder predictions and summary checks complete', flush=True)
    output = {'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.perf_counter()-started,
        'passed': not audit.failures, 'failures': audit.failures, 'numerical_inequality_flags': audit.flags,
        'hash_checks': audit.hashes, 'maximum_absolute_residuals': audit.maximum_residual, 'recomputed_counts': dict(audit.counts),
        'frozen_expected_counts': expected_counts, 'likelihood_replay': replay,
        'inequality_uncertainty': {'confidence': .95, 'bonferroni_multiplicity': global_multiplicity,
            'methods': 'Hoeffding and empirical Bernstein assessed separately; no unbudgeted intersection',
            'flags_scope': 'Simultaneous conditional integration diagnostics; flags retained separately from implementation/hash failures'},
        'scope': 'Independent scientific/statistical/artifact audit conditional on declared finite state banks; no real EEG or anatomy-population guarantee',
        'limitations': ['Numerical rank residuals and floating-point checks are diagnostics, not interval-arithmetic proofs.',
            'Likelihood replay covers first64 epochs in8 prespecified physical cases; every stored draw endpoint/statistic is recomputed but not every raw likelihood replayed.',
            'Exact binary Gaussian F uses an independently coded256-node quadrature; convergence is separately covered by mathematical/unit checks.',
            'Prediction and regret criteria are descriptive frozen-grid benchmarks; neither low regret nor two anatomies proves population transfer.',
            'Learner Bayes reference knows true finite law while fitted models estimate development parameters; matched representation/zero-calibration side information does not remove this law advantage.',
            'Decoder head intervals remain approximate diagnostics over synthetic states; fixed-stratum cross-entropy uses valid Hoeffding, not iid empirical Bernstein.'],
        'audit_source_sha256': sha(__file__), 'run_manifest_sha256': sha(OUT / 'run_manifest.json'),
        'config_sha256': sha(ROOT / 'config/ceiling_benchmark.json'), 'prior_manifest_baselines': PRIOR_SHA,
        'packages': {'numpy': np.__version__}}
    EVIDENCE.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'passed': output['passed'], 'failures': len(audit.failures), 'numerical_flags': len(audit.flags),
                      'seconds': output['elapsed_seconds'], 'evidence': str(EVIDENCE)}, indent=2), flush=True)
    if audit.failures: sys.exit(1)


if __name__ == '__main__':
    main()
