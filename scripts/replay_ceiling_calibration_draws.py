#!/usr/bin/env python3
"""Same-seed calibration replay with saved integrands; no outcome changes.

This closes a serialization gap in the completed supplementary calculation.
It does not call calibration_information or its summary functions. It retains
the frozen Gaussian channel implementation, independently tested elsewhere.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '1')
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np

spec = importlib.util.spec_from_file_location('runner', ROOT / 'scripts/run_ceiling_benchmark.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
OUT = ROOT / 'results/ceiling_benchmark_v1_calibration_replay'


def entropy(p):
    return -(p * np.log2(np.maximum(p, 1e-300)) + (1-p) * np.log2(np.maximum(1-p, 1e-300)))


def posterior(channel, samples, weights=None):
    return np.concatenate([channel.posterior(samples[a:a+512],
        None if weights is None else weights[a:a+512]) for a in range(0, len(samples), 512)])


def main():
    if (OUT / 'manifest.json').exists():
        raise RuntimeError('Completed replay cannot be overwritten')
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    config = json.loads(runner.CONFIG.read_text())
    source_rows_path = ROOT / 'results/ceiling_benchmark_v1_supplements/calibration_information.json'
    rows = json.loads(source_rows_path.read_text())
    banks = runner.load_banks(config)
    matches = []
    regime = config['noise_regimes']['joint']
    cache = {}
    for row in rows:
        family = (row['bank'], row['task'], row['domain'])
        if family not in cache:
            bank = banks[row['bank']]
            experiment = runner.experiment(bank, runner.source_means(bank, row['task'], row['domain'], config), regime, 19, config)
            cache[family] = experiment.channel
        channel = cache[family]
        n = row['full']['n_samples']
        m = row['calibration_epochs']
        rng = np.random.default_rng(row['seed'])
        heads = rng.choice(channel.n_heads, n, p=channel.probabilities)
        calibration_labels = rng.integers(0, 2, (n, m)) * 2 - 1
        calibration = channel.draw(calibration_labels.ravel(), np.repeat(heads, m), rng).reshape(n, m, channel.dimension)
        weights = channel.posterior_heads(calibration, calibration_labels)
        labels = rng.integers(0, 2, n) * 2 - 1
        samples = channel.draw(labels, heads, rng)
        p = posterior(channel, samples, weights)
        p0 = posterior(channel, samples)
        ph = np.concatenate([channel.known_head_posterior(samples[a:a+512], heads[a:a+512]) for a in range(0, n, 512)])
        arrays = dict(heads=heads, labels=labels, calibration_labels=calibration_labels,
            full_information=1-entropy(p), full_risk=np.minimum(p, 1-p),
            known_information=1-entropy(ph), known_risk=np.minimum(ph, 1-ph),
            zero_information=1-entropy(p0), zero_risk=np.minimum(p0, 1-p0),
            state_posterior_entropy=-np.sum(weights * np.log2(np.maximum(weights, 1e-300)), axis=1))
        checks = {
            'full_bits': abs(arrays['full_information'].mean() - row['full']['bits']),
            'full_risk': abs(arrays['full_risk'].mean() - row['full']['bayes_error']),
            'known_bits': abs(arrays['known_information'].mean() - row['known_head_paired']['bits']),
            'known_risk': abs(arrays['known_risk'].mean() - row['known_head_paired']['bayes_error']),
            'paired_gain': abs((arrays['full_information']-arrays['zero_information']).mean() - row['paired_calibration_information_gain']['estimate']),
            'paired_gain_se': abs((arrays['full_information']-arrays['zero_information']).std(ddof=1)/np.sqrt(n) - row['paired_calibration_information_gain']['standard_error']),
            'posterior_entropy': abs(arrays['state_posterior_entropy'].mean()-row['head_posterior_entropy_bits']),
        }
        if max(checks.values()) > 5e-13:
            raise RuntimeError(f'Replay mismatch {family}/{m}: {checks}')
        filename = '__'.join(family) + f'__cal{m}.npz'
        np.savez_compressed(OUT / filename, **arrays)
        matches.append(dict(bank=row['bank'], task=row['task'], domain=row['domain'], calibration_epochs=m,
            seed=row['seed'], path=str((OUT/filename).relative_to(ROOT)), residuals=checks))
        print(f'Replayed {filename}', flush=True)
    runner.dump(OUT / 'matches.json', matches)
    dependencies = [Path(__file__), runner.CONFIG, source_rows_path, ROOT/'scripts/run_ceiling_benchmark.py',
        ROOT/'src/tdo_sim/insight.py', ROOT/'src/tdo_sim/ceiling_benchmark.py']
    runner.dump(OUT / 'manifest.json', dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-start, conditions=len(matches),
        maximum_summary_residual=max(max(r['residuals'].values()) for r in matches),
        input_source_sha256={str(p.relative_to(ROOT)):runner.sha(p) for p in dependencies},
        output_sha256={str(p.relative_to(ROOT)):runner.sha(p) for p in OUT.glob('*') if p.is_file()},
        scope='Same-seed integrand replay after outcomes; supplement unchanged, no new scientific conditions or selection'))


if __name__ == '__main__':
    main()
