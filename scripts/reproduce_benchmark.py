#!/usr/bin/env python3
"""Run the unchanged primary benchmark in a separate, fresh directory."""
from pathlib import Path
import argparse
import json
import math
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def compare(a, b, location='summary', differences=None):
    if differences is None:
        differences = []
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise RuntimeError('Different keys: ' + location)
        for key in a:
            compare(a[key], b[key], location + '.' + key, differences)
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            raise RuntimeError('Different list: ' + location)
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f'{location}[{i}]', differences)
    elif isinstance(a, (float, int)) and not isinstance(a, bool):
        if not isinstance(b, (float, int)) or not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-10):
            raise RuntimeError(f'Different numeric value: {location}: {a} versus {b}')
        differences.append(abs(a - b))
    elif a != b:
        raise RuntimeError('Different value: ' + location)
    return differences

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', type=Path, default=ROOT / '_reproduction')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    target = args.output_directory.resolve()
    if not args.verify_only:
        if target == ROOT or (target.exists() and any(target.iterdir())):
            raise RuntimeError('Reproduction requires a fresh directory, separate from the repository.')
        target.mkdir(parents=True, exist_ok=True)
        for folder in ['src', 'tests', 'config', 'scripts', 'resources/ceiling_banks_v1']:
            shutil.copytree(ROOT / folder, target / folder, ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(ROOT / 'CEILING_BENCHMARK_PROTOCOL.md', target / 'CEILING_BENCHMARK_PROTOCOL.md')
        for study in ['pilot_v1', 'full_study_v1', 'insight_study_v1']:
            source = ROOT / 'results' / study / 'run_manifest.json'
            destination = target / 'results' / study / 'run_manifest.json'
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        (target / 'evidence').mkdir(exist_ok=True)
        print('Running the original primary benchmark in ' + str(target), flush=True)
        subprocess.run([sys.executable, str(target / 'scripts/run_ceiling_benchmark.py')], cwd=target, check=True)
    print('Running the independent primary audit', flush=True)
    subprocess.run([sys.executable, str(target / 'scripts/verify_reproduced_benchmark.py')], cwd=target, check=True)
    reference = json.loads((ROOT / 'results/ceiling_benchmark_v1/summary.json').read_text())
    replica = json.loads((target / 'results/ceiling_benchmark_v1/summary.json').read_text())
    differences = compare(reference, replica)
    record = {'status': 'passed', 'scope': 'Primary benchmark; original execution and reproduction kept separate',
        'numeric_summary_values_compared': len(differences),
        'maximum_absolute_difference': max(differences, default=0.0)}
    (target / 'evidence/reproduction_comparison.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2), flush=True)

if __name__ == '__main__':
    main()
