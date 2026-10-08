#!/usr/bin/env python3
"""Verify published files, original scientific source hashes and operator banks."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    publication = json.loads((ROOT / 'reference/manifest.json').read_text())
    assert sha(ROOT / 'reference/frozen_results.tar.gz') == publication['archive_sha256']
    for name, digest in publication['files'].items():
        assert sha(ROOT / name) == digest, name
    manifest = json.loads((ROOT / 'results/ceiling_benchmark_v1/run_manifest.json').read_text())
    checked = 0
    for name, digest in manifest['source_sha256_before'].items():
        assert sha(ROOT / name) == digest, name
        checked += 1
    physical = json.loads((ROOT / 'resources/ceiling_banks_v1/manifest.json').read_text())
    assert sha(ROOT / 'config/ceiling_benchmark.json') == physical['config_sha256']
    for name, digest in physical['output_sha256'].items():
        assert sha(ROOT / name) == digest, name
    assert physical['operators'] == 144
    missing = [name for name in manifest['output_sha256'] if not (ROOT / name).is_file()]
    print(json.dumps({'status': 'passed', 'published_reference_files': len(publication['files']),
        'original_scientific_sources_verified': checked, 'prepared_operators': 144,
        'large_primary_files_not_distributed': len(missing),
        'scope': 'Published subset and scientific source integrity; the original dense replay audit requires a full reproduction.'}, indent=2))

if __name__ == '__main__':
    main()
