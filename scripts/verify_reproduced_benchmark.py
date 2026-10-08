#!/usr/bin/env python3
"""Audit primary numerical outputs using prepared banks, with explicit upstream scope.

The original independent auditor remains unchanged. Its numerical checks and all
primary source/output/operator hash checks run in full. Optional checks of raw
anatomical inputs and unrelated prior-study artefacts check any available files,
while recording absent upstream files as not repeated, rather than as a claim of
successful reproduction of those external studies.
"""
from pathlib import Path
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    path = ROOT / 'scripts/verify_ceiling_benchmark_outputs.py'
    spec = importlib.util.spec_from_file_location('original_independent_auditor', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.Audit
    not_repeated = {}

    class PreparedBankAudit(original):
        def mapping(self, label, mapping, base=ROOT):
            optional = label == 'physical_inputs' or label.startswith(('prior_outputs:', 'prior_sources:', 'prior_inputs:'))
            if optional:
                missing = [name for name in mapping if not (base / name).is_file()]
                if missing:
                    not_repeated[label] = missing
                    mapping = {name: digest for name, digest in mapping.items() if name not in missing}
            # Available files remain strict hash checks, including all primary files.
            super().mapping(label, mapping, base)

    module.Audit = PreparedBankAudit
    code = 0
    try:
        module.main()
    except SystemExit as error:
        code = error.code
    evidence = ROOT / 'evidence/ceiling_benchmark_integrity_checks.json'
    result = json.loads(evidence.read_text())
    result['publication_reproduction_scope'] = {
        'primary_numerical_checks': 'All original checks retained',
        'primary_source_output_and_prepared_operator_hashes': 'All original checks retained',
        'upstream_checks_not_repeated': not_repeated,
        'reason': 'Raw anatomy and historical pilot outputs are not distributed or needed to reproduce the primary experiment from the verified prepared banks.'}
    evidence.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'primary_audit_passed': result['passed'],
        'upstream_files_not_repeated': sum(map(len, not_repeated.values())),
        'numerical_check_failures': result['failures']}, indent=2))
    if code:
        raise SystemExit(code)

if __name__ == '__main__':
    main()
