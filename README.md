# Task-conditioned information ceilings in scalp EEG

Simulation code, numerical results and reproducibility instructions for **Task-conditioned information ceilings in scalp EEG: implications for transferable representations**.

The study separates information in the sensor observation, information retained by a fixed encoding, uncertainty about the subject state, and the performance of a finite-data decoder. It evaluates six source-location and event-waveform distinctions using two spherical banks and two anatomical BEM geometries. Locally useful encoders can discard substantial task evidence when transferred to a different anatomical/source family. Compression can nevertheless improve a finite-data decoder, so prediction quality alone does not establish information retention.

The information values are conditional on the declared source, acquisition, noise and side-information laws. They are not total EEG information, channel capacity, or a universal ceiling for foundation models. No recorded EEG is required or included.

## Install and check

Python 3.12 was used for the audited execution. The exact third-party versions are recorded in `requirements-ceiling.lock`; a compatible scientific environment can instead be installed from `pyproject.toml`.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-ceiling.lock
.venv/bin/python scripts/install_reference.py
.venv/bin/python scripts/verify_release.py
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

The published reference archive contains frozen summaries, encoder matrices, fitted models and selected sensitivity arrays. It preserves the original execution manifests. Large primary integration, epoch and precision arrays are omitted from Git and can be regenerated. The lightweight release check verifies the distributed subset and scientific source hashes; it does not claim to repeat the original dense likelihood replay without those arrays.

## Reproduce the primary simulation

Run the unchanged benchmark in a separate fresh directory:

```sh
.venv/bin/python scripts/reproduce_benchmark.py
```

This creates `_reproduction/`, runs the primary simulation and independent audit, and compares its numerical summary with the published result. The original execution took approximately 6.5 minutes on the study computer; other runtimes vary. Allow several GB for regenerated outputs. The supplied prepared forward banks permit this run without downloading MRI data or rebuilding the BEM models. The reproduction preserves the original reference files. The checked release rerun matched all 472 primary summary values exactly; its independent audit and explicit upstream provenance scope are stored in `reference/reproduction_audit.json`.

## Contents

- `src/tdo_sim/`: forward modelling, reference coordinates, covariance propagation, Gaussian and finite-mixture information, calibration and decoders.
- `scripts/`: unchanged scientific runners and independent checks, plus publication-installation and reproduction helpers.
- `tests/`: mathematical and protocol tests, including dense likelihood, rank, covariance and leakage checks.
- `config/`: frozen settings and deterministic seeds.
- `resources/ceiling_banks_v1/`: 144 prepared forward operators, source/sensor coordinates and anatomical provenance hashes.
- `reference/`: compressed published numerical output subset and per-file SHA-256 manifest.
- `docs/`: reproduction scope and anatomical provenance.

Source discretization remains a stated limitation: the 32-source basis and coefficient-vector normalization have not been validated as a resolution-independent cortical current law. Two anatomical geometries establish a controlled transfer counterexample, not its prevalence across people.

Public anatomical-resource provenance is recorded separately. This repository does not assert a new license for those resources.
