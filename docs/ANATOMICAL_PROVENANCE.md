# Anatomical and forward-model provenance

Only public anatomical surfaces, cortical source spaces and coordinate transforms were used. No EEG or MEG recordings were used as study data. The prepared operator banks contain source/sensor coordinates and lead fields for the declared simulations.

- Template anatomy: MNE/FreeSurfer `fsaverage`, three conducting compartments, 32 selected cortical-normal sources.
- Individual anatomy: public MNE sample subject, three conducting compartments, 32 selected cortical-normal sources.
- Primary conductor surfaces: ico3, 1,280 triangles per layer; brain/skull/scalp conductivities 0.3/0.006/0.3 S/m.
- Contact names: BioSemi64; initial standard `fsaverage_1005` locations were projected to the respective scalp. Contact locations are modelled, not measured.
- Spherical banks: independent cap/conductor perturbations on a four-layer sphere with 32 radial source positions and 256 native contacts.

MNE's anatomical resource documentation: <https://mne.tools/stable/auto_tutorials/forward/35_eeg_no_mri.html>. Sample resources: <https://mne.tools/stable/generated/mne.datasets.sample.data_path.html>. FreeSurfer attribution: <https://surfer.nmr.mgh.harvard.edu/>.

`resources/ceiling_banks_v1/manifest.json` records the exact original anatomical input SHA-256 values and derived operator hashes. Each bank has metadata and acquisition-state records. `scripts/prepare_ceiling_assets.py` preserves the exact source-selection and geometry operations and requires the original staged MNE archives and Git-tree checks. Those large upstream anatomical files are not redistributed by this code release.

The nominal-cap conductor sensitivity used available finer surfaces. For the individual geometry, the supplied coarse surfaces were not an exact downsampling of the native surfaces: nearest-vertex discrepancies reached 5.74 mm. The sensitivity changes geometry as well as conductor resolution and retains the same cortical basis. It is not a source-discretization convergence test.
