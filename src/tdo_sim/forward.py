"""Genuine synthetic spherical EEG physics and separately named abstract fixtures.

MNE documentation checked against installed 1.13.2:
https://mne.tools/stable/generated/mne.make_sphere_model.html
https://mne.tools/stable/generated/mne.make_forward_dipole.html
No datasets, subjects, atlases, or downloads are used.
"""

from dataclasses import dataclass, asdict
import os
import tempfile
from pathlib import Path
import numpy as np
from .geometry import Geometry


@dataclass(frozen=True)
class HeadParameters:
    head_radius_m: float = 0.09
    center_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    relative_radii: tuple[float, float, float, float] = (0.90, 0.92, 0.97, 1.0)
    conductivities_s_m: tuple[float, float, float, float] = (0.33, 1.0, 0.004, 0.33)


@dataclass(frozen=True)
class ForwardModel:
    gain_v_per_am: np.ndarray
    metadata: dict


def make_spherical_forward(geometry: Geometry, parameters: HeadParameters | None = None,
                           cache_path: str | Path | None = None) -> ForwardModel:
    """MNE multilayer spherical EEG forward with explicit fixed dipole orientations.

    Column j is volts for a 1 A·m dipole in orientation j. Unit dipole amplitudes
    affect the returned source estimate only, not the forward gain. The sphere
    is a simplified physical conductor, not an individual anatomical head.
    """
    parameters = parameters or HeadParameters()
    if parameters.head_radius_m <= 0 or any(s <= 0 for s in parameters.conductivities_s_m):
        raise ValueError("Positive head radius and conductivities required")
    radii = np.asarray(parameters.relative_radii)
    if len(radii) != 4 or np.any(np.diff(radii) <= 0) or not np.isclose(radii[-1], 1):
        raise ValueError("Require four ordered normalized conductor radii ending at one")
    source_radius = np.linalg.norm(geometry.source_positions_m - parameters.center_m, axis=1)
    if np.any(source_radius >= parameters.head_radius_m * radii[0] - 0.005):
        raise ValueError("Source dipoles must lie safely inside the innermost conductor")
    # MNE 1.13 reads its config in r+ mode. Keep its optional config/cache wholly
    # in an isolated writable temporary home instead of touching user settings.
    if "_MNE_FAKE_HOME_DIR" not in os.environ:
        os.environ["_MNE_FAKE_HOME_DIR"] = tempfile.mkdtemp(prefix="tdo-mne-")
    try:
        import mne
    except ImportError as exc:
        raise RuntimeError("Physical forward generation requires optional dependency mne") from exc
    names = list(geometry.sensor_names)
    info = mne.create_info(names, sfreq=128.0, ch_types="eeg")
    montage = mne.channels.make_dig_montage(
        ch_pos=dict(zip(names, geometry.sensor_positions_m)), coord_frame="head")
    info.set_montage(montage, on_missing="raise", verbose=False)
    sphere = mne.make_sphere_model(r0=parameters.center_m, head_radius=parameters.head_radius_m,
                                   relative_radii=parameters.relative_radii,
                                   sigmas=parameters.conductivities_s_m, verbose=False)
    n = len(geometry.source_positions_m)
    dipole = mne.Dipole(times=np.zeros(n), pos=geometry.source_positions_m,
                        amplitude=np.ones(n), ori=geometry.source_orientations,
                        gof=np.ones(n) * 100, verbose=False)
    forward, _ = mne.make_forward_dipole(dipole, sphere, info, n_jobs=1, verbose=False)
    gain = np.asarray(forward["sol"]["data"], dtype=float)
    if gain.shape != (len(names), n) or not np.all(np.isfinite(gain)):
        raise RuntimeError("MNE returned invalid fixed-orientation EEG gain")
    metadata = {
        "kind": "MNE four-layer spherical EEG forward; fully synthetic",
        "mne_version": mne.__version__, "gain_units": "V/(A m)",
        "position_units": "m", "dipole_amplitude_units": "A m",
        "source_orientation": "fixed global unit vectors, identical across heads",
        "source_geometry": "fixed synthetic points; no anatomical correspondence claimed",
        "parameters": asdict(parameters), "sensor_count": len(names), "source_count": n,
        "sensor_positions_m": geometry.sensor_positions_m.tolist(),
        "source_positions_m": geometry.source_positions_m.tolist(),
        "source_orientations": geometry.source_orientations.tolist(),
        "maximum_sensor_scalp_distance_m": float(np.max(np.abs(
            np.linalg.norm(geometry.sensor_positions_m - parameters.center_m, axis=1)
            - parameters.head_radius_m))),
        "sensors_on_conductor_surface": bool(np.allclose(
            np.linalg.norm(geometry.sensor_positions_m - parameters.center_m, axis=1),
            parameters.head_radius_m, atol=1e-7, rtol=0)),
        "no_recordings_or_downloads": True,
    }
    if cache_path is not None:
        import json
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, gain_v_per_am=gain, metadata_json=json.dumps(metadata),
                            sensor_positions_m=geometry.sensor_positions_m,
                            source_positions_m=geometry.source_positions_m,
                            source_orientations=geometry.source_orientations)
    return ForwardModel(gain, metadata)


def make_abstract_forward(geometry: Geometry, seed: int = 0, rank: int = 12) -> ForwardModel:
    """Gaussian low-rank numerical fixture, explicitly not physical EEG evidence."""
    rng = np.random.default_rng(seed)
    m, p = len(geometry.sensor_positions_m), len(geometry.source_positions_m)
    rank = min(rank, m, p)
    gain = rng.normal(size=(m, rank)) @ rng.normal(size=(rank, p)) / np.sqrt(rank)
    return ForwardModel(gain, {"kind": "abstract Gaussian low-rank verification fixture",
                                "gain_units": "arbitrary", "physical_eeg": False,
                                "seed": seed, "rank_cap": rank})
