"""Fixed synthetic sensor/source geometry, in meters and head coordinates."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Geometry:
    sensor_positions_m: np.ndarray
    source_positions_m: np.ndarray
    source_orientations: np.ndarray
    montages: dict[int, np.ndarray]
    sensor_names: tuple[str, ...]

    def metadata(self) -> dict:
        return {
            "kind": "synthetic fixed spherical-cap geometry; no anatomical atlas",
            "coordinate_frame": "head; x right, y anterior, z superior",
            "position_units": "m", "sensor_count": len(self.sensor_positions_m),
            "source_count": len(self.source_positions_m),
            "source_meaning": "fixed point dipoles with fixed global orientations across heads",
            "sensor_cap_min_z_fraction": -0.25,
            "montage_indices": {str(k): v.tolist() for k, v in self.montages.items()},
        }


def _cap_points(n: int, z_min: float) -> np.ndarray:
    # Equal-area Fibonacci cap; no imported montage and no face-hole adaptation.
    z = 1 - (1 - z_min) * (np.arange(n) + 0.5) / n
    angle = np.arange(n) * np.pi * (3 - np.sqrt(5))
    radius = np.sqrt(1 - z * z)
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))


def _farthest_order(points: np.ndarray) -> np.ndarray:
    n = len(points)
    order = np.empty(n, dtype=int)
    order[0] = int(np.argmax(points[:, 2]))
    distances = np.sum((points - points[order[0]]) ** 2, axis=1)
    for j in range(1, n):
        distances[order[:j]] = -1
        order[j] = int(np.argmax(distances))
        distances = np.minimum(distances, np.sum((points - points[order[j]]) ** 2, axis=1))
    return order


def make_nested_geometry(n_sources: int = 32, seed: int = 20261008,
                         scalp_radius_m: float = 0.09) -> Geometry:
    """Nested 19⊂64⊂256 electrodes, selected before any task/noise observations.

    Sources occupy a 52–64 mm internal shell. Their radial orientations are
    defined once in the global head frame and stay fixed as conductor parameters
    change. With 32 sources, the 256-channel map need not have a source nullspace.
    """
    if n_sources < 2 or scalp_radius_m <= 0:
        raise ValueError("Require at least two sources and positive scalp radius")
    sensors = scalp_radius_m * _cap_points(256, -0.25)
    order = _farthest_order(sensors)
    directions = _cap_points(n_sources, -0.6)
    rng = np.random.default_rng(seed)
    rotation = rng.uniform(0, 2 * np.pi)
    rot = np.array([[np.cos(rotation), -np.sin(rotation), 0],
                    [np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]])
    directions = directions @ rot.T
    depths = np.linspace(0.052, 0.064, n_sources)[rng.permutation(n_sources)]
    return Geometry(sensors, directions * depths[:, None], directions.copy(),
                    {n: order[:n].copy() for n in (19, 64, 256)},
                    tuple(f"SYN{i + 1:03d}" for i in range(256)))
