from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from math import cos, pi, sin, sqrt
from typing import Iterable

import numpy as np

HBAR2_OVER_2M_MEV_A2 = 2.072142


@dataclass(frozen=True)
class Lattice:
    a: float
    b: float
    c: float
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0
    orient1: tuple[float, float, float] = (1.0, 0.0, 0.0)
    orient2: tuple[float, float, float] = (0.0, 1.0, 0.0)


@dataclass(frozen=True)
class SpurionCandidate:
    kind: str
    label: str
    distance_q: float
    energy_mismatch: float | None
    harmonic: tuple[int, int] | None
    bragg_hkl: tuple[int, int, int] | None
    qxy: tuple[float, float]
    q_magnitude: float
    effective_energy: float | None
    source: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SpurionCheck:
    q_hkl: tuple[float, float, float]
    energy_transfer: float
    fixed_energy: float
    fixed_energy_mode: str
    closed: bool
    ki: tuple[float, float, float]
    kf: tuple[float, float, float]
    qxy: tuple[float, float, float]
    candidates: tuple[SpurionCandidate, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        data = asdict(self)
        data["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        return data


def k_from_energy(energy_mev: float) -> float:
    if energy_mev < 0:
        return float("nan")
    return sqrt(energy_mev / HBAR2_OVER_2M_MEV_A2)


def energy_from_k(k: float | np.ndarray) -> float | np.ndarray:
    return HBAR2_OVER_2M_MEV_A2 * np.asarray(k) ** 2


def reciprocal_vectors(lattice: Lattice) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha = np.deg2rad(lattice.alpha)
    beta = np.deg2rad(lattice.beta)
    gamma = np.deg2rad(lattice.gamma)
    a1 = lattice.a * np.array([1.0, 0.0, 0.0])
    a2 = lattice.b * np.array([cos(gamma), sin(gamma), 0.0])
    cx = cos(beta)
    cy = (cos(alpha) - cos(beta) * cos(gamma)) / sin(gamma)
    cz = sqrt(max(0.0, 1.0 - cx * cx - cy * cy))
    a3 = lattice.c * np.array([cx, cy, cz])
    volume = float(np.dot(a1, np.cross(a2, a3)))
    b1 = 2 * pi * np.cross(a2, a3) / volume
    b2 = 2 * pi * np.cross(a3, a1) / volume
    b3 = 2 * pi * np.cross(a1, a2) / volume
    return b1, b2, b3


def hkl_to_qxy(hkl: Iterable[float], lattice: Lattice) -> np.ndarray:
    h, k, l = [float(value) for value in hkl]
    b1, b2, b3 = reciprocal_vectors(lattice)
    q = h * b1 + k * b2 + l * b3
    o1 = lattice.orient1[0] * b1 + lattice.orient1[1] * b2 + lattice.orient1[2] * b3
    o2 = lattice.orient2[0] * b1 + lattice.orient2[1] * b2 + lattice.orient2[2] * b3
    xhat = _unit(o1)
    zhat = _unit(np.cross(_unit(o1), _unit(o2)))
    yhat = _unit(np.cross(zhat, xhat))
    return np.array([float(np.dot(q, xhat)), float(np.dot(q, yhat)), float(np.dot(q, zhat))])


def get_kikf(ei: float, ef: float, qxy: Iterable[float]) -> tuple[np.ndarray, np.ndarray, bool]:
    qxy_vec = np.asarray(list(qxy), dtype=float)
    if qxy_vec.size == 2:
        qxy_vec = np.array([qxy_vec[0], qxy_vec[1], 0.0])
    q = float(np.linalg.norm(qxy_vec))
    ki_mag = k_from_energy(ei)
    kf_mag = k_from_energy(ef)
    if not np.isfinite(ki_mag) or not np.isfinite(kf_mag) or q == 0:
        return _nan_vec(), _nan_vec(), False
    check = (ki_mag**2 + kf_mag**2 - q**2) / (2 * ki_mag * kf_mag)
    if abs(check) > 1 or not np.isfinite(check):
        return _nan_vec(), _nan_vec(), False
    theta_q = float(np.arctan2(qxy_vec[1], qxy_vec[0]))
    arg = np.clip((ki_mag**2 + q**2 - kf_mag**2) / (2 * ki_mag * q), -1.0, 1.0)
    theta_i = theta_q - float(np.arccos(arg))
    ki = ki_mag * np.array([cos(theta_i), sin(theta_i), 0.0])
    kf = ki - qxy_vec
    return ki, kf, True


def closed_scattering_triangle(ei: float, ef: float, qxy: Iterable[float]) -> bool:
    _, _, closed = get_kikf(ei, ef, qxy)
    return closed


def check_spurions(
    q_hkl: Iterable[float],
    energy_transfer: float,
    fixed_energy: float,
    lattice: Lattice,
    *,
    fixed_energy_mode: str = "fixed_final_energy",
    q_tolerance: float = 0.08,
    energy_tolerance: float = 0.5,
    hkl_max: int = 4,
    harmonics: int = 4,
    include_powder_rings: bool = True,
    include_incoherent: bool = True,
) -> SpurionCheck:
    q_tuple = tuple(float(value) for value in q_hkl)
    qxy = hkl_to_qxy(q_tuple, lattice)
    if "incident" in fixed_energy_mode.lower():
        ei = fixed_energy
        ef = fixed_energy - energy_transfer
    else:
        ef = fixed_energy
        ei = fixed_energy + energy_transfer
    ki, kf, closed = get_kikf(ei, ef, qxy)
    warnings: list[str] = []
    if abs(float(qxy[2])) > q_tolerance:
        warnings.append("Q has a nonzero component perpendicular to the scattering plane.")
    if not closed:
        warnings.append("Requested Ei/Ef/Q cannot close the scattering triangle.")
        return SpurionCheck(q_tuple, energy_transfer, fixed_energy, fixed_energy_mode, False, tuple(ki), tuple(kf), tuple(qxy), (), tuple(warnings))

    candidates: list[SpurionCandidate] = []
    bragg_points = list(_integer_hkl_points(lattice, hkl_max))
    for n, m in product(range(1, harmonics + 1), repeat=2):
        q_harm = n * ki - m * kf
        e_eff = float((n * n) * ei - (m * m) * ef)
        for hkl, bragg_qxy in bragg_points:
            if abs(float(bragg_qxy[2])) > q_tolerance:
                continue
            dist = float(np.linalg.norm(q_harm[:2] - bragg_qxy[:2]))
            e_mismatch = abs(e_eff) if (n, m) != (1, 1) else abs(e_eff - energy_transfer)
            if dist <= q_tolerance and e_mismatch <= max(abs(e_eff), energy_tolerance):
                kind = "reciprocal_lattice_primary" if (n, m) == (1, 1) else "reciprocal_lattice_harmonic"
                candidates.append(
                    SpurionCandidate(
                        kind=kind,
                        label=f"({n}ki,{m}kf) near RL {hkl}",
                        distance_q=dist,
                        energy_mismatch=e_mismatch,
                        harmonic=(n, m),
                        bragg_hkl=hkl,
                        qxy=(float(bragg_qxy[0]), float(bragg_qxy[1])),
                        q_magnitude=float(np.linalg.norm(bragg_qxy[:2])),
                        effective_energy=e_eff,
                        source="DAVE Spurion2 harmonic Bragg logic",
                    )
                )

    if include_incoherent and abs(energy_transfer) > 0:
        qana, qmono = incoherent_q(ki, kf)
        for label, qinc in [("Qana incoherent", qana), ("Qmono incoherent", qmono)]:
            for hkl, bragg_qxy in bragg_points:
                if abs(float(bragg_qxy[2])) > q_tolerance:
                    continue
                dist = float(np.linalg.norm(qinc[:2] - bragg_qxy[:2]))
                if dist <= q_tolerance:
                    candidates.append(
                        SpurionCandidate(
                            kind="reciprocal_lattice_incoherent",
                            label=f"{label} near RL {hkl}",
                            distance_q=dist,
                            energy_mismatch=None,
                            harmonic=None,
                            bragg_hkl=hkl,
                            qxy=(float(bragg_qxy[0]), float(bragg_qxy[1])),
                            q_magnitude=float(np.linalg.norm(bragg_qxy[:2])),
                            effective_energy=0.0,
                            source="DAVE Spurion2 calc_incoherent logic",
                        )
                    )

    if include_powder_rings:
        theta = float(np.arctan2(qxy[1], qxy[0]))
        for n, m in product(range(1, harmonics + 1), repeat=2):
            q_harm = n * ki - m * kf
            e_eff = float((n * n) * ei - (m * m) * ef)
            for material, hkl, ring_q in powder_ring_q_values():
                ring_xy = ring_q * np.array([cos(theta), sin(theta)])
                dist = float(np.linalg.norm(q_harm[:2] - ring_xy))
                if dist <= q_tolerance:
                    candidates.append(
                        SpurionCandidate(
                            kind=f"{material.lower()}_powder_ring_primary" if (n, m) == (1, 1) else f"{material.lower()}_powder_ring_harmonic",
                            label=f"({n}ki,{m}kf) near {material} powder ring {hkl}",
                            distance_q=dist,
                            energy_mismatch=None,
                            harmonic=(n, m),
                            bragg_hkl=hkl,
                            qxy=(float(ring_xy[0]), float(ring_xy[1])),
                            q_magnitude=ring_q,
                            effective_energy=e_eff,
                            source="DAVE Spurion2 alewald powder-ring table",
                        )
                    )

    candidates.sort(key=lambda item: (item.distance_q, item.kind, item.label))
    return SpurionCheck(q_tuple, energy_transfer, fixed_energy, fixed_energy_mode, True, tuple(ki), tuple(kf), tuple(qxy), tuple(candidates), tuple(warnings))


def incoherent_q(ki: np.ndarray, kf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ki_mag = float(np.linalg.norm(ki[:2]))
    kf_mag = float(np.linalg.norm(kf[:2]))
    if ki_mag == 0 or kf_mag == 0:
        return _nan_vec(), _nan_vec()
    ki_incoherent = kf_mag * ki[:2] / ki_mag
    kf_incoherent = ki_mag * kf[:2] / kf_mag
    qana = np.array([ki[0] - kf_incoherent[0], ki[1] - kf_incoherent[1], 0.0])
    qmono = np.array([ki_incoherent[0] - kf[0], ki_incoherent[1] - kf[1], 0.0])
    return qana, qmono


def powder_ring_q_values() -> list[tuple[str, tuple[int, int, int], float]]:
    values: list[tuple[str, tuple[int, int, int], float]] = []
    for material, a, peaks in [
        ("Al", 4.04096, [(1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1), (2, 2, 2), (4, 0, 0), (3, 3, 1), (4, 2, 0), (4, 2, 2), (3, 3, 3), (4, 4, 0), (5, 3, 1), (6, 0, 0)]),
        ("Cu", 3.6149, [(1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1), (2, 2, 2), (4, 0, 0), (3, 3, 1), (4, 2, 0), (4, 2, 2), (3, 3, 3), (4, 4, 0), (5, 3, 1), (6, 0, 0)]),
        ("Mo", 3.147, [(1, 1, 0), (2, 0, 0), (2, 1, 1), (2, 2, 0), (3, 1, 0), (2, 2, 2), (3, 2, 1), (4, 0, 0), (3, 3, 0), (4, 2, 0), (3, 3, 2), (4, 2, 2), (4, 3, 1)]),
    ]:
        for hkl in peaks:
            values.append((material, hkl, 2 * pi / a * sqrt(sum(index * index for index in hkl))))
    return values


def _integer_hkl_points(lattice: Lattice, hkl_max: int) -> Iterable[tuple[tuple[int, int, int], np.ndarray]]:
    for h, k, l in product(range(-hkl_max, hkl_max + 1), repeat=3):
        if h == k == l == 0:
            continue
        yield (h, k, l), hkl_to_qxy((h, k, l), lattice)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("Cannot normalize zero-length vector")
    return vector / norm


def _nan_vec() -> np.ndarray:
    return np.array([float("nan"), float("nan"), float("nan")])
