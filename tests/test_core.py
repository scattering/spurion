import math
import json
from pathlib import Path

import numpy as np

from tas_spurion import (
    Lattice,
    check_spurions,
    closed_scattering_triangle,
    energy_from_k,
    get_kikf,
    hkl_to_qxy,
    k_from_energy,
)
from tas_spurion.core import powder_ring_q_values


def test_energy_wavevector_round_trip():
    k = k_from_energy(14.7)

    assert math.isclose(float(energy_from_k(k)), 14.7, rel_tol=1e-12)


def test_hkl_to_qxy_cubic_orientation():
    qxy = hkl_to_qxy((1, 0, 0), Lattice(a=2 * math.pi, b=2 * math.pi, c=2 * math.pi))

    assert np.allclose(qxy, [1.0, 0.0, 0.0], atol=1e-12)


def test_get_kikf_closes_elastic_triangle():
    qxy = np.array([1.0, 0.0, 0.0])
    ki, kf, closed = get_kikf(14.7, 14.7, qxy)

    assert closed
    assert np.allclose(ki - kf, qxy)
    assert math.isclose(float(np.linalg.norm(ki)), k_from_energy(14.7), rel_tol=1e-12)
    assert math.isclose(float(np.linalg.norm(kf)), k_from_energy(14.7), rel_tol=1e-12)
    assert closed_scattering_triangle(14.7, 14.7, qxy)


def test_get_kikf_rejects_unclosed_triangle():
    _, _, closed = get_kikf(1.0, 1.0, [100.0, 0.0, 0.0])

    assert not closed


def test_check_spurions_returns_primary_rl_candidate_for_elastic_bragg_point():
    result = check_spurions(
        q_hkl=(1, 0, 0),
        energy_transfer=0.0,
        fixed_energy=14.7,
        lattice=Lattice(a=2 * math.pi, b=2 * math.pi, c=2 * math.pi),
        q_tolerance=1e-8,
        hkl_max=1,
        include_powder_rings=False,
        include_incoherent=False,
    )

    assert result.closed
    assert any(candidate.kind == "reciprocal_lattice_primary" and candidate.bragg_hkl == (1, 0, 0) for candidate in result.candidates)


def test_check_spurions_returns_powder_ring_candidate():
    al_111 = next(q for material, hkl, q in powder_ring_q_values() if material == "Al" and hkl == (1, 1, 1))
    result = check_spurions(
        q_hkl=(al_111, 0, 0),
        energy_transfer=0.0,
        fixed_energy=14.7,
        lattice=Lattice(a=2 * math.pi, b=2 * math.pi, c=2 * math.pi),
        q_tolerance=1e-8,
        hkl_max=1,
        harmonics=1,
        include_powder_rings=True,
        include_incoherent=False,
    )

    assert result.closed
    assert any(candidate.kind == "al_powder_ring_primary" and candidate.bragg_hkl == (1, 1, 1) for candidate in result.candidates)


def test_check_spurions_returns_incoherent_candidate():
    result = check_spurions(
        q_hkl=(1, 0, 0),
        energy_transfer=1.0,
        fixed_energy=14.7,
        lattice=Lattice(a=2 * math.pi, b=2 * math.pi, c=2 * math.pi),
        q_tolerance=0.1,
        hkl_max=1,
        harmonics=1,
        include_powder_rings=False,
        include_incoherent=True,
    )

    assert result.closed
    assert any(candidate.kind == "reciprocal_lattice_incoherent" for candidate in result.candidates)


def test_check_spurions_reports_unclosed_request():
    result = check_spurions(
        q_hkl=(100, 0, 0),
        energy_transfer=0.0,
        fixed_energy=1.0,
        lattice=Lattice(a=2 * math.pi, b=2 * math.pi, c=2 * math.pi),
    )

    assert not result.closed
    assert not result.candidates
    assert result.warnings == ("Requested Ei/Ef/Q cannot close the scattering triangle.",)


def test_json_regression_fixtures_load_and_match():
    fixture_path = Path(__file__).parent / "fixtures" / "spurion_regression_cases.json"
    cases = json.loads(fixture_path.read_text(encoding="utf-8"))

    assert cases
    for case in cases:
        args = case["arguments"]
        lattice = Lattice(**args["lattice"])
        result = check_spurions(
            q_hkl=args["q_hkl"],
            energy_transfer=args["energy_transfer"],
            fixed_energy=args["fixed_energy"],
            lattice=lattice,
            q_tolerance=args["q_tolerance"],
            hkl_max=args["hkl_max"],
            harmonics=args["harmonics"],
            include_powder_rings=args["include_powder_rings"],
            include_incoherent=args["include_incoherent"],
        )
        expected = case["expected"]
        assert result.closed is expected["closed"]
        if "candidate_count" in expected:
            assert len(result.candidates) == expected["candidate_count"]
        if "first_candidate_kind" in expected:
            assert result.candidates[0].kind == expected["first_candidate_kind"]
            assert result.candidates[0].bragg_hkl == tuple(expected["first_candidate_hkl"])
        if "candidate_kind" in expected:
            assert any(candidate.kind == expected["candidate_kind"] and candidate.bragg_hkl == tuple(expected["candidate_hkl"]) for candidate in result.candidates)
