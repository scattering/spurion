from tas_copilot_contracts import ToolCall, ToolResultStatus

from tas_spurion.copilot_plugin import get_tool_specs, run_tool


def _arguments():
    return {
        "q_point": {"h": 1.0, "k": 0.0, "l": 0.0},
        "energy_transfer": {"value": {"value": 0.0, "unit": "meV"}},
        "instrument_config": {
            "fixed_energy_mode": "fixed_final_energy",
            "fixed_energy": {"value": 14.7, "unit": "meV"},
        },
        "lattice": {
            "a": 6.283185307179586,
            "b": 6.283185307179586,
            "c": 6.283185307179586,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
            "orient1": [[1, 0, 0]],
            "orient2": [[0, 1, 0]],
        },
        "q_tolerance": 1e-8,
        "hkl_max": 1,
        "harmonics": 1,
        "include_powder_rings": False,
        "include_incoherent": False,
    }


def test_get_tool_specs_exposes_python_spurion_tool():
    specs = get_tool_specs()

    assert [(spec.namespace, spec.name) for spec in specs] == [("tas.spurion", "check_spurions")]


def test_check_spurions_tool_returns_typed_result():
    result = run_tool(
        ToolCall(
            tool_namespace="tas.spurion",
            tool_name="check_spurions",
            arguments=_arguments(),
            caller_workflow="test",
        )
    )

    assert result.status == ToolResultStatus.success
    assert result.output["spurion_check"]["summary"].startswith("1 candidate")
    assert result.output["details"]["candidates"][0]["kind"] == "reciprocal_lattice_primary"
