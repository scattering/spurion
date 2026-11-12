from __future__ import annotations

from typing import Any

from .core import Lattice, check_spurions


def get_tool_specs():
    from tas_copilot_contracts import Availability, ImplementationType, PermissionLevel, SideEffectLevel, ToolSpec

    return [
        ToolSpec(
            name="check_spurions",
            namespace="tas.spurion",
            version="0.1.0",
            description="Check TAS Q/E requests for harmonic, powder-ring, and incoherent spurion candidates using a Python port of DAVE Spurion2 core logic.",
            input_schema={"type": "object", "required": ["q_point", "energy_transfer", "instrument_config", "lattice"]},
            output_schema={"type": "object"},
            side_effect_level=SideEffectLevel.none,
            availability=Availability.available,
            implementation_type=ImplementationType.python_entrypoint,
            required_dependencies=["tas-spurion", "numpy"],
            preconditions=["Q/E, fixed-energy mode, and lattice/orientation are supplied."],
            postconditions=["No files, instrument state, or live controls are modified."],
            provenance_policy="Record Q/E/configuration, DAVE Spurion2 source-file provenance, tolerances, and candidate list.",
            permissions_required=[PermissionLevel.read_only],
            enabled_by_default=True,
            owner_repo="git@github.com:scattering/tas-spurion.git",
            documentation_url="https://github.com/scattering/tas-spurion",
            maintainer="William Ratcliff",
        )
    ]


def run_tool(call):
    from tas_copilot_contracts import ProvenanceRecord, SpurionCheckResult, ToolError, ToolResult, ToolResultStatus

    if (call.tool_namespace, call.tool_name) != ("tas.spurion", "check_spurions"):
        return _result(call, ToolResult, ToolResultStatus.validation_failed, errors=[ToolError(code="unknown_tool", message="Unknown tas-spurion tool")])
    try:
        args = call.arguments
        check = check_spurions(
            q_hkl=_q(args),
            energy_transfer=_energy(args),
            fixed_energy=_fixed_energy(args),
            fixed_energy_mode=str((args.get("instrument_config") or {}).get("fixed_energy_mode") or "fixed_final_energy"),
            lattice=_lattice(args),
            q_tolerance=float(args.get("q_tolerance", 0.08)),
            energy_tolerance=float(args.get("energy_tolerance", 0.5)),
            hkl_max=int(args.get("hkl_max", 4)),
            harmonics=int(args.get("harmonics", 4)),
            include_powder_rings=bool(args.get("include_powder_rings", True)),
            include_incoherent=bool(args.get("include_incoherent", True)),
        )
        provenance = [
            ProvenanceRecord(
                source_type="code",
                source_id="DAVE Spurion2",
                source_uri="DAVE/programs/modules/TAS/Spurion2",
                method="Python port of calc_q.pro, calc_incoherent.pro, hkl_to_qxy.pro, and spurion2_2dlattice_data__define.pro",
            )
        ]
        summary = f"{len(check.candidates)} candidate spurion(s) found." if check.closed else "Scattering triangle does not close; spurion candidates were not evaluated."
        typed = SpurionCheckResult(summary=summary, warnings=list(check.warnings), provenance=provenance)
        return _result(
            call,
            ToolResult,
            ToolResultStatus.success,
            output={"spurion_check": typed.model_dump(mode="json"), "details": check.to_dict()},
            provenance=provenance,
        )
    except Exception as exc:
        return _result(call, ToolResult, ToolResultStatus.error, errors=[ToolError(code=exc.__class__.__name__, message=str(exc))])


def _q(args: dict[str, Any]) -> tuple[float, float, float]:
    q = args.get("q_point") or {}
    return float(q["h"]), float(q["k"]), float(q["l"])


def _energy(args: dict[str, Any]) -> float:
    energy = args.get("energy_transfer") or {}
    value = energy.get("value", energy)
    if isinstance(value, dict):
        return float(value["value"])
    return float(value)


def _fixed_energy(args: dict[str, Any]) -> float:
    instrument = args.get("instrument_config") or {}
    fixed = instrument.get("fixed_energy", 14.7)
    if isinstance(fixed, dict):
        return float(fixed["value"])
    return float(fixed)


def _lattice(args: dict[str, Any]) -> Lattice:
    lattice = args.get("lattice") or {}
    return Lattice(
        a=float(lattice["a"]),
        b=float(lattice["b"]),
        c=float(lattice["c"]),
        alpha=float(lattice.get("alpha", 90.0)),
        beta=float(lattice.get("beta", 90.0)),
        gamma=float(lattice.get("gamma", 90.0)),
        orient1=tuple(lattice.get("orient1", [[1, 0, 0]])[0]),
        orient2=tuple(lattice.get("orient2", [[0, 1, 0]])[0]),
    )


def _result(call, ToolResult, status, output=None, errors=None, provenance=None):
    return ToolResult(
        status=status,
        output=output,
        errors=errors or [],
        tool_name=call.tool_name,
        tool_namespace=call.tool_namespace,
        tool_version="0.1.0",
        provenance=provenance or [],
    )
