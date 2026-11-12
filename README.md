# TAS Spurion

Python port of the non-GUI computational core from DAVE Spurion2 for triple-axis spurion checks.

This package is intentionally separate from `tascopilot`. It exposes a thin `tas_copilot.tools` plugin so the co-pilot can call spurion checks through typed contracts, permissions, provenance, and audit logging without depending on IDL at runtime.

Initial scope:

- TAS wave-vector and energy conversions
- Scattering-triangle closure
- HKL to scattering-plane projection
- Harmonic Bragg spurion candidates
- Al/Cu/Mo powder-ring candidates from the DAVE `alewald` tables
- Incoherent analyzer/monochromator spurion candidates

Tests include deterministic JSON regression fixtures in `tests/fixtures/`. These are derived from the ported DAVE formulas and tables; they are not yet IDL runtime captures. Future DAVE/IDL golden captures should be added alongside those fixtures with source version, input values, and captured output metadata.

The implementation ports formulas from:

```text
DAVE/programs/modules/TAS/Spurion2
```

The GUI, plotting, and IDL object-container layers are not ported.

## Reproducibility Tests

Install the shared contracts from the
[TASCopilot paper repository](https://github.com/scattering/tascopilot-paper)
and this package into a Python 3.11 or newer environment:

```sh
TASCOPILOT_PAPER=/absolute/path/to/tascopilot-paper
python -m pip install -e "$TASCOPILOT_PAPER/packages/tas_copilot_contracts"
python -m pip install -e . pytest
python -m pytest -q -rs
```

Tests use local fixtures. Optional upstream runtimes or explicitly supplied
archive fixtures are reported as skipped when unavailable. No instrument
connection is required. No full papers, private corpus, or development
handoff documents are included.

## License

Repository contributions are available under the [MIT License](LICENSE).
The NIST DAVE-derived public-domain material retains its public-domain
status. Source attribution is to the DAVE Spurion2 computational routines;
no IDL runtime or GUI implementation is distributed.
