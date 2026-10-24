# Changelog — `bpipe-lob-engine`

All notable engineering changes to `bpipe-lob-engine` (`bpipe_lob`) are documented in this file.

## [0.3.0] — 2026-10-24
### Added
- Documented CLI diagnostic usage (`python -m bpipe_lob --self-test --json`) and deterministic invariant verification workflows in `README.md`.
- Added release history and semantic versioning audit trail in `CHANGELOG.md`.

## [0.2.1] — 2026-10-23
### Added
- Added automated CLI and runtime invariant regression suite in `tests/test_diagnostics_cli.py` covering seed determinism, negative-seed rejection, JSON output schema, and `--self-test` exit codes.

## [0.2.0] — 2026-10-21
### Added
- Added `src/bpipe_lob/diagnostics.py` (`run_diagnostic_suite`, `validate_runtime_invariants`) and module CLI runner `src/bpipe_lob/__main__.py`.
- Exported diagnostic helpers in `bpipe_lob.__all__`.

## [0.1.0] — 2026-10-19
### Added
- Initial architecture and core test suite: High-frequency L2/L3 Limit Order Book (LOB) reconstruction engine, B-PIPE / FIX tick normalization, price-time priority matching, and VPIN/microprice analytics.
