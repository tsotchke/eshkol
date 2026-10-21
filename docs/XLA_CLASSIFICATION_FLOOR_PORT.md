# XLA classification and floor plan port status

This checkout is an isolated clone of canonical connection-port/source at
7f01cd9707d4368dc6731ad6d91ae8d1eec6a76a. It carries the original bounded
classification/parity-plan layer and the focused signed-floor plan regression.
The source-derived Python plan is an input to future parity work; generating it
does not execute a StableHLO lowering, device kernel, or GPU test.

## Reconciliation status

The canonical `tests/coverage/language_surface.json` has 1,058 builtin records;
the imported classification YAML records 1,043. It is therefore intentionally
not yet a reconciled/generated artifact. The classification generator and
checker are included to reconcile it when the verification lane is granted.
Review any newly classified names and retain the generator's explicit host
classifications for arbitrary-precision/exact numeric operations. The s64
floor rows cover fixed-width lowering data only. They do not authorize coercing
exact integers, bignums, or rationals into f64; unsupported type/domain cases
must remain refused. The signed quotient correction and remainder adjustment
are distinct programs: truncating quotient/remainder semantics must not be
conflated with floor quotient/modulo semantics.

The plan's inputs include all four sign quadrants, zero numerators, exact
divisibility with unlike signs, and cases that detect omission of correction.
`test_floor_builtin_plan.py` checks the host-side integer plan evaluator only.

## Remaining integration and proof dependencies

- Device lowering declarations/implementation, StableHLO emitter, XLA runtime
dispatch, CMake target wiring, and the C++ `deviceLowering` switch are not
ported here. The table cannot by itself lower or execute an operation.
- Region formation/execution and `region_tables.inc` are queued until their
region implementation exists in the target; no floor region-table rows were
copied.
- The source XLA parity harness and its CMake target are absent, so no plan was
executed against a device and no actual kernel parity is established.
- Current SDK availability, PJRT device availability, target hardware results,
and physical GPU execution remain unverified.

## Verification record

Source-only import is complete. No generator, checker, unit test, native build,
SDK, model, service, or device command has been run. After the root grants a
verification lane, run the Python classification generator, classification
checker, parity-plan generator with source reference execution where supported,
and `python3 -m unittest tests/xla/test_floor_builtin_plan.py`; preserve exact
commands, outputs, and negative-control results in the task evidence record.
