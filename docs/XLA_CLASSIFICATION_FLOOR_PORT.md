# XLA classification and floor plan port status

This checkout is an isolated clone of canonical connection-port/source at
7f01cd9707d4368dc6731ad6d91ae8d1eec6a76a. It carries the original bounded
classification/parity-plan layer and the focused signed-floor plan regression.
The source-derived Python plan is an input to future parity work; generating it
does not execute a StableHLO lowering, device kernel, or GPU test.

## Reconciliation and semantics

The canonical `tests/coverage/language_surface.json` has 1,058 builtin names.
The original classification had 1,043: exact reconciliation found 15
additions and no removals, with no label changes among the 1,043 shared names.
All 15 additions are explicitly host-only in the generator, each with a
source-grounded reason; category defaults alone do not claim device support.
The `device` label records planned S-fragment eligibility; it is not a count
of implemented or validated device kernels.
The generated classification has 1,058 entries (233 device, 816 host,
9 host-with-device-inner), and the checker reports no missing or stale names.

The s64 floor rows cover fixed-width plan data only. They do not authorize
coercing exact integers, bignums, or rationals into f64; those operations stay
host-only and are absent from the lowering table. Signed `quotient` and
`remainder` use truncation toward zero; `modulo`, `mod`, `floor-remainder`, and
`floor-quotient` use floor semantics. The plan validator requires all six
rows, their s64 binary type, safe nonzero fixtures, and distinct correction
steps. `s64 MIN / -1` is explicitly outside this plan domain because its
mathematical quotient is not representable in s64; it is rejected by the
validator. Exact/bignum type requests stay host-only rather than being narrowed
or rounded.

`test_floor_builtin_plan.py` checks the host-side integer plan evaluator only,
including negative controls for a missing row, wrong dtype, and a table entry
whose correction program was replaced by truncating division. It also checks
that s64 plan text round-trips both signed endpoints without passing through a
float.

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

Verified in the granted Python lane with `ICC_PYTHON=/Users/tyr/Desktop/infinite_context_coder/.venv/bin/python`:

- `$ICC_PYTHON scripts/gen_builtin_classification.py`: 1,058 builtins; 233 device,
  816 host, 9 host-with-device-inner.
- `$ICC_PYTHON scripts/check_builtin_classification.py`: PASS; 0 unclassified,
  0 stale.
- `$ICC_PYTHON tests/xla/test_floor_builtin_plan.py`: 8 tests PASS, including the
  three table negative controls and exact-domain checks.
- `XDG_CACHE_HOME=/private/tmp/xla-floor-xdg-cache $ICC_PYTHON
  scripts/gen_builtin_parity_plan.py --eshkol-run /usr/local/bin/eshkol-run
  --out docs/xla-classification-floor-evidence/floor-reference-plan.txt
  --work docs/xla-classification-floor-evidence/work --builtin modulo --builtin
  mod --builtin floor-remainder --builtin floor-quotient --builtin quotient
  --builtin remainder`: 6 host references, 0 missing. The durable Eshkol
  source program, log, and plan are under
  `docs/xla-classification-floor-evidence/`.

The plan generator sets `ESHKOL_XLA_DEVICE=0`; these are language-side host
references paired with proposed table operations. No StableHLO lowering was
executed, no GPU kernel parity was measured, and no native build, SDK, model,
service, or device command was run.

## Current candidate host references

Root regenerated all six references using the built canonical engine at
`7f01cd9707d4368dc6731ad6d91ae8d1eec6a76a`, SHA-256
`4782240036dae1b33f77fbaea3aba60767c3d483c3aa0983e836ff03c8ce37e5`,
with JIT cache disabled and the matching candidate standard library. The
actual process exited zero and the plan contains six `REF` rows and zero
`NOREF` rows. The command, engine identity, and plan are preserved under
`docs/xla-classification-floor-evidence/current-7f01cd97/`. These remain host
references; they do not establish device lowering or accelerator parity.
The earlier `/usr/local/bin/eshkol-run` evidence is preserved above with its
original command; its engine revision was not recorded, so it is not used as
current candidate proof.
