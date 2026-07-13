# Language-surface coverage tracking

This directory is the backbone for **TOTAL-LANGUAGE exposure-engine coverage**:
the mechanism that measures which of Eshkol's 1,056 user-facing constructs the
generative exposure engines actually exercise, names the gap, and turns
"cover the whole language" into a measurable ICC completion-oracle criterion.

## Artifacts

| File | Producer | What it is |
|---|---|---|
| `language_surface.json` | `scripts/gen_language_surface.py` | Ground-truth manifest: every core builtin, tracked Agent FFI API, special form, AST op, and prelude fn, each categorised by risk. Extracted from source — never hand-maintained. |
| `coverage_policy.json` | monotonic ratchet | Minimum covered count/fraction and the categories that must reach zero uncovered before TOTAL-LANGUAGE completion. The floor can only increase. |
| `coverage_run.json` | `scripts/language_coverage.py` | Per-run sidecar: covered / total, covered fraction, covered + uncovered names by category. |
| `coverage_gap.md` | analysis | Human-readable gap report ranked by silent-wrong risk. |

Regenerate everything:

```sh
python3 scripts/gen_language_surface.py     # -> language_surface.json
python3 scripts/language_coverage.py        # -> coverage_run.json  (+ prints summary)
./scripts/run_language_coverage.sh          # check manifest + policy; write ICC JSONL
```

Both scripts are pure functions of the source tree and the (deterministic,
seeded) generators, so output is reproducible and diffable in CI.

## How the manifest is built (ground truth)

`gen_language_surface.py` parses four sources directly so the surface can never
silently drift from the compiler:

1. **Native first-class closure table** — `lib/backend/eshkol_compiler.c`
   `static const BuiltinDef BUILTINS[]` (`{name, native_id, arity}`).
2. **Bytecode VM table** — `lib/backend/eshkol_vm.c` `BUILTINS[]`.
3. **LLVM AOT dispatch** — every `func_name == "name"` in
   `lib/backend/llvm_codegen.cpp`. This is the AOT intrinsic surface and adds
   ~299 builtins absent from the id-tables (R7RS IO/mutation, the NN/optimizer/
   linalg surface, atomics, the extended numeric tower).
4. **Special forms** — `lib/frontend/parser.cpp` `get_operator_type`
   (keyword → `eshkol_op_t`) plus the directly-dispatched forms
   (`begin`, `define-library`, `delay`, `named-let`, ...), and the
   `eshkol_op_t` enum from `inc/eshkol/eshkol.h`.
5. **Tracked Agent FFI APIs** — the Moonlab `provide` surfaces in
   `lib/agent/quantum.esk` and `lib/agent/pqc.esk`. These entries are marked
   `agent_ffi`, not falsely attributed to the core VM/native builtin tables.

Each builtin records which backend(s) register it (`native`, `vm`,
`native_llvm`) so a construct that exists in only one backend is visible.

## How coverage is measured (dynamic)

`language_coverage.py` is the "ICC tracks the language dynamically" mechanism:

1. Import both generative engines in-process, and read the complete deterministic
   CI corpus (including the quantum acceptance corpus and explicit VM/AOT
   extension suites) that the workflow compiles and runs:
   `gen_generative_corpus.generate_programs()` and
   `gen_ad_adversarial.Gen().generate()` (plus their in-language preludes,
   which are compiled and executed as part of every program), the complete
   `run_all_tests.sh` Scheme corpus, and the opt-in `tests/quantum/*.esk` files.
2. Scan the concatenated generated source with a small s-expression head
   collector: the symbol immediately following each `(` is an
   application/operator head, and the reader macros `'` `` ` `` `,` `,@` and
   `#(` are mapped to `quote`/`quasiquote`/`unquote`/`unquote-splicing`/
   `vector`. This yields the exact set of constructs the corpus *invokes*.
3. Intersect with the manifest surface → **covered**; the complement →
   **uncovered**. Emit `coverage_run.json` and a ranked summary.

### Why source-scan rather than runtime instrumentation

The head-collector reads the generated *source*, which for these engines is
exactly what gets compiled and run (the generators emit closed, total programs;
nothing is dead). This needs no compiler changes and no execution, so it runs
in CI in milliseconds and is engine-agnostic — any new generator is measured by
adding one source accessor. The sidecar records `exercised_by_quantum_tests`
separately so Moonlab-only coverage remains auditable.

### Upgrade path: true execution-time instrumentation

For engines whose generated programs branch (so that "contains `X`" overstates
"executed `X`"), the same sidecar shape is produced by a runtime pass instead:
each `NATIVE_CALL`/intrinsic emission records its builtin id, the VM/AOT runtime
appends the exercised ids to a per-program coverage log, and
`language_coverage.py` unions those logs instead of scanning source. The
manifest already keys builtins by `native_id`, so the log→name join is direct.
The JSON contract (`covered`, `surface_total`, `covered_fraction`,
`uncovered_by_category`) is identical, so downstream ICC wiring does not change.

## Wiring into the ICC completion-oracle

The coverage fraction is designed to be an oracle criterion, not a one-off
report. `language_coverage.py --emit-runtime-event` prints two long-form ICC
`runtime_event` records: the monotonic floor and the final high-risk-complete
criterion. `--trace PATH` writes them as fresh JSONL evidence:

```json
{"kind": "runtime_event", "event": "language_surface_coverage",
 "name": "language_surface_coverage", "value": "PASS",
 "covered_fraction": 0.7661, "covered": 809, "surface_total": 1056,
 "status": "PASSED"}
```

The integration mirrors `define_loop_flat_rss_aot` and the other release
pillars:

1. `scripts/run_language_coverage.sh` first proves the checked-in manifest is
   source-current, then runs the tracker and writes
   `scripts/icc_traces/language_surface_coverage.jsonl`.
2. `coverage_policy.json` owns the one-way floor. A command-line threshold may
   raise it for an exploratory run but cannot lower it.
3. `eshkol-compiler-readiness` requires the floor event to PASS.
4. `total-language-coverage` additionally requires
   `language_surface_high_risk_complete=PASS`. Phase 4 closes both that
   criterion and the complete surface: all 1,056 constructs now have
   deterministic execution evidence and the monotonic policy floor is 100%.
5. Gate the campaign after regenerating traces:
   `icc readiness --repo eshkol --target total-language-coverage --trace-dir scripts/icc_traces`.

Because the threshold ratchets, the oracle enforces monotonic progress toward
total-language coverage: any engine change that drops a previously-covered
construct fails the gate, and the only way to raise the bar is to genuinely
exercise more of the surface.

## Categories (risk buckets)

Every construct is tagged with one category, ordered by silent-wrong risk for
prioritisation: `numeric`, `tensor_ad`, `geometry`, `control_flow`,
`consciousness`, `higher_order`, `list_pair`, `vector`, `string_char`, `hash`,
`predicate`, `io_port`, `binding_form`, `macro_syntax`, `module`,
`memory_region`, `misc_core`, `ffi_system`, `misc`. See `coverage_gap.md` for
the remaining lower-risk surface and the next monotonic ratchet.
