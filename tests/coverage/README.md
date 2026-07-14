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

Collect evidence and regenerate everything:

```sh
python3 scripts/gen_language_surface.py     # -> language_surface.json
# With both a default test build and an opt-in quantum build present, this
# creates fresh isolated traces, runs the complete deterministic corpus, and
# proves the 100% policy without reusing evidence from an earlier shell:
BUILD_DIR=build QUANTUM_BUILD_DIR=build-quantum \
  ./scripts/run_language_coverage.sh

# Existing trace directories may still be aggregated explicitly for forensic
# comparison or CI artifact replay:
LANGUAGE_COVERAGE_RUNTIME_TRACE_DIRS=/tmp/core-trace:/tmp/quantum-trace \
  BUILD_DIR=build ./scripts/run_language_coverage.sh
```

The corpus and instrumentation are deterministic; the trace is per-process TSV
so concurrent test workers can append evidence without sharing mutable state.

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

1. The parser records exact source spelling and location (`P`) only when
   `ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR` is set before compilation. It also
   records fully accepted top-level forms (`A`) and expected compile-time
   rejections (`R`, currently `syntax-error`).
2. LLVM code generation records reached AST nodes (`G`) and injects lightweight
   runtime hooks into that instrumented module. Executed operations (`O`) and
   direct calls (`C`) are emitted by the running JIT/AOT program.
3. The bytecode compiler serializes two exact dispatch witnesses. Native calls
   carry their native-ID alias marker (`V name`), while direct Scheme closure
   calls carry a stable 31-bit FNV-1a marker (`V hash @call`). The VM validates
   each marker immediately beside the actual `CALL`/`TAIL_CALL` dispatch;
   `language_coverage.py` resolves hashes only against the checked-in manifest
   and rejects collisions rather than granting ambiguous credit.
4. `language_coverage.py` grants ordinary builtins and runtime forms credit only
   from `O`/`C`/validated `V`. A parser spelling is joined to execution by normalized
   source+line+column, so aliases and reader forms remain auditable after parser
   lowering. `A`/`G` can credit only an explicit allowlist of forms whose
   semantics are compile-time (for example `define`, `require`, and
   `define-syntax`); negative forms require an `R` event.
5. The source-head collector remains as a diagnostic. Its
   `source_exposed_only_names` receive **zero release credit**. A call in an
   untaken branch has `P` and `G`, but no `O`/`C`, and is therefore uncovered.
6. The regression test `scripts/test_runtime_language_coverage.py` exercises a
   real untaken branch, exact ESKB native aliases, exact serialized direct
   Scheme calls, collision rejection, and an unset trace environment.

Normal generated programs contain no hooks unless tracing was enabled in the
compiler process. Parser dispatch has one cached false branch in production;
trace formatting/allocation occurs only in an opt-in run. Trace writes are
deduplicated per process and flushed in batches.

## Wiring into the ICC completion-oracle

The coverage fraction is designed to be an oracle criterion, not a one-off
report. `language_coverage.py --emit-runtime-event` prints two long-form ICC
`runtime_event` records: the monotonic floor and the final high-risk-complete
criterion. `--trace PATH` writes them as fresh JSONL evidence:

```json
{"kind": "runtime_event", "event": "language_surface_coverage",
 "name": "language_surface_coverage", "value": "PASS",
 "covered_fraction": 1.0, "covered": 1056, "surface_total": 1056,
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
