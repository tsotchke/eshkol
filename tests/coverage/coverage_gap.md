# Language-surface coverage completion — phases 3–4

Ground truth is generated from compiler/runtime sources into
`language_surface.json`; deterministic execution evidence is written to the
gitignored `build/coverage/coverage_run.json`. Enforce the floor with:

```sh
./scripts/run_language_coverage.sh --require-zero-high-risk
```

That run leaves the working tree clean. To also refresh the **committed**
sidecar `coverage_run.json` — a deliberate artifact, e.g. at a release cut or
after the policy floor moves — add the explicit flag:

```sh
./scripts/run_language_coverage.sh --require-zero-high-risk --update-committed-run
```

## Phase result (historical — the #277 phase-3/4 campaign)

The numbers in this section are the record of that campaign, not the live
contract. The current surface and floor are **1,078/1,078** — see
"A1 — execution-backed certification" below and `coverage_policy.json`.

| Metric | Before | At the close of #277 |
|---|---:|---:|
| User-facing surface | 1,057 | 1,056 |
| Deterministically exercised | 137 (13.0%) | **1,056 (100.0%)** |
| Uncovered high-risk constructs | hundreds | **0** |
| Uncovered constructs (all categories) | 920 | **0** |

The one-entry surface correction removed `det`, which existed only in a
commented-out C++ dispatch example and was never callable. The manifest
generator now strips C/C++ comments before extracting LLVM dispatch names.

Coverage is not inferred from dead examples or documentation. Evidence comes
from deterministic programs that CI actually compiles and runs: the portable
differential corpus, AD adversarial programs, the complete Scheme test corpus,
the opt-in quantum suite, and explicit AOT/VM surface suites. Parser-lowered
promise helpers are credited only when their corresponding `delay` or
`delay-force` forms execute.

## High-risk closure

All constructs in the monotonic policy's silent-wrong or memory-safety buckets
now have execution evidence:

- numeric
- tensor and automatic differentiation
- geometry
- control flow
- consciousness / neuro-symbolic runtime
- macro syntax and hygiene
- region-memory ownership

The campaign exposed and fixed real defects rather than weakening the policy:
cross-representation vector mutation/equality, VM symbol identity, VM complex
and rational dispatch, dead dual-number aliases, a non-failing `syntax-error`,
invalid `tile` IR, and incorrect `tensor-apply` arithmetic/return packing.

## Full-surface closure (as of #277)

All 1,056 manifest constructs of that campaign occurred in deterministic
programs that their mandatory CI harness compiles and executes.  The final
lower-risk
closure added hermetic native and VM probes for port lifecycles, file/process
operations, atomics and raw-pointer FFI, image resizing, condition variables,
futures, polling, and immediate process termination.  Those probes exposed and
fixed implementation defects instead of receiving token-only credit: native
`directory-walk` returned a packed string rather than a list, `current-jiffy`
lost exactness and nanosecond precision, string conversions returned untagged
buffers, file wrapper arities and current-port rebinding were incomplete, and
VM image results were incorrectly freed despite arena ownership.

The policy floor was therefore ratcheted to 1,056/1,056 at that point, and has
since been re-baselined upward to **1,078/1,078** (below).  Any construct removed
from the executable corpus fails CI; the floor cannot be lowered by a
command-line threshold.

## A1 — execution-backed certification, no lexical credit (2026-07-24)

The coverage verdict is certified by **executed behaviour only**. The lexical
source-head collector (`collect_heads`) is a diagnostic: a construct whose name
merely appears in a generator or the corpus earns zero release credit. Both
numbers are reported every run and checked in to `coverage_run.json`.

| Metric | Value |
|---|---:|
| User-facing surface (manifest) | 1,078 |
| **Execution-backed covered** (the only gated number) | **1,078 (100.0%)** |
| Lexical exposure (diagnostic only, zero credit) | 1,076 (99.8%) |
| Spelled-but-unproven (in corpus text, no execution) | 0 |
| Execution-backed but not lexically spelled | 2 (`%make-lazy-promise`, `%make-lazy-promise-force`) |

The truth gap AR1 feared — execution-backed trailing lexical name-presence —
does not exist: execution-backed coverage (100.0%) exceeds lexical exposure
(99.8%). The two execution-only constructs are compiler-generated promise
helpers that no source spells.

The surface grew from the 1,056 proven in #277 to 1,078: the `i128` numeric
tower (20 builtins), `linear-solve`, and `string`/`ptr` conversions landed as
new core builtins after that campaign, while the committed sidecar and policy
floor still read 1,056. The gate always measures against the live manifest, so
the honest floor is re-baselined here to 1,078/1,078 rather than left stale, and
all 22 agent-FFI (quantum/PQC) constructs are proven on the quantum build.

Guarantees added by this work:

- `verify_execution_backed_invariant` refuses to certify any construct that
  lacks runtime/compile-time evidence — a permanent tripwire against routing
  lexical name-presence back into the gate.
- `execution_deficit.json` is a monotonic deficit ledger and work queue: the
  gate fails if execution-backed coverage drops **or** the named deficit grows.
  `--write-execution-deficit` refuses to record a larger deficit without an
  explicit `--allow-deficit-growth`, so the claim is never walked down silently.
- `scripts/test_language_coverage_gate.py` pins both properties build-free.
- The ICC `execution_backed_coverage` criterion consumes the dedicated
  `execution_backed_language_coverage` runtime_event.
