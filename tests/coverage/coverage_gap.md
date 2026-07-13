# Language-surface coverage completion — phases 3–4

Ground truth is generated from compiler/runtime sources into
`language_surface.json`; deterministic execution evidence is generated into
`coverage_run.json`. Regenerate and enforce both with:

```sh
./scripts/run_language_coverage.sh --require-zero-high-risk
```

## Phase result

| Metric | Before | Current |
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

## Full-surface closure

All 1,056 manifest constructs now occur in deterministic programs that their
mandatory CI harness actually compiles and executes.  The final lower-risk
closure added hermetic native and VM probes for port lifecycles, file/process
operations, atomics and raw-pointer FFI, image resizing, condition variables,
futures, polling, and immediate process termination.  Those probes exposed and
fixed implementation defects instead of receiving token-only credit: native
`directory-walk` returned a packed string rather than a list, `current-jiffy`
lost exactness and nanosecond precision, string conversions returned untagged
buffers, file wrapper arities and current-port rebinding were incomplete, and
VM image results were incorrectly freed despite arena ownership.

The policy floor is therefore ratcheted to 1,056/1,056.  Any construct removed
from the executable corpus now fails CI; the floor cannot be lowered by a
command-line threshold.
