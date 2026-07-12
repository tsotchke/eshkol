# Language-surface coverage gap — phases 3–4

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
| Deterministically exercised | 137 (13.0%) | 809 (76.6%) |
| Uncovered high-risk constructs | hundreds | **0** |

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

## Remaining surface: 247 lower-risk constructs

| Category | Uncovered | Notes |
|---|---:|---|
| ffi_system | 175 | OS, process, network, terminal, compression, tree-sitter, atomics |
| predicate | 21 | port/error predicates and ordering variants |
| string_char | 12 | mutation and formatting helpers |
| io_port | 12 | file/port redirection and byte/string readers |
| hash | 8 | deletion, enumeration, count/default helpers |
| misc_core | 8 | environment/error/sum-type helpers |
| vector | 3 | append/string conversion/bytevector copy mutation |
| misc | 3 | internal public helpers |
| binding_form | 2 | `define-values`, explicit `named-let` head |
| higher_order | 1 | `fold-left` alias |
| list_pair | 1 | `remv` |
| module | 1 | `include-ci` |

These are the next one-way ratchets toward the aspirational 100% surface. They
are intentionally not marked complete: most FFI/system calls need hermetic
fixtures or capability-gated integration tests so their evidence remains real,
portable, and production-relevant.
