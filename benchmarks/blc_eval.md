# Normal-order evaluation through core.blc

Both builds in this benchmark call `(blc-eval term)`. Each trial evaluates a prebuilt term in a fresh native process and checks the full normal form or expected limit diagnostic. One binary uses the released substitution evaluator. The other uses lexical environments from this checkout. The public API and the one-step oracle `blc-step` are the same.

## The observable budget

Consider `(λx. x x) (I y)`, where `I` is identity and `y` is free. The first contraction produces `(I y) (I y)`. Reducing the operator and the operand costs one contraction apiece. At bound two the call must report the public step-bound error. At three it must return `y y`. A neutral variable lookup, visiting a binder, or constructing an output node consumes no contractions. The test suite also reduces this term under a lambda, discards Ω as an unused argument, and drives Ω itself to the default bound.

A suspension stores the argument and its lexical environment. Each demand evaluates the suspended code again. Readback visits the arguments of a neutral head from left to right and enters unapplied lambdas. The initial normal-form check stops at the first redex. At contraction the machine validates the raw body and argument that the reference would traverse. Encountering noncanonical syntax sends the original input through the reference loop for its precise diagnostics and permissive behavior.

An exhausted left branch can hide a large shared right argument. At bound zero, `((I I) right)` must signal the limit after attempting the left contraction. The native benchmark constructs `right` with sixteen applications that each reuse their child twice. The regression suite also puts two distinct wrappers around each shared child at forty levels. It checks exhaustion at zero and positive bounds across AOT, JIT and WASM. The process timeout catches an accidental traversal of the trillion-leaf expansion without imposing a machine-specific timing threshold.

## What the theorem covers

`BLC.Costed.bounded_correspondence` relates an independent leftmost-outermost single-step reducer to an unshared closure machine for every finite lambda term and natural beta budget. Its answer is either the full normal form with the exact contraction count, or the limit error. The machine has a separate internal allowance for administrative transitions. The proof establishes that a sufficient finite allowance always exists for the bounded observation. A decreasing measure rules out endless administrative work between contractions. The proof also allows beta steps whose result is syntactically identical to their input, as in Ω.

`exact_cost` specializes the theorem to completion in exactly k contractions. `same_limit_error` specializes it to exhaustion. Lean reports `propext` and `Quot.sound` for the main theorem. The model uses zero-based terms and functional environments. The runtime uses one-based tagged lists, linked environments, and native or VM execution. The differential tests exercise that boundary. A mechanized refinement of the executing Eshkol code, concrete allocation, aliases and malformed values is outside the theorem. [Lean model and proof](../verification/blc/Costed.lean)

## Native measurements

Linux x86-64, AMD Ryzen 5 5600H, Eshkol v1.3.6-evolve x64 lite, AOT `-O 2`. Each median covers nine fresh processes. The harness alternates the launch order and times the public evaluation and, for the bounded case, its error handler. Both builds use the same compiler, benchmark and prebuilt input. The released module and upstream base `eb224260b` have identical source bytes.

| Workload | Released µs | Candidate µs | Released / candidate |
|---|---:|---:|---:|
| Tromp U, identity, marker K | 49,786.3 | 143.6 | 346.60× |
| K I Ω | 26.4 | 69.6 | 0.38× |
| Church 2 applied to Church 5 | 389.2 | 137.0 | 2.84× |
| Second byte before Ω | 627.1 | 212.5 | 2.95× |
| 32 binders, one contraction | 32.9 | 50.8 | 0.65× |
| 256 binders, one contraction | 267.1 | 350.8 | 0.76× |
| 256 binders, already normal | 156.5 | 43.0 | 3.64× |
| Shared right argument, 16 levels, bound zero | 17.4 | 15.7 | 1.11× |
| Two nested duplicators | 313.2 | 59.8 | 5.24× |
| Five nested duplicators | 4,476.6 | 390.5 | 11.46× |
| Eight nested duplicators | 142,210.6 | 3,192.5 | 44.55× |

Tromp’s interpreter benefits from avoiding repeated substitution over large syntax trees. With one contraction under 256 binders, validating the input and rebuilding the result cost more than the source operation saves. For the already-normal input the initial scan returns the original tree. The shared-bound row measures the limit diagnostic, including its handler. Ratios below one indicate a slowdown.

## Reproduce

Use Linux x86-64, Python 3.10+, Node 22, Lean 4.35.0 and the Eshkol v1.3.6-evolve Linux x64 lite distribution with its matching LLVM 21 libraries and native linker. Put the extracted `bin` directory on PATH. The benchmark checks the installed reference module against SHA-256 `25048643f33b8b24f5e1602e4ec13d85f745bf88fda49c504dce13138cf38d6c`. The release archive has SHA-256 `a1c467edf5ee519ab36d3f1a24df2eb2cba80211e303fec5be25a49f223d5f3b`. [Eshkol release](https://github.com/tsotchke/eshkol/releases/tag/v1.3.6-evolve) · [Lean release](https://github.com/leanprover/lean4/releases/tag/v4.35.0)

```sh
python3 benchmarks/check_blc_eval.py
```

The runner checks the model, then compares the public result at the exact bound and one below it on 1,560 frozen binder cases in AOT and JIT. A separate replay runs the same cases on the bundled WASM VM. Each row includes an independently generated one-step contractum. The uncompressed corpus SHA-256 is `97bf65fc95fd4c7fe5aa306b488660e4c45af3242a20397046c0b7d0ad9051be`. An off-by-one runtime mutation must fail the exact-bound assertion on the duplicated argument. The runner outputs the nine individual timings and median per workload and build as JSON.
