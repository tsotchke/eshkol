# Eshkol v1.3.1 — flat memory for long-running programs, and a fully documented public API

**Eshkol v1.3.1 builds on v1.3.0-evolve's arbitrary-order Taylor-tower automatic differentiation and 100% R7RS conformance with what a differentiable systems language needs to run unattended: flat memory in long-running resident/daemon workloads, and a comprehensive documentation pass across every public header and implementation module.**

Eshkol is Scheme (R7RS) compiled to native code via LLVM, with automatic differentiation as a compiler primitive rather than a library. Where v1.3.0-evolve took AD from "correct at first and second order" to "correct, exact, and validated at *every* order" and closed out a long list of robustness gaps found by a new permanent adversarial-testing program, v1.3.1 closes the remaining gap between "runs correctly" and "runs unattended for days": a memory-growth class that survived the v1.3.0-evolve TCO-loop hardening, and a native-stack overflow on very large persisted data.

Full engineering detail lives in [CHANGELOG.md](CHANGELOG.md); the user-facing summary is [RELEASE_NOTES.md](RELEASE_NOTES.md); the complete AD walkthrough is the [Automatic Differentiation guide](docs/guide/AUTOMATIC_DIFFERENTIATION.md). This document is the release announcement — quotable, and every claim below is grounded in shipped code and a real, run command.

---

## What's new in v1.3.1

### Flat memory for resident and daemon workloads

v1.3.0-evolve already gave named-let TCO loops automatic, zero-annotation per-iteration arena-scope reclamation (ESH-0214). v1.3.1 closes the remaining case in that same bug class and fixes a second, unrelated way a long-running process's native stack could blow out:

- **Self-tail-recursive `define` loops, including a catch-all guard body** (ESH-0214b): the escape analysis that makes per-iteration arena reclamation safe now also recognizes self-tail-recursive functions written with plain `define` rather than named-let, and accepts a catch-all guard body in that analysis rather than declining to optimize it. Verified on a 1,000,000-iteration loop: RSS goes from 1,369 MB (unbounded growth) to 224 MB (flat).
- **Iterative S-expression reader** (ESH-0191): `read_list`, the core parser routine for reading persisted S-expression data back in, was rewritten from one native stack frame per list element to an iterative loop. A program that reads back a very large persisted data structure — a long checkpoint, a big literal table, a large logged fact base — no longer risks a native-stack overflow on load. Verified: the pre-fix reader crashed with SIGBUS at 20 million elements; the post-fix reader completes cleanly at the same size.

Together, these are the difference between a program that degrades over a long run and one that doesn't: an Eshkol process built as a resident worker or daemon — a long-lived agent loop, a persistent server, a checkpoint/restore pipeline — can now run with flat memory instead of monitoring RSS and scheduling restarts.

### Comprehensive C-API and implementation documentation

Every public embedding header and every previously-undocumented implementation file gets Doxygen-format documentation in this release: 50 of the 64 public headers under `inc/eshkol/` (backend codegen, runtime core, type system, XLA backend, subprocess/macro-expander/qLLM-bridge surfaces, thread pool and work-stealing deque, logger, model I/O, platform runtime, runtime exports) across six commits, plus 56 previously-undocumented implementation files under `lib/` (agent FFI, the type checker, the parser, the REPL, core non-runtime modules, the quantum RNG, FFI bridges) across three commits. Combined, that is 116 files and roughly 12,600 lines of new documentation — comments only, no behavior changes. Anyone embedding `libeshkol` or reading the compiler's own source now has a documented contract to read instead of guessing from call sites.

### A navigable documentation reference index

A per-subsystem reference index now organizes the language surface for lookup rather than linear reading: [`docs/reference/language/`](docs/reference/language/INDEX.md), [`ad/`](docs/reference/ad/INDEX.md), [`runtime/`](docs/reference/runtime/INDEX.md), [`tensors/`](docs/reference/tensors/INDEX.md), [`stdlib/`](docs/reference/stdlib/INDEX.md), and [`agent/`](docs/reference/agent/INDEX.md), each an example-verified index into the corresponding function and syntax reference. It is linked from [README.md](README.md#documentation).

---

## The foundation this release runs on: v1.3.0-evolve

### Arbitrary-order automatic differentiation

Eshkol's forward-mode, reverse-mode, and symbolic AD were already exact at first and second order. v1.3.0-evolve added a second axis on top of that: **order**. A Taylor-tower engine — designed in [`docs/design/AD_TAYLOR_TOWER.md`](docs/design/AD_TAYLOR_TOWER.md) and delivered across thirteen gated phases, P0 through P12 — computes *every* derivative up to an arbitrary order `k` in one pass, using closed Taylor recurrences (`lib/core/taylor_recurrences.def`, `lib/core/runtime_taylor.c`) instead of nested dual numbers. Nested/hyper-dual AD doubles its representation with every additional order (2^k); Taylor-mode is `k+1` coefficients and O(k²) work — polynomial, not exponential, in the order.

That is the same design JAX's `jax.experimental.jet` uses — and Eshkol's implementation goes further in three ways mainstream AD/ML frameworks (JAX, PyTorch) do not offer at the language level:

- **Exact coefficients.** When the seed point is an exact number (an integer or rational, not a `double`) and the differentiated function only uses exact-preserving operators (`+ - * /` and integer `expt`), `derivative-n` and `taylor` return **exact arbitrary-precision** (bignum/rational) results — not floating-point approximations. The moment a transcendental primitive (`exp`, `sin`, ...) enters the computation, the tower gracefully demotes to the ordinary double tower, matching R7RS's own exactness-contagion discipline.
- **Validated enclosures.** `taylor-model`, `tm-range`, and `tm-eval` pair the Taylor polynomial with a rigorous interval-remainder bound, giving a *provable* enclosure of a function's range — not just a point estimate.
- **AD as a language property, not a library call.** Towers are tensor-valued (differentiate through `matmul`/`conv2d`/activations), compose through reverse-mode (checkpointed reverse-over-Taylor for memory-efficient high-order gradients), recover sparse high-order structure via graph coloring (`sparse-hessian`), and work correctly through `if`/`cond`/named-let/recursion — differentiable control flow, not just differentiable expressions.

Perturbation confusion (the classic nested-derivative correctness trap) is handled structurally: every differentiation context gets its own epoch tag carried in the tower's header, so an inner and outer `derivative` can never silently cross-contaminate.

Here it is, run for real — bignum-exact derivatives and a full Taylor series, straight from `eshkol-run -r`:

```scheme
;; f(x) = x^30 — arbitrary order, exact
(define (f x) (expt x 30))

(display "f^(12)(7), exact bignum: ")
(display (derivative-n f 7 12))
(newline)
(display "  exact? ") (display (exact? (derivative-n f 7 12))) (newline)

;; 1/(1-x) — exact rational-point derivative
(define (geom x) (/ 1 (- 1 x)))
(display "geom^(6)(1/2), exact: ")
(display (derivative-n geom (/ 1 2) 6))
(newline)

;; Full Taylor series, order 5, exact rational coefficients
(display "taylor(f, x0=2, order 5): ")
(display (taylor f 2 5))
(newline)
```

Real output, on this release:

```
f^(12)(7), exact bignum: 67465815595294257109436307840000
  exact? #t
geom^(6)(1/2), exact: 92160
taylor(f, x0=2, order 5): (1073741824 16106127360 116769423360 544923975680 1839118417920 4781707886592)
```

Every value is exact — `(exact? ...)` returns `#t` — and the first Taylor coefficient, `1073741824`, is `2^30` on the nose, confirming the series is centered correctly. No floating-point error anywhere in the pipeline. See the [Automatic Differentiation guide](docs/guide/AUTOMATIC_DIFFERENTIATION.md) for the full operator-by-operator walkthrough with verified output for all thirteen phases, and [`docs/reference/ad/INDEX.md`](docs/reference/ad/INDEX.md) for the API reference.

### 100% R7RS conformance on the portable corpus

A reference-Scheme differential oracle runs the same portable R7RS-small program on Eshkol and on **chibi-scheme 0.12.0** — the strictest mainstream R7RS-small implementation — and diffs the output. It started the v1.3.0-evolve cycle at 27/34 (79.4%) programs agreeing; every divergence found was fixed: `apply` with leading arguments before the final list, multi-vector `vector-map`/`vector-for-each`, quasiquoted vector literals, `cond`/`case` `=>` arrow clauses, an allocating `vector-copy`, and more. Current state, unchanged and re-verified for this release:

```
Total         : 34
AGREE         : 34
ESHKOL-DIVERGES : 0
Agreement rate: 100.0%
Gate          : PASS  (PASS iff every program AGREES)
```

(`scripts/run_reference_differential.sh`, corpus in `tests/reference-diff/corpus/`.) Separately, Eshkol implements roughly 95% of the broader R7RS-small procedure surface (232 of 244 procedures), covering the full numeric tower, continuations, exceptions, promises, `eval`, records, bytevectors, and hygienic macros.

### Robustness: tail calls and shutdown

A cluster of fixes from v1.3.0-evolve targets the class of bug that only shows up in a program that runs for a long time or recurses deeply, not in a quick test — the same class this release's arena-reclamation and reader fixes continue to close:

- **Proper mutual tail calls** (ESH-0102): a call in tail position to another function is emitted as an LLVM `musttail` call, so mutually tail-recursive functions (`even?`/`odd?`-style state machines) run in O(1) stack instead of overflowing after ~300k hops.
- **Safe teardown** (ESH-0216): `eshkol_runtime_shutdown()` stops and joins the global parallel thread pool before running shutdown hooks, closing a use-after-free race that could `SIGSEGV` well after a graceful `SIGTERM` was already logged; AOT-compiled binaries now also emit the paired runtime shutdown call they were previously skipping entirely.

Full root-cause detail for each: [CHANGELOG.md](CHANGELOG.md).

### A hardened, permanent adversarial-testing program

v1.3.0-evolve shipped the testing infrastructure that found and closed the gaps above, wired permanently into the ICC release oracle rather than run once and discarded: a multi-path differential harness with a seeded fuzzer, a feature-pair edge matrix, an AD finite-difference oracle, a stress harness with RSS/time budgets, a VM-parity ratchet, depth-parametric sweeps, and the external reference-Scheme differential oracle described above. See [`docs/TESTING.md`](docs/TESTING.md).

---

## Why it matters

Differentiable programming today mostly means "trace a Python function with a library and hope the trace is faithful." Eshkol takes the opposite bet: make the derivative operator part of the language the compiler already understands, so it composes with closures, recursion, control flow, and the numeric tower the same way `+` does — and, since v1.3.0-evolve, at any order, exactly, with a provable error bound when you need one. That is a meaningfully larger AD surface than what JAX or PyTorch expose to user code today, delivered from an ahead-of-time native compiler rather than a runtime tracer.

The robustness work in this release matters for a different reason: a differentiable systems language is only useful if the programs built on it can run unattended. Proper tail calls, flat memory in long-running loops, a reader that doesn't overflow the stack on large persisted state, and a shutdown path that doesn't race are the difference between a research demo and something you can leave running as a daemon.

## Get started

```bash
brew tap tsotchke/eshkol && brew install eshkol
```

Or build from source (LLVM 21+, C++20, CMake 3.14+) — see [README.md](README.md#quick-start) and [`docs/tutorials/00_FIRST_5_MINUTES.md`](docs/tutorials/00_FIRST_5_MINUTES.md).

Try the AD example above yourself:

```bash
eshkol-run -r your_file.esk
```

- **[Automatic Differentiation guide](docs/guide/AUTOMATIC_DIFFERENTIATION.md)** — the full P0–P12 walkthrough, every example verified against a real build.
- **[Documentation reference index](docs/reference/language/INDEX.md)** — navigable, example-verified language/AD/runtime/tensor/stdlib/agent reference.
- **[Eshkol Language Guide](docs/ESHKOL_LANGUAGE_GUIDE.md)** — tutorial introduction to the language.
- **[CHANGELOG.md](CHANGELOG.md)** — itemized engineering detail, phase by phase.
- **[RELEASE_NOTES.md](RELEASE_NOTES.md)** — the user-facing release summary, with the full gate matrix.

## Under the hood

The Taylor tower represents a function's local behavior as a truncated power series, `f(x0 + t) = Σ c_k · t^k`, so `f⁽ⁿ⁾(x0) = n! · c_n`. Each primitive operator (`+ - * / exp log sin cos sqrt tan atan tanh pow ...`) has a closed recurrence for producing its output series' coefficients from its input series' coefficients — for example, Cauchy convolution for multiplication (`s_k = Σ_{j=0..k} u_j · w_{k-j}`) and a linear recurrence for `exp`/`log`/`sin`/`cos`. Composing these recurrences through a program the way the LLVM backend already composes arithmetic gives arbitrary-order differentiation at `O(k²)` cost, with **zero heap allocation** on the common path: when the order `k` is a literal at the call site — the overwhelmingly common case in a compiler — the entire tower is unrolled into stack-allocated, branch-free SSA IR at compile time. A runtime heap-allocated tower (`HEAP_SUBTYPE_TAYLOR`) is the correctness fallback for a dynamically chosen order.

The per-iteration arena-reclamation fix in this release uses the same style of static escape analysis: a self-tail-recursive loop body (named-let or plain `define`) is proven safe to reclaim per iteration by conservatively checking that nothing escapes the loop's arena scope across the back-edge — a catch-all guard clause is now accepted into that analysis rather than causing the optimization to bail out. The iterative reader fix replaces per-element native recursion in `read_list` with an explicit loop and heap-allocated work list, so the reader's stack usage no longer scales with the length of the list being read.

---

*Eshkol v1.3.1. MIT License. [github.com/tsotchke/eshkol](https://github.com/tsotchke/eshkol)*
