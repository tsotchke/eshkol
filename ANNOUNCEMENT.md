# Eshkol v1.3.0-evolve — arbitrary-order automatic differentiation, and a compiler you can trust with long-running programs

**Eshkol v1.3.0-evolve ships a Taylor-tower automatic-differentiation engine that computes exact derivatives of *any* order — including exact bignum/rational coefficients no double-based system can produce — alongside 100% conformance with a portable R7RS corpus verified against chibi-scheme, and a hardening pass across tail calls, memory, and shutdown that targets the kind of bug that only shows up in a program that runs for a long time.**

Eshkol is Scheme (R7RS) compiled to native code via LLVM, with automatic differentiation as a compiler primitive rather than a library. v1.3.0-evolve is the "evolve" release: it takes AD from "correct at first and second order" to "correct, exact, and validated at *every* order," and closes out a long list of robustness gaps found by a new permanent adversarial-testing program shipped in the same release.

Full engineering detail lives in [CHANGELOG.md](CHANGELOG.md); the user-facing summary is [RELEASE_NOTES.md](RELEASE_NOTES.md); the complete AD walkthrough is the [Automatic Differentiation guide](docs/guide/AUTOMATIC_DIFFERENTIATION.md) (PR #180). This document is the release announcement — quotable, and every claim below is grounded in shipped code and a real, run command.

---

## What's new

### Arbitrary-order automatic differentiation (the headline)

Eshkol's forward-mode, reverse-mode, and symbolic AD were already exact at first and second order. v1.3.0-evolve adds a second axis on top of that: **order**. A new Taylor-tower engine — designed in [`docs/design/AD_TAYLOR_TOWER.md`](docs/design/AD_TAYLOR_TOWER.md) and delivered across thirteen gated phases, P0 through P12 (PRs #147, #158, #160, #162, #163, #167–#169, #173, #174, #177, #178) — computes *every* derivative up to an arbitrary order `k` in one pass, using closed Taylor recurrences (`lib/core/taylor_recurrences.def`, `lib/core/runtime_taylor.c`) instead of nested dual numbers. Nested/hyper-dual AD doubles its representation with every additional order (2^k); Taylor-mode is `k+1` coefficients and O(k²) work — polynomial, not exponential, in the order.

That matters because it is the same design JAX's `jax.experimental.jet` uses — and Eshkol's implementation goes further in three ways mainstream AD/ML frameworks (JAX, PyTorch) do not offer at the language level:

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

A new reference-Scheme differential oracle (P7a) runs the same portable R7RS-small program on Eshkol and on **chibi-scheme 0.12.0** — the strictest mainstream R7RS-small implementation — and diffs the output. It started the release cycle at 27/34 (79.4%) programs agreeing; every divergence found is now fixed: `apply` with leading arguments before the final list, multi-vector `vector-map`/`vector-for-each`, quasiquoted vector literals, `cond`/`case` `=>` arrow clauses, an allocating `vector-copy`, and more. Re-run live for this announcement:

```
Total         : 34
AGREE         : 34
ESHKOL-DIVERGES : 0
Agreement rate: 100.0%
Gate          : PASS  (PASS iff every program AGREES)
```

(`scripts/run_reference_differential.sh`, corpus in `tests/reference-diff/corpus/`.)

### Robustness: tail calls, memory, and shutdown

A cluster of fixes targets the class of bug that only shows up in a program that runs for a long time or recurses deeply, not in a quick test:

- **Proper mutual tail calls** (ESH-0102): a call in tail position to another function is now emitted as an LLVM `musttail` call, so mutually tail-recursive functions (`even?`/`odd?`-style state machines) run in O(1) stack instead of overflowing after ~300k hops.
- **Bounded-RSS long-running loops** (ESH-0214/ESH-0214b): named-let TCO loops get automatic, zero-annotation per-iteration arena-scope reclamation, closing a production-triggered class of unbounded memory growth.
- **Safe teardown** (ESH-0216): `eshkol_runtime_shutdown()` now stops and joins the global parallel thread pool before running shutdown hooks, closing a use-after-free race that could `SIGSEGV` well after a graceful `SIGTERM` was already logged; AOT-compiled binaries now also emit the paired runtime shutdown call they were previously skipping entirely.

Full root-cause detail for each: [CHANGELOG.md](CHANGELOG.md).

### A hardened, permanent adversarial-testing program

This release also ships the testing infrastructure that found and closed the gaps above, wired permanently into the ICC release oracle rather than run once and discarded: a multi-path differential harness with a seeded fuzzer, a feature-pair edge matrix, an AD finite-difference oracle, a stress harness with RSS/time budgets, a VM-parity ratchet, depth-parametric sweeps, and now the external reference-Scheme differential oracle described above. See [`docs/TESTING.md`](docs/TESTING.md).

---

## Why it matters

Differentiable programming today mostly means "trace a Python function with a library and hope the trace is faithful." Eshkol takes the opposite bet: make the derivative operator part of the language the compiler already understands, so it composes with closures, recursion, control flow, and the numeric tower the same way `+` does — and, as of this release, at any order, exactly, with a provable error bound when you need one. That is a meaningfully larger AD surface than what JAX or PyTorch expose to user code today, delivered from an ahead-of-time native compiler rather than a runtime tracer.

The robustness work matters for a different reason: a differentiable systems language is only useful if the programs built on it can run unattended. Proper tail calls, bounded memory in long-running loops, and a shutdown path that doesn't race are the difference between a research demo and something you can leave running.

## Get started

```bash
brew tap tsotchke/eshkol && brew install eshkol
```

Or build from source (LLVM 21+, C++20, CMake 3.14+) — see [README.md](README.md#quick-start) and [`docs/tutorials/00_FIRST_5_MINUTES.md`](docs/tutorials/00_FIRST_5_MINUTES.md).

Try the AD example above yourself:

```bash
eshkol-run -r your_file.esk
```

- **[Automatic Differentiation guide](docs/guide/AUTOMATIC_DIFFERENTIATION.md)** — the full P0–P12 walkthrough, every example verified against a real build (PR #180).
- **[Eshkol Language Guide](ESHKOL_LANGUAGE_GUIDE.md)** — tutorial introduction to the language.
- **[CHANGELOG.md](CHANGELOG.md)** — itemized engineering detail, phase by phase.
- **[RELEASE_NOTES.md](RELEASE_NOTES.md)** — the user-facing release summary, with the full gate matrix.

## Under the hood

The Taylor tower represents a function's local behavior as a truncated power series, `f(x0 + t) = Σ c_k · t^k`, so `f⁽ⁿ⁾(x0) = n! · c_n`. Each primitive operator (`+ - * / exp log sin cos sqrt tan atan tanh pow ...`) has a closed recurrence for producing its output series' coefficients from its input series' coefficients — for example, Cauchy convolution for multiplication (`s_k = Σ_{j=0..k} u_j · w_{k-j}`) and a linear recurrence for `exp`/`log`/`sin`/`cos`. Composing these recurrences through a program the way the LLVM backend already composes arithmetic gives arbitrary-order differentiation at `O(k²)` cost, with **zero heap allocation** on the common path: when the order `k` is a literal at the call site — the overwhelmingly common case in a compiler — the entire tower is unrolled into stack-allocated, branch-free SSA IR at compile time. A runtime heap-allocated tower (`HEAP_SUBTYPE_TAYLOR`) is the correctness fallback for a dynamically chosen order. Every recurrence in the shared table (`taylor_recurrences.def`) auto-generates its own per-primitive analytic correctness gate, so adding a new differentiable primitive adds its own test.

---

*Eshkol v1.3.0-evolve. MIT License. [github.com/tsotchke/eshkol](https://github.com/tsotchke/eshkol)*
