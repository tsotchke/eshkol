# Eshkol v1.3.3-evolve — exact AD gradients for first-class losses, and the region-evacuator closeout

**Eshkol v1.3.3-evolve corrects an overstated automatic-differentiation claim from v1.3.2-evolve with the real fix — exact gradients through the `input2` path for first-class losses and vector/learnable gamma — and completes the region-escape evacuator series (ESH-0214) by covering the last heap subtype, `PROMISE`.**

Eshkol is Scheme (R7RS) compiled to native code via LLVM, with automatic differentiation as a compiler primitive rather than a library. v1.3.2-evolve's CHANGELOG entry for issue #212 claimed the `input2` gradient path — the second operand of `conv2d`/`batchnorm`/`layernorm`/`attention` (kernel / gamma / K / V) — was complete. An adversarial audit run during this cycle found that claim was wrong: the v1.3.2 change landed a test and a roadmap update, but no gradient code. v1.3.3-evolve ships the actual fix, corrects the record in CHANGELOG.md, and — separately — finishes the region-evacuator hardening that v1.3.1-evolve and v1.3.2-evolve had been closing out one heap subtype at a time.

Full engineering detail lives in [CHANGELOG.md](CHANGELOG.md); the user-facing summary is [RELEASE_NOTES.md](RELEASE_NOTES.md); the complete AD walkthrough is the [Automatic Differentiation guide](docs/guide/AUTOMATIC_DIFFERENTIATION.md). This document is the release announcement — quotable, and every claim below is grounded in shipped code and a real, run command.

---

## What's new since v1.3.1-evolve

### v1.3.3-evolve: exact AD gradients, and the region-evacuator series closes out

- **Exact tensor AD gradients for first-class losses and vector/learnable gamma; silent-zero backward paths now error instead of returning zero** (#229). The real `input2` fix, in three parts: (1) a loss with no compile-time `Function*` — a first-class/higher-order loss — fell to the forward-mode-dual closure path, which loses the tangent for tensor ops and silently returned a zero gradient; a reverse-mode tensor path was added to the closure branch of `AutodiffCodegen::gradient` to close that. (2) Batch-norm/layer-norm now wire per-feature gamma/beta as individual AD nodes instead of a single scalar, so vector/learnable gamma differentiates correctly rather than only a scalar approximation. (3) Any remaining unsupported tensor-op backward path now raises an explicit unsupported-op error instead of silently returning zero, honoring exact-AD-or-error rather than exact-AD-or-silently-wrong. Finite-difference-verified exact in both literal and first-class forms across matmul/conv2d/attention-K-V/vector-gamma; autodiff suite 54/54, the `input2` gradient gate 24/24 under both JIT and AOT.
- **Region escape evacuator now covers the `PROMISE` heap subtype** (ESH-0214e, #230). Adversarial-audit follow-up to ESH-0214d: `PROMISE` was left a shallow-copied leaf despite carrying interior pointers (a thunk and a cached value). A `delay`/`make-promise` created inside `with-region` that escaped outward dangled after `region_pop`, observable as a segfault or `car: not a pair` under `ESHKOL_ARENA_POISON=1` when the promise was later forced. Fixed by adding an evacuation case that walks both interior slots; verified flat at ~116 MB under poison over escape-then-force for both `delay` and `make-promise`. This closes out the ESH-0214 region-evacuator series (ESH-0214a through e) that started with named-let/`define`-loop arena reclamation in v1.3.1-evolve.
- **ICC release-oracle hardening** (#232): the completion oracle now checks region-evacuator poison coverage and the corrected `input2` gradient gate, so both fixes above are load-bearing release gates going forward, not one-off verifications.
- **Subprocess `process-wait` kqueue lost-wakeup race — documented.** The fix itself shipped in v1.3.2-evolve (commit `8443ddae`) but was never recorded in the CHANGELOG until now: on macOS, a child that exited before its `kevent` `EVFILT_PROC` filter was registered could make `process-wait` block for the full timeout and misreport a dead process as still running. The fix probes once with `waitpid(WNOHANG)` right after registering the filter (and again on the timeout branch) so an already-dead child is reaped immediately instead of falling through to a spurious timeout.

### v1.3.2-evolve: thread-safe regions, deeper evacuation, and new tooling

- **Region escape evacuator now covers logic and workspace subtypes** (ESH-0214d, #226): `SUBSTITUTION`, `FACT`, `KNOWLEDGE_BASE`, `FACTOR_GRAPH`, and `WORKSPACE` — the state a resident tick loop mutates via the neuro-symbolic stack — are now deep-walked on region escape instead of shallow-copied, so a `with-region`-wrapped tick loop can reclaim transient garbage per iteration while its escaping knowledge-base/workspace state is promoted intact. `arena_destroy` is now poisoned under `ESHKOL_ARENA_POISON` so a region use-after-free crashes loudly instead of passing by luck.
- **Thread-safe region scope stack** (#217): `parallel-map`/future callbacks that opened a `with-region` raced on the shared current-arena slot under concurrency; the region hijack moved into the runtime with a parallel-scope guard.
- **`eshkol-doc` — API reference generator** (#213): harvests Doxygen `/** @brief */` comments from `inc/` and `lib/` and generates `docs/api/`.
- **`core.blc` — Binary Lambda Calculus, plus a universal machine** (#218): a pure-Eshkol implementation of John Tromp's Binary Lambda Calculus — De Bruijn-indexed terms, `blc-encode`/`blc-decode`, normal-order `blc-eval` — deepened with `(blc-U)`, Tromp's 232-bit self-interpreter, BLC8 byte I/O, and ASCII lambda diagrams.
- **Three deferred latent bugs triaged**: ESH-0223 (named-let stack overflow at high iteration counts), ESH-0227 (apply-loop SIGBUS), ESH-0228 (`sleep-ms` argument type check) (#215).

---

## The foundation this release runs on: v1.3.1-evolve and v1.3.0-evolve

### Flat memory for resident and daemon workloads (v1.3.1-evolve)

The per-iteration arena-reclamation work that the region-evacuator series above builds on started here: self-tail-recursive `define` loops, including a catch-all guard body, gained the same automatic per-iteration arena-scope reclamation that named-let loops already had (ESH-0214b) — verified on a 1,000,000-iteration loop, RSS goes from 1,369 MB (unbounded growth) to 224 MB (flat) — and the S-expression reader (`read_list`) was rewritten from one native stack frame per list element to an iterative loop, so reading back a very large persisted data structure no longer risks a native-stack overflow (ESH-0191). v1.3.1-evolve also shipped a comprehensive Doxygen documentation pass across every public embedding header and most implementation files, plus a navigable per-subsystem documentation index.

### Arbitrary-order automatic differentiation (v1.3.0-evolve)

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

A cluster of fixes from v1.3.0-evolve targets the class of bug that only shows up in a program that runs for a long time or recurses deeply, not in a quick test — the same class the arena-reclamation, reader, and region-evacuator fixes since have continued to close:

- **Proper mutual tail calls** (ESH-0102): a call in tail position to another function is emitted as an LLVM `musttail` call, so mutually tail-recursive functions (`even?`/`odd?`-style state machines) run in O(1) stack instead of overflowing after ~300k hops.
- **Safe teardown** (ESH-0216): `eshkol_runtime_shutdown()` stops and joins the global parallel thread pool before running shutdown hooks, closing a use-after-free race that could `SIGSEGV` well after a graceful `SIGTERM` was already logged; AOT-compiled binaries now also emit the paired runtime shutdown call they were previously skipping entirely.

Full root-cause detail for each: [CHANGELOG.md](CHANGELOG.md).

### A hardened, permanent adversarial-testing program

v1.3.0-evolve shipped the testing infrastructure that found and closed the gaps above — and that this release's adversarial audit used to catch the v1.3.2-evolve `input2` overstatement — wired permanently into the ICC release oracle rather than run once and discarded: a multi-path differential harness with a seeded fuzzer, a feature-pair edge matrix, an AD finite-difference oracle, a stress harness with RSS/time budgets, a VM-parity ratchet, depth-parametric sweeps, and the external reference-Scheme differential oracle described above. See [`docs/TESTING.md`](docs/TESTING.md).

---

## Why it matters

Differentiable programming today mostly means "trace a Python function with a library and hope the trace is faithful." Eshkol takes the opposite bet: make the derivative operator part of the language the compiler already understands, so it composes with closures, recursion, control flow, and the numeric tower the same way `+` does — and, since v1.3.0-evolve, at any order, exactly, with a provable error bound when you need one. v1.3.3-evolve's correction matters for the same reason: "exact AD" is a claim that has to hold for every path through the compiler, including the first-class-loss path that v1.3.2-evolve missed, or it isn't exact AD at all. That is a meaningfully larger AD surface than what JAX or PyTorch expose to user code today, delivered from an ahead-of-time native compiler rather than a runtime tracer.

The region-evacuator work matters for a different, complementary reason: a differentiable systems language is only useful if the programs built on it can run unattended. Proper tail calls, flat memory in long-running loops, a reader that doesn't overflow the stack on large persisted state, a shutdown path that doesn't race, and — as of this release — a region evacuator that correctly promotes every heap subtype a program can allocate (including a `delay`d promise) across a `with-region` boundary are the difference between a research demo and something you can leave running as a daemon.

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
- **[CHANGELOG.md](CHANGELOG.md)** — itemized engineering detail, release by release.
- **[RELEASE_NOTES.md](RELEASE_NOTES.md)** — the user-facing release summary, with the full gate matrix.

## Under the hood

The Taylor tower represents a function's local behavior as a truncated power series, `f(x0 + t) = Σ c_k · t^k`, so `f⁽ⁿ⁾(x0) = n! · c_n`. Each primitive operator (`+ - * / exp log sin cos sqrt tan atan tanh pow ...`) has a closed recurrence for producing its output series' coefficients from its input series' coefficients — for example, Cauchy convolution for multiplication (`s_k = Σ_{j=0..k} u_j · w_{k-j}`) and a linear recurrence for `exp`/`log`/`sin`/`cos`. Composing these recurrences through a program the way the LLVM backend already composes arithmetic gives arbitrary-order differentiation at `O(k²)` cost, with **zero heap allocation** on the common path: when the order `k` is a literal at the call site — the overwhelmingly common case in a compiler — the entire tower is unrolled into stack-allocated, branch-free SSA IR at compile time. A runtime heap-allocated tower (`HEAP_SUBTYPE_TAYLOR`) is the correctness fallback for a dynamically chosen order.

The region-evacuator series (ESH-0214a through e) uses a complementary style of static-plus-runtime safety: a self-tail-recursive loop body (named-let or plain `define`) is proven safe to reclaim per iteration by conservatively checking that nothing escapes the loop's arena scope across the back-edge; when something *does* escape a `with-region` boundary — a cons cell, a knowledge base, a factor graph, a workspace, or, as of v1.3.3-evolve, a promise — the evacuator deep-walks its interior pointers and promotes them into the surviving arena rather than leaving them dangling. `ESHKOL_ARENA_POISON=1` turns any remaining gap in that coverage into an immediate, loud crash instead of a silent corruption.

---

*Eshkol v1.3.3-evolve. MIT License. [github.com/tsotchke/eshkol](https://github.com/tsotchke/eshkol)*
