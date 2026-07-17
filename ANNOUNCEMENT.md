# Eshkol v1.3.3-evolve — quantum circuits you can differentiate, post-quantum crypto, and incremental dataflow

**Eshkol v1.3.3-evolve introduces an opt-in quantum computing stack — state-vector simulation, VQE whose gradients flow through Eshkol's native automatic differentiation, a CHSH Bell-inequality gate, Bell-verified quantum randomness, and ML-KEM (FIPS 203) post-quantum cryptography — alongside real `make-parameter`/`parameterize` dynamic parameters, the `core.dbsp` incremental-dataflow module, bignum-capable exact rationals, one-pass reverse-mode gradients, and a silent-wrong-answer correctness campaign driven by two new generative exposure engines.**

Eshkol is R7RS Scheme compiled to native code through LLVM 21, with automatic differentiation as a compiler primitive rather than a tracing library. v1.3.3-evolve extends that thesis in a new direction: a quantum circuit is just another differentiable computation. Build with `-DESHKOL_QUANTUM_ENABLED=ON` and `(vqe-energy ...)` — a variational quantum eigensolver energy evaluated on the Moonlab state-vector simulator — composes with `gradient` and ordinary optimizer code exactly like `matmul` does, via new custom-VJP AD tape nodes carrying Moonlab's exact adjoint gradient. The same release corrects an overstated AD claim from v1.3.2-evolve with the real fix (exact `input2` gradients for first-class losses and vector/learnable gamma) and completes the region-escape evacuator series (ESH-0214).

Full engineering detail lives in [CHANGELOG.md](CHANGELOG.md); the user-facing summary and gate matrix are in [RELEASE_NOTES.md](RELEASE_NOTES.md); the complete AD walkthrough is the [Automatic Differentiation guide](docs/guide/AUTOMATIC_DIFFERENTIATION.md). Every claim below is tied to shipped code and an executable gate.

---

## What's new since v1.3.1-evolve

### v1.3.3-evolve: quantum computing, post-quantum crypto, incremental dataflow

- **Quantum circuit simulation, opt-in** (`-DESHKOL_QUANTUM_ENABLED=ON`, #261/#268/#273). The `agent.quantum` module binds the [Moonlab](https://github.com/tsotchke/moonlab) state-vector core: state creation/teardown, Hadamard/Pauli/CNOT/rotation gates, `measure`, `expectation-z`, molecular Hamiltonians (H2/LiH/H2O), and a `with-quantum-state` auto-destroy helper. Two acceptance gates prove the integration produces genuine quantum behavior rather than a classical imitation: a Bell-pair smoke gate (measuring one qubit of an entangled pair always matches the other — 200/200 shots) and a permanent 16K-shot CHSH Bell-inequality gate (`bell-chsh`) that measures S ≈ 2.86 — beyond the classical bound of 2 — with the gate requiring 2.4 < S ≤ 2.95 on every run.
- **VQE, differentiable through Eshkol AD** (#268/#270). Variational-quantum-eigensolver builtins — H2/LiH/H2O Hamiltonians, exact and variational energy, gradients, optimization — with the H2 ground energy matching exact diagonalization to `4.4e-16`. New **custom-VJP AD tape nodes** (`AD_NODE_CUSTOM`) bridge Moonlab's exact reverse-mode adjoint gradient into Eshkol's tape, so `(gradient (lambda (params) (vqe-energy ...)) params)` just works: the release gate requires the custom-VJP adjoint to match Moonlab's native adjoint to within `1e-8` and a central finite difference to within `1e-4`, and a VQE probe now sits inside the AD adversarial oracle permanently.
- **ML-KEM (FIPS 203) post-quantum cryptography** (#272). `agent.pqc` provides `mlkem-keygen`/`mlkem-encaps`/`mlkem-decaps` at the 512/768/1024 security levels over R7RS bytevectors, with QRNG-seeded key generation and encapsulation, verified against NIST KAT fingerprints.
- **Honest quantum randomness** (#261/#262). `quantum-random`/`quantum-random-int`/`quantum-random-range` now draw from Moonlab's Bell-verified QRNG when quantum is enabled — and are honestly labeled as a classical fallback when it is not (`eshkol_qrng_source_label()` reports which source is live). The VM and native backends now share one generator, and `quantum-random-int` honors its bound on every path (it previously returned raw 64-bit draws on the LLVM path).
- **Real dynamic parameters** (#271, #267). `make-parameter`/`parameterize` are wired to genuine runtime parameter objects — converters, a proper dynamic binding stack, correct unwinding, and region write barriers — on both the native and VM execution paths.
- **`core.dbsp` — incremental dataflow** (#233). Z-sets (weighted multisets) as a commutative group, the `z^-1`/`D`/`I` stream operators (D and I mutual inverses), incremental relational operators — linear map/filter/project/union, the three-term incremental join, multiplicity-correct `distinct` — and the generic incrementalizer `Q^Δ = D ∘ lift(Q) ∘ I`, in pure Eshkol with zero compiler changes. The first shipped slice of the incremental-dataflow spine (ADR 0009); acceptance gate 27/27 under JIT and AOT.
- **Exact rationals at any magnitude** (#247). The exact rational type is now a canonical discriminated union: a zero-allocation int64 fast path, with a bignum numerator/denominator path taken only on overflow — so no silent degradation to double when a numerator or denominator overflows int64, and bignum-magnitude rational literals parse. `(/ 1 (expt 10 19))` stays exact; verified byte-identical to Python's `Fraction` on reference workloads.
- **One-pass reverse gradients + AD introspection** (#234). Staged-kernel Phase A collapses the per-component gradient replay into one primal + one reverse pass reading every input gradient from the tape (verified: N primal calls become 1, checked at N=4 and N=64), with new `(ad-counters)`/`(ad-primal-calls)` introspection builtins; persisted AOT/shared-lib artifacts now default to O2 (~1.7x/41% speedup verified on float-heavy code).
- **A silent-wrong-answer correctness campaign**, driven by two new generative exposure engines wired permanently into the ICC release oracle: a multi-oracle differential harness (generated R7RS programs cross-checked against chibi-scheme, JIT, AOT at O0/O2, and the bytecode VM) and an AD-vs-finite-difference adversarial oracle (147 probes / 436 component checks across 21 generated files under JIT and AOT; a zero AD gradient where FD is nonzero is a hard failure). Together they drove root fixes for: every known silent-zero AD path (Jacobian/Hessian through inner forward-mode derivatives, gradients at `(vector ...)`-constructed points, Hessians through tensor ops and through `reshape`/2-D `tensor-matmul`); real bignum-aware VM arithmetic and comparisons; TCO for `cond`/`case`/`when`/`unless`/`and`/`or` in tail position (previously SIGBUS around 2M iterations unless written with `if`; now verified to 2,000,000 iterations under both JIT and AOT); numeric-tower exactness (`gcd`, bignum division identities, structural `eqv?`/`equal?` on rationals/complex); a stable, tail-recursive, memory-bounded `sort`/`filter` (2M elements at ~362 MiB peak instead of ~32 GB); and a 26x `--wasm` size regression (5.57 MB → 60 KB, 635 → 21 functions, dead-stripped, with a CI size gate).

### v1.3.3-evolve also closes out the AD-exactness correction and the evacuator series

- **Exact tensor AD gradients for first-class losses and vector/learnable gamma; silent-zero backward paths now error instead of returning zero** (#229). The real `input2` fix, in three parts: (1) a loss with no compile-time `Function*` — a first-class/higher-order loss — fell to the forward-mode-dual closure path, which loses the tangent for tensor ops and silently returned a zero gradient; a reverse-mode tensor path was added to the closure branch of `AutodiffCodegen::gradient` to close that. (2) Batch-norm/layer-norm now wire per-feature gamma/beta as individual AD nodes instead of a single scalar, so vector/learnable gamma differentiates correctly rather than only a scalar approximation. (3) Any remaining unsupported tensor-op backward path now raises an explicit unsupported-op error instead of silently returning zero, honoring exact-AD-or-error rather than exact-AD-or-silently-wrong. Finite-difference-verified exact in both literal and first-class forms across matmul/conv2d/attention-K-V/vector-gamma; autodiff suite 54/54, the `input2` gradient gate 24/24 under both JIT and AOT.
- **Region escape evacuator now covers the `PROMISE` heap subtype** (ESH-0214e, #230). Adversarial-audit follow-up to ESH-0214d: `PROMISE` was left a shallow-copied leaf despite carrying interior pointers (a thunk and a cached value). A `delay`/`make-promise` created inside `with-region` that escaped outward dangled after `region_pop`, observable as a segfault or `car: not a pair` under `ESHKOL_ARENA_POISON=1` when the promise was later forced. Fixed by adding an evacuation case that walks both interior slots; verified flat at ~116 MB under poison over escape-then-force for both `delay` and `make-promise`. This closes out the ESH-0214 region-evacuator series (ESH-0214a through e) that started with named-let/`define`-loop arena reclamation in v1.3.1-evolve.
- **ICC release-oracle hardening** (#232): the completion oracle now checks region-evacuator poison coverage and the corrected `input2` gradient gate, and is extended with the two generative exposure engines above as permanent release gates — not one-off verifications.
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

Differentiable programming today mostly means "trace a Python function with a library and hope the trace is faithful." Eshkol takes the opposite bet: make the derivative operator part of the language the compiler already understands, so it composes with closures, recursion, control flow, and the numeric tower the same way `+` does — and, since v1.3.0-evolve, at any order, exactly, with a provable error bound when you need one. v1.3.3-evolve extends that composition across a hardware boundary: a variational quantum circuit's energy is now just another node on the reverse-mode tape, carrying Moonlab's exact adjoint gradient instead of a finite-difference approximation — hybrid quantum-classical optimization written as ordinary `gradient` code, in one language, run against a build whose quantum behavior is checked against a Bell inequality rather than taken on faith. The custom-VJP mechanism that enables it is general: any foreign computation with a known adjoint can participate exactly in Eshkol AD. And the release's correction discipline matters for the same reason: "exact AD" is a claim that has to hold for every path through the compiler — including the first-class-loss path that v1.3.2-evolve missed and the silent-zero gradient paths the new adversarial oracle flushed out — or it isn't exact AD at all.

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
