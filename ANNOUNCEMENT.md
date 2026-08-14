# Eshkol v1.3.4-evolve — resident correctness, and a wrong answer that can no longer look like a right one

**Run an Eshkol daemon for 100,000 ticks and its memory stays flat. That is the first half of v1.3.4-evolve in one line: every defect that long-duration resident workloads surfaced, fixed at the architectural root. A loop that mutates persistent state every tick now reclaims memory on its own, matching explicit `with-region` byte for byte. `parallel-map` is race-free for closures that allocate and return collections. Gradients are exact through every callable form — indirect, wrapped, curried — with no finite-difference fallback in the gradient path. Printed floats round-trip to the identical `double` (R7RS 6.2.6). The strict type checker finally accepts idiomatic dynamic-but-validated code, through a checked `(the <type> expr)` ascription and predicate-guarded narrowing. High-precision numerics land too: Ozaki-II exact and reduced-precision GEMM tiers, a mixed-precision linear solver, and a native 128-bit integer type. Moonlab is pinned to v1.2.0 for quantum natural gradients, hosted-VM tensor-matmul parity is complete, and a linear `Qubit` type turns no-cloning into a type error.**

**The second half is a consumer-hardening correctness wave, and it has one organising principle: a wrong answer must not be able to look like a right one. The change that made the rest findable is that an emitted compile-time error now prevents artifact emission and execution — the compiler used to diagnose a program and then build and run it anyway. That turned a family of silent wrong answers into build failures, and the wave that followed fixed them at the root: exactness is now decided from an operand's runtime tag rather than from a result's value shape, on both the native flonum integer-division family and the bytecode VM's whole numeric surface; differentiation answers exactly at exact points, survives per-iteration nursery reclamation, and no longer returns zeros above gradient arity 16; `define-library` and `import` resolve same-unit libraries on all three back ends; and `--shared-lib` links a real, C-ABI-correct shared library instead of exiting zero with no artifact. Alongside those corrections the release adds a portable event loop (kqueue / epoll / IOCP), a fixed-point and `i128` exact-accumulation engine, the qLLM bridge implementation its documented backward rules had been waiting for, embedding and Fréchet-mean backward passes, and a release gate that finally reads CTest results as evidence.**

Eshkol is R7RS Scheme compiled to native code through LLVM 21. Automatic differentiation is a compiler primitive, not a tracing library. The line between a research demo and a daemon you can walk away from is correctness that survives long-duration, resident, and parallel workloads. This release is that line.

- **Automatic memory reclamation matches explicit regions.** This closes the ESH-0214 memory-management series. Take a resident tick loop that mutates persistent state every iteration — a knowledge base, a workspace, a growing list. It used to reclaim nothing at all, because the static escape analysis rejected any loop body containing a persistent mutation. One iteration's transient garbage stayed forever: about 3,366 bytes per tick, roughly 355 MB over 100,000 ticks. The compiler now lowers that loop with a per-loop nursery region, reusing the same deep-transitive escape-promotion path `with-region` already uses. Each iteration allocates into the nursery. The existing structural write barriers promote any persistent-mutation escapee out at the store. Each tail-call back edge promotes the loop-carried out-values and resets the nursery. The loop exit escapes the result and tears the nursery down. It is a generational minor collection made deterministic, with no tracing pause. That same tick loop now sits flat at 34 MB, byte-for-byte identical to its explicit `with-region` twin. `with-region` is still there for scratch regions. You just no longer need it to get flat RSS.
- **Race-free `parallel-map` for collection-valued closures.** Map a closure whose body uses per-iteration scope reclamation — an internal named-let loop, or a builtin such as `memv`. Cross the parallel threshold and it could hand back dangling or overlapping structure. On a pool worker over the shared thread-safe arena, scope reclamation now degrades to commit-only: allocations are retained, and the shared scope stack is never rewound. Results match serial `map`. ThreadSanitizer reports zero arena data races where it previously reported dozens.
- **Exact gradients through every callable form.** Reach `(gradient f point)` through a function parameter or a wrapper, or curry it as `((gradient f) point)`. Both now come back byte-identical to the direct call, for scalar multi-argument, vector, and non-polynomial losses. The operator recovers the callable's arity from its closure metadata instead of assuming a single tensor argument, so a multi-parameter scalar loss no longer gets misdispatched. Nothing in the gradient path falls back to finite differences. Every form is exact reverse-mode AD.
- **Round-tripping floats.** `display`, `write`, and `number->string` emit the shortest decimal that reads back as the identical `double` (R7RS 6.2.6). Integral doubles keep the no-`.0` form. The native compiler and the bytecode VM share one portable-C routine, so their output is byte-identical. `(sqrt 2.0)` prints `1.4142135623730951`.
- **Strict-mode types that accept idiomatic code.** A checked `(the <type> expr)` ascription asserts a type to the checker as a trusted assertion, and costs nothing at runtime. Predicate-guarded narrowing teaches the checker that a value tested by one of eight type predicates is that type inside the guarded branch. Narrowing is honored across `if` and `and`, and cancelled at `set!`. Sum-type annotations are honored on named-let parameters. A numeric-tower join gives recursive accumulators their least-upper-bound numeric type. A linear `Qubit` type enforces use-exactly-once on declared linear parameters, which turns the no-cloning guarantee into a type error rather than a convention.
- **High-precision numerics.** Ozaki-II recovers full-f64 `C = A*B` out of reduced-precision tensor cores. It is certified against an independent CPU f64 reference, and against the new native 128-bit integer path. A CUDA INT8 tier and a Metal reduced-precision fully-GPU fast tier are both opt-in and default off. `linear-solve` is a full-f64 dense solver with a mixed-precision iterative-refinement fast path. That path certifies a full-f64 residual, and drops back to a plain-f64 LU when it cannot.
- **Quantum and VM parity.** Moonlab is pinned to v1.2.0. The pin brings `vqe_compute_qgt` (quantum geometric tensor / quantum natural gradient) and a smooth first-principles H2/LiH potential-energy surface. The H2 equilibrium oracle at 0.735 Å is updated to `-1.142200155381` Ha. Hosted-VM tensor matmul parity is complete: `arange` in 1/2/3-argument forms, nested-literal tensor operands, and multi-dimensional `tensor-ref`/`tensor-set!`.

## The correctness wave

The first half of the cycle fixed what resident workloads surfaced. The second half fixed what *nobody* had surfaced, because the compiler had been hiding it.

- **A diagnosed program no longer builds and runs.** The compiler used to print `ERROR: …` and then emit, link and execute a binary anyway, so a diagnosed program produced a wrong answer instead of a failed build. This is the mechanism that kept the rest of this release's defects quiet: each of them was reported at compile time, and every report was ignored. Reporting an error is one call at any of 805 sites; propagating one is a return path through every enclosing frame. All 805 sites now funnel through four logging primitives, so an authoritative error state lives in one place and every path contributes to it. Downstream, this converted a family of silent wrong answers into build failures — including several fixed in this release, which is how they were found.
- **Exactness is decided by an operand's tag, not by a result's shape.** Native flonum `modulo`, `remainder`, `quotient` and the floor-division family were wrong across the board: the double path converted through `int64` and packed the result *exact*, so `(quotient 7.0 2.0)` reported `exact? #t`; a `1e20` clamp constant leaked into results and division by `0.0` returned the clamp. Independently, the bytecode VM decided a numeric result's exactness from its value rather than from its operands' tags, so inexact arithmetic silently became exact across the VM's whole numeric surface. Both are fixed at the root, and a zero divisor now raises uniformly across `quotient` / `remainder` / `modulo` on every route.
- **Differentiation is exact at exact points, and survives automatic reclamation.** At a rational or bignum point, `derivative`, `gradient` and `hessian` returned garbage, `#()`, or a SIGSEGV. They now route through the same Taylor-tower pass as `derivative-n`, with identity in value *and* exactness: `(derivative f x)` equals `(derivative-n f x 1)`, `(hessian f x)` equals `(derivative-n f x 2)`, and the tier demotes on the first transcendental. Separately, a loop that filled a vector with `vector-set!` produced a silently corrupted gradient: the per-iteration nursery recycled a dual number the write barrier failed to recognise, so the primal survived and the tangent was quietly wrong. One shared "carries an arena pointer" predicate now governs the write barrier, the `with-region` escape and the nursery recycle. And a gradient of a runtime closure with declared arity 17–32 crashed, returned all zeros, or raised a type error — the declared ceiling was 32 but an internal clamp of 16 made everything above it undefined.
- **Modules, and a shared library that is actually a shared library.** `define-library` validated its library name and threw it away, so an `import` could not see a library defined in the same file. Fixed on all three back ends — and the VM lane was the real gap, since it knew none of `define-library` / `import` / `export`; three latent VM defects were fixed with it. `op:IMPORT`, `op:PROVIDE` and `op:REQUIRE` move to `vm-supported` with no new waivers. `--shared-lib` never linked a shared library and exited 0 anyway; it now produces a real, C-ABI-correct one. Module and `(load …)` resolution ran from four private copies of the same resolver on the JIT, AOT and import paths; they are unified behind one implementation whose search order is the documented one.
- **New capability alongside the corrections.** A portable event loop over kqueue / epoll / IOCP, fail-closed on WASM. A fixed-point and `i128` exact-accumulation engine with order-independent, bit-exact reductions. The qLLM bridge implementation its documented backward rules had been waiting for, plus embedding and Fréchet-mean backward passes with gradient checks. User-reachable region handles — `region-open` / `region-close` / `region-open?` — for loop shapes where a lexical block is awkward, with a generation-counted handle so every stale token is detectably stale.
- **The release gate reads CTest.** No completion-oracle criterion consumed a CTest result at all, so a red CTest run could not turn the release gate red. CTest now emits per-test, per-group and whole-suite trace events, and a group whose regex matches no configured test is reported ABSENT and **fails** the gate — a pillar cannot quietly stop being covered because its tests were renamed or configured out.

## Evidence

- **Every declared executable surface has evidence, and the evidence is execution-backed.** The language-surface manifest is 1,091/1,091. Every one of the 1,091 declared constructs earned its row by running: it dispatched or executed in a passing run. Lexical name-presence no longer counts — it is a diagnostic now, and earns no release credit. The surface was honestly re-baselined upward across the cycle, from 1,056, as the `i128` tower, `linear-solve`, the string/pointer conversions, the portable event loop and the region-handle surface landed as new core builtins. The coverage floor and deficit ledger are ratcheted to 1,091, so nobody can walk the number back quietly.
- **The gates, measured on the release cut.** The aggregate suite is 45/45 suites and 770 individual tests. CTest is 180/181, and as of this release it is completion-oracle evidence rather than advice. The SICP full-book gate is 88/88 probes across all five chapters under both `-r` and AOT. The reference-Scheme differential oracle is 34/34 AGREE against chibi-scheme 0.12.0. The VM parity differential is 184/184. The qLLM oracle gate is 10/10. ICC readiness reports 100, verdict `ready`.
- **VM results are explicit, not overstated.** The parity manifest is 951 rows: 578 `vm-supported`, 44 `native-only-justified`, and 329 `gap`. Of those gaps, 17 are verified behavioral divergences with reproducible programs. `op:GRADIENT` and `op:DERIVATIVE` move to `vm-supported` this release, so `gradient` now runs on the bytecode VM byte-identically to native, and the R7RS module forms follow. Higher-order nesting stays native-only, and region *reclamation* stays native-only even though the region surface, its validation and its error messages are byte-identical on both substrates. This is not a claim of complete backend parity. `PARITY.tsv` separately ratchets intentionally native-only facilities and known VM gaps, so unsupported behavior cannot masquerade as success.
- **Release packages are portable and honestly labelled.** The release payload is 15 platform packages plus `SHA256SUMS.txt`: 16 published files. Linux and Windows x64/ARM64, plus macOS x64/ARM64, cover the Lite and XLA tiers. Real CUDA packages cover Linux x64/ARM64 and Windows x64. Windows ARM64 CUDA is not advertised, because NVIDIA does not provide the required supported toolkit.

Full engineering detail lives in [CHANGELOG.md](CHANGELOG.md). The user-facing summary and gate matrix are in [RELEASE_NOTES.md](RELEASE_NOTES.md). The complete AD walkthrough is the [Automatic Differentiation guide](docs/guide/AUTOMATIC_DIFFERENTIATION.md), and the memory model is documented in [docs/reference/runtime/memory-model.md](docs/reference/runtime/memory-model.md). Known limitations, including the ones this release's own hardening surfaced, are in [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md). Every claim above is tied to shipped code and an executable gate.

---

# Eshkol v1.3.3-evolve — quantum circuits you can differentiate, post-quantum crypto, and incremental dataflow

**Eshkol v1.3.3-evolve introduces an opt-in quantum computing stack: state-vector simulation, VQE, a CHSH Bell-inequality gate, Bell-verified quantum randomness, and ML-KEM (FIPS 203) post-quantum cryptography. VQE gradients flow through Eshkol's native automatic differentiation. The same release brings real `make-parameter`/`parameterize` dynamic parameters, the `core.dbsp` incremental-dataflow module, bignum-capable exact rationals, one-pass reverse-mode gradients, and a silent-wrong-answer correctness campaign driven by two new generative exposure engines.**

Eshkol is R7RS Scheme compiled to native code through LLVM 21. Automatic differentiation is a compiler primitive, not a tracing library. v1.3.3-evolve points that thesis somewhere new: a quantum circuit is just another differentiable computation. Build with `-DESHKOL_QUANTUM_ENABLED=ON`. Then `(vqe-energy ...)` composes with `gradient` and ordinary optimizer code exactly like `matmul` does. That call evaluates a variational quantum eigensolver energy on the Moonlab state-vector simulator, and new custom-VJP AD tape nodes carry Moonlab's exact adjoint gradient back through it. The same release also corrects an overstated AD claim from v1.3.2-evolve with the real fix: exact `input2` gradients for first-class losses and vector/learnable gamma. It closes the region-escape evacuator series (ESH-0214).

Full engineering detail lives in [CHANGELOG.md](CHANGELOG.md). The user-facing summary and gate matrix are in [RELEASE_NOTES.md](RELEASE_NOTES.md), and the complete AD walkthrough is the [Automatic Differentiation guide](docs/guide/AUTOMATIC_DIFFERENTIATION.md). Every claim below is tied to shipped code and an executable gate.

---

## What's new since v1.3.1-evolve

### v1.3.3-evolve: quantum computing, post-quantum crypto, incremental dataflow

- **Quantum circuit simulation, opt-in** (`-DESHKOL_QUANTUM_ENABLED=ON`, #261/#268/#273). The `agent.quantum` module binds the [Moonlab](https://github.com/tsotchke/moonlab) state-vector core: state creation/teardown, Hadamard/Pauli/CNOT/rotation gates, `measure`, `expectation-z`, molecular Hamiltonians (H2/LiH/H2O), and a `with-quantum-state` auto-destroy helper. Two acceptance gates prove this is genuine quantum behavior, not a classical imitation. The Bell-pair smoke gate measures one qubit of an entangled pair, and it always matches the other: 200/200 shots. The permanent 16K-shot CHSH Bell-inequality gate (`bell-chsh`) measures S ≈ 2.86, past the classical bound of 2, and demands 2.4 < S ≤ 2.95 on every run.
- **VQE, differentiable through Eshkol AD** (#268/#270). Variational-quantum-eigensolver builtins — H2/LiH/H2O Hamiltonians, exact and variational energy, gradients, optimization — with the H2 ground energy matching exact diagonalization to `4.4e-16`. New **custom-VJP AD tape nodes** (`AD_NODE_CUSTOM`) bridge Moonlab's exact reverse-mode adjoint gradient into Eshkol's tape. So `(gradient (lambda (params) (vqe-energy ...)) params)` just works. The release gate requires that custom-VJP adjoint to match Moonlab's native adjoint to within `1e-8`, and a central finite difference to within `1e-4`. A VQE probe now sits inside the AD adversarial oracle permanently.
- **ML-KEM (FIPS 203) post-quantum cryptography** (#272). `agent.pqc` provides `mlkem-keygen`/`mlkem-encaps`/`mlkem-decaps` at the 512/768/1024 security levels over R7RS bytevectors, with QRNG-seeded key generation and encapsulation, verified against NIST KAT fingerprints.
- **Honest quantum randomness** (#261/#262). `quantum-random`/`quantum-random-int`/`quantum-random-range` now draw from Moonlab's Bell-verified QRNG when quantum is enabled — and are honestly labeled as a classical fallback when it is not (`eshkol_qrng_source_label()` reports which source is live). The VM and native backends now share one generator, and `quantum-random-int` honors its bound on every path (it previously returned raw 64-bit draws on the LLVM path).
- **Real dynamic parameters** (#271, #267). `make-parameter`/`parameterize` are wired to genuine runtime parameter objects — converters, a proper dynamic binding stack, correct unwinding, and region write barriers — on both the native and VM execution paths.
- **`core.dbsp` — incremental dataflow** (#233). Z-sets (weighted multisets) as a commutative group, and the `z^-1`/`D`/`I` stream operators, where D and I are mutual inverses. Incremental relational operators: linear map/filter/project/union, the three-term incremental join, and a multiplicity-correct `distinct`. On top of them, the generic incrementalizer `Q^Δ = D ∘ lift(Q) ∘ I`. All of it is pure Eshkol, with zero compiler changes. This is the first shipped slice of the incremental-dataflow spine (ADR 0009), and its acceptance gate is 27/27 under JIT and AOT.
- **Exact rationals at any magnitude** (#247). The exact rational type is now a canonical discriminated union: a zero-allocation int64 fast path, with a bignum numerator/denominator path taken only on overflow. Overflowing an int64 numerator or denominator no longer degrades silently to double, and bignum-magnitude rational literals parse. `(/ 1 (expt 10 19))` stays exact, and is verified byte-identical to Python's `Fraction` on reference workloads.
- **One-pass reverse gradients + AD introspection** (#234). Staged-kernel Phase A collapses the per-component gradient replay into one primal pass plus one reverse pass, reading every input gradient off the tape. N primal calls become 1, checked at N=4 and N=64. New `(ad-counters)` and `(ad-primal-calls)` builtins let you see it happen. Persisted AOT and shared-lib artifacts now default to O2, a ~1.7x/41% speedup verified on float-heavy code.
- **A silent-wrong-answer correctness campaign**, driven by two new generative exposure engines wired permanently into the ICC release oracle. The first is a multi-oracle differential harness: generated R7RS programs, cross-checked against chibi-scheme, JIT, AOT at O0/O2, and the bytecode VM. The second is an AD-vs-finite-difference adversarial oracle, running 147 probes and 436 component checks across 21 generated files under JIT and AOT. A zero AD gradient where FD is nonzero is a hard failure. Together the two engines drove root fixes for:
    - every known silent-zero AD path — Jacobian/Hessian through inner forward-mode derivatives, gradients at `(vector ...)`-constructed points, and Hessians through tensor ops and through `reshape`/2-D `tensor-matmul`;
    - real bignum-aware VM arithmetic and comparisons;
    - TCO for `cond`/`case`/`when`/`unless`/`and`/`or` in tail position. This previously hit SIGBUS around 2M iterations unless you wrote it with `if`; it is now verified to 2,000,000 iterations under both JIT and AOT;
    - numeric-tower exactness — `gcd`, bignum division identities, and structural `eqv?`/`equal?` on rationals and complex numbers;
    - a stable, tail-recursive, memory-bounded `sort`/`filter`: 2M elements at ~362 MiB peak instead of ~32 GB;
    - and a 26x `--wasm` size regression — 5.57 MB → 60 KB, 635 → 21 functions, dead-stripped, with a CI size gate.

### v1.3.3-evolve also closes out the AD-exactness correction and the evacuator series

- **Exact tensor AD gradients for first-class losses and vector/learnable gamma; silent-zero backward paths now error instead of returning zero** (#229). The real `input2` fix, in three parts: (1) a loss with no compile-time `Function*` — a first-class/higher-order loss — fell to the forward-mode-dual closure path, which loses the tangent for tensor ops and silently returned a zero gradient; a reverse-mode tensor path was added to the closure branch of `AutodiffCodegen::gradient` to close that. (2) Batch-norm/layer-norm now wire per-feature gamma/beta as individual AD nodes instead of a single scalar, so vector/learnable gamma differentiates correctly rather than only a scalar approximation. (3) Any remaining unsupported tensor-op backward path now raises an explicit unsupported-op error instead of silently returning zero, honoring exact-AD-or-error rather than exact-AD-or-silently-wrong. Finite-difference-verified exact in both literal and first-class forms across matmul/conv2d/attention-K-V/vector-gamma; autodiff suite 54/54, the `input2` gradient gate 24/24 under both JIT and AOT.
- **Region escape evacuator now covers the `PROMISE` heap subtype** (ESH-0214e, #230). This is an adversarial-audit follow-up to ESH-0214d. `PROMISE` was left a shallow-copied leaf, despite carrying two interior pointers: a thunk and a cached value. Create a `delay` or `make-promise` inside `with-region`, let it escape outward, and it dangled after `region_pop`. Force it later and you saw a segfault, or `car: not a pair` under `ESHKOL_ARENA_POISON=1`. The fix adds an evacuation case that walks both interior slots. Escape-then-force is now verified flat at ~116 MB under poison, for both `delay` and `make-promise`. This closes out the ESH-0214 region-evacuator series (ESH-0214a through e), which started with named-let and `define`-loop arena reclamation in v1.3.1-evolve.
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

The per-iteration arena-reclamation work that the region-evacuator series builds on started here. Self-tail-recursive `define` loops, including a catch-all guard body, gained the same automatic per-iteration arena-scope reclamation that named-let loops already had (ESH-0214b). On a 1,000,000-iteration loop, RSS drops from 1,369 MB of unbounded growth to a flat 224 MB. The S-expression reader (`read_list`) was rewritten too, from one native stack frame per list element to an iterative loop, so reading back a very large persisted data structure no longer risks a native-stack overflow (ESH-0191). v1.3.1-evolve also shipped a comprehensive Doxygen documentation pass across every public embedding header and most implementation files, plus a navigable per-subsystem documentation index.

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

A reference-Scheme differential oracle runs the same portable R7RS-small program on Eshkol and on **chibi-scheme 0.12.0** — the strictest mainstream R7RS-small implementation — and diffs the output. It started the v1.3.0-evolve cycle at 27/34 (79.4%) programs agreeing. Every divergence it found was fixed: `apply` with leading arguments before the final list, multi-vector `vector-map`/`vector-for-each`, quasiquoted vector literals, `cond`/`case` `=>` arrow clauses, an allocating `vector-copy`, and more. Current state, unchanged and re-verified for this release:

```
Total         : 34
AGREE         : 34
ESHKOL-DIVERGES : 0
Agreement rate: 100.0%
Gate          : PASS  (PASS iff every program AGREES)
```

(`scripts/run_reference_differential.sh`, corpus in `tests/reference-diff/corpus/`.) Separately, Eshkol implements roughly 95% of the broader R7RS-small procedure surface (232 of 244 procedures), covering the full numeric tower, continuations, exceptions, promises, `eval`, records, bytevectors, and hygienic macros.

### Robustness: tail calls and shutdown

Some bugs only show up in a program that runs for a long time, or recurses deeply. They never appear in a quick test. A cluster of fixes from v1.3.0-evolve targets exactly that class — the same one the arena-reclamation, reader, and region-evacuator fixes have kept closing since:

- **Proper mutual tail calls** (ESH-0102): a call in tail position to another function is emitted as an LLVM `musttail` call, so mutually tail-recursive functions (`even?`/`odd?`-style state machines) run in O(1) stack instead of overflowing after ~300k hops.
- **Safe teardown** (ESH-0216): `eshkol_runtime_shutdown()` stops and joins the global parallel thread pool before running shutdown hooks, closing a use-after-free race that could `SIGSEGV` well after a graceful `SIGTERM` was already logged; AOT-compiled binaries now also emit the paired runtime shutdown call they were previously skipping entirely.

Full root-cause detail for each: [CHANGELOG.md](CHANGELOG.md).

### A hardened, permanent adversarial-testing program

v1.3.0-evolve shipped the testing infrastructure that found and closed the gaps above. The same infrastructure is what this release's adversarial audit used to catch the v1.3.2-evolve `input2` overstatement. It is wired permanently into the ICC release oracle, not run once and discarded: a multi-path differential harness with a seeded fuzzer, a feature-pair edge matrix, an AD finite-difference oracle, a stress harness with RSS/time budgets, a VM-parity ratchet, depth-parametric sweeps, and the external reference-Scheme differential oracle described above. See [`docs/TESTING.md`](docs/TESTING.md).

---

## Why it matters

Differentiable programming today mostly means "trace a Python function with a library and hope the trace is faithful." Eshkol takes the opposite bet: make the derivative operator part of the language the compiler already understands. Then it composes with closures, recursion, control flow, and the numeric tower the same way `+` does — and, since v1.3.0-evolve, at any order, exactly, with a provable error bound when you need one.

v1.3.3-evolve extends that composition across a hardware boundary. A variational quantum circuit's energy is now just another node on the reverse-mode tape, carrying Moonlab's exact adjoint gradient instead of a finite-difference approximation. That is hybrid quantum-classical optimization written as ordinary `gradient` code, in one language — run against a build whose quantum behavior is checked against a Bell inequality rather than taken on faith. The custom-VJP mechanism behind it is general: any foreign computation with a known adjoint can participate exactly in Eshkol AD.

The release's correction discipline matters for the same reason. "Exact AD" is a claim that has to hold for every path through the compiler — including the first-class-loss path that v1.3.2-evolve missed, and the silent-zero gradient paths the new adversarial oracle flushed out. Otherwise it isn't exact AD at all.

The region-evacuator work matters for a different, complementary reason. A differentiable systems language is only useful if the programs built on it can run unattended. That takes proper tail calls, flat memory in long-running loops, a reader that doesn't overflow the stack on large persisted state, and a shutdown path that doesn't race. As of this release it also takes a region evacuator that correctly promotes every heap subtype a program can allocate — including a `delay`d promise — across a `with-region` boundary. Those are the difference between a research demo and something you can leave running as a daemon.

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

The Taylor tower represents a function's local behavior as a truncated power series, `f(x0 + t) = Σ c_k · t^k`, so `f⁽ⁿ⁾(x0) = n! · c_n`. Each primitive operator (`+ - * / exp log sin cos sqrt tan atan tanh pow ...`) has a closed recurrence for producing its output series' coefficients from its input series' coefficients — for example, Cauchy convolution for multiplication (`s_k = Σ_{j=0..k} u_j · w_{k-j}`) and a linear recurrence for `exp`/`log`/`sin`/`cos`. Compose those recurrences through a program the way the LLVM backend already composes arithmetic, and you get arbitrary-order differentiation at `O(k²)` cost — with **zero heap allocation** on the common path. When the order `k` is a literal at the call site, which is the overwhelmingly common case in a compiler, the entire tower unrolls into stack-allocated, branch-free SSA IR at compile time. A runtime heap-allocated tower (`HEAP_SUBTYPE_TAYLOR`) is the correctness fallback for a dynamically chosen order.

The region-evacuator series (ESH-0214a through e) uses a complementary style of static-plus-runtime safety. A self-tail-recursive loop body, named-let or plain `define`, is proven safe to reclaim per iteration by conservatively checking that nothing escapes the loop's arena scope across the back-edge. When something *does* escape a `with-region` boundary — a cons cell, a knowledge base, a factor graph, a workspace, or, as of v1.3.3-evolve, a promise — the evacuator deep-walks its interior pointers and promotes them into the surviving arena rather than leaving them dangling. `ESHKOL_ARENA_POISON=1` turns any remaining gap in that coverage into an immediate, loud crash instead of a silent corruption.

---

*Eshkol v1.3.4-evolve. MIT License. [github.com/tsotchke/eshkol](https://github.com/tsotchke/eshkol)*
