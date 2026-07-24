# Eshkol — Description Copy

## A compiled Scheme with a constructive proof that a transformer is an interpreter

Eshkol is a compiled programming language for mathematical and cognitive computing.
The repository ships v1.3.4-evolve (July 2026) of the compiler — an arbitrary-order
automatic-differentiation system with exact gradients now verified through the
tensor-op `input2` path and through every callable form (indirect, wrapped, and
curried, with no finite-difference fallback), 100% conformance on a portable R7RS
differential corpus, and a region-escape evacuator that makes long-running/resident
programs safe to leave running across every heap subtype the language can allocate.
v1.3.4-evolve is a resident-correctness release: automatic per-iteration memory
reclamation now matches explicit `with-region` even for loops that mutate persistent
state every tick (closing the ESH-0214 memory-management series); `parallel-map` is
race-free for closures that allocate and return collections; printed floats round-trip
to the identical `double` (R7RS 6.2.6); the strict type checker accepts a checked
`(the <type> expr)` ascription and predicate-guarded narrowing; and the high-precision
numerics wave lands (Ozaki-II exact and reduced-precision GEMM tiers, a mixed-precision
`linear-solve`, and a native 128-bit integer type), alongside a Moonlab v1.2.0 quantum
pin with quantum-natural-gradient support, full hosted-VM tensor-matmul parity, and a
linear `Qubit` type.
The earlier v1.3.3-evolve added an opt-in quantum computing stack — Moonlab state-vector
simulation, a variational quantum eigensolver whose gradients flow through the
language's own automatic differentiation via new custom-VJP tape nodes, a CHSH
Bell-inequality gate, Bell-verified quantum randomness, and ML-KEM (FIPS 203)
post-quantum cryptography — alongside real `make-parameter`/`parameterize`
dynamic parameters, the `core.dbsp` incremental-dataflow module, bignum-capable
exact rationals, and one-pass reverse-mode gradients. The same release ran a
silent-wrong-answer correctness campaign driven by two new generative exposure
engines, closing several bytecode-VM silent-miscompile classes, tail-call gaps
in six special forms, and a 26x `--wasm` size regression. The repository carries
all of this together with the reproducibility artefact for
*The Self-Differentiating Neural Computer: Computable Transformers via
Analytical Weight Construction* (tsotchke, 2026), in which a six-layer
transformer with 12.22 million analytically-constructed parameters executes
a bounded 83-opcode bytecode VM bit-identically. The result is a
constructive, not statistical, demonstration that a fixed-weight transformer
can be an interpreter when its weights are derived from an instruction-set
specification rather than fit by gradient descent.

---

## Lede

Eshkol is an R7RS-compatible Scheme dialect that compiles through LLVM 21 to native
binaries on macOS, Linux, and Windows, and to WebAssembly for browser execution.
The language treats automatic differentiation, arena memory, and a neuro-symbolic
computation layer as compiler primitives rather than library add-ons. Differentiation
is available in symbolic, forward, and reverse modes alongside eight vector-calculus
operators — and, as of v1.3.0-evolve, at arbitrary order via a Taylor-tower engine that
returns exact bignum/rational derivatives when the math supports it, with the
reverse-mode tensor-op gradient path (`conv2d`/`batchnorm`/`layernorm`/`attention`)
now verified exact for first-class losses and vector/learnable gamma as of
v1.3.3-evolve; memory is
allocated through Ownership-Aware Lexical Regions with deterministic, per-scope
deallocation and no garbage collector; the consciousness engine exposes twenty-two
builtins covering unification, factor-graph belief propagation, free-energy
minimisation, and global-workspace softmax competition.

The flagship demonstration is the SDNC paper artefact: a single shell invocation
regenerates the 12.22M-parameter weight tensor and verifies that a reference C
interpreter, a simulated transformer, and a matrix-based forward pass agree on
123 of 123 traced programs at every step on every dimension of the 256-dimensional
state vector. The artefact lives in the same repository as the compiler that
hosts it.

---

## Differentiating capabilities

Each item below cites the file or measurement that grounds the claim.

- **Constructive proof of a transformer as an interpreter.** Six layers, d_model = 256,
  feed-forward width 2304, 16 attention heads, 12.22M parameters. The artefact covers
  82 of the 83 canonical opcodes; the one remaining external boundary is `OP_NATIVE_CALL`,
  the deliberate dispatch point for host-runtime services.
  See *docs/SDNC.md* and *lib/backend/weight_matrices.c*. The reproduction harness is
  *scripts/paper/run_paper_suite.sh*; expected wall time is under five minutes on a 2023 M2 Max.

- **Arbitrary-order automatic differentiation.** A Taylor-tower engine (thirteen gated
  phases, P0-P12) computes every derivative up to an arbitrary order `k` in one pass —
  `k+1` coefficients and O(k²) work, not the 2^k blow-up of nested dual numbers. When the
  seed point is exact and the function only uses exact-preserving operators, `derivative-n`
  and `taylor` return exact arbitrary-precision (bignum/rational) results rather than
  floating-point approximations; `taylor-model`/`tm-range`/`tm-eval` pair the polynomial
  with a rigorous interval-remainder bound for a provable range enclosure. Towers are
  tensor-valued, compose through reverse-mode (checkpointed reverse-over-Taylor), recover
  sparse Hessian structure via graph coloring, and work through `if`/`cond`/named-let/
  recursion. See *lib/core/taylor_recurrences.def*, *lib/core/runtime_taylor.c*, and the
  [Automatic Differentiation guide](../docs/guide/AUTOMATIC_DIFFERENTIATION.md).

- **Exact reverse-mode tensor-op gradients for first-class losses (v1.3.3-evolve).**
  `gradient` on `conv2d`/`batchnorm`/`layernorm`/`attention` now propagates an exact
  gradient to the second operand (kernel/gamma-beta/K-V) whether the loss is a
  compile-time-known function or a first-class value with no compile-time `Function*` —
  the latter previously fell to a forward-mode-dual closure path that silently returned
  a zero gradient. Batch-norm/layer-norm gamma/beta are now differentiated per-feature
  rather than as one scalar, and any remaining unsupported tensor-op backward path
  raises an explicit error instead of a silent zero. This corrects a prior CHANGELOG
  entry (#212) that an adversarial finite-difference audit found to be a no-op; the real
  fix is #229, finite-difference-verified in both literal and first-class forms.

- **Quantum computing, opt-in and differentiable (v1.3.3-evolve,
  `-DESHKOL_QUANTUM_ENABLED=ON`).** The `agent.quantum` module binds the Moonlab
  state-vector simulator: state creation/teardown, Hadamard/Pauli/CNOT/rotation
  gates, `measure`, `expectation-z`, and a `with-quantum-state` auto-destroy
  helper, plus VQE builtins with H2/LiH/H2O molecular Hamiltonians whose
  energies differentiate through Eshkol's own AD — new custom-VJP tape nodes
  bridge Moonlab's exact adjoint gradient into the reverse tape, so
  `(vqe-energy ...)` composes with ordinary `gradient`/optimizer code (the
  release gate requires the bridged adjoint to match Moonlab's native adjoint
  to within `1e-8` and a central finite difference to within `1e-4`). A
  permanent 16K-shot CHSH Bell-inequality gate (`bell-chsh`) measures S ≈ 2.86,
  beyond the classical bound of 2, proving genuine quantum correlations rather
  than a classical imitation, and `quantum-random` draws from Moonlab's
  Bell-verified QRNG when quantum is enabled, with an honestly-labeled
  classical fallback otherwise. The companion `agent.pqc` module provides
  ML-KEM (FIPS 203) post-quantum key encapsulation — `mlkem-keygen`/
  `mlkem-encaps`/`mlkem-decaps` at the 512/768/1024 security levels over
  R7RS bytevectors, QRNG-seeded, verified against NIST KAT fingerprints.
  See *lib/agent/quantum.esk*, *lib/agent/pqc.esk*, and
  *lib/agent/c/agent_quantum.c*.

- **Incremental dataflow (`core.dbsp`, v1.3.3-evolve).** Z-sets (weighted
  multisets) as a commutative group, the `z^-1`/`D`/`I` stream operators (D
  and I mutual inverses), incremental relational operators — linear
  map/filter/project/union, join via the discrete three-term product rule,
  multiplicity-correct `distinct` — and the generic incrementalizer
  `Q^Δ = D ∘ lift(Q) ∘ I`, in pure Eshkol with zero compiler changes; the
  first shipped slice of the incremental-dataflow spine (ADR 0009).
  Acceptance gate 27/27 under JIT and AOT. See *lib/core/dbsp.esk*.

- **Compiler-integrated automatic differentiation (order ≤ 2).** Three modes: symbolic AST
  rewriting at compile time using twelve differentiation rules; forward mode through
  16-byte dual numbers `{value, derivative}`; reverse mode through a computational
  graph spanning more than twenty AD node types with a 32-level tape stack for
  nested gradients. Eight vector-calculus operators — `derivative`, `gradient`,
  `jacobian`, `hessian`, `divergence`, `curl`, `laplacian`,
  `directional-derivative` — are language primitives. As of v1.3.3-evolve,
  custom-VJP tape nodes (`AD_NODE_CUSTOM`) carry an externally supplied
  vector-Jacobian product, so a foreign computation with a known adjoint
  participates exactly in reverse-mode AD (first user: Moonlab's VQE
  gradient), and the per-component gradient replay is collapsed into one
  primal plus one reverse pass reading every input gradient from the tape
  (verified: N primal calls become 1, checked at N=4 and N=64), with new
  `(ad-counters)` introspection exposing primal-call/reverse-pass/tape
  counters. See *lib/backend/autodiff_codegen.cpp*
  (≈12,600 lines) and *docs/DESIGN.md §Automatic Differentiation*.

- **100% R7RS conformance on the portable differential corpus.** A reference-Scheme
  oracle runs the same 34-program portable R7RS-small corpus on Eshkol and on
  chibi-scheme 0.12.0 and diffs the output: 34 of 34 AGREE (100%), up from 27/34 at the
  start of the v1.3.0-evolve cycle. Separately, Eshkol implements roughly 95% of the
  broader R7RS-small procedure surface (232 of 244 procedures) — full numeric tower,
  continuations, exceptions, promises, `eval`, records, bytevectors, hygienic macros.
  See *scripts/run_reference_differential.sh* and *tests/reference-diff/corpus/*.

- **Full R7RS numeric tower.** int64, arbitrary-precision bignum (with automatic
  overflow promotion and demotion), exact rational with GCD reduction, IEEE 754 double,
  and complex numbers with Smith's-formula division. Exactness tracked via a flags
  byte on each 16-byte tagged value. As of v1.3.3-evolve exact rationals are
  bignum-capable: a canonical discriminated union with a zero-allocation int64
  fast path and a bignum numerator/denominator path taken only on overflow, so
  exact fractions no longer degrade to double at bignum magnitudes — verified
  byte-identical against Python `Fraction`. See *lib/backend/arithmetic_codegen.cpp* and
  *inc/eshkol/eshkol.h §Heap subtypes*.

- **Flat memory for resident and daemon workloads.** Self-tail-recursive loops — both
  named-let and plain `define`, including a catch-all guard body — get automatic,
  zero-annotation per-iteration arena-scope reclamation, verified to hold RSS flat
  (1,369 MB unbounded growth to 224 MB flat on a 1,000,000-iteration test loop). The
  S-expression reader was rewritten from per-element native recursion to an iterative
  loop, so reading back a very large persisted data structure no longer risks a
  native-stack overflow (verified clean at 20 million elements, where the prior
  implementation crashed with SIGBUS). See *lib/backend/llvm_codegen.cpp* and
  *lib/core/runtime_reader_hosted.cpp*.

- **Region-escape evacuator closes out across every heap subtype (ESH-0214a-e,
  complete as of v1.3.3-evolve).** A value allocated inside `with-region` that escapes
  the region — is returned, stored outward, or captured by a closure — is deep-walked
  and promoted into the surviving arena instead of being left dangling after the region
  pops. Coverage was built out subtype by subtype: cons/vector/hash/tensor/exception/
  closure first, then the logic and global-workspace subtypes (substitution, fact,
  knowledge base, factor graph, workspace) in v1.3.2-evolve, and `PROMISE` — a
  `delay`/`make-promise` thunk plus its cached value — in v1.3.3-evolve, completing
  the series. `ESHKOL_ARENA_POISON=1` poisons freed arena memory so any remaining gap
  crashes loudly instead of corrupting silently.

- **Neuro-symbolic stack as compiler primitives.** Twenty-two builtins:
  `unify`, `walk`, `make-substitution`, `make-fact`, `make-kb`, `kb-assert!`,
  `kb-query`, `logic-var?`, `substitution?`, `kb?`, `fact?`,
  `make-factor-graph`, `fg-add-factor!`, `fg-infer!`, `fg-update-cpt!`,
  `free-energy`, `expected-free-energy`, `factor-graph?`,
  `make-workspace`, `ws-register!`, `ws-step!`, `workspace?`.
  Runtime implementations: *lib/core/logic.cpp* (≈1,180 lines), *lib/core/inference.cpp*
  (≈1,200 lines), *lib/core/workspace.cpp* (354 lines), lineage Robinson 1965 / Friston 2010 /
  Baars 1988.

- **Deterministic arena memory (OALR).** Single global arena with 8 KB minimum blocks,
  O(1) bump-pointer allocation, batch reset, 8-byte headers prepended to every heap
  object. Per-thread arenas (1 MB, lazily allocated) isolate parallel workers.
  See *lib/core/arena_memory.h* and the *lib/core/runtime_arena_\*.cpp* modules, and *docs/breakdown/PARALLEL_COMPUTING.md §2.1*.

- **Work-stealing parallelism.** Chase-Lev deques per worker (Chase and Lev, 2005)
  with epoch-based reclamation. Measured 4–12× speed-up of `parallel-map` on 24 cores
  per *docs/breakdown/ROADMAP.md §1.1-accelerate completed*. Primitives:
  `parallel-map`, `parallel-fold`, `parallel-filter`, `parallel-for-each`,
  `future` / `force`.

- **GPU acceleration with cost-model dispatch.** SIMD micro-kernels for small tensors,
  Apple Accelerate cBLAS at the AMX peak (≈1,100 GFLOPS measured), Metal with double-
  double SF64 emulation for native float64 absence, and a CUDA path through cuBLAS.
  Backend chosen per operation by *lib/backend/blas_backend.cpp*, configurable via
  `ESHKOL_GPU_PRECISION`, `ESHKOL_BLAS_PEAK_GFLOPS`, `ESHKOL_GPU_PEAK_GFLOPS`.

- **Native agent FFI.** libcurl-backed HTTP client (*lib/agent/c/agent_http_client.c*),
  sqlite3 (*lib/agent/c/agent_sqlite.c*), `posix_spawn` subprocess execution with
  argv arrays (*lib/agent/c/agent_subprocess.c*; the `popen("sh -c …")` path was
  removed in v1.2 to eliminate shell-metacharacter exposure), kqueue/inotify
  filesystem watching (*lib/agent/c/agent_watch.c*).

- **Comprehensively documented public API and implementation.** v1.3.1-evolve added
  Doxygen-format documentation across 50 of the 64 public headers under
  `inc/eshkol/` (~4,650 lines) and 56 previously-undocumented implementation files
  under `lib/` (~7,478 lines) — 116 files and roughly 12,600 lines total, comments
  only; v1.3.2-evolve added `eshkol-doc`, which harvests those comments automatically
  into a generated `docs/api/` reference. A navigable per-subsystem reference index
  (*docs/reference/{language,ad,runtime,tensors,stdlib,agent}/INDEX.md*) organizes
  the language surface for lookup.

- **Hardened, permanent adversarial-testing program.** A multi-pillar adversarial
  harness — differential, feature-pair edge matrix, AD finite-difference oracle,
  stress (RSS/time budgets), VM-parity ratchet, depth-parametric sweeps, and the
  external reference-Scheme differential oracle — is wired permanently into the ICC
  release oracle rather than run once and discarded; this is the same infrastructure
  whose adversarial audit caught the v1.3.2-evolve `input2` overstatement corrected
  above, and the ICC oracle itself was hardened in v1.3.3-evolve to gate on that
  correction and on region-evacuator poison coverage. v1.3.3-evolve added two
  generative exposure engines to the program, both wired permanently into the
  release oracle: a multi-oracle differential harness (deterministically grown
  R7RS-subset programs cross-checked against chibi-scheme, JIT, AOT at O0/O2,
  and the bytecode VM) and an AD-vs-finite-difference adversarial oracle (147
  probes / 436 component checks across 21 generated files under JIT and AOT; a
  zero AD gradient where finite differences are nonzero is a hard failure).
  Release gates green on the v1.3.3-evolve SHA: ICC readiness oracle 100/100
  (trace-verified); the aggregate suite 44/44 suites and 716/716 tests; SICP
  full-book gate 88/88 probes across all five chapters under both `-r` and
  AOT; reference-Scheme differential oracle 34/34 AGREE. See
  *docs/TESTING.md*.

- **Binary Lambda Calculus (`core.blc`, v1.3.2-evolve).** A pure-Eshkol
  implementation of John Tromp's BLC: De Bruijn-indexed terms as homoiconic
  s-expressions, self-delimiting bit encode/decode, normal-order evaluation, a
  decoded 232-bit universal machine, BLC8 byte I/O, and ASCII lambda diagrams.
  Loaded on demand via `(require core.blc)`. See
  *docs/guide/BINARY_LAMBDA_CALCULUS.md*.

---

## Example

The training loop below is verbatim from *README.md §Why Eshkol*. It uses
the language's `derivative` primitive to fit `y = 2x` from five points.
Nothing here is a library import or a framework call — `derivative` is in
the compiler.

```scheme
(define training-data '((1.0 2.0) (2.0 4.0) (3.0 6.0) (4.0 8.0) (5.0 10.0)))

(define (predict w x) (* w x))

(define (loss w)
  (fold-left (lambda (total pair)
    (let ((error (- (predict w (car pair)) (cadr pair))))
      (+ total (* error error))))
    0.0 training-data))

(define (train w lr steps)
  (if (= steps 0) w
    (train (- w (* lr (derivative loss w))) lr (- steps 1))))

(display (train 0.0 0.01 200))  ;; => 2.0
```

Arbitrary-order AD, run for real (`eshkol-run -r`):

```scheme
(define (f x) (expt x 30))
(display (derivative-n f 7 12))   ;; => 67465815595294257109436307840000 (exact bignum)
(display (exact? (derivative-n f 7 12)))  ;; => #t
```

---

## Dual backend

Eshkol ships two production execution backends with the same language
semantics and independent value representations. The LLVM backend compiles
to native ARM64 or x86-64 (or WebAssembly) and is the default for
`eshkol-run`. The bytecode VM (*lib/backend/eshkol_vm.c* plus its 32 *vm_\*.c*
modules) is a 63-opcode register-plus-stack interpreter with more than 250
native call IDs, an ESKB binary file format with LEB128 encoding and CRC32
checksums, and full coverage of the language including continuations,
exception handling, tensors, complex / rational / bignum, the consciousness
engine, and I/O. The browser REPL runs the bytecode VM (compiled to
WebAssembly via Emscripten); forward-mode AD via dual numbers works through
the same arithmetic opcodes. A v1.3.3-evolve parity campaign, driven by a
generative multi-oracle differential harness cross-checking chibi-scheme,
JIT, AOT at O0/O2, and the VM, closed several silent-miscompile classes —
including real bignum-aware VM arithmetic and comparisons.

The weight-matrix transformer artefact (*lib/backend/weight_matrices.c*,
≈7,500 lines) is the third execution surface — the one that proves the
SDNC theorem by being a transformer that runs the same VM through its
forward and backward passes.

---

## Lineage and references

R7RS Scheme (the language definition); Homotopy Type Theory (the type-system
foundation, gradual rather than strict); LLVM 21 (the code generation target,
hard version-enforced in *cmake/LLVMToolchain.cmake*); Robinson's resolution
principle, 1965; Friston's free-energy principle, 2010; Baars' global workspace
theory, 1988; Chase and Lev, *Dynamic Circular Work-Stealing Deque*, 2005.

The SDNC paper provides the constructive proof that ties the language's
gradient infrastructure to the transformer artefact — *docs/SDNC.md* and
*docs/breakdown/COMPUTABLE_TRANSFORMER.md*.

---

## Reproducibility

The SDNC artefact reproduces in one command:

```bash
scripts/paper/run_paper_suite.sh
```

Outputs land under `artifacts/paper/outputs/` with stable SHA-256 hashes printed
by the harness. A current successful run produces, among others,
`weights.qlmw = 381599e7…3f0c`, `vm-traces.jsonl = 4239cbb9…4801` (the transformer
trace agrees bitwise: same SHA), and `comparison-report.json = 80aa6fed…4105`.
Platform divergence is treated as a bug.

The compiler itself is bit-reproducible at link time: two back-to-back
release builds produce byte-identical `build/stdlib.bc` and `build/eshkol-run`
(*docs/HARDENING.md §`#184`*).

---

## Repository and version

| | |
|:---|:---|
| Project | Eshkol |
| Version | v1.3.4-evolve |
| Release date | 23 July 2026 (builds on v1.3.3-evolve, 16 July 2026; v1.3.2-evolve, 9 July 2026; v1.3.1-evolve and v1.3.0-evolve, 7 July 2026) |
| Implementation | C17 runtime, C++20 compiler |
| Backend | LLVM 21 (version-enforced) |
| Platforms | macOS Intel and Apple Silicon, Linux x86-64 and ARM64, Windows x86-64 and ARM64 via Visual Studio 2022 + ClangCL |
| WebAssembly target | yes (`eshkol-run --wasm`) |
| Licence | MIT |
| Source | https://github.com/tsotchke/eshkol |
| Website | https://eshkol.ai |
| Paper companion | *docs/SDNC.md*, artefact `artifacts/paper/` |
