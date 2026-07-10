# Eshkol — Description Copy

## A compiled Scheme with a constructive proof that a transformer is an interpreter

Eshkol is a compiled programming language for mathematical and cognitive computing.
The repository ships v1.3.3-evolve (July 2026) of the compiler — an arbitrary-order
automatic-differentiation system with exact gradients now verified through the
tensor-op `input2` path, 100% conformance on a portable R7RS differential corpus,
and a region-escape evacuator that makes long-running/resident programs safe to
leave running across every heap subtype the language can allocate — together
with the reproducibility artefact for
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

- **Compiler-integrated automatic differentiation (order ≤ 2).** Three modes: symbolic AST
  rewriting at compile time using twelve differentiation rules; forward mode through
  16-byte dual numbers `{value, derivative}`; reverse mode through a computational
  graph spanning more than twenty AD node types with a 32-level tape stack for
  nested gradients. Eight vector-calculus operators — `derivative`, `gradient`,
  `jacobian`, `hessian`, `divergence`, `curl`, `laplacian`,
  `directional-derivative` — are language primitives. See *lib/backend/autodiff_codegen.cpp*
  (9,205 lines) and *docs/DESIGN.md §Automatic Differentiation*.

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
  byte on each 16-byte tagged value. See *lib/backend/arithmetic_codegen.cpp* and
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
  Runtime implementations: *lib/core/logic.cpp* (805 lines), *lib/core/inference.cpp*
  (912 lines), *lib/core/workspace.cpp* (308 lines), lineage Robinson 1965 / Friston 2010 /
  Baars 1988.

- **Deterministic arena memory (OALR).** Single global arena with 8 KB minimum blocks,
  O(1) bump-pointer allocation, batch reset, 8-byte headers prepended to every heap
  object. Per-thread arenas (1 MB, lazily allocated) isolate parallel workers.
  See *lib/core/arena_memory.cpp* (6,186 lines) and *docs/breakdown/PARALLEL_COMPUTING.md §2.1*.

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
  correction and on region-evacuator poison coverage. Release gates green on the
  v1.3.0-evolve SHA: ICC readiness oracle 100/100 (trace-verified); CI 14/14 lanes
  including windows-arm64; SICP full-book gate 88/88 probes across all five chapters
  under both `-r` and AOT; reference-Scheme differential oracle 34/34 AGREE. See
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
`eshkol-run`. The bytecode VM (*lib/backend/eshkol_vm.c*, 8,457 lines) is a
63-opcode register-plus-stack interpreter with more than 250 native call
IDs, an ESKB binary file format with LEB128 encoding and CRC32 checksums,
and full coverage of the language including continuations, exception
handling, tensors, complex / rational / bignum, the consciousness engine,
and I/O. The browser REPL runs the bytecode VM (compiled to WebAssembly
via Emscripten); forward-mode AD via dual numbers works through the same
arithmetic opcodes.

The weight-matrix transformer artefact (*lib/backend/weight_matrices.c*,
≈6,800 lines) is the third execution surface — the one that proves the
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
| Version | v1.3.3-evolve |
| Release date | 10 July 2026 (builds on v1.3.2-evolve, 9 July 2026; v1.3.1-evolve and v1.3.0-evolve, 7 July 2026) |
| Implementation | C17 runtime, C++20 compiler |
| Backend | LLVM 21 (version-enforced) |
| Platforms | macOS Intel and Apple Silicon, Linux x86-64 and ARM64, Windows x86-64 and ARM64 via Visual Studio 2022 + ClangCL |
| WebAssembly target | yes (`eshkol-run --wasm`) |
| Licence | MIT |
| Source | https://github.com/tsotchke/eshkol |
| Website | https://eshkol.ai |
| Paper companion | *docs/SDNC.md*, repository `noesis`, artefact `artifacts/paper/` |
