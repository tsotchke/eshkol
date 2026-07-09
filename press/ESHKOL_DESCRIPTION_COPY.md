# Eshkol — Description Copy

## A compiled Scheme with a constructive proof that a transformer is an interpreter

Eshkol is a compiled programming language for mathematical and cognitive computing.
The repository ships v1.2.1-scale (May 2026) of the compiler together with the
reproducibility artefact for *The Self-Differentiating Neural Computer: Computable
Transformers via Analytical Weight Construction* (tsotchke, 2026), in which a
six-layer transformer with 12.22 million analytically-constructed parameters
executes a bounded 83-opcode bytecode VM bit-identically. The result is a constructive,
not statistical, demonstration that a fixed-weight transformer can be an interpreter
when its weights are derived from an instruction-set specification rather than fit
by gradient descent.

---

## Lede

Eshkol is an R7RS-compatible Scheme dialect that compiles through LLVM 21 to native
binaries on macOS, Linux, and Windows, and to WebAssembly for browser execution.
The language treats automatic differentiation, arena memory, and a neuro-symbolic
computation layer as compiler primitives rather than library add-ons. Differentiation
is available in symbolic, forward, and reverse modes alongside eight vector-calculus
operators; memory is allocated through Ownership-Aware Lexical Regions with
deterministic, per-scope deallocation and no garbage collector; the consciousness
engine exposes twenty-two builtins covering unification, factor-graph belief
propagation, free-energy minimisation, and global-workspace softmax competition.

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

- **Compiler-integrated automatic differentiation.** Three modes: symbolic AST
  rewriting at compile time using twelve differentiation rules; forward mode through
  16-byte dual numbers `{value, derivative}`; reverse mode through a computational
  graph spanning more than twenty AD node types with a 32-level tape stack for
  nested gradients. Eight vector-calculus operators — `derivative`, `gradient`,
  `jacobian`, `hessian`, `divergence`, `curl`, `laplacian`,
  `directional-derivative` — are language primitives. See *lib/backend/autodiff_codegen.cpp*
  (9,205 lines) and *docs/DESIGN.md §Automatic Differentiation*.

- **Full R7RS numeric tower.** int64, arbitrary-precision bignum (with automatic
  overflow promotion and demotion), exact rational with GCD reduction, IEEE 754 double,
  and complex numbers with Smith's-formula division. Exactness tracked via a flags
  byte on each 16-byte tagged value. See *lib/backend/arithmetic_codegen.cpp* and
  *inc/eshkol/eshkol.h §Heap subtypes*.

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
  filesystem watching (*lib/agent/c/agent_watch.c*). Path-traversal, TOCTOU, and
  Windows command-line buffer issues are addressed in *docs/HARDENING.md §`#192`/`#193`*.

- **Hardened release pipeline.** Fourteen audit blockers closed in v1.2-scale,
  seven critical/high security fixes (shell-injection, Python-FFI AST-injection,
  integer-overflow guards on arena/KB/image, path-traversal, TOCTOU,
  ReDoS, SQLi, URL/header CRLF). 87 edge/security regression tests pass; the master
  suite reports 37 of 37 sub-suites and 528 of 528 self-reported tests passing
  (*RELEASE_NOTES.md §v1.2.1 Noesis M0 Closeout Addendum*). ASan + UBSan clean on
  196 checks / 9 suites; TSan clean on the seven parallel-execution suites
  (*docs/HARDENING.md §`#178`, `#180`*).

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

## Hardening

v1.2-scale closes fourteen audit blockers and seven critical/high security
fixes (subprocess shell-injection, Python-FFI AST-injection, integer-overflow
guards on arena, KB persistence, and image I/O, path-traversal and TOCTOU
on the file copy path, a Windows subprocess command-line buffer issue,
ReDoS, SQL-injection guards, and URL/header CRLF). The eighty-seven-test
edge-and-security suite at *tests/v1_2_edge_cases/* regression-covers each
fix. The master test runner reports 37 of 37 sub-suites and 528 of 528
self-reported tests passing; ASan and UBSan are clean across 196 checks
in 9 suites; TSan is clean across the parallel-execution surface
(*docs/HARDENING.md §`#178`, `#180`*). Two back-to-back release builds produce
byte-identical `build/stdlib.bc` and `build/eshkol-run` (*docs/HARDENING.md
§`#184`*).

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
| Version | v1.2.1-scale |
| Release date | base 1 May 2026, Noesis-M0 closeout 20 May 2026 |
| Implementation | C17 runtime, C++20 compiler |
| Backend | LLVM 21 (version-enforced) |
| Platforms | macOS Intel and Apple Silicon, Linux x86-64 and ARM64, Windows x86-64 and ARM64 via Visual Studio 2022 + ClangCL |
| WebAssembly target | yes (`eshkol-run --wasm`) |
| Licence | MIT |
| Source | https://github.com/tsotchke/eshkol |
| Website | https://eshkol.ai |
| Paper companion | *docs/SDNC.md*, repository `noesis`, artefact `artifacts/paper/` |
