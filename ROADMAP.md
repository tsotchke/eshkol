# Eshkol Development Roadmap

> **This is the canonical roadmap.** It is the single source of truth for
> Eshkol's release plan. Other roadmap-shaped documents are derived
> views — they exist to answer specific questions and should not
> contradict this file:
>
> - [`docs/COMPILER_ROADMAP.md`](docs/COMPILER_ROADMAP.md) — engineering
>   detail by release line (M0–M4 task tiers), including the per-release
>   work breakdown for compiler engineers.
> - [`docs/breakdown/ROADMAP.md`](docs/breakdown/ROADMAP.md) — short
>   summary for readers in the breakdown / per-subsystem documentation
>   set.
> - [`docs/vision/FUTURE_ROADMAP.md`](docs/vision/FUTURE_ROADMAP.md) —
>   forward-looking vision and long-horizon research items.
> - [`docs/NOESIS_TRAJECTORY.md`](docs/NOESIS_TRAJECTORY.md) — the
>   Noesis-readiness view, tracked separately because Noesis has a
>   distinct downstream cadence.
>
> When any of those docs disagrees with this one, this one is correct
> and the others should be updated.

This roadmap tracks Eshkol's evolution from the **completed v1.0-foundation release** through upcoming versions that will establish Eshkol as the definitive platform for gradient-based computing and integrated AI.

> **Parallel platform program**: The internal freestanding / kernel / embedded architecture work begins during `v1.2-scale` as a mergeable infrastructure program and converges publicly at `v1.8-platform`. See [docs/platform/README.md](docs/platform/README.md) and [docs/platform/ROADMAP_ALIGNMENT.md](docs/platform/ROADMAP_ALIGNMENT.md).

---

## v1.0-foundation (2025) - COMPLETED

**Production Release Delivered**

### Completed Core Implementation
- [x] LLVM-based modular backend with 21 specialized codegen modules
- [x] Recursive descent parser with HoTT type expression support
- [x] Bidirectional type checker with gradual typing
- [x] Ownership and escape analysis for memory optimization
- [x] Module system with dependency resolution and cycle detection
- [x] Hygienic macro system (syntax-rules)
- [x] R7RS Scheme compatibility (subset)

### Completed Automatic Differentiation
- [x] Forward-mode AD (dual numbers)
- [x] Reverse-mode AD (computational graphs)
- [x] Nested gradients (32-level tape stack)
- [x] Vector calculus operators (8 total): derivative, gradient, jacobian, hessian, divergence, curl, laplacian, directional-derivative
- [x] Polymorphic arithmetic (int64/double/dual/tensor/AD-node)

### Completed Memory Management
- [x] Arena allocation with OALR (Ownership-Aware Lexical Regions)
- [x] Escape analysis (stack/region/shared allocation decisions)
- [x] with-region syntax for lexical memory scopes
- [x] Ownership tracking (owned, moved, borrowed states)
- [x] Zero garbage collection - fully deterministic

### Completed Data Structures
- [x] 16-byte tagged values with consolidated types
- [x] 32-byte cons cells supporting mixed-type lists
- [x] N-dimensional tensors with autodiff integration
- [x] Hash tables (FNV-1a hashing, open addressing)
- [x] Heterogeneous vectors
- [x] Exception handling (guard/raise)

### Completed Standard Library
- [x] 60+ list operations
- [x] 30+ string utilities
- [x] Functional programming (compose, curry, flip)
- [x] JSON/CSV/Base64 support
- [x] Math library (linear algebra, numerical methods, statistics)

### Completed Development Tools
- [x] Interactive REPL with LLVM ORC JIT
- [x] Standalone compiler (eshkol-run)
- [x] Library compilation mode
- [x] Comprehensive test suite (430+ files)
- [x] CMake build system
- [x] Docker containers

---

## v1.1-accelerate (Q1 2026) - COMPLETED

**Focus:** Performance acceleration through XLA, SIMD, and parallelism

### XLA Backend Integration
- [x] XLA type system and codegen infrastructure
- [x] XLA fusion for tensor operation chains
- [x] Automatic kernel generation
- [x] CPU/GPU code generation from single source
- [x] JIT compilation for dynamic shapes

### SIMD Vectorization
- [x] SSE/AVX/NEON instruction generation
- [x] Loop vectorization for tensor operations
- [x] Memory alignment optimization
- [x] Platform-specific tuning

### Concurrency Primitives
- [x] `parallel-map` for data parallelism
- [x] `parallel-fold` for parallel reduction
- [x] `future` for asynchronous computation
- [x] Work-stealing thread pool scheduler
- [x] Thread-safe memory management

### Extended Math Library
- [x] Complex numbers with autodiff
- [x] FFT/IFFT operations
- [x] Signal processing filters
- [x] Statistical distributions
- [x] Optimization algorithms (L-BFGS, conjugate gradient)

### Arbitrary-Precision Arithmetic (Added)
- [x] Bignum (arbitrary-precision integers) — full R7RS compliance
- [x] Rational numbers (exact fractions)
- [x] Overflow detection and automatic promotion (int64 → bignum)
- [x] Bignum demotion (normalize back to int64 when possible)
- [x] Bitwise operations on bignums (two's complement semantics)
- [x] All arithmetic, comparison, predicate, equality ops for bignums

### Consciousness Engine (Added)
- [x] Logic programming primitives (unification, substitutions, knowledge base)
- [x] Active inference engine (factor graphs, belief propagation, free energy)
- [x] Global workspace theory implementation (modules, softmax competition)
- [x] 22 builtin operations for logic, inference, and workspace

### R7RS Compliance Extensions (Added)
- [x] call/cc and dynamic-wind
- [x] guard/raise exception handling
- [x] Bytevectors
- [x] let-syntax / syntax-rules hygienic macros
- [x] Tail call optimization validation
- [x] Symbol operations
- [x] `(load "path")` R7RS file loading support

### Dual Backend Architecture (Added)
- [x] Bytecode VM — 63-opcode register+stack interpreter with 250+ native call IDs
- [x] ESKB binary format — section-based bytecode container with CRC32
- [x] `-B` flag for bytecode emission from eshkol-run
- [x] VM compiler integration — eshkol_vm.c linked into compiler build
- [x] Weight matrix transformer — programs as neural network weights (126/126 inline, 123/123 traced, 3-way verified)
- [x] qLLM bridge — Eshkol↔qLLM tensor conversion with AD integration *(design-era claim; the implementation landed in v1.3.4-evolve, #386/#392 — see below)*

### GPU Acceleration (Added)
- [x] Metal SF64/DF64/F32/FP24/FP53 precision tiers
- [x] Ozaki-II CRT-based exact matrix multiplication
- [x] CUDA backend with cuBLAS
- [x] Cost-model dispatch (SIMD → cBLAS → GPU)
- [x] Occupancy-aware kernel configuration

### Signal Processing (Added)
- [x] FFT/IFFT (Cooley-Tukey radix-2)
- [x] Window functions (Hamming, Hann, Blackman, Kaiser)
- [x] FIR/IIR filters
- [x] Butterworth filter design (lowpass, highpass, bandpass)

### Web Platform (Added)
- [x] WebAssembly compilation target
- [x] Browser-based REPL (web/)
- [x] JavaScript interop

### v1.1.12 Additions (April 2026)
- [x] Production bytecode VM (555+ builtins, 176/176 tests)
- [x] Forward-mode AD in bytecode VM (dual number propagation through all opcodes)
- [x] eshkol.ai website written in Eshkol, compiled to WASM (8 pages, browser REPL)
- [x] Interactive documentation with runnable code examples
- [x] GitHub Pages deployment workflow
- [x] R7RS control flow in VM: call/cc, guard/raise, dynamic-wind, values
- [x] Exact arithmetic in VM: rational literals, +nan.0/+inf.0/-inf.0

### v1.1.13 Additions (April 2026)
- [x] Native Windows ARM64 build path (VS 2022 + ClangCL + LLVM 21 aarch64 SDK)
- [x] 16-lane release matrix (linux/macos/windows × x64/arm64 × lite/xla/cuda)
- [x] Per-arch LLVM SDK caching on Windows runners
- [x] Two critical VM closure bug fixes (named-let nested closure PC offset, native 252 upvalue relay)
- [x] Windows setjmp hardening: x64 frameaddress, ARM64 sponentry, dynamic jmp_buf sizing
- [x] Runtime symbol renames (eshkol_fopen, eshkol_access, eshkol_remove, etc.) for MSVC POSIX shim disambiguation
- [x] Codegen fatal error flag — fail hard on undefined functions instead of generating runtime stubs
- [x] Mobile-responsive website (hamburger nav, internal scroll for code blocks, table responsiveness)
- [x] Browser REPL error display for invalid input
- [x] Consciousness engine in VM: KB pattern matching, factor graphs, workspace
- [x] Top-level mutual recursion via letrec-style group compilation

### Windows Platform Support (Added)
- [x] Native Windows build via MSYS2/MinGW64 (PR #9, mattneel)
- [x] UTF-8-safe REPL console output
- [x] Platform runtime abstraction layer

---

## Architecture Dependency Chain

```
v1.1 (COMPLETE)
 ├─ Consciousness engine (logic + inference + workspace)
 ├─ XLA/GPU backend
 ├─ Bytecode VM (production)
 ├─ Continuations + exact arithmetic
 └─ Web platform (WASM)
       │
v1.2 ──┤ Model serialization (requires tensors from v1.1)
       ├ Python bindings (requires stable API from v1.1)
       └ Per-thread arenas (requires OALR from v1.0)
             │
v1.3 ────────┤ R7RS library system (requires module system from v1.0)
             ├ String interpolation (parser extension)
             └ PGO (requires stable codegen from v1.1+)
                   │
v1.4 ──────────────┤ TCP/UDP + TLS (requires per-thread arenas from v1.2)
                   ├ Event loop (requires non-blocking I/O)
                   └ Linear resource types (requires HoTT from v1.0)
                         │
v1.5 ────────────────────┤ Symbol embeddings (requires tensors + KB from v1.1)
                         ├ Differentiable logic (requires AD + logic from v1.1)
                         └ LSTM/GRU (requires tensor backprop from v1.1)
                               │
v1.6 ──────────────────────────┤ Backward chaining (requires logic from v1.1)
                               ├ Knowledge graphs (requires KB + embeddings from v1.5)
                               └ Constraint solving (requires logic engine)
                                     │
v1.7 ────────────────────────────────┤ Neural-guided search (requires v1.5 bridge)
                                     ├ GNN (requires graph + tensor ops)
                                     └ Program synthesis (requires type system)
                                           │
v1.8 ──────────────────────────────────────┤ Windowing + event system (requires v1.4 I/O)
                                           ├ Real-time audio (requires signal from v1.1)
                                           └ Embedded targets (requires bare-metal LLVM)
                                                 │
v1.9 ────────────────────────────────────────────┤ Linear dependent types (requires HoTT)
                                                 ├ Effect types (requires type checker)
                                                 └ Algebraic effects (requires continuations)
                                                       │
v2.0 ──────────────────────────────────────────────────┤ Quantum types (requires linear dep types)
                                                       ├ Quantum gates + measurement
                                                       ├ Hybrid VQE/QAOA (requires AD + quantum)
                                                       └ Formal verification (requires dep types)
```

**Arbitrary-order AD (Taylor-tower) track — SHIPPED ahead of schedule:** the
campaign was originally planned to be threaded through the version themes
above as enabling substrate spread from v1.3.1 through v2.0 (P1 in v1.3.1,
P2/P3 in v1.3.2, P4/P6/P11 in v1.4, P5/P7/P9 in v1.5, P10 bridging v1.5-v1.7,
P12 in v1.6, P8 in v2.0). Instead, **all 13 phases (P0-P12) shipped complete
in v1.3.0-evolve** — see [CHANGELOG.md](CHANGELOG.md) and
[`docs/AD_CAMPAIGN.md`](docs/AD_CAMPAIGN.md) for the as-shipped detail. The
version rows below still show the original staging plan for historical
context; treat the AD line item in each as already delivered.

---

## v1.2-scale (May 2026) - SHIPPED

**Focus:** Get models into production. Save them, load them, deploy them — and stop being surprised by edge cases.

- [x] Model serialization (`.eshkol-model` ESKB-extended binary format)
- [x] Stable C FFI header + Python bindings (pybind11; numpy zero-copy)
- [x] Per-thread arenas (safe concurrent memory allocation)
- [x] Deep recursion: 512 MB main-thread stack on Darwin/Linux/Windows
      (linker flags wired into both single-step and compiled-files
      link paths); 100K-frame recursion-depth check with typed
      exception
- [x] Image I/O (PNG/JPEG/WebP/BMP read/write/resize) — backed by
      native platform/system codec APIs (ImageIO/CoreGraphics on
      macOS, system libpng/libjpeg/libwebp on Linux, GDI+ on
      Windows) so the active backend no longer depends on vendored
      third-party media decoders
- [x] CSV/DataFrame (tabular data loading for ML pipelines)
- [x] Improved error messages with file:line:col + caret underlines
      (preserves newlines in stripped comments + cumulative file-line
      tracking across `parse_next_ast` calls; 5-case regression suite)
- [x] Terminal plotting (`sparkline`, `bar-chart` in pure Eshkol stdlib)
- [x] Codegen modularisation: `tensor_codegen.cpp` 19,940 → 1,280 lines at
      the time of the v1.2 split (94% reduction) across 13 focused
      per-domain split files; re-measured for this documentation wave at
      1,867 lines as of commit `694c3179` (still-active file, grown with
      subsequent feature work — the 94% reduction was a point-in-time
      measurement, not an invariant). The
      remaining `llvm_codegen.cpp` extractions
      (module_init_codegen.cpp, builtin_factory_codegen.cpp,
      repl_resolution_codegen.cpp) need the `EshkolLLVMCodeGen` class
      header exposed first and are tracked as v1.3 carry-forward.
- [x] v1.2 edge-case + security regression suite (62 tests) wired into
      `run_all_tests.sh` and a new `linux-x64-asan-ubsan` CI lane.
      Includes 3 shell-style tests for compile-time diagnostics.
- [x] Tagged release artifact contract: the GitHub release workflow
      validates the full 16-asset platform matrix before publishing
      (Linux x64/ARM64 lite/XLA/CUDA, macOS arm64/x64 lite/XLA,
      Windows x64/ARM64 lite/XLA/CUDA) and emits `SHA256SUMS.txt`;
      published as the `v1.2.3-scale` packaging closeout.
- [x] Stdlib correctness: user `(define (foo …))` after `(require
      stdlib)` cleanly shadows stdlib's `foo` at link time
      (LinkOnceODR linkage on stdlib functions) and at call-site
      lowering (variadic-info hygiene clears stale entries on
      redefine).
- [x] `--wasm` is self-contained: WASM emit no longer falls through
      to native clang++ link.
- [x] AD scalar derivative on inline lambdas: `(derivative
      (lambda (x) …) point)` inside a wrapper function correctly
      flows through the runtime closure dispatch.  AD value-typed
      captures pass LLVM IR verification when capturing
      function-parameter `tagged_value` Arguments.
- [x] M1 stdlib finalised: `core.json_schema` (Draft 7 subset),
      reflection (`procedure-arity`, `record-fields`, `describe`),
      memoization/LRU, PRNG seeding + deterministic replay, lazy
      streams (SRFI 41), time API (ISO-8601), regex capture groups,
      CLI argument parser, structured logging (JSON-L),
      Prometheus metrics, extra AD ops
      (atan2 / asin / acos / softmax / gelu / silu / sinh / cosh),
      priority queues / sets / deques.
- [x] Hardening: subprocess shell-injection fix (CRITICAL), Python
      FFI AST-injection fix (CRITICAL), 3 integer-overflow guards
      (HIGH), 4 path-traversal/TOCTOU/Windows-buffer fixes (HIGH),
      36 silent-swallow sites surfaced (HIGH), ReDoS protection +
      SQL-injection guards + URL validator (MEDIUM).

---

## v1.3.0-evolve (July 2026) - SHIPPED

**Focus:** Make the language a joy to use day-to-day — and it grew into
much more: a full arbitrary-order automatic-differentiation system.

- [~] Full R7RS library system: `define-library` exports work end-to-end;
      `(rename (m) (a b))` import works.  `(prefix (m) p-)` currently
      requires an explicit `only` or `rename` clause — bare prefix
      over the module's whole export list is the remaining gap.
- [x] String interpolation (`~{expr}` within strings)
- [x] Named keyword arguments (`(f #:key value)`)
- [x] Pattern matching in `let` bindings (destructuring `let-match`)
- [~] Profile-guided optimization — build-time scaffold landed:
      `cmake -DESHKOL_PGO=generate` instruments, `-DESHKOL_PGO=use
      -DESHKOL_PGO_PROFILE=<merged.profdata>` consumes.  Workload
      selection + canonical merge step (the "what do we train on?"
      side) is the remaining gap; the codegen-side machinery is ready.
- [ ] Whole-program optimization (cross-module inlining and dead code elimination)
- [x] **Native media handling, no vendoring**: image I/O uses native
      platform/system codec APIs (ImageIO/CoreGraphics on macOS,
      system libpng/libjpeg/libwebp on Linux, GDI+ on Windows).
      Going forward the project does not vendor third-party media
      decoders.
- [x] AD second-operand (`input2`) gradient plumbing for every tensor
      op — `tensor-matmul`, `conv2d`, `batch-norm`, `layer-norm`, and
      `scaled-dot-attention`. Each op's AD forward path unrolls into the
      scalar reverse-mode graph (`recordADNodeBinary`), so gradients flow
      to the second differentiable operand (matmul kernel, conv2d kernel,
      norm gamma, attention K/V) with no monolithic tape node left with a
      null `input2`. Verified end-to-end by the finite-difference AD oracle
      `tests/v1_3_edge_cases/ad_input2_test.esk` and smoke probes
      `ad_input2_conv2d_grad_works` / `ad_input2_batchnorm_grad_works` /
      `ad_input2_layernorm_grad_works` / `ad_input2_attention_grad_works`
      (JIT `-r` and AOT), which compare the AD gradient to central finite
      differences at tight tolerance.
- [x] **Arbitrary-order AD Taylor-tower campaign — fully delivered, all 13
      phases (P0-P12), well ahead of the original P1-only-in-v1.3 plan
      below**: runtime tower + `taylor`/`derivative-n` (P1); no-heap
      compile-time-K monomorphization (P2); JET8-subsumption analysis (P3);
      GUW arbitrary-order multivariate mixed partials, `mixed-partial`/
      `gradient-n` (P4); reverse-over-Taylor (P5); exact bignum/rational
      coefficient towers (P6); tensor-valued towers through
      `matmul`/`conv2d` (P7); validated Taylor models with interval-remainder
      bounds, `taylor-model` (P8); differentiable control flow (P9);
      checkpointed high-order reverse-mode (P10); tower-based user numerics,
      `taylor-ode-solve`/`taylor-root` (P11); sparse high-order tensors,
      `sparse-hessian` (P12). See [CHANGELOG.md](CHANGELOG.md), the
      [Automatic Differentiation guide](docs/guide/AUTOMATIC_DIFFERENTIATION.md),
      and `docs/AD_CAMPAIGN.md`. This closes ESH-0118 and supersedes the
      P1-v1.3.1 / P2-P3-v1.3.2 staging plan below — P4 through P12, originally
      spread across v1.4 through v2.0 in the Architecture Dependency Chain,
      shipped complete in this release instead.
- [x] **Full R7RS conformance on the portable corpus**: a new
      reference-Scheme differential oracle (P7a) reached 34/34 (100%) AGREE
      vs. chibi-scheme 0.12.0 on its 34-program corpus, fixing `apply` with
      leading arguments, multi-vector `vector-map`, quasiquoted vectors,
      `cond`/`case` `=>`, allocating `vector-copy` (incl. `#(...)` literals),
      the `error-object` family, `write` escaping, nested `syntax-rules`
      ellipsis, and 2-arg `substring` along the way.
- [x] **Robustness hardening**: proper mutual-tail-call TCO (AArch64),
      named-let TCO in every tail position (incl. through `guard`), the
      closure-capture ceiling raised 16→64, automatic per-iteration arena
      reclamation for bounded-RSS long-running loops, a shutdown-teardown
      race fix, a deep-recursion `SIGILL`-with-no-diagnostic fix, and
      transitive-dependency AOT/JIT cache invalidation.
- [x] **Permanent adversarial-testing infrastructure**: differential
      harness+fuzzer (P1), feature-pair edge matrix (P2), AD
      finite-difference oracle (P3), stress harness (P4), VM parity ratchet
      (P5), six depth-parametric sweep families (P6), and external oracles —
      reference-Scheme differential, sanitizer fuzzing, metamorphic-law
      checking (P7) — all wired into the ICC readiness oracle. See
      `docs/TESTING.md`.

Original v1.3.1/v1.3.2 staging plan for reference (superseded — all landed
in v1.3.0-evolve): P1 runtime `derivative^n` closes ESH-0118; P2
compile-time-K monomorphization and P3 JET8 subsumption gated by `ad-depth`
and `mono-equiv`.

---

## Development workstreams (v1.3.5 → v2.0)

**Re-dated 2026-08-24 (maintainer ruling R1, executed).** Every date from v1.4
onward in the previous published roadmap was stale — some already slipped,
the rest were not going to be hit at measured velocity (the v1.3.1→v1.3.4
line averaged roughly five weeks per point release, including hardening
waves). Rather than keep publishing dates the project would miss serially,
the ladder below is re-dated to what the shipped velocity supports: v2.0
moves from the previously published "Q1 2027" to **~Q4 2028**. The
per-version sections that follow, and the Release Timeline table, use the
re-dated ladder. Compression is possible (the v1.3.4 endgame proved
multi-lane parallel throughput), but the published dates should be ones the
project can hit.

Every release from v1.3.5 forward ships work from some mix of six standing
workstreams rather than a single theme:

- **W1 — Resident/DBSP spine.** `core.dbsp` incremental dataflow (shipped as
  a first slice in v1.3.3-evolve) grows toward a v1.5.0 GA and a unified
  `differentiate` primitive (`numeric` and `incremental` interpretations
  over the closed world) at v2.0.
- **W2 — Assurance.** The ADR-0010 gap ledger (A1-A13) closes on a
  per-version schedule, plus the adversarial-capability ramp: harness CI
  lanes, oracle/ledger schema checks, a documentation-truth ratchet, a SymPy
  external oracle, and a machine-checked-invariants track that begins with a
  Taylor-tower semantics proof sketch.
- **W3 — Performance.** Public, third-party-runnable benchmarks on Eshkol's
  own axes (exact-AD cost curves, flat-RSS resident loops) from v1.3.5
  onward, building toward native PGO (v1.5.0), a staged dense graph
  (v1.6.x), closed-world whole-program optimization (v1.8.1), and
  training-grade performance gates at v2.0 (>=80% of vendor-BLAS on
  GEMM-dominated staged throughput, 10k steps with no recompile, zero
  post-warmup allocations).
- **W4 — Codebase health.** One monolithic file decomposed per release
  behind a parity gate (`vm_run.c`, then `runtime_regions.cpp`, then
  `bignum.cpp`, then `vm_geometric.c`), shell-hardening, dead-code
  liveness sweeps, and a single semantic-tooling core underneath the
  compiler's own dev tools.
- **W5 — Interop & adoption.** The locked 2026-08-20 interop-first
  sequence: boundary exactness across the Python/NumPy edge, a
  silent-demotion CI gate, benchmarks on Eshkol's own axes, and a
  definition-of-done rule for every new AD/quantum feature (an
  external-oracle case plus a Python one-liner). ONNX/StableHLO export
  ships only once there is a training win worth exporting — not on a fixed
  date. **Amended negatives:** no chasing SciPy API parity, no
  ResNet/float64-training gates as an adoption bar; distributed computing
  is explicitly *not* one of these negatives (see W6).
- **W6 — Distributed computing (maintainer ruling 2026-08-20; two tiers).**
  Promoted to a first-class workstream, not an on-demand item. Eshkol
  already emits StableHLO and compiles/executes through XLA
  (`lib/backend/xla/`), but only single-device today — no PJRT, no
  sharding, no replicas in-tree. The thesis is differentiated, not
  parity-chasing: deterministic, exact, bitwise-reproducible distributed
  computing, with raw throughput kept honest by delegating to vendor
  collectives and vendor GEMM rather than rebuilding them.
  - **Tier 1 — scale (ride XLA).** Become a PJRT client; add sharding
    annotations to the staged dense graph so XLA's GSPMD partitioning,
    collectives, and multi-host machinery carry Eshkol at XLA-class
    distributed throughput on GPU/TPU clusters, as bridge work rather than
    runtime build-out.
  - **Tier 2 — truth (native mesh).** Exact-accumulation deterministic
    allreduce (i128/fixed-point), multi-node bit-identity as a mesh-CI
    parity gate (node count as a 4th parity axis), typed communication over
    the v1.4.0 sockets, no-GC tail latency — structurally unavailable to a
    pure-XLA client, since XLA re-associates reductions by design.
  - One workload dials between the two tiers (fast vs. exact) — "exactness
    is an axis," applied to distribution. Staging: v1.4.0 PJRT client spike
    + XLA multi-device single-host + native collectives over sockets;
    v1.5.0 Tier-1 data-parallel + Tier-2 mesh bit-identity gate; v1.6.x
    sharding annotations on the staged dense graph -> GSPMD multi-host +
    distributed DBSP; v1.8.x fault tolerance/elasticity; v2.0 gates per
    tier (Tier 1 >=85% scaling efficiency at 8 devices; Tier 2
    bit-identical gradients at any node count, zero post-warmup
    allocations per rank).

---

## v1.3.5 — the consolidation release (target: late Sep 2026) - PLANNED

**Flagship: SHIPPED (#461).** VM OALR Stage-1 evacuator port (SW-14
ruling) — the full heap-tag space deep-walked on the bytecode VM (a
compile-time-checked 33-wide table over the 28 `HeapType` members plus
the manifold-tag macros and unassigned slots), poison and flat-RSS
validation, so `with-region` reclaims on the VM the way it already does
on native codegen. Re-measured for this documentation wave against a
from-source build of the merge commit (`487c2a62`): flat 25-27 MB across
1,000/4,000/16,000 iterations of the same fixture vs. 793 MB with the
evacuator disabled and 704 MB for an unwrapped control — see
[docs/breakdown/RUNTIME_CONFIGURATION.md](docs/breakdown/RUNTIME_CONFIGURATION.md#bytecode-vm-region-reclamation).
The user-reachable region **handle** surface (`region-open`/`region-close`)
remains bookkeeping-only on the VM (Stage-2, not yet scheduled to a
release).

- Interop wave 1: **H1 NumPy capsule-lifetime fix, SHIPPED (#458)** — the
  Python bindings' zero-copy tensor array now holds a strong reference to
  its owning `Context` via its NumPy capsule, so the array stays valid past
  the `Context` object's own lifetime (closed `.icc/silent-wrong-ledger.yaml`
  SW-44). See [docs/reference/bindings/python.md](docs/reference/bindings/python.md).
  The separate exactness-across-the-Python-boundary design doc remains a
  v1.4.0-connection item (implementation, not this fix).
- AD: SW-05 forward-over-reverse; ESH-0101 (recursion-depth guard coverage
  for top-level `define`d self-recursive functions, maintainer ruling
  2026-08-13 — see KNOWN_ISSUES.md); P6/P11 exact-coefficient and
  user-numerics re-cut on post-P5 master (verify exact-coefficient and
  reverse-Taylor suites together — both were shipped complete in
  v1.3.0-evolve and this is a re-verification pass, not new staging).
- Correctness debt: the `(or X null)` miscompile lineage (#229), the REPL
  no-return verifier (#244), and an `ArithmeticCodegen::mod` srem-vs-modulo
  audit.
- Assurance (W2, ADR-0010 v1.3.5 set): CI lanes for previously-unwired
  harnesses, oracle/ledger schema checks in CI (`completion-oracles.yaml`
  parse + ID-uniqueness), a self-verdict scanner, build fingerprints, and
  ICC adversarial eval scenarios (dirty worktrees, stale artifacts,
  model-server outage, disk pressure, failed gates).
- Performance (W3): benchmarks-on-our-axes wave 1, published and
  reproducible (exact-AD cost curves; flat-RSS resident loops).
- Codebase (W4): decompose `vm_run.c` (the file every VM fix touches); the
  docs-only CI-context fix; KNOWN_ISSUES version targets re-pinned; this
  ROADMAP re-dated.

---

## v1.4.0-connection — the systems profile (target: Nov 2026) - PLANNED

**Focus:** A resource-sound systems profile — connect to the outside world
under the same discipline that made `Qubit` linear.

- [ ] TCP/UDP sockets with linear resource types (guaranteed close)
- [ ] TLS/SSL via system libraries
- [x] Non-blocking I/O with event loop (kqueue / epoll / IOCP) - SHIPPED in v1.3.4-evolve
- [ ] Unix domain sockets for local IPC
- [ ] HTTP client/server and WebSocket, on the shipped event loop
- [ ] Linear types for all handles: `open → borrowed → closed` with compile-time tracking
- [ ] Borrow pattern for temporary resource access
- [ ] Agent runtime unblocked: SSE streaming, subprocess pipe contract,
      durable session persistence
- [ ] W5 interop wave 2: exactness across the Python/NumPy boundary and a
      silent-demotion CI gate; the definition-of-done rule (external-oracle
      case + Python one-liner per new AD/quantum feature) goes live
- [ ] qLLM backward completion: `input2` wiring for conv2d/batchnorm/
      layernorm/attention tape nodes
- [ ] Assurance: ADR-0010 v1.4 set (A10-A13), TSan-required lane, SymPy
      oracle pilot on the exact-AD surface
- [ ] Performance: benchmarks wave 2 (Ozaki CRT vs. cuBLAS/Accelerate,
      accuracy and throughput, pinned hardware)
- [ ] Codebase: native image I/O dependency removal; decompose
      `runtime_regions.cpp`; doc-truth ratchet phase 1 (rank + quota, not
      yet release-blocking)
- [ ] W6 distributed: PJRT client spike + XLA multi-device single-host +
      native collectives over sockets

---

## v1.4.1 — the ABI release (target: Dec 2026) - PLANNED

- [ ] OALR ABI v2 (32-byte header, layout descriptors, escape ledgers,
      transfer capsules); portable tail transfer (musttail + bounded-stack)
- [ ] PGO: canonical training workload + one-shot `llvm-profdata` merge
- [ ] Codebase: decompose `bignum.cpp`; shell-hardening epic wave 1

---

## v1.5.0-intelligence (target: Q1 2027) - PLANNED

**Focus:** Neural and symbolic computation flow bidirectionally.

Informed by the [Neuro-Symbolic Architecture](docs/future/NEURO_SYMBOLIC_COMPLETE_ARCHITECTURE.md).

**Flagship:** `core.dbsp` GA (W1) + native-product PGO in the release
workflow (ADR-0007 Phase 1).

- [ ] Symbol embeddings (learnable vector representations of KB symbols)
- [ ] Soft unification (differentiable similarity — gradients flow through matching)
- [ ] LSTM and GRU cells (standard recurrent neural architectures)
- [ ] Differentiable logic programs (gradients flow through rule application)
- [ ] Attention over knowledge base (neural query mechanism over symbolic facts)
- [ ] Gradient estimators for discrete operations (Gumbel-Softmax, straight-through)
- [ ] Noesis M2 surface: HNSW, BPE, sparse tensors, int/complex tensors
- [ ] Assurance: A6 + A8 full race matrix; SymPy oracle becomes a release
      gate; machine-checked-invariants ramp begins (Taylor-tower semantics
      proof sketch)
- [ ] Codebase: decompose `vm_geometric.c`; ADR-0008 tooling core lands
      (`eshkol check` + LSP on a shared workspace-analysis core)
- [ ] W6 distributed: Tier-1 data-parallel + Tier-2 mesh bit-identity gate

Note: the arbitrary-order AD substrate this section used to stage here
(P5/P7/P9/P10) shipped complete in v1.3.0-evolve, ahead of this plan — see
[Arbitrary-order AD track](#arbitrary-order-ad-taylor-tower-track----shipped-ahead-of-schedule)
below.

---

## v1.5.1 (target: Q1-Q2 2027) - PLANNED

- [ ] DBSP circuits + resident A/B sessions with a hard steady-state bound
- [ ] Doc-truth gate becomes release-blocking (ratchet reaches zero
      unsupported claims in release-critical docs)

---

## v1.6.0-reasoning (target: Q2 2027) - PLANNED

**Focus:** Make the logic engine production-grade.

- [ ] Backward chaining inference (Prolog-style goal-directed proof search with backtracking)
- [ ] Forward chaining inference (production rules with fixed-point derivation)
- [ ] Constraint solving (finite domain constraints, SAT solver integration)
- [ ] Knowledge graphs (RDF-style triple store with SPO/POS/OSP indexing)
- [ ] Knowledge graph embeddings (entity-relation-entity triples as learnable vectors)
- [ ] Staged AD ABI (ADR-0002b Phase G): the dense primitive registry as a
      real table, cotangent-layout and error ABI first-class, strict-mode
      kernel flag
- [ ] W6 distributed: sharding annotations on the staged dense graph ->
      GSPMD multi-host + distributed DBSP

Note: sparse high-order AD tensors (P12), originally staged here, shipped
complete in v1.3.0-evolve.

---

## v1.6.1 (target: Q3 2027) - PLANNED

- [ ] DBSP traces + staged scratch plan (ADR-0007 Phase 2: staged dense
      graph + static memory plan)
- [ ] Region-safety machine-checked-invariant work begins

---

## v1.7.0-synthesis (target: Q3-Q4 2027) - PLANNED

**Focus:** Programs that write and improve programs.

- [ ] Neural-guided program search (beam search with neural scoring for candidate ranking)
- [ ] Type-directed synthesis holes (`??` syntax — compiler searches for well-typed completions)
- [ ] Graph Neural Networks (message passing, neighborhood aggregation, graph attention)
- [ ] Synthesis from input-output examples (inductive programming)
- [ ] Neural theorem provers (neural heuristic guides symbolic proof search, using v1.5 embeddings + v1.6 chaining)
- [ ] Recursive IVM; staged optimizer; program-capsule foundations (ADR-0005)

---

## v1.8.0-platform (target: Q4 2027) - PLANNED

**Focus:** Eshkol runs on everything, controls everything.

Informed by the [Multimedia System Architecture](docs/future/MULTIMEDIA_SYSTEM_ARCHITECTURE.md).

- [ ] Cross-platform windowing (X11/Wayland, Cocoa, Win32)
- [ ] Event system (keyboard, mouse, touch, window events)
- [ ] Real-time audio (CoreAudio, ALSA, WASAPI with callback-based I/O)
- [ ] MIDI input/output for instrument control
- [ ] Vulkan Compute for cross-platform GPU (beyond Metal/CUDA)
- [ ] Embedded cross-compilation (ARM bare-metal, RISC-V)
- [ ] `core.memory` as a Z-set; resident recurrent AD

Multi-GPU / multi-node dispatch is no longer gated behind "demonstrated
demand" here — it is W6's ladder (Tier 1 PJRT/XLA multi-device lands at
v1.4.0; GSPMD multi-host at v1.6.x). This section keeps only the
single-machine platform surface.

---

## v1.8.1 (target: Q1 2028) - PLANNED

- [ ] Resident-agent circuit pilot
- [ ] Closed-world whole-program optimization (ADR-0007 Phase 3)
- [ ] W6 distributed: fault tolerance / elasticity (checkpoint/restart
      pulled forward from what was v1.9.1 below)

---

## v1.9.0-types (target: Q1-Q2 2028) - PLANNED

**Focus:** The type system becomes a proof system.

- [ ] Full dependent type enforcement (compile-time errors, not just warnings)
- [ ] Refinement types (`(Refine Integer (> x 0))` with SMT solver integration)
- [ ] Effect types (tracking `Pure`, `IO`, `State`, `Exception` at the type level)
- [ ] Algebraic effects and handlers (structured side-effect management)
- [ ] Row polymorphism for records (structural subtyping)
- [ ] Higher-rank types (rank-2 polymorphism for combinators)
- [ ] Session types for communication protocols
- [ ] `IncrementalizePass` (DBSP incrementalization as a compiler pass)

---

## v1.9.1 (target: Q2 2028) - PLANNED

- [ ] Checkpoint/restart; session protocol
- [ ] AD-aware debugger/profiler (inspect dual numbers, reverse tape,
      Taylor coefficients, region lifetimes) on the ADR-0008 execution core

---

## v1.9.2 (target: Q3 2028) - PLANNED

- [ ] Spill tier
- [ ] Reflective self-modification (capsule `psi_program` updates)

---

## v2.0-starlight (target: Q4 2028) - RESEARCH

**Focus:** Quantum computing meets formal verification, and the AD/DBSP
lines converge into one primitive.

Leverages OALR linear types (no-cloning theorem) and AD (variational circuits).

### Unified differentiation
- [ ] Unified `differentiate` primitive: `numeric` and `incremental`
      interpretations over the closed world (W1 endpoint)
- [ ] Typed-static-reverse (#216) north-star work begins only after the
      resident tape (#214) has a public training win

### Quantum Type System
- [x] Qubit type with linear resource tracking (no-cloning enforced at compile time) - SHIPPED in v1.3.4-evolve
- [ ] Quantum register types `qreg<n>` with compile-time dimension
- [ ] `define-quantum-region` scoping for qubit allocation and deallocation
- [ ] Quantum region compilation, QAOA — on the quantitative types from v1.9

### Quantum Operations
- [x] Gate primitives: H, CNOT, Rz, T, S, SWAP, Toffoli, arbitrary unitaries - SHIPPED in v1.3.3-evolve
- [x] Measurement with classical outcome - SHIPPED in v1.3.3-evolve
- [ ] Circuit compilation and optimization (gate fusion, qubit mapping)
- [x] AD integration for variational algorithms - SHIPPED in v1.3.3-evolve (custom-VJP tape nodes carry Moonlab's exact adjoint)

### Hybrid Classical-Quantum
- [x] Variational Quantum Eigensolver (VQE) - SHIPPED in v1.3.3-evolve
- [ ] Quantum Approximate Optimization Algorithm (QAOA)
- [x] Quantum machine learning (parameterized circuits with AD) - SHIPPED in v1.3.3-evolve
- [x] Integration with Moonlab quantum simulator - SHIPPED in v1.3.3-evolve; pinned to Moonlab v1.2.0 in v1.3.4-evolve

### Performance gates (W3 endpoint)
- [ ] >=10k steps with no recompile, 1 primal + 1 reverse pass, zero
      post-warmup allocations, GEMM-dominated staged throughput >=80% of
      native vendor-BLAS; application/kernel IR PGO (ADR-0007 Phase 4)

### Distributed gates (W6 endpoint)
- [ ] Tier 1: >=85% scaling efficiency at 8 devices
- [ ] Tier 2: bit-identical gradients at any node count, zero post-warmup
      allocations per rank

### Formal Verification
- [ ] Integration with proof assistants (Lean) for certified compilation —
      the Lean kernel export re-checks the compiler on the normative
      corpus, and "HoTT-inspired" is retired as a claim in favor of this
      concrete, checkable one (ADR-0004)
- [ ] Quantitative type theory for unified linear/quantum resource tracking
- [ ] Lean-certified formal verification of the validated-AD Taylor models
      (P8) already shipped in v1.3.0-evolve (`taylor-model`, `tm-range`,
      `tm-eval`) — proving the interval-remainder enclosures sound and
      order-tightening, beyond the current dense-sampling remainder
      estimate

---

## Release Timeline

| Version | Date | Theme | Key Deliverables |
|---------|------|-------|-----------------|
| **v1.1.13** | Apr 2026 | Accelerate | Windows ARM64, 16-lane release matrix, VM closure fixes, mobile site |
| **v1.2** | May 2026 | Scale | Model serialization, Python bindings, image I/O |
| **v1.3.0-evolve** | Jul 2026 | Evolve | **SHIPPED.** R7RS libraries, string interpolation; arbitrary-order AD **P0–P12 complete** (Taylor towers, exact coefficients, GUW multivariate, reverse-over-Taylor, tensor towers, Taylor models, sparse tensors — closes ESH-0118, delivered ahead of the original P1-only plan); full R7RS conformance (34/34 vs. chibi-scheme); TCO/closure/memory robustness hardening; permanent adversarial-testing infrastructure |
| **v1.3.1 → v1.3.4-evolve** | Jul-Aug 2026 | Evolve | **SHIPPED 2026-08-19** (tag `v1.3.4-evolve`, commit `694c3179`). v1.3.1: flat memory for resident/daemon loops, iterative reader. v1.3.2: thread-safe regions, deeper evacuation. v1.3.3: opt-in differentiable quantum computing (Moonlab VQE/CHSH), ML-KEM post-quantum crypto, `core.dbsp` incremental dataflow, 100% executable language coverage. v1.3.4: automatic per-iteration reclamation matching explicit regions (ESH-0214e), race-free `parallel-map`, exact gradients through every callable form, shortest-round-trip float printing, checked `(the <type> expr)` ascription + predicate narrowing, linear `Qubit`, high-precision numerics (Ozaki-II exact/fast GEMM, mixed-precision `linear-solve`, `i128`), Moonlab v1.2.0 (QGT/QNG), full hosted-VM tensor-matmul parity. Plus the consumer-hardening correctness wave: fatal compile diagnostics, tag-decided exactness on both engines, exact-point differentiation, same-unit `define-library` on all three back ends, a real `--shared-lib` (#377), the portable event loop, the fixed-point/`i128` accumulation engine, region handles, **the qLLM bridge implementation (#386/#392 — the completion the v1.1 line above claimed early)**, and embedding/Fréchet-mean backward passes. **Release gates** (RELEASE_NOTES.md, measured on the release cut): aggregate suite 45/45 suites / 770 tests; CTest 183/183; executable language coverage 1,091/1,091 (100.0%); SICP full-book gate 88/88; reference-Scheme differential 34/34 AGREE vs. chibi-scheme 0.12.0; VM parity differential 184/184 over a 956-row manifest; qLLM oracle gate 10/10; ICC readiness 100, verdict `ready` |
| **v1.3.5** | late Sep 2026 | Consolidation | VM OALR Stage-1 evacuator, **SHIPPED (#461)**; H1 Python-bindings capsule-lifetime fix, **SHIPPED (#458)**; assurance wave 1 (ledger-integrity/oracle-schema gates), **SHIPPED (#454)**; docs-only CI fix, **SHIPPED (#455)**; AD re-verification wave; correctness debt (#229/#244/mod-srem); W3 benchmarks wave 1; W4 `vm_run.c` decomposition — see "Development workstreams" above |
| **v1.4.0-connection** | Nov 2026 | Systems profile | TCP/UDP/TLS, Unix sockets, HTTP/WebSocket, linear resource types; W5 interop wave 2; W6 PJRT spike *(AD substrate P4/P6/P11 already delivered in v1.3.0-evolve, ahead of schedule)* |
| **v1.4.1** | Dec 2026 | ABI | OALR ABI v2, portable tail transfer, PGO training workload, `bignum.cpp` decomposition |
| **v1.5.0-intelligence** | Q1 2027 | Intelligence | `core.dbsp` GA, native PGO, Noesis M2 surface, symbol embeddings, differentiable logic, LSTM/GRU; W6 Tier-1 data-parallel + Tier-2 mesh bit-identity gate *(high-order AD P5/P7/P9/P10 already delivered in v1.3.0-evolve, ahead of schedule)* |
| **v1.5.1** | Q1-Q2 2027 | — | DBSP circuits, resident A/B sessions, doc-truth gate becomes release-blocking |
| **v1.6.0-reasoning** | Q2 2027 | Reasoning | Backward/forward chaining, constraint solving, knowledge graphs, staged AD ABI; W6 GSPMD multi-host *(sparse high-order AD tensors P12 already delivered in v1.3.0-evolve, ahead of schedule)* |
| **v1.6.1** | Q3 2027 | — | DBSP traces + staged scratch plan, region-safety machine-checked-invariant work begins |
| **v1.7.0-synthesis** | Q3-Q4 2027 | Synthesis | Neural-guided search, program synthesis, GNN, recursive IVM |
| **v1.8.0-platform** | Q4 2027 | Platform | Windowing, audio, Vulkan, embedded targets, `core.memory` as Z-set |
| **v1.8.1** | Q1 2028 | — | Resident-agent circuit pilot, closed-world WPO, W6 fault tolerance/elasticity |
| **v1.9.0-types** | Q1-Q2 2028 | Types | Dependent types, effects, algebraic effects, session types, `IncrementalizePass` |
| **v1.9.1** | Q2 2028 | — | Checkpoint/restart, session protocol, AD-aware debugger/profiler |
| **v1.9.2** | Q3 2028 | — | Spill tier, reflective self-modification |
| **v2.0-starlight** | Q4 2028 | Starlight | Unified `differentiate` primitive, quantum region compilation, QAOA, formal verification (Lean kernel export); training-grade performance gates; W6 gates per tier |

> **Re-dating note (maintainer ruling R1, executed 2026-08-24):** every date
> from v1.4 onward supersedes the previously published table. The previous
> table's dates (v1.4 "Jul 2026" through v2.0 "Q1 2027") were not going to be
> hit at measured velocity; this table is deliberately coarser and
> velocity-anchored instead. See
> the "Development workstreams" section above for the
> six workstreams every release now draws from, and the point-release rows
> (v1.4.1, v1.5.1, v1.6.1, v1.8.1, v1.9.1, v1.9.2) for the finer-grained
> staging.

> **Arbitrary-order AD (Taylor-tower) track — SHIPPED.** Phases P0–P12 were
> originally planned to thread through the version themes above as enabling
> substrate; instead all 13 phases landed complete in v1.3.0-evolve. See
> [`docs/AD_CAMPAIGN.md`](docs/AD_CAMPAIGN.md) for the as-planned phase →
> version → ICC-gate map and [CHANGELOG.md](CHANGELOG.md) for the as-shipped
> detail. Each phase was gated by an `ad-*` ICC oracle criterion.

---

## Component Status

### Core Compiler
- [x] Parser - Complete
- [x] Type Checker - Complete
- [x] LLVM Backend - Complete (34,928 lines)
- [x] Module System - Complete
- [x] Macro System - Complete

### Automatic Differentiation
- [x] Forward Mode - Complete
- [x] Reverse Mode - Complete
- [x] Nested Gradients - Complete (32-level tape stack)
- [x] Vector Calculus - Complete (8 operators)
- [x] Arbitrary-Order Taylor Towers (v1.3.0-evolve) - Complete (P0-P12: exact
      coefficients, GUW multivariate, reverse-over-Taylor, tensor towers,
      validated Taylor models, sparse tensors, differentiable control flow,
      checkpointed reverse, tower-based numerics)

### Memory Management
- [x] Arena Allocation - Complete
- [x] OALR System - Complete
- [x] Ownership Tracking - Complete
- [x] Escape Analysis - Complete

### Standard Library (v1.1)
- [x] Core Functions (60+ list ops, 30+ string utils) - Complete
- [x] Math Library (linear algebra, statistics, ODE solvers) - Complete
- [x] Signal Processing (FFT, filters, window functions) - Complete
- [x] ML Library (optimizers, activations, normalization) - Complete
- [x] Web Platform (80+ DOM API functions, WASM target) - Complete
- [x] JSON/CSV/Base64 Support - Complete

### Development Tools
- [x] REPL with JIT (stdlib preloading, cross-eval persistence) - Complete
- [x] Compiler (eshkol-run, AOT + script mode) - Complete
- [x] Package Manager (eshkol-pkg, TOML manifest) - Complete
- [x] LSP Server (diagnostics, completion, hover) - Complete
- [x] VSCode Extension (syntax highlighting, LSP client) - Complete
- [x] Test Suite (45 suites, 770 tests) - Complete

### v1.1-accelerate (Complete)
- [x] XLA Backend (StableHLO/MLIR + LLVM-direct) - Complete
- [x] GPU Acceleration (Metal SF64 + CUDA cuBLAS) - Complete
- [x] SIMD Vectorization (SSE/AVX/NEON) - Complete
- [x] Parallel Primitives (work-stealing thread pool) - Complete
- [x] Exact Arithmetic (bignums + rationals + complex) - Complete
- [x] Consciousness Engine (logic, inference, workspace — 22 builtins) - Complete
- [x] ML Framework (75+ builtins: activations, losses, optimizers, CNN, transformers) - Complete
- [x] Signal Processing (FFT, filters, window functions) - Complete
- [x] R7RS Extensions (call/cc, dynamic-wind, bytevectors) - Complete

### v1.1.12 Additions
- [x] Production Bytecode VM (555+ builtins, 176/176 tests, dual number AD) - Complete
- [x] eshkol.ai Website (Eshkol→WASM, browser REPL, interactive tutorials) - Complete
- [x] GitHub Pages Deployment - Complete

### v1.1.13 Additions
- [x] Native Windows ARM64 (VS 2022 + ClangCL + LLVM 21 aarch64) - Complete
- [x] 16-lane release matrix with per-arch LLVM SDK caching - Complete
- [x] VM closure bug fixes (named-let nested closure PC + native 252 upvalue relay) - Complete
- [x] Windows setjmp hardening (x64 frameaddress, ARM64 sponentry) - Complete
- [x] Mobile-responsive website + browser REPL error display - Complete

### Planned (v1.3.5+)
- [x] Model Serialization + Python Bindings — v1.2 (shipped)
- [x] R7RS Library System + String Interpolation + arbitrary-order AD — v1.3.0-evolve (shipped)
- [x] VM region evacuator (with-region reclaims on the bytecode VM) — v1.3.5 (shipped, #461)
- [ ] Networking + Linear Resource Types — v1.4.0-connection
- [ ] Distributed computing, two-tier (W6: PJRT/XLA scale + native-mesh exact
      allreduce) — spike at v1.4.0, gates at v2.0 (no longer gated behind
      "demonstrated demand"; see the "Development workstreams" section above)
- [ ] Neuro-Symbolic Bridge — v1.5.0-intelligence
- [ ] Backward Chaining + Knowledge Graphs — v1.6.0-reasoning
- [ ] Program Synthesis + Neural Search — v1.7.0-synthesis
- [ ] Platform Abstraction (windows, audio, embedded) — v1.8.0-platform
- [ ] Advanced Type Theory (dependent, effects, algebraic) — v1.9.0-types
- [ ] Quantum Computing + Formal Verification — v2.0-starlight

---

## Research Directions

**Active Research:**
- Polyhedral optimization for nested tensor loops
- Linear type systems for hardware resources — shipped for `Qubit` in
  v1.3.4-evolve; generalizing to sockets/handles in v1.4.0-connection
- Neuro-symbolic bridging — differentiable symbolic operations (v1.5.0)
- Effect systems for purity tracking and algebraic effects (v1.9.0/v2.0)
- Exact, bitwise-reproducible distributed computing (W6 Tier 2: i128/
  fixed-point deterministic allreduce, multi-node bit-identity as a mesh-CI
  parity gate) — differentiated from throughput-only distributed ML, not a
  parity chase against it

**Exploratory Research:**
- Quantum machine learning — AD through parameterized quantum circuits
- Probabilistic programming with exact inference via factor graphs
- Formal verification of automatic differentiation correctness
- Hardware-software co-design for quantum-classical hybrid systems
- Self-improving programs via gradient descent on code embeddings

---

## Community Engagement

**Open Source Development:**
- GitHub repository with MIT license
- Active issue tracking and PR reviews
- Quarterly release cycle
- Community contribution guidelines

**Academic Partnerships:**
- University curriculum integration
- Research collaborations
- Conference presentations
- Student project sponsorship

**Enterprise Support:**
- Professional consulting
- Custom feature development
- Training and workshops
- Priority support

---

## How to Contribute

We welcome contributions in all areas:

**Core Development:**
- Implement planned features
- Optimize existing code
- Fix bugs and issues
- Improve test coverage

**Research:**
- Explore new AD techniques
- Investigate type system extensions
- Study memory management innovations
- Publish findings

**Documentation:**
- Improve user guides
- Write tutorials
- Create examples
- Update specifications

**Ecosystem:**
- Develop libraries
- Create tools
- Build integrations
- Share use cases

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

---

*Last Updated: August 2026 (v1.3.5 documentation wave — re-dated ladder,
six standing workstreams, distributed computing promoted to W6)*

*The arbitrary-order automatic-differentiation (Taylor-tower) campaign — phases
P0–P12, spanning core high-order AD, exact-coefficient and tensor-valued towers,
GUW multivariate recovery, differentiable control flow, and validated Taylor
models — shipped complete in v1.3.0-evolve rather than being threaded through
the version themes above as originally planned. It is the
differentiable-programming substrate for the neuro-symbolic (v1.5.0–v1.7.0) and
quantum/formal-verification (v2.0) arc, and its successor, the unified
`differentiate` primitive (W1), is the v2.0 endpoint. See
[`docs/AD_CAMPAIGN.md`](docs/AD_CAMPAIGN.md).*

*Eshkol v1.1-accelerate is complete with 47/47 roadmap items delivered plus the v1.1.12 and v1.1.13 additions (production VM, web platform, browser AD, Windows ARM64, mobile site). The v1.3 line shipped complete through v1.3.4-evolve (tagged 2026-08-19). The roadmap progresses through data & deployment (v1.2-scale), language maturity (v1.3-evolve), consolidation (v1.3.5), networking & resources (v1.4.0-connection), the ABI release (v1.4.1), neuro-symbolic intelligence (v1.5.0-intelligence), symbolic reasoning (v1.6.0-reasoning), program synthesis (v1.7.0-synthesis), platform & hardware (v1.8.0-platform), advanced type theory (v1.9.0-types), and quantum computing with formal verification (v2.0-starlight) — with a two-tier distributed-computing workstream (W6) running underneath the whole v1.4.0→v2.0 span rather than confined to one release.*
