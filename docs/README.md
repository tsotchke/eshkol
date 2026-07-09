# Eshkol Documentation Index

This page is the hub for every document under `docs/` (and the adjacent
`examples/`, `tests/`, `tools/`, `press/`, and `nix/` documentation). If you
followed a link from the repository root [README.md](../README.md), you are
one hop away from all of it. Start here if you are not sure where to look.

Sections below group documents by topic rather than by directory, because a
few directories (`docs/breakdown/`, `docs/reference/*/`, `docs/platform/`)
already contain their own local index page — this hub links to those index
pages and lets them fan out to their siblings.

## Getting Started

- [QUICKSTART](QUICKSTART.md) — 15-minute hands-on tutorial (lists to closures to tensors to gradients)
- [Tutorials index](tutorials/README.md) — all 27 tutorials, from first program through full projects
- [First 5 Minutes](tutorials/00_FIRST_5_MINUTES.md) — install, hello world, five wow moments
- [Examples](../examples/README.md) — thirteen runnable programs (AD, parallelism, consciousness engine, streaming, simulation)
- [FAQ](FAQ.md) — installation, troubleshooting, common questions
- [Getting Started (breakdown)](breakdown/GETTING_STARTED.md) — installation and first programs, implementation-level detail
- [Overview](breakdown/OVERVIEW.md) — design philosophy and competitive positioning, start here for "why Eshkol"

## Language Guide & Reference

- [Language Guide](ESHKOL_LANGUAGE_GUIDE.md) — tutorial-style introduction to the language
- [Language Reference index](reference/language/INDEX.md) — complete, example-verified function and syntax reference (binding/mutation, control flow, error handling, pattern matching, modules, continuations, and more)
- [Complete Language Specification](COMPLETE_LANGUAGE_SPECIFICATION.md) — full technical specification, v1.3.0-evolve
- [Quick Reference](ESHKOL_QUICK_REFERENCE.md) — one-page cheat sheet, 555+ builtins
- [API Reference](API_REFERENCE.md) — comprehensive function documentation
- [Standard Library API index](reference/stdlib/INDEX.md) — module-by-function map of the standard library (58 modules, 638 symbols)
- [Standard Library API (v1.2 surface notes)](STDLIB_V1_2_API.md) — stdlib module surfaces, including infrastructure modules
- [Design Document](DESIGN.md) — v1.3.0-evolve design document
- [Feature Matrix](FEATURE_MATRIX.md) — implementation status table
- [Known Issues](KNOWN_ISSUES.md) — current limitations and v1.3+ items
- [Scheme Compatibility](breakdown/SCHEME_COMPATIBILITY.md) — R7RS compliance and migration guide
- [Function Composition](breakdown/FUNCTION_COMPOSITION.md) — closures, lambda registry, mutable captures
- [Long-Running Loops](LONG_RUNNING_LOOPS.md) — tail-call and loop behaviour for long-lived processes

## Automatic Differentiation

- [Automatic Differentiation guide](guide/AUTOMATIC_DIFFERENTIATION.md) — the full v1.3 AD surface: arbitrary order, exact, validated, tensor, sparse, checkpointed, example-driven
- [AD reference index](reference/ad/INDEX.md) — machine-verified operator reference, fans out to:
  - [operators.md](reference/ad/operators.md) — every AD operator, signatures, capture and composition rules
  - [architecture.md](reference/ad/architecture.md) — forward jet, reverse tape, mixed reverse-over-forward mode
  - [support-matrix.md](reference/ad/support-matrix.md) — the AD-oracle support matrix and open bugs
- [AD Taylor-Tower design](design/AD_TAYLOR_TOWER.md) — arbitrary-order AD design (forward/Phase-0 POC, P1-P12 ledger)
- [Automatic Differentiation (breakdown)](breakdown/AUTODIFF.md) — mathematical foundations and implementation (symbolic, forward, reverse modes)
- [AD Campaign](AD_CAMPAIGN.md) — campaign tracking for the AD adversarial-testing pillars

## VM, REPL, and Runtime

- [Runtime reference index](reference/runtime/INDEX.md) — fans out to:
  - [eshkol-run](reference/runtime/eshkol-run.md) — the compiler and JIT driver, every CLI flag
  - [eshkol-repl](reference/runtime/eshkol-repl.md) — interactive REPL and the `--machine` EREPL protocol
  - [eshkol-vm-standalone](reference/runtime/eshkol-vm-standalone.md) — the bytecode VM and ESKB binary format
  - [environment-variables.md](reference/runtime/environment-variables.md) — the full user-facing environment variable set
  - [memory-model.md](reference/runtime/memory-model.md) — tagged values, arena allocator, `with-region` semantics
  - [parallelism.md](reference/runtime/parallelism.md) — `parallel-map`/`-fold`/`-filter`/`-execute`, work-stealing pool
  - [jit-internals.md](reference/runtime/jit-internals.md) — run cache, stdlib object cache, large code model
- [Bytecode VM (breakdown)](breakdown/BYTECODE_VM.md) — 64-opcode core interpreter, ESKB format, WebAssembly execution
- [REPL JIT (breakdown)](breakdown/REPL_JIT.md) — LLVM OrcJIT, opt-level matching, stdlib preloading, hot reload
- [VM Parity](VM_PARITY.md) — bytecode-VM vs native-codegen parity ratchet
- [VM Memory Ops as Weight Matrices](breakdown/VM_MEMORY_OPS_AS_WEIGHT_MATRICES.md) — lifting VM opcodes into transformer weight constructions
- [Computable Transformer](breakdown/COMPUTABLE_TRANSFORMER.md) — programs as neural network weight matrices, 6-layer transformer VM encoding
- [Command-Line Reference](breakdown/COMMAND_LINE_REFERENCE.md) — complete flag reference for `eshkol-run` and `eshkol-repl`
- [Runtime Configuration](breakdown/RUNTIME_CONFIGURATION.md) — environment variables, config files, resource limits

## Tensors and Machine Learning

- [Tensors reference index](reference/tensors/INDEX.md) — fans out to:
  - [creation.md](reference/tensors/creation.md) — `vector` vs `tensor`, literals, dtypes
  - [operations.md](reference/tensors/operations.md) — shape ops, linear algebra, conv/pooling, attention, activations
  - [gpu.md](reference/tensors/gpu.md) — GPU dispatch status, cost-model threshold
  - [ml-modules.md](reference/tensors/ml-modules.md) — `ml.optimization`, `core.manifold`, `signal.fft`
- [Machine Learning (breakdown)](breakdown/MACHINE_LEARNING.md) — 75+ ML builtins: activations, losses, optimizers, CNN layers, transformer ops
- [Vector Operations (breakdown)](breakdown/VECTOR_OPERATIONS.md) — Scheme vectors vs. tensors
- [Signal Processing (breakdown)](breakdown/SIGNAL_PROCESSING.md) — FFT/IFFT, window functions, FIR/IIR filters
- [Quantum RNG (breakdown)](breakdown/QUANTUM_RNG.md) — quantum-inspired random number generation

## Numeric Tower

- [Exact Arithmetic (breakdown)](breakdown/EXACT_ARITHMETIC.md) — bignum/rational numeric tower, overflow detection, R7RS exactness
- [Numeric tower (language reference)](reference/language/numeric-tower.md) — user-facing numeric tower reference
- [Math Standard Library (breakdown)](breakdown/MATH_STDLIB.md) — special functions, ODE solvers, statistics, PRNG

## Memory Management (OALR)

- [Memory Management (breakdown)](breakdown/MEMORY_MANAGEMENT.md) — OALR system, arena allocation, linear types, reference counting
- [Memory model (runtime reference)](reference/runtime/memory-model.md) — tagged values, arena allocator, `with-region`
- [core.memory faculty](reference/agent/memory-faculty.md) — content-addressed, CRDT-merged event log

## Macros and Modules

- [Macros tutorial](tutorials/10_MACROS.md) — pattern matching with ellipsis, hygienic expansion
- [Module System (breakdown)](breakdown/MODULE_SYSTEM.md) — `require`/`provide`, precompiled stdlib, `LinkOnceODR` override
- [Module system (stdlib reference)](reference/stdlib/module-system.md) — `require` resolution and `stdlib.o` precompilation
- [Special forms (language reference)](reference/language/special-forms.md)
- [Pattern matching (language reference)](reference/language/pattern-matching.md)
- [Quote and quasiquote (language reference)](reference/language/quote-and-quasiquote.md)

## Agent FFI

- [Agent & FFI reference index](reference/agent/INDEX.md) — fans out to:
  - [ffi.md](reference/agent/ffi.md) — the `extern`/`:real` declaration syntax, C ABI mapping, AOT linking
  - [capabilities.md](reference/agent/capabilities.md) — the capability allow-list policy
  - [http.md](reference/agent/http.md) — HTTP client and server
  - [sqlite.md](reference/agent/sqlite.md) — embedded SQLite
  - [subprocess.md](reference/agent/subprocess.md) — process spawning, ownership/cleanup contract
  - [crypto.md](reference/agent/crypto.md) — SHA-256, HMAC, random bytes, UUIDv4, base64url
  - [eagle.md](reference/agent/eagle.md) — native linear-head training
  - [memory-faculty.md](reference/agent/memory-faculty.md) — `core.memory`, `core.memory-store`
  - [platform-utilities.md](reference/agent/platform-utilities.md) — regex, globbing, file watching, secrets, terminal/TUI, git
- [Agent FFI (breakdown)](breakdown/AGENT_FFI.md) — native HTTP/SQLite/subprocess/fs-watch surfaces and AOT link wiring

## GPU Acceleration

- [GPU Acceleration (breakdown)](breakdown/GPU_ACCELERATION.md) — Metal compute, SF64 software float64, CUDA, cost model dispatch
- [Tensors GPU reference](reference/tensors/gpu.md) — honest GPU-dispatch status per builtin
- [XLA Backend (breakdown)](breakdown/XLA_BACKEND.md) — tensor runtime operations, GPU dispatch, broadcasting
- [Eshkol on Jetson AGX Xavier](../nix/jetson/README.md) — building and running the CUDA GPU backend on NixOS/Jetson

## WebAssembly / Web Platform

- [Web Platform (breakdown)](breakdown/WEB_PLATFORM.md) — WASM compilation, DOM/Canvas/event API functions, integer handle system
- [Web Platform tutorial](tutorials/18_WEB_PLATFORM.md)

## Platform / Build

- [Platform program index](platform/README.md) — freestanding/kernel/embedded platform program, fans out to program overview, architecture, roadmap alignment, governance, workstreams, milestones, decisions
- [Build-system integration](BUILD_INTEGRATION.md) — compiling `.esk` sources from a CMake project (e.g. embedding Eshkol in a larger C/C++ project)
- [Build notes](platform/BUILD_NOTES.md)
- [CI lanes](platform/CI_LANES.md) — required vs. non-required CI lanes and what each covers
- [ICC contract surface](platform/ICC_CONTRACT_SURFACE.md)
- [Windows x86 under KVM](platform/WINDOWS_X86_KVM.md)
- [Target Support Matrix](platform/TARGET_SUPPORT_MATRIX.md)
- [Architecture documentation index](architecture/README.md) — master architecture doc, component architecture, active platform architecture
- [Development documentation index](development/README.md) — build from source, running tests, code structure, contribution workflow
- [Docker (breakdown)](breakdown/DOCKER.md) — CUDA, Debian, Ubuntu, and XLA Docker images
- [CI/CD Pipelines (breakdown)](breakdown/CI_CD.md) — GitHub Actions workflows, build matrix, release artifacts
- [VS Code Extension (breakdown)](breakdown/VSCODE_EXTENSION.md)
- [Developer Tools (breakdown)](breakdown/DEVELOPER_TOOLS.md) — LSP server, VSCode extension, `--dump-ir`/`--dump-ast`, package manager
- [Package Manager (breakdown)](breakdown/PACKAGE_MANAGER.md) — `eshkol-pkg` CLI, TOML manifest, git-based registry

## Design and Internals

- [Architecture deep-dives index](breakdown/README.md) — per-subsystem technical breakdowns (37 docs): compiler architecture, type system, memory management, closures, AD, consciousness engine, ML, signal processing, GPU, parallel computing, exact arithmetic, continuations, XLA, bytecode VM, module system, REPL JIT, web platform, package manager, developer tools, and more
- [Components documentation index](components/README.md) — component-level implementation pointers (parser, macro expander, type checker, and more)
- [Master Architecture](ESHKOL_V1_ARCHITECTURE.md) — comprehensive technical architecture from the full codebase analysis
- [Compiler Architecture (breakdown)](breakdown/COMPILER_ARCHITECTURE.md) — LLVM backend, 21 specialized codegen modules, JIT system
- [Type System (breakdown)](breakdown/TYPE_SYSTEM.md) — runtime tagged values, HoTT gradual typing, formal inference rules
- [Consciousness Engine (breakdown)](breakdown/CONSCIOUSNESS_ENGINE.md) — logic programming, active inference, global workspace theory
- [Continuations (breakdown)](breakdown/CONTINUATIONS.md) — `call/cc`, `dynamic-wind`, `guard`/`raise`
- [Mutual Tail Calls](design/MUTUAL_TAIL_CALLS.md) — proper mutual tail calls (ESH-0102/ESH-0171), AArch64 and beyond
- [Parallel Computing (breakdown)](breakdown/PARALLEL_COMPUTING.md) — work-stealing thread pool, per-thread arena isolation
- [`parallel-map` performance analysis](PARALLEL_MAP_PERFORMANCE_ANALYSIS.md) — root-cause analysis and AOT fix for real parallelism
- [Compilation Guide (breakdown)](breakdown/COMPILATION_GUIDE.md) — LLVM compilation, debugging, troubleshooting
- [Hardening Status](HARDENING.md) — per-module hardening audit and mitigation record
- [Self-Differentiating Neural Computer (SDNC)](SDNC.md) — v1.2 paper artefact

## Testing and Quality Gates

- [Testing & Adversarial Harnesses](TESTING.md) — SICP gate plus the adversarial harnesses and how to run them
- [Test Coverage](TEST_COVERAGE.md) — what the suite gate verifies
- [Depth-coverage matrix](DEPTH_COVERAGE_MATRIX.md) — whole-language depth-parametric testing (pillar P6 auditor)

Adversarial-campaign pillar harnesses (`tests/`):

- [P1 Differential execution](../tests/differential/README.md)
- [P2 Edge matrix](../tests/edge_matrix/README.md) ([feature axes](../tests/edge_matrix/FEATURES.md))
- [P3 AD composition oracle](../tests/ad_oracle/README.md)
- [P5 VM parity ratchet](../tests/vm_parity/README.md)
- [P6a Depth-parametric AD oracle](../tests/ad_depth/README.md)
- [P7 Sanitizer/libFuzzer harnesses](../tests/fuzz/README.md)
- [P7c Metamorphic/property oracle](../tests/metamorphic/README.md)
- [Stress tests](../tests/stress/README.md)

Engineering reports (`docs/reports/`) — see the [reports directory index](reports/README.md) for scope; individual reports:

- [SICP-Completeness Report](reports/SICP_COMPLETENESS_REPORT.md) — release gate: 100% of the SICP book, executable
- [Reference-Implementation Differential Report](reports/REFERENCE_DIFFERENTIAL_REPORT.md) — external R7RS ground-truth conformance oracle (P7a)
- [Sanitizer Fuzz Report](reports/SANITIZER_FUZZ_REPORT.md)
- [Metamorphic Report](reports/METAMORPHIC_REPORT.md) — P7c property-style harness results
- [Sweep Gate Report](reports/SWEEP_GATE_REPORT.md) — full gate matrix re-verification
- [AD Depth Report](reports/AD_DEPTH_REPORT.md) — P6a depth-parametric AD oracle results
- [Recursion Depth Report](reports/RECURSION_DEPTH_REPORT.md) — P6b recursion/control depth sweep
- [Nesting Depth Report](reports/NESTING_DEPTH_REPORT.md) — P6c syntax/data nesting-depth sweep
- [Numeric Depth Report](reports/NUMERIC_DEPTH_REPORT.md) — P6d numeric-tower depth/scale sweep
- [Metaprogramming Depth Report](reports/METAPROG_DEPTH_REPORT.md) — P6e metaprogramming + module depth sweep
- [Tensor/Collection Depth Report](reports/TENSOR_COLLECTION_DEPTH_REPORT.md) — P6f tensor/collection/string depth sweep
- [ICC Complete Review (2026-07-01)](reports/ESHKOL_ICC_COMPLETE_REVIEW_2026-07-01.md) — full ICC-led campaign review

## Vision and Roadmap

- [ROADMAP](../ROADMAP.md) — canonical, time-ordered release plan
- [Roadmap (breakdown, short form)](breakdown/ROADMAP.md) — summary surfaced from the breakdown set
- [Compiler Roadmap](COMPILER_ROADMAP.md)
- [Vision documentation index](vision/README.md) — architectural baseline and mission, fans out to:
  - [Purpose and Vision](vision/PURPOSE_AND_VISION.md)
  - [AI Focus](vision/AI_FOCUS.md)
  - [Scientific Computing](vision/SCIENTIFIC_COMPUTING.md)
  - [Technical White Paper](vision/TECHNICAL_WHITE_PAPER.md)
  - [Technical White Paper Addendum](vision/ADDENDUM_TECHNICAL_WHITE_PAPER_V1.md)
  - [Differentiation Analysis](vision/DIFFERENTIATION_ANALYSIS.md) — comparison with Scheme, Julia, Python/JAX
  - [Future Roadmap (vision view)](vision/FUTURE_ROADMAP.md) — long-horizon, capability-grouped roadmap
- [Noesis Trajectory](NOESIS_TRAJECTORY.md) — trajectory notes for the Noesis integration effort
- [Multimedia System Architecture](future/MULTIMEDIA_SYSTEM_ARCHITECTURE.md) — forward-looking design
- [Neuro-Symbolic Complete Architecture](future/NEURO_SYMBOLIC_COMPLETE_ARCHITECTURE.md) — forward-looking design for bidirectional neural/symbolic integration
- [V1.1 Scope](V1.1_SCOPE.md)

## Press / Outreach

- [Description Copy](../press/ESHKOL_DESCRIPTION_COPY.md)
- [Press Information Sheet](../press/ESHKOL_PRESS_INFORMATION_SHEET.md)

## Internal / Process

These documents are written for maintainers and cross-project collaborators
rather than end users. They are kept in the repository for traceability but
are not part of the primary documentation path.

- [Release Readiness Report](internal/RELEASE_READINESS_REPORT.md) — release-candidate gate status for the current line
- [Eshkol V1 Language Reference (internal)](internal/ESHKOL_V1_LANGUAGE_REFERENCE.md) — comprehensive internal feature reference
- [icc_extras tooling](../tools/icc_extras/README.md) — Eshkol-specific tooling on top of `infinite_context_coder`
- [LLVM verifier coverage audit](../tools/icc_extras/verifier_coverage.md)

## Development Tooling

- [Contributing Guide](../CONTRIBUTING.md) — architecture overview and development workflow
- [Test Coverage](TEST_COVERAGE.md)
- [Benchmarking (breakdown)](breakdown/BENCHMARKING.md) — matrix multiplication, activations, convolution, GPU vs CPU

## Historical / Superseded

<details>
<summary>Dated planning notes, resolved bug-investigation handoffs, and cross-project
correspondence. Kept for traceability; superseded by the current documents linked
above or by closed issues/PRs referenced inline. Click to expand.</summary>

- [v1.3 -> v1.5 Dependency Plan (2026-06-28)](V1_3_TO_V1_5_DEPENDENCY_PLAN_2026-06-28.md) — superseded by the canonical [ROADMAP.md](../ROADMAP.md)
- [Eshkol Language Update Brief (2026-06-17)](internal/ESHKOL_LANGUAGE_UPDATE_BRIEF.md) — one-time modernization request memo, since actioned
- [AD nested-reuse root cause (2026-06-26)](reports/AD_NESTED_REUSE_ROOTCAUSE_2026-06-26.md) — superseded by the shipped fix; see [AD reference](reference/ad/INDEX.md) for current behaviour
- [Eshkol -> Noesis: nested-AD report (2026-06-26)](reports/ESHKOL_TO_NOESIS_nested_ad_2026-06-26.md)
- [Eshkol -> Noesis: nested-AD, root cause pinpointed (update, 2026-06-26)](reports/ESHKOL_TO_NOESIS_nested_ad_update_2026-06-26.md)
- [Eshkol -> Noesis: nested-AD, FIXED and verified (2026-06-26)](reports/ESHKOL_TO_NOESIS_nested_ad_FIXED_2026-06-26.md) — resolution of the thread above
- [Noesis -> Eshkol: higher-order/nested AD (2026-06-26)](reports/NOESIS_TO_ESHKOL_nested_ad_2026-06-26.md)
- [Noesis -> Eshkol: static-lib getenv report (2026-06-26, SUPERSEDED)](reports/NOESIS_TO_ESHKOL_static_getenv_2026-06-26_SUPERSEDED.md) — explicitly marked superseded by the note below
- [Eshkol -> Noesis: getenv-returns-#f, RESOLVED (2026-06-26)](reports/ESHKOL_TO_NOESIS_static_getenv_RESOLVED_2026-06-26.md) — root cause was the capability policy, not a regression
- [Noesis -> Eshkol: capability-gated getenv, CORRECTED (2026-06-26)](reports/NOESIS_TO_ESHKOL_capability_env_2026-06-26.md) — final correction in the thread above

Note: `notes/release-hygiene-2026-07-07/noesis-notes/` contains cross-project
scratch notes exchanged during the 2026-07-07 release-hygiene pass. That
directory is listed in `.gitignore` (`notes/`) and is not tracked in the
repository, so it is intentionally not linked here — those files exist only
in local working copies, not in the committed tree.

</details>

---

Missing something? This index links every document tracked under `docs/`, plus
the adjacent `examples/`, `tests/`, `tools/icc_extras/`, `press/`, and
`nix/jetson/` documentation. If you land on a page that isn't reachable from
here, that's a bug in this index — please open an issue or fix it directly.
