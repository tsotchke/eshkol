# Eshkol: Development Roadmap (vision view)

> **Position in the doc set:** this is the long-horizon, vision-oriented
> view of the roadmap, grouped by capability rather than by release.
> For the canonical, time-ordered release plan see
> [`ROADMAP.md`](../../ROADMAP.md) at the repo root; for engineering
> detail see [`docs/COMPILER_ROADMAP.md`](../COMPILER_ROADMAP.md).
> When this document disagrees with `ROADMAP.md`, `ROADMAP.md` is
> correct.

This document outlines Eshkol's evolution from the **completed v1.0-foundation** through upcoming releases that will establish Eshkol as the definitive platform for gradient-based computing.

## v1.0-foundation: COMPLETED (2025)

**Status:** **Production Release**

Eshkol v1.0-foundation delivers a production-ready compiler that tightly integrates automatic differentiation, deterministic arena memory management, and homoiconic native-code execution.

### Completed Achievements

**Core Compiler:**
- LLVM-based modular backend (21 specialized codegen modules)
- Recursive descent parser with HoTT type expressions
- Bidirectional type checker with gradual typing
- Ownership and escape analysis
- Module system with dependency resolution and cycle detection
- Macro expansion (syntax-rules with hygienic macros)
- Exception handling (guard/raise with R7RS semantics)

**Automatic Differentiation:**
- Forward-mode AD with dual numbers
- Reverse-mode AD with computational graphs
- Nested gradients up to 32 levels deep
- Vector calculus operators: derivative, gradient, jacobian, hessian, divergence, curl, laplacian
- Polymorphic arithmetic supporting int64/double/dual/tensor/AD-node

**Memory Management:**
- Arena allocation with OALR (Ownership-Aware Lexical Regions)
- Escape analysis determining stack/region/shared allocation
- with-region syntax for lexical memory scopes
- Ownership tracking (owned, moved, borrowed states)
- Deterministic deallocation (no garbage collection)

**Data Structures:**
- 16-byte tagged values with consolidated type system
- 32-byte cons cells supporting mixed-type lists
- N-dimensional tensors with element-wise operations
- Hash tables with FNV-1a hashing
- Heterogeneous vectors
- UTF-8 strings with escape sequences

**Standard Library:**
- 60+ list operations (map, filter, fold, etc.)
- 30+ string utilities
- Functional programming (compose, curry, flip)
- JSON parsing and serialization
- CSV processing
- Base64 encoding/decoding
- Math library (linear algebra: det, inv, solve; numerical integration; root finding; statistics)

**Development Tools:**
- Interactive REPL with LLVM ORC JIT
- Standalone compiler (eshkol-run)
- Library compilation mode
- Comprehensive test suite (170+ test files)
- CMake build system
- Docker containers

**Documentation:**
- Complete language specification (300+ features)
- Language reference with examples
- Architecture documentation
- API reference
- Quickstart guide

**What v1.0-foundation Proves:**
- Compiler-integrated AD is achievable and practical
- Homoiconic closures can execute at native LLVM speed
- Arena memory provides deterministic performance without GC
- Gradual typing enables both exploration and production hardening

## Development Timeline

| Release | Timeframe | Status | Focus |
|---------|-----------|--------|-------|
| **v1.0-foundation** | **2025** | **COMPLETE** | **Core compiler, AD system, arena memory, REPL** |
| **v1.1-accelerate** | **Q1 2026** | **COMPLETE** | **XLA backend, GPU acceleration, parallelism, consciousness engine** |
| **v1.2-scale** | **Q2 2026** | **COMPLETE** | **Full numeric tower (bignum / rational / complex), Metal + CUDA, XLA dual-mode, FFI hardening (HTTP / SQLite / subprocess / fs-watch), 22-builtin consciousness engine** |
| **v1.2.1-scale** | **2026-05-20** | **COMPLETE** | **Edge-case hardening: stdlib `LinkOnceODR`, parser line-tracking, closure capture in `dynamic-wind` / `call-cc` / `guard`, TCO context preservation, agent-FFI link wiring** |
| **v1.3.0-evolve** | **Jul 2026** | **SHIPPED** | **Arbitrary-order Taylor-tower AD matrix (all 13 phases P0–P12: exact bignum/rational coefficients, GUW multivariate mixed partials, reverse-over-Taylor, tensor towers, validated Taylor models, sparse high-order tensors, differentiable control flow, checkpointed reverse, tower numerics); full R7RS conformance (34/34 vs. chibi-scheme); TCO/closure/memory robustness hardening; permanent adversarial-testing infrastructure. See [`../../ROADMAP.md`](../../ROADMAP.md) (canonical) and [`../../CHANGELOG.md`](../../CHANGELOG.md).** |
| **v1.3.1 → v1.3.4-evolve** | **Jul–Aug 2026** | **SHIPPED** | **Flat memory for resident loops, thread-safe regions, opt-in differentiable quantum computing, ML-KEM, `core.dbsp`, exact gradients through every callable form, linear `Qubit`, high-precision numerics, the portable event loop** |
| **v1.3.5-evolve** | **2026-09-22** | **SHIPPED** | **A parser with no recursion budget, dense tensor AD end to end, VM region reclamation, multi-shot continuations on all engines, certified enclosures, the exact tower and `core.exact_linalg`** |
| **v1.3.6-evolve** | **2026-10-19** | **SHIPPED** | **The 46 mathematics examples documented with the scope of each result, certified mixed exact/inexact intervals, exact browser numerics, measurement-bound publication** |
| **v1.4.0-connection** | **2026-11-25** | **RELEASE CANDIDATE** | **Linear networking (TCP/UDP/TLS/Unix-domain handles under `--strict-resources`), O(1) event-loop HTTP/WebSocket/SSE, write-ahead-logged agent persistence, socket collectives, subprocess sandboxing, WebGPU browser VM dispatch (ADR-0029), exact-rational tensors (ADR-0058), the Full Mathematics Programme with differentiable SBLC search proposals (`core.sblc.relax`, ADR-0017 item 5), 395 machine-checked Lean 4.12.0 theorems, 86 verified examples** |
| **v1.4.1 / v1.4.5-accelerate** | **Joint rebaseline** | **Planned** | **The ABI release; then the accelerator device runtime, which publishes before v1.5.0** |
| **v1.5.0-intelligence** | **After v1.4.5; joint rebaseline** | **Planned** | **Symbol embeddings, differentiable logic, LSTM/GRU, neuro-symbolic integration, VM-as-transformer Stage 3 (lift remaining delegated opcodes into weight matrices)** |
| **v1.9.0-types** | **Q1–Q2 2028** | **Planned** | **Effect / refinement type system, dependent types, algebraic effects, session types** |
| **v2.0-starlight** | **Q4 2028** | **Research** | **Quantum computing, formal verification (Lean-certified validated-AD Taylor models), fully verified compilation chain** |

> **Note:** items previously slotted under "v1.3" in this vision view — effect/refinement
> types, PGO + whole-program optimization, distributed training, automatic C-header
> bindings — were never part of what v1.3.0-evolve actually shipped. Effect/refinement
> types now track to **v1.9-types**; PGO and whole-program opt are ongoing backend work;
> C-header binding generation is a stdlib/tooling item. The canonical
> [`ROADMAP.md`](../../ROADMAP.md) is authoritative for release contents.

## v1.1-accelerate: Performance and Parallelism (Q1 2026)

**Status:** **COMPLETE (Production Release)**

Building on the v1.0 foundation, this release focuses on **computational acceleration** through XLA integration, automatic vectorization, and multi-core parallelism.

### XLA Backend Integration

**Accelerated Linear Algebra Compiler:**
- XLA fusion for tensor operation chains
- Automatic kernel generation
- CPU/GPU code generation from single source
- JIT compilation for dynamic shapes

**Benefits:**
- 10-100x speedup for tensor-heavy workloads
- Unified code path for CPU and GPU
- Automatic optimization of computation graphs

### SIMD Vectorization

**Automatic Parallelization:**
- SSE/AVX/NEON instruction generation
- Loop vectorization for tensor operations
- Memory alignment optimization
- Platform-specific tuning (x86_64, ARM64)

**Target Operations:**
- Element-wise tensor arithmetic
- Reduction operations (sum, mean, etc.)
- Matrix-vector products
- Activation functions (sigmoid, tanh, relu)

### Concurrency and Parallelism

**Multi-Core Primitives:**
- `(parallel-map func lst...)` - Data parallelism
- `(parallel-fold func init lst)` - Parallel reduction
- `(future expr)` - Asynchronous computation
- Thread pool management

**Work-Stealing Scheduler:**
- Dynamic load balancing
- Cache-aware task distribution
- Minimal overhead for fine-grained tasks

**Memory Safety:**
- Ownership tracking across threads
- Immutable data sharing
- Message passing for coordination

### Extended Math Library

**Numerical Methods:**
- Complex numbers with autodiff support
- Arbitrary precision arithmetic
- Differential equation solvers (ODE, PDE)
- Optimization algorithms (L-BFGS, conjugate gradient)

**Signal Processing:**
- FFT/IFFT operations
- Convolution and correlation
- Filtering (low-pass, high-pass, band-pass)
- Spectral analysis

**Statistics:**
- Extended statistical distributions
- Hypothesis testing
- Regression analysis
- Time series analysis

## v1.2-scale: Numeric Tower, GPU, Agent FFI, Production Deployment (Q2 2026)

**Status:** **COMPLETE (Production Release).** Closed out as v1.2.1-scale on 2026-05-20.

The release combined a production-grade numeric tower with the GPU dispatch infrastructure (Metal + CUDA + XLA) and the native agent FFI surface, plus the hardening pass that brought sanitiser-clean status across the full test suite.

### Numeric tower

- Arbitrary-precision integers (bignum) with R7RS exact-arithmetic semantics
- Exact rationals (16-byte int64 numerator / int64 denominator, auto-reduced by GCD; bignum-pair rationals remain a future extension)
- Full complex numbers with Smith's-formula division and overflow-safe magnitude
- Dual numbers for forward-mode AD; tape-based reverse-mode AD spanning more than twenty AD node types

### GPU acceleration (shipped)

- **Metal backend (macOS, Apple Silicon)** — full SF64 / DF64 / F32 / FP24 / FP53 precision tiers with software-float64 emulation when the GPU lacks native fp64.
- **CUDA backend (Linux + Windows with NVIDIA)** — cuBLAS dispatch, dynamic feature detection.
- **XLA backend (StableHLO via MLIR; LLVM-direct fallback)** — cost-model dispatch on tensor size.
- **Cost-model dispatcher** — SIMD → cBLAS → GPU based on element count and runtime-calibrated FLOPS estimates.

### Agent FFI (shipped)

- Native libcurl HTTP client (TLS verify-on by default, redirect-follow, NOSIGNAL for thread safety)
- sqlite3 binding with dynamic column-text allocation (column text is no longer truncated at a fixed width)
- posix_spawn subprocess (no shell metacharacter exposure on the safe path); macOS uses kqueue+EVFILT_PROC, Linux falls back to pthread drainers
- inotify / FSEvents / ReadDirectoryChangesW filesystem watching

### Production deployment (shipped)

- `.eshkol-model` binary serialiser (ESKM, CRC32-checksummed) for save-on-training / load-on-inference workflows
- Stable C FFI in `inc/eshkol/eshkol_ffi.h`; Python bindings via pybind11
- `eshkol-pkg` package manager with central registry, dependency resolution, lock files

### v1.2.1 hardening (shipped)

- Stdlib `LinkOnceODR` linkage so user `(define …)` can shadow stdlib names without duplicate-symbol errors
- Parser line tracking through `(load …)` and per-form REPL streams
- Free-variable capture across `dynamic-wind` / `call-cc` / `guard` / `raise`
- TCO context preservation across nested `letrec`
- Agent-FFI link wiring through `ESHKOL_HOST_AGENT_FFI_LINK_ARGS`
- 87/87 v1.2 edge-case suite; 37/37 aggregate suite gate; ASan + UBSan + TSan clean

### Items considered for v1.2 but deferred to v1.3+

- **Vulkan compute shaders** — Metal + CUDA cover the v1.2 platform matrix; Vulkan deferred.
- **Multi-node distributed training (MPI / gRPC / NCCL)** — out of scope; deferred to v1.3.
- **ONNX / Core ML / TF Lite export** — Eshkol's native `.eshkol-model` ships; cross-framework exports deferred.
- **Multi-GPU dispatch** — single-GPU works; multi-GPU deferred.

## v1.5-intelligence: Neuro-Symbolic Integration (after v1.4.5-accelerate)

**Status:** **Planned.** Publication order is mandatory: v1.4.0 → v1.4.1 →
v1.4.5 → v1.5.0. Consciousness Engine v2 and the Eshkol OS track are forward
work beyond v1.4.0-connection and are not part of the current release.

Advanced AI capabilities building on v1.0's homoiconic foundation.

### Symbolic Reasoning Primitives

**Logic Programming:**
- Unification and pattern matching
- Backtracking search
- Constraint solving
- Knowledge representation

**Integration with Neural:**
- Neural-guided symbolic search
- Symbolic constraints in neural training
- Explainable AI through symbolic traces

### Advanced Neural Architectures

**Built-in Primitives:**
- Attention mechanisms
- Convolution operations
- Recurrent structures
- Transformer blocks

**Neuro-Symbolic Models:**
- Knowledge graph embeddings
- Differentiable logic programs
- Neural theorem provers

## v2.0-starlight: Advanced Research Features (target Q4 2028)

**Status:** **Research Phase**

Long-term research directions extending Eshkol's capabilities into emerging computational paradigms.

### Quantum Computing Integration

**Quantum Circuit Support:**
- Qubit allocation and manipulation
- Quantum gate operations
- Measurement and state collapse
- Hybrid classical-quantum algorithms

**Quantum-Classical Interface:**
- Automatic optimization across boundary
- State preparation from classical data
- Measurement result processing

### Advanced Type System
- Effect system for tracking and controlling side effects
- Dependent types for more expressive type-level programming
- Linear types for resource management
- Refinement types for stronger correctness guarantees

**Whole-Program Optimization:**
- Cross-module inlining and specialization
- Dead code elimination across boundaries
- Profile-guided optimization

**Advanced Optimizations:**
- Polyhedral optimization for nested loops
- Automatic parallelization
- Cache-aware transformations

### Expanded Research Directions

**Formal Methods:**
- Dependent types for correctness proofs
- Refinement types for stronger guarantees
- Integration with proof assistants (Coq, Lean)

**Program Synthesis:**
- Inductive program synthesis from examples
- Neural program synthesis with symbolic constraints
- Automatic algorithm discovery

**New Application Domains:**
- Computational biology and bioinformatics
- Robotics and control systems
- Quantitative finance
- Digital signal processing

## Research Directions

Beyond the planned features, we are actively researching several areas that may influence Eshkol's future development:

### Programming Language Theory
- Effect systems and algebraic effects
- Gradual typing systems
- Linear and affine type systems
- Dependent type theory

### Compiler Technology
- Whole-program optimization techniques
- Just-in-time compilation strategies
- Heterogeneous computing compilation
- Automatic parallelization

### Memory Management
- Region-based memory management
- Ownership and borrowing systems
- Real-time garbage collection
- Hardware-assisted memory management

### Artificial Intelligence
- Neuro-symbolic integration techniques
- Differentiable programming models
- Program synthesis and induction
- Explainable AI methods

### Scientific Computing
- Domain-specific languages for scientific domains
- High-performance computing optimizations
- Numerical stability and precision
- Reproducible scientific computing

## Community Development

The success of Eshkol depends not only on technical excellence but also on building a vibrant and inclusive community. Our plans for community development include:

### Governance
- Establishing a transparent governance model
- Creating a technical steering committee
- Developing a code of conduct
- Setting up contribution guidelines

### Education and Outreach
- Developing educational materials
- Conducting workshops and tutorials
- Engaging with academic institutions
- Participating in conferences and events

### Industry Adoption
- Identifying key industry use cases
- Developing case studies and success stories
- Providing enterprise support options
- Building partnerships with industry leaders

### Open Source Ecosystem
- Supporting community-developed packages
- Recognizing and rewarding contributors
- Funding critical infrastructure development
- Ensuring long-term sustainability

## Getting Involved

We welcome contributions from researchers, engineers, and enthusiasts who share our vision for Eshkol. There are many ways to get involved:

### Development
- Implementing language features
- Developing libraries and tools
- Writing documentation
- Creating examples and tutorials

### Research
- Exploring new language features
- Investigating performance optimizations
- Applying Eshkol to new domains
- Publishing papers and articles

### Community
- Answering questions and providing support
- Organizing events and meetups
- Mentoring new contributors
- Spreading the word about Eshkol

### Usage
- Building projects with Eshkol
- Providing feedback on features and usability
- Filing issue reports and suggesting improvements
- Sharing your experiences and use cases

## Near-Term Priorities

**Immediate (v1.4.0-connection — release candidate, 2026-11-25):**
1. Final exact-commit qualification and publication of the release candidate
2. The resource-sound systems profile: linear TCP/UDP/TLS handles, O(1)
   event-loop HTTP/WebSocket/SSE, WAL agent persistence, socket collectives
3. The Full Mathematics Programme, differentiable SBLC search proposals
   (`core.sblc.relax`), and independent Lean proof replay

**Next (v1.4.1, then v1.4.5-accelerate):**
1. The stable public C ABI and embedding SDK (v1.4.1)
2. The accelerator device runtime through PJRT/StableHLO (v1.4.5-accelerate)
3. Multi-GPU dispatch

**Following (v1.5.0-intelligence, after v1.4.5):**
1. Advanced neural network primitives and symbolic reasoning integration
2. Symbol embeddings, soft unification, differentiable logic programs
3. Consciousness Engine v2 and the Eshkol OS track (forward work beyond
   v1.4.0-connection; not part of the current release)

## Community and Ecosystem Growth

**Open Source Development:**
- Active development on GitHub
- MIT license for maximum adoption
- Community contributions welcomed
- Regular release cycle (quarterly)

**Package Ecosystem:**
- Central package repository (the `eshkol-pkg` client shipped in v1.1; a hosted central registry is Research)
- Dependency management system
- Versioning and compatibility
- Documentation generation

**Enterprise Support:**
- Professional consulting available
- Training and workshops
- Custom feature development
- Priority issue resolution

**Academic Engagement:**
- University partnerships for curriculum integration
- Research collaborations on compiler design and AD
- Conference presentations and publications
- Student project sponsorship

## Technical Research Directions

**Active Research:**
- Polyhedral optimization for nested loops
- Automatic differentiation of recursive algorithms
- Linear type systems for resource management
- Effect systems for side effect tracking

**Exploratory Research:**
- Probabilistic programming integration
- Formal verification of AD correctness
- Hardware-software co-design for AD acceleration
- Novel memory management strategies

## Conclusion

Eshkol v1.0-foundation **proves** that compiler-integrated automatic differentiation, deterministic memory management, and homoiconic native code can coexist in a production system. The completed implementation demonstrates technical leadership in automatic differentiation architecture and establishes a foundation for GPU acceleration, parallelism, and distributed computing.

The roadmap positions Eshkol to **dominate** gradient-based computing through:
- **Near-term**: publishing v1.4.0-connection — the resource-sound systems profile, the Full Mathematics Programme, and machine-checked verification — then the v1.4.1 ABI release
- **Medium-term**: the v1.4.5-accelerate device runtime (PJRT/StableHLO, multi-GPU) rivaling PyTorch/JAX on accelerators
- **Long-term**: v1.5.0-intelligence neuro-symbolic integration and v2.0-starlight quantum computing establishing new paradigms

We invite researchers, engineers, and organizations to join this evolution - whether through code contributions, research collaborations, or production deployments. Eshkol represents the future of computational science where mathematical elegance and uncompromising performance converge.
