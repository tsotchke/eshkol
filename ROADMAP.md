# Eshkol Development Roadmap

This roadmap tracks Eshkol's evolution from the **completed v1.0-foundation release** through upcoming versions that will establish Eshkol as the definitive platform for gradient-based computing and integrated AI.

---

## ✅ v1.0-foundation (2025) - COMPLETED

**Production Release Delivered**

### Completed Core Implementation
- ✅ LLVM-based modular backend with 21 specialized codegen modules
- ✅ Recursive descent parser with HoTT type expression support
- ✅ Bidirectional type checker with gradual typing
- ✅ Ownership and escape analysis for memory optimization
- ✅ Module system with dependency resolution and cycle detection
- ✅ Hygienic macro system (syntax-rules)
- ✅ R7RS Scheme compatibility (subset)

### Completed Automatic Differentiation
- ✅ Forward-mode AD (dual numbers)
- ✅ Reverse-mode AD (computational graphs)
- ✅ Nested gradients (32-level tape stack)
- ✅ Vector calculus operators (8 total): derivative, gradient, jacobian, hessian, divergence, curl, laplacian, directional-derivative
- ✅ Polymorphic arithmetic (int64/double/dual/tensor/AD-node)

### Completed Memory Management
- ✅ Arena allocation with OALR (Ownership-Aware Lexical Regions)
- ✅ Escape analysis (stack/region/shared allocation decisions)
- ✅ with-region syntax for lexical memory scopes
- ✅ Ownership tracking (owned, moved, borrowed states)
- ✅ Zero garbage collection - fully deterministic

### Completed Data Structures
- ✅ 16-byte tagged values with consolidated types
- ✅ 32-byte cons cells supporting mixed-type lists
- ✅ N-dimensional tensors with autodiff integration
- ✅ Hash tables (FNV-1a hashing, open addressing)
- ✅ Heterogeneous vectors
- ✅ Exception handling (guard/raise)

### Completed Standard Library
- ✅ 60+ list operations
- ✅ 30+ string utilities
- ✅ Functional programming (compose, curry, flip)
- ✅ JSON/CSV/Base64 support
- ✅ Math library (linear algebra, numerical methods, statistics)

### Completed Development Tools
- ✅ Interactive REPL with LLVM ORC JIT
- ✅ Standalone compiler (eshkol-run)
- ✅ Library compilation mode
- ✅ Comprehensive test suite (170+ files)
- ✅ CMake build system
- ✅ Docker containers

---

## ✅ v1.1-accelerate (Q1 2026) - COMPLETED

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
- [x] Weight matrix transformer — programs as neural network weights (55/55, 3-way verified)
- [x] qLLM bridge — Eshkol↔qLLM tensor conversion with AD integration

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

### Windows Platform Support (Added)
- [x] Native Windows build via MSYS2/MinGW64 (PR #9, mattneel)
- [x] UTF-8-safe REPL console output
- [x] Platform runtime abstraction layer

---

## 📋 v1.2-scale (Q2 2026) - PLANNED

**Focus:** Data I/O, model deployment, distributed computing, and ecosystem expansion

### Multimedia & Data I/O
- [ ] Image I/O (PNG, JPEG read/write via stb or libpng)
- [ ] Audio I/O (WAV read/write, PCM streaming)
- [ ] Video frame extraction (FFmpeg integration)
- [ ] Typed data buffers (pixel formats: RGBA8/BGRA8/GRAY8/RGBAF; audio sample formats: I16/I24/F32/F64)
- [ ] Stream abstraction for file/network/audio data pipelines with combinators (map, filter, take, concat)
- [ ] DataFrame / table abstraction (CSV/Parquet backed)
- [ ] Plotting / visualization (SVG output, terminal sparklines)

### GPU & Compute
- [ ] Vulkan Compute for cross-platform GPU (beyond Metal/CUDA)
- [ ] Multi-GPU support (device selection, peer-to-peer transfer)
- [ ] Mixed-precision training (fp16/bf16 accumulation with fp32 master weights)
- [ ] Basic networking (TCP/UDP sockets for distributed training prerequisites)

### Model Deployment
- [ ] Model serialization (save/load weight tensors to binary format)
- [ ] ONNX export (computational graph + weights)
- [ ] Inference optimization (operator fusion, weight quantization)
- [ ] Core ML export for Apple platforms

### Ecosystem
- [ ] Python bindings (call Eshkol from Python, call Python from Eshkol)
- [ ] Improved error messages with source location spans
- [ ] Visual debugger integration (DAP protocol)
- [ ] Per-thread arenas for concurrent memory management

### Distributed Training
- [ ] Data parallelism across machines (AllReduce)
- [ ] Model parallelism for large networks
- [ ] MPI integration for HPC clusters
- [ ] gRPC for cloud deployments

---

## 📋 v1.3-evolve (Q3 2026) - PLANNED

**Focus:** Language maturity, advanced type system, and compiler optimization

### Language Extensions
- [ ] Full R7RS library system (`define-library` / `import` with renaming and prefixing)
- [ ] Syntax-rules with ellipsis patterns and template matching
- [ ] String interpolation (`~{expr}` or `$expr` within strings)
- [ ] Named keyword arguments (`(f #:key value)`)
- [ ] Pattern matching in `let` bindings (destructuring `let-match`)

### Advanced Type System
- [ ] Full dependent type enforcement (beyond gradual — compile-time errors for type violations)
- [ ] Refinement types (e.g., `(: x (Refine Integer (> x 0)))`) with SMT solver integration
- [ ] Effect types for tracking side effects (`Pure`, `IO`, `State`, `Exception`)
- [ ] Row polymorphism for records (structural subtyping)
- [ ] Higher-rank types (rank-2 polymorphism for combinators like `fold`)

### Compiler Optimization
- [ ] Profile-guided optimization (PGO) — runtime profiling feeds back into codegen
- [ ] Whole-program optimization across modules (cross-module inlining, dead code elimination)
- [ ] Polyhedral loop optimization for nested tensor operations
- [ ] Automatic parallelization of independent tensor operation chains

---

## 📋 v1.4-connection (Q4 2026) - PLANNED

**Focus:** Platform abstraction, interactive systems, and hardware connectivity

Informed by the [Multimedia System Architecture](docs/future/MULTIMEDIA_SYSTEM_ARCHITECTURE.md)
specification, v1.4 extends Eshkol from a computation-focused language into a platform for
interactive applications, real-time systems, and embedded/robotics development.

### Platform Abstraction Layer
- [ ] Cross-platform window system (X11/Wayland on Linux, Cocoa on macOS, Win32 on Windows, framebuffer on embedded)
- [ ] Event system (keyboard, mouse, touch, window resize, timer events)
- [ ] Event loop pattern with poll/wait semantics and callback registration
- [ ] `with-window` bracket pattern for automatic resource cleanup

### Real-Time Audio
- [ ] Audio device management (ALSA on Linux, CoreAudio on macOS, WASAPI on Windows)
- [ ] Callback-based and queue-based audio I/O for low-latency processing
- [ ] Real-time audio synthesis (integration with v1.1 signal processing stdlib)
- [ ] MIDI input/output for musical instrument control
- [ ] `with-audio` bracket pattern for automatic device release

### Networking
- [ ] TCP/UDP socket handles with linear resource management (guaranteed close)
- [ ] Stream-based network I/O with typed buffers from v1.2
- [ ] Unix domain sockets for local IPC
- [ ] Non-blocking I/O with event loop integration
- [ ] TLS/SSL support via system libraries

### Embedded & Robotics
- [ ] GPIO pin control (Linux sysfs, Raspberry Pi direct memory-mapped I/O)
- [ ] I2C/SPI/UART bus communication for sensor and actuator interfacing
- [ ] PWM output for motor control and LED dimming
- [ ] ADC/DAC for analog sensor reading and signal generation
- [ ] Camera and LIDAR device handles
- [ ] Mobile/embedded cross-compilation targets (ARM bare-metal, RISC-V)

### Type System Extensions for Resources
- [ ] Linear types for hardware handles (guaranteed cleanup — no leaked file descriptors or sockets)
- [ ] Borrow pattern for temporary resource access without ownership transfer
- [ ] Handle lifecycle states: `invalid → open → borrowed → closed` with compile-time tracking
- [ ] Typed buffer system for zero-copy data exchange (pixel formats, audio samples, complex numbers)
- [ ] Stream combinators (map, filter, take, drop, concat) over typed streams

---

## 📋 v1.5-intelligence (Q1 2027) - PLANNED

**Focus:** Neuro-symbolic integration and advanced AI primitives

Informed by the [Neuro-Symbolic Architecture](docs/future/NEURO_SYMBOLIC_COMPLETE_ARCHITECTURE.md)
specification, v1.5 extends the v1.1 consciousness engine into a complete neuro-symbolic
programming framework where neural (differentiable) and symbolic (discrete) computation
flow bidirectionally through a unified substrate.

### Symbolic Reasoning (expand v1.1 foundation)
- [x] Unification engine with occurs check — Done in v1.1
- [x] Knowledge base with pattern-matching query — Done in v1.1
- [ ] Backward chaining inference (Prolog-style goal-directed proof search with backtracking)
- [ ] Forward chaining inference (production rules with fixed-point derivation)
- [ ] Constraint solving (finite domain constraints, SAT solver integration)
- [ ] Triple store / knowledge graphs (RDF-style with SPO/POS/OSP indexing for efficient query)

### Neural-Symbolic Bridge
- [ ] Symbol embeddings (learnable vector representations of symbols in the knowledge base)
- [ ] Soft unification (differentiable similarity via cosine distance — gradients flow through matching)
- [ ] Attention over knowledge base (neural query mechanism over symbolic facts)
- [ ] Gradient estimators for discrete operations (Gumbel-Softmax, straight-through estimator)
- [ ] Differentiable logic programs (gradients flow through symbolic rule application)

### Advanced Neural Architectures
- [x] Attention mechanisms — Done in v1.1 (scaled-dot, multi-head attention)
- [x] Convolution operations — Done in v1.1 (conv1d/2d/3d, pooling)
- [ ] Recurrent structures (LSTM cells, GRU cells with forget/update gates)
- [x] Transformer blocks — Done in v1.1 (feed-forward, positional encoding, RoPE)
- [ ] Graph Neural Networks (message passing, neighborhood aggregation, graph attention)

### Program Synthesis
- [ ] Type-directed synthesis holes (`??` syntax — compiler searches for well-typed completions)
- [ ] Program enumeration with type-directed pruning (enumerate candidates up to depth N)
- [ ] Neural-guided program search (beam search with neural heuristic scoring for candidate ranking)
- [ ] Synthesis from input-output examples (inductive programming)

### Neuro-Symbolic Models
- [ ] Knowledge graph embeddings (embed entity-relation-entity triples as vectors)
- [ ] Neural theorem provers (neural heuristic guides symbolic proof search)
- [ ] Explainable neural networks (symbolic rule extraction from trained weights)
- [ ] Self-improving code (gradient descent on program embeddings for program optimization)
- [ ] Knowledge-grounded language models (KB-biased generation for factual consistency)

---

## 🔬 v2.0-starlight (2027+) - RESEARCH

**Focus:** Quantum computing, formal verification, and next-generation type system

Informed by the [Quantum Computing Architecture](docs/future/external/eshkol_quantum_computing.md)
specification, v2.0 introduces first-class quantum computing support, leveraging Eshkol's
existing linear type infrastructure (OALR) and automatic differentiation system to enable
hybrid classical-quantum algorithms with compile-time safety guarantees.

### Quantum Type System
- [ ] Qubit type with linear resource tracking (no-cloning theorem enforced at compile time)
- [ ] Quantum register types `qreg<n>` with compile-time dimension tracking
- [ ] Quantum state types `qstate<H>` parameterized by Hilbert space
- [ ] Quantum operation types `qop<H1, H2>` for maps between Hilbert spaces
- [ ] `define-quantum-region` scoping for qubit allocation and automatic deallocation

### Quantum Operations
- [ ] Quantum gate primitives: Hadamard (H), CNOT, Rz, T, S, SWAP, Toffoli, arbitrary unitaries
- [ ] Measurement operations with classical outcome (collapses superposition, returns bit)
- [ ] Quantum control flow (`qif` for superposition-based branching)
- [ ] Quantum circuit compilation and optimization (gate fusion, qubit mapping, routing)
- [ ] Integration with AD for variational quantum algorithms (gradients of parameterized circuits)

### Hybrid Classical-Quantum Algorithms
- [ ] Variational Quantum Eigensolver (VQE) with classical optimizer integration
- [ ] Quantum Approximate Optimization Algorithm (QAOA)
- [ ] Quantum machine learning (parameterized quantum circuits with automatic differentiation)
- [ ] Hybrid classical-quantum loop patterns (classical optimization drives quantum circuit parameters)
- [ ] Quantum error mitigation techniques (zero-noise extrapolation, probabilistic error cancellation)

### Formal Verification
- [ ] Integration with proof assistants (Coq, Lean) for certified compilation
- [ ] Quantum program verification (unitary equivalence checking)
- [ ] Certified quantum circuit compilation (provably correct gate decomposition)
- [ ] Session types for quantum communication protocols

### Next-Generation Type System
- [ ] Full dependent type enforcement with decidable type checking
- [ ] Session types for concurrent communication patterns
- [ ] Algebraic effects and handlers (structured side-effect management)
- [ ] Quantitative type theory (resource-tracking types for quantum and linear resources)

---

## Development Priorities

### Immediate (v1.2 — Q2 2026)
1. Data I/O (image, audio, typed buffers, streams)
2. Model serialization and ONNX export
3. Vulkan Compute for cross-platform GPU
4. Python bindings and distributed training (MPI)

### Near-Term (v1.3 — Q3 2026)
1. Full R7RS library system with `define-library`
2. Refinement types and effect types
3. Profile-guided optimization (PGO)
4. Higher-rank types

### Medium-Term (v1.4 — Q4 2026)
1. Platform abstraction layer (windows, events, audio)
2. Real-time audio and MIDI integration
3. Networking (TCP/UDP with linear resource management)
4. Embedded/robotics targets (GPIO, I2C, SPI)

### Long-Term (v1.5 — Q1 2027)
1. Neuro-symbolic bridge (soft unification, symbol embeddings)
2. Graph Neural Networks and recurrent structures
3. Program synthesis with type-directed search
4. Knowledge graph embeddings

### Research (v2.0 — 2027+)
1. Quantum computing integration (qubit types, gates, VQE/QAOA)
2. Formal verification with proof assistant integration
3. Session types and algebraic effects
4. Quantum machine learning (AD through parameterized circuits)

---

## Component Status

### Core Compiler
- ✅ Parser - Complete
- ✅ Type Checker - Complete
- ✅ LLVM Backend - Complete (34,928 lines)
- ✅ Module System - Complete
- ✅ Macro System - Complete

### Automatic Differentiation
- ✅ Forward Mode - Complete
- ✅ Reverse Mode - Complete
- ✅ Nested Gradients - Complete (32-level tape stack)
- ✅ Vector Calculus - Complete (8 operators)

### Memory Management
- ✅ Arena Allocation - Complete
- ✅ OALR System - Complete
- ✅ Ownership Tracking - Complete
- ✅ Escape Analysis - Complete

### Standard Library (v1.1)
- ✅ Core Functions (60+ list ops, 30+ string utils) - Complete
- ✅ Math Library (linear algebra, statistics, ODE solvers) - Complete
- ✅ Signal Processing (FFT, filters, window functions) - Complete
- ✅ ML Library (optimizers, activations, normalization) - Complete
- ✅ Web Platform (80+ DOM API functions, WASM target) - Complete
- ✅ JSON/CSV/Base64 Support - Complete

### Development Tools
- ✅ REPL with JIT (stdlib preloading, cross-eval persistence) - Complete
- ✅ Compiler (eshkol-run, AOT + script mode) - Complete
- ✅ Package Manager (eshkol-pkg, TOML manifest) - Complete
- ✅ LSP Server (diagnostics, completion, hover) - Complete
- ✅ VSCode Extension (syntax highlighting, LSP client) - Complete
- ✅ Test Suite (35 suites, 434 tests) - Complete

### v1.1-accelerate (Complete)
- ✅ XLA Backend (StableHLO/MLIR + LLVM-direct) - Complete
- ✅ GPU Acceleration (Metal SF64 + CUDA cuBLAS) - Complete
- ✅ SIMD Vectorization (SSE/AVX/NEON) - Complete
- ✅ Parallel Primitives (work-stealing thread pool) - Complete
- ✅ Exact Arithmetic (bignums + rationals + complex) - Complete
- ✅ Consciousness Engine (logic, inference, workspace — 22 builtins) - Complete
- ✅ ML Framework (75+ builtins: activations, losses, optimizers, CNN, transformers) - Complete
- ✅ Signal Processing (FFT, filters, window functions) - Complete
- ✅ R7RS Extensions (call/cc, dynamic-wind, bytevectors) - Complete

### Planned (v1.2+)
- 📋 Data I/O (image, audio, typed buffers) — v1.2
- 📋 Distributed Computing (MPI, AllReduce) — v1.2
- 📋 Vulkan Compute — v1.2
- 📋 Platform Abstraction (windows, audio, networking) — v1.4
- 📋 Neuro-Symbolic Bridge — v1.5
- 📋 Quantum Computing — v2.0

---

## Research Directions

**Active Research:**
- Polyhedral optimization for nested tensor loops (v1.3)
- Linear type systems for hardware resources (v1.4)
- Neuro-symbolic bridging — differentiable symbolic operations (v1.5)
- Effect systems for purity tracking and algebraic effects (v1.3/v2.0)

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

*Last Updated: March 2026*

*Eshkol v1.1-accelerate is now complete, delivering XLA, SIMD, GPU acceleration, parallel primitives, exact arithmetic, consciousness engine, 75+ ML builtins, signal processing, and full R7RS extensions. The roadmap progresses through data I/O and deployment (v1.2-scale), language maturity (v1.3-evolve), platform abstraction and hardware connectivity (v1.4-connection), neuro-symbolic integration (v1.5-intelligence), and quantum computing with formal verification (v2.0-starlight).*
