# Eshkol Development Roadmap

Eshkol's evolution from foundation through production acceleration and into the research frontier. Versions v1.0 and v1.1 are **complete and shipping**; v1.2+ is planned on a monthly cadence.

---

## Completed Releases

### v1.0-foundation (2025)

The core language: LLVM backend, R7RS Scheme compatibility, arena memory (OALR — zero GC), forward/reverse-mode autodiff with 8 vector-calculus operators, 16-byte tagged values, N-dimensional tensors, hygienic macros, module system, REPL with ORC JIT, and a 170+ file test suite.

### v1.1-accelerate (Q1 2026)

Performance, breadth, and platform reach. Delivered 47 roadmap items plus major unplanned additions:

- **XLA + GPU** — StableHLO/MLIR backend, Metal SF64 + CUDA cuBLAS, SIMD (SSE/AVX/NEON), cost-model dispatch
- **Parallelism** — `parallel-map`, `parallel-fold`, `future`, work-stealing thread pool
- **Exact arithmetic** — bignums, rationals, complex numbers with full autodiff integration
- **Consciousness engine** — logic programming (unification, KB, facts), active inference (factor graphs, belief propagation, free energy), global workspace (softmax competition) — 22 builtins
- **ML framework** — 555+ builtins: activations, losses, optimizers, CNN primitives, transformers
- **Signal processing** — FFT/IFFT, FIR/IIR filters, Butterworth design, window functions
- **R7RS extensions** — call/cc, dynamic-wind, guard/raise, bytevectors, let-syntax
- **Bytecode VM** — 63-opcode ISA, 555+ native call IDs, ESKB binary format, dual-number AD in VM
- **Web platform** — WASM compilation target, eshkol.ai website written in Eshkol, browser REPL
- **Windows** — native x64 + ARM64 via VS 2022 + ClangCL + LLVM 21; MSYS2/MinGW64 path
- **CI/CD** — 16-lane release matrix (linux/macos/windows × x64/arm64 × lite/xla/cuda)
- **REPL hot reload** — PLT/GOT-style slot indirection for all top-level redefinitions (functions, variables, lambdas, R7RS-correct dynamic name resolution in closures)
- **Package manager security** — `fork+execvp` / `CreateProcessW` subprocess launcher, dep-name validation, git URL allowlist

---

## Planned Releases

### v1.2-scale (May 2026)

**Focus:** Get models into production.

- [ ] Model serialization (save/load tensor weights to binary format)
- [ ] Python bindings (call Eshkol functions from Python via FFI bridge)
- [ ] Per-thread arenas (safe concurrent memory allocation)
- [ ] Image I/O (PNG/JPEG read/write via stb_image)
- [ ] CSV/DataFrame (tabular data loading for ML pipelines)
- [ ] Improved error messages with source location spans
- [ ] Terminal plotting (sparklines + bar charts for quick visualization)

### v1.3-evolve (June 2026)

**Focus:** Make the language a joy to use day-to-day.

- [ ] Full R7RS library system (`define-library` / `import` with renaming and prefixing)
- [ ] String interpolation (`~{expr}` within strings)
- [ ] Named keyword arguments (`(f #:key value)`)
- [ ] Pattern matching in `let` bindings (destructuring `let-match`)
- [ ] Profile-guided optimization (runtime profiling feeds codegen)
- [ ] Whole-program optimization (cross-module inlining and dead code elimination)

### v1.4-connection (July 2026)

**Focus:** Connect to the outside world with compile-time safety.

- [ ] TCP/UDP sockets with linear resource types (guaranteed close)
- [ ] TLS/SSL via system libraries
- [ ] Non-blocking I/O with event loop (epoll/kqueue)
- [ ] HTTP client (built on sockets + TLS)
- [ ] Linear types for all handles: `open → borrowed → closed` with compile-time tracking

### v1.5-intelligence (August 2026)

**Focus:** Neural and symbolic computation flow bidirectionally.

- [ ] Symbol embeddings (learnable vector representations of KB symbols)
- [ ] Soft unification (differentiable similarity — gradients flow through matching)
- [ ] LSTM and GRU cells (standard recurrent neural architectures)
- [ ] Differentiable logic programs (gradients flow through rule application)
- [ ] Attention over knowledge base (neural query mechanism over symbolic facts)

### v1.6-reasoning (September 2026)

**Focus:** Make the logic engine production-grade.

- [ ] Backward chaining inference (Prolog-style goal-directed proof search)
- [ ] Forward chaining inference (production rules with fixed-point derivation)
- [ ] Constraint solving (finite domain constraints, SAT solver integration)
- [ ] Knowledge graphs (RDF-style triple store with SPO/POS/OSP indexing)

### v1.7-synthesis (October 2026)

**Focus:** Programs that write and improve programs.

- [ ] Neural-guided program search (beam search with neural scoring)
- [ ] Type-directed synthesis holes (`??` — compiler searches for well-typed completions)
- [ ] Graph Neural Networks (message passing, graph attention)
- [ ] Synthesis from input-output examples (inductive programming)

### v1.8-platform (November 2026)

**Focus:** Eshkol runs on everything, controls everything.

- [ ] Cross-platform windowing (X11/Wayland, Cocoa, Win32)
- [ ] Real-time audio (CoreAudio, ALSA, WASAPI)
- [ ] Vulkan Compute for cross-platform GPU
- [ ] Embedded cross-compilation (ARM bare-metal, RISC-V)

### v1.9-types (December 2026)

**Focus:** The type system becomes a proof system.

- [ ] Full dependent type enforcement (compile-time errors, not warnings)
- [ ] Refinement types with SMT solver integration
- [ ] Effect types (tracking `Pure`, `IO`, `State`, `Exception` at the type level)
- [ ] Algebraic effects and handlers
- [ ] Session types for communication protocols

### v2.0-starlight (Q1 2027)

**Focus:** Quantum computing meets formal verification.

- [ ] Qubit type with linear resource tracking (no-cloning enforced at compile time)
- [ ] Gate primitives: H, CNOT, Rz, T, S, SWAP, Toffoli, arbitrary unitaries
- [ ] VQE, QAOA, parameterized quantum circuits with AD (parameter-shift rule)
- [ ] Integration with proof assistants (Lean) for certified compilation

---

## Release Timeline

| Version | Date | Theme | Key Deliverables |
|---------|------|-------|-----------------|
| **v1.0** | 2025 | Foundation | Core language, LLVM backend, autodiff, arena memory |
| **v1.1** | Q1 2026 | Accelerate | XLA/GPU, parallelism, bignums, consciousness engine, VM, web, Windows |
| **v1.2** | May 2026 | Scale | Model serialization, Python bindings, image I/O |
| **v1.3** | Jun 2026 | Evolve | R7RS libraries, string interpolation, PGO |
| **v1.4** | Jul 2026 | Connection | Networking, TLS, event loop, linear resource types |
| **v1.5** | Aug 2026 | Intelligence | Symbol embeddings, differentiable logic, LSTM/GRU |
| **v1.6** | Sep 2026 | Reasoning | Backward chaining, constraint solving, knowledge graphs |
| **v1.7** | Oct 2026 | Synthesis | Neural-guided search, program synthesis, GNN |
| **v1.8** | Nov 2026 | Platform | Windowing, audio, Vulkan, embedded targets |
| **v1.9** | Dec 2026 | Types | Dependent types, effects, algebraic effects, session types |
| **v2.0** | Q1 2027 | Starlight | Quantum types, VQE/QAOA, formal verification |

---

## Architecture Dependency Chain

```
v1.0 + v1.1 (COMPLETE)
 ├─ Core language + LLVM + autodiff + arena
 ├─ XLA/GPU + parallelism + SIMD
 ├─ Consciousness engine (logic + inference + workspace)
 ├─ Bytecode VM + web platform
 └─ Windows + 16-lane CI matrix
       │
v1.2 ──┤ Model serialization (requires tensors)
       ├ Python bindings (requires stable API)
       └ Per-thread arenas (requires OALR)
             │
v1.3 ────────┤ R7RS library system (requires module system)
             └ PGO (requires stable codegen)
                   │
v1.4 ──────────────┤ Networking + TLS (requires per-thread arenas)
                   └ Linear resource types (requires HoTT)
                         │
v1.5 ────────────────────┤ Neuro-symbolic bridge (requires AD + logic + tensors)
                               │
v1.6 ──────────────────────────┤ Advanced reasoning (requires logic engine)
                                     │
v1.7 ────────────────────────────────┤ Program synthesis (requires type system + neural)
                                           │
v1.8 ──────────────────────────────────────┤ Platform abstraction (requires I/O from v1.4)
                                                 │
v1.9 ────────────────────────────────────────────┤ Advanced types (requires HoTT + continuations)
                                                       │
v2.0 ──────────────────────────────────────────────────┤ Quantum + verification (requires linear dep types)
```

---

## Contributing

We welcome contributions in all areas — core development, research, documentation, and ecosystem. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

*Last updated: April 2026*
