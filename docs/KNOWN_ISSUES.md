# Known Issues and Limitations - Eshkol v1.0-Architecture

**Status**: v1.0.0-foundation - Production-ready with known limitations

This document catalogs known limitations in v1.0-architecture and planned future enhancements.

---

## Not Yet Implemented (Planned for Future Releases)

### Language Features

❌ **`eval` function** - Dynamic code evaluation
- **Status**: Planned for v1.1
- **Workaround**: Homoiconicity provides partial alternative (lambdas preserve S-expressions)
- **Technical note**: Will leverage lambda registry and JIT compiler

❌ **Full `call/cc` support** - First-class continuations
- **Status**: AST structures defined, implementation planned for v1.1  
- **Current**: Limited continuation support
- **Use case**: Non-local control flow

❌ **Exact arithmetic** - Arbitrary precision integers and rationals
- **Status**: Planned for v1.2
- **Current**: Primarily uses IEEE 754 doubles for performance
- **Workaround**: Symbolic mode for exact differentiation

❌ **Bytevectors** - R7RS bytevector operations
- **Status**: Type defined (`HEAP_SUBTYPE_BYTEVECTOR`), operations planned for v1.1
- **Current**: Use regular vectors or tensors

❌ **Records** - User-defined record types
- **Status**: Type defined (`HEAP_SUBTYPE_RECORD`), syntax planned for v1.2
- **Current**: Use vectors or hash tables

### Ecosystem

❌ **Package manager** - Central package repository
- **Status**: Planned for v1.1
- **Current**: Manual module management via `require`/`provide`

❌ **Language Server Protocol (LSP)** - IDE integration
- **Status**: Planned for v1.1
- **Current**: Basic syntax highlighting only

❌ **Visual debugger** - Source-level debugging UI
- **Status**: Planned for v1.2
- **Current**: Use GDB/LLDB with generated binaries

❌ **Foreign Function Interface (FFI)** - C/C++ library integration
- **Status**: Planned for v1.2
- **Current**: Must link C libraries at build time

### Performance Features

❌ **GPU acceleration** - CUDA/Metal/Vulkan backends
- **Status**: Planned for v1.3
- **Architecture**: Already designed (see docs/private/MULTIMEDIA_SYSTEM_ARCHITECTURE.md)

❌ **Distributed computing** - Multi-node training
- **Status**: Planned for v1.3
- **Use case**: Large-scale neural network training

❌ **Profile-Guided Optimization (PGO)** - Runtime profile-based optimization
- **Status**: Planned for v1.2
- **Current**: LLVM -O3 optimizations only

---

## Known Limitations (Design Trade-offs)

### Memory Model

**No garbage collector** (by design)
- **Trade-off**: Deterministic timing vs. automatic memory management convenience
- **Mitigation**: OALR system provides automatic arena deallocation
- **Impact**: Users must understand arena lifetimes for complex programs

**Single global arena** (current implementation)
- **Trade-off**: Simplicity vs. per-thread arenas
- **Impact**: All allocations go through one arena (fine for single-threaded scientific computing)
- **Future**: Per-thread arenas planned when threading is added

### Type System

**Gradual typing warnings (not errors)**
- **Trade-off**: Rapid prototyping vs. compile-time safety
- **Impact**: Type errors can reach production if warnings ignored
- **Mitigation**: Code review should address all type warnings

**No higher-rank types** (current limitation)
- **Trade-off**: Implementation complexity vs. type system power
- **Impact**: Rank-1 polymorphism only (`forall` at outermost level)
- **Future**: Planned for v1.3

**Limited dependent types**
- **Trade-off**: Complexity vs. safety
- **Current**: Tensor dimensions only
- **Future**: Full dependent types planned for v2.0

### Numeric Computing

**No exact arithmetic by default**
- **Trade-off**: Performance vs. mathematical exactness
- **Impact**: Uses IEEE 754 doubles (inexact)
- **Mitigation**: Symbolic differentiation for exact derivatives

**No complex number type** (as first-class value)
- **Trade-off**: Implementation simplicity vs. mathematical completeness
- **Current**: Complex numbers work in AD context
- **Future**: Full complex type planned for v1.2

### Platform Support

**Windows support is Tier 2** (best-effort, not fully tested)
- **Tier 1**: Linux x86-64, macOS x86-64/ARM64
- **Tier 2**: Windows x86-64 (community-tested)
- **Future**: Windows to become Tier 1 in v1.1

**No mobile/embedded targets** (currently)
- **Status**: Planned for v1.4
- **Architecture**: Designed but not implemented

---

## Resolved Issues (Fixed in v1.0)

✅ **Pointer Consolidation** - COMPLETE
- Consolidated 8 pointer types into 2 (HEAP_PTR, CALLABLE)
- Object headers enable type/subtype system
- Frees 6 bits of type tag space for future use

✅ **Phase 3B Tagged Cons** - COMPLETE
- 32-byte cons cells with direct tagged value storage
- Improved cache efficiency
- Simplified codegen

✅ **Closure Mutable Captures** - COMPLETE
- Pointers to captured variables (not values)
- Enables stateful closures
- Critical for iterative algorithms

✅ **AD 32-Level Tape Stack** - COMPLETE
- Nested gradient computation
- Hessian calculation
- Meta-learning support

✅ **Modular LLVM Backend** - COMPLETE
- 19 specialized codegen modules
- Callback-based architecture
- Independent testing

---

## Reporting Issues

For bugs or feature requests:

1. Check [Feature Matrix](FEATURE_MATRIX.md) for implementation status
2. Review this document for known limitations
3. File issue on GitHub: https://github.com/tsotchke/eshkol/issues
4. Provide: Eshkol version, platform, minimal reproduction

---

## See Also

- [Feature Matrix](FEATURE_MATRIX.md) - Implementation status (✅ / ⚠️ / ❌)
- [Roadmap](docs/aidocs/ROADMAP.md) - Future development plans
- [Master Architecture](ESHKOL_V1_ARCHITECTURE.md) - What is implemented