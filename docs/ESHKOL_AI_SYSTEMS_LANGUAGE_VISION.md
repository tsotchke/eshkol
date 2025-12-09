# Eshkol: The AI Systems Language

## A Comprehensive Technical Vision

**Date:** December 7, 2025
**Version:** 1.0
**Status:** Strategic Architecture Document

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Analysis](#2-current-state-analysis)
   - 2.1 [Eshkol's Existing Capabilities](#21-eshkols-existing-capabilities)
   - 2.2 [The qLLM Training Challenge](#22-the-qllm-training-challenge)
   - 2.3 [Gap Analysis](#23-gap-analysis)
3. [XLA Integration: The Game Changer](#3-xla-integration-the-game-changer)
   - 3.1 [What XLA Provides](#31-what-xla-provides)
   - 3.2 [Architecture with XLA](#32-architecture-with-xla)
   - 3.3 [Hybrid Approach: XLA + Existing qLLM](#33-hybrid-approach-xla--existing-qllm)
4. [The Case for Rewriting qLLM in Eshkol](#4-the-case-for-rewriting-qllm-in-eshkol)
   - 4.1 [Strategic Analysis](#41-strategic-analysis)
   - 4.2 [What qLLM in Eshkol Would Look Like](#42-what-qllm-in-eshkol-would-look-like)
   - 4.3 [Effort Comparison](#43-effort-comparison)
5. [Eshkol as a General-Purpose AI Language](#5-eshkol-as-a-general-purpose-ai-language)
   - 5.1 [Competitive Landscape](#51-competitive-landscape)
   - 5.2 [Eshkol's Unique Position](#52-eshkols-unique-position)
6. [Tier 1: Essential Features](#6-tier-1-essential-features)
   - 6.1 [N-Dimensional Tensors with Dependent Types](#61-n-dimensional-tensors-with-dependent-types)
   - 6.2 [Automatic Batching (vmap)](#62-automatic-batching-vmap)
   - 6.3 [Parallelism Primitives](#63-parallelism-primitives)
   - 6.4 [Mixed Precision and Quantization](#64-mixed-precision-and-quantization)
   - 6.5 [Gradient Checkpointing](#65-gradient-checkpointing)
7. [Tier 2: Advanced Features](#7-tier-2-advanced-features)
   - 7.1 [Probabilistic Programming](#71-probabilistic-programming)
   - 7.2 [Symbolic-Neural Integration](#72-symbolic-neural-integration)
   - 7.3 [Differentiable Data Structures](#73-differentiable-data-structures)
   - 7.4 [Neural ODEs and Continuous-Time Models](#74-neural-odes-and-continuous-time-models)
   - 7.5 [Effect System for AI](#75-effect-system-for-ai)
8. [Tier 3: Cutting-Edge Features](#8-tier-3-cutting-edge-features)
   - 8.1 [Quantum Computing Integration](#81-quantum-computing-integration)
   - 8.2 [Formal Verification of Neural Networks](#82-formal-verification-of-neural-networks)
   - 8.3 [Self-Improving Code Generation](#83-self-improving-code-generation)
   - 8.4 [Federated and Privacy-Preserving Learning](#84-federated-and-privacy-preserving-learning)
9. [Tier 4: Ecosystem](#9-tier-4-ecosystem)
   - 9.1 [Python Interoperability](#91-python-interoperability)
   - 9.2 [ONNX Import/Export](#92-onnx-importexport)
   - 9.3 [Visualization and Debugging](#93-visualization-and-debugging)
   - 9.4 [Model Hub Integration](#94-model-hub-integration)
10. [Implementation Specifications](#10-implementation-specifications)
    - 10.1 [Type System Extensions](#101-type-system-extensions)
    - 10.2 [AD System Extensions](#102-ad-system-extensions)
    - 10.3 [XLA Lowering](#103-xla-lowering)
    - 10.4 [Required Chain Rules](#104-required-chain-rules)
11. [Development Roadmap](#11-development-roadmap)
12. [Conclusion](#12-conclusion)

---

## 1. Executive Summary

This document presents a comprehensive vision for Eshkol as the premier AI systems programming language. Through detailed analysis of the current codebase, the semi-classical qLLM training requirements, and the broader AI language landscape, we establish a strategic path forward.

### Key Findings

1. **Eshkol's foundation is sound**: The existing HoTT type system, native autodiff, and LLVM backend provide a solid base for AI workloads.

2. **XLA integration is transformative**: With planned XLA support, many infrastructure gaps (N-D tensors, GPU, parallelism) are solved automatically.

3. **Rewriting qLLM in Eshkol is viable**: Once XLA is integrated, rewriting the qLLM in Eshkol becomes the strategically superior choice over maintaining a bridge.

4. **Eshkol can be uniquely positioned**: No existing AI language combines dependent types, native autodiff, XLA execution, probabilistic programming, and symbolic integration.

### Strategic Recommendation

Invest in making Eshkol a complete AI systems language. The combination of HoTT types + native AD + XLA + probabilistic/symbolic capabilities would create a genuinely novel and powerful platform with no direct competitor.

---

## 2. Current State Analysis

### 2.1 Eshkol's Existing Capabilities

Eshkol v1.0.0-foundation provides a sophisticated language implementation with features uniquely suited to AI workloads.

#### 2.1.1 HoTT Type System

The Homotopy Type Theory foundation in `lib/types/hott_types.cpp:56-177` provides:

```cpp
// Universe hierarchy for type stratification (lib/types/hott_types.cpp:20-28)
enum class Universe { U0, U1, U2, UOmega };
// U0: Ground types (values)
// U1: Type constructors (List, Vector, Tensor, Function)
// U2: Proposition types (Eq, LessThan, Bounded)
// UOmega: Universe polymorphism

// Runtime representation tracking (lib/types/hott_types.cpp:30-40)
enum class RuntimeRep { Int64, Float64, Pointer, TaggedValue, Struct, Erased };

// Type environment initialization (lib/types/hott_types.cpp:60-177)
void TypeEnvironment::initializeBuiltinTypes() {
    // Universe types (meta-types)
    registerBuiltinType(TypeU0.id, "Type0", Universe::U1, 0, RuntimeRep::Erased);

    // Numeric tower: Number → Integer/Real → Int64/Float64
    registerBuiltinType(Number.id, "Number", Universe::U0, 0, RuntimeRep::TaggedValue, Value);
    registerBuiltinType(Integer.id, "Integer", Universe::U0, TYPE_FLAG_EXACT, RuntimeRep::Int64, Number);
    registerBuiltinType(Real.id, "Real", Universe::U0, 0, RuntimeRep::Float64, Number);

    // Type families (parameterized types)
    registerTypeFamily(List.id, "List", Universe::U1, {"a"}, RuntimeRep::Pointer);
    registerTypeFamily(Vector.id, "Vector", Universe::U1, {"a"}, RuntimeRep::Pointer);
    registerTypeFamily(Tensor.id, "Tensor", Universe::U1, {"a", "shape"}, RuntimeRep::Pointer);
    registerTypeFamily(Function.id, "->", Universe::U1, {"a", "b"}, RuntimeRep::Pointer);

    // Autodiff types
    registerTypeFamily(DualNumber.id, "Dual", Universe::U1, {"a"}, RuntimeRep::Struct);
    registerTypeFamily(ADNode.id, "ADNode", Universe::U1, {"a"}, RuntimeRep::Pointer);

    // Linear types (must be consumed exactly once)
    registerTypeFamily(Handle.id, "Handle", Universe::U1, {"k"}, RuntimeRep::Pointer);
    types_[Handle.id].id.flags |= TYPE_FLAG_LINEAR;

    // Proposition types (erased at runtime)
    registerBuiltinType(Eq.id, "Eq", Universe::U2, TYPE_FLAG_PROOF, RuntimeRep::Erased);
}
```

**CodegenContext Integration** (`inc/eshkol/backend/codegen_context.h:77-79`):
```cpp
class CodegenContext {
    hott::TypeEnvironment hott_types_;  // HoTT type environment
    // ...
};
```

**Implications for AI:**
- Tensor shapes as type-level values: `Tensor Float64 [batch, seq_len, dim]`
- Compile-time shape mismatch detection via type checking
- Linear types for resource management (GPU memory, file handles)
- Proof types for verified invariants (erased at runtime)

#### 2.1.2 Native Automatic Differentiation

The autodiff system in `lib/backend/autodiff_codegen.cpp` implements both forward and reverse mode:

**Forward Mode (Dual Numbers):**
```cpp
// Dual number: (value, derivative)
// Arithmetic follows chain rule automatically
llvm::Value* dualMul(llvm::Value* a, llvm::Value* b) {
    // (a, a') * (b, b') = (a*b, a'*b + a*b')
    Value* value = builder.CreateFMul(a_val, b_val);
    Value* deriv = builder.CreateFAdd(
        builder.CreateFMul(a_prime, b_val),
        builder.CreateFMul(a_val, b_prime));
    return createDualNumber(value, deriv);
}
```

**Reverse Mode (Tape-Based):**
```cpp
// Backpropagation through computational graph
void propagateGradient(llvm::Value* node_ptr) {
    switch (node_type) {
        case AD_NODE_ADD:  // dL/dx = dL/dz, dL/dy = dL/dz
            accumulateGradient(input1, node_grad);
            accumulateGradient(input2, node_grad);
            break;
        case AD_NODE_MUL:  // dL/dx = dL/dz * y, dL/dy = dL/dz * x
            accumulateGradient(input1, node_grad * input2_val);
            accumulateGradient(input2, node_grad * input1_val);
            break;
        // ... more operations
    }
}
```

**Supported Operations:**
| Operation | Forward Mode | Reverse Mode |
|-----------|--------------|--------------|
| ADD | (a+b, a'+b') | dL/dx = dL/dz |
| SUB | (a-b, a'-b') | dL/dx = dL/dz, dL/dy = -dL/dz |
| MUL | (ab, a'b+ab') | dL/dx = dL/dz * y |
| DIV | (a/b, (a'b-ab')/b²) | dL/dx = dL/dz / y |
| SIN | (sin(a), a'cos(a)) | dL/dx = dL/dz * cos(x) |
| COS | (cos(a), -a'sin(a)) | dL/dx = dL/dz * -sin(x) |
| EXP | (exp(a), a'exp(a)) | dL/dx = dL/dz * exp(x) |
| LOG | (log(a), a'/a) | dL/dx = dL/dz / x |
| POW | Complex | Complex |
| TANH | (tanh(a), a'sech²(a)) | dL/dx = dL/dz * sech²(x) |

#### 2.1.3 Memory Management

Arena-based allocation in `lib/core/arena_memory.cpp:88-175`:

```cpp
// Create arena with configurable block size
arena_t* arena_create(size_t default_block_size) {
    arena_t* arena = (arena_t*)malloc(sizeof(arena_t));
    arena->current_block = create_arena_block(default_block_size);
    arena->current_scope = nullptr;
    arena->default_block_size = default_block_size;
    arena->total_allocated = default_block_size;
    arena->alignment = DEFAULT_ALIGNMENT;  // 8 bytes
    return arena;
}

// Fast aligned bump allocation
void* arena_allocate_aligned(arena_t* arena, size_t size, size_t alignment) {
    size_t aligned_size = align_size(size, alignment);
    arena_block_t* block = arena->current_block;
    size_t current_used = align_size(block->used, alignment);

    if (current_used + aligned_size > block->size) {
        // Allocate new block (grows as needed)
        arena_block_t* new_block = create_arena_block(
            std::max(aligned_size, arena->default_block_size));
        new_block->next = arena->current_block;
        arena->current_block = new_block;
        block = new_block;
        current_used = 0;
    }

    void* ptr = block->memory + current_used;
    block->used = current_used + aligned_size;
    return ptr;
}

// Scoped allocation for temporary computation
void arena_push_scope(arena_t* arena);   // Save current state
void arena_pop_scope(arena_t* arena);    // Restore to saved state (O(1) bulk free)
```

**Global AD State** (`lib/core/arena_memory.cpp:24-50`):
```cpp
// Shared across all JIT modules in REPL
ad_tape_t* __current_ad_tape = nullptr;
bool __ad_mode_active = false;
ad_tape_t* __ad_tape_stack[32] = {nullptr};  // Nested gradient support
uint64_t __ad_tape_depth = 0;
arena_t* __repl_shared_arena = nullptr;       // Persistent REPL memory
```

#### 2.1.4 LLVM Code Generation

**JIT Compilation** (`lib/repl/repl_jit.cpp:59-120`):
```cpp
// REPL uses LLVM ORC LLJIT for real-time native compilation
void ReplJITContext::initializeJIT() {
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();

    // Create LLJIT with dynamic library search
    auto jit_or_err = LLJITBuilder()
        .setNumCompileThreads(1)
        .create();
    jit_ = std::move(*jit_or_err);

    // Register runtime symbols for JIT code to call
    registerRuntimeSymbols();  // arena_*, printf, math functions

    // Create shared arena for all REPL evaluations
    __repl_shared_arena = arena_create(8192);
}

// Symbol persistence across evaluations
void ReplJITContext::registerSymbol(const std::string& name, uint64_t address) {
    symbol_table_[name] = address;
    // Also register in JIT dylib for cross-module linking
    orc::SymbolMap symbols;
    symbols[ES.intern(mangled)] = {
        orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(address)),
        JITSymbolFlags::Callable | JITSymbolFlags::Exported
    };
    main_dylib.define(orc::absoluteSymbols(symbols));
}
```

**AOT Compilation** (`lib/backend/llvm_codegen.cpp`):
```cpp
// Generate native executables via LLVM TargetMachine
std::unique_ptr<TargetMachine> target_machine(
    target->createTargetMachine(target_triple, cpu_name, features,
                                opt, reloc_model));

// Emit object file
legacy::PassManager pass;
target_machine->addPassesToEmitFile(pass, dest, nullptr,
                                     CodeGenFileType::ObjectFile);
pass.run(module);
```

**Key Architectural Features**:
- ✅ Zero interpreter overhead (all code compiles to native)
- ✅ Symbol persistence across REPL evaluations
- ✅ Shared AD tape context for gradient operations
- ✅ Arena memory for fast computation graph allocation

### 2.2 The qLLM Training Challenge

The semi-classical qLLM in `/Users/tyr/Desktop/semiclassical_qllm/` is a sophisticated transformer implementation with geometric attention mechanisms.

#### 2.2.1 qLLM Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    qLLM Architecture                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input Embedding                                            │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            Transformer Block (×N)                    │   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │  Multi-Head Geodesic Attention              │    │   │
│  │  │  - Queries/Keys in Poincaré ball            │    │   │
│  │  │  - Hyperbolic distance for scores           │    │   │
│  │  │  - Tangent space projections                │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  │       │                                              │   │
│  │       ▼                                              │   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │  Layer Normalization                        │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  │       │                                              │   │
│  │       ▼                                              │   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │  Feed-Forward Network (GELU)                │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  │       │                                              │   │
│  │       ▼                                              │   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │  Layer Normalization                        │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
│       │                                                     │
│       ▼                                                     │
│  Output Projection                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 2.2.2 Key qLLM Components

**Tensor System** (`src/core/tensor.c` - 2643 lines):
```c
struct qllm_tensor {
    size_t dims;           // Number of dimensions
    size_t* shape;         // Shape array
    size_t* strides;       // Stride array for indexing
    void* data;            // Raw data pointer
    qllm_tensor_type_t type;  // float32, float16, bfloat16
    qllm_memory_strategy_t memory_strategy;
};
```

**Geometric Operations** (`src/geometric/hyperbolic_core.c`):
```c
// Poincaré ball model operations
float hyperbolic_distance(const qllm_manifold_point_t* p1,
                          const qllm_manifold_point_t* p2,
                          float curvature);

qllm_manifold_point_t* poincare_exp_map(
    const qllm_manifold_point_t* base,
    const qllm_tangent_vector_t* tangent,
    float curvature);

qllm_tangent_vector_t* poincare_log_map(
    const qllm_manifold_point_t* base,
    const qllm_manifold_point_t* target,
    float curvature);
```

**Riemannian Optimizer** (`src/optimization/riemannian_adam.c`):
```c
// THE CRITICAL GAP: This function expects gradients as input
// But qLLM has no autodiff system to compute them!
bool riemannian_adam_step(
    qllm_riemannian_optimizer_t* opt,
    qllm_tensor_t* parameters,
    const qllm_tensor_t* gradients  // WHERE DO THESE COME FROM?
);
```

#### 2.2.3 The Missing Piece

The qLLM has:
- ✅ Forward pass (inference)
- ✅ Loss computation
- ✅ Optimizer (Riemannian Adam)
- ❌ **Backward pass (gradient computation)**

This is where Eshkol's autodiff system becomes essential.

### 2.3 Gap Analysis

#### 2.3.1 Current Eshkol Tensor Capabilities

**IMPORTANT UPDATE**: Eshkol already has comprehensive N-dimensional tensor support in `lib/backend/tensor_codegen.cpp`:

| Capability | Status | Implementation |
|------------|--------|----------------|
| N-D tensor creation | ✅ Complete | `#[1 2 3]`, `#[[1 2] [3 4]]`, `zeros`, `ones`, `eye` |
| Tensor arithmetic | ✅ Complete | `tensor-add`, `tensor-sub`, `tensor-mul`, `tensor-div` |
| Element access | ✅ Complete | `tensor-get`, `tensor-set`, `vref` |
| Shape operations | ✅ Complete | `tensor-shape`, `transpose`, `reshape` |
| Reductions | ✅ Complete | `tensor-reduce-all`, `tensor-reduce`, `tensor-sum`, `tensor-mean` |
| Functional ops | ✅ Complete | `tensor-apply` (map function over elements) |
| 1D Dot product | ✅ Complete | `tensor-dot` for vectors |
| 2D Matrix multiply | ⚠️ Partial | Placeholder exists, needs full implementation |

#### 2.3.2 What's Missing for qLLM Integration

| Requirement | Eshkol Status | Gap |
|-------------|---------------|-----|
| Scalar AD | ✅ Complete | None |
| N-D tensors | ✅ **Complete** | None (tensor_codegen.cpp) |
| Tensor reshaping | ✅ Complete | None |
| Tensor broadcasting | ⚠️ Partial | Needs verification |
| 2D MatMul | ⚠️ Partial | Needs completion (placeholder exists) |
| Tensor AD | ❌ Missing | Need tensor chain rules |
| Float32/16 | ❌ Double only | Need type conversion |
| GPU execution | ❌ CPU only | Need XLA or CUDA backend |
| Softmax | ❌ Missing | Need forward + backward |
| LayerNorm | ❌ Missing | Need forward + backward |
| Attention | ❌ Missing | Need forward + backward |
| Hyperbolic ops | ❌ Missing | Need forward + backward |

#### 2.3.3 Revised Effort Estimates

With existing tensor infrastructure, effort is significantly reduced:

| Component | Effort |
|-----------|--------|
| ~~N-D tensor type~~ | ~~4-6 weeks~~ ✅ Already done |
| Complete 2D matmul | 1 week |
| Float16/32 support | 2 weeks |
| GPU backend (XLA) | 4-6 weeks (vs 8-12 without existing tensors) |
| Tensor AD operations | 4-6 weeks (can extend existing AD) |
| Softmax/LayerNorm/Attention | 2-3 weeks |
| **Total** | **13-18 weeks** (vs 26-36 before) |

---

## 3. XLA Integration: The Game Changer

### 3.1 What XLA Provides

XLA (Accelerated Linear Algebra) is Google's compiler for ML workloads. With XLA integration, Eshkol gains:

| Feature | Description |
|---------|-------------|
| **N-D Tensors** | Full broadcasting, reshaping, slicing |
| **Hardware Backends** | CPU (AVX-512), GPU (CUDA/ROCm), TPU |
| **Float16/BFloat16** | Native mixed precision |
| **Op Fusion** | Automatic kernel fusion |
| **Memory Optimization** | Buffer assignment, rematerialization |
| **Parallelism** | SPMD partitioning |

### 3.2 Architecture with XLA

```
┌─────────────────────────────────────────────────────────────┐
│                      Eshkol Source                          │
│                                                             │
│  (define (transformer x params)                             │
│    (let* ((attn (attention x params.attn))                  │
│           (ff (feed-forward attn params.ff)))               │
│      (layer-norm ff params.ln)))                            │
│                                                             │
│  ;; AD is automatic                                         │
│  (gradient (lambda (p) (loss (transformer x p) y)) params)  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Eshkol Compiler                           │
│                                                             │
│  1. Parse to AST                                            │
│  2. Type check (HoTT)                                       │
│  3. AD transformation                                       │
│  4. Lower to StableHLO                                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     StableHLO IR                            │
│                                                             │
│  func.func @transformer(%x: tensor<32x512xf32>,             │
│                         %params: ...) -> tensor<32x512xf32> │
│  {                                                          │
│    %0 = stablehlo.dot_general %x, %w ...                   │
│    %1 = stablehlo.custom_call "softmax" %0 ...             │
│    ...                                                      │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      XLA Compiler                           │
│                                                             │
│  • Algebraic simplification                                 │
│  • Common subexpression elimination                         │
│  • Op fusion                                                │
│  • Buffer assignment                                        │
│  • Layout optimization                                      │
│  • Target-specific codegen                                  │
└─────────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
           ┌─────┐      ┌─────┐      ┌─────┐
           │ CPU │      │ GPU │      │ TPU │
           │AVX512│     │CUDA │      │     │
           └─────┘      └─────┘      └─────┘
```

### 3.3 Hybrid Approach: XLA + Existing qLLM

XLA supports **custom calls** that invoke external C/C++ code. This enables a hybrid approach:

```scheme
;; Standard ops use XLA's optimized implementations
(define (standard-attention Q K V)
  (let* ([scores (/ (matmul Q (transpose K)) (sqrt d_k))]
         [weights (softmax scores)]
         [output (matmul weights V)])
    output))

;; Specialized ops call existing qLLM code
(xla/register-custom-call "hyperbolic_distance"
  :library "libqllm_geometric.so"
  :function "hyperbolic_distance"
  :input-types [(Tensor f32 [? dim]) (Tensor f32 [? dim]) f32]
  :output-type (Tensor f32 [?]))

(define (geodesic-attention Q K V curvature)
  (let* ([scores (hyperbolic-distance Q K curvature)]  ; Custom call
         [weights (softmax scores)]
         [output (matmul weights V)])
    output))
```

**Benefits of Hybrid Approach:**
1. Leverage XLA's optimized standard ops
2. Keep specialized qLLM geometric code
3. No need to reimplement everything
4. Gradual migration path

---

## 4. The Case for Rewriting qLLM in Eshkol

### 4.1 Strategic Analysis

#### 4.1.1 Without XLA: Bridge is Better

| Factor | Bridge | Rewrite |
|--------|--------|---------|
| Effort | 12 weeks | 30-40 weeks |
| Risk | Medium (FFI bugs) | High (reimplementing) |
| Performance | Good | Unknown |
| Maintenance | Two codebases | One codebase |

**Verdict: Bridge approach is more practical**

#### 4.1.2 With XLA: Rewrite is Better

| Factor | Bridge | Rewrite |
|--------|--------|---------|
| Effort | 12 weeks | 12-18 weeks |
| Risk | Medium | Low (XLA is proven) |
| Performance | Good | Excellent (XLA optimized) |
| Maintenance | Two codebases | One codebase |
| GPU/TPU | Manual | Automatic |
| Type safety | Partial | Full (HoTT) |
| Future extensibility | Limited | High |

**Verdict: Rewrite is strategically superior**

### 4.2 What qLLM in Eshkol Would Look Like

```scheme
;;;; qllm.esk - Semi-Classical Quantum LLM in Eshkol
;;;; A complete implementation with native AD and XLA execution

(module qllm
  (require eshkol/xla)
  (require eshkol/autodiff)
  (require eshkol/types)

  ;; ═══════════════════════════════════════════════════════════════
  ;; TYPE DEFINITIONS
  ;; ═══════════════════════════════════════════════════════════════

  ;; Tensor type with compile-time shape checking
  (define-type (Tensor dtype shape)
    :where (and (DType? dtype) (Shape? shape)))

  ;; Model configuration
  (define-type TransformerConfig
    (Record
      [vocab-size : Nat]
      [d-model : Nat]
      [n-heads : Nat]
      [n-layers : Nat]
      [d-ff : Nat]
      [max-seq-len : Nat]
      [dropout : Float]
      [curvature : Float]))  ; For hyperbolic attention

  ;; Layer parameters
  (define-type AttentionParams
    (Record
      [W-Q : (Tensor f32 [d-model d-model])]
      [W-K : (Tensor f32 [d-model d-model])]
      [W-V : (Tensor f32 [d-model d-model])]
      [W-O : (Tensor f32 [d-model d-model])]))

  (define-type FFNParams
    (Record
      [W1 : (Tensor f32 [d-model d-ff])]
      [b1 : (Tensor f32 [d-ff])]
      [W2 : (Tensor f32 [d-ff d-model])]
      [b2 : (Tensor f32 [d-model])]))

  (define-type LayerNormParams
    (Record
      [gamma : (Tensor f32 [d-model])]
      [beta : (Tensor f32 [d-model])]))

  ;; ═══════════════════════════════════════════════════════════════
  ;; HYPERBOLIC GEOMETRY (Custom XLA ops calling qLLM C code)
  ;; ═══════════════════════════════════════════════════════════════

  ;; Register qLLM functions as custom XLA operations
  (xla/register-custom-call "hyperbolic_distance"
    :library "libqllm_geometric.so"
    :function "hyperbolic_distance"
    :input-types [(Tensor f32 [batch seq dim])
                  (Tensor f32 [batch seq dim])
                  f32]
    :output-type (Tensor f32 [batch seq seq]))

  (xla/register-custom-call "poincare_exp_map"
    :library "libqllm_geometric.so"
    :function "poincare_exp_map")

  (xla/register-custom-call "poincare_log_map"
    :library "libqllm_geometric.so"
    :function "poincare_log_map")

  (xla/register-custom-call "project_to_tangent"
    :library "libqllm_geometric.so"
    :function "project_to_tangent")

  ;; Define gradients for custom ops (VJP rules)
  (xla/defvjp hyperbolic-distance
    (lambda (p1 p2 curvature g)
      ;; Gradient of hyperbolic distance
      (let* ([d (hyperbolic-distance p1 p2 curvature)]
             [sqnorm-p1 (sum (square p1) -1 :keepdim #t)]
             [sqnorm-p2 (sum (square p2) -1 :keepdim #t)]
             [alpha (- 1.0 sqnorm-p1)]
             [beta (- 1.0 sqnorm-p2)]
             [diff (- p1 p2)]
             [sqnorm-diff (sum (square diff) -1 :keepdim #t)]
             [gamma (+ 1.0 (/ (* 2.0 sqnorm-diff) (* alpha beta)))]
             [denom (* alpha beta (sqrt (- (square gamma) 1.0)))]
             [coef (/ 4.0 denom)]
             [grad-p1 (* coef g (+ (/ diff alpha)
                                   (/ (* sqnorm-diff p1) (square alpha))))]
             [grad-p2 (* coef g (+ (/ (- diff) beta)
                                   (/ (* sqnorm-diff p2) (square beta))))])
        [grad-p1 grad-p2 #f])))  ; No gradient for curvature

  ;; ═══════════════════════════════════════════════════════════════
  ;; ATTENTION MECHANISMS
  ;; ═══════════════════════════════════════════════════════════════

  ;; Standard scaled dot-product attention
  (define (scaled-dot-product-attention Q K V mask)
    (let* ([d-k (shape Q -1)]
           [scores (/ (matmul Q (transpose K -1 -2)) (sqrt (cast d-k f32)))]
           [masked-scores (if mask
                              (+ scores (* (- 1.0 mask) -1e9))
                              scores)]
           [weights (softmax masked-scores -1)]
           [output (matmul weights V)])
      output))

  ;; Geodesic attention using hyperbolic geometry
  (define (geodesic-attention Q K V curvature mask)
    (let* (;; Project to Poincaré ball
           [Q-hyp (project-to-ball Q curvature)]
           [K-hyp (project-to-ball K curvature)]
           ;; Compute hyperbolic distances as attention scores
           [scores (- (hyperbolic-distance Q-hyp K-hyp curvature))]
           [masked-scores (if mask
                              (+ scores (* (- 1.0 mask) -1e9))
                              scores)]
           [weights (softmax masked-scores -1)]
           [output (matmul weights V)])
      output))

  ;; Multi-head attention
  (define (multi-head-attention x params config use-geodesic)
    (let* ([batch (shape x 0)]
           [seq (shape x 1)]
           [d-model config.d-model]
           [n-heads config.n-heads]
           [d-head (/ d-model n-heads)]
           ;; Linear projections
           [Q (matmul x params.W-Q)]
           [K (matmul x params.W-K)]
           [V (matmul x params.W-V)]
           ;; Reshape for multi-head: [batch, seq, heads, d_head]
           [Q-heads (reshape Q [batch seq n-heads d-head])]
           [K-heads (reshape K [batch seq n-heads d-head])]
           [V-heads (reshape V [batch seq n-heads d-head])]
           ;; Transpose to [batch, heads, seq, d_head]
           [Q-t (transpose Q-heads 1 2)]
           [K-t (transpose K-heads 1 2)]
           [V-t (transpose V-heads 1 2)]
           ;; Apply attention
           [attn-out (if use-geodesic
                         (geodesic-attention Q-t K-t V-t config.curvature #f)
                         (scaled-dot-product-attention Q-t K-t V-t #f))]
           ;; Transpose back and reshape
           [attn-t (transpose attn-out 1 2)]
           [attn-concat (reshape attn-t [batch seq d-model])]
           ;; Output projection
           [output (matmul attn-concat params.W-O)])
      output))

  ;; ═══════════════════════════════════════════════════════════════
  ;; LAYER COMPONENTS
  ;; ═══════════════════════════════════════════════════════════════

  ;; Layer normalization
  (define (layer-norm x params [eps 1e-5])
    (let* ([mean (mean x -1 :keepdim #t)]
           [variance (variance x -1 :keepdim #t)]
           [normalized (/ (- x mean) (sqrt (+ variance eps)))]
           [output (+ (* params.gamma normalized) params.beta)])
      output))

  ;; Feed-forward network with GELU
  (define (feed-forward x params)
    (let* ([hidden (gelu (+ (matmul x params.W1) params.b1))]
           [output (+ (matmul hidden params.W2) params.b2)])
      output))

  ;; Single transformer block
  (define (transformer-block x attn-params ffn-params ln1-params ln2-params config)
    (let* (;; Multi-head attention with residual
           [attn-out (multi-head-attention x attn-params config #t)]
           [x1 (layer-norm (+ x attn-out) ln1-params)]
           ;; Feed-forward with residual
           [ff-out (feed-forward x1 ffn-params)]
           [x2 (layer-norm (+ x1 ff-out) ln2-params)])
      x2))

  ;; ═══════════════════════════════════════════════════════════════
  ;; FULL MODEL
  ;; ═══════════════════════════════════════════════════════════════

  (define-type TransformerModel
    (Record
      [embedding : (Tensor f32 [vocab-size d-model])]
      [pos-encoding : (Tensor f32 [max-seq-len d-model])]
      [layers : (List TransformerLayerParams)]
      [output-proj : (Tensor f32 [d-model vocab-size])]
      [config : TransformerConfig]))

  (define (forward model input-ids)
    (let* ([batch (shape input-ids 0)]
           [seq-len (shape input-ids 1)]
           ;; Embedding lookup
           [embedded (gather model.embedding input-ids)]
           ;; Add positional encoding
           [pos-enc (slice model.pos-encoding [0 seq-len])]
           [x (+ embedded pos-enc)]
           ;; Apply transformer layers
           [x-final (fold (lambda (x layer-params)
                           (transformer-block x
                                              layer-params.attention
                                              layer-params.ffn
                                              layer-params.ln1
                                              layer-params.ln2
                                              model.config))
                         x
                         model.layers)]
           ;; Output projection
           [logits (matmul x-final model.output-proj)])
      logits))

  ;; ═══════════════════════════════════════════════════════════════
  ;; LOSS AND TRAINING
  ;; ═══════════════════════════════════════════════════════════════

  ;; Cross-entropy loss
  (define (cross-entropy-loss logits targets)
    (let* ([log-probs (log-softmax logits -1)]
           [gathered (gather-nd log-probs targets)]
           [loss (- (mean gathered))])
      loss))

  ;; Training step with automatic differentiation
  (define (train-step model batch optimizer)
    (let* ([inputs batch.input-ids]
           [targets batch.target-ids]
           ;; Define loss function over parameters
           [loss-fn (lambda (params)
                      (let ([logits (forward (set-params model params) inputs)])
                        (cross-entropy-loss logits targets)))]
           ;; Compute gradients automatically!
           [current-params (get-params model)]
           [loss (loss-fn current-params)]
           [grads (gradient loss-fn current-params)]
           ;; Riemannian optimizer step
           ;; Projects gradients to tangent space, updates via exp map
           [new-params (riemannian-adam-step optimizer current-params grads)])
      (values (set-params model new-params) loss)))

  ;; Full training loop
  (define (train model dataset config)
    (let ([optimizer (make-riemannian-adam config.learning-rate
                                            :beta1 0.9
                                            :beta2 0.999
                                            :curvature config.curvature)])
      (for/fold ([model model]
                 [losses '()])
                ([epoch (range config.epochs)]
                 [batch (batches dataset config.batch-size)])
        (let-values ([(new-model loss) (train-step model batch optimizer)])
          (when (= (mod batch.index 100) 0)
            (printf "Epoch ~a, Batch ~a, Loss: ~a~n" epoch batch.index loss))
          (values new-model (cons loss losses))))))

) ;; end module qllm
```

### 4.3 Effort Comparison

```
Option A: Bridge Eshkol AD to qLLM C code
├── FFI bridge implementation      2 weeks
├── Type conversion (double↔f32)   1 week
├── Tensor AD operations           4 weeks
├── Chain rules for all ops        3 weeks
├── Testing and integration        2 weeks
└── Total                          12 weeks

Option B: Rewrite qLLM in Eshkol (with XLA)
├── XLA integration                4-6 weeks (separate project)
├── qLLM model translation         4 weeks
├── Geometric ops as custom calls  2 weeks
├── VJP rules for custom ops       2 weeks
├── Testing and validation         2 weeks
└── Total                          14-16 weeks (after XLA)

Option C: Port AD to C++ (in qLLM codebase)
├── Variable class implementation  2 weeks
├── Tensor AD operations           3 weeks
├── Integration with optimizer     2 weeks
├── Testing                        1 week
└── Total                          8 weeks
```

**Recommendation:**
- **Short term:** Option C (fastest path to training)
- **Long term:** Option B (best architecture with XLA)

---

## 5. Eshkol as a General-Purpose AI Language

### 5.1 Competitive Landscape

| Language | Type System | AD | Hardware | Probability | Symbolic |
|----------|-------------|-----|----------|-------------|----------|
| Python+PyTorch | Dynamic | Library | GPU | Library | No |
| Python+JAX | Dynamic | Library | XLA | Library | No |
| Julia | Dynamic | Library | GPU | Library | Limited |
| Mojo | Static | Library | GPU | No | No |
| Swift (TF) | Static | Library | XLA | No | No |
| **Eshkol** | **HoTT** | **Native** | **XLA** | **Native** | **Native** |

### 5.2 Eshkol's Unique Position

No existing language combines:

1. **Dependent Types for Tensors**
   - Compile-time shape checking
   - Type-safe broadcasting
   - Provably correct reshaping

2. **Native Automatic Differentiation**
   - Not a library, part of language semantics
   - Works on arbitrary code
   - Higher-order derivatives

3. **HoTT Type System**
   - Path types for equivalences
   - Higher inductive types
   - Proof-carrying code

4. **XLA Execution**
   - GPU/TPU without manual optimization
   - Automatic op fusion
   - Production-grade performance

5. **Probabilistic Types**
   - Distributions as first-class values
   - Inference as a primitive
   - Differentiable probabilistic programs

6. **Symbolic Integration**
   - Homoiconic representation
   - Term rewriting
   - Neural-symbolic reasoning

---

## 6. Tier 1: Essential Features

These features are required for Eshkol to be a viable AI language.

### 6.1 N-Dimensional Tensors with Dependent Types

#### 6.1.1 Type Definition

```scheme
;; Tensor type parameterized by dtype and shape
(define-type (Tensor [dtype : DType] [shape : (List Nat)])
  : (Universe 0)
  :representation (XLA-Buffer dtype shape))

;; Shape is a type-level list of natural numbers
(define-type Shape (List Nat))

;; DType enumeration
(define-type DType
  (Union Float64 Float32 Float16 BFloat16 Float8
         Int64 Int32 Int16 Int8
         UInt64 UInt32 UInt16 UInt8
         Bool Complex64 Complex128))
```

#### 6.1.2 Shape-Polymorphic Operations

```scheme
;; Matrix multiplication with dependent types
(define (matmul [A : (Tensor dtype [... m k])]    ; Batched [*, m, k]
                [B : (Tensor dtype [... k n])])   ; Batched [*, k, n]
  : (Tensor dtype [... m n])                      ; Result [*, m, n]
  (xla/dot-general A B
    :lhs-contracting-dims [-1]
    :rhs-contracting-dims [-2]))

;; Compile-time shape checking
(define x : (Tensor f32 [32 784]))    ; Batch of 784-dim vectors
(define W : (Tensor f32 [784 256]))   ; Weight matrix

(matmul x W)  ; OK: (Tensor f32 [32 256])

(define y : (Tensor f32 [32 100]))
(matmul x y)  ; TYPE ERROR: Cannot unify 784 with 100
```

#### 6.1.3 Broadcasting Semantics

```scheme
;; Broadcasting as a type-level function
(define-type-function (broadcast-shape s1 s2)
  (match (s1 s2)
    [(() s) s]
    [(s ()) s]
    [((cons d1 r1) (cons d2 r2))
     (cons (if (= d1 d2) d1
               (if (= d1 1) d2
                   (if (= d2 1) d1
                       (error "incompatible shapes"))))
           (broadcast-shape r1 r2))]))

;; Element-wise operations broadcast automatically
(define (add [A : (Tensor dtype shape-a)]
             [B : (Tensor dtype shape-b)])
  : (Tensor dtype (broadcast-shape shape-a shape-b))
  (xla/add A B))
```

#### 6.1.4 Implementation Plan

**Files to modify:**
- `inc/eshkol/eshkol.h` - Add tensor type tag
- `lib/types/hott_types.cpp` - Register Tensor type family
- `lib/types/dependent.cpp` - Add shape arithmetic
- `lib/backend/tensor_codegen.cpp` - XLA lowering for tensor ops

**Effort:** 3-4 weeks

### 6.2 Automatic Batching (vmap)

#### 6.2.1 Semantics

`vmap` transforms a function that operates on single examples to work on batches:

```scheme
;; Single-example function
(define (predict x)
  : (-> (Tensor f32 [784]) (Tensor f32 [10]))
  (softmax (matmul (relu (matmul x W1)) W2)))

;; Automatically batched
(define predict-batch (vmap predict))
;; Type: (-> (Tensor f32 [batch 784]) (Tensor f32 [batch 10]))

;; Works on any batch size
(predict-batch (randn [32 784]))   ; 32 examples
(predict-batch (randn [128 784]))  ; 128 examples
```

#### 6.2.2 Composability

```scheme
;; vmap composes with autodiff
(define batched-gradients
  (vmap (lambda (x y)
          (gradient (lambda (p) (loss (predict x p) y)) params))))

;; vmap composes with itself (nested batching)
(define double-batched
  (vmap (vmap single-fn :in-axes 0) :in-axes 1))
```

#### 6.2.3 Implementation

vmap is implemented as a source transformation:

```scheme
;; Transform:
(vmap (lambda (x) (+ x 1)) :in-axes 0)

;; Into:
(lambda (xs)
  (let ([batch-size (shape xs 0)])
    (xla/map (lambda (x) (+ x 1)) xs :dimensions [0])))
```

**Effort:** 3-4 weeks

### 6.3 Parallelism Primitives

#### 6.3.1 pmap: Data Parallelism

```scheme
;; Distribute computation across devices
(define (data-parallel-gradient loss-fn params data)
  (let* ([devices (available-devices)]
         [n-devices (length devices)]
         [sharded-data (shard data :axis 0 :num-shards n-devices)]
         ;; Compute gradients in parallel on each device
         [local-grads (pmap (lambda (device-data)
                              (gradient loss-fn params device-data))
                            sharded-data
                            :devices devices)]
         ;; All-reduce to combine gradients
         [total-grad (all-reduce local-grads :op :mean)])
    total-grad))
```

#### 6.3.2 shard: Model Parallelism

```scheme
;; Tensor parallelism for large layers
(define (tensor-parallel-linear x W devices)
  (let* ([W-shards (shard W :axis 1 :devices devices)]
         [local-results (pmap (lambda (W-shard)
                                (matmul x W-shard))
                              W-shards
                              :devices devices)]
         [result (all-gather local-results :axis 1)])
    result))

;; Pipeline parallelism
(define (pipeline-forward x layers devices micro-batches)
  (let* ([layer-assignments (zip layers devices)]
         [micro-batched-x (split x :axis 0 :num micro-batches)])
    (pipeline-execute layer-assignments micro-batched-x)))
```

#### 6.3.3 Implementation

Uses XLA's SPMD partitioner:

```scheme
;; Eshkol pmap lowers to XLA sharding annotations
(pmap f data :devices ["gpu:0" "gpu:1"])

;; Becomes:
;; sharding={devices=[0,1], replicated={}}
;; + XLA collective operations
```

**Effort:** 4-5 weeks

### 6.4 Mixed Precision and Quantization

#### 6.4.1 Precision Control

```scheme
;; First-class dtype specification
(define W : (Tensor bf16 [1024 4096]))

;; Automatic mixed precision context
(with-precision bf16
  :compute bf16       ; Compute in bf16
  :accumulate f32     ; Accumulate in f32
  :master f32         ; Keep master weights in f32
  (train-step model batch))

;; Loss scaling handled automatically
(with-loss-scaling :initial-scale 65536
                   :growth-interval 2000
  (train-step model batch))
```

#### 6.4.2 Quantization

```scheme
;; Post-training quantization
(define quantized-model
  (quantize trained-model
    :dtype int8
    :calibration-data calibration-set
    :method :minmax))  ; or :percentile, :mse

;; Quantization-aware training
(define (qat-forward model x)
  (let* ([W-fake-quant (fake-quantize model.W :bits 8)]
         [x-fake-quant (fake-quantize x :bits 8)])
    (matmul x-fake-quant W-fake-quant)))
```

**Effort:** 2-3 weeks

### 6.5 Gradient Checkpointing

#### 6.5.1 Memory-Compute Tradeoff

```scheme
;; Checkpoint activations to reduce memory
(define (memory-efficient-transformer x layers)
  (fold (lambda (x layer)
          (checkpoint  ; Recompute forward on backward pass
            (transformer-block x layer)))
        x
        layers))

;; Fine-grained control
(define (selective-checkpoint layer-idx)
  (if (= (mod layer-idx 2) 0)
      checkpoint      ; Checkpoint every other layer
      identity))      ; Store activations normally
```

#### 6.5.2 Implementation

Checkpointing is implemented as an AD transformation:

```scheme
;; checkpoint wraps a function to recompute on backward
(define-syntax checkpoint
  (syntax-rules ()
    [(checkpoint expr)
     (let ([saved-input (current-input)])
       ;; Forward: compute and return result, don't save intermediates
       ;; Backward: recompute forward to get intermediates, then backprop
       (with-recomputation saved-input expr))]))
```

**Effort:** 2 weeks

---

## 7. Tier 2: Advanced Features

These features differentiate Eshkol from other AI languages.

### 7.1 Probabilistic Programming

#### 7.1.1 Distributions as Types

```scheme
;; Distribution type constructor
(define-type (Distribution A)
  :where (Measurable? A)
  : (Universe 1))

;; Built-in distributions
(define (normal mu sigma) : (Distribution Real))
(define (bernoulli p) : (Distribution Bool))
(define (categorical logits) : (Distribution Nat))
(define (dirichlet alpha) : (Distribution (Simplex n)))

;; Sampling is an effect
(define-effect Sample)

(define (sample [d : (Distribution A)]) : {Sample} A
  (primitive-sample d))
```

#### 7.1.2 Probabilistic Models

```scheme
;; Bayesian neural network
(define (bayesian-nn x)
  : {Sample} (Distribution (Tensor f32 [10]))
  (let* ([W1 (sample (normal (zeros [784 256])
                             (ones [784 256])))]
         [W2 (sample (normal (zeros [256 10])
                             (ones [256 10])))]
         [h (relu (matmul x W1))]
         [logits (matmul h W2)])
    (categorical logits)))

;; Inference
(define posterior
  (infer bayesian-nn observations
    :method :hmc
    :num-samples 1000
    :warmup 500))

;; Posterior predictive
(define (predict-with-uncertainty x)
  (let ([samples (map (lambda (params)
                        (forward-with-params bayesian-nn x params))
                      (sample-posterior posterior 100))])
    (values (mean samples) (std samples))))
```

#### 7.1.3 Differentiable Inference

```scheme
;; Variational inference with reparameterization
(define (variational-objective model data variational-params)
  (let* ([q (variational-distribution variational-params)]
         [z (rsample q)]  ; Reparameterized sample (differentiable)
         [log-p (log-prob (joint model data) z)]
         [log-q (log-prob q z)]
         [elbo (- log-p log-q)])
    elbo))

;; Optimize variational parameters with AD
(define (train-vi model data)
  (gradient-descent
    (lambda (params) (- (variational-objective model data params)))
    initial-params
    :steps 10000))
```

**Effort:** 6-8 weeks

### 7.2 Symbolic-Neural Integration

#### 7.2.1 Symbolic Expressions

```scheme
;; Expression type (homoiconic - code as data)
(define-type Expr
  (Union
    (Sym Symbol)              ; Variable
    (Num Number)              ; Constant
    (App Symbol (List Expr))  ; Function application
    (Lam (List Symbol) Expr)  ; Lambda
    (If Expr Expr Expr)))     ; Conditional

;; Quote creates symbolic expressions
(define my-expr '(+ (* x x) (* 2 x) 1))
;; Type: Expr

;; Symbolic differentiation
(define (symbolic-diff expr var)
  (match expr
    [(Num _) (Num 0)]
    [(Sym v) (if (eq? v var) (Num 1) (Num 0))]
    [(App '+ args)
     (App '+ (map (lambda (a) (symbolic-diff a var)) args))]
    [(App '* [a b])
     (App '+ [(App '* [a (symbolic-diff b var)])
              (App '* [(symbolic-diff a var) b])])]
    [(App 'sin [a])
     (App '* [(App 'cos [a]) (symbolic-diff a var)])]
    ...))

;; Simplification
(define (simplify expr)
  (match expr
    [(App '+ [(Num 0) a]) (simplify a)]
    [(App '* [(Num 1) a]) (simplify a)]
    [(App '* [(Num 0) _]) (Num 0)]
    [_ expr]))
```

#### 7.2.2 Neural-Symbolic Reasoning

```scheme
;; Embed symbolic expressions as vectors
(define (embed-expr expr)
  (match expr
    [(Sym s) (embedding-lookup symbol-embeddings s)]
    [(Num n) (number-embedding n)]
    [(App f args)
     (let ([f-emb (embedding-lookup function-embeddings f)]
           [arg-embs (map embed-expr args)])
       (transformer-encode (cons f-emb arg-embs)))]))

;; Neural-guided symbolic search
(define (neural-symbolic-solve problem)
  (let loop ([state (initial-state problem)]
             [depth 0])
    (cond
      [(solved? state) (extract-solution state)]
      [(> depth max-depth) #f]
      [else
       (let* ([embedding (embed-state state)]
              [action-logits (policy-network embedding)]
              [action (sample (categorical action-logits))]
              [new-state (apply-action state action)])
         (loop new-state (+ depth 1)))])))

;; Differentiable theorem prover
(define (soft-unify t1 t2)
  (let ([e1 (embed-expr t1)]
        [e2 (embed-expr t2)])
    (sigmoid (- 1.0 (cosine-distance e1 e2)))))
```

**Effort:** 8-10 weeks

### 7.3 Differentiable Data Structures

#### 7.3.1 Soft Attention Dictionaries

```scheme
;; Differentiable key-value store
(define-type (DiffDict K V)
  (Record
    [keys : (Tensor f32 [n k-dim])]
    [values : (Tensor f32 [n v-dim])]))

(define (diff-write dict key value)
  (let* ([similarities (softmax (matmul key (transpose dict.keys)))]
         [erase-weights (- 1.0 similarities)]
         [new-values (+ (* dict.values (expand-dims erase-weights -1))
                        (* (expand-dims similarities -1) value))])
    (make-DiffDict dict.keys new-values)))

(define (diff-read dict query)
  (let* ([scores (matmul query (transpose dict.keys))]
         [weights (softmax scores)])
    (matmul weights dict.values)))
```

#### 7.3.2 Differentiable Sorting

```scheme
;; Soft sort using optimal transport
(define (soft-sort x temperature)
  (let* ([n (shape x 0)]
         [pairwise-diff (- (expand-dims x 1) (expand-dims x 0))]
         [perm-matrix (sinkhorn (/ pairwise-diff temperature))]
         [sorted (matmul perm-matrix x)])
    sorted))

;; Differentiable argsort
(define (soft-argsort x temperature)
  (let ([perm-matrix (soft-sort-perm x temperature)])
    (matmul perm-matrix (cast (range (shape x 0)) f32))))
```

**Effort:** 4-5 weeks

### 7.4 Neural ODEs and Continuous-Time Models

#### 7.4.1 ODE Solvers

```scheme
;; Differentiable ODE solver
(define (ode-solve dynamics y0 t-span [method 'dopri5])
  : (-> (-> (Tensor f32 [d]) f32 (Tensor f32 [d]))  ; dynamics
        (Tensor f32 [d])                             ; initial state
        (Pair f32 f32)                               ; time span
        (Tensor f32 [d]))                            ; final state
  (xla/custom-call "ode_solve"
    :adjoint-fn (adjoint-ode-solve dynamics)
    dynamics y0 t-span method))

;; Adjoint method for memory-efficient gradients
(define (adjoint-ode-solve dynamics)
  (lambda (y0 t-span g)
    ;; Solve augmented system backward in time
    (let* ([T (second t-span)]
           [augmented-dynamics
            (lambda (state t)
              (let ([y (slice state [0 d])]
                    [adj (slice state [d (* 2 d)])]
                    [params-grad (slice state [(* 2 d) ...])])
                (let* ([dy/dt (dynamics y t)]
                       [dadj/dt (- (vjp dynamics y adj))]
                       [dparams/dt (- (vjp-params dynamics y adj))])
                  (concat [dy/dt dadj/dt dparams/dt]))))]
           [final-state (ode-solve augmented-dynamics
                                   (concat [y-final g (zeros param-shape)])
                                   (pair T 0.0))])
      (slice final-state [(* 2 d) ...]))))
```

#### 7.4.2 Continuous Normalizing Flows

```scheme
;; Neural ODE for density estimation
(define (cnf-log-likelihood x dynamics t-span)
  (let* (;; Augment state with log-det term
         [augmented-dynamics
          (lambda (state t)
            (let ([z (slice state [0 d])]
                  [log-det (slice state [d])])
              (let* ([dz/dt (dynamics z t)]
                     [trace (hutchinson-trace-estimate
                             (jacobian dynamics z) z)])
                (concat [dz/dt (tensor [trace])]))))]
         [result (ode-solve augmented-dynamics
                           (concat [x (tensor [0.0])])
                           t-span)]
         [z-final (slice result [0 d])]
         [log-det (slice result [d])])
    (- (base-log-prob z-final) log-det)))
```

**Effort:** 4-5 weeks

### 7.5 Effect System for AI

#### 7.5.1 Effect Declarations

```scheme
;; Define effects for AI computations
(define-effect Random)       ; Uses random sampling
(define-effect Gradient)     ; Requires gradient tracking
(define-effect Device)       ; Has device placement
(define-effect Checkpoint)   ; Uses checkpointing
(define-effect Distributed)  ; Requires communication

;; Functions declare their effects
(define (dropout x p)
  : {Random Gradient} (-> (Tensor f32 shape) f32 (Tensor f32 shape))
  (let ([mask (sample (bernoulli (- 1.0 p)) (shape x))])
    (* x (/ mask (- 1.0 p)))))

(define (batch-norm x params training)
  : {Gradient (if training Stateful Pure)}
  (-> (Tensor f32 [...]) BatchNormParams Bool (Tensor f32 [...]))
  ...)
```

#### 7.5.2 Effect Handlers

```scheme
;; Handler for inference mode (no randomness, no gradients)
(define-handler inference-mode
  [Random -> deterministic-mode]   ; Use mean instead of sampling
  [Gradient -> no-gradient])       ; Don't track gradients

;; Handler for device placement
(define-handler (on-device device)
  [Device -> (place device)])

;; Usage
(with-handler inference-mode
  (forward model test-data))

(with-handler (on-device "cuda:0")
  (train-step model batch))
```

**Effort:** 5-6 weeks

---

## 8. Tier 3: Cutting-Edge Features

These features position Eshkol for future AI paradigms.

### 8.1 Quantum Computing Integration

#### 8.1.1 Quantum Circuits

```scheme
;; Quantum circuit as a first-class value
(define-type (QuantumCircuit [n-qubits : Nat]))

;; Circuit construction
(define (variational-ansatz params n-qubits depth)
  : (QuantumCircuit n-qubits)
  (circuit
    (for ([layer (range depth)])
      ;; Rotation layer
      (for ([q (range n-qubits)])
        (ry (aref params layer q 0) q)
        (rz (aref params layer q 1) q))
      ;; Entanglement layer
      (for ([q (range (- n-qubits 1))])
        (cnot q (+ q 1))))))

;; Quantum-classical hybrid
(define (quantum-kernel x1 x2 params)
  (let* ([circuit (quantum-feature-map x1 x2 params)]
         [result (execute circuit :backend "ibm_qiskit")]
         [measurement (measure result)])
    (expectation measurement)))
```

#### 8.1.2 Quantum Gradients

```scheme
;; Parameter shift rule for quantum gradients
(define (quantum-gradient circuit params)
  (let ([shift (/ pi 2)])
    (for/tensor ([i (range (length params))])
      (let ([params+ (update params i (+ (aref params i) shift))]
            [params- (update params i (- (aref params i) shift))])
        (/ (- (execute-expectation circuit params+)
              (execute-expectation circuit params-))
           2.0)))))

;; Variational quantum eigensolver
(define (vqe hamiltonian ansatz initial-params)
  (gradient-descent
    (lambda (params)
      (expectation (execute (ansatz params)) hamiltonian))
    initial-params
    :optimizer :adam))
```

**Effort:** 8-12 weeks

### 8.2 Formal Verification of Neural Networks

#### 8.2.1 Contracts and Specifications

```scheme
;; Bounded output contract
(define/contract (bounded-classifier network x)
  (-> (Tensor f32 [... d]) (Tensor f32 [... c])
      #:post (lambda (y) (all (<= 0.0 y 1.0))))
  (softmax (network x)))

;; Lipschitz constraint
(define/lipschitz spectral-normed-layer
  (lambda (x) (matmul (spectral-normalize W) x))
  #:lipschitz-constant 1.0)

;; Monotonicity constraint
(define/monotonic (monotonic-network x)
  #:with-respect-to x
  (certified-monotonic-forward x))
```

#### 8.2.2 Abstract Interpretation

```scheme
;; Interval bound propagation
(define (ibp-forward network x-lower x-upper)
  (fold (lambda (bounds layer)
          (let ([lower (first bounds)]
                [upper (second bounds)])
            (abstract-layer-forward layer lower upper)))
        (list x-lower x-upper)
        network.layers))

;; Certified robustness
(define (certified-robust? network x epsilon target)
  (let* ([x-lower (- x epsilon)]
         [x-upper (+ x epsilon)]
         [bounds (ibp-forward network x-lower x-upper)]
         [output-lower (first bounds)]
         [output-upper (second bounds)])
    ;; Check that target class is always highest
    (for/and ([c (range num-classes)])
      (or (= c target)
          (> (aref output-lower target)
             (aref output-upper c))))))
```

**Effort:** 10-12 weeks

### 8.3 Self-Improving Code Generation

#### 8.3.1 Metaprogramming for Optimization

```scheme
;; Analyze computation and generate optimized kernel
(define-macro (optimize-kernel expr target)
  (let* ([ast (quote expr)]
         [analysis (analyze-computation ast)]
         [candidates (generate-implementations analysis target)]
         [best (select-best candidates target)])
    best))

;; Usage
(optimize-kernel
  (lambda (A B)
    (matmul (transpose A) B))
  :target 'cuda
  :constraints {:shared-memory 48000})

;; Generates optimized CUDA kernel with tiling
```

#### 8.3.2 Neural Program Synthesis

```scheme
;; Synthesize function from examples
(define (synthesize spec examples)
  (let* ([encoder (program-encoder)]
         [decoder (program-decoder)]
         ;; Encode specification and examples
         [spec-embedding (encoder.encode-spec spec)]
         [example-embeddings (map encoder.encode-example examples)]
         [context (transformer-encode
                   (cons spec-embedding example-embeddings))]
         ;; Decode program tokens
         [program-tokens (beam-search decoder context :beam-width 10)]
         ;; Parse and verify
         [programs (filter-map parse-program program-tokens)]
         [valid (filter (lambda (p) (satisfies? p spec examples)) programs)])
    (first valid)))
```

**Effort:** 12-16 weeks

### 8.4 Federated and Privacy-Preserving Learning

#### 8.4.1 Differential Privacy

```scheme
;; Differential privacy effect
(define-effect (DifferentiallyPrivate epsilon delta))

(define (private-gradient loss params data epsilon)
  : {(DifferentiallyPrivate epsilon 1e-5)} (Tensor f32 param-shape)
  (let* ([grads (gradient loss params data)]
         [clipped (clip-by-global-norm grads 1.0)]
         [noise-scale (compute-noise-scale epsilon 1e-5 1.0)]
         [noised (+ clipped (sample (normal 0.0 noise-scale)
                                    (shape clipped)))])
    noised))

;; Privacy accounting
(with-privacy-budget [epsilon 1.0] [delta 1e-5]
  (for ([epoch (range 100)])
    (let ([grad (private-gradient loss params batch epsilon-per-step)])
      (update-params params grad))))
```

#### 8.4.2 Federated Learning

```scheme
;; Federated averaging
(define (federated-round global-model client-data)
  (let* ([client-updates
          (federated-map
            (lambda (local-data)
              (local-sgd global-model local-data :steps 10))
            client-data
            :secure #t)]
         ;; Secure aggregation
         [aggregated (secure-aggregate client-updates :method :secagg)])
    (federated-average global-model aggregated)))

;; Secure aggregation primitive
(define (secure-aggregate updates method)
  : {Encrypted} (Tensor f32 model-shape)
  (match method
    [:secagg (secagg-protocol updates)]
    [:he (homomorphic-aggregate updates)]
    [:mpc (mpc-aggregate updates)]))
```

**Effort:** 8-10 weeks

---

## 9. Tier 4: Ecosystem

These features enable adoption and productivity.

### 9.1 Python Interoperability

```scheme
;; Import Python modules
(import-python numpy :as np)
(import-python torch)
(import-python transformers :only [AutoModel AutoTokenizer])

;; Use Python libraries
(define tokenizer
  (AutoTokenizer.from_pretrained "bert-base-uncased"))

(define (tokenize text)
  (tokenizer text
    :return_tensors "np"
    :padding #t
    :truncation #t))

;; Convert between Eshkol and NumPy
(define (eshkol->numpy t)
  (np.array (tensor->list t) :dtype np.float32))

(define (numpy->eshkol arr)
  (tensor (np.tolist arr)))

;; Export Eshkol functions to Python
(export-to-python my-model
  :name "EshkolTransformer"
  :signature [(Tensor f32 [batch seq]) -> (Tensor f32 [batch seq vocab])])
```

**Effort:** 4-5 weeks

### 9.2 ONNX Import/Export

```scheme
;; Export to ONNX
(define (export-onnx model path sample-input)
  (let* ([traced (trace model sample-input)]
         [onnx-graph (to-onnx-graph traced)]
         [onnx-model (make-onnx-model onnx-graph
                       :opset-version 17
                       :producer-name "Eshkol")])
    (save-onnx onnx-model path)))

;; Import from ONNX
(define (import-onnx path)
  (let* ([onnx-model (load-onnx path)]
         [eshkol-fn (from-onnx-graph onnx-model.graph)])
    ;; Returns a callable with AD support
    eshkol-fn))

;; Use imported model
(define bert (import-onnx "bert.onnx"))
(define embeddings (bert input-ids attention-mask))
(define grads (gradient (lambda (x) (sum (bert x mask))) input-ids))
```

**Effort:** 3-4 weeks

### 9.3 Visualization and Debugging

```scheme
;; Computation graph visualization
(define (visualize-model model sample-input)
  (let ([traced (trace model sample-input)])
    (render-graph traced
      :format 'svg
      :show-shapes #t
      :show-dtypes #t
      :highlight-bottlenecks #t)))

;; Gradient debugging
(define (debug-gradients loss-fn params)
  (let* ([grads (gradient loss-fn params)]
         [stats (gradient-statistics grads)])
    (when (has-nan? grads)
      (warn "NaN gradients detected"))
    (when (has-vanishing? stats)
      (warn "Vanishing gradients in: ~a" (vanishing-params stats)))
    (when (has-exploding? stats)
      (warn "Exploding gradients in: ~a" (exploding-params stats)))
    (plot-gradient-distribution grads)))

;; Profiling
(define (profile-model model input)
  (let ([result (with-profiling (model input))])
    (print-profile result
      :show-time #t
      :show-memory #t
      :show-flops #t
      :show-memory-bandwidth #t)))
```

**Effort:** 4-5 weeks

### 9.4 Model Hub Integration

```scheme
;; Load from hub
(define model
  (load-pretrained "eshkol-hub/llama-7b"
    :dtype 'bf16
    :device "cuda:0"))

;; Fine-tuning with LoRA
(define fine-tuned
  (lora model
    :rank 8
    :alpha 16
    :target-modules ["q_proj" "v_proj" "k_proj" "o_proj"]))

;; Train
(train fine-tuned training-data
  :epochs 3
  :learning-rate 1e-4)

;; Push to hub
(push-to-hub fine-tuned
  :repo "my-org/my-fine-tuned-llama"
  :private #t
  :readme (generate-model-card fine-tuned))
```

**Effort:** 3-4 weeks

---

## 10. Implementation Specifications

### 10.1 Type System Extensions

#### 10.1.1 Tensor Type Family

**File:** `lib/types/hott_types.cpp`

```cpp
// Register Tensor as a type family
void HoTTTypeSystem::registerTensorType() {
    TypeFamily tensor;
    tensor.id = TypeId::Tensor;
    tensor.name = "Tensor";
    tensor.universe = Universe::U0;
    tensor.params = {"dtype", "shape"};  // Dependent on dtype and shape
    tensor.representation = RuntimeRep::XLABuffer;

    registerTypeFamily(tensor);

    // Register shape arithmetic
    registerTypeLevelFunction("broadcast-shape", broadcastShapeImpl);
    registerTypeLevelFunction("matmul-shape", matmulShapeImpl);
    registerTypeLevelFunction("transpose-shape", transposeShapeImpl);
}

// Shape type (list of naturals)
TypeId HoTTTypeSystem::shapeType() {
    return makeListType(TypeId::Nat);
}

// Check tensor shape compatibility
bool HoTTTypeSystem::checkMatmulShapes(TypeId A_shape, TypeId B_shape) {
    // A: [... m k], B: [... k n] -> [... m n]
    auto A_dims = extractShapeDims(A_shape);
    auto B_dims = extractShapeDims(B_shape);

    if (A_dims.size() < 2 || B_dims.size() < 2) return false;

    size_t A_k = A_dims[A_dims.size() - 1];
    size_t B_k = B_dims[B_dims.size() - 2];

    return unifyNats(A_k, B_k);  // k dimensions must match
}
```

#### 10.1.2 Effect Types

**File:** `lib/types/effects.cpp`

```cpp
// Effect type representation
struct Effect {
    std::string name;
    std::vector<TypeId> params;  // Effect parameters
};

// Effect row (set of effects)
struct EffectRow {
    std::set<Effect> effects;
    bool is_open;  // Can have more effects
};

// Function type with effects
struct FunctionType {
    std::vector<TypeId> param_types;
    TypeId return_type;
    EffectRow effects;
};

// Check effect compatibility
bool effectsCompatible(const EffectRow& required, const EffectRow& provided) {
    for (const auto& eff : required.effects) {
        if (provided.effects.find(eff) == provided.effects.end()) {
            return false;
        }
    }
    return true;
}
```

### 10.2 AD System Extensions

#### 10.2.1 New AD Node Types

**File:** `inc/eshkol/eshkol.h`

```c
// Extended AD node types for tensors
typedef enum {
    // Existing scalar types (0-15)
    AD_NODE_CONST = 0,
    AD_NODE_VAR = 1,
    AD_NODE_ADD = 2,
    AD_NODE_SUB = 3,
    AD_NODE_MUL = 4,
    AD_NODE_DIV = 5,
    AD_NODE_SIN = 6,
    AD_NODE_COS = 7,
    AD_NODE_EXP = 8,
    AD_NODE_LOG = 9,
    AD_NODE_POW = 10,
    AD_NODE_SQRT = 11,
    AD_NODE_TANH = 12,
    AD_NODE_NEG = 13,

    // Tensor operations (16-31)
    AD_NODE_MATMUL = 16,
    AD_NODE_SOFTMAX = 17,
    AD_NODE_LAYERNORM = 18,
    AD_NODE_ATTENTION = 19,
    AD_NODE_TRANSPOSE = 20,
    AD_NODE_REDUCE_SUM = 21,
    AD_NODE_REDUCE_MEAN = 22,
    AD_NODE_BROADCAST = 23,
    AD_NODE_RESHAPE = 24,
    AD_NODE_GATHER = 25,
    AD_NODE_SCATTER = 26,
    AD_NODE_CONV2D = 27,
    AD_NODE_MAXPOOL = 28,
    AD_NODE_BATCHNORM = 29,
    AD_NODE_GELU = 30,
    AD_NODE_DROPOUT = 31,

    // Geometric operations (32-47)
    AD_NODE_HYPERBOLIC_DIST = 32,
    AD_NODE_POINCARE_EXP = 33,
    AD_NODE_POINCARE_LOG = 34,
    AD_NODE_TANGENT_PROJECT = 35,
    AD_NODE_GEODESIC_ATTN = 36,

    // Custom operations (48+)
    AD_NODE_CUSTOM = 48
} ad_node_type_t;

// Extended AD node structure for tensors
typedef struct eshkol_ad_node {
    uint32_t type;
    uint64_t node_id;

    // For scalar operations
    double scalar_value;
    double scalar_grad;

    // For tensor operations
    void* tensor_data;      // xla::Tensor* or qllm_tensor_t*
    void* tensor_grad;
    size_t* shape;
    size_t dims;

    // Graph structure
    struct eshkol_ad_node* inputs[4];  // Up to 4 inputs
    size_t num_inputs;

    // Operation-specific data
    union {
        struct { int axis; } reduce;
        struct { int dim; } softmax;
        struct { float eps; } layernorm;
        struct { int num_heads; float scale; } attention;
        struct { float curvature; } hyperbolic;
        struct { void* custom_data; } custom;
    } op_data;

    // Cached forward pass values (for backward)
    void* cached[4];
} eshkol_ad_node_t;
```

#### 10.2.2 Tensor Gradient Propagation

**File:** `lib/backend/autodiff_codegen.cpp`

```cpp
void AutodiffCodegen::propagateGradientTensor(llvm::Value* node_ptr) {
    // Load node type
    llvm::Value* type = loadNodeType(node_ptr);
    llvm::Value* grad = loadTensorGrad(node_ptr);

    // Create switch for tensor ops
    llvm::SwitchInst* sw = builder.CreateSwitch(type, default_block, 16);

    // MATMUL: dA = dC @ B^T, dB = A^T @ dC
    sw->addCase(ConstantInt::get(i32, AD_NODE_MATMUL), matmul_block);
    builder.SetInsertPoint(matmul_block);
    {
        llvm::Value* A = loadInput(node_ptr, 0);
        llvm::Value* B = loadInput(node_ptr, 1);

        llvm::Value* B_T = callXLA("transpose", B, {-1, -2});
        llvm::Value* grad_A = callXLA("matmul", grad, B_T);
        accumulateTensorGrad(A, grad_A);

        llvm::Value* A_T = callXLA("transpose", A, {-1, -2});
        llvm::Value* grad_B = callXLA("matmul", A_T, grad);
        accumulateTensorGrad(B, grad_B);
    }
    builder.CreateBr(done_block);

    // SOFTMAX: dx = y * (dy - sum(dy * y, dim, keepdim=True))
    sw->addCase(ConstantInt::get(i32, AD_NODE_SOFTMAX), softmax_block);
    builder.SetInsertPoint(softmax_block);
    {
        llvm::Value* y = loadCached(node_ptr, 0);  // Softmax output
        llvm::Value* dim = loadOpDataInt(node_ptr, "softmax.dim");

        llvm::Value* prod = callXLA("mul", grad, y);
        llvm::Value* sum_prod = callXLA("reduce_sum", prod, dim, true);
        llvm::Value* diff = callXLA("sub", grad, sum_prod);
        llvm::Value* grad_x = callXLA("mul", y, diff);

        accumulateTensorGrad(loadInput(node_ptr, 0), grad_x);
    }
    builder.CreateBr(done_block);

    // LAYERNORM: Complex gradient (see implementation)
    sw->addCase(ConstantInt::get(i32, AD_NODE_LAYERNORM), layernorm_block);
    // ... implementation ...

    // ATTENTION: Multi-step chain rule
    sw->addCase(ConstantInt::get(i32, AD_NODE_ATTENTION), attention_block);
    // ... implementation ...

    // HYPERBOLIC_DIST: Poincare ball gradient
    sw->addCase(ConstantInt::get(i32, AD_NODE_HYPERBOLIC_DIST), hyperbolic_block);
    builder.SetInsertPoint(hyperbolic_block);
    {
        // Custom call to qLLM gradient function
        llvm::Value* p1 = loadInput(node_ptr, 0);
        llvm::Value* p2 = loadInput(node_ptr, 1);
        llvm::Value* curvature = loadOpDataFloat(node_ptr, "hyperbolic.curvature");

        llvm::Function* grad_fn = getQllmFunction("hyperbolic_distance_grad");
        llvm::Value* grads = builder.CreateCall(grad_fn, {p1, p2, curvature, grad});

        llvm::Value* grad_p1 = builder.CreateExtractValue(grads, 0);
        llvm::Value* grad_p2 = builder.CreateExtractValue(grads, 1);

        accumulateTensorGrad(p1, grad_p1);
        accumulateTensorGrad(p2, grad_p2);
    }
    builder.CreateBr(done_block);
}
```

### 10.3 XLA Lowering

#### 10.3.1 StableHLO Emission

**File:** `lib/backend/xla_lowering.cpp`

```cpp
class XLALowering {
public:
    // Lower Eshkol AST to StableHLO
    mlir::func::FuncOp lowerFunction(const eshkol_ast_t* ast) {
        mlir::OpBuilder builder(context);

        // Create function signature
        auto funcType = lowerFunctionType(ast->type);
        auto funcOp = builder.create<mlir::func::FuncOp>(
            loc, ast->name, funcType);

        // Lower body
        builder.setInsertionPointToStart(funcOp.addEntryBlock());
        mlir::Value result = lowerExpr(ast->body, builder);
        builder.create<mlir::func::ReturnOp>(loc, result);

        return funcOp;
    }

    mlir::Value lowerExpr(const eshkol_ast_t* expr, mlir::OpBuilder& builder) {
        switch (expr->type) {
            case AST_LITERAL:
                return lowerLiteral(expr, builder);

            case AST_BINARY_OP:
                return lowerBinaryOp(expr, builder);

            case AST_CALL:
                return lowerCall(expr, builder);

            case AST_MATMUL:
                return lowerMatmul(expr, builder);

            case AST_SOFTMAX:
                return lowerSoftmax(expr, builder);

            // ... more cases
        }
    }

    mlir::Value lowerMatmul(const eshkol_ast_t* expr, mlir::OpBuilder& builder) {
        mlir::Value lhs = lowerExpr(expr->children[0], builder);
        mlir::Value rhs = lowerExpr(expr->children[1], builder);

        // Create stablehlo.dot_general
        auto dotDimensionNumbers = mlir::stablehlo::DotDimensionNumbersAttr::get(
            context,
            /*lhsBatchingDimensions=*/{},
            /*rhsBatchingDimensions=*/{},
            /*lhsContractingDimensions=*/{-1},
            /*rhsContractingDimensions=*/{-2});

        return builder.create<mlir::stablehlo::DotGeneralOp>(
            loc, resultType, lhs, rhs, dotDimensionNumbers, nullptr);
    }

    mlir::Value lowerCustomCall(const eshkol_ast_t* expr, mlir::OpBuilder& builder) {
        // For qLLM geometric operations
        std::vector<mlir::Value> operands;
        for (auto child : expr->children) {
            operands.push_back(lowerExpr(child, builder));
        }

        return builder.create<mlir::stablehlo::CustomCallOp>(
            loc, resultTypes, operands,
            /*call_target_name=*/expr->custom_call_name,
            /*has_side_effect=*/false,
            /*backend_config=*/"",
            /*api_version=*/1);
    }
};
```

### 10.4 Required Chain Rules

#### 10.4.1 Complete Backward Pass Specifications

| Operation | Forward | Backward |
|-----------|---------|----------|
| **MatMul** | C = A @ B | dA = dC @ B^T, dB = A^T @ dC |
| **Softmax** | y = softmax(x, dim) | dx = y * (dy - sum(dy * y, dim)) |
| **LayerNorm** | y = (x-μ)/σ * γ + β | Complex (see below) |
| **Attention** | Multi-step | 5-step chain rule |
| **GELU** | y = 0.5x(1+tanh(√(2/π)(x+0.044715x³))) | dx = dy * gelu'(x) |
| **Hyperbolic Dist** | d = arccosh(1+2||x-y||²/((1-||x||²)(1-||y||²))) | See Poincaré gradient |

**LayerNorm Backward:**
```
Input: x, Output: y = (x - μ) / σ * γ + β
μ = mean(x), σ = sqrt(var(x) + ε)

Gradients:
dγ = sum(dy * (x - μ) / σ, batch_dims)
dβ = sum(dy, batch_dims)

dx = (1/σ) * (dy * γ - mean(dy * γ) - (x-μ)/σ² * mean(dy * γ * (x-μ)/σ))
```

**Attention Backward:**
```
Forward:
  scores = Q @ K^T / √d_k
  weights = softmax(scores)
  context = weights @ V
  output = context @ W_O

Backward:
  1. d_context = d_output @ W_O^T
     d_W_O = context^T @ d_output

  2. d_weights = d_context @ V^T
     d_V = weights^T @ d_context

  3. d_scores = softmax_backward(d_weights, weights)

  4. d_scores_scaled = d_scores / √d_k

  5. d_Q = d_scores_scaled @ K
     d_K = d_scores_scaled^T @ Q
```

---

## 11. Development Roadmap

### Phase 1: Core ML Infrastructure (Q1 - 12 weeks)

```
Week 1-4: XLA Integration Foundation
├── Set up StableHLO emission pipeline
├── Basic tensor operations (add, mul, matmul)
├── JIT compilation working
└── Milestone: matmul executes on GPU

Week 5-8: N-D Tensors with Dependent Types
├── Tensor type family in HoTT
├── Shape polymorphism
├── Broadcasting type rules
└── Milestone: Shape errors caught at compile time

Week 9-12: Automatic Batching and Parallelism
├── vmap implementation
├── pmap for data parallelism
├── Integration with XLA SPMD
└── Milestone: Multi-GPU training works
```

### Phase 2: Advanced AD (Q2 - 10 weeks)

```
Week 13-16: Tensor AD Operations
├── MatMul backward
├── Softmax backward
├── LayerNorm backward
├── Attention backward
└── Milestone: Transformer training works

Week 17-20: Higher-Order AD
├── Hessian computation
├── Neural ODE adjoint method
├── Implicit differentiation
└── Milestone: Second-order optimization works

Week 21-22: Custom Operations
├── Custom VJP/JVP registration
├── qLLM geometric ops integration
├── Gradient verification framework
└── Milestone: qLLM trains end-to-end
```

### Phase 3: Probabilistic and Symbolic (Q3 - 8 weeks)

```
Week 23-26: Probabilistic Programming
├── Distribution types
├── Sample effect
├── HMC inference
├── Variational inference
└── Milestone: Bayesian neural network trains

Week 27-30: Symbolic Integration
├── Symbolic expression type
├── Term rewriting
├── Neural-symbolic search
└── Milestone: Neural theorem prover demo
```

### Phase 4: Ecosystem (Q4 - 8 weeks)

```
Week 31-34: Interoperability
├── Python FFI
├── ONNX import/export
├── NumPy compatibility
└── Milestone: Import PyTorch model, train in Eshkol

Week 35-38: Developer Experience
├── Visualization tools
├── Gradient debugging
├── Profiler
├── Model hub
└── Milestone: Complete development workflow
```

### Phase 5: Cutting Edge (Year 2)

```
Q1: Quantum Computing
├── Quantum circuit primitives
├── Parameter shift rule
├── VQE demo

Q2: Formal Verification
├── Interval bound propagation
├── Certified robustness
├── Lipschitz constraints

Q3: Privacy and Distribution
├── Differential privacy
├── Federated learning
├── Secure aggregation

Q4: Self-Improvement
├── Neural program synthesis
├── Auto-tuning compilation
├── Meta-learning integration
```

---

## 12. Conclusion

### 12.1 Summary

This document has presented a comprehensive vision for Eshkol as the premier AI systems programming language. The key insights are:

1. **Eshkol's foundation is uniquely suited for AI**: The combination of HoTT types, native autodiff, and functional purity provides capabilities no other language offers.

2. **XLA integration is the critical enabler**: With XLA, Eshkol gains production-grade tensor operations, GPU/TPU execution, and automatic optimization without reimplementing low-level infrastructure.

3. **The qLLM training problem is solvable**: Whether through bridging or rewriting, Eshkol can provide the missing autodiff capability for the semi-classical qLLM.

4. **A complete AI language is achievable**: The roadmap shows a realistic path to implementing all necessary features within 12-18 months.

### 12.2 Unique Value Proposition

Eshkol would be the first language to combine:

| Capability | Benefit |
|------------|---------|
| **Dependent tensor types** | Compile-time shape errors |
| **Native autodiff** | Differentiate any code |
| **HoTT foundation** | Provably correct transformations |
| **XLA backend** | Production performance |
| **Probabilistic types** | First-class uncertainty |
| **Symbolic integration** | Neural-symbolic AI |
| **Effect system** | Controlled side effects |
| **Homoiconicity** | Metaprogramming and synthesis |

### 12.3 Strategic Recommendation

**Invest in making Eshkol a complete AI systems language.**

The ML/AI language landscape is evolving rapidly, but no existing solution combines type safety, autodiff, performance, and expressiveness the way Eshkol can. The HoTT type system is a genuine innovation that could enable new programming paradigms for AI.

The recommended approach:
1. **Immediate**: Complete XLA integration
2. **Short-term**: Implement Tier 1 features (tensors, vmap, pmap)
3. **Medium-term**: Add Tier 2 features (probabilistic, symbolic)
4. **Long-term**: Build ecosystem and cutting-edge capabilities

With this investment, Eshkol can become the language that AI researchers and engineers reach for when they need correctness, performance, and expressiveness together.

---

## Appendix A: File Listing

### Eshkol Files Referenced

| File | Purpose |
|------|---------|
| `inc/eshkol/eshkol.h` | Core type definitions |
| `lib/types/hott_types.cpp` | HoTT type system |
| `lib/backend/autodiff_codegen.cpp` | AD implementation |
| `lib/backend/autodiff_codegen.h` | AD interface |
| `lib/backend/llvm_codegen.cpp` | LLVM code generation |
| `lib/core/arena_memory.cpp` | Memory management |

### qLLM Files Referenced

| File | Purpose |
|------|---------|
| `src/core/tensor.c` | Tensor implementation |
| `src/model/transformer.c` | Transformer blocks |
| `src/model/attention.c` | Attention mechanisms |
| `src/geometric/hyperbolic_core.c` | Poincaré ball operations |
| `src/optimization/riemannian_adam.c` | Riemannian optimizer |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **AD** | Automatic Differentiation |
| **HoTT** | Homotopy Type Theory |
| **VJP** | Vector-Jacobian Product (reverse mode AD) |
| **JVP** | Jacobian-Vector Product (forward mode AD) |
| **XLA** | Accelerated Linear Algebra (Google's ML compiler) |
| **StableHLO** | Stable High-Level Operations (XLA IR) |
| **vmap** | Vectorizing map (automatic batching) |
| **pmap** | Parallel map (data parallelism) |
| **SPMD** | Single Program Multiple Data |

---

*Document generated: December 7, 2025*
*Eshkol Version: 1.0.0-foundation*
