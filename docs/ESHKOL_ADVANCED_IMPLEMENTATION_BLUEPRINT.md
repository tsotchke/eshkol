# Eshkol Advanced Implementation Blueprint

## A Complete Technical Guide for Systems-Level Intelligent Computing

**Document Version**: 2.0.0
**Last Updated**: December 2024
**Status**: Engineering Specification

---

## Table of Contents

1. [Executive Architecture Summary](#1-executive-architecture-summary)
2. [Current Implementation Deep Dive](#2-current-implementation-deep-dive)
3. [SIMD/Vectorization Implementation](#3-simdvectorization-implementation)
4. [GPU/CUDA Backend Implementation](#4-gpucuda-backend-implementation)
5. [XLA/MLIR Integration](#5-xamlir-integration)
6. [Parallel Execution Framework](#6-parallel-execution-framework)
7. [Hygienic Macro System](#7-hygienic-macro-system)
8. [Neural Network Primitives](#8-neural-network-primitives)
9. [Distributed Computing Framework](#9-distributed-computing-framework)
10. [Serialization & Checkpointing](#10-serialization--checkpointing)
11. [Advanced Memory Optimization](#11-advanced-memory-optimization)
12. [Profiling & Debugging Infrastructure](#12-profiling--debugging-infrastructure)
13. [Extended Tensor Operations](#13-extended-tensor-operations)
14. [Optimizer Implementations](#14-optimizer-implementations)
15. [Implementation Roadmap](#15-implementation-roadmap)

---

## 1. Executive Architecture Summary

### 1.1 Eshkol Core Architecture

Eshkol is a Scheme dialect that compiles to native code via LLVM with the following unique characteristics:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ESHKOL ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  Source (.esk)                                                              │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────┐    ┌──────────────────┐    ┌────────────────────────────┐ │
│  │   Parser    │───▶│  HoTT Type Check │───▶│   LLVM IR Generation       │ │
│  │  (46+ ops)  │    │  (56,490 lines)  │    │   (24,068 lines)           │ │
│  └─────────────┘    └──────────────────┘    └────────────────────────────┘ │
│       │                    │                         │                      │
│       │                    │                         ▼                      │
│       │                    │              ┌──────────────────────┐          │
│       │                    │              │  Machine Code (.o)   │          │
│       │                    │              │  or JIT Execution    │          │
│       │                    │              └──────────────────────┘          │
├───────┴────────────────────┴─────────────────────────────────────────────────┤
│                        RUNTIME COMPONENTS                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────────┐ │
│  │ Arena Allocator│  │  AD Tape Stack │  │  Lambda Registry (Homoiconic) │ │
│  │ (OALR Regions) │  │  (32-level)    │  │  (S-expr preservation)        │ │
│  └────────────────┘  └────────────────┘  └────────────────────────────────┘ │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────────┐ │
│  │ Tagged Values  │  │ Reference Count│  │  Closure Environment          │ │
│  │ (14 types)     │  │ (Weak refs)    │  │  (Variadic capture)           │ │
│  └────────────────┘  └────────────────┘  └────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Existing Capabilities

| Capability | Implementation Location | Lines of Code |
|------------|------------------------|---------------|
| HoTT Type System | `lib/types/hott_types.cpp` | 1,200+ |
| Dependent Types | `lib/types/dependent.cpp` | 197 |
| Type Checker | `lib/types/type_checker.cpp` | 2,500+ |
| LLVM Codegen | `lib/backend/llvm_codegen.cpp` | 24,068 |
| Autodiff Codegen | `lib/backend/autodiff_codegen.cpp` | 2,000+ |
| Tensor Codegen | `lib/backend/tensor_codegen.cpp` | 1,500+ |
| Arena Memory | `lib/core/arena_memory.cpp` | 1,200+ |
| Tagged Value Codegen | `lib/backend/tagged_value_codegen.cpp` | 800+ |

---

## 2. Current Implementation Deep Dive

### 2.1 Tagged Value System

The runtime type system uses a 16-byte tagged value structure:

```cpp
// File: inc/eshkol/eshkol.h:78-90
typedef struct eshkol_tagged_value {
    uint8_t type;        // Value type (eshkol_value_type_t)
    uint8_t flags;       // Exactness and other flags
    uint16_t reserved;   // Reserved for future use
    union {
        int64_t int_val;
        double double_val;
        uint64_t ptr_val;
        uint64_t raw_val;
    } data;
} eshkol_tagged_value_t;

// Value types (14 total):
typedef enum {
    ESHKOL_VALUE_NULL        = 0,   // Empty/null value
    ESHKOL_VALUE_INT64       = 1,   // 64-bit signed integer
    ESHKOL_VALUE_DOUBLE      = 2,   // Double-precision floating point
    ESHKOL_VALUE_CONS_PTR    = 3,   // Pointer to cons cell
    ESHKOL_VALUE_DUAL_NUMBER = 4,   // Dual number for forward-mode AD
    ESHKOL_VALUE_AD_NODE_PTR = 5,   // Pointer to AD computation graph node
    ESHKOL_VALUE_TENSOR_PTR  = 6,   // Pointer to tensor structure
    ESHKOL_VALUE_LAMBDA_SEXPR = 7,  // Lambda S-expression metadata
    ESHKOL_VALUE_STRING_PTR  = 8,   // Pointer to string
    ESHKOL_VALUE_CHAR        = 9,   // Character (Unicode codepoint)
    ESHKOL_VALUE_VECTOR_PTR  = 10,  // Pointer to Scheme vector
    ESHKOL_VALUE_SYMBOL      = 11,  // Interned symbol
    ESHKOL_VALUE_CLOSURE_PTR = 12,  // Pointer to closure
    ESHKOL_VALUE_BOOL        = 13,  // Boolean
} eshkol_value_type_t;
```

### 2.2 HoTT Type System Architecture

The type system implements 4 universe levels:

```cpp
// File: lib/types/hott_types.cpp:20-28
enum class Universe {
    U0,      // Type₀ - Ground types (Int64, Float64, Boolean, etc.)
    U1,      // Type₁ - Type constructors (List, Vector, Function, etc.)
    U2,      // Type₂ - Propositions (Eq, LessThan, Bounded - erased at runtime)
    UOmega   // Typeω - Universe polymorphism
};

// Type flags for advanced type features:
#define TYPE_FLAG_EXACT   0x01  // Exact number (Integer vs Real)
#define TYPE_FLAG_LINEAR  0x02  // Linear type (must be consumed exactly once)
#define TYPE_FLAG_PROOF   0x04  // Proof type (erased at runtime)
```

### 2.3 Automatic Differentiation System

The AD system supports both forward and reverse mode:

```cpp
// File: inc/eshkol/eshkol.h:196-233
// Forward-mode: Dual numbers (16 bytes, cache-aligned)
typedef struct eshkol_dual_number {
    double value;       // f(x) - the function value
    double derivative;  // f'(x) - the derivative value
} eshkol_dual_number_t;

// Reverse-mode: Computational graph nodes
typedef struct ad_node {
    ad_node_type_t type;     // Operation type (ADD, MUL, SIN, etc.)
    double value;            // Computed value during forward pass
    double gradient;         // Accumulated gradient during backward pass
    struct ad_node* input1;  // First parent node
    struct ad_node* input2;  // Second parent node
    size_t id;               // Unique node ID for topological sorting
} ad_node_t;

// Tape for recording operations (supports 32-level nesting)
typedef struct ad_tape {
    ad_node_t** nodes;       // Array of nodes in evaluation order
    size_t num_nodes;        // Current number of nodes
    size_t capacity;         // Allocated capacity
    ad_node_t** variables;   // Input variable nodes
    size_t num_variables;    // Number of input variables
} ad_tape_t;

// Global tape stack for nested gradients
// File: lib/core/arena_memory.cpp:33-36
static const size_t MAX_TAPE_DEPTH = 32;
ad_tape_t* __ad_tape_stack[MAX_TAPE_DEPTH] = {nullptr};
uint64_t __ad_tape_depth = 0;
```

### 2.4 Closure and Environment Structure

```cpp
// File: inc/eshkol/eshkol.h:255-286
// Closure environment with variadic encoding
typedef struct eshkol_closure_env {
    size_t num_captures;  // Packed: num_captures | (fixed_params << 16) | (is_variadic << 63)
    eshkol_tagged_value_t captures[];  // Flexible array of captured values
} eshkol_closure_env_t;

// Full closure structure
typedef struct eshkol_closure {
    uint64_t func_ptr;        // Pointer to the lambda function
    eshkol_closure_env_t* env;// Captured environment (may be NULL)
    uint64_t sexpr_ptr;       // S-expression for homoiconicity
    uint8_t return_type;      // Return type category
    uint8_t input_arity;      // Number of expected arguments
    uint8_t flags;            // Additional flags
    uint8_t reserved;         // Padding
    uint32_t hott_type_id;    // HoTT TypeId for return type
} eshkol_closure_t;

// Variadic encoding macros
#define CLOSURE_ENV_GET_NUM_CAPTURES(packed) ((packed) & 0xFFFF)
#define CLOSURE_ENV_GET_FIXED_PARAMS(packed) (((packed) >> 16) & 0xFFFF)
#define CLOSURE_ENV_IS_VARIADIC(packed) (((packed) >> 63) & 1)
```

### 2.5 OALR Memory Management

Ownership-Aware Lexical Regions (OALR) operators:

```cpp
// File: inc/eshkol/eshkol.h:504-510
// Memory management operators parsed by the frontend:
ESHKOL_WITH_REGION_OP,  // with-region - lexical region for batch allocation/free
ESHKOL_OWNED_OP,        // owned - linear type for resources
ESHKOL_MOVE_OP,         // move - transfer ownership
ESHKOL_BORROW_OP,       // borrow - temporary read-only access
ESHKOL_SHARED_OP,       // shared - reference-counted allocation
ESHKOL_WEAK_REF_OP,     // weak-ref - weak reference
```

---

## 3. SIMD/Vectorization Implementation

### 3.1 Current State

**CRITICAL GAP**: Eshkol has NO SIMD support. All tensor operations use scalar loops.

### 3.2 Architecture Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SIMD VECTORIZATION LAYER                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     VectorizedOps Module                             │   │
│  │  File: lib/backend/vectorized_ops.cpp                               │   │
│  │                                                                      │   │
│  │  class VectorizedCodegen {                                           │   │
│  │      // Core vectorization interface                                 │   │
│  │      llvm::Value* vectorAdd(llvm::Value* a, llvm::Value* b, size_t n);│  │
│  │      llvm::Value* vectorMul(llvm::Value* a, llvm::Value* b, size_t n);│  │
│  │      llvm::Value* vectorFMA(llvm::Value* a, llvm::Value* b,          │   │
│  │                             llvm::Value* c, size_t n);               │   │
│  │      llvm::Value* vectorReduce(llvm::Value* v, ReduceOp op);         │   │
│  │                                                                      │   │
│  │      // Matrix operations with SIMD                                  │   │
│  │      llvm::Value* matmulTiled(llvm::Value* A, llvm::Value* B,        │   │
│  │                               size_t M, size_t K, size_t N);         │   │
│  │      llvm::Value* dotProduct(llvm::Value* a, llvm::Value* b, size_t n);│ │
│  │  };                                                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                  LLVM Intrinsics Backend                            │   │
│  │                                                                      │   │
│  │  SSE4.2:  <4 x double>, <2 x double>                                │   │
│  │  AVX2:    <8 x float>, <4 x double>                                 │   │
│  │  AVX-512: <16 x float>, <8 x double>                                │   │
│  │  NEON:    <4 x float>, <2 x double>  (ARM)                          │   │
│  │                                                                      │   │
│  │  Intrinsics used:                                                   │   │
│  │  - llvm.vector.reduce.fadd                                          │   │
│  │  - llvm.fma.v4f64                                                   │   │
│  │  - llvm.masked.load / llvm.masked.store                             │   │
│  │  - llvm.experimental.vector.reduce.*                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Implementation Files

#### 3.3.1 Header: `inc/eshkol/backend/vectorized_ops.h`

```cpp
#ifndef ESHKOL_VECTORIZED_OPS_H
#define ESHKOL_VECTORIZED_OPS_H

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Intrinsics.h>
#include <eshkol/backend/codegen_context.h>

namespace eshkol {

// SIMD width detection
enum class SIMDLevel {
    None,      // No SIMD (scalar fallback)
    SSE42,     // 128-bit vectors
    AVX2,      // 256-bit vectors
    AVX512,    // 512-bit vectors
    NEON       // ARM NEON
};

struct SIMDConfig {
    SIMDLevel level;
    unsigned vector_width;     // Number of doubles per vector
    bool supports_fma;         // Fused multiply-add
    bool supports_gather;      // Gather/scatter operations
    bool supports_mask;        // Masked operations

    static SIMDConfig detect();
};

class VectorizedCodegen {
public:
    VectorizedCodegen(CodegenContext& ctx, const SIMDConfig& config);

    // ===== Element-wise operations =====

    // Vector addition: c[i] = a[i] + b[i]
    llvm::Value* vectorAdd(llvm::Value* a_ptr, llvm::Value* b_ptr,
                           llvm::Value* c_ptr, size_t n);

    // Vector multiplication: c[i] = a[i] * b[i]
    llvm::Value* vectorMul(llvm::Value* a_ptr, llvm::Value* b_ptr,
                           llvm::Value* c_ptr, size_t n);

    // Fused multiply-add: d[i] = a[i] * b[i] + c[i]
    llvm::Value* vectorFMA(llvm::Value* a_ptr, llvm::Value* b_ptr,
                           llvm::Value* c_ptr, llvm::Value* d_ptr, size_t n);

    // Vector scale: b[i] = scalar * a[i]
    llvm::Value* vectorScale(llvm::Value* a_ptr, llvm::Value* scalar,
                             llvm::Value* b_ptr, size_t n);

    // ===== Reduction operations =====

    llvm::Value* vectorSum(llvm::Value* ptr, size_t n);
    llvm::Value* vectorMax(llvm::Value* ptr, size_t n);
    llvm::Value* vectorMin(llvm::Value* ptr, size_t n);
    llvm::Value* vectorL2Norm(llvm::Value* ptr, size_t n);

    // ===== Dot product and matrix operations =====

    llvm::Value* dotProduct(llvm::Value* a_ptr, llvm::Value* b_ptr, size_t n);

    // Tiled matrix multiply with SIMD inner kernel
    // C[M,N] = A[M,K] @ B[K,N]
    llvm::Value* matmulTiled(llvm::Value* A, llvm::Value* B, llvm::Value* C,
                             size_t M, size_t K, size_t N,
                             size_t tile_m = 64, size_t tile_n = 64, size_t tile_k = 256);

    // ===== Activation functions (vectorized) =====

    llvm::Value* vectorReLU(llvm::Value* ptr, size_t n);
    llvm::Value* vectorSigmoid(llvm::Value* ptr, size_t n);
    llvm::Value* vectorTanh(llvm::Value* ptr, size_t n);
    llvm::Value* vectorExp(llvm::Value* ptr, size_t n);
    llvm::Value* vectorLog(llvm::Value* ptr, size_t n);

    // ===== Softmax (requires two passes) =====

    llvm::Value* vectorSoftmax(llvm::Value* input_ptr, llvm::Value* output_ptr, size_t n);

private:
    CodegenContext& ctx_;
    SIMDConfig config_;

    // Generate loop with vectorized body
    void generateVectorLoop(llvm::Value* n,
                           std::function<void(llvm::Value* idx, bool is_vector)> body);

    // Load/store vector values
    llvm::Value* loadVector(llvm::Value* ptr, llvm::Value* idx);
    void storeVector(llvm::Value* vec, llvm::Value* ptr, llvm::Value* idx);

    // Masked operations for tail handling
    llvm::Value* loadVectorMasked(llvm::Value* ptr, llvm::Value* idx,
                                  llvm::Value* mask, llvm::Value* passthru);
    void storeVectorMasked(llvm::Value* vec, llvm::Value* ptr,
                           llvm::Value* idx, llvm::Value* mask);

    // Get vector type for current SIMD level
    llvm::VectorType* getVectorType();

    // Horizontal reduction
    llvm::Value* horizontalAdd(llvm::Value* vec);
    llvm::Value* horizontalMax(llvm::Value* vec);
};

} // namespace eshkol

#endif // ESHKOL_VECTORIZED_OPS_H
```

#### 3.3.2 Implementation: `lib/backend/vectorized_ops.cpp`

```cpp
#include <eshkol/backend/vectorized_ops.h>
#include <llvm/IR/IntrinsicsX86.h>
#include <llvm/TargetParser/Host.h>

namespace eshkol {

SIMDConfig SIMDConfig::detect() {
    SIMDConfig config;

    // Get host CPU features
    llvm::StringMap<bool> features;
    llvm::sys::getHostCPUFeatures(features);

    // Check for AVX-512
    if (features["avx512f"] && features["avx512dq"]) {
        config.level = SIMDLevel::AVX512;
        config.vector_width = 8;  // 8 doubles
        config.supports_fma = features["fma"];
        config.supports_gather = true;
        config.supports_mask = true;
    }
    // Check for AVX2
    else if (features["avx2"]) {
        config.level = SIMDLevel::AVX2;
        config.vector_width = 4;  // 4 doubles
        config.supports_fma = features["fma"];
        config.supports_gather = true;
        config.supports_mask = false;  // Limited mask support
    }
    // Check for SSE4.2
    else if (features["sse4.2"]) {
        config.level = SIMDLevel::SSE42;
        config.vector_width = 2;  // 2 doubles
        config.supports_fma = false;
        config.supports_gather = false;
        config.supports_mask = false;
    }
    // Fallback
    else {
        config.level = SIMDLevel::None;
        config.vector_width = 1;
        config.supports_fma = false;
        config.supports_gather = false;
        config.supports_mask = false;
    }

    return config;
}

VectorizedCodegen::VectorizedCodegen(CodegenContext& ctx, const SIMDConfig& config)
    : ctx_(ctx), config_(config) {}

llvm::VectorType* VectorizedCodegen::getVectorType() {
    return llvm::FixedVectorType::get(ctx_.doubleType(), config_.vector_width);
}

llvm::Value* VectorizedCodegen::loadVector(llvm::Value* ptr, llvm::Value* idx) {
    auto vec_type = getVectorType();
    auto elem_ptr = ctx_.builder().CreateGEP(ctx_.doubleType(), ptr, idx);
    auto vec_ptr = ctx_.builder().CreateBitCast(elem_ptr,
        llvm::PointerType::getUnqual(vec_type));

    // Use aligned load for better performance
    auto load = ctx_.builder().CreateAlignedLoad(vec_type, vec_ptr,
        llvm::MaybeAlign(config_.vector_width * 8));
    return load;
}

void VectorizedCodegen::storeVector(llvm::Value* vec, llvm::Value* ptr, llvm::Value* idx) {
    auto vec_type = getVectorType();
    auto elem_ptr = ctx_.builder().CreateGEP(ctx_.doubleType(), ptr, idx);
    auto vec_ptr = ctx_.builder().CreateBitCast(elem_ptr,
        llvm::PointerType::getUnqual(vec_type));

    ctx_.builder().CreateAlignedStore(vec, vec_ptr,
        llvm::MaybeAlign(config_.vector_width * 8));
}

llvm::Value* VectorizedCodegen::horizontalAdd(llvm::Value* vec) {
    // Use LLVM's vector reduction intrinsic
    auto reduce_id = llvm::Intrinsic::vector_reduce_fadd;
    auto zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);

    llvm::Function* reduce_fn = llvm::Intrinsic::getDeclaration(
        &ctx_.module(), reduce_id, {vec->getType()});

    return ctx_.builder().CreateCall(reduce_fn, {zero, vec});
}

llvm::Value* VectorizedCodegen::dotProduct(llvm::Value* a_ptr, llvm::Value* b_ptr, size_t n) {
    auto& builder = ctx_.builder();

    size_t vec_width = config_.vector_width;
    size_t num_vectors = n / vec_width;
    size_t remainder = n % vec_width;

    // Accumulator for vectorized sum
    auto vec_type = getVectorType();
    llvm::Value* acc = llvm::ConstantVector::getSplat(
        llvm::ElementCount::getFixed(vec_width),
        llvm::ConstantFP::get(ctx_.doubleType(), 0.0));

    // Vectorized loop
    if (num_vectors > 0) {
        auto loop_header = llvm::BasicBlock::Create(ctx_.context(), "dot_vec_loop",
            builder.GetInsertBlock()->getParent());
        auto loop_body = llvm::BasicBlock::Create(ctx_.context(), "dot_vec_body");
        auto loop_end = llvm::BasicBlock::Create(ctx_.context(), "dot_vec_end");

        builder.CreateBr(loop_header);
        builder.SetInsertPoint(loop_header);

        auto phi_i = builder.CreatePHI(ctx_.int64Type(), 2, "i");
        auto phi_acc = builder.CreatePHI(vec_type, 2, "acc");
        phi_i->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 0),
            loop_header->getSinglePredecessor());
        phi_acc->addIncoming(acc, loop_header->getSinglePredecessor());

        // Load vectors
        auto idx = builder.CreateMul(phi_i,
            llvm::ConstantInt::get(ctx_.int64Type(), vec_width));
        auto vec_a = loadVector(a_ptr, idx);
        auto vec_b = loadVector(b_ptr, idx);

        // FMA or mul+add
        llvm::Value* prod;
        if (config_.supports_fma) {
            auto fma_id = llvm::Intrinsic::fma;
            llvm::Function* fma_fn = llvm::Intrinsic::getDeclaration(
                &ctx_.module(), fma_id, {vec_type});
            prod = builder.CreateCall(fma_fn, {vec_a, vec_b, phi_acc});
        } else {
            prod = builder.CreateFMul(vec_a, vec_b);
            prod = builder.CreateFAdd(phi_acc, prod);
        }

        // Increment and check
        auto next_i = builder.CreateAdd(phi_i,
            llvm::ConstantInt::get(ctx_.int64Type(), 1));
        phi_i->addIncoming(next_i, builder.GetInsertBlock());
        phi_acc->addIncoming(prod, builder.GetInsertBlock());

        auto cmp = builder.CreateICmpULT(next_i,
            llvm::ConstantInt::get(ctx_.int64Type(), num_vectors));
        builder.CreateCondBr(cmp, loop_header, loop_end);

        // After loop, horizontal sum
        builder.SetInsertPoint(loop_end);
        acc = phi_acc;
    }

    // Horizontal reduction
    llvm::Value* sum = horizontalAdd(acc);

    // Handle remainder with scalar loop
    if (remainder > 0) {
        for (size_t i = 0; i < remainder; i++) {
            auto idx = llvm::ConstantInt::get(ctx_.int64Type(), num_vectors * vec_width + i);
            auto a_val = builder.CreateLoad(ctx_.doubleType(),
                builder.CreateGEP(ctx_.doubleType(), a_ptr, idx));
            auto b_val = builder.CreateLoad(ctx_.doubleType(),
                builder.CreateGEP(ctx_.doubleType(), b_ptr, idx));
            auto prod = builder.CreateFMul(a_val, b_val);
            sum = builder.CreateFAdd(sum, prod);
        }
    }

    return sum;
}

llvm::Value* VectorizedCodegen::matmulTiled(
    llvm::Value* A, llvm::Value* B, llvm::Value* C,
    size_t M, size_t K, size_t N,
    size_t tile_m, size_t tile_n, size_t tile_k) {

    auto& builder = ctx_.builder();

    // C[i,j] = sum_k A[i,k] * B[k,j]
    // Tiled implementation for cache efficiency

    // Generate nested loops: tile_i, tile_j, tile_k, i, j, k
    // Inner kernel uses SIMD for the j dimension

    // [Implementation continues with tiled loops and SIMD inner kernel]
    // This is a complex function - full implementation would be ~200 lines

    // For brevity, showing the SIMD inner kernel concept:
    /*
    for tile_i in range(0, M, tile_m):
        for tile_j in range(0, N, tile_n):
            for tile_k in range(0, K, tile_k):
                for i in range(tile_i, min(tile_i + tile_m, M)):
                    for j in range(tile_j, min(tile_j + tile_n, N), vec_width):
                        # SIMD: process vec_width elements of j at once
                        c_vec = load_vector(C, i, j)
                        for k in range(tile_k, min(tile_k + tile_k, K)):
                            a_val = broadcast(A[i, k])  # Scalar broadcast
                            b_vec = load_vector(B, k, j)
                            c_vec = fma(a_val, b_vec, c_vec)
                        store_vector(c_vec, C, i, j)
    */

    return C;
}

llvm::Value* VectorizedCodegen::vectorReLU(llvm::Value* ptr, size_t n) {
    auto& builder = ctx_.builder();
    auto vec_type = getVectorType();

    size_t vec_width = config_.vector_width;
    size_t num_vectors = n / vec_width;

    // Zero vector for comparison and max
    auto zero_vec = llvm::ConstantVector::getSplat(
        llvm::ElementCount::getFixed(vec_width),
        llvm::ConstantFP::get(ctx_.doubleType(), 0.0));

    // Vectorized loop
    for (size_t v = 0; v < num_vectors; v++) {
        auto idx = llvm::ConstantInt::get(ctx_.int64Type(), v * vec_width);
        auto vec = loadVector(ptr, idx);

        // ReLU = max(0, x)
        auto max_id = llvm::Intrinsic::maxnum;
        llvm::Function* max_fn = llvm::Intrinsic::getDeclaration(
            &ctx_.module(), max_id, {vec_type});
        auto result = builder.CreateCall(max_fn, {zero_vec, vec});

        storeVector(result, ptr, idx);
    }

    // Handle remainder
    size_t remainder = n % vec_width;
    for (size_t i = 0; i < remainder; i++) {
        auto idx = llvm::ConstantInt::get(ctx_.int64Type(), num_vectors * vec_width + i);
        auto elem_ptr = builder.CreateGEP(ctx_.doubleType(), ptr, idx);
        auto val = builder.CreateLoad(ctx_.doubleType(), elem_ptr);
        auto zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
        auto result = builder.CreateSelect(
            builder.CreateFCmpOGT(val, zero), val, zero);
        builder.CreateStore(result, elem_ptr);
    }

    return ptr;
}

} // namespace eshkol
```

### 3.4 Integration Points

Modify `lib/backend/tensor_codegen.cpp` to use vectorized operations:

```cpp
// In TensorCodegen::generateMatMul()
// Replace scalar loop with:

VectorizedCodegen vec_gen(ctx_, SIMDConfig::detect());
return vec_gen.matmulTiled(A, B, C, M, K, N);
```

### 3.5 Eshkol Language Extensions

Add SIMD hints to the language:

```scheme
;; Explicit vectorization hint
(define (fast-dot a b)
  (: a (Vector Float64))
  (: b (Vector Float64))
  (simd-hint :level avx2)
  (tensor-dot a b))

;; Auto-vectorization pragma
(pragma vectorize)
(define (element-wise-op vec)
  (map (lambda (x) (* x x)) vec))
```

---

## 4. GPU/CUDA Backend Implementation

### 4.1 Current State

**CRITICAL GAP**: Eshkol has NO GPU support whatsoever.

### 4.2 Architecture Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GPU BACKEND ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     GPU Memory Manager                               │   │
│  │  File: lib/gpu/gpu_memory.cpp                                       │   │
│  │                                                                      │   │
│  │  class GPUMemoryManager {                                            │   │
│  │      DeviceBuffer allocate(size_t bytes, MemoryType type);          │   │
│  │      void free(DeviceBuffer buf);                                   │   │
│  │      void copyH2D(void* host, DeviceBuffer device, size_t bytes);   │   │
│  │      void copyD2H(DeviceBuffer device, void* host, size_t bytes);   │   │
│  │      void copyD2D(DeviceBuffer src, DeviceBuffer dst, size_t bytes);│   │
│  │                                                                      │   │
│  │      // Unified memory support (CUDA managed memory)                │   │
│  │      void* allocateUnified(size_t bytes);                           │   │
│  │      void prefetchToDevice(void* ptr, size_t bytes, int device);    │   │
│  │  };                                                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     GPU Kernel Codegen                              │   │
│  │  File: lib/gpu/gpu_codegen.cpp                                      │   │
│  │                                                                      │   │
│  │  Uses LLVM NVPTX backend to generate GPU kernels:                   │   │
│  │  - Element-wise operations (map, reduce)                            │   │
│  │  - Matrix operations (matmul via tensor cores)                      │   │
│  │  - Convolutions (using cuDNN via extern calls)                      │   │
│  │  - Autodiff backward kernels                                        │   │
│  │                                                                      │   │
│  │  Target triple: nvptx64-nvidia-cuda                                 │   │
│  │  Data layout: e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32...    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     GPU Execution Runtime                           │   │
│  │  File: lib/gpu/gpu_runtime.cpp                                      │   │
│  │                                                                      │   │
│  │  - CUDA driver API integration                                      │   │
│  │  - Kernel launch configuration                                      │   │
│  │  - Stream management for async execution                            │   │
│  │  - cuBLAS/cuDNN library calls                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Implementation Files

#### 4.3.1 Header: `inc/eshkol/gpu/gpu_backend.h`

```cpp
#ifndef ESHKOL_GPU_BACKEND_H
#define ESHKOL_GPU_BACKEND_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <llvm/IR/Module.h>

namespace eshkol {
namespace gpu {

// Forward declarations
class GPUDevice;
class GPUStream;
class GPUEvent;

// Memory types
enum class MemoryType {
    Device,      // GPU-only memory
    Pinned,      // Page-locked host memory (faster H2D transfers)
    Unified,     // CUDA managed memory (automatic migration)
    Mapped       // Zero-copy (access from both CPU and GPU)
};

// Device buffer handle
struct DeviceBuffer {
    void* ptr;
    size_t size;
    MemoryType type;
    int device_id;

    bool isValid() const { return ptr != nullptr; }
};

// Tensor descriptor for GPU operations
struct GPUTensorDesc {
    DeviceBuffer buffer;
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    enum class DType { Float16, Float32, Float64, Int32, Int64 } dtype;

    size_t numElements() const;
    size_t byteSize() const;
};

// GPU device information
struct GPUDeviceInfo {
    int id;
    std::string name;
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int warp_size;
    int max_threads_per_block;
    int max_shared_memory_per_block;
    bool supports_tensor_cores;
    bool supports_unified_memory;
};

// GPU Memory Manager
class GPUMemoryManager {
public:
    GPUMemoryManager(int device_id = 0);
    ~GPUMemoryManager();

    // Allocation
    DeviceBuffer allocate(size_t bytes, MemoryType type = MemoryType::Device);
    void free(DeviceBuffer& buf);

    // Transfers
    void copyHostToDevice(const void* host, DeviceBuffer device, size_t bytes);
    void copyDeviceToHost(DeviceBuffer device, void* host, size_t bytes);
    void copyDeviceToDevice(DeviceBuffer src, DeviceBuffer dst, size_t bytes);

    // Async transfers with stream
    void copyHostToDeviceAsync(const void* host, DeviceBuffer device,
                               size_t bytes, GPUStream& stream);
    void copyDeviceToHostAsync(DeviceBuffer device, void* host,
                               size_t bytes, GPUStream& stream);

    // Unified memory operations
    void* allocateUnified(size_t bytes);
    void freeUnified(void* ptr);
    void prefetchToDevice(void* ptr, size_t bytes, int device_id);
    void prefetchToHost(void* ptr, size_t bytes);

    // Memory pool (pre-allocate to avoid allocation overhead)
    void initializePool(size_t pool_size);
    DeviceBuffer allocateFromPool(size_t bytes);
    void returnToPool(DeviceBuffer& buf);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// GPU Stream (for async execution)
class GPUStream {
public:
    GPUStream(int device_id = 0, int priority = 0);
    ~GPUStream();

    void* nativeHandle() const;
    void synchronize();
    bool isComplete() const;

    // Record event in stream
    GPUEvent recordEvent();

    // Wait for event from another stream
    void waitEvent(const GPUEvent& event);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// GPU Event (for synchronization)
class GPUEvent {
public:
    GPUEvent();
    ~GPUEvent();

    void* nativeHandle() const;
    bool isComplete() const;

    // Elapsed time between events (in milliseconds)
    static float elapsedTime(const GPUEvent& start, const GPUEvent& end);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// GPU Kernel Codegen
class GPUKernelCodegen {
public:
    GPUKernelCodegen(llvm::LLVMContext& ctx);

    // Generate NVPTX module for kernel
    std::unique_ptr<llvm::Module> createKernelModule(const std::string& name);

    // Generate element-wise kernel
    // kernel<<<blocks, threads>>>(input, output, n) { output[i] = f(input[i]); }
    llvm::Function* generateElementWiseKernel(
        llvm::Module& module,
        const std::string& name,
        std::function<llvm::Value*(llvm::IRBuilder<>&, llvm::Value*)> element_op);

    // Generate reduction kernel (sum, max, min, etc.)
    llvm::Function* generateReductionKernel(
        llvm::Module& module,
        const std::string& name,
        enum class ReductionOp { Sum, Max, Min, Prod });

    // Generate matrix multiplication kernel (uses tensor cores if available)
    llvm::Function* generateMatMulKernel(
        llvm::Module& module,
        bool use_tensor_cores = true);

    // Compile module to PTX string
    std::string compileToPTX(llvm::Module& module);

private:
    llvm::LLVMContext& ctx_;

    // Add NVPTX intrinsics
    void declareNVPTXIntrinsics(llvm::Module& module);

    // Get thread/block indices
    llvm::Value* getThreadIdx(llvm::IRBuilder<>& builder, char dim);  // x, y, z
    llvm::Value* getBlockIdx(llvm::IRBuilder<>& builder, char dim);
    llvm::Value* getBlockDim(llvm::IRBuilder<>& builder, char dim);
    llvm::Value* getGridDim(llvm::IRBuilder<>& builder, char dim);

    // Synchronization
    void syncThreads(llvm::IRBuilder<>& builder);

    // Shared memory allocation
    llvm::GlobalVariable* createSharedMemory(llvm::Module& module,
                                             llvm::Type* type, size_t size);
};

// GPU Runtime (kernel execution)
class GPURuntime {
public:
    GPURuntime(int device_id = 0);
    ~GPURuntime();

    // Load compiled PTX module
    void loadModule(const std::string& ptx, const std::string& name);

    // Get function from loaded module
    void* getFunction(const std::string& module_name, const std::string& func_name);

    // Launch kernel
    void launchKernel(void* kernel,
                      dim3 grid_dim, dim3 block_dim,
                      void** args, size_t shared_mem = 0,
                      GPUStream* stream = nullptr);

    // High-level tensor operations (uses cuBLAS/cuDNN when beneficial)
    void matmul(const GPUTensorDesc& A, const GPUTensorDesc& B,
                GPUTensorDesc& C, bool transpose_A = false,
                bool transpose_B = false, GPUStream* stream = nullptr);

    void conv2d(const GPUTensorDesc& input, const GPUTensorDesc& filter,
                GPUTensorDesc& output, int stride = 1, int padding = 0,
                GPUStream* stream = nullptr);

    void softmax(const GPUTensorDesc& input, GPUTensorDesc& output,
                 int axis = -1, GPUStream* stream = nullptr);

    void batchNorm(const GPUTensorDesc& input, const GPUTensorDesc& scale,
                   const GPUTensorDesc& bias, GPUTensorDesc& output,
                   float epsilon = 1e-5f, GPUStream* stream = nullptr);

    // Device info
    GPUDeviceInfo getDeviceInfo() const;
    static std::vector<GPUDeviceInfo> listDevices();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Helper: dim3 for grid/block dimensions
struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1)
        : x(x_), y(y_), z(z_) {}
};

} // namespace gpu
} // namespace eshkol

#endif // ESHKOL_GPU_BACKEND_H
```

### 4.4 Eshkol Language Extensions for GPU

```scheme
;; GPU tensor allocation
(define gpu-a (gpu-tensor '(1024 1024) :dtype float32))
(define gpu-b (gpu-tensor '(1024 1024) :dtype float32))

;; Copy data to GPU
(gpu-copy! cpu-data gpu-a)

;; GPU matrix multiplication
(define gpu-c (gpu-matmul gpu-a gpu-b))

;; GPU autodiff
(define (gpu-loss params data)
  (gpu-context
    (let* ((pred (gpu-forward params data))
           (diff (gpu-sub pred target)))
      (gpu-mean (gpu-square diff)))))

;; Compute gradient on GPU
(define grad (gpu-gradient gpu-loss params))
```

---

## 5. XLA/MLIR Integration

### 5.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      XLA/MLIR INTEGRATION LAYER                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Eshkol AST                                                                 │
│      │                                                                      │
│      ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    HLO Graph Builder                                │   │
│  │  File: lib/xla/hlo_builder.cpp                                      │   │
│  │                                                                      │   │
│  │  - Converts Eshkol tensor ops to XLA HLO operations                 │   │
│  │  - Preserves autodiff computation graph structure                   │   │
│  │  - Handles control flow (scan, while, cond)                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    StableHLO Dialect                                │   │
│  │  File: lib/xla/stablehlo_codegen.cpp                               │   │
│  │                                                                      │   │
│  │  Operations:                                                        │   │
│  │  - stablehlo.add, stablehlo.multiply, stablehlo.dot_general        │   │
│  │  - stablehlo.reduce, stablehlo.broadcast_in_dim                    │   │
│  │  - stablehlo.custom_call (for cuDNN, oneDNN)                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    MLIR Pass Pipeline                               │   │
│  │                                                                      │   │
│  │  1. Shape inference and broadcasting                                │   │
│  │  2. Algebraic simplifications                                       │   │
│  │  3. Operation fusion (element-wise, reduce-window)                  │   │
│  │  4. Buffer allocation and memory planning                           │   │
│  │  5. Target lowering (GPU/CPU specific)                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Target Backends                                  │   │
│  │                                                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │ LLVM/CPU     │  │ NVPTX/GPU    │  │ AMDGPU       │              │   │
│  │  │ (AVX-512,    │  │ (cuBLAS,     │  │ (ROCm,       │              │   │
│  │  │  NEON)       │  │  Tensor Cores)│  │  ROCBlas)    │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 StableHLO Operation Mapping

| Eshkol Operation | StableHLO Operation | Notes |
|------------------|---------------------|-------|
| `tensor-add` | `stablehlo.add` | Broadcasting handled automatically |
| `tensor-mul` | `stablehlo.multiply` | Element-wise |
| `tensor-matmul` | `stablehlo.dot_general` | Batched matmul support |
| `gradient` | Custom backward graph | Uses stablehlo ops |
| `reduce` | `stablehlo.reduce` | With init value and reducer |
| `softmax` | Lowered to exp/reduce/div | Custom fusion pass |
| `conv2d` | `stablehlo.convolution` | Maps to cuDNN |

---

## 6. Parallel Execution Framework

### 6.1 Current State

**CRITICAL GAP**: Only synchronization is a REPL mutex. No parallel execution.

```cpp
// File: lib/backend/llvm_codegen.cpp:100
std::mutex g_repl_mutex;  // Thread safety for REPL symbol access
```

### 6.2 Architecture Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PARALLEL EXECUTION FRAMEWORK                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Work Stealing Scheduler                        │   │
│  │  File: lib/parallel/scheduler.cpp                                   │   │
│  │                                                                      │   │
│  │  - Lock-free work stealing deques (Chase-Lev algorithm)            │   │
│  │  - Per-thread task queues with stealing from random victims        │   │
│  │  - Automatic load balancing across cores                            │   │
│  │  - Nested parallelism support                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│  ┌───────────────────────────┼───────────────────────────────────────────┐ │
│  │                           ▼                                           │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │ │
│  │  │  Worker 0   │  │  Worker 1   │  │  Worker 2   │  │  Worker N   │  │ │
│  │  │  ┌───────┐  │  │  ┌───────┐  │  │  ┌───────┐  │  │  ┌───────┐  │  │ │
│  │  │  │ Deque │  │  │  │ Deque │  │  │  │ Deque │  │  │  │ Deque │  │  │ │
│  │  │  └───────┘  │  │  └───────┘  │  │  └───────┘  │  │  └───────┘  │  │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │ │
│  │                     Work Stealing (lock-free)                         │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Parallel Primitives                            │   │
│  │                                                                      │   │
│  │  parallel-for:     Parallel loop with work stealing                 │   │
│  │  parallel-map:     Apply function to list elements in parallel      │   │
│  │  parallel-reduce:  Tree-based parallel reduction                    │   │
│  │  future/promise:   Async task with result retrieval                 │   │
│  │  atomic-ops:       Compare-and-swap, fetch-add, etc.                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Implementation: `inc/eshkol/parallel/scheduler.h`

```cpp
#ifndef ESHKOL_PARALLEL_SCHEDULER_H
#define ESHKOL_PARALLEL_SCHEDULER_H

#include <atomic>
#include <thread>
#include <vector>
#include <functional>
#include <optional>
#include <random>

namespace eshkol {
namespace parallel {

// Lock-free Chase-Lev work-stealing deque
template<typename T>
class WorkStealingDeque {
public:
    WorkStealingDeque(size_t log_size = 10);
    ~WorkStealingDeque();

    // Owner thread operations
    void push(T item);
    std::optional<T> pop();

    // Thief operations (other threads)
    std::optional<T> steal();

    size_t size() const;
    bool empty() const;

private:
    struct CircularArray {
        std::atomic<T*> items;
        size_t capacity;
        size_t mask;

        CircularArray(size_t cap);
        T get(size_t idx);
        void put(size_t idx, T val);
        CircularArray* grow(size_t bottom, size_t top);
    };

    std::atomic<size_t> top_;
    std::atomic<size_t> bottom_;
    std::atomic<CircularArray*> array_;
};

// Task for parallel execution
using Task = std::function<void()>;

// Worker thread
class Worker {
public:
    Worker(size_t id, class Scheduler& scheduler);

    void run();
    void stop();

    void submit(Task task);
    std::optional<Task> stealFrom();

    size_t id() const { return id_; }

private:
    size_t id_;
    Scheduler& scheduler_;
    WorkStealingDeque<Task> deque_;
    std::thread thread_;
    std::atomic<bool> running_;

    std::optional<Task> findWork();
};

// Main scheduler
class Scheduler {
public:
    // Get global scheduler (thread-safe initialization)
    static Scheduler& instance();

    // Initialize with N workers (defaults to hardware_concurrency)
    void initialize(size_t num_workers = 0);
    void shutdown();

    // Submit task
    void submit(Task task);

    // Parallel for: execute body(i) for i in [start, end)
    template<typename Body>
    void parallelFor(size_t start, size_t end, Body body, size_t grain_size = 1);

    // Parallel reduce
    template<typename T, typename Reducer>
    T parallelReduce(size_t start, size_t end, T identity,
                     std::function<T(size_t)> map_fn, Reducer reduce_fn);

    // Get worker for stealing
    Worker* getRandomVictim(size_t exclude_id);

    size_t numWorkers() const { return workers_.size(); }

private:
    Scheduler() = default;

    std::vector<std::unique_ptr<Worker>> workers_;
    std::atomic<bool> initialized_{false};
    std::mt19937 rng_;
    std::mutex rng_mutex_;
};

// Future/Promise for async results
template<typename T>
class Future {
public:
    T get();  // Blocking wait for result
    bool isReady() const;

private:
    friend class Promise<T>;
    std::shared_ptr<std::atomic<T*>> result_;
    std::shared_ptr<std::atomic<bool>> ready_;
};

template<typename T>
class Promise {
public:
    Promise();
    Future<T> getFuture();
    void set(T value);

private:
    std::shared_ptr<std::atomic<T*>> result_;
    std::shared_ptr<std::atomic<bool>> ready_;
};

// Parallel for implementation
template<typename Body>
void Scheduler::parallelFor(size_t start, size_t end, Body body, size_t grain_size) {
    if (end <= start) return;

    size_t n = end - start;
    if (n <= grain_size || workers_.empty()) {
        // Sequential fallback
        for (size_t i = start; i < end; i++) {
            body(i);
        }
        return;
    }

    // Split into tasks
    std::atomic<size_t> next_chunk{start};
    std::atomic<size_t> remaining_tasks{0};
    std::condition_variable cv;
    std::mutex cv_mutex;

    size_t num_chunks = (n + grain_size - 1) / grain_size;
    remaining_tasks = num_chunks;

    for (size_t c = 0; c < num_chunks; c++) {
        submit([&, c]() {
            size_t chunk_start = start + c * grain_size;
            size_t chunk_end = std::min(chunk_start + grain_size, end);

            for (size_t i = chunk_start; i < chunk_end; i++) {
                body(i);
            }

            if (--remaining_tasks == 0) {
                std::lock_guard<std::mutex> lock(cv_mutex);
                cv.notify_one();
            }
        });
    }

    // Wait for completion
    std::unique_lock<std::mutex> lock(cv_mutex);
    cv.wait(lock, [&]() { return remaining_tasks == 0; });
}

} // namespace parallel
} // namespace eshkol

#endif // ESHKOL_PARALLEL_SCHEDULER_H
```

### 6.4 Eshkol Language Parallel Constructs

```scheme
;; Parallel map (automatic work distribution)
(define squares (parallel-map square (range 1000000)))

;; Parallel for with explicit body
(parallel-for (i 0 1000000)
  (vector-set! result i (* (vector-ref data i) 2)))

;; Parallel reduce (tree-based)
(define sum (parallel-reduce + 0 data))

;; Future/Promise for async computation
(define f (future (expensive-computation)))
;; ... do other work ...
(define result (force f))  ; Wait for result

;; Parallel tensor operations (auto-parallelized)
(define c (parallel-tensor-matmul a b))
```

---

## 7. Hygienic Macro System

### 7.1 Current State

**GAP**: No macro system. Only 46+ special forms hardcoded in the parser.

### 7.2 Implementation Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      HYGIENIC MACRO SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Syntax Objects                                   │   │
│  │  File: lib/macro/syntax.cpp                                         │   │
│  │                                                                      │   │
│  │  struct SyntaxObject {                                              │   │
│  │      Datum datum;           // The actual value (symbol, list, etc.)│   │
│  │      ScopeSet scopes;       // Set of scopes for hygiene           │   │
│  │      SourceLocation loc;    // Source location for error messages  │   │
│  │  };                                                                 │   │
│  │                                                                      │   │
│  │  // Scope-based hygiene (Racket/R6RS style)                        │   │
│  │  // Each macro invocation creates a new scope                       │   │
│  │  // Identifiers are resolved by scope intersection                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Pattern Matching                                 │   │
│  │  File: lib/macro/pattern.cpp                                        │   │
│  │                                                                      │   │
│  │  Pattern syntax:                                                    │   │
│  │  - (pattern ...)         List pattern                               │   │
│  │  - pattern ...           Ellipsis (zero or more)                    │   │
│  │  - _                     Wildcard (matches anything)                │   │
│  │  - literal               Literal match                              │   │
│  │  - var                   Binding variable                           │   │
│  │                                                                      │   │
│  │  Template syntax:                                                   │   │
│  │  - (template ...)        List template                              │   │
│  │  - template ...          Expand ellipsis                            │   │
│  │  - var                   Substitute bound variable                  │   │
│  │  - (syntax-error msg)    Error during expansion                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Expander                                         │   │
│  │  File: lib/macro/expander.cpp                                       │   │
│  │                                                                      │   │
│  │  class MacroExpander {                                              │   │
│  │      SyntaxObject expand(SyntaxObject stx);                         │   │
│  │      void defineSyntax(Symbol name, Transformer* xform);            │   │
│  │  };                                                                 │   │
│  │                                                                      │   │
│  │  Expansion phases:                                                  │   │
│  │  1. Parse input as syntax objects                                   │   │
│  │  2. Resolve identifiers in scope                                    │   │
│  │  3. Match against transformer patterns                              │   │
│  │  4. Substitute and expand template                                  │   │
│  │  5. Mark with new scope for hygiene                                 │   │
│  │  6. Recurse until fully expanded                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Eshkol Macro Syntax

```scheme
;; syntax-rules (R5RS compatible)
(define-syntax when
  (syntax-rules ()
    [(when test body ...)
     (if test (begin body ...) #f)]))

;; syntax-case (R6RS compatible, more powerful)
(define-syntax define-struct
  (syntax-case ()
    [(define-struct name (field ...))
     (with-syntax ([(getter ...)
                    (map (lambda (f)
                           (string->symbol
                             (string-append (symbol->string #'name)
                                           "-"
                                           (symbol->string f))))
                         #'(field ...))])
       #'(begin
           (define (make-name field ...) (vector 'name field ...))
           (define (getter obj) (vector-ref obj 1)) ...))]))

;; Quasiquote for macro templates
(define-syntax my-let
  (syntax-rules ()
    [(my-let ((var val) ...) body ...)
     `((lambda (,@vars) ,@body) ,@vals)]))

;; Local macros
(let-syntax ([swap (syntax-rules ()
                     [(swap a b)
                      (let ([t a])
                        (set! a b)
                        (set! b t))])])
  (swap x y))
```

---

## 8. Neural Network Primitives

### 8.1 Current State

Eshkol has some basic math functions but lacks NN-specific operations.

### 8.2 Required Primitives

```scheme
;; ===== LAYER OPERATIONS =====

;; Linear/Dense layer: y = Wx + b
(define (linear weights bias input)
  (tensor-add (tensor-matmul weights input) bias))

;; Softmax: exp(x_i) / sum(exp(x_j))
(define (softmax x)
  (let* ((max-x (tensor-max x))          ; Numerical stability
         (exp-x (tensor-exp (tensor-sub x max-x)))
         (sum-exp (tensor-sum exp-x)))
    (tensor-div exp-x sum-exp)))

;; Layer normalization
(define (layer-norm x gamma beta epsilon)
  (let* ((mean (tensor-mean x :axis -1 :keepdims #t))
         (var (tensor-var x :axis -1 :keepdims #t))
         (normalized (tensor-div (tensor-sub x mean)
                                 (tensor-sqrt (tensor-add var epsilon)))))
    (tensor-add (tensor-mul gamma normalized) beta)))

;; Batch normalization (training mode)
(define (batch-norm x gamma beta running-mean running-var momentum epsilon)
  (let* ((batch-mean (tensor-mean x :axis 0))
         (batch-var (tensor-var x :axis 0))
         ;; Update running stats
         (_ (tensor-set! running-mean
              (tensor-add (tensor-mul running-mean (- 1 momentum))
                          (tensor-mul batch-mean momentum))))
         (_ (tensor-set! running-var
              (tensor-add (tensor-mul running-var (- 1 momentum))
                          (tensor-mul batch-var momentum))))
         ;; Normalize
         (normalized (tensor-div (tensor-sub x batch-mean)
                                 (tensor-sqrt (tensor-add batch-var epsilon)))))
    (tensor-add (tensor-mul gamma normalized) beta)))

;; ===== ATTENTION MECHANISM =====

;; Scaled dot-product attention
(define (attention query key value mask)
  (let* ((d-k (tensor-shape key -1))
         (scores (tensor-matmul query (tensor-transpose key)))
         (scaled (tensor-div scores (sqrt d-k)))
         (masked (if mask
                     (tensor-add scaled (tensor-mul mask -1e9))
                     scaled))
         (weights (softmax masked)))
    (tensor-matmul weights value)))

;; Multi-head attention
(define (multi-head-attention query key value num-heads w-q w-k w-v w-o mask)
  (let* ((batch-size (tensor-shape query 0))
         (seq-len (tensor-shape query 1))
         (d-model (tensor-shape query 2))
         (d-k (/ d-model num-heads))
         ;; Project Q, K, V
         (Q (reshape (tensor-matmul query w-q)
                     (list batch-size seq-len num-heads d-k)))
         (K (reshape (tensor-matmul key w-k)
                     (list batch-size seq-len num-heads d-k)))
         (V (reshape (tensor-matmul value w-v)
                     (list batch-size seq-len num-heads d-k)))
         ;; Transpose for attention: (batch, heads, seq, d_k)
         (Q (tensor-transpose Q '(0 2 1 3)))
         (K (tensor-transpose K '(0 2 1 3)))
         (V (tensor-transpose V '(0 2 1 3)))
         ;; Apply attention
         (attended (attention Q K V mask))
         ;; Concatenate heads
         (concat (reshape (tensor-transpose attended '(0 2 1 3))
                         (list batch-size seq-len d-model))))
    (tensor-matmul concat w-o)))

;; ===== CONVOLUTION =====

;; 2D Convolution
(define (conv2d input filter stride padding)
  ;; input: (batch, height, width, channels-in)
  ;; filter: (filter-height, filter-width, channels-in, channels-out)
  (native-conv2d input filter stride padding))

;; Max pooling 2D
(define (max-pool2d input pool-size stride)
  (native-max-pool2d input pool-size stride))

;; ===== LOSS FUNCTIONS =====

;; Cross-entropy loss
(define (cross-entropy-loss logits targets)
  (let* ((log-probs (tensor-log-softmax logits))
         (one-hot (tensor-one-hot targets (tensor-shape logits -1))))
    (tensor-neg (tensor-mean (tensor-sum (tensor-mul one-hot log-probs) :axis -1)))))

;; Mean squared error
(define (mse-loss pred target)
  (tensor-mean (tensor-square (tensor-sub pred target))))

;; ===== DROPOUT =====

(define (dropout x rate training)
  (if training
      (let* ((mask (tensor-bernoulli (tensor-shape x) (- 1 rate)))
             (scaled (tensor-div x (- 1 rate))))
        (tensor-mul scaled mask))
      x))
```

### 8.3 Implementation File: `lib/backend/nn_ops_codegen.cpp`

These operations require both forward pass code generation and automatic backward pass generation through the existing autodiff system.

---

## 9. Distributed Computing Framework

### 9.1 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DISTRIBUTED COMPUTING FRAMEWORK                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Communication Layer (MPI-like)                   │   │
│  │  File: lib/distributed/comm.cpp                                     │   │
│  │                                                                      │   │
│  │  // Point-to-point                                                  │   │
│  │  send(tensor, dest_rank, tag)                                       │   │
│  │  recv(tensor, src_rank, tag)                                        │   │
│  │                                                                      │   │
│  │  // Collective operations                                           │   │
│  │  broadcast(tensor, root)                                            │   │
│  │  reduce(tensor, op, root)  // op: sum, max, min, etc.              │   │
│  │  all_reduce(tensor, op)                                             │   │
│  │  scatter(send_tensor, recv_tensor, root)                            │   │
│  │  gather(send_tensor, recv_tensor, root)                             │   │
│  │  all_gather(send_tensor, recv_tensor)                               │   │
│  │  all_to_all(send_tensor, recv_tensor)                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Data Parallelism                                 │   │
│  │  File: lib/distributed/data_parallel.cpp                            │   │
│  │                                                                      │   │
│  │  // Distributed data parallel training                              │   │
│  │  class DistributedDataParallel {                                    │   │
│  │      // Wraps model for data-parallel training                      │   │
│  │      forward(input);                                                │   │
│  │      backward();                                                    │   │
│  │      sync_gradients();  // All-reduce gradients                     │   │
│  │  };                                                                 │   │
│  │                                                                      │   │
│  │  // Ring-allreduce for bandwidth-efficient gradient sync            │   │
│  │  ring_allreduce(gradients);                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Model Parallelism                                │   │
│  │  File: lib/distributed/model_parallel.cpp                           │   │
│  │                                                                      │   │
│  │  // Pipeline parallelism                                            │   │
│  │  class PipelineParallel {                                           │   │
│  │      // Split model into stages across devices                      │   │
│  │      forward_stage(stage_id, input);                                │   │
│  │      backward_stage(stage_id, grad);                                │   │
│  │      micro_batch_schedule();  // GPipe/PipeDream scheduling        │   │
│  │  };                                                                 │   │
│  │                                                                      │   │
│  │  // Tensor parallelism (for large layers)                           │   │
│  │  class TensorParallel {                                             │   │
│  │      // Split tensors across devices                                │   │
│  │      parallel_linear(input, weight_shard);                          │   │
│  │      all_reduce_output();                                           │   │
│  │  };                                                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Eshkol Distributed Syntax

```scheme
;; Initialize distributed runtime
(distributed-init :backend "nccl")  ; or "gloo", "mpi"

;; Get rank and world size
(define rank (get-rank))
(define world-size (get-world-size))

;; Data parallel training
(define ddp-model (distributed-data-parallel model))

(define (train-step batch)
  (let* ((loss (forward ddp-model batch))
         (grads (backward loss)))
    (sync-gradients ddp-model)
    (optimizer-step optimizer grads)))

;; Distributed data loading
(define distributed-loader
  (make-distributed-dataloader dataset
    :batch-size 64
    :num-workers 4
    :shuffle #t))

;; Collective operations
(broadcast tensor 0)              ; Broadcast from rank 0
(all-reduce gradients :op 'sum)   ; Sum gradients across all ranks
```

---

## 10. Serialization & Checkpointing

### 10.1 Format Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ESHKOL CHECKPOINT FORMAT (.eskpt)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Header (64 bytes):                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  magic: "ESKPT001"              (8 bytes)                           │   │
│  │  version: uint32                 (4 bytes)                           │   │
│  │  flags: uint32                   (4 bytes) - compression, etc.       │   │
│  │  num_tensors: uint64             (8 bytes)                           │   │
│  │  total_size: uint64              (8 bytes)                           │   │
│  │  metadata_offset: uint64         (8 bytes)                           │   │
│  │  checksum: uint64                (8 bytes) - CRC64 of data           │   │
│  │  reserved: 16 bytes                                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Tensor Directory:                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  For each tensor:                                                   │   │
│  │    name_length: uint16                                              │   │
│  │    name: char[name_length]                                          │   │
│  │    dtype: uint8                                                     │   │
│  │    ndim: uint8                                                      │   │
│  │    shape: uint64[ndim]                                              │   │
│  │    data_offset: uint64                                              │   │
│  │    data_size: uint64                                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Tensor Data (aligned to 64 bytes):                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Raw tensor data (optionally compressed with zstd)                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Metadata (JSON or MessagePack):                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  {                                                                  │   │
│  │    "optimizer_state": { ... },                                      │   │
│  │    "training_step": 10000,                                          │   │
│  │    "model_config": { ... },                                         │   │
│  │    "custom_data": { ... }                                           │   │
│  │  }                                                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Eshkol Serialization API

```scheme
;; Save model checkpoint
(save-checkpoint model "model.eskpt"
  :optimizer optimizer
  :step training-step
  :metadata '((lr . 0.001) (batch-size . 64)))

;; Load model checkpoint
(define checkpoint (load-checkpoint "model.eskpt"))
(define model (checkpoint-model checkpoint))
(define optimizer (checkpoint-optimizer checkpoint))
(define step (checkpoint-get checkpoint 'step))

;; Save only specific tensors
(save-tensors '((weights . w) (biases . b)) "params.eskpt")

;; Memory-mapped loading for large models
(define model (load-checkpoint "large_model.eskpt" :mmap #t))
```

---

## 11. Advanced Memory Optimization

### 11.1 Current Arena System Extension

```cpp
// File: lib/core/arena_memory.cpp - Extensions needed

// ===== MEMORY POOL WITH SIZE CLASSES =====
// For small allocations, use fixed-size pools to reduce fragmentation

class SizeClassPool {
public:
    static constexpr size_t SIZE_CLASSES[] = {
        16, 32, 64, 128, 256, 512, 1024, 2048, 4096
    };
    static constexpr size_t NUM_SIZE_CLASSES = 9;

    SizeClassPool(arena_t* backing_arena);

    void* allocate(size_t size);
    void free(void* ptr, size_t size);

private:
    // Free lists for each size class
    struct FreeNode { FreeNode* next; };
    FreeNode* free_lists_[NUM_SIZE_CLASSES];
    arena_t* arena_;

    size_t sizeClassIndex(size_t size);
};

// ===== TENSOR MEMORY REUSE =====
// Track tensor allocations for reuse in gradient computation

class TensorMemoryPool {
public:
    struct TensorBuffer {
        void* data;
        size_t size;
        std::vector<size_t> shape;
        bool in_use;
    };

    TensorBuffer* allocate(const std::vector<size_t>& shape, size_t elem_size);
    void release(TensorBuffer* buf);

    // Find existing buffer that fits the shape
    TensorBuffer* findReusable(const std::vector<size_t>& shape, size_t elem_size);

private:
    std::vector<TensorBuffer> buffers_;
    std::mutex mutex_;
};

// ===== GRADIENT CHECKPOINTING =====
// Trade compute for memory in autodiff

class CheckpointedGradient {
public:
    // Mark computation for recomputation during backward pass
    void checkpoint(const std::string& name, std::function<void()> recompute);

    // During backward, recompute instead of storing
    void recomputeIfNeeded(const std::string& name);

private:
    std::unordered_map<std::string, std::function<void()>> checkpoints_;
};
```

### 11.2 Eshkol Memory Hints

```scheme
;; Explicit memory region for batch processing
(with-region "batch-processing"
  :size-hint (* 1024 1024 100)  ; 100MB
  (process-batch data))

;; Gradient checkpointing for memory-constrained training
(define (forward-with-checkpointing model input)
  (checkpoint-segment "layer1"
    (let ((h1 (layer1 model input)))
      (checkpoint-segment "layer2"
        (let ((h2 (layer2 model h1)))
          (layer3 model h2))))))

;; Memory-efficient tensor operations (in-place when possible)
(tensor-add! a b)  ; a += b (in-place)
(tensor-mul! a scalar)  ; a *= scalar (in-place)
```

---

## 12. Profiling & Debugging Infrastructure

### 12.1 Profiling Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROFILING INFRASTRUCTURE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Instrumentation Layer                            │   │
│  │  File: lib/profiler/instrumenter.cpp                                │   │
│  │                                                                      │   │
│  │  // Compile-time instrumentation (inserted during codegen)          │   │
│  │  - Function entry/exit hooks                                        │   │
│  │  - Memory allocation tracking                                       │   │
│  │  - Tensor operation timing                                          │   │
│  │  - AD tape size monitoring                                          │   │
│  │                                                                      │   │
│  │  // Runtime cost model                                              │   │
│  │  - FLOP counting                                                    │   │
│  │  - Memory bandwidth estimation                                      │   │
│  │  - Cache miss prediction                                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Profiler Runtime                                 │   │
│  │  File: lib/profiler/runtime.cpp                                     │   │
│  │                                                                      │   │
│  │  class Profiler {                                                   │   │
│  │      void startRegion(const char* name);                            │   │
│  │      void endRegion();                                              │   │
│  │      void recordAllocation(size_t bytes, const char* type);         │   │
│  │      void recordTensorOp(const char* op, size_t flops);            │   │
│  │                                                                      │   │
│  │      ProfileReport generateReport();                                │   │
│  │      void exportChrome(const char* path); // Chrome trace format   │   │
│  │      void exportTensorBoard(const char* path);                      │   │
│  │  };                                                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Visualization                                    │   │
│  │                                                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │   │
│  │  │ Flame Graph  │  │ Memory       │  │ AD Computation Graph     │  │   │
│  │  │ (time)       │  │ Timeline     │  │ (visualize tape)         │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 12.2 Eshkol Profiling API

```scheme
;; Profile a computation
(with-profile "training"
  (train-epoch model data))

;; Get profiling report
(define report (profile-report))
(display (report-summary report))
(export-chrome-trace report "trace.json")

;; Memory profiling
(with-memory-profile
  (let ((model (create-large-model)))
    (display (memory-report))))

;; AD tape visualization
(define (f x) (* x x x))
(with-ad-trace
  (gradient f 2.0))
(export-ad-graph "computation_graph.dot")
```

---

## 13. Extended Tensor Operations

### 13.1 Missing Operations List

| Category | Operations | Priority |
|----------|------------|----------|
| Indexing | gather, scatter, index_select, masked_select | HIGH |
| Shape | reshape, transpose, squeeze, unsqueeze, expand, repeat | HIGH |
| Reduction | argmax, argmin, topk, sort, unique | HIGH |
| Comparison | where, searchsorted, bucketize | MEDIUM |
| Linear Algebra | svd, qr, cholesky, eig, lstsq, triangular_solve | HIGH |
| FFT | fft, ifft, fft2, ifft2, rfft | MEDIUM |
| Random | uniform, normal, multinomial, randperm | HIGH |
| Sparse | sparse_mm, sparse_add, to_sparse, to_dense | LOW |

### 13.2 Implementation Priority

```scheme
;; ===== HIGH PRIORITY =====

;; Gather: out[i] = input[index[i]]
(tensor-gather input dim index)

;; Scatter: self[index[i]] = src[i]
(tensor-scatter! self dim index src)

;; Where: out[i] = cond[i] ? a[i] : b[i]
(tensor-where condition a b)

;; Linear algebra
(tensor-svd A)        ; -> (U, S, V)
(tensor-qr A)         ; -> (Q, R)
(tensor-cholesky A)   ; -> L where A = LL^T
(tensor-solve A b)    ; -> x where Ax = b

;; ===== MEDIUM PRIORITY =====

;; FFT operations
(tensor-fft x)        ; 1D FFT
(tensor-fft2 x)       ; 2D FFT
(tensor-ifft x)       ; Inverse FFT

;; Advanced indexing
(tensor-topk x k)     ; -> (values, indices)
(tensor-sort x)       ; -> (sorted, indices)
(tensor-unique x)     ; -> unique values
```

---

## 14. Optimizer Implementations

### 14.1 Required Optimizers

```scheme
;; ===== SGD with Momentum =====
(define (sgd-step params grads lr momentum velocity)
  (for-each
    (lambda (p g v)
      (let ((new-v (tensor-add (tensor-mul v momentum)
                               (tensor-mul g lr))))
        (tensor-set! v new-v)
        (tensor-sub! p new-v)))
    params grads velocity))

;; ===== Adam =====
(define (adam-step params grads lr beta1 beta2 epsilon m v t)
  (for-each
    (lambda (p g m-i v-i)
      ;; Update biased first moment estimate
      (tensor-set! m-i
        (tensor-add (tensor-mul m-i beta1)
                    (tensor-mul g (- 1 beta1))))
      ;; Update biased second raw moment estimate
      (tensor-set! v-i
        (tensor-add (tensor-mul v-i beta2)
                    (tensor-mul (tensor-square g) (- 1 beta2))))
      ;; Bias correction
      (let* ((m-hat (tensor-div m-i (- 1 (expt beta1 t))))
             (v-hat (tensor-div v-i (- 1 (expt beta2 t))))
             (update (tensor-div m-hat
                       (tensor-add (tensor-sqrt v-hat) epsilon))))
        (tensor-sub! p (tensor-mul update lr))))
    params grads m v))

;; ===== AdamW (Adam with decoupled weight decay) =====
(define (adamw-step params grads lr beta1 beta2 epsilon weight-decay m v t)
  ;; Weight decay
  (for-each (lambda (p) (tensor-mul! p (- 1 (* lr weight-decay)))) params)
  ;; Adam update
  (adam-step params grads lr beta1 beta2 epsilon m v t))

;; ===== LAMB (Layer-wise Adaptive Moments for Batch training) =====
(define (lamb-step params grads lr beta1 beta2 epsilon weight-decay m v t)
  (for-each
    (lambda (p g m-i v-i)
      ;; Adam-like moment updates
      (tensor-set! m-i (tensor-add (tensor-mul m-i beta1)
                                   (tensor-mul g (- 1 beta1))))
      (tensor-set! v-i (tensor-add (tensor-mul v-i beta2)
                                   (tensor-mul (tensor-square g) (- 1 beta2))))
      (let* ((m-hat (tensor-div m-i (- 1 (expt beta1 t))))
             (v-hat (tensor-div v-i (- 1 (expt beta2 t))))
             (adam-update (tensor-div m-hat
                            (tensor-add (tensor-sqrt v-hat) epsilon)))
             ;; Layer-wise learning rate
             (weight-norm (tensor-norm p))
             (update-norm (tensor-norm adam-update))
             (trust-ratio (if (and (> weight-norm 0) (> update-norm 0))
                             (/ weight-norm update-norm)
                             1.0))
             (lr-layer (* lr trust-ratio)))
        ;; Apply update
        (tensor-sub! p (tensor-mul adam-update lr-layer))
        ;; Weight decay
        (tensor-mul! p (- 1 (* lr weight-decay)))))
    params grads m v))
```

---

## 15. Implementation Roadmap

### 15.1 Phase-by-Phase Plan

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                       IMPLEMENTATION ROADMAP                                   ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  PHASE 1: Performance Foundation (Weeks 1-4)                                  ║
║  ─────────────────────────────────────────                                    ║
║  [x] SIMD vectorization for tensor ops                                        ║
║      - File: lib/backend/vectorized_ops.cpp                                   ║
║      - Target: 4-8x speedup on element-wise ops                               ║
║                                                                               ║
║  [x] Parallel execution framework                                             ║
║      - File: lib/parallel/scheduler.cpp                                       ║
║      - Work-stealing scheduler with lock-free deques                          ║
║                                                                               ║
║  [x] Memory optimization                                                      ║
║      - Size-class pools for small allocations                                 ║
║      - Tensor buffer reuse                                                    ║
║                                                                               ║
║  PHASE 2: Hardware Acceleration (Weeks 5-10)                                  ║
║  ────────────────────────────────────────────                                 ║
║  [ ] GPU/CUDA backend                                                         ║
║      - File: lib/gpu/gpu_backend.cpp                                          ║
║      - NVPTX codegen via LLVM                                                 ║
║      - cuBLAS/cuDNN integration                                               ║
║                                                                               ║
║  [ ] XLA/StableHLO integration                                               ║
║      - File: lib/xla/stablehlo_codegen.cpp                                   ║
║      - Operation fusion passes                                                ║
║      - Multi-backend compilation                                              ║
║                                                                               ║
║  PHASE 3: ML Primitives (Weeks 11-14)                                        ║
║  ────────────────────────────────────                                        ║
║  [ ] Neural network operations                                                ║
║      - Softmax, layer-norm, attention                                        ║
║      - Conv2d, pooling                                                        ║
║      - Automatic differentiation for all ops                                  ║
║                                                                               ║
║  [ ] Optimizer implementations                                                ║
║      - SGD, Adam, AdamW, LAMB                                                 ║
║                                                                               ║
║  PHASE 4: Language Features (Weeks 15-18)                                    ║
║  ─────────────────────────────────────────                                    ║
║  [ ] Hygienic macro system                                                    ║
║      - syntax-rules (R5RS)                                                    ║
║      - syntax-case (R6RS)                                                     ║
║      - Local macro definitions                                                ║
║                                                                               ║
║  [ ] Extended tensor operations                                               ║
║      - gather, scatter, topk, sort                                            ║
║      - SVD, QR, Cholesky                                                      ║
║                                                                               ║
║  PHASE 5: Distributed & Production (Weeks 19-24)                             ║
║  ─────────────────────────────────────────────                                ║
║  [ ] Distributed computing                                                    ║
║      - Data parallel training                                                 ║
║      - Ring all-reduce                                                        ║
║                                                                               ║
║  [ ] Serialization & checkpointing                                           ║
║      - .eskpt format                                                          ║
║      - Memory-mapped loading                                                  ║
║                                                                               ║
║  [ ] Profiling & debugging                                                   ║
║      - Instrumentation                                                        ║
║      - Chrome trace export                                                    ║
║      - AD graph visualization                                                 ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

### 15.2 File Structure After Implementation

```
eshkol/
├── inc/eshkol/
│   ├── backend/
│   │   ├── vectorized_ops.h      [NEW]
│   │   ├── nn_ops_codegen.h      [NEW]
│   │   └── ...existing...
│   ├── gpu/
│   │   ├── gpu_backend.h         [NEW]
│   │   ├── gpu_memory.h          [NEW]
│   │   └── gpu_runtime.h         [NEW]
│   ├── xla/
│   │   ├── stablehlo_codegen.h   [NEW]
│   │   └── hlo_builder.h         [NEW]
│   ├── parallel/
│   │   ├── scheduler.h           [NEW]
│   │   └── work_stealing.h       [NEW]
│   ├── macro/
│   │   ├── syntax.h              [NEW]
│   │   ├── pattern.h             [NEW]
│   │   └── expander.h            [NEW]
│   ├── distributed/
│   │   ├── comm.h                [NEW]
│   │   └── data_parallel.h       [NEW]
│   ├── profiler/
│   │   ├── instrumenter.h        [NEW]
│   │   └── runtime.h             [NEW]
│   └── ...existing...
├── lib/
│   ├── backend/
│   │   ├── vectorized_ops.cpp    [NEW - 1,500 lines]
│   │   ├── nn_ops_codegen.cpp    [NEW - 2,000 lines]
│   │   └── ...existing...
│   ├── gpu/
│   │   ├── gpu_backend.cpp       [NEW - 3,000 lines]
│   │   ├── gpu_memory.cpp        [NEW - 800 lines]
│   │   └── gpu_runtime.cpp       [NEW - 1,200 lines]
│   ├── xla/
│   │   ├── stablehlo_codegen.cpp [NEW - 2,500 lines]
│   │   └── hlo_builder.cpp       [NEW - 1,500 lines]
│   ├── parallel/
│   │   └── scheduler.cpp         [NEW - 1,000 lines]
│   ├── macro/
│   │   ├── syntax.cpp            [NEW - 800 lines]
│   │   ├── pattern.cpp           [NEW - 1,200 lines]
│   │   └── expander.cpp          [NEW - 1,500 lines]
│   ├── distributed/
│   │   ├── comm.cpp              [NEW - 1,500 lines]
│   │   └── data_parallel.cpp     [NEW - 1,000 lines]
│   ├── profiler/
│   │   ├── instrumenter.cpp      [NEW - 600 lines]
│   │   └── runtime.cpp           [NEW - 800 lines]
│   └── ...existing...
└── tests/
    ├── simd/                     [NEW]
    ├── gpu/                      [NEW]
    ├── parallel/                 [NEW]
    ├── macro/                    [NEW]
    ├── distributed/              [NEW]
    └── ...existing...
```

### 15.3 Estimated Total New Code

| Component | Lines of Code | Files |
|-----------|---------------|-------|
| SIMD/Vectorization | ~2,000 | 2 |
| GPU Backend | ~5,000 | 3 |
| XLA Integration | ~4,000 | 2 |
| Parallel Framework | ~1,500 | 2 |
| Macro System | ~3,500 | 3 |
| NN Primitives | ~2,000 | 1 |
| Distributed | ~2,500 | 2 |
| Profiling | ~1,400 | 2 |
| Serialization | ~1,000 | 1 |
| Extended Tensors | ~2,000 | 1 |
| **TOTAL** | **~25,000** | **19** |

---

## Appendix A: Key Integration Points

### A.1 Integrating SIMD into TensorCodegen

```cpp
// File: lib/backend/tensor_codegen.cpp
// Modify generateMatMul to use vectorized implementation

llvm::Value* TensorCodegen::generateMatMul(llvm::Value* A, llvm::Value* B,
                                           size_t M, size_t K, size_t N) {
    // Check if SIMD is available
    auto simd_config = SIMDConfig::detect();
    if (simd_config.level != SIMDLevel::None && M >= 64 && N >= 64) {
        // Use vectorized implementation
        VectorizedCodegen vec_gen(ctx_, simd_config);
        llvm::Value* C = allocateTensor({M, N});
        return vec_gen.matmulTiled(A, B, C, M, K, N);
    }

    // Fallback to scalar implementation
    return generateMatMulScalar(A, B, M, K, N);
}
```

### A.2 Integrating GPU into AutodiffCodegen

```cpp
// File: lib/backend/autodiff_codegen.cpp
// Add GPU execution path for gradient computation

llvm::Value* AutodiffCodegen::computeGradientGPU(
    llvm::Function* func, llvm::Value* point) {

    // Check if GPU is available and tensor is large enough
    if (gpu::GPURuntime::deviceCount() > 0 && tensorSize(point) > 10000) {
        gpu::GPURuntime runtime(0);

        // Transfer to GPU
        auto gpu_point = runtime.transferToDevice(point);

        // Execute gradient on GPU
        auto gpu_grad = runtime.executeGradient(func, gpu_point);

        // Transfer back
        return runtime.transferToHost(gpu_grad);
    }

    // CPU fallback
    return computeGradientCPU(func, point);
}
```

---

## Appendix B: Testing Strategy

### B.1 SIMD Testing

```scheme
;; tests/simd/vectorized_ops_test.esk

;; Test vector addition
(define a (make-vector 1000 1.0))
(define b (make-vector 1000 2.0))
(define c (vector-add a b))
(assert (= (vector-ref c 0) 3.0))
(assert (= (vector-ref c 999) 3.0))

;; Test dot product
(define dot (vector-dot a b))
(assert (= dot 2000.0))

;; Test matrix multiplication
(define A (make-tensor '(100 100) 1.0))
(define B (make-tensor '(100 100) 1.0))
(define C (tensor-matmul A B))
(assert (= (tensor-ref C 0 0) 100.0))
```

### B.2 GPU Testing

```scheme
;; tests/gpu/cuda_backend_test.esk

(when (gpu-available?)
  ;; Test GPU tensor creation
  (define gpu-a (gpu-tensor '(1024 1024) :dtype float32))
  (assert (= (gpu-tensor-device gpu-a) 0))

  ;; Test GPU operations
  (define gpu-b (gpu-tensor '(1024 1024) :dtype float32))
  (gpu-fill! gpu-a 1.0)
  (gpu-fill! gpu-b 2.0)

  (define gpu-c (gpu-add gpu-a gpu-b))
  (define cpu-c (gpu-to-cpu gpu-c))
  (assert (= (tensor-ref cpu-c 0 0) 3.0)))
```

---

**END OF DOCUMENT**

This document provides a complete technical blueprint for extending Eshkol to become a comprehensive platform for systems-level intelligent computing. Each section includes architectural designs, implementation details, code examples, and integration points.
