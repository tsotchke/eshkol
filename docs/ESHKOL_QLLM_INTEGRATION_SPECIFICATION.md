# Eshkol-qLLM Integration Specification

## Exact Requirements for Using Eshkol as qLLM Training Engine

**Date:** December 7, 2025
**Status:** Implementation Specification
**Based on:** Direct code analysis of both codebases

---

## 1. Executive Summary

This document specifies **exactly** what must be added to Eshkol to enable it as the training engine for the semi-classical qLLM. The requirements are organized by component with specific file modifications, new types, and implementation details.

---

## 2. Current State Analysis

### 2.1 What Eshkol Already Has

#### Core Infrastructure

| Component | Location | Status |
|-----------|----------|--------|
| **JIT Compilation** | `lib/repl/repl_jit.cpp:59-120` | ✅ Complete - LLVM ORC LLJIT |
| **AOT Compilation** | `lib/backend/llvm_codegen.cpp` | ✅ Complete - TargetMachine |
| Forward-mode AD (dual numbers) | `lib/backend/autodiff_codegen.cpp:33-201` | ✅ Complete |
| Reverse-mode AD (tape-based) | `lib/backend/autodiff_codegen.cpp:1131-1250` | ✅ Complete (scalars) |
| Backpropagation | `lib/backend/autodiff_codegen.cpp:1141-1250` | ✅ Complete (scalars) |
| Chain rules: ADD, SUB, MUL, DIV | `lib/backend/autodiff_codegen.cpp:1282-1410` | ✅ Complete |
| Chain rules: SIN, COS | `lib/backend/autodiff_codegen.cpp:1412-1448` | ✅ Complete |
| Dual math: sin, cos, exp, log, sqrt, pow, tanh | `lib/backend/autodiff_codegen.cpp:240-500` | ✅ Complete |
| Nested gradient support | `lib/backend/autodiff_codegen.cpp:1454-1550` | ✅ Complete |
| AD node type enum | `inc/eshkol/eshkol.h` | ✅ Defined |
| Tape structure | `inc/eshkol/eshkol.h` | ✅ Defined |
| HoTT type system | `lib/types/hott_types.cpp` | ✅ Complete |
| Arena memory | `lib/core/arena_memory.cpp` | ✅ Complete |

#### Compilation Pipeline (Key Architectural Insight)

Eshkol's compilation pipeline is already production-ready:

```cpp
// REPL JIT Compilation (lib/repl/repl_jit.cpp:74-88)
auto jit_or_err = LLJITBuilder()
    .setNumCompileThreads(1)
    .create();
// All REPL evaluations compile to native code via LLVM ORC

// Symbol Persistence (lib/repl/repl_jit.cpp:262-283)
uint64_t ReplJITContext::lookupSymbol(const std::string& name) {
    // First check local symbol table
    auto it = symbol_table_.find(name);
    if (it != symbol_table_.end()) return it->second;
    // Then JIT lookup
    auto symbol = jit_->lookup(name);
    // ...
}

// Shared AD Context (lib/repl/repl_jit.cpp:196-211)
// Global AD tape pointer shared across JIT modules
symbols[ES.intern("__current_ad_tape")] = {
    orc::ExecutorAddr::fromPtr((void*)&::__current_ad_tape),
    JITSymbolFlags::Exported
};
```

This means Eshkol already has **zero interpreter overhead** - all code executes as native machine code.

#### Deep Architecture Analysis

**Tagged Value System (`inc/eshkol/eshkol.h:80-90`)**

Eshkol uses a 16-byte tagged value representation that preserves type information at runtime:

```c
typedef struct eshkol_tagged_value {
    uint8_t type;        // Value type (eshkol_value_type_t)
    uint8_t flags;       // Exactness and other flags
    uint16_t reserved;   // Reserved for future use (alignment)
    union {
        int64_t int_val;
        double double_val;
        uint64_t ptr_val;
        uint64_t raw_val;
    } data;
} eshkol_tagged_value_t;  // 16 bytes total
```

Type tags (`inc/eshkol/eshkol.h:43-60`):
| Tag | Value | Description |
|-----|-------|-------------|
| NULL | 0 | Empty/null value |
| INT64 | 1 | 64-bit signed integer |
| DOUBLE | 2 | Double-precision float |
| CONS_PTR | 3 | List cons cell pointer |
| DUAL_NUMBER | 4 | Forward-mode AD dual |
| AD_NODE_PTR | 5 | Reverse-mode AD node |
| TENSOR_PTR | 6 | Tensor structure |
| LAMBDA_SEXPR | 7 | Lambda S-expr (homoiconicity) |
| STRING_PTR | 8 | String pointer |
| CHAR | 9 | Unicode character |
| VECTOR_PTR | 10 | Scheme vector |
| SYMBOL | 11 | Interned symbol |
| CLOSURE_PTR | 12 | Closure with captures |
| BOOL | 13 | Boolean value |

**AD Node Structure (`inc/eshkol/eshkol.h:216-223`)**

The current AD node structure for reverse-mode differentiation:

```c
typedef struct ad_node {
    ad_node_type_t type;     // Operation type (ADD=2, MUL=4, SIN=6, etc.)
    double value;            // Forward pass computed value
    double gradient;         // Accumulated backward pass gradient
    struct ad_node* input1;  // First parent (null for constants/variables)
    struct ad_node* input2;  // Second parent (null for unary ops)
    size_t id;               // Unique node ID for topological sort
} ad_node_t;
```

AD Node Types (`inc/eshkol/eshkol.h:199-212`):
- `AD_NODE_CONSTANT` (0), `AD_NODE_VARIABLE` (1)
- `AD_NODE_ADD` (2), `AD_NODE_SUB` (3), `AD_NODE_MUL` (4), `AD_NODE_DIV` (5)
- `AD_NODE_SIN` (6), `AD_NODE_COS` (7), `AD_NODE_EXP` (8), `AD_NODE_LOG` (9)
- `AD_NODE_POW` (10), `AD_NODE_NEG` (11)

**AD Tape Structure (`inc/eshkol/eshkol.h:227-233`)**

```c
typedef struct ad_tape {
    ad_node_t** nodes;       // Array of nodes in evaluation order
    size_t num_nodes;        // Current number of nodes
    size_t capacity;         // Allocated capacity
    ad_node_t** variables;   // Input variable nodes
    size_t num_variables;    // Number of input variables
} ad_tape_t;
```

**Global AD State (`lib/core/arena_memory.cpp:24-47`)**

The AD system uses global state shared across JIT-compiled modules:

```cpp
// Global tape pointer for AD operations (shared across JIT modules)
ad_tape_t* __current_ad_tape = nullptr;

// Global AD mode flag (shared so lambdas see AD mode set by outer scope)
bool __ad_mode_active = false;

// NESTED GRADIENT: Tape stack for arbitrary-depth nested gradients
static const size_t MAX_TAPE_DEPTH = 32;
ad_tape_t* __ad_tape_stack[MAX_TAPE_DEPTH] = {nullptr};
uint64_t __ad_tape_depth = 0;

// DOUBLE BACKWARD: Storage for outer AD node in nested gradient
void* __outer_ad_node_storage = nullptr;
void* __inner_var_node_ptr = nullptr;
uint64_t __gradient_x_degree = 0;  // Tracks derivative order
```

**CodegenContext (`inc/eshkol/backend/codegen_context.h:51-338`)**

The `CodegenContext` class centralizes all code generation state:

```cpp
class CodegenContext {
    // LLVM infrastructure
    llvm::LLVMContext& context_;
    llvm::Module& module_;
    llvm::IRBuilder<>& builder_;

    // Type system integration
    TypeSystem& types_;
    FunctionCache& funcs_;
    MemoryCodegen& memory_;
    hott::TypeEnvironment hott_types_;  // HoTT type system

    // Symbol management
    std::vector<std::unordered_map<std::string, llvm::Value*>> scope_stack_;
    std::unordered_map<std::string, llvm::Value*> global_symbols_;
    std::unordered_map<std::string, llvm::Function*> function_table_;

    // AD state globals
    llvm::GlobalVariable* ad_mode_active_;
    llvm::GlobalVariable* current_ad_tape_;
    llvm::GlobalVariable* ad_tape_stack_;
    llvm::GlobalVariable* ad_tape_depth_;

    // Double backward support
    llvm::GlobalVariable* inner_var_node_ptr_;
    llvm::GlobalVariable* gradient_x_degree_;
};
```

**Closure Structure with Homoiconicity (`inc/eshkol/eshkol.h:277-286`)**

```c
typedef struct eshkol_closure {
    uint64_t func_ptr;           // Pointer to lambda function
    eshkol_closure_env_t* env;   // Captured environment (may be NULL)
    uint64_t sexpr_ptr;          // S-expression for homoiconicity!
    uint8_t return_type;         // Return type category
    uint8_t input_arity;         // Expected argument count
    uint8_t flags;               // Additional flags
    uint8_t reserved;            // Padding
    uint32_t hott_type_id;       // HoTT TypeId for return type
} eshkol_closure_t;
```

This structure enables **full homoiconicity**: `(display (list my-lambda))` shows the original source code because every closure carries its S-expression representation.

#### Existing N-Dimensional Tensor Support

**IMPORTANT**: Eshkol already has comprehensive N-dimensional tensor support implemented in `lib/backend/tensor_codegen.cpp` (1200+ lines):

```cpp
// TensorCodegen handles N-dimensional arrays with:
// - Arbitrary dimensions (1D vectors, 2D matrices, 3D+ tensors)
// - Element-wise arithmetic with broadcasting
// - Shape operations and transformations

class TensorCodegen {
    // Creation
    llvm::Value* createTensor(const eshkol_ast_t* ast);           // #[1 2 3] or #[[1 2] [3 4]]
    llvm::Value* zeros(const eshkol_operations_t* op);            // (zeros dim1 dim2 ...)
    llvm::Value* ones(const eshkol_operations_t* op);             // (ones dim1 dim2 ...)
    llvm::Value* eye(const eshkol_operations_t* op);              // (eye n) - identity matrix
    llvm::Value* arange(const eshkol_operations_t* op);           // (arange start stop step)
    llvm::Value* linspace(const eshkol_operations_t* op);         // (linspace start stop num)

    // Access
    llvm::Value* tensorGet(const eshkol_operations_t* op);        // (tensor-get tensor idx...)
    llvm::Value* tensorSet(const eshkol_operations_t* op);        // (tensor-set tensor val idx...)

    // Arithmetic (element-wise)
    llvm::Value* tensorArithmetic(..., "add"/"sub"/"mul"/"div");  // tensor-add, tensor-sub, etc.
    llvm::Value* tensorDot(const eshkol_operations_t* op);        // 1D dot product, 2D matmul (partial)

    // Transformations
    llvm::Value* tensorApply(const eshkol_operations_t* op);      // (tensor-apply tensor func)
    llvm::Value* tensorReduceAll(const eshkol_operations_t* op);  // (tensor-reduce-all tensor func init)
    llvm::Value* tensorReduceWithDim(...);                         // (tensor-reduce tensor func init dim)
    llvm::Value* transpose(const eshkol_operations_t* op);        // (transpose tensor)
    llvm::Value* reshape(const eshkol_operations_t* op);          // (reshape tensor new-dims...)

    // Statistics
    llvm::Value* tensorSum(const eshkol_operations_t* op);        // (tensor-sum tensor)
    llvm::Value* tensorMean(const eshkol_operations_t* op);       // (tensor-mean tensor)
    llvm::Value* tensorShape(const eshkol_operations_t* op);      // (tensor-shape tensor)
};
```

**Internal Tensor Structure** (`inc/eshkol/eshkol.h` - inferred from codegen):
```c
struct eshkol_tensor {
    size_t* dims;           // Dimension sizes array
    size_t num_dims;        // Number of dimensions (rank)
    double* elements;       // Contiguous data array
    size_t total_elements;  // Product of all dimensions
};
```

#### Pure Eshkol Mathematical Capabilities (ALL TESTS PASS)

Eshkol can implement arbitrarily complex mathematical algorithms in pure Eshkol that **automatically work with the AD system**. The comprehensive stress tests (`tests/features/*.esk`) demonstrate:

**Numerical Methods** (autodiff-compatible):
```scheme
;;; Newton-Raphson with autodiff (ultimate_math_stress.esk)
(define (newton-solve f x0 iterations)
  (iterate x0 iterations
    (lambda (x) (- x (/ (f x) (derivative f x))))))

;;; Runge-Kutta 4th order ODE solver
(define (rk4-solve f x0 y0 xf steps) ...)

;;; Simpson's rule integration, Lagrange interpolation, DFT
```

**Neural Networks with Backpropagation**:
```scheme
;;; Gradient descent using autodiff (ultimate_math_stress.esk)
(define (gradient-descent f start learning-rate iterations)
  (define (step point)
    (let* ((grad (gradient f point))  ;; Automatic differentiation!
           (gx (tensor-get grad 0))
           (gy (tensor-get grad 1)))
      (vector (- x (* learning-rate gx))
              (- y (* learning-rate gy)))))
  ...)

;;; Full neural network gradient computation
(define nn-grad (gradient nn-forward (vector 1.0 0.5)))
```

**Linear Algebra** (`lib/math.esk`):
```scheme
(det M n)         ; Determinant via LU O(n³)
(inv M n)         ; Matrix inverse via Gauss-Jordan
(solve A b n)     ; Solve Ax=b
(dot u v)         ; Dot product
(normalize v)     ; Unit vector
```

**Advanced Functional Programming** (extreme_stress_test_v2.esk):
- Y combinator, CPS transformations, Church encodings
- 10-level nested closures, trampolines for deep recursion
- Variadic functions, higher-order composition

### 2.2 What qLLM Requires (from `semiclassical_qllm/src/`)

| Component | Location | Eshkol Status | Gradient Support |
|-----------|----------|---------------|------------------|
| N-D Tensor ops | `core/tensor.c` | ✅ **IMPLEMENTED** | AD extension needed |
| 1D Dot product | `core/tensor.c` | ✅ **IMPLEMENTED** (`tensor-dot`, `dot`) | AD extension needed |
| 2D Matrix multiplication | `core/tensor.c:1800+` | ✅ **IMPLEMENTED** (`matmul` [M×K]@[K×N]→[M×N]) | AD extension needed |
| Matrix inverse | N/A | ✅ **IMPLEMENTED** (`inv` in lib/math.esk) | Works with gradient! |
| Determinant | N/A | ✅ **IMPLEMENTED** (`det` in lib/math.esk) | Works with gradient! |
| Linear solve | N/A | ✅ **IMPLEMENTED** (`solve` in lib/math.esk) | Works with gradient! |
| Softmax | `model/attention.c` | ❌ Not implemented | Needs implementation |
| LayerNorm | `model/transformer.c:400+` | ❌ Not implemented | Needs implementation |
| Multi-head attention | `model/attention.c` | ❌ Not implemented | Needs implementation |
| Transformer blocks | `model/transformer.c` | ❌ Not implemented | Needs implementation |
| Hyperbolic operations | `geometric/hyperbolic_core.c` | ❌ Not implemented | Needs implementation |
| Riemannian Adam | `optimization/riemannian_adam.c` | ❌ Not implemented | Expects gradients as input |

**Key Insight**: Pure Eshkol functions like `det`, `inv`, `solve` can be differentiated automatically using the existing gradient/jacobian/hessian operators!

---

## 3. Required Additions to Eshkol

### 3.1 New AD Node Types

**File:** `inc/eshkol/eshkol.h`

Add the following tensor operation types to the AD node type enum:

```c
// Existing scalar types (0-15 reserved):
// AD_NODE_CONST = 0
// AD_NODE_VAR = 1
// AD_NODE_ADD = 2
// AD_NODE_SUB = 3
// AD_NODE_MUL = 4
// AD_NODE_DIV = 5
// AD_NODE_SIN = 6
// AD_NODE_COS = 7
// AD_NODE_EXP = 8
// AD_NODE_LOG = 9
// AD_NODE_POW = 10
// AD_NODE_SQRT = 11
// AD_NODE_TANH = 12
// AD_NODE_NEG = 13

// NEW: Tensor operation types (16-31)
#define AD_NODE_TENSOR_MATMUL       16
#define AD_NODE_TENSOR_SOFTMAX      17
#define AD_NODE_TENSOR_LAYERNORM    18
#define AD_NODE_TENSOR_ATTENTION    19
#define AD_NODE_TENSOR_TRANSPOSE    20
#define AD_NODE_TENSOR_SUM          21
#define AD_NODE_TENSOR_BROADCAST_ADD 22
#define AD_NODE_TENSOR_BROADCAST_MUL 23
#define AD_NODE_TENSOR_GELU         24
#define AD_NODE_TENSOR_DROPOUT      25

// Geometric operations for qLLM (32-47)
#define AD_NODE_HYPERBOLIC_DISTANCE 32
#define AD_NODE_POINCARE_EXP_MAP    33
#define AD_NODE_POINCARE_LOG_MAP    34
#define AD_NODE_TANGENT_PROJECT     35
#define AD_NODE_GEODESIC_ATTENTION  36
```

### 3.2 Extended AD Node Structure

**File:** `inc/eshkol/eshkol.h`

The current AD node structure handles scalars. Extend it to support tensor metadata:

```c
typedef struct eshkol_ad_node {
    uint32_t type;           // Operation type (from enum above)
    double value;            // Scalar value (for scalar ops)
    double gradient;         // Scalar gradient (for scalar ops)
    struct eshkol_ad_node* input1;  // First input
    struct eshkol_ad_node* input2;  // Second input (for binary ops)
    uint64_t node_id;        // Unique node ID

    // NEW: Tensor support fields
    void* tensor_data;       // Pointer to qllm_tensor_t (for tensor ops)
    void* tensor_grad;       // Pointer to gradient tensor
    size_t* shape;           // Tensor shape
    size_t dims;             // Number of dimensions
    int reduction_dim;       // For softmax, sum, etc.
    float scale_factor;      // For attention scaling

    // NEW: Cached values for backward
    void* cached_forward;    // Cached forward pass values (e.g., softmax output)
    void* cached_aux;        // Auxiliary cached data (e.g., normalized values for layernorm)
} eshkol_ad_node_t;
```

### 3.3 New Value Type Tag

**File:** `inc/eshkol/eshkol.h`

Add a new tagged value type for qLLM tensors:

```c
typedef enum {
    // ... existing types ...
    ESHKOL_VALUE_QLLM_TENSOR = 20,  // Pointer to qllm_tensor_t
    ESHKOL_VALUE_QLLM_AD_TENSOR = 21,  // AD-tracked qLLM tensor
} eshkol_value_type_t;
```

### 3.4 FFI Bridge Header

**New File:** `inc/eshkol/bridge/qllm_bridge.h`

```c
#ifndef ESHKOL_QLLM_BRIDGE_H
#define ESHKOL_QLLM_BRIDGE_H

#include <eshkol/eshkol.h>

// Forward declaration (actual definition in qLLM)
typedef struct qllm_tensor qllm_tensor_t;

// === Tensor Conversion ===

// Convert Eshkol tensor to qLLM tensor (double -> float32)
qllm_tensor_t* eshkol_to_qllm_tensor(const eshkol_tagged_value_t* eshkol_tensor);

// Convert qLLM tensor to Eshkol tagged value
eshkol_tagged_value_t qllm_to_eshkol_tensor(const qllm_tensor_t* qllm_tensor);

// === AD-Tracked Tensor Operations ===
// These record operations on the tape and return AD nodes

// Matrix multiplication: C = A @ B
eshkol_ad_node_t* ad_tensor_matmul(
    eshkol_ad_node_t* A,  // [M, K] or batched [B, M, K]
    eshkol_ad_node_t* B   // [K, N] or batched [B, K, N]
);

// Softmax: y = softmax(x, dim)
eshkol_ad_node_t* ad_tensor_softmax(
    eshkol_ad_node_t* x,
    int dim
);

// Layer normalization: y = (x - mean) / std * gamma + beta
eshkol_ad_node_t* ad_tensor_layernorm(
    eshkol_ad_node_t* x,
    eshkol_ad_node_t* gamma,
    eshkol_ad_node_t* beta,
    float eps
);

// Multi-head attention (full operation with caching)
typedef struct {
    eshkol_ad_node_t* output;
    qllm_tensor_t* attention_weights;  // Cached for backward
    qllm_tensor_t* scores;             // Cached for backward
} ad_attention_result_t;

ad_attention_result_t ad_multi_head_attention(
    eshkol_ad_node_t* Q,
    eshkol_ad_node_t* K,
    eshkol_ad_node_t* V,
    eshkol_ad_node_t* mask,  // Optional
    int num_heads
);

// === Geometric Operations for qLLM ===

// Hyperbolic distance in Poincare ball
eshkol_ad_node_t* ad_hyperbolic_distance(
    eshkol_ad_node_t* p1,
    eshkol_ad_node_t* p2,
    float curvature
);

// Exponential map (tangent space -> manifold)
eshkol_ad_node_t* ad_poincare_exp_map(
    eshkol_ad_node_t* base_point,
    eshkol_ad_node_t* tangent_vector,
    float curvature
);

// Logarithmic map (manifold -> tangent space)
eshkol_ad_node_t* ad_poincare_log_map(
    eshkol_ad_node_t* base_point,
    eshkol_ad_node_t* target_point,
    float curvature
);

// Project gradient to tangent space (for Riemannian optimization)
eshkol_ad_node_t* ad_project_to_tangent(
    eshkol_ad_node_t* point,
    eshkol_ad_node_t* euclidean_gradient,
    float curvature
);

// === Backward Pass Functions ===
// These compute gradients for tensor operations

void tensor_matmul_backward(
    const eshkol_ad_node_t* node,
    qllm_tensor_t** grad_A,
    qllm_tensor_t** grad_B
);

void tensor_softmax_backward(
    const eshkol_ad_node_t* node,
    qllm_tensor_t** grad_input
);

void tensor_layernorm_backward(
    const eshkol_ad_node_t* node,
    qllm_tensor_t** grad_input,
    qllm_tensor_t** grad_gamma,
    qllm_tensor_t** grad_beta
);

void tensor_attention_backward(
    const eshkol_ad_node_t* node,
    qllm_tensor_t** grad_Q,
    qllm_tensor_t** grad_K,
    qllm_tensor_t** grad_V
);

// === Integration with Riemannian Optimizer ===

// Extract gradient tensor from AD node (for optimizer input)
qllm_tensor_t* extract_gradient_tensor(const eshkol_ad_node_t* node);

// Run backward pass starting from loss node
void tensor_backpropagate(eshkol_ad_node_t* loss_node);

#endif // ESHKOL_QLLM_BRIDGE_H
```

### 3.5 Tensor Backward Pass Implementation

**New File:** `lib/bridge/tensor_backward.cpp`

Add tensor gradient propagation to the switch statement in `propagateGradient`:

```cpp
void AutodiffCodegen::propagateGradientTensor(llvm::Value* node_ptr) {
    // Load node type
    llvm::Value* type_ptr = ctx_.builder().CreateStructGEP(ad_node_type, node_ptr, 0);
    llvm::Value* node_type = ctx_.builder().CreateLoad(ctx_.int32Type(), type_ptr);

    // Load tensor gradient (pointer to qllm_tensor_t)
    llvm::Value* grad_tensor_ptr = ctx_.builder().CreateStructGEP(ad_node_type, node_ptr,
        OFFSET_TENSOR_GRAD);
    llvm::Value* grad_tensor = ctx_.builder().CreateLoad(ctx_.ptrType(), grad_tensor_ptr);

    // Switch on tensor operation type
    llvm::SwitchInst* sw = ctx_.builder().CreateSwitch(node_type, done_block, 6);
    sw->addCase(llvm::ConstantInt::get(ctx_.int32Type(), AD_NODE_TENSOR_MATMUL), matmul_block);
    sw->addCase(llvm::ConstantInt::get(ctx_.int32Type(), AD_NODE_TENSOR_SOFTMAX), softmax_block);
    sw->addCase(llvm::ConstantInt::get(ctx_.int32Type(), AD_NODE_TENSOR_LAYERNORM), layernorm_block);
    sw->addCase(llvm::ConstantInt::get(ctx_.int32Type(), AD_NODE_TENSOR_ATTENTION), attention_block);
    sw->addCase(llvm::ConstantInt::get(ctx_.int32Type(), AD_NODE_HYPERBOLIC_DISTANCE), hyperbolic_block);
    sw->addCase(llvm::ConstantInt::get(ctx_.int32Type(), AD_NODE_POINCARE_EXP_MAP), exp_map_block);

    // MATMUL backward: dL/dA = dL/dC @ B^T, dL/dB = A^T @ dL/dC
    ctx_.builder().SetInsertPoint(matmul_block);
    {
        // Load input tensors
        llvm::Value* input1_ptr = loadNodeInput1(node_ptr);  // A
        llvm::Value* input2_ptr = loadNodeInput2(node_ptr);  // B
        llvm::Value* A_data = loadTensorData(input1_ptr);
        llvm::Value* B_data = loadTensorData(input2_ptr);

        // Call qLLM tensor operations (via FFI)
        llvm::Function* transpose_fn = getQllmFunction("qllm_tensor_transpose");
        llvm::Function* matmul_fn = getQllmFunction("qllm_tensor_matmul");

        // B^T
        llvm::Value* B_T = ctx_.builder().CreateCall(transpose_fn, {B_data,
            llvm::ConstantInt::get(ctx_.int32Type(), -1),
            llvm::ConstantInt::get(ctx_.int32Type(), -2)});

        // dL/dA = dL/dC @ B^T
        llvm::Value* grad_A = ctx_.builder().CreateCall(matmul_fn, {grad_tensor, B_T});
        accumulateTensorGradient(input1_ptr, grad_A);

        // A^T
        llvm::Value* A_T = ctx_.builder().CreateCall(transpose_fn, {A_data,
            llvm::ConstantInt::get(ctx_.int32Type(), -1),
            llvm::ConstantInt::get(ctx_.int32Type(), -2)});

        // dL/dB = A^T @ dL/dC
        llvm::Value* grad_B = ctx_.builder().CreateCall(matmul_fn, {A_T, grad_tensor});
        accumulateTensorGradient(input2_ptr, grad_B);
    }
    ctx_.builder().CreateBr(done_block);

    // SOFTMAX backward: dL/dx = y * (dL/dy - sum(dL/dy * y))
    ctx_.builder().SetInsertPoint(softmax_block);
    {
        llvm::Value* input_ptr = loadNodeInput1(node_ptr);
        llvm::Value* cached_output = loadCachedForward(node_ptr);  // softmax output y
        llvm::Value* dim = loadReductionDim(node_ptr);

        llvm::Function* mul_fn = getQllmFunction("qllm_tensor_mul");
        llvm::Function* sum_fn = getQllmFunction("qllm_tensor_sum");
        llvm::Function* sub_fn = getQllmFunction("qllm_tensor_sub");

        // prod = dL/dy * y
        llvm::Value* prod = ctx_.builder().CreateCall(mul_fn, {grad_tensor, cached_output});

        // dot = sum(prod, dim, keepdim=true)
        llvm::Value* dot = ctx_.builder().CreateCall(sum_fn, {prod, dim,
            llvm::ConstantInt::get(ctx_.int1Type(), 1)});

        // diff = dL/dy - dot (broadcasted)
        llvm::Value* diff = ctx_.builder().CreateCall(sub_fn, {grad_tensor, dot});

        // grad_input = y * diff
        llvm::Value* grad_input = ctx_.builder().CreateCall(mul_fn, {cached_output, diff});
        accumulateTensorGradient(input_ptr, grad_input);
    }
    ctx_.builder().CreateBr(done_block);

    // LAYERNORM backward (complex - see full derivation)
    ctx_.builder().SetInsertPoint(layernorm_block);
    {
        // ... implementation per Section 6.1 of ESHKOL_QLLM_TRAINING_SYSTEM_ANALYSIS.md
    }
    ctx_.builder().CreateBr(done_block);

    // ATTENTION backward (multi-step chain rule)
    ctx_.builder().SetInsertPoint(attention_block);
    {
        // ... implementation per Section 6.1 of ESHKOL_QLLM_TRAINING_SYSTEM_ANALYSIS.md
    }
    ctx_.builder().CreateBr(done_block);
}
```

### 3.6 LLVM Codegen Modifications

**File:** `lib/backend/llvm_codegen.cpp`

Add FFI function declarations and tensor operation handling:

```cpp
// Add to initializeQllmFunctions()
void LLVMCodegen::initializeQllmFunctions() {
    // Tensor creation and manipulation
    declareQllmFunction("qllm_tensor_create", ptrType(),
        {int64Type(), ptrType(), int32Type()});  // dims, shape, dtype
    declareQllmFunction("qllm_tensor_destroy", voidType(), {ptrType()});
    declareQllmFunction("qllm_tensor_clone", ptrType(), {ptrType()});

    // Basic tensor operations
    declareQllmFunction("qllm_tensor_matmul", ptrType(), {ptrType(), ptrType()});
    declareQllmFunction("qllm_tensor_transpose", ptrType(),
        {ptrType(), int32Type(), int32Type()});
    declareQllmFunction("qllm_tensor_add", ptrType(), {ptrType(), ptrType()});
    declareQllmFunction("qllm_tensor_sub", ptrType(), {ptrType(), ptrType()});
    declareQllmFunction("qllm_tensor_mul", ptrType(), {ptrType(), ptrType()});
    declareQllmFunction("qllm_tensor_div", ptrType(), {ptrType(), ptrType()});
    declareQllmFunction("qllm_tensor_sum", ptrType(),
        {ptrType(), int32Type(), int1Type()});  // tensor, dim, keepdim
    declareQllmFunction("qllm_tensor_mean", ptrType(),
        {ptrType(), int32Type(), int1Type()});

    // Neural network operations
    declareQllmFunction("qllm_tensor_softmax", ptrType(), {ptrType(), int32Type()});
    declareQllmFunction("qllm_tensor_gelu", ptrType(), {ptrType()});
    declareQllmFunction("qllm_layer_norm", ptrType(),
        {ptrType(), ptrType(), ptrType(), doubleType()});  // x, gamma, beta, eps

    // Hyperbolic operations
    declareQllmFunction("hyperbolic_distance", ptrType(),
        {ptrType(), ptrType(), doubleType()});  // p1, p2, curvature
    declareQllmFunction("poincare_exp_map", ptrType(),
        {ptrType(), ptrType(), doubleType()});
    declareQllmFunction("poincare_log_map", ptrType(),
        {ptrType(), ptrType(), doubleType()});
    declareQllmFunction("project_to_tangent", ptrType(),
        {ptrType(), ptrType(), doubleType()});
}
```

### 3.7 Type Conversion Functions

**New File:** `lib/bridge/type_conversion.cpp`

```cpp
#include <eshkol/bridge/qllm_bridge.h>
#include <semiclassical_qllm/tensor.h>

// Convert Eshkol tensor (double) to qLLM tensor (float32)
qllm_tensor_t* eshkol_to_qllm_tensor(const eshkol_tagged_value_t* eshkol_val) {
    if (eshkol_val->type != ESHKOL_VALUE_TENSOR_PTR) {
        return nullptr;
    }

    eshkol_tensor_t* e_tensor = (eshkol_tensor_t*)eshkol_val->data.ptr_val;

    // Create qLLM tensor with same shape
    size_t* shape = e_tensor->shape;
    size_t dims = e_tensor->dims;
    size_t total = 1;
    for (size_t i = 0; i < dims; i++) {
        total *= shape[i];
    }

    qllm_tensor_t* q_tensor = qllm_tensor_create(dims, shape, QLLM_FLOAT32);

    // Convert double -> float32
    float* q_data = (float*)qllm_tensor_data(q_tensor);
    double* e_data = (double*)eshkol_tensor_data(e_tensor);

    for (size_t i = 0; i < total; i++) {
        q_data[i] = (float)e_data[i];
    }

    return q_tensor;
}

// Convert qLLM tensor (float32) to Eshkol tensor (double)
eshkol_tagged_value_t qllm_to_eshkol_tensor(const qllm_tensor_t* q_tensor) {
    size_t dims = qllm_tensor_dims(q_tensor);
    const size_t* shape = qllm_tensor_shape(q_tensor);
    size_t total = 1;
    for (size_t i = 0; i < dims; i++) {
        total *= shape[i];
    }

    // Create Eshkol tensor
    eshkol_tensor_t* e_tensor = eshkol_tensor_create(dims, shape);

    // Convert float32 -> double
    double* e_data = (double*)eshkol_tensor_data(e_tensor);
    const float* q_data = (const float*)qllm_tensor_data(q_tensor);

    for (size_t i = 0; i < total; i++) {
        e_data[i] = (double)q_data[i];
    }

    eshkol_tagged_value_t result;
    result.type = ESHKOL_VALUE_TENSOR_PTR;
    result.data.ptr_val = (uint64_t)e_tensor;
    return result;
}
```

---

## 4. Required Chain Rule Implementations

### 4.1 MatMul Backward

Given C = A @ B where A is [M,K], B is [K,N], C is [M,N]:

```
dL/dA = dL/dC @ B^T     (result: [M,K])
dL/dB = A^T @ dL/dC     (result: [K,N])
```

**Batched version** (batch dim preserved):
```
dL/dA[b] = dL/dC[b] @ B[b]^T
dL/dB[b] = A[b]^T @ dL/dC[b]
```

### 4.2 Softmax Backward

Given y = softmax(x) where y[i] = exp(x[i]) / sum(exp(x)):

```
dL/dx[i] = y[i] * (dL/dy[i] - sum_j(dL/dy[j] * y[j]))
```

Efficient implementation:
```
dot = sum(dL/dy * y, dim=-1, keepdim=True)
dL/dx = y * (dL/dy - dot)
```

### 4.3 LayerNorm Backward

Given y = (x - mu) / sigma * gamma + beta:

```
# Intermediate values
x_hat = (x - mu) / sigma

# Gradients
dL/d_gamma = sum(dL/dy * x_hat, batch_dims)
dL/d_beta = sum(dL/dy, batch_dims)

# Input gradient (complex formula)
N = feature_dim
dL/d_xhat = dL/dy * gamma
dL/d_sigma = sum(dL/d_xhat * (x - mu) * -0.5 * sigma^(-3), feature_dim)
dL/d_mu = sum(dL/d_xhat * -1/sigma, feature_dim) + dL/d_sigma * sum(-2*(x-mu)/N, feature_dim)
dL/dx = dL/d_xhat / sigma + dL/d_sigma * 2*(x-mu)/N + dL/d_mu / N
```

### 4.4 Attention Backward

Given:
```
scores = Q @ K^T / sqrt(d_k)
attention_weights = softmax(scores)
context = attention_weights @ V
output = context @ W_O
```

Backward pass:
```
# Step 1: Through output projection
dL/d_context = dL/d_output @ W_O^T
dL/d_W_O = context^T @ dL/d_output

# Step 2: Through attention @ V
dL/d_attn = dL/d_context @ V^T
dL/d_V = attention_weights^T @ dL/d_context

# Step 3: Through softmax
dL/d_scores = softmax_backward(dL/d_attn, attention_weights)

# Step 4: Through scaling
dL/d_scores_unscaled = dL/d_scores / sqrt(d_k)

# Step 5: Through Q @ K^T
dL/d_Q = dL/d_scores_unscaled @ K
dL/d_K = dL/d_scores_unscaled^T @ Q
```

### 4.5 Hyperbolic Distance Backward (Poincare Ball)

Given d = arccosh(1 + 2 * ||x-y||^2 / ((1-||x||^2)(1-||y||^2))):

```
# Let:
# sqnorm_x = ||x||^2
# sqnorm_y = ||y||^2
# sqnorm_diff = ||x-y||^2
# alpha = 1 - sqnorm_x
# beta = 1 - sqnorm_y
# gamma = 1 + 2 * sqnorm_diff / (alpha * beta)

# Gradients:
dL/d_x = dL/d_d * (4 / (alpha * beta * sqrt(gamma^2 - 1))) *
         ((x - y) / alpha + sqnorm_diff * x / alpha^2)

dL/d_y = dL/d_d * (4 / (alpha * beta * sqrt(gamma^2 - 1))) *
         ((y - x) / beta + sqnorm_diff * y / beta^2)
```

### 4.6 Poincare Exponential Map Backward

Given exp_p(v) = p + (tanh(lambda_p * ||v|| / 2) / ||v||) * v:

```
# Where lambda_p = 2 / (1 - ||p||^2)

# The backward pass requires careful handling of:
# 1. Gradient through tanh
# 2. Gradient through normalization
# 3. Gradient through Mobius addition
```

---

## 5. Build System Modifications

### 5.1 CMakeLists.txt Additions

```cmake
# Add qLLM as external dependency
find_package(qLLM REQUIRED PATHS ${QLLM_DIR})

# Add bridge library
add_library(eshkol_qllm_bridge
    lib/bridge/type_conversion.cpp
    lib/bridge/tensor_backward.cpp
    lib/bridge/qllm_primitives.cpp
)

target_link_libraries(eshkol_qllm_bridge
    eshkol_core
    qllm::tensor
    qllm::geometric
)

# Add include paths
target_include_directories(eshkol_qllm_bridge PUBLIC
    ${CMAKE_SOURCE_DIR}/inc
    ${QLLM_INCLUDE_DIR}
)
```

### 5.2 Linker Configuration

The following qLLM libraries must be linked:
- `qllm_tensor` - Core tensor operations
- `qllm_model` - Transformer components
- `qllm_geometric` - Hyperbolic operations
- `qllm_optimization` - Riemannian Adam (runtime only)

---

## 6. Runtime Integration

### 6.1 Training Loop Structure

```scheme
;; Example training loop in Eshkol syntax
(define (train-qllm model data epochs learning-rate)
  (define optimizer (create-riemannian-adam learning-rate))

  (do ((epoch 0 (+ epoch 1)))
      ((>= epoch epochs))

    ;; Forward pass (recorded on tape)
    (let* ((batch (get-batch data))
           (inputs (batch-inputs batch))
           (targets (batch-targets batch))
           (outputs (model-forward model inputs))
           (loss (cross-entropy-loss outputs targets)))

      ;; Backward pass (uses tape)
      (tensor-backpropagate loss)

      ;; Extract gradients and update
      (let ((params (model-parameters model))
            (grads (map extract-gradient-tensor params)))
        (riemannian-adam-step optimizer params grads))

      ;; Clear tape for next iteration
      (clear-tape))))
```

### 6.2 Integration with Riemannian Adam

The qLLM Riemannian Adam optimizer expects:
```c
bool riemannian_adam_step(
    qllm_riemannian_optimizer_t* opt,
    qllm_tensor_t* parameters,      // Model parameters
    const qllm_tensor_t* gradients  // Computed by Eshkol AD
);
```

Eshkol's role is to compute the `gradients` tensor via automatic differentiation.

---

## 7. Implementation Priority and Effort

### Phase 1: Core Infrastructure (2 weeks)
1. Add tensor AD node types to `eshkol.h`
2. Extend AD node structure for tensor metadata
3. Create FFI bridge header
4. Implement type conversion functions

### Phase 2: Basic Tensor Operations (2 weeks)
1. MatMul forward + backward
2. Element-wise operations forward + backward
3. Sum/Mean reduction forward + backward
4. Transpose forward + backward

### Phase 3: Neural Network Operations (2 weeks)
1. Softmax forward + backward
2. LayerNorm forward + backward
3. GELU forward + backward
4. Dropout forward + backward

### Phase 4: Attention Mechanism (2 weeks)
1. Scaled dot-product attention forward + backward
2. Multi-head attention forward + backward
3. Attention caching for efficiency

### Phase 5: Geometric Operations (2 weeks)
1. Hyperbolic distance forward + backward
2. Poincare exp/log maps forward + backward
3. Tangent projection forward + backward
4. Geodesic attention forward + backward

### Phase 6: Integration and Testing (2 weeks)
1. Connect to Riemannian Adam optimizer
2. Implement training loop
3. Gradient verification against numerical gradients
4. Performance optimization

**Total Estimated Effort: 12 weeks**

---

## 8. Verification Strategy

### 8.1 Numerical Gradient Checking

For each tensor operation, verify:
```
|analytical_gradient - numerical_gradient| / (|numerical_gradient| + eps) < tolerance
```

Where numerical gradient uses central differences:
```
numerical_grad[i] = (f(x + h*e_i) - f(x - h*e_i)) / (2*h)
```

### 8.2 Test Cases

1. **MatMul**: Random matrices, verify against numpy
2. **Softmax**: Edge cases (uniform, one-hot, extreme values)
3. **LayerNorm**: Various batch sizes and feature dims
4. **Attention**: Compare against PyTorch implementation
5. **Hyperbolic**: Compare against geoopt library

---

## 9. Summary

To use Eshkol as the qLLM training engine, the following must be implemented:

| Component | Files to Modify/Create | Effort |
|-----------|------------------------|--------|
| New AD node types | `eshkol.h` | 1 day |
| Extended AD node struct | `eshkol.h` | 1 day |
| FFI bridge header | `qllm_bridge.h` (new) | 2 days |
| Type conversion | `type_conversion.cpp` (new) | 3 days |
| MatMul backward | `tensor_backward.cpp` (new) | 3 days |
| Softmax backward | `tensor_backward.cpp` | 2 days |
| LayerNorm backward | `tensor_backward.cpp` | 3 days |
| Attention backward | `tensor_backward.cpp` | 5 days |
| Hyperbolic ops backward | `tensor_backward.cpp` | 5 days |
| LLVM codegen extensions | `llvm_codegen.cpp` | 5 days |
| Build system | `CMakeLists.txt` | 1 day |
| Testing framework | Various | 5 days |

**Total: ~36 person-days (~7-8 weeks at full-time)**

The key insight is that Eshkol's existing AD architecture is sound - it just needs to be extended from scalars to tensors with proper chain rules for the specific operations used in the qLLM transformer architecture.
