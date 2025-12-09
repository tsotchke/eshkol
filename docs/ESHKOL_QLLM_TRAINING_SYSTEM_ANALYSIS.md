# Eshkol-qLLM Training System Compatibility Analysis

## Technical Assessment Report

**Date:** December 7, 2025
**Scope:** Evaluation of Eshkol v1.0.0-foundation for building the semi-classical qLLM training system
**Methodology:** Deep code analysis of both codebases (62k+ lines Eshkol, 267k+ lines qLLM)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Eshkol Architecture Deep Dive](#2-eshkol-architecture-deep-dive)
   - 2.1 [Core Type System](#21-core-type-system)
   - 2.2 [HoTT Type System Implementation](#22-hott-type-system-implementation)
   - 2.3 [Automatic Differentiation System](#23-automatic-differentiation-system)
   - 2.4 [Memory Management](#24-memory-management)
   - 2.5 [LLVM Code Generation](#25-llvm-code-generation)
3. [qLLM Architecture Deep Dive](#3-qllm-architecture-deep-dive)
   - 3.1 [Tensor System](#31-tensor-system)
   - 3.2 [Transformer Architecture](#32-transformer-architecture)
   - 3.3 [Geometric Operations](#33-geometric-operations)
   - 3.4 [Riemannian Optimization](#34-riemannian-optimization)
4. [Compatibility Analysis](#4-compatibility-analysis)
   - 4.1 [Architectural Alignment](#41-architectural-alignment)
   - 4.2 [Type System Mapping](#42-type-system-mapping)
   - 4.3 [AD System Comparison](#43-ad-system-comparison)
   - 4.4 [Gap Analysis](#44-gap-analysis)
5. [Integration Pathways](#5-integration-pathways)
   - 5.1 [Path A: Eshkol as Training DSL](#51-path-a-eshkol-as-training-dsl)
   - 5.2 [Path B: Port AD to C++](#52-path-b-port-ad-to-c)
   - 5.3 [Path C: Hybrid Approach](#53-path-c-hybrid-approach)
6. [Implementation Specifications](#6-implementation-specifications)
   - 6.1 [Required Tensor AD Operations](#61-required-tensor-ad-operations)
   - 6.2 [FFI Bridge Design](#62-ffi-bridge-design)
   - 6.3 [Gradient Verification Framework](#63-gradient-verification-framework)
7. [Conclusions and Recommendations](#7-conclusions-and-recommendations)

---

## 1. Executive Summary

This report presents a comprehensive technical analysis of the Eshkol programming language's capability to serve as the foundation for the semi-classical qLLM training system. The analysis is based on direct examination of source code, not documentation.

### Key Findings

| Aspect | Eshkol Status | qLLM Requirement | Compatibility |
|--------|---------------|------------------|---------------|
| Reverse-mode AD | Fully implemented | Required | **High** |
| Forward-mode AD | Fully implemented | For verification | **High** |
| Tape-based backprop | Implemented | Required | **High** |
| Tensor operations | 1D vectors/tensors | N-D tensors | **Medium** |
| Memory management | Arena-based | Arena/pool | **High** |
| Type system | HoTT with universes | C structs | **Eshkol superior** |
| Neural network primitives | Demo implementations | Production required | **Medium** |

### Verdict

**Eshkol cannot directly build the qLLM training system out-of-the-box**, but possesses the correct architectural foundations. The reverse-mode AD system, tape structure, and backpropagation logic are mathematically equivalent to what qLLM requires. The gap is primarily in:

1. Data type adaptation (double → float32/float16)
2. Tensor dimensionality (1D → N-D with broadcasting)
3. FFI bridge to qLLM's C tensor library
4. Tensor-level chain rules (matmul, softmax, attention)

---

## 2. Eshkol Architecture Deep Dive

### 2.1 Core Type System

Eshkol implements a sophisticated tagged value system that enables runtime polymorphism while maintaining type safety. The core representation is defined in `inc/eshkol/eshkol.h`:

```c
// 16-byte tagged value - the universal runtime representation
typedef struct eshkol_tagged_value {
    uint8_t type;        // Value type tag (4 bits base + 4 bits flags)
    uint8_t flags;       // Exactness and other flags
    uint16_t reserved;   // Alignment padding
    union {
        int64_t int_val;
        double double_val;
        uint64_t ptr_val;
        uint64_t raw_val;
    } data;
} eshkol_tagged_value_t;

_Static_assert(sizeof(eshkol_tagged_value_t) <= 16,
               "Tagged value must fit in 16 bytes for efficiency");
```

#### Value Type Tags

```c
typedef enum {
    ESHKOL_VALUE_NULL        = 0,   // Empty/null value
    ESHKOL_VALUE_INT64       = 1,   // 64-bit signed integer
    ESHKOL_VALUE_DOUBLE      = 2,   // Double-precision floating point
    ESHKOL_VALUE_CONS_PTR    = 3,   // Pointer to cons cell (lists)
    ESHKOL_VALUE_DUAL_NUMBER = 4,   // Forward-mode AD dual number
    ESHKOL_VALUE_AD_NODE_PTR = 5,   // Reverse-mode AD computation graph node
    ESHKOL_VALUE_TENSOR_PTR  = 6,   // Pointer to tensor structure
    ESHKOL_VALUE_LAMBDA_SEXPR = 7,  // Lambda S-expression (homoiconicity)
    ESHKOL_VALUE_STRING_PTR  = 8,   // Pointer to string
    ESHKOL_VALUE_CHAR        = 9,   // Unicode codepoint
    ESHKOL_VALUE_VECTOR_PTR  = 10,  // Scheme vector (heterogeneous array)
    ESHKOL_VALUE_SYMBOL      = 11,  // Interned symbol
    ESHKOL_VALUE_CLOSURE_PTR = 12,  // Closure (func_ptr + captured env)
    ESHKOL_VALUE_BOOL        = 13,  // Boolean (#t or #f)
} eshkol_value_type_t;
```

#### Exactness Tracking (Scheme R7RS Compliance)

```c
#define ESHKOL_VALUE_EXACT_FLAG   0x10
#define ESHKOL_VALUE_INEXACT_FLAG 0x20

// Type checking macros
#define ESHKOL_IS_EXACT(type)   (((type) & ESHKOL_VALUE_EXACT_FLAG) != 0)
#define ESHKOL_IS_INEXACT(type) (((type) & ESHKOL_VALUE_INEXACT_FLAG) != 0)
```

This exactness tracking is critical for numerical computation - exact integers preserve precision while inexact floats propagate uncertainty through computations.

### 2.2 HoTT Type System Implementation

Eshkol implements a Homotopy Type Theory-inspired type system in `lib/types/hott_types.cpp`. This is significantly more sophisticated than typical dynamic language type systems.

#### Universe Hierarchy

```cpp
namespace eshkol::hott {

enum class Universe {
    U0,      // Type₀ - ground types (Int64, Float64, Boolean, etc.)
    U1,      // Type₁ - type constructors (List, Vector, Tensor, ->)
    U2,      // Type₂ - propositions and proofs
    UOmega   // Typeω - universe polymorphism
};

// Type flags
constexpr uint8_t TYPE_FLAG_EXACT   = 0x01;  // Scheme exactness
constexpr uint8_t TYPE_FLAG_LINEAR  = 0x02;  // Linear types (use exactly once)
constexpr uint8_t TYPE_FLAG_PROOF   = 0x04;  // Erased at runtime

// Runtime representation hints
enum class RuntimeRep {
    Int64,        // 64-bit integer
    Float64,      // 64-bit float
    Pointer,      // Heap pointer
    TaggedValue,  // Full tagged value
    Struct,       // Compound structure
    Erased        // No runtime representation (proofs)
};

}
```

#### Type Hierarchy Implementation

```cpp
void TypeEnvironment::initializeBuiltinTypes() {
    // Value hierarchy (root supertype)
    registerBuiltinType(Value.id, "Value", Universe::U0, 0, RuntimeRep::TaggedValue);

    // Numeric tower
    registerBuiltinType(Number.id, "Number", Universe::U0, 0,
                        RuntimeRep::TaggedValue, Value);
    registerBuiltinType(Integer.id, "Integer", Universe::U0, TYPE_FLAG_EXACT,
                        RuntimeRep::Int64, Number);
    registerBuiltinType(Int64.id, "Int64", Universe::U0, TYPE_FLAG_EXACT,
                        RuntimeRep::Int64, Integer);
    registerBuiltinType(Real.id, "Real", Universe::U0, 0,
                        RuntimeRep::Float64, Number);
    registerBuiltinType(Float64.id, "Float64", Universe::U0, 0,
                        RuntimeRep::Float64, Real);

    // Type constructors (U1)
    registerTypeFamily(List.id, "List", Universe::U1, {"a"}, RuntimeRep::Pointer);
    registerTypeFamily(Vector.id, "Vector", Universe::U1, {"a"}, RuntimeRep::Pointer);
    registerTypeFamily(Tensor.id, "Tensor", Universe::U1, {"a", "shape"}, RuntimeRep::Pointer);
    registerTypeFamily(Function.id, "->", Universe::U1, {"a", "b"}, RuntimeRep::Pointer);

    // Autodiff types
    registerTypeFamily(DualNumber.id, "Dual", Universe::U1, {"a"}, RuntimeRep::Struct);
    registerTypeFamily(ADNode.id, "ADNode", Universe::U1, {"a"}, RuntimeRep::Pointer);

    // Linear resource types
    registerTypeFamily(Handle.id, "Handle", Universe::U1, {"k"}, RuntimeRep::Pointer);
    types_[Handle.id].id.flags |= TYPE_FLAG_LINEAR;

    // Proposition types (erased at runtime)
    registerBuiltinType(Eq.id, "Eq", Universe::U2, TYPE_FLAG_PROOF, RuntimeRep::Erased);
    registerBuiltinType(Subtype.id, "<:", Universe::U2, TYPE_FLAG_PROOF, RuntimeRep::Erased);
}
```

#### Subtype Checking with Caching

```cpp
bool TypeEnvironment::isSubtype(TypeId sub, TypeId super) const {
    // Check cache first
    auto key = std::make_pair(sub.id, super.id);
    auto it = subtype_cache_.find(key);
    if (it != subtype_cache_.end()) {
        return it->second;
    }

    bool result = isSubtypeUncached(sub, super);
    subtype_cache_[key] = result;
    return result;
}

bool TypeEnvironment::isSubtypeUncached(TypeId sub, TypeId super) const {
    if (sub == super) return true;  // Reflexivity

    // Walk supertype chain
    const TypeNode* node = getTypeNode(sub);
    while (node->supertype.has_value()) {
        if (node->supertype.value() == super) return true;
        node = getTypeNode(node->supertype.value());
        if (!node) return false;
    }
    return false;
}
```

#### Type Promotion for Arithmetic

```cpp
TypeId TypeEnvironment::promoteForArithmetic(TypeId a, TypeId b) const {
    if (a == b) return a;

    // Integer + Real → Real (Float64)
    if ((isSubtype(a, Integer) && isSubtype(b, Real)) ||
        (isSubtype(a, Real) && isSubtype(b, Integer))) {
        return Float64;
    }

    // Both integers → Int64
    if (isSubtype(a, Integer) && isSubtype(b, Integer)) {
        return Int64;
    }

    // Both reals → Float64
    if (isSubtype(a, Real) && isSubtype(b, Real)) {
        return Float64;
    }

    auto lcs = leastCommonSupertype(a, b);
    return lcs.value_or(Number);
}
```

### 2.3 Automatic Differentiation System

The AD system is Eshkol's most sophisticated component for ML applications, implemented across `autodiff_codegen.cpp` and `llvm_codegen.cpp`.

#### 2.3.1 Forward-Mode AD (Dual Numbers)

```c
// Dual number structure - 16 bytes for cache efficiency
typedef struct eshkol_dual_number {
    double value;       // f(x) - the primal value
    double derivative;  // f'(x) - the tangent/derivative
} eshkol_dual_number_t;

_Static_assert(sizeof(eshkol_dual_number_t) == 16,
               "Dual number must be 16 bytes for cache efficiency");
```

Forward-mode propagates derivatives alongside values using the chain rule:

```cpp
// From autodiff_codegen.cpp - Dual number arithmetic
llvm::Value* AutodiffCodegen::dualMul(llvm::Value* dual_a, llvm::Value* dual_b) {
    // Product rule: d(a*b) = a'*b + a*b'
    auto [a, a_prime] = unpackDualNumber(dual_a);
    auto [b, b_prime] = unpackDualNumber(dual_b);

    llvm::Value* value = ctx_.builder().CreateFMul(a, b);
    llvm::Value* term1 = ctx_.builder().CreateFMul(a_prime, b);
    llvm::Value* term2 = ctx_.builder().CreateFMul(a, b_prime);
    llvm::Value* deriv = ctx_.builder().CreateFAdd(term1, term2);

    return createDualNumber(value, deriv);
}

llvm::Value* AutodiffCodegen::dualDiv(llvm::Value* dual_a, llvm::Value* dual_b) {
    // Quotient rule: d(a/b) = (a'*b - a*b') / b²
    auto [a, a_prime] = unpackDualNumber(dual_a);
    auto [b, b_prime] = unpackDualNumber(dual_b);

    llvm::Value* value = ctx_.builder().CreateFDiv(a, b);
    llvm::Value* term1 = ctx_.builder().CreateFMul(a_prime, b);
    llvm::Value* term2 = ctx_.builder().CreateFMul(a, b_prime);
    llvm::Value* num = ctx_.builder().CreateFSub(term1, term2);
    llvm::Value* b_sq = ctx_.builder().CreateFMul(b, b);
    llvm::Value* deriv = ctx_.builder().CreateFDiv(num, b_sq);

    return createDualNumber(value, deriv);
}

// Transcendental functions
llvm::Value* AutodiffCodegen::dualSin(llvm::Value* dual_x) {
    // d(sin(x)) = cos(x) * x'
    auto [x, x_prime] = unpackDualNumber(dual_x);

    llvm::Function* sin_intrinsic = llvm::Intrinsic::getOrInsertDeclaration(
        &ctx_.module(), llvm::Intrinsic::sin, {ctx_.doubleType()});
    llvm::Function* cos_intrinsic = llvm::Intrinsic::getOrInsertDeclaration(
        &ctx_.module(), llvm::Intrinsic::cos, {ctx_.doubleType()});

    llvm::Value* value = ctx_.builder().CreateCall(sin_intrinsic, {x});
    llvm::Value* cos_x = ctx_.builder().CreateCall(cos_intrinsic, {x});
    llvm::Value* deriv = ctx_.builder().CreateFMul(cos_x, x_prime);

    return createDualNumber(value, deriv);
}

llvm::Value* AutodiffCodegen::dualExp(llvm::Value* dual_x) {
    // d(exp(x)) = exp(x) * x'
    auto [x, x_prime] = unpackDualNumber(dual_x);

    llvm::Function* exp_intrinsic = llvm::Intrinsic::getOrInsertDeclaration(
        &ctx_.module(), llvm::Intrinsic::exp, {ctx_.doubleType()});

    llvm::Value* value = ctx_.builder().CreateCall(exp_intrinsic, {x});
    llvm::Value* deriv = ctx_.builder().CreateFMul(value, x_prime);

    return createDualNumber(value, deriv);
}

llvm::Value* AutodiffCodegen::dualLog(llvm::Value* dual_x) {
    // d(log(x)) = x' / x
    auto [x, x_prime] = unpackDualNumber(dual_x);

    llvm::Function* log_intrinsic = llvm::Intrinsic::getOrInsertDeclaration(
        &ctx_.module(), llvm::Intrinsic::log, {ctx_.doubleType()});

    llvm::Value* value = ctx_.builder().CreateCall(log_intrinsic, {x});
    llvm::Value* deriv = ctx_.builder().CreateFDiv(x_prime, x);

    return createDualNumber(value, deriv);
}
```

#### 2.3.2 Reverse-Mode AD (Computational Graph)

```c
// AD node types for computational graph
typedef enum {
    AD_NODE_CONSTANT,   // Leaf node: constant value
    AD_NODE_VARIABLE,   // Leaf node: input variable (has gradient)
    AD_NODE_ADD,        // Binary: z = x + y
    AD_NODE_SUB,        // Binary: z = x - y
    AD_NODE_MUL,        // Binary: z = x * y
    AD_NODE_DIV,        // Binary: z = x / y
    AD_NODE_SIN,        // Unary: z = sin(x)
    AD_NODE_COS,        // Unary: z = cos(x)
    AD_NODE_EXP,        // Unary: z = exp(x)
    AD_NODE_LOG,        // Unary: z = log(x)
    AD_NODE_POW,        // Binary: z = x^y
    AD_NODE_NEG         // Unary: z = -x
} ad_node_type_t;

// Computational graph node
typedef struct ad_node {
    ad_node_type_t type;     // Operation type
    double value;            // Forward pass result
    double gradient;         // Accumulated gradient (backward pass)
    struct ad_node* input1;  // First parent (NULL for leaves)
    struct ad_node* input2;  // Second parent (NULL for unary ops)
    size_t id;               // Unique ID for topological sorting
} ad_node_t;

// Tape for recording computation order
typedef struct ad_tape {
    ad_node_t** nodes;       // Nodes in evaluation order
    size_t num_nodes;        // Current count
    size_t capacity;         // Allocated capacity
    ad_node_t** variables;   // Input variable nodes (for gradient extraction)
    size_t num_variables;
} ad_tape_t;
```

#### Backpropagation Implementation

From `llvm_codegen.cpp`:

```cpp
void AutodiffCodegen::backpropagate(llvm::Value* tape, llvm::Value* output_node) {
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();

    // Create validity check blocks
    llvm::BasicBlock* check_validity = llvm::BasicBlock::Create(ctx_.context(),
        "backward_check_valid", current_func);
    llvm::BasicBlock* backward_valid = llvm::BasicBlock::Create(ctx_.context(),
        "backward_valid", current_func);
    llvm::BasicBlock* backward_skip = llvm::BasicBlock::Create(ctx_.context(),
        "backward_skip", current_func);

    ctx_.builder().CreateBr(check_validity);

    // Check if output node and tape are valid
    ctx_.builder().SetInsertPoint(check_validity);
    llvm::Value* output_int = ctx_.builder().CreatePtrToInt(output_node, ctx_.int64Type());
    llvm::Value* tape_int = ctx_.builder().CreatePtrToInt(tape, ctx_.int64Type());
    llvm::Value* output_valid = ctx_.builder().CreateICmpNE(output_int,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* tape_valid = ctx_.builder().CreateICmpNE(tape_int,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* both_valid = ctx_.builder().CreateAnd(output_valid, tape_valid);
    ctx_.builder().CreateCondBr(both_valid, backward_valid, backward_skip);

    ctx_.builder().SetInsertPoint(backward_valid);

    // STEP 1: Initialize output gradient = 1.0 (seed for backpropagation)
    storeNodeGradient(output_node, llvm::ConstantFP::get(ctx_.doubleType(), 1.0));

    // STEP 2: Get number of nodes in tape
    llvm::Function* get_count_func = mem_.getArenaTapeGetNodeCount();
    llvm::Value* num_nodes = ctx_.builder().CreateCall(get_count_func, {tape});

    // STEP 3: Allocate loop counter for backward traversal
    llvm::Value* counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr,
        "backward_counter");
    ctx_.builder().CreateStore(num_nodes, counter);  // Start at end

    // Create loop blocks
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(),
        "backward_loop_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(),
        "backward_loop_body", current_func);
    llvm::BasicBlock* propagate_block = llvm::BasicBlock::Create(ctx_.context(),
        "backward_propagate", current_func);
    llvm::BasicBlock* loop_exit = llvm::BasicBlock::Create(ctx_.context(),
        "backward_loop_exit", current_func);

    ctx_.builder().CreateBr(loop_cond);

    // Loop condition: while (counter > 0)
    ctx_.builder().SetInsertPoint(loop_cond);
    llvm::Value* counter_val = ctx_.builder().CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* counter_gt_zero = ctx_.builder().CreateICmpUGT(counter_val,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    ctx_.builder().CreateCondBr(counter_gt_zero, loop_body, loop_exit);

    // Loop body: process node at index (counter - 1)
    ctx_.builder().SetInsertPoint(loop_body);
    llvm::Value* counter_minus_1 = ctx_.builder().CreateSub(counter_val,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(counter_minus_1, counter);

    // Get node at index
    llvm::Function* get_node_func = mem_.getArenaTapeGetNode();
    llvm::Value* node_ptr = ctx_.builder().CreateCall(get_node_func,
        {tape, counter_minus_1});

    // Propagate gradient for this node
    ctx_.builder().SetInsertPoint(propagate_block);
    propagateGradient(node_ptr);
    ctx_.builder().CreateBr(loop_cond);  // Continue to next iteration

    ctx_.builder().SetInsertPoint(loop_exit);
    ctx_.builder().CreateBr(backward_skip);

    ctx_.builder().SetInsertPoint(backward_skip);
}
```

#### Gradient Propagation Rules

```cpp
void AutodiffCodegen::propagateGradient(llvm::Value* node_ptr) {
    llvm::StructType* ad_node_type = ctx_.adNodeType();
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();

    // Load node type and gradient
    llvm::Value* type_ptr = ctx_.builder().CreateStructGEP(ad_node_type, node_ptr, 0);
    llvm::Value* node_type = ctx_.builder().CreateLoad(ctx_.int32Type(), type_ptr);
    llvm::Value* node_grad = loadNodeGradient(node_ptr);

    // Load input pointers
    llvm::Value* input1 = loadNodeInput1(node_ptr);
    llvm::Value* input2 = loadNodeInput2(node_ptr);

    // Create blocks for each operation type
    llvm::BasicBlock* add_block = llvm::BasicBlock::Create(ctx_.context(), "grad_add", current_func);
    llvm::BasicBlock* sub_block = llvm::BasicBlock::Create(ctx_.context(), "grad_sub", current_func);
    llvm::BasicBlock* mul_block = llvm::BasicBlock::Create(ctx_.context(), "grad_mul", current_func);
    llvm::BasicBlock* div_block = llvm::BasicBlock::Create(ctx_.context(), "grad_div", current_func);
    llvm::BasicBlock* sin_block = llvm::BasicBlock::Create(ctx_.context(), "grad_sin", current_func);
    llvm::BasicBlock* cos_block = llvm::BasicBlock::Create(ctx_.context(), "grad_cos", current_func);
    llvm::BasicBlock* done_block = llvm::BasicBlock::Create(ctx_.context(), "grad_done", current_func);

    // Switch on node type
    // ADD (type=2): gradient flows equally to both inputs
    llvm::Value* is_add = ctx_.builder().CreateICmpEQ(node_type,
        llvm::ConstantInt::get(ctx_.int32Type(), 2));
    llvm::BasicBlock* check_sub = llvm::BasicBlock::Create(ctx_.context(), "check_sub", current_func);
    ctx_.builder().CreateCondBr(is_add, add_block, check_sub);

    // ADD: dL/dx = dL/dz * 1, dL/dy = dL/dz * 1
    ctx_.builder().SetInsertPoint(add_block);
    if (input1) accumulateGradient(input1, node_grad);
    if (input2) accumulateGradient(input2, node_grad);
    ctx_.builder().CreateBr(done_block);

    // SUB (type=3): dL/dx = dL/dz, dL/dy = -dL/dz
    ctx_.builder().SetInsertPoint(check_sub);
    llvm::Value* is_sub = ctx_.builder().CreateICmpEQ(node_type,
        llvm::ConstantInt::get(ctx_.int32Type(), 3));
    llvm::BasicBlock* check_mul = llvm::BasicBlock::Create(ctx_.context(), "check_mul", current_func);
    ctx_.builder().CreateCondBr(is_sub, sub_block, check_mul);

    ctx_.builder().SetInsertPoint(sub_block);
    if (input1) accumulateGradient(input1, node_grad);
    if (input2) {
        llvm::Value* neg_grad = ctx_.builder().CreateFNeg(node_grad);
        accumulateGradient(input2, neg_grad);
    }
    ctx_.builder().CreateBr(done_block);

    // MUL (type=4): dL/dx = dL/dz * y, dL/dy = dL/dz * x
    ctx_.builder().SetInsertPoint(check_mul);
    llvm::Value* is_mul = ctx_.builder().CreateICmpEQ(node_type,
        llvm::ConstantInt::get(ctx_.int32Type(), 4));
    llvm::BasicBlock* check_div = llvm::BasicBlock::Create(ctx_.context(), "check_div", current_func);
    ctx_.builder().CreateCondBr(is_mul, mul_block, check_div);

    ctx_.builder().SetInsertPoint(mul_block);
    if (input1 && input2) {
        llvm::Value* input1_val = loadNodeValue(input1);
        llvm::Value* input2_val = loadNodeValue(input2);

        llvm::Value* grad_input1 = ctx_.builder().CreateFMul(node_grad, input2_val);
        llvm::Value* grad_input2 = ctx_.builder().CreateFMul(node_grad, input1_val);

        accumulateGradient(input1, grad_input1);
        accumulateGradient(input2, grad_input2);
    }
    ctx_.builder().CreateBr(done_block);

    // DIV (type=5): dL/dx = dL/dz / y, dL/dy = dL/dz * (-x/y²)
    ctx_.builder().SetInsertPoint(check_div);
    llvm::Value* is_div = ctx_.builder().CreateICmpEQ(node_type,
        llvm::ConstantInt::get(ctx_.int32Type(), 5));
    llvm::BasicBlock* check_sin = llvm::BasicBlock::Create(ctx_.context(), "check_sin", current_func);
    ctx_.builder().CreateCondBr(is_div, div_block, check_sin);

    ctx_.builder().SetInsertPoint(div_block);
    if (input1 && input2) {
        llvm::Value* x = loadNodeValue(input1);
        llvm::Value* y = loadNodeValue(input2);

        // dL/dx = dL/dz / y
        llvm::Value* grad_x = ctx_.builder().CreateFDiv(node_grad, y);
        accumulateGradient(input1, grad_x);

        // dL/dy = dL/dz * (-x / y²)
        llvm::Value* y_sq = ctx_.builder().CreateFMul(y, y);
        llvm::Value* neg_x = ctx_.builder().CreateFNeg(x);
        llvm::Value* neg_x_over_y_sq = ctx_.builder().CreateFDiv(neg_x, y_sq);
        llvm::Value* grad_y = ctx_.builder().CreateFMul(node_grad, neg_x_over_y_sq);
        accumulateGradient(input2, grad_y);
    }
    ctx_.builder().CreateBr(done_block);

    // SIN (type=6): dL/dx = dL/dz * cos(x)
    ctx_.builder().SetInsertPoint(check_sin);
    llvm::Value* is_sin = ctx_.builder().CreateICmpEQ(node_type,
        llvm::ConstantInt::get(ctx_.int32Type(), 6));
    ctx_.builder().CreateCondBr(is_sin, sin_block, check_cos);

    ctx_.builder().SetInsertPoint(sin_block);
    if (input1) {
        llvm::Value* x = loadNodeValue(input1);
        llvm::Function* cos_intrinsic = llvm::Intrinsic::getOrInsertDeclaration(
            &ctx_.module(), llvm::Intrinsic::cos, {ctx_.doubleType()});
        llvm::Value* cos_x = ctx_.builder().CreateCall(cos_intrinsic, {x});
        llvm::Value* grad_x = ctx_.builder().CreateFMul(node_grad, cos_x);
        accumulateGradient(input1, grad_x);
    }
    ctx_.builder().CreateBr(done_block);

    // COS (type=7): dL/dx = dL/dz * (-sin(x))
    ctx_.builder().SetInsertPoint(check_cos);
    llvm::Value* is_cos = ctx_.builder().CreateICmpEQ(node_type,
        llvm::ConstantInt::get(ctx_.int32Type(), 7));
    ctx_.builder().CreateCondBr(is_cos, cos_block, done_block);

    ctx_.builder().SetInsertPoint(cos_block);
    if (input1) {
        llvm::Value* x = loadNodeValue(input1);
        llvm::Function* sin_intrinsic = llvm::Intrinsic::getOrInsertDeclaration(
            &ctx_.module(), llvm::Intrinsic::sin, {ctx_.doubleType()});
        llvm::Value* sin_x = ctx_.builder().CreateCall(sin_intrinsic, {x});
        llvm::Value* neg_sin_x = ctx_.builder().CreateFNeg(sin_x);
        llvm::Value* grad_x = ctx_.builder().CreateFMul(node_grad, neg_sin_x);
        accumulateGradient(input1, grad_x);
    }
    ctx_.builder().CreateBr(done_block);

    ctx_.builder().SetInsertPoint(done_block);
}
```

#### Higher-Order Derivatives (Double Backward)

Eshkol supports computing gradients of gradients for Hessian computation:

```cpp
// From llvm_codegen.cpp - Double backward support in MUL gradient
ctx_.builder().SetInsertPoint(mul_block);
if (input1 && input2) {
    // ... normal gradient computation ...

    // DOUBLE BACKWARD: Track degree when multiplying by variable value
    llvm::GlobalVariable* inner_var_node_ptr = ctx_.innerVarNodePtr();
    llvm::GlobalVariable* gradient_x_degree = ctx_.gradientXDegree();

    if (inner_var_node_ptr && gradient_x_degree) {
        // Load stored variable node for comparison
        llvm::Value* stored_var_node = ctx_.builder().CreateLoad(
            llvm::PointerType::getUnqual(ctx_.context()), inner_var_node_ptr);
        llvm::Value* stored_var_is_valid = ctx_.builder().CreateICmpNE(stored_var_node,
            llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(ctx_.context())));

        // Track how many times we multiply by the variable value
        // This enables correct degree tracking for polynomial expressions
        llvm::Value* current_degree = ctx_.builder().CreateLoad(ctx_.int64Type(), gradient_x_degree);
        llvm::Value* inc = ctx_.builder().CreateSelect(is_var,
            llvm::ConstantInt::get(ctx_.int64Type(), 1),
            llvm::ConstantInt::get(ctx_.int64Type(), 0));
        llvm::Value* new_degree = ctx_.builder().CreateAdd(current_degree, inc);
        ctx_.builder().CreateStore(new_degree, gradient_x_degree);
    }
}
```

#### Gradient Operator Implementation

The `(gradient f point)` form computes ∇f at a given point:

```cpp
Value* LLVMCodegen::codegenGradient(const eshkol_operations_t* op) {
    // Higher-order form: (gradient f) returns a closure
    if (!op->gradient_op.point) {
        return codegenGradientHigherOrder(op);
    }

    // Resolve the function to differentiate
    Value* func = resolveLambdaFunction(op->gradient_op.function);

    // Get the evaluation point
    Value* point_val = codegenAST(op->gradient_op.point);

    // For each dimension i:
    //   1. Create input vector with dual numbers: x_i has tangent=1, others have tangent=0
    //   2. Call function with dual input
    //   3. Extract tangent from result = ∂f/∂x_i
    //   4. Store in gradient vector

    // Allocate result tensor
    Value* result_ptr = builder->CreateCall(malloc_func, {result_size});

    // Outer loop: for each dimension i
    for (i = 0; i < n; i++) {
        // Inner loop: create dual vector
        for (j = 0; j < n; j++) {
            Value* primal = load_input(point_val, j);
            Value* tangent = (i == j) ? 1.0 : 0.0;
            Value* dual = packDualNumber(primal, tangent);
            store_dual(dual_vec, j, dual);
        }

        // Call function with dual input
        Value* result = call_function(func, dual_vec);

        // Extract derivative (tangent component)
        Value* deriv = getDualTangent(result);
        store_gradient(result_ptr, i, deriv);
    }

    return packTensorPtr(result_ptr);
}
```

#### Jacobian Matrix Computation

```cpp
Value* LLVMCodegen::codegenJacobian(const eshkol_operations_t* op) {
    // For function f: R^n → R^m, compute m×n Jacobian matrix
    // J[i,j] = ∂f_i/∂x_j

    // Get input dimension n and output dimension m
    // For each output dimension i:
    //   For each input dimension j:
    //     J[i,j] = gradient of f_i with respect to x_j

    // This uses reverse-mode AD for efficiency when m < n
    // For each output component:
    //   - Set output gradient seed to 1 for component i, 0 elsewhere
    //   - Run backward pass
    //   - Extract gradients for all input variables
}
```

### 2.4 Memory Management

Eshkol uses arena-based allocation for efficient memory management:

```cpp
// From arena_memory.cpp

struct arena_block {
    uint8_t* data;           // Block memory
    size_t size;             // Block size
    size_t used;             // Bytes used
    struct arena_block* next; // Next block
};

struct arena {
    arena_block* current;     // Current allocation block
    arena_block* first;       // First block in chain
    size_t default_block_size; // Size for new blocks
    size_t total_allocated;   // Total bytes allocated
    size_t scope_stack[MAX_SCOPES]; // Scope markers for push/pop
    size_t scope_depth;
};

// Arena allocation - O(1) for most allocations
void* arena_allocate(arena* a, size_t size) {
    // Align size to 8 bytes
    size = (size + 7) & ~7;

    // Check if current block has space
    if (a->current->used + size <= a->current->size) {
        void* ptr = a->current->data + a->current->used;
        a->current->used += size;
        return ptr;
    }

    // Allocate new block
    size_t block_size = (size > a->default_block_size) ? size : a->default_block_size;
    arena_block* new_block = malloc(sizeof(arena_block));
    new_block->data = malloc(block_size);
    new_block->size = block_size;
    new_block->used = size;
    new_block->next = NULL;

    a->current->next = new_block;
    a->current = new_block;

    return new_block->data;
}

// Scope-based cleanup
void arena_push_scope(arena* a) {
    a->scope_stack[a->scope_depth++] = a->current->used;
}

void arena_pop_scope(arena* a) {
    size_t saved = a->scope_stack[--a->scope_depth];
    // Reset to saved position (doesn't free memory, just allows reuse)
    a->current->used = saved;
}
```

### 2.5 LLVM Code Generation

The main codegen file (`llvm_codegen.cpp`) is 24,068 lines implementing:

- AST traversal and code generation
- All special forms (define, lambda, let, if, cond, etc.)
- Arithmetic and comparison operators
- List operations (cons, car, cdr, map, filter, fold)
- Tensor operations
- Autodiff operators
- Closure capture and invocation
- Tagged value packing/unpacking

---

## 3. qLLM Architecture Deep Dive

### 3.1 Tensor System

The qLLM tensor system (`src/core/tensor.c`, 2643 lines) provides N-dimensional array support:

```c
// Maximum dimensions supported
#define QLLM_MAX_DIMS 16

// Data types
typedef enum {
    QLLM_DTYPE_FLOAT32,   // 4 bytes
    QLLM_DTYPE_FLOAT16,   // 2 bytes
    QLLM_DTYPE_INT32,     // 4 bytes
    QLLM_DTYPE_INT16,     // 2 bytes
    QLLM_DTYPE_INT8,      // 1 byte
    QLLM_DTYPE_UINT8,     // 1 byte
    QLLM_DTYPE_BOOL,      // 1 byte
    QLLM_DTYPE_COMPLEX64, // 8 bytes (2x float32)
    QLLM_DTYPE_BF16,      // 2 bytes (bfloat16)
    QLLM_DTYPE_COUNT
} qllm_dtype_t;

// Memory strategies
typedef enum {
    QLLM_MEMORY_DEFAULT,  // Standard aligned allocation
    QLLM_MEMORY_POOL,     // Fixed-size memory pool
    QLLM_MEMORY_ARENA,    // Arena allocation
    QLLM_MEMORY_ALIGNED   // Explicitly aligned
} qllm_memory_strategy_t;

// Tensor structure
struct qllm_tensor {
    size_t dims;                   // Number of dimensions
    size_t* shape;                 // Array of dimension sizes
    size_t* strides;               // Array of strides for indexing
    size_t size;                   // Total number of elements
    qllm_tensor_options_t options; // dtype, device, memory strategy
    qllm_tensor_type_t type;       // DENSE, SPARSE, etc.
    void* data;                    // Pointer to data buffer
    bool owns_data;                // Whether tensor owns its data
    bool is_view;                  // Whether tensor is a view
};

// Stride calculation for row-major layout
static void calculate_strides(size_t dims, const size_t* shape, size_t* strides) {
    if (dims == 0) return;

    strides[dims - 1] = 1;
    for (size_t i = dims - 1; i > 0; i--) {
        strides[i - 1] = strides[i] * shape[i];
    }
}
```

#### Float16 Conversion (IEEE 754 Half-Precision)

```c
// Float16 to Float32
static float fp16_to_fp32(uint16_t half) {
    uint32_t sign = (half >> 15) & 0x1;
    uint32_t exp = (half >> 10) & 0x1f;
    uint32_t mant = half & 0x3ff;

    if (exp == 0) {
        if (mant == 0) {
            return (sign) ? -0.0f : 0.0f;  // Zero
        } else {
            // Denormalized
            float val = (float)mant / 1024.0f / 16384.0f;
            return (sign) ? -val : val;
        }
    } else if (exp == 31) {
        return (mant == 0) ? ((sign) ? -INFINITY : INFINITY) : NAN;
    } else {
        // Normalized: convert exponent
        exp = exp - 15 + 127;
        uint32_t result = (sign << 31) | (exp << 23) | (mant << 13);
        return *(float*)&result;
    }
}

// Float32 to Float16
static uint16_t fp32_to_fp16(float value) {
    uint32_t bits = *(uint32_t*)&value;
    uint32_t sign = (bits >> 31) & 0x1;
    uint32_t exp = (bits >> 23) & 0xff;
    uint32_t mant = bits & 0x7fffff;

    if (exp == 0) {
        return (uint16_t)(sign << 15);  // Zero/denorm
    } else if (exp == 255) {
        return (mant == 0) ?
            (uint16_t)((sign << 15) | 0x7c00) :  // Infinity
            (uint16_t)((sign << 15) | 0x7c00 | (mant >> 13));  // NaN
    } else {
        int32_t new_exp = (int32_t)exp - 127 + 15;
        if (new_exp <= 0) return (uint16_t)(sign << 15);  // Underflow
        if (new_exp >= 31) return (uint16_t)((sign << 15) | 0x7c00);  // Overflow
        return (uint16_t)((sign << 15) | (new_exp << 10) | (mant >> 13));
    }
}
```

### 3.2 Transformer Architecture

#### Transformer Block Structure

```c
struct qllm_transformer_block {
    qllm_transformer_type_t type;        // STANDARD, GEOMETRIC, SPARSE
    size_t dim;                          // Model dimension (e.g., 512)
    size_t hidden_dim;                   // FFN hidden dimension (e.g., 2048)
    size_t num_heads;                    // Number of attention heads (e.g., 8)
    qllm_attention_scoring_t scoring;    // DOT_PRODUCT, ADDITIVE, GEODESIC
    qllm_activation_type_t activation;   // GELU, RELU, SWISH
    qllm_feedforward_type_t ff_type;     // STANDARD, GATED, KAN
    float dropout_rate;                  // Dropout probability
    bool causal;                         // Causal masking for autoregressive
    bool pre_norm;                       // Pre-norm vs post-norm

    qllm_attention_t* attention;         // Attention mechanism
    qllm_feedforward_t* feedforward;     // Feed-forward network

    // Layer normalization parameters
    qllm_tensor_t* attn_norm_gamma;      // Scale (initialized to 1)
    qllm_tensor_t* attn_norm_beta;       // Shift (initialized to 0)
    qllm_tensor_t* ff_norm_gamma;
    qllm_tensor_t* ff_norm_beta;

    qllm_manifold_t* manifold;           // For geometric attention
};
```

#### Transformer Model Structure

```c
struct qllm_transformer_model {
    size_t dim;                          // Model dimension
    size_t hidden_dim;                   // FFN hidden dimension
    size_t num_heads;                    // Attention heads
    size_t num_layers;                   // Number of transformer layers
    size_t vocab_size;                   // Vocabulary size
    size_t max_seq_len;                  // Maximum sequence length

    qllm_transformer_block_t** blocks;   // Array of transformer blocks

    // Embedding layers
    qllm_tensor_t* embedding_weights;    // [vocab_size, dim]
    qllm_tensor_t* output_weights;       // [dim, vocab_size] (tied or separate)

    // Final layer norm
    qllm_tensor_t* final_norm_gamma;
    qllm_tensor_t* final_norm_beta;

    qllm_manifold_t* manifold;
};
```

#### Layer Normalization Implementation

```c
static qllm_tensor_t* apply_layernorm(const qllm_tensor_t* input,
                                      const qllm_tensor_t* gamma,
                                      const qllm_tensor_t* beta) {
    qllm_tensor_t* result = qllm_tensor_clone(input);

    size_t input_dims;
    size_t* input_shape;
    qllm_tensor_get_shape(input, &input_dims, &input_shape);

    float* input_data = (float*)qllm_tensor_get_data_const(input);
    float* result_data = (float*)qllm_tensor_get_data(result);
    float* gamma_data = (float*)qllm_tensor_get_data_const(gamma);
    float* beta_data = (float*)qllm_tensor_get_data_const(beta);

    const float eps = 1e-5f;
    size_t last_dim = input_shape[input_dims - 1];
    size_t batch_size = qllm_tensor_get_size(input) / last_dim;

    for (size_t b = 0; b < batch_size; b++) {
        // Compute mean
        float mean = 0.0f;
        for (size_t i = 0; i < last_dim; i++) {
            mean += input_data[b * last_dim + i];
        }
        mean /= (float)last_dim;

        // Compute variance
        float var = 0.0f;
        for (size_t i = 0; i < last_dim; i++) {
            float diff = input_data[b * last_dim + i] - mean;
            var += diff * diff;
        }
        var /= (float)last_dim;

        // Normalize, scale, and shift
        float std = sqrtf(var + eps);
        for (size_t i = 0; i < last_dim; i++) {
            float norm = (input_data[b * last_dim + i] - mean) / std;
            result_data[b * last_dim + i] = gamma_data[i] * norm + beta_data[i];
        }
    }

    free(input_shape);
    return result;
}
```

### 3.3 Geometric Operations

#### Attention Structure

```c
struct qllm_attention {
    qllm_attention_type_t type;         // SINGLE_HEAD, MULTI_HEAD, CAUSAL
    qllm_attention_scoring_t scoring;   // DOT_PRODUCT, ADDITIVE, GEODESIC
    size_t dim;                         // Attention dimension
    size_t num_heads;                   // Number of heads
    bool use_positional_encoding;
    bool causal;
    float dropout_rate;
    qllm_device_t device;
    qllm_manifold_t* manifold;          // For geodesic attention

    // Projection matrices
    qllm_tensor_t* query_weights;       // [input_dim, dim]
    qllm_tensor_t* key_weights;         // [input_dim, dim]
    qllm_tensor_t* value_weights;       // [input_dim, dim]
    qllm_tensor_t* output_weights;      // [dim, output_dim]
};
```

#### Dot-Product Attention Scores

```c
static qllm_tensor_t* compute_dot_product_scores(const qllm_attention_t* attention,
                                                 const qllm_tensor_t* queries,
                                                 const qllm_tensor_t* keys) {
    // Input shapes: [batch, seq_len, dim]

    // Transpose keys: [batch, dim, seq_len]
    qllm_tensor_t* keys_transposed = qllm_tensor_transpose(keys, 1, 2);

    // Compute Q @ K^T: [batch, seq_q, seq_k]
    qllm_tensor_t* scores = qllm_tensor_matmul(queries, keys_transposed);
    qllm_tensor_destroy(keys_transposed);

    // Scale by 1/sqrt(d_k)
    float scale = 1.0f / sqrtf((float)attention->dim);
    qllm_tensor_t* scaled_scores = qllm_tensor_mul_scalar(scores, scale);
    qllm_tensor_destroy(scores);

    return scaled_scores;
}
```

#### Geodesic (Hyperbolic) Attention

```c
static float stable_hyperbolic_distance(const float* q_data, const float* k_data,
                                        size_t dim, float curvature) {
    // Compute norms with numerical stability
    float q_norm_sq = 0.0f, k_norm_sq = 0.0f, diff_norm_sq = 0.0f;

    for (size_t i = 0; i < dim; i++) {
        float diff = q_data[i] - k_data[i];
        q_norm_sq += q_data[i] * q_data[i];
        k_norm_sq += k_data[i] * k_data[i];
        diff_norm_sq += diff * diff;
    }

    // Safety margins for numerical stability
    float safety_margin = 1e-6f + 1e-8f * sqrtf((float)dim);
    float max_norm_sq = (1.0f - safety_margin) * (1.0f - safety_margin);

    q_norm_sq = fminf(q_norm_sq, max_norm_sq);
    k_norm_sq = fminf(k_norm_sq, max_norm_sq);

    // Poincaré ball distance formula
    float denom = (1.0f - q_norm_sq) * (1.0f - k_norm_sq);
    denom = fmaxf(denom, safety_margin);

    // Dimension-aware scaling
    float scale = 1.0f / (1.0f + 0.001f * logf((float)dim));

    // Argument for acosh
    float arg = 1.0f + 2.0f * diff_norm_sq / denom * scale;

    // Distance: d(q, k) = 2 * acosh(arg) / sqrt(-K)
    float distance = 2.0f * stable_acosh(arg) / sqrtf(-curvature);

    // Return negative for softmax (closer = higher attention)
    return -distance;
}
```

#### Hyperbolic Manifold Operations

```c
// Point projection onto Poincaré ball
qllm_manifold_point_t* hyperbolic_project(const qllm_manifold_t* manifold,
                                          const qllm_tensor_t* tensor) {
    size_t dim = manifold->dim;
    float norm_squared = compute_norm_squared(tensor, dim);

    hyperbolic_data_t* data = get_hyperbolic_data(manifold);

    // Dimension-dependent safety margin
    float safety_margin = 0.01f + 0.005f * sqrtf((float)dim);
    if (dim >= 100) safety_margin += 0.02f;
    if (dim >= 1000) safety_margin += 0.05f;

    float target_radius = data->ball_radius * (1.0f - safety_margin);

    float norm = sqrtf(norm_squared);
    if (norm < target_radius) {
        return hyperbolic_point_from_tensor(manifold, tensor);
    }

    // Scale to boundary
    float scale_factor = target_radius / norm;
    qllm_tensor_t* scaled = qllm_tensor_mul_scalar(tensor, scale_factor);
    qllm_manifold_point_t* point = hyperbolic_point_from_tensor(manifold, scaled);
    qllm_tensor_destroy(scaled);

    return point;
}

// Geodesic distance in Poincaré ball
float hyperbolic_distance(const qllm_manifold_point_t* p1,
                          const qllm_manifold_point_t* p2) {
    // d(x, y) = 2 * R * acosh(1 + 2 * ||x - y||² / ((1 - ||x||²)(1 - ||y||²)))

    struct hyperbolic_point* hp1 = as_hyperbolic_point(p1);
    struct hyperbolic_point* hp2 = as_hyperbolic_point(p2);

    qllm_tensor_t* diff = qllm_tensor_sub(p1->coords, p2->coords);
    float diff_norm_sq = compute_norm_squared(diff, manifold->dim);
    qllm_tensor_destroy(diff);

    float x_term = 1.0f - hp1->norm_squared;
    float y_term = 1.0f - hp2->norm_squared;
    float denom = fmaxf(x_term * y_term, threshold);

    float arg = 1.0f + 2.0f * diff_norm_sq / denom;

    // Dimension-specific dampening
    if (dim >= 100) {
        float dampening = 1.0f / (1.0f + 0.001f * logf((float)dim));
        arg = 1.0f + (arg - 1.0f) * dampening;
    }

    return 2.0f * radius * acoshf(fmaxf(arg, 1.0f));
}
```

### 3.4 Riemannian Optimization

#### Riemannian Adam Implementation

```c
bool riemannian_adam_step(qllm_riemannian_optimizer_t* opt,
                          qllm_tensor_t* parameters,
                          const qllm_tensor_t* gradients) {

    // 1. Project Euclidean gradient to Riemannian gradient
    qllm_tensor_t* riem_grad = project_gradient_to_tangent(
        gradients, parameters, opt->manifold);

    // 2. Initialize moments on first step
    if (!opt->first_moment || !opt->second_moment) {
        size_t param_ndim;
        size_t* param_shape;
        qllm_tensor_get_shape(parameters, &param_ndim, &param_shape);

        opt->first_moment = qllm_tensor_zeros(param_ndim, param_shape, &options);
        opt->second_moment = qllm_tensor_zeros(param_ndim, param_shape, &options);
        free(param_shape);
    }

    // 3. Update first moment: m_t = β₁·m_{t-1} + (1-β₁)·grad
    qllm_tensor_t* beta1_m = qllm_tensor_mul_scalar(opt->first_moment, opt->beta1);
    qllm_tensor_t* one_beta1_grad = qllm_tensor_mul_scalar(riem_grad, 1.0f - opt->beta1);
    qllm_tensor_t* new_first_moment = qllm_tensor_add(beta1_m, one_beta1_grad);

    // 4. Update second moment: v_t = β₂·v_{t-1} + (1-β₂)·grad²
    qllm_tensor_t* grad_squared = qllm_tensor_mul(riem_grad, riem_grad);
    qllm_tensor_t* beta2_v = qllm_tensor_mul_scalar(opt->second_moment, opt->beta2);
    qllm_tensor_t* one_beta2_grad_sq = qllm_tensor_mul_scalar(grad_squared, 1.0f - opt->beta2);
    qllm_tensor_t* new_second_moment = qllm_tensor_add(beta2_v, one_beta2_grad_sq);

    // 5. Bias correction
    opt->timestep++;
    float bc1 = 1.0f - powf(opt->beta1, (float)opt->timestep);
    float bc2 = 1.0f - powf(opt->beta2, (float)opt->timestep);
    qllm_tensor_t* m_hat = qllm_tensor_div_scalar(new_first_moment, bc1);
    qllm_tensor_t* v_hat = qllm_tensor_div_scalar(new_second_moment, bc2);

    // 6. Compute update: -η · m̂ / (√v̂ + ε)
    qllm_tensor_t* v_sqrt = qllm_tensor_sqrt(v_hat);
    qllm_tensor_t* v_sqrt_eps = qllm_tensor_add_scalar(v_sqrt, opt->epsilon);
    qllm_tensor_t* update_unnorm = qllm_tensor_div(m_hat, v_sqrt_eps);
    qllm_tensor_t* update = qllm_tensor_mul_scalar(update_unnorm, -opt->learning_rate);

    // 7. Apply update via exponential map
    qllm_manifold_point_t* current = qllm_manifold_point_from_tensor(opt->manifold, parameters);
    qllm_manifold_tangent_t* tangent = qllm_manifold_tangent_from_tensor(current, update);
    qllm_manifold_point_t* new_point = qllm_manifold_exp_map(tangent);

    // 8. Copy new point to parameters
    qllm_tensor_t* new_params = qllm_manifold_point_to_tensor(new_point);
    memcpy(qllm_tensor_get_data(parameters),
           qllm_tensor_get_data(new_params),
           qllm_tensor_get_size(parameters) * sizeof(float));

    // Cleanup...
    return true;
}
```

#### Gradient Projection to Tangent Space

```c
static qllm_tensor_t* project_gradient_to_tangent(
    const qllm_tensor_t* euclidean_grad,
    const qllm_tensor_t* point,
    const qllm_manifold_t* manifold) {

    qllm_manifold_type_t type = qllm_manifold_get_type(manifold);

    switch (type) {
        case QLLM_MANIFOLD_HYPERBOLIC: {
            // Scale by conformal factor squared: λ² = (2 / (1 - ||x||²))²
            qllm_tensor_t* x_squared = qllm_tensor_mul(point, point);
            qllm_tensor_t* x_norm_sq = qllm_tensor_sum(x_squared, -1, true);
            float norm_sq = *(float*)qllm_tensor_get_data(x_norm_sq);

            float denom = fmaxf(1.0f - norm_sq, 1e-7f);
            float lambda = 2.0f / denom;
            float scale = fminf(lambda * lambda, 100.0f);  // Numerical stability

            return qllm_tensor_mul_scalar(euclidean_grad, scale);
        }

        case QLLM_MANIFOLD_SPHERICAL: {
            // Project to tangent space: grad - (grad · x) * x
            qllm_tensor_t* dot = qllm_tensor_mul(euclidean_grad, point);
            qllm_tensor_t* dot_sum = qllm_tensor_sum(dot, -1, true);
            float dot_product = *(float*)qllm_tensor_get_data(dot_sum);

            qllm_tensor_t* projection = qllm_tensor_mul_scalar(point, dot_product);
            qllm_tensor_t* riem_grad = qllm_tensor_sub(euclidean_grad, projection);

            return riem_grad;
        }

        case QLLM_MANIFOLD_EUCLIDEAN:
        default:
            return qllm_tensor_clone(euclidean_grad);
    }
}
```

---

## 4. Compatibility Analysis

### 4.1 Architectural Alignment

Both systems share fundamental architectural patterns:

| Pattern | Eshkol | qLLM | Alignment |
|---------|--------|------|-----------|
| Tape-based reverse AD | `ad_tape_t` with linked nodes | Needs `Variable` with backward_fn | **Conceptually identical** |
| Forward pass recording | Build graph during evaluation | Need to track operations | **Same approach** |
| Backward traversal | Iterate tape in reverse | Need same traversal | **Identical** |
| Gradient accumulation | `accumulateGradient()` | Need += semantics | **Same** |
| Arena memory | `arena_allocate()` | `qllm_memory_arena_alloc()` | **Both implemented** |

### 4.2 Type System Mapping

| Eshkol Type | qLLM Equivalent | Mapping Strategy |
|-------------|-----------------|------------------|
| `double` (64-bit) | `float` (32-bit) | Type narrowing, precision considerations |
| `eshkol_tagged_value_t` | `qllm_tensor_t*` | Wrap tensor pointer as tagged value |
| `ad_node_t` | `Variable<tensor>` | Extend node to hold tensor |
| `ad_tape_t` | Implicit tape | Make tape explicit in wrapper |
| `tensor_t` (1D, double) | `qllm_tensor_t` (N-D, float32) | Shape abstraction layer |

### 4.3 AD System Comparison

#### Forward-Mode

| Operation | Eshkol Implementation | qLLM Need |
|-----------|----------------------|-----------|
| Dual number | `{value, deriv}` struct | Same for scalar verification |
| Arithmetic | All rules implemented | Same rules |
| Transcendentals | sin, cos, exp, log, pow | Same functions |
| Use case | Gradient verification | Numerical gradient checking |

#### Reverse-Mode

| Operation | Eshkol Implementation | qLLM Need |
|-----------|----------------------|-----------|
| ADD backward | `dL/dx = dL/dz, dL/dy = dL/dz` | Same |
| SUB backward | `dL/dx = dL/dz, dL/dy = -dL/dz` | Same |
| MUL backward | `dL/dx = dL/dz * y, dL/dy = dL/dz * x` | Same, but batched |
| DIV backward | `dL/dx = dL/dz / y, dL/dy = -dL/dz * x / y²` | Same |
| **MATMUL backward** | N/A | `dL/dA = dL/dC @ B^T, dL/dB = A^T @ dL/dC` |
| **SOFTMAX backward** | N/A | `dL/dx = softmax(x) * (dL/dy - sum(dL/dy * softmax(x)))` |
| **LAYERNORM backward** | N/A | Mean/variance gradient flow |

### 4.4 Gap Analysis

#### Critical Gaps

1. **Tensor-Level Operations**
   - Eshkol AD operates on scalar `double` values
   - qLLM needs batched operations on `[batch, seq, dim]` tensors
   - Solution: Extend `ad_node_t` to hold `qllm_tensor_t*`

2. **MatMul Backward**
   ```
   Forward: C = A @ B
   Backward:
     dL/dA = dL/dC @ B^T
     dL/dB = A^T @ dL/dC
   ```
   - Not in Eshkol (no matrix multiply primitive)
   - Critical for attention and feedforward layers

3. **Softmax Backward**
   ```
   Forward: y = exp(x) / sum(exp(x))
   Backward: Complex Jacobian-vector product
     J[i,j] = y[i] * (δ[i,j] - y[j])
     dL/dx = y * (dL/dy - sum(dL/dy * y))
   ```
   - Eshkol has no built-in softmax
   - Could be implemented but needs tensor support

4. **Attention Backward**
   - Combines matmul, softmax, and scaling
   - Must propagate gradients through QKV projections
   - Causal masking complicates gradient flow

5. **FFI Bridge**
   - No mechanism to call C functions from Eshkol
   - Would need to extend LLVM codegen to emit C function calls
   - Or implement qLLM tensor ops as Eshkol primitives

#### Non-Critical Gaps

1. **Float32 vs Float64**
   - Eshkol uses double, qLLM uses float
   - Can adapt with type conversion
   - May affect numerical stability

2. **Shape Handling**
   - Eshkol tensors are 1D with size
   - qLLM tensors have N-D shape and strides
   - Requires shape abstraction layer

---

## 5. Integration Pathways

### 5.1 Path A: Eshkol as Training DSL

#### Overview

Use Eshkol as a high-level language for expressing training logic, with FFI calls to qLLM's C tensor library for actual computation.

#### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Eshkol Training DSL                       │
│  (define (train-step model input target)                    │
│    (let* ((output (qllm-forward model input))               │
│           (loss (cross-entropy output target))              │
│           (grads (gradient loss model-params)))             │
│      (qllm-optimizer-step optimizer model-params grads)))   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      FFI Bridge Layer                        │
│  - Wrap qllm_tensor_t* as ESHKOL_VALUE_TENSOR_PTR          │
│  - Implement qllm-forward, qllm-backward as primitives     │
│  - Bridge arena memory between systems                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     qLLM C Library                           │
│  - qllm_tensor_* functions                                  │
│  - qllm_attention_forward, qllm_feedforward_forward        │
│  - qllm_optimizer_step                                      │
└─────────────────────────────────────────────────────────────┘
```

#### Implementation Steps

1. **Add qLLM tensor type to Eshkol** (1 week)
   ```c
   // In eshkol.h
   #define ESHKOL_VALUE_QLLM_TENSOR_PTR 14
   ```

2. **Implement FFI bridge** (2 weeks)
   ```cpp
   // In llvm_codegen.cpp
   Value* LLVMCodegen::codegenQllmCall(const char* func_name,
                                        std::vector<Value*> args) {
       // Look up qLLM function
       Function* qllm_func = getOrDeclareQllmFunction(func_name);
       // Convert Eshkol values to C values
       std::vector<Value*> c_args = convertArgs(args);
       // Call function
       return builder->CreateCall(qllm_func, c_args);
   }
   ```

3. **Implement tensor AD primitives** (3 weeks)
   ```scheme
   ;; New Eshkol primitives
   (qllm-tensor-matmul A B)      ; Returns tensor, records for backward
   (qllm-tensor-softmax x dim)   ; Softmax with backward support
   (qllm-tensor-layernorm x gamma beta) ; LayerNorm with backward
   ```

4. **Extend reverse-mode AD for tensors** (2 weeks)
   ```cpp
   // New AD node types
   AD_NODE_TENSOR_MATMUL,
   AD_NODE_TENSOR_SOFTMAX,
   AD_NODE_TENSOR_LAYERNORM,

   // Propagate gradients for tensor ops
   void propagateGradientTensor(llvm::Value* node_ptr) {
       switch (node_type) {
           case AD_NODE_TENSOR_MATMUL:
               // dL/dA = dL/dC @ B^T
               grad_A = qllm_tensor_matmul(grad_C, qllm_tensor_transpose(B));
               // dL/dB = A^T @ dL/dC
               grad_B = qllm_tensor_matmul(qllm_tensor_transpose(A), grad_C);
               break;
           // ...
       }
   }
   ```

#### Effort Estimate

| Task | Duration |
|------|----------|
| Add qLLM tensor type | 1 week |
| FFI bridge layer | 2 weeks |
| Tensor AD primitives | 3 weeks |
| Extend reverse-mode AD | 2 weeks |
| Testing and integration | 2 weeks |
| **Total** | **10 weeks** |

### 5.2 Path B: Port AD to C++

#### Overview

Translate Eshkol's AD architecture to C++ classes that wrap `qllm_tensor_t*`.

#### Design

```cpp
// Variable class wrapping qllm tensor
class Variable {
public:
    qllm_tensor_t* data;
    qllm_tensor_t* grad;
    std::function<void()> backward_fn;
    std::vector<Variable*> inputs;
    bool requires_grad;

    Variable(qllm_tensor_t* tensor, bool requires_grad = false);
    ~Variable();

    // Operations that build computation graph
    static Variable* matmul(Variable* a, Variable* b);
    static Variable* add(Variable* a, Variable* b);
    static Variable* softmax(Variable* x, int dim);
    static Variable* layernorm(Variable* x, Variable* gamma, Variable* beta);

    // Backward pass
    void backward();
};

// Global tape (thread-local for parallelism)
thread_local std::vector<Variable*> g_tape;

// MatMul implementation
Variable* Variable::matmul(Variable* a, Variable* b) {
    // Forward pass
    qllm_tensor_t* result = qllm_tensor_matmul(a->data, b->data);
    Variable* c = new Variable(result, a->requires_grad || b->requires_grad);

    if (c->requires_grad) {
        c->inputs = {a, b};
        c->backward_fn = [a, b, c]() {
            if (a->requires_grad && a->grad) {
                // dL/dA = dL/dC @ B^T
                qllm_tensor_t* b_t = qllm_tensor_transpose(b->data, -1, -2);
                qllm_tensor_t* grad_a = qllm_tensor_matmul(c->grad, b_t);
                qllm_tensor_destroy(b_t);
                // Accumulate
                if (a->grad) {
                    qllm_tensor_t* new_grad = qllm_tensor_add(a->grad, grad_a);
                    qllm_tensor_destroy(a->grad);
                    qllm_tensor_destroy(grad_a);
                    a->grad = new_grad;
                } else {
                    a->grad = grad_a;
                }
            }
            if (b->requires_grad && b->grad) {
                // dL/dB = A^T @ dL/dC
                qllm_tensor_t* a_t = qllm_tensor_transpose(a->data, -1, -2);
                qllm_tensor_t* grad_b = qllm_tensor_matmul(a_t, c->grad);
                qllm_tensor_destroy(a_t);
                // Accumulate
                if (b->grad) {
                    qllm_tensor_t* new_grad = qllm_tensor_add(b->grad, grad_b);
                    qllm_tensor_destroy(b->grad);
                    qllm_tensor_destroy(grad_b);
                    b->grad = new_grad;
                } else {
                    b->grad = grad_b;
                }
            }
        };
        g_tape.push_back(c);
    }

    return c;
}

// Backward pass (like Eshkol's backpropagate)
void Variable::backward() {
    // Seed gradient = 1
    if (!this->grad) {
        this->grad = qllm_tensor_ones_like(this->data);
    }

    // Traverse tape in reverse
    for (auto it = g_tape.rbegin(); it != g_tape.rend(); ++it) {
        Variable* v = *it;
        if (v->backward_fn) {
            v->backward_fn();
        }
    }
}
```

#### Effort Estimate

| Task | Duration |
|------|----------|
| Variable class core | 1 week |
| Basic ops (add, mul, div) | 1 week |
| MatMul backward | 1 week |
| Softmax backward | 1 week |
| LayerNorm backward | 1 week |
| Attention backward | 1 week |
| Testing framework | 1 week |
| Integration with optimizer | 1 week |
| **Total** | **8 weeks** |

### 5.3 Path C: Hybrid Approach (Recommended)

#### Overview

Use Eshkol for prototyping and verification, C++ for production.

#### Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                Phase 1: Prototype in Eshkol                  │
│  - Implement gradient formulas using existing AD            │
│  - Test against numerical gradients                         │
│  - Verify mathematical correctness                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               Phase 2: Port to C++ Variable                  │
│  - Translate verified formulas to C++                       │
│  - Use Eshkol as test oracle                                │
│  - Benchmark performance                                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               Phase 3: Production Integration                │
│  - Integrate with Riemannian optimizer                      │
│  - Add data loading pipeline                                │
│  - Full training loop                                        │
└─────────────────────────────────────────────────────────────┘
```

#### Verification in Eshkol

```scheme
;; Test matmul backward in Eshkol
(define (test-matmul-backward)
  (let* ((A (tensor 2 3 '(1 2 3 4 5 6)))
         (B (tensor 3 2 '(7 8 9 10 11 12)))
         (dC (tensor 2 2 '(1 1 1 1))))

    ;; Analytical gradient (from formula)
    (define dA-analytical (matmul dC (transpose B)))
    (define dB-analytical (matmul (transpose A) dC))

    ;; Numerical gradient (finite differences)
    (define (loss-fn-A A-flat)
      (tensor-sum (matmul (reshape A-flat '(2 3)) B)))
    (define dA-numerical (numerical-gradient loss-fn-A (flatten A)))

    ;; Compare
    (assert-close dA-analytical dA-numerical 1e-5)))
```

---

## 6. Implementation Specifications

### 6.1 Required Tensor AD Operations

#### MatMul Backward

```cpp
// Forward: C[i,j] = Σ_k A[i,k] * B[k,j]
// Backward:
//   dL/dA[i,k] = Σ_j dL/dC[i,j] * B[k,j] = (dL/dC @ B^T)[i,k]
//   dL/dB[k,j] = Σ_i A[i,k] * dL/dC[i,j] = (A^T @ dL/dC)[k,j]

void matmul_backward(const qllm_tensor_t* grad_output,
                     const qllm_tensor_t* A,
                     const qllm_tensor_t* B,
                     qllm_tensor_t** grad_A,
                     qllm_tensor_t** grad_B) {
    // Batched matmul with broadcasting
    qllm_tensor_t* B_T = qllm_tensor_transpose(B, -1, -2);
    *grad_A = qllm_tensor_matmul(grad_output, B_T);
    qllm_tensor_destroy(B_T);

    qllm_tensor_t* A_T = qllm_tensor_transpose(A, -1, -2);
    *grad_B = qllm_tensor_matmul(A_T, grad_output);
    qllm_tensor_destroy(A_T);
}
```

#### Softmax Backward

```cpp
// Forward: y[i] = exp(x[i]) / Σ_j exp(x[j])
// Backward: dL/dx[i] = y[i] * (dL/dy[i] - Σ_j dL/dy[j] * y[j])

void softmax_backward(const qllm_tensor_t* grad_output,
                      const qllm_tensor_t* softmax_output,
                      int dim,
                      qllm_tensor_t** grad_input) {
    // dot = sum(grad_output * softmax_output, dim, keepdim=True)
    qllm_tensor_t* prod = qllm_tensor_mul(grad_output, softmax_output);
    qllm_tensor_t* dot = qllm_tensor_sum(prod, dim, true);
    qllm_tensor_destroy(prod);

    // grad_input = softmax_output * (grad_output - dot)
    qllm_tensor_t* diff = qllm_tensor_sub(grad_output, dot);
    qllm_tensor_destroy(dot);
    *grad_input = qllm_tensor_mul(softmax_output, diff);
    qllm_tensor_destroy(diff);
}
```

#### LayerNorm Backward

```cpp
// Forward: y = (x - μ) / σ * γ + β
//   where μ = mean(x), σ = sqrt(var(x) + ε)
// Backward: Complex chain rule through mean, variance, and normalization

void layernorm_backward(const qllm_tensor_t* grad_output,
                        const qllm_tensor_t* input,
                        const qllm_tensor_t* gamma,
                        float eps,
                        qllm_tensor_t** grad_input,
                        qllm_tensor_t** grad_gamma,
                        qllm_tensor_t** grad_beta) {
    size_t N = /* batch size */;
    size_t D = /* feature dim */;

    // Recompute forward quantities
    qllm_tensor_t* mean = qllm_tensor_mean(input, -1, true);
    qllm_tensor_t* centered = qllm_tensor_sub(input, mean);
    qllm_tensor_t* var = qllm_tensor_var(input, -1, true);
    qllm_tensor_t* std = qllm_tensor_sqrt(qllm_tensor_add_scalar(var, eps));
    qllm_tensor_t* normalized = qllm_tensor_div(centered, std);

    // grad_gamma = sum(grad_output * normalized, batch_dims)
    qllm_tensor_t* prod_gamma = qllm_tensor_mul(grad_output, normalized);
    *grad_gamma = qllm_tensor_sum(prod_gamma, 0, false);

    // grad_beta = sum(grad_output, batch_dims)
    *grad_beta = qllm_tensor_sum(grad_output, 0, false);

    // grad_input (complex formula)
    // dL/dx = (1/σ) * (dL/dy * γ - mean(dL/dy * γ) - normalized * mean(dL/dy * γ * normalized))
    qllm_tensor_t* dy_gamma = qllm_tensor_mul(grad_output, gamma);
    qllm_tensor_t* mean_dy_gamma = qllm_tensor_mean(dy_gamma, -1, true);
    qllm_tensor_t* dy_gamma_norm = qllm_tensor_mul(dy_gamma, normalized);
    qllm_tensor_t* mean_dy_gamma_norm = qllm_tensor_mean(dy_gamma_norm, -1, true);

    qllm_tensor_t* term1 = qllm_tensor_sub(dy_gamma, mean_dy_gamma);
    qllm_tensor_t* term2 = qllm_tensor_mul(normalized, mean_dy_gamma_norm);
    qllm_tensor_t* diff = qllm_tensor_sub(term1, term2);
    *grad_input = qllm_tensor_div(diff, std);

    // Cleanup...
}
```

#### Attention Backward

```cpp
void attention_backward(const qllm_tensor_t* grad_output,  // [batch, seq, dim]
                        const qllm_tensor_t* Q,             // [batch, seq, dim]
                        const qllm_tensor_t* K,             // [batch, seq, dim]
                        const qllm_tensor_t* V,             // [batch, seq, dim]
                        const qllm_tensor_t* scores,        // [batch, seq, seq]
                        const qllm_tensor_t* attention_weights, // [batch, seq, seq]
                        const qllm_tensor_t* W_Q,           // [dim, dim]
                        const qllm_tensor_t* W_K,           // [dim, dim]
                        const qllm_tensor_t* W_V,           // [dim, dim]
                        const qllm_tensor_t* W_O,           // [dim, dim]
                        AttentionGradients* grads) {

    // 1. Gradient through output projection
    // context = attention_weights @ V
    // output = context @ W_O
    // dL/d(context) = dL/d(output) @ W_O^T
    qllm_tensor_t* W_O_T = qllm_tensor_transpose(W_O, 0, 1);
    qllm_tensor_t* grad_context = qllm_tensor_matmul(grad_output, W_O_T);

    // dL/dW_O = context^T @ dL/d(output)
    // (summed over batch)

    // 2. Gradient through attention @ V
    // context = attention_weights @ V
    // dL/d(attention_weights) = dL/d(context) @ V^T
    qllm_tensor_t* V_T = qllm_tensor_transpose(V, -1, -2);
    qllm_tensor_t* grad_attn_weights = qllm_tensor_matmul(grad_context, V_T);

    // dL/dV = attention_weights^T @ dL/d(context)
    qllm_tensor_t* attn_T = qllm_tensor_transpose(attention_weights, -1, -2);
    qllm_tensor_t* grad_V = qllm_tensor_matmul(attn_T, grad_context);

    // 3. Gradient through softmax
    // attention_weights = softmax(scores)
    qllm_tensor_t* grad_scores;
    softmax_backward(grad_attn_weights, attention_weights, -1, &grad_scores);

    // 4. Gradient through scaling
    // scores = (Q @ K^T) / sqrt(d_k)
    float scale = 1.0f / sqrtf((float)dim);
    qllm_tensor_t* grad_scores_scaled = qllm_tensor_mul_scalar(grad_scores, scale);

    // 5. Gradient through Q @ K^T
    // dL/dQ = dL/d(scores_scaled) @ K
    qllm_tensor_t* grad_Q = qllm_tensor_matmul(grad_scores_scaled, K);

    // dL/dK = dL/d(scores_scaled)^T @ Q
    qllm_tensor_t* grad_scores_T = qllm_tensor_transpose(grad_scores_scaled, -1, -2);
    qllm_tensor_t* grad_K = qllm_tensor_matmul(grad_scores_T, Q);

    // 6. Gradient through projections
    // Q = input @ W_Q, K = input @ W_K, V = input @ W_V
    // dL/d(input) = dL/dQ @ W_Q^T + dL/dK @ W_K^T + dL/dV @ W_V^T

    // Store results...
}
```

### 6.2 FFI Bridge Design

```cpp
// eshkol_qllm_bridge.h
#ifndef ESHKOL_QLLM_BRIDGE_H
#define ESHKOL_QLLM_BRIDGE_H

#include <eshkol/eshkol.h>
#include <semiclassical_qllm/tensor.h>

// Convert Eshkol tensor to qLLM tensor
qllm_tensor_t* eshkol_to_qllm_tensor(eshkol_tagged_value_t value);

// Convert qLLM tensor to Eshkol tagged value
eshkol_tagged_value_t qllm_to_eshkol_tensor(qllm_tensor_t* tensor);

// Wrap qLLM function for Eshkol
typedef eshkol_tagged_value_t (*eshkol_qllm_wrapper_fn)(eshkol_tagged_value_t* args, size_t num_args);

// Register qLLM operations as Eshkol primitives
void eshkol_register_qllm_ops(void);

// Example wrapper
eshkol_tagged_value_t eshkol_qllm_matmul(eshkol_tagged_value_t* args, size_t num_args) {
    qllm_tensor_t* A = eshkol_to_qllm_tensor(args[0]);
    qllm_tensor_t* B = eshkol_to_qllm_tensor(args[1]);
    qllm_tensor_t* C = qllm_tensor_matmul(A, B);
    return qllm_to_eshkol_tensor(C);
}

#endif
```

### 6.3 Gradient Verification Framework

```scheme
;; gradient_verification.esk
;; Framework for testing gradient implementations

(define epsilon 1e-5)
(define tolerance 1e-4)

;; Numerical gradient using central differences
(define (numerical-gradient f x)
  (let ((n (tensor-size x))
        (grad (tensor-zeros-like x)))
    (do ((i 0 (+ i 1)))
        ((>= i n) grad)
      (let ((x-plus (tensor-copy x))
            (x-minus (tensor-copy x)))
        (tensor-set! x-plus i (+ (tensor-ref x i) epsilon))
        (tensor-set! x-minus i (- (tensor-ref x i) epsilon))
        (tensor-set! grad i
          (/ (- (f x-plus) (f x-minus))
             (* 2 epsilon)))))))

;; Compare analytical and numerical gradients
(define (check-gradient f x analytical-grad)
  (let ((numerical-grad (numerical-gradient f x)))
    (let ((diff (tensor-sub analytical-grad numerical-grad))
          (norm-diff (tensor-norm diff))
          (norm-num (tensor-norm numerical-grad)))
      (if (< (/ norm-diff (+ norm-num 1e-8)) tolerance)
          (begin
            (display "PASS: Gradient check passed")
            (newline)
            #t)
          (begin
            (display "FAIL: Gradient check failed")
            (newline)
            (display "  Analytical: ") (display analytical-grad) (newline)
            (display "  Numerical:  ") (display numerical-grad) (newline)
            (display "  Relative error: ") (display (/ norm-diff norm-num)) (newline)
            #f)))))

;; Test suite for matmul backward
(define (test-matmul-backward)
  (display "Testing MatMul Backward...") (newline)

  (let* ((A (tensor-random 2 3))
         (B (tensor-random 3 4))
         (dC (tensor-random 2 4)))

    ;; Test dL/dA
    (define (loss-A A-flat)
      (let ((A-mat (tensor-reshape A-flat '(2 3))))
        (tensor-sum (tensor-mul dC (tensor-matmul A-mat B)))))

    (let ((dA-analytical (tensor-matmul dC (tensor-transpose B))))
      (check-gradient loss-A (tensor-flatten A) (tensor-flatten dA-analytical)))

    ;; Test dL/dB
    (define (loss-B B-flat)
      (let ((B-mat (tensor-reshape B-flat '(3 4))))
        (tensor-sum (tensor-mul dC (tensor-matmul A B-mat)))))

    (let ((dB-analytical (tensor-matmul (tensor-transpose A) dC)))
      (check-gradient loss-B (tensor-flatten B) (tensor-flatten dB-analytical)))))

;; Run all gradient tests
(define (run-gradient-tests)
  (display "=== Gradient Verification Suite ===") (newline)
  (test-matmul-backward)
  (test-softmax-backward)
  (test-layernorm-backward)
  (test-attention-backward)
  (display "=== All tests complete ===") (newline))
```

---

## 7. Conclusions and Recommendations

### 7.1 Summary of Findings

1. **Eshkol's AD system is architecturally sound** for the qLLM training task. The tape-based reverse-mode implementation with backpropagation correctly implements the chain rule for scalar operations.

2. **The type systems are compatible** at a conceptual level. Eshkol's tagged values can wrap qLLM tensor pointers, and the HoTT type system could express tensor shapes with dependent types.

3. **The primary gaps are practical, not conceptual**:
   - Tensor-level operations (matmul, softmax, layernorm, attention)
   - FFI bridge between Eshkol and qLLM C library
   - Float64 → Float32 type adaptation

4. **The mathematical foundations in both systems align**:
   - Both use Riemannian geometry for optimization
   - Both support hyperbolic manifolds
   - Both have arena-based memory management

### 7.2 Recommended Path

**Adopt the Hybrid Approach (Path C)**:

1. **Immediate** (Weeks 1-2): Use Eshkol to prototype and verify gradient formulas
2. **Short-term** (Weeks 3-6): Port verified formulas to C++ Variable class
3. **Medium-term** (Weeks 7-10): Integrate with Riemannian optimizer and training loop
4. **Long-term** (Optional): Develop Eshkol FFI for training DSL

### 7.3 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Numerical precision issues | Medium | Medium | Test with float64 first, then adapt |
| FFI complexity | High | Low | Port to C++ instead of FFI |
| Tensor broadcasting bugs | Medium | Medium | Comprehensive test suite |
| Memory leaks in AD graph | Low | High | Arena-based allocation |

### 7.4 Success Metrics

1. **Gradient accuracy**: All analytical gradients match numerical gradients within 1e-4 relative error
2. **Training convergence**: Loss decreases monotonically on toy problems
3. **Performance**: Forward+backward pass completes in < 100ms for micro model
4. **Memory efficiency**: No memory growth over 1000 training steps

---

## Appendix A: Code Statistics

| Component | Files | Lines of Code |
|-----------|-------|---------------|
| Eshkol C++ | 47 | 47,951 |
| Eshkol .esk stdlib | 12 | 21,162 |
| qLLM src/ | 49 | 51,640 |
| qLLM include/ | 42 | 12,389 |
| qLLM tests/ | 798 | 203,353 |

## Appendix B: Key File References

### Eshkol
- `inc/eshkol/eshkol.h` - Core type definitions
- `lib/types/hott_types.cpp` - HoTT type system
- `lib/backend/llvm_codegen.cpp` - Main code generator
- `lib/backend/autodiff_codegen.cpp` - AD implementation
- `lib/backend/tensor_codegen.cpp` - Tensor operations
- `lib/math.esk` - Math stdlib
- `tests/neural/nn_complete.esk` - Neural network example

### qLLM
- `src/core/tensor.c` - Tensor implementation
- `src/model/transformer.c` - Transformer architecture
- `src/model/attention.c` - Attention mechanism
- `src/geometric/hyperbolic_core.c` - Hyperbolic operations
- `src/optimization/riemannian_adam.c` - Riemannian optimizer

---

*Document generated from source code analysis on December 7, 2025*
