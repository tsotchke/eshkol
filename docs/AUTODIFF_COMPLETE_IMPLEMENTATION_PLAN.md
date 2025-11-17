# Complete Automatic Differentiation Implementation Plan for Eshkol
**Date**: November 17, 2025  
**Status**: APPROVED - Ready for Implementation  
**Estimated Timeline**: 35 sessions  
**Scope**: Build complete autodiff system from scratch

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Architecture Overview](#architecture-overview)
4. [Implementation Phases](#implementation-phases)
5. [Technical Specifications](#technical-specifications)
6. [Testing Strategy](#testing-strategy)
7. [Success Criteria](#success-criteria)
8. [References](#references)

---

## Executive Summary

### What Exists Today

**Working** ✅:
- Basic symbolic `diff` operator ([`llvm_codegen.cpp:4993-5116`](../lib/backend/llvm_codegen.cpp:4993))
- Handles: constants, variables, +, -, *, sin, cos
- Returns: scalar int64 only
- ~100 lines of code

**Missing** ❌:
- `gradient`, `derivative`, `jacobian`, `hessian` functions
- `divergence`, `curl`, `laplacian` operators  
- Dual numbers for forward-mode AD
- Computational graphs for reverse-mode AD
- Vector return types
- Type inference for autodiff
- Lambda function differentiation
- Proper chain rule implementation

### Scope of Work

Build complete automatic differentiation system with:
- Forward-mode AD using dual numbers
- Reverse-mode AD using computational graphs
- Vector calculus operators (gradient, jacobian, divergence, curl, laplacian)
- Type-safe gradients with proper inference
- Integration with existing tensor and tagged value systems
- Performance optimization (SIMD, sparsity, compile-time optimization)

**Timeline**: 35 sessions across 6 phases  
**Complexity**: High - building from scratch  
**Dependencies**: Tagged value system ✅, Tensor operations ✅, Arena memory ✅

---

## Current State Analysis

### Existing Implementation

**Location**: [`lib/backend/llvm_codegen.cpp:4993-5116`](../lib/backend/llvm_codegen.cpp:4993)

**Functions**:
```cpp
Value* codegenDiff(const eshkol_operations_t* op);           // Lines 4994-5011
Value* differentiate(const eshkol_ast_t* expr, const char* var); // Lines 5013-5040
Value* differentiateOperation(const eshkol_operations_t* op, const char* var); // Lines 5042-5116
```

**Supported Operations**:
| Operation | Rule | Status |
|-----------|------|--------|
| `diff c x` | 0 | ✅ Complete |
| `diff x x` | 1 | ✅ Complete |
| `diff y x` | 0 | ✅ Complete |
| `diff (+ f g) x` | f' + g' | ✅ Complete |
| `diff (- f g) x` | f' - g' | ✅ Complete |
| `diff (* f g) x` | f'g + fg' | ⚠️ Partial (only handles x*x specially) |
| `diff (sin f) x` | f' | ⚠️ Wrong (should be cos(f)·f') |
| `diff (cos f) x` | -f' | ⚠️ Wrong (should be -sin(f)·f') |

**Bugs to Fix**:
1. **Type Mismatch** (line 5084): Always creates `i64` constants for double expressions
2. **Incomplete Product Rule** (lines 5076-5086): Only handles `x*x` case
3. **Simplified Trig** (lines 5099-5111): Missing chain rule multiplication

### Documentation vs Reality Gap

**Documented** (but doesn't exist):
- [`docs/aidocs/AUTODIFF.md`](../docs/aidocs/AUTODIFF.md) - Claims full AD system
- [`examples/vector_calculus.esk`](../examples/vector_calculus.esk) - Uses undefined operators
- [`examples/autodiff_example.esk`](../examples/autodiff_example.esk) - Uses dual numbers (not implemented)

**Working Examples**:
- [`examples/test_autodiff.esk`](../examples/test_autodiff.esk) - Basic `diff` only

---

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Eshkol Autodiff System                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌────────────────┐         ┌──────────────────┐          │
│  │ Symbolic Diff  │         │  Forward-Mode AD │          │
│  │  (diff expr)   │         │  (dual numbers)  │          │
│  └────────┬───────┘         └────────┬─────────┘          │
│           │                           │                    │
│           │                           │                    │
│  ┌────────▼───────┐         ┌────────▼─────────┐          │
│  │   Parser       │         │  Derivative Op   │          │
│  │   Extensions   │◄────────┤  (derivative f)  │          │
│  └────────┬───────┘         └────────┬─────────┘          │
│           │                           │                    │
│           │    ┌──────────────────────┴──────┐            │
│           │    │                              │            │
│  ┌────────▼────▼───┐         ┌───────────────▼────┐       │
│  │  LLVM Codegen   │         │  Reverse-Mode AD   │       │
│  │  (dual arith)   │         │  (comp graphs)     │       │
│  └────────┬────────┘         └────────┬───────────┘       │
│           │                           │                    │
│           │                  ┌────────▼───────────┐        │
│           │                  │  Gradient Operator │        │
│           │                  │  (gradient f)      │        │
│           │                  └────────┬───────────┘        │
│           │                           │                    │
│  ┌────────▼───────────────────────────▼────┐              │
│  │        Vector Calculus Operators         │              │
│  │  gradient │ jacobian │ divergence │ curl │              │
│  └──────────────────────┬───────────────────┘              │
│                         │                                   │
│            ┌────────────▼──────────────┐                   │
│            │  Tagged Value Integration │                   │
│            │  Tensor System Integration│                   │
│            │  Arena Memory Management  │                   │
│            └───────────────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

### Type System Hierarchy

```
eshkol_value_type_t (enum in eshkol.h)
├── ESHKOL_VALUE_NULL (0)
├── ESHKOL_VALUE_INT64 (1)
├── ESHKOL_VALUE_DOUBLE (2)
├── ESHKOL_VALUE_CONS_PTR (3)
├── ESHKOL_VALUE_DUAL_NUMBER (4)     ← NEW in Phase 2
├── ESHKOL_VALUE_AD_NODE_PTR (5)     ← NEW in Phase 3
└── ESHKOL_VALUE_MAX (15)

eshkol_tagged_value_t (struct in eshkol.h:68-78)
├── uint8_t type
├── uint8_t flags (exact/inexact)
├── uint16_t reserved
└── union data {int64, double, ptr, raw}
```

### Memory Layout

```
Phase 2: Dual Number Storage
┌──────────────────────────────┐
│ eshkol_dual_number_t         │
│ ┌──────────────────────────┐ │
│ │ double value             │ │  8 bytes
│ ├──────────────────────────┤ │
│ │ double derivative        │ │  8 bytes
│ └──────────────────────────┘ │
└──────────────────────────────┘  16 bytes total

Phase 3: AD Node Storage
┌──────────────────────────────┐
│ ad_node_t                    │
│ ┌──────────────────────────┐ │
│ │ ad_node_type_t type      │ │  4 bytes
│ ├──────────────────────────┤ │
│ │ double value             │ │  8 bytes
│ ├──────────────────────────┤ │
│ │ double gradient          │ │  8 bytes
│ ├──────────────────────────┤ │
│ │ ad_node_t* inputs[2]     │ │  16 bytes
│ ├──────────────────────────┤ │
│ │ size_t input_count       │ │  8 bytes
│ └──────────────────────────┘ │
└──────────────────────────────┘  48 bytes total
```

---

## Implementation Phases

### PHASE 0: Foundation Fixes (Sessions 1-3)
**Goal**: Fix existing `diff` operator as foundation for new features

#### Session 1-2: Fix Type System Bugs (SCH-008)

**File**: [`lib/backend/llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp)

**Bug 1: Type Mismatch in Derivatives** (Line 5084)
```cpp
// CURRENT (BROKEN):
Value* two = ConstantInt::get(Type::getInt64Ty(*context), 2);
return builder->CreateMul(two, f);  // f might be double!

// FIX:
Value* detectExpressionType(const eshkol_ast_t* expr) {
    // Recursively determine if expression is int64 or double
}

Value* two = (isDoubleExpression(expr)) ?
    ConstantFP::get(Type::getDoubleTy(*context), 2.0) :
    ConstantInt::get(Type::getInt64Ty(*context), 2);
```

**Bug 2: Incomplete Product Rule** (Lines 5076-5092)
```cpp
// CURRENT: Only handles x*x
if (op->call_op.variables[0].variable.id == var &&
    op->call_op.variables[1].variable.id == var) {
    // Special case
}

// FIX: General product rule
Value* f = codegenAST(&op->call_op.variables[0]);
Value* g = codegenAST(&op->call_op.variables[1]);
Value* f_prime = differentiate(&op->call_op.variables[0], var);
Value* g_prime = differentiate(&op->call_op.variables[1], var);

// f'*g + f*g' (with proper type handling)
Value* term1 = createMul(f_prime, g);  // Type-aware multiply
Value* term2 = createMul(f, g_prime);
return createAdd(term1, term2);
```

**Bug 3: Simplified Trig Functions** (Lines 5099-5111)
```cpp
// CURRENT:
// d/dx(sin(f)) = f'  (WRONG!)

// FIX:
// d/dx(sin(f)) = cos(f) * f'
Value* f = codegenAST(&op->call_op.variables[0]);
Value* f_prime = differentiate(&op->call_op.variables[0], var);
Value* cos_f = builder->CreateCall(function_table["cos"], {f});
return createMul(cos_f, f_prime);  // Type-aware multiply
```

#### Session 3: Expand Diff Coverage

**Add Missing Differentiation Rules**:
```cpp
// Division rule: d/dx(f/g) = (f'g - fg')/g²
else if (func_name == "/" && op->call_op.num_vars == 2) {
    Value* f = codegenAST(&op->call_op.variables[0]);
    Value* g = codegenAST(&op->call_op.variables[1]);
    Value* f_prime = differentiate(&op->call_op.variables[0], var);
    Value* g_prime = differentiate(&op->call_op.variables[1], var);
    
    // (f'*g - f*g') / (g*g)
    Value* numerator = createSub(createMul(f_prime, g), createMul(f, g_prime));
    Value* denominator = createMul(g, g);
    return createDiv(numerator, denominator);
}

// Exponential: d/dx(exp(f)) = exp(f) * f'
else if (func_name == "exp" && op->call_op.num_vars == 1) {
    Value* f = codegenAST(&op->call_op.variables[0]);
    Value* f_prime = differentiate(&op->call_op.variables[0], var);
    Value* exp_f = builder->CreateCall(function_table["exp"], {f});
    return createMul(exp_f, f_prime);
}

// Natural log: d/dx(log(f)) = f' / f
else if (func_name == "log" && op->call_op.num_vars == 1) {
    Value* f = codegenAST(&op->call_op.variables[0]);
    Value* f_prime = differentiate(&op->call_op.variables[0], var);
    return createDiv(f_prime, f);
}

// Power rule: d/dx(f^n) = n * f^(n-1) * f' (constant exponent)
else if (func_name == "pow" && op->call_op.num_vars == 2) {
    // Check if exponent is constant
    if (isConstant(&op->call_op.variables[1])) {
        Value* f = codegenAST(&op->call_op.variables[0]);
        Value* n = codegenAST(&op->call_op.variables[1]);
        Value* f_prime = differentiate(&op->call_op.variables[0], var);
        
        // n * f^(n-1) * f'
        Value* n_minus_1 = createSub(n, getConstant(1.0));
        Value* f_power = builder->CreateCall(function_table["pow"], {f, n_minus_1});
        return createMul(createMul(n, f_power), f_prime);
    }
    // Otherwise: general f^g rule (more complex)
}
```

**Test Coverage**:
```scheme
;; tests/autodiff_phase0_comprehensive.esk
(test-diff (* 3.5 x) x)        ; Should return 3.5 (double)
(test-diff (/ x 2.0) x)        ; Should return 0.5 (quotient rule)
(test-diff (sin (* 2 x)) x)    ; Should return 2*cos(2x) (chain rule)
(test-diff (exp (* x x)) x)    ; Should return 2x*exp(x²)
(test-diff (log (+ x 1)) x)    ; Should return 1/(x+1)
(test-diff (pow x 3) x)        ; Should return 3x²
```

**Deliverables Phase 0**:
- ✅ Fixed type handling in `diff`
- ✅ Complete product rule
- ✅ Proper trig derivatives with chain rule
- ✅ Division, exp, log, power rules
- ✅ Comprehensive test suite

---

### PHASE 1: Type System & Infrastructure (Sessions 4-8)
**Goal**: Build foundational types and data structures for AD

#### Session 4: Dual Number Type Definition

**File**: [`inc/eshkol/eshkol.h`](../inc/eshkol/eshkol.h)

**Add Dual Number Structure**:
```c
// Dual number for forward-mode automatic differentiation
// Stores value and derivative simultaneously
typedef struct eshkol_dual_number {
    double value;       // f(x)
    double derivative;  // f'(x)
} eshkol_dual_number_t;

// Compile-time size check
_Static_assert(sizeof(eshkol_dual_number_t) == 16,
               "Dual number must be 16 bytes for cache efficiency");
```

**Add to Value Type Enum**:
```c
typedef enum {
    ESHKOL_VALUE_NULL     = 0,
    ESHKOL_VALUE_INT64    = 1,
    ESHKOL_VALUE_DOUBLE   = 2,
    ESHKOL_VALUE_CONS_PTR = 3,
    ESHKOL_VALUE_DUAL_NUMBER = 4,     // NEW
    ESHKOL_VALUE_AD_NODE_PTR = 5,     // NEW (Phase 3)
    ESHKOL_VALUE_MAX      = 15
} eshkol_value_type_t;
```

**Helper Functions**:
```c
// Create dual number
static inline eshkol_dual_number_t eshkol_make_dual(double value, double derivative) {
    eshkol_dual_number_t result = {value, derivative};
    return result;
}

// Extract components
static inline double eshkol_dual_value(const eshkol_dual_number_t* d) {
    return d->value;
}

static inline double eshkol_dual_derivative(const eshkol_dual_number_t* d) {
    return d->derivative;
}
```

#### Session 5: Dual Number LLVM IR Types

**File**: [`lib/backend/llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp)

**Create LLVM Dual Number Type**:
```cpp
class EshkolLLVMCodeGen {
private:
    StructType* dual_number_type;  // NEW member variable
    
    void createBuiltinFunctions() {
        // ... existing code ...
        
        // Initialize dual number struct type
        std::vector<Type*> dual_fields;
        dual_fields.push_back(Type::getDoubleTy(*context));  // value
        dual_fields.push_back(Type::getDoubleTy(*context));  // derivative
        dual_number_type = StructType::create(*context, dual_fields, "dual_number");
        
        eshkol_debug("Created dual_number LLVM type");
    }
};
```

**Dual Number Helpers**:
```cpp
// Pack value and derivative into dual number
Value* packDualNumber(Value* value, Value* derivative) {
    Value* dual_ptr = builder->CreateAlloca(dual_number_type, nullptr, "dual");
    
    Value* value_ptr = builder->CreateStructGEP(dual_number_type, dual_ptr, 0);
    builder->CreateStore(value, value_ptr);
    
    Value* deriv_ptr = builder->CreateStructGEP(dual_number_type, dual_ptr, 1);
    builder->CreateStore(derivative, deriv_ptr);
    
    return builder->CreateLoad(dual_number_type, dual_ptr);
}

// Unpack dual number components
std::pair<Value*, Value*> unpackDualNumber(Value* dual) {
    Value* dual_ptr = builder->CreateAlloca(dual_number_type, nullptr, "temp_dual");
    builder->CreateStore(dual, dual_ptr);
    
    Value* value_ptr = builder->CreateStructGEP(dual_number_type, dual_ptr, 0);
    Value* value = builder->CreateLoad(Type::getDoubleTy(*context), value_ptr);
    
    Value* deriv_ptr = builder->CreateStructGEP(dual_number_type, dual_ptr, 1);
    Value* deriv = builder->CreateLoad(Type::getDoubleTy(*context), deriv_ptr);
    
    return {value, deriv};
}
```

#### Session 6: Computational Graph Node Types

**File**: [`inc/eshkol/eshkol.h`](../inc/eshkol/eshkol.h)

**Add AD Node Structures**:
```c
// AD node types for reverse-mode differentiation
typedef enum {
    AD_NODE_CONSTANT,
    AD_NODE_VARIABLE,
    AD_NODE_ADD,
    AD_NODE_SUB,
    AD_NODE_MUL,
    AD_NODE_DIV,
    AD_NODE_SIN,
    AD_NODE_COS,
    AD_NODE_EXP,
    AD_NODE_LOG,
    AD_NODE_POW,
    AD_NODE_NEG
} ad_node_type_t;

// Computational graph node for reverse-mode AD
typedef struct ad_node {
    ad_node_type_t type;
    double value;               // Computed during forward pass
    double gradient;            // Accumulated during backward pass
    struct ad_node* input1;     // First parent (null for constants/variables)
    struct ad_node* input2;     // Second parent (null for unary ops)
    size_t id;                  // Unique node ID for topological sort
} ad_node_t;

// Computational graph/tape
typedef struct ad_tape {
    ad_node_t** nodes;          // Array of nodes in evaluation order
    size_t num_nodes;
    size_t capacity;
    ad_node_t** variables;      // Input variable nodes
    size_t num_variables;
} ad_tape_t;
```

#### Session 7: AD Memory Management

**File**: [`lib/core/arena_memory.h`](../lib/core/arena_memory.h) and `.cpp`

**Add AD-Specific Allocation**:
```c
// Dual number allocation
eshkol_dual_number_t* arena_allocate_dual_number(arena_t* arena);
eshkol_dual_number_t* arena_allocate_dual_batch(arena_t* arena, size_t count);

// AD node allocation  
ad_node_t* arena_allocate_ad_node(arena_t* arena);
ad_node_t* arena_allocate_ad_batch(arena_t* arena, size_t count);

// Tape allocation
ad_tape_t* arena_allocate_tape(arena_t* arena, size_t initial_capacity);
void arena_tape_add_node(ad_tape_t* tape, ad_node_t* node);
void arena_tape_reset(ad_tape_t* tape);
```

**Implementation**:
```c
// In arena_memory.cpp
eshkol_dual_number_t* arena_allocate_dual_number(arena_t* arena) {
    eshkol_dual_number_t* dual = (eshkol_dual_number_t*)
        arena_allocate_aligned(arena, sizeof(eshkol_dual_number_t), 8);
    
    if (dual) {
        dual->value = 0.0;
        dual->derivative = 0.0;
    }
    return dual;
}

ad_node_t* arena_allocate_ad_node(arena_t* arena) {
    ad_node_t* node = (ad_node_t*)
        arena_allocate_aligned(arena, sizeof(ad_node_t), 8);
    
    if (node) {
        node->type = AD_NODE_CONSTANT;
        node->value = 0.0;
        node->gradient = 0.0;
        node->input1 = nullptr;
        node->input2 = nullptr;
        node->id = 0;
    }
    return node;
}
```

#### Session 8: Parser Extensions for AD Operators

**File**: [`lib/frontend/parser.cpp`](../lib/frontend/parser.cpp)

**Add New Operators**:
```cpp
static eshkol_op_t get_operator_type(const std::string& op) {
    // ... existing ...
    if (op == "diff") return ESHKOL_DIFF_OP;
    if (op == "derivative") return ESHKOL_DERIVATIVE_OP;     // NEW
    if (op == "gradient") return ESHKOL_GRADIENT_OP;         // NEW
    if (op == "jacobian") return ESHKOL_JACOBIAN_OP;         // NEW
    if (op == "hessian") return ESHKOL_HESSIAN_OP;           // NEW
    if (op == "divergence") return ESHKOL_DIVERGENCE_OP;     // NEW
    if (op == "curl") return ESHKOL_CURL_OP;                 // NEW
    if (op == "laplacian") return ESHKOL_LAPLACIAN_OP;       // NEW
    return ESHKOL_CALL_OP;
}
```

**Parse Derivative Syntax**:
```cpp
// (derivative function point)
if (ast.operation.op == ESHKOL_DERIVATIVE_OP) {
    // Parse function (can be lambda or function name)
    token = tokenizer.nextToken();
    eshkol_ast_t function = parse_expression_or_lambda(tokenizer, token);
    
    // Parse evaluation point
    token = tokenizer.nextToken();
    eshkol_ast_t point = parse_expression(tokenizer, token);
    
    // Store in derivative_op structure
    ast.operation.derivative_op.function = new eshkol_ast_t;
    *ast.operation.derivative_op.function = function;
    ast.operation.derivative_op.point = new eshkol_ast_t;
    *ast.operation.derivative_op.point = point;
}
```

**Deliverables Phase 1**:
- ✅ Dual number type in [`eshkol.h`](../inc/eshkol/eshkol.h)
- ✅ AD node types and structures
- ✅ LLVM IR dual number type
- ✅ Arena allocation for AD structures
- ✅ Parser support for new operators

---

### PHASE 2: Forward-Mode AD (Sessions 9-14)
**Goal**: Implement dual number arithmetic and derivative operator

#### Session 9-10: Dual Number Arithmetic

**File**: [`lib/backend/llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp)

**Implement All Operations**:
```cpp
// Addition: (a,a') + (b,b') = (a+b, a'+b')
Value* dualAdd(Value* dual_a, Value* dual_b) {
    auto [a, a_prime] = unpackDualNumber(dual_a);
    auto [b, b_prime] = unpackDualNumber(dual_b);
    
    Value* value = builder->CreateFAdd(a, b);
    Value* deriv = builder->CreateFAdd(a_prime, b_prime);
    
    return packDualNumber(value, deriv);
}

// Multiplication: (a,a') * (b,b') = (a*b, a'*b + a*b')
Value* dualMul(Value* dual_a, Value* dual_b) {
    auto [a, a_prime] = unpackDualNumber(dual_a);
    auto [b, b_prime] = unpackDualNumber(dual_b);
    
    Value* value = builder->CreateFMul(a, b);
    Value* term1 = builder->CreateFMul(a_prime, b);
    Value* term2 = builder->CreateFMul(a, b_prime);
    Value* deriv = builder->CreateFAdd(term1, term2);
    
    return packDualNumber(value, deriv);
}

// Division: (a,a') / (b,b') = (a/b, (a'*b - a*b')/b²)
Value* dualDiv(Value* dual_a, Value* dual_b) {
    auto [a, a_prime] = unpackDualNumber(dual_a);
    auto [b, b_prime] = unpackDualNumber(dual_b);
    
    Value* value = builder->CreateFDiv(a, b);
    
    Value* numerator_term1 = builder->CreateFMul(a_prime, b);
    Value* numerator_term2 = builder->CreateFMul(a, b_prime);
    Value* numerator = builder->CreateFSub(numerator_term1, numerator_term2);
    Value* denominator = builder->CreateFMul(b, b);
    Value* deriv = builder->CreateFDiv(numerator, denominator);
    
    return packDualNumber(value, deriv);
}

// Sin: sin(a,a') = (sin(a), a'*cos(a))
Value* dualSin(Value* dual_a) {
    auto [a, a_prime] = unpackDualNumber(dual_a);
    
    Value* value = builder->CreateCall(function_table["sin"], {a});
    Value* cos_a = builder->CreateCall(function_table["cos"], {a});
    Value* deriv = builder->CreateFMul(a_prime, cos_a);
    
    return packDualNumber(value, deriv);
}

// Cos: cos(a,a') = (cos(a), -a'*sin(a))
Value* dualCos(Value* dual_a) {
    auto [a, a_prime] = unpackDualNumber(dual_a);
    
    Value* value = builder->CreateCall(function_table["cos"], {a});
    Value* sin_a = builder->CreateCall(function_table["sin"], {a});
    Value* neg_sin = builder->CreateFNeg(sin_a);
    Value* deriv = builder->CreateFMul(a_prime, neg_sin);
    
    return packDualNumber(value, deriv);
}

// Exp: exp(a,a') = (exp(a), a'*exp(a))
Value* dualExp(Value* dual_a) {
    auto [a, a_prime] = unpackDualNumber(dual_a);
    
    Value* exp_a = builder->CreateCall(function_table["exp"], {a});
    Value* deriv = builder->CreateFMul(a_prime, exp_a);
    
    return packDualNumber(exp_a, deriv);
}

// Log: log(a,a') = (log(a), a'/a)
Value* dualLog(Value* dual_a) {
    auto [a, a_prime] = unpackDualNumber(dual_a);
    
    Value* value = builder->CreateCall(function_table["log"], {a});
    Value* deriv = builder->CreateFDiv(a_prime, a);
    
    return packDualNumber(value, deriv);
}

// Pow: (a,a')^(b,b') = (a^b, a^b * (b'*log(a) + b*a'/a))
Value* dualPow(Value* dual_a, Value* dual_b) {
    auto [a, a_prime] = unpackDualNumber(dual_a);
    auto [b, b_prime] = unpackDualNumber(dual_b);
    
    Value* value = builder->CreateCall(function_table["pow"], {a, b});
    
    Value* log_a = builder->CreateCall(function_table["log"], {a});
    Value* term1 = builder->CreateFMul(b_prime, log_a);
    Value* term2 = builder->CreateFMul(b, builder->CreateFDiv(a_prime, a));
    Value* sum = builder->CreateFAdd(term1, term2);
    Value* deriv = builder->CreateFMul(value, sum);
    
    return packDualNumber(value, deriv);
}
```

#### Session 11-12: Derivative Operator Implementation

**Add to AST** ([`eshkol.h`](../inc/eshkol/eshkol.h)):
```c
typedef enum {
    // ... existing ...
    ESHKOL_DERIVATIVE_OP,    // NEW
} eshkol_op_t;

struct {
    struct eshkol_ast *function;  // Function to differentiate
    struct eshkol_ast *point;     // Point to evaluate derivative
    uint8_t mode;                 // 0=forward, 1=reverse, 2=auto
} derivative_op;
```

**Codegen** ([`llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp)):
```cpp
Value* codegenDerivative(const eshkol_operations_t* op) {
    // Get function (can be lambda or function reference)
    Value* func = resolveLambdaFunction(op->derivative_op.function);
    if (!func) return nullptr;
    
    Function* func_ptr = dyn_cast<Function>(func);
    
    // Get evaluation point
    Value* x = codegenAST(op->derivative_op.point);
    
    // Create dual number with seed derivative = 1.0
    Value* x_dual = packDualNumber(x, ConstantFP::get(Type::getDoubleTy(*context), 1.0));
    
    // Call function with dual number
    Value* result_dual = builder->CreateCall(func_ptr, {x_dual});
    
    // Extract derivative component
    auto [value, derivative] = unpackDualNumber(result_dual);
    
    return derivative;  // Return just the derivative
}
```

**Polymorphic Arithmetic Extension**:
```cpp
// Extend existing polymorphic functions to handle dual numbers
Value* polymorphicAdd(Value* left_tagged, Value* right_tagged) {
    Value* left_type = getTaggedValueType(left_tagged);
    Value* right_type = getTaggedValueType(right_tagged);
    
    // Check if either is dual number
    Value* left_is_dual = builder->CreateICmpEQ(left_base,
        ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_DUAL_NUMBER));
    Value* right_is_dual = builder->CreateICmpEQ(right_base,
        ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_DUAL_NUMBER));
    Value* any_dual = builder->CreateOr(left_is_dual, right_is_dual);
    
    // Branch: dual path vs regular path
    BasicBlock* dual_path = BasicBlock::Create(*context, "add_dual_path", current_func);
    BasicBlock* regular_path = BasicBlock::Create(*context, "add_regular_path", current_func);
    BasicBlock* merge = BasicBlock::Create(*context, "add_merge", current_func);
    
    builder->CreateCondBr(any_dual, dual_path, regular_path);
    
    // Dual path: extract dual numbers and use dual arithmetic
    builder->SetInsertPoint(dual_path);
    Value* left_dual = unpackDualFromTaggedValue(left_tagged);
    Value* right_dual = unpackDualFromTaggedValue(right_tagged);
    Value* dual_result = dualAdd(left_dual, right_dual);
    Value* tagged_dual_result = packDualToTaggedValue(dual_result);
    builder->CreateBr(merge);
    
    // Regular path: existing int/double logic
    builder->SetInsertPoint(regular_path);
    // ... existing code ...
    
    // Merge results
    builder->SetInsertPoint(merge);
    PHINode* result_phi = builder->CreatePHI(tagged_value_type, 2);
    result_phi->addIncoming(tagged_dual_result, dual_path);
    result_phi->addIncoming(tagged_regular_result, regular_path);
    
    return result_phi;
}
```

#### Session 13-14: Chain Rule & Function Composition

**Handle Nested Functions**:
```cpp
// Automatic chain rule through dual number propagation
// User writes: (derivative (lambda (x) (sin (* x x))) 2.0)
// Compiler generates:
// 1. x_dual = (2.0, 1.0)
// 2. x_squared_dual = (* x_dual x_dual) = (4.0, 4.0)  ; via dualMul
// 3. result_dual = (sin x_squared_dual) = (sin(4.0), 4.0*cos(4.0))
// 4. Extract derivative: 4.0*cos(4.0)
```

**Lambda Differentiation**:
```cpp
Value* codegenLambdaForwardAD(const eshkol_operations_t* lambda_op, Value* x_dual) {
    // Generate lambda body with dual number arithmetic
    // All operations automatically use dual variants
    // Return result dual number
}
```

**Deliverables Phase 2**:
- ✅ Complete dual number arithmetic (all math ops)
- ✅ `derivative` operator working
- ✅ Chain rule automatic via dual propagation  
- ✅ Lambda differentiation support
- ✅ Polymorphic arithmetic extended for duals

---

### PHASE 3: Reverse-Mode AD (Sessions 15-23)
**Goal**: Implement computational graphs and backpropagation

#### Session 15-16: Graph Construction (Forward Pass)

**File**: [`lib/backend/llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp)

**Graph Building During Execution**:
```cpp
class ComputationalGraphBuilder {
private:
    ad_tape_t* tape;
    std::map<std::string, ad_node_t*> variable_nodes;
    size_t next_node_id;
    
public:
    // Record operations during forward pass
    ad_node_t* recordAdd(ad_node_t* a, ad_node_t* b) {
        ad_node_t* node = arena_allocate_ad_node(current_arena);
        node->type = AD_NODE_ADD;
        node->value = a->value + b->value;  // Compute value
        node->gradient = 0.0;               // Initialize gradient
        node->input1 = a;
        node->input2 = b;
        node->id = next_node_id++;
        
        arena_tape_add_node(tape, node);
        return node;
    }
    
    ad_node_t* recordMul(ad_node_t* a, ad_node_t* b) {
        ad_node_t* node = arena_allocate_ad_node(current_arena);
        node->type = AD_NODE_MUL;
        node->value = a->value * b->value;
        node->gradient = 0.0;
        node->input1 = a;
        node->input2 = b;
        node->id = next_node_id++;
        
        arena_tape_add_node(tape, node);
        return node;
    }
    
    // ... similar for all operations
};
```

**LLVM Implementation**:
```cpp
Value* codegenADNodeAdd(Value* node_a_ptr, Value* node_b_ptr) {
    // Allocate new node
    Value* new_node = builder->CreateCall(arena_allocate_ad_node_func, {getArenaPtr()});
    
    // Set type
    Value* type_ptr = builder->CreateStructGEP(ad_node_type, new_node, 0);
    builder->CreateStore(
        ConstantInt::get(Type::getInt32Ty(*context), AD_NODE_ADD),
        type_ptr
    );
    
    // Compute and store value
    Value* a_value = loadNodeValue(node_a_ptr);
    Value* b_value = loadNodeValue(node_b_ptr);
    Value* result_value = builder->CreateFAdd(a_value, b_value);
    
    Value* value_ptr = builder->CreateStructGEP(ad_node_type, new_node, 1);
    builder->CreateStore(result_value, value_ptr);
    
    // Store input pointers
    Value* input1_ptr = builder->CreateStructGEP(ad_node_type, new_node, 3);
    builder->CreateStore(node_a_ptr, input1_ptr);
    
    Value* input2_ptr = builder->CreateStructGEP(ad_node_type, new_node, 4);
    builder->CreateStore(node_b_ptr, input2_ptr);
    
    // Add to tape
    recordNodeInTape(new_node);
    
    return new_node;
}
```

#### Session 17-18: Backward Pass (Backpropagation)

**Gradient Accumulation**:
```cpp
void codegenBackward(Value* output_node_ptr) {
    // Initialize output gradient = 1.0
    Value* output_grad_ptr = builder->CreateStructGEP(ad_node_type, output_node_ptr, 2);
    builder->CreateStore(ConstantFP::get(Type::getDoubleTy(*context), 1.0), output_grad_ptr);
    
    // Traverse tape in reverse order
    // For each node, propagate gradient to inputs
    
    // Get tape
    Value* tape_ptr = getCurrentTape();
    Value* num_nodes = getTapeNodeCount(tape_ptr);
    
    // Loop backwards through nodes
    Value* counter = builder->CreateAlloca(Type::getInt64Ty(*context));
    builder->CreateStore(num_nodes, counter);
    
    BasicBlock* loop_cond = BasicBlock::Create(*context, "backward_cond", current_func);
    BasicBlock* loop_body = BasicBlock::Create(*context, "backward_body", current_func);
    BasicBlock* loop_exit = BasicBlock::Create(*context, "backward_exit", current_func);
    
    builder->CreateBr(loop_cond);
    
    builder->SetInsertPoint(loop_cond);
    Value* i = builder->CreateLoad(Type::getInt64Ty(*context), counter);
    Value* i_gt_zero = builder->CreateICmpUGT(i, ConstantInt::get(Type::getInt64Ty(*context), 0));
    builder->CreateCondBr(i_gt_zero, loop_body, loop_exit);
    
    builder->SetInsertPoint(loop_body);
    Value* i_minus_1 = builder->CreateSub(i, ConstantInt::get(Type::getInt64Ty(*context), 1));
    builder->CreateStore(i_minus_1, counter);
    
    // Get node at index i-1
    Value* node_ptr = getTapeNode(tape_ptr, i_minus_1);
    
    // Propagate gradient based on operation type
    propagateGradient(node_ptr);
    
    builder->CreateBr(loop_cond);
    
    builder->SetInsertPoint(loop_exit);
}

void propagateGradient(Value* node_ptr) {
    // Load node type
    Value* type_ptr = builder->CreateStructGEP(ad_node_type, node_ptr, 0);
    Value* node_type = builder->CreateLoad(Type::getInt32Ty(*context), type_ptr);
    
    // Load node gradient
    Value* grad_ptr = builder->CreateStructGEP(ad_node_type, node_ptr, 2);
    Value* grad = builder->CreateLoad(Type::getDoubleTy(*context), grad_ptr);
    
    // Branch on operation type
    // For ADD: gradient flows equally to both inputs
    // For MUL: gradient multiplied by other input value
    // etc.
    
    // Example for ADD:
    Value* is_add = builder->CreateICmpEQ(node_type,
        ConstantInt::get(Type::getInt32Ty(*context), AD_NODE_ADD));
    
    BasicBlock* add_case = BasicBlock::Create(*context, "grad_add", current_func);
    BasicBlock* mul_case = BasicBlock::Create(*context, "grad_mul", current_func);
    // ... more cases
    
    builder->CreateCondBr(is_add, add_case, next_check);
    
    builder->SetInsertPoint(add_case);
    // For z = x + y:
    // dL/dx = dL/dz * dz/dx = dL/dz * 1
    // dL/dy = dL/dz * dz/dy = dL/dz * 1
    Value* input1_ptr = loadNodeInput1(node_ptr);
    Value* input2_ptr = loadNodeInput2(node_ptr);
    accumulateGradient(input1_ptr, grad);  // Add grad to input1's gradient
    accumulateGradient(input2_ptr, grad);  // Add grad to input2's gradient
    builder->CreateBr(done);
}

void accumulateGradient(Value* node_ptr, Value* gradient_to_add) {
    // Load current gradient
    Value* grad_ptr = builder->CreateStructGEP(ad_node_type, node_ptr, 2);
    Value* current_grad = builder->CreateLoad(Type::getDoubleTy(*context), grad_ptr);
    
    // Add incoming gradient
    Value* new_grad = builder->CreateFAdd(current_grad, gradient_to_add);
    
    // Store updated gradient
    builder->CreateStore(new_grad, grad_ptr);
}
```

#### Session 19-20: Gradient Operator

**Parse Gradient** ([`parser.cpp`](../lib/frontend/parser.cpp)):
```cpp
// (gradient function vector)
if (ast.operation.op == ESHKOL_GRADIENT_OP) {
    // Parse function
    token = tokenizer.nextToken();
    eshkol_ast_t function = parse_expression_or_lambda(tokenizer, token);
    
    // Parse input vector
    token = tokenizer.nextToken();
    eshkol_ast_t vector = parse_expression(tokenizer, token);
    
    ast.operation.gradient_op.function = new eshkol_ast_t;
    *ast.operation.gradient_op.function = function;
    ast.operation.gradient_op.point = new eshkol_ast_t;
    *ast.operation.gradient_op.point = vector;
}
```

**Codegen Gradient**:
```cpp
Value* codegenGradient(const eshkol_operations_t* op) {
    // For function f: ℝⁿ → ℝ, compute ∇f at point v
    // Returns vector of partial derivatives
    
    Function* func = resolveLambdaFunction(op->gradient_op.function);
    Value* vector_ptr = codegenAST(op->gradient_op.point);
    
    // Get vector dimensions
    size_t n = getTensorDimension(vector_ptr, 0);
    
    // Create result vector for gradient
    Value* gradient_vector = createTensor({n});
    
    // For each component, compute partial derivative
    for (size_t i = 0; i < n; i++) {
        // Create AD variable node for component i with seed gradient
        Value* var_node = createADVariable(i, getVectorElement(vector_ptr, i));
        
        // Build computational graph by calling function
        Value* output_node = callFunctionWithADNodes(func, vector_ptr, i);
        
        // Run backward pass
        codegenBackward(output_node);
        
        // Extract gradient w.r.t. variable i
        Value* partial_i = extractNodeGradient(var_node);
        
        // Store in result vector
        setTensorElement(gradient_vector, i, partial_i);
        
        // Reset tape for next partial derivative
        resetTape();
    }
    
    return gradient_vector;
}
```

**Vector Return Type Construction**:
```cpp
Value* createGradientVector(const std::vector<Value*>& partial_derivatives) {
    size_t n = partial_derivatives.size();
    
    // Create 1D tensor to hold gradient
    std::vector<Type*> tensor_fields;
    tensor_fields.push_back(PointerType::getUnqual(*context)); // dimensions
    tensor_fields.push_back(Type::getInt64Ty(*context));       // num_dimensions
    tensor_fields.push_back(PointerType::getUnqual(*context)); // elements
    tensor_fields.push_back(Type::getInt64Ty(*context));       // total_elements
    StructType* tensor_type = StructType::create(*context, tensor_fields, "tensor");
    
    // Allocate tensor structure
    Value* tensor_ptr = allocateTensor(1, {n});
    
    // Fill with partial derivatives
    Value* elements_ptr = getTensorElements(tensor_ptr);
    for (size_t i = 0; i < n; i++) {
        Value* elem_ptr = builder->CreateGEP(Type::getDoubleTy(*context), 
                                             elements_ptr, 
                                             ConstantInt::get(Type::getInt64Ty(*context), i));
        builder->CreateStore(partial_derivatives[i], elem_ptr);
    }
    
    return tensor_ptr;
}
```

#### Session 21-22: Higher-Order Derivatives

**Jacobian Operator**:
```cpp
Value* codegenJacobian(const eshkol_operations_t* op) {
    // For function F: ℝⁿ → ℝᵐ
    // Jacobian is m×n matrix of partial derivatives
    // J[i,j] = ∂Fᵢ/∂xⱼ
    
    Function* func = resolveLambdaFunction(op->jacobian_op.function);
    Value* vector_ptr = codegenAST(op->jacobian_op.point);
    
    size_t n = getTensorDimension(vector_ptr, 0);  // Input dimension
    
    // Call function once to determine output dimension
    Value* output = builder->CreateCall(func, {vector_ptr});
    size_t m = getTensorDimension(output, 0);  // Output dimension
    
    // Create m×n matrix for Jacobian
    Value* jacobian_matrix = createTensor({m, n});
    Value* elements_ptr = getTensorElements(jacobian_matrix);
    
    // For each output component i
    for (size_t i = 0; i < m; i++) {
        // For each input component j
        for (size_t j = 0; j < n; j++) {
            // Build graph with variable j as seed
            Value* output_node = callFunctionWithADNodes(func, vector_ptr, j);
            
            // Extract component i of output
            Value* output_i_node = getOutputComponent(output_node, i);
            
            // Backward pass
            codegenBackward(output_i_node);
            
            // Get ∂Fᵢ/∂xⱼ
            Value* partial = extractGradient(getVariableNode(j));
            
            // Store in matrix[i,j]
            Value* linear_idx = builder->CreateAdd(
                builder->CreateMul(ConstantInt::get(Type::getInt64Ty(*context), i),
                                 ConstantInt::get(Type::getInt64Ty(*context), n)),
                ConstantInt::get(Type::getInt64Ty(*context), j)
            );
            Value* elem_ptr = builder->CreateGEP(Type::getDoubleTy(*context),
                                                 elements_ptr, linear_idx);
            builder->CreateStore(partial, elem_ptr);
            
            resetTape();
        }
    }
    
    return jacobian_matrix;
}
```

**Hessian Operator** (second derivatives):
```cpp
Value* codegenHessian(const eshkol_operations_t* op) {
    // Hessian is Jacobian of gradient
    // H[i,j] = ∂²f/∂xᵢ∂xⱼ
    
    // Approach: gradient returns vector function
    // Jacobian of that vector function is Hessian
    
    // Or: use forward-over-reverse mode
    // Forward pass: dual numbers
    // Reverse pass: on dual numbers
}
```

#### Session 23: Graph Memory & Tape Management

**Tape Operations**:
```c
// In arena_memory.h/cpp
typedef struct ad_tape {
    ad_node_t** nodes;
    size_t num_nodes;
    size_t capacity;
    ad_node_t** variables;
    size_t num_variables;
    arena_t* arena;  // Arena for allocations
} ad_tape_t;

ad_tape_t* ad_tape_create(arena_t* arena, size_t initial_capacity);
void ad_tape_add_node(ad_tape_t* tape, ad_node_t* node);
void ad_tape_reset(ad_tape_t* tape);
void ad_tape_destroy(ad_tape_t* tape);

ad_node_t* ad_tape_get_node(const ad_tape_t* tape, size_t index);
size_t ad_tape_get_node_count(const ad_tape_t* tape);
```

**Gradient Reset**:
```cpp
void resetNodeGradients(ad_tape_t* tape) {
    for (size_t i = 0; i < tape->num_nodes; i++) {
        tape->nodes[i]->gradient = 0.0;
    }
}

void clearTape(ad_tape_t* tape) {
    tape->num_nodes = 0;
    // Don't free nodes - arena will handle that
}
```

**Deliverables Phase 3**:
- ✅ Complete computational graph construction
- ✅ Backward pass/backpropagation engine
- ✅ `gradient` operator returning vectors
- ✅ `jacobian` operator returning matrices
- ✅ `hessian` operator for second derivatives
- ✅ Efficient tape management

---

### PHASE 4: Vector Calculus Operators (Sessions 24-28)
**Goal**: Implement differential geometry operators

#### Session 24-25: Gradient & Divergence

**Gradient** (already implemented in Phase 3):
```scheme
;; Gradient: ∇f for scalar field f: ℝⁿ → ℝ
;; Returns vector field: ∇f: ℝⁿ → ℝⁿ
(define f (lambda (v) (dot v v)))  ; f(v) = v·v
(gradient f #(1.0 2.0 3.0))        ; Returns #(2.0 4.0 6.0)
```

**Divergence**:
```scheme
;; Divergence: ∇·F for vector field F: ℝⁿ → ℝⁿ
;; Returns scalar field: ∇·F: ℝⁿ → ℝ
;; ∇·F = ∂F₁/∂x₁ + ∂F₂/∂x₂ + ... + ∂Fₙ/∂xₙ
(define F (lambda (v) v))  ; Identity field
(divergence F #(1.0 2.0 3.0))  ; Returns 3.0 (sum of diagonals)
```

**Implementation**:
```cpp
Value* codegenDivergence(const eshkol_operations_t* op) {
    Function* field_func = resolveLambdaFunction(op->divergence_op.function);
    Value* vector_ptr = codegenAST(op->divergence_op.point);
    
    size_t n = getTensorDimension(vector_ptr, 0);
    
    // Compute Jacobian (n×n matrix of partial derivatives)
    Value* jacobian = computeJacobianMatrix(field_func, vector_ptr);
    
    // Sum diagonal elements: ∂F₁/∂x₁ + ∂F₂/∂x₂ + ...
    Value* divergence = ConstantFP::get(Type::getDoubleTy(*context), 0.0);
    Value* elements_ptr = getTensorElements(jacobian);
    
    for (size_t i = 0; i < n; i++) {
        // Get diagonal element [i,i]
        Value* linear_idx = builder->CreateMul(
            ConstantInt::get(Type::getInt64Ty(*context), i),
            ConstantInt::get(Type::getInt64Ty(*context), n + 1)
        );
        Value* elem_ptr = builder->CreateGEP(Type::getDoubleTy(*context),
                                             elements_ptr, linear_idx);
        Value* elem = builder->CreateLoad(Type::getDoubleTy(*context), elem_ptr);
        
        divergence = builder->CreateFAdd(divergence, elem);
    }
    
    return divergence;
}
```

#### Session 26-27: Curl & Laplacian

**Curl** (3D only):
```scheme
;; Curl: ∇×F for vector field F: ℝ³ → ℝ³
;; Returns vector field: ∇×F: ℝ³ → ℝ³
;; (∇×F) = [∂F₃/∂x₂ - ∂F₂/∂x₃,
;;          ∂F₁/∂x₃ - ∂F₃/∂x₁,
;;          ∂F₂/∂x₁ - ∂F₁/∂x₂]
(define F (lambda (v) (vector (* (vref v 1) (vref v 2))
                               (* (vref v 2) (vref v 0))
                               (* (vref v 0) (vref v 1)))))
(curl F #(1.0 2.0 3.0))
```

**Implementation**:
```cpp
Value* codegenCurl(const eshkol_operations_t* op) {
    Function* field_func = resolveLambdaFunction(op->curl_op.function);
    Value* vector_ptr = codegenAST(op->curl_op.point);
    
    size_t n = getTensorDimension(vector_ptr, 0);
    
    // Validate dimension = 3
    if (n != 3) {
        eshkol_error("curl only defined for 3D vector fields, got dimension %zu", n);
        return nullptr;
    }
    
    // Compute Jacobian (3×3 matrix)
    Value* jacobian = computeJacobianMatrix(field_func, vector_ptr);
    Value* J = getTensorElements(jacobian);
    
    // Extract specific partial derivatives
    // J[i,j] = ∂Fᵢ/∂xⱼ
    Value* dF3_dx2 = loadMatrixElement(J, 2, 1, 3);  // ∂F₃/∂x₂
    Value* dF2_dx3 = loadMatrixElement(J, 1, 2, 3);  // ∂F₂/∂x₃
    Value* dF1_dx3 = loadMatrixElement(J, 0, 2, 3);  // ∂F₁/∂x₃
    Value* dF3_dx1 = loadMatrixElement(J, 2, 0, 3);  // ∂F₃/∂x₁
    Value* dF2_dx1 = loadMatrixElement(J, 1, 0, 3);  // ∂F₂/∂x₁
    Value* dF1_dx2 = loadMatrixElement(J, 0, 1, 3);  // ∂F₁/∂x₂
    
    // Compute curl components
    Value* curl_x = builder->CreateFSub(dF3_dx2, dF2_dx3);
    Value* curl_y = builder->CreateFSub(dF1_dx3, dF3_dx1);
    Value* curl_z = builder->CreateFSub(dF2_dx1, dF1_dx2);
    
    // Create result vector
    return createTensor({3}, {curl_x, curl_y, curl_z});
}
```

**Laplacian**:
```scheme
;; Laplacian: ∇²f for scalar field f: ℝⁿ → ℝ
;; Returns scalar: ∇²f = ∂²f/∂x₁² + ∂²f/∂x₂² + ...
(define f (lambda (v) (dot v v)))
(laplacian f #(1.0 2.0 3.0))  ; Returns 6.0 (constant second derivatives)
```

**Implementation**:
```cpp
Value* codegenLaplacian(const eshkol_operations_t* op) {
    Function* func = resolveLambdaFunction(op->laplacian_op.function);
    Value* vector_ptr = codegenAST(op->laplacian_op.point);
    
    size_t n = getTensorDimension(vector_ptr, 0);
    
    // Compute Hessian (n×n matrix of second derivatives)
    Value* hessian = computeHessianMatrix(func, vector_ptr);
    
    // Sum diagonal elements: ∂²f/∂x₁² + ∂²f/∂x₂² + ...
    Value* laplacian = ConstantFP::get(Type::getDoubleTy(*context), 0.0);
    Value* elements_ptr = getTensorElements(hessian);
    
    for (size_t i = 0; i < n; i++) {
        // Get diagonal element [i,i]
        Value* linear_idx = builder->CreateMul(
            ConstantInt::get(Type::getInt64Ty(*context), i),
            ConstantInt::get(Type::getInt64Ty(*context), n + 1)
        );
        Value* elem_ptr = builder->CreateGEP(Type::getDoubleTy(*context),
                                             elements_ptr, linear_idx);
        Value* elem = builder->CreateLoad(Type::getDoubleTy(*context), elem_ptr);
        
        laplacian = builder->CreateFAdd(laplacian, elem);
    }
    
    return laplacian;
}
```

#### Session 28: Directional Derivatives

**Directional Derivative**:
```scheme
;; D_v f = ∇f · v (gradient dotted with direction)
(directional-derivative f #(1.0 2.0 3.0) #(1.0 0.0 0.0))
;; Derivative in x-direction
```

**Implementation**:
```cpp
Value* codegenDirectionalDerivative(const eshkol_operations_t* op) {
    // Compute gradient
    Value* grad = codegenGradient(op);
    
    // Get direction vector
    Value* direction = codegenAST(op->directional_deriv_op.direction);
    
    // Dot product: ∇f · v
    return codegenTensorDot({grad, direction});
}
```

**Deliverables Phase 4**:
- ✅ `gradient` operator (from Phase 3)
- ✅ `divergence` operator
- ✅ `curl` operator (3D only, with dimension check)
- ✅ `laplacian` operator
- ✅ `directional-derivative` operator
- ✅ Dimension validation
- ✅ Mathematical correctness tests

---

### PHASE 5: Integration & Optimization (Sessions 29-32)
**Goal**: Integrate with existing systems and optimize performance

#### Session 29: Tensor Integration

**Autodiff for Tensor Operations**:
```cpp
// Make tensor-apply differentiable
Value* codegenTensorApplyAD(const eshkol_operations_t* op) {
    // Apply function to each element
    // If function uses AD nodes, track full computation graph
    // Enable gradient w.r.t. tensor elements
}

// Make tensor-reduce differentiable
Value* codegenTensorReduceAD(const eshkol_operations_t* op) {
    // Reduction with gradient backpropagation
    // Gradient flows to all elements that contributed to reduction
}
```

**Broadcasting Support**:
```cpp
// Handle broadcasting in gradients
// If f(x) where x is broadcasted, gradient must reduce appropriately
Value* handleBroadcastGradient(Value* grad, Shape original_shape, Shape broadcast_shape);
```

#### Session 30: Type Inference & Safety

**Type Inference Rules**:
```cpp
// Gradient type inference
// f: (Vector n -> Scalar) implies gradient: (Vector n -> Vector n)
Type* inferGradientType(Function* f) {
    FunctionType* ft = f->getFunctionType();
    Type* input_type = ft->getParamType(0);
    
    // If input is vector, gradient returns vector of same dimension
    if (isTensorType(input_type)) {
        return input_type;  // Same type as input
    }
    
    return Type::getDoubleTy(*context);  // Scalar case
}

// Jacobian type inference  
// F: (Vector n -> Vector m) implies jacobian: (Vector n -> Matrix m n)
Type* inferJacobianType(Function* F);

// Hessian type inference
// f: (Vector n -> Scalar) implies hessian: (Vector n -> Matrix n n)
Type* inferHessianType(Function* f);
```

**Type Safety Checks**:
```cpp
void validateGradientTypes(Value* func, Value* input) {
    // Check: function must be differentiable
    // Check: input must be numeric (int64, double, vector)
    // Check: function must return scalar for gradient
    // Emit compile-time error if validation fails
}
```

#### Session 31: Performance Optimization

**SIMD Vectorization for Gradients**:
```cpp
// When computing gradient of f: ℝⁿ → ℝ
// Multiple partial derivatives can be computed in parallel

Value* computeGradientVectorized(Function* f, Value* vector, size_t n) {
    // Batch partial derivative computations
    // Use SIMD for parallel evaluation
    // Especially effective for large n
}
```

**Sparse Gradient Support**:
```cpp
// Many gradients are sparse (most components = 0)
// Detect and exploit sparsity patterns

typedef struct sparse_gradient {
    size_t* indices;      // Non-zero indices
    double* values;       // Non-zero values
    size_t nnz;          // Number of non-zeros
    size_t dimension;    // Full dimension
} sparse_gradient_t;
```

**Compile-Time Constant Folding**:
```cpp
// For simple functions with known derivatives, compute at compile time
// E.g., (derivative (lambda (x) (* x x)) 5.0) → constant 10.0

Value* tryConstantFoldDerivative(const eshkol_ast_t* func, Value* point) {
    // Analyze function body
    // If pure arithmetic with constants, compute derivative symbolically
    // Return constant instead of runtime computation
}
```

#### Session 32: Error Handling & Diagnostics

**Runtime Checks**:
```cpp
// NaN/Inf detection
void checkNumericalStability(Value* value, const char* operation) {
    Value* is_nan = builder->CreateFCmpUNO(value, value);
    Value* is_inf = builder->CreateFCmpOEQ(
        builder->CreateCall(fabs_func, {value}),
        ConstantFP::get(Type::getDoubleTy(*context), INFINITY)
    );
    Value* is_invalid = builder->CreateOr(is_nan, is_inf);
    
    // Emit warning if invalid
    emitRuntimeWarning(is_invalid, 
        "Numerical instability detected in " + std::string(operation));
}
```

**Dimension Mismatch Errors**:
```cpp
void validateVectorDimensions(Value* v1, Value* v2, const char* operation) {
    size_t dim1 = getTensorDimension(v1, 0);
    size_t dim2 = getTensorDimension(v2, 0);
    
    if (dim1 != dim2) {
        eshkol_error("%s requires vectors of same dimension, got %zu and %zu",
                    operation, dim1, dim2);
        return nullptr;
    }
}
```

**Helpful Error Messages**:
```cpp
// When gradient fails, provide context
void reportGradientError(const char* reason, Function* func, Value* point) {
    std::string func_name = func->getName().str();
    std::string error_msg = std::string("Gradient computation failed for function ") +
                           func_name + ": " + reason;
    eshkol_error("%s", error_msg.c_str());
    
    // Suggest fixes
    eshkol_info("Hint: Ensure function is differentiable and inputs are numeric");
}
```

**Deliverables Phase 5**:
- ✅ Tensor operation autodiff integration
- ✅ Type inference for all AD operators
- ✅ SIMD optimization for gradient computation
- ✅ Sparse gradient support
- ✅ Comprehensive error handling
- ✅ Runtime numerical stability checks

---

### PHASE 6: Advanced Features (Sessions 33-35)
**Goal**: Complete the system with advanced capabilities

#### Session 33: Lambda & Closure Differentiation

**Closure Capture with Autodiff**:
```scheme
;; Differentiate functions with captured variables
(define make-quadratic
  (lambda (a b c)
    (lambda (x) (+ (* a x x) (* b x) c))))

(define f (make-quadratic 1.0 2.0 3.0))
(derivative f 5.0)  ; Should work with closure
```

**Implementation**:
```cpp
Value* codegenClosureDerivative(Function* closure, Value* x) {
    // Detect captured variables
    // Treat captured variables as constants during differentiation
    // Only differentiate w.r.t. explicit parameter
    
    std::vector<std::string> captured_vars = getCapturedVariables(closure);
    
    // Mark captured variables as constants for AD
    for (const std::string& var : captured_vars) {
        markAsConstant(var);
    }
    
    // Proceed with normal derivative computation
    return codegenDerivative(closure, x);
}
```

**Higher-Order Functions**:
```scheme
;; Differentiate functions that return functions
(define make-derivative
  (lambda (f)
    (lambda (x) (derivative f x))))

(define df/dx (make-derivative (lambda (x) (* x x))))
(df/dx 5.0)  ; Returns 10.0
```

#### Session 34: Automatic Mode Selection

**Compile-Time Analysis**:
```cpp
enum ADMode {
    AD_MODE_FORWARD,   // Best for few inputs, many outputs
    AD_MODE_REVERSE,   // Best for many inputs, few outputs
    AD_MODE_MIXED      // Hybrid approach
};

ADMode selectOptimalMode(Function* func) {
    // Analyze function signature
    size_t num_inputs = getNumInputs(func);
    size_t num_outputs = getNumOutputs(func);
    
    // Forward mode: cost = O(num_outputs)
    // Reverse mode: cost = O(num_inputs)
    
    if (num_outputs > num_inputs) {
        return AD_MODE_REVERSE;  // Cheaper to do one backward pass
    } else {
        return AD_MODE_FORWARD;  // Cheaper to do forward passes
    }
}

Value* codegenAutoGradient(const eshkol_operations_t* op) {
    Function* func = resolveLambdaFunction(op->gradient_op.function);
    ADMode mode = selectOptimalMode(func);
    
    switch (mode) {
        case AD_MODE_FORWARD:
            return codegenGradientForward(op);
        case AD_MODE_REVERSE:
            return codegenGradientReverse(op);
        case AD_MODE_MIXED:
            return codegenGradientMixed(op);
    }
}
```

**Mode-Specific Optimizations**:
```cpp
// Forward mode for Jacobian when m < n
Value* codegenJacobianForward(Function* F, Value* v) {
    // Use forward-mode AD m times (once per output)
    // More efficient when outputs < inputs
}

// Reverse mode for Jacobian when m > n
Value* codegenJacobianReverse(Function* F, Value* v) {
    // Use reverse-mode AD n times (once per input)
    // More efficient when outputs > inputs
}
```

#### Session 35: Final Integration & Testing

**Comprehensive Test Suite**:
```scheme
;; tests/autodiff_comprehensive.esk

;; Phase 0: Symbolic diff
(test-suite "symbolic-differentiation"
  (test (diff (* x x) x) ...)
  (test (diff (sin x) x) ...))

;; Phase 2: Forward-mode AD
(test-suite "forward-mode-ad"
  (test (derivative (lambda (x) (* x x x)) 2.0) 12.0)
  (test (derivative (lambda (x) (exp (* x x))) 1.0) ...))

;; Phase 3: Reverse-mode AD
(test-suite "reverse-mode-ad"
  (test (gradient (lambda (v) (dot v v)) #(1.0 2.0 3.0)) #(2.0 4.0 6.0)))

;; Phase 4: Vector calculus
(test-suite "vector-calculus"
  (test (divergence ...) ...)
  (test (curl ...) ...)
  (test (laplacian ...) ...))
```

**Performance Benchmarks**:
```scheme
;; Benchmark against other systems
(benchmark "gradient-computation"
  :implementations (eshkol pytorch jax)
  :test (lambda () (gradient large-neural-net params)))

(benchmark "jacobian-computation"
  :implementations (eshkol tensorflow jax)
  :test (lambda () (jacobian vector-field point)))
```

**Documentation Updates**:
- Update [`AUTODIFF.md`](../docs/aidocs/AUTODIFF.md) to reflect reality
- Add API reference for all operators
- Create tutorial examples
- Update [`vector_calculus.esk`](../examples/vector_calculus.esk) to use real implementation

**Example Programs**:
```scheme
;; examples/autodiff_neural_network.esk
;; Complete neural network with backpropagation

;; examples/autodiff_physics.esk
;; Physics simulation with force computation

;; examples/autodiff_optimization.esk
;; Gradient descent and Newton's method
```

**Deliverables Phase 6**:
- ✅ Closure differentiation
- ✅ Automatic mode selection
- ✅ Comprehensive test suite
- ✅ Performance benchmarks
- ✅ Complete documentation
- ✅ Example programs

---

## Technical Specifications

### 1. Parser Extensions

**New AST Operation Types** ([`eshkol.h`](../inc/eshkol/eshkol.h:140-155)):
```c
typedef enum {
    // ... existing ...
    ESHKOL_DIFF_OP,          // Existing
    ESHKOL_DERIVATIVE_OP,    // NEW
    ESHKOL_GRADIENT_OP,      // NEW
    ESHKOL_JACOBIAN_OP,      // NEW
    ESHKOL_HESSIAN_OP,       // NEW
    ESHKOL_DIVERGENCE_OP,    // NEW
    ESHKOL_CURL_OP,          // NEW
    ESHKOL_LAPLACIAN_OP,     // NEW
    ESHKOL_DIRECTIONAL_DERIV_OP  // NEW
} eshkol_op_t;
```

**New Operation Structures**:
```c
struct {
    struct eshkol_ast *function;  // Function to differentiate
    struct eshkol_ast *point;     // Evaluation point
    uint8_t mode;                 // 0=forward, 1=reverse, 2=auto
} derivative_op;

struct {
    struct eshkol_ast *function;  // Vector field or scalar field
    struct eshkol_ast *point;     // Evaluation point
} gradient_op, divergence_op, curl_op, laplacian_op;

struct {
    struct eshkol_ast *function;
    struct eshkol_ast *point;
} jacobian_op, hessian_op;

struct {
    struct eshkol_ast *function;
    struct eshkol_ast *point;
    struct eshkol_ast *direction;
} directional_deriv_op;
```

### 2. LLVM Code Generation Functions

**New Functions in** [`llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp):

```cpp
// Dual number operations
Value* packDualNumber(Value* value, Value* derivative);
std::pair<Value*, Value*> unpackDualNumber(Value* dual);
Value* dualAdd(Value* a, Value* b);
Value* dualSub(Value* a, Value* b);
Value* dualMul(Value* a, Value* b);
Value* dualDiv(Value* a, Value* b);
Value* dualSin(Value* a);
Value* dualCos(Value* a);
Value* dualExp(Value* a);
Value* dualLog(Value* a);
Value* dualPow(Value* a, Value* b);
Value* dualNeg(Value* a);

// AD node operations
Value* createADNode(ad_node_type_t type, Value* value, Value* input1, Value* input2);
Value* loadNodeValue(Value* node_ptr);
Value* loadNodeGradient(Value* node_ptr);
void storeNodeGradient(Value* node_ptr, Value* gradient);
Value* loadNodeInput1(Value* node_ptr);
Value* loadNodeInput2(Value* node_ptr);

// Graph operations
Value* codegenADNodeAdd(Value* node_a, Value* node_b);
Value* codegenADNodeMul(Value* node_a, Value* node_b);
// ... all operations

// Tape management
Value* getCurrentTape();
void recordNodeInTape(Value* node_ptr);
void resetTape();

// Main AD operators
Value* codegenDerivative(const eshkol_operations_t* op);
Value* codegenGradient(const eshkol_operations_t* op);
Value* codegenJacobian(const eshkol_operations_t* op);
Value* codegenHessian(const eshkol_operations_t* op);
Value* codegenDivergence(const eshkol_operations_t* op);
Value* codegenCurl(const eshkol_operations_t* op);
Value* codegenLaplacian(const eshkol_operations_t* op);
Value* codegenDirectionalDerivative(const eshkol_operations_t* op);

// Backward pass
void codegenBackward(Value* output_node_ptr);
void propagateGradient(Value* node_ptr);
void accumulateGradient(Value* node_ptr, Value* gradient_to_add);

// Utilities
Value* createTensorFromValues(const std::vector<Value*>& values);
Value* computeJacobianMatrix(Function* F, Value* vector);
Value* computeHessianMatrix(Function* f, Value* vector);
ADMode selectOptimalMode(Function* func);
```

### 3. Type System Integration

**Tagged Value Extensions**:
```cpp
// Pack dual number into tagged value
Value* packDualToTaggedValue(Value* dual) {
    Value* dual_ptr = builder->CreateAlloca(dual_number_type, nullptr, "dual_temp");
    builder->CreateStore(dual, dual_ptr);
    
    // Cast to uint64 for storage in tagged value
    Value* dual_as_ptr = builder->CreatePtrToInt(dual_ptr, Type::getInt64Ty(*context));
    
    return packPtrToTaggedValue(
        builder->CreateIntToPtr(dual_as_ptr, builder->getPtrTy()),
        ESHKOL_VALUE_DUAL_NUMBER
    );
}

// Unpack dual number from tagged value
Value* unpackDualFromTaggedValue(Value* tagged) {
    Value* type = getTaggedValueType(tagged);
    // Verify it's a dual number type
    
    Value* ptr_val = unpackPtrFromTaggedValue(tagged);
    Value* dual_ptr = builder->CreateIntToPtr(ptr_val, 
                                              PointerType::get(dual_number_type, 0));
    
    return builder->CreateLoad(dual_number_type, dual_ptr);
}
```

### 4. Mathematical Correctness

**Differentiation Rules Reference**:

| Function | Derivative | Implementation Status |
|----------|------------|----------------------|
| `c` | `0` | ✅ Phase 0 |
| `x` | `1` | ✅ Phase 0 |
| `f + g` | `f' + g'` | ✅ Phase 0 |
| `f - g` | `f' - g'` | ✅ Phase 0 |
| `f * g` | `f'g + fg'` | 🔧 Fix in Phase 0 |
| `f / g` | `(f'g - fg')/g²` | ❌ Phase 0 |
| `sin(f)` | `cos(f)·f'` | 🔧 Fix in Phase 0 |
| `cos(f)` | `-sin(f)·f'` | 🔧 Fix in Phase 0 |
| `exp(f)` | `exp(f)·f'` | ❌ Phase 0 |
| `log(f)` | `f'/f` | ❌ Phase 0 |
| `f^n` | `n·f^(n-1)·f'` | ❌ Phase 0 |
| `f^g` | `f^g·(g'log(f) + g·f'/f)` | ❌ Phase 2 |

**Vector Calculus Formulas**:

| Operator | Formula | Dimension |
|----------|---------|-----------|
| Gradient | `∇f = [∂f/∂x₁, ..., ∂f/∂xₙ]` | `ℝⁿ → ℝ` → `ℝⁿ` |
| Divergence | `∇·F = Σᵢ ∂Fᵢ/∂xᵢ` | `ℝⁿ → ℝⁿ` → `ℝ` |
| Curl | `(∇×F)ᵢ = εᵢⱼₖ ∂Fₖ/∂xⱼ` | `ℝ³ → ℝ³` → `ℝ³` |
| Laplacian | `∇²f = Σᵢ ∂²f/∂xᵢ²` | `ℝⁿ → ℝ` → `ℝ` |
| Jacobian | `J[i,j] = ∂Fᵢ/∂xⱼ` | `ℝⁿ → ℝᵐ` → `ℝᵐˣⁿ` |
| Hessian | `H[i,j] = ∂²f/∂xᵢ∂xⱼ` | `ℝⁿ → ℝ` → `ℝⁿˣⁿ` |

---

## Testing Strategy

### Unit Tests (Per Phase)

**Phase 0 Tests**:
```scheme
;; tests/autodiff_phase0_test.esk
(define (test-phase-0)
  ;; Type handling
  (assert-equal (diff (* 3.5 x) x) 3.5)
  (assert-type (diff (* 3.5 x) x) 'double)
  
  ;; Product rule
  (assert-equal (diff (* (+ x 1) (+ x 2)) x) ...)
  
  ;; Trig with chain rule
  (assert-approx (diff (sin (* 2 x)) x) ...)
  
  ;; Division
  (assert-equal (diff (/ x 2.0) x) 0.5)
  
  ;; Exp/log
  (assert-approx (diff (exp x) x) ...)
  (assert-approx (diff (log x) x) ...))
```

**Phase 2 Tests**:
```scheme
;; tests/autodiff_phase2_forward_test.esk
(define (test-forward-mode)
  ;; Basic derivative
  (assert-equal (derivative (lambda (x) (* x x)) 5.0) 10.0)
  
  ;; Chain rule
  (assert-approx (derivative (lambda (x) (sin (* x x))) 2.0) ...)
  
  ;; Composition
  (assert-approx (derivative (lambda (x) (exp (sin x))) 1.0) ...))
```

**Phase 3 Tests**:
```scheme
;; tests/autodiff_phase3_reverse_test.esk
(define (test-reverse-mode)
  ;; Vector gradient
  (assert-vector-equal 
    (gradient (lambda (v) (dot v v)) #(1.0 2.0 3.0))
    #(2.0 4.0 6.0))
  
  ;; Multi-variable
  (assert-vector-equal
    (gradient (lambda (x y) (+ (* x x) (* y y))) 2.0 3.0)
    #(4.0 6.0))
  
  ;; Jacobian
  (assert-matrix-equal
    (jacobian (lambda (v) (vector (* (vref v 0) (vref v 1))
                                   (pow (vref v 0) 2)))
              #(2.0 3.0))
    [[3.0 2.0]
     [4.0 0.0]]))
```

**Phase 4 Tests**:
```scheme
;; tests/autodiff_phase4_vector_calculus_test.esk
(define (test-vector-calculus)
  ;; Divergence
  (assert-equal (divergence (lambda (v) v) #(1.0 2.0 3.0)) 3.0)
  
  ;; Curl (3D only)
  (assert-vector-equal (curl ...) ...)
  
  ;; Laplacian
  (assert-equal (laplacian (lambda (v) (dot v v)) #(1.0 2.0 3.0)) 6.0))
```

### Integration Tests

**Neural Network Training**:
```scheme
;; tests/integration/neural_network_ad.esk
(define (test-nn-training)
  (define (forward input weights)
    (matrix-vector-multiply weights input))
  
  (define (loss prediction target)
    (dot (vector-sub prediction target)
         (vector-sub prediction target)))
  
  ;; Compute weight gradients
  (define grad-weights 
    (gradient (lambda (w) (loss (forward input w) target))
              initial-weights))
  
  ;; Update weights
  (define new-weights 
    (vector-sub initial-weights 
                (vector-scale grad-weights learning-rate)))
  
  ;; Verify gradient descent reduces loss
  (assert (< (loss (forward input new-weights) target)
            (loss (forward input initial-weights) target))))
```

**Physics Simulation**:
```scheme
;; tests/integration/physics_simulation_ad.esk
(define (test-physics-ad)
  ;; Potential energy function
  (define (potential q)
    (+ (* 0.5 k (dot q q))      ; Spring potential
       (* -1 m g (vref q 1))))  ; Gravitational potential
  
  ;; Force = -∇U
  (define force (gradient potential position))
  (define force-neg (vector-scale force -1.0))
  
  ;; Verify force is correct
  (assert-vector-approx force-neg expected-force 1e-6))
```

**Optimization Algorithms**:
```scheme
;; tests/integration/optimization_ad.esk
(define (test-optimization)
  ;; Rosenbrock function
  (define (rosenbrock v)
    (let ((x (vref v 0))
          (y (vref v 1)))
      (+ (pow (- 1 x) 2)
         (* 100 (pow (- y (* x x)) 2)))))
  
  ;; Gradient descent
  (define optimized 
    (gradient-descent rosenbrock 
                     #(-1.0 1.0)   ; Start point
                     0.001         ; Learning rate
                     10000))       ; Iterations
  
  ;; Should converge to (1,1)
  (assert-vector-approx optimized #(1.0 1.0) 1e-2))
```

### Performance Benchmarks

**Benchmark Suite**:
```scheme
;; benchmarks/autodiff_performance.esk

(define (benchmark-gradient-sizes)
  (for-each (lambda (n)
    (let* ((f (lambda (v) (dot v v)))
           (v (make-vector n 1.0))
           (start-time (current-time)))
      
      (for i 1 1000
        (gradient f v))
      
      (let ((elapsed (- (current-time) start-time)))
        (printf "n=%d, time=%f ms, throughput=%f grad/sec\n"
                n (* elapsed 1000) (/ 1000 elapsed)))))
    '(10 100 1000 10000)))

(define (benchmark-vs-pytorch)
  ;; Compare performance with PyTorch on same operations
  ;; Expected: Within 2x of PyTorch for reverse-mode
  )
```

**Target Performance**:
- Forward-mode: < 3x overhead vs manual derivatives
- Reverse-mode: < 2x overhead vs PyTorch
- Memory: O(graph_size) for reverse-mode
- Compilation: Add < 15% to total compile time

---

## Success Criteria

### Phase Completion Checklist

**Phase 0: Foundation Fixes** ✅
- [ ] `diff` handles int64 and double correctly
- [ ] No type conflicts in LLVM IR
- [ ] Product rule works for general case
- [ ] Trig functions use proper chain rule
- [ ] Division, exp, log, pow rules implemented
- [ ] All basic diff tests pass

**Phase 1: Type System** ✅  
- [ ] Dual number type defined in [`eshkol.h`](../inc/eshkol/eshkol.h)
- [ ] AD node types defined
- [ ] LLVM IR types created
- [ ] Arena allocation for AD structures
- [ ] Parser recognizes new operators
- [ ] Type system tests pass

**Phase 2: Forward-Mode AD** ✅
- [ ] All math operations support dual numbers
- [ ] `derivative` operator works for scalar functions
- [ ] Chain rule automatic via dual propagation
- [ ] Lambda differentiation works
- [ ] Closure capture handled correctly
- [ ] Forward-mode tests pass

**Phase 3: Reverse-Mode AD** ✅
- [ ] Computational graph construction works
- [ ] Forward pass records operations
- [ ] Backward pass computes gradients correctly
- [ ] `gradient` returns vectors
- [ ] `jacobian` returns matrices
- [ ] `hessian` computes second derivatives
- [ ] Tape management efficient
- [ ] Reverse-mode tests pass

**Phase 4: Vector Calculus** ✅
- [ ] `divergence` works for vector fields
- [ ] `curl` works for 3D fields (with dimension check)
- [ ] `laplacian` works for scalar fields
- [ ] `directional-derivative` works
- [ ] Mathematical correctness verified
- [ ] Vector calculus tests pass

**Phase 5: Integration** ✅
- [ ] Tensor operations work with AD
- [ ] Type inference correct for all operators
- [ ] Performance optimization implemented
- [ ] Error handling comprehensive
- [ ] All integration tests pass
- [ ] Performance benchmarks meet targets

**Phase 6: Polish** ✅
- [ ] Closure differentiation works
- [ ] Automatic mode selection optimal
- [ ] Comprehensive test suite passes (100%)
- [ ] Performance competitive with PyTorch/JAX
- [ ] Documentation complete and accurate
- [ ] Example programs work

### Final System Requirements

**Functional Requirements**:
1. ✅ Forward-mode AD for scalar functions
2. ✅ Reverse-mode AD for vector functions
3. ✅ All standard math operations differentiable
4. ✅ Vector calculus operators (gradient, divergence, curl, laplacian)
5. ✅ Higher-order derivatives (jacobian, hessian)
6. ✅ Lambda and closure differentiation
7. ✅ Type-safe gradients with inference

**Performance Requirements**:
1. Forward-mode: < 3x overhead
2. Reverse-mode: < 2x overhead vs PyTorch
3. Memory: O(graph_size), not O(n²)
4. SIMD vectorization where applicable

**Quality Requirements**:
1. Mathematical correctness (verified against analytical derivatives)
2. Numerical stability (NaN/Inf detection)
3. Comprehensive error messages
4. 100% test coverage for AD operations

---

## Session Schedule & Milestones

### Timeline Overview

```
Month 1: Foundation & Forward-Mode AD
├─ Week 1 (Sessions 1-7)
│  ├─ Phase 0: Fix diff bugs (Sessions 1-3)
│  └─ Phase 1: Type system (Sessions 4-7)
│
├─ Week 2 (Sessions 8-14)
│  ├─ Phase 1: Parser extensions (Session 8)
│  └─ Phase 2: Forward-mode AD (Sessions 9-14)
│
Month 2: Reverse-Mode AD & Vector Calculus
├─ Week 3 (Sessions 15-21)
│  └─ Phase 3: Reverse-mode AD (Sessions 15-21)
│
├─ Week 4 (Sessions 22-28)
│  ├─ Phase 3: Higher-order (Sessions 22-23)
│  └─ Phase 4: Vector calculus (Sessions 24-28)
│
Month 3: Integration & Polish
└─ Week 5 (Sessions 29-35)
   ├─ Phase 5: Integration (Sessions 29-32)
   └─ Phase 6: Advanced features (Sessions 33-35)
```

### Milestone Checkpoints

**Milestone 1** (After Session 8): Foundation Complete
- Type system ready
- Parser supports new operators
- Infrastructure in place
- **Decision Point**: Proceed or adjust?

**Milestone 2** (After Session 14): Forward-Mode Works
- Derivative operator functional
- Dual arithmetic complete
- Basic AD working
- **Decision Point**: Performance acceptable?

**Milestone 3** (After Session 23): Reverse-Mode Works
- Gradient operator functional
- Computational graphs working
- Core AD complete
- **Decision Point**: Continue to vector calculus?

**Milestone 4** (After Session 28): Vector Calculus Complete
- All operators implemented
- Mathematical correctness verified
- **Decision Point**: Polish or ship?

**Milestone 5** (After Session 35): System Complete
- Full autodiff system operational
- Performance benchmarks met
- Documentation complete
- **READY FOR RELEASE**

---

## File Modification Checklist

### Files to Create

**New Header Files**:
- [ ] `lib/autodiff/dual_number.h` - Dual number operations
- [ ] `lib/autodiff/ad_node.h` - AD node structures
- [ ] `lib/autodiff/tape.h` - Tape management
- [ ] `lib/autodiff/gradient_engine.h` - Main gradient computation

**New Implementation Files**:
- [ ] `lib/autodiff/dual_number.cpp` - Dual arithmetic
- [ ] `lib/autodiff/ad_node.cpp` - Node operations
- [ ] `lib/autodiff/tape.cpp` - Tape operations
- [ ] `lib/autodiff/gradient_engine.cpp` - Gradient computation

**New Test Files**:
- [ ] `tests/autodiff/phase0_diff_fixes.esk`
- [ ] `tests/autodiff/phase1_types.esk`
- [ ] `tests/autodiff/phase2_forward.esk`
- [ ] `tests/autodiff/phase3_reverse.esk`
- [ ] `tests/autodiff/phase4_vector_calc.esk`
- [ ] `tests/autodiff/integration_nn.esk`
- [ ] `tests/autodiff/integration_physics.esk`
- [ ] `tests/autodiff/integration_optimization.esk`

### Files to Modify

**Core Headers**:
- [ ] [`inc/eshkol/eshkol.h`](../inc/eshkol/eshkol.h) - Add AD types and operations
- [ ] [`lib/core/arena_memory.h`](../lib/core/arena_memory.h) - Add AD allocators

**Core Implementation**:
- [ ] [`lib/core/arena_memory.cpp`](../lib/core/arena_memory.cpp) - Implement AD allocators
- [ ] [`lib/frontend/parser.cpp`](../lib/frontend/parser.cpp) - Parse new operators
- [ ] [`lib/backend/llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp) - Implement all AD codegen

**Documentation**:
- [ ] [`docs/aidocs/AUTODIFF.md`](../docs/aidocs/AUTODIFF.md) - Update to reflect reality
- [ ] [`docs/type_system/AUTODIFF.md`](../docs/type_system/AUTODIFF.md) - Update type system integration
- [ ] [`docs/aidocs/VECTOR_OPERATIONS.md`](../docs/aidocs/VECTOR_OPERATIONS.md) - Add AD integration

**Examples**:
- [ ] [`examples/vector_calculus.esk`](../examples/vector_calculus.esk) - Use real implementation
- [ ] [`examples/autodiff_example.esk`](../examples/autodiff_example.esk) - Update to working code
- [ ] Create `examples/autodiff_neural_network.esk`
- [ ] Create `examples/autodiff_physics.esk`
- [ ] Create `examples/autodiff_optimization.esk`

---

## Risk Management

### Technical Risks

**Risk 1: LLVM IR Complexity**
- **Probability**: High
- **Impact**: Medium (delays)
- **Mitigation**: Incremental implementation with frequent testing
- **Contingency**: Use simpler runtime-based approach for complex cases

**Risk 2: Type System Conflicts**
- **Probability**: Medium
- **Impact**: High (breaks existing code)
- **Mitigation**: Use existing `tagged_value` system, extend carefully
- **Contingency**: Rollback to safe checkpoint if conflicts arise

**Risk 3: Memory Leaks in Graphs**
- **Probability**: Medium
- **Impact**: High (production blocker)
- **Mitigation**: Arena allocation with automatic cleanup
- **Contingency**: Add manual reset points, memory profiling

**Risk 4: Performance Issues**
- **Probability**: Low
- **Impact**: Medium (user dissatisfaction)
- **Mitigation**: Profile early, optimize hot paths, use SIMD
- **Contingency**: Document performance characteristics, optimize in v1.1

### Schedule Risks

**Risk 1: Underestimated Complexity**
- **Probability**: Medium
- **Impact**: High (timeline slip)
- **Mitigation**: Buffer time, incremental checkpoints
- **Contingency**: Reduce scope to core features only

**Risk 2: Integration Issues**
- **Probability**: Medium
- **Impact**: Medium (rework needed)
- **Mitigation**: Frequent integration testing, keep existing tests passing
- **Contingency**: Isolate AD system, reduce coupling

**Risk 3: Testing Overhead**
- **Probability**: High
- **Impact**: Low (takes time but manageable)
- **Mitigation**: Automate test generation, property-based testing
- **Contingency**: Focus on critical path tests only

---

## Dependencies & Prerequisites

### Required Systems (All ✅)

1. **Tagged Value System** ✅
   - Location: [`eshkol.h:68-78`](../inc/eshkol/eshkol.h:68)
   - Status: Complete
   - Used for: Storing dual numbers and AD nodes

2. **Tensor Operations** ✅
   - Location: [`llvm_codegen.cpp:3782-4991`](../lib/backend/llvm_codegen.cpp:3782)
   - Status: Complete
   - Used for: Vector/matrix return types

3. **Arena Memory** ✅
   - Location: [`arena_memory.h/cpp`](../lib/core/arena_memory.h)
   - Status: Complete
   - Used for: Allocating AD nodes and tapes

4. **Lambda System** ✅
   - Location: [`llvm_codegen.cpp:3650-3780`](../lib/backend/llvm_codegen.cpp:3650)
   - Status: Complete
   - Used for: Differentiating function expressions

5. **Polymorphic Arithmetic** ✅
   - Location: [`llvm_codegen.cpp:1668-2079`](../lib/backend/llvm_codegen.cpp:1668)
   - Status: Complete
   - Used for: Mixed-type gradient computation

### External Dependencies

1. **LLVM Libraries**: Already integrated ✅
2. **Math Library (libm)**: Already linked for sin, cos, exp, log ✅
3. **Standard Library**: For memory allocation ✅

---

## References

### Internal Documentation

1. [`AUTODIFF_TYPE_ANALYSIS.md`](AUTODIFF_TYPE_ANALYSIS.md) - Current state analysis
2. [`docs/aidocs/AUTODIFF.md`](../docs/aidocs/AUTODIFF.md) - AD documentation (aspirational)
3. [`docs/type_system/AUTODIFF.md`](../docs/type_system/AUTODIFF.md) - Type system integration
4. [`docs/tasks/task_008_automatic_differentiation.md`](../docs/tasks/task_008_automatic_differentiation.md) - Original task
5. [`docs/vision/SCIENTIFIC_COMPUTING.md`](../docs/vision/SCIENTIFIC_COMPUTING.md) - Scientific computing vision

### Academic References

1. **"Automatic Differentiation in Machine Learning: a Survey"** - Baydin et al., 2018
   - Comprehensive AD overview
   - Forward and reverse mode comparison
   - Applications in ML

2. **"Efficient Implementation of Automatic Differentiation"** - Griewank & Walther
   - Classic AD textbook
   - Mathematical foundations
   - Implementation strategies

3. **"Automatic Differentiation of Algorithms"** - Rall, 1981
   - Original dual numbers paper
   - Forward-mode AD theory

4. **JAX Documentation** - Google Research
   - Modern AD system design
   - Functional programming approach
   - Composable transformations

5. **PyTorch Autograd Documentation** - Facebook AI
   - Reverse-mode AD in practice
   - Dynamic computational graphs
   - Gradient accumulation

### Code Examples

**From Research**: JAX gradient implementation
```python
# JAX approach (for reference)
def gradient(f):
    def grad_f(x):
        y, vjp_fn = vjp(f, x)
        return vjp_fn(1.0)
    return grad_f
```

**From Research**: PyTorch autograd
```python
# PyTorch approach (for reference)
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()
grad = x.grad  # [2.0, 4.0, 6.0]
```

**Eshkol Target**:
```scheme
;; Our implementation
(define f (lambda (v) (dot v v)))
(define grad (gradient f #(1.0 2.0 3.0)))
;; Returns #(2.0 4.0 6.0) efficiently
```

---

## Appendix: Quick Reference

### Operator Summary

| Operator | Syntax | Input | Output | Mode |
|----------|--------|-------|--------|------|
| `diff` | `(diff expr var)` | expression, symbol | scalar | symbolic |
| `derivative` | `(derivative f x)` | function, scalar | scalar | forward |
| `gradient` | `(gradient f v)` | function, vector | vector | reverse |
| `jacobian` | `(jacobian F v)` | function, vector | matrix | reverse |
| `hessian` | `(hessian f v)` | function, vector | matrix | mixed |
| `divergence` | `(divergence F v)` | function, vector | scalar | reverse |
| `curl` | `(curl F v)` | function, vector(3) | vector(3) | reverse |
| `laplacian` | `(laplacian f v)` | function, vector | scalar | reverse |

### Code Locations

| Component | File | Lines |
|-----------|------|-------|
| Existing diff | [`llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp) | 4993-5116 |
| Tagged values | [`eshkol.h`](../inc/eshkol/eshkol.h) | 41-137 |
| Tensor ops | [`llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp) | 3782-4991 |
| Arena memory | [`arena_memory.h`](../lib/core/arena_memory.h) | 1-200 |
| Parser | [`parser.cpp`](../lib/frontend/parser.cpp) | 195-209 |

### Key Algorithms

**Forward-Mode AD**:
1. Inject dual number at input: `(x, 1.0)`
2. Propagate through operations
3. Extract derivative component
4. Cost: O(num_outputs)

**Reverse-Mode AD**:
1. Forward pass: Build computational graph
2. Initialize output gradient: `∇L = 1.0`
3. Backward pass: Propagate gradients via chain rule
4. Extract input gradients
5. Cost: O(num_inputs)

**Mode Selection**:
- Use forward when: `num_outputs > num_inputs`
- Use reverse when: `num_inputs > num_outputs`
- Use mixed when: Similar dimensions or multiple passes needed

---

## Next Steps

1. **Review & Approve** this implementation plan
2. **Start Phase 0** - Fix existing `diff` bugs (Sessions 1-3)
3. **Build incrementally** - Complete each phase before next
4. **Test continuously** - Keep all existing tests passing
5. **Document progress** - Update this plan as we go

**Ready to begin implementation?** Switch to Code mode to start Phase 0.

---

**END OF IMPLEMENTATION PLAN**