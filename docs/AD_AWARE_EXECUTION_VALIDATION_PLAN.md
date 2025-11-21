# AD-Aware Execution Validation Plan
**Date**: November 20, 2025  
**Status**: CRITICAL for v1.0-foundation Release  
**Context**: Phase 3/4 Tensor-AD Integration  
**Target**: Systematic validation of computational graph construction

---

## Executive Summary

This plan provides **systematic validation** of Eshkol's AD-aware execution mode, which enables automatic differentiation through computational graphs built during normal program execution. The system is **mostly implemented** but requires comprehensive testing to ensure all polymorphic operations correctly detect and propagate AD node types.

### Current Implementation Status

**Infrastructure Complete** ✅:
- Tagged value type system with `ESHKOL_VALUE_AD_NODE_PTR` (type 5)
- Dual numbers for forward-mode AD (type 4)
- AD node structures (`ad_node_t`) with gradient storage
- Tape management (`ad_tape_t`) for operation recording
- Backward pass implementation ([`codegenBackward()`](../lib/backend/llvm_codegen.cpp:6918))
- Gradient propagation rules for all operations

**Polymorphic Operations Are AD-Aware** ✅:
- [`polymorphicAdd()`](../lib/backend/llvm_codegen.cpp:1861) - Lines 1876-1880 detect AD nodes
- [`polymorphicSub()`](../lib/backend/llvm_codegen.cpp:2057) - Lines 2070-2075 detect AD nodes  
- [`polymorphicMul()`](../lib/backend/llvm_codegen.cpp:2244) - Lines 2257-2262 detect AD nodes
- [`polymorphicDiv()`](../lib/backend/llvm_codegen.cpp:2431) - Lines 2445-2450 detect AD nodes
- [`codegenVectorRef()`](../lib/backend/llvm_codegen.cpp:5059) - Lines 5096-5123 preserve AD nodes

**Critical Gap** ⚠️:
- Gradient operator has safety check (lines 7567-7588) that skips backward pass if function returns non-AD scalar
- This prevents gradient computation for simple test functions like `(lambda (v) 0)`
- Need validation that **real** AD-using functions work correctly

---

## Validation Architecture

### Three-Layer Testing Strategy

```
┌─────────────────────────────────────────────────────────┐
│              Layer 1: Unit Tests                         │
│  Test individual polymorphic operations with AD nodes   │
│  ✓ Verify type detection (AD_NODE_PTR recognized)      │
│  ✓ Verify graph construction (nodes recorded to tape)   │
│  ✓ Verify backward pass (gradients propagate)           │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│         Layer 2: Integration Tests                       │
│  Test lambda→tensor→gradient full pipeline              │
│  ✓ Lambda with vref creates AD nodes                    │
│  ✓ Arithmetic on AD nodes builds graph                  │
│  ✓ Gradient extraction returns correct vectors          │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│        Layer 3: Production Tests                         │
│  Test real scientific computing use cases                │
│  ✓ Neural network gradient descent                      │
│  ✓ Physics simulations with forces                      │
│  ✓ Optimization algorithms (gradient-based)             │
└─────────────────────────────────────────────────────────┘
```

---

## Layer 1: Unit Tests for Polymorphic Operations

### Test 1.1: Addition with AD Nodes

**File**: `tests/autodiff/unit_test_add_ad_nodes.esk`

```scheme
;; Test that polymorphicAdd detects AD nodes and builds graph correctly

;; Setup: Create two AD variable nodes with values 3.0 and 5.0
;; Expectation: Addition should create AD_NODE_ADD and record to tape

(define (test-add-ad-nodes)
  (display "Test 1.1: Addition with AD nodes")
  (newline)
  
  ;; This test will use gradient to force AD node creation
  (define f (lambda (v) (+ (vref v 0) (vref v 1))))
  (define grad (gradient f (vector 3.0 5.0)))
  
  ;; Expected: gradient of (x + y) is [1.0, 1.0]
  (display "Gradient result: ")
  (display grad)
  (newline)
  
  ;; Validation
  (display "Expected: vector(1.0, 1.0)")
  (newline)
  0)
```

**Validation Points**:
- ✓ `polymorphicAdd` detects `ESHKOL_VALUE_AD_NODE_PTR` (line 1876)
- ✓ Calls `recordADNodeBinary(2, ...)` for `AD_NODE_ADD` (line 1959)
- ✓ Returns AD node pointer packed in tagged value
- ✓ Backward pass propagates gradient correctly (line 7053-7056)

---

### Test 1.2: Multiplication with AD Nodes (Product Rule)

**File**: `tests/autodiff/unit_test_mul_ad_nodes.esk`

```scheme
;; Test product rule: d/dx(x*y) = y, d/dy(x*y) = x

(define (test-mul-ad-nodes)
  (display "Test 1.2: Multiplication with AD nodes (product rule)")
  (newline)
  
  (define f (lambda (v) (* (vref v 0) (vref v 1))))
  (define grad (gradient f (vector 3.0 5.0)))
  
  ;; Expected: gradient of (x * y) at (3,5) is [5.0, 3.0]
  ;; ∂(x*y)/∂x = y = 5.0
  ;; ∂(x*y)/∂y = x = 3.0
  (display "Gradient result: ")
  (display grad)
  (newline)
  
  (display "Expected: vector(5.0, 3.0)")
  (newline)
  0)
```

**Validation Points**:
- ✓ `polymorphicMul` detects AD nodes (line 2257-2262)
- ✓ Calls `recordADNodeBinary(4, ...)` for `AD_NODE_MUL` (line 2337)
- ✓ Backward pass applies product rule correctly (lines 7080-7091)
- ✓ Gradient matches analytical derivative

---

### Test 1.3: Vector Reference Preserves AD Nodes

**File**: `tests/autodiff/unit_test_vref_ad_preservation.esk`

```scheme
;; Critical test: vref must preserve AD node pointers, not convert to scalars

(define (test-vref-preserves-ad)
  (display "Test 1.3: vref preserves AD node types")
  (newline)
  
  ;; Function that extracts and uses vector components
  (define f (lambda (v) 
    (let ((x (vref v 0))
          (y (vref v 1)))
      (+ (* x x) (* y y)))))
  
  (define grad (gradient f (vector 2.0 3.0)))
  
  ;; Expected: gradient of (x² + y²) at (2,3) is [2x, 2y] = [4.0, 6.0]
  (display "Gradient result: ")
  (display grad)
  (newline)
  
  (display "Expected: vector(4.0, 6.0)")
  (newline)
  0)
```

**Validation Points**:
- ✓ `codegenVectorRef` checks for AD nodes (lines 5096-5099)
- ✓ Returns `ESHKOL_VALUE_AD_NODE_PTR` packed in tagged value (line 5110)
- ✓ AD nodes flow through let bindings
- ✓ Arithmetic operations build complete computational graph

---

## Layer 2: Integration Tests (Lambda→Tensor→Gradient)

### Test 2.1: Complete Gradient Pipeline

**File**: `tests/autodiff/integration_gradient_pipeline.esk`

```scheme
;; End-to-end test of the full gradient computation pipeline
;; Tests: parser → lambda → tensor → vref → arithmetic → backward pass

(define (test-complete-pipeline)
  (display "Test 2.1: Complete gradient computation pipeline")
  (newline)
  
  ;; Test 1: Simple quadratic form f(v) = v·v
  (define quadratic (lambda (v)
    (+ (* (vref v 0) (vref v 0))
       (* (vref v 1) (vref v 1))
       (* (vref v 2) (vref v 2)))))
  
  (define grad1 (gradient quadratic (vector 1.0 2.0 3.0)))
  (display "Gradient of v·v at (1,2,3): ")
  (display grad1)
  (newline)
  (display "Expected: vector(2.0, 4.0, 6.0)")
  (newline)
  (newline)
  
  ;; Test 2: Linear function f(v) = 2x + 3y
  (define linear (lambda (v)
    (+ (* 2.0 (vref v 0))
       (* 3.0 (vref v 1)))))
  
  (define grad2 (gradient linear (vector 5.0 7.0)))
  (display "Gradient of (2x + 3y) at (5,7): ")
  (display grad2)
  (newline)
  (display "Expected: vector(2.0, 3.0)")
  (newline)
  (newline)
  
  ;; Test 3: Non-linear f(v) = x²y
  (define nonlinear (lambda (v)
    (* (* (vref v 0) (vref v 0))
       (vref v 1))))
  
  (define grad3 (gradient nonlinear (vector 3.0 4.0)))
  (display "Gradient of x²y at (3,4): ")
  (display grad3)
  (newline)
  (display "Expected: vector(24.0, 9.0)")  ;; ∂(x²y)/∂x = 2xy = 24, ∂(x²y)/∂y = x² = 9
  (newline)
  
  0)
```

**Critical Validation Points**:
1. **Parser** correctly builds `ESHKOL_GRADIENT_OP` AST (lines 1100-1160)
2. **Lambda resolution** finds function in symbol table (lines 10153-10296)
3. **Tensor creation** from vector literal works (lines 4820-4897)
4. **Variable nodes** created with input values (lines 7468-7488)
5. **vref** extracts AD nodes from tensor (lines 5086-5123)
6. **Arithmetic** builds computational graph via `recordADNodeBinary` (lines 1959, 2337)
7. **Backward pass** propagates gradients correctly (lines 6918-7151)
8. **Result extraction** returns proper gradient vector (lines 7614-7619)

---

### Test 2.2: Nested Operations Build Complete Graph

**File**: `tests/autodiff/integration_nested_ops.esk`

```scheme
;; Test that nested operations build full computational graph

(define (test-nested-graph-construction)
  (display "Test 2.2: Nested operations build complete graph")
  (newline)
  
  ;; f(x,y,z) = sin(x²) + cos(y²) + exp(z)
  ;; Complex nested operations to test graph depth
  (define complex-f (lambda (v)
    (+ (+ (sin (* (vref v 0) (vref v 0)))
          (cos (* (vref v 1) (vref v 1))))
       (exp (vref v 2)))))
  
  (define grad (gradient complex-f (vector 1.0 2.0 3.0)))
  
  (display "Gradient of complex function: ")
  (display grad)
  (newline)
  
  ;; Expected gradients:
  ;; ∂f/∂x = cos(x²) * 2x at x=1: cos(1) * 2 ≈ 1.08
  ;; ∂f/∂y = -sin(y²) * 2y at y=2: -sin(4) * 4 ≈ 3.03
  ;; ∂f/∂z = exp(z) at z=3: exp(3) ≈ 20.09
  (display "Expected: vector(~1.08, ~3.03, ~20.09)")
  (newline)
  
  0)
```

**Graph Construction Flow**:
```
vref(v,0) → AD_NODE_VARIABLE(x)
  ↓
x * x → AD_NODE_MUL → stores both inputs
  ↓
sin(x²) → AD_NODE_SIN → stores input
  ↓
+ operations → AD_NODE_ADD chain
  ↓
Backward pass traverses in reverse order
```

---

## Layer 3: Production Tests

### Test 3.1: Neural Network Gradient Descent

**File**: `tests/autodiff/production_neural_network.esk`

```scheme
;; Simplified neural network training with gradient descent

(define (test-neural-network-training)
  (display "Test 3.1: Neural network gradient descent")
  (newline)
  
  ;; Single-layer network: y = w₁x₁ + w₂x₂
  ;; Loss: (y - target)²
  ;; Gradient descent: w := w - α∇L
  
  (define input (vector 1.0 2.0))      ;; x = [1, 2]
  (define target 7.0)                  ;; Expected: 1*2 + 2*2.5 = 7
  (define weights (vector 0.0 0.0))    ;; Initial: w = [0, 0]
  (define learning-rate 0.01)
  
  ;; Forward pass: compute prediction
  (define (forward w)
    (+ (* (vref w 0) (vref input 0))
       (* (vref w 1) (vref input 1))))
  
  ;; Loss function: L(w) = (forward(w) - target)²
  (define (loss w)
    (let ((pred (forward w))
          (error (- pred target)))
      (* error error)))
  
  ;; Compute gradient of loss w.r.t. weights
  (define grad (gradient loss weights))
  
  (display "Gradient of loss at initial weights: ")
  (display grad)
  (newline)
  
  ;; Expected: gradient should point in direction to reduce loss
  ;; For (0,0) → (2,2.5): gradient = dL/dw = 2(pred-target)*x
  ;; At w=(0,0): pred=0, gradient = 2(-7)*[1,2] = [-14, -28]
  (display "Expected: vector(~-14, ~-28)")
  (newline)
  
  0)
```

---

## Critical Issue Analysis

### Issue 1: Safety Check May Skip Valid Computations

**Location**: [`codegenGradient()`](../lib/backend/llvm_codegen.cpp:7567-7588)

**Current Code**:
```cpp
// Line 7567-7571: Check if output is valid AD node pointer
Value* output_is_valid_ptr = builder->CreateICmpUGT(output_node_int,
    ConstantInt::get(Type::getInt64Ty(*context), 1000));

BasicBlock* has_valid_output = ...;
BasicBlock* invalid_output = ...;

builder->CreateCondBr(output_is_valid_ptr, has_valid_output, invalid_output);
```

**Problem**: Functions like `(lambda (v) 0)` return scalar `0`, not AD node, causing check to fail

**Impact**:
- `test_ad_aware_execution.esk` tests will fail (return zero gradients)
- Simple test functions won't build computational graphs
- Only functions that ALREADY use AD operations will work

**Solution Options**:

**Option A**: Remove safety check (trust function to build graph)
```cpp
// Just run backward pass unconditionally
codegenBackward(output_node_ptr, partial_tape);
```

**Option B**: Make functions AD-aware by default
```cpp
// In lambda codegen, detect if called from gradient context
// Automatically create AD variable nodes for parameters
```

**Option C**: Better heuristic for valid AD nodes
```cpp
// Check if output is AD node by type tag, not pointer value
Value* output_type = getTaggedValueType(output_tagged);
Value* is_ad_node = builder->CreateICmpEQ(output_type, 
    ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_AD_NODE_PTR));
```

**Recommended**: Option C - Type-based detection is more robust

---

### Issue 2: Lambda Functions Don't Auto-Create AD Nodes

**Current Behavior**:
- When `gradient` calls a lambda, it passes vector of AD nodes
- Lambda's `vref` operations extract these AD nodes ✅
- But simple arithmetic like `(+ 1 2)` creates regular scalars, not AD nodes ❌

**Example Failure**:
```scheme
(define f (lambda (v) 3))  ;; Returns constant 3, not AD node
(gradient f (vector 1.0))  ;; Fails: no graph built, gradient = 0
```

**Root Cause**:
- Constants like `3` are codegen'd as `ConstantInt`, not wrapped in AD nodes
- Lambda doesn't know it's being called in gradient context
- No automatic wrapping of scalars → AD nodes

**Solution**: Context-aware constant creation
```cpp
// In codegenAST for ESHKOL_INT64/ESHKOL_DOUBLE:
if (current_tape_ptr != nullptr) {
    // We're in AD context - wrap constant in AD node
    Value* const_val = ConstantFP::get(...);
    return packPtrToTaggedValue(createADConstant(const_val), ESHKOL_VALUE_AD_NODE_PTR);
} else {
    // Normal execution - return raw value
    return ConstantInt::get(...);
}
```

---

## Systematic Validation Test Suite

### Suite 1: Type Detection Tests

**File**: `tests/autodiff/validation_01_type_detection.esk`

```scheme
;; Validate that each polymorphic operation correctly detects AD node types

(define (test-type-detection)
  (display "=== Suite 1: Type Detection Validation ===")
  (newline)
  
  ;; Test 1: Addition detects AD nodes
  (display "Test: Addition with mixed types")
  (newline)
  (define f1 (lambda (v) (+ (vref v 0) 2.0)))
  (gradient f1 (vector 3.0))  ;; Should build graph: vref→AD_NODE, 2.0→AD_CONST, +→AD_ADD
  
  ;; Test 2: Multiplication detects AD nodes  
  (display "Test: Multiplication with mixed types")
  (newline)
  (define f2 (lambda (v) (* (vref v 0) (vref v 0))))
  (gradient f2 (vector 4.0))  ;; Should apply product rule: 2x = 8.0
  
  ;; Test 3: Division detects AD nodes
  (display "Test: Division with AD nodes")
  (newline)
  (define f3 (lambda (v) (/ (vref v 0) 2.0)))
  (gradient f3 (vector 6.0))  ;; Should return 0.5 (quotient rule)
  
  ;; Test 4: Subtraction detects AD nodes
  (display "Test: Subtraction with AD nodes")
  (newline)
  (define f4 (lambda (v) (- (vref v 0) (vref v 1))))
  (gradient f4 (vector 5.0 3.0))  ;; Should return [1.0, -1.0]
  
  (display "All type detection tests completed")
  (newline)
  0)
```

---

### Suite 2: Graph Construction Tests

**File**: `tests/autodiff/validation_02_graph_construction.esk`

```scheme
;; Validate that operations correctly build computational graphs

(define (test-graph-construction)
  (display "=== Suite 2: Graph Construction Validation ===")
  (newline)
  
  ;; Test 1: Linear operations build simple graph
  (display "Test: Linear graph (x + y + z)")
  (newline)
  (define linear (lambda (v)
    (+ (+ (vref v 0) (vref v 1)) (vref v 2))))
  (gradient linear (vector 1.0 2.0 3.0))  ;; Expected: [1, 1, 1]
  
  ;; Test 2: Chain of multiplications
  (display "Test: Multiplicative chain (x * y * z)")
  (newline)
  (define multiplicative (lambda (v)
    (* (* (vref v 0) (vref v 1)) (vref v 2))))
  (gradient multiplicative (vector 2.0 3.0 4.0))  ;; Expected: [yz, xz, xy] = [12, 8, 6]
  
  ;; Test 3: Nested operations
  (display "Test: Nested operations ((x+y)*(x-y))")
  (newline)
  (define nested (lambda (v)
    (* (+ (vref v 0) (vref v 1))
       (- (vref v 0) (vref v 1)))))
  (gradient nested (vector 5.0 3.0))  ;; Expected: [2x, -2y] = [10, -6]
  
  (display "All graph construction tests completed")
  (newline)
  0)
```

---

### Suite 3: Backward Pass Tests

**File**: `tests/autodiff/validation_03_backward_pass.esk`

```scheme
;; Validate gradient propagation through backward pass

(define (test-backward-propagation)
  (display "=== Suite 3: Backward Pass Validation ===")
  (newline)
  
  ;; Test 1: Single variable (sanity check)
  (display "Test: Single variable x²")
  (newline)
  (define f1 (lambda (v) (* (vref v 0) (vref v 0))))
  (define grad1 (gradient f1 (vector 5.0)))
  (display "Result: ")
  (display grad1)
  (display " Expected: vector(10.0)")
  (newline)
  
  ;; Test 2: Chain rule through multiple operations
  (display "Test: Chain rule sin(x²)")
  (newline)
  (define f2 (lambda (v) (sin (* (vref v 0) (vref v 0)))))
  (define grad2 (gradient f2 (vector 2.0)))
  ;; ∂sin(x²)/∂x = cos(x²)*2x at x=2: cos(4)*4
  (display "Result: ")
  (display grad2)
  (newline)
  
  ;; Test 3: Multiple inputs with interactions
  (display "Test: Multi-input x²+xy+y²")
  (newline)
  (define f3 (lambda (v)
    (+ (+ (* (vref v 0) (vref v 0))
          (* (vref v 0) (vref v 1)))
       (* (vref v 1) (vref v 1)))))
  (define grad3 (gradient f3 (vector 3.0 4.0)))
  ;; ∂f/∂x = 2x+y = 10, ∂f/∂y = x+2y = 11
  (display "Result: ")
  (display grad3)
  (display " Expected: vector(10.0, 11.0)")
  (newline)
  
  (display "All backward pass tests completed")
  (newline)
  0)
```

---

## Tensor-AD Integration Validation

### Critical Path Tests

**Test 4.1**: Vector operations in gradient context
**Test 4.2**: Tensor element access preserves AD types
**Test 4.3**: Dot product with gradients
**Test 4.4**: Matrix operations (future phase 4)

**File**: `tests/autodiff/validation_04_tensor_ad_integration.esk`

```scheme
;; Validate tensor operations work correctly with AD nodes

(define (test-tensor-ad-integration)
  (display "=== Suite 4: Tensor-AD Integration ===")
  (newline)
  
  ;; Test 1: Vector dot product gradient
  (display "Test: Gradient of dot product")
  (newline)
  (define dot-grad (gradient 
    (lambda (v) (+ (* (vref v 0) (vref v 0))
                   (* (vref v 1) (vref v 1))))
    (vector 3.0 4.0)))
  (display "∇(v·v) at (3,4): ")
  (display dot-grad)
  (display " Expected: vector(6.0, 8.0)")
  (newline)
  
  ;; Test 2: Weighted sum gradient
  (display "Test: Gradient of weighted sum")
  (newline)
  (define weights-grad (gradient
    (lambda (w) (+ (* 2.0 (vref w 0))
                   (* 3.0 (vref w 1))
                   (* 5.0 (vref w 2))))
    (vector 1.0 1.0 1.0)))
  (display "∇(2w₁+3w₂+5w₃): ")
  (display weights-grad)
  (display " Expected: vector(2.0, 3.0, 5.0)")
  (newline)
  
  (display "Tensor-AD integration tests completed")
  (newline)
  0)
```

---

## Known Issues and Workarounds

### SCH-006: Type Inference for Autodiff Incomplete

**Symptom**: Type conflicts when mixing int64 and double in derivatives

**Current Workaround**: Type-aware helpers in `differentiateOperation()` (lines 6145-6280)

**Permanent Fix Needed**:
- Unify type inference across symbolic diff and runtime AD
- Ensure `createTypedMul/Add/Sub/Div` handles all AD node combinations
- Add compile-time type checking for gradient operations

---

### SCH-007: Vector Return Types Not Handled

**Symptom**: Functions returning vectors can't be properly differentiated

**Current Status**: 
- Jacobian operator EXISTS (lines 7652-7983)
- Returns m×n matrix correctly ✅
- But may have issues with vector-valued function outputs

**Test Needed**:
```scheme
;; Vector-valued function F: ℝ² → ℝ²
(define F (lambda (v)
  (vector (* (vref v 0) (vref v 1))     ;; F₁ = xy
          (* (vref v 0) (vref v 0)))))   ;; F₂ = x²

(jacobian F (vector 2.0 3.0))
;; Expected Jacobian:
;; J = [[∂F₁/∂x  ∂F₁/∂y]  = [[y   x ]  = [[3  2]
;;      [∂F₂/∂x  ∂F₂/∂y]]    [2x  0 ]]    [4  0]]
```

---

### SCH-008: Type Conflicts in Generated Code

**Symptom**: LLVM IR verification errors due to type mismatches

**Known Locations**:
- Mixed int64/double in derivatives (fixed in Phase 0)
- Tagged value struct vs scalar type confusion
- Pointer type mismatches in AD node access

**Validation**: Run LLVM verification on all gradient tests
```bash
./build/eshkol-run tests/autodiff/validation_*.esk 2>&1 | grep "LLVM module verification"
```

---

## Implementation Checklist

### Pre-Release Validation (v1.0-foundation)

- [ ] **Unit Tests** (Suite 1): All polymorphic ops handle AD nodes ✓
- [ ] **Integration Tests** (Suite 2): Full gradient pipeline works ✓
- [ ] **Production Tests** (Suite 3): Real use cases validated ✓
- [ ] **Tensor-AD Tests** (Suite 4): Vector operations work with gradients ✓

### Code Fixes Required

**Priority 1: Fix Gradient Safety Check**
- [ ] Location: [`llvm_codegen.cpp:7567-7588`](../lib/backend/llvm_codegen.cpp:7567)
- [ ] Change: Use type-based detection instead of pointer value heuristic
- [ ] Test: `test_ad_aware_execution.esk` should pass

**Priority 2: Context-Aware Constants**
- [ ] Location: [`codegenAST()`](../lib/backend/llvm_codegen.cpp:2829) cases for INT64/DOUBLE
- [ ] Change: Check `current_tape_ptr != nullptr` and wrap in AD nodes
- [ ] Test: `(gradient (lambda (v) 5) (vector 1.0))` should return [0.0]

**Priority 3: Lambda AD Context Propagation**
- [ ] Location: [`codegenLambda()`](../lib/backend/llvm_codegen.cpp:4580)
- [ ] Change: Mark lambda as "AD-aware" when called from gradient
- [ ] Test: All arithmetic in lambda body should use AD nodes

---

## Validation Methodology

### Phase 1: Structural Validation (Static Analysis)

**Check 1**: All polymorphic operations have AD node detection
```bash
grep -n "ESHKOL_VALUE_AD_NODE_PTR" lib/backend/llvm_codegen.cpp
```
**Expected**: 4+ occurrences in polymorphic{Add,Sub,Mul,Div}

**Check 2**: All operations call recordADNode functions
```bash
grep -n "recordADNodeBinary\|recordADNodeUnary" lib/backend/llvm_codegen.cpp
```
**Expected**: Calls in all AD paths of polymorphic operations

**Check 3**: Backward pass handles all operation types
```bash
grep -n "AD_NODE_ADD\|AD_NODE_MUL\|AD_NODE_DIV" lib/backend/llvm_codegen.cpp
```
**Expected**: Switch cases in `propagateGradient()` for each type

---

### Phase 2: Runtime Validation (Dynamic Testing)

**Test Protocol**:
1. Build test: `cmake --build build`
2. Run each validation suite sequentially
3. Check LLVM IR verification passes
4. Verify numerical results match analytical derivatives
5. Check no memory leaks with valgrind

**Numerical Tolerance**: `|computed - expected| < 1e-6`

**Memory Check**:
```bash
valgrind --leak-check=full ./build/eshkol-run tests/autodiff/validation_*.esk
```

---

### Phase 3: Mathematical Validation

Compare computed gradients against analytical solutions:

| Function | Point | Analytical Gradient | Test File |
|----------|-------|---------------------|-----------|
| f(x) = x² | x=5 | 10 | validation_01 |
| f(v) = v·v | (1,2,3) | (2,4,6) | validation_02 |
| f(x,y) = xy | (3,5) | (5,3) | validation_02 |
| f(v) = sin(v₀²) | v₀=2 | 4cos(4) | validation_03 |
| f(w) = 2w₁+3w₂ | (0,0) | (2,3) | validation_04 |

---

## Test Execution Plan

### Step 1: Run Existing AD Tests
```bash
cd /Users/tyr/Desktop/eshkol
./build/eshkol-run tests/autodiff/test_ad_aware_execution.esk
./build/eshkol-run tests/autodiff/phase3_complete_test.esk
./build/eshkol-run tests/autodiff/phase4_simple_test.esk
```

**Expected**: Some tests may fail due to safety check issue

---

### Step 2: Create and Run Validation Suite
```bash
# Create validation tests
# (Files specified above in Suite 1-4)

# Run systematically
for test in tests/autodiff/validation_*.esk; do
    echo "Running $test"
    ./build/eshkol-run "$test" || echo "FAILED: $test"
done
```

---

### Step 3: Fix Issues and Re-validate
```bash
# After code fixes
cmake --build build --clean-first
bash scripts/run_all_tests.sh
```

**Success Criteria**:
- 100% test pass rate (66/66 existing + new validation tests)
- All gradients match analytical solutions within tolerance
- No LLVM IR verification errors
- No memory leaks

---

## Critical Path for v1.0-foundation

### Immediate Actions (This Week)

1. **Create validation test files** (1-2 hours)
   - validation_01_type_detection.esk
   - validation_02_graph_construction.esk
   - validation_03_backward_pass.esk
   - validation_04_tensor_ad_integration.esk

2. **Run baseline tests** (30 min)
   - Document current failures
   - Identify specific failure modes

3. **Fix gradient safety check** (1 hour)
   - Implement Option C (type-based detection)
   - Test with simple functions

4. **Validate fixes** (2 hours)
   - Re-run all validation suites
   - Verify numerical correctness
   - Check memory safety

---

## HoTT Integration Notes (Future Work)

### Current System (Runtime Tagged Values)
```cpp
struct eshkol_tagged_value_t {
    uint8_t type;        // Runtime type tag
    uint8_t flags;       // Exactness flags
    uint16_t reserved;
    union data;          // 8-byte value storage
};
```

### Future System (Compile-Time Proofs + Runtime Erasure)
```cpp
template<TypeCode CarCode, TypeCode CdrCode>
struct HottConsCell {
    // Compile-time type information
    using car_type = Interpret<CarCode>;
    using cdr_type = Interpret<CdrCode>;
    using safety_proof = TypeSafetyProof<CarCode, CdrCode>;
    
    // Runtime data only (proofs erased)
    car_type car_data;
    cdr_type cdr_data;
};
```

**Key Difference**: HoTT moves type checking to compile-time with template metaprogramming, eliminating runtime overhead while providing stronger guarantees.

**For v1.0-foundation**: Continue with current tagged value system - it works well and is fully implemented.

**For v1.1+**: Gradual migration to HoTT with backward compatibility layer (16-21 weeks estimated).

---

## Success Metrics

### Quantitative Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Test Pass Rate | 100% | 84.8% (56/66) | 🟡 Needs work |
| AD Tests Pass | 100% | ~60% est. | 🔴 Needs validation |
| Gradient Accuracy | <1e-6 error | Unknown | ⚪ Needs testing |
| Memory Leaks | 0 | Unknown | ⚪ Needs valgrind |

### Qualitative Metrics

- [ ] All polymorphic operations detect AD nodes correctly
- [ ] Computational graphs build for all arithmetic combinations
- [ ] Backward pass propagates gradients accurately
- [ ] Tensor operations integrate seamlessly with AD
- [ ] Type inference works for mixed int64/double gradients
- [ ] Vector return types handled correctly
- [ ] No LLVM IR verification errors

---

## Documentation Requirements

### For v1.0-foundation Release

**Update Required**:
1. [`docs/aidocs/AUTODIFF.md`](../docs/aidocs/AUTODIFF.md) - Reflect actual implementation status
2. [`README.md`](../README.md) - Accurate feature claims about AD capabilities
3. [`examples/autodiff_example.esk`](../examples/autodiff_example.esk) - Working examples only

**New Documentation**:
- `AD_AWARE_EXECUTION_GUIDE.md` - How the system works internally
- `GRADIENT_COMPUTATION_TUTORIAL.md` - User-facing examples
- `AD_TROUBLESHOOTING.md` - Common issues and solutions

---

## References

### Internal Documentation
- [`AUTODIFF_COMPLETE_IMPLEMENTATION_PLAN.md`](AUTODIFF_COMPLETE_IMPLEMENTATION_PLAN.md) - Overall plan
- [`AUTODIFF_PHASE3_PRODUCTION_IMPLEMENTATION.md`](AUTODIFF_PHASE3_PRODUCTION_IMPLEMENTATION.md) - Phase 3 specs
- [`AUTODIFF_TYPE_ANALYSIS.md`](AUTODIFF_TYPE_ANALYSIS.md) - Type system integration

### Code Locations
- **Polymorphic Arithmetic**: Lines 1861-2604 (Add/Sub/Mul/Div with AD detection)
- **AD Node Management**: Lines 6598-6913 (Create/record/load/store AD nodes)
- **Backward Pass**: Lines 6918-7151 (Gradient propagation)
- **Gradient Operator**: Lines 7223-7647 (Full gradient computation)
- **Vector Reference**: Lines 5059-5126 (AD-aware vref implementation)

### Test Files
- **Existing**: `tests/autodiff/test_ad_aware_execution.esk`
- **Phase 3**: `tests/autodiff/phase3_complete_test.esk`
- **Phase 4**: `tests/autodiff/phase4_simple_test.esk`
- **New**: validation_01 through validation_04 (to be created)

---

## Conclusion

Eshkol's AD-aware execution system is **architecturally sound** with proper type detection, graph construction, and gradient propagation. The main issues are:

1. **Safety checks too conservative** - preventing valid computations
2. **Context propagation incomplete** - lambdas don't auto-create AD nodes
3. **Test coverage gaps** - need systematic validation

**Estimated Effort to Fix**: 8-12 hours
**Risk Level**: LOW - infrastructure exists, just needs refinement
**Blocker for v1.0**: YES - AD is a core feature claim

**Recommendation**: Create validation tests first, then fix issues, then re-validate. This ensures nothing breaks during fixes.

---

**Document Status**: Active validation plan  
**Next Action**: Create validation test suite
**Owner**: Eshkol Development Team