# Autodiff Type System Analysis - Session 21 Investigation
**Date**: November 17, 2025  
**Investigator**: v1.0-foundation Session 21  
**Status**: 🚨 CRITICAL FINDINGS - Major Scope Discovery

---

## Executive Summary

**CRITICAL DISCOVERY**: The autodiff system described in documentation and execution plan **DOES NOT EXIST**. 

### What Actually Exists:
- ✅ Basic `diff` symbolic differentiation operator (100 lines of code)
- ✅ Handles: constants, variables, +, -, *, sin, cos
- ✅ Returns: scalar int64 only

### What Documentation Claims Exists (But Doesn't):
- ❌ `gradient` function (reverse-mode AD)
- ❌ `d/dx` operator (forward-mode AD)
- ❌ `derivative` function
- ❌ `jacobian`, `divergence`, `curl`, `laplacian` operators
- ❌ Dual numbers system
- ❌ Computational graph system
- ❌ Vector return types
- ❌ Type inference for autodiff
- ❌ Lambda function differentiation

### Impact on v1.0-foundation Timeline:
**Original Plan**: 10 sessions (21-30) to "fix three bugs"  
**Reality**: Need to **BUILD entire autodiff system from scratch** (~20-30 sessions)

---

## Detailed Analysis

### 1. Current Autodiff Implementation

**Location**: [`lib/backend/llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp:4993-5116)

**Lines 4993-5116**: `codegenDiff()` and `differentiate()` functions

#### Implementation Details:

```cpp
// Lines 4993-5011: codegenDiff() - Entry point
Value* codegenDiff(const eshkol_operations_t* op) {
    // Calls differentiate() for symbolic differentiation
    Value* derivative_result = differentiate(op->diff_op.expression, 
                                             op->diff_op.variable);
    // Always returns scalar Value* (int64)
}

// Lines 5013-5040: differentiate() - Core differentiation
Value* differentiate(const eshkol_ast_t* expr, const char* var) {
    switch (expr->type) {
        case ESHKOL_INT64:
        case ESHKOL_DOUBLE:
            return ConstantInt::get(Type::getInt64Ty(*context), 0); // d/dx(c) = 0
        case ESHKOL_VAR:
            if (strcmp(expr->variable.id, var) == 0)
                return ConstantInt::get(Type::getInt64Ty(*context), 1); // d/dx(x) = 1
            else
                return ConstantInt::get(Type::getInt64Ty(*context), 0); // d/dx(y) = 0
        case ESHKOL_OP:
            return differentiateOperation(&expr->operation, var);
    }
}

// Lines 5042-5116: differentiateOperation() - Handle operations
Value* differentiateOperation(const eshkol_operations_t* op, const char* var) {
    // Addition: d/dx(f + g) = f' + g'
    // Subtraction: d/dx(f - g) = f' - g'
    // Product: d/dx(f * g) = f'*g + f*g' (SPECIAL CASE: x*x = 2x)
    // sin: d/dx(sin(f)) = f' (SIMPLIFIED - should be cos(f)*f')
    // cos: d/dx(cos(f)) = -f' (SIMPLIFIED - should be -sin(f)*f')
}
```

#### Supported Operations:
| Operation | Rule | Implementation Status |
|-----------|------|----------------------|
| `diff c x` | 0 | ✅ Complete |
| `diff x x` | 1 | ✅ Complete |
| `diff y x` | 0 | ✅ Complete |
| `diff (+ f g) x` | f' + g' | ✅ Complete |
| `diff (- f g) x` | f' - g' | ✅ Complete |
| `diff (* f g) x` | f'*g + f*g' | ⚠️ Partial (special case for x*x) |
| `diff (sin f) x` | f' | ⚠️ Simplified (should be cos(f)*f') |
| `diff (cos f) x` | -f' | ⚠️ Simplified (should be -sin(f)*f') |

#### Type System:
- **Input**: AST expression + variable name (string)
- **Output**: LLVM `Value*` of type `i64` (always scalar integer)
- **Type Inference**: NONE - always assumes int64
- **Vector Support**: NONE

---

### 2. Documentation vs Reality Gap

#### Documentation Claims (docs/aidocs/AUTODIFF.md):

```scheme
;; DOCUMENTED (but doesn't exist):
(define (f x) (* x x))
(define df (gradient f))              ; ❌ NOT IMPLEMENTED
(df 4)  ; Should return derivative

(gradient (lambda (v) (dot v v)) #(1 2 3))  ; ❌ NOT IMPLEMENTED

(derivative g 2.0)                    ; ❌ NOT IMPLEMENTED
```

#### What Actually Works:

```scheme
;; REALITY (only this works):
(diff (* x x) x)  ; Returns 2*x at compile time (symbolic)
(diff (+ x 3) x)  ; Returns 1
(diff 42 x)       ; Returns 0
```

#### Examples Analysis:

| File | Claims | Reality |
|------|--------|---------|
| `examples/vector_calculus.esk` | `gradient`, `derivative`, `jacobian`, `divergence`, `curl`, `laplacian` | ❌ NONE exist |
| `examples/autodiff_example.esk` | Dual numbers, forward/reverse mode | ❌ NOT implemented |
| `examples/test_autodiff.esk` | Basic `diff` only | ✅ Works |
| `docs/aidocs/AUTODIFF.md` | Complete AD system | ❌ Aspirational |

---

### 3. Bug Analysis

#### SCH-006: Type Inference Incomplete

**Original Description**: "Type inference doesn't work for autodiff functions"

**Reality**: 
- There IS NO type inference to be incomplete
- The `diff` operator always returns `i64`
- No inference needed because no runtime autodiff exists

**Root Cause**: Not a bug - a missing feature

**Example That Would Fail** (if gradient existed):
```scheme
(define f (lambda (x) (* x x)))
(gradient f #(1.0 2.0 3.0))  ; Would need to infer: vector -> scalar, so gradient returns vector
```

**Fix Strategy**: 
1. Implement `gradient` function
2. Add type inference: `inputType -> returnType` implies gradient returns `inputType`
3. Handle scalar→scalar, vector→scalar cases

---

#### SCH-007: Vector Return Types Not Handled

**Original Description**: "Gradient should return vector but returns scalar"

**Reality**:
- There IS NO `gradient` function at all
- No vector operations in autodiff code
- No LLVM vector type construction for autodiff

**Root Cause**: Not a bug - the entire feature is missing

**What Needs Implementation**:
```cpp
// Pseudocode for missing gradient function:
Value* codegenGradient(const eshkol_operations_t* op) {
    // 1. Get function to differentiate
    // 2. Get input vector
    // 3. Compute partial derivatives for each component
    // 4. Pack into LLVM vector type
    // 5. Return vector
    //
    // Status: NOT IMPLEMENTED AT ALL
}
```

**Fix Strategy**:
1. Implement computational graph system
2. Implement reverse-mode AD (backpropagation)
3. Implement LLVM vector return types
4. Add gradient operator to AST
5. Wire up in codegen

---

#### SCH-008: Type Conflicts in Generated Code

**Original Description**: "Type conflicts between scalar and vector in LLVM IR"

**Reality**:
- This might be a real issue with `diff` operator
- `diff` returns `i64`, but might be used with `double` expressions
- Example: `(diff (* 3.5 x) x)` should handle double arithmetic

**Root Cause**: 
- `differentiate()` always returns `ConstantInt::get(Type::getInt64Ty()...)`
- No handling of double-typed expressions
- No type promotion in differentiation

**Example That Fails**:
```scheme
(define x 3.5)
(diff (* x x) x)  ; Tries to multiply i64 with double, type conflict
```

**Actual Code** (lines 5084-5092):
```cpp
// Always creates i64 constants, even for double expressions!
Value* two = ConstantInt::get(Type::getInt64Ty(*context), 2);
return builder->CreateMul(two, f); // f might be double!
```

**Fix Strategy**:
1. Detect expression type (int64 vs double)
2. Return appropriately typed constants
3. Use `ConstantFP::get()` for double derivatives
4. Properly handle mixed arithmetic in derivatives

---

### 4. What Documentation Describes (But Doesn't Exist)

#### Forward-Mode AD (Dual Numbers)

**Documented** (lines 29-46 in AUTODIFF.md):
```scheme
(define x (dual 3.0 1.0))  ; Value + derivative
(define result (f x))
(define derivative (dual-derivative result))
```

**Implementation Status**: ❌ **NOT IMPLEMENTED**
- No `dual` type
- No dual number arithmetic
- No `dual-derivative` extraction

---

#### Reverse-Mode AD (Computational Graph)

**Documented** (lines 48-65 in AUTODIFF.md):
```scheme
(define result (f (node 2.0) (node 3.0)))
(backward result)
(define dx (gradient (node 2.0)))
```

**Implementation Status**: ❌ **NOT IMPLEMENTED**
- No `node` type
- No computational graph
- No `backward` pass
- No gradient accumulation

---

#### Vector Calculus Operators

**Documented** (vector_calculus.esk):
```scheme
(gradient f v)      ; ∇f
(divergence F v)    ; ∇·F
(curl F v)          ; ∇×F
(laplacian f v)     ; ∇²f
(derivative g x)    ; df/dx
(jacobian F v)      ; Jacobian matrix
```

**Implementation Status**: ❌ **ALL NOT IMPLEMENTED**

---

### 5. Scope of Work Required

#### To Actually Implement Autodiff (Proper Estimate):

**Phase 1: Foundation (8-10 sessions)**
1. Design type system for autodiff
2. Implement dual number type
3. Implement computational graph nodes
4. Add autodiff operators to AST
5. Implement type inference rules

**Phase 2: Forward-Mode AD (6-8 sessions)**
1. Dual number arithmetic (+, -, *, /, sin, cos, exp, log)
2. `derivative` function implementation
3. Chain rule for composition
4. Lambda differentiation

**Phase 3: Reverse-Mode AD (8-10 sessions)**
1. Computational graph construction
2. Backward pass implementation
3. `gradient` function
4. Gradient accumulation
5. Memory management for graph

**Phase 4: Vector Operations (6-8 sessions)**
1. Vector return types
2. `jacobian` matrix construction
3. `divergence`, `curl`, `laplacian`
4. Matrix operations

**TOTAL**: ~28-36 sessions (vs 10 planned)

---

### 6. Critical Decision Point

#### Option A: Build Full Autodiff System
- **Time**: 28-36 sessions (triple original estimate)
- **Benefit**: Meet documentation claims
- **Risk**: Delays v1.0-foundation by 3-4 weeks
- **Complexity**: High - requires new type systems

#### Option B: Fix Only `diff` Operator
- **Time**: 2-4 sessions (as planned)
- **Benefit**: Stay on schedule
- **Fix**: Make `diff` work with mixed types correctly
- **Document**: Clearly state modern autodiff is planned for v1.1

#### Option C: Hybrid Approach
- **Time**: 6-8 sessions
- **Implement**: `diff` + basic `gradient` for vectors
- **Skip**: Dual numbers, computational graph, jacobian
- **Benefit**: Partial autodiff capability
- **Risk**: Still behind schedule

---

### 7. Actual Bugs in `diff` Implementation

#### Bug 1: No Double Support
**Line 5021**: Always returns `ConstantInt::get(Type::getInt64Ty(*context), 0)`
**Should**: Check if expression is double, return `ConstantFP::get()`

#### Bug 2: Incorrect Product Rule
**Lines 5076-5086**: Special case for `x*x` only
**Should**: General product rule `f'*g + f*g'`

#### Bug 3: Simplified Trig Functions
**Lines 5099-5111**: `d/dx(sin(f)) = f'` (wrong!)
**Should**: `d/dx(sin(f)) = cos(f) * f'`

#### Bug 4: No Composition Support
**Missing**: Can't differentiate `(f (g x))`
**Should**: Implement chain rule

---

### 8. Test Case Analysis

#### Test Case 1: What Currently Works

```scheme
;; examples/test_corrected_diff.esk - WORKS
(diff 42 x)        ; Returns 0 ✅
(diff x x)         ; Returns 1 ✅
(diff y x)         ; Returns 0 ✅
(diff (+ x 3) x)   ; Returns 1 ✅
(diff (* x x) x)   ; Returns 2*x ✅ (special case)
```

#### Test Case 2: What Doesn't Exist

```scheme
;; examples/vector_calculus.esk - FAILS (functions don't exist)
(gradient f v)     ; ❌ Undefined function
(derivative g x)   ; ❌ Undefined function  
(jacobian F v)     ; ❌ Undefined function
```

#### Test Case 3: Type Issues (SCH-008)

```scheme
;; This will fail - type mismatch
(define x 3.5)               ; double
(diff (* x x) x)             ; Returns i64 2*3.5, but i64 can't hold 7.0
                             ; Type conflict: i64 constant * double variable

;; This should work but might not
(diff (* 3.5 x) x)           ; Need to return double constant 3.5
```

---

### 9. Minimal Reproducible Test Cases

#### SCH-006: Type Inference (Actually: Missing Feature)

```scheme
;; tests/autodiff_debug/test_sch_006_type_inference.esk
;; This test will FAIL because gradient doesn't exist

(define f (lambda (x) (* x x)))

;; This should infer:
;; - f: double -> double
;; - gradient: (double -> double) -> (double -> double)  
;; - Result: derivative function
(define df (gradient f))  ; ❌ gradient not implemented
(display (df 5.0))        ; Should return 10.0

;; EXPECTED: Type inference works
;; ACTUAL: Compile error - 'gradient' undefined
```

#### SCH-007: Vector Returns (Actually: Missing Feature)

```scheme
;; tests/autodiff_debug/test_sch_007_vector_returns.esk
;; This test will FAIL because gradient doesn't exist

;; Gradient of dot product: ∇(v·v) = 2v
(define f (lambda (v) (dot v v)))
(define grad (gradient f #(1.0 2.0 3.0)))  ; ❌ gradient not implemented

;; EXPECTED: grad = #(2.0 4.0 6.0)
;; ACTUAL: Compile error - 'gradient' undefined
```

#### SCH-008: Type Conflicts (Actually: Real Bug in diff)

```scheme
;; tests/autodiff_debug/test_sch_008_type_conflicts.esk
;; This test will FAIL with type mismatch

(define x 3.5)  ; double variable
(define y 2.0)

;; This should work but causes type conflict
(display (diff (* x x) x))  ; diff returns i64, but x is double
                            ; Type conflict in CreateMul(i64_constant, double_var)

;; EXPECTED: Returns 7.0 (2 * 3.5)
;; ACTUAL: LLVM IR verification error - type mismatch
```

---

### 10. Root Cause Summary

### SCH-006 Root Cause:
**NOT A BUG** - It's a missing feature. The `gradient` function doesn't exist, so there's no type inference to be incomplete.

**Actual Issue**: Documentation promises a modern autodiff system that was never built.

### SCH-007 Root Cause:
**NOT A BUG** - It's a missing feature. No vector return capability exists because there's no gradient/jacobian implementation.

**Actual Issue**: No reverse-mode AD, no computational graph, no vector returns.

### SCH-008 Root Cause:
**REAL BUG** - The `diff` operator doesn't properly handle double-typed expressions.

**Location**: Lines 5084-5092 in [`llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp:5084)

**Issue**: Always creates `i64` constants even when differentiating double expressions:
```cpp
Value* two = ConstantInt::get(Type::getInt64Ty(*context), 2);  // Always i64!
return builder->CreateMul(two, f);  // f might be double - TYPE CONFLICT
```

---

### 11. Fix Strategies

#### Strategy A: Build Full Autodiff System (28-36 sessions)

**Scope**: Implement everything documented

**Implementation Plan**:
1. **Design Phase** (2 sessions)
   - Type system design for dual numbers
   - Computational graph architecture
   - Integration with existing type system

2. **Forward-Mode AD** (8-10 sessions)
   - Dual number type in LLVM IR
   - Overloaded arithmetic operators
   - `derivative` function
   - Chain rule implementation
   - Lambda differentiation

3. **Reverse-Mode AD** (10-12 sessions)
   - Computational graph nodes
   - Graph construction during forward pass
   - Backward pass (backpropagation)
   - `gradient` function
   - Gradient accumulation
   - Memory management

4. **Vector Calculus** (6-8 sessions)
   - Vector return types
   - `jacobian` matrix construction
   - `divergence` operator
   - `curl` operator
   - `laplacian` operator

5. **Testing** (2-4 sessions)
   - Comprehensive test suite
   - Performance benchmarks
   - Integration with existing code

**Pros**:
- ✅ Meets documentation claims
- ✅ Full scientific computing capability
- ✅ Competitive with other systems

**Cons**:
- ❌ 3x timeline overrun
- ❌ Delays v1.0-foundation by 3-4 weeks
- ❌ High complexity and risk

---

#### Strategy B: Fix Only `diff` + Document Reality (2-4 sessions)

**Scope**: Make existing `diff` work correctly, document limitations

**Implementation Plan**:
1. **Fix SCH-008** (2 sessions)
   - Detect expression type (int64 vs double)
   - Return appropriately typed constants
   - Handle mixed arithmetic properly
   - Fix trig function derivatives

2. **Documentation Update** (1 session)
   - Update AUTODIFF.md to reflect reality
   - Remove aspirational claims
   - Document `diff` operator only
   - Add "Future Work" section for gradient/etc.

3. **Update Examples** (1 session)
   - Remove examples that use undefined functions
   - Keep only `diff` examples
   - Add realistic autodiff examples

**Pros**:
- ✅ Stays on schedule
- ✅ Accurate documentation
- ✅ `diff` works reliably
- ✅ Clear about limitations

**Cons**:
- ❌ No modern autodiff in v1.0-foundation
- ❌ Limited scientific computing appeal
- ❌ Deferred feature to v1.1+

---

#### Strategy C: Implement Basic gradient Only (6-8 sessions)

**Scope**: Add `gradient` for vector→scalar functions only

**Implementation Plan**:
1. **Gradient Function** (4 sessions)
   - Implement `gradient` for simple functions
   - Compute partial derivatives
   - Return LLVM vector type
   - Basic type inference

2. **Fix diff** (2 sessions)  
   - Same as Strategy B

3. **Testing** (2 sessions)
   - Test gradient on simple cases
   - Integration tests

**Pros**:
- ✅ Partial autodiff capability
- ✅ Meets minimum expectations
- ✅ Foundation for future work

**Cons**:
- ❌ 20% timeline overrun (tolerable)
- ❌ Still limited (no jacobian, etc.)
- ❌ Incomplete system

---

### 12. Recommended Path Forward

**RECOMMENDATION**: **Strategy B** (Fix `diff` + Document Reality)

**Rationale**:
1. **Timeline**: Critical path analysis shows Week 3 is already constrained
2. **Risk**: Building autodiff from scratch is too risky for v1.0-foundation
3. **Value**: Honest documentation > broken promises
4. **Future**: Can build full autodiff in v1.1-scientific-computing

**Session Allocation**:
- **Session 21-22**: Investigation (COMPLETE ✅)
- **Session 23-24**: Fix SCH-008 in `diff` operator
- **Session 25-26**: Update documentation to reality
- **Session 27-28**: Update examples (remove non-working ones)
- **Session 29-30**: Test existing `diff` thoroughly

**Impact on v1.0-foundation**:
- ✅ Still ships on time
- ✅ Feature matrix accurate
- ✅ No false promises
- ⚠️ "Autodiff: 30%" instead of "95%" on feature matrix

---

### 13. Type Flow Analysis (for diff operator)

#### Current Flow:
```
User Code: (diff (* x x) x)
    ↓
Parser: Creates DIFF_OP AST node
    ↓
codegenDiff(): Calls differentiate()
    ↓
differentiate(): Analyzes AST recursively
    ↓
Returns: ConstantInt::get(Type::getInt64Ty(), result)
    ↓
Type: Always i64, regardless of expression type
```

#### Fixed Flow (Strategy B):
```
User Code: (diff (* x 3.5) x)
    ↓
Parser: Creates DIFF_OP AST node
    ↓
codegenDiff(): Calls differentiate() with type detection
    ↓
detectExpressionType(): Returns DOUBLE
    ↓
differentiate(): Analyzes AST, detects result type
    ↓
Returns: ConstantFP::get(Type::getDoubleTy(), 3.5)
    ↓
Type: Matches expression type (double in this case)
```

---

### 14. Immediate Actions Required

#### For Session 22 (Completion):
1. ✅ Create this analysis document
2. ⏳ Create minimal test cases (in progress)
3. ⏳ Create tests/autodiff_debug/ directory
4. ⏳ Test and verify issues reproduce

#### For Decision Meeting:
1. Present three strategies to stakeholders
2. Get approval for chosen path (recommend Strategy B)
3. Update execution plan based on decision
4. Adjust v1.0-foundation scope if needed

---

### 15. Deliverables

#### Session 21-22 Deliverables:
- ✅ This document: `docs/AUTODIFF_TYPE_ANALYSIS.md`
- ⏳ Test directory: `tests/autodiff_debug/`
- ⏳ SCH-006 test: `test_sch_006_type_inference.esk`
- ⏳ SCH-007 test: `test_sch_007_vector_returns.esk`
- ⏳ SCH-008 test: `test_sch_008_type_conflicts.esk`
- ⏳ Test results: Verify all three issues

#### Next Steps (Session 23+):
**IF Strategy B Approved**:
- Session 23-24: Fix SCH-008 (type conflicts in diff)
- Session 25-26: Update documentation
- Session 27-28: Clean up examples
- Session 29-30: Comprehensive diff testing

**IF Strategy A Approved**:
- Extend timeline by 18-26 sessions
- Redesign autodiff architecture
- Build from scratch

**IF Strategy C Approved**:
- Extend timeline by 4-6 sessions  
- Implement basic gradient only
- Document limitations

---

### 16. Code Locations Reference

#### Autodiff Implementation:
- [`lib/backend/llvm_codegen.cpp:4993-5116`](../lib/backend/llvm_codegen.cpp:4993) - `codegenDiff()`, `differentiate()`, `differentiateOperation()`

#### Documentation (Aspirational):
- [`docs/aidocs/AUTODIFF.md`](../docs/aidocs/AUTODIFF.md) - Claims full AD system
- [`examples/vector_calculus.esk`](../examples/vector_calculus.esk) - Uses undefined operators
- [`examples/autodiff_example.esk`](../examples/autodiff_example.esk) - Uses dual numbers (don't exist)

#### Working Examples:
- [`examples/test_autodiff.esk`](../examples/test_autodiff.esk) - Basic `diff` only
- [`examples/test_corrected_diff.esk`](../examples/test_corrected_diff.esk) - Simple derivatives

---

### 17. Conclusion

**The "three autodiff bugs" are actually**:
1. **SCH-006**: Not a bug - missing feature (`gradient` function)
2. **SCH-007**: Not a bug - missing feature (vector returns)  
3. **SCH-008**: **Real bug** - `diff` doesn't handle double types

**The execution plan assumptions were incorrect**:
- Assumed modern autodiff exists ❌
- Assumed we're fixing bugs ❌
- Assumed 10 sessions sufficient ❌

**Actual situation**:
- Basic symbolic `diff` exists ✅
- No forward/reverse-mode AD ❌
- Documentation is aspirational ❌
- Need ~30 sessions to build full system ❌

**Recommendation**: 
Accept reality, fix `diff` properly (2-4 sessions), document limitations honestly, defer modern autodiff to v1.1-scientific-computing.

---

**Analysis Status**: COMPLETE ✅  
**Next Session**: Decision on strategy + test case creation  
**Critical Path Impact**: HIGH - affects entire Month 2 planning

---

**Related Documents**:
- [`V1_0_FOUNDATION_EXECUTION_PLAN.md`](V1_0_FOUNDATION_EXECUTION_PLAN.md) - Original plan (needs revision)
- [`V1_0_FOUNDATION_CRITICAL_PATH.md`](V1_0_FOUNDATION_CRITICAL_PATH.md) - Timeline (at risk)
- [`V1_0_FOUNDATION_ARCHITECTURE_DIAGRAMS.md`](V1_0_FOUNDATION_ARCHITECTURE_DIAGRAMS.md) - Architecture

**END OF ANALYSIS**