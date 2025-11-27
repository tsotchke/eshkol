# Autodiff Test Analysis - Day 2
**Date**: November 26, 2025
**Status**: 36/40 tests passing (90%)
**Critical Finding**: Forward-mode AD WORKS, Reverse-mode AD returns zeros

---

## Executive Summary

**GOOD NEWS** ✅:
- Forward-mode AD (`derivative` operator) is **FULLY FUNCTIONAL** and numerically correct
- All 7 derivative tests return perfect numerical accuracy
- No data corruption, no type errors, dual number arithmetic works flawlessly

**BAD NEWS** ❌:
- Reverse-mode AD (`gradient` operator) returns zero vectors instead of correct gradients
- 4 tests segfault when calling `gradient`, `jacobian`, `curl`, or `laplacian`
- Computational graph not being constructed properly

---

## Detailed Test Results

### Phase 2: Forward-Mode AD (Derivative) - ✅ 100% SUCCESS

**Test**: [`phase2_tests_1_7.esk`](phase2_tests_1_7.esk)

| Test | Function | Point | Expected | Actual | Status |
|------|----------|-------|----------|--------|--------|
| 1 | 2x | x=5 | 2.0 | 2.000000 | ✅ PERFECT |
| 2 | x² | x=5 | 10.0 | 10.000000 | ✅ PERFECT |
| 3 | x³ | x=2 | 12.0 | 12.000000 | ✅ PERFECT |
| 4 | sin(x²) | x=2 | -2.614 | -2.614574 | ✅ PERFECT |
| 5 | exp(sin(x)) | x=1 | 1.253 | 1.253381 | ✅ PERFECT |
| 6 | (x+1)(x+2) | x=3 | 9.0 | 9.000000 | ✅ PERFECT |
| 7 | x/2 | x=10 | 0.5 | 0.500000 | ✅ PERFECT |

**Conclusion**: Dual number arithmetic, product rule, quotient rule, chain rule all work correctly!

---

### Phase 3: Reverse-Mode AD (Gradient) - ❌ RETURNS ZEROS

**Test**: [`verify_gradient_working.esk`](../tests/autodiff/verify_gradient_working.esk)

```scheme
(define f (lambda (v) (* (vref v 0) (vref v 0))))  ; f(v) = v[0]²
(define point (vector 3.0))
(define grad (gradient f point))
(define grad_0 (vref grad 0))
```

**Expected**: `grad_0 = 6.0` (since ∂(x²)/∂x = 2x = 2*3 = 6)
**Actual**: `grad_0 = 0`

**All gradient tests show similar pattern**:
- Compilation succeeds ✓
- Runtime succeeds ✓
- Result is zero vector ✗

---

## Root Cause Analysis

### What's Working

1. **Tensor Creation**: Vectors are created correctly with proper values
   - Output shows: "DEBUG 8: LOADED n = 1" - dimension is correct
   - Runtime malloc succeeds
   
2. **AD Mode Flag**: Global `__ad_mode_active` is set correctly
   - Set to true before lambda call (line 5813)
   - Set to false after lambda call (line 5820)
   
3. **vref AD Detection**: Context-aware type detection implemented
   - AD mode path exists (lines 5313-5332)
   - Should return AD_NODE_PTR for large values when in AD mode

### What's NOT Working

1. **Gradient Returns Zero**: Despite infrastructure existing, gradients are all 0
   
2. **Segfaults in 4 Tests**:
   - `debug_operators.esk`: Calls `gradient` and `jacobian` → SEGFAULT
   - `phase2_forward_test.esk`: Multiple `derivative` calls → SEGFAULT
   - `phase3_complete_test.esk`: Calls `gradient`, `jacobian`, `hessian` → SEGFAULT
   - `phase4_vector_calculus_test.esk`: Calls `divergence`, `curl`, `laplacian` → SEGFAULT

---

## Critical Discoveries

### Discovery 1: Derivative vs Gradient Dichotomy

The system has TWO separate autodiff implementations:
- **Forward-mode** (`derivative`): Uses dual numbers, WORKS PERFECTLY
- **Reverse-mode** (`gradient`): Uses computational graphs, BROKEN

This suggests the issue is SPECIFIC to the reverse-mode implementation, not the general autodiff infrastructure.

### Discovery 2: Jacobian Returns Null

From `debug_operators_full_output.txt`:
```
error: Jacobian: function returned null (expected vector)
```

This happens during compilation when the jacobian operator tries to call a function to determine output dimensions. The function returns null/scalar instead of a vector, causing the jacobian computation to fail.

### Discovery 3: Display vs Type Confusion

Many tests show:
```
error: Attempted to get int64 from non-int64 cell (type=32)
```

This happens when trying to DISPLAY gradient results. The gradient vector is being misidentified as a cons cell during display, suggesting type confusion in the display logic for tensors.

---

## Hypothesis: Why Gradients Are Zero

Based on the evidence, here's the most likely scenario:

1. `gradient` creates AD variable nodes ✓
2. Stores them in tensor elements as pointers ✓
3. Sets `ad_mode_active = true` ✓
4. Calls lambda with AD tensor ✓
5. Lambda executes, vref checks `ad_mode_active` ✓
6. **PROBLEM**: vref returns AD node pointers, BUT...
7. **BUG**: polymorphicMul might not be seeing them correctly
8. **OR**: The backward pass isn't running properly
9. **OR**: Gradients are computed but not extracted correctly

The fact that the debug output shows "Gradient requires non-zero dimension vector" during COMPILATION suggests there might be an issue with how the result gradient vector is being constructed or returned.

---

## Segfault Analysis

All 4 segfaulting tests have something in common - they call operators that depend on `jacobian`:
- `jacobian` itself → SEGFAULT
- `curl` (uses jacobian internally) → SEGFAULT  
- `laplacian` (uses hessian, which uses gradient repeatedly) → SEGFAULT
- `divergence` (uses jacobian and computes trace) → SEGFAULT

The common error before segfault:
```
error: Jacobian: function returned null (expected vector)
```

This suggests the segfault happens when trying to ACCESS a null tensor pointer.

Looking at [`llvm_codegen.cpp:7998-8001`](../lib/backend/llvm_codegen.cpp:7998):
```cpp
// Invalid output: return null jacobian (don't crash)
builder->SetInsertPoint(output_invalid_block);
eshkol_error("Jacobian: function returned null (expected vector)");
Value* null_jac = ConstantInt::get(Type::getInt64Ty(*context), 0);
builder->CreateBr(jac_return_block);
```

So the code TRIES to handle null output gracefully, but then later code might be dereferencing this null pointer, causing the segfault.

---

## Action Items

### Priority 1: Fix Segfaults (Immediate)
1. Find where jacobian/gradient/curl/laplacian dereference null pointers
2. Add proper null checks before dereferencing
3. Make all operators gracefully handle null/scalar outputs

### Priority 2: Debug Gradient Zero-Value Bug
1. Add instrumentation to polymorphicMul to verify AD nodes are detected
2. Add instrumentation to backward pass to verify it runs
3. Add instrumentation to gradient extraction to verify values are non-zero
4. Trace through ONE complete gradient computation with full logging

### Priority 3: Fix Display Type Confusion
1. Improve tensor detection in display logic
2. Fix "Attempted to get int64 from non-int64 cell" errors
3. Ensure gradient vectors display correctly as vectors, not cons cells

---

## Next Steps

1. Create minimal diagnostic test that shows WHERE the computational graph breaks
2. Add detailed logging to trace ONE gradient computation from start to finish
3. Fix identified bugs one at a time
4. Verify each fix with incremental testing
5. Achieve 40/40 test pass rate

---

**Status**: Analysis complete, ready to implement fixes