# CURL CRITICAL FIX PLAN

**Priority**: 🔴 **CRITICAL** - Must fix immediately  
**Date**: 2025-11-27  
**Status**: Curl returns scalar `(0)` instead of 3D vector

---

## Problem Statement

**Current Behavior**:
```scheme
(curl F (vector 1.0 2.0 3.0))  ; F(x,y,z) = [0, 0, xy]
→ (0)  ; WRONG - returns scalar!
```

**Required Behavior**:
```scheme
(curl F (vector 1.0 2.0 3.0))
→ #(curl_x curl_y curl_z)  ; 3D vector
```

For `F(x,y,z) = [0, 0, xy]` at `(1,2,3)`:
- curl_x = ∂F₃/∂y - ∂F₂/∂z = ∂(xy)/∂y - 0 = x = 1
- curl_y = ∂F₁/∂z - ∂F₃/∂x = 0 - ∂(xy)/∂x = -y = -2
- curl_z = ∂F₂/∂x - ∂F₁/∂y = 0 - 0 = 0

**Expected**: `#(1 -2 0)` or `#(2 -1 0)` depending on evaluation point

---

## Root Cause Analysis

### Hypothesis 1: Return Type Issue (MOST LIKELY)

**Location**: [`llvm_codegen.cpp:10030-10220`](lib/backend/llvm_codegen.cpp:10030-10220)

**Code Flow**:
```cpp
// Line 10030: Function signature
Value* codegenCurl(const eshkol_operations_t* op)

// Line 10068-10076: Invalid dimension path
BasicBlock* dim_invalid = ...;
builder->SetInsertPoint(dim_invalid);
eshkol_error("Curl only defined for 3D vector fields");
Value* null_result = ConstantInt::get(Type::getInt64Ty(*context), 0);  // ⚠️ SCALAR!
builder->CreateBr(curl_done);

// Line 10105-10109: Null Jacobian path  
builder->SetInsertPoint(jac_invalid);
eshkol_error("Curl: Jacobian returned null, returning null vector");
Value* null_curl = ConstantInt::get(Type::getInt64Ty(*context), 0);  // ⚠️ SCALAR!
builder->CreateBr(curl_done);

// Line 10164-10209: Valid computation path
// Creates 3D result tensor with curl_x, curl_y, curl_z
Value* curl_result = builder->CreatePtrToInt(typed_result_ptr, Type::getInt64Ty(*context));
builder->CreateBr(curl_done);

// Line 10212-10217: PHI node merges 3 paths
builder->SetInsertPoint(curl_done);
PHINode* result_phi = builder->CreatePHI(Type::getInt64Ty(*context), 3, "curl_result");
result_phi->addIncoming(null_result, dim_invalid);      // Path 1: scalar 0
result_phi->addIncoming(null_curl, jac_invalid);        // Path 2: scalar 0
result_phi->addIncoming(curl_result, dim_valid_exit);   // Path 3: tensor pointer
```

**PROBLEM**: PHI node mixes scalars (0) with tensor pointer (curl_result)!

**At Runtime**: Test is taking `jac_invalid` path, returning scalar 0

---

### Investigation: Why Is Jacobian Null?

**Test Function**: `F(x,y,z) = [0, 0, xy]`

**Code**:
```scheme
(define (test-curl-3d)
  (let ((F (lambda (v) 
             (vector 0.0                            ; F₁ = 0
                     0.0                            ; F₂ = 0
                     (* (vref v 0) (vref v 1))))))  ; F₃ = xy
```

**Jacobian Call** (line 10086-10087):
```cpp
Value* jacobian_tagged = codegenJacobian(&jacobian_temp);
```

**Check** (line 10097-10109):
```cpp
Value* jac_is_null = builder->CreateICmpEQ(jacobian_ptr_int,
    ConstantInt::get(Type::getInt64Ty(*context), 0));

builder->CreateCondBr(jac_is_null, jac_invalid, jac_valid);
```

**Runtime Evidence**:
```
JACOBIAN: Output element is NOT AD node (constant), gradient=0.0
```

**Analysis**: 
- Function DOES return a vector: `(vector 0.0 0.0 (* (vref v 0) (vref v 1)))`
- But `0.0` constants don't create AD nodes
- Jacobian computation sees constants and returns zeros
- **Jacobian itself is VALID** (zero Jacobian for constant components)
- But curl interprets valid zero Jacobian as "null"

**Conclusion**: Curl is taking jac_invalid path incorrectly!

---

## The Real Bug

**Location**: Lines 10096-10109

**Current Code**:
```cpp
// WRONG: Treats zero Jacobian pointer as null/invalid
Value* jac_is_null = builder->CreateICmpEQ(jacobian_ptr_int,
    ConstantInt::get(Type::getInt64Ty(*context), 0));

BasicBlock* jac_valid = BasicBlock::Create(*context, "curl_jac_valid", current_func);
BasicBlock* jac_invalid = BasicBlock::Create(*context, "curl_jac_invalid", current_func);

builder->CreateCondBr(jac_is_null, jac_invalid, jac_valid);

builder->SetInsertPoint(jac_invalid);
eshkol_error("Curl: Jacobian returned null, returning null vector");
Value* null_curl = ConstantInt::get(Type::getInt64Ty(*context), 0);  // Returns scalar!
builder->CreateBr(curl_done);
```

**Problem**: 
- Jacobian CAN return a valid tensor of zeros
- Code treats "zero pointer" as invalid
- But Jacobian returns TENSOR_PTR (non-zero), not scalar 0
- Something else is making jac_is_null true

**Investigation Needed**: What is `jacobian_ptr_int` value at runtime?

---

## Fix Strategy

### Step 1: Verify Jacobian Return Type

**Add Runtime Debugging**:
```cpp
// After line 10094
Function* printf_func = function_table["printf"];
if (printf_func) {
    builder->CreateCall(printf_func, {
        codegenString("CURL DEBUG 1: jacobian_ptr_int = %lld\n"),
        jacobian_ptr_int
    });
    
    Value* jacobian_type = getTaggedValueType(jacobian_tagged);
    Value* jacobian_base_type = builder->CreateAnd(jacobian_type,
        ConstantInt::get(Type::getInt8Ty(*context), 0x0F));
    
    builder->CreateCall(printf_func, {
        codegenString("CURL DEBUG 2: jacobian type tag = %d (TENSOR_PTR=6)\n"),
        builder->CreateZExt(jacobian_base_type, Type::getInt32Ty(*context))
    });
}
```

**Run Test**: See what's actually being returned

---

### Step 2: Fix Null Check Logic

**Current** (line 10097-10103):
```cpp
Value* jac_is_null = builder->CreateICmpEQ(jacobian_ptr_int,
    ConstantInt::get(Type::getInt64Ty(*context), 0));

builder->CreateCondBr(jac_is_null, jac_invalid, jac_valid);
```

**Problem**: This checks if pointer IS zero, but Jacobian likely returns non-zero tensor pointer

**Fix**: Check type tag instead of pointer value
```cpp
// Check if jacobian is actually a tensor (not null tagged value)
Value* jacobian_type = getTaggedValueType(jacobian_tagged);
Value* jacobian_base_type = builder->CreateAnd(jacobian_type,
    ConstantInt::get(Type::getInt8Ty(*context), 0x0F));

Value* jac_is_tensor = builder->CreateICmpEQ(jacobian_base_type,
    ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_TENSOR_PTR));

// Branch: if tensor, proceed; if not tensor, return null vector
builder->CreateCondBr(jac_is_tensor, jac_valid, jac_invalid);
```

---

### Step 3: Fix Null Return Values

**Current** (lines 10075 and 10108):
```cpp
// dim_invalid path:
Value* null_result = ConstantInt::get(Type::getInt64Ty(*context), 0);  // SCALAR!

// jac_invalid path:
Value* null_curl = ConstantInt::get(Type::getInt64Ty(*context), 0);    // SCALAR!
```

**Problem**: Returns scalar 0, not null 3D vector

**Fix**: Create actual null tensor (3D vector of zeros)
```cpp
// Helper function: Create null 3D vector
Value* createNull3DVector() {
    Function* malloc_func = function_table["malloc"];
    
    // Allocate tensor structure
    Value* tensor_size = ConstantInt::get(Type::getInt64Ty(*context),
        module->getDataLayout().getTypeAllocSize(tensor_type));
    Value* tensor_ptr = builder->CreateCall(malloc_func, {tensor_size});
    Value* typed_tensor_ptr = builder->CreatePointerCast(tensor_ptr, builder->getPtrTy());
    
    // Set dimensions [3]
    Value* dims_size = ConstantInt::get(Type::getInt64Ty(*context), sizeof(uint64_t));
    Value* dims_ptr = builder->CreateCall(malloc_func, {dims_size});
    Value* typed_dims_ptr = builder->CreatePointerCast(dims_ptr, builder->getPtrTy());
    builder->CreateStore(ConstantInt::get(Type::getInt64Ty(*context), 3), typed_dims_ptr);
    
    // Store tensor metadata
    builder->CreateStore(typed_dims_ptr,
        builder->CreateStructGEP(tensor_type, typed_tensor_ptr, 0));
    builder->CreateStore(ConstantInt::get(Type::getInt64Ty(*context), 1),
        builder->CreateStructGEP(tensor_type, typed_tensor_ptr, 1));
    builder->CreateStore(ConstantInt::get(Type::getInt64Ty(*context), 3),
        builder->CreateStructGEP(tensor_type, typed_tensor_ptr, 3));
    
    // Allocate and zero elements
    Value* elems_size = ConstantInt::get(Type::getInt64Ty(*context), 3 * sizeof(double));
    Value* elems_ptr = builder->CreateCall(malloc_func, {elems_size});
    Value* typed_elems_ptr = builder->CreatePointerCast(elems_ptr, 
        PointerType::get(Type::getDoubleTy(*context), 0));
    
    // Zero all elements
    builder->CreateStore(ConstantFP::get(Type::getDoubleTy(*context), 0.0),
        builder->CreateGEP(Type::getDoubleTy(*context), typed_elems_ptr, ConstantInt::get(Type::getInt64Ty(*context), 0)));
    builder->CreateStore(ConstantFP::get(Type::getDoubleTy(*context), 0.0),
        builder->CreateGEP(Type::getDoubleTy(*context), typed_elems_ptr, ConstantInt::get(Type::getInt64Ty(*context), 1)));
    builder->CreateStore(ConstantFP::get(Type::getDoubleTy(*context), 0.0),
        builder->CreateGEP(Type::getDoubleTy(*context), typed_elems_ptr, ConstantInt::get(Type::getInt64Ty(*context), 2)));
    
    builder->CreateStore(typed_elems_ptr,
        builder->CreateStructGEP(tensor_type, typed_tensor_ptr, 2));
    
    return builder->CreatePtrToInt(typed_tensor_ptr, Type::getInt64Ty(*context));
}

// Use in error paths:
Value* null_result = createNull3DVector();  // Not scalar 0!
Value* null_curl = createNull3DVector();    // Not scalar 0!
```

---

### Step 4: Verify Return Path

**Current** (line 10212-10217):
```cpp
PHINode* result_phi = builder->CreatePHI(Type::getInt64Ty(*context), 3, "curl_result");
result_phi->addIncoming(null_result, dim_invalid);
result_phi->addIncoming(null_curl, jac_invalid);
result_phi->addIncoming(curl_result, dim_valid_exit);

return result_phi;  // Returns i64 (tensor pointer OR scalar)
```

**After Fix** (all paths return tensor pointer):
```cpp
PHINode* result_phi = builder->CreatePHI(Type::getInt64Ty(*context), 3, "curl_result");
result_phi->addIncoming(null_3d_vector, dim_invalid);     // Tensor pointer
result_phi->addIncoming(null_3d_vector, jac_invalid);     // Tensor pointer
result_phi->addIncoming(curl_result, dim_valid_exit);      // Tensor pointer

return result_phi;  // Always returns tensor pointer (consistent!)
```

---

## Implementation Checklist

### Phase 1: Investigation (30 min)
- [ ] Add debug prints to curl function (see Step 1)
- [ ] Run `phase4_real_vector_test.esk` with debugging
- [ ] Identify which path is being taken (dim_invalid vs jac_invalid vs valid)
- [ ] Check jacobian_ptr_int value at runtime

### Phase 2: Fix Null Checks (1 hour)
- [ ] Replace pointer value check with type tag check (Step 2)
- [ ] Verify Jacobian type detection works
- [ ] Test with constant function (should return zero Jacobian, not null)

### Phase 3: Fix Null Returns (2 hours)
- [ ] Create `createNull3DVector()` helper function
- [ ] Replace scalar 0 returns with null vector calls
- [ ] Ensure PHI node always receives tensor pointers
- [ ] Test error paths still work (wrong dimension, invalid function)

### Phase 4: Verification (1 hour)
- [ ] Test curl with working function: `F(x,y,z) = [0, 0, xy]`
- [ ] Verify output is 3D vector (even if all zeros)
- [ ] Test curl with non-3D input (should return null 3D vector, not crash)
- [ ] Check all 3 curl components are calculated correctly

---

## Expected Results After Fix

### Test Case 1: Constant Components
```scheme
(curl (lambda (v) (vector 0.0 0.0 0.0)) (vector 1.0 2.0 3.0))
→ #(0 0 0)  ; Zero curl for constant field
```

### Test Case 2: Linear Function
```scheme
(curl (lambda (v) (vector 0.0 0.0 (* (vref v 0) (vref v 1)))) (vector 1.0 2.0 3.0))
→ #(x -y 0) at (1,2,3) = #(1 -2 0)
```

### Test Case 3: General 3D Field
```scheme
(curl (lambda (v) 
        (vector (vref v 1)              ; F₁ = y
                (vref v 0)              ; F₂ = x  
                (vref v 2)))            ; F₃ = z
      (vector 1.0 2.0 3.0))
→ #(0 0 0)  ; Zero curl for gradient field
```

---

## Code Changes Required

### File: [`lib/backend/llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp)

#### Change 1: Add Helper Function (before `codegenCurl`)
```cpp
// Around line 10025, add:
Value* createNull3DVectorTensor() {
    // (Implementation from Step 3 above)
    // Returns pointer to 3D zero vector
}
```

#### Change 2: Fix Dimension Invalid Path (line 10075)
```cpp
// BEFORE:
Value* null_result = ConstantInt::get(Type::getInt64Ty(*context), 0);

// AFTER:
Value* null_result = createNull3DVectorTensor();
```

#### Change 3: Fix Jacobian Type Check (lines 10096-10103)
```cpp
// BEFORE:
Value* jac_is_null = builder->CreateICmpEQ(jacobian_ptr_int,
    ConstantInt::get(Type::getInt64Ty(*context), 0));

// AFTER:
Value* jacobian_type = getTaggedValueType(jacobian_tagged);
Value* jacobian_base_type = builder->CreateAnd(jacobian_type,
    ConstantInt::get(Type::getInt8Ty(*context), 0x0F));
Value* jac_is_tensor = builder->CreateICmpEQ(jacobian_base_type,
    ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_TENSOR_PTR));

builder->CreateCondBr(jac_is_tensor, jac_valid, jac_invalid);
```

#### Change 4: Fix Jacobian Invalid Path (line 10108)
```cpp
// BEFORE:
Value* null_curl = ConstantInt::get(Type::getInt64Ty(*context), 0);

// AFTER:
Value* null_curl = createNull3DVectorTensor();
```

---

## Testing Plan

### Test 1: Basic Curl
```scheme
;; File: tests/autodiff/test_curl_basic.esk
(define F (lambda (v) (vector 0.0 0.0 (* (vref v 0) (vref v 1)))))
(define curl-result (curl F (vector 1.0 2.0 3.0)))
(display "Curl result: ")
(display curl-result)
(newline)

;; Expected: #(1 -2 0) or similar 3D vector
;; NOT: (0)
```

### Test 2: Zero Curl
```scheme
;; Gradient fields have zero curl
(define G (lambda (v) (vector (vref v 0) (vref v 1) (vref v 2))))
(define zero-curl (curl G (vector 1.0 2.0 3.0)))
(display zero-curl)
;; Expected: #(0 0 0)
```

### Test 3: Invalid Dimension
```scheme
;; 2D input should fail gracefully
(define F2D (lambda (v) (vector (vref v 0) (vref v 1))))
(define invalid-curl (curl F2D (vector 1.0 2.0)))
;; Expected: #(0 0 0) with error message, NOT crash
```

---

## Success Criteria

✅ Curl returns 3D vector (tensor pointer) for valid 3D input  
✅ Curl returns null 3D vector (not scalar) for invalid input  
✅ All 3 curl components calculated correctly  
✅ PHI node type-consistent (all branches return tensor)  
✅ Works with constant function components  
✅ Works with variable function components  
✅ Test `phase4_real_vector_test.esk` passes with vector output  

---

## Time Estimate

- Investigation: 30 min
- Implement fixes: 2 hours
- Testing: 1 hour
- **Total**: 3.5 hours

---

## Priority

🔴 **CRITICAL** - This must be fixed before autodiff can be considered production-ready.

Curl is a fundamental vector calculus operator used in:
- Fluid dynamics (vorticity)
- Electromagnetism (magnetic field)
- Computer graphics (rotation fields)
- Physics simulations

A broken curl makes the entire Phase 4 vector calculus system incomplete.

---

## Related Documents

- [`AUTODIFF_COMPLETE_TEST_ANALYSIS.md`](AUTODIFF_COMPLETE_TEST_ANALYSIS.md) - Full test analysis
- [`RECURSIVE_TENSOR_DISPLAY_ARCHITECTURE.md`](RECURSIVE_TENSOR_DISPLAY_ARCHITECTURE.md) - Display fix (helps curl visualization)
- [`AUTODIFF_TEST_FINDINGS_EXECUTIVE_SUMMARY.md`](AUTODIFF_TEST_FINDINGS_EXECUTIVE_SUMMARY.md) - Overall findings

---

## Next Steps

1. Switch to Code mode
2. Implement fixes in order (Step 1 → Step 4)
3. Test each change incrementally
4. Verify curl produces 3D vector output
5. Update test expectations to validate correct curl values