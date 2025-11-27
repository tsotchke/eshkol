# CURL N-DIMENSIONAL FIX SPECIFICATION

**Date**: 2025-11-27  
**Status**: 🔴 CRITICAL - Curl returns scalar `(0)` instead of n-D vector  
**Scope**: N-dimensional solution (NOT hardcoded 3D)

---

## Problem Analysis

### Current Behavior
```
error: Jacobian: function returned null (expected vector)
error: Curl: Jacobian returned null, returning null vector
Test 3 - Curl of F(x,y,z)=[0,0,xy] computed
(0)  ← WRONG: Returns scalar 0
```

### Expected Behavior
```
Test 3 - Curl of F(x,y,z)=[0,0,xy] computed
#(curl_x curl_y curl_z)  ← Correct: Returns 3D vector as tensor
```

### Jacobian Display Format (Working Correctly)
```
#((3 2) (4 0))  ← Jacobian as tensor containing row lists
```

**Key Insight**: Jacobian NOW returns a tensor whose elements are list pointers (nested structure). This is the correct n-dimensional format!

---

## Root Cause

### Issue 1: Jacobian Type Validation Failure

**Location**: [`llvm_codegen.cpp:10524-10530`](../lib/backend/llvm_codegen.cpp:10524-10530)

**Current Code**:
```cpp
Value* jac_is_null = builder->CreateICmpEQ(jacobian_ptr_int,
    ConstantInt::get(Type::getInt64Ty(*context), 0));
builder->CreateCondBr(jac_is_null, jac_invalid, jac_valid);
```

**Problem**: Checks pointer value == 0, but should check TYPE TAG

**Evidence from runtime**: 
- `error: Jacobian: function returned null (expected vector)`  
- This error comes from INSIDE Jacobian (line 9280), not curl!
- Jacobian's type check is failing for the vector output

**Real Issue**: Lambda returns vector with mixed content (constants + AD nodes)
- Component 0: `0.0` → constant (NOT AD node)
- Component 1: `0.0` → constant (NOT AD node)  
- Component 2: `(* (vref v 0) (vref v 1))` → AD node

When Jacobian processes output, it sees non-AD components and returns null.

### Issue 2: Error Paths Return Scalars

**Locations**:
- Line 10502: `Value* null_result = ConstantInt::get(..., 0);` (dim_invalid)
- Line 10535: `Value* null_curl = ConstantInt::get(..., 0);` (jac_invalid)

**Problem**: Both return scalar 0 instead of n-dimensional zero vector

---

## N-Dimensional Solution Strategy

### Key Principle

**NEVER hardcode dimensions** - always work with runtime dimension values from input tensors.

### Step 1: Create N-Dimensional Null Vector Helper

**Function Signature**:
```cpp
Value* createNullVectorTensor(Value* dimension)
```

**Purpose**: Create n-dimensional zero vector at runtime (dimension determined at runtime, NOT compile-time!)

**Implementation Pattern**:
```cpp
Value* createNullVectorTensor(Value* dimension) {
    // dimension is a runtime LLVM Value* (could be 2, 3, 4, n...)
    // NOT a compile-time constant!
    
    // 1. Allocate tensor structure
    // 2. Set dimensions array with single element: dimension
    // 3. Set total_elements = dimension  
    // 4. Allocate elements array (dimension * sizeof(double))
    // 5. Zero all elements using LOOP (not fixed count!)
    // 6. Return tensor pointer
}
```

**Critical**: Use loops for zeroing, not fixed counts!

### Step 2: Update dim_invalid Path

**Current** (hardcoded):
```cpp
Value* null_result = ConstantInt::get(Type::getInt64Ty(*context), 0);
```

**N-Dimensional Fix**:
```cpp
// Use RUNTIME dimension value from input validation
// At this point, we have 'n' from input tensor
Value* null_result = createNullVectorTensor(
    ConstantInt::get(Type::getInt64Ty(*context), 3)  // Curl always 3D
);
```

**Note**: Curl IS 3D-specific (mathematical requirement), but implementation should still use the pattern for consistency.

### Step 3: Update jac_invalid Path

**Current** (hardcoded):
```cpp
Value* null_curl = ConstantInt::get(Type::getInt64Ty(*context), 0);
```

**N-Dimensional Fix**:
```cpp
// Create 3D zero vector (curl dimensionality)
Value* null_curl = createNullVectorTensor(
    ConstantInt::get(Type::getInt64Ty(*context), 3)
);
```

### Step 4: Fix Jacobian Type Check

**Current** (checks pointer == 0):
```cpp
Value* jac_is_null = builder->CreateICmpEQ(jacobian_ptr_int,
    ConstantInt::get(Type::getInt64Ty(*context), 0));
```

**N-Dimensional Fix** (checks type tag):
```cpp
// Check if Jacobian is actually a TENSOR
Value* jacobian_type = getTaggedValueType(jacobian_tagged);
Value* jacobian_base_type = builder->CreateAnd(jacobian_type,
    ConstantInt::get(Type::getInt8Ty(*context), 0x0F));
Value* jac_is_tensor = builder->CreateICmpEQ(jacobian_base_type,
    ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_TENSOR_PTR));

// If IS tensor, proceed; if NOT tensor, use error path
builder->CreateCondBr(jac_is_tensor, jac_valid, jac_invalid);
```

---

## Deeper Issue: Jacobian Type Check Failing

### Investigation: Why Does Jacobian Return "null"?

From user's test output (jacobian display test):
```
JACOBIAN: Type check passed, extracting dimensions from tensor
...
#((3 2) (4 0))  ← SUCCESS
```

But from curl test:
```
error: Jacobian: function returned null (expected vector)
```

**Hypothesis**: Different code paths or input causes Jacobian type check to fail.

### Jacobian Type Validation Code

**Location**: [`llvm_codegen.cpp:9254-9282`](../lib/backend/llvm_codegen.cpp:9254-9282)

```cpp
// Line 9250: Test call to get output dimension
Value* test_output_tagged = builder->CreateCall(func_ptr, {vector_tagged});

// Line 9254: Extract type tag
Value* output_type = getTaggedValueType(test_output_tagged);
Value* output_base_type = builder->CreateAnd(output_type,
    ConstantInt::get(Type::getInt8Ty(*context), 0x0F));

// Line 9260: Check if TENSOR_PTR
Value* output_is_tensor = builder->CreateICmpEQ(output_base_type,
    ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_TENSOR_PTR));

// Line 9272: Branch
builder->CreateCondBr(output_is_tensor, output_valid_block, output_invalid_block);

// Line 9279: Invalid path
builder->SetInsertPoint(output_invalid_block);
eshkol_error("Jacobian: function returned null (expected vector)");  // ← THIS ERROR!
builder->CreateBr(jac_return_block);
```

**This means**: Lambda output is NOT being tagged as TENSOR_PTR!

### Lambda Return Tagging

From [`llvm_codegen.cpp:5791-5805`](../lib/backend/llvm_codegen.cpp:5791-5805):

```cpp
if (body_result->getType() == tagged_value_type) {
    eshkol_error("DEBUG: Lambda returning tagged_value directly");
    builder->CreateRet(body_result);  // Returns tagged_value as-is
} else {
    TypedValue typed = detectValueType(body_result);
    Value* tagged = typedValueToTaggedValue(typed);
    builder->CreateRet(tagged);
}
```

From [`llvm_codegen.cpp:6135`](../lib/backend/llvm_codegen.cpp:6135):

```cpp
// codegenTensorOperation returns:
return packPtrToTaggedValue(tensor_int, ESHKOL_VALUE_TENSOR_PTR);
```

**So vector SHOULD be tagged as TENSOR_PTR**. ✓

**Why does check fail then?**

### Possible Cause: Multiple Jacobian Calls

Looking at error output:
```
error: DEBUG: codegenTensorOperation called! num_dims=1, total_elems=2  ← Test vector
error: DEBUG: codegenTensorOperation called! num_dims=1, total_elems=2  ← Lambda output
error: Jacobian: function returned null (expected vector)  ← FIRST Jacobian call FAILS
```

Then later (successful jacobian):
```
JACOBIAN: Type check passed, extracting dimensions from tensor
...
#((3 2) (4 0))  ← SECOND Jacobian call SUCCEEDS
```

**Conclusion**: Curl's internal Jacobian call is DIFFERENT from test's direct call!

---

## The REAL Root Cause

### Curl's Jacobian Construction

**Location**: [`llvm_codegen.cpp:10508-10514`](../lib/backend/llvm_codegen.cpp:10508-10514)

```cpp
// Compute Jacobian matrix (3×3)
eshkol_operations_t jacobian_temp;
jacobian_temp.op = ESHKOL_JACOBIAN_OP;
jacobian_temp.jacobian_op.function = op->curl_op.function;  // ← AST pointer
jacobian_temp.jacobian_op.point = op->curl_op.point;        // ← AST pointer

Value* jacobian_tagged = codegenJacobian(&jacobian_temp);
```

**Problem**: Uses original AST pointers, NOT evaluated values!

**But curl already has evaluated values**:
- Line 10466: `Value* vector_val = codegenAST(op->curl_op.point);`
- Line 10474: `Value* vector_ptr_int = safeExtractInt64(vector_val);`

**Curl should pass these to Jacobian, NOT re-evaluate from AST!**

However, `codegenJacobian` expects `eshkol_operations_t*` with AST fields, not evaluated Values.

**Structural Issue**: Jacobian is designed to evaluate from AST, but curl has already evaluated.

This causes:
- Re-evaluation of same expression
- Possible state inconsistency  
- Different tape/AD context

---

## N-Dimensional Fix Implementation

### Fix 1: Create N-Dimensional Null Vector Helper

**Add to [`llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp) before `codegenCurl` (~line 10330)**:

```cpp
// Create n-dimensional null vector (all zeros)
// dimension is runtime LLVM Value*, not compile-time constant
Value* createNullVectorTensor(Value* dimension) {
    Function* malloc_func = function_table["malloc"];
    if (!malloc_func) {
        eshkol_error("malloc not found for null vector creation");
        return ConstantInt::get(Type::getInt64Ty(*context), 0);
    }
    
    // Allocate tensor structure
    Value* tensor_size = ConstantInt::get(Type::getInt64Ty(*context),
        module->getDataLayout().getTypeAllocSize(tensor_type));
    Value* tensor_ptr = builder->CreateCall(malloc_func, {tensor_size});
    Value* typed_tensor_ptr = builder->CreatePointerCast(tensor_ptr, builder->getPtrTy());
    
    // Allocate dimensions array (1D vector of given dimension)
    Value* dims_size = ConstantInt::get(Type::getInt64Ty(*context), sizeof(uint64_t));
    Value* dims_ptr = builder->CreateCall(malloc_func, {dims_size});
    Value* typed_dims_ptr = builder->CreatePointerCast(dims_ptr, builder->getPtrTy());
    builder->CreateStore(dimension, typed_dims_ptr);  // Runtime dimension
    
    // Store tensor metadata
    builder->CreateStore(typed_dims_ptr,
        builder->CreateStructGEP(tensor_type, typed_tensor_ptr, 0));  // dimensions
    builder->CreateStore(ConstantInt::get(Type::getInt64Ty(*context), 1),
        builder->CreateStructGEP(tensor_type, typed_tensor_ptr, 1));  // num_dimensions = 1
    builder->CreateStore(dimension,
        builder->CreateStructGEP(tensor_type, typed_tensor_ptr, 3));  // total_elements = dimension
    
    // Allocate elements array (dimension * sizeof(double))
    Value* elems_size = builder->CreateMul(dimension,
        ConstantInt::get(Type::getInt64Ty(*context), sizeof(double)));
    Value* elems_ptr = builder->CreateCall(malloc_func, {elems_size});
    Value* typed_elems_ptr = builder->CreatePointerCast(elems_ptr, builder->getPtrTy());
    
    // Zero all elements using LOOP (n-dimensional!)
    Function* current_func = builder->GetInsertBlock()->getParent();
    BasicBlock* zero_loop_cond = BasicBlock::Create(*context, "null_vec_zero_cond", current_func);
    BasicBlock* zero_loop_body = BasicBlock::Create(*context, "null_vec_zero_body", current_func);
    BasicBlock* zero_loop_exit = BasicBlock::Create(*context, "null_vec_zero_exit", current_func);
    
    Value* zero_idx = builder->CreateAlloca(Type::getInt64Ty(*context), nullptr, "zero_idx");
    builder->CreateStore(ConstantInt::get(Type::getInt64Ty(*context), 0), zero_idx);
    builder->CreateBr(zero_loop_cond);
    
    builder->SetInsertPoint(zero_loop_cond);
    Value* idx = builder->CreateLoad(Type::getInt64Ty(*context), zero_idx);
    Value* idx_less = builder->CreateICmpULT(idx, dimension);
    builder->CreateCondBr(idx_less, zero_loop_body, zero_loop_exit);
    
    builder->SetInsertPoint(zero_loop_body);
    Value* elem_ptr = builder->CreateGEP(Type::getDoubleTy(*context), typed_elems_ptr, idx);
    builder->CreateStore(ConstantFP::get(Type::getDoubleTy(*context), 0.0), elem_ptr);
    Value* next_idx = builder->CreateAdd(idx, ConstantInt::get(Type::getInt64Ty(*context), 1));
    builder->CreateStore(next_idx, zero_idx);
    builder->CreateBr(zero_loop_cond);
    
    builder->SetInsertPoint(zero_loop_exit);
    
    builder->CreateStore(typed_elems_ptr,
        builder->CreateStructGEP(tensor_type, typed_tensor_ptr, 2));  // elements
    
    // Return tensor pointer as i64
    return builder->CreatePtrToInt(typed_tensor_ptr, Type::getInt64Ty(*context));
}
```

### Fix 2: Update dim_invalid Path (Line 10502)

**Before**:
```cpp
Value* null_result = ConstantInt::get(Type::getInt64Ty(*context), 0);
```

**After** (n-dimensional):
```cpp
// Curl is mathematically 3D-specific, but use consistent pattern
Value* null_result = createNullVectorTensor(
    ConstantInt::get(Type::getInt64Ty(*context), 3)
);
```

### Fix 3: Fix Jacobian Type Validation (Lines 10524-10530)

**Before**:
```cpp
Value* jac_is_null = builder->CreateICmpEQ(jacobian_ptr_int,
    ConstantInt::get(Type::getInt64Ty(*context), 0));
builder->CreateCondBr(jac_is_null, jac_invalid, jac_valid);
```

**After** (type-based check):
```cpp
// Check type tag, not pointer value
Value* jacobian_type = getTaggedValueType(jacobian_tagged);
Value* jacobian_base_type = builder->CreateAnd(jacobian_type,
    ConstantInt::get(Type::getInt8Ty(*context), 0x0F));
Value* jac_is_tensor = builder->CreateICmpEQ(jacobian_base_type,
    ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_TENSOR_PTR));

// If IS tensor, proceed; if NOT tensor, error
builder->CreateCondBr(jac_is_tensor, jac_valid, jac_invalid);
```

### Fix 4: Update jac_invalid Path (Line 10535)

**Before**:
```cpp
Value* null_curl = ConstantInt::get(Type::getInt64Ty(*context), 0);
```

**After** (n-dimensional):
```cpp
// Create 3D zero vector (curl output dimension)
Value* null_curl = createNullVectorTensor(
    ConstantInt::get(Type::getInt64Ty(*context), 3)
);
```

---

## Critical Insight: Jacobian's Nested List Format

### Current Jacobian Return Format

From successful test:
```
#((3 2) (4 0))  ← Tensor containing list pointers as elements
```

**Structure**:
- Outer tensor: 1D, m elements
- Each element: pointer to list (row of Jacobian)
- Each row list: n double values

**This is the CORRECT n-dimensional format!**

### How Curl Accesses Jacobian Elements

**Current code** ([`llvm_codegen.cpp:10543-10583`](../lib/backend/llvm_codegen.cpp:10543-10583)):

```cpp
// Get Jacobian elements
Value* jac_elements_field = builder->CreateStructGEP(tensor_type, jacobian_ptr, 2);
Value* jac_elements_ptr = builder->CreateLoad(..., jac_elements_field);
Value* typed_jac_elements = builder->CreatePointerCast(jac_elements_ptr, ...);

// Extract specific elements: J[i,j] at linear index i*n + j
Value* j21_idx = ConstantInt::get(..., 2*3 + 1);  // row 2, col 1
Value* j21_ptr = builder->CreateGEP(..., typed_jac_elements, j21_idx);
Value* dF3_dx2 = builder->CreateLoad(Type::getDoubleTy(*context), j21_ptr);
```

**PROBLEM**: This assumes elements are doubles!

**But Jacobian elements are LIST POINTERS** (from nested structure)!

**This is why curl fails even when Jacobian succeeds!**

---

## The Complete Fix Strategy

### Fix A: Handle Jacobian's Nested List Format

Curl must extract doubles from Jacobian's nested list structure, not treat elements as doubles directly.

**Current Extraction** (WRONG):
```cpp
// Assumes elements[i*n+j] is a double
Value* dF3_dx2 = builder->CreateLoad(Type::getDoubleTy(*context), j21_ptr);
```

**Correct Extraction** (handle nested lists):
```cpp
// Element is a list pointer (int64), need to extract doubles from lists
Value* row_list_int = builder->CreateLoad(Type::getInt64Ty(*context), row_ptr);

// Traverse row list to get column j element
// Extract j-th element from row list using list traversal
Value* elem_double = extractDoubleFromListAtIndex(row_list_int, col_idx);
```

**OR BETTER**: Change Jacobian to return flat 2D tensor (double elements), not nested lists!

---

## Recommended Solution Path

### Option 1: Fix Jacobian Return Format (PREFERRED)

**Modify Jacobian** to return **flat 2D tensor** instead of tensor-of-lists:

```cpp
// Current: Returns 1D tensor containing m list pointers
// Each list pointer points to row of n doubles

// Proposed: Return 2D tensor with m×n double elements
// Dimensions: [m, n]
// Elements: flat array of m*n doubles
```

**Benefits**:
- Simpler for curl to access elements
- More efficient (no list traversal)
- More traditional matrix format
- Easier to extend to other operators

**Changes Required**:
1. Modify Jacobian's result tensor creation (lines 9972-9974)
2. Change from nested list building to flat array filling
3. Update display to handle 2D tensors properly

### Option 2: Fix Curl's Element Extraction (ALTERNATIVE)

Keep Jacobian's nested list format, update curl to handle it:

```cpp
// For each J[i,j], extract from nested structure:
// 1. Get row i from tensor elements
// 2. Traverse row list to position j
// 3. Extract car (the double value)
```

**Drawbacks**:
- More complex
- Less efficient  
- Harder to maintain

---

## Recommended Implementation: Option 1

### Step 1: Simplify Jacobian Return (Primary Fix)

**Location**: [`llvm_codegen.cpp:9796-9984`](../lib/backend/llvm_codegen.cpp:9796-9984)

**Current Approach** (lines 9796-9970):
```cpp
// Build nested list structure: ((row1) (row2) ...)
// Then pack into 1D tensor with list pointers as elements
```

**New Approach**:
```cpp
// Skip nested list building entirely
// Store partial derivatives directly in 2D tensor

// After outer_exit (line 9796):
// Result is already computed in jac_elems (m×n doubles)
// Just return it as 2D tensor!

// NO list conversion needed!
```

**Specific Changes**:

1. **Remove lines 9799-9970** (nested list conversion logic)
2. **Keep Jacobian matrix** as computed (already in `typed_jac_elems`)
3. **Return 2D tensor directly**:

```cpp
builder->SetInsertPoint(outer_exit);

// Jacobian matrix is complete in typed_jac_elems (m×n doubles)
// Return as 2D tensor (dimensions already set correctly)
Value* jac_result_int = builder->CreatePtrToInt(typed_jac_ptr, Type::getInt64Ty(*context));
Value* jac_result = packPtrToTaggedValue(jac_result_int, ESHKOL_VALUE_TENSOR_PTR);
builder->CreateBr(jac_return_block);
```

### Step 2: Add N-Dimensional Null Vector Helper

(Implementation from Fix 1 above)

### Step 3: Fix Curl Error Paths

(Implementations from Fix 2, 3, 4 above)

### Step 4: Update Display for 2D Tensors

**Current**: Display handles 1D tensors well  
**Needed**: Display should recognize 2D tensors and format appropriately

**Enhancement** (optional, for better UX):
```cpp
// In display logic, detect num_dimensions == 2
// Format as matrix: #((row1...) (row2...))
// Instead of flat: #(e1 e2 e3 e4...)
```

---

## Implementation Plan

### Priority 1: Fix Jacobian Return Format (CRITICAL)

**File**: [`lib/backend/llvm_codegen.cpp:9796-9984`](../lib/backend/llvm_codegen.cpp:9796-9984)

**Changes**:
1. Remove nested list conversion loop (lines 9799-9970)
2. Return 2D tensor directly with flat double array
3. Update dimensions to [m, n] (2D, not 1D)

### Priority 2: Add Null Vector Helper (CRITICAL)

**File**: [`lib/backend/llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp) before `codegenCurl`

**Add**: `createNullVectorTensor(Value* dimension)` function

### Priority 3: Fix Curl Error Paths (CRITICAL)

**File**: [`lib/backend/llvm_codegen.cpp:10457-10647`](../lib/backend/llvm_codegen.cpp:10457-10647)

**Changes**:
1. Line 10502: Use `createNullVectorTensor(ConstantInt::get(..., 3))`
2. Lines 10524-10530: Change to type tag check
3. Line 10535: Use `createNullVectorTensor(ConstantInt::get(..., 3))`

### Priority 4: Update Curl Element Access (if keeping nested lists)

**Only if NOT doing Priority 1**:
- Change curl's Jacobian element extraction to handle nested lists

---

## Testing Strategy

### Test 1: Verify Jacobian Format Change
```scheme
(define F (lambda (v) (vector (* (vref v 0) (vref v 1)) 
                              (* (vref v 0) (vref v 0)))))
(display (jacobian F (vector 2.0 3.0)))

;; Before fix: #((3 2) (4 0))  ← Nested lists
;; After fix:  #(3 2 4 0) or #((3 2) (4 0))  ← Depends on display
```

### Test 2: Verify Curl Works
```scheme
(define F (lambda (v) (vector 0.0 0.0 (* (vref v 0) (vref v 1)))))
(display (curl F (vector 1.0 2.0 3.0)))

;; Expected: #(1 -2 0) or #(2 -1 0)  ← 3D vector
;; Current:  (0)  ← Scalar
```

### Test 3: Error Path
```scheme
(define F2D (lambda (v) (vector (vref v 0) (vref v 1))))
(display (curl F2D (vector 1.0 2.0)))

;; Expected: #(0 0 0)  ← Null 3D vector with error
;; Current:  (0)  ← Scalar
```

---

## Success Criteria

✅ Jacobian returns flat 2D tensor (m×n doubles) OR nested list tensor (choice depends on display preferences)  
✅ Curl can extract Jacobian elements correctly  
✅ Curl returns 3D vector in ALL code paths (error or success)  
✅ No scalar returns from vector calculus operators  
✅ PHI nodes type-consistent across all branches  
✅ All tests pass with proper vector outputs  

---

## Next Actions

1. **Decide on Jacobian format**: Flat 2D tensor vs. nested lists
   - Flat 2D: Simpler for operators, may need display update
   - Nested lists: Works with current display, complex for operators

2. **Implement chosen approach** in Code mode

3. **Test incrementally**:
   - Jacobian standalone
   - Curl with working Jacobian
   - Error paths

4. **Verify n-dimensional correctness**: No hardcoded dimensions!

---

## Recommendation

**Use Option 1**: Simplify Jacobian to return flat 2D tensor

**Rationale**:
- Curl, Hessian, other operators need matrix element access
- Nested lists add complexity without benefit
- Display can handle 2D tensors (may need minor enhancement)
- More mathematically natural (Jacobian IS a matrix!)

---

**End of N-Dimensional Specification**