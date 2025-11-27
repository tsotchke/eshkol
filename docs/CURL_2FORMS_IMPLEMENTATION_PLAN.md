# N-DIMENSIONAL CURL: DIFFERENTIAL 2-FORMS IMPLEMENTATION

**Date**: 2025-11-27  
**Status**: 🔴 CRITICAL BUG + Mathematical Generalization  
**Scope**: Phase 4.1 (Immediate Fix) + Phase 4.2 (N-Dimensional Theory)

---

## Executive Summary

Traditional curl is **mathematically restricted to 3D**. For true n-dimensional vector calculus, we must implement the **exterior derivative of 1-forms**, which produces **differential 2-forms** (antisymmetric tensors).

**Immediate Problem**: Curl returns scalar `(0)` instead of vector  
**Long-term Goal**: N-dimensional curl using proper differential geometry

---

## Mathematical Foundation

### What is Curl in Different Dimensions?

#### 2D: Scalar Curl (z-component only)
```
F: ℝ² → ℝ²,  F = [F₁, F₂]
curl(F) = ∂F₂/∂x - ∂F₁/∂y  (single scalar)
```

**Returns**: Scalar (magnitude of rotation about z-axis)

#### 3D: Vector Curl (Traditional)
```
F: ℝ³ → ℝ³,  F = [F₁, F₂, F₃]
curl(F) = [∂F₃/∂y - ∂F₂/∂z,  ∂F₁/∂z - ∂F₃/∂x,  ∂F₂/∂x - ∂F₁/∂y]
```

**Returns**: 3-vector (axis of rotation + magnitude)

#### N-D: Antisymmetric 2-Form (Generalized)
```
F: ℝⁿ → ℝⁿ
curl(F) = antisymmetric matrix A where A[i,j] = ∂Fⱼ/∂xᵢ - ∂Fᵢ/∂xⱼ
```

**Returns**: n×n antisymmetric matrix

**Properties**:
- A[i,j] = -A[j,i] (antisymmetry)
- A[i,i] = 0 (diagonal zeros)
- n(n-1)/2 independent components

**Relationship to 3D**: In 3D, the 3×3 antisymmetric matrix has 3 independent components, which map to the 3-vector curl!

### Mathematical Interpretation

The curl is the **exterior derivative** of a 1-form (vector field):

```
d: Ω¹(ℝⁿ) → Ω²(ℝⁿ)
```

Where:
- Ω¹ = space of 1-forms (vector fields)
- Ω² = space of 2-forms (antisymmetric (0,2)-tensors)
- d = exterior derivative operator

---

## Implementation Strategy

### Phase 4.1: Immediate Bug Fix (2-4 hours)

**Goal**: Make curl work correctly in 3D (fix current test)

**Changes**:
1. Fix Jacobian type validation in curl
2. Fix error paths to return tensors (not scalars)
3. Add null vector helper function
4. Verify 3D curl computation

**Deliverable**: `curl` operator returns proper 3D vector

### Phase 4.2: N-Dimensional Generalization (Future)

**Goal**: Implement proper n-dimensional curl as 2-form

**Mathematical Design**:
- 2D input → scalar output (z-component)
- 3D input → 3-vector output (traditional curl)
- nD input → n×n antisymmetric matrix output (2-form)

**API Design**:
```scheme
;; 2D case: Returns scalar
(curl F (vector x y))  → scalar

;; 3D case: Returns 3-vector  
(curl F (vector x y z))  → #(curl_x curl_y curl_z)

;; 4D case: Returns 4×4 antisymmetric matrix
(curl F (vector x y z w))  → #((0 A01 A02 A03)
                               (-A01 0 A12 A13)
                               (-A02 -A12 0 A23)
                               (-A03 -A13 -A23 0))
```

**Implementation Approach**:
1. Detect input dimension n from input tensor
2. Branch on dimension:
   - n=2: Compute scalar (∂F₂/∂x - ∂F₁/∂y)
   - n=3: Compute 3-vector (current implementation)
   - n≥4: Compute n×n antisymmetric matrix
3. Build appropriate result structure

---

## Phase 4.1: Immediate Fix Implementation

### Fix 1: Create N-Dimensional Null Vector Helper

**Location**: Add before `codegenCurl` in [`llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp) (~line 10330)

**Code**:
```cpp
// Create n-dimensional null vector (all zeros)
// Handles ANY dimension at runtime
Value* createNullVectorTensor(Value* dimension) {
    Function* malloc_func = function_table["malloc"];
    if (!malloc_func) {
        eshkol_error("malloc not found for null vector");
        return ConstantInt::get(Type::getInt64Ty(*context), 0);
    }
    
    Function* current_func = builder->GetInsertBlock()->getParent();
    
    // Allocate tensor structure
    Value* tensor_size = ConstantInt::get(Type::getInt64Ty(*context),
        module->getDataLayout().getTypeAllocSize(tensor_type));
    Value* tensor_ptr = builder->CreateCall(malloc_func, {tensor_size});
    Value* typed_tensor_ptr = builder->CreatePointerCast(tensor_ptr, builder->getPtrTy());
    
    // Allocate dimensions array
    Value* dims_size = ConstantInt::get(Type::getInt64Ty(*context), sizeof(uint64_t));
    Value* dims_ptr = builder->CreateCall(malloc_func, {dims_size});
    Value* typed_dims_ptr = builder->CreatePointerCast(dims_ptr, builder->getPtrTy());
    builder->CreateStore(dimension, typed_dims_ptr);  // Runtime dimension!
    
    // Store tensor metadata
    builder->CreateStore(typed_dims_ptr,
        builder->CreateStructGEP(tensor_type, typed_tensor_ptr, 0));
    builder->CreateStore(ConstantInt::get(Type::getInt64Ty(*context), 1),
        builder->CreateStructGEP(tensor_type, typed_tensor_ptr, 1));
    builder->CreateStore(dimension,
        builder->CreateStructGEP(tensor_type, typed_tensor_ptr, 3));
    
    // Allocate elements array
    Value* elems_size = builder->CreateMul(dimension,
        ConstantInt::get(Type::getInt64Ty(*context), sizeof(double)));
    Value* elems_ptr = builder->CreateCall(malloc_func, {elems_size});
    Value* typed_elems_ptr = builder->CreatePointerCast(elems_ptr, builder->getPtrTy());
    
    // Zero all elements using RUNTIME LOOP (n-dimensional!)
    BasicBlock* zero_cond = BasicBlock::Create(*context, "null_vec_zero_cond", current_func);
    BasicBlock* zero_body = BasicBlock::Create(*context, "null_vec_zero_body", current_func);
    BasicBlock* zero_exit = BasicBlock::Create(*context, "null_vec_zero_exit", current_func);
    
    Value* idx_ptr = builder->CreateAlloca(Type::getInt64Ty(*context), nullptr, "zero_idx");
    builder->CreateStore(ConstantInt::get(Type::getInt64Ty(*context), 0), idx_ptr);
    builder->CreateBr(zero_cond);
    
    builder->SetInsertPoint(zero_cond);
    Value* idx = builder->CreateLoad(Type::getInt64Ty(*context), idx_ptr);
    Value* idx_less = builder->CreateICmpULT(idx, dimension);
    builder->CreateCondBr(idx_less, zero_body, zero_exit);
    
    builder->SetInsertPoint(zero_body);
    Value* elem_ptr = builder->CreateGEP(Type::getDoubleTy(*context), typed_elems_ptr, idx);
    builder->CreateStore(ConstantFP::get(Type::getDoubleTy(*context), 0.0), elem_ptr);
    Value* next_idx = builder->CreateAdd(idx, ConstantInt::get(Type::getInt64Ty(*context), 1));
    builder->CreateStore(next_idx, idx_ptr);
    builder->CreateBr(zero_cond);
    
    builder->SetInsertPoint(zero_exit);
    
    builder->CreateStore(typed_elems_ptr,
        builder->CreateStructGEP(tensor_type, typed_tensor_ptr, 2));
    
    return builder->CreatePtrToInt(typed_tensor_ptr, Type::getInt64Ty(*context));
}
```

### Fix 2: Update dim_invalid Path

**Location**: [`llvm_codegen.cpp:10502`](../lib/backend/llvm_codegen.cpp:10502)

**Before**:
```cpp
Value* null_result = ConstantInt::get(Type::getInt64Ty(*context), 0);
```

**After**:
```cpp
// Create 3D null vector (curl output dimension)
Value* null_result = createNullVectorTensor(
    ConstantInt::get(Type::getInt64Ty(*context), 3)
);
```

### Fix 3: Fix Jacobian Type Validation

**Location**: [`llvm_codegen.cpp:10524-10530`](../lib/backend/llvm_codegen.cpp:10524-10530)

**Before**:
```cpp
Value* jac_is_null = builder->CreateICmpEQ(jacobian_ptr_int,
    ConstantInt::get(Type::getInt64Ty(*context), 0));
builder->CreateCondBr(jac_is_null, jac_invalid, jac_valid);
```

**After**:
```cpp
// Check type tag (TENSOR_PTR), not pointer value
Value* jacobian_type = getTaggedValueType(jacobian_tagged);
Value* jacobian_base_type = builder->CreateAnd(jacobian_type,
    ConstantInt::get(Type::getInt8Ty(*context), 0x0F));
Value* jac_is_tensor = builder->CreateICmpEQ(jacobian_base_type,
    ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_TENSOR_PTR));

// If IS tensor, proceed; if NOT, error
builder->CreateCondBr(jac_is_tensor, jac_valid, jac_invalid);
```

### Fix 4: Update jac_invalid Path

**Location**: [`llvm_codegen.cpp:10535`](../lib/backend/llvm_codegen.cpp:10535)

**Before**:
```cpp
Value* null_curl = ConstantInt::get(Type::getInt64Ty(*context), 0);
```

**After**:
```cpp
// Create 3D null vector (curl output dimension)
Value* null_curl = createNullVectorTensor(
    ConstantInt::get(Type::getInt64Ty(*context), 3)
);
```

---

## Phase 4.2: N-Dimensional Curl Design

### Mathematical Theory: Differential 2-Forms

#### Exterior Derivative

For vector field F: ℝⁿ → ℝⁿ, the exterior derivative produces a 2-form:

```
dF = Σᵢ<ⱼ (∂Fⱼ/∂xᵢ - ∂Fᵢ/∂xⱼ) dxᵢ ∧ dxⱼ
```

This is represented as an **antisymmetric matrix** A:

```
A[i,j] = ∂Fⱼ/∂xᵢ - ∂Fᵢ/∂xⱼ
```

#### Special Cases

**2D** (n=2):
```
A = [  0    A₁₂ ]  where A₁₂ = ∂F₂/∂x - ∂F₁/∂y
    [-A₁₂   0  ]
```
Only 1 independent component → return as scalar

**3D** (n=3):
```
A = [  0    A₁₂  A₁₃ ]  where A₁₂ = ∂F₂/∂x - ∂F₁/∂y
    [-A₁₂   0   A₂₃ ]        A₁₃ = ∂F₃/∂x - ∂F₁/∂z  
    [-A₁₃ -A₂₃   0  ]        A₂₃ = ∂F₃/∂y - ∂F₂/∂z
```

**Hodge Dual in 3D**: Maps to curl vector
```
curl = [A₂₃, -A₁₃, A₁₂]ᵀ
```

This is why 3D curl is special - the 3 components of curl exactly match the 3 independent components of the 3×3 antisymmetric matrix!

**4D+** (n≥4):
```
A = n×n antisymmetric matrix
# of components = n(n-1)/2
```

No natural vector representation exists. Must return full matrix.

### Implementation Strategy

#### Dimension Detection Runtime
```cpp
// Extract dimension from input tensor
Value* n = <load from input tensor dimensions>

// Branch based on dimension
Value* is_2d = builder->CreateICmpEQ(n, ConstantInt::get(..., 2));
Value* is_3d = builder->CreateICmpEQ(n, ConstantInt::get(..., 3));

BasicBlock* curl_2d = ...; // Compute scalar curl
BasicBlock* curl_3d = ...; // Compute vector curl (current)
BasicBlock* curl_nd = ...; // Compute antisymmetric matrix
```

#### 2D Implementation
```cpp
// Curl in 2D: scalar value ∂F₂/∂x - ∂F₁/∂y
Value* dF2_dx = J[1,0];  // From Jacobian
Value* dF1_dy = J[0,1];
Value* curl_2d = builder->CreateFSub(dF2_dx, dF1_dy);

// Return as scalar double (not vector!)
return curl_2d;
```

#### 3D Implementation (Current)
```cpp
// Curl in 3D: 3-vector as currently implemented
// (After fixing immediate bugs)
```

#### N-D Implementation (n≥4)
```cpp
// Curl in nD: n×n antisymmetric matrix

// Allocate n×n matrix tensor
Value* matrix_dims = malloc([n, n]);
Value* matrix_elems = malloc(n*n * sizeof(double));

// Fill antisymmetric matrix
for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
        if (i == j) {
            A[i,j] = 0.0;  // Diagonal zeros
        } else {
            A[i,j] = J[j,i] - J[i,j];  // Antisymmetry
        }
    }
}

// Return as 2D tensor
```

---

## Unified Curl API

### Return Type Strategy

**Option A: Dimension-Dependent Returns**
- 2D → double (scalar)
- 3D → 1D tensor (3-vector)
- nD → 2D tensor (n×n matrix)

**Option B: Always Return Tensor (PREFERRED)**
- 2D → 1D tensor [curl_z]
- 3D → 1D tensor [curl_x, curl_y, curl_z]
- nD → 2D tensor (n×n antisymmetric matrix)

**Recommendation**: Option B for API consistency

### Usage Examples

```scheme
;; 2D curl (returns 1-vector containing z-component)
(define F2D (lambda (v) (vector (vref v 1) (- (vref v 0)))))
(curl F2D (vector 1.0 2.0))  → #(-2.0)

;; 3D curl (returns 3-vector)
(define F3D (lambda (v) (vector 0.0 0.0 (* (vref v 0) (vref v 1)))))
(curl F3D (vector 1.0 2.0 3.0))  → #(2.0 -1.0 0.0)

;; 4D curl (returns 4×4 antisymmetric matrix)
(define F4D (lambda (v) (vector (vref v 0) (vref v 1) (vref v 2) (vref v 3))))
(curl F4D (vector 1.0 2.0 3.0 4.0))  → #((0 0 0 0) (0 0 0 0) (0 0 0 0) (0 0 0 0))
```

---

## Immediate Implementation Plan (Phase 4.1)

### File Changes: [`lib/backend/llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp)

#### Change 1: Add Helper Function (~line 10330)
```cpp
Value* createNullVectorTensor(Value* dimension) {
    // (Full implementation from Fix 1 above)
}
```

#### Change 2: Fix dim_invalid Path (line 10502)
```cpp
-  Value* null_result = ConstantInt::get(Type::getInt64Ty(*context), 0);
+  Value* null_result = createNullVectorTensor(
+      ConstantInt::get(Type::getInt64Ty(*context), 3));
```

#### Change 3: Fix Jacobian Validation (lines 10524-10530)
```cpp
-  Value* jac_is_null = builder->CreateICmpEQ(jacobian_ptr_int,
-      ConstantInt::get(Type::getInt64Ty(*context), 0));
-  builder->CreateCondBr(jac_is_null, jac_invalid, jac_valid);
+  Value* jacobian_type = getTaggedValueType(jacobian_tagged);
+  Value* jacobian_base_type = builder->CreateAnd(jacobian_type,
+      ConstantInt::get(Type::getInt8Ty(*context), 0x0F));
+  Value* jac_is_tensor = builder->CreateICmpEQ(jacobian_base_type,
+      ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_TENSOR_PTR));
+  builder->CreateCondBr(jac_is_tensor, jac_valid, jac_invalid);
```

#### Change 4: Fix jac_invalid Path (line 10535)
```cpp
-  Value* null_curl = ConstantInt::get(Type::getInt64Ty(*context), 0);
+  Value* null_curl = createNullVectorTensor(
+      ConstantInt::get(Type::getInt64Ty(*context), 3));
```

---

## Future Work: N-Dimensional Curl (Phase 4.2)

### Implementation Approach

**Modify** [`codegenCurl`](../lib/backend/llvm_codegen.cpp:10457) to:

1. **Extract dimension** from input vector (runtime value)
2. **Branch on dimension**:
   ```cpp
   Value* is_2d = builder->CreateICmpEQ(n, ConstantInt::get(..., 2));
   Value* is_3d = builder->CreateICmpEQ(n, ConstantInt::get(..., 3));
   
   builder->CreateCondBr(is_2d, curl_2d_block, check_3d);
   builder->SetInsertPoint(check_3d);
   builder->CreateCondBr(is_3d, curl_3d_block, curl_nd_block);
   ```

3. **Implement each case**:
   - `curl_2d_block`: Compute scalar, return as 1-vector
   - `curl_3d_block`: Current 3D implementation
   - `curl_nd_block`: Compute antisymmetric matrix

4. **Merge results** with consistent return type (all tensors)

### Antisymmetric Matrix Construction

**For n≥4**:
```cpp
// Allocate n×n tensor
Value* matrix_total = builder->CreateMul(n, n);
Value* matrix_elems_size = builder->CreateMul(matrix_total,
    ConstantInt::get(..., sizeof(double)));
Value* matrix_elems_ptr = builder->CreateCall(malloc_func, {matrix_elems_size});

// Double loop: compute A[i,j] = J[j,i] - J[i,j]
for (i=0; i<n; i++) {
    for (j=0; j<n; j++) {
        if (i == j) {
            A[i,j] = 0.0;
        } else {
            // Extract J[i,j] and J[j,i] from Jacobian
            Value* Jij = extractJacobianElement(jacobian, i, j, n);
            Value* Jji = extractJacobianElement(jacobian, j, i, n);
            Value* Aij = builder->CreateFSub(Jji, Jij);
            storeMatrixElement(matrix_elems_ptr, i, j, n, Aij);
        }
    }
}
```

---

## Testing Plan

### Phase 4.1 Tests (Immediate)

**Test 1**: 3D Curl with mixed components
```scheme
(curl (lambda (v) (vector 0.0 0.0 (* (vref v 0) (vref v 1)))) 
      (vector 1.0 2.0 3.0))
;; Expected: #(2.0 -1.0 0.0) or similar 3-vector
```

**Test 2**: Error path validation
```scheme
(curl (lambda (v) (vector (vref v 0) (vref v 1))) 
      (vector 1.0 2.0))
;; Expected: #(0 0 0) with "only defined for 3D" error
```

### Phase 4.2 Tests (Future)

**Test 3**: 2D Curl
```scheme
(curl (lambda (v) (vector (vref v 1) (- (vref v 0)))) 
      (vector 1.0 2.0))
;; Expected: #(-2.0)  ← Scalar wrapped in vector
```

**Test 4**: 4D Curl (antisymmetric matrix)
```scheme
(curl (lambda (v) (vector (vref v 1) (vref v 0) (vref v 3) (vref v 2)))
      (vector 1.0 2.0 3.0 4.0))
;; Expected: 4×4 antisymmetric matrix
```

---

## Success Criteria

### Phase 4.1 (Immediate)
✅ Curl returns 3D vector (not scalar) for valid input  
✅ Error paths return null vectors (not scalars)  
✅ Jacobian validation uses type tags  
✅ All PHI nodes type-consistent  
✅ Test `phase4_real_vector_test.esk` passes

### Phase 4.2 (Future)
✅ 2D curl returns scalar (or 1-vector)  
✅ 3D curl returns 3-vector (traditional)  
✅ nD curl returns antisymmetric n×n matrix  
✅ Mathematical properties verified (antisymmetry, etc.)  
✅ Comprehensive test suite for all dimensions

---

## References

### Mathematical
- Lee, "Introduction to Smooth Manifolds" (Chapter 14: Differential Forms)
- Spivak, "Calculus on Manifolds" (Chapter 4: Integration on Chains)
- Wikipedia: "Exterior derivative", "Differential form"

### Implementation
- Current curl: [`llvm_codegen.cpp:10457-10647`](../lib/backend/llvm_codegen.cpp:10457-10647)
- Jacobian: [`llvm_codegen.cpp:9191-9984`](../lib/backend/llvm_codegen.cpp:9191-9984)
- Tensor type: [`llvm_codegen.cpp:467-475`](../lib/backend/llvm_codegen.cpp:467-475)

---

## Next Steps

1. **Immediate**: Implement Phase 4.1 fixes (switch to Code mode)
2. **Short-term**: Test 3D curl thoroughly
3. **Medium-term**: Design Phase 4.2 API and mathematical validation
4. **Long-term**: Implement full n-dimensional differential forms calculus

---

**End of 2-Forms Implementation Plan**