# Autodiff Critical Bugs: Implementation Fix Plan
**Date**: November 27, 2025  
**Status**: READY FOR IMPLEMENTATION  
**Estimated Time**: 6-8 hours total  

---

## Overview

This document provides **exact line numbers and code changes** to fix the three critical autodiff bugs:
1. Tensor type shadowing (29 instances)
2. Missing TENSOR_PTR type tag
3. Display function type confusion

---

## Phase 1: Fix Tensor Type Shadowing (3-4 hours)

### Goal
Replace all local `tensor_type` variables with the class member `this->tensor_type`

### Pre-Implementation Verification

**Confirm class member exists** at [`llvm_codegen.cpp:96`](../lib/backend/llvm_codegen.cpp:96):
```cpp
StructType* tensor_type;  // ✓ Class member declared
```

**Confirm initialization** at [`llvm_codegen.cpp:468`](../lib/backend/llvm_codegen.cpp:468):
```cpp
tensor_type = StructType::create(*context, tensor_fields, "tensor");  // ✓ Created once
```

### Fix Pattern

For each function, replace:
```cpp
// BEFORE (broken):
std::vector<Type*> tensor_fields;
tensor_fields.push_back(PointerType::getUnqual(*context));
tensor_fields.push_back(Type::getInt64Ty(*context));
tensor_fields.push_back(PointerType::getUnqual(*context));
tensor_fields.push_back(Type::getInt64Ty(*context));
StructType* tensor_type = StructType::create(*context, tensor_fields, "tensor");

// AFTER (fixed):
// Use this->tensor_type (already initialized in createBuiltinFunctions)
// DELETE the above 6 lines completely
```

### All 28 Fix Locations (excluding line 468 class init)

| Function | Line Range | Lines to Delete |
|----------|------------|-----------------|
| `codegenDisplay()` | 3893-3899 | 7 lines |
| `codegenTensor()` | 5106-5112 | 7 lines |
| `codegenTensorOperation()` | 5188-5195 | 8 lines |
| `codegenTensorGet()` | 5297-5303 | 7 lines |
| `codegenVectorRef()` | 5359-5365 | 7 lines |
| `codegenTensorSet()` | 5476-5482 | 7 lines |
| `codegenTensorArithmetic()` | 5534-5540 | 7 lines |
| `codegenTensorDot()` | 5638-5644 | 7 lines |
| `codegenTensorShape()` | 5902-5908 | 7 lines |
| `codegenTensorApply()` | 5946-5952 | 7 lines |
| `codegenTensorReduceAll()` | 6089-6095 | 7 lines |
| `codegenTensorReduceWithDim()` | 6207-6213 | 7 lines |
| `codegenGradient()` | 7635-7642 | 8 lines |
| `codegenJacobian()` | 8018-8024 | 7 lines |
| `codegenHessian()` | 8575-8581 | 7 lines |
| `codegenDivergence()` | 8933-8939 | 7 lines |
| `codegenCurl()` | 9034-9040 | 7 lines |
| `codegenLaplacian()` | 9255-9261 | 7 lines |
| `codegenDirectionalDerivative()` | 9363-9369 | 7 lines |
| `codegenVectorToString()` | 9779-9785 | 7 lines |
| `codegenMatrixToString()` | 9936-9942 | 7 lines |

**Total**: ~147 lines to delete across 21 functions

### Implementation Steps

1. **Backup**: Create git branch `fix/autodiff-type-shadowing`

2. **For each function** (in order of test priority):
   a. Find the `std::vector<Type*> tensor_fields;` declaration
   b. Find the `StructType::create()` call
   c. Delete the entire block (typically 6-8 lines)
   d. All subsequent uses of `tensor_type` will now use `this->tensor_type`

3. **Critical check**: After each fix, verify:
   - No undefined `tensor_type` variable errors
   - Function still compiles
   - Uses class member, not local variable

4. **Test after each major function**:
   - Tensor creation functions first (codegenTensor, codegenTensorOperation)
   - Then access functions (vref, tensor-get)
   - Then autodiff functions (gradient, jacobian)

### Example Fix (codegenJacobian)

**BEFORE** (lines 8018-8024):
```cpp
// CRITICAL FIX: Create tensor_type ONCE at start, BEFORE any loops (like gradient does)
// This prevents LLVM IR name conflicts and invalid pointer casts in nested loops
std::vector<Type*> tensor_fields;
tensor_fields.push_back(PointerType::getUnqual(*context)); // uint64_t* dimensions
tensor_fields.push_back(Type::getInt64Ty(*context));       // uint64_t num_dimensions
tensor_fields.push_back(PointerType::getUnqual(*context)); // double* elements
tensor_fields.push_back(Type::getInt64Ty(*context));       // uint64_t total_elements
StructType* tensor_type = StructType::create(*context, tensor_fields, "tensor");
```

**AFTER**:
```cpp
// Use class member tensor_type (already created in createBuiltinFunctions)
// All tensor operations now share the same LLVM StructType
```

**All subsequent uses** of `tensor_type` in the function automatically use `this->tensor_type`

---

## Phase 2: Add TENSOR_PTR Type Tag (1-2 hours)

### Step 1: Update Type Enum

**File**: [`eshkol.h:41-50`](../inc/eshkol/eshkol.h:41)

**BEFORE**:
```c
typedef enum {
    ESHKOL_VALUE_NULL        = 0,
    ESHKOL_VALUE_INT64       = 1,
    ESHKOL_VALUE_DOUBLE      = 2,
    ESHKOL_VALUE_CONS_PTR    = 3,
    ESHKOL_VALUE_DUAL_NUMBER = 4,
    ESHKOL_VALUE_AD_NODE_PTR = 5,
    // Reserved for future expansion
    ESHKOL_VALUE_MAX         = 15
} eshkol_value_type_t;
```

**AFTER**:
```c
typedef enum {
    ESHKOL_VALUE_NULL        = 0,
    ESHKOL_VALUE_INT64       = 1,
    ESHKOL_VALUE_DOUBLE      = 2,
    ESHKOL_VALUE_CONS_PTR    = 3,
    ESHKOL_VALUE_DUAL_NUMBER = 4,
    ESHKOL_VALUE_AD_NODE_PTR = 5,
    ESHKOL_VALUE_TENSOR_PTR  = 6,  // ← NEW: Tensor pointer type
    // Reserved for future expansion
    ESHKOL_VALUE_MAX         = 15
} eshkol_value_type_t;
```

### Step 2: Update Helper Macros

**File**: [`eshkol.h:144`](../inc/eshkol/eshkol.h:144) - Add after `ESHKOL_IS_AD_NODE_PTR_TYPE`:
```c
#define ESHKOL_IS_TENSOR_PTR_TYPE(type) (((type) & 0x0F) == ESHKOL_VALUE_TENSOR_PTR)
```

### Step 3: Update detectValueType()

**File**: [`llvm_codegen.cpp:1846-1851`](../lib/backend/llvm_codegen.cpp:1846)

**BEFORE**:
```cpp
if (isa<PtrToIntInst>(llvm_val)) {
    eshkol_debug("detectValueType: i64 from PtrToInt, treating as CONS_PTR");
    return TypedValue(llvm_val, ESHKOL_VALUE_CONS_PTR, true);
}
```

**AFTER**:
```cpp
if (isa<PtrToIntInst>(llvm_val)) {
    // Check if this came from tensor operation or cons cell
    PtrToIntInst* ptoi = dyn_cast<PtrToIntInst>(llvm_val);
    if (ptoi) {
        // Heuristic: tensor operations have specific naming
        // For production: use explicit tagging in tensor operations
        std::string name = ptoi->getName().str();
        if (name.find("tensor") != std::string::npos || 
            name.find("vector") != std::string::npos) {
            eshkol_debug("detectValueType: i64 from tensor PtrToInt");
            return TypedValue(llvm_val, ESHKOL_VALUE_TENSOR_PTR, true);
        }
    }
    eshkol_debug("detectValueType: i64 from PtrToInt, treating as CONS_PTR");
    return TypedValue(llvm_val, ESHKOL_VALUE_CONS_PTR, true);
}
```

### Step 4: Tag Tensor Returns

**Locations to update**:

| Function | Line | Change |
|----------|------|--------|
| `codegenTensorOperation()` | 5281 | Return tagged tensor pointer |
| `codegenGradient()` | 7995 | Return tagged tensor pointer |
| `codegenJacobian()` | 8528 | Return tagged tensor pointer |
| `codegenHessian()` | 8880 | Return tagged tensor pointer |

**Pattern**:
```cpp
// BEFORE:
return builder->CreatePtrToInt(typed_result_tensor_ptr, Type::getInt64Ty(*context));

// AFTER:
Value* tensor_int = builder->CreatePtrToInt(typed_result_tensor_ptr, Type::getInt64Ty(*context));
return packPtrToTaggedValue(
    builder->CreateIntToPtr(tensor_int, builder->getPtrTy()),
    ESHKOL_VALUE_TENSOR_PTR
);
```

---

## Phase 3: Fix Display Function (2-3 hours)

### Goal
Make `display()` handle tensors separately from cons cells

### Current Bug Location

**File**: [`llvm_codegen.cpp:3853-4181`](../lib/backend/llvm_codegen.cpp:3853)

The function has this flow:
```
1. Extract int64 value (line 3854)
2. Check if large enough to be pointer (line 3859)
3. Try to read as cons cell (line 3874) ← BUG: Assumes cons cell
4. Display as list if valid type
5. Try as tensor if cons type invalid (line 3891)
```

### The Fix

**Insert type check BEFORE memory probing**:

**Location**: After line 3856, before line 3859

**INSERT**:
```cpp
// PHASE 3 FIX: If arg is tagged_value with TENSOR_PTR type, handle specially
if (arg->getType() == tagged_value_type) {
    Value* arg_type = getTaggedValueType(arg);
    Value* arg_base_type = builder->CreateAnd(arg_type,
        ConstantInt::get(Type::getInt8Ty(*context), 0x0F));
    
    Value* is_tensor = builder->CreateICmpEQ(arg_base_type,
        ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_TENSOR_PTR));
    
    BasicBlock* tensor_display_block = BasicBlock::Create(*context, "display_tensor_tagged", current_func);
    BasicBlock* continue_normal = BasicBlock::Create(*context, "display_continue_normal", current_func);
    
    builder->CreateCondBr(is_tensor, tensor_display_block, continue_normal);
    
    // Tensor display path
    builder->SetInsertPoint(tensor_display_block);
    Value* tensor_ptr_from_tagged = unpackInt64FromTaggedValue(arg);
    // Jump to existing tensor display code at line 3923
    // (or inline tensor display here)
    builder->CreateBr(display_tensor); // Reuse existing tensor display code
    
    // Continue with normal display logic
    builder->SetInsertPoint(continue_normal);
    // Fall through to existing code
}
```

### Alternative: Simpler Fix

Just check type tag before calling cons cell functions:

**Location**: Line 3872, BEFORE calling `arena_tagged_cons_get_type_func`

**INSERT**:
```cpp
// CRITICAL FIX: Don't call cons functions on tensor pointers!
// Check if this might be a tensor first
if (arg->getType() == tagged_value_type) {
    Value* arg_type = getTaggedValueType(arg);
    Value* arg_base = builder->CreateAnd(arg_type, ConstantInt::get(Type::getInt8Ty(*context), 0x0F));
    Value* is_tensor = builder->CreateICmpEQ(arg_base,
        ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_TENSOR_PTR));
    
    BasicBlock* skip_cons_check = BasicBlock::Create(*context, "skip_cons_tensor", current_func);
    BasicBlock* do_cons_check = BasicBlock::Create(*context, "do_cons_check", current_func);
    
    builder->CreateCondBr(is_tensor, skip_cons_check, do_cons_check);
    
    builder->SetInsertPoint(skip_cons_check);
    // Go directly to tensor display (line 3923)
    builder->CreateBr(try_tensor_check);
    
    builder->SetInsertPoint(do_cons_check);
    // Continue with cons cell check
}
```

---

## Phase 4: Validation & Testing (1 hour)

### Test Sequence

1. **Compile test**: `cmake --build build`
   - Should complete without errors
   - No LLVM type warnings

2. **Basic tensor test**: `examples/minimal_tensor_test.esk`
   - Should display vectors correctly
   - No type errors

3. **Gradient test**: `tests/autodiff/debug_gradient_only.esk`
   - Should return gradient vector
   - Display should show `#(...)` not `(0)`

4. **Jacobian test**: `tests/autodiff/debug_operators.esk`
   - Should NOT segfault
   - Should compute jacobian matrix
   - Display should work

5. **Full autodiff suite**: `scripts/run_autodiff_tests.sh`
   - All tests should pass
   - No memory corruption
   - Correct numerical results

---

## Detailed Fix Instructions for Each Function

### 1. codegenDisplay() - Line 3893-3899

**DELETE**:
```cpp
std::vector<Type*> tensor_fields;
tensor_fields.push_back(PointerType::getUnqual(*context));
tensor_fields.push_back(Type::getInt64Ty(*context));
tensor_fields.push_back(PointerType::getUnqual(*context));
tensor_fields.push_back(Type::getInt64Ty(*context));
StructType* tensor_type = StructType::create(*context, tensor_fields, "tensor");
```

**RESULT**: All uses of `tensor_type` in this function now refer to `this->tensor_type`

---

### 2. codegenTensor() - Line 5106-5112

**DELETE**:
```cpp
std::vector<Type*> tensor_fields;
tensor_fields.push_back(PointerType::getUnqual(*context)); // dimensions array
tensor_fields.push_back(Type::getInt64Ty(*context)); // num_dimensions
tensor_fields.push_back(PointerType::getUnqual(*context)); // elements array  
tensor_fields.push_back(Type::getInt64Ty(*context)); // total_elements
StructType* tensor_type = StructType::create(*context, tensor_fields, "tensor");
```

---

### 3. codegenTensorOperation() - Line 5188-5195

**DELETE**:
```cpp
std::vector<Type*> tensor_fields;
tensor_fields.push_back(PointerType::getUnqual(*context)); // dimensions array
tensor_fields.push_back(Type::getInt64Ty(*context)); // num_dimensions
tensor_fields.push_back(PointerType::getUnqual(*context)); // elements array
tensor_fields.push_back(Type::getInt64Ty(*context)); // total_elements
StructType* tensor_type = StructType::create(*context, tensor_fields, "tensor");
```

---

### 4-21. [Remaining Functions Follow Same Pattern]

All functions follow the **identical pattern**:
- Delete `std::vector<Type*> tensor_fields;` declaration
- Delete 4 `tensor_fields.push_back()` calls  
- Delete `StructType* tensor_type = StructType::create()` call
- Keep all other code unchanged

---

## CRITICAL: codegenJacobian Special Fix

### Why This is Different

Jacobian has **nested tensor operations** with multiple StructGEP calls. The type shadowing creates the MOST severe corruption here.

### Specific Fix Location

**File**: [`llvm_codegen.cpp:8018-8024`](../lib/backend/llvm_codegen.cpp:8018)

**Line 8018 comment says**:
```cpp
// CRITICAL FIX: Create tensor_type ONCE at start, BEFORE any loops (like gradient does)
// This prevents LLVM IR name conflicts and invalid pointer casts in nested loops
```

**THIS COMMENT IS WRONG!** The fix isn't to create it locally, it's to use the class member!

**DELETE lines 8018-8024 completely**:
```cpp
// CRITICAL FIX: Create tensor_type ONCE at start, BEFORE any loops (like gradient does)
// This prevents LLVM IR name conflicts and invalid pointer casts in nested loops
std::vector<Type*> tensor_fields;
tensor_fields.push_back(PointerType::getUnqual(*context)); // uint64_t* dimensions
tensor_fields.push_back(Type::getInt64Ty(*context));       // uint64_t num_dimensions
tensor_fields.push_back(PointerType::getUnqual(*context)); // double* elements
tensor_fields.push_back(Type::getInt64Ty(*context));       // uint64_t total_elements
StructType* tensor_type = StructType::create(*context, tensor_fields, "tensor");
```

**REPLACE WITH**:
```cpp
// Use class member tensor_type (shared by ALL tensor operations)
// This prevents LLVM IR type conflicts in nested operations
```

---

## Quick Reference: All Line Numbers to Fix

```
llvm_codegen.cpp deletions:
- Lines 3893-3899   (codegenDisplay)
- Lines 5106-5112   (codegenTensor)  
- Lines 5188-5195   (codegenTensorOperation) ← Critical: used by gradient/jacobian
- Lines 5297-5303   (codegenTensorGet)
- Lines 5359-5365   (codegenVectorRef) ← Critical: used in AD lambdas
- Lines 5476-5482   (codegenTensorSet)
- Lines 5534-5540   (codegenTensorArithmetic)
- Lines 5638-5644   (codegenTensorDot)
- Lines 5902-5908   (codegenTensorShape)
- Lines 5946-5952   (codegenTensorApply)
- Lines 6089-6095   (codegenTensorReduceAll)
- Lines 6207-6213   (codegenTensorReduceWithDim)
- Lines 7635-7642   (codegenGradient) ← Critical: creates result tensors
- Lines 8018-8024   (codegenJacobian) ← Critical: SEGFAULT HERE
- Lines 8575-8581   (codegenHessian)
- Lines 8933-8939   (codegenDivergence)
- Lines 9034-9040   (codegenCurl)
- Lines 9255-9261   (codegenLaplacian)
- Lines 9363-9369   (codegenDirectionalDerivative)
- Lines 9779-9785   (codegenVectorToString)
- Lines 9936-9942   (codegenMatrixToString)

Total: 21 functions, ~147 lines to delete
```

---

## Expected Outcomes

### After Phase 1 (Type Shadowing Fix)

✅ **LLVM IR**:
- Only ONE `%tensor` type in entire module
- No `%tensor.0`, `%tensor.1` conflicts
- All StructGEP operations use same type

✅ **Runtime**:
- No segfaults in jacobian
- Memory operations are type-safe
- Optimizer can work correctly

✅ **Tests**:
- `debug_operators.esk` completes without crash
- Jacobian computes (even if results wrong)

### After Phase 2 (Type Tag)

✅ **Type System**:
- Tensor pointers distinguishable from cons pointers
- `display()` can check type before operations
- Type safety throughout system

✅ **Tests**:
- No more "type=32" errors
- Display shows correct format

### After Phase 3 (Display Fix)

✅ **User Experience**:
- `(display (gradient ...))` shows `#(10.0)` not `(0)`
- Tensor formatting correct
- No cons cell operations on tensors

---

## Risk Mitigation

### Potential Issues

**Issue 1**: Missing a tensor_type usage
- **Symptom**: Compilation error "undeclared identifier"
- **Fix**: Search for "tensor_type" in function, ensure using class member

**Issue 2**: Breaking non-autodiff tensor operations
- **Symptom**: Tests that worked before now fail
- **Fix**: Test incrementally, one function at a time

**Issue 3**: LLVM verification failures
- **Symptom**: "Type mismatch in StructGEP"
- **Fix**: Verify all deletions complete, no local shadows remain

### Rollback Plan

If fixes cause regressions:
1. `git checkout lib/backend/llvm_codegen.cpp` (revert changes)
2. `git checkout inc/eshkol/eshkol.h` (revert type enum)
3. Fix one function at a time instead of bulk changes
4. Add comprehensive logging to identify exact failure point

---

## Success Criteria

### Must Pass

- [ ] `cmake --build build` completes without errors
- [ ] `debug_operators.esk` runs without segfault
- [ ] Gradient returns correct values (not zero)
- [ ] Jacobian computes 2x2 matrix
- [ ] Display shows tensors as `#(...)`
- [ ] No "type=32" errors
- [ ] All 40 autodiff tests pass

### LLVM IR Validation

- [ ] Only ONE `%tensor` type in generated IR
- [ ] No `%tensor.0` or `%tensor.1` in IR
- [ ] StructGEP operations type-consistent

---

## Implementation Checklist

### Setup
- [ ] Create git branch: `fix/autodiff-critical-bugs`
- [ ] Run baseline tests, document current state
- [ ] Review all line numbers are current

### Phase 1 Execution  
- [ ] Fix codegenTensorOperation (most critical - used by gradient/jacobian)
- [ ] Fix codegenVectorRef (most critical - used in lambdas)
- [ ] Fix codegenGradient (creates result tensors)
- [ ] Fix codegenJacobian (segfault location)
- [ ] Fix remaining 17 functions
- [ ] Verify compilation
- [ ] Run jacobian test - should not crash

### Phase 2 Execution
- [ ] Update eshkol.h enum
- [ ] Update eshkol.h macro
- [ ] Update detectValueType
- [ ] Update tensor return statements (4 locations)
- [ ] Test gradient display

### Phase 3 Execution
- [ ] Add type check in display
- [ ] Add tensor display branch
- [ ] Test with various tensor types
- [ ] Validate no cons operations on tensors

### Final Validation
- [ ] Run full autodiff test suite
- [ ] Check memory with valgrind (no leaks/corruption)
- [ ] Verify numerical correctness
- [ ] Update documentation

---

**Status**: Implementation plan complete, ready for Code mode execution  
**Recommended Mode**: Switch to Code mode for implementation  
**Estimated Time**: 6-8 hours total (can be done in one day)