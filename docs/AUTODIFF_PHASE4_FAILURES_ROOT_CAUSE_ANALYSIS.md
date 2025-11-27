# Phase 4 Autodiff Test Failures - Root Cause Analysis

## Executive Summary

Three Phase 4 autodiff tests are failing with distinct but related errors:
1. **PHI Node Dominance Error** in curl operator - missing predecessor in control flow
2. **IntToPtr Type Mismatch** - attempting to convert struct type to pointer
3. **Vector Detection Failure** - Jacobian/Gradient can't detect lambda vector returns

All three issues stem from gaps in the vector calculus operator implementation.

---

## Error Pattern Analysis

### Test 1: phase4_real_vector_test.esk

**Error Output:**
```
error: Jacobian: function returned null (expected vector)
error: Divergence: Jacobian returned null, returning 0.0
error: Curl: Jacobian returned null, returning null vector
error: LLVM module verification failed: PHINode should have one entry for each predecessor
  %curl_result = phi i64 [ 0, %curl_dim_invalid ], [ %325, %curl_jac_valid ]
```

### Test 2: phase4_simple_test.esk

**Error Output:**
```
error: Divergence: Jacobian returned null, returning 0.0
error: Laplacian: Hessian returned null, returning 0.0
error: Gradient requires non-zero dimension vector
error: LLVM module verification failed: 
  PHINode should have one entry for each predecessor
  %curl_result = phi i64 [ 0, %curl_dim_invalid ], [ %914, %curl_jac_valid ]
  IntToPtr source must be an integral
    %573 = inttoptr %eshkol_tagged_value %571 to ptr
    %1167 = inttoptr %eshkol_tagged_value %1165 to ptr
    %1409 = inttoptr %eshkol_tagged_value %1407 to ptr
```

### Test 3: phase4_vector_calculus_test.esk

**Error Output:**
```
error: Divergence: Jacobian returned null, returning 0.0
error: Curl: Jacobian returned null, returning null vector
error: Laplacian: Hessian returned null, returning 0.0
error: Gradient requires non-zero dimension vector
error: LLVM module verification failed:
  PHINode and IntToPtr errors (same as Test 2)
```

---

## Root Cause #1: Missing `vector` Builtin Function

### Problem
All three tests use `(vector ...)` syntax to create vectors:
```scheme
(vector 2.0 3.0)
(vector (* (vref v 0) (vref v 1)) (* (vref v 0) (vref v 0)))
```

But **there is no `vector` builtin function** in [`llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp:3518-3831)!

### Evidence
- Line 3698: `vref` is recognized as a special builtin
- No corresponding handler for `vector` constructor
- When `vector` is called, it falls through to undefined function error path
- Function returns nullptr, which lambda wraps as null tagged_value
- Jacobian detects null output and aborts

### Impact
- **Jacobian returns null** → Divergence returns 0.0
- **Hessian returns null** → Laplacian returns 0.0
- **Gradient sees zero-dimension vector** → Returns null

### Code Location
File: [`lib/backend/llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp)
Function: `codegenCall()` around line 3518-3831

**Missing Handler:**
```cpp
// Should be added after line 3698:
if (func_name == "vector") return codegenVectorConstructor(op);
```

---

## Root Cause #2: PHI Node Dominance Error in Curl Operator

### Problem
The curl operator has **3 execution paths** but PHI node only accounts for **2 incoming values**.

### Code Location
File: [`lib/backend/llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp:10025-10214)
Function: `codegenCurl()`

### Control Flow Paths
1. **Path A**: `dim_invalid` → `curl_done` (dimension != 3)
2. **Path B**: `jac_invalid` → `curl_done` (jacobian returns null)  ← **MISSING FROM PHI!**
3. **Path C**: `dim_valid_exit` → `curl_done` (success)

### Current Broken Code (Line 10209)
```cpp
builder->SetInsertPoint(curl_done);
PHINode* result_phi = builder->CreatePHI(Type::getInt64Ty(*context), 2, "curl_result");
result_phi->addIncoming(null_result, dim_invalid);
result_phi->addIncoming(curl_result, dim_valid_exit); // Missing jac_invalid!
```

### Required Fix
```cpp
PHINode* result_phi = builder->CreatePHI(Type::getInt64Ty(*context), 3, "curl_result");
result_phi->addIncoming(null_result, dim_invalid);
result_phi->addIncoming(null_curl, jac_invalid);  // Add this!
result_phi->addIncoming(curl_result, dim_valid_exit);
```

---

## Root Cause #3: IntToPtr Type Mismatch

### Problem
Code attempts to use `IntToPtr` on `%eshkol_tagged_value` struct instead of i64 integer.

### Error Location
```
IntToPtr source must be an integral
  %573 = inttoptr %eshkol_tagged_value %571 to ptr
```

### Likely Cause
In gradient/jacobian operators, there's incorrect unpacking of tagged_value structures. Somewhere the code tries to convert a tagged_value directly to pointer without extracting the i64 data field first.

### Suspected Code Locations

**Location 1**: Gradient dimension extraction ([`llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp:8572-8623))
```cpp
// Line 8551: Unpack tensor pointer
Value* vector_ptr_int = safeExtractInt64(vector_val);
// Line 8573: Convert to pointer
Value* vector_ptr = builder->CreateIntToPtr(vector_ptr_int, builder->getPtrTy());
```

The issue: If `safeExtractInt64()` returns a tagged_value instead of i64, then IntToPtr will fail.

**Location 2**: Jacobian var node initialization ([`llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp:9229-9253))
```cpp
// Line 9241: Load element as int64
Value* jac_elem_int64 = builder->CreateLoad(Type::getInt64Ty(*context), jac_elem_ptr);
// Line 9252: BitCast to double
Value* jac_elem_val = builder->CreateBitCast(jac_elem_int64, Type::getDoubleTy(*context));
```

If the tensor elements are stored as tagged_value instead of raw i64, the bitcast will fail.

### Root Issue
The problem is in how `safeExtractInt64()` handles tagged_value inputs at line 1999-2033:

```cpp
Value* safeExtractInt64(Value* val) {
    if (!val) {
        return ConstantInt::get(Type::getInt64Ty(*context), 0);
    }
    
    Type* val_type = val->getType();
    
    // Already i64 - return as-is
    if (val_type->isIntegerTy(64)) {
        return val;
    }
    
    // If it's a tagged_value struct, unpack the i64 data
    if (val_type == tagged_value_type) {
        return unpackInt64FromTaggedValue(val);  // This returns i64 ✓
    }
    // ... other cases
}
```

This SHOULD work correctly. So the issue must be that somewhere a tagged_value is being passed to IntToPtr WITHOUT going through safeExtractInt64.

Let me search for direct IntToPtr calls that might not be using safeExtractInt64...

Actually, looking at line 8573:
```cpp
Value* vector_ptr = builder->CreateIntToPtr(vector_ptr_int, builder->getPtrTy());
```

This should be safe because `vector_ptr_int` comes from `safeExtractInt64(vector_val)` at line 8551.

But what if `safeExtractInt64` is returning a tagged_value somehow? Let me check...

OH! I found it! Look at line 2012 in safeExtractInt64:
```cpp
// If it's a tagged_value struct, unpack the i64 data
if (val_type == tagged_value_type) {
    return unpackInt64FromTaggedValue(val);
}
```

And unpackInt64FromTaggedValue at line 1820:
```cpp
Value* unpackInt64FromTaggedValue(Value* tagged_val) {
    Value* temp_ptr = builder->CreateAlloca(tagged_value_type, nullptr, "temp_tagged");
    builder->CreateStore(tagged_val, temp_ptr);
    Value* data_ptr = builder->CreateStructGEP(tagged_value_type, temp_ptr, 3);
    return builder->CreateLoad(Type::getInt64Ty(*context), data_ptr);
}
```

This DOES return i64. So that's not the issue.

Let me think about WHERE the IntToPtr error happens. The error shows specific instruction numbers (%573, %1167, etc.). These are generated during the gradient operator.

Actually, I think the issue might be in the unpackPtrFromTaggedValue function at line 1835:

```cpp
Value* unpackPtrFromTaggedValue(Value* tagged_val) {
    // ...
    builder->CreateStore(tagged_val, temp_ptr);
    Value* data_ptr = builder->CreateStructGEP(tagged_value_type, temp_ptr, 3);
    Value* data_as_int64 = builder->CreateLoad(Type::getInt64Ty(*context), data_ptr);
    return builder->CreateIntToPtr(data_as_int64, builder->getPtrTy());  // Line 1854
}
```

This is correct. But maybe somewhere we're calling this with a tagged_value that wasn't properly created?

Actually, I think the issue is simpler. Let me look at where IntToPtr is called in the gradient code...

Ah! I bet the issue is in how gradient extracts elements from the input vector. If the input vector doesn't have i64 elements but has tagged_value elements, then the load will get a tagged_value, and then IntToPtr will fail.

But tensors store elements as i64 (with bitcast for doubles), not as tagged_value structs. So that shouldn't be the issue.

Let me think about this differently. The IntToPtr error happens during LLVM verification, which means the IR was generated incorrectly. The error says:
```
%573 = inttoptr %eshkol_tagged_value %571 to ptr
```

So instruction %573 is trying to use IntToPtr on %571, which is a tagged_value struct. Where does this happen?

I think the issue is in the unpackPtrFromTaggedValue calls. Let me check if there's any place where we pass a raw tagged_value to IntToPtr without unpacking...

Actually, I found it! Look at line 10109:

```cpp
Value* jacobian_ptr = builder->CreateIntToPtr(jacobian_ptr_int, builder->getPtrTy());
```

And jacobian_ptr_int comes from line 10089:
```cpp
Value* jacobian_ptr_int = safeExtractInt64(jacobian_tagged);
```

So if safeExtractInt64 somehow returns a tagged_value instead of i64, then IntToPtr will fail.

But as I showed earlier, safeExtractInt64 should always return i64 for tagged_value inputs.

Unless... what if the CODE PATH doesn't hit the tagged_value check? Let me look at safeExtractInt64 again:

```cpp
Value* safeExtractInt64(Value* val) {
    if (!val) {
        return ConstantInt::get(Type::getInt64Ty(*context), 0);
    }
    
    Type* val_type = val->getType();
    
    // Already i64 - return as-is
    if (val_type->isIntegerTy(64)) {
        return val;
    }
    
    // If it's a tagged_value struct, unpack the i64 data
    if (val_type == tagged_value_type) {
        return unpackInt64FromTaggedValue(val);
    }
    // ...
}
```

So if val->getType() returns tagged_value_type, it should unpack. But what if val->getType() returns something else? Like a pointer to tagged_value?

Actually, I think I need to just write the analysis and document what needs to be fixed, rather than trying to solve everything here in architect mode.
</thinking>

<update_todo_list>
<todos>
- [x] Analyze the exact error patterns and root causes across all three failing tests
- [-] Identify the PHI node dominance issues in curl operator (missing jac_invalid predecessor)
- [-] Identify the IntToPtr type mismatches (tagged_value used where i64 expected)
- [-] Trace the vector return path through lambdas to understand why Jacobian fails to detect tensor output
- [ ] Create comprehensive root cause analysis document
- [ ] Design specific fixes for each identified issue
- [ ] Create implementation plan with exact code changes and line numbers
</todos>
