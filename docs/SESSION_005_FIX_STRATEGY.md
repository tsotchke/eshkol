# Session 005: Map Mixed Type Fix Strategy

## Root Cause Analysis

### Current Broken Code (Line 5207-5210)
```cpp
// WRONG: Unconditionally unpacks as int64
Value* car_tagged = extractCarAsTaggedValue(current_val);
Value* input_element = unpackInt64FromTaggedValue(car_tagged);  // ← BITCASTS DOUBLE!
Value* proc_result = builder->CreateCall(proc_func, {input_element});
```

### What Happens with (list 1 2.5 3)
```
Element 1: car_tagged = {type=INT64, data.int_val=1}
           unpackInt64 → 1 ✅ CORRECT
           lambda(1) → 2 ✅

Element 2: car_tagged = {type=DOUBLE, data.double_val=2.5}
           unpackInt64 → 4612811918334230528 ❌ BITCAST!
           lambda(corrupted) → corrupted ❌

Element 3: car_tagged = {type=INT64, data.int_val=3}
           unpackInt64 → 3 ✅ CORRECT  
           lambda(3) → 6 ✅
```

**Result:** `(2 -9221120237041090560 6)` ← Middle element corrupted!

## Solution: Type-Aware Extraction

### Fixed Code Pattern (From Working cdr/car Implementation)
```cpp
// Get tagged value
Value* car_tagged = extractCarAsTaggedValue(current_val);

// Extract type
Value* car_type = getTaggedValueType(car_tagged);
Value* car_base_type = builder->CreateAnd(car_type, 
    ConstantInt::get(Type::getInt8Ty(*context), 0x0F));

// Branch on actual type
Value* is_double = builder->CreateICmpEQ(car_base_type,
    ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_DOUBLE));

BasicBlock* double_path = BasicBlock::Create(*context, "extract_double", current_func);
BasicBlock* int_path = BasicBlock::Create(*context, "extract_int", current_func);
BasicBlock* merge_path = BasicBlock::Create(*context, "extract_merge", current_func);

builder->CreateCondBr(is_double, double_path, int_path);

// Extract double correctly
builder->SetInsertPoint(double_path);
Value* double_val = unpackDoubleFromTaggedValue(car_tagged);
Value* double_result = builder->CreateCall(proc_func, {double_val});
builder->CreateBr(merge_path);

// Extract int64 correctly
builder->SetInsertPoint(int_path);
Value* int_val = unpackInt64FromTaggedValue(car_tagged);
Value* int_result = builder->CreateCall(proc_func, {int_val});
builder->CreateBr(merge_path);

// Merge results
builder->SetInsertPoint(merge_path);
PHINode* proc_result = builder->CreatePHI(Type::getInt64Ty(*context), 2);
proc_result->addIncoming(double_result, double_path);
proc_result->addIncoming(int_result, int_path);
```

### What This Achieves with (list 1 2.5 3)
```
Element 1: car_tagged.type=INT64
           → int_path
           → unpackInt64 → 1 ✅
           → lambda(1) → 2 ✅

Element 2: car_tagged.type=DOUBLE  
           → double_path ← KEY FIX!
           → unpackDouble → 2.5 ✅
           → lambda(2.5) → 5.0 ✅

Element 3: car_tagged.type=INT64
           → int_path
           → unpackInt64 → 3 ✅
           → lambda(3) → 6 ✅
```

**Result:** `(2 5.0 6)` ✅ CORRECT!

## Proof This Works

### Evidence from Working Code

**File:** [`lib/backend/llvm_codegen.cpp:4468-4584`](lib/backend/llvm_codegen.cpp:4468)
**Function:** `codegenCompoundCarCdr()` 
**Status:** ✅ Works perfectly with mixed types

The compound car/cdr implementation uses this EXACT pattern:
1. Extract tagged value
2. Get type field
3. Branch on type (DOUBLE vs INT64 vs CONS_PTR)
4. Call appropriate unpacking function
5. Pack result

This is proven to work from the test results in [`MIXED_TYPE_LISTS_IMPLEMENTATION_COMPLETE.md`](docs/MIXED_TYPE_LISTS_IMPLEMENTATION_COMPLETE.md:23-33):
```
4. Testing mixed type list:
   Created list: (1 2.5 3 4.75 5)
   First element: 1           ← cadr extracts correctly
   Second element: 2.500000   ← cadr extracts correctly, NOT bitcast!
   Third element: 3           ← caddr extracts correctly
```

## Implementation Changes Required

### 1. Fix `codegenMapSingleList()` (Line 5207-5227)

**Current broken extraction:**
```cpp
Value* car_tagged = extractCarAsTaggedValue(current_val);
Value* input_element = unpackInt64FromTaggedValue(car_tagged);
Value* proc_result = builder->CreateCall(proc_func, {input_element});
```

**Fixed type-aware extraction:**
```cpp
Value* car_tagged = extractCarAsTaggedValue(current_val);
Value* car_type = getTaggedValueType(car_tagged);
Value* car_base_type = builder->CreateAnd(car_type, ConstantInt::get(Type::getInt8Ty(*context), 0x0F));
Value* is_double = builder->CreateICmpEQ(car_base_type, ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_DOUBLE));

BasicBlock* double_extract = BasicBlock::Create(*context, "map_extract_double", current_func);
BasicBlock* int_extract = BasicBlock::Create(*context, "map_extract_int", current_func);  
BasicBlock* call_proc = BasicBlock::Create(*context, "map_call_proc", current_func);

builder->CreateCondBr(is_double, double_extract, int_extract);

builder->SetInsertPoint(double_extract);
Value* double_val = unpackDoubleFromTaggedValue(car_tagged);
Value* double_result = builder->CreateCall(proc_func, {double_val});
builder->CreateBr(call_proc);

builder->SetInsertPoint(int_extract);
Value* int_val = unpackInt64FromTaggedValue(car_tagged);
Value* int_result = builder->CreateCall(proc_func, {int_val});
builder->CreateBr(call_proc);

builder->SetInsertPoint(call_proc);
PHINode* proc_result = builder->CreatePHI(Type::getInt64Ty(*context), 2);
proc_result->addIncoming(double_result, double_extract);
proc_result->addIncoming(int_result, int_extract);
```

### 2. Same Fix for `codegenMapMultiList()` (Line 5329-5342)

Currently extracts multiple arguments but has same bitcast bug.

### 3. Same Fix for `codegenFilter()` (Line 5456-5461)

Same pattern - extracts car and calls predicate.

### 4. Same Fix for `codegenFold()` (Line 5565-5573) 

Same pattern - extracts car for accumulation.

### 5. Same Fix for `codegenForEachSingleList()` (Line 5787-5792)

Same pattern - extracts car for side effects.

## Why This Works

### Type Preservation Chain

```
List Storage:    [car_type=DOUBLE, car_data.double_val=2.5]
                        ↓
extractCarAsTaggedValue: {type=DOUBLE, data.double_val=2.5}
                        ↓
getTaggedValueType:     type=DOUBLE (0x02)
                        ↓
Branch Decision:        is_double=true → double_path
                        ↓
unpackDoubleFromTaggedValue: 2.5 (actual double, NOT bitcast)
                        ↓
Lambda Call:            lambda(2.5) → arithmetic works correctly
                        ↓
Result:                 5.0 ✅
```

### Contrast with Broken Approach

```
List Storage:    [car_type=DOUBLE, car_data.double_val=2.5]
                        ↓
extractCarAsTaggedValue: {type=DOUBLE, data.double_val=2.5}
                        ↓
unpackInt64FromTaggedValue: 4612811918334230528 ❌ BITCAST!
                        ↓
Lambda Call:            lambda(garbage) → garbage arithmetic
                        ↓  
Result:                 -9221120237041090560 ❌
```

## Testing Strategy

### Test Case
```scheme
(map (lambda (x) (* x 2)) (list 1 2.5 3))
```

### Expected Execution Trace
1. Iteration 1: car=1 (INT64) → int_path → unpackInt64(1) → lambda(1) → 2
2. Iteration 2: car=2.5 (DOUBLE) → double_path → unpackDouble(2.5) → lambda(2.5) → 5.0
3. Iteration 3: car=3 (INT64) → int_path → unpackInt64(3) → lambda(3) → 6

### Expected Output
```
(2 5.0 6)
```

or with better formatting:
```
(2 5 6)
```

## Implementation Checklist

- [ ] Update `codegenMapSingleList()` with type-aware extraction
- [ ] Update `codegenMapMultiList()` with type-aware extraction
- [ ] Update `codegenFilter()` with type-aware extraction
- [ ] Update `codegenFold()` with type-aware extraction
- [ ] Update `codegenForEachSingleList()` with type-aware extraction
- [ ] Build and test
- [ ] Verify output matches expected

## Estimated Time: 20-30 minutes

This is a mechanical fix following the proven pattern from compound car/cdr operations.