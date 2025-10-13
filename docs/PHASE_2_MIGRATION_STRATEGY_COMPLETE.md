# Phase 2: Complete Migration Strategy for Higher-Order List Operations

## Critical Finding

**URGENT:** We have a **MIXED ALLOCATION** problem:
- ✅ `cons`, `list` → Allocate 24-byte TAGGED cells
- ❌ Most other functions → Read as 16-byte UNTYPED cells
- ⚠️ Some functions still use `codegenArenaConsCell()` → Allocate 16-byte cells

**Risk:** Functions reading 24-byte cells with 16-byte layout will read garbage data!

## Functions Analysis (24 Total Using Old Cells)

### Category 1: List Traversal Functions (Type-Agnostic)
These functions only traverse lists via cdr, don't care about value types:

1. **`codegenLength()`** (line 4522) - Counts elements
2. **`codegenListTail()`** (line 4865) - Skips n elements
3. **`codegenDrop()`** (line 6003) - Drops first n elements
4. **`codegenLast()`** (line 6544) - Finds last element
5. **`codegenLastPair()`** (line 6610) - Finds last pair

**Migration Strategy:** 
- Don't need type info, just need to traverse cdr chain
- Can use `arena_tagged_cons_get_ptr()` to get cdr as pointer
- Alternatively: accept reading garbage from cdr_data since it's at offset 12 (still in union)

### Category 2: List Construction Functions (Must Preserve Types)
These functions build new lists and must preserve element types:

6. **`codegenIterativeAppend()`** (line 4609) - Concatenates lists
7. **`codegenReverse()`** (line 4717) - Reverses list order
8. **`codegenTake()`** (line 5908) - Takes first n elements
9. **`codegenSplitAt()`** (line 6324) - Splits list at index
10. **`codegenListStar()`** (line 5754) - Uses `codegenArenaConsCell()` directly!
11. **`codegenAcons()`** (line 5775) - Uses `codegenArenaConsCell()` directly!

**Migration Strategy:**
- MUST use `codegenTaggedArenaConsCell()` instead of `codegenArenaConsCell()`
- Need to extract car with proper type (use TypedValue)
- Currently extract car as raw int64 - loses type info!

### Category 3: Higher-Order Functions (Complex Type Handling)
These apply functions to elements and must handle result types:

12. **`codegenMapSingleList()`** (line 5099)
13. **`codegenMapMultiList()`** (line 5206)
14. **`codegenFilter()`** (line 5334)
15. **`codegenFold()`** (line 5442)
16. **`codegenForEachSingleList()`** (line 5666)
17. **`codegenPartition()`** (line 6188)
18. **`codegenFind()`** (line 6069)
19. **`codegenRemove()`** (line 6427)

**Migration Strategy:**
- Extract car using car/cdr helper functions (returns tagged value)
- Pass to procedure (may need to unpack)
- Result may be different type than input
- Store result using `codegenTaggedArenaConsCell()`

### Category 4: Mutable Operations (Direct Memory Access)
These mutate existing cells:

20. **`codegenSetCar()`** (line 4890)
21. **`codegenSetCdr()`** (line 4911)

**Migration Strategy:**
- Must use `arena_tagged_cons_set_*()` helpers
- Need type information for value being set
- Currently just stores raw int64 - loses type!

### Category 5: Membership/Association (Comparison Functions)
These search lists comparing values:

22. **`codegenMember()`** (line 5568)
23. **`codegenAssoc()`** (line 5729)

**Migration Strategy:**
- Extract values for comparison
- Type doesn't matter for equality check (comparing int64 representations)
- BUT should preserve list structure types

### Category 6: Deprecated Old Allocation
24. **`codegenArenaConsCell()`** (lines 995-1004) - **REMOVE**

## Memory Layout Issue Analysis

### Current Situation

**24-byte Tagged Cell (what we allocate in cons/list):**
```
Offset  Field       Size    Value Example
0       car_type    1 byte  0x01 (INT64)
1       cdr_type    1 byte  0x03 (CONS_PTR)
2       flags       2 bytes 0x0010 (EXACT)
4       car_data    8 bytes 42 (the actual value)
12      cdr_data    8 bytes 0x00007f8... (pointer to next cell)
```

**16-byte Old Cell (what functions try to read):**
```
Offset  Field   Size    What They Read
0       car     8 bytes Reads bytes 0-7: gets 0x0010_03_01_00_00_00_00 (GARBAGE!)
8       cdr     8 bytes Reads bytes 8-15: gets car_data (first 8 bytes) - WRONG!
```

**Result:** Complete corruption when reading tagged cells with old layout!

### Why Some Functions Still Work

Functions like `length` work by accident:
- They read "cdr" at offset 8
- This happens to be first 4 bytes of car_data union (offset 4-7) + first 4 bytes of cdr_data (offset 12-15)
- By luck, for simple integer lists this might give semi-reasonable pointer values
- BUT: This is COMPLETELY UNRELIABLE and will fail unpredictably!

## Comprehensive Migration Plan

### Strategy: Three-Phase Approach

#### Phase 2A: Fix Critical Data Corruption (URGENT)

**Goal:** Ensure all functions read cells correctly

**Functions to Migrate:**
1. All list construction functions (append, reverse, take, etc.)
2. All higher-order functions (map, filter, fold, etc.)
3. All traversal functions (length, list-ref, etc.)

**Migration Pattern:**

**OLD CODE:**
```cpp
StructType* arena_cons_type = StructType::get(
    Type::getInt64Ty(*context), Type::getInt64Ty(*context));
Value* cons_ptr = builder->CreateIntToPtr(current_val, builder->getPtrTy());
Value* car_ptr = builder->CreateStructGEP(arena_cons_type, cons_ptr, 0);
Value* car_val = builder->CreateLoad(Type::getInt64Ty(*context), car_ptr);
```

**NEW CODE (Simple Traversal):**
```cpp
// For cdr traversal only (like length):
Value* cons_ptr = builder->CreateIntToPtr(current_val, builder->getPtrTy());
Value* is_cdr = ConstantInt::get(Type::getInt1Ty(*context), 1);
Value* cdr_val = builder->CreateCall(arena_tagged_cons_get_ptr_func,
    {cons_ptr, is_cdr});
```

**NEW CODE (Type-Preserving Extraction):**
```cpp
// For functions that need typed values:
Value* cons_ptr = builder->CreateIntToPtr(current_val, builder->getPtrTy());
Value* is_car = ConstantInt::get(Type::getInt1Ty(*context), 0);

// Get type
Value* car_type = builder->CreateCall(arena_tagged_cons_get_type_func,
    {cons_ptr, is_car});
Value* base_type = builder->CreateAnd(car_type, 
    ConstantInt::get(Type::getInt8Ty(*context), 0x0F));

// Branch on type and extract appropriately
Value* is_double = builder->CreateICmpEQ(base_type, 
    ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_DOUBLE));
Value* is_ptr = builder->CreateICmpEQ(base_type,
    ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_CONS_PTR));

// Create blocks and extract with correct function
// (Similar to codegenCompoundCarCdr pattern)
```

#### Phase 2B: Replace Old Allocation

**Functions Using `codegenArenaConsCell()` (16-byte):**
- codegenListStar (line 5754)
- codegenAcons (line 5775)
- codegenArenaConsCell itself (deprecated)
- codegenIterativeAppend (line 4541)
- codegenReverse (line 4648)
- codegenMapSingleList (line 5034)
- codegenMapMultiList (line 5150)
- codegenFilter (line 5274)
- codegenTake (line 5840)
- codegenPartition (lines 6128, 6154)
- codegenRemove (line 6373)

**Must Replace With:**
```cpp
// OLD:
Value* new_cons = codegenArenaConsCell(car_val, cdr_val);

// NEW:
TypedValue car_typed(car_val, ESHKOL_VALUE_INT64, true); // Or detect type
TypedValue cdr_typed(cdr_val, ESHKOL_VALUE_CONS_PTR, true);
Value* new_cons = codegenTaggedArenaConsCell(car_typed, cdr_typed);
```

#### Phase 2C: Enhanced Type Detection

**Add Helper Function:**
```cpp
TypedValue detectValueType(Value* llvm_val) {
    Type* val_type = llvm_val->getType();
    if (val_type->isIntegerTy(64)) {
        return TypedValue(llvm_val, ESHKOL_VALUE_INT64, true);
    } else if (val_type->isDoubleTy()) {
        return TypedValue(llvm_val, ESHKOL_VALUE_DOUBLE, false);
    } else if (val_type->isPointerTy()) {
        Value* as_int = builder->CreatePtrToInt(llvm_val, Type::getInt64Ty(*context));
        return TypedValue(as_int, ESHKOL_VALUE_CONS_PTR, true);
    }
    return TypedValue(llvm_val, ESHKOL_VALUE_NULL, true);
}
```

## Detailed Function-by-Function Migration Plan

### Group A: Simple Traversal (Lowest Risk)

**Functions:** length, list-tail, drop, last, last-pair

**Current Code Pattern:**
```cpp
// Move to cdr
StructType* arena_cons_type = StructType::get(Type::getInt64Ty(*context), Type::getInt64Ty(*context));
Value* cons_ptr = builder->CreateIntToPtr(current_val, builder->getPtrTy());
Value* cdr_ptr = builder->CreateStructGEP(arena_cons_type, cons_ptr, 1);
Value* cdr_val = builder->CreateLoad(Type::getInt64Ty(*context), cdr_ptr);
```

**New Code Pattern:**
```cpp
// Move to cdr using tagged cons cell helper
Value* cons_ptr = builder->CreateIntToPtr(current_val, builder->getPtrTy());
Value* is_cdr = ConstantInt::get(Type::getInt1Ty(*context), 1);
Value* cdr_val = builder->CreateCall(arena_tagged_cons_get_ptr_func,
    {cons_ptr, is_cdr});
```

**Lines to Change:**
- Line 4522 (codegenLength)
- Line 4784 (codegenListRef) 
- Line 4865 (codegenListTail)
- Line 6003 (codegenDrop)
- Line 6544 (codegenLast)
- Line 6610 (codegenLastPair)

### Group B: Type-Preserving Construction (Medium Risk)

**Functions:** append, reverse, take, split-at

**Current Pattern:**
```cpp
// Get car
Value* car_ptr = builder->CreateStructGEP(arena_cons_type, cons_ptr, 0);
Value* car_val = builder->CreateLoad(Type::getInt64Ty(*context), car_ptr);

// Create new cons
Value* new_cons = codegenArenaConsCell(car_val, cdr_val);
```

**New Pattern:**
```cpp
// Extract car with type info - create synthetic call to car()
eshkol_operations_t car_op;
car_op.op = ESHKOL_CALL_OP;
car_op.call_op.num_vars = 1;
eshkol_ast_t current_ast;
current_ast.type = ESHKOL_INT64;
current_ast.int64_val = 0; // Placeholder
car_op.call_op.variables = &current_ast;
Value* tagged_car = codegenCar(&car_op); // Returns tagged value

// Extract cdr similarly
Value* tagged_cdr = codegenCdr(&cdr_op);

// Create new cons with tagged values
// Need to convert tagged_value back to TypedValue
TypedValue car_typed = extractTypedValueFromTagged(tagged_car);
TypedValue cdr_typed = extractTypedValueFromTagged(tagged_cdr);
Value* new_cons = codegenTaggedArenaConsCell(car_typed, cdr_typed);
```

**Better Pattern:**
```cpp
// Directly use helper functions to extract
Value* cons_ptr = builder->CreateIntToPtr(current_val, builder->getPtrTy());
Value* is_car = ConstantInt::get(Type::getInt1Ty(*context), 0);
Value* is_cdr = ConstantInt::get(Type::getInt1Ty(*context), 1);

// Get types
Value* car_type = builder->CreateCall(arena_tagged_cons_get_type_func, {cons_ptr, is_car});
Value* cdr_type = builder->CreateCall(arena_tagged_cons_get_type_func, {cons_ptr, is_cdr});

// Extract based on type (need branching like in codegenCompoundCarCdr)
// This is complex - see detailed implementation below
```

**SIMPLEST Pattern (Acceptable for Now):**
```cpp
// Just extract as int64/ptr regardless of actual type
// This works because all values fit in 8 bytes
Value* cons_ptr = builder->CreateIntToPtr(current_val, builder->getPtrTy());
Value* is_car = ConstantInt::get(Type::getInt1Ty(*context), 0);

// Try to get as int64 first, fall back to ptr
Value* car_val = builder->CreateCall(arena_tagged_cons_get_int64_func, {cons_ptr, is_car});
// If this errors, we know it's a different type

// Actually, we need the type to decide which getter to use
// So we MUST check type first
```

**RECOMMENDED Pattern:**
Create a helper function `extractCarAsTypedValue()` that encapsulates the branching logic.

### Group C: Higher-Order Functions (Complex)

**Functions:** map (2 variants), filter, fold, for-each, partition, find, remove, member, assoc

**Current Pattern:**
```cpp
// Extract element
Value* input_car_ptr = builder->CreateStructGEP(arena_cons_type, input_cons_ptr, 0);
Value* input_element = builder->CreateLoad(Type::getInt64Ty(*context), input_car_ptr);

// Apply function
Value* proc_result = builder->CreateCall(proc_func, {input_element});

// Store result
Value* new_cons = codegenArenaConsCell(proc_result, ...);
```

**Issues:**
1. `input_element` extracted as raw int64 - loses type
2. `proc_result` may be different type than input
3. `codegenArenaConsCell()` creates untyped 16-byte cell

**New Pattern:**
```cpp
// Extract element with type (returns tagged_value struct)
Value* tagged_element = extractCarWithType(input_cons_ptr);

// Unpack for function call (functions expect raw values currently)
Value* input_element = unpackInt64FromTaggedValue(tagged_element);

// Apply function
Value* proc_result = builder->CreateCall(proc_func, {input_element});

// Detect result type
TypedValue result_typed = detectValueType(proc_result);

// Create typed cons cell
TypedValue null_typed(ConstantInt::get(Type::getInt64Ty(*context), 0), 
                     ESHKOL_VALUE_NULL, true);
Value* new_cons = codegenTaggedArenaConsCell(result_typed, null_typed);
```

### Group D: Mutable Operations

**Functions:** set-car!, set-cdr!

**Current Pattern:**
```cpp
Value* car_ptr = builder->CreateStructGEP(arena_cons_type, cons_ptr, 0);
builder->CreateStore(new_value, car_ptr);
```

**New Pattern:**
```cpp
// Need to determine type of new_value
TypedValue new_val_typed = detectValueType(new_value);

Value* is_car = ConstantInt::get(Type::getInt1Ty(*context), 0);
uint8_t type_with_flags = new_val_typed.type;
if (new_val_typed.isInt64() && new_val_typed.is_exact) {
    type_with_flags |= ESHKOL_VALUE_EXACT_FLAG;
}

if (new_val_typed.isInt64()) {
    builder->CreateCall(arena_tagged_cons_set_int64_func,
        {cons_ptr, is_car, new_val_typed.llvm_value, 
         ConstantInt::get(Type::getInt8Ty(*context), type_with_flags)});
} else if (new_val_typed.isDouble()) {
    builder->CreateCall(arena_tagged_cons_set_double_func, ...);
}
// etc.
```

## Helper Functions Needed

### 1. `extractCarWithType()` - Extract car returning tagged value

```cpp
Value* extractCarWithType(Value* cons_ptr_int) {
    Value* cons_ptr = builder->CreateIntToPtr(cons_ptr_int, builder->getPtrTy());
    Value* is_car = ConstantInt::get(Type::getInt1Ty(*context), 0);
    
    Value* car_type = builder->CreateCall(arena_tagged_cons_get_type_func,
        {cons_ptr, is_car});
    Value* base_type = builder->CreateAnd(car_type, 
        ConstantInt::get(Type::getInt8Ty(*context), 0x0F));
    
    // Branch on type
    Function* current_func = builder->GetInsertBlock()->getParent();
    BasicBlock* double_block = BasicBlock::Create(*context, "extract_double", current_func);
    BasicBlock* ptr_block = BasicBlock::Create(*context, "extract_ptr", current_func);
    BasicBlock* int_block = BasicBlock::Create(*context, "extract_int", current_func);
    BasicBlock* merge_block = BasicBlock::Create(*context, "extract_merge", current_func);
    
    Value* is_double = builder->CreateICmpEQ(base_type, 
        ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_DOUBLE));
    Value* is_ptr = builder->CreateICmpEQ(base_type,
        ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_CONS_PTR));
    
    BasicBlock* check_ptr = BasicBlock::Create(*context, "check_ptr", current_func);
    builder->CreateCondBr(is_double, double_block, check_ptr);
    
    builder->SetInsertPoint(check_ptr);
    builder->CreateCondBr(is_ptr, ptr_block, int_block);
    
    // Extract double
    builder->SetInsertPoint(double_block);
    Value* double_val = builder->CreateCall(arena_tagged_cons_get_double_func,
        {cons_ptr, is_car});
    Value* tagged_double = packDoubleToTaggedValue(double_val);
    builder->CreateBr(merge_block);
    
    // Extract ptr
    builder->SetInsertPoint(ptr_block);
    Value* ptr_val = builder->CreateCall(arena_tagged_cons_get_ptr_func,
        {cons_ptr, is_car});
    Value* tagged_ptr = packInt64ToTaggedValue(ptr_val, true);
    builder->CreateBr(merge_block);
    
    // Extract int64
    builder->SetInsertPoint(int_block);
    Value* int_val = builder->CreateCall(arena_tagged_cons_get_int64_func,
        {cons_ptr, is_car});
    Value* tagged_int = packInt64ToTaggedValue(int_val, true);
    builder->CreateBr(merge_block);
    
    // Merge
    builder->SetInsertPoint(merge_block);
    PHINode* phi = builder->CreatePHI(tagged_value_type, 3);
    phi->addIncoming(tagged_double, double_block);
    phi->addIncoming(tagged_ptr, ptr_block);
    phi->addIncoming(tagged_int, int_block);
    
    return phi;
}
```

### 2. `extractCdrWithType()` - Extract cdr returning tagged value

Same as above but with `is_cdr = 1`.

### 3. `detectValueType()` - Convert LLVM Value* to TypedValue

```cpp
TypedValue detectValueType(Value* llvm_val) {
    if (!llvm_val) return TypedValue();
    
    // Check if already a tagged_value struct
    if (llvm_val->getType() == tagged_value_type) {
        // Extract type from struct
        Value* type_field = getTaggedValueType(llvm_val);
        // ... complex extraction logic
        // For now, return as-is
        return TypedValue(llvm_val, ESHKOL_VALUE_INT64, true); // Placeholder
    }
    
    Type* val_type = llvm_val->getType();
    if (val_type->isIntegerTy(64)) {
        return TypedValue(llvm_val, ESHKOL_VALUE_INT64, true);
    } else if (val_type->isDoubleTy()) {
        return TypedValue(llvm_val, ESHKOL_VALUE_DOUBLE, false);
    } else if (val_type->isPointerTy()) {
        Value* as_int = builder->CreatePtrToInt(llvm_val, Type::getInt64Ty(*context));
        return TypedValue(as_int, ESHKOL_VALUE_CONS_PTR, true);
    }
    return TypedValue(llvm_val, ESHKOL_VALUE_NULL, true);
}
```

## Migration Priority Order

### Priority 1: Stop Data Corruption (CRITICAL)

These functions are called frequently and WILL fail with tagged cells:

1. **length** - Used everywhere, MUST work with tagged cells
2. **append** - Common operation, creates new cells
3. **reverse** - Common operation, creates new cells
4. **list-ref** - Used in tests, must extract correct values

### Priority 2: Higher-Order Functions (HIGH)

These are core Scheme functionality:

5. **map** (both variants) - Core functional programming
6. **filter** - Core functional programming
7. **fold** - Core functional programming
8. **for-each** - Side effects

### Priority 3: Utility Functions (MEDIUM)

9. **member/assoc** - Search functions
10. **take/drop/split-at** - List slicing
11. **partition/find/remove** - List manipulation
12. **set-car!/set-cdr!** - Mutation

### Priority 4: Cleanup (LOW)

13. **Remove codegenArenaConsCell()** - Deprecated
14. **Remove old struct definitions** - Cleanup

## Implementation Approach

### Option A: Minimal Migration (Fast, Risky)

Just fix the struct GEP calls to read from correct offsets:
```cpp
// Hack: Read cdr_data directly at offset 12
Value* cdr_data_ptr = builder->CreateGEP(Type::getInt8Ty(*context), 
    cons_ptr_as_bytes, ConstantInt::get(Type::getInt64Ty(*context), 12));
Value* cdr_val_ptr = builder->CreateBitCast(cdr_data_ptr, 
    PointerType::get(Type::getInt64Ty(*context), 0));
Value* cdr_val = builder->CreateLoad(Type::getInt64Ty(*context), cdr_val_ptr);
```

**Pros:** Quick fix  
**Cons:** Loses type safety, fragile, not future-proof

### Option B: Helper Function Migration (Medium, Balanced)

Create `extractCarWithType()` and `extractCdrWithType()` helpers, use them everywhere:

**Pros:** Type-safe, maintainable  
**Cons:** More code changes, testing needed

### Option C: Full Tagged Value Pipeline (Slow, Best)

Convert all operations to work with tagged values throughout:

**Pros:** Maximum type safety, future-proof  
**Cons:** Large refactor, high risk

## RECOMMENDED: Option B with Phased Rollout

**Week 1:** Implement helper functions + migrate Groups A & B  
**Week 2:** Migrate Group C (higher-order functions)  
**Week 3:** Cleanup and optimization

## Testing Strategy

### Test Suite Progression

**Level 1: Basic Operations** ✅ DONE
- cons, car, cdr, list
- compound car/cdr
- display with mixed types

**Level 2: Utility Functions** (NEXT)
- length with mixed types
- append mixed lists
- reverse mixed lists
- list-ref extracts correct types

**Level 3: Higher-Order** (AFTER LEVEL 2)
- map over mixed lists
- filter mixed lists
- fold with type promotion

**Level 4: Edge Cases** (FINAL)
- Empty lists
- Single elements
- Deep nesting
- Large lists

## Risk Assessment

**HIGH RISK** functions (complex logic, type handling):
- map, filter, fold
- partition, find

**MEDIUM RISK** functions (construction, must preserve types):
- append, reverse
- take, split-at

**LOW RISK** functions (simple traversal):
- length, list-tail
- drop, last

## Success Criteria

- [ ] All 24 functions migrated to tagged cons cells
- [ ] No uses of old 16-byte struct layout remain
- [ ] `codegenArenaConsCell()` removed
- [ ] All tests pass with mixed type lists
- [ ] No data corruption or segfaults
- [ ] Type information preserved through all operations

## Estimated Effort

**Helper Functions:** 2 hours  
**Group A Migration:** 2 hours  
**Group B Migration:** 3 hours  
**Group C Migration:** 4 hours  
**Testing & Debug:** 3 hours  
**Total:** ~14 hours for complete migration

## Next Session Action Items

1. Implement `extractCarWithType()` helper
2. Implement `extractCdrWithType()` helper  
3. Implement `detectValueType()` helper
4. Migrate `codegenLength()` as proof of concept
5. Test length with mixed type lists
6. When successful, proceed with rest of Group A

---

**Current Status:** Phase 1 Complete ✅  
**Next Phase:** Phase 2A - Helper Functions + Group A Migration  
**Timeline:** Ready to begin implementation