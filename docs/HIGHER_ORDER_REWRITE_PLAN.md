# Higher-Order Functions Rewrite Plan
## Eshkol v1.0-foundation - Phase 1, Month 1, Week 1

**Sessions: 005-020 (16 sessions)**  
**Status: Analysis Complete - Ready for Implementation**  
**Created: 2025-11-13**

---

## Executive Summary

This document provides a complete, line-by-line migration plan for all 15 higher-order list functions in [`lib/backend/llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp). The migration converts legacy `CreateStructGEP` direct memory access to type-safe tagged cons cell helper functions, enabling proper support for mixed-type lists (integers, doubles, and cons pointers).

**Status Breakdown:**
- ✅ **2 Complete**: [`codegenLast()`](../lib/backend/llvm_codegen.cpp:6607), [`codegenLastPair()`](../lib/backend/llvm_codegen.cpp:6705) - Already using tagged helpers
- 🔨 **13 Need Migration**: All use direct `CreateStructGEP` on `arena_cons_type`
- 🚧 **1 Stub Implementation**: [`codegenFoldRight()`](../lib/backend/llvm_codegen.cpp:5807) - Needs full implementation

---

## Complete Function Analysis

### Group A: Map Functions (Priority: CRITICAL)

#### 1. codegenMapSingleList (Lines 5182-5273) 🔴 CRITICAL
**Current Implementation:**
- **Line 5218-5223**: Direct `CreateStructGEP(arena_cons_type, input_cons_ptr, 0)` for car extraction
- **Line 5250-5252**: Direct `CreateStructGEP(arena_cons_type, tail_cons_ptr, 1)` for tail cdr update
- **Line 5258-5259**: Direct `CreateStructGEP(arena_cons_type, input_cons_ptr, 1)` for input cdr iteration

**Migration Required:**
```cpp
// BEFORE (Line 5218-5223):
Value* input_car_ptr = builder->CreateStructGEP(arena_cons_type, input_cons_ptr, 0);
Value* input_element = builder->CreateLoad(Type::getInt64Ty(*context), input_car_ptr);

// AFTER:
Value* is_car = ConstantInt::get(Type::getInt1Ty(*context), 0);
Value* car_type = builder->CreateCall(arena_tagged_cons_get_type_func, {input_cons_ptr, is_car});
Value* car_base = builder->CreateAnd(car_type, ConstantInt::get(Type::getInt8Ty(*context), 0x0F));
Value* is_double = builder->CreateICmpEQ(car_base, ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_DOUBLE));

BasicBlock* double_car = BasicBlock::Create(*context, "map_car_double", current_func);
BasicBlock* int_car = BasicBlock::Create(*context, "map_car_int", current_func);
BasicBlock* merge_car = BasicBlock::Create(*context, "map_car_merge", current_func);

builder->CreateCondBr(is_double, double_car, int_car);

builder->SetInsertPoint(double_car);
Value* car_double = builder->CreateCall(arena_tagged_cons_get_double_func, {input_cons_ptr, is_car});
builder->CreateBr(merge_car);

builder->SetInsertPoint(int_car);
Value* car_int = builder->CreateCall(arena_tagged_cons_get_int64_func, {input_cons_ptr, is_car});
builder->CreateBr(merge_car);

builder->SetInsertPoint(merge_car);
PHINode* input_element = builder->CreatePHI(Type::getInt64Ty(*context), 2);
input_element->addIncoming(builder->CreateBitCast(car_double, Type::getInt64Ty(*context)), double_car);
input_element->addIncoming(car_int, int_car);
```

**Changes Required:**
1. Replace car extraction (line 5218-5223) with tagged helper + type branching
2. Replace tail cdr access (line 5250-5252) with `arena_tagged_cons_set_ptr_func`
3. Replace input cdr iteration (line 5258-5259) with `arena_tagged_cons_get_ptr_func`
4. Update cons cell creation to use `codegenTaggedArenaConsCell()` for type preservation

**Test Requirements:**
- Mixed-type list: `(map + (list 1 2.0 3) (list 4.0 5 6))`
- Integer-only: `(map (lambda (x) (* x 2)) (list 1 2 3))`
- Double-only: `(map sqrt (list 1.0 4.0 9.0))`

---

#### 2. codegenMapMultiList (Lines 5276-5393) 🔴 CRITICAL
**Current Implementation:**
- **Lines 5329-5334**: Direct `CreateStructGEP(arena_cons_type, cons_ptr, 0)` for multi-list car extraction (loop)
- **Line 5367-5368**: Direct `CreateStructGEP(arena_cons_type, tail_cons_ptr, 1)` for tail cdr update
- **Lines 5376-5379**: Direct `CreateStructGEP(arena_cons_type, cons_ptr, 1)` for cdr iteration (loop over all lists)

**Migration Required:**
Same pattern as `codegenMapSingleList`, but must handle multiple lists:

```cpp
// BEFORE (Lines 5329-5334):
for (size_t i = 0; i < current_ptrs.size(); i++) {
    Value* current_val = builder->CreateLoad(Type::getInt64Ty(*context), current_ptrs[i]);
    Value* cons_ptr = builder->CreateIntToPtr(current_val, builder->getPtrTy());
    Value* car_ptr = builder->CreateStructGEP(arena_cons_type, cons_ptr, 0);
    Value* element = builder->CreateLoad(Type::getInt64Ty(*context), car_ptr);
    proc_args.push_back(element);
}

// AFTER:
for (size_t i = 0; i < current_ptrs.size(); i++) {
    Value* current_val = builder->CreateLoad(Type::getInt64Ty(*context), current_ptrs[i]);
    Value* cons_ptr = builder->CreateIntToPtr(current_val, builder->getPtrTy());
    
    // Extract element with type checking
    TypedValue element = extractCarAsTaggedValue(cons_ptr); // Use existing helper
    proc_args.push_back(typedValueToLLVM(element));
}
```

**Changes Required:**
1. Replace multi-list car extraction loop (lines 5329-5334) with type-aware extraction
2. Replace tail cdr access (line 5367-5368) with `arena_tagged_cons_set_ptr_func`
3. Replace all cdr iterations (lines 5376-5379) with `arena_tagged_cons_get_ptr_func`
4. Handle type promotion when calling procedure with mixed-type arguments

**Test Requirements:**
- Two mixed-type lists: `(map + (list 1 2.0) (list 3.0 4))`
- Three lists: `(map (lambda (x y z) (+ x y z)) (list 1 2) (list 3 4) (list 5 6))`
- Type preservation: Verify results maintain correct types

---

### Group B: Filter/Search Functions (Priority: HIGH)

#### 3. codegenFilter (Lines 5396-5510) 🟡 HIGH
**Current Implementation:**
- **Lines 5453-5458**: Direct `CreateStructGEP(arena_cons_type, input_cons_ptr, 0)` for car extraction
- **Line 5487-5489**: Direct `CreateStructGEP(arena_cons_type, tail_cons_ptr, 1)` for tail cdr update
- **Lines 5494-5496**: Direct `CreateStructGEP(arena_cons_type, input_cons_ptr, 1)` for input cdr iteration

**Migration Pattern:**
Identical to `codegenMapSingleList` - extract car with type checking, iterate with `arena_tagged_cons_get_ptr_func`.

**Changes Required:**
1. Replace car extraction (lines 5453-5458) with tagged helper + type branching
2. Replace tail cdr updates (line 5487-5489) with `arena_tagged_cons_set_ptr_func`
3. Replace input cdr iteration (lines 5494-5496) with `arena_tagged_cons_get_ptr_func`

**Test Requirements:**
- Filter mixed types: `(filter (lambda (x) (> x 5)) (list 1 2.0 10 3.5 20))`
- Predicate with doubles: `(filter (lambda (x) (< x 10.0)) (list 5.0 15.0 7.5))`

---

#### 4. codegenMember (Lines 5653-5723) 🟡 HIGH
**Current Implementation:**
- **Lines 5687-5692**: Direct `CreateStructGEP(arena_cons_type, cons_ptr, 0)` for car extraction
- **Lines 5711-5713**: Direct `CreateStructGEP(arena_cons_type, alist_cons_ptr, 1)` for cdr iteration

**Migration Required:**
```cpp
// BEFORE (Lines 5687-5692):
Value* car_ptr = builder->CreateStructGEP(arena_cons_type, cons_ptr, 0);
Value* current_element = builder->CreateLoad(Type::getInt64Ty(*context), car_ptr);

// AFTER:
Value* is_car = ConstantInt::get(Type::getInt1Ty(*context), 0);
Value* current_element = builder->CreateCall(arena_tagged_cons_get_int64_func, {cons_ptr, is_car});
// Note: Assumes int64 for simplicity; full implementation needs type checking
```

**Changes Required:**
1. Replace car extraction (lines 5687-5692) with type-aware helper
2. Replace cdr iteration (lines 5711-5713) with `arena_tagged_cons_get_ptr_func`
3. Add type-aware comparison for `equal` vs `eq` semantics

**Test Requirements:**
- Mixed-type member: `(member 2.0 (list 1 2.0 3))`
- Exact match: `(memq 2 (list 1 2 3))`
- Value equality: `(memv 2.0 (list 1.0 2.0 3.0))`

---

#### 5. codegenFind (Lines 6138-6240) 🟡 HIGH
**Current Implementation:**
- **Lines 6184-6189**: Direct `CreateStructGEP(arena_cons_type, cons_ptr, 0)` for car extraction
- **Lines 6203-6205**: Direct `CreateStructGEP(arena_cons_type, cons_ptr, 1)` for cdr iteration
- **Lines 6224-6226**: Direct `CreateStructGEP(arena_cons_type, found_cons_ptr, 0)` for found element car

**Changes Required:**
1. Replace all car extractions with type-aware helpers
2. Replace cdr iterations with `arena_tagged_cons_get_ptr_func`
3. Return found element with proper type (currently assumes int64)

**Test Requirements:**
- Find double: `(find (lambda (x) (> x 5.0)) (list 1.0 2.0 10.0))`
- Find integer: `(find even? (list 1 3 4 5))`
- Not found: `(find (lambda (x) (< x 0)) (list 1 2 3))`

---

### Group C: Reduction Functions (Priority: HIGH)

#### 6. codegenFold (Lines 5513-5585) 🟡 HIGH
**Current Implementation:**
- **Lines 5561-5566**: Direct `CreateStructGEP(arena_cons_type, input_cons_ptr, 0)` for car extraction
- **Lines 5575-5577**: Direct `CreateStructGEP(arena_cons_type, input_cons_ptr, 1)` for input cdr iteration

**Migration Required:**
Must handle mixed-type accumulator and elements:

```cpp
// BEFORE (Lines 5561-5566):
Value* input_car_ptr = builder->CreateStructGEP(arena_cons_type, input_cons_ptr, 0);
Value* input_element = builder->CreateLoad(Type::getInt64Ty(*context), input_car_ptr);

// AFTER:
TypedValue input_element = extractCarAsTypedValue(input_cons_ptr);
// Then pass to procedure with type-aware calling convention
```

**Changes Required:**
1. Replace car extraction (lines 5561-5566) with type-aware helper
2. Replace cdr iteration (lines 5575-5577) with `arena_tagged_cons_get_ptr_func`
3. Handle type promotion in accumulator (int64 + double → double)
4. Modify procedure call to accept `TypedValue` arguments

**Test Requirements:**
- Mixed-type fold: `(fold + 0 (list 1 2.0 3))` → 6.0 (promoted to double)
- Integer-only: `(fold * 1 (list 2 3 4))` → 24
- Custom accumulator: `(fold cons '() (list 1 2 3))` → reverse

---

#### 7. codegenFoldRight (Lines 5807-5810) 🚧 STUB - NEEDS FULL IMPLEMENTATION
**Current Implementation:**
```cpp
Value* codegenFoldRight(const eshkol_operations_t* op) {
    eshkol_warn("fold-right not yet implemented");
    return ConstantInt::get(Type::getInt64Ty(*context), 0);
}
```

**Implementation Required:**
Must implement from scratch using tagged helpers:

1. **Recursive or Two-Pass Approach**: Either recursively fold from right, or:
   - Pass 1: Reverse the list
   - Pass 2: Apply left fold on reversed list
   - Reverse result if needed

2. **Type-Aware Implementation**:
   - Use `arena_tagged_cons_get_type_func` for type detection
   - Use `arena_tagged_cons_get_double_func` / `arena_tagged_cons_get_int64_func` for extraction
   - Use `arena_tagged_cons_get_ptr_func` for cdr iteration
   - Handle type promotion in accumulator

**Example Semantics:**
```scheme
(fold-right cons '() (list 1 2 3))  → (1 2 3)  ; identity
(fold-right + 0 (list 1 2 3))       → 6        ; same as fold for +
(fold-right - 0 (list 1 2 3))       → 2        ; 1 - (2 - (3 - 0))
```

**Changes Required:**
1. Implement complete function body with tagged helpers
2. Add type promotion logic
3. Choose recursive vs two-pass approach
4. Add proper arena scoping

**Test Requirements:**
- Basic: `(fold-right cons '() (list 1 2 3))`
- Mixed types: `(fold-right + 0 (list 1 2.0 3))`
- Non-commutative: `(fold-right - 0 (list 10 3 2))` → 9

---

### Group D: Iteration Functions (Priority: MEDIUM)

#### 8. codegenForEachSingleList (Lines 5759-5804) 🟠 MEDIUM
**Current Implementation:**
- **Lines 5784-5789**: Direct `CreateStructGEP(arena_cons_type, input_cons_ptr, 0)` for car extraction
- **Lines 5794-5796**: Direct `CreateStructGEP(arena_cons_type, input_cons_ptr, 1)` for input cdr iteration

**Migration Required:**
Simple pattern - extract car with type awareness, iterate with cdr helper:

```cpp
// BEFORE (Lines 5784-5789):
Value* input_car_ptr = builder->CreateStructGEP(arena_cons_type, input_cons_ptr, 0);
Value* input_element = builder->CreateLoad(Type::getInt64Ty(*context), input_car_ptr);

// AFTER:
TypedValue input_element = extractCarAsTypedValue(input_cons_ptr);
```

**Changes Required:**
1. Replace car extraction (lines 5784-5789) with type-aware helper
2. Replace cdr iteration (lines 5794-5796) with `arena_tagged_cons_get_ptr_func`
3. Pass typed values to procedure (may need type conversion)

**Test Requirements:**
- Side effects: `(for-each display (list 1 2.0 3))` → outputs "12.03"
- Mixed types: `(for-each (lambda (x) (display x) (newline)) (list 1 2.0))`

---

### Group E: Association Functions (Priority: MEDIUM)

#### 9. codegenAssoc (Lines 5813-5923) 🟠 MEDIUM
**Current Implementation:**
- **Lines 5848-5853**: Direct `CreateStructGEP(arena_cons_type, alist_cons_ptr, 0)` for alist car (pair)
- **Lines 5863-5866**: Direct `CreateStructGEP(arena_cons_type, pair_cons_ptr, 0)` for pair key
- **Lines 5884-5886**: Direct `CreateStructGEP(arena_cons_type, alist_cons_ptr, 1)` for alist cdr iteration
- **Lines 5907-5909**: Direct `CreateStructGEP(arena_cons_type, found_cons_ptr, 0)` for found pair

**Migration Required:**
Complex - two levels of cons cells (alist contains pairs):

```cpp
// Extract pair from alist (BEFORE - Lines 5848-5853):
Value* alist_car_ptr = builder->CreateStructGEP(arena_cons_type, alist_cons_ptr, 0);
Value* current_pair = builder->CreateLoad(Type::getInt64Ty(*context), alist_car_ptr);

// AFTER:
Value* is_car = ConstantInt::get(Type::getInt1Ty(*context), 0);
Value* current_pair = builder->CreateCall(arena_tagged_cons_get_ptr_func, {alist_cons_ptr, is_car});

// Then extract key from pair (BEFORE - Lines 5863-5866):
Value* pair_key_ptr = builder->CreateStructGEP(arena_cons_type, pair_cons_ptr, 0);
Value* pair_key = builder->CreateLoad(Type::getInt64Ty(*context), pair_key_ptr);

// AFTER:
TypedValue pair_key = extractCarAsTypedValue(pair_cons_ptr);
```

**Changes Required:**
1. Replace alist car extraction (lines 5848-5853) with `arena_tagged_cons_get_ptr_func`
2. Replace pair key extraction (lines 5863-5866) with type-aware helper
3. Replace alist cdr iteration (lines 5884-5886) with `arena_tagged_cons_get_ptr_func`
4. Replace found pair extraction (lines 5907-5909) with `arena_tagged_cons_get_ptr_func`
5. Add type-aware key comparison (not just `CreateICmpEQ`)

**Test Requirements:**
- Mixed-type alist: `(assoc 2.0 (list (cons 1 "a") (cons 2.0 "b")))`
- Integer keys: `(assoc 2 (list (cons 1 10) (cons 2 20)))`
- Not found: `(assoc 5 (list (cons 1 10) (cons 2 20)))`

---

### Group F: List Slicing Functions (Priority: MEDIUM)

#### 10. codegenTake (Lines 5979-6079) 🟠 MEDIUM
**Current Implementation:**
- **Lines 6027-6032**: Direct `CreateStructGEP(arena_cons_type, input_cons_ptr, 0)` for car extraction
- **Lines 6054-6056**: Direct `CreateStructGEP(arena_cons_type, tail_cons_ptr, 1)` for tail cdr update
- **Lines 6061-6063**: Direct `CreateStructGEP(arena_cons_type, input_cons_ptr, 1)` for input cdr iteration

**Migration Pattern:**
Same as `codegenMapSingleList` - extract, build, iterate.

**Changes Required:**
1. Replace car extraction (lines 6027-6032) with type-aware helper
2. Replace tail cdr updates (lines 6054-6056) with `arena_tagged_cons_set_ptr_func`
3. Replace input cdr iteration (lines 6061-6063) with `arena_tagged_cons_get_ptr_func`
4. Use `codegenTaggedArenaConsCell()` instead of `codegenArenaConsCell()`

**Test Requirements:**
- Mixed types: `(take (list 1 2.0 3 4.0) 2)` → `(1 2.0)`
- Edge case: `(take (list 1 2) 5)` → `(1 2)` (list shorter than n)
- Zero: `(take (list 1 2 3) 0)` → `()`

---

#### 11. codegenSplitAt (Lines 6392-6496) 🟠 MEDIUM
**Current Implementation:**
- **Lines 6440-6445**: Direct `CreateStructGEP(arena_cons_type, input_cons_ptr, 0)` for car extraction
- **Lines 6467-6469**: Direct `CreateStructGEP(arena_cons_type, prefix_tail_cons_ptr, 1)` for prefix tail cdr
- **Lines 6474-6476**: Direct `CreateStructGEP(arena_cons_type, input_cons_ptr, 1)` for input cdr iteration

**Migration Pattern:**
Similar to `codegenTake`, but returns pair `(prefix . suffix)`.

**Changes Required:**
1. Replace car extraction (lines 6440-6445) with type-aware helper
2. Replace prefix tail cdr updates (lines 6467-6469) with `arena_tagged_cons_set_ptr_func`
3. Replace input cdr iteration (lines 6474-6476) with `arena_tagged_cons_get_ptr_func`
4. Create result pair using `codegenTaggedArenaConsCell()`

**Test Requirements:**
- Mixed types: `(split-at (list 1 2.0 3 4.0) 2)` → `((1 2.0) . (3 4.0))`
- Empty prefix: `(split-at (list 1 2 3) 0)` → `(() . (1 2 3))`
- Full list: `(split-at (list 1 2) 5)` → `((1 2) . ())`

---

#### 12. codegenPartition (Lines 6242-6389) 🟠 MEDIUM
**Current Implementation:**
- **Lines 6303-6308**: Direct `CreateStructGEP(arena_cons_type, input_cons_ptr, 0)` for car extraction
- **Lines 6337-6339**: Direct `CreateStructGEP(arena_cons_type, true_tail_cons_ptr, 1)` for true tail cdr
- **Lines 6363-6365**: Direct `CreateStructGEP(arena_cons_type, false_tail_cons_ptr, 1)` for false tail cdr
- **Lines 6370-6372**: Direct `CreateStructGEP(arena_cons_type, input_cons_ptr, 1)` for input cdr iteration

**Migration Pattern:**
Two result lists (true/false), similar to filter but dual outputs.

**Changes Required:**
1. Replace car extraction (lines 6303-6308) with type-aware helper
2. Replace true tail cdr updates (lines 6337-6339) with `arena_tagged_cons_set_ptr_func`
3. Replace false tail cdr updates (lines 6363-6365) with `arena_tagged_cons_set_ptr_func`
4. Replace input cdr iteration (lines 6370-6372) with `arena_tagged_cons_get_ptr_func`
5. Create both result lists using `codegenTaggedArenaConsCell()`

**Test Requirements:**
- Mixed types: `(partition (lambda (x) (> x 5)) (list 1 2.0 10 3.5))`
- All true: `(partition (lambda (x) #t) (list 1 2 3))`
- All false: `(partition (lambda (x) #f) (list 1 2 3))`

---

#### 13. codegenRemove (Lines 6499-6604) 🟠 MEDIUM
**Current Implementation:**
- **Lines 6542-6547**: Direct `CreateStructGEP(arena_cons_type, input_cons_ptr, 0)` for car extraction
- **Lines 6582-6584**: Direct `CreateStructGEP(arena_cons_type, tail_cons_ptr, 1)` for tail cdr update
- **Lines 6589-6591**: Direct `CreateStructGEP(arena_cons_type, input_cons_ptr, 1)` for input cdr iteration

**Migration Pattern:**
Inverse of filter - keep non-matching elements.

**Changes Required:**
1. Replace car extraction (lines 6542-6547) with type-aware helper
2. Replace tail cdr updates (lines 6582-6584) with `arena_tagged_cons_set_ptr_func`
3. Replace input cdr iteration (lines 6589-6591) with `arena_tagged_cons_get_ptr_func`
4. Add type-aware comparison for equality

**Test Requirements:**
- Mixed types: `(remove 2.0 (list 1 2.0 3 2.0 4))` → `(1 3 4)`
- Value equality: `(remove 2 (list 1 2 3 2 4))` → `(1 3 4)`
- No match: `(remove 5 (list 1 2 3))` → `(1 2 3)`

---

### Group G: Boundary Functions (Priority: LOW - Already Complete)

#### 14. codegenLast (Lines 6607-6702) ✅ COMPLETE
**Status:** Already using tagged helpers!

**Current Implementation:**
- **Line 6658-6660**: ✅ Uses `arena_tagged_cons_get_ptr_func(cons_ptr, is_cdr)`
- **Lines 6668-6671**: ✅ Uses `arena_tagged_cons_get_type_func(last_cons_ptr, is_car)`
- **Lines 6681-6686**: ✅ Uses `arena_tagged_cons_get_double_func` and `arena_tagged_cons_get_int64_func`

**No Changes Required** - This function serves as the reference implementation for others!

**Test Requirements (Already Passing):**
- Mixed types: `(last (list 1 2.0 3))` → `3`
- Double: `(last (list 1.0 2.0))` → `2.0`
- Single element: `(last (list 42))` → `42`

---

#### 15. codegenLastPair (Lines 6705-6768) ✅ COMPLETE
**Status:** Already using tagged helpers!

**Current Implementation:**
- **Lines 6744-6746**: ✅ Uses `arena_tagged_cons_get_ptr_func(cons_ptr, is_cdr)`

**No Changes Required** - Another reference implementation!

**Test Requirements (Already Passing):**
- Normal: `(last-pair (list 1 2 3))` → `(3)`
- Single element: `(last-pair (list 42))` → `(42)`
- Improper list: `(last-pair (list* 1 2 3))` → `(2 . 3)`

---

## Migration Priority Order

### Phase 1: Critical Path (Sessions 005-010)
**Goal:** Enable mixed-type map operations

1. **Session 005**: [`codegenMapSingleList()`](../lib/backend/llvm_codegen.cpp:5182) - Foundation for all map operations
2. **Session 006**: [`codegenMapMultiList()`](../lib/backend/llvm_codegen.cpp:5276) - Multi-list map support
3. **Session 007-008**: Create comprehensive map test suite
4. **Session 009**: [`codegenFilter()`](../lib/backend/llvm_codegen.cpp:5396) - Similar to map, high value
5. **Session 010**: Create filter test suite

### Phase 2: Reduction & Search (Sessions 011-014)
**Goal:** Enable accumulator-based and search operations

6. **Session 011**: [`codegenFold()`](../lib/backend/llvm_codegen.cpp:5513) - Core reduction primitive
7. **Session 012**: [`codegenFoldRight()`](../lib/backend/llvm_codegen.cpp:5807) - Full implementation needed
8. **Session 013**: [`codegenFind()`](../lib/backend/llvm_codegen.cpp:6138) - Search primitive
9. **Session 014**: Create fold/find test suite

### Phase 3: Association & Membership (Sessions 015-016)
**Goal:** Enable alist and membership operations

10. **Session 015**: [`codegenMember()`](../lib/backend/llvm_codegen.cpp:5653) - Membership testing
11. **Session 016**: [`codegenAssoc()`](../lib/backend/llvm_codegen.cpp:5813) - Association lists
12. **Session 016**: Create assoc/member test suite

### Phase 4: List Slicing & Partitioning (Sessions 017-019)
**Goal:** Enable list manipulation operations

13. **Session 017**: [`codegenTake()`](../lib/backend/llvm_codegen.cpp:5979) - Prefix extraction
14. **Session 018**: [`codegenSplitAt()`](../lib/backend/llvm_codegen.cpp:6392) - Two-way split
15. **Session 019**: [`codegenPartition()`](../lib/backend/llvm_codegen.cpp:6242) - Predicate-based split
16. **Session 019**: [`codegenRemove()`](../lib/backend/llvm_codegen.cpp:6499) - Element removal

### Phase 5: Validation & Integration (Session 020)
**Goal:** Comprehensive testing and documentation

17. **Session 020**: 
    - Run all test suites
    - Update [`docs/BUILD_STATUS.md`](BUILD_STATUS.md)
    - Verify [`codegenLast()`](../lib/backend/llvm_codegen.cpp:6607) and [`codegenLastPair()`](../lib/backend/llvm_codegen.cpp:6705) remain working
    - Integration testing with real-world examples

---

## Detailed Change Specifications

### Common Migration Pattern

**All functions (except codegenLast, codegenLastPair, codegenFoldRight) follow this pattern:**

#### Step 1: Car Extraction Replacement
```cpp
// OLD (Direct memory access):
StructType* arena_cons_type = StructType::get(Type::getInt64Ty(*context), Type::getInt64Ty(*context));
Value* cons_ptr = builder->CreateIntToPtr(current_val, builder->getPtrTy());
Value* car_ptr = builder->CreateStructGEP(arena_cons_type, cons_ptr, 0);
Value* element = builder->CreateLoad(Type::getInt64Ty(*context), car_ptr);

// NEW (Type-safe tagged helper):
Value* cons_ptr = builder->CreateIntToPtr(current_val, builder->getPtrTy());
Value* is_car = ConstantInt::get(Type::getInt1Ty(*context), 0);
Value* car_type = builder->CreateCall(arena_tagged_cons_get_type_func, {cons_ptr, is_car});
Value* car_base = builder->CreateAnd(car_type, ConstantInt::get(Type::getInt8Ty(*context), 0x0F));
Value* is_double = builder->CreateICmpEQ(car_base, ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_DOUBLE));

BasicBlock* double_block = BasicBlock::Create(*context, "car_double", current_func);
BasicBlock* int_block = BasicBlock::Create(*context, "car_int", current_func);
BasicBlock* merge_block = BasicBlock::Create(*context, "car_merge", current_func);

builder->CreateCondBr(is_double, double_block, int_block);

builder->SetInsertPoint(double_block);
Value* car_double = builder->CreateCall(arena_tagged_cons_get_double_func, {cons_ptr, is_car});
builder->CreateBr(merge_block);

builder->SetInsertPoint(int_block);
Value* car_int = builder->CreateCall(arena_tagged_cons_get_int64_func, {cons_ptr, is_car});
builder->CreateBr(merge_block);

builder->SetInsertPoint(merge_block);
// Merge results - either keep as separate types or promote to common type
```

#### Step 2: Cdr Iteration Replacement
```cpp
// OLD (Direct memory access):
Value* cdr_ptr = builder->CreateStructGEP(arena_cons_type, cons_ptr, 1);
Value* cdr_val = builder->CreateLoad(Type::getInt64Ty(*context), cdr_ptr);

// NEW (Type-safe tagged helper):
Value* is_cdr = ConstantInt::get(Type::getInt1Ty(*context), 1);
Value* cdr_val = builder->CreateCall(arena_tagged_cons_get_ptr_func, {cons_ptr, is_cdr});
```

#### Step 3: Tail Cdr Update Replacement
```cpp
// OLD (Direct memory access):
Value* tail_cdr_ptr = builder->CreateStructGEP(arena_cons_type, tail_cons_ptr, 1);
builder->CreateStore(new_result_cons, tail_cdr_ptr);

// NEW (Type-safe tagged helper):
Value* is_cdr = ConstantInt::get(Type::getInt1Ty(*context), 1);
Value* new_result_as_uint64 = new_result_cons; // Already int64
builder->CreateCall(arena_tagged_cons_set_ptr_func, 
    {tail_cons_ptr, is_cdr, new_result_as_uint64, 
     ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_CONS_PTR)});
```

#### Step 4: Cons Cell Creation
```cpp
// OLD:
Value* new_cons = codegenArenaConsCell(car_val, cdr_val);

// NEW:
TypedValue car_typed = detectValueType(car_val);
TypedValue cdr_typed = detectValueType(cdr_val);
Value* new_cons = codegenTaggedArenaConsCell(car_typed, cdr_typed);
```

---

## Per-Function Change Specifications

### 1. codegenMapSingleList

**Lines to Change:**
- **5218-5223**: Car extraction → Use type-aware helper with branching
- **5227**: Procedure call → May need type conversion for mixed args
- **5229**: Cons creation → Use `codegenTaggedArenaConsCell()`
- **5250-5252**: Tail cdr update → Use `arena_tagged_cons_set_ptr_func`
- **5258-5259**: Input cdr iteration → Use `arena_tagged_cons_get_ptr_func`

**Estimated Changes:** ~40 lines (add type branching logic)

---

### 2. codegenMapMultiList

**Lines to Change:**
- **5329-5334**: Multi-list car extraction loop → Use type-aware helpers
- **5342**: Procedure call → Handle mixed-type arguments
- **5345**: Cons creation → Use `codegenTaggedArenaConsCell()`
- **5367-5368**: Tail cdr update → Use `arena_tagged_cons_set_ptr_func`
- **5376-5379**: All cdr iterations → Use `arena_tagged_cons_get_ptr_func` in loop

**Estimated Changes:** ~50 lines (add type branching for each list)

---

### 3. codegenFilter

**Lines to Change:**
- **5453-5458**: Car extraction → Use type-aware helper
- **5461**: Predicate call → Handle typed argument
- **5469**: Cons creation → Use `codegenTaggedArenaConsCell()`
- **5487-5489**: Tail cdr update → Use `arena_tagged_cons_set_ptr_func`
- **5494-5496**: Input cdr iteration → Use `arena_tagged_cons_get_ptr_func`

**Estimated Changes:** ~35 lines

---

### 4. codegenFold

**Lines to Change:**
- **5561-5566**: Car extraction → Use type-aware helper
- **5572**: Procedure call → Handle mixed-type accumulator and element
- **5575-5577**: Input cdr iteration → Use `arena_tagged_cons_get_ptr_func`

**Estimated Changes:** ~30 lines (+ accumulator type tracking)

---

### 5. codegenFoldRight

**Implementation Required:** Complete function (currently stub at lines 5807-5810)

**Recommended Approach:**
```cpp
Value* codegenFoldRight(const eshkol_operations_t* op) {
    // Validate arguments
    if (op->call_op.num_vars != 3) {
        eshkol_warn("fold-right requires 3 arguments: proc, init, list");
        return nullptr;
    }
    
    // Option 1: Two-pass (reverse, fold, reverse)
    // 1. Reverse input list
    // 2. Apply left fold
    // 3. Reverse result if needed (depends on operation)
    
    // Option 2: Build result list during traversal, then process
    // 1. Convert list to vector
    // 2. Iterate from end to start
    // 3. Build result
}
```

**Estimated Changes:** ~120 lines (new implementation)

---

### 6. codegenForEachSingleList

**Lines to Change:**
- **5784-5789**: Car extraction → Use type-aware helper
- **5792**: Procedure call → Handle typed argument (discard result)
- **5794-5796**: Input cdr iteration → Use `arena_tagged_cons_get_ptr_func`

**Estimated Changes:** ~25 lines

---

### 7. codegenMember

**Lines to Change:**
- **5687-5692**: Car extraction → Use type-aware helper
- **5696-5701**: Comparison → Type-aware equality check
- **5711-5713**: Cdr iteration → Use `arena_tagged_cons_get_ptr_func`

**Estimated Changes:** ~30 lines (+ type-aware comparison)

---

### 8. codegenAssoc

**Lines to Change:**
- **5848-5853**: Alist car (pair) extraction → Use `arena_tagged_cons_get_ptr_func`
- **5863-5866**: Pair key extraction → Use type-aware helper
- **5872-5875**: Key comparison → Type-aware equality
- **5884-5886**: Alist cdr iteration → Use `arena_tagged_cons_get_ptr_func`
- **5907-5909**: Found pair extraction → Use `arena_tagged_cons_get_ptr_func`

**Estimated Changes:** ~40 lines (nested cons cells)

---

### 9. codegenTake

**Lines to Change:**
- **6027-6032**: Car extraction → Use type-aware helper
- **6035**: Cons creation → Use `codegenTaggedArenaConsCell()`
- **6054-6056**: Tail cdr update → Use `arena_tagged_cons_set_ptr_func`
- **6061-6063**: Input cdr iteration → Use `arena_tagged_cons_get_ptr_func`

**Estimated Changes:** ~35 lines

---

### 10. codegenFind

**Lines to Change:**
- **6184-6189**: Car extraction → Use type-aware helper
- **6192**: Predicate call → Handle typed argument
- **6203-6205**: Cdr iteration → Use `arena_tagged_cons_get_ptr_func`
- **6224-6226**: Found element car → Use type-aware helper for return

**Estimated Changes:** ~40 lines (return type handling)

---

### 11. codegenPartition

**Lines to Change:**
- **6303-6308**: Car extraction → Use type-aware helper
- **6311**: Predicate call → Handle typed argument
- **6318-6319**: True cons creation → Use `codegenTaggedArenaConsCell()`
- **6337-6339**: True tail cdr → Use `arena_tagged_cons_set_ptr_func`
- **6345**: False cons creation → Use `codegenTaggedArenaConsCell()`
- **6363-6365**: False tail cdr → Use `arena_tagged_cons_set_ptr_func`
- **6370-6372**: Input cdr iteration → Use `arena_tagged_cons_get_ptr_func`
- **6383**: Result pair → Use `codegenTaggedArenaConsCell()`

**Estimated Changes:** ~45 lines (dual result lists)

---

### 12. codegenSplitAt

**Lines to Change:**
- **6440-6445**: Car extraction → Use type-aware helper
- **6448**: Cons creation → Use `codegenTaggedArenaConsCell()`
- **6467-6469**: Prefix tail cdr → Use `arena_tagged_cons_set_ptr_func`
- **6474-6476**: Input cdr iteration → Use `arena_tagged_cons_get_ptr_func`
- **6490**: Result pair → Use `codegenTaggedArenaConsCell()`

**Estimated Changes:** ~40 lines

---

### 13. codegenRemove

**Lines to Change:**
- **6542-6547**: Car extraction → Use type-aware helper
- **6550-6556**: Comparison → Type-aware equality
- **6564**: Cons creation → Use `codegenTaggedArenaConsCell()`
- **6582-6584**: Tail cdr update → Use `arena_tagged_cons_set_ptr_func`
- **6589-6591**: Input cdr iteration → Use `arena_tagged_cons_get_ptr_func`

**Estimated Changes:** ~35 lines

---

## Helper Function Dependencies

### Existing Helpers (Ready to Use)
✅ [`arena_tagged_cons_get_type_func`](../lib/backend/llvm_codegen.cpp:664) - Get type tag from car/cdr  
✅ [`arena_tagged_cons_get_int64_func`](../lib/backend/llvm_codegen.cpp:518) - Extract int64 value  
✅ [`arena_tagged_cons_get_double_func`](../lib/backend/llvm_codegen.cpp:538) - Extract double value  
✅ [`arena_tagged_cons_get_ptr_func`](../lib/backend/llvm_codegen.cpp:558) - Extract cons pointer  
✅ [`arena_tagged_cons_set_int64_func`](../lib/backend/llvm_codegen.cpp:578) - Set int64 value  
✅ [`arena_tagged_cons_set_double_func`](../lib/backend/llvm_codegen.cpp:600) - Set double value  
✅ [`arena_tagged_cons_set_ptr_func`](../lib/backend/llvm_codegen.cpp:622) - Set cons pointer  
✅ [`arena_tagged_cons_set_null_func`](../lib/backend/llvm_codegen.cpp:644) - Set null value  
✅ [`codegenTaggedArenaConsCell()`](../lib/backend/llvm_codegen.cpp:1012) - Create tagged cons cell  
✅ [`extractCarAsTaggedValue()`](../lib/backend/llvm_codegen.cpp:1168) - Extract car as tagged value  
✅ [`extractCdrAsTaggedValue()`](../lib/backend/llvm_codegen.cpp:1207) - Extract cdr as tagged value  
✅ [`detectValueType()`](../lib/backend/llvm_codegen.cpp:1246) - Detect LLVM value type  

### Helper to Create (If Needed)

**extractCarAsTypedValue()** - Simplified version that doesn't pack into tagged_value_type:
```cpp
TypedValue extractCarAsTypedValue(Value* cons_ptr_int) {
    Value* cons_ptr = builder->CreateIntToPtr(cons_ptr_int, builder->getPtrTy());
    Value* is_car = ConstantInt::get(Type::getInt1Ty(*context), 0);
    Value* car_type = builder->CreateCall(arena_tagged_cons_get_type_func, {cons_ptr, is_car});
    Value* car_base = builder->CreateAnd(car_type, ConstantInt::get(Type::getInt8Ty(*context), 0x0F));
    
    // Branch and extract based on type
    // ... (similar to extractCarAsTaggedValue but return TypedValue instead)
}
```

---

## Test Coverage Plan

### Test Suite Structure

**Location:** `tests/higher_order_migration_tests.esk`

#### Suite 1: Map Operations (Sessions 007-008)
```scheme
; Single-list map
(define test-map-single-int (map (lambda (x) (* x 2)) (list 1 2 3)))
(define test-map-single-double (map sqrt (list 1.0 4.0 9.0)))
(define test-map-single-mixed (map (lambda (x) (+ x 1)) (list 1 2.0 3)))

; Multi-list map
(define test-map-multi-2-int (map + (list 1 2 3) (list 4 5 6)))
(define test-map-multi-2-mixed (map + (list 1 2.0) (list 3.0 4)))
(define test-map-multi-3-lists (map (lambda (x y z) (+ x y z)) 
                                     (list 1 2) (list 3 4) (list 5 6)))

; Edge cases
(define test-map-empty (map (lambda (x) x) '()))
(define test-map-single-elem (map (lambda (x) (* x 2)) (list 42)))
```

#### Suite 2: Filter Operations (Session 010)
```scheme
; Basic filter
(define test-filter-int (filter (lambda (x) (> x 5)) (list 1 3 7 2 9)))
(define test-filter-double (filter (lambda (x) (< x 10.0)) (list 5.0 15.0 7.5)))
(define test-filter-mixed (filter (lambda (x) (> x 5)) (list 1 2.0 10 3.5 20)))

; Edge cases
(define test-filter-none (filter (lambda (x) #f) (list 1 2 3)))
(define test-filter-all (filter (lambda (x) #t) (list 1 2 3)))
```

#### Suite 3: Fold Operations (Session 014)
```scheme
; Fold left
(define test-fold-sum-int (fold + 0 (list 1 2 3 4)))
(define test-fold-sum-mixed (fold + 0 (list 1 2.0 3)))
(define test-fold-product (fold * 1 (list 2 3 4)))
(define test-fold-reverse (fold cons '() (list 1 2 3)))

; Fold right (once implemented)
(define test-fold-right-cons (fold-right cons '() (list 1 2 3)))
(define test-fold-right-subtract (fold-right - 0 (list 10 3 2)))
(define test-fold-right-mixed (fold-right + 0 (list 1 2.0 3)))
```

#### Suite 4: Search Operations (Session 014)
```scheme
; Find
(define test-find-exists (find (lambda (x) (> x 5)) (list 1 3 7 2)))
(define test-find-not-found (find (lambda (x) (< x 0)) (list 1 2 3)))
(define test-find-mixed (find (lambda (x) (> x 5.0)) (list 1.0 2.0 10.0)))

; Member
(define test-member-int (member 2 (list 1 2 3)))
(define test-member-double (member 2.0 (list 1.0 2.0 3.0)))
(define test-member-mixed (member 2.0 (list 1 2.0 3)))
```

#### Suite 5: Association Lists (Session 016)
```scheme
; Assoc
(define test-alist (list (cons 1 "a") (cons 2 "b") (cons 3 "c")))
(define test-assoc-found (assoc 2 test-alist))
(define test-assoc-not-found (assoc 5 test-alist))
(define test-assoc-mixed-keys (assoc 2.0 (list (cons 1 "a") (cons 2.0 "b"))))
```

#### Suite 6: List Slicing (Session 019)
```scheme
; Take
(define test-take-normal (take (list 1 2 3 4 5) 3))
(define test-take-mixed (take (list 1 2.0 3 4.0) 2))
(define test-take-too-many (take (list 1 2) 5))

; Split-at
(define test-split-normal (split-at (list 1 2 3 4) 2))
(define test-split-mixed (split-at (list 1 2.0 3 4.0) 2))
(define test-split-zero (split-at (list 1 2 3) 0))

; Partition
(define test-partition-even (partition even? (list 1 2 3 4 5 6)))
(define test-partition-mixed (partition (lambda (x) (> x 5)) (list 1 2.0 10 3.5)))

; Remove
(define test-remove-single (remove 2 (list 1 2 3 2 4)))
(define test-remove-mixed (remove 2.0 (list 1 2.0 3 2.0 4)))
```

#### Suite 7: Integration Tests (Session 020)
```scheme
; Chained operations
(define test-chain-1 (map (lambda (x) (* x 2)) 
                          (filter (lambda (x) (> x 0)) 
                                 (list -1 2 -3 4))))

(define test-chain-2 (fold + 0 
                           (map (lambda (x) (* x x)) 
                                (list 1 2 3))))

; Mixed-type throughout
(define test-mixed-chain (filter (lambda (x) (> x 5.0))
                                 (map (lambda (x) (+ x 2.5))
                                      (list 1 2.0 3 4.0))))
```

---

## Validation Criteria

### Per-Function Validation
Each migrated function must pass:

1. **Type Preservation Test**: Verify correct type propagation through operations
2. **Mixed Type Test**: At least one test with int64 and double values
3. **Edge Case Test**: Empty list, single element, null handling
4. **Integration Test**: Works correctly when chained with other operations

### Overall System Validation
After all migrations:

1. **No Regressions**: All existing integer-only tests still pass
2. **Mixed-Type Support**: New test suites pass (Suites 1-7 above)
3. **Memory Safety**: No arena corruption or invalid pointer accesses
4. **Type Correctness**: Results maintain expected types (int64 vs double)

---

## Risk Assessment & Mitigation

### Critical Risks

**Risk 1: Arena Memory Corruption**
- **Cause**: Improper scope management in higher-order functions
- **Mitigation**: Follow [`codegenLast()`](../lib/backend/llvm_codegen.cpp:6607) pattern - no scope for results that must persist
- **Detection**: Segfaults, incorrect car/cdr values, test failures

**Risk 2: Type Promotion Logic Errors**
- **Cause**: Incorrect handling of int64 ↔ double conversions
- **Mitigation**: Comprehensive test suite with explicit type assertions
- **Detection**: Wrong results, type mismatches in PHI nodes

**Risk 3: Function Call Arity Mismatches**
- **Cause**: Mixed-type arguments changing expected parameter types
- **Mitigation**: Use `TypedValue` throughout, explicit type conversion at call sites
- **Detection**: LLVM verification errors, runtime crashes

### Medium Risks

**Risk 4: Performance Degradation**
- **Cause**: Type checking overhead in inner loops
- **Mitigation**: Profile before/after, optimize hot paths
- **Detection**: Benchmarking reveals slowdowns

**Risk 5: Complex Control Flow Bugs**
- **Cause**: Type branching creating incorrect PHI node patterns
- **Mitigation**: Careful block structure, test each branch independently
- **Detection**: Wrong results for specific type combinations

---

## Session-by-Session Breakdown

### Sessions 005-006: Map Foundation
- **005**: Migrate [`codegenMapSingleList()`](../lib/backend/llvm_codegen.cpp:5182)
  - Replace all 3 `CreateStructGEP` sites
  - Add type branching for car extraction
  - Update cons cell creation
  - Create basic test suite
  
- **006**: Migrate [`codegenMapMultiList()`](../lib/backend/llvm_codegen.cpp:5276)
  - Replace multi-list car extraction loop
  - Handle synchronized type checking across lists
  - Update cons cell creation
  - Extend test suite for multi-list cases

### Sessions 007-008: Map Testing & Validation
- **007**: Create comprehensive map test suite
  - Single-list: int-only, double-only, mixed
  - Multi-list: 2 lists, 3 lists, mixed types
  - Edge cases: empty, single element, unequal lengths
  
- **008**: Debug and validate map operations
  - Run test suite
  - Fix any type promotion issues
  - Verify memory safety (no corruption)
  - Document any discovered edge cases

### Sessions 009-010: Filter & Search
- **009**: Migrate [`codegenFilter()`](../lib/backend/llvm_codegen.cpp:5396)
  - Same pattern as map (car extraction, iteration)
  - Add predicate result handling
  
- **010**: Create filter test suite
  - Mixed-type filtering
  - Edge cases (all pass, none pass)

### Sessions 011-012: Fold Operations
- **011**: Migrate [`codegenFold()`](../lib/backend/llvm_codegen.cpp:5513)
  - Handle accumulator type promotion
  - Type-aware procedure calls
  
- **012**: Implement [`codegenFoldRight()`](../lib/backend/llvm_codegen.cpp:5807) from scratch
  - Choose implementation strategy (two-pass recommended)
  - Implement with tagged helpers from start
  - Add comprehensive type handling

### Sessions 013-014: Reduction Testing
- **013**: Migrate [`codegenFind()`](../lib/backend/llvm_codegen.cpp:6138)
  - Type-aware element return
  
- **014**: Create fold/find test suite
  - Type promotion in fold
  - Fold-right correctness
  - Find with mixed types

### Sessions 015-016: Association & Membership
- **015**: Migrate [`codegenMember()`](../lib/backend/llvm_codegen.cpp:5653)
  - Type-aware comparison
  - Handle memq, memv, member variants
  
- **016**: Migrate [`codegenAssoc()`](../lib/backend/llvm_codegen.cpp:5813)
  - Nested cons cell handling (alist of pairs)
  - Type-aware key comparison
  - Create test suite

### Sessions 017-018: List Manipulation
- **017**: Migrate [`codegenTake()`](../lib/backend/llvm_codegen.cpp:5979)
  - Prefix extraction with type preservation
  
- **018**: Migrate [`codegenSplitAt()`](../lib/backend/llvm_codegen.cpp:6392)
  - Two-way split returning pair
  - Type preservation in both parts

### Session 019: Partitioning & Removal
- **019**: Migrate remaining functions:
  - [`codegenPartition()`](../lib/backend/llvm_codegen.cpp:6242) - Dual output lists
  - [`codegenRemove()`](../lib/backend/llvm_codegen.cpp:6499) - Element removal
  - [`codegenForEachSingleList()`](../lib/backend/llvm_codegen.cpp:5759) - Side effects only
  - Create test suites

### Session 020: Final Validation
- **020**: Comprehensive system validation
  - Run ALL test suites (Suites 1-7)
  - Verify [`codegenLast()`](../lib/backend/llvm_codegen.cpp:6607) and [`codegenLastPair()`](../lib/backend/llvm_codegen.cpp:6705) still work
  - Integration testing with complex examples
  - Update [`docs/BUILD_STATUS.md`](BUILD_STATUS.md)
  - Document any remaining issues
  - Prepare for Phase 1, Month 1, Week 2

---

## Code Quality Guidelines

### Consistency Rules
1. **Always use tagged helpers** - Never use `CreateStructGEP` on cons cells
2. **Type checking pattern** - Always check type before extraction:
   ```cpp
   Value* type = builder->CreateCall(arena_tagged_cons_get_type_func, {ptr, is_car_or_cdr});
   Value* base_type = builder->CreateAnd(type, ConstantInt::get(Type::getInt8Ty(*context), 0x0F));
   ```
3. **PHI node discipline** - Always merge same types (int64↔int64 or double↔double)
4. **Error handling** - Check for null before dereferencing (already done in most functions)

### Naming Conventions
- Loop blocks: `{function}_loop_cond`, `{function}_loop_body`, `{function}_loop_exit`
- Type blocks: `{operation}_double`, `{operation}_int`, `{operation}_merge`
- Result vars: `{function}_result_head`, `{function}_result_tail`

### Documentation Standards
- Add comment before each migration: `// MIGRATION: Using tagged helper for car/cdr`
- Reference this document: `// See docs/HIGHER_ORDER_REWRITE_PLAN.md`
- Document type handling: `// Handle mixed int64/double types with promotion`

---

## Success Metrics

### Per-Session Metrics
- **Code Coverage**: All `CreateStructGEP` sites replaced
- **Test Pass Rate**: 100% for migrated function's test suite
- **No Regressions**: Existing tests continue to pass
- **Type Correctness**: Mixed-type tests verify correct type propagation

### Overall Project Metrics
- **13 Functions Migrated**: All non-stub, non-complete functions updated
- **1 Function Implemented**: `codegenFoldRight` complete
- **2 Functions Validated**: `codegenLast` and `codegenLastPair` verified still working
- **7 Test Suites Created**: Comprehensive coverage of all functions
- **0 CreateStructGEP Remaining**: Complete elimination from higher-order functions

---

## Rollback Plan

If critical issues arise during migration:

1. **Per-Function Rollback**: Git revert individual function changes
2. **Test-Driven Recovery**: Re-run test suite to identify exact failure
3. **Incremental Fix**: Address specific type handling issue
4. **Validation**: Re-test before proceeding

**Rollback Triggers:**
- Segmentation faults during testing
- Widespread test failures (>10% of suite)
- Arena memory corruption detected
- LLVM verification failures

---

## Dependencies & Prerequisites

### Completed (Sessions 001-002)
✅ Tagged cons cell C helpers implemented in [`lib/core/arena_memory.cpp`](../lib/core/arena_memory.cpp)  
✅ Function declarations in [`lib/backend/llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp:499-682)  
✅ Helper functions: `extractCarAsTaggedValue()`, `codegenTaggedArenaConsCell()`  
✅ Reference implementations: [`codegenLast()`](../lib/backend/llvm_codegen.cpp:6607), [`codegenLastPair()`](../lib/backend/llvm_codegen.cpp:6705)  

### Required for Success
- Arena memory system stable
- Tagged cons cell helpers tested
- `TypedValue` infrastructure working
- Test framework ready

---

## Appendix A: Quick Reference

### Function Status Matrix

| Function | Lines | Status | CreateStructGEP Sites | Priority | Session |
|----------|-------|--------|---------------------|----------|---------|
| [`codegenMapSingleList`](../lib/backend/llvm_codegen.cpp:5182) | 5182-5273 | 🔴 TODO | 3 sites | CRITICAL | 005 |
| [`codegenMapMultiList`](../lib/backend/llvm_codegen.cpp:5276) | 5276-5393 | 🔴 TODO | 3 sites | CRITICAL | 006 |
| [`codegenFilter`](../lib/backend/llvm_codegen.cpp:5396) | 5396-5510 | 🔴 TODO | 3 sites | HIGH | 009 |
| [`codegenFold`](../lib/backend/llvm_codegen.cpp:5513) | 5513-5585 | 🔴 TODO | 2 sites | HIGH | 011 |
| [`codegenFoldRight`](../lib/backend/llvm_codegen.cpp:5807) | 5807-5810 | 🚧 STUB | N/A | HIGH | 012 |
| [`codegenForEachSingleList`](../lib/backend/llvm_codegen.cpp:5759) | 5759-5804 | 🔴 TODO | 2 sites | MEDIUM | 019 |
| [`codegenMember`](../lib/backend/llvm_codegen.cpp:5653) | 5653-5723 | 🔴 TODO | 2 sites | MEDIUM | 015 |
| [`codegenAssoc`](../lib/backend/llvm_codegen.cpp:5813) | 5813-5923 | 🔴 TODO | 4 sites | MEDIUM | 016 |
| [`codegenTake`](../lib/backend/llvm_codegen.cpp:5979) | 5979-6079 | 🔴 TODO | 3 sites | MEDIUM | 017 |
| [`codegenFind`](../lib/backend/llvm_codegen.cpp:6138) | 6138-6240 | 🔴 TODO | 3 sites | HIGH | 013 |
| [`codegenPartition`](../lib/backend/llvm_codegen.cpp:6242) | 6242-6389 | 🔴 TODO | 4 sites | MEDIUM | 019 |
| [`codegenSplitAt`](../lib/backend/llvm_codegen.cpp:6392) | 6392-6496 | 🔴 TODO | 3 sites | MEDIUM | 018 |
| [`codegenRemove`](../lib/backend/llvm_codegen.cpp:6499) | 6499-6604 | 🔴 TODO | 3 sites | MEDIUM | 019 |
| [`codegenLast`](../lib/backend/llvm_codegen.cpp:6607) | 6607-6702 | ✅ DONE | 0 (tagged) | - | - |
| [`codegenLastPair`](../lib/backend/llvm_codegen.cpp:6705) | 6705-6768 | ✅ DONE | 0 (tagged) | - | - |

### Helper Function Index

| Helper | Line | Purpose |
|--------|------|---------|
| [`arena_tagged_cons_get_type_func`](../lib/backend/llvm_codegen.cpp:664) | 664-682 | Get type tag from car/cdr |
| [`arena_tagged_cons_get_int64_func`](../lib/backend/llvm_codegen.cpp:518) | 518-536 | Extract int64 value |
| [`arena_tagged_cons_get_double_func`](../lib/backend/llvm_codegen.cpp:538) | 538-556 | Extract double value |
| [`arena_tagged_cons_get_ptr_func`](../lib/backend/llvm_codegen.cpp:558) | 558-576 | Extract cons pointer |
| [`arena_tagged_cons_set_int64_func`](../lib/backend/llvm_codegen.cpp:578) | 578-598 | Set int64 value |
| [`arena_tagged_cons_set_double_func`](../lib/backend/llvm_codegen.cpp:600) | 600-620 | Set double value |
| [`arena_tagged_cons_set_ptr_func`](../lib/backend/llvm_codegen.cpp:622) | 622-642 | Set cons pointer |
| [`arena_tagged_cons_set_null_func`](../lib/backend/llvm_codegen.cpp:644) | 644-662 | Set null value |
| [`codegenTaggedArenaConsCell()`](../lib/backend/llvm_codegen.cpp:1012) | 1012-1092 | Create tagged cons cell |
| [`extractCarAsTaggedValue()`](../lib/backend/llvm_codegen.cpp:1168) | 1168-1205 | Extract car as tagged value |
| [`extractCdrAsTaggedValue()`](../lib/backend/llvm_codegen.cpp:1207) | 1207-1244 | Extract cdr as tagged value |
| [`detectValueType()`](../lib/backend/llvm_codegen.cpp:1246) | 1246-1269 | Detect LLVM value type |

---

## Appendix B: Common Patterns

### Pattern 1: Type-Aware Car Extraction
```cpp
Value* cons_ptr = builder->CreateIntToPtr(cons_int, builder->getPtrTy());
Value* is_car = ConstantInt::get(Type::getInt1Ty(*context), 0);
Value* car_type = builder->CreateCall(arena_tagged_cons_get_type_func, {cons_ptr, is_car});
Value* base_type = builder->CreateAnd(car_type, ConstantInt::get(Type::getInt8Ty(*context), 0x0F));
Value* is_double = builder->CreateICmpEQ(base_type, ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_DOUBLE));

// Branch and extract...
```

### Pattern 2: Cdr Iteration
```cpp
Value* is_cdr = ConstantInt::get(Type::getInt1Ty(*context), 1);
Value* next_cons = builder->CreateCall(arena_tagged_cons_get_ptr_func, {cons_ptr, is_cdr});
```

### Pattern 3: Tail Update
```cpp
Value* is_cdr = ConstantInt::get(Type::getInt1Ty(*context), 1);
builder->CreateCall(arena_tagged_cons_set_ptr_func,
    {tail_cons_ptr, is_cdr, new_cons_int, 
     ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_CONS_PTR)});
```

---

## Next Steps

**Immediate Actions (Session 005):**
1. Read this document thoroughly
2. Review [`codegenLast()`](../lib/backend/llvm_codegen.cpp:6607) as reference implementation
3. Begin migration of [`codegenMapSingleList()`](../lib/backend/llvm_codegen.cpp:5182)
4. Create initial test file for map operations

**Success Criteria for Session 005:**
- [`codegenMapSingleList()`](../lib/backend/llvm_codegen.cpp:5182) fully migrated
- All 3 `CreateStructGEP` sites replaced
- Basic single-list map tests passing
- No regressions in existing tests

---

**Document Version:** 1.0  
**Last Updated:** 2025-11-13  
**Next Review:** After Session 010 (Mid-Phase checkpoint)