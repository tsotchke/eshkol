# Phase 3B: Tagged Cons Cell Storage Architecture Update

## Executive Summary

**Update `arena_tagged_cons_cell_t` to store full `eshkol_tagged_value_t` structs**

This eliminates the lossy conversion between runtime tagged_value and storage,
enabling perfect value preservation for scientific computing.

## Current Architecture (Phase 3 - Has Issues)

```c
// Current: 24 bytes per cons cell
typedef struct {
    uint8_t car_type;     // 1 byte
    uint8_t cdr_type;     // 1 byte
    uint8_t car_flags;    // 1 byte
    uint8_t cdr_flags;    // 1 byte
    union {
        int64_t i;
        double d;
        uint64_t ptr;
    } car_data;           // 8 bytes
    union {
        int64_t i;
        double d;
        uint64_t ptr;
    } cdr_data;           // 8 bytes
    // Total: 4 + 8 + 8 = 20 bytes, aligned to 24
} arena_tagged_cons_cell_t;
```

**Problem:** Type and flags are stored separately from data.
**Result:** Complex conversion logic, potential value loss.

## Phase 3B Architecture (Proposed - Robust)

```c
// Phase 3B: 32 bytes per cons cell
typedef struct {
    eshkol_tagged_value_t car;  // 12 bytes (type + flags + reserved + data)
    eshkol_tagged_value_t cdr;  // 12 bytes (type + flags + reserved + data)
    uint64_t padding;            // 8 bytes (align to 32)
    // Total: 12 + 12 + 8 = 32 bytes
} arena_tagged_cons_cell_t;
```

**Benefits:**
- ✅ Direct storage of tagged_value (zero conversion)
- ✅ Perfect value preservation
- ✅ Simpler get/set functions
- ✅ Foundation for future optimizations

## Implementation Plan

### Step 1: Update Structure Definitions

#### File: `inc/eshkol/eshkol.h`
```c
// Update struct definition (around line 50-70)
typedef struct {
    eshkol_tagged_value_t car;
    eshkol_tagged_value_t cdr;
    uint64_t padding;  // Ensure 32-byte alignment
} arena_tagged_cons_cell_t;
```

#### File: `lib/core/arena_memory.h`
```c
// Update struct definition (should match eshkol.h)
typedef struct {
    eshkol_tagged_value_t car;
    eshkol_tagged_value_t cdr;
    uint64_t padding;
} arena_tagged_cons_cell_t;
```

### Step 2: Simplify C Helper Functions

#### File: `lib/core/arena_memory.cpp`

**BEFORE (Complex):**
```c
int64_t arena_tagged_cons_get_int64(const arena_tagged_cons_cell_t* cell, bool is_cdr) {
    if (is_cdr) {
        return cell->cdr_data.i;
    }
    return cell->car_data.i;
}
```

**AFTER (Simple - Direct Access):**
```c
// Get full tagged_value directly
eshkol_tagged_value_t arena_tagged_cons_get_car(const arena_tagged_cons_cell_t* cell) {
    return cell->car;
}

eshkol_tagged_value_t arena_tagged_cons_get_cdr(const arena_tagged_cons_cell_t* cell) {
    return cell->cdr;
}

// Set full tagged_value directly
void arena_tagged_cons_set_car(arena_tagged_cons_cell_t* cell, 
                                eshkol_tagged_value_t value) {
    cell->car = value;
}

void arena_tagged_cons_set_cdr(arena_tagged_cons_cell_t* cell,
                                eshkol_tagged_value_t value) {
    cell->cdr = value;
}
```

**Impact:**
- Functions reduced from 7 helpers to 4 helpers
- Zero type checking needed (handled by tagged_value itself)
- Direct struct copy (compiler optimizes)

### Step 3: Update LLVM Function Declarations

#### File: `lib/backend/llvm_codegen.cpp`

**Replace current 7 function declarations with 4:**

```cpp
// BEFORE (7 functions):
arena_tagged_cons_get_int64_func
arena_tagged_cons_get_double_func
arena_tagged_cons_get_ptr_func
arena_tagged_cons_set_int64_func
arena_tagged_cons_set_double_func
arena_tagged_cons_set_ptr_func
arena_tagged_cons_set_null_func

// AFTER (4 functions):
arena_tagged_cons_get_car_func:  eshkol_tagged_value_t (cell*)
arena_tagged_cons_get_cdr_func:  eshkol_tagged_value_t (cell*)
arena_tagged_cons_set_car_func:  void (cell*, eshkol_tagged_value_t)
arena_tagged_cons_set_cdr_func:  void (cell*, eshkol_tagged_value_t)
```

### Step 4: Simplify Helper Functions

**extractCarAsTaggedValue becomes:**
```cpp
Value* extractCarAsTaggedValue(Value* cons_ptr_int) {
    Value* cons_ptr = builder->CreateIntToPtr(cons_ptr_int, builder->getPtrTy());
    
    // Direct call - returns full tagged_value struct!
    return builder->CreateCall(arena_tagged_cons_get_car_func, {cons_ptr});
}

Value* extractCdrAsTaggedValue(Value* cons_ptr_int) {
    Value* cons_ptr = builder->CreateIntToPtr(cons_ptr_int, builder->getPtrTy());
    
    // Direct call - returns full tagged_value struct!
    return builder->CreateCall(arena_tagged_cons_get_cdr_func, {cons_ptr});
}
```

**No branching, no type checking, no conversion - PERFECT!**

### Step 5: Simplify Cons Cell Creation

**codegenTaggedArenaConsCellFromTaggedValue becomes:**
```cpp
Value* codegenTaggedArenaConsCellFromTaggedValue(Value* car_tagged, Value* cdr_tagged) {
    Value* arena_ptr = getArenaPtr();
    Value* cons_ptr = builder->CreateCall(arena_allocate_tagged_cons_cell_func, {arena_ptr});
    
    // Store car and cdr directly - no conversion!
    builder->CreateCall(arena_tagged_cons_set_car_func, {cons_ptr, car_tagged});
    builder->CreateCall(arena_tagged_cons_set_cdr_func, {cons_ptr, cdr_tagged});
    
    return builder->CreatePtrToInt(cons_ptr, Type::getInt64Ty(*context));
}
```

**From 70+ lines to 7 lines!**

## Testing Strategy

### Test 1: Basic Map (Integer)
```scheme
(map (lambda (x) (+ x 1)) (list 1 2 3))
;; Expected: (2 3 4)
;; Verifies: Int64 value preservation
```

### Test 2: Map with Doubles
```scheme
(map (lambda (x) (* x 2.0)) (list 1.5 2.5 3.5))
;; Expected: (3.0 5.0 7.0)
;; Verifies: Double value preservation
```

### Test 3: Multi-List Map (Mixed Types)
```scheme
(map + (list 1 2.0 3) (list 4 5.0 6))
;; Expected: (5.0 7.0 9.0)
;; Verifies: Type promotion in multi-list operations
```

### Test 4: Complex Chain
```scheme
(fold + 0 (map (lambda (x) (* x 2)) (list 1 2.0 3)))
;; Expected: 12.0
;; Verifies: Value preservation through pipeline
```

## Implementation Checklist

- [ ] Update `eshkol_tagged_value_t` in `inc/eshkol/eshkol.h`
- [ ] Update `arena_tagged_cons_cell_t` in `inc/eshkol/eshkol.h`
- [ ] Update `arena_tagged_cons_cell_t` in `lib/core/arena_memory.h`
- [ ] Implement new get/set functions in `lib/core/arena_memory.cpp`
- [ ] Update LLVM function declarations in `lib/backend/llvm_codegen.cpp`
- [ ] Simplify `extractCarAsTaggedValue` helper
- [ ] Simplify `extractCdrAsTaggedValue` helper
- [ ] Simplify `codegenTaggedArenaConsCellFromTaggedValue`
- [ ] Remove old get_int64/get_double/get_ptr functions
- [ ] Remove old set_int64/set_double/set_ptr functions
- [ ] Build and test
- [ ] Run full test suite
- [ ] Verify Valgrind clean
- [ ] Commit with comprehensive message

## Expected Results

### Code Simplification
- Helper functions: 70+ lines → ~20 lines (-71%)
- Type conversion logic: ELIMINATED
- Branching complexity: ELIMINATED

### Performance
- Direct struct copy: Compiler-optimized
- Zero conversion overhead
- Better cache locality (contiguous storage)

### Correctness
- Perfect value preservation
- Zero truncation/rounding errors
- Mixed-type operations: FULLY OPERATIONAL

### Scientific Computing
- Float64 precision: PRESERVED
- Type promotion: CORRECT
- Numerical stability: GUARANTEED

## Risk Assessment

**Risk: LOW**
- Structure update is mechanical
- Helper functions are straightforward
- LLVM codegen is already prepared (Phase 3 work)
- Clear rollback path if needed

**Estimated Time: 2-3 hours**
- Structure updates: 30 min
- C helper functions: 45 min
- LLVM updates: 45 min
- Testing/validation: 45 min

## Success Criteria

- ✅ `(map (lambda (x) (+ x 1)) (list 1 2 3))` returns `(2 3 4)`
- ✅ `(map + (list 1 2.0 3) (list 4 5.0 6))` returns `(5.0 7.0 9.0)`
- ✅ Mixed-type fold operations produce correct results
- ✅ Valgrind reports zero leaks
- ✅ All existing tests continue to pass

## Integration with v1.0-architecture

This completes:
- ✅ Phase 1: Core Language Features (with mixed types!)
- ✅ Foundation for Phase 2: Numeric Computing
- ✅ Preparation for HoTT dependent types

**Phase 3B delivers the final piece needed for v1.0-architecture Phase 1 completion.**

## Conclusion

Phase 3 established the **safety infrastructure**.
Phase 3B will deliver **perfect correctness**.

Together, they complete the transformation to a scientifically robust programming language.