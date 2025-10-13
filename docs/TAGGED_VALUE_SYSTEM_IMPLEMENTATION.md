# Tagged Value System Implementation Plan

## Overview

This document outlines the implementation of a universal tagged value system where ALL values in Eshkol carry runtime type information. This eliminates type casting issues and provides a solid foundation for mixed-type operations and future scientific data types.

## Core Design

### Tagged Value Structure

```c
// Runtime representation for ALL Eshkol values (16 bytes aligned)
typedef struct eshkol_tagged_value {
    uint8_t type;        // ESHKOL_VALUE_INT64, ESHKOL_VALUE_DOUBLE, etc.
    uint8_t flags;       // ESHKOL_VALUE_EXACT_FLAG, ESHKOL_VALUE_INEXACT_FLAG
    uint16_t reserved;   // Reserved for future use
    union {
        int64_t int_val;
        double double_val;
        uint64_t ptr_val;
    } data;              // 8 bytes
} eshkol_tagged_value_t;  // Total: 12 bytes (pads to 16 for alignment)
```

### Key Principles

1. **Universal Representation**: EVERY value is a tagged_value
2. **No Implicit Casting**: Type conversions only happen during arithmetic promotion
3. **Type Preservation**: car/cdr/list operations maintain exact types
4. **Efficient Storage**: 16-byte aligned for cache efficiency

## Implementation Phases

### Phase 1: Core Infrastructure (COMPLETED)

- ✅ Added `eshkol_tagged_value_t` to [`inc/eshkol/eshkol.h`](inc/eshkol/eshkol.h:67-122)
- ✅ Added helper functions: `eshkol_make_int64()`, `eshkol_make_double()`, `eshkol_make_ptr()`
- ✅ Added unpacking functions: `eshkol_unpack_int64()`, `eshkol_unpack_double()`, `eshkol_unpack_ptr()`
- ✅ Created LLVM struct type in codegen constructor

### Phase 2: LLVM Helper Functions (NEXT)

Add helper functions in [`lib/backend/llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp):

```cpp
// Pack int64 into tagged value
Value* packInt64ToTaggedValue(Value* int64_val, bool is_exact = true);

// Pack double into tagged value  
Value* packDoubleToTaggedValue(Value* double_val);

// Pack pointer into tagged value
Value* packPtrToTaggedValue(Value* ptr_val, eshkol_value_type_t type);

// Unpack int64 from tagged value
Value* unpackInt64FromTaggedValue(Value* tagged_val_ptr);

// Unpack double from tagged value
Value* unpackDoubleFromTaggedValue(Value* tagged_val_ptr);

// Unpack pointer from tagged value
Value* unpackPtrFromTaggedValue(Value* tagged_val_ptr);

// Get type from tagged value
Value* getTaggedValueType(Value* tagged_val_ptr);
```

### Phase 3: Update Core Operations

#### 3.1 Literals
```cpp
// Old: return ConstantInt::get(Type::getInt64Ty(*context), ast->int64_val);
// New: return packInt64ToTaggedValue(ConstantInt::get(...), true);
```

#### 3.2 car() and cdr()
```cpp
// Old: Returns raw int64 (broken for doubles)
// New: Returns tagged_value struct with correct type
Value* codegenCar(const eshkol_operations_t* op) {
    // Get cons cell pointer
    // Call arena_tagged_cons_get_type() to get type
    // If DOUBLE: call arena_tagged_cons_get_double() and pack to tagged_value
    // If INT64: call arena_tagged_cons_get_int64() and pack to tagged_value
    // Return tagged_value
}
```

#### 3.3 cons() and list()
```cpp
// Already creates tagged cons cells - no changes needed!
// These already use codegenTaggedArenaConsCell() which preserves types
```

#### 3.4 Arithmetic Operations
```cpp
Value* codegenArithmetic(const eshkol_operations_t* op, const std::string& operation) {
    // Unpack first operand from tagged value
    // For each subsequent operand:
    //   - Unpack from tagged value
    //   - Promote to common type (int64→double if needed)
    //   - Perform operation
    // Pack result into tagged value with correct type
}
```

#### 3.5 display()
```cpp
Value* codegenDisplay(const eshkol_operations_t* op) {
    // Get tagged value
    // Extract type field
    // Switch based on type:
    //   - INT64: unpack and print with %lld
    //   - DOUBLE: unpack and print with %f
    //   - STRING: unpack ptr and print with %s
}
```

### Phase 4: Update All List Operations

Need to update to work with tagged values:
- append, reverse, map, filter, fold
- list-ref, list-tail, length
- member, assoc, find
- All compound car/cdr operations

### Phase 5: Update Function Calls

Function signatures need to change:
```cpp
// Old: int64_t func(int64_t x, int64_t y)
// New: eshkol_tagged_value_t func(eshkol_tagged_value_t x, eshkol_tagged_value_t y)
```

This affects:
- User-defined functions
- Lambda expressions  
- Builtin arithmetic functions

## Migration Strategy

### Incremental Approach

1. **Phase 2A**: Implement helper functions (today)
2. **Phase 2B**: Update car/cdr to return tagged values (today)
3. **Phase 2C**: Update display to handle tagged values (today)
4. **Phase 2D**: Test basic mixed-type lists (today)
5. **Phase 3**: Update arithmetic operations (tomorrow)
6. **Phase 4**: Update all list operations (1-2 days)
7. **Phase 5**: Update function signatures (1-2 days)

### Compatibility

During migration:
- Old integer-only code continues working (int64 is just a tagged value with INT64 type)
- New mixed-type code works correctly
- No breaking changes to existing tests

## Expected Results After Implementation

### Test Output Should Show:
```
2. Testing double cons cell:
   Created cons with (3.14159 . 2.71828)
   car: 3.141590        ← CORRECT (not 4614256650576692846)
   cdr: 2.718280        ← CORRECT (not 4613303441197561744)

4. Testing mixed type list:
   Created list: (1 2.5 3 4.75 5)
   First element: 1
   Second element: 2.500000    ← CORRECT (not 4612811918334230528)
   Third element: 3
```

### Benefits

1. **Type Safety**: No more bit-pattern confusion
2. **Precision**: Doubles stay doubles, ints stay ints
3. **Extensibility**: Easy to add complex numbers, rationals, etc.
4. **HoTT Ready**: Tagged values map directly to dependent types
5. **Performance**: 16-byte structs are cache-friendly

## Implementation Status

- ✅ Phase 1: Core infrastructure
- 🔄 Phase 2A: Helper functions (in progress)
- ⏳ Phase 2B-D: car/cdr/display updates
- ⏳ Phase 3-5: Full system integration

## Next Steps

1. Implement LLVM helper functions for pack/unpack
2. Update car() to return tagged_value
3. Update cdr() to return tagged_value  
4. Update display() to unpack and print based on type
5. Test with [`tests/mixed_type_lists_basic_test.esk`](tests/mixed_type_lists_basic_test.esk)

This is the correct, mathematically rigorous solution that will serve as the foundation for all future type system work.