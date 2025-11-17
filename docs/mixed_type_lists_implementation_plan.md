# Mixed Type Lists Implementation Plan

## Overview

This document outlines the comprehensive plan for implementing mixed type lists in Eshkol, supporting both integers (int64) and floating-point numbers (double) within the same list structure using a tagged union approach.

## Current System Analysis (COMPLETED)

### Existing Arena Memory System
- **Current cons cell**: `arena_cons_cell_t` with two int64 fields (car, cdr)
- **Memory management**: Arena-based allocation with automatic cleanup
- **List operations**: 275+ references in LLVM codegen with comprehensive functionality
- **Performance**: Excellent cache locality with 16-byte aligned structures

### List Operations Currently Supported
- **Basic operations**: cons, car, cdr, list, null?, pair?
- **Access operations**: list-ref, list-tail, length, compound car/cdr (cadr, caddr, etc.)
- **Construction operations**: append, reverse, make-list, list*
- **Mutable operations**: set-car!, set-cdr!
- **Higher-order functions**: map, filter, fold, for-each
- **Utility functions**: member, assoc, find, partition, take, drop, etc.

## Proposed Tagged Union Design (COMPLETED)

### New Tagged Cons Cell Structure
```c
typedef struct arena_tagged_cons_cell {
    uint8_t car_type;     // Type tag for car value
    uint8_t cdr_type;     // Type tag for cdr value  
    uint16_t padding;     // Alignment padding (reserved for future use)
    union {
        int64_t int_val;
        double double_val;
        uint64_t ptr_val;   // For cons cell pointers
    } car_data;
    union {
        int64_t int_val;
        double double_val;
        uint64_t ptr_val;
    } cdr_data;
} arena_tagged_cons_cell_t;
```

### Type Constants
```c
#define ESHKOL_TYPE_NULL     0  // Empty/null value
#define ESHKOL_TYPE_INT64    1  // 64-bit signed integer
#define ESHKOL_TYPE_DOUBLE   2  // Double-precision floating point
#define ESHKOL_TYPE_CONS_PTR 3  // Pointer to another cons cell
```

### Memory Layout Analysis
- **Size**: 24 bytes (vs current 16 bytes) - 50% increase
- **Alignment**: 8-byte aligned for optimal cache performance
- **Overhead**: 4 bytes of type information per cons cell
- **Benefits**: Complete type safety, no ambiguity, supports future extensions

## Implementation Phases

### Phase 1: Arena Memory System Updates (NEXT)

#### 1.1 Update Header File (lib/core/arena_memory.h)
- Add tagged cons cell structure definition
- Add type constants and helper macros
- Add new allocation functions for tagged cons cells
- Maintain backward compatibility with existing functions

#### 1.2 Update Implementation (lib/core/arena_memory.cpp)
- Implement `arena_allocate_tagged_cons_cell()` function
- Add helper functions for type checking and data access
- Update arena statistics to account for new structure sizes

#### 1.3 Type Safety Helper Functions
```c
// Type checking helpers
static inline bool is_int64_type(uint8_t type) { return type == ESHKOL_TYPE_INT64; }
static inline bool is_double_type(uint8_t type) { return type == ESHKOL_TYPE_DOUBLE; }
static inline bool is_cons_ptr_type(uint8_t type) { return type == ESHKOL_TYPE_CONS_PTR; }

// Data access helpers
static inline int64_t get_int64_value(const tagged_cons_union_t* data, uint8_t type);
static inline double get_double_value(const tagged_cons_union_t* data, uint8_t type);
```

### Phase 2: Basic List Operations Updates

#### 2.1 Core Operations (cons, car, cdr)
- Update `codegenCons()` to create tagged cons cells
- Update `codegenCar()` and `codegenCdr()` to handle type information
- Generate LLVM IR for type checking and data extraction

#### 2.2 List Construction (list, make-list)
- Update `codegenList()` to preserve type information through construction
- Handle mixed type arguments in list creation
- Ensure proper type tagging for each element

#### 2.3 Type Predicates
- Enhance `codegenPairCheck()` and `codegenNullCheck()`
- Add type-specific predicates for integer? and real? checks

### Phase 3: List Access and Manipulation

#### 3.1 Access Operations
- Update `codegenListRef()`, `codegenListTail()`, `codegenLength()`
- Update compound car/cdr operations (`codegenCompoundCarCdr()`)
- Ensure type information is preserved through access operations

#### 3.2 Construction Operations  
- Update `codegenAppend()` and `codegenReverse()`
- Update `codegenListStar()`, `codegenAcons()`
- Handle mixed type lists in construction operations

#### 3.3 Mutable Operations
- Update `codegenSetCar()` and `codegenSetCdr()`
- Support type changes through mutation
- Ensure type safety in mutable operations

### Phase 4: Higher-Order Functions and Arithmetic

#### 4.1 Higher-Order Functions
- Update `codegenMap()`, `codegenFilter()`, `codegenFold()`
- Handle type promotion in function applications
- Support mixed type inputs and outputs

#### 4.2 Arithmetic Operations with Type Promotion
- Update arithmetic codegen functions (`+`, `-`, `*`, `/`)
- Implement Scheme numeric tower promotion rules:
  - `int64 + int64` → `int64`
  - `int64 + double` → `double`
  - `double + int64` → `double`  
  - `double + double` → `double`

#### 4.3 Comparison Operations
- Update comparison operations (`=`, `<`, `>`, `<=`, `>=`)
- Handle mixed type comparisons with appropriate conversion

### Phase 5: Advanced List Utilities

#### 5.1 Search and Membership
- Update `codegenMember()`, `codegenAssoc()`, `codegenFind()`
- Handle type-aware equality checking
- Support different comparison predicates (eq?, eqv?, equal?)

#### 5.2 List Processing
- Update `codegenTake()`, `codegenDrop()`, `codegenPartition()`
- Update `codegenSplitAt()`, `codegenRemove()`
- Ensure type preservation through list processing operations

### Phase 6: Parser and Language Integration

#### 6.1 Parser Updates
- Preserve type information during parsing
- Handle numeric literals with correct types
- Support mixed type list literals

#### 6.2 Type System Integration
- Update AST nodes to carry type information
- Integrate with existing type inference system
- Maintain compatibility with gradual typing

### Phase 7: Testing and Validation

#### 7.1 Comprehensive Test Suite
- Create mixed type specific test cases
- Test all operation combinations (int64/double × int64/double)
- Performance benchmarks comparing old vs new implementation

#### 7.2 Backward Compatibility Testing
- Verify all existing functionality continues working
- Test migration path from current to new system
- Validate memory usage and performance characteristics

## Type Promotion Strategy

### Automatic Promotion Rules (Scheme R7RS Compatible)
1. **Integer operations**: Result is integer if all operands are integers
2. **Mixed operations**: Any floating-point operand promotes result to floating-point
3. **Division special case**: Division always produces floating-point result (following Scheme semantics)
4. **Comparison operations**: Automatic conversion for comparison, but preserve original types

### LLVM Code Generation Strategy
```llvm
; Example: Mixed type addition (int64 + double)
%car_type = load i8, %cons_car_type_ptr
%cdr_type = load i8, %cons_cdr_type_ptr

; Type dispatch for arithmetic
switch i8 %car_type, label %error [
    i8 1, label %int64_car     ; ESHKOL_TYPE_INT64
    i8 2, label %double_car    ; ESHKOL_TYPE_DOUBLE
]

int64_car:
    switch i8 %cdr_type, label %error [
        i8 1, label %int64_add     ; int64 + int64 → int64
        i8 2, label %mixed_add     ; int64 + double → double
    ]

double_car:
    ; double + anything → double (with conversion)
    br label %double_add
```

## Migration Strategy

### Compatibility Approach
1. **Dual implementation**: Keep both old and new cons cell structures
2. **Feature flags**: Enable tagged cons cells via compilation flags
3. **Gradual migration**: Convert operations one by one with extensive testing
4. **Legacy support**: Maintain old API for existing code

### Performance Considerations
- **Memory overhead**: ~50% increase in cons cell size
- **CPU overhead**: 1-2 additional instructions per type check
- **Cache impact**: Larger structures may reduce cache efficiency
- **Optimization**: LLVM will optimize redundant type checks

## Success Criteria

### Functionality
- [ ] All existing list operations work with mixed types
- [ ] Proper type promotion follows Scheme semantics
- [ ] No regression in existing integer-only list performance
- [ ] Comprehensive error handling for type mismatches

### Performance
- [ ] Mixed type operations perform within 20% of current integer-only operations
- [ ] Memory overhead stays within 60% increase (target: 50%)
- [ ] No significant impact on compilation speed

### Quality
- [ ] 100% test coverage for mixed type scenarios
- [ ] Full backward compatibility with existing code
- [ ] Clear documentation and migration guide
- [ ] Production-ready error messages and debugging support

## Implementation Timeline

1. **Phase 1 (Arena Updates)**: 2-3 days
2. **Phase 2 (Basic Operations)**: 3-4 days  
3. **Phase 3 (Access/Manipulation)**: 4-5 days
4. **Phase 4 (Higher-Order/Arithmetic)**: 5-6 days
5. **Phase 5 (Advanced Utilities)**: 3-4 days
6. **Phase 6 (Parser Integration)**: 2-3 days
7. **Phase 7 (Testing/Validation)**: 4-5 days

**Total estimated timeline**: 3-4 weeks for complete implementation and testing.

## Next Steps

The immediate next step is updating the arena memory system to support tagged cons cells while maintaining full backward compatibility with the existing system.