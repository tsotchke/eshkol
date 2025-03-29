# Equality Predicates Implementation Roadmap

Last Updated: 2025-03-29

## Overview

This document outlines the roadmap for implementing equality predicates in Eshkol. Equality predicates are essential for comparing values and determining their equivalence, which is a fundamental operation in Scheme programming.

## Implementation Status

| Function | Priority | Status | Target Date | Dependencies |
|----------|----------|--------|-------------|--------------|
| `eq?` | P0 | Planned | Q2 2025 | Type system |
| `eqv?` | P0 | Planned | Q2 2025 | Type system, `eq?` |
| `equal?` | P0 | Planned | Q2 2025 | Type system, `eqv?` |

## Implementation Plan

### Phase 1: Design and Infrastructure (Week 1)

1. **Type System Integration**
   - Ensure type tags are consistently applied to all objects
   - Define equality semantics for each type
   - Create a unified equality checking mechanism

2. **Error Handling**
   - Define error cases for equality predicates
   - Implement error reporting for invalid inputs

3. **Testing Framework**
   - Set up test cases for all equality predicates
   - Include edge cases and error cases

### Phase 2: Core Implementation (Week 2)

1. **Implement Basic Equality Predicates**
   - `eq?`: Identity equality (pointer comparison)
   - `eqv?`: Equivalence relation (type-specific comparison)
   - `equal?`: Recursive equivalence (deep comparison)

2. **Documentation**
   - Document each equality predicate
   - Update function status documentation
   - Update test coverage documentation

### Phase 3: Integration and Testing (Week 3)

1. **Integration Testing**
   - Test equality predicates in combination with other language features
   - Test equality predicates with edge cases
   - Test equality predicates with error cases

2. **Performance Optimization**
   - Optimize equality checking for common cases
   - Ensure equality predicates are efficient

3. **Final Documentation**
   - Update all documentation to reflect implementation
   - Create examples demonstrating equality predicates

## Implementation Details

### eq? Implementation

The `eq?` predicate tests for object identity. Two objects are `eq?` if and only if they are the same object (i.e., they have the same memory address).

```c
// eq? predicate
bool eshkol_is_eq(EshkolObject* obj1, EshkolObject* obj2) {
    // NULL check
    if (obj1 == NULL || obj2 == NULL) {
        return obj1 == obj2;
    }
    
    // Pointer comparison
    return obj1 == obj2;
}
```

### eqv? Implementation

The `eqv?` predicate is a more refined version of `eq?`. It returns true if the objects are `eq?`, or if they are numbers or characters with the same value.

```c
// eqv? predicate
bool eshkol_is_eqv(EshkolObject* obj1, EshkolObject* obj2) {
    // If objects are eq?, they are eqv?
    if (eshkol_is_eq(obj1, obj2)) {
        return true;
    }
    
    // NULL check
    if (obj1 == NULL || obj2 == NULL) {
        return false;
    }
    
    // Check if both objects have the same type
    if (obj1->type != obj2->type) {
        return false;
    }
    
    // Type-specific comparison
    switch (obj1->type) {
        case ESHKOL_TYPE_INTEGER:
            return obj1->value.integer == obj2->value.integer;
        case ESHKOL_TYPE_FLOAT:
            return obj1->value.floating == obj2->value.floating;
        case ESHKOL_TYPE_CHARACTER:
            return obj1->value.character == obj2->value.character;
        case ESHKOL_TYPE_BOOLEAN:
            return obj1->value.boolean == obj2->value.boolean;
        default:
            // For other types, eqv? is the same as eq?
            return false;
    }
}
```

### equal? Implementation

The `equal?` predicate is a recursive version of `eqv?`. It returns true if the objects are `eqv?`, or if they are pairs or strings with the same structure and contents.

```c
// equal? predicate
bool eshkol_is_equal(EshkolObject* obj1, EshkolObject* obj2) {
    // If objects are eqv?, they are equal?
    if (eshkol_is_eqv(obj1, obj2)) {
        return true;
    }
    
    // NULL check
    if (obj1 == NULL || obj2 == NULL) {
        return false;
    }
    
    // Check if both objects have the same type
    if (obj1->type != obj2->type) {
        return false;
    }
    
    // Type-specific comparison
    switch (obj1->type) {
        case ESHKOL_TYPE_PAIR:
            // Recursively compare car and cdr
            return eshkol_is_equal(eshkol_car(obj1), eshkol_car(obj2)) &&
                   eshkol_is_equal(eshkol_cdr(obj1), eshkol_cdr(obj2));
        case ESHKOL_TYPE_STRING:
            // Compare string contents
            return strcmp(obj1->value.string, obj2->value.string) == 0;
        case ESHKOL_TYPE_VECTOR:
            // Compare vector contents
            if (obj1->value.vector.length != obj2->value.vector.length) {
                return false;
            }
            for (size_t i = 0; i < obj1->value.vector.length; i++) {
                if (!eshkol_is_equal(obj1->value.vector.elements[i], obj2->value.vector.elements[i])) {
                    return false;
                }
            }
            return true;
        default:
            // For other types, equal? is the same as eqv?
            return false;
    }
}
```

### Error Handling

Equality predicates should handle NULL pointers and invalid objects gracefully:

```c
bool eshkol_is_eq(EshkolObject* obj1, EshkolObject* obj2) {
    // NULL check
    if (obj1 == NULL || obj2 == NULL) {
        return obj1 == obj2;
    }
    
    // Check if the objects have valid type tags
    if (!eshkol_has_valid_type_tag(obj1) || !eshkol_has_valid_type_tag(obj2)) {
        eshkol_report_error("Invalid object: no type tag");
        return false;
    }
    
    // Pointer comparison
    return obj1 == obj2;
}
```

### Testing Strategy

Each equality predicate will be tested with:

1. **Normal Cases**: Objects of the same type and value
2. **Edge Cases**: Objects of similar types or values
3. **Error Cases**: NULL pointers, invalid objects

Example test cases:

```c
void test_eq_predicate() {
    // Normal cases
    EshkolObject* obj1 = eshkol_create_integer(42);
    EshkolObject* obj2 = obj1;
    assert(eshkol_is_eq(obj1, obj2));
    
    // Edge cases
    EshkolObject* obj3 = eshkol_create_integer(42);
    assert(!eshkol_is_eq(obj1, obj3));
    
    // Error cases
    assert(!eshkol_is_eq(obj1, NULL));
    assert(!eshkol_is_eq(NULL, obj2));
    assert(eshkol_is_eq(NULL, NULL));
    
    // Cleanup
    eshkol_free_object(obj1);
    eshkol_free_object(obj3);
}

void test_eqv_predicate() {
    // Normal cases
    EshkolObject* int1 = eshkol_create_integer(42);
    EshkolObject* int2 = eshkol_create_integer(42);
    assert(eshkol_is_eqv(int1, int2));
    
    // Edge cases
    EshkolObject* float1 = eshkol_create_float(42.0);
    assert(!eshkol_is_eqv(int1, float1));
    
    // Error cases
    assert(!eshkol_is_eqv(int1, NULL));
    assert(!eshkol_is_eqv(NULL, int2));
    assert(eshkol_is_eqv(NULL, NULL));
    
    // Cleanup
    eshkol_free_object(int1);
    eshkol_free_object(int2);
    eshkol_free_object(float1);
}

void test_equal_predicate() {
    // Normal cases
    EshkolObject* list1 = eshkol_create_list(3, 
                                           eshkol_create_integer(1),
                                           eshkol_create_integer(2),
                                           eshkol_create_integer(3));
    EshkolObject* list2 = eshkol_create_list(3, 
                                           eshkol_create_integer(1),
                                           eshkol_create_integer(2),
                                           eshkol_create_integer(3));
    assert(eshkol_is_equal(list1, list2));
    
    // Edge cases
    EshkolObject* list3 = eshkol_create_list(3, 
                                           eshkol_create_integer(1),
                                           eshkol_create_integer(2),
                                           eshkol_create_integer(4));
    assert(!eshkol_is_equal(list1, list3));
    
    // Error cases
    assert(!eshkol_is_equal(list1, NULL));
    assert(!eshkol_is_equal(NULL, list2));
    assert(eshkol_is_equal(NULL, NULL));
    
    // Cleanup
    eshkol_free_object(list1);
    eshkol_free_object(list2);
    eshkol_free_object(list3);
}
```

## Dependencies

The implementation of equality predicates depends on:

1. **Type System**: The type system must be in place to support type tags
2. **Memory Management**: Objects must be properly allocated and managed
3. **Error Handling**: Error reporting must be in place
4. **List Operations**: The `car` and `cdr` functions must be implemented for `equal?` to work with pairs

## Next Steps After Implementation

After implementing the equality predicates, the next steps are:

1. **Implement Additional List Processing Functions**: `length`, `append`, `reverse`, etc.
2. **Implement Additional Type Predicates**: `integer?`, `real?`, `complex?`, etc.
3. **Implement Numeric Operations**: `zero?`, `positive?`, `negative?`, etc.

## Conclusion

The implementation of equality predicates is a crucial step in the Scheme compatibility roadmap. It provides the foundation for comparing values and determining their equivalence, which is essential for many Scheme programs.
