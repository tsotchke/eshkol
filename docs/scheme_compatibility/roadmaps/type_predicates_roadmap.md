# Type Predicates Implementation Roadmap

Last Updated: 2025-03-29

## Overview

This document outlines the roadmap for implementing basic type predicates in Eshkol. Type predicates are essential for type checking and conditional execution based on data types.

## Implementation Status

| Function | Priority | Status | Target Date | Dependencies |
|----------|----------|--------|-------------|--------------|
| `boolean?` | P0 | Planned | Q2 2025 | Type system |
| `symbol?` | P0 | Planned | Q2 2025 | Type system |
| `number?` | P0 | Planned | Q2 2025 | Type system |
| `string?` | P0 | Planned | Q2 2025 | Type system |
| `char?` | P0 | Planned | Q2 2025 | Type system |
| `procedure?` | P0 | Planned | Q2 2025 | Type system, Closure system |
| `vector?` | P0 | Planned | Q2 2025 | Type system |

## Implementation Plan

### Phase 1: Design and Infrastructure (Week 1)

1. **Type System Integration**
   - Ensure type tags are consistently applied to all objects
   - Define type constants for all basic types
   - Create a unified type checking mechanism

2. **Error Handling**
   - Define error cases for type predicates
   - Implement error reporting for invalid inputs

3. **Testing Framework**
   - Set up test cases for all type predicates
   - Include edge cases and error cases

### Phase 2: Core Implementation (Week 2)

1. **Implement Basic Type Predicates**
   - `boolean?`: Check if object is a boolean value
   - `symbol?`: Check if object is a symbol
   - `number?`: Check if object is a number (integer or float)
   - `string?`: Check if object is a string
   - `char?`: Check if object is a character
   - `procedure?`: Check if object is a procedure
   - `vector?`: Check if object is a vector

2. **Documentation**
   - Document each type predicate
   - Update function status documentation
   - Update test coverage documentation

### Phase 3: Integration and Testing (Week 3)

1. **Integration Testing**
   - Test type predicates in combination with other language features
   - Test type predicates with edge cases
   - Test type predicates with error cases

2. **Performance Optimization**
   - Optimize type checking for common cases
   - Ensure type predicates are efficient

3. **Final Documentation**
   - Update all documentation to reflect implementation
   - Create examples demonstrating type predicates

## Implementation Details

### Type Tag System

The type predicates will rely on a consistent type tag system. Each object in Eshkol will have a type tag that identifies its type. The type tag system will be implemented as follows:

```c
// Type tags
typedef enum {
    ESHKOL_TYPE_BOOLEAN,
    ESHKOL_TYPE_SYMBOL,
    ESHKOL_TYPE_INTEGER,
    ESHKOL_TYPE_FLOAT,
    ESHKOL_TYPE_STRING,
    ESHKOL_TYPE_CHARACTER,
    ESHKOL_TYPE_PROCEDURE,
    ESHKOL_TYPE_VECTOR,
    ESHKOL_TYPE_PAIR,
    ESHKOL_TYPE_NULL,
    // Add more types as needed
} EshkolType;

// Type checking function
bool eshkol_is_type(EshkolObject* obj, EshkolType type);
```

### Type Predicate Implementation

Each type predicate will be implemented as a function that checks the type tag of the object:

```c
// Boolean predicate
bool eshkol_is_boolean(EshkolObject* obj) {
    return eshkol_is_type(obj, ESHKOL_TYPE_BOOLEAN);
}

// Symbol predicate
bool eshkol_is_symbol(EshkolObject* obj) {
    return eshkol_is_type(obj, ESHKOL_TYPE_SYMBOL);
}

// Number predicate
bool eshkol_is_number(EshkolObject* obj) {
    return eshkol_is_type(obj, ESHKOL_TYPE_INTEGER) || 
           eshkol_is_type(obj, ESHKOL_TYPE_FLOAT);
}

// String predicate
bool eshkol_is_string(EshkolObject* obj) {
    return eshkol_is_type(obj, ESHKOL_TYPE_STRING);
}

// Character predicate
bool eshkol_is_char(EshkolObject* obj) {
    return eshkol_is_type(obj, ESHKOL_TYPE_CHARACTER);
}

// Procedure predicate
bool eshkol_is_procedure(EshkolObject* obj) {
    return eshkol_is_type(obj, ESHKOL_TYPE_PROCEDURE);
}

// Vector predicate
bool eshkol_is_vector(EshkolObject* obj) {
    return eshkol_is_type(obj, ESHKOL_TYPE_VECTOR);
}
```

### Error Handling

Type predicates should handle NULL pointers and invalid objects gracefully:

```c
bool eshkol_is_type(EshkolObject* obj, EshkolType type) {
    if (obj == NULL) {
        return false;
    }
    
    // Check if the object has a valid type tag
    if (!eshkol_has_valid_type_tag(obj)) {
        eshkol_report_error("Invalid object: no type tag");
        return false;
    }
    
    return obj->type == type;
}
```

### Testing Strategy

Each type predicate will be tested with:

1. **Normal Cases**: Objects of the correct type
2. **Edge Cases**: Objects of similar types
3. **Error Cases**: NULL pointers, invalid objects

Example test cases:

```c
void test_boolean_predicate() {
    // Normal cases
    EshkolObject* true_obj = eshkol_create_boolean(true);
    EshkolObject* false_obj = eshkol_create_boolean(false);
    assert(eshkol_is_boolean(true_obj));
    assert(eshkol_is_boolean(false_obj));
    
    // Edge cases
    EshkolObject* number_obj = eshkol_create_integer(0);
    assert(!eshkol_is_boolean(number_obj));
    
    // Error cases
    assert(!eshkol_is_boolean(NULL));
    
    // Cleanup
    eshkol_free_object(true_obj);
    eshkol_free_object(false_obj);
    eshkol_free_object(number_obj);
}
```

## Dependencies

The implementation of type predicates depends on:

1. **Type System**: The type system must be in place to support type tags
2. **Memory Management**: Objects must be properly allocated and managed
3. **Error Handling**: Error reporting must be in place

## Next Steps After Implementation

After implementing the basic type predicates, the next steps are:

1. **Implement Equality Predicates**: `eq?`, `eqv?`, `equal?`
2. **Implement Additional List Processing Functions**: `length`, `append`, `reverse`, etc.
3. **Implement Additional Type Predicates**: `integer?`, `real?`, `complex?`, etc.

## Conclusion

The implementation of basic type predicates is a crucial step in the Scheme compatibility roadmap. It provides the foundation for type checking and conditional execution based on data types, which are essential for many Scheme programs.
