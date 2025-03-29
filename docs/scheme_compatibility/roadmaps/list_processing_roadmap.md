# List Processing Functions Implementation Roadmap

Last Updated: 2025-03-29

## Overview

This document outlines the roadmap for implementing additional list processing functions in Eshkol. These functions build upon the core list operations (cons, car, cdr) to provide more advanced list manipulation capabilities.

## Implementation Status

| Function | Priority | Status | Target Date | Dependencies |
|----------|----------|--------|-------------|--------------|
| `length` | P1 | Planned | Q2 2025 | Core list operations |
| `append` | P1 | Planned | Q2 2025 | Core list operations |
| `reverse` | P1 | Planned | Q2 2025 | Core list operations |
| `list-ref` | P1 | Planned | Q2 2025 | Core list operations |
| `list-tail` | P1 | Planned | Q2 2025 | Core list operations |
| `list-set!` | P1 | Planned | Q2 2025 | Core list operations |
| `memq`, `memv`, `member` | P2 | Planned | Q3 2025 | Core list operations, Equality predicates |
| `assq`, `assv`, `assoc` | P2 | Planned | Q3 2025 | Core list operations, Equality predicates |

## Implementation Plan

### Phase 1: Basic List Processing Functions (Week 1)

1. **Implement Length and Access Functions**
   - `length`: Count the number of elements in a list
   - `list-ref`: Access an element by index
   - `list-tail`: Get a sublist starting at a specified index

2. **Documentation**
   - Document each function
   - Update function status documentation
   - Update test coverage documentation

### Phase 2: List Transformation Functions (Week 2)

1. **Implement List Transformation Functions**
   - `append`: Concatenate lists
   - `reverse`: Reverse a list
   - `list-set!`: Set an element at a specified index

2. **Documentation**
   - Document each function
   - Update function status documentation
   - Update test coverage documentation

### Phase 3: List Search Functions (Week 3)

1. **Implement List Search Functions**
   - `memq`, `memv`, `member`: Find an element in a list
   - `assq`, `assv`, `assoc`: Find an association in an association list

2. **Documentation**
   - Document each function
   - Update function status documentation
   - Update test coverage documentation

## Implementation Details

### length Implementation

The `length` function counts the number of elements in a list.

```c
// length function
size_t eshkol_length(EshkolObject* list) {
    // NULL check
    if (list == NULL) {
        return 0;
    }
    
    // Empty list check
    if (eshkol_is_null(list)) {
        return 0;
    }
    
    // Type check
    if (!eshkol_is_pair(list)) {
        eshkol_report_error("length: argument must be a list");
        return 0;
    }
    
    // Count elements
    size_t count = 0;
    while (eshkol_is_pair(list)) {
        count++;
        list = eshkol_cdr(list);
    }
    
    // Check if the list is proper
    if (!eshkol_is_null(list)) {
        eshkol_report_error("length: argument must be a proper list");
        return 0;
    }
    
    return count;
}
```

### append Implementation

The `append` function concatenates lists.

```c
// append function
EshkolObject* eshkol_append(EshkolObject* list1, EshkolObject* list2) {
    // NULL check
    if (list1 == NULL) {
        return list2;
    }
    
    // Empty list check
    if (eshkol_is_null(list1)) {
        return list2;
    }
    
    // Type check
    if (!eshkol_is_pair(list1)) {
        eshkol_report_error("append: first argument must be a list");
        return NULL;
    }
    
    // Create a new list
    EshkolObject* result = eshkol_cons(eshkol_car(list1), NULL);
    EshkolObject* current = result;
    list1 = eshkol_cdr(list1);
    
    // Copy elements from list1
    while (eshkol_is_pair(list1)) {
        eshkol_set_cdr(current, eshkol_cons(eshkol_car(list1), NULL));
        current = eshkol_cdr(current);
        list1 = eshkol_cdr(list1);
    }
    
    // Check if list1 is proper
    if (!eshkol_is_null(list1)) {
        eshkol_report_error("append: first argument must be a proper list");
        return NULL;
    }
    
    // Append list2
    eshkol_set_cdr(current, list2);
    
    return result;
}
```

### reverse Implementation

The `reverse` function reverses a list.

```c
// reverse function
EshkolObject* eshkol_reverse(EshkolObject* list) {
    // NULL check
    if (list == NULL) {
        return NULL;
    }
    
    // Empty list check
    if (eshkol_is_null(list)) {
        return list;
    }
    
    // Type check
    if (!eshkol_is_pair(list)) {
        eshkol_report_error("reverse: argument must be a list");
        return NULL;
    }
    
    // Reverse the list
    EshkolObject* result = eshkol_create_empty_list();
    while (eshkol_is_pair(list)) {
        result = eshkol_cons(eshkol_car(list), result);
        list = eshkol_cdr(list);
    }
    
    // Check if the list is proper
    if (!eshkol_is_null(list)) {
        eshkol_report_error("reverse: argument must be a proper list");
        return NULL;
    }
    
    return result;
}
```

### list-ref Implementation

The `list-ref` function accesses an element by index.

```c
// list-ref function
EshkolObject* eshkol_list_ref(EshkolObject* list, size_t index) {
    // NULL check
    if (list == NULL) {
        eshkol_report_error("list-ref: list is NULL");
        return NULL;
    }
    
    // Empty list check
    if (eshkol_is_null(list)) {
        eshkol_report_error("list-ref: index out of bounds");
        return NULL;
    }
    
    // Type check
    if (!eshkol_is_pair(list)) {
        eshkol_report_error("list-ref: argument must be a list");
        return NULL;
    }
    
    // Access element by index
    size_t current_index = 0;
    while (current_index < index && eshkol_is_pair(list)) {
        list = eshkol_cdr(list);
        current_index++;
    }
    
    // Check if the index is valid
    if (!eshkol_is_pair(list)) {
        eshkol_report_error("list-ref: index out of bounds");
        return NULL;
    }
    
    return eshkol_car(list);
}
```

### list-tail Implementation

The `list-tail` function returns a sublist starting at a specified index.

```c
// list-tail function
EshkolObject* eshkol_list_tail(EshkolObject* list, size_t index) {
    // NULL check
    if (list == NULL) {
        eshkol_report_error("list-tail: list is NULL");
        return NULL;
    }
    
    // Empty list check
    if (eshkol_is_null(list) && index > 0) {
        eshkol_report_error("list-tail: index out of bounds");
        return NULL;
    }
    
    // Type check
    if (!eshkol_is_pair(list) && !eshkol_is_null(list)) {
        eshkol_report_error("list-tail: argument must be a list");
        return NULL;
    }
    
    // Access sublist by index
    size_t current_index = 0;
    while (current_index < index && eshkol_is_pair(list)) {
        list = eshkol_cdr(list);
        current_index++;
    }
    
    // Check if the index is valid
    if (current_index < index) {
        eshkol_report_error("list-tail: index out of bounds");
        return NULL;
    }
    
    return list;
}
```

### list-set! Implementation

The `list-set!` function sets an element at a specified index.

```c
// list-set! function
void eshkol_list_set(EshkolObject* list, size_t index, EshkolObject* value) {
    // NULL check
    if (list == NULL) {
        eshkol_report_error("list-set!: list is NULL");
        return;
    }
    
    // Empty list check
    if (eshkol_is_null(list)) {
        eshkol_report_error("list-set!: index out of bounds");
        return;
    }
    
    // Type check
    if (!eshkol_is_pair(list)) {
        eshkol_report_error("list-set!: argument must be a list");
        return;
    }
    
    // Access element by index
    size_t current_index = 0;
    while (current_index < index && eshkol_is_pair(list)) {
        list = eshkol_cdr(list);
        current_index++;
    }
    
    // Check if the index is valid
    if (!eshkol_is_pair(list)) {
        eshkol_report_error("list-set!: index out of bounds");
        return;
    }
    
    // Set the element
    eshkol_set_car(list, value);
}
```

### memq, memv, member Implementation

The `memq`, `memv`, and `member` functions find an element in a list.

```c
// memq function
EshkolObject* eshkol_memq(EshkolObject* obj, EshkolObject* list) {
    // NULL check
    if (list == NULL) {
        return eshkol_create_boolean(false);
    }
    
    // Empty list check
    if (eshkol_is_null(list)) {
        return eshkol_create_boolean(false);
    }
    
    // Type check
    if (!eshkol_is_pair(list)) {
        eshkol_report_error("memq: second argument must be a list");
        return NULL;
    }
    
    // Search for the element
    while (eshkol_is_pair(list)) {
        if (eshkol_is_eq(obj, eshkol_car(list))) {
            return list;
        }
        list = eshkol_cdr(list);
    }
    
    // Check if the list is proper
    if (!eshkol_is_null(list)) {
        eshkol_report_error("memq: second argument must be a proper list");
        return NULL;
    }
    
    return eshkol_create_boolean(false);
}

// memv function
EshkolObject* eshkol_memv(EshkolObject* obj, EshkolObject* list) {
    // NULL check
    if (list == NULL) {
        return eshkol_create_boolean(false);
    }
    
    // Empty list check
    if (eshkol_is_null(list)) {
        return eshkol_create_boolean(false);
    }
    
    // Type check
    if (!eshkol_is_pair(list)) {
        eshkol_report_error("memv: second argument must be a list");
        return NULL;
    }
    
    // Search for the element
    while (eshkol_is_pair(list)) {
        if (eshkol_is_eqv(obj, eshkol_car(list))) {
            return list;
        }
        list = eshkol_cdr(list);
    }
    
    // Check if the list is proper
    if (!eshkol_is_null(list)) {
        eshkol_report_error("memv: second argument must be a proper list");
        return NULL;
    }
    
    return eshkol_create_boolean(false);
}

// member function
EshkolObject* eshkol_member(EshkolObject* obj, EshkolObject* list) {
    // NULL check
    if (list == NULL) {
        return eshkol_create_boolean(false);
    }
    
    // Empty list check
    if (eshkol_is_null(list)) {
        return eshkol_create_boolean(false);
    }
    
    // Type check
    if (!eshkol_is_pair(list)) {
        eshkol_report_error("member: second argument must be a list");
        return NULL;
    }
    
    // Search for the element
    while (eshkol_is_pair(list)) {
        if (eshkol_is_equal(obj, eshkol_car(list))) {
            return list;
        }
        list = eshkol_cdr(list);
    }
    
    // Check if the list is proper
    if (!eshkol_is_null(list)) {
        eshkol_report_error("member: second argument must be a proper list");
        return NULL;
    }
    
    return eshkol_create_boolean(false);
}
```

### assq, assv, assoc Implementation

The `assq`, `assv`, and `assoc` functions find an association in an association list.

```c
// assq function
EshkolObject* eshkol_assq(EshkolObject* obj, EshkolObject* alist) {
    // NULL check
    if (alist == NULL) {
        return eshkol_create_boolean(false);
    }
    
    // Empty list check
    if (eshkol_is_null(alist)) {
        return eshkol_create_boolean(false);
    }
    
    // Type check
    if (!eshkol_is_pair(alist)) {
        eshkol_report_error("assq: second argument must be an association list");
        return NULL;
    }
    
    // Search for the association
    while (eshkol_is_pair(alist)) {
        EshkolObject* pair = eshkol_car(alist);
        if (eshkol_is_pair(pair) && eshkol_is_eq(obj, eshkol_car(pair))) {
            return pair;
        }
        alist = eshkol_cdr(alist);
    }
    
    // Check if the list is proper
    if (!eshkol_is_null(alist)) {
        eshkol_report_error("assq: second argument must be a proper list");
        return NULL;
    }
    
    return eshkol_create_boolean(false);
}

// assv function
EshkolObject* eshkol_assv(EshkolObject* obj, EshkolObject* alist) {
    // NULL check
    if (alist == NULL) {
        return eshkol_create_boolean(false);
    }
    
    // Empty list check
    if (eshkol_is_null(alist)) {
        return eshkol_create_boolean(false);
    }
    
    // Type check
    if (!eshkol_is_pair(alist)) {
        eshkol_report_error("assv: second argument must be an association list");
        return NULL;
    }
    
    // Search for the association
    while (eshkol_is_pair(alist)) {
        EshkolObject* pair = eshkol_car(alist);
        if (eshkol_is_pair(pair) && eshkol_is_eqv(obj, eshkol_car(pair))) {
            return pair;
        }
        alist = eshkol_cdr(alist);
    }
    
    // Check if the list is proper
    if (!eshkol_is_null(alist)) {
        eshkol_report_error("assv: second argument must be a proper list");
        return NULL;
    }
    
    return eshkol_create_boolean(false);
}

// assoc function
EshkolObject* eshkol_assoc(EshkolObject* obj, EshkolObject* alist) {
    // NULL check
    if (alist == NULL) {
        return eshkol_create_boolean(false);
    }
    
    // Empty list check
    if (eshkol_is_null(alist)) {
        return eshkol_create_boolean(false);
    }
    
    // Type check
    if (!eshkol_is_pair(alist)) {
        eshkol_report_error("assoc: second argument must be an association list");
        return NULL;
    }
    
    // Search for the association
    while (eshkol_is_pair(alist)) {
        EshkolObject* pair = eshkol_car(alist);
        if (eshkol_is_pair(pair) && eshkol_is_equal(obj, eshkol_car(pair))) {
            return pair;
        }
        alist = eshkol_cdr(alist);
    }
    
    // Check if the list is proper
    if (!eshkol_is_null(alist)) {
        eshkol_report_error("assoc: second argument must be a proper list");
        return NULL;
    }
    
    return eshkol_create_boolean(false);
}
```

### Testing Strategy

Each list processing function will be tested with:

1. **Normal Cases**: Lists of various lengths
2. **Edge Cases**: Empty lists, singleton lists, circular lists
3. **Error Cases**: NULL pointers, non-list arguments, improper lists

Example test cases:

```c
void test_length() {
    // Normal cases
    EshkolObject* list1 = eshkol_create_list(3, 
                                           eshkol_create_integer(1),
                                           eshkol_create_integer(2),
                                           eshkol_create_integer(3));
    assert(eshkol_length(list1) == 3);
    
    // Edge cases
    EshkolObject* empty_list = eshkol_create_empty_list();
    assert(eshkol_length(empty_list) == 0);
    
    EshkolObject* singleton_list = eshkol_cons(eshkol_create_integer(1), eshkol_create_empty_list());
    assert(eshkol_length(singleton_list) == 1);
    
    // Error cases
    assert(eshkol_length(NULL) == 0);
    
    EshkolObject* improper_list = eshkol_cons(eshkol_create_integer(1), eshkol_create_integer(2));
    // Should report an error and return 0
    assert(eshkol_length(improper_list) == 0);
    
    // Cleanup
    eshkol_free_object(list1);
    eshkol_free_object(empty_list);
    eshkol_free_object(singleton_list);
    eshkol_free_object(improper_list);
}
```

## Dependencies

The implementation of list processing functions depends on:

1. **Core List Operations**: The `cons`, `car`, `cdr`, `set-car!`, `set-cdr!` functions must be implemented
2. **Type Predicates**: The `pair?`, `null?`, `list?` functions must be implemented
3. **Equality Predicates**: The `eq?`, `eqv?`, `equal?` functions must be implemented for the search functions

## Next Steps After Implementation

After implementing the list processing functions, the next steps are:

1. **Implement Higher-Order Functions**: `map`, `for-each`, `filter`, etc.
2. **Implement Vector Operations**: `vector`, `vector-ref`, `vector-set!`, etc.
3. **Implement String Operations**: `string`, `string-ref`, `string-set!`, etc.

## Conclusion

The implementation of list processing functions is a crucial step in the Scheme compatibility roadmap. It provides the foundation for more advanced list manipulation capabilities, which are essential for many Scheme programs.
