# Higher-Order Functions Implementation Roadmap

Last Updated: 2025-03-29

## Overview

This document outlines the roadmap for implementing higher-order functions in Eshkol. Higher-order functions are functions that take other functions as arguments or return functions as results. They are a fundamental part of functional programming and are essential for Scheme compatibility.

## Implementation Status

| Function | Priority | Status | Target Date | Dependencies |
|----------|----------|--------|-------------|--------------|
| `map` | P1 | Planned | Q2 2025 | Core list operations, Lambda support |
| `for-each` | P1 | Planned | Q2 2025 | Core list operations, Lambda support |
| `filter` | P2 | Planned | Q3 2025 | Core list operations, Lambda support |
| `fold-left` | P2 | Planned | Q3 2025 | Core list operations, Lambda support |
| `fold-right` | P2 | Planned | Q3 2025 | Core list operations, Lambda support |
| `compose` | P3 | Planned | Q4 2025 | Lambda support |
| `curry` | P3 | Planned | Q4 2025 | Lambda support |

## Implementation Plan

### Phase 1: Basic Higher-Order Functions (Week 1-2)

1. **Implement Map and For-Each**
   - `map`: Apply a function to each element of a list and return a new list
   - `for-each`: Apply a function to each element of a list for side effects

2. **Documentation**
   - Document each function
   - Update function status documentation
   - Update test coverage documentation

### Phase 2: Filtering and Folding (Week 3-4)

1. **Implement Filter and Fold Functions**
   - `filter`: Filter a list based on a predicate
   - `fold-left`: Fold a list from left to right
   - `fold-right`: Fold a list from right to left

2. **Documentation**
   - Document each function
   - Update function status documentation
   - Update test coverage documentation

### Phase 3: Function Composition and Currying (Week 5-6)

1. **Implement Function Composition and Currying**
   - `compose`: Compose two or more functions
   - `curry`: Convert a function that takes multiple arguments into a sequence of functions that each take a single argument

2. **Documentation**
   - Document each function
   - Update function status documentation
   - Update test coverage documentation

## Implementation Details

### map Implementation

The `map` function applies a function to each element of a list and returns a new list.

```c
// map function
EshkolObject* eshkol_map(EshkolObject* func, EshkolObject* list) {
    // NULL check
    if (func == NULL || list == NULL) {
        eshkol_report_error("map: function or list is NULL");
        return NULL;
    }
    
    // Empty list check
    if (eshkol_is_null(list)) {
        return eshkol_create_empty_list();
    }
    
    // Type check
    if (!eshkol_is_procedure(func)) {
        eshkol_report_error("map: first argument must be a procedure");
        return NULL;
    }
    
    if (!eshkol_is_pair(list) && !eshkol_is_null(list)) {
        eshkol_report_error("map: second argument must be a list");
        return NULL;
    }
    
    // Apply the function to each element
    EshkolObject* result = eshkol_create_empty_list();
    EshkolObject* last_pair = NULL;
    
    while (eshkol_is_pair(list)) {
        // Apply the function to the current element
        EshkolObject* args = eshkol_cons(eshkol_car(list), eshkol_create_empty_list());
        EshkolObject* value = eshkol_apply(func, args);
        eshkol_free_object(args);
        
        // Add the result to the new list
        EshkolObject* new_pair = eshkol_cons(value, eshkol_create_empty_list());
        
        if (eshkol_is_null(result)) {
            result = new_pair;
            last_pair = new_pair;
        } else {
            eshkol_set_cdr(last_pair, new_pair);
            last_pair = new_pair;
        }
        
        // Move to the next element
        list = eshkol_cdr(list);
    }
    
    // Check if the list is proper
    if (!eshkol_is_null(list)) {
        eshkol_report_error("map: second argument must be a proper list");
        eshkol_free_object(result);
        return NULL;
    }
    
    return result;
}
```

### for-each Implementation

The `for-each` function applies a function to each element of a list for side effects.

```c
// for-each function
void eshkol_for_each(EshkolObject* func, EshkolObject* list) {
    // NULL check
    if (func == NULL || list == NULL) {
        eshkol_report_error("for-each: function or list is NULL");
        return;
    }
    
    // Empty list check
    if (eshkol_is_null(list)) {
        return;
    }
    
    // Type check
    if (!eshkol_is_procedure(func)) {
        eshkol_report_error("for-each: first argument must be a procedure");
        return;
    }
    
    if (!eshkol_is_pair(list) && !eshkol_is_null(list)) {
        eshkol_report_error("for-each: second argument must be a list");
        return;
    }
    
    // Apply the function to each element
    while (eshkol_is_pair(list)) {
        // Apply the function to the current element
        EshkolObject* args = eshkol_cons(eshkol_car(list), eshkol_create_empty_list());
        EshkolObject* value = eshkol_apply(func, args);
        eshkol_free_object(args);
        eshkol_free_object(value);
        
        // Move to the next element
        list = eshkol_cdr(list);
    }
    
    // Check if the list is proper
    if (!eshkol_is_null(list)) {
        eshkol_report_error("for-each: second argument must be a proper list");
    }
}
```

### filter Implementation

The `filter` function filters a list based on a predicate.

```c
// filter function
EshkolObject* eshkol_filter(EshkolObject* pred, EshkolObject* list) {
    // NULL check
    if (pred == NULL || list == NULL) {
        eshkol_report_error("filter: predicate or list is NULL");
        return NULL;
    }
    
    // Empty list check
    if (eshkol_is_null(list)) {
        return eshkol_create_empty_list();
    }
    
    // Type check
    if (!eshkol_is_procedure(pred)) {
        eshkol_report_error("filter: first argument must be a procedure");
        return NULL;
    }
    
    if (!eshkol_is_pair(list) && !eshkol_is_null(list)) {
        eshkol_report_error("filter: second argument must be a list");
        return NULL;
    }
    
    // Filter the list
    EshkolObject* result = eshkol_create_empty_list();
    EshkolObject* last_pair = NULL;
    
    while (eshkol_is_pair(list)) {
        // Apply the predicate to the current element
        EshkolObject* args = eshkol_cons(eshkol_car(list), eshkol_create_empty_list());
        EshkolObject* value = eshkol_apply(pred, args);
        eshkol_free_object(args);
        
        // If the predicate returns true, add the element to the result
        if (eshkol_is_true(value)) {
            EshkolObject* new_pair = eshkol_cons(eshkol_car(list), eshkol_create_empty_list());
            
            if (eshkol_is_null(result)) {
                result = new_pair;
                last_pair = new_pair;
            } else {
                eshkol_set_cdr(last_pair, new_pair);
                last_pair = new_pair;
            }
        }
        
        eshkol_free_object(value);
        
        // Move to the next element
        list = eshkol_cdr(list);
    }
    
    // Check if the list is proper
    if (!eshkol_is_null(list)) {
        eshkol_report_error("filter: second argument must be a proper list");
        eshkol_free_object(result);
        return NULL;
    }
    
    return result;
}
```

### fold-left Implementation

The `fold-left` function folds a list from left to right.

```c
// fold-left function
EshkolObject* eshkol_fold_left(EshkolObject* func, EshkolObject* init, EshkolObject* list) {
    // NULL check
    if (func == NULL || list == NULL) {
        eshkol_report_error("fold-left: function or list is NULL");
        return NULL;
    }
    
    // Type check
    if (!eshkol_is_procedure(func)) {
        eshkol_report_error("fold-left: first argument must be a procedure");
        return NULL;
    }
    
    if (!eshkol_is_pair(list) && !eshkol_is_null(list)) {
        eshkol_report_error("fold-left: third argument must be a list");
        return NULL;
    }
    
    // Fold the list
    EshkolObject* result = init;
    
    while (eshkol_is_pair(list)) {
        // Apply the function to the accumulator and the current element
        EshkolObject* args = eshkol_cons(result, eshkol_cons(eshkol_car(list), eshkol_create_empty_list()));
        EshkolObject* value = eshkol_apply(func, args);
        eshkol_free_object(args);
        
        // Update the accumulator
        result = value;
        
        // Move to the next element
        list = eshkol_cdr(list);
    }
    
    // Check if the list is proper
    if (!eshkol_is_null(list)) {
        eshkol_report_error("fold-left: third argument must be a proper list");
        return NULL;
    }
    
    return result;
}
```

### fold-right Implementation

The `fold-right` function folds a list from right to left.

```c
// fold-right function
EshkolObject* eshkol_fold_right(EshkolObject* func, EshkolObject* init, EshkolObject* list) {
    // NULL check
    if (func == NULL || list == NULL) {
        eshkol_report_error("fold-right: function or list is NULL");
        return NULL;
    }
    
    // Empty list check
    if (eshkol_is_null(list)) {
        return init;
    }
    
    // Type check
    if (!eshkol_is_procedure(func)) {
        eshkol_report_error("fold-right: first argument must be a procedure");
        return NULL;
    }
    
    if (!eshkol_is_pair(list)) {
        eshkol_report_error("fold-right: third argument must be a list");
        return NULL;
    }
    
    // Recursively fold the list
    EshkolObject* rest = eshkol_fold_right(func, init, eshkol_cdr(list));
    
    // Apply the function to the current element and the result of folding the rest
    EshkolObject* args = eshkol_cons(eshkol_car(list), eshkol_cons(rest, eshkol_create_empty_list()));
    EshkolObject* result = eshkol_apply(func, args);
    eshkol_free_object(args);
    
    return result;
}
```

### compose Implementation

The `compose` function composes two or more functions.

```c
// compose function
EshkolObject* eshkol_compose(EshkolObject* f, EshkolObject* g) {
    // NULL check
    if (f == NULL || g == NULL) {
        eshkol_report_error("compose: functions are NULL");
        return NULL;
    }
    
    // Type check
    if (!eshkol_is_procedure(f) || !eshkol_is_procedure(g)) {
        eshkol_report_error("compose: arguments must be procedures");
        return NULL;
    }
    
    // Create a new lambda that applies g to the arguments and then applies f to the result
    EshkolObject* lambda = eshkol_create_lambda(
        eshkol_create_symbol("args"),
        eshkol_create_application(
            f,
            eshkol_cons(
                eshkol_create_application(
                    g,
                    eshkol_create_symbol("args")
                ),
                eshkol_create_empty_list()
            )
        )
    );
    
    return lambda;
}
```

### curry Implementation

The `curry` function converts a function that takes multiple arguments into a sequence of functions that each take a single argument.

```c
// curry function
EshkolObject* eshkol_curry(EshkolObject* func, int n) {
    // NULL check
    if (func == NULL) {
        eshkol_report_error("curry: function is NULL");
        return NULL;
    }
    
    // Type check
    if (!eshkol_is_procedure(func)) {
        eshkol_report_error("curry: argument must be a procedure");
        return NULL;
    }
    
    // Create a curried function
    EshkolObject* curried = func;
    
    for (int i = 0; i < n; i++) {
        EshkolObject* arg_name = eshkol_create_symbol("arg");
        
        curried = eshkol_create_lambda(
            arg_name,
            eshkol_create_lambda(
                eshkol_create_symbol("args"),
                eshkol_create_application(
                    curried,
                    eshkol_cons(
                        arg_name,
                        eshkol_create_symbol("args")
                    )
                )
            )
        );
    }
    
    return curried;
}
```

### Testing Strategy

Each higher-order function will be tested with:

1. **Normal Cases**: Lists of various lengths, different types of functions
2. **Edge Cases**: Empty lists, singleton lists, identity functions
3. **Error Cases**: NULL pointers, non-function arguments, non-list arguments, improper lists

Example test cases:

```c
void test_map() {
    // Create a function that doubles its argument
    EshkolObject* double_func = eshkol_create_lambda(
        eshkol_create_symbol("x"),
        eshkol_create_application(
            eshkol_create_symbol("*"),
            eshkol_cons(
                eshkol_create_symbol("x"),
                eshkol_cons(
                    eshkol_create_integer(2),
                    eshkol_create_empty_list()
                )
            )
        )
    );
    
    // Normal cases
    EshkolObject* list1 = eshkol_create_list(3, 
                                           eshkol_create_integer(1),
                                           eshkol_create_integer(2),
                                           eshkol_create_integer(3));
    EshkolObject* result1 = eshkol_map(double_func, list1);
    assert(eshkol_equal(result1, eshkol_create_list(3,
                                                  eshkol_create_integer(2),
                                                  eshkol_create_integer(4),
                                                  eshkol_create_integer(6))));
    
    // Edge cases
    EshkolObject* empty_list = eshkol_create_empty_list();
    EshkolObject* result2 = eshkol_map(double_func, empty_list);
    assert(eshkol_is_null(result2));
    
    // Error cases
    assert(eshkol_map(NULL, list1) == NULL);
    assert(eshkol_map(double_func, NULL) == NULL);
    assert(eshkol_map(eshkol_create_integer(42), list1) == NULL);
    
    // Cleanup
    eshkol_free_object(double_func);
    eshkol_free_object(list1);
    eshkol_free_object(result1);
    eshkol_free_object(empty_list);
    eshkol_free_object(result2);
}
```

## Dependencies

The implementation of higher-order functions depends on:

1. **Core List Operations**: The `cons`, `car`, `cdr`, `set-car!`, `set-cdr!` functions must be implemented
2. **Lambda Support**: The ability to create and apply lambda functions
3. **Type Predicates**: The `procedure?`, `pair?`, `null?` functions must be implemented
4. **Equality Predicates**: The `eq?`, `eqv?`, `equal?` functions must be implemented for testing

## Next Steps After Implementation

After implementing the higher-order functions, the next steps are:

1. **Implement Advanced Higher-Order Functions**: `call-with-current-continuation`, `dynamic-wind`, etc.
2. **Implement Macro System**: `syntax-rules`, `define-syntax`, etc.
3. **Implement Module System**: `import`, `export`, etc.

## Conclusion

The implementation of higher-order functions is a crucial step in the Scheme compatibility roadmap. It provides the foundation for functional programming in Eshkol, which is essential for many Scheme programs.
