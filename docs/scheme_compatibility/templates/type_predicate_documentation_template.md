# Type Predicate Function Documentation Template

This template should be used for documenting type predicate functions in the Eshkol project.

## Function Name: `function-name?`

- **Signature**: `(function-name? obj) -> boolean`
- **Description**: Returns `#t` if `obj` is of the specified type, and `#f` otherwise.
- **Standard Reference**: R7RS section X.X
- **Implementation File**: src/core/utils/type_predicates.c
- **Implementation Details**: 
  - Brief description of how the function is implemented
  - Any special algorithms or techniques used
  - Performance considerations
- **Edge Cases Handled**:
  - NULL pointers
  - Invalid objects
  - Type conversion considerations
- **Known Limitations**:
  - Any known limitations or edge cases not handled
  - Performance issues
  - Compatibility issues
- **Test File**: tests/unit/test_type_predicates.c
- **Dependencies**:
  - List of other functions or modules this function depends on

## Example Usage

```scheme
;; Check if a value is of the specified type
(define value 42)
(function-name? value) ;; => #t or #f

;; Use in conditional expressions
(if (function-name? value)
    (do-something-with-value value)
    (handle-other-type value))

;; Use with higher-order functions
(filter function-name? some-list)
```

## Implementation

```c
// C implementation of the function
bool eshkol_is_function_name(EshkolObject* obj) {
    // NULL check
    if (obj == NULL) {
        return false;
    }
    
    // Check if the object has a valid type tag
    if (!eshkol_has_valid_type_tag(obj)) {
        eshkol_report_error("Invalid object: no type tag");
        return false;
    }
    
    // Type-specific check
    return obj->type == ESHKOL_TYPE_FUNCTION_NAME;
}
```

## Test Cases

```c
void test_function_name_predicate() {
    // Normal cases
    EshkolObject* obj1 = eshkol_create_function_name_object();
    assert(eshkol_is_function_name(obj1));
    
    // Edge cases
    EshkolObject* obj2 = eshkol_create_other_type_object();
    assert(!eshkol_is_function_name(obj2));
    
    // Error cases
    assert(!eshkol_is_function_name(NULL));
    
    // Cleanup
    eshkol_free_object(obj1);
    eshkol_free_object(obj2);
}
```

## Error Handling

The function should handle the following error cases:

1. **NULL Pointers**: Return `false` for NULL pointers.
2. **Invalid Type Tags**: Report an error and return `false` for objects with invalid type tags.
3. **Type Conversion**: Do not attempt to convert types; simply return `false` for objects of the wrong type.

## Performance Considerations

- The function should be efficient, with a time complexity of O(1).
- The function should not allocate memory.
- The function should not modify the object.

## Related Functions

- List of related functions or predicates
- Functions that use this predicate
- Functions that are used by this predicate

## Notes

- Any additional notes or considerations
- Historical context or design decisions
- Future improvements or changes
