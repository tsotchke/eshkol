# Function Documentation Template

This template should be used when documenting Scheme functions in Eshkol.

## Function: `function-name`

### Standard Reference
- **Standard**: R7RS
- **Section**: X.Y.Z
- **Page**: N

### Signature
```scheme
(function-name arg1 arg2 ...)
```

### Description
A clear and concise description of what the function does.

### Parameters
- `arg1`: Description of the first argument
- `arg2`: Description of the second argument
- ...

### Return Value
Description of the return value.

### Examples
```scheme
(function-name 1 2)  ; => 3
(function-name "a" "b")  ; => "ab"
```

### Edge Cases
- What happens with empty input?
- What happens with invalid input?
- Any other special cases?

### Error Conditions
- When does this function signal an error?
- What type of error is signaled?

### Implementation Notes
- Any implementation-specific details
- Performance considerations
- Memory usage
- Optimizations

### Dependencies
- List of functions this function depends on
- Any other dependencies

### Compatibility Notes
- Any differences from the standard
- Compatibility with R5RS vs. R7RS
- Compatibility with other Scheme implementations

### Test Cases
- List of test cases that should be implemented
- Include normal cases, edge cases, and error cases

### Implementation Status
- **Status**: Planned/In Progress/Implemented/Tested
- **Priority**: P0/P1/P2/P3/P4
- **Target Phase**: Phase X
- **Implementation File**: src/path/to/file.c
- **Test File**: tests/path/to/test.c
- **Implemented By**: Developer Name
- **Implemented On**: YYYY-MM-DD
- **Last Updated**: YYYY-MM-DD

### Related Functions
- List of related functions
- Functions that are often used together with this function
- Alternative functions that provide similar functionality
