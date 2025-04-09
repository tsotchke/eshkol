# Task 005: Implement Scheme Higher-Order Functions

---

## Description

Implement core Scheme higher-order functions in Eshkol:

- `map`
- `for-each`
- `apply`
- `filter`
- `fold-left`
- `fold-right`

These functions enable functional programming patterns, data processing, and are essential for Scheme compatibility.

---

## Dependencies

- **Task 001: Type predicates**
- **Task 002: Equality predicates**
- **Task 003: List processing**
- Basic lambda/function support must be implemented (already done).
- Variadic argument handling must be functional.

---

## Resources

- [R5RS Standard, Section 6.4](https://schemers.org/Documents/Standards/R5RS/HTML/r5rs-Z-H-9.html#%_sec_6.4)
- [R7RS-small, Section 6.5](https://small.r7rs.org/attachment/r7rs.pdf)
- Eshkol docs: `docs/scheme_compatibility/IMPLEMENTATION_PLAN.md`
- Eshkol docs: `docs/scheme_compatibility/roadmaps/higher_order_functions_roadmap.md` (if exists)
- Example files: `examples/higher_order_functions.esk`

---

## Detailed Instructions

1. **Design**

   - Review function calling conventions in Eshkol.
   - Define semantics per R5RS/R7RS:
     - `map`: apply function to elements of one or more lists, return new list.
     - `for-each`: like `map` but for side effects, returns unspecified.
     - `apply`: apply function to argument list, handle variadic spreading.
     - `filter`: return list of elements satisfying predicate.
     - `fold-left`/`fold-right`: accumulate values over list.
   - Handle edge cases (empty lists, multiple lists, improper lists).

2. **Implementation**

   - Implement `map` and `for-each` with support for multiple lists.
   - Implement `apply` with correct argument spreading.
   - Implement `filter` using predicate.
   - Implement `fold-left` and `fold-right` recursively or iteratively.
   - Optimize common cases (single list, small lists).

3. **Testing**

   - Unit tests for all functions:
     - Normal cases
     - Empty lists
     - Multiple lists
     - Edge cases
     - Error cases (mismatched list lengths)
   - Compliance tests from R5RS/R7RS.
   - Integration tests with other language features.

4. **Documentation**

   - For each function, document:
     - Signature
     - Description
     - Examples
     - Edge cases
     - Errors

---

## Success Criteria

- Functions behave per R5RS/R7RS specifications.
- Pass all unit, integration, and compliance tests.
- Correctly handle edge/error cases.
- Well-documented with examples.
- Enables downstream features (macros, libraries, user code).

---

## Dependencies for Next Tasks

- **Required by:**  
  - Macro system  
  - User libraries  
  - Advanced Scheme features

---

## Status

_Not started_

---

## Notes

- Consider tail-recursive or iterative implementations for performance.
- Optimize for common cases.
- Ensure compatibility with gradual typing and memory model.
