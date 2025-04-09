# Task 003: Implement Scheme List Processing Functions

---

## Description

Implement core Scheme list processing functions in Eshkol:

- `length`
- `append`
- `reverse`
- `list-ref`
- `list-tail`
- `list-set!`
- `memq`
- `memv`
- `member`
- `assq`
- `assv`
- `assoc`

These functions enable fundamental list manipulations, searches, and updates, essential for Scheme compatibility.

---

## Dependencies

- **Task 001: Type predicates** must be complete.
- **Task 002: Equality predicates** must be complete.
- Core pairs/lists implementation must be functional.
- Basic memory management must be stable.

---

## Resources

- [R5RS Standard, Section 6.3](https://schemers.org/Documents/Standards/R5RS/HTML/r5rs-Z-H-9.html#%_sec_6.3)
- [R7RS-small, Section 6.4](https://small.r7rs.org/attachment/r7rs.pdf)
- Eshkol docs: `docs/scheme_compatibility/IMPLEMENTATION_PLAN.md`
- Eshkol docs: `docs/scheme_compatibility/roadmaps/list_processing_roadmap.md` (if exists)
- Example file: `examples/list_operations.esk`

---

## Detailed Instructions

1. **Design**

   - Review list representation (pairs, null).
   - Define semantics per R5RS/R7RS, including error cases (e.g., out-of-bounds).
   - Decide on iterative vs recursive implementations for performance.

2. **Implementation**

   - `length`: count elements, error on improper list (optional).
   - `append`: concatenate multiple lists, create new list.
   - `reverse`: create reversed copy.
   - `list-ref`: access element by index, error if out-of-bounds.
   - `list-tail`: return sublist starting at index, error if out-of-bounds.
   - `list-set!`: mutate element at index, error if out-of-bounds.
   - `memq`, `memv`, `member`: search list using `eq?`, `eqv?`, `equal?` respectively, return sublist starting at match or `#f`.
   - `assq`, `assv`, `assoc`: search association list (list of pairs) using `eq?`, `eqv?`, `equal?`, return matching pair or `#f`.
   - Handle edge cases (empty lists, improper lists).

3. **Testing**

   - Unit tests for all functions:
     - Normal cases
     - Empty lists
     - Nested lists
     - Out-of-bounds errors
     - Mutation correctness (`list-set!`)
     - Search success/failure
   - Compliance tests from R5RS/R7RS.
   - Integration tests with other Scheme features.

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
- Enables downstream features (macros, control flow, libraries).

---

## Dependencies for Next Tasks

- **Required by:**  
  - Control flow macros  
  - Macro system  
  - Advanced Scheme features  
  - User libraries

---

## Status

_Not started_

---

## Notes

- Consider tail-recursive or iterative implementations for performance.
- Optimize common cases (e.g., appending empty list).
- Ensure compatibility with gradual typing and memory model.
