# Task 002: Implement Scheme Equality Predicates

---

## Description

Implement the core Scheme equality predicates in Eshkol:

- `eq?` — identity comparison (pointer equality)
- `eqv?` — equivalence relation, type-aware
- `equal?` — deep structural equality

These predicates are fundamental for Scheme semantics, enabling correct behavior of conditionals, data structure comparisons, and many language features.

---

## Dependencies

- **Task 001: Type predicates** (`boolean?`, `symbol?`, etc.) must be implemented.
- Core data structures (pairs, symbols, numbers, strings, vectors) must be functional.
- Basic memory management and type tagging must be in place.

---

## Resources

- [R5RS Standard, Section 6.1](https://schemers.org/Documents/Standards/R5RS/HTML/r5rs-Z-H-9.html#%_sec_6.1)
- [R7RS-small, Section 6.1](https://small.r7rs.org/attachment/r7rs.pdf)
- Eshkol docs: `docs/scheme_compatibility/IMPLEMENTATION_PLAN.md`
- Eshkol docs: `docs/scheme_compatibility/roadmaps/equality_predicates_roadmap.md` (if exists)
- Example file: `examples/equality_predicates.esk`

---

## Detailed Instructions

1. **Design**

   - Review how Eshkol represents different types internally.
   - Define clear semantics for each predicate:
     - `eq?`: pointer/reference equality.
     - `eqv?`: same as `eq?` for most types, but numeric/char equality for numbers and characters.
     - `equal?`: recursive deep equality for lists, vectors, strings, etc.
   - Handle special cases (e.g., NaN, infinities, mutable structures).

2. **Implementation**

   - Implement `eq?` as a fast pointer comparison.
   - Implement `eqv?`:
     - For numbers: compare numeric value, including exactness.
     - For characters: compare code points.
     - Else: fallback to `eq?`.
   - Implement `equal?`:
     - Recursively compare pairs, vectors, strings.
     - Use `eqv?` for atomic elements.
     - Avoid infinite loops on cyclic structures (optional for initial version).
   - Optimize common cases (e.g., identical references).

3. **Testing**

   - Create unit tests covering:
     - Same and different objects of all types.
     - Numeric edge cases (exact/inexact, NaN, infinities).
     - Deeply nested lists/vectors.
     - Edge cases (empty lists, empty vectors, empty strings).
   - Use compliance tests from R5RS/R7RS.
   - Add to integration suite.

4. **Documentation**

   - For each predicate, document:
     - Signature
     - Description
     - Examples
     - Edge cases
     - Known limitations (e.g., cyclic structures)

---

## Success Criteria

- Predicates behave per R5RS/R7RS specifications.
- Passes all unit and integration tests.
- Correctly distinguishes identity, equivalence, and deep equality.
- Well-documented with examples.
- Enables downstream features (list processing, macros, control flow).

---

## Dependencies for Next Tasks

- **Required by:**  
  - List processing functions (`memq`, `memv`, `member`, `assq`, `assv`, `assoc`)  
  - Macro system  
  - Control flow constructs  
  - Advanced Scheme features

---

## Status

_Not started_

---

## Notes

- Consider cycle detection in `equal?` for robustness.
- Optimize for common cases (e.g., identical references).
- Ensure compatibility with gradual typing and memory model.
