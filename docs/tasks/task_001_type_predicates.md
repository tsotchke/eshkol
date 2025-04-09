# Task 001: Implement Scheme Type Predicates

---

## Description

Implement the core Scheme type predicate functions in Eshkol:

- `boolean?`
- `symbol?`
- `number?`
- `string?`
- `char?`
- `procedure?`
- `vector?`

These predicates test the type of a given value and return `#t` or `#f` accordingly. They are essential for Scheme compatibility and enable many other language features.

---

## Dependencies

- Core data structures (pairs, symbols, numbers, strings, vectors) must be implemented.
- Basic memory management and type tagging must be functional.
- None of these predicates depend on other Scheme functions, so this task can be started immediately.

---

## Resources

- [R5RS Standard, Section 6.2](https://schemers.org/Documents/Standards/R5RS/HTML/r5rs-Z-H-9.html#%_sec_6.2)
- [R7RS-small, Section 6.3](https://small.r7rs.org/attachment/r7rs.pdf)
- Eshkol docs: `docs/scheme_compatibility/IMPLEMENTATION_PLAN.md`
- Eshkol docs: `docs/scheme_compatibility/roadmaps/type_predicates_roadmap.md` (if exists)
- Example file: `examples/type_predicates.esk`

---

## Detailed Instructions

1. **Design**

   - Review existing type tagging or runtime type info in Eshkol.
   - Define or confirm unique tags/identifiers for booleans, symbols, numbers, strings, chars, procedures, vectors.
   - Ensure predicates can distinguish these types reliably.

2. **Implementation**

   - Implement each predicate as a primitive or built-in function.
   - Each should accept one argument and return `#t` if the argument matches the type, else `#f`.
   - Use efficient type checks (bitmask, tag compare, etc.).
   - Handle edge cases (e.g., `null` is not a pair, but is a list).

3. **Testing**

   - Create unit tests covering:
     - Correct detection of each type.
     - False negatives (e.g., `boolean?` on a number returns `#f`).
     - Edge cases (empty list, nested structures).
   - Use compliance tests from R5RS/R7RS where applicable.
   - Add tests to integration suite.

4. **Documentation**

   - For each predicate, document:
     - Signature
     - Description
     - Examples
     - Edge cases
     - Errors (if any)

---

## Success Criteria

- All predicates return correct results per R5RS/R7RS.
- Passes unit and integration tests.
- No false positives or negatives.
- Well-documented with examples.
- Enables downstream tasks (equality predicates, list processing, macros).

---

## Dependencies for Next Tasks

- **Required by:**  
  - Equality predicates (`eq?`, `eqv?`, `equal?`)  
  - List processing functions  
  - Control flow macros  
  - Advanced Scheme features

---

## Status

_Not started_

---

## Notes

- Consider optimizing for inlining or JIT in future.
- Ensure compatibility with gradual typing system.
- Coordinate with memory/tagging design for future-proofing.
