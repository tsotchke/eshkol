# Task 004: Implement Scheme Control Flow Macros

---

## Description

Implement core Scheme control flow constructs as macros or primitives in Eshkol:

- `cond`
- `case`
- `and`
- `or`
- `not`
- `when`
- `unless`

These constructs enable expressive conditional logic and are essential for Scheme compatibility.

---

## Dependencies

- **Task 001: Type predicates** (for `not`)
- **Task 003: List processing** (for macro argument handling)
- Macro system or macro-like transformation support must be available.
- Basic `if` expression must be implemented (already done).

---

## Resources

- [R5RS Standard, Section 4.2](https://schemers.org/Documents/Standards/R5RS/HTML/r5rs-Z-H-9.html#%_sec_4.2)
- [R7RS-small, Section 4.2](https://small.r7rs.org/attachment/r7rs.pdf)
- Eshkol docs: `docs/scheme_compatibility/IMPLEMENTATION_PLAN.md`
- Example files using conditionals

---

## Detailed Instructions

1. **Design**

   - Review macro system capabilities in Eshkol.
   - Define macro expansions per R5RS/R7RS:
     - `cond` → nested `if` expressions
     - `case` → nested `if` with `eqv?` comparisons
     - `and` → short-circuit nested `if`
     - `or` → short-circuit nested `if`
     - `when` → `if` with implicit `begin`
     - `unless` → negated `if`
   - Handle edge cases (empty clauses, else clauses).

2. **Implementation**

   - Implement as hygienic macros if possible.
   - Ensure short-circuit behavior for `and`/`or`.
   - Support arbitrary number of clauses.
   - Optimize expansions for performance/readability.

3. **Testing**

   - Unit tests for all constructs:
     - Normal cases
     - Edge cases (empty, else, single clause)
     - Nested conditionals
     - Short-circuit behavior
   - Compliance tests from R5RS/R7RS.
   - Integration tests with other language features.

4. **Documentation**

   - For each construct, document:
     - Macro expansion
     - Usage examples
     - Edge cases
     - Known limitations

---

## Success Criteria

- Macros expand correctly per Scheme standards.
- Pass all unit, integration, and compliance tests.
- Correct short-circuit behavior.
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

- Consider macro hygiene and variable capture.
- Optimize common cases.
- Ensure compatibility with gradual typing and memory model.
