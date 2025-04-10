# Task 006: Implement Scheme Advanced Numeric Operations

---

## Description

Implement advanced numeric predicates and operations in Eshkol:

- Predicates: `zero?`, `positive?`, `negative?`, `odd?`, `even?`
- Operations: `max`, `min`, `abs`, `quotient`, `remainder`, `modulo`, `gcd`, `lcm`

These functions extend basic arithmetic, enabling richer numeric computations per Scheme standards.

---

## Dependencies

- Basic numeric types and arithmetic (`+`, `-`, `*`, `/`, `=`, `<`, `>`, etc.) must be implemented.
- **Task 001: Type predicates** (for `number?`)
- Type system integration for numeric types should be functional.

---

## Resources

- [R5RS Standard, Section 6.2](https://schemers.org/Documents/Standards/R5RS/HTML/r5rs-Z-H-9.html#%_sec_6.2)
- [R7RS-small, Section 6.2](https://small.r7rs.org/attachment/r7rs.pdf)
- Eshkol docs: `docs/scheme_compatibility/IMPLEMENTATION_PLAN.md`
- Example files with numeric operations

---

## Detailed Instructions

1. **Design**

   - Review numeric type hierarchy (integers, reals, rationals, complex).
   - Define semantics per R5RS/R7RS, including exactness and error cases.
   - Decide on integer division semantics (`quotient`, `remainder`, `modulo` differences).

2. **Implementation**

   - Implement predicates using numeric comparisons.
   - Implement `max` and `min` with variable arguments.
   - Implement `abs` with sign handling.
   - Implement integer division functions:
     - `quotient`: truncated division
     - `remainder`: remainder with sign of dividend
     - `modulo`: remainder with sign of divisor
   - Implement `gcd` and `lcm` using Euclidean algorithm.
   - Handle edge cases (zero, negatives, mixed exact/inexact).

3. **Testing**

   - Unit tests for all functions:
     - Normal cases
     - Edge cases (zero, negatives)
     - Exact and inexact numbers
     - Error cases (division by zero)
   - Compliance tests from R5RS/R7RS.
   - Integration tests with other numeric features.

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
- Enables downstream features (scientific computing, AI).

---

## Dependencies for Next Tasks

- **Required by:**  
  - Scientific computing extensions  
  - AI features  
  - User libraries

---

## Status

_Not started_

---

## Notes

- Consider performance optimizations for common cases.
- Ensure compatibility with gradual typing and numeric tower.
- Plan for future support of rationals and complex numbers.
