# Sub-Task 002-04: Handle Numeric Equality Edge Cases in `eqv?` and `equal?`

---

## Description

Implement correct handling of numeric edge cases in `eqv?` and `equal?` predicates, including:

- Exact vs. inexact numbers
- NaN (Not a Number)
- Positive and negative infinity
- Signed zero (+0.0 vs -0.0)
- Complex numbers (future support)

---

## Dependencies

- Numeric type representations and comparison functions.
- Basic `eqv?` and `equal?` implementations.

---

## Instructions

- Follow R5RS/R7RS semantics for numeric equality.
- Ensure NaN is not equal to anything, including itself.
- Handle infinities correctly.
- Distinguish or equate signed zeros per standard.
- Plan for future complex/rational support.
- Add tests for all edge cases.

---

## Success Criteria

- Correct behavior per Scheme standards.
- Passes all edge case tests.
- Well-documented with examples.

---

## Status

_Not started_

---

## Notes

- Coordinate with numeric tower design.
- Consider IEEE 754 compliance.
- Ensure compatibility with gradual typing.
