# Sub-Task 002-05: Handle Character Equality Edge Cases in `eqv?` and `equal?`

---

## Description

Implement correct handling of character equality in `eqv?` and `equal?` predicates, including:

- Unicode code point comparisons
- Case sensitivity
- Normalization (optional, future)
- Edge cases (null, invalid characters)

---

## Dependencies

- Character type representation and comparison functions.
- Basic `eqv?` and `equal?` implementations.

---

## Instructions

- Compare Unicode code points directly.
- Ensure case-sensitive comparison per Scheme standards.
- Plan for future Unicode normalization support.
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

- Coordinate with character encoding design.
- Consider Unicode normalization in future.
- Ensure compatibility with gradual typing.
