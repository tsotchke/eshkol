# Sub-Task 001-03: Implement `number?` Predicate

---

## Description

Implement the Scheme predicate `(number? obj)` in Eshkol, which returns `#t` if `obj` is a number (integer, float, etc.), else `#f`.

---

## Dependencies

- Core numeric type representations must be defined.
- Basic function definition support.

---

## Instructions

- Check the runtime type tag of `obj`.
- Return `#t` if it matches any numeric type tag, else `#f`.
- Optimize for fast type check.
- Handle edge cases (null, other types).

---

## Success Criteria

- Correctly identifies numbers.
- Passes unit tests for numbers and non-numbers.
- Well-documented with examples.

---

## Status

_Not started_

---

## Notes

- Coordinate with overall type tagging scheme.
- Ensure compatibility with gradual typing and numeric tower.
