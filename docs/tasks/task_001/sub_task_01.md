# Sub-Task 001-01: Implement `boolean?` Predicate

---

## Description

Implement the Scheme predicate `(boolean? obj)` in Eshkol, which returns `#t` if `obj` is a boolean (`#t` or `#f`), else `#f`.

---

## Dependencies

- Core boolean type representation must be defined.
- Basic function definition support.

---

## Instructions

- Check the runtime type tag of `obj`.
- Return `#t` if it matches the boolean tag, else `#f`.
- Optimize for fast type check (bitmask, tag compare).
- Handle edge cases (null, other types).

---

## Success Criteria

- Correctly identifies booleans.
- Passes unit tests for booleans and non-booleans.
- Well-documented with examples.

---

## Status

_Not started_

---

## Notes

- Coordinate with overall type tagging scheme.
- Ensure compatibility with gradual typing.
