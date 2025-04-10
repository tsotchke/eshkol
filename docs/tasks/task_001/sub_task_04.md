# Sub-Task 001-04: Implement `string?` Predicate

---

## Description

Implement the Scheme predicate `(string? obj)` in Eshkol, which returns `#t` if `obj` is a string, else `#f`.

---

## Dependencies

- Core string type representation must be defined.
- Basic function definition support.

---

## Instructions

- Check the runtime type tag of `obj`.
- Return `#t` if it matches the string tag, else `#f`.
- Optimize for fast type check.
- Handle edge cases (null, other types).

---

## Success Criteria

- Correctly identifies strings.
- Passes unit tests for strings and non-strings.
- Well-documented with examples.

---

## Status

_Not started_

---

## Notes

- Coordinate with overall type tagging scheme.
- Ensure compatibility with gradual typing.
