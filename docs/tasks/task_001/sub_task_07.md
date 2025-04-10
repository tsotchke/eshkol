# Sub-Task 001-07: Implement `vector?` Predicate

---

## Description

Implement the Scheme predicate `(vector? obj)` in Eshkol, which returns `#t` if `obj` is a vector, else `#f`.

---

## Dependencies

- Core vector type representation must be defined.
- Basic function definition support.

---

## Instructions

- Check the runtime type tag of `obj`.
- Return `#t` if it matches the vector tag, else `#f`.
- Optimize for fast type check.
- Handle edge cases (null, other types).

---

## Success Criteria

- Correctly identifies vectors.
- Passes unit tests for vectors and non-vectors.
- Well-documented with examples.

---

## Status

_Not started_

---

## Notes

- Coordinate with overall type tagging scheme.
- Ensure compatibility with gradual typing.
