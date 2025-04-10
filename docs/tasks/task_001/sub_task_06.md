# Sub-Task 001-06: Implement `procedure?` Predicate

---

## Description

Implement the Scheme predicate `(procedure? obj)` in Eshkol, which returns `#t` if `obj` is a procedure (function, lambda, primitive), else `#f`.

---

## Dependencies

- Core procedure/function representation must be defined.
- Basic function definition support.

---

## Instructions

- Check the runtime type tag of `obj`.
- Return `#t` if it matches the procedure tag, else `#f`.
- Optimize for fast type check.
- Handle edge cases (null, other types).

---

## Success Criteria

- Correctly identifies procedures.
- Passes unit tests for procedures and non-procedures.
- Well-documented with examples.

---

## Status

_Not started_

---

## Notes

- Coordinate with overall type tagging scheme.
- Ensure compatibility with gradual typing.
