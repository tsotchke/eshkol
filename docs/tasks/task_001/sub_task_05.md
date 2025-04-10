# Sub-Task 001-05: Implement `char?` Predicate

---

## Description

Implement the Scheme predicate `(char? obj)` in Eshkol, which returns `#t` if `obj` is a character, else `#f`.

---

## Dependencies

- Core character type representation must be defined.
- Basic function definition support.

---

## Instructions

- Check the runtime type tag of `obj`.
- Return `#t` if it matches the character tag, else `#f`.
- Optimize for fast type check.
- Handle edge cases (null, other types).

---

## Success Criteria

- Correctly identifies characters.
- Passes unit tests for characters and non-characters.
- Well-documented with examples.

---

## Status

_Not started_

---

## Notes

- Coordinate with overall type tagging scheme.
- Ensure compatibility with gradual typing.
