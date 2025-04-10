# Sub-Task 002-02: Implement `eqv?` Predicate

---

## Description

Implement the Scheme predicate `(eqv? obj1 obj2)` in Eshkol, which returns `#t` if `obj1` and `obj2` are equivalent (identity for most types, value equality for numbers and characters), else `#f`.

---

## Dependencies

- Core object representation and numeric/char comparison must be defined.
- Basic function definition support.

---

## Instructions

- If `obj1` and `obj2` are the same object (pointer equality), return `#t`.
- For numbers: return `#t` if numerically equal, else `#f`.
- For characters: return `#t` if code points equal, else `#f`.
- For other types: fallback to pointer equality.
- Handle edge cases (NaN, infinities, null).

---

## Success Criteria

- Correctly identifies equivalence per Scheme semantics.
- Passes unit tests for all core types.
- Well-documented with examples.

---

## Status

_Not started_

---

## Notes

- Coordinate with numeric tower and character encoding.
- Ensure compatibility with gradual typing.
