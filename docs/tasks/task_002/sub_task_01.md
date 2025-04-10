# Sub-Task 002-01: Implement `eq?` Predicate

---

## Description

Implement the Scheme predicate `(eq? obj1 obj2)` in Eshkol, which returns `#t` if `obj1` and `obj2` are the same object (pointer equality), else `#f`.

---

## Dependencies

- Core object representation and pointer comparison must be defined.
- Basic function definition support.

---

## Instructions

- Compare pointers or unique object IDs of `obj1` and `obj2`.
- Return `#t` if they are identical, else `#f`.
- Optimize for fast identity check.
- Handle edge cases (null, uninitialized).

---

## Success Criteria

- Correctly identifies identical vs. distinct objects.
- Passes unit tests for all core types.
- Well-documented with examples.

---

## Status

_Not started_

---

## Notes

- Coordinate with memory management and object allocation.
- Ensure compatibility with gradual typing.
