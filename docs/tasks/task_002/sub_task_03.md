# Sub-Task 002-03: Implement `equal?` Predicate

---

## Description

Implement the Scheme predicate `(equal? obj1 obj2)` in Eshkol, which returns `#t` if `obj1` and `obj2` are deeply structurally equal, else `#f`.

---

## Dependencies

- Core data structures (pairs, vectors, strings, numbers, symbols, chars).
- Basic function definition support.

---

## Instructions

- For atomic types, use `eqv?`.
- For pairs: recursively compare `car` and `cdr`.
- For vectors: compare length, then elements recursively.
- For strings: compare length and characters.
- Avoid infinite loops on cyclic structures (optional for initial version).
- Handle edge cases (null, empty structures).

---

## Success Criteria

- Correctly identifies deep equality per Scheme semantics.
- Passes unit tests for nested structures and atomic types.
- Well-documented with examples.

---

## Status

_Not started_

---

## Notes

- Consider cycle detection for robustness.
- Optimize common cases (identical references).
- Ensure compatibility with gradual typing.
