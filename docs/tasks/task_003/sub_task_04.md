# Sub-Task 003-04: Implement `list-ref` Function

---

## Description

Implement the Scheme function `(list-ref list k)` in Eshkol, which returns the k-th element of `list` (0-based index).

---

## Dependencies

- Pair/list data structures and accessors.
- Basic function definition support.

---

## Instructions

- Traverse the list, counting elements.
- Return the element at index `k`.
- Raise an error if `k` is out of bounds or list is improper.
- Add tests for valid, invalid, empty, and nested lists.

---

## Success Criteria

- Correctly returns the k-th element per Scheme semantics.
- Passes unit tests for various list cases.
- Well-documented with examples.

---

## Status

_Not started_

---

## Notes

- Optimize for tail recursion or iteration.
- Plan for future support of improper lists.
- Ensure compatibility with gradual typing.
