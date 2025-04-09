# Sub-Task 003-06: Implement `list-set!` Function

---

## Description

Implement the Scheme function `(list-set! list k obj)` in Eshkol, which mutates the k-th element of `list` to be `obj`.

---

## Dependencies

- Pair/list data structures and accessors.
- Basic function definition support.
- Mutable pairs support.

---

## Instructions

- Traverse the list to index `k`.
- Mutate the `car` of the k-th pair to `obj`.
- Raise an error if `k` is out of bounds or list is improper.
- Add tests for valid, invalid, empty, and nested lists.

---

## Success Criteria

- Correctly mutates the k-th element per Scheme semantics.
- Passes unit tests for various list cases.
- Well-documented with examples.

---

## Status

_Not started_

---

## Notes

- Optimize for tail recursion or iteration.
- Plan for future support of improper lists.
- Ensure compatibility with gradual typing and immutability rules.
