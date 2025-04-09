# Sub-Task 003-08: Implement `memv` Function

---

## Description

Implement the Scheme function `(memv obj list)` in Eshkol, which searches `list` for an element `eqv?` to `obj` and returns the sublist starting at the match, or `#f` if not found.

---

## Dependencies

- Pair/list data structures and accessors.
- `eqv?` predicate.
- Basic function definition support.

---

## Instructions

- Traverse the list.
- Use `eqv?` to compare each element with `obj`.
- Return the sublist starting at the first match.
- Return `#f` if no match is found.
- Add tests for empty, nested, and no-match lists.

---

## Success Criteria

- Correctly finds or rejects matches per Scheme semantics.
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
