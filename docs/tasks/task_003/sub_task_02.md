# Sub-Task 003-02: Implement `append` Function

---

## Description

Implement the Scheme function `(append list1 list2 ...)` in Eshkol, which concatenates multiple lists into a new list.

---

## Dependencies

- Pair/list data structures and accessors.
- Basic function definition support.

---

## Instructions

- Traverse each list except the last, copying elements.
- Link the last list directly (do not copy).
- Handle empty lists correctly.
- Support multiple argument lists.
- Add tests for empty, nested, and multiple lists.

---

## Success Criteria

- Correctly concatenates lists per Scheme semantics.
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
