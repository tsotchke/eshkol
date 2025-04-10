# Sub-Task 003-10: Implement Association List Functions (`assq`, `assv`, `assoc`)

---

## Description

Implement the Scheme functions:

- `(assq obj alist)` — search association list using `eq?`
- `(assv obj alist)` — search association list using `eqv?`
- `(assoc obj alist)` — search association list using `equal?`

Each returns the first matching pair or `#f` if not found.

---

## Dependencies

- Pair/list data structures and accessors.
- `eq?`, `eqv?`, `equal?` predicates.
- Basic function definition support.

---

## Instructions

- Traverse the association list (list of pairs).
- For each pair, compare `car` with `obj` using the appropriate predicate.
- Return the matching pair if found.
- Return `#f` if no match is found.
- Add tests for empty, nested, and no-match lists.

---

## Success Criteria

- Correctly finds or rejects matches per Scheme semantics.
- Passes unit tests for various alist cases.
- Well-documented with examples.

---

## Status

_Not started_

---

## Notes

- Optimize for tail recursion or iteration.
- Plan for future support of improper lists.
- Ensure compatibility with gradual typing.
