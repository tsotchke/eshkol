# Sub-Task 001-02: Implement `symbol?` Predicate

---

## Description

Implement the Scheme predicate `(symbol? obj)` in Eshkol, which returns `#t` if `obj` is a symbol, else `#f`.

---

## Dependencies

- Core symbol type representation must be defined.
- Basic function definition support.

---

## Instructions

- Check the runtime type tag of `obj`.
- Return `#t` if it matches the symbol tag, else `#f`.
- Optimize for fast type check.
- Handle edge cases (null, other types).

---

## Success Criteria

- Correctly identifies symbols.
- Passes unit tests for symbols and non-symbols.
- Well-documented with examples.

---

## Status

_Not started_

---

## Notes

- Coordinate with overall type tagging scheme.
- Ensure compatibility with gradual typing.
