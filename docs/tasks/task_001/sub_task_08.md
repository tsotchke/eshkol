# Sub-Task 001-08: Design Type Tagging Scheme

---

## Description

Design the runtime type tagging scheme for Eshkol to support efficient and correct implementation of type predicates and gradual typing.

---

## Dependencies

- Knowledge of all core data types (booleans, symbols, numbers, strings, chars, procedures, vectors, pairs, etc.)
- Understanding of memory layout and performance considerations.

---

## Instructions

- Define unique tags or bit patterns for each core type.
- Plan how to embed tags in object headers or pointer bits.
- Ensure compatibility with gradual typing and type inference.
- Optimize for fast type checks (bitmasking, pointer tagging).
- Document the tagging scheme clearly.
- Plan for extensibility (user types, future features).

---

## Success Criteria

- Tagging scheme supports all core types.
- Enables fast, reliable type checks.
- Compatible with gradual typing.
- Well-documented with diagrams/examples.

---

## Status

_Not started_

---

## Notes

- Coordinate with memory management design.
- Consider alignment and platform-specific constraints.
- Plan for future extensions (complex types, user-defined types).
