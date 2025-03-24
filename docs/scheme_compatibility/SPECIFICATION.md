# Eshkol Scheme Compatibility - Specification

Last Updated: 2025-03-23

This document serves as the definitive reference for Scheme compatibility in Eshkol. It outlines which Scheme standards we follow, how we handle differences between standards, and any intentional deviations from the standards.

## Scheme Standards

Eshkol aims to be compatible with the following Scheme standards:

1. **R5RS**: The Revised⁵ Report on the Algorithmic Language Scheme
2. **R7RS-small**: The Revised⁷ Report on the Algorithmic Language Scheme (small language)

Where these standards differ, we follow R7RS-small by default, but provide options for R5RS compatibility where feasible.

## Core Language Features

### Lexical Conventions

Eshkol follows the lexical conventions of R7RS-small, including:

- Case sensitivity for identifiers
- Support for the full range of numeric literals
- String and character literals as specified in R7RS
- Comments using semicolons (`;`)
- Block comments using `#|` and `|#`
- Datum comments using `#;`

### Data Types

Eshkol supports all standard Scheme data types:

- **Booleans**: `#t` and `#f`
- **Numbers**: Integers, floating-point, and exact/inexact numbers
- **Characters**: Unicode characters
- **Strings**: Unicode strings
- **Symbols**: Case-sensitive symbols
- **Pairs and Lists**: Constructed with `cons`, `list`, etc.
- **Vectors**: Fixed-length arrays
- **Procedures**: First-class functions

### Expression Types

Eshkol supports all standard Scheme expression types:

- **Self-evaluating expressions**: Numbers, strings, characters, booleans
- **Variable references**: Identifiers that refer to bindings
- **Quotation**: `quote` and `'` syntax
- **Procedure calls**: `(procedure arg1 arg2 ...)`
- **Lambda expressions**: `(lambda (params) body)`
- **Conditionals**: `if`, `cond`, `case`, etc.
- **Assignments**: `set!`
- **Derived expressions**: `and`, `or`, `when`, `unless`, etc.

### Standard Procedures

Eshkol implements the standard procedures as specified in R7RS-small. See the [Implementation Plan](./IMPLEMENTATION_PLAN.md) for details on which procedures are implemented in each phase.

## Compatibility Notes

### R5RS vs. R7RS Differences

Where R5RS and R7RS-small differ, we handle the differences as follows:

1. **Libraries**: We support the R7RS library system, but also allow R5RS-style top-level definitions.
2. **Character and String Encoding**: We use Unicode as specified in R7RS, rather than the unspecified encoding in R5RS.
3. **Exact/Inexact Numbers**: We follow the R7RS approach to exact and inexact numbers.
4. **Additional Procedures**: We implement the additional procedures specified in R7RS.
5. **Exception Handling**: We use the R7RS exception system.

### Implementation-Specific Features

Eshkol includes the following implementation-specific features:

1. **Scientific Computing Extensions**: Additional vector and matrix operations for scientific computing.
2. **Automatic Differentiation**: Support for automatic differentiation of functions.
3. **SIMD Optimization**: Automatic use of SIMD instructions for vector operations.
4. **C Interoperability**: Direct interface with C code.
5. **Type Annotations**: Optional static type annotations for improved performance and safety.

### Intentional Deviations

Eshkol intentionally deviates from the Scheme standards in the following ways:

1. **Performance Optimizations**: We may use different internal representations for data structures to improve performance.
2. **Memory Management**: We use a hybrid memory management approach rather than traditional garbage collection.
3. **Compilation Model**: We compile to C rather than interpreting or compiling directly to machine code.

## Standard Compliance

For each feature, we track compliance with the relevant standards:

- **Full Compliance**: The feature behaves exactly as specified in the standard.
- **Partial Compliance**: The feature behaves mostly as specified, but with some limitations or differences.
- **Non-Compliance**: The feature behaves differently from the standard, or is not implemented.

See the [Registry](./REGISTRY.md) for detailed compliance information for each feature.

## Testing Against Standards

We test Eshkol against standard Scheme test suites to ensure compatibility:

1. **R7RS Test Suite**: Tests for compliance with R7RS-small.
2. **SRFI Test Suites**: Tests for compliance with relevant SRFIs.
3. **Real-World Code**: Tests with real-world Scheme code to ensure practical compatibility.

## Future Standards

As Scheme evolves, we will update Eshkol to maintain compatibility with new standards:

1. **R7RS-large**: We will evaluate and implement relevant parts of R7RS-large as they are finalized.
2. **SRFIs**: We will implement relevant Scheme Requests for Implementation as they become widely adopted.

## References

1. [R5RS Specification](https://schemers.org/Documents/Standards/R5RS/r5rs.pdf)
2. [R7RS-small Specification](https://small.r7rs.org/attachment/r7rs.pdf)
3. [SRFI Index](https://srfi.schemers.org/)

## Revision History

- 2025-03-23: Initial specification created
