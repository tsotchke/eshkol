# Eshkol Scheme Compatibility - Known Issues and Limitations

Last Updated: 2025-03-23

This document tracks known issues, limitations, and compatibility notes for Scheme support in Eshkol. It serves as a reference for users and developers to understand the current state of Scheme compatibility.

## Current Issues

| ID | Description | Severity | Affected Functions | Status | Target Fix |
|----|-------------|----------|-------------------|--------|------------|
| SCH-001 | Type system integration with Scheme is incomplete | Medium | Type annotations, type checking | Known limitation | Phase 4 |
| SCH-002 | No proper tail call optimization | Medium | Recursive functions | Known limitation | Phase 6 |
| SCH-003 | Limited support for the full numeric tower | Medium | Numeric operations | Known limitation | Phase 7 |
| SCH-004 | No hygienic macro system | Medium | Macros | Known limitation | Phase 7 |
| SCH-005 | No continuations support | Medium | call/cc, dynamic-wind | Known limitation | Phase 5 |

## Compatibility Notes

### R5RS vs. R7RS Differences

Eshkol aims to be compatible with both R5RS and R7RS-small, but there are some differences in how these standards are implemented:

1. **Character and String Encoding**: Eshkol uses Unicode (as specified in R7RS) rather than the unspecified encoding in R5RS.
2. **Exact/Inexact Numbers**: Eshkol follows the R7RS approach to exact and inexact numbers.
3. **Additional Procedures**: Eshkol implements the additional procedures specified in R7RS.
4. **Exception Handling**: Eshkol uses the R7RS exception system.

### Implementation-Specific Behavior

Eshkol has some implementation-specific behavior that may differ from other Scheme implementations:

1. **Memory Management**: Eshkol uses a hybrid memory management approach rather than traditional garbage collection.
2. **Compilation Model**: Eshkol compiles to C rather than interpreting or compiling directly to machine code.
3. **Type Annotations**: Eshkol supports optional static type annotations, which are not part of standard Scheme.
4. **Scientific Computing Extensions**: Eshkol includes additional vector and matrix operations for scientific computing.
5. **Automatic Differentiation**: Eshkol includes support for automatic differentiation, which is not part of standard Scheme.

### Standard Library Limitations

The following standard library features are not yet fully implemented:

1. **Complex Numbers**: Complex number support is planned for Phase 7.
2. **Rational Numbers**: Rational number support is planned for Phase 7.
3. **Library System**: The R7RS library system is not yet fully implemented.
4. **Dynamic FFI**: Dynamic foreign function interface is not yet implemented.

## Deferred Features

The following features are intentionally deferred to later phases:

1. **Full Continuations**: Full continuations support is deferred to Phase 5.
2. **Hygienic Macros**: Hygienic macro system is deferred to Phase 7.
3. **Full Numeric Tower**: Complete numeric tower is deferred to Phase 7.
4. **Advanced I/O**: Advanced I/O features are deferred to Phase 4.
5. **Module System**: Enhanced module system is deferred to Phase 7.

## Workarounds

### Tail Call Optimization

Until proper tail call optimization is implemented, you can work around stack overflow issues by:

1. Using iterative algorithms instead of recursive ones
2. Manually implementing trampolines for deeply recursive functions
3. Using loop constructs like `do` instead of recursion

### Type System Integration

Until the type system is fully integrated with Scheme, you can:

1. Use untyped Scheme code for most functionality
2. Use type annotations only for performance-critical code
3. Be aware that type errors may not be caught until runtime

### Numeric Tower Limitations

Until the full numeric tower is implemented, you can:

1. Use floating-point numbers for most calculations
2. Be aware of precision limitations
3. Implement your own rational or complex number types if needed

## Reporting Issues

If you encounter an issue with Scheme compatibility in Eshkol, please report it by:

1. Checking if the issue is already known (listed in this document)
2. Creating a new issue in the issue tracker with:
   - A minimal reproducible example
   - Expected behavior
   - Actual behavior
   - Eshkol version
   - Any relevant context

## Planned Improvements

See the [Evolution Roadmap](./EVOLUTION.md) for planned improvements to Scheme compatibility in Eshkol.

## Revision History

| Date | Changes |
|------|---------|
| 2025-03-23 | Initial document created |
