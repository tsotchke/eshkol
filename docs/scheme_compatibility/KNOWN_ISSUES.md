# Eshkol Scheme Compatibility - Known Issues and Limitations

Last Updated: 2025-03-29

This document tracks known issues, limitations, and compatibility notes for Scheme support in Eshkol. It serves as a reference for users and developers to understand the current state of Scheme compatibility.

## Current Issues

| ID | Description | Severity | Affected Functions | Status | Target Fix |
|----|-------------|----------|-------------------|--------|------------|
| SCH-001 | Type system integration with Scheme is incomplete | Medium | Type annotations, type checking | Known limitation | Phase 4 |
| SCH-002 | No proper tail call optimization | Medium | Recursive functions | Known limitation | Phase 6 |
| SCH-003 | Limited support for the full numeric tower | Medium | Numeric operations | Known limitation | Phase 7 |
| SCH-004 | No hygienic macro system | Medium | Macros | Known limitation | Phase 7 |
| SCH-005 | No continuations support | Medium | call/cc, dynamic-wind | Known limitation | Phase 5 |
| SCH-006 | Type inference for autodiff functions is incomplete | High | autodiff-forward, autodiff-reverse, gradient functions | In Progress | Phase 4 |
| SCH-007 | Vector return types not correctly handled | High | Vector operations, gradient functions | In Progress | Phase 4 |
| SCH-008 | Type conflicts in generated C code | High | Autodiff and vector functions | In Progress | Phase 4 |
| SCH-009 | Inconsistent type handling between scalar and vector operations | Medium | Vector math operations | In Progress | Phase 4 |
| SCH-010 | Implicit conversions between numeric types not fully supported | Medium | Numeric operations | Known limitation | Phase 4 |
| SCH-011 | Lambda capture analysis for autodiff functions is incomplete | High | Higher-order autodiff functions | In Progress | Phase 4 |
| SCH-012 | MCP tools for type analysis use simplified implementations | Medium | analyze-types tool | In Progress | N/A |
| SCH-013 | Core list operations implemented | High | cons, car, cdr | Implemented | Phase 7 |
| SCH-014 | Basic type predicates not yet implemented | Medium | pair?, null?, list? | Planning | Phase 7 |
| SCH-015 | Mutual recursion handling in type inference is incomplete | Medium | Mutually recursive functions | In Progress | Phase 4 |
| SCH-016 | Equality predicates not yet implemented | Medium | eq?, eqv?, equal? | Planning | Phase 7 |
| SCH-017 | Higher-order functions not yet implemented | Medium | map, for-each, filter | Planning | Phase 7 |
| SCH-018 | List processing functions not yet implemented | Medium | length, append, reverse | Planning | Phase 7 |
| SCH-019 | Function composition not working correctly | High | compose, higher-order functions | In Progress | Phase 4 |

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
6. **MCP Tools Integration**: Eshkol provides MCP tools for analyzing Scheme code, which are not part of standard Scheme.

### Standard Library Limitations

The following standard library features are not yet fully implemented:

1. **List Operations**: 
   - Core list operations (cons, car, cdr) have been implemented.
   - Additional list processing functions (length, append, reverse, etc.) are planned for Q2 2025.
   - See [list_processing_roadmap.md](./roadmaps/list_processing_roadmap.md) for details.

2. **Type Predicates**: 
   - Basic type predicates (pair?, null?, list?) are in the planning stage.
   - Implementation roadmap has been created.
   - See [type_predicates_roadmap.md](./roadmaps/type_predicates_roadmap.md) for details.

3. **Equality Predicates**:
   - Equality predicates (eq?, eqv?, equal?) are in the planning stage.
   - Implementation roadmap has been created.
   - See [equality_predicates_roadmap.md](./roadmaps/equality_predicates_roadmap.md) for details.

4. **Higher-Order Functions**:
   - Higher-order functions (map, for-each, filter, etc.) are in the planning stage.
   - Implementation roadmap has been created.
   - See [higher_order_functions_roadmap.md](./roadmaps/higher_order_functions_roadmap.md) for details.

5. **Complex Numbers**: Complex number support is planned for Phase 7.

6. **Rational Numbers**: Rational number support is planned for Phase 7.

7. **Library System**: The R7RS library system is not yet fully implemented.

8. **Dynamic FFI**: Dynamic foreign function interface is not yet implemented.

## Deferred Features

The following features are intentionally deferred to later phases:

1. **Full Continuations**: Full continuations support is deferred to Phase 5.
2. **Hygienic Macros**: Hygienic macro system is deferred to Phase 7.
3. **Full Numeric Tower**: Complete numeric tower is deferred to Phase 7.
4. **Advanced I/O**: Advanced I/O features are deferred to Phase 5.
5. **Module System**: Enhanced module system is deferred to Phase 7.

## MCP Tools Issues

The following issues affect the MCP tools used for Scheme compatibility analysis:

1. **Type Analysis Simplification**: The analyze-types tool uses a simplified implementation that doesn't fully parse the code.
2. **Binding Analysis Limitations**: The analyze-bindings and analyze-lambda-captures tools may not correctly handle complex binding patterns.
3. **Recursion Analysis Limitations**: The analyze-scheme-recursion and analyze-tscheme-recursion tools may not correctly handle all forms of mutual recursion.
4. **Integration Issues**: The MCP tools are not fully integrated with the compiler pipeline.

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

### Autodiff and Vector Type Issues

To work around the current issues with automatic differentiation and vector types:

1. Use explicit type annotations for functions that return vectors
   ```scheme
   (: sum-of-squares-gradient-forward (-> float float (Vector float)))
   (define (sum-of-squares-gradient-forward x y)
     (autodiff-forward-gradient sum-of-squares (vector x y)))
   ```

2. Avoid mixing scalar and vector operations in complex expressions
   ```scheme
   ;; Instead of this:
   (define result (+ (vector-ref gradient 0) (* 2 (vector-ref gradient 1))))
   
   ;; Do this:
   (define g0 (vector-ref gradient 0))
   (define g1 (vector-ref gradient 1))
   (define result (+ g0 (* 2 g1)))
   ```

3. Use intermediate variables with explicit types for autodiff results
   ```scheme
   (define gradient (autodiff-forward-gradient f input-vec))
   (: dx float)
   (define dx (vector-ref gradient 0))
   (: dy float)
   (define dy (vector-ref gradient 1))
   ```

4. For complex autodiff operations, consider breaking them down into simpler steps with explicit types

5. Use the MCP tools to analyze your code for potential issues:
   ```bash
   # Analyze type inference for an Eshkol file
   use_mcp_tool eshkol-tools analyze-types '{"filePath": "examples/autodiff_example.esk", "detail": "detailed"}'
   
   # Analyze lambda captures for an Eshkol file
   use_mcp_tool eshkol-tools analyze-lambda-captures '{"filePath": "examples/autodiff_example.esk", "detail": "detailed"}'
   ```

### Lambda Capture Issues

To work around issues with lambda captures in autodiff functions:

1. Avoid capturing mutable variables in lambdas passed to autodiff functions
2. Use explicit parameter passing instead of capturing variables
3. Break complex higher-order functions into simpler functions
4. Use the analyze-lambda-captures MCP tool to identify potential issues

### Numeric Tower Limitations

Until the full numeric tower is implemented, you can:

1. Use floating-point numbers for most calculations
2. Be aware of precision limitations
3. Implement your own rational or complex number types if needed

### MCP Tools Limitations

To work around limitations in the MCP tools:

1. Use the tools as guidance rather than relying on them for complete analysis
2. Verify tool results manually, especially for complex code
3. Use multiple tools to get a more complete picture
4. Report issues with the tools to help improve them

## Reporting Issues

If you encounter an issue with Scheme compatibility in Eshkol, please report it by:

1. Checking if the issue is already known (listed in this document)
2. Creating a new issue in the issue tracker with:
   - A minimal reproducible example
   - Expected behavior
   - Actual behavior
   - Eshkol version
   - Any relevant context
   - MCP tool analysis results, if applicable

## Planned Improvements

See the [Evolution Roadmap](./EVOLUTION.md) for planned improvements to Scheme compatibility in Eshkol.

## Example Files

The following example files demonstrate the use of Scheme features in Eshkol:

1. **Type Predicates**: [type_predicates.esk](../../examples/type_predicates.esk)
2. **Equality Predicates**: [equality_predicates.esk](../../examples/equality_predicates.esk)
3. **List Operations**: [list_operations.esk](../../examples/list_operations.esk)
4. **Function Composition**: [function_composition.esk](../../examples/function_composition.esk)
5. **Mutual Recursion**: [mutual_recursion.esk](../../examples/mutual_recursion.esk)

These example files provide a reference for how to use Scheme features in Eshkol, even if some of the features are not yet fully implemented.

## Revision History

| Date | Changes |
|------|---------|
| 2025-03-29 | Added new issues: equality predicates, higher-order functions, list processing functions (SCH-016, SCH-017, SCH-018) |
| 2025-03-29 | Added links to implementation roadmaps |
| 2025-03-29 | Added example files section |
| 2025-03-29 | Updated standard library limitations section |
| 2025-03-28 | Updated SCH-013: Core list operations implemented |
| 2025-03-28 | Comprehensive update with new issues and workarounds |
| 2025-03-28 | Added MCP tools issues and workarounds |
| 2025-03-28 | Added lambda capture issues (SCH-011) |
| 2025-03-28 | Added MCP tools issues (SCH-012) |
| 2025-03-28 | Added core list operations and type predicates issues (SCH-013, SCH-014) |
| 2025-03-28 | Added mutual recursion handling issue (SCH-015) |
| 2025-03-24 | Added autodiff and vector calculus type issues (SCH-006 to SCH-010) |
| 2025-03-24 | Added workarounds for autodiff and vector type issues |
| 2025-03-23 | Initial document created |
