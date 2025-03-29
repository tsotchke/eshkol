# Eshkol Scheme Compatibility - Implementation Plan

Last Updated: 2025-03-23

This document outlines the comprehensive plan for implementing Scheme language compatibility in Eshkol. The implementation is organized into phases, with each phase building on the previous one to ensure a systematic approach to achieving full Scheme compatibility.

## Implementation Priorities

Our implementation strategy follows these priority levels:

- **P0**: Critical core features required for basic Scheme functionality
- **P1**: Essential features needed for practical programming
- **P2**: Important features for comprehensive language support
- **P3**: Advanced features for complete language compatibility
- **P4**: Extended/specialized features

## Phase 1: Core Data Types and Fundamental Operations (Release 1.0)

### P0: Pairs and Lists
| Function | Description | Implementation Notes | Status |
|----------|-------------|---------------------|--------|
| `cons` | Create a pair | Core memory structure for lists | Implemented |
| `car` | Get first element of pair | Direct memory access | Implemented |
| `cdr` | Get second element of pair | Direct memory access | Implemented |
| `list` | Create a list from elements | Built on cons | Implemented |
| `pair?` | Test if object is a pair | Type checking | Implemented |
| `null?` | Test for empty list | Special case checking | Implemented |
| `list?` | Test if object is a proper list | Recursive checking | Implemented |

### P0: Basic Type Predicates
| Function | Description | Implementation Notes |
|----------|-------------|---------------------|
| `boolean?` | Test for boolean | Type checking |
| `symbol?` | Test for symbol | Type checking |
| `number?` | Test for number | Type checking |
| `string?` | Test for string | Type checking |
| `char?` | Test for character | Type checking |
| `procedure?` | Test for procedure | Type checking |
| `vector?` | Test for vector | Type checking |

### P0: Equality Predicates
| Function | Description | Implementation Notes |
|----------|-------------|---------------------|
| `eq?` | Identity equality | Memory address comparison |
| `eqv?` | Equivalence relation | Type-specific comparison |
| `equal?` | Recursive equivalence | Deep comparison |

### P0: Basic Arithmetic
| Function | Description | Implementation Notes |
|----------|-------------|---------------------|
| `+` | Addition | Support variable arguments |
| `-` | Subtraction | Support variable arguments |
| `*` | Multiplication | Support variable arguments |
| `/` | Division | Support variable arguments |
| `=` | Numeric equality | Support variable arguments |
| `<` | Less than | Support variable arguments |
| `>` | Greater than | Support variable arguments |
| `<=` | Less than or equal | Support variable arguments |
| `>=` | Greater than or equal | Support variable arguments |

### P1: Extended Pair Operations
| Function | Description | Implementation Notes | Status |
|----------|-------------|---------------------|--------|
| `set-car!` | Modify first element of pair | Mutation operation | Implemented |
| `set-cdr!` | Modify second element of pair | Mutation operation | Implemented |
| `caar`, `cadr`, etc. | Nested car/cdr operations | Up to 4 levels deep | Implemented |

## Phase 2: List Processing and Control Flow (Release 1.1)

### P1: List Processing
| Function | Description | Implementation Notes |
|----------|-------------|---------------------|
| `length` | Count list elements | Iterative implementation |
| `append` | Concatenate lists | Handle variable arguments |
| `reverse` | Reverse a list | Create new list |
| `list-ref` | Access by index | Error on out-of-bounds |
| `list-tail` | Get sublist | Error on out-of-bounds |
| `list-set!` | Set element at index | Mutation operation |
| `memq`, `memv`, `member` | Find element | Return sublist or #f |
| `assq`, `assv`, `assoc` | Association list lookup | Return pair or #f |

### P1: Control Flow
| Function | Description | Implementation Notes |
|----------|-------------|---------------------|
| `if` | Conditional (already implemented) | Core syntax |
| `cond` | Multi-way conditional | Macro on top of if |
| `case` | Dispatch on values | Macro on top of cond |
| `and`, `or` | Logical operations (already implemented) | Short-circuit evaluation |
| `not` | Logical negation | Boolean operation |
| `when`, `unless` | One-sided conditionals | Macros on top of if |

### P1: Advanced Numeric Operations
| Function | Description | Implementation Notes |
|----------|-------------|---------------------|
| `zero?`, `positive?`, `negative?` | Sign predicates | Numeric comparison |
| `odd?`, `even?` | Parity predicates | Numeric comparison |
| `max`, `min` | Extrema | Support variable arguments |
| `abs` | Absolute value | Sign handling |
| `quotient`, `remainder`, `modulo` | Integer division | Handle division by zero |
| `gcd`, `lcm` | Greatest common divisor, least common multiple | Euclidean algorithm |

## Phase 3: Higher-Order Functions and Data Structures (Release 1.2)

### P1: Higher-Order Functions
| Function | Description | Implementation Notes |
|----------|-------------|---------------------|
| `map` | Apply procedure to elements | Support multiple lists |
| `for-each` | Apply procedure for side effects | Support multiple lists |
| `apply` | Apply procedure to list of arguments | Handle proper argument spreading |
| `filter` | Select elements satisfying predicate | Create new list |
| `fold-left`, `fold-right` | Accumulate values | Handle empty list cases |

### P2: String Operations
| Function | Description | Implementation Notes |
|----------|-------------|---------------------|
| `string-length` | Get string length | Direct access |
| `string-ref` | Access character by index | Bounds checking |
| `string-set!` | Modify character at index | Mutation operation |
| `string=?`, `string<?`, etc. | String comparison | Lexicographic ordering |
| `substring` | Extract substring | Bounds checking |
| `string-append` | Concatenate strings | Handle variable arguments |
| `string->list`, `list->string` | Convert between string and list | Character handling |
| `string-copy`, `string-fill!` | Copy and fill operations | Mutation operations |

### P2: Character Operations
| Function | Description | Implementation Notes |
|----------|-------------|---------------------|
| `char=?`, `char<?`, etc. | Character comparison | ASCII/Unicode handling |
| `char-alphabetic?`, `char-numeric?`, etc. | Character classification | ASCII/Unicode handling |
| `char->integer`, `integer->char` | Character conversion | ASCII/Unicode handling |
| `char-upcase`, `char-downcase` | Case conversion | ASCII/Unicode handling |

### P2: Vector Operations
| Function | Description | Implementation Notes |
|----------|-------------|---------------------|
| `make-vector` | Create vector | Initialization options |
| `vector` | Create vector from elements | Variable arguments |
| `vector-length` | Get vector length | Direct access |
| `vector-ref` | Access element by index | Bounds checking |
| `vector-set!` | Modify element at index | Mutation operation |
| `vector->list`, `list->vector` | Convert between vector and list | Full conversion |
| `vector-fill!` | Fill vector with value | Mutation operation |

## Phase 4: I/O and System Interface (Release 1.3)

### P2: Basic I/O
| Function | Description | Implementation Notes |
|----------|-------------|---------------------|
| `display`, `write` | Output values | Different formatting rules |
| `newline` | Output newline | Convenience function |
| `read`, `read-char` | Input parsing | Handle EOF |
| `peek-char` | Look ahead at input | Non-consuming read |
| `current-input-port`, `current-output-port` | Get current ports | Global state |

### P2: File I/O
| Function | Description | Implementation Notes |
|----------|-------------|---------------------|
| `open-input-file`, `open-output-file` | Open file ports | Error handling |
| `close-input-port`, `close-output-port` | Close ports | Resource management |
| `with-input-from-file`, `with-output-to-file` | File operations with automatic closing | Dynamic binding |
| `call-with-input-file`, `call-with-output-file` | File operations with procedure | Resource management |

### P3: System Interface
| Function | Description | Implementation Notes |
|----------|-------------|---------------------|
| `load` | Load and evaluate file | File handling and evaluation |
| `eval` | Evaluate expression | Full evaluation cycle |
| `error` | Signal error | Error handling |
| `exit` | Exit program | System interaction |

## Phase 5: Advanced Features (Release 1.4+)

### P3: Continuations and Exceptions
| Function | Description | Implementation Notes |
|----------|-------------|---------------------|
| `call-with-current-continuation` (call/cc) | Capture continuation | Complex control flow |
| `dynamic-wind` | Protected dynamic context | Entry/exit guards |
| `with-exception-handler` | Exception handling | Error management |
| `raise`, `raise-continuable` | Signal exceptions | Error signaling |

### P3: Environments and Evaluation
| Function | Description | Implementation Notes |
|----------|-------------|---------------------|
| `environment` | Create environment | Namespace management |
| `scheme-report-environment` | Standard environment | R5RS compatibility |
| `null-environment` | Empty environment | Minimal environment |

### P3: Delayed Evaluation
| Function | Description | Implementation Notes |
|----------|-------------|---------------------|
| `delay`, `force` | Lazy evaluation | Thunk management |
| `promise?` | Test for promise | Type checking |
| `make-promise` | Create promise | Explicit promise creation |

### P4: Advanced Numeric Support
| Function | Description | Implementation Notes |
|----------|-------------|---------------------|
| Complex numbers | Support for complex arithmetic | Full numeric tower |
| Rational numbers | Support for rational arithmetic | Full numeric tower |
| `exact?`, `inexact?` | Exactness predicates | Numeric precision |
| `exact->inexact`, `inexact->exact` | Conversion functions | Precision conversion |

## Implementation Strategy

For each phase, we will follow this process:

1. **Analysis**: Review existing code to identify what's already implemented and what needs to be added
2. **Design**: Create detailed specifications for each function, including edge cases and error handling
3. **Implementation**: Develop the functions in priority order, ensuring they meet the specifications
4. **Testing**: Create comprehensive test suites for each function, including edge cases
5. **Documentation**: Update documentation to reflect new capabilities
6. **Integration**: Ensure new functions work with existing codebase
7. **Release**: Package and release the updated version

## Testing Strategy

Each function will be tested with:

1. **Unit tests**: Verify correct behavior for normal inputs
2. **Edge case tests**: Verify correct behavior for boundary conditions
3. **Error tests**: Verify appropriate error handling
4. **Integration tests**: Verify correct interaction with other functions
5. **Compliance tests**: Verify compatibility with Scheme standards

## Documentation Strategy

For each implemented function, we will provide:

1. **Function signature**: Parameters and return values
2. **Description**: Purpose and behavior
3. **Examples**: Usage examples
4. **Edge cases**: Behavior in special situations
5. **Errors**: Conditions that trigger errors
6. **Implementation notes**: Any Eshkol-specific details

## Roadmap Timeline

- **Release 1.0 (Phase 1)**: Core data types and fundamental operations - Q2 2025
- **Release 1.1 (Phase 2)**: List processing and control flow - Q3 2025
- **Release 1.2 (Phase 3)**: Higher-order functions and data structures - Q4 2025
- **Release 1.3 (Phase 4)**: I/O and system interface - Q1 2026
- **Release 1.4+ (Phase 5)**: Advanced features - Q2 2026 and beyond

## Dependencies Between Components

Understanding the dependencies between Scheme features is crucial for planning the implementation order:

1. **Core Memory Management**: Required for all data structures
2. **Pairs and Lists**: Foundation for most data structures and operations
3. **Type System**: Required for type predicates and operations
4. **Evaluation**: Required for control flow and higher-order functions
5. **I/O System**: Required for input/output operations
6. **Exception Handling**: Required for error management

## Compatibility Considerations

We will maintain compatibility with:

1. **R5RS**: The Revised⁵ Report on the Algorithmic Language Scheme
2. **R7RS-small**: The Revised⁷ Report on the Algorithmic Language Scheme (small language)

Where these standards differ, we will follow R7RS-small by default, but provide options for R5RS compatibility where feasible.

## Conclusion

This implementation plan provides a structured approach to achieving Scheme compatibility in Eshkol. By following this plan, we will systematically implement the Scheme language features, ensuring a solid foundation for future development.
