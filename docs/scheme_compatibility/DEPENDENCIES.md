# Eshkol Scheme Compatibility - Dependencies

Last Updated: 2025-03-23

This document maps the dependencies between Scheme features in Eshkol. Understanding these dependencies is crucial for planning the implementation order and ensuring that features are implemented in a logical sequence.

## Core Dependencies

```
Memory Management ← Pairs and Lists ← Most Other Features
Type System ← Type Predicates ← Type-Specific Operations
Evaluation ← Special Forms ← Macros
I/O System ← File Operations ← Loading and Evaluation
```

## Detailed Dependency Graph

### Memory Management

The memory management system is the foundation for all data structures and operations:

```
Memory Management
├── Arena Allocator
├── Object Pool
└── Memory Tracking
```

### Type System

The type system depends on memory management and provides the foundation for type predicates and operations:

```
Type System
├── Type Representation
├── Type Checking
└── Type Inference
```

### Pairs and Lists

Pairs and lists are the fundamental data structures in Scheme and depend on memory management:

```
Pairs and Lists
├── Memory Management
├── cons, car, cdr
├── list
└── list operations (length, append, etc.)
```

### Special Forms

Special forms provide the core control flow and binding mechanisms:

```
Special Forms
├── define
├── if
├── lambda
├── begin
├── quote
├── set!
└── let
```

### Derived Expressions

Derived expressions are built on top of special forms:

```
Derived Expressions
├── Special Forms
├── cond, case
├── and, or
└── when, unless
```

### Procedures

Procedures depend on the evaluation system and memory management:

```
Procedures
├── Memory Management
├── Evaluation
└── Lambda
```

### Higher-Order Functions

Higher-order functions depend on procedures and lists:

```
Higher-Order Functions
├── Procedures
├── Pairs and Lists
├── map, for-each
└── apply
```

### I/O System

The I/O system depends on memory management and provides the foundation for file operations:

```
I/O System
├── Memory Management
├── Ports
├── display, write, read
└── File Operations
```

### Advanced Features

Advanced features depend on multiple core components:

```
Advanced Features
├── Continuations
│   ├── Evaluation
│   └── Memory Management
├── Exceptions
│   ├── Continuations
│   └── I/O System
└── Delayed Evaluation
    ├── Procedures
    └── Memory Management
```

## Function-Level Dependencies

### Pairs and Lists

| Function | Dependencies |
|----------|--------------|
| `cons` | Memory Management |
| `car` | `cons` |
| `cdr` | `cons` |
| `list` | `cons` |
| `pair?` | Type System, `cons` |
| `null?` | Type System |
| `list?` | `pair?`, `null?` |
| `set-car!` | `cons`, `car` |
| `set-cdr!` | `cons`, `cdr` |
| `length` | `null?`, `cdr` |
| `append` | `null?`, `cons`, `car`, `cdr` |
| `reverse` | `null?`, `cons`, `car`, `cdr` |
| `list-ref` | `car`, `cdr` |
| `list-tail` | `cdr` |
| `memq`, `memv`, `member` | `null?`, `car`, `cdr`, `eq?`/`eqv?`/`equal?` |
| `assq`, `assv`, `assoc` | `null?`, `car`, `cdr`, `eq?`/`eqv?`/`equal?` |

### Type Predicates

| Function | Dependencies |
|----------|--------------|
| `boolean?` | Type System |
| `symbol?` | Type System |
| `number?` | Type System |
| `string?` | Type System |
| `char?` | Type System |
| `procedure?` | Type System |
| `vector?` | Type System |

### Equality Predicates

| Function | Dependencies |
|----------|--------------|
| `eq?` | Memory Management |
| `eqv?` | `eq?`, Type System |
| `equal?` | `eqv?`, Recursion |

### Numeric Operations

| Function | Dependencies |
|----------|--------------|
| `+`, `-`, `*`, `/` | Type System, Number Implementation |
| `=`, `<`, `>`, `<=`, `>=` | Type System, Number Implementation |
| `zero?`, `positive?`, `negative?` | `=`, `<`, `>` |
| `odd?`, `even?` | Remainder Operation |
| `max`, `min` | `<`, `>` |
| `abs` | `<`, `-` |
| `quotient`, `remainder`, `modulo` | `/` |
| `gcd`, `lcm` | `remainder` |

### Higher-Order Functions

| Function | Dependencies |
|----------|--------------|
| `map` | `null?`, `cons`, `car`, `cdr`, Procedure Application |
| `for-each` | `null?`, `car`, `cdr`, Procedure Application |
| `apply` | Procedure Application, `list->vector` (for vector arguments) |
| `filter` | `null?`, `cons`, `car`, `cdr`, Procedure Application |
| `fold-left`, `fold-right` | `null?`, `car`, `cdr`, Procedure Application |

### String Operations

| Function | Dependencies |
|----------|--------------|
| `string-length` | Type System, String Implementation |
| `string-ref` | Type System, String Implementation |
| `string-set!` | Type System, String Implementation |
| `string=?`, `string<?`, etc. | Type System, String Implementation |
| `substring` | Type System, String Implementation |
| `string-append` | Type System, String Implementation |
| `string->list`, `list->string` | `cons`, `null?`, String Implementation |

### Vector Operations

| Function | Dependencies |
|----------|--------------|
| `make-vector` | Type System, Vector Implementation |
| `vector` | Type System, Vector Implementation |
| `vector-length` | Type System, Vector Implementation |
| `vector-ref` | Type System, Vector Implementation |
| `vector-set!` | Type System, Vector Implementation |
| `vector->list`, `list->vector` | `cons`, `null?`, Vector Implementation |

### I/O Operations

| Function | Dependencies |
|----------|--------------|
| `display`, `write` | I/O System |
| `read`, `read-char` | I/O System, Parser |
| `open-input-file`, `open-output-file` | I/O System, File System |
| `close-input-port`, `close-output-port` | I/O System |

### System Interface

| Function | Dependencies |
|----------|--------------|
| `load` | I/O System, Evaluation |
| `eval` | Parser, Evaluation |
| `error` | I/O System |
| `exit` | System Interface |

### Advanced Features

| Feature | Dependencies |
|---------|--------------|
| `call/cc` | Evaluation, Memory Management |
| `dynamic-wind` | Evaluation, Procedures |
| `with-exception-handler` | Evaluation, Procedures |
| `delay`, `force` | Procedures, Memory Management |

## Implementation Order Considerations

Based on these dependencies, the following implementation order is recommended:

1. **Core Infrastructure**:
   - Memory management system
   - Type system
   - Basic evaluation

2. **Fundamental Data Types**:
   - Booleans, numbers, characters, strings, symbols
   - Pairs and lists (cons, car, cdr)
   - Type predicates

3. **Core Special Forms**:
   - define, if, lambda, begin, quote, set!

4. **Basic Operations**:
   - Equality predicates (eq?, eqv?, equal?)
   - Numeric operations (+, -, *, /, =, <, >, <=, >=)
   - List operations (list, pair?, null?, list?)

5. **Derived Expressions**:
   - cond, case, and, or, when, unless
   - let, let*, letrec

6. **Advanced List Operations**:
   - length, append, reverse
   - list-ref, list-tail
   - memq, memv, member
   - assq, assv, assoc

7. **Higher-Order Functions**:
   - map, for-each, apply
   - filter, fold-left, fold-right

8. **Data Structure Operations**:
   - String operations
   - Vector operations
   - Character operations

9. **I/O and System Interface**:
   - Basic I/O (display, write, read)
   - File I/O
   - System interface (load, eval, error, exit)

10. **Advanced Features**:
    - Continuations
    - Exceptions
    - Delayed evaluation

This order ensures that each feature is implemented after its dependencies, minimizing the need for placeholder implementations and reducing the risk of having to refactor code later.
