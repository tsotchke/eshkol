# Eshkol Scheme Compatibility - Registry

Last Updated: 2025-03-23

This registry tracks the implementation status of all Scheme functions, syntax, and features in Eshkol. It serves as the single source of truth for what is implemented and how it complies with the Scheme standards.

## Special Forms

| Form | Standard Reference | Status | Implementation File | Test File | Compatibility Notes |
|------|-------------------|--------|---------------------|-----------|---------------------|
| `define` | R7RS 4.1.2 | Implemented | src/frontend/parser/parser.c | tests/unit/test_parser.c | Supports both variable and function definitions |
| `if` | R7RS 4.1.5 | Implemented | src/frontend/parser/parser.c | tests/unit/test_parser.c | Full compliance |
| `lambda` | R7RS 4.1.4 | Implemented | src/frontend/parser/parser.c | tests/unit/test_parser.c | Full compliance |
| `begin` | R7RS 4.2.3 | Implemented | src/frontend/parser/parser.c | tests/unit/test_parser.c | Full compliance |
| `quote` | R7RS 4.1.2 | Implemented | src/frontend/parser/parser.c | tests/unit/test_parser.c | Full compliance |
| `set!` | R7RS 4.1.6 | Implemented | src/frontend/parser/parser.c | tests/unit/test_parser.c | Full compliance |
| `let` | R7RS 4.2.2 | Implemented | src/frontend/parser/parser.c | tests/unit/test_parser.c | Basic let only, no named let yet |
| `and` | R7RS 4.2.1 | Implemented | src/frontend/parser/parser.c | tests/unit/test_parser.c | Full compliance |
| `or` | R7RS 4.2.1 | Implemented | src/frontend/parser/parser.c | tests/unit/test_parser.c | Full compliance |
| `cond` | R7RS 4.2.1 | Planned | - | - | Planned for Phase 2 |
| `case` | R7RS 4.2.1 | Planned | - | - | Planned for Phase 2 |
| `when` | R7RS 4.2.1 | Planned | - | - | Planned for Phase 2 |
| `unless` | R7RS 4.2.1 | Planned | - | - | Planned for Phase 2 |
| `let*` | R7RS 4.2.2 | Planned | - | - | Planned for Phase 2 |
| `letrec` | R7RS 4.2.2 | Planned | - | - | Planned for Phase 2 |
| `do` | R7RS 4.2.4 | Planned | - | - | Planned for Phase 2 |

## Pairs and Lists

| Function | Standard Reference | Status | Implementation File | Test File | Compatibility Notes |
|----------|-------------------|--------|---------------------|-----------|---------------------|
| `cons` | R7RS 6.4 | Planned | - | - | Planned for Phase 1 |
| `car` | R7RS 6.4 | Planned | - | - | Planned for Phase 1 |
| `cdr` | R7RS 6.4 | Planned | - | - | Planned for Phase 1 |
| `list` | R7RS 6.4 | Planned | - | - | Planned for Phase 1 |
| `pair?` | R7RS 6.4 | Planned | - | - | Planned for Phase 1 |
| `null?` | R7RS 6.4 | Planned | - | - | Planned for Phase 1 |
| `list?` | R7RS 6.4 | Planned | - | - | Planned for Phase 1 |
| `set-car!` | R7RS 6.4 | Planned | - | - | Planned for Phase 1 |
| `set-cdr!` | R7RS 6.4 | Planned | - | - | Planned for Phase 1 |
| `caar`, `cadr`, etc. | R7RS 6.4 | Planned | - | - | Planned for Phase 1 |
| `length` | R7RS 6.4 | Planned | - | - | Planned for Phase 2 |
| `append` | R7RS 6.4 | Planned | - | - | Planned for Phase 2 |
| `reverse` | R7RS 6.4 | Planned | - | - | Planned for Phase 2 |
| `list-ref` | R7RS 6.4 | Planned | - | - | Planned for Phase 2 |
| `list-tail` | R7RS 6.4 | Planned | - | - | Planned for Phase 2 |
| `memq`, `memv`, `member` | R7RS 6.4 | Planned | - | - | Planned for Phase 2 |
| `assq`, `assv`, `assoc` | R7RS 6.4 | Planned | - | - | Planned for Phase 2 |

## Type Predicates

| Function | Standard Reference | Status | Implementation File | Test File | Compatibility Notes |
|----------|-------------------|--------|---------------------|-----------|---------------------|
| `boolean?` | R7RS 6.3 | Planned | - | - | Planned for Phase 1 |
| `symbol?` | R7RS 6.3 | Planned | - | - | Planned for Phase 1 |
| `number?` | R7RS 6.3 | Planned | - | - | Planned for Phase 1 |
| `string?` | R7RS 6.3 | Planned | - | - | Planned for Phase 1 |
| `char?` | R7RS 6.3 | Planned | - | - | Planned for Phase 1 |
| `procedure?` | R7RS 6.3 | Planned | - | - | Planned for Phase 1 |
| `vector?` | R7RS 6.3 | Planned | - | - | Planned for Phase 1 |

## Equality Predicates

| Function | Standard Reference | Status | Implementation File | Test File | Compatibility Notes |
|----------|-------------------|--------|---------------------|-----------|---------------------|
| `eq?` | R7RS 6.1 | Planned | - | - | Planned for Phase 1 |
| `eqv?` | R7RS 6.1 | Planned | - | - | Planned for Phase 1 |
| `equal?` | R7RS 6.1 | Planned | - | - | Planned for Phase 1 |

## Numeric Operations

| Function | Standard Reference | Status | Implementation File | Test File | Compatibility Notes |
|----------|-------------------|--------|---------------------|-----------|---------------------|
| `+` | R7RS 6.2.6 | Planned | - | - | Planned for Phase 1 |
| `-` | R7RS 6.2.6 | Planned | - | - | Planned for Phase 1 |
| `*` | R7RS 6.2.6 | Planned | - | - | Planned for Phase 1 |
| `/` | R7RS 6.2.6 | Planned | - | - | Planned for Phase 1 |
| `=` | R7RS 6.2.6 | Planned | - | - | Planned for Phase 1 |
| `<` | R7RS 6.2.6 | Planned | - | - | Planned for Phase 1 |
| `>` | R7RS 6.2.6 | Planned | - | - | Planned for Phase 1 |
| `<=` | R7RS 6.2.6 | Planned | - | - | Planned for Phase 1 |
| `>=` | R7RS 6.2.6 | Planned | - | - | Planned for Phase 1 |
| `zero?` | R7RS 6.2.6 | Planned | - | - | Planned for Phase 2 |
| `positive?` | R7RS 6.2.6 | Planned | - | - | Planned for Phase 2 |
| `negative?` | R7RS 6.2.6 | Planned | - | - | Planned for Phase 2 |
| `odd?` | R7RS 6.2.6 | Planned | - | - | Planned for Phase 2 |
| `even?` | R7RS 6.2.6 | Planned | - | - | Planned for Phase 2 |
| `max` | R7RS 6.2.6 | Planned | - | - | Planned for Phase 2 |
| `min` | R7RS 6.2.6 | Planned | - | - | Planned for Phase 2 |
| `abs` | R7RS 6.2.6 | Planned | - | - | Planned for Phase 2 |
| `quotient` | R7RS 6.2.6 | Planned | - | - | Planned for Phase 2 |
| `remainder` | R7RS 6.2.6 | Planned | - | - | Planned for Phase 2 |
| `modulo` | R7RS 6.2.6 | Planned | - | - | Planned for Phase 2 |
| `gcd` | R7RS 6.2.6 | Planned | - | - | Planned for Phase 2 |
| `lcm` | R7RS 6.2.6 | Planned | - | - | Planned for Phase 2 |

## Higher-Order Functions

| Function | Standard Reference | Status | Implementation File | Test File | Compatibility Notes |
|----------|-------------------|--------|---------------------|-----------|---------------------|
| `map` | R7RS 6.4 | Planned | - | - | Planned for Phase 3 |
| `for-each` | R7RS 6.4 | Planned | - | - | Planned for Phase 3 |
| `apply` | R7RS 6.4 | Planned | - | - | Planned for Phase 3 |
| `filter` | SRFI-1 | Planned | - | - | Planned for Phase 3 |
| `fold-left` | SRFI-1 | Planned | - | - | Planned for Phase 3 |
| `fold-right` | SRFI-1 | Planned | - | - | Planned for Phase 3 |

## String Operations

| Function | Standard Reference | Status | Implementation File | Test File | Compatibility Notes |
|----------|-------------------|--------|---------------------|-----------|---------------------|
| `string-length` | R7RS 6.7 | Planned | - | - | Planned for Phase 3 |
| `string-ref` | R7RS 6.7 | Planned | - | - | Planned for Phase 3 |
| `string-set!` | R7RS 6.7 | Planned | - | - | Planned for Phase 3 |
| `string=?` | R7RS 6.7 | Planned | - | - | Planned for Phase 3 |
| `string<?` | R7RS 6.7 | Planned | - | - | Planned for Phase 3 |
| `string>?` | R7RS 6.7 | Planned | - | - | Planned for Phase 3 |
| `string<=?` | R7RS 6.7 | Planned | - | - | Planned for Phase 3 |
| `string>=?` | R7RS 6.7 | Planned | - | - | Planned for Phase 3 |
| `substring` | R7RS 6.7 | Planned | - | - | Planned for Phase 3 |
| `string-append` | R7RS 6.7 | Planned | - | - | Planned for Phase 3 |
| `string->list` | R7RS 6.7 | Planned | - | - | Planned for Phase 3 |
| `list->string` | R7RS 6.7 | Planned | - | - | Planned for Phase 3 |
| `string-copy` | R7RS 6.7 | Planned | - | - | Planned for Phase 3 |
| `string-fill!` | R7RS 6.7 | Planned | - | - | Planned for Phase 3 |

## Character Operations

| Function | Standard Reference | Status | Implementation File | Test File | Compatibility Notes |
|----------|-------------------|--------|---------------------|-----------|---------------------|
| `char=?` | R7RS 6.6 | Planned | - | - | Planned for Phase 3 |
| `char<?` | R7RS 6.6 | Planned | - | - | Planned for Phase 3 |
| `char>?` | R7RS 6.6 | Planned | - | - | Planned for Phase 3 |
| `char<=?` | R7RS 6.6 | Planned | - | - | Planned for Phase 3 |
| `char>=?` | R7RS 6.6 | Planned | - | - | Planned for Phase 3 |
| `char-alphabetic?` | R7RS 6.6 | Planned | - | - | Planned for Phase 3 |
| `char-numeric?` | R7RS 6.6 | Planned | - | - | Planned for Phase 3 |
| `char-whitespace?` | R7RS 6.6 | Planned | - | - | Planned for Phase 3 |
| `char-upper-case?` | R7RS 6.6 | Planned | - | - | Planned for Phase 3 |
| `char-lower-case?` | R7RS 6.6 | Planned | - | - | Planned for Phase 3 |
| `char->integer` | R7RS 6.6 | Planned | - | - | Planned for Phase 3 |
| `integer->char` | R7RS 6.6 | Planned | - | - | Planned for Phase 3 |
| `char-upcase` | R7RS 6.6 | Planned | - | - | Planned for Phase 3 |
| `char-downcase` | R7RS 6.6 | Planned | - | - | Planned for Phase 3 |

## Vector Operations

| Function | Standard Reference | Status | Implementation File | Test File | Compatibility Notes |
|----------|-------------------|--------|---------------------|-----------|---------------------|
| `make-vector` | R7RS 6.8 | Planned | - | - | Planned for Phase 3 |
| `vector` | R7RS 6.8 | Planned | - | - | Planned for Phase 3 |
| `vector-length` | R7RS 6.8 | Planned | - | - | Planned for Phase 3 |
| `vector-ref` | R7RS 6.8 | Planned | - | - | Planned for Phase 3 |
| `vector-set!` | R7RS 6.8 | Planned | - | - | Planned for Phase 3 |
| `vector->list` | R7RS 6.8 | Planned | - | - | Planned for Phase 3 |
| `list->vector` | R7RS 6.8 | Planned | - | - | Planned for Phase 3 |
| `vector-fill!` | R7RS 6.8 | Planned | - | - | Planned for Phase 3 |

## I/O Operations

| Function | Standard Reference | Status | Implementation File | Test File | Compatibility Notes |
|----------|-------------------|--------|---------------------|-----------|---------------------|
| `display` | R7RS 6.13.3 | Planned | - | - | Planned for Phase 4 |
| `write` | R7RS 6.13.3 | Planned | - | - | Planned for Phase 4 |
| `newline` | R7RS 6.13.3 | Planned | - | - | Planned for Phase 4 |
| `read` | R7RS 6.13.2 | Planned | - | - | Planned for Phase 4 |
| `read-char` | R7RS 6.13.2 | Planned | - | - | Planned for Phase 4 |
| `peek-char` | R7RS 6.13.2 | Planned | - | - | Planned for Phase 4 |
| `open-input-file` | R7RS 6.13.1 | Planned | - | - | Planned for Phase 4 |
| `open-output-file` | R7RS 6.13.1 | Planned | - | - | Planned for Phase 4 |
| `close-input-port` | R7RS 6.13.1 | Planned | - | - | Planned for Phase 4 |
| `close-output-port` | R7RS 6.13.1 | Planned | - | - | Planned for Phase 4 |

## System Interface

| Function | Standard Reference | Status | Implementation File | Test File | Compatibility Notes |
|----------|-------------------|--------|---------------------|-----------|---------------------|
| `load` | R7RS 6.13.1 | Planned | - | - | Planned for Phase 4 |
| `eval` | R7RS 6.12 | Planned | - | - | Planned for Phase 4 |
| `error` | R7RS 6.11 | Planned | - | - | Planned for Phase 4 |
| `exit` | R7RS 6.11 | Planned | - | - | Planned for Phase 4 |

## Advanced Features

| Feature | Standard Reference | Status | Implementation File | Test File | Compatibility Notes |
|---------|-------------------|--------|---------------------|-----------|---------------------|
| `call/cc` | R7RS 6.10 | Planned | - | - | Planned for Phase 5 |
| `dynamic-wind` | R7RS 6.10 | Planned | - | - | Planned for Phase 5 |
| `with-exception-handler` | R7RS 6.11 | Planned | - | - | Planned for Phase 5 |
| `raise` | R7RS 6.11 | Planned | - | - | Planned for Phase 5 |
| `delay` | R7RS 6.9 | Planned | - | - | Planned for Phase 5 |
| `force` | R7RS 6.9 | Planned | - | - | Planned for Phase 5 |

## Eshkol-Specific Extensions

| Feature | Description | Status | Implementation File | Test File | Notes |
|---------|------------|--------|---------------------|-----------|-------|
| Type Annotations | Optional static type annotations | Implemented | src/frontend/parser/type_parser.c | tests/unit/test_type.c | Eshkol-specific extension |
| Vector Calculus | Vector calculus operations | Implemented | src/core/utils/vector.c | tests/unit/test_vector.c | Eshkol-specific extension |
| Automatic Differentiation | Automatic differentiation of functions | Implemented | src/core/utils/autodiff.c | tests/unit/test_autodiff.c | Eshkol-specific extension |
| SIMD Optimization | Automatic SIMD optimization | Implemented | src/core/utils/simd.c | tests/unit/test_simd.c | Eshkol-specific extension |

## Implementation History

| Date | Version | Changes |
|------|---------|---------|
| 2025-03-23 | Initial | Created registry with current implementation status |
