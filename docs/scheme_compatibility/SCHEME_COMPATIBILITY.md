# Scheme Compatibility in Eshkol

This document outlines the Scheme compatibility features in Eshkol, focusing on the core language features and standard library functions that are supported.

## Core Language Features

Eshkol supports the following core Scheme language features:

- **Variables and Definitions**: `define` for defining variables and functions
- **Conditionals**: `if`, `cond`, `and`, `or`, `not`
- **Lexical Scoping**: `let`, `let*`, `letrec`
- **Procedures**: Lambda expressions, procedure application
- **Recursion**: Both direct and mutual recursion
- **Arithmetic Operations**: `+`, `-`, `*`, `/`, `=`, `<`, `>`, `<=`, `>=`
- **Type Predicates**: `number?`, `boolean?`, `string?`, `symbol?`, `pair?`, `null?`, `procedure?`
- **I/O Operations**: `display`, `newline`, `read`

## Data Types

Eshkol supports the following Scheme data types:

- **Numbers**: Integers and floating-point numbers
- **Booleans**: `#t` and `#f`
- **Characters**: Character literals
- **Strings**: String literals
- **Symbols**: Symbol literals
- **Pairs and Lists**: Cons cells, proper and improper lists
- **Vectors**: Vector literals and operations

## Standard Library Functions

### Numeric Functions

- `+`, `-`, `*`, `/`: Basic arithmetic operations
- `=`, `<`, `>`, `<=`, `>=`: Numeric comparisons
- `abs`, `quotient`, `remainder`, `modulo`: Integer operations
- `floor`, `ceiling`, `truncate`, `round`: Rounding operations
- `exp`, `log`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`: Transcendental functions
- `sqrt`, `expt`: Power operations

### List Operations

- `cons`, `car`, `cdr`: Basic list operations
- `list`: List constructor
- `length`: List length
- `append`: List concatenation
- `reverse`: List reversal
- `list-ref`: List element access
- `map`, `for-each`: List iteration

### Vector Operations

- `vector`: Vector constructor
- `vector-length`: Vector length
- `vector-ref`: Vector element access
- `vector-set!`: Vector element mutation
- `vector->list`, `list->vector`: Conversion between vectors and lists

### String Operations

- `string`: String constructor
- `string-length`: String length
- `string-ref`: String element access
- `string-append`: String concatenation
- `string->number`, `number->string`: Conversion between strings and numbers

### Symbol Operations

- `symbol->string`, `string->symbol`: Conversion between symbols and strings

### Control Flow

- `if`, `cond`, `case`: Conditional expressions
- `and`, `or`, `not`: Logical operations
- `begin`: Sequence of expressions

### Procedures

- `lambda`: Procedure constructor
- `apply`: Procedure application
- `procedure?`: Procedure predicate

## Extended Features

In addition to standard Scheme features, Eshkol provides the following extensions:

### Vector Calculus

- `v+`, `v-`, `v*`: Vector arithmetic operations
- `dot`: Dot product
- `cross`: Cross product
- `norm`: Vector magnitude
- `gradient`: Gradient of a scalar field
- `divergence`: Divergence of a vector field
- `curl`: Curl of a vector field
- `laplacian`: Laplacian of a scalar field

### Automatic Differentiation

- `autodiff-forward`: Forward-mode automatic differentiation
- `autodiff-reverse`: Reverse-mode automatic differentiation
- `autodiff-forward-gradient`: Forward-mode gradient computation
- `autodiff-reverse-gradient`: Reverse-mode gradient computation
- `autodiff-jacobian`: Jacobian matrix computation
- `autodiff-hessian`: Hessian matrix computation
- `derivative`: Derivative of a function at a point

## Examples

### Basic Scheme Examples

```scheme
;; Define a function
(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

;; Use let for local variables
(define (sum-of-squares x y)
  (let ((x-squared (* x x))
        (y-squared (* y y)))
    (+ x-squared y-squared)))

;; Use recursion with lists
(define (sum-list lst)
  (if (null? lst)
      0
      (+ (car lst) (sum-list (cdr lst)))))
```

### Vector Calculus Examples

```scheme
;; Define vectors
(define v1 (vector 1.0 2.0 3.0))
(define v2 (vector 4.0 5.0 6.0))

;; Vector operations
(define v-sum (v+ v1 v2))        ;; [5.0, 7.0, 9.0]
(define v-diff (v- v1 v2))       ;; [-3.0, -3.0, -3.0]
(define dot-prod (dot v1 v2))    ;; 32.0
(define cross-prod (cross v1 v2)) ;; [-3.0, 6.0, -3.0]
(define v1-norm (norm v1))       ;; 3.74

;; Define a scalar field
(define (f v)
  (let ((x (vector-ref v 0))
        (y (vector-ref v 1))
        (z (vector-ref v 2)))
    (+ (* x x) (* y y) (* z z))))

;; Compute gradient
(define grad-f (gradient f v1))  ;; [2.0, 4.0, 6.0]
```

### Automatic Differentiation Examples

```scheme
;; Define a function
(define (g x)
  (* x x x))  ;; g(x) = x^3

;; Compute derivative at x=2.0
(define dg/dx (derivative g 2.0))  ;; 12.0

;; Define a multivariate function
(define (h v)
  (let ((x (vector-ref v 0))
        (y (vector-ref v 1)))
    (+ (* x x y) (* y y))))  ;; h(x,y) = x^2*y + y^2

;; Compute gradient at [1.0, 2.0]
(define grad-h (gradient h (vector 1.0 2.0)))  ;; [4.0, 5.0]
```

## Known Limitations

- The type system is not fully implemented in this release
- Tail call optimization is not guaranteed
- Continuations are not supported
- Macros are not supported
- The full numeric tower (complex numbers, rational numbers) is not implemented
