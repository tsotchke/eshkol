# Eshkol Language Specification

**Version 1.0.0 | May 20, 2025**

---

## Table of Contents

1. [Introduction](#1-introduction)
   1. [Purpose of This Document](#11-purpose-of-this-document)
   2. [Relationship to Scheme](#12-relationship-to-scheme)
   3. [Design Philosophy](#13-design-philosophy)

2. [Scheme Compatibility](#2-scheme-compatibility)
   1. [Base Standards](#21-base-standards)
   2. [Compliance Level](#22-compliance-level)
   3. [Known Deviations](#23-known-deviations)

3. [Lexical Structure](#3-lexical-structure)
   1. [Character Set](#31-character-set)
   2. [Comments](#32-comments)
   3. [Identifiers](#33-identifiers)
   4. [Keywords](#34-keywords)
   5. [Literals](#35-literals)
   6. [Eshkol-specific Extensions](#36-eshkol-specific-extensions)

4. [Type System](#4-type-system)
   1. [Design Principles](#41-design-principles)
   2. [Base Types](#42-base-types)
   3. [Type Annotations](#43-type-annotations)
      1. [Implicit Typing](#431-implicit-typing)
      2. [Inline Explicit Typing](#432-inline-explicit-typing)
      3. [Separate Type Declarations](#433-separate-type-declarations)
   4. [Complex Types](#44-complex-types)
      1. [Function Types](#441-function-types)
      2. [Parametric Types](#442-parametric-types)
      3. [Vector and Matrix Types](#443-vector-and-matrix-types)
      4. [Tensor Types](#444-tensor-types)
   5. [Type Classes and Constraints](#45-type-classes-and-constraints)
   6. [Gradual Typing](#46-gradual-typing)
   7. [Type Inference](#47-type-inference)

5. [Core Language Features](#5-core-language-features)
   1. [Expressions](#51-expressions)
   2. [Special Forms](#52-special-forms)
   3. [Binding Constructs](#53-binding-constructs)
   4. [Control Flow](#54-control-flow)
   5. [Eshkol-specific Extensions](#55-eshkol-specific-extensions)

6. [Functions and Closures](#6-functions-and-closures)
   1. [Function Definition](#61-function-definition)
   2. [Lambda Expressions](#62-lambda-expressions)
   3. [Function Application](#63-function-application)
   4. [Higher-Order Functions](#64-higher-order-functions)
   5. [Function Composition](#65-function-composition)
   6. [Mutual Recursion](#66-mutual-recursion)

7. [Scientific Computing Features](#7-scientific-computing-features)
   1. [Vector Operations](#71-vector-operations)
   2. [Matrix Operations](#72-matrix-operations)
   3. [Tensor Operations](#73-tensor-operations)
   4. [Numerical Algorithms](#74-numerical-algorithms)
   5. [SIMD Optimization](#75-simd-optimization)

8. [Automatic Differentiation](#8-automatic-differentiation)
   1. [Forward Mode](#81-forward-mode)
   2. [Reverse Mode](#82-reverse-mode)
   3. [Higher-Order Derivatives](#83-higher-order-derivatives)
   4. [Vector and Matrix Derivatives](#84-vector-and-matrix-derivatives)
   5. [Integration with Type System](#85-integration-with-type-system)

9. [Stochastic Lambda Calculus](#9-stochastic-lambda-calculus)
   1. [Probabilistic Functions](#91-probabilistic-functions)
   2. [Sampling Operations](#92-sampling-operations)
   3. [Distribution Types](#93-distribution-types)
   4. [Inference Algorithms](#94-inference-algorithms)

10. [Quantum Computing Extensions](#10-quantum-computing-extensions)
    1. [Quantum Types](#101-quantum-types)
    2. [Quantum Operations](#102-quantum-operations)
    3. [Measurement](#103-measurement)
    4. [Quantum Control Flow](#104-quantum-control-flow)

11. [Memory Management](#11-memory-management)
    1. [Model Overview](#111-model-overview)
    2. [Resource Management](#112-resource-management)
    3. [Effects on Programming Style](#113-effects-on-programming-style)

12. [Macro System](#12-macro-system)
    1. [Hygiene](#121-hygiene)
    2. [Syntax-Rules](#122-syntax-rules)
    3. [Syntax-Case](#123-syntax-case)
    4. [Procedural Macros](#124-procedural-macros)
    5. [Eshkol-specific Extensions](#125-eshkol-specific-extensions)

13. [Module System](#13-module-system)
    1. [Module Definitions](#131-module-definitions)
    2. [Imports and Exports](#132-imports-and-exports)
    3. [Namespace Management](#133-namespace-management)

14. [Standard Library](#14-standard-library)
    1. [Core Procedures](#141-core-procedures)
    2. [Scientific Computing Modules](#142-scientific-computing-modules)
    3. [Automatic Differentiation Modules](#143-automatic-differentiation-modules)
    4. [Stochastic Computation Modules](#144-stochastic-computation-modules)
    5. [Quantum Computing Modules](#145-quantum-computing-modules)

15. [Appendices](#15-appendices)
    1. [Grammar (EBNF)](#151-grammar-ebnf)
    2. [Standard Library Reference](#152-standard-library-reference)
    3. [Type System Reference](#153-type-system-reference)
    4. [Example Programs](#154-example-programs)

---

## 1. Introduction

### 1.1 Purpose of This Document

This specification defines the complete Eshkol programming language. It is intended for language implementers, tool developers, and advanced Eshkol programmers. The document serves as the definitive reference for:

- The syntax and semantics of all Eshkol language constructs
- The type system and its rules
- The scientific computing, automatic differentiation, and other advanced features
- The standard library interfaces
- The extensions beyond standard Scheme

Language implementers should be able to build a complete Eshkol implementation based solely on this specification.

### 1.2 Relationship to Scheme

Eshkol is fundamentally an extension of the Scheme programming language, specifically targeting R5RS and R7RS-small compliance. Where not otherwise specified, Eshkol follows the semantics of R7RS-small Scheme. This specification focuses primarily on the ways Eshkol extends and diverges from standard Scheme.

The key principle governing Eshkol's relationship to Scheme is "conservative extension" - all valid standard Scheme programs should also be valid Eshkol programs with identical semantics. Extensions are introduced in ways that avoid breaking existing Scheme code.

### 1.3 Design Philosophy

Eshkol's design is guided by several core principles:

1. **Scientific Computing Focus**: Eshkol extends Scheme to excel in scientific computing, numerical analysis, and artificial intelligence applications.

2. **Gradual Typing**: The type system is optional and gradual, allowing programmers to add type information incrementally without disrupting existing code.

3. **Performance Without Sacrifice**: Performance optimizations are introduced without sacrificing the clarity and expressiveness of Scheme.

4. **First-class Automatic Differentiation**: Gradient computation is a fundamental language feature, not just a library.

5. **Scheme Compatibility**: Wherever possible, Eshkol maintains compatibility with standard Scheme to leverage the existing ecosystem.

## 2. Scheme Compatibility

### 2.1 Base Standards

Eshkol is based on the following Scheme standards:

- **R5RS**: The Revised⁵ Report on the Algorithmic Language Scheme
- **R7RS-small**: The Revised⁷ Report on the Algorithmic Language Scheme (small language)

Where these standards differ, Eshkol follows R7RS-small by default but provides options for R5RS compatibility where feasible.

### 2.2 Compliance Level

Eshkol aims for full compliance with R7RS-small, with the following implementation status:

- **Core Language Features**: Complete implementation
- **Standard Procedures**: Mostly implemented, with ongoing work
- **Input/Output**: Partially implemented
- **Standard Library**: Partially implemented

See the [Implementation Plan](https://github.com/tsotchke/eshkol/blob/master/docs/scheme_compatibility/IMPLEMENTATION_PLAN.md) for details on the current implementation status.

### 2.3 Known Deviations

While Eshkol strives for Scheme compatibility, there are several intentional deviations:

1. **Type System**: Eshkol adds an optional type system with static type checking.

2. **Memory Management**: Eshkol uses a hybrid memory management approach rather than traditional garbage collection.

3. **Scientific Computing Extensions**: Eshkol adds vector and matrix operations not found in standard Scheme.

4. **Automatic Differentiation**: Eshkol includes built-in support for automatic differentiation.

5. **Compilation Model**: Eshkol compiles to optimized machine code via LLVM rather than being interpreted.

These deviations are designed to enhance performance and capabilities without breaking compatibility with standard Scheme.

## 3. Lexical Structure

### 3.1 Character Set

Eshkol uses Unicode (UTF-8) for all source code, supporting the full range of Unicode characters in identifiers, strings, and comments. This is consistent with R7RS-small but a deviation from R5RS.

### 3.2 Comments

Eshkol supports all standard Scheme comment forms:

- **Line comments**: Begin with a semicolon (`;`) and continue to the end of the line.

  ```scheme
  ; This is a line comment
  (define x 5) ; This is an end-of-line comment
  ```

- **Block comments**: Enclosed between `#|` and `|#` and can span multiple lines.

  ```scheme
  #| This is a block comment
     that spans multiple lines |#
  ```

- **Datum comments**: Begin with `#;` and cause the following datum (expression) to be ignored.

  ```scheme
  (+ 1 #;(complex-calculation) 2)  ; The complex-calculation is ignored
  ```

### 3.3 Identifiers

Identifiers in Eshkol follow R7RS-small rules:

- Case-sensitive
- May consist of letters, digits, and special characters (`! $ % & * + - . / : < = > ? @ ^ _ ~`)
- May not begin with a digit
- May include Unicode characters
- Cannot be a keyword

Examples of valid identifiers:

````scheme
x
calculate-result
make-vector!
string->number
λ-calculus
````

### 3.4 Keywords

Eshkol includes all standard Scheme keywords, plus additional keywords for its extensions:

**Standard Scheme Keywords**:

- `define`, `lambda`, `if`, `cond`, `case`, `let`, `let*`, `letrec`, `set!`, `begin`, `quote`, `quasiquote`, `unquote`, `unquote-splicing`, `and`, `or`, `when`, `unless`

**Eshkol Type System Keywords**:

- `:` (type annotation marker)
- `->` (function type constructor)
- `forall` (universal quantifier for parametric types)
- `where` (constraint introducer for type classes)
- `define-type` (type definition)
- `define-type-class` (type class definition)
- `define-type-class-instance` (type class instance definition)

**Eshkol Scientific Computing Keywords**:

- `vector-add`, `vector-multiply`, `vector-scale`, `vector-dot`, etc.
- `matrix-add`, `matrix-multiply`, `matrix-transpose`, etc.
- `tensor-contract`, `tensor-product`, etc.

**Eshkol Automatic Differentiation Keywords**:

- `autodiff-gradient`, `autodiff-jacobian`, `autodiff-hessian`, etc.
- `forward-mode`, `reverse-mode`

### 3.5 Literals

Eshkol supports all standard Scheme literals, plus extensions for scientific computing:

**Boolean Literals**:

- `#t` or `#true`: Boolean true
- `#f` or `#false`: Boolean false

**Number Literals**:

- Integers: `42`, `-7`
- Floating-point: `3.14159`, `-0.5`, `1.0e-10`
- Radix notation: `#b1010` (binary), `#o777` (octal), `#x1a2b` (hexadecimal)
- Exactness: `#e3.14` (exact), `#i42` (inexact)

**Character Literals**:

- Basic characters: `#\a`, `#\Z`, `#\0`
- Named characters: `#\space`, `#\newline`, `#\tab`
- Unicode characters: `#\λ`, `#\u03BB` (Unicode code point)

**String Literals**:

- Basic strings: `"Hello, world!"`
- Escape sequences: `"Line 1\nLine 2"`, `"Tab\tIndented"`
- Unicode characters: `"Unicode λ character"`

**Symbol Literals**:

- Quoted symbols: `'symbol`, `'x`
- Keywords: `'if`, `'define`

**List Literals**:

- Quoted lists: `'(1 2 3)`, `'(a b c)`
- Dotted pairs: `'(1 . 2)`, `'(a . b)`
- Empty list: `'()` or `nil`

**Vector Literals**:

- Basic vectors: `#(1 2 3)`, `#(a b c)`
- Empty vector: `#()`

**Eshkol-specific Vector Literals**:

- Typed vectors: `#f64(1.0 2.0 3.0)` (64-bit float vector)
- Typed matrices: `#f64matrix((1.0 2.0) (3.0 4.0))` (64-bit float matrix)

### 3.6 Eshkol-specific Extensions

Eshkol extends Scheme's lexical structure with the following features:

**Type Annotations**:

```scheme
(: identifier type)
(define (function-name param1 : type1 param2 : type2) : return-type
  body)
```

**Vector and Matrix Literals**:

```
#f32vec(1.0 2.0 3.0)         ; 32-bit float vector
#f64vec(1.0 2.0 3.0)         ; 64-bit float vector
#i32vec(1 2 3)               ; 32-bit integer vector
#f32mat((1.0 2.0) (3.0 4.0)) ; 32-bit float matrix
```

**Tensor Literals**:

```scheme
#tensor(shape: (2 2 2), data: (((1 2) (3 4)) ((5 6) (7 8))))
```

## 4. Type System

Eshkol's type system is one of its most significant extensions to Scheme. It provides optional static typing with inference, allowing programmers to add type annotations incrementally while maintaining compatibility with untyped Scheme code.

### 4.1 Design Principles

The type system is guided by several principles:

1. **Gradual Typing**: The type system is optional. Code can be fully typed, partially typed, or completely untyped.

2. **Type Inference**: Eshkol employs sophisticated type inference to minimize the need for explicit annotations.

3. **Performance Optimization**: Types enable performance optimizations without sacrificing Scheme's semantics.

4. **Safety Guarantees**: Types prevent common errors like type mismatches and enable better static analysis.

5. **Scientific Computing Support**: The type system includes specialized types for scientific computing, such as vectors and matrices with known dimensions.

### 4.2 Base Types

Eshkol provides the following base types, which correspond to Scheme's primitive types:

| Type      | Description                      | Examples                  |
|-----------|----------------------------------|---------------------------|
| `boolean` | Boolean values                   | `#t`, `#f`                |
| `symbol`  | Symbolic identifiers             | `'a`, `'hello`            |
| `char`    | Unicode characters               | `#\a`, `#\space`          |
| `string`  | Unicode strings                  | `"hello"`, `"λ"`          |
| `number`  | Any numeric value                | `42`, `3.14`              |
| `integer` | Integer values                   | `42`, `-7`                |
| `float32` | 32-bit floating-point numbers    | `3.14f`                   |
| `float64` | 64-bit floating-point numbers    | `3.14`                    |
| `pair`    | Pair of values                   | `'(1 . 2)`                |
| `list`    | List of values                   | `'(1 2 3)`, `'()`         |
| `vector`  | Vector of values                 | `#(1 2 3)`                |
| `procedure`| Procedure (function) values     | `(lambda (x) x)`          |
| `any`     | Any value (dynamic type)         | Any Eshkol value          |

In addition, Eshkol provides scientific computing types:

| Type                  | Description                     | Examples                       |
|-----------------------|---------------------------------|--------------------------------|
| `vector<T>`           | Vector of elements of type T    | `#f64vec(1.0 2.0 3.0)`        |
| `vector<T, n>`        | Vector of known length n        | `#f64vec(1.0 2.0 3.0)`        |
| `matrix<T>`           | Matrix of elements of type T    | `#f64mat((1.0 2.0) (3.0 4.0))`|
| `matrix<T, m, n>`     | Matrix of known dimensions      | `#f64mat((1.0 2.0) (3.0 4.0))`|
| `tensor<T, shape>`    | Tensor with specified shape     | `#tensor(...)`                 |

### 4.3 Type Annotations

Eshkol supports three complementary approaches to typing, giving programmers flexibility in how they express and enforce types.

#### 4.3.1 Implicit Typing

Implicit typing relies on Eshkol's type inference system to automatically determine types based on how values are used. This approach requires no type annotations, making it ideal for rapid prototyping or when migrating existing Scheme code.

```scheme
;; No type annotations, but fully type-checked
(define (add x y)
  (+ x y))

;; The compiler infers that x and y must be numbers
;; and that the function returns a number
```

#### 4.3.2 Inline Explicit Typing

Inline explicit typing allows programmers to annotate parameters and return types directly in the function definition. This provides clear documentation and ensures type safety at the definition site.

```scheme
;; Explicitly typed parameters and return type
(define (add-integers x : integer y : integer) : integer
  (+ x y))

;; Vector operations with explicit types
(define (compute-distance point1 : vector<float64> point2 : vector<float64>) : float64
  (let ((x1 (vector-ref point1 0))
        (y1 (vector-ref point1 1))
        (x2 (vector-ref point2 0))
        (y2 (vector-ref point2 1)))
    (sqrt (+ (expt (- x2 x1) 2) (expt (- y2 y1) 2)))))
```

#### 4.3.3 Separate Type Declarations

Separate type declarations allow programmers to specify types separately from the implementation. This is particularly useful for complex type signatures or when adding types to existing code without modifying it.

```scheme
;; Separate type declaration
(: gradient-descent (-> (-> vector<float64> float64) vector<float64> float64 vector<float64>))

;; Implementation without type annotations
(define (gradient-descent f initial-point learning-rate)
  (let ((gradient (autodiff-gradient f initial-point)))
    (vector-subtract initial-point (vector-scale gradient learning-rate))))
```

### 4.4 Complex Types

Eshkol supports several advanced type constructs to express complex type relationships.

#### 4.4.1 Function Types

Function types use the arrow notation (`->`) to indicate parameter and return types:

```scheme
(-> param-type1 param-type2 ... return-type)
```

Examples:

```scheme
(: square (-> number number))  ; Takes a number, returns a number
(: cons (-> a b (pair a b)))   ; Takes values of types a and b, returns a pair
(: map (-> (-> a b) (list a) (list b)))  ; Higher-order function type
```

Function types can be nested to represent higher-order functions:

```scheme
(: compose (-> (-> b c) (-> a b) (-> a c)))  ; Function composition
```

#### 4.4.2 Parametric Types

Parametric types allow for generic code that works with multiple types:

```scheme
(forall (a b ...) type-expression)
```

Examples:

```scheme
(: id (forall (a) (-> a a)))  ; Identity function for any type
(: map (forall (a b) (-> (-> a b) (list a) (list b))))  ; Generic map
```

Type variables (like `a` and `b` above) can appear multiple times in the type expression, indicating that the same type must be used in all occurrences.

#### 4.4.3 Vector and Matrix Types

Eshkol provides specialized types for vectors and matrices, which are crucial for scientific computing:

```scheme
;; Vector types
(vector<element-type>)               ; Vector of any length
(vector<element-type, dimension>)    ; Vector of fixed length

;; Matrix types
(matrix<element-type>)               ; Matrix of any dimensions
(matrix<element-type, rows, cols>)   ; Matrix of fixed dimensions
```

Examples:

```scheme
(: v1 (vector<float64>))             ; A vector of double-precision floats
(: v2 (vector<float64, 3>))          ; A 3D vector of double-precision floats
(: m1 (matrix<float64>))             ; A matrix of double-precision floats
(: m2 (matrix<float64, 3, 3>))       ; A 3x3 matrix of double-precision floats
```

#### 4.4.4 Tensor Types

For more complex multi-dimensional data, Eshkol provides tensor types:

```scheme
(tensor<element-type, dimensions...>)  ; Tensor with fixed dimensions
```

Examples:

```scheme
(: t1 (tensor<float64, 2, 2, 2>))      ; A 2x2x2 tensor of double-precision floats
(: t2 (tensor<float32, 28, 28, 3>))    ; A tensor for a small RGB image
```

### 4.5 Type Classes and Constraints

Eshkol supports type classes (similar to Haskell) or traits (similar to Rust) for ad-hoc polymorphism:

```scheme
;; Define a Numeric type class
(define-type-class (Numeric a)
  (+ : (-> a a a))
  (- : (-> a a a))
  (* : (-> a a a))
  (/ : (-> a a a)))

;; Constrain a function to work with any Numeric type
(: sum (forall (a) (-> (vector<a>) a) (where (Numeric a))))
(define (sum vec)
  (let ((result (vector-ref vec 0)))
    (do ((i 1 (+ i 1)))
        ((= i (vector-length vec)) result)
      (set! result (+ result (vector-ref vec i))))))
```

Type classes enable powerful abstractions while maintaining type safety:

```scheme
;; Define a Differentiable type class
(define-type-class (Differentiable a)
  (gradient : (-> (-> a float64) a a)))

;; Implement for vectors
(define-type-class-instance (Differentiable (vector<float64>))
  (define (gradient f v)
    (autodiff-gradient f v)))

;; A function that works with any Differentiable type
(: optimize (forall (a) (-> (-> a float64) a float64 integer a) 
                    (where (Differentiable a))))
(define (optimize f initial-point learning-rate iterations)
  (let loop ((point initial-point)
             (iter 0))
    (if (= iter iterations)
        point
        (let ((grad (gradient f point)))
          (loop (vector-subtract point (vector-scale grad learning-rate))
                (+ iter 1))))))
```

### 4.6 Gradual Typing

Eshkol employs gradual typing, allowing programs to mix typed and untyped code seamlessly:

1. **The Dynamic Type**: The type system includes a special dynamic type (`any`) that represents dynamically typed values.

   ```scheme
   (: f (-> any any))  ; f takes and returns dynamically typed values
   ```

2. **Type Boundaries**: At the boundary between typed and untyped code, the compiler inserts runtime checks.

   ```scheme
   (: typed-function (-> number number))
   (define (typed-function x)
     (untyped-function x))  ; Runtime check inserted here
   ```

3. **Optional Annotations**: Types can be added incrementally to existing code.

   ```scheme
   (define (f x) x)  ; Untyped
   (: g (-> number number))
   (define (g x) x)  ; Typed
   ```

### 4.7 Type Inference

Eshkol employs a sophisticated type inference system based on the Hindley-Milner algorithm, allowing it to determine types automatically in many cases without explicit annotations.

The type inference system operates at several levels:

1. **Local Inference**: Types are inferred within function bodies based on how variables are used.

   ```scheme
   (define (square x)
     (* x x))  ; Infers that x must be a number
   ```

2. **Global Inference**: Types can be inferred across module boundaries.

   ```scheme
   (define (f x) (g x))  ; f's type depends on g's type
   (define (g x) (* x 2))  ; g takes and returns a number
   ```

3. **Contextual Inference**: Types can be inferred based on how functions are used.

   ```scheme
   (map square '(1 2 3))  ; Infers that square takes a number
   ```

## 5. Core Language Features

### 5.1 Expressions

Eshkol follows Scheme's expression-based paradigm, where all computations are expressed as expressions that evaluate to values. Standard Scheme expressions include:

- **Literals**: Self-evaluating values like numbers, strings, booleans, etc.
- **Variables**: Identifiers that refer to values bound in the current scope
- **Procedure applications**: `(procedure arg1 arg2 ...)`
- **Lambda expressions**: `(lambda (param1 param2 ...) body ...)`
- **Conditional expressions**: `(if test then else)`
- **Assignments**: `(set! variable value)`
- **Sequencing**: `(begin expr1 expr2 ...)`

### 5.2 Special Forms

Eshkol includes all standard Scheme special forms, plus extensions for its advanced features:

**Standard Scheme Special Forms**:

- `define`: Define variables and procedures
- `lambda`: Create anonymous procedures
- `if`: Conditional execution
- `cond`: Multi-way conditional
- `case`: Pattern matching conditional
- `and`, `or`: Short-circuit logical operations
- `let`, `let*`, `letrec`: Variable binding constructs
- `set!`: Variable assignment
- `begin`: Sequence of expressions
- `quote`, `quasiquote`, `unquote`, `unquote-splicing`: Quotation forms

**Eshkol Type System Forms**:

- `:`: Type annotation
- `define-type`: Define new types
- `define-type-class`: Define type classes
- `define-type-class-instance`: Implement type classes

**Eshkol Scientific Computing Forms**:

- Vector and matrix operation special forms

**Eshkol Automatic Differentiation Forms**:

- `autodiff-gradient`: Compute gradients
- `autodiff-jacobian`: Compute Jacobian matrices
- `autodiff-hessian`: Compute Hessian matrices

### 5.3 Binding Constructs

Eshkol supports all standard Scheme binding constructs, with extensions for typed bindings:

**Standard Scheme Bindings**:

```scheme
(let ((var1 val1) (var2 val2) ...) body ...)
(let* ((var1 val1) (var2 val2) ...) body ...)
(letrec ((var1 val1) (var2 val2) ...) body ...)
```

**Eshkol Typed Bindings**:

```scheme
(let ((var1 : type1 val1) (var2 : type2 val2) ...) body ...)
(let* ((var1 : type1 val1) (var2 : type2 val2) ...) body ...)
(letrec ((var1 : type1 val1) (var2 : type2 val2) ...) body ...)
```

### 5.4 Control Flow

Eshkol supports all standard Scheme control flow constructs:

**Conditionals**:

```scheme
(if condition then-expr else-expr)
(cond (test1 expr1 ...) ... (else default-expr ...))
(case key ((datum1 ...) expr1 ...) ... (else default-expr ...))
(when condition expr ...)
(unless condition expr ...)
```

**Iteration**:

```scheme
(do ((var1 init1 step1) ...) (test result) body ...)
```

**Exception Handling**:

```scheme
(guard (var (condition1 expr1 ...) ...) body ...)
(raise obj)
(raise-continuable obj)
```

### 5.5 Eshkol-specific Extensions

Eshkol extends the core language with several new constructs:

**Type Pattern Matching**:

```scheme
(match expr
  (pattern1 : type1 result1)
  (pattern2 : type2 result2)
  ...)
```

**Computation Expression Blocks**:

```scheme
(computation-block kind
  expr1
  expr2
  ...)
```

Where `kind` can be `autodiff`, `stochastic`, `quantum`, etc., enabling domain-specific syntax within the block.

**Pipeline Operator**:

```scheme
(|> initial-value
    (transform1 arg1 arg2)
    transform2
    (transform3 arg1))
```

Equivalent to:
```scheme
(transform3 arg1 (transform2 (transform1 initial-value arg1 arg2)))
```

This enables more readable data transformation pipelines.

## 6. Functions and Closures

Eshkol fully supports Scheme's first-class functions and lexical closures, with extensions for types, performance optimization, and scientific computing.

### 6.1 Function Definition

Eshkol provides several ways to define functions:

**Basic Function Definition**:

```scheme
(define (function-name param1 param2 ...)
  body-expressions)
```

**Lambda Expression Definition**:

```scheme
(define function-name
  (lambda (param1 param2 ...)
    body-expressions))
```

**Typed Function Definition**:

```scheme
(: function-name (-> param-type1 param-type2 ... return-type))
(define (function-name param1 param2 ...)
  body-expressions)
```

**Inline Type Annotation**:

```scheme
(define (function-name param1 : type1 param2 : type2 ...) : return-type
  body-expressions)
```

Examples:

```scheme
;; Basic function
(define (square x)
  (* x x))

;; Lambda form
(define add
  (lambda (x y)
    (+ x y)))

;; With separate type declaration
(: factorial (-> integer integer))
(define (factorial n)
  (if (zero? n)
      1
      (* n (factorial (- n 1)))))

;; With inline type annotations
(define (distance p1 : (vector<float64, 2>) p2 : (vector<float64, 2>)) : float64
  (sqrt (+ (expt (- (vector-ref p2 0) (vector-ref p1 0)) 2)
           (expt (- (vector-ref p2 1) (vector-ref p1 1)) 2))))
```

### 6.2 Lambda Expressions

Lambda expressions create anonymous procedures:

```scheme
(lambda (param1 param2 ...) body-expressions)
```

Lambda expressions capture variables from their lexical environment, forming closures:

```scheme
(define (make-adder n)
  (lambda (x) (+ x n)))  ; n is captured from the environment

(define add-5 (make-adder 5))
(add-5 10)  ; => 15
```

Lambda expressions can also include type annotations:

```scheme
;; With parameter type annotations
(lambda (x : number y : number) (+ x y))

;; With return type annotation
(lambda (x y) : number (+ x y))

;; With both
(lambda (x : number y : number) : number (+ x y))
```

### 6.3 Function Application

Functions are applied using the standard Scheme function application syntax:

```scheme
(function-name arg1 arg2 ...)
```

Eshkol also supports the standard Scheme higher-order function application operations:

```scheme
(apply function-name arglist)
(map function-name list1 list2 ...)
(for-each function-name list1 list2 ...)
```

Examples:

```scheme
(square 5)                  ; => 25
(apply + '(1 2 3 4 5))      ; => 15
(map square '(1 2 3 4 5))   ; => (1 4 9 16 25)
(for-each display '(1 2 3)) ; Displays 123
```

Type checking is performed at function application sites, ensuring that arguments match parameter types and, if necessary, inserting runtime type checks at the boundary between typed and untyped code.

### 6.4 Higher-Order Functions

Eshkol fully supports higher-order functions (functions that take other functions as arguments or return functions as results), making them a core part of the language's functional programming paradigm.

Examples of higher-order functions:

```scheme
;; Function that takes a function as an argument
(define (apply-twice f x)
  (f (f x)))

;; Function that returns a function
(define (make-multiplier factor)
  (lambda (x) (* x factor)))

;; Using higher-order functions
(define double (make-multiplier 2))
(apply-twice double 3)  ; => (double (double 3)) => (double 6) => 12
```

The type system fully supports higher-order functions through function types:

```scheme
(: apply-twice (forall (a) (-> (-> a a) a a)))
(define (apply-twice f x)
  (f (f x)))

(: make-multiplier (-> number (-> number number)))
(define (make-multiplier factor)
  (lambda (x) (* x factor)))
```

### 6.5 Function Composition

Eshkol extends Scheme with efficient function composition capabilities, allowing functions to be combined to create new functions.

**Binary Composition**:

```scheme
;; Define the compose function
(define (compose f g)
  (lambda (x) (f (g x))))

;; Example usage
(define square (lambda (x) (* x x)))
(define double (lambda (x) (+ x x)))

;; Create a composed function
(define square-then-double (compose double square))

;; Use the composed function
(square-then-double 3)  ; => (double (square 3)) => (double 9) => 18
```

**N-ary Composition**:

```scheme
;; Define the compose-n function for multiple functions
(define (compose-n . fns)
  (fold-right compose identity fns))

;; Example usage
(define square (lambda (x) (* x x)))
(define double (lambda (x) (+ x x)))
(define add1 (lambda (x) (+ x 1)))

;; Create a composed function
(define pipeline (compose-n add1 double square))

;; Use the composed function
(pipeline 3)  ; => (add1 (double (square 3))) => (add1 (double 9)) => (add1 18) => 19
```

**Composition Operator**:

```scheme
;; The . operator for function composition
(define f.g (. f g))  ; Equivalent to (compose f g)

;; Example usage
(define double.square (. double square))
(double.square 3)  ; => 18
```

Function composition is optimized in Eshkol, with the compiler automatically fusing composed functions where possible to eliminate intermediate function calls and improve performance.

### 6.6 Mutual Recursion

Eshkol supports mutual recursion, where functions call each other in a recursive cycle:

```scheme
;; Mutual recursion with define
(define (even? n)
  (if (zero? n)
      #t
      (odd? (- n 1))))

(define (odd? n)
  (if (zero? n)
      #f
      (even? (- n 1))))

;; Mutual recursion with letrec
(letrec ((even? (lambda (n)
                 (if (zero? n)
                     #t
                     (odd? (- n 1)))))
         (odd? (lambda (n)
                (if (zero? n)
                    #f
                    (even? (- n 1))))))
  (even? 10))  ; => #t
```

In typed code, mutually recursive functions can also be type-annotated:

```scheme
(: even? (-> integer boolean))
(: odd? (-> integer boolean))

(define (even? n)
  (if (zero? n)
      #t
      (odd? (- n 1))))

(define (odd? n)
  (if (zero? n)
      #f
      (even? (- n 1))))
```

The type checker ensures that mutually recursive functions have compatible types, preventing errors in the recursive cycle.

## 7. Scientific Computing Features

Eshkol extends Scheme with rich support for scientific computing, including specialized types, operations, and optimizations for numerical computing.

### 7.1 Vector Operations

Eshkol provides a comprehensive set of vector operations, all of which are optimized for performance using SIMD instructions where available.

**Vector Creation**:

```scheme
(make-vector length initial-value)  ; Standard Scheme
(vector element ...)                ; Standard Scheme
(vector<T> element ...)             ; Typed vector
(vector<float64> 1.0 2.0 3.0)       ; Explicitly float64 vector
```

**Vector Access and Mutation**:

```scheme
(vector-ref vector index)           ; Standard Scheme
(vector-set! vector index value)    ; Standard Scheme
```

**Vector Operations**:

```scheme
(vector-add v1 v2)                  ; Element-wise addition
(vector-subtract v1 v2)             ; Element-wise subtraction
(vector-multiply v1 v2)             ; Element-wise multiplication
(vector-divide v1 v2)               ; Element-wise division
(vector-scale v scalar)             ; Scale vector by scalar
(vector-dot v1 v2)                  ; Dot product
(vector-magnitude v)                ; Euclidean norm (magnitude)
(vector-normalize v)                ; Unit vector in same direction
(vector-map f v)                    ; Apply function to each element
(vector-map2 f v1 v2)               ; Apply binary function element-wise
(vector-fold f init v)              ; Fold vector with function
```

**Example**:

```scheme
(define v1 (vector<float64> 1.0 2.0 3.0))
(define v2 (vector<float64> 4.0 5.0 6.0))

(vector-add v1 v2)        ; => (vector<float64> 5.0 7.0 9.0)
(vector-dot v1 v2)        ; => 32.0
(vector-magnitude v1)     ; => 3.7416573867739413
(vector-normalize v1)     ; => (vector<float64> 0.267 0.535 0.802)
```

### 7.2 Matrix Operations

Eshkol provides comprehensive support for matrix operations.

**Matrix Creation**:

```scheme
(make-matrix rows cols initial-value)
(matrix row-vector ...)
(matrix<T> row-vector ...)
(matrix<float64> (vector<float64> 1.0 2.0) (vector<float64> 3.0 4.0))
```

**Matrix Access and Mutation**:

```scheme
(matrix-ref matrix row col)
(matrix-set! matrix row col value)
(matrix-row matrix row)
(matrix-column matrix col)
```

**Matrix Operations**:

```scheme
(matrix-add m1 m2)                  ; Element-wise addition
(matrix-subtract m1 m2)             ; Element-wise subtraction
(matrix-multiply m1 m2)             ; Matrix multiplication
(matrix-scale m scalar)             ; Scale matrix by scalar
(matrix-transpose m)                ; Transpose matrix
(matrix-determinant m)              ; Determinant (for square matrices)
(matrix-inverse m)                  ; Inverse (for invertible matrices)
(matrix-eigenvalues m)              ; Eigenvalues (for square matrices)
(matrix-eigenvectors m)             ; Eigenvectors (for square matrices)
(matrix-svd m)                      ; Singular value decomposition
(matrix-solve m v)                  ; Solve linear system Mx = v
```

**Example**:

```scheme
(define m1 (matrix<float64>
             (vector<float64> 1.0 2.0)
             (vector<float64> 3.0 4.0)))
(define m2 (matrix<float64>
             (vector<float64> 5.0 6.0)
             (vector<float64> 7.0 8.0)))

(matrix-add m1 m2)        ; => Matrix with values 6 8, 10 12
(matrix-multiply m1 m2)   ; => Matrix with values 19 22, 43 50
(matrix-determinant m1)   ; => -2.0
(matrix-inverse m1)       ; => Matrix with values -2 1, 1.5 -0.5
```

### 7.3 Tensor Operations

For multi-dimensional data beyond matrices, Eshkol provides tensor operations.

**Tensor Creation**:

```scheme
(make-tensor dimensions initial-value)
(tensor<T> dimensions data)
```

**Tensor Access and Mutation**:

```scheme
(tensor-ref tensor indices ...)
(tensor-set! tensor indices ... value)
```

**Tensor Operations**:

```scheme
(tensor-add t1 t2)                  ; Element-wise addition
(tensor-subtract t1 t2)             ; Element-wise subtraction
(tensor-multiply t1 t2)             ; Element-wise multiplication
(tensor-contract t indices)         ; Tensor contraction
(tensor-product t1 t2)              ; Tensor product
(tensor-transpose t permutation)    ; Tensor transpose
(tensor-reshape t new-dimensions)   ; Reshape tensor
```

**Example**:

```scheme
(define t1 (make-tensor '(2 2 2) 0.0))
(tensor-set! t1 0 0 0 1.0)
(tensor-set! t1 1 1 1 2.0)

(define t2 (make-tensor '(2 2 2) 1.0))
(tensor-add t1 t2)        ; => Tensor with all elements 1.0 except (0,0,0)=2.0 and (1,1,1)=3.0
```

### 7.4 Numerical Algorithms

Eshkol provides a wide range of numerical algorithms for scientific computing.

**Numerical Integration**:

```scheme
(integrate f a b)                   ; Integrate f from a to b
(integrate-adaptive f a b error)    ; Adaptive integration
(integrate-monte-carlo f a b samples) ; Monte Carlo integration
```

**Optimization**:

```scheme
(minimize f initial-guess)          ; Find minimum of f
(maximize f initial-guess)          ; Find maximum of f
(find-root f a b)                   ; Find root of f in [a,b]
(gradient-descent f initial-point learning-rate iterations)
```

**Differential Equations**:

```scheme
(ode-solve f y0 t0 t1 dt)           ; Solve ODE dy/dt = f(y,t)
(euler-step f y t dt)               ; Euler's method step
(runge-kutta-4-step f y t dt)       ; 4th-order Runge-Kutta step
```

**Linear Algebra**:

```scheme
(solve-linear-system A b)           ; Solve Ax = b
(eigendecomposition A)              ; Eigendecomposition of A
(svd A)                             ; Singular value decomposition
(qr-decomposition A)                ; QR decomposition
```

**Example**:

```scheme
(define (f x) (* x x))
(integrate f 0 1)          ; => 0.33333 (approximately 1/3)

(define (g x) (- (* x x) 2))
(find-root g 0 2)          ; => 1.4142 (approximately sqrt(2))
```

### 7.5 SIMD Optimization

Eshkol automatically optimizes vector and matrix operations using SIMD (Single Instruction, Multiple Data) instructions where available. This can provide significant performance improvements for numerical computations.

SIMD optimization is transparent to the programmer—the same high-level vector and matrix operations are compiled to use SIMD instructions when appropriate.

**SIMD Optimization Levels**:

```scheme
(set-simd-optimization-level! level) ; Set SIMD optimization level (0-3)
(current-simd-features)              ; Get available SIMD features
```

**Example of SIMD-optimized code**:

```scheme
;; This high-level code
(define (vector-add-scale v1 v2 scale)
  (vector-add v1 (vector-scale v2 scale)))

;; Might be compiled to use SIMD instructions like:
;; (in pseudocode)
;; for (i = 0; i < length; i += 4)
;;   v1_chunk = load_simd(&v1[i])
;;   v2_chunk = load_simd(&v2[i])
;;   scaled = mul_simd(v2_chunk, scale)
;;   result = add_simd(v1_chunk, scaled)
;;   store_simd(&result[i], result)
```

Eshkol's SIMD optimization is aware of the target architecture and can generate code for multiple SIMD instruction sets (e.g., SSE, AVX, AVX-512, ARM NEON) depending on the platform.

## 8. Automatic Differentiation

One of Eshkol's most distinctive features is its first-class support for automatic differentiation (AD), which enables efficient computation of derivatives of arbitrary functions. This is essential for many scientific computing and machine learning applications.

### 8.1 Forward Mode

Forward mode AD computes derivatives alongside the original computation by propagating derivative values forward through the computation.

**Basic Forward Mode**:

```scheme
(autodiff-forward f x)              ; Compute f(x) and df/dx
(autodiff-forward-n f x n)          ; Compute derivatives up to nth order
```

**Example**:

```scheme
(define (f x) (* x x))
(autodiff-forward f 3)              ; => (values 9 6)

;; Multiple arguments
(define (g x y) (+ (* x x) (* y y)))
(autodiff-forward g 3 4)            ; => (values 25 (6 8))
```

### 8.2 Reverse Mode

Reverse mode AD computes derivatives by first performing the original computation, recording a computation graph, and then propagating derivatives backward. This is more efficient for functions with many inputs and few outputs, such as neural networks.

**Basic Reverse Mode**:

```scheme
(autodiff-reverse f x)              ; Compute f(x) and df/dx
(autodiff-gradient f x)             ; Compute gradient of f at x
```

**Example**:

```scheme
(define (f x) (* x x))
(autodiff-reverse f 3)              ; => (values 9 6)

;; Vector input, scalar output
(define (g v) (vector-dot v v))     ; Sum of squares
(autodiff-gradient g (vector 3 4))  ; => (values 25 (vector 6 8))
```

### 8.3 Higher-Order Derivatives

Eshkol supports computation of higher-order derivatives through repeated application of differentiation.

**Higher-Order Derivatives**:

```scheme
(define df/dx (autodiff-gradient f))
(define d2f/dx2 (autodiff-gradient df/dx))
```

**Example**:

```scheme
(define (f x) (* x x x))            ; f(x) = x^3
(define df/dx (autodiff-gradient f))
(define d2f/dx2 (autodiff-gradient df/dx))
(define d3f/dx3 (autodiff-gradient d2f/dx2))

(f 2)                               ; => 8 (2^3)
(df/dx 2)                           ; => 12 (3*2^2)
(d2f/dx2 2)                         ; => 12 (6*2)
(d3f/dx3 2)                         ; => 6 (6)
```

### 8.4 Vector and Matrix Derivatives

Eshkol supports automatic differentiation for vector and matrix valued functions, computing Jacobians, Hessians, and other matrix derivatives.

**Vector and Matrix Derivatives**:

```scheme
(autodiff-jacobian f x)             ; Compute Jacobian matrix of f at x
(autodiff-hessian f x)              ; Compute Hessian matrix of f at x
```

**Example**:

```scheme
;; Function mapping R^2 to R^2: f(x,y) = (x^2, y^2)
(define (f v)
  (let ((x (vector-ref v 0))
        (y (vector-ref v 1)))
    (vector (* x x) (* y y))))

(autodiff-jacobian f (vector 3 4))  ; => Matrix of partial derivatives
                                    ;    [6 0]
                                    ;    [0 8]
```

### 8.5 Integration with Type System

Eshkol's automatic differentiation system is deeply integrated with its type system, enabling static verification of differentiation operations and automatic selection of the optimal differentiation mode.

**Type-Directed AD Mode Selection**:

```scheme
;; The compiler selects forward mode for this function (few inputs, one output)
(: f (-> float64 float64))
(define (f x) (* x x))

;; The compiler selects reverse mode for this function (many inputs, one output)
(: g (-> (vector float64 n) float64))
(define (g v) (vector-sum (vector-square v)))
```

**Type Safety for Derivatives**:

```scheme
;; Function type
(: f (-> float64 float64))
(define (f x) (* x x))

;; Derivative type is automatically derived
(: df/dx (-> float64 float64))
(define df/dx (autodiff-gradient f))

;; For vector functions, the gradient has a specific type
(: g (-> (vector float64 n) float64))
(define g (lambda (v) (vector-sum (vector-square v))))

;; Gradient type is automatically derived
(: grad-g (-> (vector float64 n) (vector float64 n)))
(define grad-g (autodiff-gradient g))
```

The type system ensures that derivatives are computed correctly and have the expected types, preventing common errors in numerical code.

## 9. Stochastic Lambda Calculus

Eshkol extends the lambda calculus with stochastic features, enabling probabilistic programming and stochastic simulation directly in the language.

### 9.1 Probabilistic Functions

Probabilistic functions are functions that may return different values on different calls, according to a probability distribution.

**Distribution Creation**:

```scheme
(normal-dist mean std)            ; Normal (Gaussian) distribution
(uniform-dist min max)            ; Uniform distribution
(bernoulli-dist p)                ; Bernoulli distribution
(categorical-dist probs)          ; Categorical distribution
(dirichlet-dist alphas)           ; Dirichlet distribution
```

**Sampling**:

```scheme
(sample dist)                     ; Draw one sample from the distribution
(sample-n dist n)                 ; Draw n samples from the distribution
```

**Example**:

```scheme
(define d (normal-dist 0 1))      ; Standard normal distribution
(sample d)                        ; => -0.123 (random value)
(sample-n d 3)                    ; => (vector -0.5 0.7 -0.2) (random values)
```

### 9.2 Sampling Operations

Eshkol provides operations for working with samples and constructing complex sampling processes.

**Sampling Operations**:

```scheme
(rejection-sample expr condition)  ; Sample until condition is satisfied
(importance-sample expr weight)    ; Sample with importance weight
(resample samples weights)         ; Resample according to weights
(monte-carlo expr trials)          ; Monte Carlo estimation
```

**Example**:

```scheme
;; Estimate π using Monte Carlo
(define (estimate-pi trials)
  (let ((count (monte-carlo
                 (let ((x (sample (uniform-dist -1 1)))
                       (y (sample (uniform-dist -1 1))))
                   (< (+ (* x x) (* y y)) 1))
                 trials)))
    (* 4.0 (/ count trials))))

(estimate-pi 1000000)              ; => 3.14159... (approximate)
```

### 9.3 Distribution Types

Eshkol's type system includes types for probability distributions, enabling static verification of probabilistic operations.

**Distribution Types**:

```scheme
(distribution T)                   ; Distribution over type T
(: normal-dist (-> float64 float64 (distribution float64)))
(: sample (forall (a) (-> (distribution a) a)))
```

**Example**:

```scheme
(: standard-normal (distribution float64))
(define standard-normal (normal-dist 0 1))

(: sample-standard-normal (-> float64))
(define (sample-standard-normal)
  (sample standard-normal))
```

### 9.4 Inference Algorithms

Eshkol provides built-in support for several probabilistic inference algorithms.

**Inference Algorithms**:

```scheme
(metropolis-hastings target-dist proposal-dist iterations)
(gibbs-sampler full-conditionals iterations)
(particle-filter init-dist observe-fn transition-fn observations n-particles)
```

**Example**:

```scheme
;; Bayesian inference for mean of normal distribution
(define (bayesian-mean observations)
  (let* ((n (length observations))
         (sum (fold-left + 0 observations))
         (mean (/ sum n))
         (prior (normal-dist 0 10))
         (likelihood (lambda (mu)
                       (fold-left *
                                 1
                                 (map (lambda (x) (normal-pdf x mu 1))
                                     observations))))
         (posterior (lambda (mu)
                      (* (normal-pdf mu 0 10)
                         (likelihood mu)))))
    (metropolis-hastings posterior
                        (lambda (mu) (normal-dist mu 0.1))
                        1000)))
```

## 10. Quantum Computing Extensions

Eshkol includes extensions for quantum computing, allowing quantum algorithms to be expressed directly in the language.

### 10.1 Quantum Types

Eshkol's type system includes types for quantum computing concepts.

**Quantum Types**:

```scheme
(qubit)                            ; Single qubit
(qureg n)                          ; Quantum register of n qubits
(qstate n)                         ; Quantum state of n qubits
```

**Quantum Type Operations**:

```scheme
(qubit? x)                         ; Check if x is a qubit
(qureg? x)                         ; Check if x is a quantum register
(qstate? x)                        ; Check if x is a quantum state
(qureg-size r)                     ; Number of qubits in register r
```

### 10.2 Quantum Operations

Eshkol provides operations for manipulating quantum states and registers.

**Quantum Gates**:

```scheme
(hadamard qubit)                   ; Hadamard gate
(phase-shift qubit angle)          ; Phase shift gate
(controlled-not control target)    ; CNOT gate
(toffoli control1 control2 target) ; Toffoli gate
(rotation-x qubit angle)           ; Rotation around X axis
(rotation-y qubit angle)           ; Rotation around Y axis
(rotation-z qubit angle)           ; Rotation around Z axis
```

**Quantum Register Operations**:

```scheme
(qreg-new n initial-state)         ; Create new quantum register
(qreg-apply gate qreg)             ; Apply gate to register
(qreg-apply-controlled gate control target qreg)  ; Apply controlled gate
```

**Example**:

```scheme
;; Quantum teleportation
(define (quantum-teleport q)
  (let* ((alice-aux (qreg-new 1 |0>))
         (bob-qubit (qreg-new 1 |0>))
         (epr-pair (entangle alice-aux bob-qubit))
         (alice-qubits (qreg-join q (qreg-extract-qubit epr-pair 0)))
         (bob-qubits (qreg-extract-qubit epr-pair 1)))
    
    ;; Alice's operations
    (qreg-apply (controlled-not 0 1) alice-qubits)
    (qreg-apply (hadamard 0) alice-qubits)
    (let ((measurement (qreg-measure alice-qubits)))
      
      ;; Classical communication
      
      ;; Bob's operations
      (when (bit-set? measurement 1)
        (qreg-apply pauli-x bob-qubits))
      (when (bit-set? measurement 0)
        (qreg-apply pauli-z bob-qubits))
      
      bob-qubits)))
```

### 10.3 Measurement

Eshkol provides operations for measuring quantum states, which collapses the quantum state according to quantum mechanics principles.

**Measurement Operations**:

```scheme
(measure qubit)                    ; Measure single qubit (0 or 1)
(qreg-measure qreg)                ; Measure all qubits in register
(qreg-measure-qubit qreg index)    ; Measure specific qubit in register
(qreg-measure-partial qreg indices) ; Measure subset of qubits
```

**Example**:

```scheme
(define q (qreg-new 3 |000>))      ; 3-qubit register in |000> state
(qreg-apply hadamard 0 q)          ; Apply Hadamard to first qubit
(qreg-measure q)                   ; => 0 or 4 with equal probability (|000> or |100>)
```

### 10.4 Quantum Control Flow

Eshkol extends classical control flow to quantum contexts, allowing for quantum control flow constructs.

**Quantum Control Flow**:

```scheme
(qif condition true-branch false-branch)  ; Quantum if
(qwhile condition body)                  ; Quantum while loop
```

**Example**:

```scheme
;; Quantum if
(qif (qubit-is? q 1)
     (qreg-apply hadamard target)
     (qreg-apply phase-shift target (/ pi 4)))
```

## 11. Memory Management

Eshkol uses a hybrid memory management approach that differs from traditional Scheme implementations, providing more predictable performance and better control over memory usage.

### 11.1 Model Overview

Eshkol's memory management system combines several techniques:

1. **Arena Allocation**: Most memory is allocated from arenas, which are large pre-allocated memory blocks. This provides fast allocation and bulk deallocation.

2. **Reference Counting**: For complex objects like closures and environments, reference counting tracks when objects can be safely reused or freed.

3. **Regions**: Code can create and destroy memory regions, allowing for efficient management of temporary objects.

4. **Escape Analysis**: The compiler performs escape analysis to determine the lifetime of objects and optimize allocation.

This hybrid approach avoids the unpredictable pauses associated with traditional garbage collection while providing memory safety and efficiency.

**Arena Management**:

```scheme
(define-arena name size)           ; Define a new arena
(with-arena name expr ...)         ; Evaluate expressions in arena
(arena-reset! name)                ; Reset arena (free all allocations)
```

**Memory Regions**:

```scheme
(with-region expr ...)             ; Evaluate in a new memory region
(region-reset!)                    ; Reset current region
```

**Example**:

```scheme
(define-arena computation-arena (* 1024 1024))  ; 1MB arena

(with-arena computation-arena
  (let ((result (compute-matrix-inverse large-matrix)))
    (process-result result)))  ; Arena automatically reset after this
```

### 11.2 Resource Management

Eshkol provides mechanisms for managing resources beyond memory, such as file handles, network connections, and other system resources.

**Resource Management**:

```scheme
(with-resource expr cleanup-expr ...)  ; Resource acquisition is initialization
(protect expr finally-expr ...)        ; Ensure cleanup regardless of exit
```

**Example**:

```scheme
(define (process-file filename)
  (with-resource (open-input-file filename)
                (lambda (port) (close-input-port port))
    (lambda (port)
      (let loop ((line (read-line port))
                 (count 0))
        (if (eof-object? line)
            count
            (loop ((read-line port))
            (+ count 1))))))
```

This ensures that the file port is properly closed even if an error occurs during processing.

**Resource Tracking**:

```scheme
(track-resource resource cleanup)    ; Register resource for cleanup
(untrack-resource resource)          ; Unregister resource
(cleanup-resources!)                 ; Cleanup all tracked resources
```

**Example**:

```scheme
(define (process-multiple-files filenames)
  (for-each
    (lambda (filename)
      (let ((port (open-input-file filename)))
        (track-resource port close-input-port)
        (process-file-contents port)))
    filenames)
  (cleanup-resources!))
```

### 11.3 Effects on Programming Style

Eshkol's memory management approach influences programming style in several ways:

1. **Prefer Functional Approach**: Immutable data and pure functions work well with arena allocation, as they create less fragmentation and are easier to analyze for escape behavior.

2. **Explicit Resource Handling**: Use `with-resource` and similar constructs to ensure proper cleanup of non-memory resources.

3. **Region-Based Processing**: Organize computation phases into memory regions for efficient memory management.

4. **Avoid Circular References**: While reference counting handles most cases, circular references require explicit management.

**Example of Region-Based Processing**:

```scheme
(define (process-large-dataset data)
  (with-region
    ;; Phase 1: Data transformation
    (let ((transformed-data (transform-data data)))
      
      ;; Phase 2: Analysis
      (let ((analysis-results (analyze-data transformed-data)))
        
        ;; Phase 3: Report generation
        (generate-report analysis-results)))))
```

## 12. Macro System

Eshkol supports a powerful macro system for metaprogramming, extending the language's syntax and semantics.

### 12.1 Hygiene

The Eshkol macro system is hygienic, meaning that macros cannot inadvertently capture identifiers from the context where they are used. This prevents a whole class of subtle bugs that can occur in non-hygienic macro systems.

**Hygiene Example**:

```scheme
(define-syntax let1
  (syntax-rules ()
    ((let1 var val body ...)
     (let ((var val))
       body ...))))

(let ((var 5))
  (let1 x 10
    (+ x var)))  ; => 15, not a hygiene error
```

### 12.2 Syntax-Rules

The `syntax-rules` macro system provides pattern-based macros with hygienic variable binding.

**Syntax-Rules Definition**:

```scheme
(define-syntax name
  (syntax-rules (literals ...)
    ((pattern) template)
    ...))
```

**Example**:

```scheme
(define-syntax when
  (syntax-rules ()
    ((when condition body ...)
     (if condition
         (begin body ...)
         (void)))))

(when (> 5 3)
  (display "5 is greater than 3")
  (newline))
```

### 12.3 Syntax-Case

For more advanced macro transformations, Eshkol provides `syntax-case`, which allows for programmatic manipulation of syntax.

**Syntax-Case Definition**:

```scheme
(define-syntax name
  (lambda (stx)
    (syntax-case stx (literals ...)
      (pattern1 expr1)
      (pattern2 expr2)
      ...)))
```

**Example**:

```scheme
(define-syntax define-logging-function
  (lambda (stx)
    (syntax-case stx ()
      ((_ name)
       (let ((name-str (symbol->string (syntax->datum #'name))))
         #`(define (name arg)
             (display #,name-str)
             (display ": ")
             (display arg)
             (newline)))))))

(define-logging-function log-error)
(log-error "File not found")  ; Displays: log-error: File not found
```

### 12.4 Procedural Macros

Procedural macros provide full programmatic control over syntax transformation.

**Procedural Macro Definition**:

```scheme
(define-syntax name
  (lambda (stx)
    ;; Syntax transformation code
    ))
```

**Example**:

```scheme
(define-syntax define-enum
  (lambda (stx)
    (syntax-case stx ()
      ((_ name (id val) ...)
       (with-syntax (((index ...) (generate-temporaries #'(id ...))))
         #'(begin
             (define name (make-enum 'name))
             (define id (enum-add! name 'id val)) ...
             (define (name->string x)
               (case x
                 ((id) (symbol->string 'id)) ...
                 (else (error "Invalid enum value"))))))))))

(define-enum color
  (red 0)
  (green 1)
  (blue 2))

(color->string red)  ; => "red"
```

### 12.5 Eshkol-specific Extensions

Eshkol extends the standard Scheme macro system with specialized features for scientific computing, type-level programming, and performance optimization.

**Type-Level Macros**:

```scheme
(define-type-macro name
  (lambda (stx)
    ;; Type transformation code
    ))
```

**Example**:

```scheme
(define-type-macro vector-of
  (lambda (stx)
    (syntax-case stx ()
      ((_ elem-type)
       #'(vector elem-type)))))

(: v (vector-of float64))  ; Equivalent to (: v (vector float64))
```

**Optimization Directives**:

```scheme
(define-syntax with-optimization
  (syntax-rules ()
    ((with-optimization level body ...)
     (begin
       (set-optimization-level! level)
       (let ((result (begin body ...)))
         (restore-optimization-level!)
         result)))))
```

**Domain-Specific Macros**:

```scheme
(define-syntax matrix-with-builder
  (syntax-rules ()
    ((matrix-with-builder builder rows cols)
     (let ((m (make-matrix rows cols 0.0)))
       (do ((i 0 (+ i 1)))
           ((= i rows) m)
         (do ((j 0 (+ j 1)))
             ((= j cols))
           (matrix-set! m i j (builder i j))))))))

(define identity-matrix
  (matrix-with-builder
   (lambda (i j) (if (= i j) 1.0 0.0))
   3 3))
```

## 13. Module System

Eshkol provides a comprehensive module system for organizing code into reusable, composable units.

### 13.1 Module Definitions

Modules are defined using the `define-module` form, which specifies a name, imports, and exports.

**Module Definition**:

```scheme
(define-module name
  (import module1 module2 ...)
  (export identifier1 identifier2 ...)
  
  ;; Module body
  definitions-and-expressions)
```

**Example**:

```scheme
(define-module math.vector
  (import core)
  (export make-vector vector? vector-ref vector-set!
          vector-add vector-subtract vector-scale vector-dot)
  
  ;; Vector operations implementation
  (define (vector-add v1 v2) ...)
  (define (vector-subtract v1 v2) ...)
  (define (vector-scale v scalar) ...)
  (define (vector-dot v1 v2) ...))
```

### 13.2 Imports and Exports

Modules control their interface through imports and exports, allowing fine-grained control over visibility.

**Import Forms**:

```scheme
(import module-name)               ; Import all exported identifiers
(import (only module-name id1 id2 ...))  ; Import specific identifiers
(import (prefix module-name prefix))  ; Add prefix to imported identifiers
(import (rename module-name (old new) ...))  ; Rename imported identifiers
```

**Export Forms**:

```scheme
(export id1 id2 ...)               ; Export identifiers
(export (rename (old new) ...))    ; Export with renamed identifiers
```

**Example**:

```scheme
(define-module geometry
  (import (prefix math.vector vec:)
          (only math.matrix matrix-multiply))
  (export distance area volume)
  
  (define (distance p1 p2)
    (sqrt (vec:vector-dot (vec:vector-subtract p2 p1)
                         (vec:vector-subtract p2 p1))))
  
  (define (area shape) ...)
  (define (volume shape) ...))
```

### 13.3 Namespace Management

Eshkol provides tools for managing namespaces and resolving naming conflicts.

**Namespace Forms**:

```scheme
(namespace-open module-name)       ; Open module namespace
(namespace-close module-name)      ; Close module namespace
(namespace-with module-name expr ...)  ; Evaluate with open namespace
```

**Example**:

```scheme
;; Without namespace management
(define v1 (math.vector.make-vector 3 0.0))
(math.vector.vector-set! v1 0 1.0)

;; With namespace management
(namespace-with math.vector
  (define v1 (make-vector 3 0.0))
  (vector-set! v1 0 1.0))
```

**Module Aliases**:

```scheme
(define-module-alias alias module-name)
```

**Example**:

```scheme
(define-module-alias vec math.vector)
(define v1 (vec:make-vector 3 0.0))
```

## 14. Standard Library

Eshkol includes a comprehensive standard library divided into modules for different domains.

### 14.1 Core Procedures

The core module provides fundamental Scheme procedures, including:

- **Equivalence**: `eq?`, `eqv?`, `equal?`
- **Numbers**: `+`, `-`, `*`, `/`, `=`, `<`, `>`, `<=`, `>=`, `zero?`, etc.
- **Booleans**: `not`, `boolean?`
- **Pairs and Lists**: `cons`, `car`, `cdr`, `list`, `append`, `length`, etc.
- **Symbols**: `symbol?`, `symbol->string`, `string->symbol`
- **Characters**: `char?`, `char=?`, `char<?`, etc.
- **Strings**: `string?`, `string-length`, `string-ref`, `substring`, etc.
- **Vectors**: `vector?`, `vector-length`, `vector-ref`, `vector-set!`, etc.
- **Control**: `apply`, `map`, `for-each`, etc.
- **Evaluation**: `eval`, `scheme-report-environment`, etc.
- **Input/Output**: `read`, `write`, `display`, `newline`, etc.
- **System Interface**: `file-exists?`, `delete-file`, etc.

### 14.2 Scientific Computing Modules

The scientific computing modules provide advanced numerical computation capabilities:

**Vector Module**:

```scheme
(import scientific.vector)

;; Vector operations
(vector-add v1 v2)
(vector-subtract v1 v2)
(vector-scale v scalar)
(vector-dot v1 v2)
(vector-cross v1 v2)  ; For 3D vectors
(vector-magnitude v)
(vector-normalize v)
```

**Matrix Module**:

```scheme
(import scientific.matrix)

;; Matrix operations
(matrix-add m1 m2)
(matrix-subtract m1 m2)
(matrix-multiply m1 m2)
(matrix-scale m scalar)
(matrix-transpose m)
(matrix-inverse m)
(matrix-determinant m)
```

**Numerical Analysis Module**:

```scheme
(import scientific.numerical)

;; Numerical algorithms
(integrate f a b)
(differentiate f x)
(find-root f a b)
(minimize f initial-guess)
(interpolate points x)
```

**Statistics Module**:

```scheme
(import scientific.statistics)

;; Statistical functions
(mean data)
(median data)
(variance data)
(standard-deviation data)
(covariance data1 data2)
(correlation data1 data2)
(linear-regression x-data y-data)
```

### 14.3 Automatic Differentiation Modules

The automatic differentiation modules provide tools for gradient-based optimization and machine learning:

**Forward Mode Module**:

```scheme
(import autodiff.forward)

;; Forward-mode AD
(autodiff-forward f x)
(autodiff-forward-n f x n)
(make-dual value derivative)
(dual-value d)
(dual-derivative d)
```

**Reverse Mode Module**:

```scheme
(import autodiff.reverse)

;; Reverse-mode AD
(autodiff-reverse f x)
(autodiff-gradient f x)
(make-tape)
(tape-record! tape expr)
(tape-gradient tape)
```

**AD Algorithms Module**:

```scheme
(import autodiff.algorithms)

;; AD-based algorithms
(gradient-descent f initial-point learning-rate iterations)
(newtons-method f initial-point iterations)
(adam-optimizer f initial-point iterations)
```

### 14.4 Stochastic Computation Modules

The stochastic computation modules provide tools for probabilistic programming and stochastic simulation:

**Probability Module**:

```scheme
(import stochastic.probability)

;; Distribution constructors
(normal-dist mean std)
(uniform-dist min max)
(bernoulli-dist p)
(beta-dist alpha beta)
(gamma-dist shape scale)

;; Distribution operations
(pdf dist x)
(cdf dist x)
(quantile dist p)
(sample dist)
(sample-n dist n)
```

**Inference Module**:

```scheme
(import stochastic.inference)

;; Inference algorithms
(rejection-sample proposal condition)
(importance-sample proposal weight)
(metropolis-hastings target proposal iterations)
(gibbs-sampler full-conditionals iterations)
(particle-filter init-dist observe-fn transition-fn observations n-particles)
```

**Monte Carlo Module**:

```scheme
(import stochastic.monte-carlo)

;; Monte Carlo methods
(monte-carlo-integrate f a b n)
(monte-carlo-estimate procedure trials)
(bootstrap data statistic n)
(jack-knife data statistic)
(cross-validate data-splits model-builder error-fn)
```

### 14.5 Quantum Computing Modules

The quantum computing modules provide tools for quantum algorithm development:

**Quantum State Module**:

```scheme
(import quantum.state)

;; Quantum state operations
(qubit-new)
(qureg-new n initial-state)
(qstate-amplitude state basis-state)
(qstate-probability state basis-state)
(qstate-entanglement state partition)
```

**Quantum Gates Module**:

```scheme
(import quantum.gates)

;; Quantum gates
(hadamard qubit)
(pauli-x qubit)
(pauli-y qubit)
(pauli-z qubit)
(phase-shift qubit angle)
(controlled-not control target)
(toffoli control1 control2 target)
(swap qubit1 qubit2)
```

**Quantum Algorithms Module**:

```scheme
(import quantum.algorithms)

;; Quantum algorithms
(grover-search f n iterations)
(shor-factorize n)
(quantum-fourier-transform qureg)
(phase-estimation unitary eigenstate precision)
```

## 15. Appendices

### 15.1 Grammar (EBNF)

The formal grammar of Eshkol is specified using Extended Backus-Naur Form (EBNF):

```ebnf
program         ::= form*

form            ::= definition | expression

definition      ::= variable-definition | function-definition | 
                    macro-definition | module-definition |
                    type-definition | class-definition

variable-definition ::= '(' 'define' variable expression ')'
                    | '(' 'define' variable ':' type expression ')'

function-definition ::= '(' 'define' '(' variable parameter* ')' body ')'
                    | '(' 'define' '(' variable parameter* ')' ':' type body ')'

parameter       ::= variable | '(' variable ':' type ')'

expression      ::= literal | variable | procedure-call | 
                    lambda-expression | conditional | binding-expression |
                    assignment | sequence | quotation

literal         ::= boolean | number | character | string | 
                    symbol | list-literal | vector-literal

procedure-call  ::= '(' expression expression* ')'

lambda-expression ::= '(' 'lambda' '(' parameter* ')' body ')'
                  | '(' 'lambda' '(' parameter* ')' ':' type body ')'

conditional     ::= '(' 'if' expression expression expression ')'
                 | '(' 'cond' cond-clause+ ')'
                 | '(' 'case' expression case-clause+ ')'

binding-expression ::= '(' 'let' '(' binding* ')' body ')'
                    | '(' 'let*' '(' binding* ')' body ')'
                    | '(' 'letrec' '(' binding* ')' body ')'

binding         ::= '(' variable expression ')'
                 | '(' variable ':' type expression ')'

assignment      ::= '(' 'set!' variable expression ')'

sequence        ::= '(' 'begin' expression+ ')'

quotation       ::= '\'' datum | '(' 'quote' datum ')'

type            ::= base-type | parameterized-type | function-type |
                    forall-type | constraint-type

base-type       ::= identifier

parameterized-type ::= '(' identifier type* ')'
                    | identifier '<' type (',' type)* '>'

function-type   ::= '(' '->' type* type ')'

forall-type     ::= '(' 'forall' '(' type-variable* ')' type ')'

constraint-type ::= '(' type 'where' constraint+ ')'
```

This grammar is a simplified version; the complete grammar includes additional productions for all syntactic forms.

### 15.2 Standard Library Reference

For a detailed reference of the standard library, see the separate document `STANDARD_LIBRARY_REFERENCE.md`.

### 15.3 Type System Reference

For a detailed reference of the type system, see the separate document `TYPE_SYSTEM_REFERENCE.md`.

### 15.4 Example Programs

Below are examples demonstrating various Eshkol features:

**Factorial with Type Annotations**:

```scheme
(: factorial (-> integer integer))
(define (factorial n)
  (if (zero? n)
      1
      (* n (factorial (- n 1)))))

(factorial 5)  ; => 120
```

**Vector Operations with SIMD Optimization**:

```scheme
(: dot-product (-> (vector<float64>) (vector<float64>) float64))
(define (dot-product v1 v2)
  (let ((sum 0.0))
    (do ((i 0 (+ i 1)))
        ((= i (vector-length v1)) sum)
      (set! sum (+ sum (* (vector-ref v1 i)
                         (vector-ref v2 i)))))))

(dot-product (vector<float64> 1.0 2.0 3.0)
             (vector<float64> 4.0 5.0 6.0))  ; => 32.0
```

**Gradient Descent with Automatic Differentiation**:

```scheme
(: gradient-descent (-> (-> (vector<float64>) float64)
                       (vector<float64>)
                       float64
                       integer
                       (vector<float64>)))
(define (gradient-descent f initial-point learning-rate iterations)
  (let loop ((point initial-point)
             (iter 0))
    (if (= iter iterations)
        point
        (let ((gradient (autodiff-gradient f point)))
          (loop (vector-subtract point
                                (vector-scale gradient learning-rate))
                (+ iter 1))))))

(: rosenbrock (-> (vector<float64>) float64))
(define (rosenbrock v)
  (let ((x (vector-ref v 0))
        (y (vector-ref v 1)))
    (+ (expt (- 1 x) 2)
       (* 100 (expt (- y (* x x)) 2)))))

(gradient-descent rosenbrock
                 (vector<float64> 0.0 0.0)
                 0.001
                 10000)  ; => (vector<float64> 0.99... 0.98...) (approx [1, 1])
```

**Quantum Teleportation**:

```scheme
(: quantum-teleport (-> (qstate 1) (qstate 1)))
(define (quantum-teleport input-state)
  (let* ((alice-aux (qreg-new 1 |0>))
         (bob-qubit (qreg-new 1 |0>))
         ;; Create entangled pair for Alice and Bob
         (_ (qreg-apply hadamard 0 alice-aux))
         (_ (qreg-apply (controlled-not 0 0) alice-aux bob-qubit))
         ;; Alice's system now consists of the input state and her auxiliary
         (alice-system (qreg-join input-state alice-aux))
         ;; Alice applies her operations
         (_ (qreg-apply (controlled-not 0 1) alice-system))
         (_ (qreg-apply hadamard 0 alice-system))
         ;; Alice measures her qubits
         (measurement (qreg-measure alice-system))
         ;; Classical communication (implied)
         ;; Bob applies corrections based on Alice's measurement
         (_ (when (bit-set? measurement 1)
              (qreg-apply pauli-x 0 bob-qubit)))
         (_ (when (bit-set? measurement 0)
              (qreg-apply pauli-z 0 bob-qubit))))
    ;; Bob's qubit now contains the teleported state
    bob-qubit))
```

**Bayesian Inference with Probabilistic Programming**:

```scheme
(: coin-flip-model (-> (list boolean) (distribution float64)))
(define (coin-flip-model observations)
  (let ((prior (beta-dist 1.0 1.0)))  ; Uniform prior over [0,1]
    (define (likelihood theta)
      (let ((p-heads theta)
            (p-tails (- 1.0 theta)))
        (product (map (lambda (obs)
                        (if obs p-heads p-tails))
                      observations))))
    (define (posterior-density theta)
      (* (pdf prior theta)
         (likelihood theta)))
    ;; Return posterior distribution approximated by MCMC
    (metropolis-hastings posterior-density
                        (lambda (theta) (normal-dist theta 0.1))
                        1000)))

(define results (coin-flip-model (list #t #t #t #f #t #t)))
(mean results)  ; => 0.75 (approximately)
```

This concludes the Eshkol Language Specification. For additional detail on specific topics, please refer to the referenced supplementary documentation.

