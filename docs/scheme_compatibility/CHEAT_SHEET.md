# Eshkol Scheme Compatibility Cheat Sheet

Last Updated: 2025-03-29

This cheat sheet provides a quick reference for Eshkol's Scheme compatibility features, including implemented functions, syntax, and examples.

## Core Data Types

| Type | Description | Example | Predicates |
|------|-------------|---------|------------|
| Boolean | True or false values | `#t`, `#f` | `boolean?` |
| Number | Integer or floating-point | `42`, `3.14` | `number?`, `integer?`, `real?` |
| Character | Unicode character | `#\a`, `#\space` | `char?` |
| String | Sequence of characters | `"hello"` | `string?` |
| Symbol | Identifier | `'symbol`, `'x` | `symbol?` |
| Pair | Cons cell | `(cons 1 2)`, `'(1 . 2)` | `pair?` |
| List | Linked list of pairs | `'(1 2 3)`, `(list 1 2 3)` | `list?`, `null?` |
| Vector | Array-like collection | `#(1 2 3)` | `vector?` |
| Procedure | Function | `(lambda (x) x)` | `procedure?` |

## List Operations

| Function | Description | Example | Status |
|----------|-------------|---------|--------|
| `cons` | Create a pair | `(cons 1 2)` | ✅ Implemented |
| `car` | Get first element of pair | `(car '(1 2))` | ✅ Implemented |
| `cdr` | Get rest of pair | `(cdr '(1 2))` | ✅ Implemented |
| `list` | Create a list | `(list 1 2 3)` | ⚠️ Planned |
| `length` | Get list length | `(length '(1 2 3))` | ⚠️ Planned |
| `append` | Concatenate lists | `(append '(1 2) '(3 4))` | ⚠️ Planned |
| `reverse` | Reverse a list | `(reverse '(1 2 3))` | ⚠️ Planned |
| `list-ref` | Get element at index | `(list-ref '(1 2 3) 1)` | ⚠️ Planned |
| `list-tail` | Get sublist starting at index | `(list-tail '(1 2 3) 1)` | ⚠️ Planned |
| `memq`, `memv`, `member` | Find element in list | `(memq 'a '(a b c))` | ⚠️ Planned |
| `assq`, `assv`, `assoc` | Find association in alist | `(assq 'a '((a 1) (b 2)))` | ⚠️ Planned |

## Type Predicates

| Function | Description | Example | Status |
|----------|-------------|---------|--------|
| `boolean?` | Check if value is boolean | `(boolean? #t)` | ⚠️ Planned |
| `number?` | Check if value is number | `(number? 42)` | ⚠️ Planned |
| `integer?` | Check if value is integer | `(integer? 42)` | ⚠️ Planned |
| `real?` | Check if value is real number | `(real? 3.14)` | ⚠️ Planned |
| `char?` | Check if value is character | `(char? #\a)` | ⚠️ Planned |
| `string?` | Check if value is string | `(string? "hello")` | ⚠️ Planned |
| `symbol?` | Check if value is symbol | `(symbol? 'x)` | ⚠️ Planned |
| `pair?` | Check if value is pair | `(pair? '(1 . 2))` | ⚠️ Planned |
| `list?` | Check if value is list | `(list? '(1 2 3))` | ⚠️ Planned |
| `null?` | Check if value is empty list | `(null? '())` | ⚠️ Planned |
| `vector?` | Check if value is vector | `(vector? #(1 2 3))` | ⚠️ Planned |
| `procedure?` | Check if value is procedure | `(procedure? (lambda (x) x))` | ⚠️ Planned |

## Equality Predicates

| Function | Description | Example | Status |
|----------|-------------|---------|--------|
| `eq?` | Check if values are identical | `(eq? 'a 'a)` | ⚠️ Planned |
| `eqv?` | Check if values are equivalent | `(eqv? 42 42)` | ⚠️ Planned |
| `equal?` | Check if values are structurally equal | `(equal? '(1 2) '(1 2))` | ⚠️ Planned |

## Higher-Order Functions

| Function | Description | Example | Status |
|----------|-------------|---------|--------|
| `map` | Apply function to each element | `(map square '(1 2 3))` | ⚠️ Planned |
| `for-each` | Apply function for side effects | `(for-each display '(1 2 3))` | ⚠️ Planned |
| `filter` | Select elements matching predicate | `(filter even? '(1 2 3 4))` | ⚠️ Planned |
| `fold-left` | Fold list from left to right | `(fold-left + 0 '(1 2 3))` | ⚠️ Planned |
| `fold-right` | Fold list from right to left | `(fold-right + 0 '(1 2 3))` | ⚠️ Planned |
| `compose` | Compose functions | `((compose square add1) 5)` | ⚠️ Planned |
| `curry` | Curry a function | `((curry +) 1 2)` | ⚠️ Planned |

## Control Flow

| Syntax | Description | Example | Status |
|--------|-------------|---------|--------|
| `if` | Conditional expression | `(if (> x 0) "positive" "non-positive")` | ✅ Implemented |
| `cond` | Multi-way conditional | `(cond ((> x 0) "positive") ((< x 0) "negative") (else "zero"))` | ✅ Implemented |
| `case` | Dispatch on value | `(case x ((1) "one") ((2) "two") (else "other"))` | ⚠️ Planned |
| `and` | Logical AND | `(and (> x 0) (< x 10))` | ✅ Implemented |
| `or` | Logical OR | `(or (< x 0) (> x 10))` | ✅ Implemented |
| `not` | Logical NOT | `(not (= x 0))` | ✅ Implemented |
| `when` | Conditional with multiple expressions | `(when (> x 0) (display x) (newline))` | ⚠️ Planned |
| `unless` | Negative conditional | `(unless (= x 0) (/ 1 x))` | ⚠️ Planned |

## Binding Constructs

| Syntax | Description | Example | Status |
|--------|-------------|---------|--------|
| `let` | Local binding | `(let ((x 1) (y 2)) (+ x y))` | ✅ Implemented |
| `let*` | Sequential local binding | `(let* ((x 1) (y (+ x 1))) (+ x y))` | ✅ Implemented |
| `letrec` | Recursive local binding | `(letrec ((even? (lambda (n) (if (zero? n) #t (odd? (- n 1))))) (odd? (lambda (n) (if (zero? n) #f (even? (- n 1))))) (even? 10))` | ✅ Implemented |
| `define` | Global binding | `(define x 42)` | ✅ Implemented |
| `set!` | Assignment | `(set! x 42)` | ✅ Implemented |

## Lambda Expressions

| Syntax | Description | Example | Status |
|--------|-------------|---------|--------|
| `lambda` | Create a procedure | `(lambda (x) (* x x))` | ✅ Implemented |
| `define` (procedure) | Define a procedure | `(define (square x) (* x x))` | ✅ Implemented |

## Iteration

| Syntax | Description | Example | Status |
|--------|-------------|---------|--------|
| `do` | General iteration | `(do ((i 0 (+ i 1))) ((= i 10) 'done) (display i))` | ⚠️ Planned |
| Tail recursion | Recursive iteration | `(define (sum-iter n acc) (if (zero? n) acc (sum-iter (- n 1) (+ n acc))))` | ✅ Implemented |

## Input/Output

| Function | Description | Example | Status |
|----------|-------------|---------|--------|
| `display` | Display a value | `(display "Hello")` | ✅ Implemented |
| `newline` | Output a newline | `(newline)` | ✅ Implemented |
| `read` | Read a value | `(read)` | ⚠️ Planned |
| `write` | Write a value | `(write '(1 2 3))` | ⚠️ Planned |

## Type System Integration

Eshkol extends Scheme with an optional static type system:

```scheme
;; Untyped Scheme function
(define (add x y)
  (+ x y))

;; Typed Eshkol function
(: add (-> number number number))
(define (add x y)
  (+ x y))

;; Inline type annotation
(define (add (: x number) (: y number)) : number
  (+ x y))

;; Separate type declaration
(: square (-> number number))
(define (square x)
  (* x x))
```

## Autodiff Integration

Eshkol extends Scheme with automatic differentiation:

```scheme
;; Define a function
(define (f x)
  (* x x))

;; Compute the derivative
(define df/dx (autodiff-forward f))

;; Evaluate the derivative at x=3
(df/dx 3)  ; => 6

;; Compute the gradient of a multivariate function
(define (g x y)
  (+ (* x x) (* y y)))

(define grad-g (autodiff-forward-gradient g))

;; Evaluate the gradient at (x,y)=(1,2)
(grad-g (vector 1 2))  ; => #(2 4)
```

## MCP Tools

Eshkol provides MCP tools for analyzing Scheme code:

```bash
# Analyze mutual recursion in a Scheme file
use_mcp_tool eshkol-tools analyze-scheme-recursion '{"filePath": "examples/mutual_recursion.esk", "detail": "detailed"}'

# Analyze lambda captures in a Scheme file
use_mcp_tool eshkol-tools analyze-lambda-captures '{"filePath": "examples/lambda_closure_test.esk", "detail": "detailed"}'

# Analyze bindings in a Scheme file
use_mcp_tool eshkol-tools analyze-bindings '{"filePath": "examples/binding_test.esk", "detail": "detailed"}'

# Visualize closure memory in a Scheme file
use_mcp_tool eshkol-tools visualize-closure-memory '{"filePath": "examples/closure_test.esk", "format": "mermaid"}'
```

## Example Files

Eshkol provides example files demonstrating Scheme features:

- [Type Predicates](../../examples/type_predicates.esk)
- [Equality Predicates](../../examples/equality_predicates.esk)
- [List Operations](../../examples/list_operations.esk)
- [Higher-Order Functions](../../examples/higher_order_functions.esk)
- [Function Composition](../../examples/function_composition.esk)
- [Mutual Recursion](../../examples/mutual_recursion.esk)

## Documentation

For more detailed information, see:

- [Progress Dashboard](PROGRESS_DASHBOARD.md)
- [Known Issues](KNOWN_ISSUES.md)
- [Implementation Plan](IMPLEMENTATION_PLAN.md)
- [Master Tracking](MASTER_TRACKING.md)
- [Implementation Roadmaps](roadmaps/)
