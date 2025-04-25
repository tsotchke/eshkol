# Eshkol Language Cheat Sheet

## 1. LANGUAGE OVERVIEW

Eshkol is a high-performance LISP-like language designed for scientific computing and AI applications. It features:

- Scheme-compatible syntax (R5RS/R7RS-small)
- Optional static type system
- Automatic differentiation
- SIMD vector operations
- High performance via compilation to C

## 2. SYNTAX BASICS

### Comments
```scheme
;; Single line comment
```

### S-expressions
Everything in Eshkol is an S-expression: `(operator operand1 operand2 ...)`

### Program Structure
```scheme
;; Define a function
(define (function-name param1 param2)
  body-expression)

;; Main function (program entry point)
(define (main)
  (printf "Hello, Eshkol!\n")
  0)  ; Return 0 to indicate success
```

## 3. DATA TYPES

### Basic Types
- **Boolean**: `#t`, `#f`
- **Number**: Integer (`42`), Floating-point (`3.14`)
- **Character**: `#\a`, `#\space`
- **String**: `"hello"`
- **Symbol**: `'symbol`, `'x`
- **Pair**: `(cons 1 2)`, `'(1 . 2)`
- **List**: `'(1 2 3)`, `(list 1 2 3)`
- **Vector**: `#(1 2 3)`
- **Procedure**: `(lambda (x) x)`

### Type Predicates
```scheme
(boolean? val)   ; Check if val is a boolean
(number? val)    ; Check if val is a number
(integer? val)   ; Check if val is an integer
(real? val)      ; Check if val is a real number
(char? val)      ; Check if val is a character
(string? val)    ; Check if val is a string
(symbol? val)    ; Check if val is a symbol
(pair? val)      ; Check if val is a pair
(list? val)      ; Check if val is a list
(null? val)      ; Check if val is an empty list
(vector? val)    ; Check if val is a vector
(procedure? val) ; Check if val is a procedure
```

### Equality
```scheme
(eq? a b)     ; Identity comparison (same object)
(eqv? a b)    ; Equivalent values (same type and value)
(equal? a b)  ; Structurally equal (recursive equality)
```

## 4. CONTROL FLOW

### Conditionals
```scheme
;; If expression
(if condition
    true-expression
    false-expression)

;; Multi-way conditional
(cond
  (condition1 result1)
  (condition2 result2)
  ...
  (else default-result))

;; Case expression
(case value
  ((value1) result1)
  ((value2 value3) result2)
  ...
  (else default-result))
```

### Logical Operators
```scheme
(and expr1 expr2 ...)  ; Short-circuit AND
(or expr1 expr2 ...)   ; Short-circuit OR
(not expr)             ; Logical NOT
```

### When/Unless
```scheme
(when condition         ; Like if with no else clause
  expr1 expr2 ...)

(unless condition       ; Like (when (not condition) ...)
  expr1 expr2 ...)
```

## 5. FUNCTIONS & LAMBDAS

### Function Definition
```scheme
;; Basic function definition
(define (square x)
  (* x x))

;; Alternative syntax
(define square
  (lambda (x) (* x x)))
```

### Lambda Expressions
```scheme
;; Anonymous function
(lambda (param1 param2 ...)
  body-expression)

;; Example usage
((lambda (x) (* x x)) 5)  ; => 25
```

### Closures
```scheme
;; Function that returns a function
(define (make-adder n)
  (lambda (x) (+ x n)))

(define add5 (make-adder 5))
(add5 10)  ; => 15
```

### Function Composition
```scheme
;; Compose two functions
(define (compose f g)
  (lambda (x)
    (f (g x))))

;; Example
(define (square x) (* x x))
(define (double x) (+ x x))
(define square-then-double (compose double square))
(square-then-double 3)  ; => 18
```

## 6. BINDING CONSTRUCTS

### Variable Binding
```scheme
;; Simple binding
(let ((var1 expr1)
      (var2 expr2))
  body-expression)

;; Sequential binding (each binding sees previous ones)
(let* ((x 1)
       (y (+ x 1)))
  (+ x y))  ; => 3

;; Recursive binding
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

### Assignment
```scheme
;; Change a variable's value
(set! variable-name new-value)
```

## 7. TYPE SYSTEM

### Type Annotations

#### Separate Type Declaration
```scheme
;; Declare function type separately
(: square (-> number number))
(define (square x)
  (* x x))
```

#### Inline Parameter Types
```scheme
;; Annotate parameters inline
(define (add x : number y : number)
  (+ x y))
```

#### Return Type Annotation
```scheme
;; Annotate return type
(define (add x y) : number
  (+ x y))
```

#### Combined Annotations
```scheme
;; Fully annotated function
(define (add x : number y : number) : number
  (+ x y))
```

### Complex Types
```scheme
;; Function type
(: map (-> (-> a b) (list a) (list b)))

;; Parametric types
(: id (forall (a) (-> a a)))

;; List types
(: numbers (list number))

;; Vector types
(define (compute-distance point1 : vector<float> point2 : vector<float>) : float
  (sqrt (+ (expt (- (vector-ref point2 0) (vector-ref point1 0)) 2)
           (expt (- (vector-ref point2 1) (vector-ref point1 1)) 2))))
```

## 8. LIST OPERATIONS

### Basic List Operations
```scheme
(cons a b)                ; Create a pair
(car pair)                ; Get first element of pair
(cdr pair)                ; Get rest of pair
(list a b c)              ; Create a list
(length lst)              ; Get list length
(append lst1 lst2)        ; Concatenate lists
(reverse lst)             ; Reverse a list
(list-ref lst n)          ; Get element at index
(list-tail lst n)         ; Get sublist starting at index
```

### List Searching
```scheme
(memq obj lst)            ; Find object (using eq?)
(memv obj lst)            ; Find object (using eqv?)
(member obj lst)          ; Find object (using equal?)
(assq key alist)          ; Find association (using eq?)
(assv key alist)          ; Find association (using eqv?)
(assoc key alist)         ; Find association (using equal?)
```

## 9. HIGHER-ORDER FUNCTIONS

```scheme
;; Apply function to each element
(map proc lst)

;; Apply function for side effects
(for-each proc lst)

;; Filter elements
(filter pred lst)

;; Fold from left
(fold-left proc init lst)

;; Fold from right
(fold-right proc init lst)
```

## 10. ITERATION

### Recursive Iteration
```scheme
;; Tail-recursive factorial
(define (factorial n)
  (define (fact-iter n acc)
    (if (zero? n)
        acc
        (fact-iter (- n 1) (* n acc))))
  (fact-iter n 1))
```

### Do Loop
```scheme
;; General iteration
(do ((var1 init1 step1)
     (var2 init2 step2)
     ...)
    (test result)
  body-expressions)

;; Example (sum of numbers 0 to 9)
(do ((i 0 (+ i 1))
     (sum 0 (+ sum i)))
    ((= i 10) sum)
  (display i))
```

## 11. INPUT/OUTPUT

```scheme
(display obj)             ; Display a value
(newline)                 ; Output a newline
(printf format-str args)  ; Formatted output
(read)                    ; Read a value
(write obj)               ; Write a value
```

## 12. SPECIAL FEATURES

### Automatic Differentiation

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

### Vector Operations

```scheme
;; Create vectors
(define v1 (vector 1 2 3))
(define v2 (vector 4 5 6))

;; Vector operations (SIMD-accelerated)
(vector-add v1 v2)        ; Element-wise addition
(vector-mul v1 v2)        ; Element-wise multiplication
(vector-dot v1 v2)        ; Dot product
(vector-scale v1 2)       ; Scale by scalar
(vector-magnitude v1)     ; Vector magnitude
```

## 13. COMMON PATTERNS

### Mutual Recursion
```scheme
;; Mutually recursive functions
(define (even? n)
  (if (zero? n)
      #t
      (odd? (- n 1))))

(define (odd? n)
  (if (zero? n)
      #f
      (even? (- n 1))))
```

### Function Composition Chain
```scheme
;; Chain multiple functions
(define (compose-n . fns)
  (fold-right compose identity fns))

(define pipeline (compose-n f g h))
(pipeline x)  ; => (f (g (h x)))
```

### Curry & Partial Application
```scheme
;; Curry a function
(define (curry f)
  (lambda (x)
    (lambda (y)
      (f x y))))

;; Partial application
(define (partial f . args)
  (lambda rest-args
    (apply f (append args rest-args))))
```

## 14. COMPILATION

Eshkol compiles to C for high performance:

```bash
# Compile an Eshkol file
eshkol file.esk

# Run a compiled program
./file
```
