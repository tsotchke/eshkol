# Eshkol Programming Language - Complete Reference
## Version 1.0.0-foundation

**A Comprehensive Guide to Every Feature in Pure Eshkol**

---

## Contents

1. [Language Fundamentals](#1-language-fundamentals)
2. [Data Types and Literals](#2-data-types-and-literals)
3. [Variables and Definitions](#3-variables-and-definitions)
4. [Functions and Closures](#4-functions-and-closures)
5. [Local Bindings](#5-local-bindings)
6. [Control Flow](#6-control-flow)
7. [Lists and Pairs](#7-lists-and-pairs)
8. [Vectors and Tensors](#8-vectors-and-tensors)
9. [Strings and Characters](#9-strings-and-characters)
10. [Hash Tables](#10-hash-tables)
11. [Higher-Order Functions](#11-higher-order-functions)
12. [Automatic Differentiation](#12-automatic-differentiation)
13. [Type System](#13-type-system)
14. [Pattern Matching](#14-pattern-matching)
15. [Exception Handling](#15-exception-handling)
16. [Module System](#16-module-system)
17. [Memory Management](#17-memory-management)
18. [Macros](#18-macros)
19. [I/O Operations](#19-io-operations)
20. [System Operations](#20-system-operations)
21. [Standard Library](#21-standard-library)
22. [Complete Function Index](#22-complete-function-index)

---

## 1. Language Fundamentals

### 1.1 Comments

```scheme
; This is a single-line comment

;; By convention, use double semicolon for important comments

;;; Triple semicolon for section headers
;;; Like this one

(define x 42)  ; inline comments work too
```

### 1.2 Expression Evaluation

Everything in Eshkol is an expression that returns a value:

```scheme
42                    ; => 42
(+ 1 2)              ; => 3
(if #t 1 2)          ; => 1
(begin (display "hi") 5)  ; Prints "hi", returns 5
```

### 1.3 S-Expression Syntax

```scheme
; Function calls use prefix notation
(function-name arg1 arg2 arg3)

; Nested expressions
(+ (* 2 3) (* 4 5))  ; => 26

; Lists are built from cons cells
'(1 2 3)             ; A list literal
(list 1 2 3)         ; Constructed list
```

---

## 2. Data Types and Literals

### 2.1 Numbers

#### Integers (Exact)
```scheme
42                ; Positive integer
-17               ; Negative integer
0                 ; Zero
+99               ; Explicit positive sign
```

#### Floating-Point (Inexact)
```scheme
3.14              ; Decimal notation
-2.5              ; Negative float
1.0               ; Explicit decimal point
```

#### Scientific Notation
```scheme
1.5e10            ; 1.5 × 10^10
2.3E-5            ; 2.3 × 10^-5
1e4               ; 10000.0
```

### 2.2 Booleans

```scheme
#t                ; True
#f                ; False

; All values except #f are truthy
(if 0 "yes" "no")       ; => "yes" (0 is truthy)
(if '() "yes" "no")     ; => "yes" (() is truthy)
(if #f "yes" "no")      ; => "no" (only #f is false)
```

### 2.3 Characters

```scheme
#\a               ; Lowercase 'a'
#\A               ; Uppercase 'A'
#\0               ; Digit '0'

; Named characters
#\space           ; Space character
#\newline         ; Newline
#\tab             ; Tab
#\return          ; Carriage return

; Unicode by codepoint
#\x0041           ; 'A' by hex code
```

### 2.4 Strings

```scheme
"hello world"     ; Basic string
"multi\nline"     ; With newline escape
"tab\there"       ; With tab
"quote: \""       ; Escaped quote
"backslash: \\"   ; Escaped backslash
""                ; Empty string
```

### 2.5 Symbols

```scheme
'foo              ; Symbol foo
'hello-world      ; Multi-word symbol
'+                ; Even operators can be symbols
'cadr             ; Compound names
```

### 2.6 Lists

```scheme
'()               ; Empty list (nil)
'(1 2 3)          ; List of numbers
'(a b c)          ; List of symbols
'(1 "two" #t)     ; Mixed-type list

; Dotted pairs (improper lists)
'(1 . 2)          ; Pair of 1 and 2
'(1 2 . 3)        ; List ending in 3
```

### 2.7 Vectors

```scheme
#(1 2 3)          ; Vector literal
#(a b c)          ; Vector of symbols
#()               ; Empty vector
#(1 "two" #t)     ; Mixed-type vector
```

### 2.8 Null (Empty List)

```scheme
'()               ; Empty list
()                ; Also empty list (in quoted context)

(null? '())       ; => #t
(null? (list))    ; => #t
```

---

## 3. Variables and Definitions

### 3.1 Simple Variable Definition

```scheme
(define x 42)
(define name "Alice")
(define flag #t)
(define empty '())
```

### 3.2 Function Definitions

#### Basic Function
```scheme
(define (square n)
  (* n n))

(square 5)        ; => 25
```

#### Multiple Parameters
```scheme
(define (add a b)
  (+ a b))

(define (volume width height depth)
  (* width height depth))
```

#### Multiple Expressions in Body
```scheme
(define (greet name)
  (display "Hello, ")
  (display name)
  (newline)
  #t)             ; Returns #t after printing
```

### 3.3 Variadic Functions

#### Fixed + Rest Parameters
```scheme
(define (sum-with-base base . numbers)
  (+ base (fold + 0 numbers)))

(sum-with-base 100 1 2 3)  ; => 106
```

#### All Parameters as List
```scheme
(define (variadic-sum . args)
  (fold + 0 args))

(variadic-sum 1 2 3 4 5)  ; => 15
```

### 3.4 Internal Definitions

Functions can have internal definitions (transformed to `letrec` automatically):

```scheme
(define (outer x)
  (define (helper y)
    (+ y 1))
  (define z 10)
  (+ x (helper z)))

(outer 5)         ; => 16
```

---

## 4. Functions and Closures

### 4.1 Anonymous Functions (Lambda)

#### Basic Lambda
```scheme
(lambda (x) (* x x))

; Using immediately
((lambda (x) (* x 2)) 5)  ; => 10
```

#### Multi-Parameter Lambda
```scheme
(lambda (x y) (+ x y))
(lambda (a b c) (* a b c))
```

#### Lambda with Multiple Expressions
```scheme
(lambda (x)
  (display x)
  (newline)
  (* x x))
```

### 4.2 Closures (Lexical Scoping)

```scheme
; Closure capturing outer variable
(define (make-adder n)
  (lambda (x) (+ x n)))

(define add5 (make-adder 5))
(add5 10)         ; => 15
(add5 20)         ; => 25

; Closure capturing multiple variables
(define (make-multiplier a b)
  (lambda (x) (* x a b)))

(define mul6 (make-multiplier 2 3))
(mul6 7)          ; => 42
```

### 4.3 Variadic Lambdas

#### Fixed + Rest
```scheme
(lambda (x . rest)
  (cons x rest))

((lambda (x y . rest)
   (list x y rest))
 1 2 3 4 5)       ; => (1 2 (3 4 5))
```

#### All Args as List
```scheme
(lambda args
  (length args))

((lambda args args) 1 2 3)  ; => (1 2 3)
```

### 4.4 Higher-Order Functions

```scheme
; Function returning function
(define (compose f g)
  (lambda (x) (f (g x))))

(define (double x) (* x 2))
(define (square x) (* x x))
(define double-then-square (compose square double))

(double-then-square 3)  ; => 36 (square of 6)

; Function taking function
(define (apply-twice f x)
  (f (f x)))

(apply-twice square 2)  ; => 16
```

### 4.5 First-Class Functions

```scheme
; Store in variable
(define my-func (lambda (x) (+ x 1)))

; Pass as argument
(map (lambda (x) (* x x)) '(1 2 3))  ; => (1 4 9)

; Return from function
(define (make-incrementer n)
  (lambda (x) (+ x n)))

; Store in list
(define ops (list + - * /))
((car ops) 5 3)   ; => 8 (calls +)
```

---

## 5. Local Bindings

### 5.1 `let` - Parallel Bindings

```scheme
; Bindings evaluated in parallel
(let ((x 5)
      (y 10))
  (+ x y))        ; => 15

; Later bindings CANNOT see earlier ones
(let ((x 5)
      (y x))      ; Error! x not yet bound
  (+ x y))
```

### 5.2 `let*` - Sequential Bindings

```scheme
; Bindings evaluated left-to-right
(let* ((x 5)
       (y (+ x 1)))   ; y CAN use x
  (+ x y))            ; => 11

; Useful for dependent computations
(let* ((width 10)
       (height 20)
       (area (* width height)))
  area)               ; => 200
```

### 5.3 `letrec` - Recursive Bindings

```scheme
; All bindings visible to all values
(letrec ((even? (lambda (n)
                  (if (= n 0) #t (odd? (- n 1)))))
         (odd? (lambda (n)
                 (if (= n 0) #f (even? (- n 1))))))
  (even? 42))         ; => #t

; Useful for mutually recursive functions
(letrec ((f (lambda (x) (if (= x 0) 0 (g (- x 1)))))
         (g (lambda (x) (if (= x 0) 1 (f (- x 1))))))
  (f 5))              ; => 0
```

### 5.4 Named Let (Loop Construct)

```scheme
; Named let for iteration
(let loop ((n 10) (acc 0))
  (if (= n 0)
      acc
      (loop (- n 1) (+ acc n))))  ; => 55

; Factorial using named let
(define (factorial n)
  (let loop ((i n) (result 1))
    (if (= i 0)
        result
        (loop (- i 1) (* result i)))))

(factorial 6)     ; => 720
```

---

## 6. Control Flow

### 6.1 `if` Expressions

```scheme
; Basic if
(if (> 5 3) "yes" "no")  ; => "yes"

; No else clause (returns unspecified if false)
(if (positive? 10)
    (display "positive"))

; Nested if
(if (> x 0)
    (if (even? x) "positive even" "positive odd")
    "not positive")
```

### 6.2 `cond` - Multi-Way Conditional

```scheme
(cond ((< x 0) "negative")
      ((= x 0) "zero")
      ((> x 0) "positive"))

; With else clause
(cond ((< x 0) "negative")
      ((= x 0) "zero")
      (else "positive"))

; Multiple expressions per clause
(cond ((null? lst)
       (display "empty")
       0)
      (else
       (display "not empty")
       (length lst)))
```

### 6.3 `case` - Switch on Value

```scheme
(case (+ 1 1)
  ((1) "one")
  ((2) "two")
  ((3) "three")
  (else "other"))     ; => "two"

; Multiple datums per clause
(case (car '(a b c))
  ((x y z) "xyz")
  ((a b c) "abc")
  (else "other"))     ; => "abc"
```

### 6.4 `when` and `unless`

```scheme
; Execute expressions when test is true
(when (> x 0)
  (display "positive")
  (newline))

; Execute expressions when test is false  
(unless (null? lst)
  (display "list is not empty")
  (process lst))
```

### 6.5 `and` and `or` (Short-Circuit)

```scheme
; and returns first false or last value
(and #t #t #t)           ; => #t
(and #t #f #t)           ; => #f
(and 1 2 3)              ; => 3

; or returns first true or last value
(or #f #f #t)            ; => #t
(or #f #f #f)            ; => #f
(or 1 2 3)               ; => 1

; Short-circuit evaluation
(and (not (null? lst)) (car lst))  ; Safe - won't call car on empty
```

### 6.6 `begin` - Sequencing

```scheme
(begin
  (display "Hello")
  (newline)
  (display "World")
  42)                   ; => 42 (returns last value)
```

### 6.7 `do` - Iteration

```scheme
; Basic do loop
(do ((i 0 (+ i 1))
     (sum 0 (+ sum i)))
    ((= i 10) sum))     ; => 45

; Multiple variables
(do ((x 0 (+ x 1))
     (y 10 (- y 1)))
    ((>= x y) (cons x y)))  ; => (5 . 5)

; With body expressions
(do ((i 0 (+ i 1)))
    ((= i 5))
  (display i)
  (display " "))        ; Prints: 0 1 2 3 4
```

---

## 7. Lists and Pairs

### 7.1 Creating Lists and Pairs

```scheme
; Cons cell (pair)
(cons 1 2)              ; => (1 . 2)
(cons 1 '())            ; => (1)
(cons 1 (cons 2 '()))   ; => (1 2)

; List construction
(list 1 2 3)            ; => (1 2 3)
(list)                  ; => ()
(list 1)                ; => (1)

; List with custom tail (list*)
(list* 1 2 3 '(4 5))    ; => (1 2 3 4 5)
(list* 1 2 3)           ; => (1 2 . 3)

; Quoted lists
'(1 2 3)                ; => (1 2 3)
'()                     ; => ()
'(a b c)                ; => (a b c)
```

### 7.2 Accessing List Elements

```scheme
(define lst '(10 20 30 40 50))

; Basic accessors
(car lst)               ; => 10 (first element)
(cdr lst)               ; => (20 30 40 50) (rest of list)

; Compound accessors (2-level)
(cadr lst)              ; => 20 (second element)
(caddr lst)             ; => 30 (third element)
(cadddr lst)            ; => 40 (fourth element)

(cdar lst)              ; cdr of car
(cddr lst)              ; cdr of cdr
(caar lst)              ; car of car (for nested lists)

; Three-level accessors
(caaar lst)             ; car of car of car
(caadr lst)             ; car of car of cdr
(cadar lst)             ; car of cdr of car  
(caddr lst)             ; car of cdr of cdr
; ...and so on through cdddr

; Four-level accessors (all 16 combinations)
(caaaar lst) (caaadr lst) (caadar lst) (caaddr lst)
(cadaar lst) (cadadr lst) (caddar lst) (cadddr lst)
(cdaaar lst) (cdaadr lst) (cdadar lst) (cdaddr lst)
(cddaar lst) (cddadr lst) (cdddar lst) (cddddr lst)
```

### 7.3 Positional Accessors

```scheme
(define lst '(a b c d e f g h i j))

(first lst)             ; => a
(second lst)            ; => b
(third lst)             ; => c
(fourth lst)            ; => d
(fifth lst)             ; => e
(sixth lst)             ; => f
(seventh lst)           ; => g
(eighth lst)            ; => h
(ninth lst)             ; => i
(tenth lst)             ; => j
```

### 7.4 List Predicates

```scheme
(null? '())             ; => #t
(null? '(1))            ; => #f

(pair? '(1 . 2))        ; => #t
(pair? '(1 2 3))        ; => #t
(pair? 42)              ; => #f

(list? '(1 2 3))        ; => #t
(list? '())             ; => #t
(list? '(1 . 2))        ; => #f (improper list)
```

### 7.5 List Queries

```scheme
(define lst '(10 20 30 40 50))

(length lst)            ; => 5
(length '())            ; => 0

; Get nth element (0-indexed)
(list-ref lst 0)        ; => 10
(list-ref lst 2)        ; => 30

; Get sublist from position n
(list-tail lst 2)       ; => (30 40 50)
(list-tail lst 0)       ; => (10 20 30 40 50)
```

### 7.6 List Transformations

```scheme
; Append lists
(append '(1 2) '(3 4))         ; => (1 2 3 4)
(append '(a) '(b) '(c))        ; => (a b c)
(append '() '(1 2))            ; => (1 2)

; Reverse list
(reverse '(1 2 3))             ; => (3 2 1)
(reverse '())                  ; => ()

; Take first n elements
(take '(1 2 3 4 5) 3)          ; => (1 2 3)
(take '(1 2) 10)               ; => (1 2)

; Drop first n elements
(drop '(1 2 3 4 5) 2)          ; => (3 4 5)
(drop '(1 2) 0)                ; => (1 2)

; Filter by predicate
(filter even? '(1 2 3 4 5 6))  ; => (2 4 6)
(filter positive? '(-2 -1 0 1 2))  ; => (1 2)

; Partition by predicate
(partition even? '(1 2 3 4 5))
; => ((2 4) (1 3 5))
```

### 7.7 List Searching

```scheme
; Find element (returns sublist or #f)
(member 3 '(1 2 3 4 5))       ; => (3 4 5)
(member 'x '(a b c))          ; => #f

; Check membership (returns boolean)
(member? 3 '(1 2 3))          ; => #t
(member? 'z '(a b c))         ; => #f

; memq (uses eq? - pointer equality)
(memq 'b '(a b c))            ; => (b c)

; memv (uses eqv? - value equality)
(memv 2 '(1 2 3))             ; => (2 3)

; Association lists
(define alist '((a . 1) (b . 2) (c . 3)))

(assoc 'b alist)              ; => (b . 2)
(assoc 'z alist)              ; => #f

; assq (uses eq?)
(assq 'a alist)               ; => (a . 1)

; assv (uses eqv?)
(assv 'c alist)               ; => (c . 3)
```

### 7.8 List Generation

```scheme
; iota: 0 to n-1
(iota 5)                      ; => (0 1 2 3 4)

; iota-from: n numbers starting from start
(iota-from 5 10)              ; => (10 11 12 13 14)

; iota-step: custom start and step
(iota-step 5 0 2)             ; => (0 2 4 6 8)

; repeat: n copies of value
(repeat 4 'x)                 ; => (x x x x)

; range: start to end (exclusive)
(range 1 6)                   ; => (1 2 3 4 5)
(range 0 0)                   ; => ()

; zip: combine two lists
(zip '(a b c) '(1 2 3))       ; => ((a 1) (b 2) (c 3))
```

### 7.9 List Mutation

```scheme
(define pair (cons 1 2))

(set-car! pair 99)
pair                          ; => (99 . 2)

(set-cdr! pair 88)
pair                          ; => (99 . 88)
```

### 7.10 List Sorting

```scheme
; Sort with comparison function
(sort '(3 1 4 1 5 9 2 6) <)   ; => (1 1 2 3 4 5 6 9)
(sort '(3 1 4 1 5) >)         ; => (5 4 3 1 1)

; Sort with custom comparator
(define (by-length a b)
  (< (length a) (length b)))

(sort '((1 2 3) (4) (5 6)) by-length)
; => ((4) (5 6) (1 2 3))
```

---

## 8. Vectors and Tensors

### 8.1 Vector Creation

```scheme
; Vector literal
#(1 2 3)                      ; => #(1 2 3)
#()                           ; => #()

; vector function
(vector 1 2 3)                ; => #(1 2 3)
(vector 'a 'b 'c)             ; => #(a b c)

; make-vector
(make-vector 5 0)             ; => #(0 0 0 0 0)
(make-vector 3 #f)            ; => #(#f #f #f)
```

### 8.2 Vector Access

```scheme
(define v #(10 20 30 40 50))

(vector-length v)             ; => 5
(vector-ref v 0)              ; => 10
(vector-ref v 2)              ; => 30

; vref (alias for vector-ref on tensors)
(vref v 1)                    ; => 20
```

### 8.3 Vector Mutation

```scheme
(define v (vector 1 2 3))

(vector-set! v 1 99)
v                             ; => #(1 99 3)
```

### 8.4 Vector Conversions

```scheme
; Vector to list
(vector->list #(1 2 3))       ; => (1 2 3)

; List to vector
(list->vector '(4 5 6))       ; => #(4 5 6)
```

### 8.5 Tensor Creation

```scheme
; 1D tensor (vector)
(vector 1.0 2.0 3.0)          ; => #(1.0 2.0 3.0)
#(1.0 2.0 3.0)                ; Literal form

; 2D tensor (matrix)
(matrix 2 3                    ; 2 rows, 3 columns
        1.0 2.0 3.0
        4.0 5.0 6.0)          ; => 2x3 matrix

; Generic tensor
(tensor 2 2 2                  ; 2x2x2 tensor
        1.0 2.0 3.0 4.0
        5.0 6.0 7.0 8.0)
```

### 8.6 Tensor Generators

```scheme
; Zero-filled tensor
(zeros 3)                     ; => #(0.0 0.0 0.0)
(zeros 2 3)                   ; => 2x3 matrix of zeros

; One-filled tensor
(ones 4)                      ; => #(1.0 1.0 1.0 1.0)
(ones 3 3)                    ; => 3x3 matrix of ones

; Identity matrix
(eye 3)                       ; => 3x3 identity matrix

; Range of values
(arange 1 6)                  ; => #(1.0 2.0 3.0 4.0 5.0)
(arange 0 10)                 ; => #(0.0 1.0 ... 9.0)

; Linearly spaced values
(linspace 0 1 5)              ; => #(0.0 0.25 0.5 0.75 1.0)
(linspace -1 1 3)             ; => #(-1.0 0.0 1.0)
```

### 8.7 Tensor Access

```scheme
(define v #(10.0 20.0 30.0))
(define M (matrix 2 2  1.0 2.0
                       3.0 4.0))

; 1D access
(vref v 0)                    ; => 10.0
(vref v 2)                    ; => 30.0

; N-D access
(tensor-get M 0 0)            ; => 1.0
(tensor-get M 1 1)            ; => 4.0
```

### 8.8 Tensor Operations

```scheme
(define v1 #(1.0 2.0 3.0))
(define v2 #(4.0 5.0 6.0))

; Element-wise arithmetic
(tensor-add v1 v2)            ; => #(5.0 7.0 9.0)
(tensor-sub v1 v2)            ; => #(-3.0 -3.0 -3.0)
(tensor-mul v1 v2)            ; => #(4.0 10.0 18.0)
(tensor-div v1 v2)            ; => #(0.25 0.4 0.5)

; Dot product
(tensor-dot v1 v2)            ; => 32.0 (scalar)

; Matrix multiplication
(define A (matrix 2 2  1.0 2.0
                       3.0 4.0))
(define B (matrix 2 2  5.0 6.0
                       7.0 8.0))
(matmul A B)                  ; => 2x2 result matrix

; Transpose
(transpose (matrix 2 3  1.0 2.0 3.0
                        4.0 5.0 6.0))
; => 3x2 matrix

; Reshape
(reshape #(1.0 2.0 3.0 4.0 5.0 6.0) 2 3)
; => 2x3 matrix
```

### 8.9 Tensor Reductions

```scheme
(define v #(1.0 2.0 3.0 4.0))

(tensor-sum v)                ; => 10.0
(tensor-mean v)               ; => 2.5

; Custom reduction
(tensor-reduce-all v * 1.0)   ; => 24.0 (product)
```

### 8.10 Tensor Queries

```scheme
(define M (matrix 3 4  1 2 3 4
                       5 6 7 8
                       9 10 11 12))

(tensor-shape M)              ; => (3 4)
(tensor-rank M)               ; => 2
(tensor-size M)               ; => 12
```

---

## 9. Strings and Characters

### 9.1 String Creation

```scheme
"hello"                       ; String literal
""                            ; Empty string

; make-string
(make-string 5 #\a)           ; => "aaaaa"
(make-string 0 #\x)           ; => ""

; string function
(string #\h #\i)              ; => "hi"
```

### 9.2 String Access

```scheme
(define s "hello")

(string-length s)             ; => 5
(string-length "")            ; => 0

(string-ref s 0)              ; => #\h
(string-ref s 4)              ; => #\o

; Substring
(substring s 1 4)             ; => "ell"
(substring s 0 5)             ; => "hello"
```

### 9.3 String Operations

```scheme
; Concatenation
(string-append "hello" " " "world")  ; => "hello world"
(string-append "a" "b" "c")          ; => "abc"

; String modification
(define s (string-copy "hello"))
(string-set! s 0 #\H)
s                             ; => "Hello"
```

### 9.4 String Predicates

```scheme
(string? "hello")             ; => #t
(string? 'symbol)             ; => #f

(string=? "abc" "abc")        ; => #t
(string=? "abc" "ABC")        ; => #f

(string<? "abc" "abd")        ; => #t
(string>? "xyz" "abc")        ; => #t
(string<=? "abc" "abc")       ; => #t
(string>=? "abc" "abc")       ; => #t
```

### 9.5 String Utilities (stdlib)

```scheme
; Join with delimiter
(string-join '("a" "b" "c") ",")  ; => "a,b,c"

; Split by delimiter
(string-split "a,b,c" ",")    ; => ("c" "b" "a") note: reverse order
(string-split-ordered "a,b,c" ",")  ; => ("a" "b" "c")

; Trim whitespace
(string-trim "  hello  ")     ; => "hello"
(string-trim-left "  hi")     ; => "hi"
(string-trim-right "hi  ")    ; => "hi"

; Case conversion
(string-upcase "hello")       ; => "HELLO"
(string-downcase "WORLD")     ; => "world"

; Replace
(string-replace "hello world" "o" "0")  ; => "hell0 w0rld"

; Reverse
(string-reverse "abc")        ; => "cba"

; Repeat
(string-repeat "ab" 3)        ; => "ababab"

; Tests
(string-starts-with? "hello" "hel")    ; => #t
(string-ends-with? "hello" "lo")       ; => #t
(string-contains? "hello" "ll")        ; => #t

; Find index
(string-index "hello" "ll")   ; => 2
(string-index "abc" "x")      ; => -1

; Count occurrences
(string-count "hello" "l")    ; => 2
```

### 9.6 String Conversions

```scheme
; To number
(string->number "42")         ; => 42
(string->number "3.14")       ; => 3.14
(string->number "not-a-number")  ; => #f

; From number
(number->string 42)           ; => "42"
(number->string 3.14)         ; => "3.14"

; To symbol
(string->symbol "foo")        ; => 'foo

; From symbol
(symbol->string 'bar)         ; => "bar"

; To list of characters
(string->list "abc")          ; => (#\a #\b #\c)

; From list of characters
(list->string '(#\x #\y #\z))  ; => "xyz"
```

### 9.7 Character Operations

```scheme
(char? #\a)                   ; => #t
(char? "a")                   ; => #f

; Character comparisons
(char=? #\a #\a)              ; => #t
(char<? #\a #\b)              ; => #t
(char>? #\z #\a)              ; => #t

; Conversions
(char->integer #\A)           ; => 65
(integer->char 65)            ; => #\A
```

---

## 10. Hash Tables

### 10.1 Creating Hash Tables

```scheme
; Empty hash table
(make-hash-table)             ; => #<hash:0>

; Hash with initial key-value pairs
(hash 'a 1 'b 2 'c 3)         ; => #<hash:3>
(hash "key1" "val1" "key2" "val2")
```

### 10.2 Hash Table Access

```scheme
(define h (hash 'a 1 'b 2 'c 3))

; Get value by key
(hash-ref h 'a)               ; => 1
(hash-ref h 'z)               ; => #f (not found)
(hash-ref h 'z 'default)      ; => 'default

; Set key-value pair
(hash-set! h 'd 4)
(hash-ref h 'd)               ; => 4

; Check if key exists
(hash-has-key? h 'a)          ; => #t
(hash-has-key? h 'z)          ; => #f

; Remove key
(hash-remove! h 'b)
(hash-has-key? h 'b)          ; => #f
```

### 10.3 Hash Table Queries

```scheme
(define h (hash 'a 1 'b 2 'c 3))

; Count of entries
(hash-count h)                ; => 3

; Get all keys as list
(hash-keys h)                 ; => (a b c) or similar

; Get all values as list
(hash-values h)               ; => (1 2 3) or similar

; Clear all entries
(hash-clear! h)
(hash-count h)                ; => 0
```

### 10.4 Hash Table Predicate

```scheme
(hash-table? (make-hash-table))  ; => #t
(hash-table? '(a b c))           ; => #f
```

---

## 11. Higher-Order Functions

### 11.1 `map` - Transform Each Element

```scheme
; Single list
(map (lambda (x) (* x x)) '(1 2 3 4))  ; => (1 4 9 16)
(map square '(2 3 4))                   ; => (4 9 16)

; Multiple lists (parallel iteration)
(map + '(1 2 3) '(10 20 30))           ; => (11 22 33)
(map * '(1 2 3) '(4 5 6))              ; => (4 10 18)
(map list '(a b) '(1 2))               ; => ((a 1) (b 2))

; With closures
(define (scale n)
  (lambda (x) (* n x)))
(map (scale 10) '(1 2 3))              ; => (10 20 30)
```

### 11.2 `filter` - Select Elements

```scheme
(filter even? '(1 2 3 4 5 6))          ; => (2 4 6)
(filter positive? '(-2 -1 0 1 2))      ; => (1 2)
(filter null? '(() (1) () (2)))        ; => (() ())

; With lambda
(filter (lambda (x) (> x 5)) '(1 6 3 8 2 9))  ; => (6 8 9)
```

### 11.3 `fold` / `foldl` - Left Fold

```scheme
; Sum
(fold + 0 '(1 2 3 4))         ; => 10

; Product
(fold * 1 '(2 3 4))           ; => 24

; Build list in reverse
(fold cons '() '(1 2 3))      ; => (3 2 1)

; Count elements
(fold (lambda (acc x) (+ acc 1)) 0 '(a b c))  ; => 3

; Maximum
(fold max 0 '(3 7 2 9 1))     ; => 9
```

### 11.4 `fold-right` / `foldr` - Right Fold

```scheme
; Build list in order
(fold-right cons '() '(1 2 3))  ; => (1 2 3)

; Difference with fold
(fold - 0 '(1 2 3))           ; => -6  (((0-1)-2)-3)
(fold-right - 0 '(1 2 3))     ; => 2   (1-(2-(3-0)))
```

### 11.5 `for-each` - Side Effects

```scheme
; Print each element
(for-each display '(1 2 3))   ; Prints: 123

; Multiple expressions
(for-each (lambda (x)
            (display x)
            (newline))
          '("line1" "line2" "line3"))
```

### 11.6 `apply` - Apply Function to List

```scheme
; Apply to entire list
(apply + '(1 2 3))            ; => 6
(apply max '(3 1 4 1 5))      ; => 5

; Mix fixed args and list
(apply + 1 2 '(3 4 5))        ; => 15
(apply list 'a '(b c))        ; => (a b c)

; Apply cons to 2-element list
(apply cons '(1 2))           ; => (1 . 2)
```

### 11.7 Predicates on Lists

```scheme
; any: true if any element satisfies predicate
(any even? '(1 3 4 5))        ; => #t
(any odd? '(2 4 6))           ; => #f

; every: true if all elements satisfy predicate
(every positive? '(1 2 3))    ; => #t
(every even? '(2 4 5))        ; => #f

; find: first element satisfying predicate
(find even? '(1 3 4 6 8))     ; => 4
(find negative? '(1 2 3))     ; => #f

; count-if: count matching elements
(count-if even? '(1 2 3 4 5 6))  ; => 3
(count-if null? '(() (1) () ()))  ; => 3
```

---

## 12. Automatic Differentiation

### 12.1 `derivative` - First Derivative

```scheme
; Define function
(define (f x) (* x x))        ; f(x) = x²

; Compute derivative at point
(derivative f 5.0)            ; => 10.0 (f'(5) = 2*5)

; Higher-order: get derivative function
(define df (derivative f))
(df 3.0)                      ; => 6.0

; Chain of functions
(define (g x) (sin (* x x)))
(derivative g 1.0)            ; => cos(1) * 2 ≈ 1.0806

; Second derivative
(define (h x) (* x x x))      ; h(x) = x³
(define dh (derivative h))    ; h'(x) = 3x²
(define ddh (derivative dh))  ; h''(x) = 6x
(ddh 2.0)                     ; => 12.0
```

### 12.2 `gradient` - Multivariate Gradient

```scheme
; Function ℝ² → ℝ
(define (f v)
  (+ (* (vref v 0) (vref v 0))    ; x²
     (* (vref v 1) (vref v 1))))  ; + y²

; Gradient at point
(gradient f #(3.0 4.0))       ; => #(6.0 8.0) [∂f/∂x, ∂f/∂y]

; Higher-order form
(define grad-f (gradient f))
(grad-f #(1.0 1.0))           ; => #(2.0 2.0)

; Rosenbrock function
(define (rosenbrock v)
  (let ((x (vref v 0))
        (y (vref v 1)))
    (+ (* 100.0 (* (- y (* x x)) (- y (* x x))))
       (* (- 1.0 x) (- 1.0 x)))))

(gradient rosenbrock #(1.0 1.0))  ; => #(0.0 0.0) at minimum
```

### 12.3 `jacobian` - Jacobian Matrix

```scheme
; Vector function ℝ² → ℝ²
(define (polar-to-cartesian v)
  (let ((r (vref v 0))
        (theta (vref v 1)))
    (vector (* r (cos theta))
            (* r (sin theta)))))

; Jacobian matrix
(jacobian polar-to-cartesian #(1.0 0.0))
; => 2x2 matrix of partial derivatives

; General form for ℝⁿ → ℝᵐ
(define (vector-func v)
  (vector (* (vref v 0) (vref v 1))
          (+ (vref v 0) (vref v 1))))

(jacobian vector-func #(2.0 3.0))
```

### 12.4 `hessian` - Second Derivatives

```scheme
; Scalar function ℝ² → ℝ
(define (quadratic v)
  (+ (* (vref v 0) (vref v 0))
     (* (vref v 1) (vref v 1))))

; Hessian matrix (second partials)
(hessian quadratic #(1.0 1.0))
; => #((2.0 0.0) (0.0 2.0)) constant Hessian

; General scalar field
(define (f v)
  (+ (* (vref v 0) (vref v 0) (vref v 1))
     (sin (vref v 1))))

(hessian f #(1.0 1.0))
```

### 12.5 `divergence` - Vector Field Divergence

```scheme
; Vector field ℝ³ → ℝ³
(define (radial-field v)
  (vector (vref v 0)          ; F = (x, y, z)
          (vref v 1)
          (vref v 2)))

(divergence radial-field #(1.0 2.0 3.0))
; => 3.0 (∂x/∂x + ∂y/∂y + ∂z/∂z = 1+1+1)

; 2D example
(define (field-2d v)
  (vector (* 2.0 (vref v 0))  ; F = (2x, 3y)
          (* 3.0 (vref v 1))))

(divergence field-2d #(1.0 1.0))
; => 5.0 (∂(2x)/∂x + ∂(3y)/∂y = 2+3)
```

### 12.6 `curl` - Vector Field Curl (3D)

```scheme
; Rotating field
(define (rotation v)
  (vector (- 0.0 (vref v 1))  ; F = (-y, x, 0)
          (vref v 0)
          0.0))

(curl rotation #(1.0 1.0 0.0))
; => #(0.0 0.0 2.0) rotation around z-axis

; General 3D field
(define (field-3d v)
  (vector (* (vref v 1) (vref v 2))
          (* (vref v 0) (vref v 2))
          (* (vref v 0) (vref v 1))))

(curl field-3d #(1.0 1.0 1.0))
```

### 12.7 `laplacian` - Scalar Field Laplacian

```scheme
; Harmonic function
(define (harmonic v)
  (+ (* (vref v 0) (vref v 0))
     (* (vref v 1) (vref v 1))))  ; x² + y²

(laplacian harmonic #(1.0 1.0))
; => 4.0 (∂²f/∂x² + ∂²f/∂y² = 2+2)

; General scalar field
(define (potential v)
  (exp (+ (* (vref v 0) (vref v 0))
          (* (vref v 1) (vref v 1)))))

(laplacian potential #(0.0 0.0))
```

### 12.8 `directional-derivative` - Derivative in Direction

```scheme
(define (f v)
  (+ (* (vref v 0) (vref v 0))
     (* (vref v 1) (vref v 1))))

(define point #(1.0 1.0))
(define direction #(1.0 0.0))  ; Unit x-direction

(directional-derivative f point direction)
; => 2.0 (derivative in x-direction)
```

### 12.9 Gradient Descent Example

```scheme
; Loss function
(define (loss v)
  (+ (* (- (vref v 0) 3.0) (- (vref v 0) 3.0))
     (* (- (vref v 1) 4.0) (- (vref v 1) 4.0))))

; Gradient descent step
(define point #(0.0 0.0))
(define learning-rate 0.1)

(define grad (gradient loss point))
(define new-x (- (vref point 0) (* learning-rate (vref grad 0))))
(define new-y (- (vref point 1) (* learning-rate (vref grad 1))))
(define new-point (vector new-x new-y))

(display "Initial loss: ") (display (loss point)) (newline)
(display "New loss: ") (display (loss new-point)) (newline)
; Loss decreases toward minimum at (3, 4)
```

---

## 13. Type System

### 13.1 Type Annotations

#### Standalone Type Declaration
```scheme
(: x integer)
(define x 42)

(: square (-> integer integer))
(define (square n) (* n n))
```

#### Inline Parameter Types
```scheme
(define (add (x : integer) (y : integer))
  (+ x y))

(define (distance (x : real) (y : real))
  (sqrt (+ (* x x) (* y y))))
```

#### Return Type Annotations
```scheme
(define (get-name person) : string
  (hash-ref person 'name))

(define (compute-sum lst) : integer
  (fold + 0 lst))
```

#### Lambda Type Annotations
```scheme
(lambda ((x : integer)) (* x x))

(lambda ((x : real) (y : real)) : real
  (sqrt (+ (* x x) (* y y))))
```

### 13.2 Type Expressions

#### Primitive Types
```scheme
integer         ; 64-bit signed integer
int            ; Alias for integer
int64          ; Explicit 64-bit

real           ; Double-precision float
float          ; Alias for real
double         ; Alias for real
float64        ; Explicit double

boolean        ; Boolean type
bool           ; Alias
string         ; String type
str            ; Alias
char           ; Character type
symbol         ; Symbol type

null           ; Empty list type
nil            ; Alias
```

#### Compound Types
```scheme
(list integer)              ; List of integers
(vector real)               ; Vector of reals
(tensor float64)            ; Tensor of floats
(pair integer string)       ; Pair of int and string

; Function types
(-> integer integer)        ; int → int
(-> integer real real)      ; int → real → real
(-> (list integer) integer) ; (list int) → int
```

#### Polymorphic Types
```scheme
(forall (a) (list a))       ; Generic list
(forall (a b) (-> a b b))   ; Generic function
(forall (a) (-> a a))       ; Identity type
```

### 13.3 Type Aliases

```scheme
(define-type Point (pair real real))
(define-type Name string)
(define-type Age integer)

; Parameterized type alias
(define-type (Maybe a) (+ a null))
(define-type (List a) (list a))
```

### 13.4 Type Predicates

```scheme
; Numeric types
(number? 42)              ; => #t
(integer? 42)             ; => #t
(real? 3.14)              ; => #t
(exact? 42)               ; => #t
(inexact? 3.14)           ; => #t

; Other types
(boolean? #t)             ; => #t
(char? #\a)               ; => #t
(string? "hello")         ; => #t
(symbol? 'foo)            ; => #t
(null? '())               ; => #t
(pair? '(1 . 2))          ; => #t
(list? '(1 2 3))          ; => #t
(vector? #(1 2 3))        ; => #t
(procedure? (lambda (x) x))  ; => #t
(hash-table? (make-hash-table))  ; => #t
```

---

## 14. Pattern Matching

### 14.1 `match` Expression

```scheme
; Match against literals
(match 42
  (42 "found it")
  (_ "not found"))      ; => "found it"

; Match variables (bind values)
(match 42
  (x x))                ; => 42 (x binds to 42)

; Match lists
(match '(1 2 3)
  ((list x y z) (+ x y z))
  (_ 0))                ; => 6

; Match cons patterns
(match '(1 2 3)
  ((cons first rest) (list first rest))
  (_ '()))              ; => (1 (2 3))

; Match with wildcard
(match '(1 2 3 4)
  ((list a _ _ d) (+ a d))
  (_ 0))                ; => 5 (1+4, middle two ignored)

; Nested patterns
(match '((1 2) (3 4))
  ((list (list a b) (list c d)) (+ a b c d))
  (_ 0))                ; => 10
```

### 14.2 Match with Guards

```scheme
(match x
  ((? even?) "even")
  ((? odd?) "odd")
  (_ "not a number"))

; Combining patterns
(match lst
  ((list) "empty")
  ((list x) "one element")
  ((list x y) "two elements")
  (_ "many elements"))
```

### 14.3 Match with Or Patterns

```scheme
(match x
  ((or 1 2 3) "small")
  ((or 4 5 6) "medium")
  (_ "large"))
```

---

## 15. Exception Handling

### 15.1 `guard` - Exception Handler

```scheme
; Basic exception handling
(guard (e
        ((error? e) (display "Error occurred"))
        (else (display "Unknown exception")))
  (/ 1 0))              ; Division by zero

; Multiple handlers
(guard (exc
        ((error? exc) "error")
        ((type-error? exc) "type error")
        (else "unknown"))
  (risky-operation))

; Re-raising exceptions
(guard (e
        ((error? e)
         (display "Logging error...")
         (raise e)))    ; Re-raise
  (dangerous-code))
```

### 15.2 `raise` - Raise Exception

```scheme
; Raise with message
(raise (error "Something went wrong"))

; Conditional raising
(define (safe-divide a b)
  (if (= b 0)
      (raise (error "Division by zero"))
      (/ a b)))
```

---

## 16. Module System

### 16.1 `require` - Import Modules

```scheme
; Load standard library
(require stdlib)

; Load specific modules
(require core.functional.compose)
(require core.list.higher_order)
(require core.json)

; Multiple modules
(require core.strings core.io core.data.csv)
```

### 16.2 `provide` - Export Symbols

```scheme
; In a module file: my-module.esk
(provide add-squared multiply-squared)

; Private (not exported)
(define (helper x) (* x 2))

; Public (exported)
(define (add-squared x y)
  (+ (* x x) (* y y)))

(define (multiply-squared x y)
  (* (* x x) (* y y)))
```

### 16.3 `import` - Legacy File Import

```scheme
; Import by file path
(import "lib/utils.esk")
(import "../shared/common.esk")
```

---

## 17. Memory Management

### 17.1 `with-region` - Lexical Memory Regions

```scheme
; Anonymous region
(with-region
  (define data (iota 1000))
  (process data))
; Memory freed after region exits

; Named region
(with-region 'temp
  (define large-list (iota 10000))
  (compute large-list))

; Named region with size hint
(with-region ('cache 8192)
  (define cache-data (build-cache))
  (use-cache cache-data))
```

### 17.2 `owned` - Ownership Marker

```scheme
; Mark value as owned
(define resource (owned (allocate-resource)))

; Must be consumed before scope exit
(with-region
  (define data (owned (iota 100)))
  (process data))       ; OK - data consumed
```

### 17.3 `move` - Transfer Ownership

```scheme
(define x (owned (list 1 2 3)))
(define y (move x))
; x is now invalid, y owns the list

; Error if try to use x after move
; (display x)          ; Compile error!
```

### 17.4 `borrow` - Temporary Access

```scheme
(define data (owned (iota 100)))

(borrow data
  (display (length data))     ; OK - read-only access
  (car data))                 ; OK - but can't move

; data still owned after borrow
(process data)
```

### 17.5 `shared` - Reference Counting

```scheme
; Create shared (ref-counted) value
(define s (shared (list 1 2 3)))

; Can use multiple times
(display s)
(display s)
; Reference count tracks usage
```

### 17.6 `weak-ref` - Weak References

```scheme
(define strong (shared (iota 100)))
(define weak (weak-ref strong))

; weak doesn't prevent deallocation
; Can upgrade to strong ref when needed
```

---

## 18. Macros

### 18.1 `define-syntax` - Hygienic Macros

```scheme
; Simple macro
(define-syntax when
  (syntax-rules ()
    ((when test expr ...)
     (if test (begin expr ...)))))

(when (> x 0)
  (display "positive")
  (newline))

; Macro with literals
(define-syntax my-cond
  (syntax-rules (else)
    ((my-cond (else expr ...))
     (begin expr ...))
    ((my-cond (test expr ...) clause ...)
     (if test
         (begin expr ...)
         (my-cond clause ...)))))

; Macro with ellipsis (repetition)
(define-syntax let-values
  (syntax-rules ()
    ((let-values ((vars producer) ...) body ...)
     (call-with-values
       (lambda () producer)
       (lambda vars body ...)))
    ...))
```

---

## 19. I/O Operations

### 19.1 Output

```scheme
; Display value (no quotes)
(display "hello")             ; Prints: hello
(display 42)                  ; Prints: 42
(display '(1 2 3))            ; Prints: (1 2 3)

; Newline
(newline)                     ; Prints newline

; Write (Scheme semantics - with quotes)
(write "hello")               ; Prints: "hello"
(write 'symbol)               ; Prints: symbol

; Combined output
(display "Value: ")
(display result)
(newline)
```

### 19.2 Input

```scheme
; Read S-expression
(define input (read))

; Read line as string
(define line (read-line))

; Read character
(define ch (read-char))

; Peek at next character
(define next (peek-char))
```

### 19.3 Ports

```scheme
; Open files
(define in-port (open-input-file "data.txt"))
(define out-port (open-output-file "output.txt"))

; Read from port
(define line (read-line in-port))

; Write to port
(write-string "hello\n" out-port)
(write-line "world" out-port)
(write-char #\! out-port)

; Close ports
(close-port in-port)
(close-port out-port)

; Flush output
(flush-output-port out-port)
```

### 19.4 File Operations

```scheme
; Read entire file
(define content (read-file "data.txt"))

; Write file
(write-file "output.txt" "Hello, file!")

; Append to file
(append-file "log.txt" "New log entry\n")

; Check file existence
(file-exists? "myfile.txt")   ; => #t or #f

; File properties
(file-readable? "data.txt")   ; => #t or #f
(file-writable? "data.txt")   ; => #t or #f
(file-size "data.txt")        ; => size in bytes

; File operations
(file-delete "temp.txt")
(file-rename "old.txt" "new.txt")
```

### 19.5 Directory Operations

```scheme
; Check directory
(directory-exists? "/tmp")    ; => #t

; Create directory
(make-directory "mydir")

; Delete directory
(delete-directory "olddir")

; List contents
(directory-list "/tmp")       ; => list of filenames

; Working directory
(current-directory)           ; => "/path/to/current"
(set-current-directory! "/tmp")
```

---

## 20. System Operations

### 20.1 Environment Variables

```scheme
; Get environment variable
(getenv "PATH")               ; => path string or #f
(getenv "USER")               ; => username

; Set environment variable
(setenv "MY_VAR" "my_value")

; Unset environment variable
(unsetenv "MY_VAR")
```

### 20.2 System Calls

```scheme
; Execute shell command
(system "ls -l")              ; Returns exit code
(system "echo 'Hello'")       ; Prints and returns 0

; Sleep
(sleep 2)                     ; Sleep 2 seconds

; Current time
(current-seconds)             ; => Unix timestamp

; Exit program
(exit 0)                      ; Exit with code 0
(exit 1)                      ; Exit with code 1
```

### 20.3 Command-Line Arguments

```scheme
; Get command-line args as list
(define args (command-line))  ; => ("program" "arg1" "arg2" ...)

; Process arguments
(for-each (lambda (arg)
            (display "Arg: ")
            (display arg)
            (newline))
          (cdr args))         ; Skip program name
```

---

## 21. Standard Library

### 21.1 Functional Programming

#### Composition
```scheme
(require core.functional.compose)

(define (double x) (* 2 x))
(define (square x) (* x x))

; Compose two functions
(define f (compose square double))
(f 3)                         ; => 36 (square of 6)

; Compose three functions
(define g (compose3 car reverse list))

; Identity
(identity 42)                 ; => 42

; Constantly
(define always-5 (constantly 5))
(always-5 'anything)          ; => 5
```

#### Currying
```scheme
(require core.functional.curry)

; Curry binary function
(define curried-add (curry2 add))
(define add10 (curried-add 10))
(add10 5)                     ; => 15

; Partial application
(define times2 (partial2 * 2))
(times2 21)                   ; => 42

; Generic partial
(define add-many (partial + 1 2 3))
(add-many 4 5 6)              ; => 21
```

#### Flip
```scheme
(require core.functional.flip)

(define reversed-sub (flip -))
(reversed-sub 1 10)           ; => 9 (10 - 1)

((flip cons) '(2 3) 1)        ; => (1 2 3)
```

### 21.2 List Utilities

#### Compound Accessors
```scheme
(require core.list.compound)

(define nested '((1 2) (3 4) (5 6)))

(caar nested)                 ; => 1
(cadr nested)                 ; => (3 4)
(caddr nested)                ; => (5 6)
(caadr nested)                ; => 3

; All combinations available through cddddr
```

#### Higher-Order
```scheme
(require core.list.higher_order)

; map1, map2, map3 for 1, 2, 3 lists
(map1 square '(1 2 3))        ; => (1 4 9)
(map2 + '(1 2) '(10 20))      ; => (11 22)
(map3 (lambda (a b c) (+ a b c))
      '(1 2) '(10 20) '(100 200))  ; => (111 222)

; fold and fold-right
(fold + 0 '(1 2 3))           ; => 6
(fold-right cons '() '(1 2))  ; => (1 2)

; any and every
(any even? '(1 3 4))          ; => #t
(every positive? '(1 2 3))    ; => #t
```

#### Transformations
```scheme
(require core.list.transform)

(take '(1 2 3 4 5) 3)         ; => (1 2 3)
(drop '(1 2 3 4 5) 2)         ; => (3 4 5)
(append '(1 2) '(3 4))        ; => (1 2 3 4)
(reverse '(1 2 3))            ; => (3 2 1)

(filter even? '(1 2 3 4))     ; => (2 4)

; Unzip list of pairs
(unzip '((a 1) (b 2) (c 3)))  ; => ((a b c) (1 2 3))

; Partition by predicate
(partition even? '(1 2 3 4 5 6))
; => ((2 4 6) (1 3 5))
```

### 21.3 String Utilities

```scheme
(require core.strings)

(string-join '("a" "b" "c") "-")  ; => "a-b-c"
(string-split-ordered "a,b,c" ",")  ; => ("a" "b" "c")
(string-trim "  hello  ")     ; => "hello"
(string-upcase "hello")       ; => "HELLO"
(string-downcase "WORLD")     ; => "world"
(string-replace "hello" "l" "L")  ; => "heLLo"
(string-reverse "abc")        ; => "cba"
(string-repeat "ab" 3)        ; => "ababab"

(string-starts-with? "hello" "hel")  ; => #t
(string-ends-with? "hello" "lo")     ; => #t
(string-contains? "hello" "ll")      ; => #t

(string-index "hello" "ll")   ; => 2
(string-count "hello" "l")    ; => 2
```

### 21.4 JSON Support

```scheme
(require core.json)

; Parse JSON string
(define data (json-parse "{\"name\":\"Alice\",\"age\":30}"))
(json-get data "name")        ; => "Alice"
(json-get data "age")         ; => 30

; Stringify to JSON
(define h (hash "x" 1 "y" 2))
(json-stringify h)            ; => "{\"x\":1,\"y\":2}"

; File I/O
(json-write-file "data.json" h)
(define loaded (json-read-file "data.json"))

; Association list conversion
(define alist '(("a" . 1) ("b" . 2)))
(alist->json alist)           ; => JSON string
(alist->hash-table alist)     ; => hash table
```

### 21.5 CSV Support

```scheme
(require core.data.csv)

; Parse CSV string
(csv-parse "a,b,c\n1,2,3")    ; => (("a" "b" "c") ("1" "2" "3"))

; Parse CSV file
(csv-parse-file "data.csv")   ; => list of rows

; Generate CSV
(csv-stringify '(("a" "b") ("1" "2")))  ; => "a,b\n1,2"

; Write CSV file
(csv-write-file "out.csv" rows)
```

### 21.6 Base64 Encoding

```scheme
(require core.data.base64)

; Encode string
(base64-encode-string "hello")  ; => "aGVsbG8="

; Decode string
(base64-decode-string "aGVsbG8=")  ; => "hello"

; Encode bytes
(base64-encode '(72 101 108 108 111))  ; => "SGVsbG8="

; String/byte conversion
(string->bytes "hi")          ; => (104 105)
(bytes->string '(104 105))    ; => "hi"
```

### 21.7 Trampoline (Bounded Stack Recursion)

```scheme
(require core.control.trampoline)

; Trampolined even/odd for deep recursion
(define (t-even n)
  (if (= n 0)
      #t
      (lambda () (t-odd (- n 1)))))

(define (t-odd n)
  (if (= n 0)
      #f
      (lambda () (t-even (- n 1)))))

; Execute with constant stack space
(trampoline (lambda () (t-even 10000)))  ; => #t

; bounce and done helpers
(define (factorial-cps n k)
  (if (= n 0)
      (done (k 1))
      (bounce (lambda () (factorial-cps (- n 1)
                                        (lambda (r) (k (* n r))))))))
```

### 21.8 Math Library

```scheme
(require math.esk)

; Constants
pi                            ; => 3.141592653589793
e                             ; => 2.718281828459045
epsilon                       ; => 1e-15

; Linear algebra
(define M (matrix 3 3  1 2 3
                       4 5 6
                       7 8 9))
(det M 3)                     ; Determinant
(inv M 3)                     ; Inverse (or #f if singular)
(solve A b 3)                 ; Solve Ax=b

; Vector operations
(dot #(1.0 2.0 3.0) #(4.0 5.0 6.0))  ; => 32.0
(cross #(1.0 0.0 0.0) #(0.0 1.0 0.0))  ; => #(0 0 1)
(normalize #(3.0 4.0))        ; => #(0.6 0.8)

; Numerical methods
(integrate sin 0.0 pi 100)    ; Numerical integration
(newton (lambda (x) (- (* x x) 2))   ; Root finding
        (lambda (x) (* 2 x))
        1.0 1e-10 100)        ; => 1.414... (sqrt(2))

; Eigenvalues
(power-iteration M 3 100 1e-10)  ; Dominant eigenvalue

; Statistics
(variance #(1.0 2.0 3.0 4.0))  ; Variance
(std #(1.0 2.0 3.0 4.0))       ; Standard deviation
(covariance v1 v2)             ; Covariance
```

---

## 22. Complete Function Index

This section lists EVERY function, operator, and special form in Eshkol v1.0-foundation.

### 22.1 Special Forms (39)

```scheme
define                ; Variable/function definition
lambda                ; Anonymous function
if                    ; Conditional
cond                  ; Multi-way conditional
case                  ; Switch on value
match                 ; Pattern matching
let                   ; Parallel bindings
let*                  ; Sequential bindings
letrec                ; Recursive bindings
let-values            ; Multiple value bindings
let*-values           ; Sequential multiple values
do                    ; Iteration
begin                 ; Sequencing
and                   ; Short-circuit AND
or                    ; Short-circuit OR
when                  ; One-armed if (true)
unless                ; One-armed if (false)
quote                 ; Literal data (')
quasiquote            ; Template (`)
unquote               ; Escape (,)
unquote-splicing      ; Splice (,@)
set!                  ; Mutation
define-type           ; Type alias
define-syntax         ; Macro definition
import                ; File import
require               ; Module import
provide               ; Symbol export
extern                ; External function declaration
extern-var            ; External variable declaration
with-region           ; Memory region
owned                 ; Ownership marker
move                  ; Transfer ownership
borrow                ; Temporary access
shared                ; Reference counting
weak-ref              ; Weak reference
guard                 ; Exception handler
raise                 ; Raise exception
values                ; Multiple return values
call-with-values      ; Consume multiple values
```

### 22.2 Arithmetic Operators

```scheme
; Binary/variadic
(+ a b ...)           ; Addition
(- a b ...)           ; Subtraction
(* a b ...)           ; Multiplication
(/ a b ...)           ; Division

; Unary
(- x)                 ; Negation
(abs x)               ; Absolute value

; Integer operations
(quotient a b)        ; Integer division
(remainder a b)       ; Remainder
(modulo a b)          ; Modulo

; Min/max
(min a b ...)         ; Minimum
(max a b ...)         ; Maximum

; Exponentiation
(expt base exp)       ; Power
(pow base exp)        ; Alias for expt
```

### 22.3 Math Functions

```scheme
; Trigonometric
(sin x) (cos x) (tan x)
(asin x) (acos x) (atan x)

; Hyperbolic
(sinh x) (cosh x) (tanh x)

; Exponential and logarithmic
(exp x)               ; e^x
(log x)               ; Natural log
(sqrt x)              ; Square root

; Rounding
(floor x)             ; Round down
(ceiling x)           ; Round up
(truncate x)          ; Round toward zero
(round x)             ; Round to nearest
```

### 22.4 Comparison Operators

```scheme
(= a b ...)           ; Numeric equality
(< a b ...)           ; Less than
(> a b ...)           ; Greater than
(<= a b ...)          ; Less than or equal
(>= a b ...)          ; Greater than or equal

; Numeric predicates
(zero? n)             ; Is zero?
(positive? n)         ; Is positive?
(negative? n)         ; Is negative?
(odd? n)              ; Is odd?
(even? n)             ; Is even?
(exact? n)            ; Is exact?
(inexact? n)          ; Is inexact?
```

### 22.5 Equality Predicates

```scheme
(eq? a b)             ; Pointer equality
(eqv? a b)            ; Value equality
(equal? a b)          ; Structural equality (deep)
```

### 22.6 List Functions (60+)

```scheme
; Construction
(cons a b)
(list e ...)
(list* e ... tail)
(make-list n fill)

; Access
(car pair)
(cdr pair)
(caar x) (cadr x) (cdar x) (cddr x)
(caaar x) ... (cdddr x)       ; 3-level
(caaaar x) ... (cddddr x)     ; 4-level
(first lst) ... (tenth lst)   ; Positional

; Predicates
(null? x)
(pair? x)
(list? x)

; Queries
(length lst)
(list-ref lst n)
(list-tail lst n)
(count-if pred lst)
(find pred lst)

; Transformations
(append lst ...)
(reverse lst)
(take lst n)
(drop lst n)
(filter pred lst)
(partition pred lst)
(unzip pairs)

; Search
(member x lst)
(member? x lst)
(memq x lst)
(memv x lst)
(assoc key alist)
(assq key alist)
(assv key alist)

; Higher-order
(map proc lst ...)
(map1 proc lst)
(map2 proc lst1 lst2)
(map3 proc lst1 lst2 lst3)
(fold proc init lst)
(fold-right proc init lst)
(for-each proc lst)
(any pred lst)
(every pred lst)
(apply proc lst)

; Generation
(iota n)
(iota-from n start)
(iota-step n start step)
(repeat n x)
(range start end)
(zip lst1 lst2)

; Sorting
(sort lst less?)

; Mutation
(set-car! pair val)
(set-cdr! pair val)

; Conversion
(list->vector lst)
(vector->list vec)
```

### 22.7 String Functions (30+)

```scheme
; Construction
(string char ...)
(make-string k char)
(string-append str ...)
(substring str start end)

; Access
(string-length str)
(string-ref str k)
(string-set! str k char)

; Predicates
(string? obj)
(string=? s1 s2)
(string<? s1 s2)
(string>? s1 s2)
(string<=? s1 s2)
(string>=? s1 s2)

; Utilities (stdlib)
(string-join lst delim)
(string-split str delim)
(string-split-ordered str delim)
(string-trim str)
(string-trim-left str)
(string-trim-right str)
(string-upcase str)
(string-downcase str)
(string-replace str old new)
(string-reverse str)
(string-copy str)
(string-repeat str n)
(string-starts-with? str prefix)
(string-ends-with? str suffix)
(string-contains? str substr)
(string-index str substr)
(string-last-index str substr)
(string-count str substr)

; Conversions
(string->number str)
(number->string num)
(string->symbol str)
(symbol->string sym)
(string->list str)
(list->string chars)
```

### 22.8 Vector Functions

```scheme
(vector elem ...)
(make-vector k fill)
(vector-length vec)
(vector-ref vec k)
(vector-set! vec k val)
(vector->list vec)
(list->vector lst)
(vector-fill! vec fill)
```

### 22.9 Tensor Functions (25+)

```scheme
; Creation
(vector elem ...)
(matrix rows cols elem ...)
(tensor dim ... elem ...)
(zeros dim ...)
(ones dim ...)
(eye n)
(arange start end)
(linspace start end num)

; Access
(vref tensor idx)
(tensor-get tensor idx ...)
(tensor-set tensor val idx ...)

; Arithmetic
(tensor-add t1 t2)
(tensor-sub t1 t2)
(tensor-mul t1 t2)
(tensor-div t1 t2)
(tensor-dot t1 t2)
(matmul m1 m2)

; Transformations
(transpose tensor)
(reshape tensor dims)
(tensor-apply tensor func)

; Reductions
(tensor-sum tensor)
(tensor-mean tensor)
(tensor-reduce-all tensor func init)

; Queries
(tensor-shape tensor)
(tensor-rank tensor)
(tensor-size tensor)
```

### 22.10 Hash Table Functions

```scheme
(make-hash-table)
(hash key val ...)
(hash-ref table key [default])
(hash-set! table key val)
(hash-has-key? table key)
(hash-remove! table key)
(hash-keys table)
(hash-values table)
(hash-count table)
(hash-clear! table)
(hash-table? obj)
```

### 22.11 Automatic Differentiation

```scheme
(derivative func point)       ; Forward-mode AD
(derivative func)             ; Returns derivative function
(gradient func point)         ; Multivariate gradient
(jacobian func point)         ; Jacobian matrix
(hessian func point)          ; Hessian matrix (second derivatives)
(divergence func point)       ; Vector field divergence
(curl func point)             ; Vector field curl (3D)
(laplacian func point)        ; Scalar field Laplacian
(directional-derivative func point dir)  ; Directional derivative
```

### 22.12 Type Predicates (20+)

```scheme
(null? x)
(boolean? x)
(char? x)
(string? x)
(symbol? x)
(number? x)
(integer? x)
(real? x)
(complex? x)
(pair? x)
(list? x)
(vector? x)
(procedure? x)
(hash-table? x)
(input-port? x)
(output-port? x)
(port? x)
(eof-object? x)
```

### 22.13 I/O Functions

```scheme
; Output
(display obj [port])
(write obj [port])
(newline [port])
(write-char char [port])
(write-string str [port])
(write-line str [port])
(flush-output-port port)

; Input
(read [port])
(read-line [port])
(read-char [port])
(peek-char [port])

; Ports
(open-input-file filename)
(open-output-file filename)
(close-port port)
(current-input-port)
(current-output-port)

; Files
(read-file filename)
(write-file filename content)
(append-file filename content)
(file-exists? filename)
(file-readable? filename)
(file-writable? filename)
(file-delete filename)
(file-rename old new)
(file-size filename)

; Directories
(directory-exists? path)
(make-directory path)
(delete-directory path)
(directory-list path)
(current-directory)
(set-current-directory! path)
```

### 22.14 System Functions

```scheme
(getenv name)
(setenv name value)
(unsetenv name)
(system command)
(exit [code])
(sleep seconds)
(current-seconds)
(command-line)
```

### 22.15 Conversions

```scheme
(exact->inexact num)
(inexact->exact num)
(char->integer char)
(integer->char n)
```

---

## 23. Complete Code Examples

### 23.1 Fibonacci (Recursive)

```scheme
(define (fib n)
  (if (< n 2)
      n
      (+ (fib (- n 1)) (fib (- n 2)))))

(fib 10)              ; => 55
```

### 23.2 Factorial with Named Let

```scheme
(define (factorial n)
  (let loop ((i n) (acc 1))
    (if (= i 0)
        acc
        (loop (- i 1) (* acc i)))))

(factorial 6)         ; => 720
```

### 23.3 Quicksort

```scheme
(define (qsort lst)
  (if (null? lst)
      '()
      (let ((pivot (car lst))
            (rest (cdr lst)))
        (append
          (qsort (filter (lambda (x) (< x pivot)) rest))
          (cons pivot
            (qsort (filter (lambda (x) (>= x pivot)) rest)))))))

(qsort '(3 1 4 1 5 9 2 6))  ; => (1 1 2 3 4 5 6 9)
```

### 23.4 Church Numerals

```scheme
(define church-zero (lambda (f) (lambda (x) x)))
(define (church-succ n)
  (lambda (f) (lambda (x) (f ((n f) x)))))
(define (church-add m n)
  (lambda (f) (lambda (x) ((m f) ((n f) x)))))
(define (church->int n)
  ((n (lambda (x) (+ x 1))) 0))

(define church-2 (church-succ (church-succ church-zero)))
(define church-3 (church-succ church-2))
(define church-5 (church-add church-2 church-3))

(church->int church-5)        ; => 5
```

### 23.5 Y Combinator

```scheme
(define Z
  (lambda (f)
    ((lambda (x) (f (lambda (v) ((x x) v))))
     (lambda (x) (f (lambda (v) ((x x) v)))))))

(define fact-gen
  (lambda (fact)
    (lambda (n)
      (if (= n 0) 1 (* n (fact (- n 1)))))))

((Z fact-gen) 6)      ; => 720
```

### 23.6 State Machine (Closure with Mutation)

```scheme
(define (make-counter initial)
  (let ((count initial))
    (lambda (op)
      (cond ((eq? op 'inc) (set! count (+ count 1)) count)
            ((eq? op 'dec) (set! count (- count 1)) count)
            ((eq? op 'get) count)
            ((eq? op 'reset) (set! count initial) count)))))

(define c (make-counter 0))
(c 'inc)              ; => 1
(c 'inc)              ; => 2
(c 'get)              ; => 2
(c 'reset)            ; => 0
```

### 23.7 Neural Network Forward Pass

```scheme
(require stdlib)

; Activation function
(define (sigmoid x)
  (/ 1.0 (+ 1.0 (exp (- 0.0 x)))))

; Layer forward pass
(define (layer-forward weights biases activation input)
  (let ((z (map + (map (lambda (row) (fold + 0.0 (map * row input)))
                       weights)
                  biases)))
    (map activation z)))

; Simple 2-layer network
(define w1 (list (list 0.5 0.3) (list 0.2 0.8)))
(define b1 (list 0.1 -0.1))
(define w2 (list (list 0.6 0.4)))
(define b2 (list 0.05))

(define input (list 1.0 0.5))
(define hidden (layer-forward w1 b1 sigmoid input))
(define output (layer-forward w2 b2 sigmoid hidden))

(display output)      ; Network prediction
```

### 23.8 Gradient Descent

```scheme
(require stdlib)

(define (loss v)
  (+ (* (- (vref v 0) 3.0) (- (vref v 0) 3.0))
     (* (- (vref v 1) 4.0) (- (vref v 1) 4.0))))

(define (gradient-step point learning-rate)
  (let ((grad (gradient loss point)))
    (vector (- (vref point 0) (* learning-rate (vref grad 0)))
            (- (vref point 1) (* learning-rate (vref grad 1))))))

; Training loop
(define (train point steps lr)
  (if (= steps 0)
      point
      (train (gradient-step point lr) (- steps 1) lr)))

(define start #(0.0 0.0))
(define result (train start 100 0.1))

(display "Final point: ") (display result) (newline)
(display "Final loss: ") (display (loss result)) (newline)
; Converges toward (3, 4)
```

---

## 24. Full Language Syntax Reference

### 24.1 Lexical Syntax

```scheme
; Comments
; Single line comment

; Identifiers
foo
hello-world
some->thing
+
list->vector

; Numbers
42                    ; Integer
3.14                  ; Float
1.5e10                ; Scientific notation
-17                   ; Negative

; Strings
"hello"
"multi\nline"
"escape: \""

; Characters
#\a                   ; Letter
#\space              ; Named
#\x0041              ; Unicode

; Booleans
#t #f

; Empty list
() '()

; Vectors
#(1 2 3)

; Quote
'expression
`template
,unquote
,@splice
```

### 24.2 Expression Syntax

```scheme
; Self-evaluating
42
"string"
#t

; Variable reference
variable-name

; Function call
(function arg1 arg2 ...)

; Lambda
(lambda (params...) body...)
(lambda (param . rest) body)
(lambda params body)          ; All args as list

; Let forms
(let ((var val) ...) body...)
(let* ((var val) ...) body...)
(letrec ((var val) ...) body...)
(let name ((var init) ...) body...)  ; Named let

; Conditionals
(if test then else)
(cond (test expr) ... (else expr))
(case key (datum expr) ... (else expr))
(match expr (pattern body) ...)
(when test expr ...)
(unless test expr ...)
(and expr ...)
(or expr ...)

; Sequencing
(begin expr ...)

; Iteration
(do ((var init step) ...) (test result) body...)

; Definition
(define name value)
(define (name params...) body...)
(define (name . rest) body)

; Mutation
(set! name value)

; Quotation
(quote datum)
(quasiquote template)
(unquote expr)
(unquote-splicing expr)

; Type annotation
(: name type)
(variable : type)

; Exception
(guard (var (test handler) ...) body...)
(raise exception)

; Multiple values
(values expr ...)
(call-with-values producer consumer)
(let-values (((vars ...) producer) ...) body...)

; Memory
(with-region body...)
(owned expr)
(move var)
(borrow var body...)
(shared expr)

; Module
(require module ...)
(provide name ...)
(import "file.esk")

; Macro
(define-syntax name (syntax-rules (literals...) (pattern template) ...))

; External
(extern return-type name param-types...)
```

---

## Complete Coverage Summary

This reference documents the **ENTIRE** Eshkol language v1.0-foundation:

✅ **39 Special Forms** - All control structures and binding forms  
✅ **300+ Built-in Functions** - Every operator, math function, and list operation  
✅ **Complete Standard Library** - All modules with every exported function  
✅ **Automatic Differentiation** - All 8 AD operators with examples  
✅ **Type System** - Full HoTT-based gradual typing  
✅ **Pattern Matching** - Complete match syntax  
✅ **Memory Management** - OALR with ownership tracking  
✅ **Module System** - require/provide with visibility  
✅ **Exception Handling** - guard/raise mechanism  
✅ **Hash Tables** - Complete API  
✅ **Macros** - Hygienic syntax-rules  
✅ **I/O System** - Files, ports, directories  
✅ **System Integration** - Environment, processes, time  

**Every feature is documented with working Eshkol code examples.**

---

*This reference was created by comprehensive analysis of the complete Eshkol compiler source code, test suite, and standard library implementations.*