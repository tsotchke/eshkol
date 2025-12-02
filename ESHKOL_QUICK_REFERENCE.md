# Eshkol Quick Reference Card

## Basics

```scheme
;; Variables
(define x 42)
(define pi 3.14159)

;; Functions
(define (square x) (* x x))
(define (add a b) (+ a b))

;; Lambdas
(lambda (x) (* x 2))

;; Let bindings
(let ((x 5) (y 10)) (+ x y))
```

## Control Flow

```scheme
(if condition then-expr else-expr)

(cond ((test1) result1)
      ((test2) result2)
      (else default))

(when test expr...)
(unless test expr...)

(and a b c)  ;; short-circuit
(or a b c)   ;; short-circuit
```

## Lists

```scheme
(list 1 2 3)              ;; create list
(cons 1 (list 2 3))       ;; prepend
(car lst)                 ;; first element
(cdr lst)                 ;; rest of list
(cadr lst)                ;; second element (car (cdr lst))
(caddr lst)               ;; third element
(length lst)              ;; list length
(append lst1 lst2)        ;; concatenate
(reverse lst)             ;; reverse
(list-ref lst n)          ;; nth element
(member x lst)            ;; find element
(assoc key alist)         ;; lookup in assoc list
```

## Higher-Order Functions

```scheme
(map f lst)               ;; apply f to each element
(filter pred lst)         ;; select matching elements
(fold f init lst)         ;; reduce left
(fold-right f init lst)   ;; reduce right
(for-each f lst)          ;; side effects
(any pred lst)            ;; any match?
(every pred lst)          ;; all match?
(find pred lst)           ;; first match
```

## Vectors & Tensors

```scheme
(vector 1 2 3)            ;; create vector
(vref v i)                ;; get element
(vector-set! v i x)       ;; set element
(make-vector n val)       ;; n copies of val
(vector-length v)         ;; length

;; Tensor creation
(zeros m n)               ;; m×n zeros
(ones m n)                ;; m×n ones
(eye n)                   ;; n×n identity
(arange n)                ;; 0 to n-1
(linspace a b n)          ;; n values from a to b
(reshape v m n)           ;; reshape to m×n

;; Operations
(tensor-add a b)          ;; element-wise add
(tensor-dot a b)          ;; dot product
(matmul A B)              ;; matrix multiply
(transpose M)             ;; transpose
(norm v)                  ;; L2 norm
(tensor-sum v)            ;; sum all
(tensor-mean v)           ;; mean
```

## Automatic Differentiation

```scheme
;; Symbolic (compile-time)
(diff (* x x) x)          ;; → (* 2 x)

;; Forward-mode (dual numbers)
(derivative f x)          ;; df/dx at x

;; Reverse-mode (computation graphs)
(gradient f v)            ;; ∇f at v
(jacobian f v)            ;; Jacobian matrix
(hessian f v)             ;; Hessian matrix

;; Vector calculus
(divergence F v)          ;; ∇·F at v
(curl F v)                ;; ∇×F at v (3D)
(laplacian f v)           ;; ∇²f at v
(directional-derivative f v dir)
```

## Math Functions

```scheme
;; Arithmetic
+ - * / abs floor ceiling round truncate
modulo remainder quotient gcd lcm min max expt

;; Trigonometric
sin cos tan asin acos atan atan2
sinh cosh tanh asinh acosh atanh

;; Exponential
exp log log10 log2 sqrt pow
```

## Type Predicates

```scheme
(null? x)    (pair? x)    (list? x)
(number? x)  (integer? x) (real? x)
(string? x)  (char? x)    (symbol? x)
(vector? x)  (procedure? x)
(zero? x)    (positive? x) (negative? x)
(even? x)    (odd? x)
```

## I/O

```scheme
(display x)               ;; print value
(newline)                 ;; print newline
(printf fmt args...)      ;; formatted print

;; Files
(open-input-file path)
(open-output-file path)
(read-line port)
(write-string str port)
(close-port port)
```

## C Interop

```scheme
(extern return-type name param-types...)
(extern void printf char* ...)
(extern double sin double)

;; With aliasing
(extern void log :real printf char* ...)
```

## Standard Library (stdlib.esk)

```scheme
;; Combinators
(compose f g)             ;; (f (g x))
(identity x)              ;; x
(constantly x)            ;; (lambda (_) x)
(flip f)                  ;; swap arguments
(negate pred)             ;; (not (pred x))

;; Currying
(curry2 f)                ;; curry 2-arg function
(curry3 f)                ;; curry 3-arg function
(partial2 f x)            ;; fix first arg

;; List utilities
(iota n)                  ;; (0 1 ... n-1)
(iota-from n start)       ;; (start start+1 ... start+n-1)
(repeat n x)              ;; n copies of x
(sort lst less?)          ;; merge sort
(partition pred lst)      ;; split by predicate
(all? pred lst)           ;; all satisfy?
(none? pred lst)          ;; none satisfy?
(count-if pred lst)       ;; count matches
```

## Math Library (math.esk)

```scheme
;; Linear algebra
(det M n)                 ;; determinant
(inv M n)                 ;; inverse
(solve A b n)             ;; solve Ax=b
(cross u v)               ;; 3D cross product
(dot u v)                 ;; dot product
(normalize v)             ;; unit vector

;; Numerical methods
(power-iteration A n max tol)  ;; eigenvalue
(integrate f a b n)            ;; Simpson's rule
(newton f df x0 tol max)       ;; root finding

;; Statistics
(variance v)
(std v)
(covariance u v)
```

## REPL Commands

```
:help        Show help
:quit        Exit REPL
:env         Show symbols
:load FILE   Load file
:reload      Reload last
:time EXPR   Time execution
:ast EXPR    Show AST
:clear       Clear screen
```

## CLI Options

```bash
eshkol-run [options] file.esk

-h, --help          Help
-d, --debug         Debug mode
-a, --dump-ast      Dump AST
-i, --dump-ir       Dump LLVM IR
-o, --output FILE   Output name
-c, --compile-only  Object file only
-l LIB              Link library
-L PATH             Library path
-n, --no-stdlib     Skip stdlib
```
