# Tutorial 17: Functional Programming Patterns

Eshkol is a functional-first language. This tutorial covers closures,
higher-order patterns, composition, and currying.

---

## Closures

A closure captures variables from its enclosing scope:

```scheme
(define (make-counter start)
  (let ((count start))
    (lambda ()
      (set! count (+ count 1))
      count)))

(define c (make-counter 0))
(display (c))  ;; => 1
(display (c))  ;; => 2
(display (c))  ;; => 3

;; Each counter is independent
(define c2 (make-counter 100))
(display (c2))  ;; => 101
(display (c))   ;; => 4 (c still has its own state)
```

---

## Function Composition

```scheme
(require core.functional.compose)

;; compose: (compose f g) => (lambda (x) (f (g x)))
(define inc-then-double
  (compose (lambda (x) (* x 2))
           (lambda (x) (+ x 1))))

(display (inc-then-double 3))  ;; => 8 ((3+1)*2)
(display (inc-then-double 5))  ;; => 12 ((5+1)*2)

;; pipe: left-to-right composition (opposite of compose)
;; (pipe f g h) => (lambda (x) (h (g (f x))))
```

---

## Currying

```scheme
(require core.functional.curry)

;; curry: convert a multi-arg function to a chain of single-arg functions
(define add (lambda (a b) (+ a b)))
(define add5 ((curry add) 5))
(display (add5 10))  ;; => 15
(display (add5 20))  ;; => 25

;; flip: swap the first two arguments
(define div (lambda (a b) (/ a b)))
(define div-by (flip div))
(display ((div-by 2) 10))  ;; => 5 (10/2)
```

---

## Higher-Order Patterns

### Apply a function N times

```scheme
(define (apply-n f n x)
  (if (= n 0) x
      (apply-n f (- n 1) (f x))))

(define double (lambda (x) (* x 2)))
(display (apply-n double 5 1))  ;; => 32 (2^5)
```

### Memoisation

```scheme
(define (memoize f)
  (let ((cache (make-hash-table)))
    (lambda args
      (let ((key (car args)))
        (if (hash-table-exists? cache key)
            (hash-table-ref cache key)
            (let ((result (f key)))
              (hash-table-set! cache key result)
              result))))))

(define fib
  (memoize (lambda (n)
    (if (< n 2) n
        (+ (fib (- n 1)) (fib (- n 2)))))))

(display (fib 40))  ;; instant (memoised)
```

### Church encoding (functions all the way down)

```scheme
;; Booleans as functions
(define TRUE (lambda (t f) t))
(define FALSE (lambda (t f) f))
(define IF (lambda (b t f) (b t f)))

(display (IF TRUE "yes" "no"))   ;; => yes
(display (IF FALSE "yes" "no"))  ;; => no
```

---

## First-Class Functions Everywhere

Functions are values. Pass them, return them, store them:

```scheme
;; Store functions in a list
(define ops (list + - * /))
(display ((car ops) 3 4))        ;; => 7
(display ((cadr ops) 10 3))      ;; => 7
(display ((caddr ops) 6 7))      ;; => 42

;; Return functions from functions
(define (make-adder n) (lambda (x) (+ x n)))
(define add10 (make-adder 10))
(display (add10 32))  ;; => 42

;; Map with first-class functions
(require core.list.higher_order)
(display (map add10 '(1 2 3 4 5)))  ;; => (11 12 13 14 15)
```
