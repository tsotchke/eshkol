# Tutorial 15: Pattern Matching

Eshkol provides `match` expressions for structural pattern matching —
destructure data, dispatch on shape, and bind variables in one expression.

---

## Basic Matching

```scheme
;; Match on literal values
(define (describe x)
  (match x
    (0 "zero")
    (1 "one")
    (_ "other")))

(display (describe 0))     ;; => zero
(display (describe 1))     ;; => one
(display (describe 42))    ;; => other
```

The `_` wildcard matches anything without binding.

---

## Destructuring Lists

```scheme
;; Match list structure
(define (first-two lst)
  (match lst
    ((a b . rest) (list a b))
    ((a) (list a))
    (() '())))

(display (first-two '(1 2 3 4)))   ;; => (1 2)
(display (first-two '(42)))        ;; => (42)
(display (first-two '()))          ;; => ()
```

---

## Nested Patterns

```scheme
;; Match on nested structure
(define (tree-sum tree)
  (match tree
    ((left right) (+ (tree-sum left) (tree-sum right)))
    (n n)))  ;; leaf node

(display (tree-sum '((1 2) (3 (4 5)))))
;; => 15
```

---

## Guard Expressions with cond

When `match` isn't available, `cond` provides multi-way dispatch:

```scheme
(define (classify n)
  (cond
    ((< n 0) "negative")
    ((= n 0) "zero")
    ((< n 10) "small")
    ((< n 100) "medium")
    (else "large")))

(display (classify -5))   ;; => negative
(display (classify 0))    ;; => zero
(display (classify 7))    ;; => small
(display (classify 42))   ;; => medium
(display (classify 999))  ;; => large
```
