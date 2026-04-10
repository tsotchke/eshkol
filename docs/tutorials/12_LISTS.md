# Tutorial 12: Lists and Higher-Order Functions

Lists are the fundamental data structure in Eshkol. 60+ operations are
built in, from basic cons/car/cdr to sorting and partitioning.

---

## Creating Lists

```scheme
(define xs (list 1 2 3 4 5))      ;; => (1 2 3 4 5)
(define ys (cons 0 xs))           ;; => (0 1 2 3 4 5)
(define zs '(a b c))              ;; quoted list of symbols
(define empty '())                ;; empty list
(define nested '((1 2) (3 4)))    ;; nested lists
```

---

## Basic Operations

```scheme
(car '(1 2 3))       ;; => 1 (first element)
(cdr '(1 2 3))       ;; => (2 3) (rest)
(cadr '(1 2 3))      ;; => 2 (second element)
(caddr '(1 2 3))     ;; => 3 (third element)
(length '(a b c d))  ;; => 4
(null? '())          ;; => #t
(pair? '(1 2))       ;; => #t
(list? '(1 2 3))     ;; => #t
```

---

## Transformations

```scheme
(require core.list.transform)

(reverse '(1 2 3))           ;; => (3 2 1)
(append '(1 2) '(3 4))       ;; => (1 2 3 4)
(take 3 '(a b c d e))        ;; => (a b c)
(drop 2 '(a b c d e))        ;; => (c d e)
(list-copy '(1 2 3))         ;; => (1 2 3) (fresh copy)
(filter even? '(1 2 3 4 5))  ;; => (2 4)
(partition even? '(1 2 3 4 5))
;; => two lists: (2 4) and (1 3 5)
```

---

## Higher-Order Functions

```scheme
(require core.list.higher_order)

;; Map — apply function to each element
(map (lambda (x) (* x x)) '(1 2 3 4))  ;; => (1 4 9 16)

;; Multi-list map — element-wise across lists
(map + '(1 2 3) '(10 20 30))  ;; => (11 22 33)
(map * '(1 2 3) '(4 5 6))     ;; => (4 10 18)

;; Fold (left reduce)
(fold + 0 '(1 2 3 4 5))       ;; => 15
(fold-left + 0 '(1 2 3 4 5))  ;; => 15 (same, R6RS name)

;; Fold-right
(fold-right cons '() '(1 2 3))  ;; => (1 2 3)

;; For-each (side effects, no return value)
(for-each (lambda (x) (display x) (display " "))
          '(a b c d))
;; prints: a b c d

;; Filter
(filter even? '(1 2 3 4 5 6))  ;; => (2 4 6)

;; Any / Every
(any even? '(1 3 5 4 7))     ;; => #t
(every even? '(2 4 6))       ;; => #t
(every even? '(2 3 6))       ;; => #f

;; Find
(find even? '(1 3 4 5))      ;; => 4
(find even? '(1 3 5))        ;; => #f
```

---

## Sorting

```scheme
(require core.list.sort)

(sort < '(3 1 4 1 5 9 2 6))
;; => (1 1 2 3 4 5 6 9)

(sort > '(3 1 4 1 5 9 2 6))
;; => (9 6 5 4 3 2 1 1)

(sort string<? '("banana" "apple" "cherry"))
;; => ("apple" "banana" "cherry")
```

---

## Search and Lookup

```scheme
(require core.list.search)

(member 3 '(1 2 3 4 5))      ;; => (3 4 5)
(member 9 '(1 2 3))          ;; => #f
(list-ref '(a b c d) 2)      ;; => c
(list-tail '(a b c d e) 3)   ;; => (d e)

;; Association lists (key-value pairs)
(define db '((name "Alice") (age 30) (city "NYC")))
(assoc 'age db)               ;; => (age 30)
(cadr (assoc 'name db))       ;; => "Alice"
```

---

## Generators

```scheme
(require core.list.generate)

(iota 5)                ;; => (0 1 2 3 4)
(iota-from 5 10)        ;; => (10 11 12 13 14)
(range 1 6)             ;; => (1 2 3 4 5)
(make-list 4 0)         ;; => (0 0 0 0)
(repeat 3 'x)           ;; => (x x x)
(zip '(1 2 3) '(a b c)) ;; => ((1 a) (2 b) (3 c))
```

---

## Query

```scheme
(require core.list.query)

(length '(a b c d))           ;; => 4
(count-if even? '(1 2 3 4 5)) ;; => 2
(find odd? '(2 4 5 6))        ;; => 5
```

---

## Mixed-Type Lists

Lists can hold any type — integers, strings, booleans, nested lists,
closures, even tensors:

```scheme
(define mixed (list 1 "hello" #t 3.14 '(nested) #(1 2 3)))
(display (car mixed))          ;; => 1
(display (cadr mixed))         ;; => hello
(display (caddr mixed))        ;; => #t
```
