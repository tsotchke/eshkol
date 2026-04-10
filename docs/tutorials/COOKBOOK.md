# Eshkol Cookbook

30 copy-paste recipes for common tasks. Each one works in the REPL or
as a compiled program.

---

## Basics

**Sum a list:**
```scheme
(fold-left + 0 '(1 2 3 4 5))  ;; => 15
```

**Find the maximum:**
```scheme
(fold-left max 0 '(3 7 2 9 4))  ;; => 9
```

**Reverse a string:**
```scheme
(require core.strings)
(string-reverse "hello")  ;; => "olleh"
```

**Flatten a nested list:**
```scheme
(define (flatten lst)
  (cond ((null? lst) '())
        ((pair? (car lst)) (append (flatten (car lst)) (flatten (cdr lst))))
        (else (cons (car lst) (flatten (cdr lst))))))
(flatten '(1 (2 3) (4 (5 6))))  ;; => (1 2 3 4 5 6)
```

---

## Math

**Dot product:**
```scheme
(require core.list.higher_order)
(fold-left + 0 (map * '(1 2 3) '(4 5 6)))  ;; => 32
```

**Fibonacci (fast):**
```scheme
(define (fib n)
  (let loop ((a 0) (b 1) (i 0))
    (if (= i n) a (loop b (+ a b) (+ i 1)))))
(fib 50)  ;; => 12586269025
```

**Prime check:**
```scheme
(define (prime? n)
  (if (< n 2) #f
      (let loop ((i 2))
        (cond ((> (* i i) n) #t)
              ((= 0 (remainder n i)) #f)
              (else (loop (+ i 1)))))))
(filter prime? (iota 30))  ;; => (2 3 5 7 11 13 17 19 23 29)
```

**GCD:**
```scheme
(define (my-gcd a b) (if (= b 0) a (my-gcd b (remainder a b))))
(my-gcd 48 18)  ;; => 6
```

---

## Autodiff

**Derivative:**
```scheme
(derivative (lambda (x) (* x x)) 3.0)  ;; => 6.0
```

**Gradient of 2-variable function:**
```scheme
(gradient (lambda (x y) (+ (* x x) (* y y))) 3.0 4.0)
;; => #(6.0 8.0)
```

**Newton's method (find root of f'(x)=0):**
```scheme
(define (newton f x)
  (let ((fp (derivative f x))
        (fpp (derivative (lambda (t) (derivative f t)) x)))
    (- x (/ fp fpp))))
```

---

## Lists

**Take every other element:**
```scheme
(define (every-other lst)
  (if (or (null? lst) (null? (cdr lst))) lst
      (cons (car lst) (every-other (cddr lst)))))
(every-other '(a b c d e f))  ;; => (a c e)
```

**Group by predicate:**
```scheme
(define (partition pred lst)
  (list (filter pred lst) (filter (lambda (x) (not (pred x))) lst)))
(partition even? '(1 2 3 4 5 6))  ;; => ((2 4 6) (1 3 5))
```

**Zip two lists:**
```scheme
(map list '(a b c) '(1 2 3))  ;; => ((a 1) (b 2) (c 3))
```

**Unique elements:**
```scheme
(define (unique lst)
  (fold-left (lambda (acc x) (if (member x acc) acc (append acc (list x)))) '() lst))
(unique '(1 2 3 2 1 4))  ;; => (1 2 3 4)
```

---

## Strings

**Count words:**
```scheme
(define (count-words s)
  (length (filter (lambda (c) (char=? c #\space)) (string->list s))))
;; Approximate — counts spaces
```

**Repeat a string:**
```scheme
(require core.strings)
(string-repeat "ha" 3)  ;; => "hahaha"
```

---

## Control Flow

**Try/catch:**
```scheme
(guard (exn (#t (display "Error caught")))
  (/ 1 0))
```

**Early return:**
```scheme
(define (find-first pred lst)
  (call/cc (lambda (return)
    (for-each (lambda (x) (if (pred x) (return x))) lst)
    #f)))
(find-first even? '(1 3 5 4 7))  ;; => 4
```

---

## Data

**Build a lookup table:**
```scheme
(define db '((alice 30) (bob 25) (charlie 35)))
(define (lookup name) (cadr (assoc name db)))
(lookup 'alice)  ;; => 30
```

**Frequency count:**
```scheme
(define (frequencies lst)
  (fold-left
    (lambda (acc x)
      (let ((found (assoc x acc)))
        (if found
            (map (lambda (pair) (if (eq? (car pair) x) (list x (+ 1 (cadr pair))) pair)) acc)
            (cons (list x 1) acc))))
    '() lst))
(frequencies '(a b a c b a))  ;; => ((a 3) (b 2) (c 1))
```

---

## Functional Patterns

**Compose functions:**
```scheme
(define (compose f g) (lambda (x) (f (g x))))
(define inc-double (compose (lambda (x) (* x 2)) (lambda (x) (+ x 1))))
(inc-double 3)  ;; => 8
```

**Memoize:**
```scheme
(define (memo f)
  (let ((cache '()))
    (lambda (x)
      (let ((found (assoc x cache)))
        (if found (cadr found)
            (let ((result (f x)))
              (set! cache (cons (list x result) cache))
              result))))))
```

**Apply a function N times:**
```scheme
(define (iterate f n x) (if (= n 0) x (iterate f (- n 1) (f x))))
(iterate (lambda (x) (* x 2)) 10 1)  ;; => 1024
```
