# Tutorial 8: Continuations, Exceptions, and Control Flow

Eshkol implements the full R7RS control flow toolkit: first-class
continuations (`call/cc`), dynamic wind guards (`dynamic-wind`), and
structured exception handling (`guard`/`raise`).

---

## Part 1: First-Class Continuations

`call/cc` (call-with-current-continuation) captures the current execution
state as a first-class value. Invoking the continuation jumps back to the
point where it was captured.

```scheme
;; Simple escape: return early from a computation
(define (find-first pred lst)
  (call/cc
    (lambda (return)
      (for-each (lambda (x)
                  (if (pred x) (return x)))
                lst)
      #f)))  ;; not found

(display (find-first even? '(1 3 5 4 7)))
;; => 4

(display (find-first even? '(1 3 5 7)))
;; => #f
```

### Coroutines with Continuations

```scheme
;; Two coroutines that yield control to each other
(define (make-coroutine body)
  (let ((cont #f))
    (lambda ()
      (call/cc
        (lambda (return)
          (if cont
              (cont return)
              (body (lambda (value)
                      (call/cc
                        (lambda (k)
                          (set! cont k)
                          (return value)))))))))))
```

---

## Part 2: Dynamic Wind

`dynamic-wind` guarantees that setup and cleanup code runs even when
control jumps via continuations:

```scheme
(define (with-file filename proc)
  (let ((port #f))
    (dynamic-wind
      (lambda () (set! port (open-input-file filename)))  ;; setup
      (lambda () (proc port))                              ;; body
      (lambda () (close-input-port port)))))               ;; cleanup

;; The file is ALWAYS closed, even if proc calls a continuation
;; that jumps out of the body.
```

---

## Part 3: Exception Handling

`guard` catches exceptions raised by `raise`:

```scheme
;; Raise and catch an exception
(guard (exn
        (#t (display "Caught: ")
            (display (condition-message exn))
            (newline)))
  (display "Before error")
  (newline)
  (raise (make-exception 'error "something went wrong"))
  (display "This never runs")
  (newline))

;; Output:
;; Before error
;; Caught: something went wrong
```

### Custom Exception Types

```scheme
;; Define domain-specific exceptions
(define (divide a b)
  (if (= b 0)
      (raise (make-exception 'divide-by-zero "division by zero"))
      (/ a b)))

(guard (exn
        ((eq? (condition-type exn) 'divide-by-zero)
         (display "Cannot divide by zero!")
         (newline)
         0))
  (display (divide 10 3))  ;; => 3.333...
  (newline)
  (display (divide 10 0))  ;; raises, caught by guard
  (newline))
```

### Nested Guards

```scheme
(guard (outer-exn
        (#t (display "Outer caught it")))
  (guard (inner-exn
          ((eq? (condition-type inner-exn) 'recoverable)
           (display "Inner recovered")))
    (raise (make-exception 'fatal "boom"))))
;; => Outer caught it
;; (inner guard didn't match 'fatal, so it re-raised to outer)
```

---

## Part 4: Values and Multiple Return Values

```scheme
;; Return multiple values
(define (divide-and-remainder a b)
  (values (quotient a b) (remainder a b)))

;; Receive multiple values
(call-with-values
  (lambda () (divide-and-remainder 17 5))
  (lambda (q r)
    (display "Quotient: ") (display q) (newline)
    (display "Remainder: ") (display r) (newline)))
;; Quotient: 3
;; Remainder: 2
```

---

## Part 5: Tail Call Optimization

Eshkol guarantees proper tail calls — tail-recursive functions run in
constant stack space:

```scheme
;; This runs with O(1) stack, even for n = 1,000,000
(define (sum-to n)
  (define (loop i acc)
    (if (> i n) acc
        (loop (+ i 1) (+ acc i))))
  (loop 1 0))

(display (sum-to 1000000))
;; => 500000500000
```

Mutual recursion is also optimised:

```scheme
(define (even? n) (if (= n 0) #t (odd? (- n 1))))
(define (odd? n) (if (= n 0) #f (even? (- n 1))))

(display (even? 1000000))
;; => #t (no stack overflow)
```

---

*Next: Tutorial 9 — The Module System*
