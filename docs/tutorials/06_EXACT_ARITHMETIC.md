# Tutorial 6: Exact Arithmetic

Eshkol's numeric tower includes arbitrary-precision integers (bignums),
exact rationals, and complex numbers. Arithmetic automatically promotes
when precision demands it — no overflow, no truncation.

---

## Part 1: Bignums

Standard 64-bit integers promote to bignums when they would overflow:

```scheme
;; 2^64 — overflows int64, automatically becomes a bignum
(display (expt 2 64))
(newline)
;; => 18446744073709551616

;; Factorial of 100 — 158 digits
(define (factorial n)
  (if (= n 0) 1 (* n (factorial (- n 1)))))

(display (factorial 100))
(newline)
;; => 93326215443944152681699238856266700490715968264381621468...

;; Fibonacci of 1000
(define (fib n)
  (define (loop a b i)
    (if (= i n) a (loop b (+ a b) (+ i 1))))
  (loop 0 1 0))

(display (fib 1000))
(newline)
;; => 43466557686937456435688527675040625802564...
```

Bignums support all standard arithmetic:
```scheme
(+ (expt 10 50) 1)       ;; exact
(* (expt 2 128) 3)       ;; exact
(quotient (expt 10 100) (expt 10 50))  ;; => 10^50
(gcd 1000000007 999999937)             ;; works on bignums
```

---

## Part 2: Exact Rationals

Rationals are written as `numerator/denominator` and stay exact:

```scheme
(display (+ 1/3 1/6))    ;; => 1/2
(display (* 3/4 4/3))    ;; => 1
(display (- 5/7 2/7))    ;; => 3/7

;; Rationals normalise automatically
(display (+ 1/4 1/4))    ;; => 1/2 (not 2/4)

;; Mixed exact/inexact promotes to inexact (R7RS rule)
(display (+ 1/3 0.1))    ;; => 0.4333...
(display (exact->inexact 1/3))  ;; => 0.3333...
```

---

## Part 3: Complex Numbers

```scheme
(define z (make-rectangular 3.0 4.0))  ;; 3 + 4i
(display z)              ;; => 3+4i
(display (magnitude z))  ;; => 5.0
(display (angle z))      ;; => 0.9273... (radians)

;; Arithmetic
(display (+ z (make-rectangular 1.0 -1.0)))  ;; => 4+3i
(display (* z z))        ;; => -7+24i

;; Polar form
(define w (make-polar 2.0 1.5708))  ;; r=2, theta=pi/2
(display w)              ;; => ~0+2i
```

Complex numbers integrate with autodiff — you can differentiate functions
of complex arguments.

---

## Part 4: Number Conversions and Predicates

```scheme
;; Type predicates
(display (integer? 42))      (newline)  ;; => #t
(display (rational? 1/3))    (newline)  ;; => #t
(display (real? 3.14))       (newline)  ;; => #t
(display (exact? 1/3))       (newline)  ;; => #t
(display (inexact? 3.14))    (newline)  ;; => #t
(display (zero? 0))          (newline)  ;; => #t
(display (positive? 5))      (newline)  ;; => #t
(display (negative? -3))     (newline)  ;; => #t

;; Conversion between exact and inexact
(display (exact->inexact 1/3))  (newline)  ;; => 0.333...
(display (number->string 255))  (newline)  ;; => "255"
(display (string->number "42")) (newline)  ;; => 42
```

---

*Next: Tutorial 7 — Parallel Computing*
