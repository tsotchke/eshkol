---
kind: reference
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/multimodular.esk
  - lib/core/ring.esk
  - tests/stdlib/multimodular_test.esk
---

# `core.multimodular` — exact computation over Spec Z

`(require core.multimodular)` computes exact integer and rational answers
from word-size prime fields ([ADR-0034](../../design/adr/0034-ring-carriers-functor-of-points.md)
section 7), in two distinct steps:

1. **Descent over Spec(Z/M).** For a squarefree M = p1 ... pk the Chinese
   remainder theorem is the ring isomorphism Z/M = F_p1 x ... x F_pk. An
   integer polynomial computation commutes with every reduction Z -> F_p, so
   it runs in each fibre F_p and the residues glue to the answer **modulo
   M**, and to nothing more: the finitely many closed points Spec F_p do not
   form a cover of Spec Z.
2. **Bounded reconstruction over Z or Q.** The residue modulo M determines
   the answer only inside a window: the symmetric lift recovers an integer of
   absolute value at most B when M > 2B, and Wang's rational reconstruction
   recovers n/d with |n| <= N, 0 < d <= D when M > 2ND. The proven bounds
   (Hadamard, Cramer, Mignotte; see below) are what place the answer inside
   the window. The number of primes is fixed by them before the first fibre
   runs, so the result is proven, never a probabilistic early termination.

At least one fibre always runs, so a bound of 0 still returns a value of the
shape the computation returns.

Computations are written against [`core.ring`](ring.md) and run over
`(make-zmod p)`. Primes are taken downward from 2^31 - 1, so a product of two
residues fits a signed 64-bit integer on every target, including wasm32. A
prime at which the computation needs a non-unit (it raises `ring-not-a-unit`
in that fibre) is **unlucky**: it is skipped and does not count toward the
bound.

```scheme
(require core.multimodular)
(define A '((3 -1 4) (1 5 -9) (2 6 5)))
(display (hadamard-bound A)) ;; => 426
(newline)
(display (multimodular-primes (* 2 (hadamard-bound A)))) ;; => (2147483647)
(newline)
(display (multimodular-det A)) ;; => 244
(newline)
(display (multimodular-solve A '(1 2 3))) ;; => #(26/61 21/61 1/61)
(newline)
(display (multimodular-det '((1/2 1/3) (1/5 7)))) ;; => 103/30
```

## Algorithms and the bounds they use

| Procedure | Answer | Bound fixing the prime count |
|---|---|---|
| `(multimodular-det A)` | the exact determinant of a square matrix of exact rationals | Hadamard: \|det A\| <= prod_i \|\|row_i\|\|_2, after each row is scaled by the lcm of its denominators; no fibre is unlucky |
| `(multimodular-solve A b)` | the exact solution of A x = b over Q, a vector of exact rationals | Cramer: x_i = det(A_i)/det(A); numerators at most max_i H(A_i) (column i replaced by b), denominators at most H(A); reconstruction is unique for M > 2 N D. Fibres where det A vanishes (computed first, exactly) are skipped before they run; a singular matrix is refused |
| `(multimodular-poly-gcd a b)` | the gcd in Z[x], positive leading coefficient: gcd of the contents times the gcd of the primitive parts | Mignotte: a factor g of degree m of a in Z[x] has \|\|g\|\|_inf <= \|\|g\|\|_1 <= 2^m \|\|a\|\|_2, and so does the normalised gcd h = (d / lc g) g with d the gcd of the leading coefficients; B = 2^k min(\|\|a\|\|_2, \|\|b\|\|_2) with k = min(deg a, deg b) |

For the gcd, primes dividing d are skipped, a fibre whose gcd has larger
degree than another fibre's is unlucky and is discarded, and the lift is
accepted when the primes of the least degree seen exceed 2B and it divides
both inputs exactly, which proves it (a fibre of degree 0 proves the gcd is
the content gcd at once). `(mignotte-gcd-bound a b)` and `(hadamard-bound A)`
return the bounds. Polynomials are coefficient lists of exact integers, lowest
degree first.

```scheme
(require core.multimodular)
;; (x^2 - 1)(2x + 3) and (x^2 - 1)(x - 4)
(display (multimodular-poly-gcd '(-3 -2 3 2) '(4 -1 -4 1))) ;; => (-1 0 1)
(newline)
(display (multimodular-poly-gcd '(6 12) '(4 8))) ;; => (2 4)
(newline)
(display (mignotte-gcd-bound '(-3 -2 3 2) '(4 -1 -4 1))) ;; => 48
```

## Fibre evaluation of any computation

| Procedure | Meaning |
|---|---|
| `(multimodular-integer proc bound [usable?])` | run `(proc Fp)` over usable primes until their product M exceeds 2 * bound; return the symmetric lift, the unique integer of absolute value at most `bound` in the class |
| `(multimodular-rational proc N D [usable?])` | the same with M > 2 N D, each component reconstructed by Wang's algorithm for numerators at most N and denominators at most D |
| `multimodular-integer-report` `multimodular-rational-report` | the same, returning `((value . v) (modulus . M) (primes . used) (unlucky . skipped))` |
| `(multimodular-prime i)` | the i-th prime below 2^31, counting down from 2^31 - 1 (i = 0) |
| `(multimodular-primes bound [usable?])` | the primes from the top that satisfy `usable?` until their product exceeds `bound` |
| `(chinese-remainder residues moduli)` | the residue modulo the product of pairwise coprime moduli |
| `(symmetric-residue x m)` | the representative of x mod m in (-m/2, m/2] |
| `(rational-reconstruct u m N D)` | the unique n/d with \|n\| <= N, 0 < d <= D, n = d u (mod m), when m > 2 N D |

`proc` returns an element of the fibre, or a list or vector of them, glued
componentwise. A computation refused in more than 64 fibres beyond those it
needed is refused as a whole.

```scheme
(require core.ring)
(require core.multimodular)
(define (f R x) (r+ R (r* R 3 (r-expt R x 5)) (r-int R -7)))
;; 3 * 10^100 - 7, from residues modulo primes below 2^31
(display (= (multimodular-integer (lambda (F) (f F (expt 10 20))) (* 4 (expt 10 100)))
            (- (* 3 (expt 10 100)) 7))) ;; => #t
(newline)
(display (multimodular-rational (lambda (F) (r+ F (r-inv F 3) (r-inv F 7))) 10 21)) ;; => 10/21
(newline)
;; a fibre where 2^31 - 1 must be inverted is unlucky: skipped, not counted
(define rep (multimodular-integer-report (lambda (F) (r* F (r-inv F 2147483647) 2147483647)) 1))
(display (cdr (assq 'unlucky rep))) ;; => (2147483647)
(newline)
(display (chinese-remainder '(2 3 2) '(3 5 7))) ;; => 23
(newline)
(display (rational-reconstruct 20 23 3 3)) ;; => -3
```

## Tests

```text
ctest --test-dir build --output-on-failure -R '^multimodular_(jit|aot)$'
```

`tests/stdlib/multimodular_test.esk` compares every answer with an independent
exact path: 17 determinants (1x1 to 10x10, 13-digit entries, singular) with
the bignum Bareiss `exact-det` of `core.exact_linalg`, solutions with
`exact-solve` and by substitution, and 17 gcd pairs with Euclid's algorithm
over Q[x] and exact division in `core.polynomial_exact`, including a forced
unlucky determinant, leading coefficient and gcd degree at the first prime.
Batched F_p kernels on CPU vector units, GPU and XLA are Planned for
v1.4.5-accelerate.
