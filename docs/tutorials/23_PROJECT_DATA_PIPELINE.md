# Project: Build a Data Analysis Pipeline

A complete program that reads data, processes it with higher-order
functions, computes statistics, and outputs results.

---

## The Complete Program

```scheme
;; ═══════════════════════════════════════════════════════
;; Data Analysis Pipeline
;; Uses: lists, fold, map, filter, exact arithmetic
;; ═══════════════════════════════════════════════════════

(require core.list.higher_order)

;; --- Dataset: exam scores ---
(define scores '(85 92 78 95 88 73 91 87 69 94 82 76 90 83 97))

;; --- Basic statistics ---

(define n (length scores))

;; Mean
(define total (fold-left + 0 scores))
(define mean (/ total n))
(display "Count: ") (display n) (newline)
(display "Sum: ") (display total) (newline)
(display "Mean: ") (display (exact->inexact mean)) (newline)

;; Min and Max
(define (list-min lst)
  (fold-left (lambda (a b) (if (< a b) a b)) (car lst) (cdr lst)))
(define (list-max lst)
  (fold-left (lambda (a b) (if (> a b) a b)) (car lst) (cdr lst)))

(display "Min: ") (display (list-min scores)) (newline)
(display "Max: ") (display (list-max scores)) (newline)

;; Variance and standard deviation
(define (square-diff x) (* (- x mean) (- x mean)))
(define variance (/ (fold-left + 0 (map square-diff scores)) n))
(display "Variance: ") (display (exact->inexact variance)) (newline)
(display "Std Dev: ") (display (sqrt (exact->inexact variance))) (newline)

;; --- Filtering and grouping ---

;; Students who passed (>= 80)
(define passed (filter (lambda (s) (>= s 80)) scores))
(display "Passed (>= 80): ") (display passed) (newline)
(display "Pass rate: ")
(display (exact->inexact (/ (length passed) n)))
(newline)

;; Grade distribution
(define (grade score)
  (cond ((>= score 90) 'A)
        ((>= score 80) 'B)
        ((>= score 70) 'C)
        (else 'F)))

(define grades (map grade scores))
(display "Grades: ") (display grades) (newline)

;; Count each grade
(define (count-grade g)
  (length (filter (lambda (x) (eq? x g)) grades)))

(display "A: ") (display (count-grade 'A)) (newline)
(display "B: ") (display (count-grade 'B)) (newline)
(display "C: ") (display (count-grade 'C)) (newline)
(display "F: ") (display (count-grade 'F)) (newline)

;; --- Normalisation ---

(define min-score (list-min scores))
(define max-score (list-max scores))
(define range (- max-score min-score))

(define (normalise x)
  (exact->inexact (/ (- x min-score) range)))

(define normalised (map normalise scores))
(display "Normalised: ") (display normalised) (newline)

;; --- Dot product for correlation ---
(define (dot a b) (fold-left + 0 (map * a b)))

;; Hours studied (hypothetical data, same length as scores)
(define hours '(5 8 4 9 7 3 8 6 2 9 5 4 7 6 10))

(display "Score-Hours dot product: ")
(display (dot scores hours))
(newline)

(display "Pipeline complete.")
(newline)
```

---

## Key Concepts Demonstrated

1. **fold-left** for reductions (sum, min, max, variance)
2. **map** for transformations (grading, normalisation)
3. **filter** for selection (passing students)
4. **Exact arithmetic** — integer sums stay exact, `exact->inexact` for display
5. **Higher-order composition** — small functions composed via map/fold
6. **Multi-list map** — `(map * a b)` for element-wise products
