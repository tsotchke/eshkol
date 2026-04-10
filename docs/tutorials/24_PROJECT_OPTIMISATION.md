# Project: Function Optimisation with Autodiff

Find the minimum of complex functions using gradient descent,
Newton's method, and the Hessian — all powered by compiler-native AD.

---

## Problem 1: Rosenbrock Function

The Rosenbrock function is a classic optimisation benchmark. The minimum
is at (1, 1) but the valley is narrow and curved — hard for optimisers.

```scheme
;; f(x, y) = (1 - x)^2 + 100*(y - x^2)^2
(define (rosenbrock x y)
  (+ (* (- 1.0 x) (- 1.0 x))
     (* 100.0 (* (- y (* x x)) (- y (* x x))))))

;; Gradient descent with autodiff
(define (optimise-gd x y lr steps)
  (if (= steps 0)
      (begin
        (display "Minimum at: (")
        (display x) (display ", ") (display y) (display ")")
        (newline)
        (display "f(x,y) = ") (display (rosenbrock x y))
        (newline))
      (let ((g (gradient rosenbrock x y)))
        (optimise-gd (- x (* lr (vector-ref g 0)))
                     (- y (* lr (vector-ref g 1)))
                     lr (- steps 1)))))

(display "=== Rosenbrock Optimisation ===") (newline)
(display "Starting at (0, 0):") (newline)
(optimise-gd 0.0 0.0 0.001 10000)
;; Should converge near (1, 1) with f ~ 0
```

---

## Problem 2: Newton's Method with Hessian

For faster convergence, use the Hessian (second derivatives) to take
Newton steps: x_new = x - H^(-1) * grad.

```scheme
;; Simple 1D Newton's method for finding roots of f'(x) = 0
(define (newton-1d f x steps)
  (if (= steps 0)
      (begin
        (display "Minimum at x = ") (display x)
        (display ", f(x) = ") (display (f x))
        (newline))
      (let ((fp (derivative f x))
            (fpp (derivative (lambda (t) (derivative f t)) x)))
        (if (< (abs fpp) 1e-10)
            x  ;; Hessian too small, stop
            (newton-1d f (- x (/ fp fpp)) (- steps 1))))))

;; f(x) = x^4 - 3x^2 + 2  (minima at x = +/- sqrt(3/2))
(define (quartic x) (+ (- (expt x 4) (* 3.0 (* x x))) 2.0))

(display "=== Newton's Method ===") (newline)
(newton-1d quartic 2.0 20)
```

---

## Problem 3: Gradient Descent on a Loss Landscape

```scheme
;; Fit a quadratic y = ax^2 + bx + c to noisy data
(define data '((0 1.1) (1 2.8) (2 6.2) (3 11.1) (4 17.9)))

(define (model a b c x) (+ (* a (* x x)) (* b x) c))

(define (mse a b c)
  (fold-left + 0.0
    (map (lambda (point)
           (let ((x (car point))
                 (y (cadr point))
                 (pred (model a b c x)))
             (* (- pred y) (- pred y))))
         data)))

(define (fit a b c lr steps)
  (if (= steps 0)
      (begin
        (display "Fitted: ") (display a) (display "x^2 + ")
        (display b) (display "x + ") (display c)
        (newline)
        (display "MSE: ") (display (mse a b c)) (newline))
      (let ((g (gradient mse a b c)))
        (fit (- a (* lr (vector-ref g 0)))
             (- b (* lr (vector-ref g 1)))
             (- c (* lr (vector-ref g 2)))
             lr (- steps 1)))))

(display "=== Quadratic Fit ===") (newline)
(fit 0.0 0.0 0.0 0.0001 5000)
;; Should converge near a=1, b=1, c=1 (y = x^2 + x + 1)
```

---

## Why This Matters

In most languages, gradient descent requires:
- A framework (PyTorch, JAX, TensorFlow)
- Defining a computation graph
- Calling `.backward()` or `jax.grad()`
- Managing parameter tensors

In Eshkol: `(gradient f x y)` — that's it. The compiler computes exact
derivatives of arbitrary Scheme code. You write math, the compiler
differentiates it.
