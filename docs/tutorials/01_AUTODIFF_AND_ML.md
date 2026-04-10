# Tutorial 1: Automatic Differentiation and Machine Learning

Learn to use Eshkol's compiler-native automatic differentiation to build
gradient-based machine learning from scratch. No libraries, no frameworks —
the compiler itself computes derivatives.

**Prerequisites:** Eshkol installed (`brew install tsotchke/eshkol/eshkol` or from source).

---

## Part 1: Your First Derivative

Eshkol computes derivatives at the compiler level. The `derivative` builtin
takes a function and a point, and returns the derivative at that point.

```scheme
;; f(x) = x^2
(define (square x) (* x x))

;; f'(2) = 2x = 4
(display (derivative square 2.0))
(newline)
;; => 4.0
```

Save as `deriv.esk` and run:

```bash
$ eshkol-run deriv.esk -o deriv && ./deriv
4.0
```

This isn't numerical (no epsilon, no finite differences). Eshkol uses
**forward-mode AD with dual numbers**: the compiler transforms arithmetic
to propagate both the value and its derivative simultaneously.

---

## Part 2: Gradients of Multivariate Functions

For functions of multiple variables, `gradient` returns a vector of partial
derivatives.

```scheme
;; f(x, y) = x^2 + x*y + y^2
(define (f x y) (+ (* x x) (* x y) (* y y)))

;; grad f at (1, 2) = (df/dx, df/dy) = (2x+y, x+2y) = (4, 5)
(display (gradient f 1.0 2.0))
(newline)
;; => #(4.0 5.0)
```

The gradient is returned as a tensor (vector). Each component is the
partial derivative with respect to the corresponding argument.

---

## Part 3: Higher-Order Derivatives

Eshkol supports nested differentiation up to 32 levels deep. You can
differentiate a derivative to get second derivatives:

```scheme
;; f(x) = x^3
(define (cube x) (* x x x))

;; f'(x) = 3x^2 => f'(2) = 12
(display (derivative cube 2.0))
(newline)
;; => 12.0

;; f''(x) = 6x => f''(2) = 12
(define (cube-prime x) (derivative cube x))
(display (derivative cube-prime 2.0))
(newline)
;; => 12.0
```

Or use `hessian` directly for the matrix of second partial derivatives:

```scheme
(define (g x y) (+ (* x x y) (* y y y)))

;; Hessian at (1, 2):
;; [[d2g/dxdx, d2g/dxdy], [d2g/dydx, d2g/dydy]]
;; = [[2y, 2x], [2x, 6y]]
;; = [[4, 2], [2, 12]]
(display (hessian g 1.0 2.0))
(newline)
```

---

## Part 4: Vector Calculus

Eight vector calculus operators are built in:

```scheme
;; Divergence of a vector field F(x,y,z) = (x^2, y^2, z^2)
;; div F = 2x + 2y + 2z
(define (field-x x y z) (* x x))
(define (field-y x y z) (* y y))
(define (field-z x y z) (* z z))

(display (divergence field-x field-y field-z 1.0 2.0 3.0))
(newline)
;; => 12.0  (2*1 + 2*2 + 2*3)
```

Available operators: `derivative`, `gradient`, `jacobian`, `hessian`,
`divergence`, `curl`, `laplacian`, `directional-derivative`.

---

## Part 5: Gradient Descent from Scratch

Now let's use autodiff to train a model. We'll fit a linear function
`y = w*x` to data by minimizing squared error.

```scheme
;; Loss function: (w*x - y)^2
(define (loss w x y)
  (let ((error (- (* w x) y)))
    (* error error)))

;; One gradient descent step
(define (step w lr x y)
  (let ((grad (derivative (lambda (w) (loss w x y)) w)))
    (- w (* lr grad))))

;; Train over a dataset
(define (train w lr data)
  (if (null? data)
      w
      (let ((x (car (car data)))
            (y (car (cdr (car data)))))
        (train (step w lr x y) lr (cdr data)))))

;; Dataset: y = 2x
(define data '((1 2) (2 4) (3 6) (4 8) (5 10)))

;; Start with random weight, learn rate 0.01
(define w0 0.5)

;; Train for 100 epochs
(define (train-epochs w lr data epochs)
  (if (= epochs 0) w
      (train-epochs (train w lr data) lr data (- epochs 1))))

(define w-final (train-epochs w0 0.01 data 100))
(display "Learned weight: ")
(display w-final)
(newline)
;; => ~2.0 (converges to the true slope)
```

The key insight: `(derivative (lambda (w) (loss w x y)) w)` computes the
exact gradient of the loss with respect to `w`, automatically. No manual
backprop, no computation graph — the compiler handles it.

---

## Part 6: Multi-Parameter Models

For models with multiple parameters, use `gradient`:

```scheme
;; Linear model: y = w1*x + w0 (slope + intercept)
(define (predict w0 w1 x) (+ (* w1 x) w0))

;; Mean squared error over dataset
(define (mse w0 w1 data)
  (if (null? data)
      0.0
      (let ((x (car (car data)))
            (y (car (cdr (car data))))
            (pred (predict w0 w1 x))
            (err (- pred y)))
        (+ (* err err) (mse w0 w1 (cdr data))))))

;; Compute gradient with respect to (w0, w1)
;; and update both parameters simultaneously
(define (step-2d w0 w1 lr data)
  (let ((grad (gradient (lambda (a b) (mse a b data)) w0 w1)))
    (let ((dw0 (vector-ref grad 0))
          (dw1 (vector-ref grad 1)))
      (list (- w0 (* lr dw0))
            (- w1 (* lr dw1))))))
```

---

## Part 7: Using ML Builtins

Eshkol includes 555+ builtins for ML. Here are the most useful:

### Activation Functions

```scheme
;; All take and return doubles or tensors
(sigmoid 0.0)           ;; => 0.5
(tanh 1.0)              ;; => 0.7616...
(relu -0.5)             ;; => 0.0
(relu 0.5)              ;; => 0.5
(softplus 0.0)          ;; => 0.6931... (ln(2))
(leaky-relu -0.5)       ;; => -0.005 (alpha=0.01)
(elu -1.0)              ;; => -0.6321...
(gelu 0.5)              ;; => 0.3457...
(swish 1.0)             ;; => 0.7311...
```

### Loss Functions

```scheme
;; Mean squared error
(mse-loss predicted actual)

;; Cross-entropy (for classification)
(cross-entropy-loss predicted actual)

;; Huber loss (robust to outliers)
(huber-loss predicted actual delta)
```

### Optimizers

```scheme
;; Gradient descent with momentum
(gradient-descent params grad learning-rate)

;; Adam optimizer
(adam params grad learning-rate beta1 beta2 epsilon t)

;; L-BFGS for second-order optimization
(line-search f x direction)
```

These all compose with autodiff — you can differentiate through any of
them.

---

## Part 8: Dot Product and Linear Algebra

```scheme
(require core.list.higher_order)

;; Dot product using fold-left + map
(define (dot a b)
  (fold-left + 0 (map * a b)))

(display (dot '(1 2 3) '(4 5 6)))
(newline)
;; => 32

;; Matrix-vector multiply (list of rows)
(define (matvec matrix vec)
  (map (lambda (row) (dot row vec)) matrix))

(define M '((1 2) (3 4)))
(display (matvec M '(5 6)))
(newline)
;; => (17 39)
```

---

## What's Next

- **Tutorial 2: The Bytecode VM** — compile to portable bytecode, inspect,
  run on any platform
- **Tutorial 3: Weight Matrix Transformer** — compile Eshkol programs into
  neural network weight matrices
- **Tutorial 4: Consciousness Engine** — logic programming, active
  inference, and global workspace theory

---

*All examples in this tutorial can be run with `eshkol-run file.esk -o out && ./out`
or pasted into the browser REPL at [eshkol.ai](https://eshkol.ai).*
