# Project: Train a Neural Network from Scratch

A complete, runnable program that trains a neural network to learn XOR
using only Eshkol's autodiff — no frameworks, no libraries, no magic.

Save this as `xor_nn.esk` and run with `eshkol-run xor_nn.esk -o xor_nn && ./xor_nn`.

---

## The Problem

XOR is the classic non-linear problem. A single neuron can't learn it
because XOR isn't linearly separable. We need a hidden layer.

```
Input    Output
0, 0  →  0
0, 1  →  1
1, 0  →  1
1, 1  →  0
```

---

## The Complete Program

```scheme
;; ═══════════════════════════════════════════════════════
;; XOR Neural Network — trained with compiler-native AD
;; ═══════════════════════════════════════════════════════

;; --- Activation function ---
;; Sigmoid squashes any value to (0, 1)
(define (sigmoid x)
  (/ 1.0 (+ 1.0 (exp (- 0.0 x)))))

;; --- Forward pass ---
;; 2-2-1 network: 2 inputs, 2 hidden neurons, 1 output
;; Parameters: w1-w6 (weights) + b1-b3 (biases) = 9 parameters
(define (forward x1 x2 w1 w2 w3 w4 w5 w6 b1 b2 b3)
  ;; Hidden layer (2 neurons)
  (let ((h1 (sigmoid (+ (* w1 x1) (* w2 x2) b1)))
        (h2 (sigmoid (+ (* w3 x1) (* w4 x2) b2))))
    ;; Output layer (1 neuron)
    (sigmoid (+ (* w5 h1) (* w6 h2) b3))))

;; --- Loss function ---
;; Mean squared error over all 4 XOR examples
(define (loss w1 w2 w3 w4 w5 w6 b1 b2 b3)
  (let ((e1 (- (forward 0.0 0.0 w1 w2 w3 w4 w5 w6 b1 b2 b3) 0.0))
        (e2 (- (forward 0.0 1.0 w1 w2 w3 w4 w5 w6 b1 b2 b3) 1.0))
        (e3 (- (forward 1.0 0.0 w1 w2 w3 w4 w5 w6 b1 b2 b3) 1.0))
        (e4 (- (forward 1.0 1.0 w1 w2 w3 w4 w5 w6 b1 b2 b3) 0.0)))
    (+ (* e1 e1) (* e2 e2) (* e3 e3) (* e4 e4))))

;; --- Training ---
;; Gradient descent: compute gradient of loss w.r.t. all 9 parameters,
;; then step in the negative gradient direction.
;;
;; The key insight: (gradient loss w1 w2 ... b3) computes ALL 9 partial
;; derivatives in one call. The compiler handles the chain rule through
;; sigmoid, addition, and multiplication automatically.

(define (train w1 w2 w3 w4 w5 w6 b1 b2 b3 lr epochs)
  (if (= epochs 0)
      (list w1 w2 w3 w4 w5 w6 b1 b2 b3)
      (let ((g (gradient loss w1 w2 w3 w4 w5 w6 b1 b2 b3)))
        (train (- w1 (* lr (vector-ref g 0)))
               (- w2 (* lr (vector-ref g 1)))
               (- w3 (* lr (vector-ref g 2)))
               (- w4 (* lr (vector-ref g 3)))
               (- w5 (* lr (vector-ref g 4)))
               (- w6 (* lr (vector-ref g 5)))
               (- b1 (* lr (vector-ref g 6)))
               (- b2 (* lr (vector-ref g 7)))
               (- b3 (* lr (vector-ref g 8)))
               lr (- epochs 1)))))

;; --- Initialise and train ---
;; Start with small random-ish weights (hand-picked for reproducibility)
(define params
  (train 0.5 -0.3 0.8 -0.6 0.4 -0.7
         0.1 -0.2 0.3
         2.0    ;; learning rate
         5000)) ;; epochs

;; --- Evaluate ---
(define (predict x1 x2)
  (let ((w1 (list-ref params 0)) (w2 (list-ref params 1))
        (w3 (list-ref params 2)) (w4 (list-ref params 3))
        (w5 (list-ref params 4)) (w6 (list-ref params 5))
        (b1 (list-ref params 6)) (b2 (list-ref params 7))
        (b3 (list-ref params 8)))
    (forward x1 x2 w1 w2 w3 w4 w5 w6 b1 b2 b3)))

(display "XOR Neural Network Results:") (newline)
(display "0 XOR 0 = ") (display (predict 0.0 0.0)) (newline)
(display "0 XOR 1 = ") (display (predict 0.0 1.0)) (newline)
(display "1 XOR 0 = ") (display (predict 1.0 0.0)) (newline)
(display "1 XOR 1 = ") (display (predict 1.0 1.0)) (newline)
(display "Final loss: ") (display (apply loss params)) (newline)
```

---

## What You Should See

```
XOR Neural Network Results:
0 XOR 0 = 0.02...  (close to 0)
0 XOR 1 = 0.97...  (close to 1)
1 XOR 0 = 0.97...  (close to 1)
1 XOR 1 = 0.03...  (close to 0)
Final loss: 0.001...
```

---

## What Just Happened

1. We defined a neural network as a pure function (`forward`)
2. We defined a loss function that measures error on all 4 XOR examples
3. We called `(gradient loss ...)` which computed all 9 partial derivatives
   **automatically** — the compiler propagated dual numbers through every
   multiplication, addition, sigmoid, and let-binding
4. We ran gradient descent for 5000 steps
5. The network learned XOR

No backpropagation code. No computation graph. No framework. The compiler
did all the calculus.
