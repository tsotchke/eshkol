# Project: Self-Improving Programs and Machine Reasoning

Two programs that improve themselves: one through gradient descent
(differentiable self-optimisation), one through active inference
(belief-driven reasoning). Both demonstrate capabilities unique to Eshkol.

---

## Program 1: A Function That Learns Itself

This program starts with a bad approximation of `sin(x)` and uses
autodiff to teach itself to be accurate. The parameters are part of the
program — gradient descent literally rewrites them.

```scheme
;; ═══════════════════════════════════════════════════════
;; Self-Improving Function Approximator
;;
;; Goal: learn to approximate sin(x) on [0, pi] using a
;; polynomial with trainable coefficients.
;;
;; The program optimises its OWN parameters via gradient
;; descent on the compiler's built-in autodiff.
;; ═══════════════════════════════════════════════════════

;; --- The learnable model ---
;; f(x) = c1*x + c2*x^2 + c3*x^3
;; Starts with random coefficients — a bad approximation
(define (model c1 c2 c3 x)
  (+ (* c1 x) (* c2 (* x x)) (* c3 (* x x x))))

;; --- Ground truth ---
(define (target x) (sin x))

;; --- Training data: 8 points on [0, pi] ---
(define xs '(0.0 0.449 0.898 1.347 1.571 1.796 2.244 3.14159))

;; --- Loss: mean squared error over all points ---
(define (loss c1 c2 c3)
  (define (point-error x)
    (let ((err (- (model c1 c2 c3 x) (target x))))
      (* err err)))
  (fold-left + 0.0 (map point-error xs)))

;; --- Self-improvement loop ---
(define (improve c1 c2 c3 lr steps)
  (if (= steps 0)
      (list c1 c2 c3)
      (let ((g (gradient loss c1 c2 c3)))
        (improve (- c1 (* lr (vector-ref g 0)))
                 (- c2 (* lr (vector-ref g 1)))
                 (- c3 (* lr (vector-ref g 2)))
                 lr (- steps 1)))))

;; --- Run ---
(display "=== Self-Improving Function Approximator ===") (newline)

;; Before training
(display "Before training (c1=1, c2=0, c3=0):") (newline)
(display "  f(1.57) = ") (display (model 1.0 0.0 0.0 1.571)) (newline)
(display "  sin(1.57) = ") (display (sin 1.571)) (newline)
(display "  Loss = ") (display (loss 1.0 0.0 0.0)) (newline)
(newline)

;; Train: the program improves itself
(define params (improve 1.0 0.0 0.0 0.001 3000))
(define c1 (car params))
(define c2 (cadr params))
(define c3 (caddr params))

(display "After 3000 steps of self-improvement:") (newline)
(display "  Learned coefficients: ")
(display c1) (display " ")
(display c2) (display " ")
(display c3) (newline)
(display "  f(1.57) = ") (display (model c1 c2 c3 1.571)) (newline)
(display "  sin(1.57) = ") (display (sin 1.571)) (newline)
(display "  Loss = ") (display (loss c1 c2 c3)) (newline)
(newline)

;; Show predictions at every training point
(display "Point-by-point comparison:") (newline)
(for-each
  (lambda (x)
    (display "  x=") (display x)
    (display "  model=") (display (model c1 c2 c3 x))
    (display "  sin=") (display (sin x))
    (newline))
  xs)
```

### What Makes This Self-Improving

The program doesn't just COMPUTE — it modifies its own behaviour.
The coefficients `c1, c2, c3` are the program's knowledge, and
`(gradient loss c1 c2 c3)` tells the program exactly how to change
that knowledge to reduce error. After training, the `model` function
produces different outputs than before — it has literally taught itself.

In most languages, you'd need a framework for this. In Eshkol, it's
just `gradient` — the compiler differentiates through your code.

---

## Program 2: A Reasoning Agent with Active Inference

This program models an agent that:
1. Believes something about the world (prior beliefs in a factor graph)
2. Observes evidence
3. Updates its beliefs (belief propagation)
4. Measures its surprise (free energy)
5. Adapts its model to reduce surprise (learning)
6. Shows its reasoning at each step

```scheme
;; ═══════════════════════════════════════════════════════
;; Active Inference Agent — A Thinking Process
;;
;; The agent models a simple world: "Is it raining?"
;; It has a prior belief, observes evidence (wet ground,
;; umbrella people), updates beliefs, and learns.
;;
;; Uses: factor graphs, belief propagation, free energy,
;; CPT updates — the consciousness engine's inference API.
;; ═══════════════════════════════════════════════════════

(display "=== Active Inference: A Thinking Process ===")
(newline) (newline)

;; --- World model as a factor graph ---
;; Variable 0: Weather (0=sunny, 1=rainy)
;; Variable 1: Ground (0=dry, 1=wet)
;; Variable 2: People (0=no-umbrella, 1=umbrella)
(define world (make-factor-graph 3))

;; Prior beliefs (before seeing anything):
;; P(wet ground | rainy) = 0.9, P(wet ground | sunny) = 0.1
(fg-add-factor! world 0 1 #(0.9 0.1 0.1 0.9))

;; P(umbrella | rainy) = 0.8, P(umbrella | sunny) = 0.2
(fg-add-factor! world 0 2 #(0.8 0.2 0.2 0.8))

;; --- Step 1: Initial beliefs (no evidence yet) ---
(display "Step 1: Prior beliefs (no evidence)")
(newline)
(fg-infer! world 10)
(define fe1 (free-energy world #(0 0)))
(display "  Free energy (surprise): ") (display fe1) (newline)
(display "  Agent thinks: 'I have no evidence. Could go either way.'")
(newline) (newline)

;; --- Step 2: Observe wet ground ---
(display "Step 2: Agent observes WET GROUND")
(newline)
;; Update the weather→ground factor to strongly favour rain
;; given the observation of wetness
(fg-update-cpt! world 0 #(0.95 0.05 0.05 0.95))
(fg-infer! world 10)
(define fe2 (free-energy world #(0 1)))
(display "  Free energy after update: ") (display fe2) (newline)
(display "  Agent thinks: 'Ground is wet. Rain is likely.'")
(newline) (newline)

;; --- Step 3: Also see umbrellas ---
(display "Step 3: Agent also observes UMBRELLAS")
(newline)
;; Strengthen the umbrella evidence
(fg-update-cpt! world 1 #(0.9 0.1 0.1 0.9))
(fg-infer! world 10)
(define fe3 (free-energy world #(0 1)))
(display "  Free energy after update: ") (display fe3) (newline)
(display "  Agent thinks: 'Wet ground AND umbrellas. Very confident: rain.'")
(newline) (newline)

;; --- Step 4: Compare surprise levels ---
(display "Step 4: Reasoning trace")
(newline)
(display "  Surprise with no evidence: ") (display fe1) (newline)
(display "  Surprise after wet ground: ") (display fe2) (newline)
(display "  Surprise after umbrellas:  ") (display fe3) (newline)
(display "  (Lower free energy = less surprise = better model)")
(newline) (newline)

;; --- Step 5: Expected free energy for decision-making ---
(display "Step 5: Decision — should I take an umbrella?")
(newline)
(define efe (expected-free-energy world #(0 1)))
(display "  Expected free energy of 'go outside': ") (display efe) (newline)
(display "  Agent decides: 'High EFE → take an umbrella to reduce future surprise.'")
(newline) (newline)

(display "=== Reasoning complete ===") (newline)
```

### The Thinking Process Explained

This isn't just a classifier — it's an agent that **reasons**:

1. **Prior beliefs** → the agent starts uncertain. Free energy is moderate.
2. **Evidence integration** → wet ground shifts beliefs toward rain.
   Free energy drops as the model better explains the evidence.
3. **Converging evidence** → umbrellas confirm rain. Free energy drops
   further. The agent becomes confident.
4. **Decision under uncertainty** → expected free energy predicts future
   surprise. High EFE means "this action leads to states my model can't
   predict well" → the agent acts to minimise future surprise.

This is the Free Energy Principle: the agent's "thinking" is the
process of updating a generative model to minimise the divergence
between predictions and observations.

---

## Why This Matters

### Self-improving programs are possible because:

1. **Autodiff makes programs differentiable** — gradients flow through
   any computation, so a program can ask "how should I change my
   parameters to do better?" and get an exact answer.

2. **The consciousness engine makes programs that reason** — factor
   graphs model uncertainty, belief propagation integrates evidence,
   free energy quantifies surprise, and CPT updates enable learning.

3. **The weight matrix transformer takes this further** — Eshkol
   programs literally ARE neural network weights. Optimising the
   weights optimises the program. A gradient step on the transformer
   creates a slightly different interpreter — one that might execute
   programs faster or more accurately.

These three capabilities — differentiable execution, probabilistic
reasoning, and programs-as-weights — are the foundation for machine
intelligence that improves itself.

---

*This is Tutorial 26 in the [Eshkol Tutorial Series](https://github.com/tsotchke/eshkol/blob/master/docs/tutorials/README.md).*
