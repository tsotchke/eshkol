# `core.ad.tape` — reverse-mode automatic differentiation (Wengert tape + custom ops)

**Source**: [`lib/core/ad/tape.esk`](../../../lib/core/ad/tape.esk)
**Require**: `(require core.ad.tape)` — **NOT** auto-loaded via `(require stdlib)`; require it explicitly. The stateful pure-Scheme layer (`with-tape`, `tape-mul`, …) is defined in this file and fails with *"called undefined function 'with-tape'"* if the module is not required. The low-level `ad-*` layer is codegen builtins and works even without the require, but require it anyway for clarity.

This module provides **two independent reverse-mode AD systems**:

1. **Low-level Wengert tape** (`ad-*`) — codegen builtins backed by native C. Fast, but only supports the fixed set of builtin ops. Every op takes the **tape as its first argument**.
2. **Stateful op-recording tape** (`with-tape` / `tape-*` / `record-op!`) — pure Scheme in this file. Each op is recorded with an explicit backward **closure**, so a single loss can route through builtin ops *and* hand-written custom ops (with exact or finite-difference backward), and reverse mode flows through all of them. Node values and gradients may be **scalars or vectors** (embeddings/tensors) uniformly. This is the keystone that makes reasoning ops (soft-unify, BP steps, free energy, encoder forward) differentiable.

A **node** in the stateful tape is a mutable 2-slot vector `#(value grad)`. A **tape** is a mutable `#(ops nodes)` where `ops` is stored in reverse-chronological order (already the reverse-topological order the backward pass needs).

---

## Low-level Wengert tape (`ad-*` builtins)

All `ad-*` ops take the tape first. Nodes carry a forward value; `ad-backward` seeds the output adjoint to 1 and propagates; `ad-gradient` then reads the accumulated adjoint at a node.

### `(ad-tape-new)`
Create a fresh low-level tape handle.

### `(ad-var tape value)`
Create a differentiable **variable** node (a gradient target) with the given scalar value.

### `(ad-const tape value)`
Create a **constant** node (value, no gradient accumulation as a target).

### `(ad-node-value tape node)` / `(ad-value tape node)`
Read a node's forward value. **Takes the tape as the first argument** — `(ad-node-value node)` with one arg silently returns `()`. `ad-value` is a builtin alias.

### `(ad-add tape a b)` `(ad-sub tape a b)` `(ad-mul tape a b)` `(ad-div tape a b)`
Binary arithmetic ops between two nodes; return a new node.

### `(ad-sin tape a)` `(ad-cos tape a)` `(ad-exp tape a)` `(ad-log tape a)` `(ad-sqrt tape a)` `(ad-neg tape a)` `(ad-abs tape a)` `(ad-relu tape a)` `(ad-sigmoid tape a)` `(ad-tanh tape a)`
Unary ops on a node; return a new node.

### `(ad-backward tape output-node)`
Run the reverse pass from `output-node` (seeds its adjoint to 1.0).

### `(ad-gradient tape node)`
Read the accumulated gradient at `node` after `ad-backward`. (Builtin aliases `ad-gradient-of`, `ad-value-of`, and `ad-pow`/`ad-tape-length`/`ad-tape-release` also exist but are outside this module's `provide` block; see [Extra builtins](#extra-builtins-not-in-provide).)

```scheme
;; adlow.esk
(require core.ad.tape)
(define tp (ad-tape-new))
(define x (ad-var tp 2.0))
(define y (ad-mul tp x x))          ; y = x^2
(ad-backward tp y)
(display "y=")(display (ad-node-value tp y))
(display " dy/dx=")(display (ad-gradient tp x))(newline)

(define t2 (ad-tape-new))
(define a (ad-var t2 0.5))
(define s (ad-add t2 (ad-sin t2 a) (ad-exp t2 a)))  ; sin a + exp a
(ad-backward t2 s)
(display "s=")(display (ad-node-value t2 s))
(display " ds/da=")(display (ad-gradient t2 a))(newline)
```
```
y=4 dy/dx=4
s=2.12815 ds/da=2.5263
```
(dy/dx = 2x = 4; ds/da = cos 0.5 + exp 0.5 = 2.5263.)

Edge case: **arg order matters.** Calling `ad-node-value`/`ad-value` with only the node (omitting the tape) returns `()` rather than erroring.

---

## Stateful op-recording tape

### `(make-tape)`
Create a fresh stateful tape. **Takes zero arguments.** Returns `#(ops nodes)` (both initially `'()`).

### `(with-tape name thunk)`
Bind a fresh tape as the dynamic *current tape* for the duration of `(thunk)`, then restore the previous current tape and return the thunk's result. `name` is a label (string) for readability only. Signature is `(with-tape name thunk)` — the thunk takes **no** arguments and typically calls `(current-tape)` to get the tape.

### `(current-tape)`
Return the tape currently installed by `with-tape`, or `#f` outside any `with-tape` scope.

```scheme
;; withtape.esk
(require core.ad.tape)
(display (with-tape "demo"
  (lambda ()
    (let* ((t (current-tape))
           (x (tape-input t 3.0))
           (y (tape-mul t x x)))          ; x^2
      (tape-gradient t y (list x))))))    ; => (6)
(newline)
(display "outside: ")(display (current-tape))(newline)
```
```
(6)
outside: #f
```

### `(tape-input t value)`
Create a **leaf node** (a gradient target) with `value` (scalar or vector). Registers it with the tape.

### `(tape-const t value)`
Create a leaf node used as a constant (same representation as `tape-input`; you simply don't ask for its gradient).

### `(node-value node)` / `(node-grad node)`
Read a node's forward value / its accumulated gradient. Unlike the low-level `ad-node-value`, these take **only the node** (a node is a plain `#(value grad)` vector). `node-grad` is meaningful after `tape-gradient` has run.

### `(record-op! t inputs out-value backward)` — keystone
Record a **custom op** on the tape.
- `inputs` — list of input nodes.
- `out-value` — the already-computed forward output value (scalar or vector).
- `backward` — a closure `(lambda (grad-out) -> list-of-input-grads)`, returning one gradient per input node, in the **same order** as `inputs`. The closure captures whatever forward context it needs.
Returns the new output node. This is what lets any op whose backward is not a plain builtin composition participate in reverse mode.

```scheme
;; recordop.esk — custom softplus op mixed with builtin mul/add
(require core.ad.tape)
(define (softplus v) (log (+ 1.0 (exp v))))
(define (sigmoid v)  (/ 1.0 (+ 1.0 (exp (- v)))))
(define g
  (with-tape "loss"
    (lambda ()
      (let* ((t (current-tape))
             (x (tape-input t 1.5))
             (y (tape-input t 0.5))
             (m (tape-mul t x y))                     ; builtin op (recorded)
             (s (let ((mv (node-value m)))
                  (record-op! t (list m) (softplus mv) ; CUSTOM op
                              (lambda (gr) (list (* gr (sigmoid mv)))))))
             (y2 (tape-mul t y y))
             (loss (tape-add t s y2)))                ; softplus(x*y) + y^2
        (tape-gradient t loss (list x y))))))
(display g)(newline)
```
```
(0.339589 2.01877)
```
(Matches central finite differences of `softplus(x*y)+y^2` at `(1.5, 0.5)`: `d/dx = sigmoid(xy)*y`, `d/dy = sigmoid(xy)*x + 2y`; verified in `tests/ad/stateful_tape_test.esk`.)

Custom ops also handle **vector-valued** nodes — `out-value` and each returned gradient may be a vector (used for embedding/tensor ops):
```scheme
;; a tensor dot-product custom op: grad_a = b, grad_b = a
(record-op! t (list na nb) (vdot av bv)
  (lambda (gr) (list (scale-vec gr bv) (scale-vec gr av))))
```

### `(record-fd-op! t inputs fwd)`
Record a custom op whose backward is computed by **central finite differences** on `fwd`. `fwd` takes the list of input **values** (scalars) and returns a scalar. Use when an analytic backward is unavailable or not worth deriving (e.g. differentiating through belief-propagation reconvergence / free energy). Prefer `record-op!` with an exact backward when you have one. Step size `tape-fd-eps = 1e-6`.

```scheme
;; recordfd.esk — differentiate an arbitrary scalar fn with no hand backward
(require core.ad.tape)
(define (fn vals)
  (let ((x (car vals)) (y (cadr vals)))
    (+ (* (* x x) y) (sin x))))          ; x^2*y + sin x
(display
  (with-tape "fd"
    (lambda ()
      (let* ((t (current-tape))
             (x (tape-input t 2.0))
             (y (tape-input t 3.0))
             (fo (record-fd-op! t (list x y) fn)))
        (tape-gradient t fo (list x y))))))   ; expect (2xy+cos x, x^2) = (12+cos2, 4)
(newline)
```
```
(11.5839 4)
```

### Builtin recorded ops (with backward rules)
Each returns a new output node; each takes the **tape first**.

| Op | Signature | Forward | d/dinput |
|----|-----------|---------|----------|
| `tape-add` | `(tape-add t a b)` | `a+b` | `(g, g)` |
| `tape-sub` | `(tape-sub t a b)` | `a-b` | `(g, -g)` |
| `tape-mul` | `(tape-mul t a b)` | `a*b` | `(g*b, g*a)` |
| `tape-div` | `(tape-div t a b)` | `a/b` | `(g/b, -g*a/b^2)` |
| `tape-sin` | `(tape-sin t a)` | `sin a` | `g*cos a` |
| `tape-cos` | `(tape-cos t a)` | `cos a` | `-g*sin a` |
| `tape-exp` | `(tape-exp t a)` | `exp a` | `g*exp a` |
| `tape-log` | `(tape-log t a)` | `log a` | `g/a` |
| `tape-neg` | `(tape-neg t a)` | `-a` | `-g` |
| `tape-square` | `(tape-square t a)` | `a*a` | `2*g*a` |
| `tape-sqrt` | `(tape-sqrt t a)` | `sqrt a` | `g/(2*sqrt a)` |
| `tape-tanh` | `(tape-tanh t a)` | `tanh a` | `g*(1-tanh^2)` |
| `tape-sigmoid` | `(tape-sigmoid t a)` | `1/(1+e^-a)` | `g*out*(1-out)` |
| `tape-relu` | `(tape-relu t a)` | `max(a,0)` | `g if a>0 else 0` |
| `tape-pow` | `(tape-pow t a p)` | `a^p` (**p is a constant number, not a node**) | `g*p*a^(p-1)` |

```scheme
;; buops.esk
(require core.ad.tape)
(define (g1 opf x0)
  (with-tape "o" (lambda ()
    (let* ((t (current-tape)) (x (tape-input t x0)) (o (opf t x)))
      (list (node-value o) (car (tape-gradient t o (list x))))))))
(display "square@3 (val grad) ")(display (g1 tape-square 3.0))(newline)
(display "sqrt@4 ")(display (g1 tape-sqrt 4.0))(newline)
(display "tanh@0 ")(display (g1 tape-tanh 0.0))(newline)
(display "sigmoid@0 ")(display (g1 tape-sigmoid 0.0))(newline)
(display "relu@2 ")(display (g1 tape-relu 2.0))(newline)
(display "relu@-2 ")(display (g1 tape-relu -2.0))(newline)
;; tape-pow: p is a plain number
(display "pow(3,2) ")(display (with-tape "p" (lambda ()
  (let* ((t (current-tape)) (x (tape-input t 3.0)) (o (tape-pow t x 2.0)))
    (list (node-value o) (car (tape-gradient t o (list x))))))))(newline)
```
```
square@3 (val grad) (9 6)
sqrt@4 (2 0.25)
tanh@0 (0 1)
sigmoid@0 (0.5 0.25)
relu@2 (2 1)
relu@-2 (0 0)
pow(3,2) (9 6)
```

### `(tape-gradient t output targets)`
Run reverse mode from `output` (zeros all node grads, seeds `output` adjoint to 1.0, walks recorded ops in reverse order), then return the list of gradients at the `targets` nodes, in order. Can be called repeatedly (it re-zeros each time). Handles tensor adjoints unconditionally; skips only zero scalar adjoints.

### `(tape-snapshot t)`
Return a list of the current node **values** (in tape-node order) for later replay.

### `(tape-restore t snapshot)`
Write the snapshot values back into the tape's nodes; returns the tape.

```scheme
;; snap.esk
(require core.ad.tape)
(define ts (make-tape))
(define nx (tape-input ts 3.0))
(define snap (tape-snapshot ts))
(vector-set! nx 0 99.0)          ; clobber the node value
(tape-restore ts snap)
(display (node-value nx))(newline)   ; restored to 3
```
```
3
```

## Extra builtins (not in `provide`)
These low-level builtins exist in codegen but are outside this module's `provide` block; they are exercised by `tests/vm/ad_tape_lowlevel_regression.esk`, which passes on **all three substrates — JIT, AOT and the bytecode VM**:
- `(ad-pow tape a p-node)` — power on the low-level tape (`p` here is a **node**, e.g. `(ad-const tape 3.0)`), distinct from stateful `tape-pow` whose `p` is a number.  Forward value is ordinary `pow(a, p)`; reverse derivatives are `d/da = p·a^(p-1)` and `d/dp = value·ln(a)`.
- `(ad-value-of tape node)`, `(ad-gradient-of tape node)` — aliases of `ad-value` / `ad-gradient`.
- `(ad-tape-length tape)` — number of nodes recorded.
- `(ad-tape-release tape)` — release the low-level tape; idempotent (safe to call twice).

`reverse-gradient` (a high-level, operator-overloading convenience that boxes a
point as tape variables and reads back the gradient vector) is a VM builtin and
is subsumed on the LLVM path by the `gradient` operator; the low-level surface
above is the portable, substrate-uniform contract, so the low-level regression
builds the equivalent reverse-mode gradient directly on the tape
(`ad-var`/`ad-mul`/`ad-add`/`ad-backward`/`ad-gradient`) rather than calling
`reverse-gradient`.

## Internal helpers (provided but user-facing use is uncommon)
`node-value` and `node-grad` are the documented accessors. The following are used internally by the module and generally not called directly: the `tape-*` list helpers, `val-*` value helpers, and `node-grad-set!/add!` are **not** exported.

## Related ledger entries
- The stateful tape + custom-op backward is the **TAPE** feature (memory: *"Stateful AD tape + custom-op backward"*, PR #48). Gotcha recorded there: the tape is pure Scheme because C has no closure invoker; and core modules cannot use `for-each`/`map` (AOT would fail), which is why this file carries its own `tape-for-each`/`tape-map1`.
- The native `gradient`/`hessian`/`jacobian` **operators** (a *separate* AD system from this tape) have open/closed issues **ESH-0067, ESH-0070, ESH-0071, ESH-0072, ESH-0078, ESH-0081, ESH-0095, ESH-0096, ESH-0097**. Those do **not** affect this tape module — verified independently here.

## Verification note
Full acceptance suite `tests/ad/stateful_tape_test.esk` (22/22), `tests/ad/free_energy_ad_test.esk` (2/2), `tests/vm/ad_tape_lowlevel_regression.esk`, and `tests/stdlib/v12_ad_tape_test.esk` all pass under `eshkol-run -r`.
