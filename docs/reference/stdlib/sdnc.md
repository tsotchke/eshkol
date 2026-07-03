# `core.sdnc` — self-improving DNC weight-program (bytecode-VM-as-transformer θ)

**Source**: [`lib/core/sdnc.esk`](../../../lib/core/sdnc.esk)
**Require**: `(require core.sdnc)` — **NOT** auto-loaded via `(require stdlib)`. The functions are **native codegen builtins** (heap subtype for the SDNC handle) and resolve even without the require; require it anyway for clarity. The `.esk` file only carries the `provide` list.

Exposes a **trainable weight-program θ** — a genuine transformer forward pass plus its exact weight-gradient — so programs can do gradient-based self-improvement on real SDNC weights from Eshkol. The weight kernels (forward / backward-through-weights / SGD step) mirror `lib/backend/weight_matrices.c` byte-for-byte (see `lib/core/sdnc_core.h`). All vectors are Eshkol `#(...)` float vectors.

## Functions

### `(sdnc-program name)`
Create a named, reproducible weight set θ. Same `name` ⇒ same initial weights. Returns an opaque SDNC handle.

### `(sdnc? x)`
Predicate: `#t` iff `x` is an SDNC handle.

### `(sdnc-params θ)`
Return the flattened trainable weight vector (a long `#(...)`; e.g. length 12,220,416 for the default program — the full transformer weight set).

### `(sdnc-set-params! θ vec)`
Write a flat weight vector back into θ (must match the `sdnc-params` length). Returns θ. Round-trips exactly with `sdnc-params`.

### `(sdnc-run θ input)`
Transformer forward pass. `input` is a `D`-dim `#(...)` vector (D=256 for the default program); returns the `D`-dim output vector. **Deterministic** — running twice on the same input gives bit-identical output.

### `(sdnc-weight-grad θ input target)`
Exact gradient `dL/dWEIGHTS` of the loss between `sdnc-run(θ, input)` and `target`. Returns a flat vector whose length **equals** `(vector-length (sdnc-params θ))` — this invariant is the key contract for plugging θ into an external optimizer.

### `(sdnc-improve! θ data steps lr)`
Run `steps` of SGD on the weights with learning rate `lr`. `data` is a single `(cons input target)` example. Mutates and returns θ; the loss on the example decreases.

```scheme
;; sdnc.esk
(require core.sdnc)
(define (sqnorm v)
  (let ((n (vector-length v)))
    (let loop ((i 0) (acc 0.0))
      (if (< i n) (loop (+ i 1) (+ acc (* (vector-ref v i) (vector-ref v i)))) (* 0.5 acc)))))
(define (vsub a b)
  (let* ((n (vector-length a)) (o (make-vector n 0.0)))
    (let loop ((i 0)) (if (< i n) (begin (vector-set! o i (- (vector-ref a i) (vector-ref b i))) (loop (+ i 1))) o))))

(define th (sdnc-program "fib"))
(display "sdnc? ")(display (sdnc? th))(newline)
(define p0 (sdnc-params th))
(display "params length ")(display (vector-length p0))(newline)

(define input (make-vector 256 0.0))
(vector-set! input 0 1.0)(vector-set! input 1 0.5)
(define out (sdnc-run th input))
(display "run length ")(display (vector-length out))(newline)

;; reachable target = forward(input) nudged, so loss > 0
(define target (let* ((n (vector-length out)) (o (make-vector n 0.0)))
  (let loop ((i 0)) (if (< i n) (begin (vector-set! o i (vector-ref out i)) (loop (+ i 1))) o))))
(vector-set! target 0 (+ (vector-ref target 0) 0.5))
(define g (sdnc-weight-grad th input target))
(display "grad length == params length ")(display (= (vector-length g) (vector-length p0)))(newline)

(define (loss-now) (sqnorm (vsub (sdnc-run th input) target)))
(define before (loss-now))
(sdnc-improve! th (cons input target) 30 0.01)
(display "loss ")(display before)(display " -> ")(display (loss-now))(newline)
```
```
sdnc? #t
params length 12220416
run length 256
grad length == params length #t
loss 0.125 -> 0.030797
```

## Edge cases
- `sdnc?` on a non-SDNC value returns `#f` (does not error).
- `sdnc-run` is deterministic; `(sqnorm (vsub (sdnc-run θ x) (sdnc-run θ x)))` is exactly `0.0`.
- `sdnc-set-params!` expects a vector of exactly `(sdnc-params θ)` length.

## Verification note
`tests/sdnc/sdnc_api_test.esk` passes 13/13 under `eshkol-run -r`: handle/predicate, params length > 1000, params round-trip, deterministic 256-dim forward, `sdnc-weight-grad` length == `sdnc-params` length, and `sdnc-improve!` loss decrease (observed 0.25 → 0.0616 in the suite's setup). This design is described in memory as *"VM-as-transformer memory-ops design"*. No `.swarm` ledger issues reference the SDNC builtins.
