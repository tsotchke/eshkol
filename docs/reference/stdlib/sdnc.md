# `core.sdnc` — self-improving DNC weight-program (bytecode-VM-as-transformer θ)

**Source**: [`lib/core/sdnc.esk`](../../../lib/core/sdnc.esk)
**Require**: `(require core.sdnc)` — **NOT** auto-loaded via `(require stdlib)`. `sdnc-program`, `sdnc-run`, `sdnc-weight-grad`, `sdnc-params`, `sdnc-set-params!`, `sdnc-improve!` and `sdnc?` are **native codegen builtins** (heap subtype for the SDNC handle) and resolve even without the require. The fine-tuning procedures (`sdnc-load-weights!`, `sdnc-mismatches`, `sdnc-repair!`, `sdnc-default-repair-beta`) are defined in `lib/core/sdnc.esk` over C entry points in `lib/core/sdnc_api.c` and need the require.

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

### `(sdnc-weight-grad θ input target smoothing)`
The same gradient with **backward smoothing** of the two saturating nonlinearities. `smoothing` is a positive real `beta` (gates and attention) or a pair `(gate-beta . attention-beta)`. The forward pass is unchanged; the backward pass differentiates the logistic gate as `sigma(beta * pre)` and the Layer-0 softmax as `softmax(beta * logits)`, and uses the exact forward values everywhere else. `beta = 1` is the exact gradient, bit for bit. For the constructed interpreter, whose saturated gates make the exact derivative of every control weight zero, `beta = 1/30` gives an informative derivative (see `sdnc-repair!`).

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
loss 0.125 -> 0.030796962212231688
```

## Fine-tuning the constructed interpreter

The constructed SDNC interpreter (the analytic weights of `lib/backend/weight_matrices.c`) runs every gate at a saturating scale, so its forward pass is exact and the exact derivative of its control weights (gate thresholds, gate columns, `W_Q`, `W_K`, `b_Q`) is identically zero. These procedures fine-tune the gate thresholds anyway, without softening the forward pass: the smoothed backward ranks one-unit threshold moves and the exact forward accepts a move only if more recorded transitions are reproduced byte for byte. Design: [ADR-0032](../../design/adr/0032-exact-forward-control-fine-tuning.md).

The weights and the recorded transitions come from the `weight_matrices` tool:

```sh
mkdir -p sdnc-artifacts
build/tools/weight_matrices --control-finetune --artifacts sdnc-artifacts
# writes sdnc-artifacts/constructed.qlmw, damaged.qlmw, transitions.sdtr
```

`constructed.qlmw` is the constructed interpreter, `damaged.qlmw` the same weights with the ADD dispatch switched off, and `transitions.sdtr` every forward transition of the 127-program verifier battery on the constructed weights.

### `(sdnc-load-weights! θ path)`
Replace θ's weights with a QLMW v3 weight file (the format `weight_matrices` writes; `ESHKOL_WEIGHTS_OUT` also writes it). Returns θ. Raises an error for an unreadable file or a file for another architecture.

### `(sdnc-mismatches θ transitions)`
Number of transitions in the `transitions` file (SDTR v1) whose next state θ does not reproduce byte for byte. `0` means θ executes every recorded step exactly as the teacher did.

### `(sdnc-repair! θ transitions [beta])`
Fine-tune θ's gate thresholds in place against the recorded transitions: tied construction gates first (neurons with identical gate column and bias move together), then single neurons, each move one construction unit (300). `beta` is the backward smoothing (default `sdnc-default-repair-beta` = 1/30). Returns a report alist:

| Key | Meaning |
|---|---|
| `transitions` | recorded transitions |
| `tied-gates` | tied construction gates |
| `mismatched-before` / `mismatched-after` | transitions not reproduced byte for byte |
| `gate-moves` / `neuron-moves` | accepted moves per level |
| `exact-control-gradient-zero` | `#t` if the exact derivative of every control weight was zero at the start |

```scheme
;; sdnc_repair.esk — run with ESHKOL_SDNC_ARTIFACTS=sdnc-artifacts
(require core.sdnc)
(define dir (getenv "ESHKOL_SDNC_ARTIFACTS"))
(define transitions (string-append dir "/transitions.sdtr"))
(define th (sdnc-program "interpreter"))

(sdnc-load-weights! th (string-append dir "/constructed.qlmw"))
(display "constructed: ") (display (sdnc-mismatches th transitions)) (newline)

(sdnc-load-weights! th (string-append dir "/damaged.qlmw"))
(display "damaged:     ") (display (sdnc-mismatches th transitions)) (newline)

(define report (sdnc-repair! th transitions))
(display "repaired:    ") (display (sdnc-mismatches th transitions)) (newline)
(display report) (newline)
```
```
EXAMPLE_OUTPUT
```

## Edge cases
- `sdnc?` on a non-SDNC value returns `#f` (does not error).
- `sdnc-run` is deterministic; `(sqnorm (vsub (sdnc-run θ x) (sdnc-run θ x)))` is exactly `0.0`.
- `sdnc-set-params!` expects a vector of exactly `(sdnc-params θ)` length.
- `sdnc-weight-grad` with a smoothing argument that is not a positive real or a pair of positive reals raises a type error.
- `sdnc-repair!` changes only gate thresholds, each by whole construction units; thresholds that no recorded transition exercises are left where they are.

## Verification note
`tests/sdnc/sdnc_finetune_test.esk` (ctest `sdnc_finetune_surface_jit`) runs the fine-tuning surface on the artifacts written by ctest `sdnc_control_finetune`: 15/15 checks. `tests/sdnc/sdnc_finetune_unit_test.c` (ctest `sdnc_finetune_unit`) checks the smoothing rule against finite differences and the beta = 1 bit-identity: 20/20.

`tests/sdnc/sdnc_api_test.esk` passes 13/13 under `eshkol-run -r`: handle/predicate, params length > 1000, params round-trip, deterministic 256-dim forward, `sdnc-weight-grad` length == `sdnc-params` length, and `sdnc-improve!` loss decrease (observed 0.25 → 0.0616 in the suite's setup). This design is described in memory as *"VM-as-transformer memory-ops design"*. No `.swarm` ledger issues reference the SDNC builtins.
