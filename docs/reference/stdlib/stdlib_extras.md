# `stdlib` extras — random tensors, timing, keyword-arg support

**Source**: [`lib/stdlib.esk`](../../../lib/stdlib.esk)
**Require**: `(require stdlib)` — these are defined **directly in `lib/stdlib.esk`** (not a separate module), so they are always available whenever `stdlib` is loaded.

Beyond re-exporting the core modules, `lib/stdlib.esk` defines a handful of top-level helpers: random tensor constructors, high-precision timing utilities, and the internal keyword-argument support functions the parser lowers `#:key` calls onto.

## Random tensors

### `(random-tensor dims)`
Uniform-random tensor with shape `dims` (a **list** of dimension sizes). Wraps `(apply rand dims)`.

```scheme
(require stdlib)
(display (vector-length (random-tensor (list 3)))) (newline)   ;; illustrative: values are random
```
```
3
```

### `(random-normal-tensor dims)`
Standard-normal random tensor with shape `dims` (a list). Wraps `(apply randn dims)`. A `(list 2 2)` shape yields a 4-element (flattened) tensor.

```scheme
(display (vector-length (random-normal-tensor (list 2 2)))) (newline)   ;; illustrative
```
```
4
```

> `dims` must be a **list** (it is `apply`d as the argument list to `rand`/`randn`), e.g. `(list 3)` or `(list 2 2)` — not a bare number.

## Timing utilities

All derive from the built-in `current-time-ns`. Outputs below are illustrative (wall-clock, nondeterministic); the examples display booleans so they are reproducible.

### `(current-time-us)`
Current Unix time in microseconds, `(/ (current-time-ns) 1000.0)`.

```scheme
(display (> (current-time-us) 0.0)) (newline)
```
```
#t
```

### `(time-ns thunk)`
Run `thunk` once, return elapsed nanoseconds.

```scheme
(display (> (time-ns (lambda () (+ 1 2))) 0)) (newline)
```
```
#t
```

### `(time-us thunk)`
Run `thunk` once, return elapsed microseconds (`time-ns / 1000`).

```scheme
(display (>= (time-us (lambda () (+ 1 2))) 0.0)) (newline)
```
```
#t
```

### `(time-it thunk iterations)`
Run `thunk` `iterations` times, return the **average** time per call in nanoseconds.

```scheme
(display (>= (time-it (lambda () (+ 1 2)) 100) 0.0)) (newline)
```
```
#t
```

## Keyword-argument support (internal)

`__keyword-member?`, `__keyword-args-validate`, and `__keyword-arg` are **internal parser-support functions** — you do not call them directly. When you write a call with keyword arguments, the parser lowers the keyword formals through these helpers at function/lambda entry. A user call like `(f #:scale 2)` reaches the callee as a variadic tail `(#:scale 2)`, and `__keyword-arg` looks up the value by key (with `__keyword-args-validate` / `__keyword-member?` checking the keyword is expected and has a value).

The user-facing feature is ordinary keyword arguments:

```scheme
;; example.esk
(define (scale-it x #:scale s)
  (* x s))
(display (scale-it 10 #:scale 3)) (newline)
```
```
30
```

Signatures of the internal helpers, for reference:

- `(__keyword-member? key keys)` → `#t`/`#f` — is `key` in the list `keys`.
- `(__keyword-args-validate args keys)` → `#t` or raises — every keyword in the `(key val …)` tail `args` must be in `keys` and have a following value.
- `(__keyword-arg args key)` → the value paired with `key` in `args`, or raises if missing.

## Auto-load note

`lib/stdlib.esk` re-exports the core modules plus `signal.fft`, `signal.filters`, `core.manifold`, and `ml.optimization`. It does **not** require `math`, `ml.activations`, or `core.testing` — those must be required individually (see their respective pages).
