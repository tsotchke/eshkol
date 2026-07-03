# `core.testing` — minimal test framework

**Source**: [`lib/core/testing.esk`](../../../lib/core/testing.esk)
**Require**: `(require core.testing)` — **must be required individually**. It is **NOT** auto-loaded by `(require stdlib)`.

A function-based (not macro-based) test framework: register named tests whose bodies call `check-*` assertions, then `(run-tests)` executes them in source order, prints a per-test and summary report, and returns `#t`/`#f`.

## Why it isn't part of `stdlib`

Baking `core.testing` into the precompiled `stdlib.o` triggers the symbol-renamer / external-declaration path that currently mis-handles a precompiled module's mutable internal state, producing a **duplicate `_*tests*`** symbol in the user's object file. To avoid that, `lib/stdlib.esk` deliberately omits it (see the comment block there) and users must `(require core.testing)` explicitly.

The API is procedures, not macros, on purpose: Eshkol's `define-syntax` does not propagate macro definitions across the precompiled-stdlib boundary (the parser has already expanded `testing.esk` before user code is read), so a macro-based `(define-test …)` would not expand in user source. Wrap check forms in your own `(lambda () …)` thunk to register a test.

## Functions

### `(register-test name thunk)`
Append a test: `name` is a string, `thunk` is a zero-argument procedure whose body runs the `check-*` assertions. Tests are stored newest-first and replayed in source order by `run-tests`.

### `(check-equal? actual expected)`
Structural equality (dispatches on string/number/boolean/char/symbol/null, falling back to `equal?`). On mismatch prints `FAIL: expected …, got …` and increments the fail counters.

### `(check-true v)` / `(check-false v)`
Assert `v` is truthy / false respectively.

### `(check-approx actual expected tol)`
Float equality within `tol`: passes when `|actual − expected| < tol`.

### `(check-raises thunk)`
Assert that calling `thunk` raises (via `guard`). If the thunk returns normally, that is a failure.

### `(run-tests)`
Run every registered test in source order, printing `[test] <name> … OK` / `… FAIL (n check(s))` per test and a final summary. Resets the pass/fail counters at the start. Returns `#t` if all passed, `#f` otherwise.

### `(reset-tests!)`
Clear the registry and all counters, and reset the runtime logic/predicate registry (safe even when the consciousness engine isn't loaded).

Full run:

```scheme
;; example.esk
(require core.testing)
(register-test "addition"
  (lambda () (check-equal? (+ 2 2) 4) (check-true (> 5 3))))
(register-test "approx"
  (lambda () (check-approx 3.14159 3.14 0.01)))
(register-test "raises"
  (lambda () (check-raises (lambda () (error "boom")))))
(register-test "failing"
  (lambda () (check-equal? 1 2) (check-false #t)))
(run-tests)
```
```
  [test] addition … OK
  [test] approx … OK
  [test] raises … OK
  [test] failing    FAIL: expected 2, got 1
    FAIL: expected false, got #t
 … FAIL (2 check(s))

=== Test summary ===
  Passed: 4
  Failed: 2
  RESULT: FAILURES
```

## Exported state variables

These are `provide`d so tooling can inspect results after `(run-tests)`:

- `*tests*` — the registry, a list of `(name thunk)` pairs (stored newest-first).
- `*test-pass-count*` — number of `check-*` assertions that passed in the last run.
- `*test-fail-count*` — number that failed (plus one per test whose thunk raised).
- `*current-test-fails*` — failing checks in the currently running test (reset per test).
- `*current-test-name*` — name of the currently running test.

```scheme
(require core.testing)
(register-test "t" (lambda () (check-equal? (+ 1 1) 2)))
(run-tests)
(display "pass=") (display *test-pass-count*) (newline)
```
```
  [test] t … OK

=== Test summary ===
  Passed: 1
  Failed: 0
  RESULT: OK
pass=1
```

## Internal helpers (not in `provide`)

`record-check!` and `testing-equal?` are used by the `check-*` procedures and are not exported.
