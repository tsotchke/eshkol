# Error Handling

## `raise`

```
(raise obj)
```
Raises `obj` as an exception. Any value may be raised — a symbol, number, string,
list, or an error object built with `error`.

```scheme
(define sym 'boom)
(display (guard (e (#t (list 'caught e))) (raise sym))) (newline)
(display (guard (e (#t (list 'caught e))) (raise 42)))  (newline)
(display (guard (e (#t (list 'caught e))) (raise (list 'my-error "detail")))) (newline)
```
```
(caught boom)
(caught 42)
(caught (my-error detail))
```

## `guard`

```
(guard (var clause …) body …)
```
Evaluates `body`. If an exception is raised, `var` is bound to the raised object
and the `clause`s are tried like a `cond`. The value of the matching clause
becomes the value of the `guard` form.

```scheme
(display (guard (e ((symbol? e) (list 'sym e))
                   ((string? e) (list 'str e))
                   (else (list 'other e)))
  (raise "oops"))) (newline)
```
```
(str oops)
```

### Clause shapes

A guard clause *is* a `cond` clause, so all four shapes are available (R7RS
4.2.7):

| Shape | Value of the `guard` form |
|---|---|
| `(test expr …)` | the last `expr` |
| `(test => receiver)` | `(receiver <the test's value>)` |
| `(test)` | the **test's own** value |
| `(else expr …)` | the last `expr` |

```scheme
;; => passes the TEST's value (not `e`) to the receiver
(display (guard (e ((assoc 'b e) => cdr))
  (raise (list (cons 'a 1) (cons 'b 2))))) (newline)

;; a test-only clause returns what the test computed
(display (guard (e ((memq e (list 1 2 3))))
  (raise 2))) (newline)
```
```
2
(2 3)
```

`=>` and the test-only clause are checked on every engine — JIT, AOT at `-O0`
and `-O2`, and both VM axes — by the gate described in
[TESTING.md](../../TESTING.md#guard-coverage-gate-esh-0101).

`guard` is **not** a tail context in R7RS, and a self tail call in a guard body
is not one here either — see [tail-calls.md](tail-calls.md).

### Quoted symbols inside `guard` (ESH-0106, closed)

Both quote spellings behave identically inside a `guard` form — in clause bodies
and in the `raise` argument:

```scheme
(display (guard (e (#t e)) (raise 'boom))) (newline)
(display (guard (e (#t e)) (raise (quote boom)))) (newline)
```
```
boom
boom
```

### Differential `guard` findings (closed)

Two adversarial-harness findings used to be listed here: a `guard`-caught value
that was garbage and differed per clause path, and an optimization-level-
dependent crash when a displayed `guard` catching a `raise` was followed by a
second `guard` form at `-O1`/`-O2`/`-O3`. Both were fixed by #117 and are now
pinned by `tests/differential/corpus/41_guard_value_and_double.esk`, which every
execution axis must agree on, and re-checked per-engine against an independent
golden by the guard coverage gate
([TESTING.md](../../TESTING.md#guard-coverage-gate-esh-0101)) — including at
`-O2`, which is the level that used to crash.

Earlier revisions of this page attributed those two findings to the ledger ids
`ESH-0101` and `ESH-0102`. That was a misattribution: `ESH-0101` is the
recursion-depth guard-coverage item (a deep non-tail recursion dies without a
diagnostic — see [KNOWN_ISSUES.md](../../KNOWN_ISSUES.md)) and `ESH-0102` is
mutual tail-call optimization, since closed. Neither ever denoted a `guard`
defect.

## `error`

```
(error message irritant …)
```
Constructs and raises an error object carrying a message string and zero or more
irritants. In the native code path the caught object is **opaque**: it can be
caught and re-raised but prints as `#<exception>`.

```scheme
(display (guard (e (#t (list 'caught e))) (error "bad thing" 1 2))) (newline)
```
```
(caught #<exception>)
```

### Known limitation — error-object accessors (native path)

`error-object?`, `error-object-message`, and `error-object-irritants` are
implemented in the **bytecode VM backend** but are **not available in the native
LLVM path** (`-r` / AOT), where they report `Unknown function`. If you need to
inspect message/irritants in native code, `raise` a structured value you control
(e.g. `(raise (list 'my-error "message" irritants))`) and destructure it in the
`guard` clause.

## `with-exception-handler`

```
(with-exception-handler handler thunk)
```
Installs `handler` for the dynamic extent of `thunk`. This form **works** in the
native path.

```scheme
(display
  (guard (e (#t (list 'caught e)))
    (with-exception-handler
      (lambda (e) (display "handler ") 100)
      (lambda () (raise 'warn)))))
```

### `raise-continuable` is not implemented

`raise-continuable` is **not available on any substrate** — not the native LLVM
path and not the bytecode VM. It appears in no builtin table, no special-form
dispatch and no prelude; the name occurs exactly once in the compiler, as an
entry in an iteration-scope blacklist, which is not an implementation.
`docs/COMPLETE_LANGUAGE_SPECIFICATION.md` has this right ("all raises are
non-continuable"); earlier revisions of *this* page and of
[INDEX.md](INDEX.md) said it was "VM-only", which was never true of any build.
The correction is ledgered as `SW-80`, and implementing it is a build item, not
a documented limitation.

`with-exception-handler` itself **works** in the native path, combined with a
plain (non-continuable, escaping) `raise` — including under an enclosing
`guard`, which the guard coverage gate checks on every engine.

## The capability-denied signal

When a capability policy is active (see [capabilities.md](capabilities.md)),
attempting a denied operation does not raise a catchable exception in the usual
sense — the runtime prints a diagnostic to stderr and the operation returns a
benign value (`#f`/null), letting the program continue.

```scheme
(require core.capabilities)
(capability-install-policy! '(file-read))     ; env-read NOT allowed
(display (get-environment-variable "HOME")) (newline)
(display "after") (newline)
```
```
capability denied: env-read
#f
after
```
