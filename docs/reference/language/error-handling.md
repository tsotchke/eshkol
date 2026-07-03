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
and the `clause`s are tried like a `cond` (each `(test result …)`, optional
`else`). The value of the matching clause becomes the value of the `guard` form.

```scheme
(display (guard (e ((symbol? e) (list 'sym e))
                   ((string? e) (list 'str e))
                   (else (list 'other e)))
  (raise "oops"))) (newline)
```
```
(str oops)
```

### Known issue — `'sym` sugar inside `guard` (ESH-0106)

Inside a `guard` form, the `'` reader sugar is compiled as a variable reference,
both in clause bodies and in a `raise` argument.

```scheme
;; WRONG: 'boom is read as a variable
(guard (e (#t e)) (raise 'boom))   ; => error: Undefined variable: boom
```
**Always use the explicit `(quote …)` form inside `guard`:**
```scheme
(display (guard (e (#t e)) (raise (quote boom)))) (newline)  ; => boom
```

### Known issues — differential `guard` findings (ESH-0101 / ESH-0102)

The project's differential/adversarial harness has open findings around `guard`:
value corruption across clause paths (ESH-0101) and an optimization-level-dependent
crash when a displayed `guard` catching a `raise` is followed by a second `guard`
form (ESH-0102, reported at `-O1`/`-O2`/`-O3`). Simple two-`guard` programs run
correctly at `-O0`/JIT and in many `-O2` cases; treat heavy `guard` nesting under
aggressive optimization as not-yet-hardened and test at your target `-O` level.

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
  (with-exception-handler
    (lambda (e) (display "handler ") 100)
    (lambda () (+ 1 (raise-continuable 'warn)))))
```
The above uses `raise-continuable`, which is **not available** in the native path
(`Unknown function`); it is VM-only. `with-exception-handler` combined with a plain
`raise` (non-continuable, escaping) works in native code.

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
