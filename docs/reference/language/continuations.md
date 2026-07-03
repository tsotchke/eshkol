# Continuations: `call/cc` and `dynamic-wind`

## `call/cc` / `call-with-current-continuation`

```
(call/cc proc)
(call-with-current-continuation proc)   ; same thing
```
Captures the current continuation and passes it as a one-argument escape
procedure `k` to `proc`. Calling `(k v)` returns `v` to the point where `call/cc`
was invoked.

### Escape (upward) continuations

```scheme
(display (call/cc (lambda (k) (+ 1 (k 42))))) (newline)
(display (+ 1 (call-with-current-continuation (lambda (k) (k 10) 999)))) (newline)
```
```
42
11
```
In the first line, `(k 42)` abandons the pending `(+ 1 …)` and returns `42`
directly. In the second, `(k 10)` escapes before `999` is reached, so the outer
`(+ 1 …)` sees `10`.

### Re-invocable (multi-shot) continuations

A captured continuation can be stored and invoked more than once — it is a full
re-entrant continuation, not merely a one-shot escape.

```scheme
(define k #f)
(define n 0)
(display (+ 1 (call/cc (lambda (c) (set! k c) 0))))
(newline)
(set! n (+ n 1))
(if (< n 3) (k n))     ; re-invoke the saved continuation
(display "done") (newline)
```
```
1
2
3
done
```
Re-invoking `k` re-enters the `(+ 1 …)` context repeatedly with successive values,
demonstrating multi-shot behaviour.

### Known limitation — deep CPS chains (ESH-0080)

Very deep continuation-passing chains (e.g. the SICP ch4 `amb` nondeterministic
search) can crash with SIGILL beyond a moderate depth (reported around n ≳ 16 for
that workload). Ordinary escape and modest re-entrant use are fine; do not rely on
extremely deep continuation chaining.

## `dynamic-wind`

```
(dynamic-wind before thunk after)
```
Calls `before`, then `thunk`, then `after`. `after` is guaranteed to run when
control leaves `thunk` — including when a continuation escapes out of it.

```scheme
(dynamic-wind
  (lambda () (display "before "))
  (lambda () (display "during "))
  (lambda () (display "after ")))
(newline)
```
```
before during after 
```

`after` runs even on a non-local exit via a continuation:

```scheme
(call/cc (lambda (k)
  (dynamic-wind
    (lambda () (display "in "))
    (lambda () (k 'escaped))    ; escape out of the middle thunk
    (lambda () (display "out ")))))
(newline)
```
```
in out 
```
The `out` guard runs during the escape.
