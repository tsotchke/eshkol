# Special Forms: definitions, `lambda`, `let`-family, `begin`

## `define`

```
(define name value)
(define (name arg …) body …)          ; function-definition sugar
(define (name arg … . rest) body …)   ; variadic
```

Binds `name` in the enclosing scope. The function-definition sugar
`(define (f x) …)` is equivalent to `(define f (lambda (x) …))`.

```scheme
(define pi 3.14159)
(define (square x) (* x x))
(display (square 9)) (newline)
```
```
81
```

Internal `define`s at the start of a body are treated as `letrec*` bindings
(R7RS semantics): they may refer to each other and are collected only while they
are *consecutive* at the start of the body.

```scheme
(define (f)
  (define a 1)
  (define b (+ a 1))   ; may reference the previous define
  (+ a b))
(display (f)) (newline)
```
```
3
```

### Known issues / non-support

- **Curried definition sugar is not supported.** `(define ((adder x) y) …)`
  fails to parse: `error: expected function name in define`. Write the explicit
  nested `lambda` instead: `(define (adder x) (lambda (y) (+ x y)))`.
- **ESH-0092** — top-level `define`d globals are emitted as raw C symbols. A
  top-level name that collides with a libc symbol (e.g. `free`, `log`) corrupts
  the process; `(define free 0)` runs but crashes with **SIGBUS** at teardown.
  Avoid libc names for top-level globals. See
  [binding-mutation-and-scope.md](binding-mutation-and-scope.md).

## `lambda`

```
(lambda (arg …) body …)
(lambda (arg … . rest) body …)   ; rest collects extra args into a list
(lambda args body …)             ; all args collected into a single list
```

```scheme
(display ((lambda (x y) (+ x y)) 3 4)) (newline)
(display ((lambda args args) 1 2 3)) (newline)
```
```
7
(1 2 3)
```

See [functions-and-parameters.md](functions-and-parameters.md) for variadic and
keyword-argument formals.

## `begin`

```
(begin expr …)
```
Evaluates each `expr` in order and returns the value of the last.

```scheme
(begin (display "a") (display "b") (newline))
```
```
ab
```

## `let`

```
(let ((var init) …) body …)
```
All `init` expressions are evaluated in the *enclosing* scope; the new bindings
are visible only in `body`.

```scheme
(define x 10)
(let ((a 1) (b 2)) (display (+ a b x)) (newline))
```
```
13
```

## `let*`

```
(let* ((var init) …) body …)
```
Like `let`, but each `init` can see the bindings established by the earlier
clauses.

```scheme
(let* ((a 1) (b (+ a 1))) (display b) (newline))
```
```
2
```

## `letrec` and `letrec*`

```
(letrec ((var init) …) body …)
(letrec* ((var init) …) body …)
```
All names are in scope for every `init`, enabling mutual recursion. Use for
locally-defined recursive procedures.

```scheme
(letrec ((ev? (lambda (n) (if (= n 0) #t (od? (- n 1)))))
         (od? (lambda (n) (if (= n 0) #f (ev? (- n 1))))))
  (display (ev? 10)) (newline))
```
```
#t
```

### Per-activation instance isolation (fixed, ESH-0075)

Each activation of an enclosing procedure produces a **fresh** set of `letrec`
closures with independent captured state. Previously, local `letrec`/internal-define
functions were stored in module-level globals and multiple closure instances
aliased each other's state; this is fixed.

```scheme
(define (counter-factory)
  (letrec ((count 0)
           (inc (lambda () (set! count (+ count 1)) count)))
    inc))
(define a (counter-factory))
(define b (counter-factory))
(display (a)) (display (a)) (display (b)) (newline)
```
```
121
```
`a` counts `1`, `2`; `b` is independent and counts `1`. (Before the fix the two
would have shared one backing counter.)

## Named `let` (loop)

```
(let name ((var init) …) body …)
```
Defines a local recursive procedure `name` bound to the loop, immediately called
with the `init` values. Tail-recursive calls to `name` are fully optimized (see
[tail-calls.md](tail-calls.md)).

```scheme
(define (count-up n)
  (let loop ((i 0) (acc '()))
    (if (= i n) (reverse acc) (loop (+ i 1) (cons i acc)))))
(display (count-up 5)) (newline)
```
```
(0 1 2 3 4)
```

Named-let captures are passed as per-call arguments, which makes the loop
thread-safe and correct across parallel execution.
