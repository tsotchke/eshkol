# Control Flow

## `if`

```
(if test consequent)             ; one-armed
(if test consequent alternate)   ; two-armed
```
Every value except `#f` is truthy. A one-armed `if` whose test is false returns
[the unspecified value](#the-unspecified-value).

```scheme
(if (> 3 2) (display "one-armed-if\n"))
(display (if (= 1 2) 'a 'b)) (newline)
```
```
one-armed-if
b
```

## `cond`

```
(cond (test body …) …
      (else body …))
```
Evaluates each `test` in order; runs the body of the first true clause. `else` is
the catch-all.

```scheme
(display (cond ((= 1 2) 'a) ((= 1 1) 'b) (else 'c))) (newline)
```
```
b
```

### `=>` clauses

The R7RS `(test => proc)` clause form is supported: when `test` is true its
value is passed to `proc`, and the clause's value is the result of that call.

```scheme
(display (cond (42 => (lambda (v) (* v 2))) (else 'z))) (newline)
(display (cond ((assv 2 '((1 a) (2 b))) => cadr) (else 'none))) (newline)
(display (cond (#f => (lambda (v) v)) (else 'fell-through))) (newline)
```
```
84
b
fell-through
```
`case` accepts `=>` on a clause too — the key is passed to `proc`:
```scheme
(display (case 3 ((1 2) 'low) ((3 4) => (lambda (v) (* v 10))) (else 'other))) (newline)
```
```
30
```

## `case`

```
(case key ((datum …) body …) …
          (else body …))
```
Compares `key` (with `eqv?`) against each list of data.

```scheme
(display (case 3 ((1 2) 'low) ((3 4) 'mid) (else 'hi))) (newline)
```
```
mid
```

The `=>` clause form of `case` is likewise **not supported** (same limitation as
`cond`).

## `when` / `unless`

```
(when test body …)     ; run body if test is true
(unless test body …)   ; run body if test is false
```
```scheme
(when (> 3 2) (display "when-yes") (newline))
(unless (> 2 3) (display "unless-yes") (newline))
```
```
when-yes
unless-yes
```

## `do`

```
(do ((var init step) …)
    (test result …)
  command …)
```
Iterates: each `var` starts at `init` and is updated to `step` each pass; when
`test` becomes true the `result` expressions run and the last is returned.

```scheme
(do ((i 0 (+ i 1)) (s 0 (+ s i))) ((= i 5) (display s) (newline)))
```
```
10
```

## `and` / `or`

```
(and expr …)   ; left-to-right; returns first #f, else the last value; (and) => #t
(or  expr …)   ; left-to-right; returns first truthy value, else #f; (or) => #f
```
Both short-circuit.

```scheme
(display (and 1 2 3)) (newline)
(display (or #f #f 5)) (newline)
(display (and)) (display " ") (display (or)) (newline)
```
```
3
5
#t #f
```

## Static types of the control forms

The optional type checker examines the tests, keys and every branch body of
these forms, exactly as it examines a call at top level, and gives each form
the join of its branch types (plus `#f` for a `cond`, `case`, `when` or
`unless` that may run no branch). An `if` and the equivalent `cond` have the
same type. The rules, with runnable examples, are in
[the gradual typing guide](../../guide/GRADUAL_TYPING.md#3-what-is-checked-and-where).

## The unspecified value

Every form R7RS leaves unspecified evaluates to one value, the unspecified
value: a one-armed `if` whose test is false, `when` and `unless` when they do
not run their body, `set!`, `display`, `newline`, `write`, `for-each`,
`vector-set!`, `vector-fill!`, `vector-copy!`, `set-car!`, `set-cdr!`,
`hash-table-set!` and `(void)`. It is its own value, not the empty list:

```scheme
(display (null? (when #f 1))) (newline)     ;; => #f
(display (eq? (when #f 1) '())) (newline)   ;; => #f
(define x 0)
(display (list 'a (if #f 1) 'b)) (newline)  ;; => (a  b)
(display (eq? (set! x 1) (void))) (newline)  ;; => #t
```

`display` prints nothing for it, so a list holding one shows a gap, and the
REPL shows nothing after a form that evaluates to it. The bytecode VM's own
void value is the same value: a program prints the same text on every engine.
This is [ADR-0024](../../design/adr/0024-unspecified-value.md).
