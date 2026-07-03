# Control Flow

## `if`

```
(if test consequent)             ; one-armed
(if test consequent alternate)   ; two-armed
```
Every value except `#f` is truthy. A one-armed `if` whose test is false returns an
unspecified value.

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

### Known issue — `=>` clause not supported

The R7RS `(test => proc)` clause form is **not** supported; `=>` is parsed as a
variable reference.

```scheme
(display (cond (42 => (lambda (v) (* v 2))) (else 'z))) (newline)
```
```
error: Undefined variable: =>
```
**Workaround:** bind the test value explicitly:
`(let ((v 42)) (cond (v ((lambda (x) (* x 2)) v)) (else 'z)))`.

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
