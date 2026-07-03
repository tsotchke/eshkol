# Tail-Call Optimization (TCO)

Eshkol performs proper tail-call optimization for **self-recursion** and for
tail calls within a single procedure (including named `let` loops). A tail call
reuses the current stack frame, so deep self-recursion runs in constant stack
space.

## What is a tail position

An expression is in tail position when its value is the value of the whole
procedure body — nothing further is computed after it returns. In particular:

- the last expression of a `lambda`/`define` body, `let`/`let*`/`letrec` body,
  `begin`, `when`/`unless` body;
- the `consequent` and `alternate` of an `if` in tail position;
- the body of the selected `cond`/`case`/`match` clause in tail position;
- the recursive call of a named `let` loop.

A call is **not** in tail position if its result is consumed by another operation,
e.g. `(+ 1 (f n))` — the `(+ 1 …)` runs after `f` returns.

## Self tail recursion is optimized

```scheme
(define (loop i n) (if (>= i n) i (loop (+ i 1) n)))
(display (loop 0 5000000)) (newline)
```
```
5000000
```
Five million iterations complete without stack growth. Named-`let` loops behave
the same way (their loop call is a tail call).

## Known limitation — mutual tail recursion is NOT optimized

Tail calls **between** two procedures (ping-pong / mutual recursion) are *not*
turned into jumps; each call consumes stack. Such programs work up to a moderate
depth and then crash (SIGILL, stack exhaustion).

```scheme
(define (ev? n) (if (= n 0) #t (od? (- n 1))))
(define (od? n) (if (= n 0) #f (ev? (- n 1))))
(display (ev? 400000)) (newline)   ; works
```
```
#t
```
The same program at `(ev? 500000)` crashes:
```
[process exits with signal SIGILL, return code 132]
```
Measured threshold: fine at 400k, crashes at 500k+. **Workaround:** fold mutually
recursive state machines into a single self-recursive procedure that dispatches on
a state argument, or use an explicit loop with an accumulator.

## Related known issue — non-tail stdlib list procedures (ESH-0108)

Some stdlib list procedures are themselves non-tail-recursive and will exhaust the
stack on very large inputs *even though your call to them is in tail position*.
`length` and `filter` are implemented as `(+ 1 (length (cdr lst)))`-style
non-tail recursion and crash (SIGILL, no diagnostic) around 500k–1M elements;
`map`, `reverse`, and `fold-left` are tail-safe to at least 1M.
