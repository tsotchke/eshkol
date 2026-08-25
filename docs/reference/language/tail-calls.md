# Tail-Call Optimization (TCO)

Eshkol performs proper tail-call optimization for **self-recursion**, for tail
calls within a single procedure (including named `let` loops), and for **mutual**
tail recursion between named procedures. A tail call reuses the current stack
frame, so deep tail recursion runs in constant stack space.

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

## Mutual tail recursion is optimized

Tail calls **between** procedures (ping-pong / mutual recursion) are emitted as
LLVM `musttail` calls (ESH-0102), so a mutually tail-recursive state machine
reuses one frame and runs in O(1) stack:

```scheme
(define (ev? n) (if (= n 0) #t (od? (- n 1))))
(define (od? n) (if (= n 0) #f (ev? (- n 1))))
(display (ev? 5000000)) (newline)
```
```
#t
```

### The spelling does not matter (ESH-0102b)

R7RS §3.5 makes the selected branch of `cond`, `case`, `when`, `unless` and the
last operand of `and`/`or` tail positions exactly as much as the branches of
`if`, and Eshkol optimizes all six the same way. Until ESH-0102b only the `if`
spelling was optimized: the walk that offered call sites to the mutual-tail
lowering descended solely through `if`/`begin`/`let`, so a mutual tail call
written any other way silently became an ordinary call, grew one native frame
per hop, and exhausted the stack a few million hops in — while the byte-identical
program written with `if` ran flat. All six now behave identically:

```scheme
(define (ping n acc) (cond ((= n 0) acc) (else (pong (- n 1) (+ acc n)))))
(define (pong n acc) (cond ((= n 0) acc) (else (ping (- n 1) (+ acc n)))))
(display (ping 100000000 0)) (newline)
```
```
5000000050000000
```
One hundred million hops, constant stack, under both the JIT and AOT. The
depth-parametric probes `mutual_tail_cond` and `mutual_tail_forms`
(`scripts/run_recursion_depth.sh`) pin this at 100,000,000; `mutual_tail_forms`
routes its tail call through a different one of the four forms at each hop, so a
regression in any single form fails the gate.

### What is still not optimized

These shapes fall back to an ordinary, stack-consuming call. They are correct,
and they announce themselves by crashing loudly with the fatal-signal
diagnostic ("most likely a stack overflow") rather than returning a wrong
answer — but they are bounded, and deep recursion in these shapes will exhaust
the stack:

- **Mutually recursive procedures with different signatures.** `musttail`
  requires the caller and callee to have identical parameter counts and types,
  so `(define (ping n acc) … (pong a b c))` paired with `(define (pong n acc k)
  … (ping a b))` is a bounded call in both directions. Give mutually
  tail-recursive procedures the same arity.
- **Indirect tail calls** — calling a procedure held in a variable or passed as
  an argument, rather than named directly at the call site.
- **Higher-order tail calls that forward a stack-allocated closure argument.**
  `musttail` is illegal when an argument points into the caller's frame, since
  that frame is gone before the callee runs, so these deliberately fall back.
- **Tail calls in the body or a handler clause of `guard`.** Leaving a `guard`
  owes a handler-stack pop that discarding the frame outright would skip, so
  mutual tail calls are not optimized through `guard`. (Self tail recursion
  through `guard` *is* optimized — it branches to a loop header with the handler
  bookkeeping intact, ESH-0222.)
- **Non-AArch64 targets.** Eshkol procedures return an aggregate (the by-value
  tagged value), and on LLVM 21 only the AArch64 backends lower `musttail` with
  an aggregate return; x86-64, arm32 and riscv64 reject it. There the call keeps
  the `tail` hint, which is bounded. Lifting this needs the `i128` tagged-return
  ABI tracked as ESH-0171.

**Workaround for any of the above:** fold the state machine into a single
self-recursive procedure that dispatches on a state argument, or give the
procedures a common signature.

## Related known issue — non-tail stdlib list procedures (ESH-0108)

Some stdlib list procedures are themselves non-tail-recursive and will exhaust the
stack on very large inputs *even though your call to them is in tail position*.
`length` and `filter` are implemented as `(+ 1 (length (cdr lst)))`-style
non-tail recursion and crash (SIGILL, no diagnostic) around 500k–1M elements;
`map`, `reverse`, and `fold-left` are tail-safe to at least 1M.
