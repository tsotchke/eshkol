# Tail-Call Optimization (TCO)

Eshkol performs proper tail-call optimization for **self-recursion**, for tail
calls within a single procedure (including named `let` loops), and for **mutual**
tail recursion between named procedures. A tail call reuses the current stack
frame, so deep tail recursion runs in constant stack space.

The bytecode VM follows the same R7RS tail-position contract. Its call frames
are heap-managed, and its compiler emits the same tail transfer for the final
operand of `and`/`or`, `when`/`unless` bodies, and `let`-family bodies even when
local bindings need cleanup.

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

### The arity does not matter either (ESH-0102c)

R7RS says nothing about two mutually tail-recursive procedures having the same
parameter list, and neither does Eshkol:

```scheme
(define (ping n acc)   (if (= n 0) acc (pong (- n 1) (+ acc n) 0)))
(define (pong n acc k) (if (= n 0) acc (ping (- n 1) (+ acc n))))
(display (ping 100000000 0)) (newline)
```
```
5000000050000000
```

`musttail` cannot express this shape at all — LLVM requires the caller and
callee to have byte-identical signatures — so it takes the **tail-transfer
dispatcher** instead. The depth-parametric probe `mutual_tail_arity` pins it at
100,000,000.

### How a tail call is lowered

Two mechanisms, both O(1) in native stack. Which one a call site gets is not a
choice about quality; it is a choice about what the target can express.

| Call shape | Lowering | Stack |
|---|---|---|
| Self tail call, named `let` loop | branch to a loop header | O(1) |
| Mutual tail call, same signature, AArch64 target | LLVM `musttail` | O(1) |
| Mutual tail call with **differing arities**; **any** mutual tail call on a non-AArch64 target | **tail-transfer dispatcher** | O(1) |
| Everything under "What is still not optimized" | ordinary call | one frame per hop |

The dispatcher (ADR-0006 §3) works by *not calling* the callee. The transferring
procedure copies its evaluated arguments into a per-thread transfer record,
records the callee's uniform entry, raises a `pending` flag, and **returns
normally**. A driver loop in the procedure's public entry sees the flag and runs
the transfer in the caller's stead. Because the arguments live in a record the
driver owns rather than in either frame, and because the callee's uniform entry
supplies its own parameters out of that record, neither the two signatures nor
the target's ability to lower an aggregate-return `musttail` matters any more.

One consequence is visible in a profile rather than in a program's answer: a
transfer costs a record write plus a driver bounce per hop where a `musttail`
costs a branch. Measured on the 100,000,000-hop `cond` probe, AOT, mean of three
runs on an idle arm64-darwin host: **2.99 s under `musttail`, 3.35 s under the
dispatcher — about 12%**, at 8.3 MB and 8.9 MB peak RSS respectively. It is the
same guarantee at a slightly different constant. (An earlier figure of "five
times" in this file was measured on a loaded machine and was wrong; the numbers
above are what a quiet one reports.)

### Non-AArch64 targets are no longer bounded (ESH-0171)

Eshkol procedures return an aggregate (the by-value tagged value), and on LLVM
21 only the AArch64 backends lower `musttail` with an aggregate return; x86-64,
arm32 and riscv64 reject it. Such a call used to keep a `tail` **hint**, which
the backend is free to ignore and which was therefore bounded. It now takes the
dispatcher, which returns through the ordinary aggregate return and never needs
that lowering, so those targets have the same O(1) guarantee AArch64 has.

`scripts/run_recursion_depth.sh` measures this rather than arguing it: every
`mutual_tail*` cell runs a third time in an `aot-xfer` lane with
`ESHKOL_TAIL_TRANSFER_ONLY=1`, which makes an AArch64 host lower mutual tail
calls exactly as a non-AArch64 target must.

The `i128` internal tagged-return ABI that ESH-0171 was opened for remains
worthwhile — it would let those targets take the cheaper `musttail` lowering —
but it is now a performance item, not a correctness one.

### What is still not optimized

These shapes fall back to an ordinary, stack-consuming call. They are correct,
and they announce themselves by crashing loudly with the fatal-signal
diagnostic ("most likely a stack overflow") rather than returning a wrong
answer — but they are bounded, and deep recursion in these shapes will exhaust
the stack:

- **Indirect tail calls** — calling a procedure held in a variable or passed as
  an argument, rather than named directly at the call site. A transfer names the
  callee's *uniform entry*; a procedure VALUE carries only its public entry, and
  dispatching a transfer through that re-enters the callee's own driver loop,
  which stacks one driver frame per hop instead of removing them. Lifting this
  needs the universal invoke entry ADR-0006 §3 describes — reachable from the
  callable itself, not from the call site's static knowledge of it.
- **Mutual tail calls between procedures that are closures rather than named
  top-level procedures** — including *internal* defines, which are normalized to
  `letrec*` lambdas. The walker that offers sites to the mutual-tail lowering
  runs over `define` bodies only, and a call between two `letrec`-bound lambdas
  is indirect in any case, so it is bounded for both of the reasons above.
- **Mutual tail calls made from inside a named `let` loop.** The loop body is a
  separate function, and only the enclosing `define` participates.
- **Higher-order tail calls that forward a pointer into the caller's frame.** A
  transfer copies argument *values* into a record; a pointer to storage the
  transferring frame owns would be carried across the return that destroys it
  (ADR-0006 §6 principle 8), so those sites keep the ordinary call. In this tree
  every such site is an `extern` FFI declaration taking a raw pointer, which must
  not be transferred in any case.
- **Tail calls in the body of `guard`.** R7RS 7.3 derives `guard` from
  `with-exception-handler` wrapping the body, so the handler is installed for the
  body's whole dynamic extent: a call in a guard body is **not in a tail
  context**, and the guard must still be able to catch what the callee raises.
  Optimizing it away is not merely hard, it is wrong — measured against
  chibi-scheme 0.12, transferring these sites made
  `(guard (e (#t 'caught)) (b n))` report an unhandled exception where the
  reference answers `caught`, and made a nested pair of guards answer with the
  outer handler instead of the inner one. Giving `guard` a genuine tail context
  requires its handler to be a heap-owned continuation the driver can invoke
  rather than a `setjmp` landing pad in the frame a transfer destroys
  (ADR-0006 §4). Until then this stays an ordinary call.
  (Self tail recursion through `guard` *is* transformed into a loop, ESH-0222 —
  see SW-58 in `.icc/silent-wrong-ledger.yaml` for the conformance consequence
  of that, which is a separate, pre-existing finding.)

**Workaround for any of the above:** fold the state machine into a single
self-recursive procedure that dispatches on a state argument, or give the
mutually recursive procedures top-level `define`s and name the callee directly
at the call site.

## Related known issue — non-tail stdlib list procedures (ESH-0108)

Some stdlib list procedures are themselves non-tail-recursive and will exhaust the
stack on very large inputs *even though your call to them is in tail position*.
`length` and `filter` are implemented as `(+ 1 (length (cdr lst)))`-style
non-tail recursion and crash (SIGILL, no diagnostic) around 500k–1M elements;
`map`, `reverse`, and `fold-left` are tail-safe to at least 1M.
