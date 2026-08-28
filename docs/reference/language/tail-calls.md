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

## Self tail recursion in a `guard` body

A `guard` is how a resident program draws an error boundary around one unit of
work, and the unit of work is usually a loop:

```scheme
(define (tick i acc)
  (guard (e (#t acc))                    ; the boundary: any raise ends the run
    (if (>= i 1000000)
        acc
        (tick (+ i 1) (+ acc 1)))))      ; self tail call, inside the guard
(display (tick 0 0)) (newline)
```
```
1000000
```

**A self tail call in the body of a `guard` runs in constant stack, and the
handler chain the program observes is exactly the one it would observe if every
activation kept a native frame.** Both halves of that sentence are gated:
`tests/tco/guard_tail_position_test.esk` runs a million guarded iterations under
a reduced `ulimit -s`, and the nine fixtures in `tests/tco/guard_tail_context/`
are checked against chibi-scheme 0.12's answers on the native JIT, the native
AOT path and the bytecode VM by `scripts/run_guard_tail_context.sh`.

This is a deliberate extension of R7RS, not a reading of it. R7RS 7.1.3's
tail-context grammar does not list `guard`, and it cannot: the handler's whole
purpose is to be in effect while the body runs, so at depth N a conforming
implementation has N *live* guards, each holding that activation's variables.
Every one of them is observable — a handler that re-raises, or a clause body
that raises, must find the next one out and get *its* activation's values back:

```scheme
(define (climb n)
  (guard (e ((> n 0) (raise e))          ; not the outermost: hand it further out
            (#t (list 'answered-at n)))  ; n = 0: the outermost activation answers
    (if (>= n 100000) (raise 'top) (climb (+ n 1)))))
(display (climb 0)) (newline)
```
```
(answered-at 0)
```

That condition meets one hundred thousand distinct live handlers, in order, each
answering with its own `n`, in one native frame.

### How it is lowered

The self-call is a branch back to the loop header (ESH-0222), which is what
makes the stack flat. What the back edge must not do is destroy the guards the
collapsed activations were standing in. So it does not pop them: it **leaves the
handler frames installed** — they *are* the enclosing activations' handlers —
and attaches a snapshot of the departing activation's loop parameters to each.
The guard's landing pad asks the frame that fired whether it carries a snapshot;
if it does, the raise walked out of an inner activation, and those values are put
back before the clauses read them. A re-raise then simply raises, and the next
frame down is the next activation's guard, with its own `wind_mark` and
`region_mark`, so `dynamic-wind` after-thunks and region unwinding happen in
exactly the reference's order.

Three lowerings are chosen per loop, and all three are exact:

| lowering | when | stack | handler frames |
|---|---|---|---|
| collapse | the guard has a catch-all clause and no clause test or body can raise, so the enclosing activations are unobservable | flat | one, reused |
| replay | anything else the loop's parameters can restore | flat | one per live guard, on the heap |
| ordinary call | the clauses read a binding the loop re-establishes each iteration, which no snapshot of the *parameters* can restore | one frame per activation | one per live guard, on the stack |

The collapse lowering is the resident tick-loop shape
(`docs/LONG_RUNNING_LOOPS.md`), and it is why a daemon's peak RSS is flat across
a week of ticks rather than growing by a handler frame each time round. The
replay lowering keeps the stack flat and pays one small heap frame per *live*
guard — which is the space R7RS's own semantics require to exist, moved off the
native stack. The third row is not a fallback to something weaker: it is what a
conforming implementation costs for a non-tail `guard` body, and it is chosen
only where nothing cheaper is exact.

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
- **MUTUAL tail calls in the body of `guard`.** R7RS 7.3 derives `guard` from
  `with-exception-handler` wrapping the body, so the handler is installed for the
  body's whole dynamic extent: a call in a guard body is **not in a tail
  context**, and the guard must still be able to catch what the callee raises.
  Transferring one away is not merely hard, it is wrong — measured against
  chibi-scheme 0.12, transferring these sites made
  `(guard (e (#t 'caught)) (b n))` report an unhandled exception where the
  reference answers `caught`, and made a nested pair of guards answer with the
  outer handler instead of the inner one
  (`tests/tco/guard_tail_context/01`, `03`). A transfer RETURNS from the
  transferring procedure to run the callee, which pops the guard's `setjmp`
  landing pad out of the frame; giving the mutual side a genuine tail context
  needs that landing pad to be a heap-owned continuation the driver can invoke
  instead (ADR-0006 §4). Until then a mutual tail call in a guard body stays an
  ordinary call.
  **This does not apply to SELF tail recursion**, which keeps its frame and is
  therefore fully optimized — see "Self tail recursion in a `guard` body" above.

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
