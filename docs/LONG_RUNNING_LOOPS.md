# Long-running loops with a per-iteration error boundary

This document is the canonical reference for writing a tail-recursive loop
that runs indefinitely (a resident process, a daemon, an agent's tick loop)
and wants to catch and recover from per-iteration errors without crashing
the whole process or leaking stack/heap over time. It grew out of ESH-0222
("tail calls through `guard` grow the stack"), filed after a compiled
long-running resident stack-overflowed every 45-75 minutes.

## TL;DR — the canonical pattern

```scheme
(define (resident-loop tick state)
  (if (>= tick max-ticks)
      state
      (let ((next-state
              (guard (e (#t (log-stumble! e) state))   ; guard wraps ONLY the
                (run-one-tick! state))))                ; fallible section
        (resident-loop (+ tick 1) next-state))))         ; loop call OUTSIDE guard
```

- `guard` wraps **only** the fallible computation and evaluates to a value.
- The recursive loop call is **outside** `guard`'s dynamic extent, in the
  tail position of the enclosing `let`.
- State is threaded as a single value (or a small record — see below), not
  reconstructed by consing a fresh list every iteration.

This shape was already safe before ESH-0222 and remains the recommended
default. The rest of this document explains why, what else is now *also*
safe after the ESH-0222 fix, and what to avoid.

## Q1: Does `guard` preserve tail position for its body?

R7RS does not require it to — a `guard` form's protected body is not
inherently a tail context, because conceptually the exception handler must
stay installed while the body runs. Eshkol special-cases self-recursive
tail calls textually inside a `guard`'s protected body or handler clauses
so they *can* compile to a genuine loop back-edge instead of a real call —
but two related bugs meant this either didn't happen (TCO silently
disabled) or happened unsoundly (TCO enabled but leaking) before ESH-0222:

1. **Top-level `(define (f ...) ...)` self-tail-recursion** used a family
   of helpers (`isSelfTailRecursive` → `countAllRecursiveCalls` /
   `findTailCalls` / `isInTailPosition`, in `lib/backend/llvm_codegen.cpp`)
   that had **no case for `guard`** and silently skipped over it. A
   function whose *only* recursive calls live inside a `guard` (exactly
   the shape below) had those calls become invisible to the recursion
   counter — `total_recursive_calls` came out as 0, so the compiler
   concluded "no recursion at all" and never even attempted TCO. The loop
   then ran as real, unbounded recursion — one native stack frame per
   iteration, forever:

   ```scheme
   (define (loop tick state)
     (if (>= tick max-ticks)
         state
         (guard (e (#t (loop (+ tick 1) (on-error state e))))  ; handler tail-recurses
           (run-one-tick! state)
           (loop (+ tick 1) (updated state)))))                 ; body tail-recurses
   ```

2. **Named-let TCO** used a separate analysis
   (`allSelfCallsInTailPosition` in `lib/backend/tail_call_codegen.cpp`)
   that had the *opposite* bug: it silently assumed any self-call inside a
   `guard` was tail-safe without checking. When that assumption happened
   to be true, TCO got enabled — but the code generator itself wasn't
   sound for it: `guard`'s `setjmp` buffer is a **dynamically-sized**
   `alloca` (its size comes from a runtime call, not a compile-time
   constant), so LLVM cannot hoist or reuse it across iterations of a
   hand-rolled branch-based loop. Every iteration through `guard`
   permanently consumed a little more native stack — never reclaimed,
   because the loop never returns — until the process overflowed. On top
   of that, the compiler's tail-call shortcut (an unconditional branch
   back to the loop header) bypassed the exception handler's own cleanup
   code, so the runtime's handler chain also grew by one unpopped entry
   per iteration.

**Both are fixed as of ESH-0222.** The tail-position analyses are now
honest about `guard` (they actually look inside it, in both directions),
and the code generator reclaims the per-iteration stack space
(`llvm.stacksave`/`stackrestore` bracket the loop's back-edge) and rebalances
the exception-handler push/pop across the shortcut branch. Concretely, this
means the pattern from bug (1) above — recursion in **both** the guard
body's tail position and a handler clause's tail position — now compiles
to a flat loop:

```scheme
(define (loop tick state)
  (if (>= tick max-ticks)
      state
      (guard (e (#t (loop (+ tick 1) (on-error state e))))
        (run-one-tick! state)
        (loop (+ tick 1) (updated state)))))
```

Verified flat at 1,000,000 ticks (top-level `define` form): ~45MB RSS,
~22MB peak footprint, well under half a second. See
`tests/tco/guard_loop_tail_test.esk`.

**Caveat — named-let specifically:** a separate, pre-existing bug
(ESH-0223, discovered while verifying this fix) means named-let tail loops
overflow the stack somewhere around 300-500k iterations **regardless of
whether `guard` is involved at all** — even `(let loop ((n 0)) (if (>= n N)
n (loop (+ n 1))))` with zero `guard` usage hits it. If you need a loop
that runs well past that count, prefer a **top-level `define`** (as in the
canonical pattern above) until ESH-0223 is resolved; `define`-based
tail-recursive loops are independently verified flat past 10^8 iterations
(`tests/stress/rec_tco_1e8.esk`).

Either way, **the flat-outside-guard pattern at the top of this document
remains the recommended default** regardless of which bug is or isn't
fixed on a given build — it never depended on either.

## Q2: What per-iteration-error-handling loop structure is provably tail-optimized?

In order of preference:

1. **`guard` wraps only the fallible section; the loop call is outside it**
   (the canonical pattern above). Never relies on the guard-tail-call
   mechanism at all — `guard` here is just an ordinary expression that
   produces a value, no different from any other subexpression the loop's
   tail call depends on. This is unaffected by ESH-0222 or ESH-0223 and is
   the pattern to reach for first.

2. **Self-calls in guard-body-tail and/or handler-clause-tail position**
   (the pattern ESH-0222 fixes) — use this when the error boundary
   genuinely needs to wrap the recursive step itself (e.g. different
   recovery logic depending on *where* in the tick the error occurred).
   Prefer a top-level `define` over a named-let per the ESH-0223 caveat
   above.

**Threading state without consing:** prefer a single accumulator value, or
a small `define-record-type` instance passed by reference, over
reconstructing a fresh list every iteration. Consing a new list per
iteration is extra per-tick allocation for no benefit — the arena reclaims
it at the same rate either way, but a record/scalar avoids the allocation
and the cons-traversal entirely. This project's automatic per-iteration
arena-scope reclamation work (ESH-0214/ESH-0214b, `fix/loop-arena-reclamation`,
not yet merged as of this writing) is complementary: once merged, it
further bounds RSS for named-let loops whose static shape qualifies for
automatic per-iteration scoping, independent of whether guard is used.
`guard`/`call/cc` bodies are expected to be on that feature's rejection
list for automatic scoping (their handler/continuation state must survive
past a naive iteration boundary), so the flat-outside-guard pattern remains
the most robust choice even after that feature lands.

## Q3: Is `(apply loop lst)` a proper tail call?

**No.** `call_apply_codegen.cpp` never consults the tail-call-optimization
context (`isTCOActive`/`TailCallContext`) — every `apply` call, self-recursive
or not, compiles to a genuine (non-tail) LLVM `call`. Using `apply` for a
loop's back-edge will grow the native call stack by one frame per
iteration no matter what the target function's body looks like. This is a
distinct, currently-unfixed gap (tracked separately; not part of
ESH-0222), not a special case of the guard bug — it reproduces identically
with or without `guard` in the loop body.

**Do not use `(apply loop next-args)` as a long-running loop's back-edge.**
Call the loop function directly with its arguments spelled out
(`(loop (+ tick 1) new-state)`), which does participate in TCO. If the
argument list's arity is only known at runtime, thread a single record or
vector instead of relying on `apply` to spread a list.

(A prior attempt to work around ESH-0222 by moving the recursive call
outside `guard` via `(apply loop (list ...))` ballooned RSS from 0.6GB to
3GB in about a minute and crashed — consistent with this: every iteration
made a real, non-tail `apply` call, and the per-iteration `(list ...)`
allocation was never reclaimed because the surrounding recursion was never
flat to begin with.)

## Q4: Is there a stack-size stopgap?

Yes — `ESHKOL_STACK_SIZE` (bytes, minimum 1MB) raises the process's
`RLIMIT_STACK` soft limit at startup (`eshkol_init_stack_size()`,
`lib/core/runtime_stack_hosted.cpp`). The default is 512MB. This is a
**stopgap, not a fix**: it linearly extends how long an unbounded-growth
loop survives before overflowing, it does not make the loop flat. Note
also that the OS hard limit (`ulimit -Hs`) caps how far this can actually
go — e.g. 64MB is a common macOS default hard limit, well under Eshkol's
512MB *request*, so `setrlimit` silently clamps to whatever the hard limit
allows. Don't rely on this for a process that's supposed to run
indefinitely; use the canonical pattern above instead.

## See also

- `tests/tco/guard_loop_tail_test.esk` — the regression test for this
  document's shapes.
- `.swarm/tasks/ESH-0222.json` — the guard-tail-position fix.
- `.swarm/tasks/ESH-0223.json` — the separate, still-open named-let stack
  bug referenced in the caveat above.
- `docs/KNOWN_ISSUES.md` — "Arena memory (OALR) instead of garbage
  collection": Eshkol has no tracing/stop-the-world collector, so a
  long-running loop's memory behavior is governed entirely by arena
  growth (page pressure), not GC pauses.
