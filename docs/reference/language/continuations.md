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

### Re-invoking a saved continuation — current behavior is engine-specific and single-shot in the general case

`call/cc` is implemented as a single `setjmp`/`longjmp` pair per capture on the
native backend (the `-r` JIT and AOT compiler share this codegen), and as an
operand-stack/call-frame snapshot-and-restore on the bytecode VM. **Neither
engine currently delivers general, standards-conformant multi-shot re-entrant
continuations** — invoking a saved continuation more than once is only
reliable when the re-invocation stays within the same dynamic extent that is
still live on the stack. An earlier version of this page claimed "a full
re-entrant continuation, not merely a one-shot escape" without qualification;
that claim was not measured against the classic generator/`amb` shape and did
not hold up when tested. `call/cc` being escape-only is not itself a defect —
most real uses of `call/cc` (early return, exception-style unwinding) are
escape-only, and native handles those correctly and with no runtime overhead
on the non-escaping path. The defect was the documentation claiming more than
either engine delivers, compounded by four other Eshkol docs
(`docs/breakdown/CONTINUATIONS.md`, `docs/COMPLETE_LANGUAGE_SPECIFICATION.md`,
`docs/internal/ESHKOL_V1_LANGUAGE_REFERENCE.md`,
`docs/breakdown/SCHEME_COMPATIBILITY.md`) stating single-shot without
qualification either, so all five disagreed. Measured 2026-08-25:

**Native (JIT `-r` and AOT):** genuinely single-shot / escape-only. A saved
continuation can safely be re-invoked only while the C stack frame that
captured it is still live (has not returned). Example — this program's
`call/cc` is invoked, returns normally, and `k` is re-invoked repeatedly, but
because every re-invocation happens from the *same* top-level flow that
called it (no intervening function call has returned in between), the
`jmp_buf` target is still valid stack and this specific shape happens to work:

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
Do not read this as general multi-shot support. The moment a continuation is
captured *inside a function call that has since returned* — the shape every
real generator, coroutine, or `amb`-style backtracking search needs —
re-invoking it is undefined behavior on native and reproducibly crashes:
`tests/continuations/reentry_after_function_return.esk` SIGILLs on both JIT
and AOT the instant the popped frame is re-entered, and
`tests/continuations/generator_coroutine.esk` (a call/cc-based generator)
SIGSEGVs on its second `yield`. See `.icc/silent-wrong-ledger.yaml` SW-51.

**Bytecode VM:** architecturally different — a continuation capture
snapshots the VM's own operand stack and call-frame array rather than the C
stack, which in principle *can* survive re-entry into an already-returned
Scheme-level frame (and does, correctly, for
`reentry_after_function_return.esk` above — see `tests/continuations/`). But
the VM does not correctly implement multi-shot re-entry either: running the
exact "re-invoke `k` in a loop" example above on the VM prints `1` and then
loops printing `2` forever, never reaching `3` or `done` — the VM's restore
path does not track a coherent notion of "resume exactly here," so it
re-executes later top-level forms wholesale, and its own runaway-instruction
safety net does not save you (the guard's counter is reset by every
continuation invoke, so it never trips). `generator_coroutine.esk` on the VM
does not crash but prints a garbled transcript. See
`.icc/silent-wrong-ledger.yaml` SW-52 for both, waived pending the build item
below, and `tests/continuations/README.md` for exact repro commands (these
fixtures are not wired into automated CI because the point is that they hang
or crash).

### Known limitation — deep CPS chains (ESH-0080) — FIXED

Very deep continuation-passing chains (e.g. the SICP ch4 `amb`
nondeterministic search) used to crash with SIGILL beyond a moderate depth
(observed around n ≳ 16 for that workload) on native JIT/AOT. **This was
fixed on 2026-06-29** (commit `47338adb`, "run native O0 cleanup for deep
CPS"): native `-O0` codegen was skipping LLVM's function cleanup passes
entirely, leaving oversized stack frames for deep closure chains. The fix
runs a small cleanup pipeline (sroa, early-cse, instcombine, simplifycfg) at
`-O0`. The regression test, `tests/sicp/ch4_amb_deep_cps_test.esk`, still
passes on both native JIT and AOT as of this page's last measurement.
**Note:** despite living under this "continuations" page, ESH-0080's own
regression test does not use `call/cc` at all — it is a manual-CPS/closure-
chain program, so this was never actually evidence about `call/cc`
specifically; it is unrelated to the single-shot/multi-shot question above.

### Build item: full multi-shot re-entrant continuations

Full, standards-conformant multi-shot re-entrant continuations remain a
goal, not abandoned by the corrections above. What it would take:

- **Native.** The current `jmp_buf`-per-capture design fundamentally cannot
  survive re-entry once its capturing C stack frame has been reused —
  fixing this means replacing the underlying stack representation, most
  plausibly a segmented or heap-allocated stack (so a captured continuation
  owns its own durable copy of the frames below it, the same idea the VM
  already gestures at) or lowering `call/cc`-using code through a CPS
  transform in the compiler front end so "the rest of the computation"
  becomes an ordinary heap-allocated closure instead of a raw stack
  address. Either is a substantial codegen/runtime redesign, not a local
  fix — this is why the current design was chosen for `v1.x` (zero
  overhead on the non-escaping path, no custom stack management).
- **Bytecode VM.** The snapshot-and-restore mechanism is closer in spirit
  to what full re-entry needs, but SW-52 shows its restore path does not
  track a coherent "resume point" distinct from "replay the rest of the
  program" — this needs its own design pass, and ideally a re-entry model
  shared with whatever native ends up doing, so the two engines don't
  diverge on what "resuming a continuation" means.
- **No target version yet.** This is large enough (spanning the codegen,
  the runtime stack/heap representation, and the VM's interpreter loop) that
  it is not yet scoped into a specific release; it is tracked here and in
  `.icc/silent-wrong-ledger.yaml` (SW-51, SW-52) rather than given an
  unfounded date. Acceptance criterion when it is scoped: all three fixtures
  in `tests/continuations/` produce the documented-correct output on both
  native and the VM, and the doc-typed-claims/contradiction checks in
  `docs/design/AUDIT_2026_08_25_RESOLUTION.md`'s lineage stay clean against
  this page.

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
