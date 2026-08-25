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

### Re-invoking a saved continuation — multi-shot and re-entrant

A continuation captured by `call/cc` may be invoked any number of times, from
any dynamic extent, including after the procedure that captured it has already
returned. This is what generators, coroutines and `amb`-style backtracking
search need, and it works identically on the native `-r` JIT, the native AOT
compiler, and the bytecode VM.

```scheme
(define k #f)
(define (f)
  (+ 1 (call/cc (lambda (c) (set! k c) 1))))
(display (f)) (newline)          ; 2 — f returns normally
(display (k 10)) (newline)       ; re-enters f's continuation after f returned
```

The classic shapes are exercised as fixtures in `tests/continuations/`:
`generator_multishot.esk` (a producer/consumer generator resumed repeatedly),
`amb_backtracking.esk` (McCarthy's `amb`, re-entering each choice point once
per alternative), `reentry_after_function_return.esk`, and
`region_capture_resume.esk`.

#### How each engine delivers it

**Native (JIT and AOT).** `call/cc` still records a `setjmp` point, but a
`jmp_buf` only names a stack address: once the capturing frame returns and its
memory is reused, jumping back to it would resume on top of whatever now
occupies those bytes. So a capture that may outlive its frame also copies the
live C stack — from just below the capture point up to the thread's stack base
— into the arena. Invoking the continuation copies those bytes back to *the
same addresses* and then `longjmp`s. Restoring in place is what makes this safe
without relocating anything: frame pointers, spilled registers, addresses of
locals held by closures, and the `jmp_buf` itself all still point where they
always did. The image is written once and never mutated, so re-invoking a
continuation any number of times restores the same pristine copy.

**Bytecode VM.** A capture snapshots the VM's own operand stack and call-frame
array rather than the C stack. Crucially it snapshots only the *control* half:
the VM binds every top-level definition to an operand-stack slot, and those
slots are the *store*. R7RS `call/cc` captures the control state, not the
store, so top-level bindings are excluded from both the snapshot and the
restore — `set!` and `define` effects survive re-entry rather than being rolled
back with it.

#### Escape continuations pay nothing

Most real uses of `call/cc` are escape-only — early return, exception-style
unwinding — where the continuation is invoked while the capturing frame is
still live. Those never need a stack image, so native does not take one: the
compiler classifies a capture as non-escaping when `proc` is a literal
one-parameter lambda whose parameter is only ever the operator of a direct
call, with no reference from inside a nested lambda. Such a capture compiles to
exactly the `setjmp`/`longjmp` it always did. Anything the analysis does not
model is treated as escaping, so the error direction is a copy that was not
needed rather than a missing image. Measured on an escape-only `call/cc`
captured at depth 60, AOT, best of 9 interleaved runs: within noise of the
pre-multi-shot implementation.

#### Interaction with regions

Eshkol has no garbage collector and reclaims by region at scope exit, so a
continuation captured inside `with-region` and resumed after that region exits
is an ownership question. **The rule on both engines: capturing a continuation
inside a region pins it.** The region is promoted whole rather than reclaimed,
so the failure direction is a leak, never a dangling reference. The VM says so
on stderr ("a `with-region` body could not be reclaimed and was promoted
whole"); set `ESHKOL_VM_REGION_QUIET=1` to silence the note.
`tests/continuations/region_capture_resume.esk` resumes such a continuation
three times after the region has exited, with heavy allocation churn in
between, and checks that region-allocated data still reads correctly.

#### Limits

Two shapes do not behave as R7RS specifies, both tracked in
`.icc/silent-wrong-ledger.yaml`:

- **A top-level binding established after a capture, on the bytecode VM.**
  Because the VM keeps top-level bindings in operand-stack slots, a binding
  whose slot sits above a continuation's saved stack top cannot be preserved
  and re-entered at the same time. The VM **refuses the resume with a
  diagnostic** naming the cause and the workaround (move the definition above
  the `call/cc`, or use the native backend) rather than resuming onto a
  corrupted store. Native has no such restriction.
- **A local variable mutated after capture is rolled back on re-entry**, on
  both engines, when that variable is neither a top-level binding nor captured
  by a closure. Such a variable lives directly in the restored frame, so the
  image restores its capture-time value; R7RS says the location persists and
  only the control state is captured. Making this sound needs assignment
  conversion — boxing `set!`-assigned locals — which is not yet implemented.
  See ledger SW-53.

### Resolved history — deep CPS chains (ESH-0080)

Very deep continuation-passing chains (for example the SICP ch4 `amb`
nondeterministic search) once crashed with SIGILL beyond a moderate depth on
native JIT and AOT. Fixed on 2026-06-29 in commit `47338adb`: native `-O0`
codegen was skipping LLVM's function cleanup passes entirely, leaving
oversized stack frames for deep closure chains, and the fix runs a small
cleanup pipeline (sroa, early-cse, instcombine, simplifycfg) at `-O0`. The
regression test is `tests/sicp/ch4_amb_deep_cps_test.esk`.

Recorded here only because the issue was historically filed against this page.
It never concerned `call/cc`: its regression test is a manual-CPS closure-chain
program that does not use `call/cc` at all.

### Delivered: full multi-shot re-entrant continuations

The build item recorded here — replacing the `jmp_buf`-per-capture design so a
continuation survives its frame's return — is implemented. Of the two options
this page previously named, the **heap-allocated stack copy** was taken and the
front-end CPS transform was rejected.

The reason is scoping. Eshkol's native builtins call back into Eshkol closures
(`map`, `for-each`, `sort`, the AD tape), the standard library is separately
compiled, and the FFI lets a host call in. Deciding which code a CPS transform
must cover across those boundaries needs whole-program information Eshkol does
not have, so in practice everything reachable from a `call/cc` user would need
a CPS variant — changing the calling convention globally, taxing code that
never uses `call/cc`, and invalidating rather than composing with the
tail-transfer ABI work. Stack copying keeps the existing zero-overhead escape
path untouched and adds cost only where a continuation may actually outlive its
frame.

Stack geometry is platform-specific and the continuation runtime is
freestanding core, so the hosted runtime installs a probe
(`eshkol_set_stack_base_hook`) at startup. A freestanding target that installs
no probe keeps escape-only continuations rather than copying a stack region it
cannot bound.

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
