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
that may outlive the frame it was captured in pins every region that is open at
the moment of capture.** The failure direction is a retained region, never a
dangling reference. `tests/continuations/region_capture_resume.esk` resumes
such a continuation three times after the region has exited, with heavy
allocation churn in between, and checks that region-allocated data still reads
correctly; `region_capture_resume_nested.esk` does the same two regions deep.

Five properties of that rule are worth stating precisely, because each one is
easy to assume the other way round.

**An escape-only capture does not pin at all, on native.** `(call/cc (lambda (k)
… (k v) …))` where the body only ever uses `k` as the operator of a direct call
— the early-return idiom — produces a continuation that provably cannot outlive
the frame that captured it. Every enclosing `with-region`'s dynamic extent
contains that frame's, so such a continuation can never observe the region after
it closes, and native does not pin it: the region reclaims in full.
`tests/memory/region_callcc_flat_rss_test.sh` measures that shape in a resident
loop as **exactly 0.000 bytes/tick** from 10,000 to 100,000 iterations (a 10×
horizon). Anything the
classifier does not model reads as "may escape" and pins, which is the safe
direction and costs only a retained region. **The bytecode VM does not make this
distinction**: it pins on region depth alone, so the same program retains the
region there. That is a difference in memory policy only — the printed answer is
identical, which is what `scripts/run_continuation_tests.sh` compares on all
three engines.

There is one exception on native, and it is narrow: if any region open at
capture time was opened through the **region-handle API** rather than by a
lexical `with-region`, an escape-only capture pins anyway. `(region-close h)` is
an ordinary call, so unlike a lexical region exit it can run inside the `call/cc`
procedure and close a region the capture is standing in. A lexical `with-region`
cannot do that: its exit is downstream of the `call/cc` it encloses.

**It pins all of them, not the innermost one.** `eshkol_region_pin_all` walks
the whole region stack and marks every frame. A `call/cc` nested two
`with-region`s deep can be re-entered after both have exited, and either
frame's locals may need either arena, so pinning only the innermost would leave
the outer one free to be reclaimed out from under the resumed continuation.

**A pin, once taken, is never lifted.** Neither engine has an unpin path.
Deciding that a first-class continuation can no longer be invoked needs a
tracing collector, which neither substrate has. So a pin is a Stage-1 policy
that trades "this region's memory is not returned at region exit" for "no
continuation can ever observe a reclaimed region". What is decided per capture
is whether a pin is taken at all — see the escape-only rule above.

**A pinned region is promoted, on both engines.** The region's arena blocks are
spliced whole into the arena that encloses it — the parent region's arena when
nested, the process arena at the outermost level. They are never allocated from
again, so nothing the continuation still reads can be overwritten, and they are
freed when that enclosing scope ends. Memory is therefore not returned at region
exit, and if the enclosing scope is the process arena it is not returned until
the process exits. Nested pinned regions promote in one hop per level as they
pop, innermost first, and the move is zero-copy: **no promoted object changes
address**, which is why a resumed continuation's interior pointers stay valid.
The VM does this in `vm_evac_promote_all_blocks`; native does it in
`arena_adopt_blocks`, reached from `region_destroy`.

**Both engines say so on stderr, once per process.** The VM prints —

```
eshkol-vm: note: a `with-region` body could not be reclaimed and was promoted
whole (a continuation was captured inside a region). The answer is unaffected;
the memory is not returned until the enclosing scope ends.
```

— and native prints the same kind of note naming the same tradeoff.
`ESHKOL_VM_REGION_QUIET=1` silences both. An escape-only capture on native
prints nothing, because it takes no pin; that absence is itself gated, in
`region_callcc_flat_rss_test.sh`.

**That VM note has five possible reasons, and only two of them are yours.** The
reason string in parentheses distinguishes them: `a continuation was captured
inside a region` and `a continuation crossed the region boundary` are the
continuation cases; `ESHKOL_VM_REGION_EVAC=0`, `block table allocation failed`,
`mark bitset allocation failed` and `an object or value type the evacuator does
not classify` are not. Reading the parenthetical is the difference between
"my program captured a continuation" and "the evacuator ran out of memory".

##### Two native mechanisms, not one

Native has a second, independent protection that the pin does not subsume, and
it is what makes ["escape continuations pay nothing"](#escape-continuations-pay-nothing)
true for regions as well as for stack copying.

- **At codegen**, `codegenCallCC` chooses which arena the continuation's state,
  closure and stack image are allocated from. `with-region` redirects
  `eshkol_current_arena()`, so a capture the compiler classifies as *possibly
  escaping* is allocated from the process-wide shared arena instead, which
  outlives every region. A capture classified as escape-only keeps the current
  arena, because such a continuation cannot outlive the region body that
  created it and its state is correctly reclaimed with the region.
- **At run time**, `eshkol_make_continuation_state_flags` pins every open region
  when the region depth is greater than zero *and* the capture was not
  classified escape-only, because the raw C-stack snapshot of a capture that may
  escape can hold interior pointers into any open region's arena — pointers the
  codegen path cannot see and therefore cannot redirect.

The first protects what codegen can see; the second protects what it cannot.
Both consult the same classification, so an escape-only capture inside a
`with-region` neither copies the stack nor pins, and a capture that may escape
does both.

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
  See ledger SW-62.

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
