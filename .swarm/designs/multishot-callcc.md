# Multi-shot `call/cc` via stack copying — design

Author: claude-main · 2026-05-27 · v1.3-evolve critical-path item

This is the locked design for the v1.3 multi-shot continuation work
(`ESH-NNNN to be created`). Read this before touching any code in
`codegenCallCC`, `runtime_continuations.cpp`, the `dynamic-wind`
machinery, or the OALR region refcount path. Architectural decisions
recorded here are final for v1.3; see Open Questions if you need to
revisit one.

---

## Current implementation (single-shot, what we're replacing)

`call/cc` lives at `lib/backend/llvm_codegen.cpp` around line 6395-6441.

- Continuation value is a `CALLABLE` heap object with
  `CALLABLE_SUBTYPE_CONTINUATION`.
- Continuation state struct (no name in source, layout reconstructed):
  ```
  { void* jmp_buf_ptr;        // offset 0
    tagged_value_t value;      // offset 8 — written by invoker
    void* wind_mark; }         // offset 24 — dynamic-wind unwind anchor
  ```
- Capture path (`codegenCallCC`):
  1. `alloca` a `jmp_buf`.
  2. `setjmp(jmp_buf)` → 0 first time, non-zero on `longjmp` return.
  3. Build the state struct on the heap.
  4. Pack into a CALLABLE with subtype CONTINUATION.
  5. Pass to the receiver lambda.
  6. On return, the `setjmp` site reads `state->value` and yields it.
- Invoke path (around line 6390):
  1. Detect CALLABLE_SUBTYPE_CONTINUATION at the call site.
  2. Store invoke value into `state->value`.
  3. `eshkol_unwind_dynamic_wind(wind_mark)` to run `after` thunks.
  4. `longjmp(state->jmp_buf, 1)` — single-shot semantics:
     after this, the `jmp_buf` is invalid; re-invocation is UB.

This is **single-shot** because `setjmp`/`longjmp` does **not** save the
C stack frames between the call/cc site and the current frame. After
`longjmp` returns, those frames are gone. Re-invoking the continuation
would `longjmp` to a stack frame that no longer exists.

---

## v1.3 design — multi-shot via stack copying

Approach: at capture, snapshot the entire C stack between a known
low-watermark (the Eshkol-entry frame) and the current frame into a
heap arena. At each invocation, restore that snapshot over the current
stack region.

### Stack representation

The Eshkol process has a single main stack (per thread). The
relevant region is bounded by:

- **High watermark (capture top):** the address of the local frame
  at the call/cc site. Determined by taking the address of an
  `alloca`'d sentinel variable at the top of `codegenCallCC`.
- **Low watermark (capture base):** the address of a fixed
  thread-local sentinel set up by `eshkol_init_stack_size` at
  process entry, before any user code runs. Stored as
  `pthread_specific_t g_eshkol_stack_base` (already exists for the
  100K-frame recursion-depth check — repurpose).

On x86_64 and arm64 stacks grow **downward**, so capture_top is
numerically *less* than capture_base.

### Captured state struct (replaces the current 3-field state)

```
struct eshkol_continuation_t {
    /* New: stack snapshot */
    void*    stack_low;        /* address of capture_top — bottom of snapshot */
    size_t   stack_size;       /* capture_base - capture_top */
    uint8_t* stack_copy;       /* heap-arena copy of the captured region */

    /* New: machine state at capture point */
    void*    saved_pc;         /* return address into the call/cc receiver lambda */
    void*    saved_fp;         /* frame pointer at capture */
    /* Callee-saved registers required by the ABI. arm64: x19-x28, fp, lr.
       x86_64: rbx, rbp, r12-r15. Captured via inline asm at the
       call/cc site, restored on invocation. */
    uintptr_t saved_callee_regs[12];

    /* Existing: value slot for invocation */
    tagged_value_t value;

    /* Existing: wind mark for dynamic-wind */
    void*    wind_mark;
    /* New: wind state — captured stack copy of the wind-stack so we can
       rewind correctly on re-invocation (after thunks from current pos,
       then before thunks down to captured pos) */
    void*    wind_snapshot;
    size_t   wind_snapshot_size;

    /* New: OALR region refcount pins for v1.3 */
    void**   pinned_regions;   /* array of region_descriptor* with refcount bumped */
    size_t   num_pinned_regions;

    /* New: invocation counter — debug + future GC hints */
    int      invocation_count;
};
```

### Capture sequence (replaces `codegenCallCC`)

1. Take address of a local sentinel as `capture_top`.
2. Snapshot callee-saved registers via inline asm into
   `saved_callee_regs`.
3. Compute `stack_size = (capture_base - capture_top)`.
4. Allocate `stack_copy` in the persistent continuation arena
   (NOT the OALR region — this lives independent of region scope).
5. `memcpy(stack_copy, capture_top, stack_size)`.
6. Walk the wind stack from `wind_mark` to capture_base; snapshot
   it. (Wind stack is heap-allocated already; we just record the
   active before/after thunks at each level.)
7. Walk the OALR region stack from `wind_mark` to capture_base;
   for each live region descriptor, bump its refcount and add to
   `pinned_regions`.
8. Pack the struct into a CALLABLE with `CALLABLE_SUBTYPE_CONTINUATION`.
9. Call the receiver lambda with this continuation.
10. On normal return: return value from the receiver propagates.
11. On invocation (separate path): we land at step 5 of the
    invoke sequence below.

### Invoke sequence (replaces the current longjmp path)

1. Detect CALLABLE_SUBTYPE_CONTINUATION at the call site.
2. Forbid-AD-context check: if `eshkol_ad_tape_depth() > 0`, raise
   a runtime error: `"call/cc capture inside gradient/derivative is
   not supported in v1.3"`. (See decisions.md.)
3. Unwind dynamic-wind from current `wind_mark` to a common
   ancestor of (current wind chain) and (captured wind chain),
   running `after` thunks along the way.
4. Re-wind dynamic-wind from the common ancestor down to the
   captured `wind_mark`, running `before` thunks along the way.
5. Adjust current stack pointer: discard everything between current
   SP and `capture_top`.
6. `memcpy(capture_top, stack_copy, stack_size)`.
7. Restore callee-saved registers from `saved_callee_regs` via
   inline asm.
8. Write the invocation value into the captured frame's `value`
   slot (offset known from the stack snapshot's known layout —
   the value slot lives at a known offset from `capture_top`).
9. `__builtin_longjmp` to (`saved_pc`, `saved_fp`).
   Yes, this uses `__builtin_longjmp` rather than the C library
   `longjmp` because the latter validates that the jmp_buf
   belongs to the calling thread + frame. `__builtin_longjmp` is
   the lower-level primitive that just sets pc/sp and jumps.
10. The captured frame's setjmp-like landing point reads the
    value and continues.
11. Continuation may be invoked again — the stack_copy is intact,
    so step 6 onward repeats with a fresh restore.

### Forbidden: capture inside AD tape

The AD tape's reverse-mode is fundamentally a sequence of tape-node
writes. Capturing in the middle and re-invoking later would either
replay or skip those writes, producing wrong gradients.

For v1.3 we simply trap: `eshkol_callcc_capture` checks
`eshkol_ad_tape_depth() > 0` and if so raises `"call/cc capture
inside (gradient ...) / (derivative ...) is not supported in v1.3.
Move the call/cc outside the gradient frame, or use AD with no
continuation capture."`

v1.5 will revisit this when AD is restructured for soft unification.
The intended semantic at that point: continuation capture inside AD
also snapshots the tape; invocation restores both the C stack AND
the AD tape head. But that's v1.5 work, not v1.3.

### OALR region refcount

OALR regions are scope-bound: they get popped when their lexical
scope exits. If a captured continuation outlives its enclosing region
(e.g. captured inside a `with-region`, then invoked after the
region's `pop`), invocation would `longjmp` back into a region that's
already been freed.

Fix: at capture, walk the region stack from the current frame down
to `capture_base`. For each region descriptor, bump its refcount
(new field). Store the descriptor list in `pinned_regions`. The
region's `pop` implementation decrements refcount but only frees the
arena when refcount reaches 0.

When the continuation is GC'd (which currently never happens; in
v1.3 the continuation arena is freed at program exit), the pinned
region descriptors get a matched decrement, eventually allowing
their arenas to be freed.

### dynamic-wind

The current wind-stack already supports cross-call/cc unwinding (the
`wind_mark` + `eshkol_unwind_dynamic_wind` machinery). Multi-shot
needs the additional property that a re-invocation re-runs the
`before` thunks of the captured wind chain.

Implementation:

1. At capture: snapshot the wind chain (linked list of
   before/body/after triples).
2. At invoke: walk current wind chain back to the common ancestor
   of (current chain) and (captured chain). Run `after` thunks.
3. Walk down the captured chain from the common ancestor, running
   `before` thunks.
4. Now the wind stack matches the captured state; proceed with
   stack restore.

The "common ancestor" comparison is by `wind_id` (a monotonic
counter stamped on each wind frame at push time). Frames in both
chains with the same wind_id are the common prefix; everything past
that is the diff.

### Memory cost

Per captured continuation:
- Stack snapshot: typically ~4-16 KiB (a few hundred frames worth).
- Wind snapshot: ~256 bytes plus N × 48 bytes per active wind
  frame (typically N ≤ 16).
- Pinned regions: N × 8 bytes per active OALR region (typically
  N ≤ 8).
- Struct itself: ~256 bytes.

Total: ~5-20 KiB per continuation. For most code that captures a
few continuations this is fine. Heavy users (backtracking solvers)
typically run hundreds or thousands of captures — at 20 KiB each
that's a few MiB of arena, acceptable.

### Open questions (NOT v1.3-blocking)

1. Can we share stack-copy storage across continuations that captured
   from the same outer scope (linked-frame structure-sharing)?
   Probably yes; deferred to v1.4 if profiling shows the memory cost
   matters.
2. Should `call/cc` be a Scheme-level macro on top of a primitive
   `%capture-stack` and `%restore-stack`? Marginally cleaner separation
   but more codegen surface. Stick with the current "call/cc is a
   special form" approach for v1.3.
3. Interaction with parallel-map and futures: continuations captured
   in one thread invoked in another. Forbid for v1.3 with a runtime
   trap. Same shape as the AD-tape forbid.

---

## Acceptance criteria for the multi-shot work

When the implementation lands, the following must hold:

1. **R7RS conformance**: a continuation can be invoked any number of
   times; each invocation re-enters at the call/cc site with the new
   value.
2. **dynamic-wind interaction**: re-invocation triggers the captured
   wind chain's `before` thunks; exits trigger `after` thunks of the
   current chain.
3. **OALR**: a continuation captured inside `(with-region ...)` can
   still be invoked after the region exits (the region was pinned).
4. **AD forbid**: `(call/cc ...)` inside `(gradient ...)` raises a
   clear runtime error, not a silent corruption.
5. **No regression on single-shot use cases**: every existing
   continuation-using test in `tests/v1_2_edge_cases/` still passes.
6. **Performance**: capture latency under 10 µs on M-series for a
   typical 4 KiB stack snapshot; invoke latency under 5 µs.
7. **ICC oracle**: a new probe `multishot_callcc_works` lands and
   passes.

---

## Test plan

`tests/v1_3_edge_cases/multishot_callcc_test.esk` covers:

- Single-shot baseline (regression).
- Invoke twice with different values; second invocation re-runs
  the body after the call/cc site.
- Capture inside `let`, invoke after `let` exit → bindings still
  visible (lexical capture).
- Capture inside `dynamic-wind`, invoke after exit → before/after
  thunks run correctly.
- Capture inside `with-region`, invoke after region exit → region
  was pinned.
- `(call/cc (lambda (k) (gradient (lambda (x) (k 5)) 2.0)))` → raises
  the forbid-AD error.
- 100k captures + invocations stress test → no leaks (heap RSS
  stable).
- Generators via call/cc + queue: works.
- amb implemented as a library on top of call/cc (Friedman/Felleisen
  style) → passes SICP §4.3 multiple-dwelling example with no
  metacircular evaluator.

---

## Sequencing

This is critical-path single-thread work. Do not dispatch via the
swarm. Implementation order:

1. Add `pinned_regions` refcount infrastructure (small).
2. Add wind-chain snapshot/diff machinery.
3. Replace continuation state struct.
4. Replace capture path in `codegenCallCC`.
5. Replace invoke path at the CALLABLE_SUBTYPE_CONTINUATION dispatch.
6. Add the AD-forbid trap.
7. Write the test file; iterate until all 9 cases pass.
8. Add the ICC `multishot_callcc_works` probe.
9. Run the full v1_2_edge_cases regression to confirm no break.
10. Update `SCHEME_COMPATIBILITY.md`: §6.10 call/cc — flip from
    "Limited (single-shot)" to "Full".

Estimate: 3-4 weeks for the architectural-coherent design + careful
testing. The QGTL-style sub-tasks (`pinned_regions` infra, wind
snapshot, etc.) are individually small but interlock; one mind has
to hold them.
