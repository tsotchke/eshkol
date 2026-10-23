# ADR-0030: Fibers run on stackful coroutines with a per-context dynamic state

- **Status:** Accepted — implemented on native JIT and AOT; the bytecode VM is
  Planned for v1.4.1
- **Date:** 2026-10-23
- **Decision owners:** Eshkol runtime
- **Applies to:** `inc/eshkol/core/coroutine.h`,
  `lib/core/runtime_coroutines_hosted.cpp`, `lib/core/fibers.esk`,
  `lib/core/channels.esk`, and every runtime file that keeps dynamic state in
  thread-local storage (`runtime_continuations.cpp`,
  `runtime_exceptions_hosted.cpp`, `runtime_regions.cpp`,
  `resource_limits.cpp`, `runtime_stack_hosted.cpp`,
  `lib/backend/parallel_codegen.cpp`)
- **Relates to:** ADR-0001 (per-computation memory context), the event loop
  (`inc/eshkol/core/event_loop.h`), roadmap item #159

## Context

v1.4 needs fibers: many lightweight cooperative tasks on one OS thread,
scheduled over the portable event loop, that can park on descriptor
readiness, timers, futures and channels, be joined, and be cancelled with
their `dynamic-wind` cleanup run exactly once.

The language already has re-entrant first-class continuations, and the first
design question was whether a fiber should simply be a continuation. On
native code a continuation is a copy of the C stack from the capture point to
the stack base, allocated in the arena and restored in place
(`runtime_continuations.cpp`). Three properties of that design rule it out as
the switching mechanism:

1. **Cost and memory.** Every switch would copy the stack and allocate the
   copy in the arena, which only region exit reclaims. Ten thousand fibers
   switching in a loop would grow memory without bound.
2. **Dynamic-wind.** Invoking a continuation reroots the wind chain: it runs
   the `after` thunks of the extent it leaves and the `before` thunks of the
   extent it enters. A fiber switch must run neither; a fiber parked inside a
   `dynamic-wind` has not left the extent.
3. **Host extents.** The REPL refuses a continuation whose image belongs to a
   finished top-level evaluation. A fiber spawned in one evaluation and
   resumed in a later one would be refused.

## Decision

**A fiber is a stackful coroutine; a coroutine owns a stack and an execution
context.**

- `make-coroutine` creates a coroutine for a nullary procedure. Its stack is
  mapped (with a guard region) on first resume, recycled through a bounded
  per-thread pool when it finishes, and never arena-allocated. The
  Scheme-visible coroutine is a generation-tagged integer handle, the
  convention of every kernel-resource holder.
- The machine switch saves callee-saved registers and swaps the stack pointer:
  hand-written for x86-64 System V and AArch64 (Linux, macOS, FreeBSD), the
  system Fibers API on Windows. WebAssembly has no second machine stack, so
  the surface fails closed there.
- **One record of the running computation's dynamic state.** Every runtime
  subsystem that kept "the dynamic state of the running computation" in
  thread-local storage now saves and loads it through
  `eshkol_exec_context_t` (`inc/eshkol/core/coroutine.h`): the dynamic-wind
  chain, the exception-handler chain and in-flight raise, the
  promise-evaluation chain, the parallel unwind-boundary chain, the lexical
  region stack and the allocation domain it selects, the recursion-depth
  counter, and the stack geometry. Each subsystem owns its own fields; the
  coroutine runtime calls all of them on every switch. A new coroutine starts
  from an empty context. This is ADR-0001's per-computation memory context at
  the granularity of an execution context.
- **One unwind boundary.** A coroutine body runs under the same boundary a
  `parallel-map` callback runs under (`run_in_boundary`,
  `parallel_codegen.cpp`): a raise inside it is recorded at the coroutine's
  base, after the body's winds, regions and handlers have unwound, and is
  re-delivered with its original object to whoever resumed the coroutine.
- **Continuations compose.** Inside a coroutine, `call/cc` captures only the
  coroutine's own stack (the stack-base probe reports the coroutine stack),
  and a stack image records its owning context. A continuation is re-entered
  only on the coroutine whose stack it captured; invoking it elsewhere raises
  a catchable condition before any state changes.
- **Values crossing a switch pass the region write barrier**, so neither the
  coroutine's procedure nor a transferred value can point into a region that
  closes under it.
- A switch while a reverse-mode differentiation pass is active is refused
  with a catchable condition: a tape that interleaves two computations has no
  meaning.
- **Scheduling is library code.** `core.fibers` resumes fibers from the
  resuming context: FIFO run queue, a deadline heap for sleepers, descriptor
  waiters registered with the event loop, futures and general conditions.
  Cancellation is `coroutine-raise` of a fiber-cancelled condition at the
  fiber's suspension point, so cleanup is ordinary unwinding.
- **Channels park fibers.** A blocking `channel-send!` / `channel-recv!`
  called from a fiber releases the channel mutex and parks the fiber until the
  channel can proceed; thread waiters keep their condition variables.

## Alternatives rejected

- **Fibers as continuations** — rejected for the three reasons above.
- **ucontext** (`makecontext` / `swapcontext`) — deprecated on macOS, absent
  from musl, and `swapcontext` makes a signal-mask system call on every switch.
- **One fiber per OS thread** — the existing `make-thread` already provides
  that; it does not give ten thousand cheap tasks or cooperative ordering.

## Consequences

- Suspending inside `with-region`, `parameterize` (whose bindings ride the
  dynamic-wind chain) or a `guard` works: each fiber's regions, winds and
  handlers are its own.
- A coroutine is resumed only by the thread that created it.
- A suspended coroutine whose frames are still live cannot be released; it is
  finished first (cancellation unwinds it).
- Process exit does not unwind suspended coroutines, exactly as it does not
  unwind other threads.

## Stages

1. **v1.4.0-connection (this record):** coroutines and `core.fibers` on
   native JIT and AOT on Linux and macOS (x86-64 and AArch64), the Windows
   Fibers API path, channel integration, and the fibers gate
   (`scripts/run_fibers_gate.py`, criterion `v14_fibers_coroutines`).
2. **v1.4.1:** coroutines on the bytecode VM. The VM keeps top-level bindings
   and control in one operand array, so a suspended coroutine needs its own VM
   control stack; `tests/vm_parity/PARITY.tsv` records the gap until then.
   Wake lists on channels, so a same-thread send wakes its receiver directly
   instead of through the scheduler's condition check.
3. **v1.4.2:** one scheduler per OS thread with cross-thread wakeups through
   the event loop, for fibers fed by pool threads without condition polling.
