# Coroutines reference

Status: SHIPPED on native JIT and AOT (v1.4.0-connection). Bytecode VM:
Planned for v1.4.1. Browser: the builtins answer `#f`.

A coroutine runs a procedure of no arguments on a stack of its own. The
procedure starts on the first `coroutine-resume`; `coroutine-yield` hands a
value back to whoever resumed the coroutine and suspends it; the next resume
continues from that point, and its value becomes the yield's result. This is
the asymmetric (resume/yield) shape of Lua coroutines and Python generators.
[`core.fibers`](../stdlib/fibers.md) is a scheduler built on it.

Header: [`inc/eshkol/core/coroutine.h`](../../../inc/eshkol/core/coroutine.h).
Implementation:
[`lib/core/runtime_coroutines_hosted.cpp`](../../../lib/core/runtime_coroutines_hosted.cpp).
Design: [ADR-0030](../../design/adr/0030-stackful-coroutines-for-fibers.md).

## Builtins

| Builtin | Result |
|---|---|
| `(make-coroutine thunk)` | a suspended coroutine (an integer handle) that will run `(thunk)` |
| `(coroutine-resume co [value])` | runs `co` until it yields or returns; the yielded or returned value. A raise that escapes the body is re-raised here with its original object |
| `(coroutine-raise co obj)` | resumes `co` by raising `obj` at its suspension point. A coroutine that never ran finishes without running, and `obj` is raised in the caller |
| `(coroutine-yield [value])` | suspends the running coroutine; the next resume's value. Raises outside a coroutine |
| `(coroutine-status co)` | `suspended`, `running`, `normal` (it resumed another coroutine), `done` or `failed` |
| `(coroutine-release! co)` | frees a finished or never-started coroutine's handle: `#t`, or `#f` when already released. Raises for a coroutine whose body is suspended mid-way |
| `(current-coroutine)` | the running coroutine, or `#f` on the thread's own stack |
| `(coroutine? x)` | `#t` for a handle that names an unreleased coroutine |
| `(coroutine-stats)` | association list: `live` (coroutines holding a stack), `peak-live`, `created`, `stacks-mapped`, `stacks-pooled`, `stack-bytes`, `handles-open` |

## What a switch preserves

Each coroutine is its own execution context. A switch saves the outgoing
context and loads the incoming one; it runs no Scheme code. The context is:

- the dynamic-wind chain — no `before` or `after` thunk runs because of a
  switch; a coroutine's `after` thunks run exactly once, when its body leaves
  the extent;
- the exception-handler chain and the raise in flight;
- `parameterize` bindings — a coroutine sees its own bindings, never another's;
- open regions and the allocation domain they select;
- the promise-evaluation chain and the parallel unwind boundary;
- the recursion-depth count and the stack the overflow guard measures.

A new coroutine starts from an empty context. Values held across a switch (the
procedure, and every transferred value) pass the region write barrier, so they
never point into a region that closes while they are held.

## Continuations inside a coroutine

`call/cc` works inside a coroutine and captures only that coroutine's stack. A
continuation is resumed only on the coroutine that captured it, including
after that coroutine was suspended and resumed; invoking it from anywhere else
raises a catchable condition. A continuation captured outside a coroutine and
invoked inside one ends the coroutine, and the jump completes in the context
that resumed it, the same rule a `parallel-map` callback follows.

## Limits

- A coroutine is resumed only by the thread that created it.
- A switch inside a reverse-mode differentiation pass is refused with a
  catchable condition.
- Stacks reserve `ESHKOL_FIBER_STACK_BYTES` (default 1 MiB, minimum 128 KiB)
  of address space each, committed only as touched, with a guard region below.
  Exhausting one prints a stack-overflow diagnostic naming the variable and
  exits with the stack-limit status.
- Finished stacks are kept in a per-thread pool of up to 64 for reuse; the rest
  are returned to the operating system.
- Process exit does not unwind suspended coroutines.

## Example

```scheme
;; coroutines_example.esk
(define gen
  (make-coroutine
   (lambda ()
     (coroutine-yield 1)
     (coroutine-yield 2)
     'done)))
(display (coroutine-resume gen)) (newline)
(display (coroutine-resume gen)) (newline)
(display (coroutine-resume gen)) (newline)
(display (coroutine-status gen)) (newline)
(display (coroutine-release! gen)) (newline)
```
```
1
2
done
done
#t
```

## Evidence

`tests/fibers/coroutine_primitives_test.esk` runs through JIT and O1 AOT in
`scripts/run_fibers_gate.py`.
