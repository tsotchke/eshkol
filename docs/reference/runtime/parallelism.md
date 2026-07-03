# Parallelism & Threading

Eshkol provides data-parallel primitives backed by a work-stealing thread pool
with per-thread arenas. The compiled binary contains both parallel and
sequential code paths; the runtime chooses at dispatch time and can be forced to
run sequentially.

## Primitives

| Form | Signature | Returns |
|------|-----------|---------|
| `parallel-map` | `(parallel-map fn list)` | list of `(fn item)` |
| `parallel-fold` | `(parallel-fold fn init list)` | folded value (`(fn acc item)`); sequential for non-associative ops |
| `parallel-filter` | `(parallel-filter pred list)` | filtered list |
| `parallel-execute` | `(parallel-execute thunks)` | runs a set of thunks, returns their results |

### Futures / promises

| Form | Purpose |
|------|---------|
| `(future expr)` | Spawn a lazy future; returns a handle |
| `(future-ready? h)` | Non-blocking readiness check |
| `(force x)` | Force a future **or** an R7RS promise (`delay`) to its value |

`force` handles both R7RS promises (heap `PROMISE` subtype) and lazy futures.

Parallel workers are runtime-registered by the stdlib; if the stdlib is not
loaded you get "workers not registered (stdlib not loaded?)". A lazy `dlsym`
fallback resolves them when possible.

## Thread pool

- **Threads**: default `std::thread::hardware_concurrency()` (fallback 4).
- **Per-thread arenas**: each worker allocates into its own 1 MB thread-local
  arena — zero contention (see [memory model](memory-model.md)).
- **Work-stealing**: on by default; each worker has its own deque (4096 initial
  capacity) with epoch-based reclamation and randomized victim selection. Set
  `ESHKOL_DISABLE_WORK_STEALING` to fall back to the legacy shared queue.
- **Worker stack**: 16 MB per worker by default; override with
  `ESHKOL_WORKER_STACK_BYTES` (floored at `PTHREAD_STACK_MIN`).

## Serialized state pattern

Parallel work must be **read-only over shared state**. Workers execute closure
bytecode in isolated VM/arena snapshots and publish returned values back to the
main heap; **any closure that may mutate shared VM state is run through a
serialized fallback on the main VM.** In the native (LLVM) path the same
discipline appears as result-publication-by-pointer: each task writes its result
into a pre-allocated slot (`result_ptr`) with no arena allocation during the
parallel phase — arena allocation happens only in the sequential result-assembly
phase afterward.

Practical rule: make the mapped/folded function pure. Thread an accumulator as a
loop variable rather than `set!`-ing a shared cell from inside parallel work.

## JIT warmup

Under the JIT, `parallel-map` runs `item[0]` on the calling thread first to force
ORC materialization of the whole transitive symbol set before dispatching workers
— otherwise every worker races on ORC's single materialization lock. Skip it with
`ESHKOL_PARALLEL_NO_WARMUP=1`. Raising `ESHKOL_JIT_COMPILE_THREADS` also relieves
that lock contention (see [JIT internals](jit-internals.md)).

## Controlling parallelism

| Variable | Effect |
|----------|--------|
| `ESHKOL_PARALLEL_DISABLE=1` | Force sequential inlined loop |
| `ESHKOL_PARALLEL_ENABLE=0` | Legacy disable |
| `ESHKOL_PARALLEL_NO_WARMUP=1` | Skip the ORC warmup pass |
| `ESHKOL_DISABLE_WORK_STEALING` | Use the legacy queue |
| `ESHKOL_WORKER_STACK_BYTES` | Per-worker stack size |
| `ESHKOL_DEBUG_PAR` | Print pool/task metrics |

## Known limitation — AD mode flag is a global

Automatic-differentiation state is *mostly* thread-isolated: the AD tape stack
(`__ad_tape_stack`, depth 32) is `thread_local`. However the `__ad_mode_active`
flag is a plain global, not thread-local, so one worker enabling AD mode can be
observed by another worker. **Do not run automatic differentiation
(`gradient`/`jacobian`/etc.) inside parallel workers** — compute gradients on the
main thread and parallelize only the pure numeric work.

For measured speedups and tuning, see
`docs/PARALLEL_MAP_PERFORMANCE_ANALYSIS.md`.
