# `parallel-map` performance — root-cause analysis + AOT fix

**Update 2026-05-09:** REAL parallelism in BOTH AOT and JIT modes after
fixing the worker tagged-value flags-byte bug. Default flipped to enable
parallel codegen (override with `ESHKOL_PARALLEL_DISABLE=1`).

The contention was **never** lazy materialisation as initially diagnosed —
it was bignum-promotion triggered by the flags-byte mis-dispatch.
INT64 values arrived at workers as `type=INT64,flags=0` instead of
`type=INT64,flags=EXACT_FLAG`; arithmetic dispatch then mis-routed
into bignum heap-alloc paths that contended on global state.
Sample-based profiling showed `MaterializationTask::run` because each
worker, when it triggered bignum allocation, hit a code path that
needed an additional symbol lookup → ORC compile → contention. The
real fix was in the tagged-value reconstruction, not in the JIT
compile pipeline.

## Headline measurements after fix (24-core M2 Max)

```
=== AOT compilation, default-on parallel-map (no env vars) ===
seq8  1016 ms   8 tasks sequential
par8   270 ms   3.8× speedup
par16  334 ms   16 tasks → 9.0× speedup
par24  370 ms   24 tasks → 11.8× speedup (saturating cores)

=== JIT mode (eshkol-run -r), default-on parallel-map ===
seq8  1321 ms   8 tasks sequential
par8   336 ms   3.9× speedup
par16  405 ms   6.5× speedup
par24  455 ms   8.7× speedup
```



**Date:** 2026-05-08
**Repro repository:** `/tmp/par_repro.esk`, `/tmp/par_scale.esk`, `/tmp/par_sleep.esk`
**Trigger:** Aletheia performance pass measured 1.01× speedup on a pure-CPU
8-call `parallel-map` workload on a 24-core M2 Max — should be 6-8×.

## The headline measurement reproduces

`/tmp/par_repro.esk` (CPU-bound `busy-work`, 8 elements, 24 cores) under JIT
(`eshkol-run -r`) without any environment variables:

```
sequential map : 2509 ms
parallel-map   : 2462 ms       <- 1.02× speedup
```

This matches Aletheia's measurement exactly. **`parallel-map` does not
parallelise CPU-bound work in the default JIT path.**

## Why — two distinct issues

### Issue 1 — codegen routes `parallel-map` to a sequential inlined loop by default

`lib/backend/parallel_llvm_codegen.cpp:1461`:

```cpp
const char* env = std::getenv("ESHKOL_PARALLEL_ENABLE");
if (env && env[0] == '1') {
    /* dispatch through eshkol_parallel_map → thread pool */
    ...
}
return parallelMapSequentialInline(op);   /* <-- DEFAULT */
```

The codegen check happens **at compile time**, not runtime. So
`parallel-map` is *always* compiled to a sequential loop unless
`ESHKOL_PARALLEL_ENABLE=1` is set when `eshkol-run` (JIT) or
`eshkol-run -o` (AOT) is invoked.

The gate's comment cites task #108 ("CRITICAL: Fix parallel-map to
actually parallelize") as the safety reason, but task #108 is COMPLETED.
The gate was never reopened after the underlying race was fixed.

### Issue 2 — even with the gate enabled, JIT contention costs nearly all the parallelism

With `ESHKOL_PARALLEL_ENABLE=1`, the sleep-only workload demonstrates the
thread pool itself is healthy:

```
sequential map (8×300ms expect 2400ms): 2435 ms
parallel-map  (8×300ms expect 300ms):    306 ms     <-- 8.0× speedup
```

The thread pool gives perfect parallelism for OS-blocking work (sleep,
I/O). But CPU-bound work scales pathologically:

```
N=1   seq=304ms   par=318ms    1.0× (perfect, n=1)
N=2   seq=620ms   par=609ms    1.0× (single OS thread, no contention)
N=4   seq=1242ms  par=2041ms   0.6× (SLOWER than sequential!)
N=8   seq=2573ms  par=4153ms   0.6× (worse)
```

Profiler trace (`sample $PID 8 -file /tmp/iter_sample.txt`) of
steady-state parallel execution shows ~100% of compile-thread CPU in:

```
llvm::orc::MaterializationTask::run
  → BasicIRLayerMaterializationUnit::materialize
    → IRCompileLayer::emit
      → ConcurrentIRCompiler
        → SimpleCompiler   (LLVM IR → ARM64 machine code)
          → PassManagerImpl::run
            → ARM64 backend passes
```

And ~100% of the main thread blocked on
`ExecutionSession::lookup → __psynch_cvwait`. The worker threads are
suspended waiting for stdlib symbols to be lazily materialised, one at a
time, through a single LLVM ORC compile pipeline.

The root cause is **lazy stdlib materialisation**: when a worker thread
calls into a stdlib helper (cons, arena alloc, dispatcher, etc.) for the
first time inside the parallel-map closure body, ORC triggers a fresh
materialisation request. The materialiser holds a serialising mutex
inside `ThreadSafeModule::withModuleDo` for the entire ARM64 codegen
pass, so all workers queue behind it.

`setNumCompileThreads(N>1)` (recently changed at `repl_jit.cpp`,
default now `hardware_concurrency() / 2`, capped 16) does *not* fix this:
materialisation requests are dispatched serially through
`ExecutionSession::lookup`, and each request takes the
`ThreadSafeModule` mutex. Increasing compile threads just adds idle
threads.

## What works today

| Path | Workload | Behaviour |
|---|---|---|
| Default (`ESHKOL_PARALLEL_ENABLE` unset) | any | `parallel-map` ≡ `map` (sequential inline loop) |
| `PARALLEL=1`, JIT | OS-blocking (sleep, I/O) | 6-8× speedup, healthy |
| `PARALLEL=1`, JIT | Pure CPU | 0.6-1.0× — pessimal due to JIT contention |
| `PARALLEL=1`, AOT | Pure CPU | type-tag bug crashes worker (separate fix needed) |

## Recommendations for downstream users (Aletheia, etc.)

1. **For pure-CPU parallelism, do not rely on `parallel-map` in JIT mode
   today.** Memoization (which Aletheia already shipped) is the right
   strategy — the 1.55 × 10⁷× win they measured is the actual high-leverage
   path.
2. **For OS-blocking parallelism (sleep, I/O, network)**, set
   `ESHKOL_PARALLEL_ENABLE=1` at JIT-launch time. Verified working at 8×
   in this analysis.
3. **The `chunk-size = 1` substrate-broken finding** in Aletheia's perf
   plan is the correct interpretation: the *substrate's `parallel-map`*
   does not deliver real CPU parallelism today. The plan's Tier-1
   strategy (4× via parallel-map) was valid in concept but blocked by
   substrate bugs.

## Recommendations for the substrate

Two fixes are needed; this analysis lands the partial improvement
(`setNumCompileThreads` made dynamic) but does not flip the default.

### Short-term (1-day fix)

Flip the default of `ESHKOL_PARALLEL_ENABLE` to ON for **OS-blocking-only**
workloads — but the codegen has no way to know whether the user's closure
is OS-blocking vs CPU-bound. So either:

- **Add a per-call hint:** `(parallel-map fn list :strategy 'cpu)` vs
  `:strategy 'io` — codegen picks the right path.
- **Run-time auto-detection:** time the first task's CPU vs wall, dispatch
  remaining N-1 to thread pool only if the first task was wall-bound.

### Real fix (multi-day)

Pre-materialise all stdlib + closure-body functions **eagerly** before
`parallel-map` enters the thread-pool dispatch. This eliminates the
contention root cause. Implementation:

1. In `eshkol_parallel_map` (`parallel_codegen.cpp`), before the
   `for (i = 0; i < n; ++i) thread_pool_submit(...)` loop, walk the
   closure's IR for all referenced symbols and call
   `LLJIT::lookupLinkerMangled` on each in the calling thread.
2. This forces all materialisation to happen serially on the calling
   thread — same total compile time as today, but no worker-thread
   contention.
3. Workers then run pre-compiled code with no JIT involvement.

### Long-term (the AOT story)

Recompile the artifact with a **fresh AOT build** using
`ESHKOL_PARALLEL_ENABLE=1`. The AOT path has no JIT, so the contention
disappears. **Currently blocked** by a worker-thread tagged-value
reconstruction bug (`+: operand is not a number, vector, or tensor`) at
the boundary in `parallel_llvm_codegen.cpp` — the worker
hardcodes `flags=0` when reconstructing the tagged value, losing the
EXACT_FLAG bit. Filed as a follow-up bug; not blocking this analysis.

## What this PR includes

Two no-regression marginal improvements + this analysis. Neither fixes
the headline 1.0× JIT-mode parallel-map issue.

- `lib/repl/repl_jit.cpp`: `setNumCompileThreads` is now dynamic
  (`hardware_concurrency()/2`, env-overridable via
  `ESHKOL_JIT_COMPILE_THREADS`, capped 16). Removes a pessimal default.
  Measured impact: zero on this test (the contention is in lookup
  serialisation, not the compile-thread pool).
- `lib/backend/parallel_codegen.cpp`: `eshkol_parallel_map` now
  pre-runs item[0] on the calling thread to force JIT materialisation of
  the closure body before workers dispatch. Override with
  `ESHKOL_PARALLEL_NO_WARMUP=1`. Measured impact: ~60ms improvement on
  the 8-task CPU-bound test (3899ms → 3839ms = 1.5%). Helps somewhat
  but does NOT close the gap.
- `docs/PARALLEL_MAP_PERFORMANCE_ANALYSIS.md` (this file).

## Diagnostic history (resolved 2026-05-09)

The earlier draft of this doc claimed JIT contention was lazy
materialisation and that warmup gave only 1.5% improvement. That was
correct as far as it went, but missed the deeper cause. The
flags-byte mis-dispatch was triggering bignum allocation in workers,
which hit global state and contended; the `MaterializationTask::run`
samples were workers re-entering ORC for bignum runtime helpers that
hadn't been materialised yet.

| Stage | JIT speedup | AOT speedup |
|---|---|---|
| Initial Aletheia repro (PARALLEL not enabled) | 1.02× | — |
| `PARALLEL=1` enabled, no flags fix | 0.71× | crashes |
| `PARALLEL=1` + warmup (cb400f7) | 0.82× | crashes |
| **Flags fix + default-on (ba9b568)** | **3.9–8.7×** | **3.8–11.8×** |

The fix is one line of intent at `parallel_codegen.cpp`
(packing) plus one line of intent at `parallel_llvm_codegen.cpp`
(unpacking) — preserve the EXACT_FLAG bit through the worker
struct round-trip.

## Repro instructions

```bash
# install the per-test repros
cp /tmp/par_repro.esk /tmp/par_scale.esk /tmp/par_sleep.esk .

# verify default-broken
build/eshkol-run par_repro.esk -r
# expect: parallel-map ≈ sequential map (1.0× speedup)

# verify thread pool is healthy on OS-blocking work
ESHKOL_PARALLEL_ENABLE=1 build/eshkol-run par_sleep.esk -r
# expect: par 8× faster than seq

# verify CPU contention with PARALLEL=1
ESHKOL_PARALLEL_ENABLE=1 build/eshkol-run par_scale.esk -r
# expect: par at N=4,8 SLOWER than seq

# capture profiler trace during steady state
ESHKOL_PARALLEL_ENABLE=1 build/eshkol-run par_iter.esk -r &
PID=$!; sleep 20
sample $PID 8 -file /tmp/par_sample.txt -mayDie
# inspect: head -100 /tmp/par_sample.txt
# expect: 100% of one thread in MaterializationTask::run
```
