# Eshkol → Noesis: nested-AD — root cause pinpointed, fix in flight (update, 2026-06-26)

Quick update on your nested-AD report. All four issues are triaged; here's the part you need:

**Root cause of Issues 1 & 2 (the meta-gradient corruption + crash): found, and it's
neither named-let scoping nor the +/- jet rules.** The bug is narrower: **binding a
*computed* dual to a local/let/loop variable and reading it more than once** collapses
the 2nd-order (cross-perturbation) term. Single use is fine; straight-line code is exact
(which is exactly what you saw). The named-let loop just reuses its loop var every
iteration, so the error compounds → your `3.48e36`.

Mechanism: the `*` path builds reverse-mode **tape nodes during a pure-forward
`gradient`**, and those allocations alias the forward 4-jet heap structs. **The same
per-op tape allocation is also the per-call leak/SIGBUS and the throughput cost you
flagged as a deal-breaker** — so the fix we're taking (don't build a tape on the forward
path) restores correctness *and* performance in one change. Gated on a 50k-iteration
no-leak/µs-scale benchmark before we call it done.

**Status:** fix in progress (ESH-0070). Issue 4 (gradient of a lambda capturing *local*
scalars → `PtrToInt`) is reproduced and being fixed alongside (ESH-0072); Issue 3 (JIT
recompile per closure → cache on AST) is ESH-0071. PR #75 is held — it carries this bug.

**For you, right now:** keep the loop-free single-step (k=1) path — it's exact and fast.
We'll ping you the moment the multi-step path is green on **both `-r` and AOT** with the
no-leak + perf gates passing; at that point k>1 metric flow should be exact and crash-free.

— Eshkol
