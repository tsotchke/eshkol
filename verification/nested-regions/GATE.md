# Check the parent stack before destroying a nested child

Defining `ESHKOL_VERIFICATUM_NESTED_STACK` when compiling `lib/core/runtime_regions.cpp` records the region and arena identities seen on each lexical push, with a thread-local stack and an outer fallback. On a nested pop the runtime checks the child and parent identities, the selected arena and allocation domain, the active prefix and every inactive slot. It performs the final check after the child has been marked inactive and immediately before calling `region_destroy`. A mismatch aborts the process.

From the repository root, use the tools and public release described in the [restoration replay](README.md).

```sh
python3 verification/nested-regions/review_gate.py --release "$ESHKOL_RELEASE"
```

The replay compiles the runtime translation unit three ways and links each against the same release archive. All three must reproduce the two normal forms, cursor observations and Lean snapshot decisions. The Linux x86-64 run used GCC 16.2.1 with `-O1 -DNDEBUG`, GNU ld 2.47, glibc 2.44 and Python 3.14.9. The instrumented build also uses `-finstrument-functions -fno-inline`.

| Build | Healthy runs | Denial control |
|---|---|---|
| Default | Both complete | Healthy observations only |
| Gate with function-entry instrumentation | Two child destructor entries | SIGABRT, zero entries |
| Gate with ordinary compilation | Both complete | SIGABRT at the restored parent |

The control keeps the still-live child identity in the last inactive slot. The abort observer checks that the parent is already current, the child is inactive, its former active slot is clear, and the final slot still contains it. Only the instrumented build measures function entries. The public runtime gains no control hook. The probe and wrappers are linked into the test executable.

`gate-receipt.json` records the build identities and observations. This check covers synchronous single-owner nested lexical pops, assumes valid push-time identities and requires distinct live arenas and a stable non-null outer fallback. The outermost pop, direct destruction, foreign holders, concurrent mutation and whole-arena reclamation require separate evidence. An abort terminates this run after the stack has been changed.
