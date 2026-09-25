# Constructor and exception-handler allocation hardening

This follow-up targets the pending v1.3.5 release implementation. It preserves
that implementation's promotion transaction, destination rollback, and existing
per-thread allocation-failure condition. It does not replace the promotion ABI.

Generated cons, vector, and closure constructors check allocation results before
initialization or publication. A capturing closure whose environment allocation
fails returns null; capture-count arithmetic also rejects size overflow. These
checks cover the constructor lowering paths changed here, not every allocator in
the compiler or numeric/tensor runtime.

Every handler push primes the existing allocation-failure condition before
publishing its frame. A later failed frame allocation raises to the previously
installed handler instead of continuing with a missing frame. With no handler,
the existing uncaught-error policy exits. Priming costs one small private arena
per runtime thread, shared with the release's region-promotion failure path.

## Handler reservation

```c
int64_t eshkol_runtime_reserve_exception_handlers_v1(int64_t free_count);
```

- Returns zero when at least `free_count` inactive frames are available on the
  calling thread. Zero is a no-op. Existing capacity is never reduced.
- Returns -1 for a negative count or a count exceeding
  `SIZE_MAX / sizeof(eshkol_exception_handler_t)`, before allocating or mutating
  the pool. This replaces the unshipped draft's emergency-condition contract;
  consumers must check the return value.
- Allocation failure raises the release's allocation condition to the previously
  active handler. Successfully reserved frames remain inactive and reusable, so
  retry allocates only the deficit.
- Reserved capacity covers additional simultaneous handler pushes. Pop recycles
  frames, so sequential entries do not consume the reservation permanently.
- New frames initialize the release's `replay_values` and `replay_capacity`
  fields; recycled frames retain their replay buffer and reset its active state.
  Reservation does not preallocate replay snapshots or arbitrary guard-body work.
- The pool and active exception state are thread-local in this release. The
  existing inactive pool retains peak storage and has no production destructor;
  this change does not claim leak-free transient-worker use. Test-only drain
  hooks free replay buffers as well as frames and are absent from normal builds.

JIT registration includes the reservation and allocation-failure symbols. The
browser glue supplies the new imports: allocation failure throws a host error;
positive handler reservations explicitly throw as unsupported because that
browser runtime has no native handler chain. Native allocation recovery is not a
claim about browser exception semantics.

## Verification

Configure the ordinary build with `-DESHKOL_ALLOCATION_TESTING=ON`, then run:

```sh
cmake --build build --target allocation-hardening-tests compiler_public_api_linkage
ctest --test-dir build --output-on-failure -L allocation-hardening
ctest --test-dir build --output-on-failure -R '^(region_promotion_failure_test|runtime_closure_alloc_test|compiler_public_api_linkage)$'
python3 scripts/gen_api_docs.py --check --no-trace
python3 scripts/generate_wasm_import_glue.py --check
```

The opt-in failure-injection suite requires Linux and GNU/Clang linker `--wrap`.
It tests invalid/partial reservation, retry, exact capacity, repeated reuse,
replay-buffer initialization and reuse, fresh-thread condition priming, persistent
allocation refusal, thread-local isolation, closure size overflow, and failed
closure-environment allocation. Both malloc and calloc are intercepted because
optimizers may fold zero-initializing allocation into calloc.

The optimized AOT fixture injects eight constructor/handler failures, including a
failed captured-lambda environment, and checks operand
evaluation order and reuse of an existing `apply` argument list. An IR control-flow
check covers 38 emitted constructor sites, proves the success branch dominates
pointer uses, and rejects deliberately
removed-branch and failure-store mutants. JIT and AOT also exercise the public
reservation ABI and ordinary checked constructors.

Local validation uses LLVM 21 with Clang 22 on Linux. The native failure suite and
unchanged promotion-failure regression are also built with ASan/UBSan and run
with leak detection enabled. The AOT fixture is additionally linked to the
sanitized runtime and a sanitized C++ injection shim; its generated Eshkol object
comes from the ordinary compiler build. The compiler itself is not claimed to be
sanitizer-validated by that run. Other native platforms need their CI builds;
the Linux failure-injection mechanism is not presented as cross-platform proof.

Six existing regression fixtures also pass in optimized JIT and AOT modes against
their checked-in expected output: nested guards, guard/dynamic-wind ordering,
handler interplay, captured handler snapshots, nested region continuation resume,
and assignment captured by a guard handler. Public API linkage verifies all 104
umbrella-header exports, including the new reservation symbol.

The strict WASM import smoke compiles all 13 existing surfaces and verifies 135
unique imports against both browser glue files, including the new constructor
allocation-failure import. The glue freshness/contract and import-scanner
self-tests also pass. This establishes import compatibility, not browser guard
recovery or Windows/macOS runtime behavior.

## Release-candidate performance probe

`bench/allocation_hardening_probe.esk` retains one million cons cells and then
performs one million `vector-set!` calls. Compile it with each tree's
`eshkol-run -n -O3 -o <binary> bench/allocation_hardening_probe.esk -L<build>`;
run the resulting binaries alternately on one CPU. The recorded outputs were
checked for list head `1` and final vector element `999999`.

On Linux x86-64 (Ryzen 7 3700X, Clang 22.1.6, LLVM 21.1.8), comparing release
candidate `8f75ca49` with this branch, eight serial samples per binary pinned
to CPU 7 gave median cons times of 1,545,259,529 vs. 1,539,394,269 jiffies
(PR/base 0.996), and median vector-store loop times of 83,789,535 vs.
86,969,041 jiffies (PR/base 1.038). The baseline store samples included one
122,214,689-jiffy outlier. The `fill` function's 616 disassembled instructions
are identical in both binaries after normalizing addresses. Thus this probe
finds no added per-store instruction, but the roughly 4% elapsed difference is
not enough to rule out layout, cache or background-load effects. It is not a
whole-application performance claim.
