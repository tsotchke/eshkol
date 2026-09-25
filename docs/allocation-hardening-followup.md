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

The inactive handler pool remains thread-local and retains peak storage. Fresh
frames initialize replay fields; recycled frames retain their replay buffer and
reset its active state. Test-only drain hooks free the frames and buffers. No
public handler-reservation API is introduced: the VM and browser have different
handler machinery, and this fix needs only the existing native push operation.

The JIT registers the allocation-failure symbol. Browser glue supplies its
import as a host throw; native allocation recovery does not imply browser guard
recovery.

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
It tests handler-frame failure, replay-buffer initialization and reuse,
fresh-thread condition priming, persistent allocation refusal, thread-local
isolation, closure size overflow, and failed closure-environment allocation.
Both malloc and calloc are intercepted because optimizers may fold
zero-initializing allocation into calloc.

The optimized AOT fixture injects nine constructor/handler failures, including
failed closure-object and captured-lambda environment allocations, and checks
operand evaluation order and reuse of an existing `apply` argument list. An IR
control-flow check covers 38 emitted constructor sites, proves the success
branch dominates pointer uses, and rejects deliberately removed-branch and
failure-store mutants. The runtime failpoints cover representative constructor
families; the IR check covers the other emitted paths in this fixture.

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
and assignment captured by a guard handler. Public API linkage checks the
umbrella-header exports.

The strict WASM import smoke compiles all 13 existing surfaces and verifies the
imports against both browser glue files, including the new constructor
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
to CPU 7 gave median cons times of 2,104,747,812 vs. 2,110,103,330 jiffies
(PR/base 1.003), and median vector-store loop times of 124,576,971 vs.
133,199,877 jiffies (PR/base 1.069). The `fill` function's 616 disassembled
instructions are identical in both binaries after normalizing addresses. This
probe finds no added instruction in the store loop, but the measured loop time
was about 7% higher; binary layout, cache, and background-load effects remain
possible explanations. It is not a whole-application performance claim.
