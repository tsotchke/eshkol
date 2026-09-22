# Checked native promotion and emergency transfer, version 1

This is the accepted interface for the coordinated checked-promotion workstream.
Implementation and acceptance evidence are separate. The transformer consumer
must not adopt an intermediate runtime/compiler revision. This document adds no
Eshkol source operation, provider authority, or serialized identity.

## Exact compiler-private C ABI

```c
int32_t eshkol_region_write_barrier_checked_v1(
    eshkol_tagged_value_t *out,
    const void *destination_owner,
    const eshkol_tagged_value_t *value);

/* noreturn, with a declaration valid from C and C++ */
void eshkol_runtime_emergency_raise_v1(int32_t condition);

/* Returns for every ordinary/noncanonical value, including a null pointer. */
void eshkol_runtime_emergency_rethrow_if_v1(
    const eshkol_tagged_value_t *value);
```

The checked barrier returns 0 after complete promotion or a proven no-promotion
case. Its failure results are:

| Result / emergency condition | Meaning |
|---:|---|
| 1 | Promotion target or temporary bookkeeping allocation failed. |
| 2 | Promotion encountered an unsupported or unprovable object layout. |
| 3 | Promotion size/span/capacity arithmetic overflowed. |
| 4 | Invalid promotion call/runtime state. |
| 5 | Ordinary vector/cons construction or guard-handler allocation failed. |

**5 is not a checked-barrier return code.** It is a separate static exception
identity selected only by the separately reviewed constructor/handler paths.
There are no additional `promotion_raise`, `allocation_raise`, or per-condition
rethrow entry points. An unknown emergency selector transfers the fixed invalid-
runtime-state condition 4; it never returns successfully or invents another
condition.

For every checked-barrier failure, output staging bytes, destination bytes,
original source objects, committed forwarding map/target, and committed escape
counts remain unchanged. Speculative arena allocations are scrubbed and retained
until their ordinary arena lifetime ends; that retained cost is measured. All
transaction temporaries are released before returning a failure status. No C++
exception may cross the C ABI. Inputs are valid live runtime values; this is not
an arbitrary-native-pointer validation service. Output and input may alias only
private staging storage, never the real destination or an input graph object.

The compiler emits the checked call and branches on its status before loading
staging output or storing into the destination. On failure it transfers the exact
condition after transaction cleanup. Immediate, root-owned and same/older-lifetime
values must use zero-allocation paths, including batch unwind of the fixed
emergency value. A failed promotion never publishes the original younger pointer.

## Emergency object lifetime and delivery

Five distinct process-lifetime objects have normal exception headers, static
bounded messages, existing `ESHKOL_EXCEPTION_ERROR` type, no irritants, and no
model, frame, source-pointer or caught-graph detail. Initialization uses no arena
or heap allocation. Header-to-payload offset and alignment are checked at compile
time. Runtime location/irritant mutation helpers leave these exact objects
unchanged without allocating. Copies, equal messages, tags, and ordinary exception
objects do not gain emergency identity. Concurrent exception handling is not
introduced by these immutable process-lifetime objects.

`eshkol_runtime_emergency_raise_v1` accepts exactly conditions 1 through 5 and
selects the corresponding object. It sets both current exception and raised-value
state consistently and invokes the existing exception transfer machinery. No
exception wrapper, message, irritant, handler frame, registration, or promotion
scratch is allocated by the selector. Its callers must have destroyed all C++
transaction temporaries before transfer; `longjmp` cannot bypass their cleanup.

`eshkol_runtime_emergency_rethrow_if_v1` recognizes only the exact canonical
HEAP_PTR tagged identity of one of these five payloads. For a recognized identity
it transfers that same object and value without allocation. It returns normally
for every other value, including copies and noncanonical tags. Generated explicit
`raise` evaluates its operand exactly once, spills it, and calls this helper before
ordinary wrapper/message construction or replacement of the emergency state.
Ordinary `raise` semantics remain unchanged. Existing-exception/fallthrough paths
must preserve the same reserved identity and avoid wrapping it.

Existing dynamic-wind and promise ordering remains in force. This does not make
arbitrary user callbacks, outer cleanup, or rich error construction allocation
free. The bounded no-allocation recovery proof concerns the demonstrated nearest
preinstalled fixture handlers, whose cleanup has no callbacks or new regions;
applying it to the downstream P1 guard remains a separate acceptance obligation.
Emergency unwind must detect that its kept value is already root-owned
before allocating batch scratch, otherwise repeated allocation failure could
recursively transfer instead of reaching the handler.

## Constructor and handler boundary

The separate serial constructor scope is generic `vector`, `make-vector`, cons
cells/list-cell construction, and `guard` handler push. It adds no fallback and
must preserve operand evaluation order. Every admitted null allocation branches
to condition 5 before the first initialization store, dereference, or publication.
Constructor size/overflow admission keeps its existing semantics; other allocation
families are not silently included in this scope.

The existing void `eshkol_push_exception_handler` ABI remains unchanged. If its
frame allocation fails, it transfers condition 5 before publishing the new frame;
control cannot return to generated `setjmp` or the guarded body. The previous
handler remains installed. With no enclosing handler, the existing uncaught-
exception policy applies. Free-list success and ordinary handler ordering remain
unchanged. There is no fallible published-but-uninitialized handler frame.

Condition 5 infrastructure may be defined alongside conditions 1–4. Constructor
null/push callsites may be changed only after independent review and P2 unwind
readiness; adding the fifth static object does not prove any constructor fixed.

## Escaping continuation allocation owner

The additional compiler-private C ABI is `arena_t* eshkol_root_arena_v1(void)`.
It runs the existing once initializer and returns its captured, stable process
root arena. An unavailable root transfers condition 4 before any null owner is
returned. The existing once-initialization failure policy is preserved: repeated
calls transfer the same condition without retrying root creation. This accessor
is separate from the three checked-promotion/emergency functions above.

Only the existing nonlocal `call/cc` allocation branch uses this accessor, for
its state, closure and saved stack. The previous getter,
`get_global_arena_shared()`, returns the mutable allocation slot that an active
region redirects; it did not meet that branch's stated permanent-root intent.
Its semantics and all other callers remain unchanged. Local-only continuations
continue to allocate through the current arena. Captured-frame region pinning
also remains unchanged and deliberately retains those arenas. No generic pinned
region exemption or traversal of raw continuation state has been added.

This correction does not extend condition 5 to continuation allocations.
Existing state/closure allocation failures still log and return null; snapshot
allocation or unavailable stack geometry can still leave escape-only state.
Those paths are outside the new recoverable constructor guarantee. The new
root-initialization refusal seam has its own fresh-process failure/repeat test;
no universal continuation allocation-recovery claim follows from it.

## Required integration order and evidence

P0 establishes the single transactional promotion engine. P1 adds checked scalar
codegen, these fixed emergency identities, exact rethrow, and safe scalar adapters.
P2 migrates every remaining forwarding-map writer, including prewrite vector-copy
staging and per-successful-level unwind/handle retirement, and retires the old
postwrite range ABI. P3 proves linkage, full caller inventory and supported upstream
behavior. Only the complete reviewed union can become a new transformer pin.
Constructor/handler null-failure evidence remains an additional serial gate.

Tests must cover every target and bookkeeping allocation failure, unchanged
publication and old forwarding state, cycles/shared tails, failure/retry and
legacy/checked caller interleaving, exactly one emergency transfer under continued
allocation failure, metadata mutation rejection, exact catch/clear/rethrow identity,
ordinary raise controls, and fresh optimized/AOT/JIT linkage. Measure speculative
retention and successful fixed initialization separately from no-promotion reuse.
No local compatibility run alone establishes supported-lane or downstream P1
acceptance.

## Implemented layout and alias limits

`runtime_regions.cpp` checks regional headers and declared payload spans against
used arena bytes before copying or traversing them. Size arithmetic is checked.
This requires valid live runtime inputs: an arena records used block extents, not
every allocation boundary, and nonregional native pointers are not a validated
address space. These checks do not make arbitrary native pointers safe to read.

The implemented deep walks cover cons cells, vectors/records, multiple values,
hash tables, numeric and tagged dual tensors, exceptions, closures and their
expression graphs/captures, primitives, substitutions, facts, knowledge bases,
factor graphs, workspaces, promises, big rationals, and supported Taylor
coefficients. Both produced closure subtypes, including zero-capture
`CALLABLE_SUBTYPE_LAMBDA_SEXPR`, use the closure walk. Headerless dual/complex
payloads have fixed-size raw copies. Tensor admission accepts the legacy 32-byte
numeric layout and the current 40-byte layout with a declared dtype; a partial
dtype field or unknown dtype is rejected.

Known self-contained leaf payloads have size checks. Parameters, DNC and SDNC use
private producer-owned layout queries rather than duplicate runtime structs.
Their native malloc/calloc storage is not reinterpreted as regional storage.
A regional AD node is admitted as a leaf only when every supported input,
tensor-value, tensor-gradient, saved-tensor and shape pointer is null. General
AD graphs, regional continuation objects, linear HANDLE/BUFFER/STREAM/EVENT values,
regional ports, undeclared subtypes and unproved layouts return status 2. Stable
values that need no promotion retain their normal fast path. Supporting an object
family here does not establish exhaustive success for every graph it can contain.

Raw forwarding entries record the source base, byte extent and whether the entry
is a header-backed object. The accepted bounded alias policy is:

- Repeated same-base raw requests reuse one destination if the requested extent
  fits the already-copied extent. A full tensor buffer encountered before its
  same-base prefix therefore preserves shared storage.
- A later larger extent at that base returns status 2. Known overlaps at different
  bases, including interior views, also return status 2; mixed raw/object entries
  cannot silently share a pointer-only cache entry. The comparison includes
  previously committed entries and the current transaction's candidate entries.
- A nonnull zero-length raw pointer gets a stable one-byte destination when no
  reusable entry exists. A zero-length request at a different base strictly
  inside a known nonempty raw extent is rejected; a same-base zero-length prefix
  reuses that extent, and its one-past-end address is outside the half-open span. A previously
  copied zero-length entry cannot later satisfy a larger request.

This intentionally permits encounter-order-dependent admission: a larger view
first can prove a later prefix safe, while the reverse cannot extend a committed
copy. There is no general allocation-span discovery, interior-offset remapping,
copy-on-growth, or alias-breaking fallback. Only spans encountered by this engine
are known. Unsupported overlap must be handled through the exact condition-2
path; it is not successful tensor-view promotion. The 16-case tensor alias fixture
covers both encounter orders, same-base/interior views, zero/nonzero short views,
and fresh/prior-commit states: four cases succeed and twelve return status 2,
preserving source, output, old map/target/count and prior canonical identity.
A separate zero-length endpoint case succeeds.

The runtime-private batch copy stages all roots before publishing a range and
supports ordinary source/destination overlap. All four tagged vector/dual-tensor
copy channels use it. Scalar legacy deep-promotion adapters share the checked
engine and transfer on failure. The old postwrite range-barrier symbol is retired;
old generated artifacts must be rebuilt with the matching compiler/runtime.
Forwarding target changes replace the cache only after success. Failure preserves
its map and target identity; retained speculative target bytes are a separate cost.

Successful promotion is not a constant-time extension of the existing cache.
For a target with M committed forwarding entries, creating the candidate map and
retiring the old map each perform work proportional to M. Each new source span
is compared with the known entries: N new spans can require O(NM + N²) overlap
comparisons. Interior-write validation also scans the transaction's new-span
ledger. Zero-allocation fast paths avoid this transaction work but still perform
region-ownership checks and, for batches, scan/copy the values. No throughput,
latency or scaling benchmark is claimed for this implementation; correctness,
the measured retention examples and the stated fast-path allocation counts are
separate from performance acceptance.

## Retention and allocation-observation bounds

An unsuccessful transaction zeros only its recorded unpublished target spans and
releases native scratch before returning. It does not rewind the target arena or
scan arbitrary raw roots. Aligned padding is not claimed as scrubbed payload.
In the fixed-target experiment, 1,024 failed attempts retained 81,920 arena bytes
(80 per attempt), with no increase in reserved bytes or block count. The fixture
prezeros padding. This measurement demonstrates retained failure cost, not flat
process-lifetime memory under repeated failure. Reported per-category native peaks
are not a simultaneous aggregate peak.

Parameter value stacks remain native malloc storage. Constructor, push and set
promote their stored values to root lifetime; converter assignment now explicitly
does the same even when the parameter control itself is regional. This makes a
valid control's leaf copy free of younger regional edges. Successfully promoted
converter graphs remain in the root destination until that arena's ordinary
lifetime ends, even after converter replacement or destruction of the original
parameter region. The measured two-slot/shared-string graph copied 63 bytes and
used 64 root-arena bytes; a second store of that same forwarded graph added zero.
Distinct fresh graphs may accumulate. Parameter-control/stack allocation failures
and the existing realloc warning/drop behavior are separate, unchanged cases;
a realloc failure after successful promotion can retain that promoted graph.
This work does not redesign native stack ownership or later mutation through
aliases of copied parameter controls.

The deterministic engine hooks count five allocation sites: forwarding map,
worklist, span ledger, root scratch, and target arena. A selected site's failure
persists through subsequent attempts until reset. This is synthetic injection,
not observed host OOM. Separate bounded arenas exercise real allocator-null
returns with fixed capacity.

The allocation-denial probe starts **after** the checked call has returned and
transaction temporaries have been released. Until arrival at the nearest
preinstalled handler, it observes direct linked calls to `malloc`, `calloc`,
`realloc`, `aligned_alloc`, `posix_memalign`, scalar `new` and array `new`, and
separately counts `__cxa_allocate_exception`. All observed counts are zero in
that interval. GNU linker wrapping does not intercept hidden/internal allocations
inside shared libc/libstdc++, nor every allocator API such as aligned C++ new.
The transaction's earlier C++ throw machinery lies outside the interval. These
results do not establish whole-process allocation denial, allocator-independent
C++ exception handling, or allocation-free arbitrary dynamic-wind callbacks.

## Concrete constructor coverage

The executed AOT fixture catches seven condition-5 failures: handler-frame malloc,
explicit vector, make-vector, explicit cons, acons, list*, and an indirect
variadic closure's rest-list construction. Operand-order checks are part of those
cases. A separate positive control confirms ordinary `apply` reuses its existing
rest list, leaves the armed cons failure unconsumed, and executes the callee once.
It is not the AD-gradient spread-rest lowering.

The fixture's IR checker checks 23 admitted allocator sites and rejects removed-
branch and failure-store mutants. Its four explicit exclusions are vector
allocator calls in `__eshkol_arith_add`, `__eshkol_arith_sub`,
`__eshkol_arith_mul`, and `__eshkol_arith_div`; general numeric allocation remains
outside this constructor scope. Headerless cons lowering and AD-gradient
spread-rest lowering have source-level null checks but no executed injected
failure in this fixture. Static site counts are not executed-path counts. Other
allocation families do not inherit an all-allocation guarantee from condition 5.

## Reproducing the focused gates

The supported environment used here is Ubuntu 22.04, Linux x86-64, Clang/LLVM
**21.1.8**, Ninja, and the project's ordinary development dependencies. Confirm
`/usr/bin/llvm-config-21 --version` reports that exact patch version; the CMake
major-version requirement alone does not pin it. The coordinator used container
image `eshkol-checked-promotion-llvm21:20260922`, with the source mounted read-only
at `/source` and one owned writable build directory at `/build`. The image recipe,
cache, tool versions and final artifact hashes belong to the evidence manifest;
the image tag alone is not an immutable toolchain identity.

Inside that environment, configure a fresh release build and run all eleven
focused gates (seven native, two generated AOT executables and two IR checkers):

```sh
test "$(/usr/bin/llvm-config-21 --version)" = 21.1.8
cmake -S /source -B /build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DLLVM_CONFIG_EXECUTABLE=/usr/bin/llvm-config-21 \
  -DESHKOL_REQUIRED_LLVM_MAJOR=21 \
  -DESHKOL_BUILD_TESTS=ON -DESHKOL_PROMOTION_TESTING=ON \
  -DESHKOL_BUILD_AGENT_FFI=OFF
cmake --build /build --target checked-promotion-tests -j4
ctest --test-dir /build -L checked-promotion --output-on-failure
```

`ESHKOL_PROMOTION_TESTING` defaults OFF. Its ON mode requires tests enabled and
64-bit Linux with GNU/Clang and a `--wrap`-capable linker; unavailable full-suite
coverage is a configuration error. Only the core and hosted runtime object
targets and the focused fixtures receive the hook definition. All native and AOT
execution tests set `ESHKOL_ARENA_POISON=1`. The AOT fixtures are generated by the
just-built compiler at `-O 2`; their emitted IR feeds the dominance checks and
negative mutants. Missing fixtures or failed gates are not optional green skips.

The focused AOT fixtures use `--no-stdlib`. Broader source-library regressions
also require `cmake --build /build --target eshkol-repl stdlib` and
`ESHKOL_PATH=/source/lib`; run them with `/source` as the working directory and
`ESHKOL_JIT_CACHE=0`, `ESHKOL_ARENA_POISON=1`, and
`ESHKOL_CXX_COMPILER=/usr/bin/clang++-21`. In particular the closure escape fixture
needs that source-library search path for its AOT run. Initial harness environment,
working-directory and log-scanner corrections are evidence-runner corrections,
not runtime fixes or additional passed product cases. Complete broader regression
results remain in the coordinator's final disposition.

For the measured sanitizer lane, mount a **separate empty** owned directory at
`/build`, use the same toolchain and dependencies, and run the seven native gates:

```sh
cmake -S /source -B /build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DLLVM_CONFIG_EXECUTABLE=/usr/bin/llvm-config-21 \
  -DESHKOL_REQUIRED_LLVM_MAJOR=21 \
  -DESHKOL_BUILD_TESTS=ON -DESHKOL_PROMOTION_TESTING=ON \
  -DESHKOL_BUILD_AGENT_FFI=OFF \
  -DESHKOL_ENABLE_ASAN=ON -DESHKOL_ENABLE_UBSAN=ON
cmake --build /build -j4 --target \
  runtime_promotion_transaction_test runtime_promotion_unwind_test \
  runtime_emergency_semantics_test checked_promotion_external_callers_test \
  runtime_promotion_noalloc_transfer_test runtime_promotion_layout_lifetime_test \
  runtime_root_arena_failure_test
ctest --test-dir /build --output-on-failure \
  -R '^(runtime_promotion_(transaction|unwind|noalloc_transfer|layout_lifetime)_test|runtime_emergency_semantics_test|checked_promotion_external_callers_test|runtime_root_arena_failure_test)$'
```

These targets inherit the project's directory sanitizer compile/link flags and
normal system-library dependencies. This seven-test measurement does not claim
sanitizer instrumentation of all generated AOT instructions or the broader suite.

## Evidence status at handoff

The earlier continuation regression is resolved for its existing producer. The
supported production-OFF closure runs
`tests/continuations/region_capture_resume.esk` successfully as optimized AOT and
full-file JIT with arena poisoning, and checks that escaping state, closure and
stack snapshot allocation use the immutable root getter. The local-only control
continues to use the current arena. This evidence does not add generic support for
regional continuation objects or change the continuation-allocation failure limits
stated above.

| Measurement | Recorded result and scope |
|---|---|
| Native transaction matrix | 47 synthetic allocation-failure prefixes; actual bounded-arena budgets 0 and 80; cycles/sharing, failure/retry, old/new adapter reuse and allocation-free fast paths. |
| External callers | 21 parameter and 24 tagged/dual batch failures; unchanged publication, cleanup, retry, overlap and region-exit lifetime. |
| Layout lifetime | Produced closure expression/capture lifetime, tensor dual payload lifetime, and the 16-case raw alias admission matrix described above. |
| Repeated failure | 1,024 failures retained 81,920 arena bytes in the measured graph; zero reserved/block increase. |
| Constructor AOT | Seven injected failures plus existing-list apply control; 23 admitted IR sites and four named arithmetic exclusions. |
| Supported Ubuntu 22.04 / LLVM 21.1.8 | 11/11 focused gates passed, including the fresh-process root-initialization refusal/repeat test. The preserved production-OFF closure and continuation AOT/JIT checks also passed against the source hashes in `build-promotion-production-supported/evidence/source-inputs.sha256`. |
| Supported ASan + UBSan | Seven native gates passed on the current final engine; this is the explicit native-only sanitizer scope above. |
| Local transformer compatibility | The preserved local consumer log reports 749 passing checks, but it predates the final root-getter/test/comment refresh and is not final-source evidence. Rerun it before using 749 as a final-source integration claim; it does not adopt a new pin or establish downstream P1 acceptance. |

The implementation is commit `714d20fe` on `codex/wave3-checked-promotion`.
It is an integration candidate based on `90cbd713`; it is not a release or a
downstream pin. The final condition-2 AOT fixture exercises catch, unchanged destination,
exact-identity rethrow, preserved forwarding state and scratch cleanup, followed
by a supported retry that survives region exit. The final focused logs are the
eleven-test supported run and seven-test native sanitizer run; earlier ten-test
and six-test checkpoints are superseded. The preserved source and artifact
manifests match the committed production inputs and outputs. Downstream pin approval
remains owned by the integration coordinator; no downstream adoption is asserted here.
