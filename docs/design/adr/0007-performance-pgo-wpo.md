# ADR 0007: PGO, Whole-Program Optimization, and Staged Training Throughput

Status: Proposed

Date: 2026-07-09

Audience: Eshkol compiler, runtime, release, and performance maintainers.

## Decision

Eshkol will treat performance as three connected but separately measurable
compiler products:

1. **Native product PGO** optimizes the C/C++ compiler, tools, and runtime
   archive using a checked-in workload manifest and a reproducible
   generate/train/merge/use workflow.
2. **Application IR PGO** optionally optimizes one emitted Eshkol program or
   staged-kernel specialization. It has a different profile and cache key from
   native product PGO.
3. **Closed-world staged optimization** specializes a typed tensor/AD graph,
   emits a static primal and reverse schedule, plans scratch memory, links only
   a small bitcode runtime surface, and then runs LLVM 21's LTO pipeline.

PGO is the last multiplier in this sequence, not a substitute for dense AD,
static shapes, devirtualization, fusion, or allocation removal. A profile may
change layout, inlining, and residual branch decisions; it may never select a
numerically different AD fallback.

Every performance change is guarded in two ways:

- deterministic structural assertions prove the intended execution shape;
- a statistically evaluated, same-host A/B suite detects throughput, compile
  time, memory, and code-size regressions.

Profiles used for a release fail closed on source, toolchain, target, or IR
identity mismatch. Performance training inputs and performance gate inputs are
disjoint. Floating-point reassociation and other fast-math changes remain off
unless an explicit math-mode contract is part of the specialization key.

## Context and current constraints

### The existing PGO switch is a useful probe, not a workflow

The root build has `OFF`, `generate`, and `use` modes and describes a manual
four-step process ([`CMakeLists.txt:247-265`](../../../CMakeLists.txt#L247-L265)).
Generation adds Clang instrumentation globally, while use consumes one profile
path ([`CMakeLists.txt:267-290`](../../../CMakeLists.txt#L267-L290)). That proves
the toolchain can perform instrumentation PGO, but it leaves several release
properties unspecified:

- there is no executable training manifest, merge target, profile validation,
  provenance record, baseline build, holdout comparison, or artifact promotion;
- a requested PGO build on a non-Clang compiler is only warned and then ignored
  ([`CMakeLists.txt:267-270`](../../../CMakeLists.txt#L267-L270));
- use mode suppresses both stale-profile and unprofiled-function diagnostics
  ([`CMakeLists.txt:283-289`](../../../CMakeLists.txt#L283-L289));
- `%p` distinguishes processes but not instrumented binaries, corpus cases, or
  run identities ([`CMakeLists.txt:271-275`](../../../CMakeLists.txt#L271-L275));
- the flags affect sources built by CMake, not LLVM IR later emitted for an
  arbitrary Eshkol application.

The last distinction is architectural. Eshkol's build separates the runtime
archive linked into generated programs from the larger compiler/tool archive
([`CMakeLists.txt:1518-1526`](../../../CMakeLists.txt#L1518-L1526),
[`CMakeLists.txt:1718-1739`](../../../CMakeLists.txt#L1718-L1739)). Generated
executables are then linked by a manually assembled driver invocation beginning
with a native object, not by the CMake link that received PGO flags
([`lib/backend/llvm_codegen.cpp:37197-37224`](../../../lib/backend/llvm_codegen.cpp#L37197-L37224)).
An instrumented runtime archive therefore also needs the profile-runtime link
flag propagated into every generated-executable link path.

### Generated code has an independent optimization plane

The backend has a process-global optimization level whose initial value is O0,
and maps it independently to LLVM IR and machine-code optimization levels
([`lib/backend/llvm_codegen.cpp:179-214`](../../../lib/backend/llvm_codegen.cpp#L179-L214)).
The CLI also initializes `-O` to zero and only calls the backend setter for a
positive value ([`exe/eshkol-run.cpp:3705-3718`](../../../exe/eshkol-run.cpp#L3705-L3718),
[`exe/eshkol-run.cpp:4546-4551`](../../../exe/eshkol-run.cpp#L4546-L4551)).
Consequently, a CMake `Release` build does not imply optimized Eshkol output;
all performance corpus entries must explicitly compile generated code at O3.

At O1 through O3, `optimizeModule` constructs LLVM's per-module default
pipeline. O0 runs only a small cleanup sequence (or O1 for WebAssembly)
([`lib/backend/llvm_codegen.cpp:327-377`](../../../lib/backend/llvm_codegen.cpp#L327-L377)).
The pipeline runs immediately before one module is emitted as an object
([`lib/backend/llvm_codegen.cpp:37029-37041`](../../../lib/backend/llvm_codegen.cpp#L37029-L37041)).
This gives LLVM visibility into the generated module, but not into the native
runtime archive or the object-form standard library.

There is already a path to emit bitcode
([`lib/backend/llvm_codegen.cpp:37172-37194`](../../../lib/backend/llvm_codegen.cpp#L37172-L37194)),
and the build installs both `stdlib.o` and `stdlib.bc`
([`CMakeLists.txt:1856-1860`](../../../CMakeLists.txt#L1856-L1860),
[`CMakeLists.txt:1893-1900`](../../../CMakeLists.txt#L1893-L1900)). Those are the
seeds for WPO, but the ordinary AOT path still compiles a user object and links
it with native objects and archives
([`exe/eshkol-run.cpp:4710-4733`](../../../exe/eshkol-run.cpp#L4710-L4733),
[`exe/eshkol-run.cpp:4800-4825`](../../../exe/eshkol-run.cpp#L4800-L4825)).

### The staged AD work changes the asymptotic problem

The AD handoff requires one primal, one reverse pass, all parameter gradients,
dense tensor adjoints, resident buffers, and fixed scratch for a scalar loss
([`docs/design/AD_STAGED_KERNEL_HANDOFF.md:369-408`](../AD_STAGED_KERNEL_HANDOFF.md#L369-L408)).
Today, tensor/vector gradients replay the function and allocate a tape per input
component ([`docs/design/AD_STAGED_KERNEL_HANDOFF.md:202-244`](../AD_STAGED_KERNEL_HANDOFF.md#L202-L244)),
while dense matmul, reductions, and elementwise AD are commonly scalarized
([`docs/design/AD_STAGED_KERNEL_HANDOFF.md:288-330`](../AD_STAGED_KERNEL_HANDOFF.md#L288-L330)).
No amount of branch profiling can turn those algorithms into a training
runtime.

The handoff also establishes the right ABI direction—descriptors and raw
pointers, not tagged structs by value—and a narrow first subset with static
f64 shapes ([`docs/design/AD_STAGED_KERNEL_HANDOFF.md:1059-1190`](../AD_STAGED_KERNEL_HANDOFF.md#L1059-L1190)).
Its specialization key and runtime shape guard are defined at
[`docs/design/AD_STAGED_KERNEL_HANDOFF.md:1227-1277`](../AD_STAGED_KERNEL_HANDOFF.md#L1227-L1277),
and its memory plan explicitly forbids hot tensor allocation
([`docs/design/AD_STAGED_KERNEL_HANDOFF.md:1279-1330`](../AD_STAGED_KERNEL_HANDOFF.md#L1279-L1330)).
This ADR supplies the optimization and measurement architecture around that
contract.

### Existing benchmarks are not a gate

The repository has useful operation benchmarks, but the current documentation
states that there is no unified runner and shows ad hoc shell loops
([`docs/breakdown/BENCHMARKING.md:176-195`](../../breakdown/BENCHMARKING.md#L176-L195)).
`time-it` provides warmup and per-iteration timing
([`docs/breakdown/BENCHMARKING.md:199-210`](../../breakdown/BENCHMARKING.md#L199-L210)),
but a release gate also needs raw samples, machine metadata, paired baselines,
confidence intervals, correctness checks, and structural compiler counters.

## Goals

- Make native PGO a single, reproducible command sequence suitable for CI and
  release packaging on every supported release target.
- Make stale, empty, contaminated, or incompatible profiles observable and
  non-promotable.
- Give ordinary closed-world AOT programs and staged kernels progressively more
  whole-program visibility without changing REPL, dynamic loading, weak
  override, or FFI semantics.
- Deliver compile-once/run-many `value + grad + update` kernels whose steady
  state is limited primarily by dense math and memory bandwidth rather than
  tagged dispatch, tape construction, or arena allocation.
- Detect performance regressions with repeatable relative measurements while
  keeping exact functional and AD invariants as hard gates.
- Record enough artifacts to reproduce every profile and gate decision.

## Non-goals

- PGO will not be mandatory for development builds or for every user program.
- The first implementation will not require online tiering or swap optimized
  code while a training job is running.
- WPO will not assume the REPL, `eval`, arbitrary FFI callbacks, or dynamically
  loaded modules are closed world.
- This ADR does not make GPU kernel compilation part of the CPU PGO profile.
  GPU performance needs its own device- and driver-specific lane.
- This ADR does not enable global fast math, change f64 to f32, or relax the
  exact-AD/no-hidden-finite-difference requirement.
- PGO is not used to choose between exact AD and a fallback. Unsupported staged
  operations remain explicit errors.

## Architecture overview

```text
                native C/C++ sources
                         |
        instrumented compiler + runtime archive
                         |
        native training manifest (compile + execute)
                         |
                   native.profdata
                         |
          PGO-use compiler + runtime release

Eshkol source + static signature + static configuration
                         |
        typed closed-world graph and AD transform
                         |
      primal schedule + reverse schedule + memory plan
                         |
       specialized LLVM IR + runtime bitcode capsule
                    /             \
          unprofiled O3       optional IR instrumentation
                    |              |
                    |       kernel-<key>.profdata
                    \              /
                    LLVM 21 Full LTO
                         |
                  staged kernel artifact
                         |
          structural gate, then holdout A/B gate
```

The profiles are deliberately separate:

| Plane | Optimizes | Training unit | Profile artifact | Reuse boundary |
|---|---|---|---|---|
| Native product PGO | `eshkol-run`, compiler libraries, native runtime helpers | repository corpus manifest | `native.profdata` | exact native build key |
| Application IR PGO | one ordinary AOT program | that program's representative runs | `module-<key>.profdata` | exact pre-instrumentation IR key |
| Staged-kernel IR PGO | one static shape/config family | representative training steps | `kernel-<key>.profdata` | exact staged kernel key |

Combining all three into one anonymous `.profdata` would make compatibility,
coverage, weighting, and cache invalidation unverifiable.

## Artifact identity and provenance

### Native build key

`NativeBuildKey` is a digest of:

- the clean source tree revision and submodule revisions;
- CMake cache options that affect code generation or ABI;
- build type and sanitizer state;
- C and C++ compiler executable, version, and resource directory;
- `llvm-profdata` executable and version;
- LLVM major version;
- target triple, deployment target, standard library, and CPU baseline;
- native PGO instrumentation schema and training-manifest version.

The profile directory contains `native.profdata` plus a JSON provenance file
with that key, the raw-file manifest and hashes, command outcomes, deterministic
seeds, corpus weights, timestamps, and the producing host metadata. The merged
profile is written to a temporary file, validated, fsynced, and atomically
renamed. A profile without its matching provenance file is unusable.

### Staged kernel key

The handoff's shape key is extended to form `KernelKey`:

```text
transitive source/function digest
compiler build id and staged-IR schema version
LLVM version and optimization-pipeline revision
target triple, CPU name, and target feature string
optimization level and floating-point math mode
staged ABI and supported-op-set versions
AD transform and checkpoint policy versions
dtype, rank, dims, strides, alignment, and mutability of every buffer
static configuration values and shape-specialized control-flow values
```

The CPU and feature string matter because the native object path selects host
CPU features when the module target is the host target
([`lib/backend/llvm_codegen.cpp:36921-36937`](../../../lib/backend/llvm_codegen.cpp#L36921-L36937),
[`lib/backend/llvm_codegen.cpp:37009-37022`](../../../lib/backend/llvm_codegen.cpp#L37009-L37022)).

`KernelProfileKey` additionally hashes the canonical, pre-instrumentation LLVM
IR. `OptimizedArtifactKey` hashes `KernelKey`, `KernelProfileKey`, the profile
digest, and the final pipeline revision. This prevents an old profile from
being consumed merely because a human-readable function name stayed the same.

## Native product PGO workflow

### CMake contract

Keep the existing public options:

```text
ESHKOL_PGO=OFF|generate|use
ESHKOL_PGO_RAW_DIR=<path>
ESHKOL_PGO_PROFILE=<native.profdata>
```

Add strict workflow behavior:

- Any explicit `generate` or `use` request on an unsupported compiler is a
  configure error, not a warning followed by a non-PGO build.
- Instrumentation flags are attached through target-scoped interface targets to
  the native products in the release profile, rather than accidentally
  instrumenting every test utility.
- The default raw pattern is a run-specific directory containing
  `%m-%p.profraw`; the training runner may override it with
  `LLVM_PROFILE_FILE`.
- Both generated-executable link implementations receive
  `-fprofile-instr-generate` when they link an instrumented `eshkol-runtime`.
  This is required for compiler-driven profile-runtime linkage. The common
  object/stdlib link is assembled independently at
  [`exe/eshkol-run.cpp:4800-4825`](../../../exe/eshkol-run.cpp#L4800-L4825),
  while the direct link is assembled in the backend as cited above.
- Use mode treats `profile-instr-out-of-date` as fatal. Unprofiled cold code is
  reported and budgeted rather than hidden globally; required hot symbols must
  be present with nonzero counts.
- PGO generate/use cannot be combined with sanitizers, coverage, or a different
  target CPU baseline in the release workflow.

The build exposes orchestration targets backed by one cross-platform runner:

```text
eshkol-pgo-train      run the versioned training manifest
eshkol-pgo-merge      merge weighted raw profiles and write provenance
eshkol-pgo-verify     validate identity, counters, symbols, and compatibility
eshkol-perf-gate      compare non-PGO and PGO-use holdout artifacts
```

CMake remains responsible for building; the runner owns process execution,
weights, profile files, validation, and JSON results. Encoding the corpus as a
long CMake command would make failure handling and provenance needlessly
platform-specific.

### Required end-to-end sequence

All three build directories are clean and immutable after their build step:

```text
build/perf-baseline   Release, ESHKOL_PGO=OFF
build/pgo-generate    Release, ESHKOL_PGO=generate
build/pgo-use         Release, ESHKOL_PGO=use
```

The release sequence is:

1. **Preflight.** Require a clean tree, verify Clang/LLVM/`llvm-profdata`
   versions, compute `NativeBuildKey`, reserve a unique run directory, and
   materialize the resolved corpus manifest.
2. **Baseline.** Configure and build non-PGO Release. Run correctness tests and
   build the holdout benchmark artifacts with generated Eshkol code explicitly
   at `-O3`.
3. **Generate.** Configure and build a separate instrumented Release tree. Run
   its correctness smoke tests before accepting it as a profile producer.
4. **Discard bootstrap data.** The build invokes `eshkol-run` to precompile the
   standard library
   ([`CMakeLists.txt:1882-1890`](../../../CMakeLists.txt#L1882-L1890)); delete all
   raw profiles produced during configure/build so they do not silently weight
   the training corpus.
5. **Train.** Execute each manifest entry in a fresh process with a deterministic
   seed, case-specific raw directory, timeout, expected exit status, and output
   checksum. Training includes compiler invocations and execution of binaries
   linked with the instrumented runtime.
6. **Merge.** Apply explicit manifest weights with `llvm-profdata merge`; never
   let a test family dominate merely because it launches more processes.
7. **Verify.** Reject zero-byte raw files, failed cases, malformed data, missing
   expected compiler/runtime symbols, zero total counts, toolchain mismatch, or
   a manifest/provenance mismatch. Produce a human-readable `llvm-profdata show`
   summary alongside machine-readable JSON.
8. **Use.** Configure a new build against the atomically published profile.
   Warnings that establish staleness are errors. Build and run the entire
   correctness suite.
9. **Gate.** Run baseline and PGO-use holdout binaries in paired, interleaved
   order on the same quiescent machine. Apply the gate defined below.
10. **Promote.** Publish binaries, profile, provenance, resolved manifests, raw
    sample JSON, and the comparison report as one release unit. Never publish a
    PGO binary without the report that qualified it.

Any failed step invalidates the run; an older profile is not silently reused.

### Training corpus design

The native corpus is a checked-in declarative manifest. Each entry supplies:

```text
id, family, command, environment, deterministic seed
timeout, expected exit/checksum, repetitions, profile weight
required platform/features, expected profiled symbol group
```

The initial families are:

- frontend parsing, module loading, type/ownership/escape analysis, and error
  reporting;
- O3 native code generation and linking for small, medium, and large modules;
- REPL/JIT compilation and execution, kept separate from AOT weights;
- scalar R7RS control flow, closures, lists, exact numbers, and arena use;
- dense tensor elementwise, reduction, BLAS matmul, convolution, and
  normalization runtime paths;
- exact scalar/Taylor and dense reverse AD;
- staged value/grad and value/grad/update training steps once available.

Workloads are weighted by expected product usage, not historical test count.
Very large stress tests, GPU tests, network tests, and nondeterministic fuzzing
do not train the release profile. They remain correctness or specialized
performance lanes.

## Application and staged-kernel IR PGO

Native PGO cannot profile the control-flow graph of generated Scheme functions.
Application PGO is therefore an explicit compiler feature with mutually
exclusive options such as:

```text
--codegen-pgo-generate <raw-dir>
--codegen-pgo-use <profile.profdata>
```

It obeys these rules:

1. It is supported only for O2/O3 native AOT at first. The perf and release
   workflows use O3.
2. `llvm::PassBuilder` is constructed with LLVM 21 instrumentation-PGO options,
   so generation/use happens in the correct positions inside the default or LTO
   pipeline. PGO is not bolted on after object emission.
3. Generate mode adds the profile runtime to both native link paths and names
   raw files by module/kernel key and process.
4. Use mode validates `KernelProfileKey` before constructing the pipeline and
   treats function-hash mismatch as an error.
5. One profile belongs to one ordinary closed-world program or one staged
   specialization family. It is never treated as a universal Scheme profile.
6. Instrumented performance is never reported as product performance; the
   kernel is rebuilt in use mode before measurement.

For staged training, shape specialization and the AD transform happen before IR
instrumentation. The representative run executes enough complete steps to
exercise steady-state control, but correctness/error branches are not made hot
by repeatedly feeding invalid inputs. A profile may improve hot/cold splitting,
inlining, indirect-call promotion, and code layout. The structural gate must
still show no generic fallback, finite differences, or hot allocation both with
and without the profile.

## Whole-program optimization design

### Closed-world modes

WPO is opt-in for ordinary AOT and intrinsic to staged kernels. It is not used
for the REPL or a shared library whose callers/overrides are unknown.

| Mode | Scope | LLVM strategy | Compile-time policy |
|---|---|---|---|
| `module` | current generated module | existing per-module O3 pipeline | default ordinary AOT |
| `thin` | user modules + reachable stdlib/runtime bitcode | ThinLTO pre-link and link pipelines | ordinary closed-world AOT |
| `full` | one staged kernel + small runtime capsule | unified module and Full LTO | default staged kernel |

The compiler already batches a source program's ASTs into one generated module
([`exe/eshkol-run.cpp:4514-4585`](../../../exe/eshkol-run.cpp#L4514-L4585)). The
next visibility boundary is therefore stdlib/runtime code and semantic
information that currently disappears before LLVM.

### Closed-world roots

Before internalization, the linker constructs an explicit root set:

- `main`, the staged ABI entry, and explicitly exported Scheme functions;
- `extern` symbols, address-taken functions, FFI callbacks, reflection roots,
  and dynamic lookup registrations;
- exception, shutdown, and platform entry points required by the selected
  execution profile;
- `llvm.used`, `llvm.compiler.used`, and global constructors/destructors;
- weak/override definitions that have not been resolved to a final definition.

This is essential because ordinary definitions use external linkage outside
library mode ([`lib/backend/llvm_codegen.cpp:1030-1042`](../../../lib/backend/llvm_codegen.cpp#L1030-L1042),
[`lib/backend/llvm_codegen.cpp:4430-4447`](../../../lib/backend/llvm_codegen.cpp#L4430-L4447)).
It is also essential because one direct-link path force-loads the runtime to
preserve a parallel-worker constructor
([`lib/backend/llvm_codegen.cpp:37284-37318`](../../../lib/backend/llvm_codegen.cpp#L37284-L37318)).
WPO must model that constructor as a root, not accidentally delete it. A later
explicit registration ABI may remove this special case.

After final symbol resolution, non-roots receive internal/private linkage and
the pipeline runs global dead-code elimination. REPL hot-reload and stdlib weak
override behavior remain on their current non-closed path.

### Typed whole-program transformations

The highest-return work happens before generic LLVM optimization:

1. **Reachability and closure devirtualization.** Build a call graph across the
   transitive Scheme unit. Replace calls through closures with direct calls when
   the lambda identity and capture layout are known; remove closure allocation
   when the closure does not escape.
2. **Representation specialization.** Clone hot procedures for proven scalar,
   tensor, dtype, rank, shape, and static-argument combinations. Pass unboxed
   scalars or raw tensor pointers across specialized edges; keep a generic
   tagged entry only at a public boundary.
3. **Guard hoisting.** Validate dtype, rank, dimensions, strides, alignment, and
   scratch size once in the staged wrapper. The hot body receives proven facts
   and contains no repeated shape/type dispatch.
4. **Escape and region specialization.** Preserve ownership/escape facts in the
   typed IR so nonescaping closures, pairs, descriptors, and small temporaries
   can be scalar-replaced or stack/region allocated, and loop-invariant arena
   work can be hoisted. The driver already runs ownership and escape analyses
   before IR generation
   ([`exe/eshkol-run.cpp:4492-4511`](../../../exe/eshkol-run.cpp#L4492-L4511));
   WPO must make their results actionable across the closed call graph.
5. **Control specialization.** Fold static configuration branches and bounds;
   unroll only small profitable loops; lower larger fixed-trip loops with
   constant bounds and vectorization metadata.
6. **Effect-aware fusion.** Fuse compatible elementwise chains, broadcasts,
   reductions, adjoint chains, and optimizer epilogues. Do not cross observable
   mutation, exception, FFI, checkpoint, or numerically incompatible reduction
   boundaries.
7. **AD schedule generation.** Convert the differentiated graph into an
   explicit forward schedule plus reverse schedule before LLVM lowering. Dense
   tensor ops remain graph nodes, not scalar AD-node loops.
8. **Memory planning.** Compute saved-forward values and temporary lifetimes,
   assign aligned scratch offsets, and reuse nonoverlapping slots.

LLVM should receive the facts it cannot safely rediscover:

- `internal`/`private` linkage and precise function attributes;
- `nonnull`, alignment, dereferenceable byte counts, `readonly`/`writeonly`,
  `nocapture`, and `noalias` only where the staged ABI proves them;
- alias scopes distinguishing immutable inputs, parameters, gradients, optimizer
  state, outputs, and scratch;
- range metadata and `llvm.assume` derived from a checked wrapper;
- branch weights from compatible PGO data;
- loop trip counts, vectorization legality, and alignment;
- cold attributes for checked error exits.

Attributes are proof obligations, not hints. In particular, an in-place update
buffer must not be marked disjoint from an alias the ABI permits.

### Bitcode composition and LTO

The WPO path creates two build artifacts in addition to the existing native
archives:

- a target-specific stdlib bitcode bundle with export/override metadata;
- a small `eshkol-kernel-runtime.bc` capsule containing only hot helpers that a
  staged kernel may call.

The staged compiler links the specialized module and runtime capsule in memory,
resolves roots, internalizes, and runs LLVM 21's Full LTO pipeline. Vendor BLAS,
system math, GPU drivers, and large hosted facilities remain external. General
closed-world AOT uses ThinLTO so imported hot functions can specialize without
forcing all compiler/runtime code into one enormous module.

The current compile-only flow emits an object and then bitcode from the same
module ([`exe/eshkol-run.cpp:4646-4673`](../../../exe/eshkol-run.cpp#L4646-L4673)).
WPO instead has an explicit pre-link bitcode stage; it must not depend on
incidental mutation performed by an earlier object-emission pipeline.

### Prioritized WPO opportunities

| Priority | Opportunity | Why it matters | Required proof/gate |
|---|---|---|---|
| P0 | Make perf artifacts explicitly O3 | CMake Release currently does not set generated-code O3 | result metadata records O3; IR pipeline assertion |
| P0 | Staged dense AD and static reverse schedule | removes replay and scalar-node asymptotics | exact counters and gradient oracle |
| P0 | Scratch/resident-buffer plan | removes arena/tensor allocation from every step | zero post-warmup allocations; stable high-water mark |
| P1 | Closed-world root analysis and internalization | enables inlining, devirtualization, and DCE | export/FFI/constructor conformance tests |
| P1 | Shape/representation specialization | removes tags, shape checks, and generic dispatch | guarded wrapper plus mismatch tests |
| P1 | Escape/region specialization | removes nonescaping arena objects and hoists region work | lifetime, TCO, and allocation-count tests |
| P1 | Proven alias/alignment attributes | enables vectorization and load/store motion | ABI alias tests and LLVM verifier |
| P1 | Elementwise/adjoint/update fusion | removes memory traffic and dispatch | numerical comparison and bandwidth benchmark |
| P2 | Full LTO runtime capsule for staged kernels | crosses the generated/runtime boundary cheaply | symbol-root audit and code-size gate |
| P2 | ThinLTO stdlib/runtime import for AOT | specializes hot library calls without full monolith | compile-time and binary-size budgets |
| P2 | Native and kernel PGO | improves residual inlining/layout/branches | holdout benefit and no-regression gate |

Global fast math is not on this table. A future `training-relaxed` math mode may
permit contraction or reassociation, but it requires an explicit numerical
tolerance contract, its own cache key, and separate correctness baselines.

## Staged AD kernel performance architecture

### Compilation product

The staged API described by the handoff has a pointer/out-param wrapper and
counter/status surface
([`docs/design/AD_STAGED_KERNEL_HANDOFF.md:1080-1155`](../AD_STAGED_KERNEL_HANDOFF.md#L1080-L1155)).
Internally, compilation produces:

```text
StagedKernelPlan
  signature and runtime guard plan
  closed-world typed call graph
  primal tensor schedule
  exact reverse tensor schedule
  optional optimizer/update schedule
  saved-value/checkpoint policy
  scratch slot and resident-buffer plan
  external dense-library calls
  structural cost/counter expectations
  KernelKey and optional KernelProfileKey
```

The compatibility implementation may first record one dense tape node per
tensor graph operation, as required by the handoff. The optimized staged path
then erases the generic tape when the supported graph is static:

- the compiler emits adjoints in reverse topological order;
- static loops become forward/reverse loops with fixed bounds;
- data-dependent branch predicates needed by reverse mode occupy planned
  scratch bits/bytes;
- unsupported dynamic effects reject staging or select an explicit generic
  dense-tape artifact outside the training-grade lane.

This preserves exact reverse semantics while removing node allocation, opcode
dispatch, pointer chasing, and a runtime tape-capacity decision. It also makes
saved-value liveness available to the memory planner.

### What static shapes buy

For every shape specialization, the compiler can:

- constant-fold element counts, byte sizes, strides, broadcast axes, reduction
  extents, matmul `M/K/N`, and tape/scratch capacity;
- select one dense forward and backward implementation at compile time;
- guard once at the ABI boundary and remove checks inside the kernel;
- turn rank-generic indexing into affine pointer arithmetic;
- expose fixed trip counts and alignment to LLVM's loop vectorizers;
- precompute forward values that backward must retain;
- color temporary lifetimes into one fixed scratch buffer;
- keep parameter, gradient, optimizer-state, and input buffers resident across
  steps;
- compile common batch sizes as a small, explicitly dispatched family rather
  than carry a dynamic shape branch through every operation.

The shape mismatch remains a status error; there is no reinterpretation or
silent generic fallback.

### Dense operation and fusion policy

Large GEMMs and convolutions remain calls to the selected vendor/runtime dense
kernel. WPO should eliminate wrapper overhead around them, not replace a tuned
BLAS with scalarized LLVM loops. The compiler fuses operations where memory
traffic dominates:

- bias, activation, residual, scale, and elementwise loss chains;
- broadcast-reduction adjoints;
- activation adjoints and gradient accumulation;
- gradient clearing when it can be folded into first writes;
- SGD/Adam-style update epilogues over resident buffers.

Fusion preserves the selected math contract. Reductions retain a specified
order unless an explicit relaxed mode permits reassociation.

For a static MLP-like loss, the intended steady-state cost is:

```text
T_step = forward dense calls
       + fused forward pointwise/reductions
       + backward dense calls
       + fused adjoints/accumulations
       + optional fused optimizer update
       + O(1) wrapper/guard overhead
```

It is not `O(number_of_parameters)` primal evaluations or tape rebuilds. The
first serious staged benchmark's required counters—one compile, one primal and
reverse per step, dense matmul backward, fixed scratch, and stable arena bytes—
are already specified at
[`docs/design/AD_STAGED_KERNEL_HANDOFF.md:1550-1587`](../AD_STAGED_KERNEL_HANDOFF.md#L1550-L1587).

### Training-grade definition

“Training-grade” means all of the following on a qualified CPU target:

- compile once and execute at least 10,000 measured steps without recompilation;
- one primal and one reverse pass for each scalar-loss step;
- no hidden finite differences, generic fallback, or scalar AD nodes generated
  by supported dense operations;
- zero arena/tensor allocations after warmup and constant scratch/resident
  memory for a fixed key;
- numerical results within the declared exact/default floating contract and
  analytic/scalar-AD oracle tolerances;
- GEMM-dominated staged throughput at least 80% of a same-process native
  C++/vendor-BLAS reference implementing the same math and buffer policy;
- bandwidth-dominated fused kernels at least 75% of a same-process native
  loop/vector reference;
- compilation amortized below 1% of wall time for a 10,000-step run.

The efficiency thresholds are release-qualification targets, not noisy
per-commit absolute timers. The PR gate uses paired relative comparisons; the
reference-efficiency lane runs nightly and before release.

## Performance gate

### Result protocol

One runner compiles and executes all performance cases and writes versioned
JSON. Every sample includes:

```text
schema and benchmark id
source revision, build key, profile digest, kernel key
compiler/LLVM/BLAS versions and complete effective flags
OS, target triple, CPU model/features, core count, memory
affinity/governor/power/thermal metadata where available
warmup count, iteration count, randomized A/B order
raw wall/cpu samples and checksum/correctness outcome
compile, link, code size, peak RSS, arena, scratch, and AD counters
```

Human-readable tables are derived from JSON; they are never the only artifact.

### Gate layers

#### Layer 0: correctness and structural invariants

This runs on ordinary CI and must pass before timing is interpreted:

| Invariant for a staged scalar-loss case | Required value |
|---|---:|
| `compile_count` | 1 |
| `primal_calls` | measured steps |
| `reverse_passes` | measured steps |
| `finite_difference_evals` | 0 |
| `fallback_count` | 0 |
| scalar AD nodes attributable to supported dense ops | 0 |
| post-warmup arena/tensor allocations | 0 |
| arena high-water and scratch bytes after warmup | constant |
| shape/dtype mismatch | explicit non-success status |
| analytic and tiny scalar-AD gradient oracles | pass |

The counters extend the handoff's proposed kernel counters rather than infer
execution shape from elapsed time. A faster result that violates one invariant
is a failed optimization.

#### Layer 1: dedicated-runner PR regression gate

For compiler/runtime changes labeled performance-sensitive, build the merge
base and candidate from clean trees with the same non-PGO Release configuration.
Run cases in randomized `A/B/B/A` blocks on one pinned, quiescent worker.

- Warm up until code, BLAS dispatch, pages, and caches are initialized.
- Choose iterations so each process-level sample lasts at least one second.
- Collect at least seven process-level pairs; increase automatically when the
  coefficient of variation exceeds the benchmark's noise budget.
- Compare paired log ratios and report a bootstrap 95% confidence interval.
- Fail when the lower bound of the candidate/base latency ratio exceeds 1.02
  for the weighted suite geomean, or 1.05 for any sentinel case.
- Fail when compile+link latency significantly exceeds 1.10, peak RSS exceeds
  1.05, or code size exceeds 1.05, unless an approved budget change accompanies
  the change.
- Scratch bytes and post-warmup allocation counts are exact structural values,
  not percentage budgets.

Public shared runners may execute and archive the suite, but they do not make a
hard timing decision.

#### Layer 2: nightly PGO efficacy gate

On each release target, compare same-revision PGO-use and non-PGO Release builds
against the holdout suite:

- the median weighted geomean speedup must be at least 1.03;
- the lower 95% confidence bound of the geomean speedup must be at least 1.00;
- no sentinel's significant latency regression may exceed 3%;
- compiler throughput, staged step throughput, peak RSS, and code size are
  reported separately so a runtime win cannot hide a compiler regression;
- the profile validator must find all required hot symbol groups and no stale
  profile diagnostics.

If PGO does not clear this gate, ship the correct non-PGO build and treat the
profile/corpus as defective. “PGO enabled” is not itself a release outcome.

#### Layer 3: staged training qualification

Nightly and release lanes run:

- a static f64 MLP `value_and_grad` case dominated by GEMM;
- an elementwise/broadcast/reduction loss dominated by memory bandwidth;
- an exact-coordinate-derivative PINN residual case;
- a fused `value + grad + update` case with resident optimizer state;
- small-shape cases where dispatch, guards, and allocation would otherwise
  dominate.

They enforce the training-grade criteria above and compare against both the
parent Eshkol artifact and same-math native references.

### Holdout discipline

Native PGO training and holdout manifests share operation families but not exact
programs, shapes, seeds, or datasets. For example, the native profile may train
an MLP at batch 128 and hidden width 64, while the native-PGO gate uses batch 96
and hidden width 80. This tests whether a product profile generalizes rather
than memorizing benchmark control flow.

Application and kernel IR PGO require the exact compatible IR key, so their
holdout uses the same program and static specialization but disjoint input data,
seeds, and control-flow trajectories. Kernel IR PGO is evaluated twice:

- on that exact compatible specialization with held-out data, to verify the
  profile is useful beyond the observations that created it;
- on sibling specializations in non-PGO mode, which must still meet structural
  and performance requirements rather than consume an incompatible profile.

### Noise controls

- Pin CPU affinity where supported and isolate the worker from unrelated jobs.
- Fix the power governor; record frequency and thermal throttling; abort and
  retry a thermally invalid block.
- Set BLAS/OpenMP thread counts explicitly. Use a single-thread primary lane and
  a separate scaling lane rather than inherit ambient settings.
- Disable network and background downloads; pre-stage all data.
- Keep baseline/candidate binaries, input pages, and invocation order symmetric.
- Use monotonic clocks, output checksums, process timeouts, and fresh processes
  for process-level samples.
- Store every raw sample and outlier decision. Do not replace a failed result
  with a hand-selected rerun.

Thresholds are configuration data versioned with the benchmark schema. After a
benchmark has 20 clean historical runs, its noise budget may be tightened; it
may not be loosened without a reviewed result artifact and rationale.

## Delivery phases

### Phase 0: measurements before optimization

- Implement the JSON runner, machine metadata, paired comparison, and Layer 0
  staged counters.
- Check in separate training and holdout manifests.
- Make every performance build record and assert generated-code O3.
- Establish non-PGO baselines on the release CPU classes.

Exit criterion: one command produces a reproducible baseline report, and
structural failures are distinguishable from timing regressions.

#### Phase 0 implementation status

Landed:

- The generated-code default is no longer O0 for artifacts. Paths that produce
  a persisted artifact (`-o` AOT binary, `-c` object, `--shared-lib`) now
  default to O2, closing the sleeper gap where a Release compiler still emitted
  unoptimized binaries ([`exe/eshkol-run.cpp`](../../../exe/eshkol-run.cpp), the
  post-getopt default-resolution block). Ephemeral/interactive paths (plain
  run, `-r`, `-e`, REPL) and `-g` debug builds stay at O0 for fast turnaround,
  because whole-module optimization folds in referenced stdlib and costs far
  more compile time than a single ephemeral run recovers. An explicit `-O<n>`
  always wins; performance artifacts pass `-O3` explicitly as this phase
  requires.
- A structural assertion pins that contract:
  [`scripts/run_codegen_optlevel_tests.sh`](../../../scripts/run_codegen_optlevel_tests.sh)
  parses the backend's applied-level log and fails if `-o`/`-c` stop defaulting
  to O2, if a plain run starts optimizing, or if an explicit `-O` is ignored. It
  is wired into `run_all_tests.sh`.
- A first PGO training corpus of hot-path programs is checked in at
  [`bench/pgo_corpus/`](../../../bench/pgo_corpus) (arithmetic, autodiff, lists,
  strings, tensors) for later native/IR PGO training.

Deferred to a dedicated follow-up (kept out of this change to avoid displacing
the O0->O2 fix, per "PGO is the last multiplier, not a substitute"): the
versioned JSON result runner with machine metadata and paired A/B comparison,
the separate training/holdout manifests, the Layer 0 staged counters, and the
per-CPU-class non-PGO baselines.

### Phase 1: native PGO release workflow

- Target-scope instrumentation/use flags and propagate the instrumentation
  runtime to generated links.
- Add `%m-%p` run directories, clean bootstrap profiles, weighted merge,
  provenance, strict verification, orchestration targets, and promotion
  artifacts.
- Tune corpus weights only on the training manifest; evaluate on holdout.

Exit criterion: a clean generate/train/merge/use run passes correctness and the
nightly PGO efficacy gate on at least one primary CPU target.

### Phase 2: staged dense graph and static memory

- Complete one-pass dense `value_and_grad`, strict unsupported behavior, and the
  pointer/out-param ABI.
- Introduce typed shape specialization, a static reverse schedule, resident
  buffers, and the scratch planner.
- Add fusion for pointwise, reduction-adjoint, and optimizer loops.

Exit criterion: all Layer 0 invariants pass for the first MLP and PINN kernels,
including 10,000 stable-memory steps.

### Phase 3: closed-world WPO

- Add root analysis, internalization, representation/closure specialization,
  proven LLVM attributes, and the runtime bitcode capsule.
- Use Full LTO for staged kernels and ThinLTO for an experimental ordinary AOT
  mode.
- Preserve constructor, FFI, weak override, exception, and export semantics with
  explicit conformance tests.

Exit criterion: staged qualification reaches the native-reference efficiency
targets without exceeding compile-time, memory, or code-size budgets.

### Phase 4: application and kernel IR PGO

- Add LLVM 21 PassBuilder instrumentation/use configuration and generated-link
  profile runtime support.
- Key profiles to canonical pre-instrumentation IR and specialization identity.
- Add offline generate/use commands and holdout comparisons.

Exit criterion: IR PGO either demonstrates a significant holdout benefit for a
target workload or remains an optional, non-default feature. Native PGO and
static WPO do not depend on it.

## Alternatives considered

### Keep the current CMake switch and document shell commands

Rejected. It cannot prove corpus identity, profile compatibility, correct
generated-runtime linkage, benefit on holdout, or reproducibility. It also
allows an explicitly requested PGO mode to degrade silently to OFF.

### Use one profile for compiler, runtime, and every generated program

Rejected. Native functions and generated LLVM functions have different build
identities and reuse boundaries. A universal profile would either reject valid
builds, accept stale generated IR, or hide most entries as unprofiled.

### Add more LLVM passes before fixing staged AD

Rejected. LLVM cannot infer that a source-level loop performing one tape build
and loss replay per parameter should be one reverse sweep, nor can it recover a
dense tensor graph after scalar AD-node construction has made those effects
observable.

### Full-LTO the entire compiler/runtime archive into every user program

Rejected. It increases compile time and memory, broadens the root/constructor
problem, and imports hosted facilities irrelevant to a staged kernel. A small
runtime capsule plus ThinLTO for general AOT gives the optimizer the useful
boundary without creating a monolith.

### Train and gate on the same benchmarks

Rejected. It rewards overfitting, especially for branch layout and inlining,
and cannot establish that a native product profile generalizes to real users.

### Gate on fixed wall-clock numbers from shared CI

Rejected. Absolute numbers drift with hardware, OS, thermals, BLAS, and runner
load. Paired same-host ratios plus structural invariants provide a defensible
decision.

### Enable fast math globally for training throughput

Rejected. It would change Scheme and AD numerical semantics without an explicit
contract and could invalidate exact derivative tests. Any relaxed mode must be
named, keyed, measured, and tested separately.

## Consequences

### Positive

- Release PGO becomes auditable and reproducible rather than a compiler flag
  whose benefit is assumed.
- Profiles cannot cross incompatible source, toolchain, target, or kernel IR
  boundaries silently.
- The performance gate catches algorithmic regressions even when timing noise is
  high, because AD calls, fallbacks, nodes, and allocation counts are exact.
- Static staged kernels expose the information needed for dense dispatch,
  fusion, vectorization, memory reuse, and LLVM LTO.
- Ordinary AOT gains a path to ThinLTO without imposing closed-world assumptions
  on the REPL or shared libraries.

### Costs and risks

- Release builds require three clean build trees, dedicated performance hosts,
  corpus maintenance, and retained artifacts.
- Full LTO and specialization increase compile latency and cache size; budgets
  and specialization-family limits are required.
- Incorrect root analysis or LLVM attributes can miscompile programs. Both must
  be conservative and backed by conformance/alias tests.
- Static shapes can cause code-size explosion. Only observed or declared shape
  families are compiled, with an explicit cap and eviction policy.
- Vendor library and CPU changes invalidate long-lived baselines and may require
  retraining native profiles.
- Profile-guided inlining can increase code size even when it improves runtime;
  the gate therefore reports and budgets both.

## Acceptance criteria for this ADR

The architecture is implemented when:

1. `ESHKOL_PGO=generate/use` participates in a documented one-command workflow
   that builds clean baseline/generate/use trees, clears bootstrap data, runs a
   versioned weighted corpus, merges, verifies, and writes provenance.
2. Requested but unsupported PGO and stale profile data fail closed; required
   generated-executable links correctly include the instrumentation runtime.
3. Native training and holdout manifests are separate, and the dedicated-runner
   gate publishes raw paired samples and confidence intervals.
4. Generated performance artifacts record O3 independently of CMake Release.
5. The staged kernel key includes shapes, target CPU/features, math mode,
   pipeline/AD versions, and the optional profile digest.
6. The first supported staged MLP and PINN kernels meet all Layer 0 invariants:
   one primal/reverse per step, dense adjoints, no finite differences/fallback,
   no scalarized dense AD, and stable zero-allocation steady state.
7. The staged WPO path uses an explicit root set, a small runtime bitcode
   capsule, and LLVM 21 Full LTO; general AOT ThinLTO remains separately
   selectable.
8. Native PGO passes its efficacy gate; otherwise the release automation selects
   the qualified non-PGO build rather than labeling an unproven binary as PGO.
9. Staged release qualification reaches the native reference-efficiency targets
   without violating correctness, compile-time, memory, or code-size budgets.
