# ADR 0002: Staged Dense-Tensor Reverse-Mode AD and `value_and_grad`

Status: Proposed

Date: 2026-07-09

Audience: Eshkol compiler/runtime maintainers working on the AD subsystem for
SciML and PINN training workloads.

## Position

Eshkol should make automatic differentiation a typed, shape-specialized compiler
IR feature, not a larger Scheme wrapper layer and not an LLVM-afterthought
pass. The target training primitive is:

```text
compile once:
  staged = compile_value_grad_update(loss, static_param_shapes, static_input_shapes, derivative_spec)

run many:
  status = staged(params, grads, inputs, outputs, scratch, step)
```

The default SciML/PINN path must be exact AD or an explicit unsupported-op error.
Hidden finite differences are disallowed.

The current implementation already has the right substrate:

- `ad_node_t` already has scalar fields plus tensor fields for `tensor_value`,
  `tensor_gradient`, `input3`, `input4`, `saved_tensors`, `params`, `shape`, and
  `ndim` (`inc/eshkol/eshkol.h:971-1013`).
- The codegen type system already assigns stable AD-node field indices for those
  tensor fields (`inc/eshkol/backend/type_system.h:132-152`).
- `AutodiffCodegen::recordADNodeTensor` already allocates a tensor-valued node,
  fills inputs, saved tensors, shape/rank, and appends it to the current tape
  (`lib/backend/autodiff_codegen.cpp:2334-2448`).
- `backpropagate` already seeds scalar and tensor outputs, then walks the tape in
  reverse order (`lib/backend/autodiff_codegen.cpp:9977-10017`).
- `propagateGradient` already has a tensor-gradient fast path that calls
  `eshkol_tensor_backward_dispatch` when field 7 is non-null
  (`lib/backend/autodiff_codegen.cpp:10134-10160`).
- The runtime already has dense backward kernels for important primitives,
  including the matmul wrapper over `eshkol_matmul_backward_f64`
  (`lib/backend/tensor_backward.cpp:542-559`).

The current implementation also has the wrong execution shape for training:

- `gradient` documents that tensor/vector reverse mode allocates a fresh tape and
  replays the function for each input dimension (`lib/backend/autodiff_codegen.cpp:3823-3846`),
  and the emitted loop does exactly that (`lib/backend/autodiff_codegen.cpp:5261-5880`).
- Variables are not appended to the tape; they are tracked separately
  (`lib/backend/autodiff_codegen.cpp:2500-2551`). `arena_tape_reset` zeroes
  gradients and sets `num_nodes` to zero (`lib/core/runtime_autodiff.cpp:454-479`),
  so retained-graph row sweeps need a separate zero-gradient helper.
- `jacobian` still has an `(output, input)` nested replay shape: it allocates a
  tape inside the inner loop (`lib/backend/autodiff_codegen.cpp:6575-6615`),
  backpropagates one output component, reads one input gradient, and resets the
  tape (`lib/backend/autodiff_codegen.cpp:6828-6864`).
- Dense matmul has an AD-mode scalarized loop that records scalar multiply/add
  nodes per output element (`lib/backend/llvm_codegen.cpp:27616-27745`);
  the dense `recordADNodeTensor` path exists but is guarded by
  `!after_matmul_compute`, so it is skipped when the AD branch exists
  (`lib/backend/llvm_codegen.cpp:27781-27831`).
- Tensor arithmetic similarly records scalar binary nodes per element in AD mode
  (`lib/backend/tensor_arith_codegen.cpp:490-519`).
- The tensor backward dispatcher states that unsupported tensor node types
  silently drop gradients (`lib/backend/tensor_backward.cpp:1002-1012`), and the
  default branch does exactly that (`lib/backend/tensor_backward.cpp:1362-1365`).
- Hidden finite differences remain: higher-order `gradient` uses central
  differences (`lib/backend/autodiff_codegen.cpp:3520-3560` and
  `lib/backend/autodiff_codegen.cpp:3687-3711`), vector/tensor Hessian uses an
  epsilon difference of gradients (`lib/backend/autodiff_codegen.cpp:7683-7690`,
  `lib/backend/autodiff_codegen.cpp:8168-8169`, and
  `lib/backend/autodiff_codegen.cpp:8364-8633`), and Scheme tape exposes
  `record-fd-op!` as central finite differences (`lib/core/ad/tape.esk:129-158`).

## Assessment of the Handoff Plan

The handoff is directionally right. It identifies the core architectural bug:
Eshkol has semantically rich AD but the dense tensor training path is not a
dense tensor reverse pass. It also correctly prioritizes counters, one-pass
gradient, strict unsupported tensor backward, removal of hidden finite
differences, and a staged pointer/out-param ABI.

What I would keep:

- Phase A counters are non-negotiable. Without counters for primal calls, reverse
  passes, scalar nodes, tensor nodes, tape allocations, and finite-difference
  evaluations, the project cannot distinguish a semantic fix from a performance
  placebo.
- The one-pass scalar-loss gradient is the first real semantic shape change.
  Current reverse gradient is a per-component loop; staged `value_and_grad`
  needs one primal and one reverse pass.
- Dense tensor nodes are the right abstraction. A matmul should be one AD graph
  node plus a dense VJP, not `M*N*K` scalar AD nodes.
- Strict unsupported behavior must be enforced before widening dense-node
  recording. Silent zero gradients are worse than missing features.
- Exact coordinate derivatives for PINNs should be built from the Taylor/JVP
  machinery, not finite differences.
- Optimizer/update staging should come after correct `value_and_grad`, not before.

What I would change:

- Move "strict unsupported tensor backward" ahead of broad dense-node rollout. In
  the handoff it appears inside Phase C after some dense routing. In practice,
  strictness is the guardrail for Phase C. Call it Phase C0 and make tests run in
  strict mode before recording new tensor node types.
- Treat `value_and_grad` as an internal lowering target from the start, even
  before exposing a public API. A one-pass `gradient` helper should be factored as
  the core of `value_and_grad`, not as a separate path that later gets copied.
- Split "staged ABI" into two milestones: first a raw pointer/out-param wrapper
  that may still use the arena internally, then a no-hot-loop-allocation memory
  plan. Waiting for the full memory planner before exposing the ABI will delay
  useful benchmark feedback.
- Do not rely on an environment variable as the truth for strict AD. Use a
  compile/run flag in the staged kernel and make exact-only the default for
  SciML. An env var can be a diagnostic override, not the policy.
- Put dense primitive registration before adding many op IDs. A central table
  should say whether a primitive has shape inference, forward lowering, residual
  saving, and VJP support. Scattered `switch` additions will repeat the current
  "record something, maybe drop it later" failure mode.
- Make Jacobian row-sweep a correctness/performance cleanup, but not the critical
  path for the first Phydra-style training benchmark. Scalar-loss
  `value_and_grad` plus exact coordinate derivatives matter more for PINNs than a
  general dense Jacobian.

What the handoff misses:

- Cotangent layout is an ABI decision. Every tensor primitive needs an explicit
  answer for where its upstream cotangent lives, who owns internal cotangent
  buffers, and how zeros are represented.
- The staged kernel needs an error ABI. Unsupported AD, bad shape, unsupported
  dtype, scratch-too-small, and runtime error must be status codes, not stderr
  side effects.
- Tensor gradients for resident parameters should be caller-provided buffers,
  not arena allocations hidden behind `tensor_gradient`. Current accumulation
  allocates from the global arena when `tensor_gradient` is null
  (`lib/backend/tensor_backward.cpp:1376-1397`); staged kernels should prebind
  parameter leaf gradients to `grads[p]`.
- Current backward temporaries are arena-scoped inside
  `eshkol_tensor_backward_dispatch` (`lib/backend/tensor_backward.cpp:1023-1028`
  and `lib/backend/tensor_backward.cpp:1368-1371`). That is good for non-staged
  correctness but not enough for a fixed scratch memory plan.
- There is no differentiability/effects analysis boundary. Staged kernels must
  reject unsupported mutation, escaping temporaries, dynamic rank, and unknown
  closure capture at compile time.
- The design should define how Taylor-tower values compose with dense reverse
  mode. The Taylor subsystem is exact and should remain the coordinate derivative
  engine, not be bypassed by dense F64 kernels.

## Recommended Architecture

### Adopt a JAX-like primitive IR, not Enzyme or Zygote as the primary model

Eshkol should use a JAX-like architecture at the Eshkol compiler IR boundary:

```text
Scheme AST
  -> typed Eshkol IR with static tensor shapes for staged regions
  -> primitive graph with differentiability annotations
  -> linearized primal graph with residual plan
  -> transposed/VJP graph
  -> LLVM wrapper with raw pointer ABI and scratch plan
```

The useful JAX ideas are `jaxpr`, `linearize`, primitive VJP/transpose rules,
residual capture, and compilation keyed by static shape. Eshkol should adopt
those ideas without importing Python/PyTree complexity. Start with flat parameter
leaves and explicit descriptors.

Enzyme-style LLVM-level AD should not be the primary AD system here. LLVM-level
AD is attractive for C-like leaf kernels, but Eshkol's interesting semantics
exist above LLVM: tagged values, Scheme closures, tensor shapes, Taylor towers,
arena lifetimes, and staged unsupported-op policy. By the time LLVM sees raw
pointer arithmetic, too much semantic structure has been erased or lowered into
runtime calls.

Zygote-style source-to-source AD is also the wrong primary model. It is elegant
for a high-level language, but dense tensor training needs a small set of
compiler-known primitives with explicit transpose rules and memory planning.
Source-to-source transformation would make mutation, allocation, closure capture,
and tensor residual storage harder to control.

Enzyme can still be useful later as an optional backend for leaf scalar kernels.
Zygote-like transformations can still exist for Scheme-level experiments. The
staged SciML path should be primitive-IR AD.

### Core value model

Every staged tensor expression lowers to a value carrying both a primal view and
an optional AD node:

```c
typedef enum {
    ESHKOL_DTYPE_F64 = 1
} EshkolDType;

typedef struct {
    void* data;
    EshkolDType dtype;
    uint32_t rank;
    uint32_t flags;        /* contiguous, readonly, param, input, scratch */
    uint64_t dims[8];
    uint64_t strides[8];
    uint64_t elements;
} EshkolTensorView;

typedef struct EshkolADNode EshkolADNode;

typedef struct {
    EshkolTensorView primal;
    EshkolADNode* node;    /* null when no gradient is required */
} EshkolADTensorValue;
```

This is the staged analog of the handoff's `unwrapTensorOrADTensor` helper. It
replaces pointer-range heuristics with an explicit typed convention. The current
code still has heuristic paths in places, for example tensor gradient input
handling checks pointer-like integer ranges (`lib/backend/autodiff_codegen.cpp:5320-5361`),
and Jacobian output checks integer bit patterns to decide whether an output
element is an AD node (`lib/backend/autodiff_codegen.cpp:6799-6817`). Staged AD
should not use those heuristics.

### Node representation

Short term, use the existing `ad_node_t` for dense nodes because the runtime and
codegen already agree on it. Long term, add a versioned dense payload instead of
overloading six `int64_t` params and `void** saved_tensors` forever.

Minimum current-node convention:

```c
/* Existing ad_node_t fields, used by convention. */
type             = AD_NODE_TENSOR_* or legacy AD_NODE_MATMUL/SUM/MEAN
input1..input4   = producer nodes for differentiable tensor inputs
tensor_value     = output data pointer or tensor object pointer, consistently chosen per ABI
tensor_gradient  = upstream cotangent data pointer
saved_tensors    = residual pointer array: inputs, outputs, masks, stats
num_saved        = residual count
params           = small fixed op parameters
shape, ndim      = output shape
```

Recommended v2 node for staged kernels:

```c
typedef enum {
    ESHKOL_ADOP_TENSOR_ADD,
    ESHKOL_ADOP_TENSOR_SUB,
    ESHKOL_ADOP_TENSOR_MUL,
    ESHKOL_ADOP_TENSOR_DIV,
    ESHKOL_ADOP_TENSOR_MATMUL,
    ESHKOL_ADOP_TENSOR_SUM_ALL,
    ESHKOL_ADOP_TENSOR_MEAN_ALL,
    ESHKOL_ADOP_TENSOR_TRANSPOSE,
    ESHKOL_ADOP_TENSOR_RESHAPE,
    ESHKOL_ADOP_SCALAR_EXTRACT
} EshkolADOp;

typedef struct {
    uint32_t abi_version;       /* 1 */
    uint32_t op;                /* EshkolADOp */
    uint32_t flags;             /* differentiable inputs, strict, owns_grad */
    uint32_t ninputs;
    EshkolADNode* inputs[4];
    EshkolTensorView value;
    EshkolTensorView cotangent;
    const void* params;
    uint32_t params_bytes;
    uint32_t nsaved;
    EshkolTensorView saved[8];
    uint32_t topo_id;
} EshkolADTensorNodePayload;

struct EshkolADNode {
    uint32_t kind;              /* scalar or dense tensor */
    uint32_t op;
    double scalar_value;
    double scalar_cotangent;
    EshkolADNode* input1;
    EshkolADNode* input2;
    void* dense_payload;        /* EshkolADTensorNodePayload* for dense nodes */
};
```

This can be introduced as a staged-only node array first, then unified with
`ad_node_t` later. If maintainers prefer extending `ad_node_t`, add
`void* op_data`, `uint32_t op_data_bytes`, `uint16_t dtype`, and `uint16_t flags`
after `ndim`, then update the field mapping currently fixed in
`TypeSystem` (`inc/eshkol/backend/type_system.h:132-152`).

### Primitive registry and VJP rules

Every dense primitive in staged AD must be registered with shape inference,
forward lowering, residual planning, and VJP support:

```c
typedef struct EshkolADRun EshkolADRun;

typedef int (*EshkolForwardKernel)(
    EshkolADRun* run,
    const EshkolTensorView* inputs,
    uint32_t ninputs,
    EshkolTensorView* output,
    const void* params);

typedef int (*EshkolVJPKernel)(
    EshkolADRun* run,
    const EshkolADTensorNodePayload* node);

typedef struct {
    uint32_t op;
    const char* name;
    uint32_t min_inputs;
    uint32_t max_inputs;
    uint32_t flags;        /* elementwise, linear, reduction, view, needs_residuals */
    EshkolForwardKernel forward;
    EshkolVJPKernel vjp;
} EshkolADPrimitive;
```

Rules are transposes of linearized primitives, not materialized Jacobians:

```text
Y = A + B
  dA += dY
  dB += dY

Y = A - B
  dA += dY
  dB -= dY

Y = A * B
  dA += dY * B
  dB += dY * A

Y = A / B
  dA += dY / B
  dB -= dY * A / (B * B)

C = A @ B
  dA += dC @ B^T
  dB += A^T @ dC

s = sum(A)
  dA += broadcast(dS, shape(A))

m = mean(A)
  dA += broadcast(dM / numel(A), shape(A))

Y = broadcast(A, shapeY)
  dA += reduce_sum(dY, broadcasted_axes)
```

The matmul rule should call the existing dense backward kernel. Current matmul
backward lacks overflow checks around `M*K` and `K*N` allocations
(`lib/backend/tensor_backward.cpp:1157-1172`); add those before relying on it in
strict staged mode.

The rule for recording is:

```text
if staged_ad_active and any input needs gradient:
    compute the numeric dense forward result
    create one dense AD node
    save only residuals required by the VJP
    return {primal = result_view, node = dense_node}
else:
    compute numeric dense forward result
    return {primal = result_view, node = null}
```

Never record a dense node whose VJP is absent in exact-only mode. Compile-time
rejection is preferred. Runtime strict rejection is a backstop.

### Scalar loss and parameter leaves

For `value_and_grad`, trainable parameters are tensor leaf nodes:

```c
typedef struct {
    const char* name;
    EshkolTensorView value;
    EshkolTensorView grad;
    uint32_t flags;      /* trainable, zero_grad_on_entry, accumulate */
} EshkolParamLeaf;
```

A parameter leaf's `tensor_value` or `value.data` points to the caller's resident
parameter buffer. Its `tensor_gradient` or `cotangent.data` points to the caller's
resident gradient buffer. The generated wrapper zeroes gradients according to the
call policy before the primal run. This avoids arena allocation for parameter
gradients and makes optimizer epilogues straightforward.

Variables not appended to the tape are acceptable for scalar AD today
(`lib/backend/autodiff_codegen.cpp:2500-2551`), but staged dense leaves should be
held in a parameter-leaf table. The reverse pass reads all leaves after one
backward traversal; it must not rebuild the tape per parameter.

## Staged Kernel ABI

Use raw pointers and descriptors across the C ABI. Do not pass tagged values or
Scheme objects by value through the staged boundary.

```c
typedef enum {
    ESHKOL_KERNEL_OK = 0,
    ESHKOL_KERNEL_BAD_SHAPE = 1,
    ESHKOL_KERNEL_BAD_DTYPE = 2,
    ESHKOL_KERNEL_UNSUPPORTED_OP = 3,
    ESHKOL_KERNEL_UNSUPPORTED_AD = 4,
    ESHKOL_KERNEL_SCRATCH_TOO_SMALL = 5,
    ESHKOL_KERNEL_RUNTIME_ERROR = 6
} EshkolKernelStatus;

typedef struct {
    EshkolDType dtype;
    uint32_t rank;
    uint32_t flags;
    uint64_t dims[8];
    uint64_t strides[8];
    uint64_t elements;
} EshkolTensorDesc;

typedef struct {
    uint32_t num_params;
    uint32_t num_inputs;
    uint32_t num_outputs;
    const EshkolTensorDesc* params;
    const EshkolTensorDesc* inputs;
    const EshkolTensorDesc* outputs;
} EshkolKernelSignature;

typedef struct {
    double* const* params;
    double* const* grads;
    const double* const* inputs;
    double* const* outputs;
    void* scratch;
    uint64_t scratch_bytes;
    uint64_t step;
    uint32_t flags;       /* exact_only, zero_grads, allow_accumulate */
} EshkolKernelCall;

typedef struct {
    uint64_t invocations;
    uint64_t primal_calls;
    uint64_t reverse_passes;
    uint64_t tape_allocations;
    uint64_t scalar_ad_nodes;
    uint64_t tensor_ad_nodes;
    uint64_t tensor_backward_dispatches;
    uint64_t unsupported_ad;
    uint64_t finite_difference_evals;
    uint64_t arena_bytes_high_water;
    uint64_t scratch_bytes;
} EshkolKernelCounters;

typedef struct EshkolStagedKernel EshkolStagedKernel;

EshkolKernelStatus eshkol_compile_staged_value_grad(
    const void* compiler_ir,
    const EshkolKernelSignature* sig,
    EshkolStagedKernel** out);

EshkolKernelStatus eshkol_staged_kernel_run(
    EshkolStagedKernel* kernel,
    const EshkolKernelCall* call);

EshkolKernelStatus eshkol_staged_kernel_get_counters(
    EshkolStagedKernel* kernel,
    EshkolKernelCounters* out);

void eshkol_staged_kernel_destroy(EshkolStagedKernel* kernel);
```

Generated LLVM wrapper shape:

```llvm
define i32 @__eshkol_vg_<hash>(
    ptr %runtime,
    ptr %params,        ; double**
    ptr %grads,         ; double**
    ptr %inputs,        ; double**
    ptr %outputs,       ; double**
    ptr %scratch,
    i64 %scratch_bytes,
    i64 %step,
    i32 %flags
) {
entry:
    ; validate scratch and shape-specialized assumptions
    ; bind parameter leaves: value=params[p], cotangent=grads[p]
    ; optionally memset grads[p] to zero
    ; reset per-call tape/node arena marker
    ; run one primal loss computation
    ; seed scalar loss cotangent = 1
    ; walk dense/scalar node array in reverse once
    ; write outputs[0] = loss
    ; return ESHKOL_KERNEL_OK or explicit status
}
```

First supported subset:

```text
dtype:
  f64

layout:
  contiguous row-major

shapes:
  static rank and static dims

control:
  straight-line code and simple shape-static loops

AD:
  scalar loss
  gradients w.r.t. flat parameter tensor leaves
  exact coordinate Taylor/JVP lanes for requested small coordinate variables

ops:
  elementwise add/sub/mul/div
  matmul
  full tensor sum/mean
  reshape/transpose as views where possible
  tanh/sigmoid/Stan activation if required for the first PINN

unsupported:
  dynamic rank
  unknown dtype
  unsupported broadcast VJP
  mutation escaping the staged region
  unknown closure capture
  unsupported tensor backward
  finite-difference fallback
```

## Arena and Memory Plan

The staged kernel should use three memory classes:

```text
resident:
  params, grads, optimizer state, model constants
  owned by caller or training object
  lives across invocations

scratch:
  fixed-size byte buffer selected by static shape key
  holds forward temporaries, residuals, internal cotangents, node arrays
  lives for one invocation but reused across invocations

arena:
  small dynamic metadata and legacy runtime interop during transition
  reset to a saved marker at end of invocation
  must not grow after warmup in fully staged mode
```

Compile-time memory planning should assign offsets for:

- primal temporaries,
- residuals saved for backward,
- internal cotangent buffers,
- dense AD node payloads,
- small shape/params records,
- optional optimizer state temporaries.

Sketch:

```c
typedef struct {
    uint32_t id;
    EshkolTensorDesc desc;
    uint64_t bytes;
    uint64_t alignment;
    uint32_t first_use;
    uint32_t last_use;
    uint32_t flags;       /* residual, cotangent, may_alias_input, in_place_ok */
} EshkolTemp;

typedef struct {
    uint64_t scratch_bytes;
    uint32_t num_temps;
    const EshkolTemp* temps;
    const uint64_t* offsets;
    uint32_t num_nodes;
    uint64_t node_array_offset;
} EshkolKernelMemoryPlan;
```

Rules:

- Parameter and input views alias caller buffers.
- Output buffers alias caller outputs.
- Internal primal buffers live in scratch.
- Residuals are views of params/inputs/scratch when the producer value remains
  live; otherwise the planner keeps the buffer alive until the VJP consumes it.
- Internal cotangents live in scratch and can be reused after their last reverse
  consumer.
- Parameter cotangents alias `grads[p]`.
- Legacy arena allocation is allowed only in transitional kernels and must be
  counted. A staged "no hot loop allocation" test should assert stable arena
  high-water after warmup.

This is compatible with OALR/no-GC: lifetimes are lexical per compiled kernel
invocation, and the planner proves that no temporary needs object tracing.

## Removing Hidden Finite Differences

The policy should be simple:

```text
gradient, hessian, laplacian, divergence, curl, PINN residual derivatives:
  exact AD or unsupported error

finite differences:
  explicit numeric API only
```

Actions:

1. Add an exact-only flag used by staged SciML and by tests. In exact-only mode,
   any attempt to enter finite-difference code returns `UNSUPPORTED_AD` or emits
   a compile-time diagnostic.
2. Keep explicit numerical APIs, but rename them so users know what they are:
   `numeric-gradient`, `numeric-hessian`, `record-fd-op!`, or similar. Increment
   `finite_difference_evals` every time they evaluate a perturbation.
3. Change `(gradient f)` higher-order closure generation. Today it creates a
   central-difference wrapper (`lib/backend/autodiff_codegen.cpp:3520-3560`).
   Preferred: lower to exact staged/direct gradient when the function and arity
   are known. Temporary acceptable behavior: explicit unsupported exact closure.
4. Change vector/tensor Hessian. Today it computes base gradients and perturbed
   gradients with epsilon (`lib/backend/autodiff_codegen.cpp:8168-8169` and
   `lib/backend/autodiff_codegen.cpp:8364-8633`). Preferred for PINNs: exact
   selected coordinate Hessian diagonal via Taylor/JVP. General dense Hessian of
   all tensor parameters can remain unsupported until a real mixed-mode plan
   exists.
5. Keep scalar exact paths. The scalar Hessian path now uses forward-over-forward
   dual components for `f''(x)` (`lib/backend/autodiff_codegen.cpp:7844-7862`);
   do not regress it because nearby comments still mention central differences.
6. In tensor backward dispatch, no-op/default cases become errors in strict mode.
   Current bridge tensor ops without dedicated backward explicitly skip
   (`lib/backend/tensor_backward.cpp:1347-1360`); staged exact-only mode must not
   accept those nodes.

## Taylor-Tower Composition

The Taylor system should remain the exact coordinate-derivative engine.

Existing facts:

- Taylor towers represent `c[0..K]` where `f^(n)(x0) = n! * c[n]`
  (`lib/core/runtime_taylor.c:6-17`).
- Epoch tags prevent perturbation confusion across nested tower contexts
  (`lib/core/runtime_taylor.c:24-29`).
- Exact-coefficient towers preserve R7RS exactness for exact polynomial/rational
  paths (`lib/core/runtime_taylor.c:31-42`).
- Exact point seeding creates exact towers when possible
  (`lib/core/runtime_taylor.c:1282-1324`).
- Exact derivative extraction is preserved by `eshkol_taylor_extract_tagged`
  (`lib/core/runtime_taylor.c:1391-1410`).
- Reverse-over-Taylor already lifts reverse-tape AD nodes into dual towers with
  seed tangents (`lib/core/runtime_taylor.c:1356-1388`), and codegen routes AD
  nodes through `towerLiftOperand` before normal jet lifting
  (`lib/backend/autodiff_codegen.cpp:852-1005`).

The staged dense design must not demote Taylor values to finite differences or
plain F64 by accident. For PINNs, the compiler should represent coordinate
derivative requests as a small derivative-lane bundle:

```c
typedef struct {
    uint32_t order;       /* 0, 1, or 2 initially */
    uint32_t lanes;       /* value, dx, dy, dz, dxx, dyy, dzz, etc. */
    uint32_t exact_flags; /* lane exactness where available */
} EshkolCoordDerivativeSpec;
```

Lowering strategy:

```text
for each collocation batch shape:
  seed coordinate Taylor/JVP lanes exactly
  run the primal model once over lanes
  extract requested value/derivative lanes
  build residual loss from those lanes
  reverse the scalar loss once into dense parameter leaves
```

Dense tensor primitives should be lane-polymorphic in staged PINN mode:

- For ordinary training, lane count is 1 and kernels operate on F64 tensors.
- For coordinate derivatives, a tensor element may carry a small lane bundle or
  tower value. Elementwise ops apply the Taylor recurrence per element/lane.
- Matmul is linear in both operands, so for coordinate lanes it computes the same
  dense contraction independently per lane. For parameter gradients, the VJP
  still accumulates into F64 parameter buffers using the cotangent of the scalar
  loss with respect to the lane-composed residual.
- Unsupported lane/tensor combinations fail exactly. They do not fall back to
  finite differences.

This keeps higher-order coordinate derivatives exact while preserving dense
reverse-mode gradients for large parameter tensors.

## Dependency-Ordered Implementation Sequence

1. Exact-mode instrumentation and strict runtime backstop.
   Add global and per-kernel AD counters, finite-difference counters, strict
   unsupported tensor backward behavior, and tests that prove unsupported tensor
   VJPs do not silently return zero. This is the first PR.

2. Typed AD tensor operand convention.
   Add an internal helper that returns `{numeric_tensor_ptr, ad_node_ptr,
   needs_grad}` for plain tensors and AD tensors. Replace pointer-range
   heuristics at new staged call sites first, then migrate old paths.

3. One-pass scalar-loss gradient helper.
   Factor `buildADTensorInput`, `callFunctionWithADInputs`,
   `extractScalarOutputADNode`, `backpropagate`, and `readAllLeafGradients`.
   Preserve scalar exact fast paths and nested AD. This helper is the core of
   both `gradient` and `value_and_grad`.

4. Tensor gradient zeroing without tape reset.
   Add `arena_tape_zero_gradients` and a variant that zeros variable/leaf
   gradients. This unblocks retained-graph Jacobian row sweeps and repeated
   staged reverse passes.

5. Dense primitive registry and strict matmul.
   Route AD-mode matmul to numeric dense forward plus one dense node. Add matmul
   VJP overflow checks. Assert `scalar_ad_nodes_from_matmul == 0` for supported
   dense shapes.

6. Dense reductions and elementwise ops.
   Add full sum/mean, same-shape add/sub/mul/div, then broadcast add/mul with
   reduce-over-broadcasted-axes VJPs. Unsupported axis reductions fail.

7. Internal `valueAndGradientFlat`.
   Return `(loss, flat_grad_buffers)` for flat parameter leaves. No PyTree
   ergonomics yet. Inputs are non-trainable unless explicitly marked.

8. Raw pointer staged ABI.
   Generate the LLVM wrapper with static shape guards and status returns. It may
   still use arena allocation internally but must expose counters.

9. Scratch memory planner.
   Move forward temporaries, residuals, internal cotangents, and node arrays to
   fixed scratch offsets. Enforce no hot-loop arena growth after warmup.

10. Exact coordinate derivative staging.
    Add `EshkolCoordDerivativeSpec` and compile selected first/second coordinate
    derivatives through Taylor/JVP lanes. Use this for PINN residuals before
    attempting full general tensor Hessians.

11. Staged optimizer/update epilogue.
    Add SGD first, then Adam/Rprop. Optimizer state uses resident descriptors and
    participates in the same shape key.

12. Jacobian row-sweep cleanup.
    Refactor `jacobian` to row-sweep and then retained-graph row replay. This is
    useful, but not the first training milestone.

## First Implementation PR

The smallest high-leverage first PR is not matmul. It is the invariant harness:

```text
Title:
  AD exact-mode counters and strict unsupported tensor backward

Files:
  inc/eshkol/eshkol.h
  lib/core/runtime_autodiff.cpp
  lib/backend/autodiff_codegen.cpp
  lib/backend/tensor_backward.cpp
  inc/eshkol/backend/memory_codegen.h
  lib/backend/memory_codegen.cpp
  tests/ad/no_hidden_finite_difference_test.esk
  tests/ad/tensor_unsupported_backward_test.esk

Behavior:
  add EshkolADCounters
  count primal calls, reverse passes, tape allocations, scalar nodes, tensor nodes
  count tensor backward dispatches and unsupported tensor AD
  count finite-difference evaluations in explicit numeric paths
  add exact-only mode
  make tensor backward default/no-op cases error in exact-only mode
  keep explicit finite-difference APIs available outside exact-only mode
```

Why this first:

- It directly enforces the hard SciML constraint before adding more dense nodes.
- It makes later PRs measurable.
- It is small enough to review without changing AD semantics broadly.
- It exposes current failures honestly: component replay, scalarized tensor ops,
  unsupported VJPs, and finite-difference entry points become visible.

The second PR should be one-pass scalar-loss tensor gradient. The third should be
dense matmul VJP. That order gives a fast path to a meaningful benchmark:

```text
loss(W, x) = sum((W @ x) * (W @ x))

expected counters:
  primal_calls = 1
  reverse_passes = 1
  finite_difference_evals = 0
  tensor_ad_nodes includes matmul
  scalar_ad_nodes_from_matmul = 0
```

## Decision

Build staged AD as a typed, shape-specialized primitive graph with dense
transpose rules. Keep Taylor towers as the exact higher-order coordinate engine.
Use raw pointer/out-param staged kernels, resident parameter and gradient
buffers, and a fixed scratch memory plan. Remove hidden finite differences from
the default compiler AD path. Make unsupported exact AD loud.

The immediate engineering bar is not "more AD features". It is this invariant:

```text
For a scalar tensor loss over trainable parameter buffers:
  one primal evaluation
  one reverse pass
  no finite differences
  dense tensor nodes for supported dense primitives
  explicit unsupported errors for the rest
```
