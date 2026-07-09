# COMPILER.md — Eshkol Compiler Work Needed For Phydra-Style Training

Generated: 2026-07-09

This document is a maintainer-facing compiler/runtime handoff for the Eshkol
repository. It explains what the current AD/tensor compiler already has, what is
missing for Phydra/Phydrax-like SciML workloads, and exactly what should be
implemented first.

Paths are relative to the Eshkol repository root unless otherwise stated.

Hard constraint from the consuming SciML/PINN work:

> Do not use finite differences for the SciML/PINN derivative path. The default
> compiler path must be exact AD or an explicit unsupported-operation error.

The short version:

- Eshkol already has broad semantic AD infrastructure: scalar reverse tape,
  forward-mode jets, Taylor towers, reverse-over-forward, reverse-over-Taylor,
  GUW-style higher-order helpers, tensor objects, and several tensor backward
  kernels.
- The missing piece is not another Scheme wrapper layer. The missing piece is a
  compiler/runtime path that can compile a tensor loss into a reusable staged
  `value + grad + update` kernel.
- Current vector/tensor gradient and Jacobian codegen repeatedly rebuilds tapes
  and replays the user function per component. That is semantically useful but
  not a JAX/Phydrax-like execution model.
- Current dense tensor codegen mostly scalarizes tensor AD. Matmul, reductions,
  elementwise ops, broadcasts, activations, and some normalization paths build
  many scalar AD nodes instead of recording one dense tensor node and calling
  dense backward kernels.
- Hidden finite-difference fallbacks still exist in higher-order-gradient and
  vector/tensor Hessian paths. Those must be removed from the default compiler
  AD path or made explicit numeric APIs.

## 1. Current Compiler State

### 1.1 AD infrastructure already present

Primary files:

- `lib/backend/autodiff_codegen.cpp`
- `inc/eshkol/backend/autodiff_codegen.h`
- `lib/core/runtime_autodiff.cpp`
- `inc/eshkol/eshkol.h`

The compiler/runtime already has:

- `ad_node_t` and `ad_tape_t` for scalar reverse-mode AD.
- `AutodiffCodegen::gradient`, `jacobian`, `hessian`, `laplacian`,
  `derivative`, `derivativeN`, and Taylor APIs.
- Runtime perturbation-level tracking for nested forward-mode AD.
- Runtime tape stack / AD mode globals.
- Reverse-over-forward hooks:
  - `eshkol_ad_seed_swap`
  - `eshkol_ad_seed_flag`
  - `eshkol_ad_mixed_record`
  - `AutodiffCodegen::maybeJetLiftTapeOperand`
- Reverse-over-Taylor hooks:
  - `AutodiffCodegen::towerCtxPush`
  - `AutodiffCodegen::towerCtxPop`
  - `AutodiffCodegen::towerLiftOperand`
  - `eshkol_taylor_lift_ad_node`
  - `eshkol_taylor_extract_tangent`

Important observed behavior:

- Scalar arity-1 gradients can use exact forward-mode jet paths.
- Taylor towers support arbitrary-order univariate derivatives.
- Reverse-over-Taylor is present and tested.
- Vector/tensor gradients exist semantically, but the current implementation is
  not one-pass reverse-mode over all inputs.

### 1.2 Taylor tower infrastructure already present

Primary files:

- `lib/core/runtime_taylor.c`
- `lib/core/taylor_recurrences.def`
- `lib/core/ad/guw.esk`
- `lib/core/ad/checkpoint.esk`
- `docs/design/AD_TAYLOR_TOWER.md`
- `tests/ad/taylor_tower_test.esk`
- `tests/ad/taylor_tower_mono_test.esk`
- `tests/ad/tensor_tower_test.esk`
- `tests/ad/reverse_over_taylor_test.esk`
- `tests/ad/checkpointed_reverse_test.esk`

Already implemented and useful:

- Runtime heap Taylor towers.
- Literal-K monomorphized Taylor path for selected pure scalar recurrences.
- Exact coefficient path for selected cases.
- Reverse-over-Taylor for gradients through `derivative-n`.
- Scheme-level checkpointed reverse-over-Taylor proof-of-shape.
- Tensor-valued tower experiments through Scheme-level `core.ad.tensor_tower`.

Important limitation:

- These scalar/Taylor improvements do not yet provide a staged dense tensor
  `value_and_grad` kernel for trainable parameter tensors.

### 1.3 Tensor AD substrate already present

Primary files:

- `inc/eshkol/eshkol.h`
- `inc/eshkol/backend/type_system.h`
- `lib/backend/autodiff_codegen.cpp`
- `lib/backend/tensor_backward.cpp`
- `inc/eshkol/backend/tensor_backward.h`

`ad_node_t` already has tensor fields:

```c
void* tensor_value;
void* tensor_gradient;
struct ad_node* input3;
struct ad_node* input4;
void** saved_tensors;
size_t num_saved;
params union;
int64_t* shape;
size_t ndim;
```

`TypeSystem` maps those fields as AD-node fields 6 through 14.

`AutodiffCodegen::recordADNodeTensor` already exists. It allocates a node,
stores:

- op type,
- inputs 1 through 4,
- result tensor pointer,
- saved tensor pointer array,
- saved tensor count,
- params,
- shape,
- rank,
- tape membership.

`AutodiffCodegen::propagateGradient` has a tensor-gradient fast path: when a node
has `tensor_gradient != null`, it calls:

```c
eshkol_tensor_backward_dispatch(node_ptr)
```

### 1.4 Tensor backward kernels already present

Primary file:

- `lib/backend/tensor_backward.cpp`

Runtime backward support exists for several dense operations:

- Matmul:
  - `eshkol_backward_matmul`
  - delegates to `eshkol_matmul_backward_f64`
- Full tensor sum / mean.
- Transpose / reshape.
- Conv2d.
- MaxPool2d / AvgPool2d.
- BatchNorm / LayerNorm.
- Attention / multi-head attention.
- Positional encoding.
- Embedding.
- Selected bridge-backed tensor nodes.

This is enough runtime substrate to start using dense tensor adjoints. The main
problem is that codegen mostly does not route through this substrate.

### 1.5 Arena and tensor allocation substrate already present

Primary files:

- `lib/core/runtime_arena_core.cpp`
- `lib/core/runtime_tensor_alloc.cpp`
- `inc/eshkol/backend/memory_codegen.h`
- `lib/backend/memory_codegen.cpp`

Already present:

- arena creation/reset/scope management,
- aligned arena allocation,
- tensor allocation with header,
- full tensor allocation with dims/elements,
- tape allocation and tape node access,
- arena stats such as used/total/block count.

Missing:

- compiler-owned memory plan,
- scratch-buffer assignment,
- resident parameter/gradient buffers,
- staged kernel lifecycle,
- no-hot-loop-allocation enforcement.

## 2. Hard Blockers

### 2.1 `gradient` is not one-pass for tensor/vector inputs

Current conceptual shape in `AutodiffCodegen::gradient` tensor/reverse path:

```text
for i in 0..n:
    tape = arena_allocate_tape(arena, 1024)
    vars = create n AD variables
    ad_input = tensor containing vars
    output = f(ad_input)
    backpropagate(tape, output)
    result[i] = vars[i].gradient
```

This costs approximately:

```text
n user-function calls
n tape allocations
n backprops
```

Required shape for scalar-output loss:

```text
tape = arena_allocate_tape(arena, capacity)
vars = create n AD variables
ad_input = tensor containing vars
output = f(ad_input)
backpropagate(tape, output)
for i in 0..n:
    result[i] = vars[i].gradient
```

This costs:

```text
1 user-function call
1 tape allocation
1 backprop
```

This is the first surgical compiler fix.

### 2.2 `jacobian` is not row-swept

Current conceptual shape:

```text
for output row i:
  for input column j:
    tape = arena_allocate_tape(...)
    vars = create n variables
    y = F(vars)
    backpropagate(tape, y[i])
    J[i,j] = vars[j].gradient
```

Required minimal shape:

```text
for output row i:
    tape = arena_allocate_tape(...)
    vars = create n variables
    y = F(vars)
    backpropagate(tape, y[i])
    for input column j:
        J[i,j] = vars[j].gradient
```

Better later:

```text
tape = arena_allocate_tape(...)
vars = create n variables
y = F(vars)
for output row i:
    clear all node/variable gradients without erasing tape nodes
    seed y[i]
    backpropagate(tape, y[i])
    read all n variable gradients
```

The better form requires a runtime helper that clears gradients without resetting
tape `num_nodes`.

### 2.3 Dense tensor AD scalarizes instead of recording dense nodes

Current codegen commonly turns tensor AD into per-element scalar AD:

- same-shape elementwise ops create scalar AD nodes per element,
- matmul creates scalar multiply/add nodes in nested loops,
- reductions create scalar add/div chains,
- activations and softmax create scalar graphs,
- some conv/norm paths scalarize in AD mode.

This blocks dense backward kernels and makes tape size scale with scalar
operation count instead of tensor graph operation count.

Required shape:

```text
Forward:
  compute dense numeric tensor using existing optimized path
  record one AD_NODE_TENSOR_* node
  save needed tensors/metadata

Backward:
  dispatch one dense backward kernel
  accumulate dense gradients into input tensor nodes
```

### 2.4 Matmul dense tensor backward exists but is bypassed

Runtime dense matmul backward exists.

Codegen issue:

- There is a dense `recordADNodeTensor` matmul callsite in `lib/backend/llvm_codegen.cpp`.
- The AD-mode matmul branch constructs scalarized AD computation first.
- The later tensor-recording block is guarded so it is skipped when the AD-mode
  branch exists.

Required:

- Use normal dense numeric matmul forward in AD mode.
- Record one `AD_NODE_MATMUL` tensor node.
- Use `eshkol_backward_matmul` during reverse.
- Remove or strictly fallback-gate the scalarized AD matmul loop.

### 2.5 Tensor backward dispatch can silently drop gradients

`eshkol_tensor_backward_dispatch` has default/skipped cases that do not propagate
gradients.

This is dangerous. Once codegen records more tensor nodes, an unimplemented
backward must fail loudly or fallback before recording the node.

Required:

```text
strict mode / tests:
  unsupported tensor AD node -> error

release fallback mode:
  unsupported tensor AD node -> warn once only if compiler explicitly selected fallback

never:
  silently return zero/missing gradients
```

### 2.6 Hidden finite differences remain in compiler AD paths

Finite-difference paths observed:

- `gradientHigherOrder` uses central differences.
- General vector/tensor Hessian fallback uses epsilon-based gradient differences.
- Scheme tape has explicit `record-fd-op!` for custom scalar operations.

Policy:

- Explicit numerical finite-difference APIs may remain.
- Hidden finite-difference fallback inside `gradient`, `hessian`, `laplacian`,
  or SciML/PINN operators must not be the default.
- For supported compiler-differentiated programs, use exact AD.
- For unsupported exact AD, return a clear unsupported error.

## 3. Compiler Contract To Add

The compiler/runtime should expose a staged path equivalent in shape to:

```python
loss_value, grads = jax.value_and_grad(loss)(params, batch)
```

Eshkol target:

```text
compiled = eshkol_compile_staged_value_grad(loss, param_descs, input_descs, static_config)
compiled.run(params, grads, inputs, outputs, scratch)
```

Required execution contract:

```text
one primal loss evaluation
one reverse pass for scalar loss
all parameter gradients returned together
dense tensor adjoints for supported primitives
resident parameter and gradient buffers
fixed scratch plan for static shapes
no hidden finite differences
no per-parameter tape rebuild
no scalarized matmul/reduction on supported dense tensors
explicit unsupported errors
```

For a scalar loss over `P` trainable parameters, acceptable counters should show:

```text
primal_calls_per_step = 1
reverse_passes_per_step = 1
finite_difference_evals = 0
tape_allocations = O(1)
tensor_ad_nodes proportional to tensor graph ops
scalar_ad_nodes not proportional to matmul inner-loop scalar operations
```

## 4. Patch Plan

Do the work in this order.

## Phase A — Counters and one-pass reverse gradient

### A.1 Add AD counters first

Target files:

- `lib/core/runtime_autodiff.cpp`
- `inc/eshkol/eshkol.h` or a new runtime diagnostics header
- `lib/backend/autodiff_codegen.cpp`

Add counters:

```c
typedef struct {
    uint64_t primal_calls;
    uint64_t reverse_passes;
    uint64_t forward_jvp_passes;
    uint64_t tape_allocations;
    uint64_t tape_nodes;
    uint64_t scalar_ad_nodes;
    uint64_t tensor_ad_nodes;
    uint64_t tensor_backward_dispatches;
    uint64_t tensor_backward_unsupported;
    uint64_t finite_difference_evals;
    uint64_t arena_allocations;
    uint64_t arena_bytes;
    uint64_t scratch_bytes;
    uint64_t compile_count;
    uint64_t cache_hits;
} EshkolADCounters;

void eshkol_ad_counters_reset(void);
void eshkol_ad_counters_get(EshkolADCounters* out);
```

Expose Scheme builtins later if useful:

```scheme
(ad-reset-counters!)
(ad-counters)
```

Counters are not optional. They define whether the compiler really moved from
component-loop AD to one-pass AD.

### A.2 Refactor `AutodiffCodegen::gradient`

Target files:

- `lib/backend/autodiff_codegen.cpp`
- `inc/eshkol/backend/autodiff_codegen.h`

Add internal helpers:

```cpp
struct ADInputBundle {
    llvm::Value* tape;
    llvm::Value* var_nodes;       // ad_node_t**
    llvm::Value* ad_input_tagged; // tensor/scalar/structured input
    llvm::Value* n;
    llvm::Value* shape;
    llvm::Value* ndim;
};

ADInputBundle buildADTensorInputFromTensor(llvm::Value* tensor_tagged);
ADInputBundle buildADScalarInput(llvm::Value* scalar_tagged);
llvm::Value* callFunctionWithADInputs(llvm::Function* fn, ADInputBundle& inputs, CapturePlan& captures);
llvm::Value* extractScalarOutputADNode(llvm::Value* output_tagged);
llvm::Value* allocateGradientResultLikeInput(ADInputBundle& inputs);
void emitReadAllVariableGradients(ADInputBundle& inputs, llvm::Value* result_tensor);
```

Then replace the tensor/reverse component loop with:

```cpp
ADInputBundle b = buildADTensorInputFromTensor(point);

pushTapeContext(b.tape);
setAdModeActive(true);
llvm::Value* output = callFunctionWithADInputs(func, b, captures);
setAdModeActive(false);
popTapeContext();

llvm::Value* output_node = extractScalarOutputADNode(output);
backpropagate(b.tape, output_node);

llvm::Value* grad_result = allocateGradientResultLikeInput(b);
emitReadAllVariableGradients(b, grad_result);
return grad_result;
```

Important:

- Preserve existing scalar exact fast paths.
- Preserve nested AD behavior.
- Preserve capture handling, but move it behind helper functions.
- Do not use pointer-range heuristics for AD-node detection.

### A.3 Add a typed AD tensor unwrap convention

Current tensor element code uses pointer heuristics in places. Replace with a
clear convention.

Recommended convention:

```text
Plain tensor:
  tagged base type = ESHKOL_VALUE_HEAP_PTR, heap subtype tensor

AD scalar node:
  tagged base type = ESHKOL_VALUE_CALLABLE, ad_node_t with tensor_value == null

AD tensor node:
  tagged base type = ESHKOL_VALUE_CALLABLE, ad_node_t with tensor_value != null
```

Add helper:

```cpp
struct ADTensorOperand {
    llvm::Value* numeric_tensor_ptr; // tensor_value for AD tensor, else heap tensor
    llvm::Value* ad_node_ptr;        // null for plain tensor
    llvm::Value* is_ad_tensor;
};

ADTensorOperand unwrapTensorOrADTensor(llvm::Value* tagged);
```

All tensor op codegen should use this helper.

### A.4 Add gradient clearing without tape reset

Target files:

- `lib/core/runtime_autodiff.cpp`
- `inc/eshkol/backend/memory_codegen.h`
- `lib/backend/memory_codegen.cpp`

Current `arena_tape_reset` zeroes gradients and sets `num_nodes = 0`. That is not
enough for repeated backward over a retained graph.

Add:

```c
void arena_tape_zero_gradients(ad_tape_t* tape);
void arena_tape_zero_gradients_with_vars(ad_tape_t* tape, ad_node_t** vars, size_t n);
```

Variables currently are not appended to the tape. Either pass `vars` explicitly
or start using `ad_tape_t::variables` / `num_variables` consistently.

### A.5 Tests for Phase A

Add:

```text
tests/ad/value_grad_one_pass_test.esk
tests/ad/tensor_gradient_one_pass_test.esk
```

Test shape:

```scheme
;; f(w) = sum_i w_i^2
;; grad = 2w
```

Assertions:

```text
loss correct
gradient correct
primal_calls = 1
tape_allocations = 1
reverse_passes = 1
finite_difference_evals = 0
```

For a small matrix loss:

```scheme
loss(W, x) = tensor-sum((matmul W x) * (matmul W x))
```

Compare with analytic gradient.

## Phase B — Row-sweep Jacobian

Target file:

- `lib/backend/autodiff_codegen.cpp`

Change `AutodiffCodegen::jacobian` from `(i,j)` replay to row replay.

Minimal shape:

```cpp
for each output row i:
    ADInputBundle b = buildADTensorInputFromTensor(point);
    output = callFunctionWithADInputs(func, b, captures);
    output_i_node = extractOutputComponentADNode(output, i);
    backpropagate(b.tape, output_i_node);
    for each input j:
        J[i,j] = loadNodeGradient(var_nodes[j]);
```

Better shape after `arena_tape_zero_gradients`:

```cpp
ADInputBundle b = buildADTensorInputFromTensor(point);
output = callFunctionWithADInputs(func, b, captures);
for each output row i:
    arena_tape_zero_gradients_with_vars(b.tape, vars, n);
    output_i_node = extractOutputComponentADNode(output, i);
    backpropagate(b.tape, output_i_node);
    read row i;
```

Add:

```text
tests/ad/jacobian_row_sweep_test.esk
```

Assertions:

```text
analytic Jacobian correct
primal_calls = m or 1 + m, not m*n
finite_difference_evals = 0
```

## Phase C — Dense tensor AD nodes

This is the key performance phase.

### C.1 Fix matmul routing first

Target files:

- `lib/backend/llvm_codegen.cpp`
- `lib/backend/tensor_reduce_codegen.cpp`
- `lib/backend/tensor_backward.cpp`

Required AD-mode behavior:

```text
1. Compute numeric dense C = matmul(A, B) through existing optimized path.
2. Record one AD_NODE_MATMUL tensor node.
3. Save A and B element pointers.
4. Store params M, K, N.
5. Store output shape [M, N].
6. Return a value that downstream tensor code can identify as an AD tensor.
7. Backward uses eshkol_backward_matmul.
```

Do not create scalar multiply/add AD nodes for supported dense matmul.

Also add overflow checks in matmul backward allocation:

```text
M*K
K*N
M*N
```

mirror existing conv safety helpers.

Add:

```text
tests/ad/tensor_matmul_dense_ad_test.esk
```

Assertions:

```text
analytic dA, dB correct
one tensor matmul AD node recorded
matmul tensor backward dispatch called
scalar AD multiply/add nodes from matmul = 0 or explicitly fallback-counted
```

### C.2 Route full `tensor-sum` and `tensor-mean` through dense tensor nodes

Target files:

- `lib/backend/tensor_reduce_codegen.cpp`
- `lib/backend/tensor_backward.cpp`

Required behavior:

```text
AD_NODE_SUM:
  input1 = source tensor AD node
  tensor_value = scalar/1-element output tensor
  params[0] = input_total_elements

AD_NODE_MEAN:
  same, with mean scaling
```

Backward already exists conceptually:

```text
sum:  dT_i += upstream_scalar
mean: dT_i += upstream_scalar / n
```

Add:

```text
tests/ad/tensor_reduction_dense_ad_test.esk
```

Example:

```scheme
loss(T) = tensor-sum(T * T)
```

Expected:

```text
grad(T) = 2T
```

### C.3 Add dense same-shape elementwise tensor nodes

Target files:

- `inc/eshkol/eshkol.h`
- `lib/backend/tensor_arith_codegen.cpp`
- `lib/backend/tensor_backward.cpp`
- `inc/eshkol/backend/tensor_backward.h`

Add explicit dense tensor op IDs. Do not rely on scalar op IDs when dispatching
tensor gradients.

Example enum additions:

```c
AD_NODE_TENSOR_ADD_DENSE,
AD_NODE_TENSOR_SUB_DENSE,
AD_NODE_TENSOR_MUL_DENSE,
AD_NODE_TENSOR_DIV_DENSE,
```

Backward rules:

```text
Y = A + B:
  dA += dY
  dB += dY

Y = A - B:
  dA += dY
  dB -= dY

Y = A * B:
  dA += dY * B
  dB += dY * A

Y = A / B:
  dA += dY / B
  dB -= dY * A / (B * B)
```

Add:

```text
tests/ad/tensor_elementwise_dense_ad_test.esk
```

### C.4 Add dense broadcast backward

Target files:

- `lib/backend/tensor_arith_codegen.cpp`
- `lib/backend/tensor_backward.cpp`

Broadcast backward must reduce over broadcasted axes.

Example:

```text
Y = A + b
A shape = [M, N]
b shape = [N]
Y shape = [M, N]

dA[m,n] += dY[m,n]
db[n]   += sum_m dY[m,n]
```

Add metadata sufficient to compute input reductions:

```c
typedef struct {
    uint32_t rank_a;
    uint32_t rank_b;
    uint32_t rank_y;
    uint64_t dims_a[8];
    uint64_t dims_b[8];
    uint64_t dims_y[8];
    uint64_t strides_a[8];
    uint64_t strides_b[8];
    uint64_t strides_y[8];
} EshkolBroadcastBackwardParams;
```

Add:

```text
tests/ad/tensor_broadcast_ad_test.esk
```

### C.5 Make unsupported tensor backward strict

Target file:

- `lib/backend/tensor_backward.cpp`

Required behavior:

```c
if unsupported tensor node type:
    if strict mode:
        error/fail loudly
    else:
        warn once only when compiler deliberately selected fallback
```

Add env flag:

```text
ESHKOL_AD_STRICT_TENSOR=1
```

Tests should run strict.

Add:

```text
tests/ad/tensor_unsupported_backward_test.esk
```

## Phase D — First-class `value-and-grad`

Only add this after Phase A one-pass gradient works.

Target files:

- `lib/backend/autodiff_codegen.cpp`
- `inc/eshkol/backend/autodiff_codegen.h`

Add internal primitive:

```cpp
llvm::Value* AutodiffCodegen::valueAndGradient(const eshkol_operations_t* op);
```

Required scalar-loss behavior:

```text
1. Build AD variables for all trainable parameter leaves.
2. Treat batch/input tensors as non-trainable unless requested.
3. Run loss once.
4. Preserve primal loss value.
5. Backprop once.
6. Read all parameter gradients.
7. Return value and gradient structure.
```

Start with flat parameter leaves, not arbitrary PyTree ergonomics.

Suggested internal leaf descriptor:

```c
typedef struct {
    const char* name;
    uint32_t dtype;
    uint32_t rank;
    const uint64_t* dims;
    uint64_t total_elements;
    double* value;
    double* grad;
} EshkolParamLeaf;
```

Add:

```text
tests/ad/value_and_grad_test.esk
```

Example:

```scheme
loss(params) = sum(W * W) + sum(b * b)
```

Expected:

```text
grad_W = 2W
grad_b = 2b
primal_calls = 1
reverse_passes = 1
finite_difference_evals = 0
```

## Phase E — Exact coordinate derivatives for PINNs

This phase is needed for SciML/PINN workloads.

Problem shape:

```text
u(params, x, y, z)
need:
  u
  ux, uy, uz
  uxx, uyy, uzz
  laplacian = uxx + uyy + uzz
then:
  loss(params) = residual/boundary loss
  grad_params = reverse(loss, params)
```

Required compiler mode:

```text
small coordinate variables:
  exact forward/Taylor/JVP derivatives

large trainable tensors:
  reverse-mode adjoints

scalar loss:
  one reverse pass to params
```

No finite differences.

A manually implemented proof-of-shape exists in the experiments workspace:

```text
phydra-esk/poisson3d_mlp_pinn_sdf_stan20_w20d6_fast_ad_500.esk
```

That program manually propagates coordinate jets through the MLP and performs one
reverse sweep over parameters. The compiler should make that pattern native.

Suggested compiler descriptor:

```c
typedef struct {
    uint32_t num_coord_inputs;
    const char** coord_names;
    uint32_t max_order;
    uint32_t requested_mask;
    /* e.g. value, dx, dy, dz, dxx, dyy, dzz */
} EshkolCoordDerivativeSpec;
```

Compiler behavior:

```text
for each collocation point:
    propagate coordinate Taylor/JVP state:
      value, first derivatives, selected second derivatives
    compute PDE residual exactly
accumulate scalar loss
reverse once to params
```

Add:

```text
tests/ad/pinn_coord_derivatives_exact_test.esk
```

Analytic test function:

```text
u(x,y,z) = x^2 + y^3 + sin(z)
laplacian = 2 + 6y - sin(z)
```

Assertions:

```text
ux, uy, uz correct
uxx, uyy, uzz correct
laplacian correct
param gradients correct for a parameterized small form
finite_difference_evals = 0
```

## Phase F — Remove/contain hidden finite differences

Target files:

- `lib/backend/autodiff_codegen.cpp`
- `lib/core/ad/tape.esk` documentation/API boundaries

Policy:

```text
(gradient f x)      -> exact AD or unsupported error
(hessian f x)       -> exact AD or unsupported error
(laplacian f x)     -> exact AD or unsupported error
PINN derivatives    -> exact AD or unsupported error

finite differences  -> explicit numeric API only
```

`gradientHigherOrder` options:

1. Preferred: route generated closure to exact direct-gradient machinery.
2. Temporary: rename/document as finite-difference helper and make exact
   `(gradient f)` closure unsupported until implemented.

Hessian options:

- Keep scalar/direct multiparam exact paths.
- Replace vector/tensor epsilon fallback with exact mixed-mode or unsupported.
- For PINNs, first implement exact coordinate Hessian diagonal path rather than
  full dense Hessian for all tensors.

Add:

```text
tests/ad/no_hidden_finite_difference_test.esk
```

Assertions:

```text
supported AD cases pass with finite_difference_evals = 0
unsupported exact cases fail loudly
no default hidden finite differences
```

## Phase G — Staged kernel ABI

This is the compiler/runtime boundary needed for compile-once/run-many training.

### G.1 Add a pointer/out-param C ABI

Do not pass tagged structs by value across the C ABI. Use descriptors and raw
pointers.

Add a new header:

```text
inc/eshkol/backend/staged_kernel.h
```

or a dedicated section in:

```text
inc/eshkol/llvm_backend.h
```

Recommended ABI:

```c
typedef struct EshkolStagedKernel EshkolStagedKernel;

typedef enum {
    ESHKOL_DTYPE_F64 = 1,
} EshkolDType;

typedef struct {
    EshkolDType dtype;
    uint32_t rank;
    uint64_t dims[8];
    uint64_t strides[8];
    uint64_t total_elements;
    uint32_t flags; /* contiguous, readonly, trainable, batch, etc. */
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
    const double* const* inputs;
    double* const* params;
    double* const* grads;
    double* const* outputs;
    void* scratch;
    uint64_t scratch_bytes;
    uint64_t step;
} EshkolKernelCall;

typedef struct {
    uint64_t invocations;
    uint64_t compile_ns;
    uint64_t run_ns;
    uint64_t arena_bytes_high_water;
    uint64_t scratch_bytes;
    uint64_t tape_nodes;
    uint64_t scalar_ad_nodes;
    uint64_t tensor_ad_nodes;
    uint64_t fallback_count;
    uint64_t finite_difference_evals;
} EshkolKernelCounters;

typedef enum {
    ESHKOL_KERNEL_OK = 0,
    ESHKOL_KERNEL_BAD_SHAPE = 1,
    ESHKOL_KERNEL_BAD_DTYPE = 2,
    ESHKOL_KERNEL_UNSUPPORTED_OP = 3,
    ESHKOL_KERNEL_UNSUPPORTED_AD = 4,
    ESHKOL_KERNEL_SCRATCH_TOO_SMALL = 5,
    ESHKOL_KERNEL_RUNTIME_ERROR = 6,
} EshkolKernelStatus;

int eshkol_compile_staged_value_grad(
    const eshkol_ast_t* asts,
    size_t num_asts,
    const EshkolKernelSignature* sig,
    EshkolStagedKernel** out);

int eshkol_staged_kernel_run(
    EshkolStagedKernel* kernel,
    const EshkolKernelCall* call);

int eshkol_staged_kernel_get_counters(
    EshkolStagedKernel* kernel,
    EshkolKernelCounters* out);

void eshkol_staged_kernel_destroy(EshkolStagedKernel* kernel);
```

### G.2 First supported staged subset

Start narrow:

```text
dtype:
  f64 only

shapes:
  static rank/dims
  contiguous row-major

control flow:
  straight-line and simple shape-static loops only

ops:
  elementwise add/sub/mul/div
  full tensor sum/mean
  matmul
  tanh/sigmoid/Stan activation if needed
  scalar arithmetic around loss

AD:
  scalar loss
  gradients w.r.t. param buffers
  coordinate JVP/Taylor path for selected small coordinate variables

unsupported:
  dynamic rank
  arbitrary closure capture in staged kernels
  mutation escaping kernel
  unsupported tensor backward
  finite-difference fallback
```

### G.3 Generated LLVM wrapper shape

Generate a wrapper like:

```llvm
define i32 @__eshkol_staged_vg_<hash>(
    ptr %runtime,
    ptr %params,       ; double**
    ptr %grads,        ; double**
    ptr %inputs,       ; double**
    ptr %outputs,      ; double**
    ptr %scratch,
    i64 %scratch_bytes,
    i64 %step
)
```

Wrapper responsibilities:

1. Validate shape/dtype specialization key.
2. Bind tensor descriptors to raw buffers.
3. Clear gradient buffers according to policy.
4. Run primal loss.
5. Run reverse pass.
6. Write loss/output values.
7. Optionally run optimizer update.
8. Fill counters/status.
9. Return explicit status code.

Add:

```text
tests/staged/staged_value_grad_test.esk
```

## Phase H — Static shape specialization

Target files:

- `lib/backend/llvm_codegen.cpp`
- `lib/backend/tensor_*_codegen.cpp`
- `inc/eshkol/backend/tensor_codegen.h`
- staged kernel files

Add internal shape structs:

```cpp
struct TensorShapeSpec {
    DType dtype;
    uint32_t rank;
    std::array<uint64_t, 8> dims;
    std::array<uint64_t, 8> strides;
    uint64_t total_elements;
    bool contiguous;
    bool static_shape;
};

struct StagedTensorValue {
    llvm::Value* data_ptr;
    TensorShapeSpec shape;
    llvm::Value* runtime_tensor_view;
    llvm::Value* ad_node;
};
```

Specialization key should include:

```text
source/function identity
compiler version
LLVM version
target triple
optimization level
dtype/rank/dims/strides for every param/input/output
supported-op set version
AD mode/staging flags
```

Runtime guard:

```text
if runtime shape != compiled shape:
    return ESHKOL_KERNEL_BAD_SHAPE
```

No silent reinterpretation.

## Phase I — Kernel memory plan

Current tensor codegen allocates tensors/elements inside every op. Staged kernels
need planned scratch memory.

Add:

```cpp
struct TensorTemp {
    uint32_t id;
    TensorShapeSpec shape;
    uint64_t bytes;
    uint64_t alignment;
    OpId producer;
    std::vector<OpId> consumers;
    uint32_t first_use;
    uint32_t last_use;
    bool may_alias_input;
    bool mutable_in_place_ok;
};

struct KernelMemoryPlan {
    std::vector<TensorTemp> temps;
    uint64_t scratch_bytes;
    std::unordered_map<uint32_t, uint64_t> temp_offsets;
};
```

Staged lowering policy:

```text
Do not call arena_allocate_tensor_full inside hot tensor ops.
Use preallocated scratch offsets.
Represent internal tensors as views over scratch/resident buffers.
Save only needed forward intermediates for backward.
Reuse scratch slots when lifetimes do not overlap.
```

Add:

```text
tests/staged/no_hot_loop_arena_alloc_test.esk
```

Assertions:

```text
kernel runs 1000 times
scratch bytes stable
arena used bytes stable after warmup
no per-run tensor allocations in counters
```

## Phase J — Staged optimizer/update step

After `value-and-grad`, add a train step.

Start with simple SGD/Rprop-like update, not full Optax.

Kernel shape:

```text
compiled_train_step(params, opt_state, batch, static_config)
  -> loss, updated params in-place, updated opt_state, metrics
```

Initial epilogue:

```text
for each param leaf:
    param -= lr * grad
```

Then add:

- weight decay,
- Adam first/second moment state,
- Rprop sign update,
- optional line search later.

State descriptor:

```c
typedef struct {
    uint32_t num_state_buffers;
    EshkolTensorDesc* state_descs;
    double** state_buffers;
} EshkolOptimizerState;
```

Add:

```text
tests/staged/staged_sgd_step_test.esk
```

Assertions:

```text
loss decreases over repeated calls
params move toward analytic target
no recompilation
no growing arena memory
gradients match analytic first step
```

## 5. File-by-File Change Map

### `lib/backend/autodiff_codegen.cpp`

Change:

- Refactor `AutodiffCodegen::gradient` tensor/reverse path to one-pass.
- Refactor `AutodiffCodegen::jacobian` to row-sweep.
- Remove/replace hidden vector/tensor finite-difference Hessian fallback.
- Remove/replace finite-difference `gradientHigherOrder` default.
- Add `valueAndGradient`.
- Add AD tensor unwrap helper.
- Add structured gradient readback helper.
- Add counter increments.

### `inc/eshkol/backend/autodiff_codegen.h`

Add declarations for:

```cpp
valueAndGradient(...)
buildADTensorInputFromTensor(...)
extractScalarOutputADNode(...)
emitReadAllVariableGradients(...)
unwrapTensorOrADTensor(...)
```

### `lib/core/runtime_autodiff.cpp`

Add:

```c
arena_tape_zero_gradients(...)
arena_tape_zero_gradients_with_vars(...)
eshkol_ad_counters_reset(...)
eshkol_ad_counters_get(...)
```

Consider using `ad_tape_t::variables` consistently.

### `inc/eshkol/eshkol.h`

Add explicit dense tensor AD node IDs for supported dense primitives.

Do not record op IDs whose backward defaults to no-op.

### `lib/backend/tensor_backward.cpp`

Change:

- Matmul overflow checks.
- Strict unsupported tensor AD behavior.
- Dense elementwise backward.
- Broadcast backward.
- Axis-reduction backward later.
- Warn/error once for unsupported node type.

### `inc/eshkol/backend/tensor_backward.h`

Add declarations for new dense backward kernels.

### `lib/backend/tensor_arith_codegen.cpp`

Change:

- Same-shape AD path records one dense tensor node.
- Broadcast path becomes AD-aware.
- Staged mode writes into provided output buffers.

### `lib/backend/tensor_reduce_codegen.cpp`

Change:

- Full sum/mean record dense reduction nodes.
- Matmul/dot avoid scalarization for supported dense shapes.
- Axis reductions either implement proper backward or fail explicitly.

### `lib/backend/llvm_codegen.cpp`

Change:

- `codegenMatmul` must not bypass dense `recordADNodeTensor` in AD mode.
- Add staged wrapper emission.
- Add shape guard and staged counter hooks.

### `inc/eshkol/llvm_backend.h`

Add staged compile/run APIs or include `inc/eshkol/backend/staged_kernel.h`.

### `inc/eshkol/backend/memory_codegen.h` and `lib/backend/memory_codegen.cpp`

Declare new runtime helpers for:

- tape gradient clearing,
- AD counters,
- staged scratch helpers if needed.

### New files recommended

```text
inc/eshkol/backend/staged_kernel.h
lib/backend/staged_kernel_codegen.cpp
lib/core/runtime_staged_kernel.cpp
```

Keep these narrow. Do not start with a broad framework.

## 6. Tests To Add

Minimum high-signal suite:

```text
tests/ad/value_grad_one_pass_test.esk
tests/ad/tensor_gradient_one_pass_test.esk
tests/ad/jacobian_row_sweep_test.esk
tests/ad/tensor_matmul_dense_ad_test.esk
tests/ad/tensor_reduction_dense_ad_test.esk
tests/ad/tensor_elementwise_dense_ad_test.esk
tests/ad/tensor_broadcast_ad_test.esk
tests/ad/tensor_unsupported_backward_test.esk
tests/ad/no_hidden_finite_difference_test.esk
tests/ad/pinn_coord_derivatives_exact_test.esk
tests/staged/staged_value_grad_test.esk
tests/staged/no_hot_loop_arena_alloc_test.esk
tests/staged/staged_sgd_step_test.esk
```

Existing regression tests to keep green:

```text
tests/ad/taylor_tower_test.esk
tests/ad/taylor_tower_mono_test.esk
tests/ad/tensor_tower_test.esk
tests/ad/reverse_over_taylor_test.esk
tests/ad/checkpointed_reverse_test.esk
tests/ad/mixed_mode_ad_test.esk
tests/ad/nested_ad_test.esk
tests/ad/guw_multivariate_test.esk
```

Experiments-side smoke after compiler work:

```text
phydra-esk/poisson3d_mlp_pinn_sdf_stan20_w20d6_fast_ad_500.esk
```

This file is important because it manually demonstrates the target compiler
shape: coordinate jets forward, one reverse sweep over parameters, no finite
differences.

## 7. What Not To Do First

Do not prioritize:

1. More Phydra wrapper parity. The wrapper layer is not the bottleneck.
2. More scalar Taylor examples only. Scalar/Taylor AD is already substantially
   improved.
3. Arbitrary PyTree ergonomics before flat parameter leaves work.
4. Bytecode VM redesign as the staged tensor ABI. The current VM API is not the
   right pointer/out-param tensor-kernel boundary.
5. Hidden fallback to finite differences.
6. Broad tensor-node recording before unsupported tensor backward is strict.
7. Scalarized matmul as the “temporary” performance path. Scalarized matmul is
   the current bottleneck.

## 8. Definition Of Done For First Serious Phydra Benchmark

A useful first milestone is complete when this shape works:

```text
params = {W1,b1,W2,b2,...}
batch = coordinate/collocation tensors
loss = PINN residual loss with exact coordinate derivatives

compiled = eshkol_compile_staged_value_grad(loss, static_shapes)

repeat N:
    compiled.run(params, grads, batch, scratch)
    optimizer_update(params, grads, opt_state)
```

Counters should show:

```text
compile_count = 1
finite_difference_evals = 0
primal_calls_per_step = 1
reverse_passes_per_step = 1
matmul_backward_dispatches > 0 for MLPs
scalar_ad_nodes_from_matmul = 0
arena_bytes_after_warmup stable
scratch_bytes fixed by shape key
```

Correctness should show:

```text
small analytic gradients match closed form
dense tensor gradients match scalar AD oracle on tiny shapes
coordinate Laplacian matches analytic references
unsupported exact AD paths error loudly
existing Taylor/reverse-over-Taylor tests remain green
```

## 9. Summary For Maintainer

Eshkol already has the hard semantic pieces: scalar reverse AD, forward-mode
nesting, Taylor towers, reverse-over-Taylor, tensor objects, arena allocation,
and several dense tensor backward kernels.

The next compiler work should be surgical:

1. Stop replaying the loss per input component in `gradient`.
2. Stop replaying the function per Jacobian entry.
3. Route dense matmul/reduction/elementwise/broadcast operations through tensor
   AD nodes and dense backward kernels.
4. Make unsupported tensor backward strict.
5. Remove hidden finite-difference fallbacks from default compiler AD paths.
6. Add `value-and-grad`.
7. Add a pointer/out-param staged kernel ABI with static shape specialization.
8. Add scratch memory planning and resident parameter/gradient buffers.
9. Add exact coordinate derivative staging for PINNs.
10. Add optimizer/update epilogues only after value+grad is correct.

That is the path from current Eshkol AD semantics to a Phydrax/JAX-like training
substrate. The key deliverable is not a broader public wrapper API; it is a
compiled, exact, tensor-aware, one-pass `value + grad + update` execution model.
