# ADR 0005: Lambda foundations to resident programs-as-weights

- **Status:** Proposed
- **Date:** 2026-07-09
- **Decision owners:** Eshkol compiler, AD, and SDNC maintainers
- **Scope:** Binary Lambda Calculus, homoiconic compilation, automatic
  differentiation, the VM-as-transformer, and resident training

## Decision

Eshkol will treat “programs to weights” as the construction of a **resident
program capsule**, not as unconstrained training of the interpreter and not as
a claim that a list-shaped syntax tree is differentiable.

A capsule has three disjoint parameter partitions:

1. `theta_sem`: analytically constructed, immutable weights implementing the
   bounded VM small-step semantics;
2. `psi_program`: an immutable, program-specific embedding/constant table
   compiled from canonical code and resident in the model as a weight tensor;
3. `phi_plastic`: explicitly declared trainable weights—adapters, learned
   memory, or typed program holes—that may change without changing the meaning
   of the VM.

Execution is a recurrent transformer transition

```text
s[t+1] = F(theta_sem, psi_program, phi_plastic, s[t], input[t])
```

and learning is a staged, shape-specialized kernel over a bounded rollout:

```text
(loss, dphi, next_resident_state) =
    staged_value_and_grad(loss_of_rollout,
                          phi_plastic,
                          frozen={theta_sem, psi_program},
                          state, batch, scratch)
```

Homoiconicity owns discrete program construction and revision. De Bruijn form
owns binding identity and canonicalization. The AD tower owns numeric
derivatives. The staged `value_and_grad` kernel owns efficient repeated
training. None substitutes for another.

This is a target architecture. The BLC evaluator and the analytical
VM-as-transformer exist today; the general staged `value_and_grad` kernel and
the source/ESKB-to-resident-capsule bridge do not.

## Why this decision is necessary

The repository contains all of the edges of the intended triangle, but they do
not yet meet at one semantic boundary.

### The lambda foundation is real and canonicalizable

`core.blc` represents variables, abstractions, and applications as ordinary
S-expressions using **1-based De Bruijn indices** and defines Tromp's
self-delimiting encoding (`lib/core/blc.esk:7-19`). Its constructors produce
plain lists (`lib/core/blc.esk:66-80`), its structural validator checks the
three term forms (`lib/core/blc.esk:82-90`), and its encoder/decoder implement
the bit grammar directly (`lib/core/blc.esk:99-165`).

The evaluator is not an informal example. Capture-avoiding beta reduction is
implemented by the standard shift/substitute construction
`shift(-1, 1, subst(B, 1, shift(1, 1, N)))`
(`lib/core/blc.esk:168-210`). `blc-step` chooses the leftmost outermost redex,
then descends under lambdas to reach full beta normal form
(`lib/core/blc.esk:212-244`). Evaluation repeats that deterministic step with a
divergence bound (`lib/core/blc.esk:246-261`). The test that `K I Omega`
terminates while its discarded argument diverges is an executable witness that
the contract is normal order rather than Eshkol's ordinary eager call behavior
(`tests/features/blc_test.esk:111-127`).

De Bruijn form matters beyond compactness. It removes alpha-renaming from
program identity: two alpha-equivalent closed lambda terms have the same
canonical tree and BLC bit string. That gives the weight compiler a stable
content key, a stable cache key, and a binding representation whose variable
lookup is an integer operation rather than a name comparison.

### Eshkol has a code-data-code path

The LLVM backend turns AST nodes back into quoted tagged values
(`lib/backend/homoiconic_codegen.cpp:65-146`) and reconstructs calls and lambda
forms as cons-cell S-expressions (`lib/backend/homoiconic_codegen.cpp:423-528`,
`lib/backend/homoiconic_codegen.cpp:576-639`). In the other direction, the
runtime converter maps cons-cell S-expressions directly to AST nodes without a
serialize/parse round trip, explicitly to support metaprogramming and
self-modifying code (`lib/core/sexp_to_ast.cpp:6-10`). It recognizes lambda,
binding, mutation, control, and quotation forms and otherwise constructs an
application (`lib/core/sexp_to_ast.cpp:984-1116`); the public entry point is
`eshkol_sexp_to_ast` (`lib/core/sexp_to_ast.cpp:1203-1216`).

This makes reflective revision possible, but it does **not** make cons-cell
topology differentiable. Reflection can propose a new program; AD can optimize
numeric leaves exposed by a fixed program. Crossing that boundary requires a
deliberate relaxation or a compile-and-verify cycle.

### The current VM artifact compiles an interpreter, not each program

The early weight compiler states the distinction exactly: the transformer
implements the interpreter, programs are input tokens, and one weight set
handles all programs (`lib/backend/weight_compiler.c:2-26`). The production
artifact follows that architecture. Its state is a 256-lane VM register file,
AD tape, and bounded arena (`lib/backend/weight_matrices.c:140-269`); each
instruction becomes a position/opcode/operand row
(`lib/backend/weight_matrices.c:350-359`).

`generate_weights` analytically wires attention and gated FFNs to reproduce the
VM transition (`lib/backend/weight_matrices.c:3610-3637`). One matrix forward
applies the attention/FFN stack to one state transition
(`lib/backend/weight_matrices.c:5506-5552`), while the runner constructs the
instruction rows and loops until halt (`lib/backend/weight_matrices.c:5555-5633`).
The artifact validates reference C, simulated layers, and actual matrix
execution as three separate paths (`lib/backend/weight_matrices.c:5667-5700`),
and it can load compiler-produced ESKB instructions into those rows
(`lib/backend/weight_matrices.c:7368-7446`).

Therefore the shipped map is currently:

```text
ISA -------------------------------> theta_sem
program.esk -> ESKB -> instruction rows -> runtime activations
```

It is not yet:

```text
program.esk -> one self-contained, program-specific weight artifact
```

The decision makes the latter precise by promoting the compiled instruction
row table to a resident embedding-table parameter `psi_program`. Selecting its
rows by fixed position IDs is an embedding layer, so the rows are literally a
model weight tensor; no program prompt has to be injected by a host on every
run. When `theta_sem` is frozen, Layer-0 K/V rows may also be precomputed and
stored in the capsule without changing execution.

The current QLMW export writes the `InterpreterWeights` structure and model
dimensions, but no program rows (`lib/backend/weight_matrices.c:5635-5657`).
ESKB and QLMW should remain independently inspectable; a capsule may be a
manifest-bound directory or a new outer container, but it must hash both plus
the derived `psi_program` rather than quietly changing either existing format.

### The current `core.sdnc` API is a differentiability prototype, not that bridge

The Eshkol-visible API exposes a forward, a flat gradient, parameter
get/set, and an SGD loop (`lib/core/sdnc.esk:2-30`). Its canonical flat order is
well defined (`lib/core/sdnc_api.c:70-117`), and `sdnc-weight-grad` performs a
real forward and analytic backward for a one-step squared-error objective
(`lib/core/sdnc_api.c:280-329`). The shared core can propagate an adjoint back
to the input state, which is the primitive needed for recurrent backpropagation
(`lib/core/sdnc_core.h:196-205`, `lib/core/sdnc_core.h:350-355`).

But `sdnc-program` does not compile its name. It hashes the name into a
name-seeded random position-embedding bank
(`lib/core/sdnc_api.c:211-252`), and the weight set it allocates is a
small-random initialization with only two active FFN layers, not the analytical
VM produced by `generate_weights`
(`lib/core/sdnc_core.h:389-415`). `sdnc-run` then performs one transformer
forward, not a bytecode rollout to halt (`lib/core/sdnc_api.c:258-278`). This is
a useful proof that weights are accessible and differentiable from Eshkol. It
must not be used as evidence that named Eshkol/BLC programs already compile to
resident weights.

The separate qLLM training code has the same one-step boundary: it describes
training from reference-generated `(state, target)` pairs
(`lib/backend/qllm_backward.c:1-14`). A resident mind requires gradients of a
loss over a stateful rollout, not merely independent next-state regression.

## Semantic model

### Two source semantics, one explicit strategy field

Classic BLC and full R7RS Eshkol must not be silently assigned the same
evaluation order.

- A capsule with `strategy = blc-normal` contains the pure lambda core and is
  observationally checked against `blc-eval`. Its compiled machine must use
  explicit thunks/environments or graph reduction; lowering `app` directly to
  eager `OP_CALL` is incorrect.
- A capsule with `strategy = r7rs` preserves the existing Eshkol compiler/VM
  semantics. Its lambda bindings may still be canonicalized to De Bruijn form
  internally, but this does not change argument evaluation.

For `blc-normal`, the first implementation should lower the three BLC node
forms to a **strong call-by-name abstract machine** represented in ESKB:

```text
control  = EVAL(term, env) | READBACK(value, level)
value    = closure(body, env) | neutral(level, spine)
frame    = ARG(term, env) | APPLY(value) | UNDER_LAMBDA | REBUILD_APP
```

`EVAL` reaches weak head form without evaluating unused arguments; `READBACK`
continues under lambdas and through neutral spines to produce full normal form.
The environment is indexed by De Bruijn distance. Sharing a forced thunk is an
allowed optimization because pure call-by-need has the same normal form, but
the reference oracle remains the tree reducer in `blc-step`.

Full Eshkol is represented by an **extended lambda core IR** (`LCIR`):

```text
Var(index) | Lam(body) | App(fun, arg)
Literal(v) | Prim(id, args) | LetRec(bindings, body)
Effect(capability, args) | TensorOp(id, shape, args) | ADOp(id, args)
```

Only the first three nodes serialize as classic BLC. Extensions have a
versioned canonical encoding and an effect/capability manifest. Calling the
whole R7RS language “BLC” would lose evaluation order, effects, mutation,
tensor shape, and primitive identity; LCIR keeps the foundation without making
that category error.

### Canonicalization and identity

The source-to-capsule path is:

```text
runtime/source S-expression
    -> parse and hygienically expand
    -> resolve bindings
    -> alpha-lower binders to 1-based De Bruijn LCIR
    -> validate closedness, evaluation strategy, effects, shapes, and bounds
    -> canonical encode/hash
    -> lower to canonical ESKB
    -> construct program embedding table psi_program
    -> assemble and verify resident capsule
```

The existing bytecode compiler already emits instructions and constants and
writes them into ESKB (`lib/backend/eshkol_compiler.c:5798-5806`,
`lib/backend/eshkol_compiler.c:6019-6055`). The new front half must feed that
lowering from canonical LCIR rather than making the standalone compiler's
parser representation the long-term identity format.

Capsule identity is the hash of all semantic inputs:

```text
H(format-version,
  evaluation-strategy,
  canonical-LCIR,
  primitive-ABI-version,
  VM-ISA-version,
  bounds,
  capability-manifest,
  theta_sem-hash,
  compiler-build-id)
```

Classic `blc-decode` intentionally parses one self-delimiting term and ignores
trailing bits (`lib/core/blc.esk:41-44`, `tests/features/blc_test.esk:86-89`). A
capsule decoder must instead expose the consumed position and require end of
input. Otherwise `bits` and `bits || payload` would identify the same term while
hashing as different artifacts. The capsule also rejects a free De Bruijn index
unless an explicit environment schema is part of its signature; the current
`blc-term?` checks positive indices but not closedness
(`lib/core/blc.esk:82-90`).

## Resident program capsule

The logical schema is:

```text
ResidentProgramCapsule {
  version
  identity_hash
  source_provenance
  evaluation_strategy

  canonical_lcir
  canonical_blc_bits?       ; pure BLC only, strict single term
  eskb
  entrypoints
  capability_manifest
  bounds_manifest

  theta_sem_ref             ; immutable analytical VM weights + hash
  psi_program[N, D]         ; immutable instruction embedding table
  phi_manifest[]            ; trainable leaves only

  state_schema
  optimizer_schema
  staged_kernel_keys[]
  verification_certificate
}
```

The parameter/state roles are normative:

| Role | Examples | Mutable by optimizer? | Semantic effect |
|---|---|---:|---|
| `FROZEN_SEMANTICS` | Q/K/V/O and FFN matrices from `generate_weights` | No | Defines the VM |
| `FROZEN_PROGRAM` | BLC/ESKB-derived embedding rows or precomputed K/V rows | No | Defines the program |
| `TRAINABLE_PLASTIC` | residual adapters, learned memory, typed holes | Yes | Learns within declared interface |
| `RESIDENT_STATE` | VM registers, mind state, episodic memory | No optimizer update | Evolves by execution |
| `OPTIMIZER_STATE` | moments, step, loss scale | By optimizer only | Training mechanism |
| `SCRATCH` | activations, adjoints, checkpoints | No; cleared/reused | No persistent meaning |

The existing flat `SdncWeights`/`SdncGrads` layouts prove that a canonical
parameter traversal is feasible (`lib/core/sdnc_core.h:50-96`), but a resident
capsule needs descriptors for role, dtype, shape, offset, mutability, and hash.
A flat vector alone cannot protect semantic weights from an optimizer.

### What “program to weights” means here

For a program `P`, backend `B`, and bounds `beta`, compilation returns

```text
C(P, B, beta) = (theta_sem[B, beta], psi_program[P, B, beta], certificate)
```

`theta_sem` may be shared by hash across many capsules. `psi_program` is unique
to `P` and is resident model state declared as a frozen embedding weight. This
is already the mathematical form consumed by Layer 0: program position rows
carry opcode and operand while the query derives from PC
(`lib/backend/weight_matrices.c:350-359`,
`lib/backend/weight_matrices.c:5523-5545`).

An optional semantics-preserving specialization pass may:

- precompute `K_P = W_K psi_program` and `V_P = W_V psi_program` while
  `theta_sem` is frozen;
- eliminate unreachable rows proven unreachable under the bounds;
- specialize constant operands and static control edges;
- package a smaller program-specific FFN/attention module.

Every specialization must reproduce the unspecialized capsule's state trace.
If `theta_sem` is trainable, K/V caches are invalid and must be recomputed; the
default architecture avoids that by freezing `theta_sem`.

## Three distinct differentiation levels

The word “AD” currently names different derivatives. The architecture makes
the level part of every API and artifact.

### Level O: object-program AD inside VM state

The VM ISA includes AD opcodes and stores its bounded object tape in state
lanes 36–111 (`lib/backend/weight_matrices.c:163-195`). The transformer has a
separate weight-driven backward transition for that tape
(`lib/backend/weight_matrices.c:5492-5504`). This computes a derivative asked
for **by the resident program**, such as `df/dx` in its object language.

Level O changes VM state. It is not the gradient used by the host optimizer.

### Level M: meta-gradient through transformer execution

Level M differentiates a behavioral loss with respect to model parameters:

```text
d loss / d phi_plastic
d loss / d psi_program      ; only in an explicit relaxed/sketch mode
d loss / d theta_sem        ; diagnostic only by default
```

The analytical one-step primitive already caches a forward and backpropagates
through attention and FFNs (`lib/core/sdnc_core.h:138-205`). The standalone
artifact similarly derives gradients for the transformer matrices
(`lib/backend/weight_matrices.c:5779-5795`) and demonstrates an SGD update
(`lib/backend/weight_matrices.c:5977-6001`). This derivative is of the neural
implementation, not the derivative represented on the VM's object tape.

Program-row gradients do not exist in the current training path. The qLLM
backward code explicitly treats position embeddings as fixed and notes that
learned embeddings would need their own gradients
(`lib/backend/qllm_backward.c:377-384`). A relaxed program-learning mode must
add `dpsi_program`; it must not pretend weight gradients are program gradients.

### Level H: host higher-order and mixed-mode AD

Eshkol's Taylor tower carries coefficients `c[0..K]` with
`f^(n)(x0) = n! c[n]`, uses epoch tags to prevent perturbation confusion, and
serves arbitrary-order `derivative-n`/`taylor`
(`lib/core/runtime_taylor.c:6-29`). Reverse-over-Taylor carries a parallel seed
tangent through the recurrence dispatch (`lib/core/runtime_taylor.c:849-940`),
while the reverse runtime records a mixed derivative back onto the outer tape
(`lib/core/runtime_autodiff.cpp:82-157`).

Level H is how a staged loss can contain exact coordinate derivatives or a
resident program can ask for higher-order numeric structure. It does not make
discrete `Var/Lam/App` tags differentiable. When a Level-M gradient crosses a
Level-O backward operation, the request is higher order and must be declared as
one of:

- `stop_object_gradient`: Level O is an opaque result at Level M;
- `differentiate_object_gradient`: provide a tested second-order VJP/JVP;
- `recompute_with_host_tower`: express the inner derivative using Level H.

Silent fallback among these policies is forbidden.

## Composition with staged `value_and_grad`

The staged-kernel handoff correctly identifies the missing compiler feature:
the repository has broad scalar, mixed, Taylor, and partial tensor AD, but not a
reusable dense tensor `value_and_grad` kernel
(`docs/design/AD_STAGED_KERNEL_HANDOFF.md:17-35`,
`docs/design/AD_STAGED_KERNEL_HANDOFF.md:99-103`). Its target contract is one
primal loss evaluation, one reverse pass, resident parameter/gradient buffers,
fixed scratch, and explicit unsupported errors
(`docs/design/AD_STAGED_KERNEL_HANDOFF.md:369-408`). This ADR adopts that
contract and specializes it for a recurrent resident program.

### Kernel boundary

The compile-time signature includes:

```text
capsule identity hash
entrypoint and evaluation strategy
parameter descriptors and role mask
resident-state schema
input/output shapes
rollout horizon or maximum horizon
checkpoint interval
dtype and accumulation dtype
Level-O/M/H nesting policy
capability/effect policy
```

The runtime call contains pointers only:

```text
run(params, grads, resident_state, optimizer_state,
    batch, outputs, scratch, counters) -> status
```

This extends the proposed pointer/out-parameter ABI
(`docs/design/AD_STAGED_KERNEL_HANDOFF.md:1080-1155`). Program identity and
shapes are specialization keys, never arbitrary captured closures. Parameters,
gradients, state, and scratch are non-aliasing unless a descriptor explicitly
permits an in-place update.

### Fixed-horizon recurrent primal

The compiled primal is a masked fixed-horizon scan:

```text
active[0] = 1
for t in 0 .. T-1:
    candidate = vm_step(theta_sem, psi_program, phi_plastic, state[t], input[t])
    state[t+1] = select(active[t], candidate, state[t])
    active[t+1] = active[t] && !halt(candidate)
    loss += active[t] * observation_loss(state[t+1], target[t])
loss += terminal_loss(state[T], target_terminal)
```

The fixed maximum gives the compiler a static memory plan; the active mask
preserves halt behavior. A “single primal call” means one evaluation of this
whole rollout, not one VM cycle.

The reverse is a **pathwise gradient of the executed discrete trace**. PC,
opcode choice, integer indexing, halt, and capability dispatch are control, not
continuous optimization variables; their derivative is zero/undefined at a
trace change. Learning different control flow uses a declared soft sketch or a
homoiconic recompile-and-verify transaction, never an accidental gradient
through a hard VM branch.

### Reverse and checkpointing

`vm_step` is initially a custom dense primitive whose VJP is the existing
cached analytical backward, rather than millions of scalar tape nodes. Its VJP
returns both parameter adjoints and `dstate`, so the staged reverse can chain
through time. This leverages the existing `dL_dstate` output
(`lib/core/sdnc_core.h:196-205`, `lib/core/sdnc_core.h:350-355`).

For horizon `T`, the kernel stores every state for short rollouts and uses
static checkpoint/rematerialization for long rollouts. Scratch offsets are
planned once; no tensor or tape allocation occurs in the hot loop, matching the
staged memory policy (`docs/design/AD_STAGED_KERNEL_HANDOFF.md:1307-1330`).
Truncated BPTT is permitted only when named in the kernel policy and
certificate; it is never presented as the exact full-rollout gradient.

The current one-step backward covers the normal forward stack. A meta-VJP for
the weight-driven Level-O backward transition and any native post-processing
must be added before `differentiate_object_gradient` is accepted. Until then,
those paths return `UNSUPPORTED_AD` or use an explicit stop-gradient policy.

### Dense tensor path

Eshkol already has a tensor AD-node representation capable of saving up to four
inputs, intermediates, shapes, and parameters
(`lib/backend/autodiff_codegen.cpp:2311-2415`), and its reverse pass dispatches
tensor nodes to dense runtime kernels
(`lib/backend/autodiff_codegen.cpp:10100-10160`). Dense attention and
multi-head-attention backward kernels exist
(`lib/backend/tensor_backward.cpp:1229-1293`).

However, the language-level transformer codegen currently constructs many
scalar nodes inside attention—for example its AD dot product loops record
scalar multiply/add nodes (`lib/backend/tensor_transformer_codegen.cpp:181-190`,
`lib/backend/tensor_transformer_codegen.cpp:245-280`). The staged handoff also
notes that dense codegen mostly fails to route through the available dense
backward substrate (`docs/design/AD_STAGED_KERNEL_HANDOFF.md:29-32`,
`docs/design/AD_STAGED_KERNEL_HANDOFF.md:150-172`). The resident kernel should
therefore begin with an SDNC custom VJP and converge with generic dense tensor
AD only after dense recording is real. `tensor_transformer_codegen` and the
VM-as-transformer are complementary backends, not presently the same compiler.

### Dtype contract

The exact VM artifact is float32 and depends on a specified arithmetic order
and saturated gates (`lib/backend/weight_matrices.c:59-85`). The first staged
subset proposed elsewhere is f64-only
(`docs/design/AD_STAGED_KERNEL_HANDOFF.md:1157-1189`). Resident SDNC staging
therefore requires an explicit f32 tensor descriptor and f32 primal semantics.
Gradient accumulation may use f32 or a declared f64 master accumulator, but the
primal must not be silently widened and then advertised as bit-identical VM
execution.

## Learning and self-modification

### Default: learn plastic weights, preserve semantics and code

The safe resident form is

```text
F_resident = F_exact(theta_sem, psi_program) + R(phi_plastic)
```

where `R` is connected only at declared residual lanes or learned-memory
interfaces and is zero at capsule creation. The optimizer receives only
`TRAINABLE_PLASTIC` descriptors. Hashes of `theta_sem` and `psi_program` are
checked before and after every update. This permits a mind to learn without
quietly corrupting instruction fetch, opcode dispatch, or its own program.

Training all fields returned by today's `sdnc-params` is not the default: the
API walks every matrix slot (`lib/core/sdnc_api.c:79-117`) and its SGD helper
updates all of them (`lib/core/sdnc_core.h:358-378`). That is useful for a
research experiment but destroys the analytical equivalence certificate after
the first nonzero step.

### Differentiable program sketches are explicit relaxations

Discrete opcodes, operands, De Bruijn indices, and list topology have no useful
pathwise gradient. The exact VM makes this even sharper: its indicator gates
are intentionally saturated to literal 0/1 (`lib/backend/weight_matrices.c:59-85`,
`lib/backend/weight_matrices.c:333-348`). Arbitrarily nudging opcode lanes is
neither stable program synthesis nor faithful VM execution.

A trainable program mode must declare typed holes:

- opcode logits over a finite allowed set;
- operand logits over a bounded integer set;
- differentiable numeric constants;
- soft branch or module-selection gates;
- learned embeddings whose consumers have a specified decoding rule.

The relaxed executor computes a mixture and sets
`certificate.mode = approximate-sketch`. Promotion performs argmax/projection,
rebuilds an exact `psi_program`, and reruns semantic verification. No relaxed
capsule receives an exactness certificate.

### Homoiconic revision is a discrete outer loop

General self-modification uses Eshkol's strongest property directly:

```text
observe -> quote/read canonical syntax -> propose structural rewrite
        -> resolve + De-Bruijn canonicalize -> compile capsule
        -> test in shadow -> certify -> atomically promote or reject
```

The proposal mechanism may itself be learned, but the rewrite is a discrete
transaction. Every accepted revision gets a new capsule identity and retains
its parent hash. This is more principled than a straight-through estimator over
arbitrary cons cells and gives a resident mind rollback, provenance, and a
stable answer to “which program am I running?”

## Resident-mind execution protocol

The substrate for a long-lived resident mind is a lifecycle, not merely a
gradient function.

1. **Boot:** load a verified capsule; verify `theta_sem`, `psi_program`, bounds,
   capabilities, and staged-kernel hashes.
2. **Run:** evolve `RESIDENT_STATE` through the exact program plus declared
   plastic interfaces. External effects pass through capability-checked host
   boundaries and are logged; they are never silently differentiated.
3. **Accumulate experience:** write immutable training records containing
   capsule identity, state schema version, input/output, effect receipts, and
   objective metadata.
4. **Train in shadow:** invoke the staged rollout `value_and_grad` on a snapshot
   of `phi_plastic` and resident state. Update shadow parameters and optimizer
   state; the live execution state is not aliased with training scratch.
5. **Validate:** check finite values, gradient policy, held-out behavior,
   resource bounds, capability policy, and exact semantic/program hashes. For a
   code revision, also run BLC/LCIR and VM trace oracles.
6. **Promote atomically:** install a versioned plastic checkpoint or a new
   certified capsule at a quiescent state boundary. Preserve the predecessor
   for rollback.
7. **Resume:** migrate resident state only through an explicit, versioned state
   transformer. A new program never reinterprets old state bytes by accident.

Optimizer state is not identity, scratch is not memory, and resident memory is
not a trainable parameter unless its schema explicitly says so. These
separations prevent arena reuse, gradient buffers, or Adam moments from becoming
accidental parts of the mind's durable state.

## Compiler and runtime interfaces

The intended user-facing operations are conceptually:

```scheme
(compile-resident-program code
  #:strategy 'r7rs                    ; or 'blc-normal
  #:bounds bounds
  #:capabilities caps
  #:plastic-spec plastic-spec)
    -> capsule

(resident-run capsule state input #:max-steps T)
    -> (values output next-state trace-summary)

(stage-resident-value-and-grad capsule loss-spec signature policy)
    -> kernel

(resident-train-step! kernel capsule shadow-state batch optimizer-state)
    -> (values loss metrics candidate-checkpoint)
```

The C layer should expose opaque handles but retain descriptor-based buffers,
not only a flattened vector. A `value_and_grad` call returns loss and all
declared parameter gradients from the same primal. `sdnc-weight-grad`'s current
fixed MSE target (`lib/core/sdnc_api.c:309-325`) becomes a compatibility wrapper
around a loss-specific staged kernel, not the universal training API.

## Verification gates

A capsule or staged kernel is not complete until all applicable gates pass.

### Lambda and homoiconic gates

- `decode_exact(encode(term)) = term` and consumes the complete input;
- closedness or an explicit environment schema;
- alpha-equivalent named sources canonicalize to the same De Bruijn LCIR hash;
- quote/read/LCIR round trips preserve symbol identity, literals, strategy, and
  effect annotations;
- `blc-normal` results agree structurally with `blc-eval`, including discarded
  divergent arguments and resource-bound failure.

### Compilation and transformer gates

- LCIR and ESKB reference executions agree on output and declared effects;
- ESKB instruction rows equal the canonical `embed_instruction` construction;
- analytical, simulated, and matrix paths agree on every in-contract state
  transition, extending the existing three-way harness;
- specialization and K/V precomputation agree with the unspecialized capsule;
- out-of-bounds programs fail or use a declared non-SDNC backend—never truncate;
- `OP_NATIVE_CALL` and other effects are explicit capability boundaries.

### Gradient and staging gates

- custom one-step VJP agrees with the existing analytic backward and an
  independent numerical oracle on sampled smooth parameters; finite differences
  remain a test oracle, never a runtime fallback;
- `dstate` chaining matches an unrolled small-rollout oracle;
- gradients for `FROZEN_SEMANTICS` and `FROZEN_PROGRAM` are absent or exactly
  masked, not merely ignored by the optimizer;
- program-row gradients exist only in relaxed-sketch mode;
- one rollout primal and one rollout reverse produce all requested gradients;
- compile count is one per specialization key, scratch use is constant after
  warmup, and the hot loop allocates nothing;
- unsupported dense ops or AD nesting policies return an explicit status;
- exact f32 execution remains bit-identical after staging;
- repeated promotion/rollback preserves capsule and resident-state provenance.

These requirements refine the proposed staged definition of done—one compile,
zero hidden finite differences, one primal, one reverse, stable scratch
(`docs/design/AD_STAGED_KERNEL_HANDOFF.md:1550-1587`)—for recurrent program
execution.

## Delivery sequence

### Phase 1: make the semantic artifact shareable

- Move the analytical weight layout, constructor, forward, and VJP behind one
  library boundary used by the paper tool and runtime. Remove the long-term need
  for byte-for-byte mirrored definitions described in
  `lib/core/sdnc_core.h:4-17`.
- Add a loader/constructor for analytical `theta_sem`; retain random
  initialization under an explicitly experimental API.
- Add structured parameter descriptors and immutable role masks.

### Phase 2: canonical program capsules

- Add named-source-to-De-Bruijn LCIR normalization and closedness checking.
- Add strict BLC decoding and the `blc-normal` strong evaluator lowering.
- Feed LCIR/R7RS lowering into canonical ESKB.
- Replace name-hashed random program rows with ESKB-derived
  `psi_program`; package it with hashes, bounds, capabilities, and provenance.
- Extend three-way verification to compiled capsules.

### Phase 3: resident execution and one-step VJP

- Add a pure `vm_step` boundary for in-contract, weight-implemented behavior.
- Return adjoints for state, trainable weights, and—only when requested—program
  rows. The current qLLM omission of position-embedding gradients is then closed.
- Separate host effects from the differentiable transition.
- Add fixed-horizon masked rollout and state checkpoints.

### Phase 4: staged `value_and_grad`

- Land one-pass parameter gradients and strict dense tensor backward before the
  public primitive, following the order already specified for first-class
  `value-and-grad` (`docs/design/AD_STAGED_KERNEL_HANDOFF.md:861-924`).
- Add f32 staged descriptors, fixed scratch planning, custom `vm_step` VJP, and
  recurrent reverse/checkpointing.
- Add optimizer epilogues only after value/gradient correctness, matching the
  staged optimizer sequencing (`docs/design/AD_STAGED_KERNEL_HANDOFF.md:1332-1383`).

### Phase 5: resident learning and reflective promotion

- Add plastic adapters and learned-memory interfaces with zero initial residual.
- Add shadow checkpoints, atomic promotion, rollback, state migration, and
  experience provenance.
- Add typed differentiable sketches only after the exact capsule path is
  certified.

## Alternatives rejected

### Train the entire interpreter as the resident mind

Rejected as the default. It destroys the VM equivalence proof immediately and
mixes semantic law with learned policy. Full-weight training remains a named
research mode with no exactness certificate.

### Call the current `sdnc-program` result a compiled program

Rejected. The implementation hashes a name into random embeddings and creates
random weights (`lib/core/sdnc_api.c:211-252`,
`lib/core/sdnc_core.h:389-415`). Renaming that operation would not create the
missing compiler bridge.

### Treat runtime bytecode rows as “already program weights” without packaging

Rejected. The mathematical data is present, but a host-provided activation has
no immutable role, identity, provenance, or optimizer protection. Promoting it
to an embedding-table parameter inside a capsule makes the claim testable.

### Lower BLC application directly to eager VM calls

Rejected. It fails the normal-order witness already tested by `K I Omega`
(`tests/features/blc_test.esk:111-127`). Evaluation strategy is part of the
capsule contract.

### Differentiate arbitrary homoiconic structure

Rejected. AD differentiates numeric maps of fixed shape. General syntax change
is a discrete reflective transaction; bounded typed holes are the only
continuous relaxation admitted.

### Use the VM's internal AD tape as the optimizer tape

Rejected. The object tape is bounded VM state and computes program-requested
derivatives. Model training needs a meta-gradient through transformer steps and
time, with a different tape, lifetime, and parameter domain.

### Wait for generic transformer tensor codegen before staging SDNC

Rejected. Dense runtime backward kernels exist, but the current attention
codegen still records scalar inner-loop nodes. A custom SDNC step VJP provides a
correct bounded bridge now and can later lower to the generic dense graph.

## Consequences

Positive consequences:

- Lambda terms gain stable, alpha-invariant program identity.
- “Program to weights” becomes a concrete artifact transformation rather than
  a metaphor.
- Exact VM semantics survive learning because semantic and plastic parameters
  are structurally separated.
- Homoiconic self-revision remains available with audit, certification, and
  rollback.
- The existing AD tower, analytical transformer VJP, and staged-kernel plan fit
  into one hierarchy without conflating their derivative levels.
- A resident mind can train repeatedly with stable buffers and preserve durable
  state across kernel invocations and program upgrades.

Costs and risks:

- LCIR, strict BLC packaging, and a normal-order abstract-machine lowering are
  new compiler work.
- The analytical generator/runtime mirror must be unified to prevent drift.
- Recurrent meta-gradients require checkpointing and VJPs for every admitted
  transition, including explicit policy for Level-O backward and effects.
- Exact f32 execution and higher-precision optimizer state require a careful
  mixed-dtype ABI.
- Frozen semantic weights constrain where learning can occur; useful plastic
  interfaces must be designed rather than obtained by making everything
  mutable.
- Soft program sketches are approximate until discretized and re-certified.

These costs are intentional. The substrate is meant to support a mind that can
learn and revise itself while retaining a precise account of its code,
semantics, state, gradients, and lineage. A monolithic mutable tensor cannot
provide that account; a resident program capsule can.
