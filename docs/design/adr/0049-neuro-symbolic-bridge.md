---
kind: explanation
status: current
owner-area: stdlib
since: v1.5.0
sources:
  - lib/core/ml/recurrent.esk
  - lib/core/ml/gradient_estimators.esk
  - lib/core/ml/hnsw.esk
  - lib/core/ml/bpe.esk
  - lib/core/ml/sparse.esk
  - lib/core/runtime_taylor.c
  - lib/core/runtime_arena_core.cpp
  - lib/backend/llvm_codegen.cpp
  - lib/backend/eshkol_vm.c
  - lib/backend/vm_native.c
  - scripts/wasm_exact_runtime.fragment.js
---
# ADR-0049: The v1.5.0 neuro-symbolic bridge — Noesis for the cognition, Eshkol for the substrate

**Status:** Accepted — Eshkol substrate and gap items implemented; Noesis
upstreaming planned for v1.5.0 under the boundary process below
**Scope:** The v1.5.0-intelligence items of [`ROADMAP.md`](../../../ROADMAP.md):
symbol embeddings, soft unification, LSTM and GRU cells, differentiable logic
programs, attention over the knowledge base, gradient estimators for discrete
operations, and the Noesis M2 surface (HNSW, BPE, sparse tensors, int/complex
tensors).

## Context

The design document
[`NEURO_SYMBOLIC_COMPLETE_ARCHITECTURE.md`](../../future/NEURO_SYMBOLIC_COMPLETE_ARCHITECTURE.md)
(section 2.2.4 and Part IV) describes an embedding table, recursive term
embeddings, soft unification, attention over facts, Gumbel-Softmax and a
straight-through estimator written as `stop-gradient(hard - soft) + soft`, all
integrated with `gradient`. Noesis, the Eshkol reference consumer, ships with
v1.5.0-intelligence and v1.6.0-reasoning (maintainer ruling 2026-08-26), and
already implements most of the cognition side of these items in Eshkol. What
Noesis needs from Eshkol is substrate. This record maps each item to where it
is delivered, decides what Eshkol builds, and proposes how the Noesis modules
reach the Eshkol standard library.

## Item map

| v1.5.0 item | Delivered by | Where |
|---|---|---|
| Symbol embeddings | Noesis | term embeddings (`core/geometry/term_embedding`, `core/encoder`) |
| Soft unification, gradients through matching | Noesis | `core/soft_unify` (`soft_unify`, `soft_unify_grad`, `semantic_unify`) |
| Differentiable logic programs | Noesis | differentiable belief propagation (`core/v1_5/differentiable_bp`) on its tape |
| Attention over the knowledge base | Noesis | attention unification (`core/attention_unify`) |
| LSTM and GRU cells | Eshkol (absent from Noesis) | `core.ml.recurrent` |
| Gradient estimators (Gumbel-Softmax, straight-through) | Eshkol (absent from Noesis) | `stop-gradient` builtin, `core.ml.gradient_estimators` |
| M2: HNSW | Eshkol | `core.ml.hnsw`, with `kb-query-nearest` over KB facts |
| M2: BPE | Eshkol | `core.ml.bpe` |
| M2: sparse tensors | Eshkol | `core.ml.sparse` (CSR) |
| M2: int/complex tensors | Eshkol, planned for v1.5.0 | staged design below |

## Decision

1. **Eshkol does not carry a second implementation of what Noesis delivers.**
   Symbol embeddings, soft unification, differentiable logic and KB attention
   reach Eshkol by upstreaming Noesis modules, not by rewriting them. An
   Eshkol-side implementation built before this map was drawn is kept on the
   branch `feat/v15-neurosymbolic-noesis-overlap` as reference material for
   that review and is not part of the v1.5.0 change.

2. **One parameter vector, one differentiation mechanism, for what Eshkol
   builds.** LSTM and GRU weights, categorical logits and sparse-matrix values
   are slices of a flat vector `theta`; every forward function is written over
   generic arithmetic, so `(gradient loss theta)` is the exact reverse-mode
   gradient (for a recurrent net, backpropagation through time) and
   `derivative` along any coordinate gives the same numbers. No hand-written
   backward pass exists to drift from its forward pass. The tests compare the
   two modes, compare closed forms, and assert `(ad-finite-difference-evals)`
   is 0.

3. **`stop-gradient` is a builtin.** No library function can remove an AD
   carrier: arithmetic on a carrier produces a carrier. `(stop-gradient x)`
   returns the primal of any carrier (forward jet, Taylor tower or nested
   level, reverse-mode node; an exact payload stays exact) and passes other
   values through. It reads the primal as the comparisons and branch builtins
   already do (`tagged_primal`), through one runtime entry point,
   `eshkol_stop_gradient_tagged`; the VM (builtin id 2252) reads its dual's
   primal the same way, and the browser lane has the same import. The
   straight-through identity `s + stop-gradient(h - s)` has the value `h` and
   the derivative of `s`; the estimators are built on it, with their bias and
   variance stated and tested on known distributions.

4. **The recording tape is a root for loop reclamation.** A named-let loop
   under reverse mode opens its per-iteration arena scope on the tape's arena.
   The scope used to be popped whenever the loop-carried values did not point
   into it; a node consed onto a list from the global arena is not such a
   pointer, so its span was rewound while the tape still recorded the node and
   the next iteration reused the address. Nodes are appended in creation order
   and allocated bump-ordered from the tape's arena, so a span holds a
   recorded node exactly when it holds the last one; such a span, or one
   holding a regrown node array, is now retained at a back edge, before a
   promotion, and at the loop's exit. Every list-building loop under
   `gradient` depends on this, Noesis's included.

5. **The M2 substrate is library first.** HNSW (Malkov & Yashunin 2018,
   deterministic levels from a seeded LCG, squared-L2 or cosine), BPE
   (Sennrich et al. 2016, deterministic tie-breaking, models as plain merge and
   vocabulary data) and CSR sparse matrices (values may be AD carriers;
   Gustavson products) are `core.ml` modules. `hnsw-index-kb!` and
   `kb-query-nearest` put an HNSW index beside a KB under a caller's embedding
   function, so Noesis keeps its own embeddings and gains the vector side of a
   KB query.

## Int and complex tensors (planned for v1.5.0)

Tensor storage is f64 bit patterns with a logical `dtype` field
(`runtime_tensor_dtype.cpp`); casting reduces precision through f64. Exact
int64 elements and complex elements need storage that is not f64: stage 1
adds an element-width field and int64/complex128 storage to the tensor
header behind the object-ABI ratchet (ADR-0012); stage 2 routes the
elementwise, reduction and matmul kernels by storage kind; stage 3 carries the
VM and the browser lane. It changes the 40-byte tensor layout and so moves
with the object-ABI staging, not inside this record's library change.

## Upstreaming proposal for the Noesis modules

Under the Noesis boundary process (a file-by-file enumeration of each
candidate module and an identity, secrets and history sweep, decided by the
maintainer):

- **Candidates for the Eshkol standard library / public prover engine:**
  soft unification and its gradient (`core/soft_unify`), term embeddings
  (`core/geometry/term_embedding`), attention unification
  (`core/attention_unify`), differentiable belief propagation
  (`core/v1_5/differentiable_bp`) and the factor-graph layer it runs on. These
  are mechanism, built on the consciousness engine and the AD operators, and
  are what the roadmap items name.
- **Stays in Noesis:** the product layer — the system's own cognitive state
  and the encoders trained on it.
- **Interface contract on the Eshkol side:** the upstreamed modules use
  `walk`/`unify`/`kb-query` for structure, the built-in AD operators (with
  `stop-gradient` where a custom backward is needed) instead of a private
  tape, and the substrate modules above for vector search, tokenization and
  sparse algebra.

## Alternatives rejected

- **Reimplementing the Noesis items in Eshkol.** Two implementations of the
  same cognition mechanism would diverge, and the Noesis versions are the ones
  exercised by a running system.
- **Hand-written backward passes on `core.ad.tape` for the new Eshkol
  modules.** The built-in operators already differentiate the forward code
  exactly, in two independent modes the tests compare.
- **Finite-difference gradient checks.** Exact AD in two modes and closed
  forms replace them.

## Consequences

- The language surface gains one builtin (`stop-gradient`), registered in the
  VM table, the LLVM dispatch, the XLA classification (host) and both browser
  bundles; the coverage ledger counts it at integration.
- Loops that build lists of reverse-mode nodes keep those nodes for the tape's
  lifetime, which is the lifetime the tape already requires.

## Verification

`tests/ml/recurrent_test.esk`, `tests/ml/gradient_estimators_native_test.esk`,
`tests/autodiff/loop_list_reverse_mode_test.esk`, `tests/ml/hnsw_test.esk`,
`tests/ml/bpe_test.esk` and `tests/ml/sparse_test.esk`, each under JIT and
AOT; `tests/vm_parity/corpus/102_stop_gradient.esk` for the VM.
