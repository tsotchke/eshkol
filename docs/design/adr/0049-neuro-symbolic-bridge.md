---
kind: explanation
status: current
owner-area: stdlib
since: v1.5.0
sources:
  - lib/core/ml/neurosymbolic.esk
  - lib/core/ml/diff_logic.esk
  - lib/core/ml/recurrent.esk
  - lib/core/ml/gradient_estimators.esk
  - lib/core/runtime_taylor.c
  - lib/core/runtime_arena_core.cpp
  - lib/backend/llvm_codegen.cpp
  - lib/backend/eshkol_vm.c
  - lib/backend/vm_native.c
  - scripts/wasm_exact_runtime.fragment.js
---
# ADR-0049: The neuro-symbolic bridge is differentiated by the built-in AD operators

**Status:** Accepted — implemented for v1.5.0-intelligence
**Scope:** The v1.5.0 neuro-symbolic items of [`ROADMAP.md`](../../../ROADMAP.md):
symbol embeddings, soft unification, LSTM and GRU cells, differentiable logic
programs, attention over the knowledge base, and gradient estimators for
discrete operations. Library modules `core.ml.neurosymbolic`,
`core.ml.diff_logic`, `core.ml.recurrent` and `core.ml.gradient_estimators`,
one builtin (`stop-gradient`) on the native backend, the bytecode VM and the
browser WASM lane, and the loop-reclamation rule for reverse-mode tapes.

## Context

The consciousness engine is exact: `unify` matches two ground symbols only
when they are the same symbol, `kb-query` returns the substitutions of the
facts that unify, and inference is discrete. The roadmap's v1.5.0 items ask
for the opposite direction as well: symbols with learnable vector meanings,
matching that has a gradient, rules whose application has a gradient, and
estimators for the gradient of a discrete choice. The design document
[`NEURO_SYMBOLIC_COMPLETE_ARCHITECTURE.md`](../../future/NEURO_SYMBOLIC_COMPLETE_ARCHITECTURE.md)
(section 2.2.4 and Part IV) specifies the surface: an embedding table,
recursive term embeddings, cosine-similarity soft unification, attention over
facts, Gumbel-Softmax and a straight-through estimator written as
`stop-gradient(hard - soft) + soft`, all integrated with `gradient`.

Eshkol already has three exact differentiation engines: forward jets
(`derivative`), Taylor towers (`derivative-n`, `taylor`) and the reverse-mode
tape (`gradient`, `jacobian`). It also has `core.ad.tape`, a pure-Scheme tape
where every op carries a hand-written backward closure; the first slice of
`core.ml.neurosymbolic` and `core.ml.gradient_estimators` was built on it.

## Decision

1. **One parameter vector, one differentiation mechanism.** Every learnable
   quantity of the bridge (symbol embeddings, rule-template logits, LSTM and
   GRU weights, categorical logits) is a slice of a flat vector `theta`, and
   every forward function is written over generic arithmetic. A loss is an
   ordinary `(lambda (theta) ...)`; `(gradient loss theta)` is its exact
   reverse-mode gradient (for a recurrent net: backpropagation through time),
   and `derivative` along any coordinate gives the same numbers in forward
   mode. No forward function has a hand-written backward pass, so there is no
   second derivative implementation to keep in agreement. The tape-based
   functions of the first slice stay as the `core.ad.tape` interface.

2. **Embeddings are attached to the knowledge base, not copied out of it.**
   `embedding-attach-kb!` registers every predicate and constant of the facts
   of a schema; facts are read with the engine's own `kb-query` through fresh
   logic variables (`kb-facts`). The table has a live view (`embed!`) and the
   parameter view (`embedding-params`, `param-embedding`, `embedding-load!`).
   Compound terms are embedded as holographic reduced representations
   (Plate 1995): `emb(f) + sum_i role_i (*) emb(a_i)` with fixed unit-norm role
   vectors and circular convolution.

3. **Soft unification extends the engine's unification.** `soft-unify-terms`
   walks both terms with the engine's `walk`, binds logic variables with the
   engine's `unify` (occurs check included), unifies pairs componentwise with
   the product t-norm, and scores two distinct ground atoms with a similarity
   of their embeddings (RBF kernel, mapped cosine, or sigmoid of the scaled dot
   product). Whenever `unify` succeeds, soft unification returns the same
   bindings with score exactly 1. A structural mismatch still fails.

4. **Attention over the KB is attention over proofs.** `neural-query` answers
   a pattern with one variable by a softmax over the facts whose logits are
   `log(score)/temperature` of the soft unification of the pattern with each
   fact (Neural Theorem Provers, Rocktaschel & Riedel 2017). Arguments equal in
   the pattern and a fact contribute exactly 1, so a learned predicate
   similarity carries over to entities never seen in training. Generic
   scaled dot-product attention (`attend`), Gaussian-kernel attention
   (`attend-rbf`) and term-embedding attention over facts (`attend-facts`) are
   provided for key-value use.

5. **Differentiable logic programs are forward chaining with soft truth.**
   `core.ml.diff_logic` holds binary relations over a finite domain read from
   the KB as `n x n` truth matrices; conjunction is the product t-norm,
   disjunction the probabilistic sum, the existential the noisy-or. Rule
   templates (`chain`, `conj`, `inverse`, `copy`) leave their body predicates
   to be learned as a softmax over candidates (Evans & Grefenstette 2018; the
   relation mixture is the Neural LP relaxation). A head may be its own
   candidate, so recursive programs are expressed directly, and `steps` bounds
   the derivation depth. `dl-extract-rules` reads the learned program back as
   symbolic clauses and `dl-assert-derived!` asserts the derived atoms into the
   KB, where the exact engine queries them.

6. **`stop-gradient` is a builtin.** The straight-through estimators need a
   value that every enclosing AD pass treats as a constant, and no library
   function can remove a carrier: arithmetic on a carrier produces a carrier.
   `(stop-gradient x)` returns the primal of any carrier (forward jet, Taylor
   tower or nested level, reverse-mode node; an exact payload stays exact) and
   passes every other value through. It reads the primal exactly as the
   comparisons and branch builtins do (`tagged_primal`), through one runtime
   entry point `eshkol_stop_gradient_tagged`; the VM (builtin id 2252) reads
   its dual's primal the same way, and the browser lane has the same import.
   The straight-through identity `s + stop-gradient(h - s)` has the value `h`
   and the derivative of `s`.

7. **The recording tape is a root for loop reclamation.** A named-let loop
   under reverse mode opens its per-iteration arena scope on the tape's arena.
   The scope used to be popped whenever the loop-carried values did not point
   into it; a node consed onto a list from the global arena is not such a
   pointer, so its span was rewound while the tape still recorded the node,
   and the next iteration reused the address (every earlier entry then read
   the newest node). Nodes are appended to the tape in creation order and
   allocated bump-ordered from the tape's arena, so a span holds a recorded
   node exactly when it holds the last one; such a span, or one holding a
   regrown node array, is now retained at a back edge, before a promotion, and
   at the loop's exit.

## Alternatives rejected

- **Hand-written backward passes on `core.ad.tape` for every new op.** Each
  op would carry a second implementation of its derivative that has to agree
  with the forward pass forever; the built-in operators already differentiate
  the forward code exactly, in two independent modes that the tests compare.
- **Finite-difference gradient checks.** The checks compare reverse mode with
  forward mode and with closed forms, and assert
  `(ad-finite-difference-evals)` is 0 for the whole program.
- **Dot-product attention for `neural-query`.** Superposed term embeddings
  make the logit depend on cross terms between the predicate and the entity,
  so a predicate similarity learned on some entities does not transfer to
  others; proof-score logits factor over argument positions and do.
- **A tensor-typed embedding table.** The bridge's parameter counts are in the
  tens to thousands, and the vector representation keeps every function on the
  carriers that all three AD engines differentiate today. A dense tensor
  parameter store is the v1.5.x follow-on once ADR-0002's staged kernels carry
  these losses.

## Consequences

- Every bridge function runs on plain numbers and on every AD carrier; the
  tests check reverse == forward to 1e-12 and the closed forms to 1e-14.
- The language surface gains one builtin (`stop-gradient`), registered in the
  VM table, the LLVM dispatch, the XLA classification (host) and both browser
  bundles.
- Loops that build lists of reverse-mode nodes keep those nodes for the tape's
  lifetime instead of reclaiming them per iteration, which is the lifetime the
  tape already requires.

## Verification

`tests/ml/neurosymbolic_bridge_test.esk`, `tests/ml/diff_logic_test.esk`,
`tests/ml/recurrent_test.esk`, `tests/ml/gradient_estimators_native_test.esk`
and `tests/autodiff/loop_list_reverse_mode_test.esk`, each under JIT and AOT;
examples `examples/neurosymbolic_kinship.esk` and
`examples/neurosymbolic_relation_paraphrase.esk`.
