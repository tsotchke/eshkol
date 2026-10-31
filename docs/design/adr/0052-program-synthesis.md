# ADR-0052: Program synthesis, synthesis holes and graph neural networks (v1.7.0-synthesis)

**Status:** Proposed
**Rung:** v1.7.0-synthesis
**Scope:** `core.synthesis` (example-based synthesis), `ml.autograd` and
`ml.gnn` (graph neural networks), compiler completion of `??` holes
(`lib/types/hole_synthesis.cpp`, the hole rules in `lib/types/type_checker.cpp`),
and the plan for the learned-search items of the rung.

## Context

The v1.7.0-synthesis rung of the roadmap lists five capabilities: neural-guided
program search, type-directed synthesis holes (`??`), graph neural networks,
synthesis from input-output examples, and neural theorem provers that guide
symbolic proof search with v1.5 embeddings and v1.6 chaining.

Noesis, the reference Eshkol application that ships as part of v1.5.0 and
v1.6.0, already implements part of this surface in Eshkol. Its modules were
mapped read-only before any design decision here:

| Roadmap item | Noesis | Eshkol before this ADR |
|---|---|---|
| Neural-guided program search | Implemented: Sigma PCFG search with beam, sampling and MCTS moves and an MDL score (`src/core/sigma/`), learned template memory (`sigma_learning.esk`), differentiable program selection (`diff_synthesis/`), attention-program synthesis with JSD ranking (`attention_unify/attention_program_synthesis.esk`) | none |
| `??` typed holes | Design only (`docs/design/faculties/poiesis/01-type-directed-synth.md`) | none |
| Graph neural networks | Partial: a deterministic compositional term encoder (`encoder/encoder.esk`); the learned GNN encoder is deferred there | none |
| IO-example synthesis | Partial: an enumerator over sequences of unary operators (`synthesizer/synthesizer.esk`) used by bounded code synthesis with counterexample revision (`poiesis/code_synthesis.esk`) | none |
| Neural theorem prover | Implemented: learned tactic and lemma ranking from a shared embedding table (`aletheia/learned_prover.esk`), geodesic proof-flow ranking (`aletheia/proof_flow_ranker.esk`, `geometry/proof_flow.esk`) | none |

## Decision

Eshkol builds the parts that are substrate or that Noesis lacks, and does not
reimplement Noesis's learned search or its prover.

### 1. Synthesis from examples: `core.synthesis`

- **Typed DSL.** Types are `int`, `bool`, `list`, `str` and three function
  types. A component carries its name, signature, a total implementation and a
  positive integer cost. Partial operations (head of the empty list, a value past
  the magnitude bound) return the bottom marker instead of raising, so search
  never needs exception handling.
- **Bottom-up enumeration by cost** with observational-equivalence pruning: a
  candidate whose outputs on all example inputs equal those of a cheaper program
  of the same type is discarded, so the bank holds one program per behaviour.
  Components are applied only to arguments of their declared types; the first
  program whose outputs equal the examples' is a cheapest consistent program.
- **Deductive splitting.** FlashFill for strings (concatenations of constants
  and substrings delimited by token, from-end or absolute positions, with the
  atoms consistent with every example found jointly by a cheapest-path search
  over the product of output positions), and map / filter / filter-then-map for
  lists, whose element functions are synthesised from derived per-element
  examples.
- **Benchmark set.** 42 tasks (integers, DeepCoder-style lists, deduction-style
  lists, FlashFill-style strings). A task is solved only when the program also
  reproduces held-out examples it never saw.

### 2. The guidance hook (substrate for learned search)

Program cost is the sum of component costs, and enumeration is in cost order.
A learned scorer guides the search by setting costs (for example
`round(-log2 p)` from predicted component probabilities) with
`syn-library-costs`; completeness is unchanged. This is the interface a Noesis
scorer, or any other, drives. Eshkol does not ship its own trained scorer.

### 3. `??` synthesis holes in the compiler

`??` is an ordinary identifier to the reader. When it is unbound at its use,
the type checker treats it as a hole:

- **Expected type.** `TypeChecker::expectAt` records the type a position
  requires for a hole: a function body under a return annotation, an argument
  of a callee with a known signature, an operand of `+ - * /` (a number), and an
  `if` test (a boolean). `checkTask` reports a hole checked against a type
  directly. Elsewhere the expected type is unknown (`Value`).
- **Observer.** The checker calls `HoleObserver::onHole` with the hole and its
  expected type while the hole's scope is live, so `Context::visibleBindings`
  is exactly what is in scope there.
- **Search.** `HoleSynthesizer` enumerates terms bottom-up by size (default at
  most 5 nodes) from the bindings in scope and a typed operator table
  instantiated at the numeric types present, applying a component only to
  arguments whose types are subtypes of its parameters. It never offers the
  definition that contains the hole. Candidates of the expected type are ranked
  by the number of distinct local bindings used, then constants used, then
  size, then construction order (deterministic), and each is accepted only if
  `TypeChecker::checkSpeculative` (Γ ⊢ e ⇐ τ with every effect rolled back)
  succeeds.
- **Completion.** In the codegen type-check phase, after a top-level form is
  checked its holes are replaced by their best candidates, a warning lists the
  completion and the other candidates, and the form is checked again. A hole
  with no candidate is an error and no code is generated (JIT and AOT).
- **Interface with the checking-direction work (ADR-0042).** The hole rule is
  in `checkTask`; `expectAt` covers the positions the synthesis rules already
  type. When the checker gains checking rules for more forms, each of those
  calls `check` on its subterms and the hole receives the expected type with no
  change here, and the corresponding `expectAt` call retires.

### 4. Graph neural networks: `ml.autograd`, `ml.gnn`

`ml.autograd` records dense matrix operations (matmul, bias, activations,
concatenation, sparse propagation, softmax cross-entropy, sigmoid
cross-entropy, L2) on the `core.ad.tape` reverse-mode tape with hand-derived
backward rules, plus Adam. `ml.gnn` adds graphs, host message passing with any
message, aggregate and update functions, differentiable sum / mean / max
neighbourhood aggregation, GCN (Kipf and Welling 2017), GraphSAGE-style message
passing (Hamilton et al. 2017) and graph attention (Velickovic et al. 2018),
and two-layer node classifiers. Every backward rule is checked against central
differences; no finite differences are used in training.

### 5. Neural-guided search and neural theorem proving: upstreaming proposal

Eshkol does not build these engines; Noesis has them. Proposal for the owner
and tsotchke-chan to decide through the boundary enumeration process (no
Noesis file has been copied or edited):

- **Candidates for the public Eshkol stdlib / prover engine:** the Sigma PCFG
  search core (grammar, candidate scoring, beam and MCTS moves) as a
  grammar-guided search library; `diff_synthesis` as a differentiable
  program-selection example; `aletheia/learned_prover.esk` and
  `aletheia/proof_flow_ranker.esk` as a learned proof-step ranker over a
  pluggable proof-search interface; `synthesizer/synthesizer.esk` is subsumed
  by `core.synthesis` and need not move.
- **Stays in Noesis:** learning sessions and equation memory tied to its own
  state (`sigma_learning.esk`), attention-program synthesis over its attention
  objects, and anything bound to cognition, self-model or personality state.
- **Eshkol interfaces they would bind to:** the `core.synthesis` cost hook for
  a learned scorer; `ml.autograd` for training scorers with exact gradients;
  the v1.6 backward chainer's single-step resolution primitive (ADR-0051) for a
  learned proof-step ordering.

## Consequences

- Synthesis from examples, graph neural networks and `??` completion ship as
  Eshkol facilities with JIT and AOT tests.
- `??` is read as an identifier (before this ADR the reader took it as a
  logic variable named `?`). It is completed by the native front end (JIT and
  AOT). The bytecode compiler reports an unbound `??` as a hole it does not
  complete, and the REPL does not complete holes; programs with holes are
  compiled natively. Planned for a later rung: the same elaboration before the VM
  compiler, and example-directed hole filling that evaluates candidates on
  user-supplied examples with `core.synthesis`.
- Type-directed completion ranks by relevance and size; when several
  completions share a rank (for example `(+ x y)` and `(* x y)`), the
  diagnostic lists them so the author can choose.
