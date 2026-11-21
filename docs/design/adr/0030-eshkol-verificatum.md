---
kind: explanation
status: current
owner-area: language
since: v1.3.6-evolve
sources:
  - verification/authorization/Refinement.lean
  - verification/authorization/Authority.lean
  - verification/authorization/authority.py
  - verification/lean-kernel/R0.toml
  - verification/lean-kernel/Typing.lean
  - verification/lean-kernel/replay.py
---
# ADR-0030: Eshkol Verificatum — Independent Proof Checking and Verified Refinement

- Status: proposed
- **Date:** 2026-11-20
- **Scope:** Proof admission, model-to-implementation claims, and evidence consumed by the compiler and release gates.

## Context

A checker can validate a sound derivation of the wrong proposition. For example, a declaration can still be called `mt` after its producer changes the meaning of negation and supplies a well-typed body. R0 includes exactly this control. The reference kernel accepts the altered export in its own environment. Eshkol rejects it when it compares the theorem with an independently authored statement of modus tollens. That comparison is the basis for this decision. [R0 experiment and replay instructions](../../../verification/README.md)

ADR 0004 gives the source language one quantitative typing judgment. The elaborator checks value types, effects and ownership together, emits typed HIR, and discharges proof obligations before erasure. It distinguishes definitional equality, static subtyping, gradual consistency and proposition entailment. Its v2.0 plan already calls for a small auditable dependent kernel and export of normalized Eshkol judgments to Lean. Univalence and higher inductive types can be deferred. [ADR 0004, core static language and v2.0 gates](0004-type-system-trajectory.md) · [ADR 0000, staged trajectory](0000-unified-trajectory.md)

Verificatum starts at the proof boundary. In R0, Lean produces a proof term and Eshkol checks it against a separately authored proposition. In the planned forward direction, Eshkol produces a normalized judgment for Lean to check. To use either result for a source program, the consumer must bind the proposition to the particular source obligation, including every hypothesis it relies on. Source elaboration and authorization to lower typed HIR stay with ADR 0004.

## Decision

**Only a fresh acceptance by the named proof checker, against the independently supplied target and the fragment's declared premises, may discharge a Verificatum proof obligation.** The gradual checker may discover obligations and supply candidate terms. Its success, coercions, inferred types, and cached results confer no proof authority. Erasing evidence or using a refinement requires authorization for the specific action under the source judgment. A dynamic check may still be appropriate under the language rules. Its outcome remains a dynamic observation. [Gradual type relation](0013-gradual-type-relation.md)

Each fragment defines its input format, term and declaration grammar, conversion rules, axiom policy, resource bounds and observable verdicts. The checker admits the supplied declarations under that policy, resolves names structurally, and checks the selected body against the independently bound target. A receipt identifies the checked declarations and any assumptions introduced by the input. Changing a transparent definition used in the statement must trigger target comparison again, even when the theorem's name survives.

| Verdict | Meaning for the consumer |
|---|---|
| `ACCEPT` | The derivation establishes the requested target under the recorded fragment and premises. |
| `REJECT` | A supported rule, declaration policy or target comparison failed. |
| `UNSUPPORTED` | The input needs a format or rule outside this checker profile. |
| `ERROR` | Malformed input, resource exhaustion, or execution failure prevented a verdict about the proof. |

Only `ACCEPT` discharges the obligation. In particular, a timeout leaves it open.

## Contracts and computational authority

A consumer independently fixes the subject, observation, action, permitted variation and required evidence for each obligation. Exact equality, refinement, bounded error, trace agreement, risk and resource bounds describe different observations. For example, a grid-point error budget can permit a different arithmetic schedule. A risk bound identifies the distribution, event alignment, recovery policy and measured outcome. Conjoining obligations requires evidence for each one.

A proof checker decides whether the proposition follows. A consumer gate decides whether the checked claims and their applicability premises suffice for the proposed action in the current state. Only the gate issues a capability. Every receipt carries an explicit permission class. IR review, lowering, native release, execution and publication are distinct. The consuming entry point requires its own class. A change of class or policy requires fresh authorization. The ticket binds the class, policy generation, contract, artifact, checked evidence IDs and live premises. Consumption compares the independently expected bytes with the privately retained artifact. A producer may submit bytes and certificates. The consumer owns the requirements, checker registry and revocation state. Under ADR 0004 the compiler still determines which obligations its source judgment generates.

| Evidence level | Scope of the claim |
|---|---|
| Model-proved | A theorem establishes a property of the stated model and assumptions. |
| Executable-observed | A named execution produced the specified observations. |
| Artifact-validated | A checker established the requested relation on specific artifact bytes. |
| Refinement-established | A proved relation connects the named representations or execution models. |

The gate requires the level specified for each obligation. A refinement claim needs its own evidence. An authorization decision identifies its contract, artifact, context and evidence nodes by content digest. Each observation identifies its checker inputs. A consumer may reuse a checked observation only after validating its entire dependency closure under the applicable trust policy. Executable-specific premises can include linked helper bytes, ABI, allocator generation, model and caller effects. A descriptor generation must advance across intervening writes even when bytes later regain their original values.

On failure, record the first violated rule, required and observed values, and witness location when available. Missing premises, unsupported semantics and exhausted resources yield an inconclusive decision. Either outcome withholds permission. A consumer choosing a fallback obtains a separate decision for that artifact. A receipt explains the decision. Only a live capability can release the scoped artifact.

The first implementation is a sequential host-side gate shared by three consumers. Its Lean model establishes authorization soundness conditional on checker soundness and consumer-policy sufficiency. The replay translates the Python decision loop and byte-release predicate used by the gate into Lean and verifies them against the model. This boundary assumes a faithful translation of the admitted Python subset, correctly bound observations, and a trusted issuer that constructs the scope and looks up single-use tickets. Persistent graph storage, cross-build reuse and compiler integration each require further evidence. [Contract and authorization R0](../../../verification/authorization/README.md)

### Exercising authority

Execution validity is a separate claim. A consuming entry point identifies its expected artifact and sink, checks live applicability, and binds the actual operation to the released snapshot. Each implementation names the interval over which its policy and execution context remain stable. A restorable policy record requires independent live authority before a consumer can rely on its generation.

R1 exercises the shipped ABI fingerprint operation on R2's reviewed folded CLI. A replaced pathname gives identical output on the reference executable. The replay distinguishes them using the kernel's executable identity, the argument and environment block, and loaded file identities. The demonstrated scope is one Linux release, a process-local issuer and a trusted loader environment. [R1 reproduction and limits](../../../verification/consumption/README.md)

## Trusted computing base and evidence

| Claim | What warrants it | What remains trusted or open |
|---|---|---|
| A judgment holds in the Lean model | Lean checks the theorem and reports its axiom dependencies. | Lean's kernel, the recorded axioms, the definitions of the calculus, and the choice of statement. |
| This Eshkol executable accepted this target | The replay runs the pinned checker on proof and target bytes, records the declaration closure and tests altered inputs. | The independent target and premises, Eshkol importer, kernel and VM, Node launcher and byte transport, host libraries, OS and hardware. The exporter and reference checker are used to prepare and cross-check fixtures. The candidate executes independently of them. |
| Executable acceptance refines modeled acceptance | A refinement argument connects byte decoding, declaration admission, target binding, conversion and executable behavior to the modeled judgment. | Each unproved link stays explicit. R0 leaves this end-to-end argument open. |

The current Lean theorems establish soundness of successful modeled term checking, binding of the modeled export to a portable target in a valid environment, and commutation of binder erasure with beta contraction. Their axiom reports are distinct from the axiom allowlist on imported declarations. R0 records both. [Typing model](../../../verification/lean-kernel/Typing.lean) · [Binder model](../../../verification/lean-kernel/Binders.lean) · [R0 premises and digests](../../../verification/lean-kernel/R0.toml)

## Receipts and release use

ADR 0010 records two ways a green result can mislead. A coverage counter once credited a construct when its spelling appeared in a generator. VM and native outputs can agree because both paths share a defective reader or printer. For each claimed guarantee, exercise the real checker, record which components the oracles share, and keep a control that would fail for the relevant semantic defect. For target binding, a valid proof of a changed proposition must reach target comparison and fail there. A parser error would leave this property untested. [ADR 0010, evidence and independent oracles](0010-closed-loop-assurance.md)

Every claimed fragment supplies a versioned receipt and a runnable verifier. Bind the target, input bytes or their reproducible derivation, checker sources and runtime artifacts, specification, toolchain versions, assumptions, bounds, expected verdicts and reasons. Include a valid proof, an invalid derivation, and a valid derivation of a changed target. Record which properties have a model proof, a replay on the executing implementation, or a proved refinement. ADR 0014 governs measurement of runtime and release invariants. This proposal applies its fail-closed receipt discipline to proof checking. A release gate replays the receipt against the named artifacts. A changed digest, missing control or unexpected verdict leaves the claim unqualified. [Closed-loop assurance](0010-closed-loop-assurance.md) · [Release invariant contracts](0014-release-invariant-contracts.md)

R0 is one worked example. Its replay checks 16 verdicts across all four outcomes, audits the Lean theorem dependencies and compares binder operations on 1,560 generated rows. Its scope covers a restricted dependent-function calculus with a pinned empty inductive package, safe monomorphic declarations and bounded conversion. General Lean inductives, arbitrary universe conversion and the runtime correctness of compiled Eshkol remain separate obligations. The binder rows exercise substitution. They do not establish soundness for an entire evaluator. [Executable replay](../../../verification/lean-kernel/replay.py)

## Staging and alternatives

The first refinement milestone is faithful decoding of the supported export into structured names, levels, declarations, terms and the selected target. The second connects successful Eshkol substitution, conversion and inference to the corresponding Lean judgments. A later exact-arithmetic solver can supply a certificate to another small checker with its own soundness theorem and receipt.

1. Retain R0's separately authored target, changed-meaning control and runnable receipt as the admission baseline. New rules or formats get a new receipt that names their additional premises.
2. For an optimized BLC evaluator, first compare complete one-based de Bruijn normal forms on open terms, including reductions beneath binders. For a verified refinement claim, define how an environment, thunk heap and continuation denote source terms. Show that transitions preserve that denotation up to beta equivalence, that completed readback is normal, and that completion on normalizable inputs is preserved in the stated resource model. Fuel exhaustion remains inconclusive. ADR 0017 concerns probabilistic choice, exact bounded outcome distributions and bounded program search. A future stochastic claim would need its own observation relation and receipt. R0 uses ordinary deterministic BLC. [Lambda foundations](0005-lambda-foundations-programs-to-weights.md) · [Stochastic BLC](0017-stochastic-binary-lambda-calculus.md)
3. When ADR 0004 exports its first normalized judgment, bind that export to an independently specified Lean proposition and replay it. Any compiler optimization or erasure that consumes this evidence must name the source judgment, the observable behavior it preserves, and its implementation boundary.

Calling the reference kernel from the Eshkol candidate would test delegation while leaving the Eshkol kernel unexercised. Treating a model theorem as an executable guarantee would leave importer and runtime errors invisible. Independent replay costs a second implementation of each admitted rule and maintenance of reproducible evidence. It also makes disagreements observable at the boundary where a compiler or release decision would consume them.
