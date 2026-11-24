# The Eshkol Constitution

Eshkol Verificatum records which semantic guarantees hold for which programs. A guarantee earns implementation coverage when evidence connects the named production entry point to the specification. Each claim identifies its fragment, observable result, assumptions, and a way to detect a meaningful change in behavior.

| Commitment | Obligation to establish |
|---|---|
| Binding | Substitution preserves the referent of every surviving variable, including free variables moved under a binder. |
| Types | Each evaluation step of the admitted typed fragment preserves the typing judgment. The specification identifies possible errors and dynamic checks. |
| Compilation | Every observable behavior of the compiled program is allowed by the source semantics. For a deterministic, terminating pure fragment, the two results agree. Effects, errors, and divergence need explicit observations before extending the claim. |
| Mathematics | An exact result satisfies the stated mathematical relation. An approximate result identifies its domain and error bound. For an affine solution family, check the particular solution, independence of the reported basis, and that it spans the entire nullspace. |
| Evidence | Acceptance binds a checked derivation to the independently specified proposition and its declarations, under an explicit axiom policy. A different well-typed theorem must fail the target comparison. |
| Resources | For each supported completion, error, or cancellation path, resource transitions satisfy the ownership protocol. Eventual cleanup also needs explicit scheduling or termination assumptions. |

**Computational authority is derived.** The consumer fixes the contract and proposed action. Proof acceptance establishes the target under its assumptions. The action gate checks coverage, applicability to the actual artifact and current premises, and releases authority for that action alone. A changed live premise can expire permission while the theorem continues to hold. ADR 0004 defines the source-language judgment. [Authorization decision and replay](authorization/README.md)

**Authority is exercised at consumption.** Proof acceptance establishes the claim, authorization binds it to an action, and the consuming subsystem must establish that it performed that action on the authorized artifact under the required context. The ABI fingerprint handoff observes the executable actually selected by the OS. [Verified consumption R1](consumption/README.md)

## R0 — Independent proof replay

Eshkol checks the Lean 4.35.0 export of modus tollens using the bundled WASM VM. It accepts the genuine derivation, rejects the well-scoped wrong body at its application-domain check, and rejects a valid theorem named `mt` after changing the definition of negation. In the last case con-ron accepts the altered declaration in its own environment. Eshkol refuses it at comparison with the independently specified proposition.

| Fixture | SHA-256 | Reference verdict |
|---|---|---|
| [mt.ndjson](lean-kernel/mt.ndjson) | `d4cb6b03bc24222a3c3c1608158da7aedbc466602d63bfee40ac31a41eb7d664` | con-ron accepts three declarations |
| [mt-wrong-body.ndjson](lean-kernel/mt-wrong-body.ndjson) | `f834bcebd6b66b0d4a6350b99b3d446dc619e154c0951bec44104462b1dd5b5d` | con-ron rejects an application type mismatch in mt |

The original 50-record export comes from `leanexport Init.Core -- mt` with format 3.1.0. It contains the empty inductive and its generated recursor, the definition of negation, and the theorem, with no axiom declarations. In the second fixture a single argument reference gives the inner application a different function. Both terms are well scoped. The genuine body is `g (f a)`. The changed body is `g (g a)`, where `g` needs a B and `a` has type A.

The [target](lean-kernel/Target.lean) states modus tollens with negation expanded. The checker validates the candidate and target in separate environments and unfolds checked transparent definitions before comparison. Shared use of `False` requires the canonical empty inductive package, including its generated recursor. The theorem replay reports the declarations `False`, `False.rec`, `Not`, and `mt`, with an empty list of assumptions introduced by the export.

### Run the experiment

With Node, Python 3.11+ (or 3.10 with tomli), and the official Lean 4.35.0 tools on PATH:

```sh
python3 verification/lean-kernel/replay.py
```

The [replay](lean-kernel/replay.py) authenticates the [R0 receipt](lean-kernel/R0.toml), hashes the artifacts, re-exports the genuine proof, audits the model theorems' axioms, checks the binder corpus, then checks every required verdict. A different receipt, missing evidence, or changed digest fails the run. The receipt binds this release to specific sources and WASM artifacts. Preserve the original corpus and independent target across releases.

For standalone use, supply a proof export and a separate export containing the expected proposition as a definition:

```sh
verification/lean-kernel/check mt.ndjson mt target.ndjson Verificatum.target
```

The [launcher](lean-kernel/check) uses Node for file I/O, strict UTF-8 decoding, scalar transport, limits, and running the pinned VM. Only numeric character data enters VM-generated input expressions. The [Eshkol importer](lean-kernel/import.esk) reads JSON and the [kernel](lean-kernel/kernel.esk) checks declarations. Each check starts from an empty environment. The kernel admits safe monomorphic definitions and theorems, concrete universes and dependent functions, plus the specific empty inductive package needed here. Recursor application and general inductive admission are outside R0. The CLI selects string-only dotted names; inside both imports string components, numeric components, and their boundaries are preserved.

The result is one JSON object with `status`, `reason`, `closure` and `assumptions`. Only `ACCEPT` attests the checked environment. A supported typing, declaration or target violation gives `REJECT`. Rules outside the profile give `UNSUPPORTED`; malformed data, exceeded budgets or runtime failure give `ERROR`. Exit codes are 0, 1, 2 and 3 respectively. R0 checks the entire supplied declaration stream, including declarations unused by the theorem. Both frozen proofs were also replayed in a read-only filesystem namespace with no Lean toolchain or reference kernel executable mounted.

### What Lean proves

[Binders.lean](lean-kernel/Binders.lean) proves `erase_beta` for arbitrary open lambda bodies and arguments in the pure annotated fragment. The translation erases annotations and increments every variable index. A separate lemma equates fused substitution with the three passes modeled on Eshkol's BLC reducer. The [differential runner](lean-kernel/binder-replay.cjs) compares the full output term of both actual Eshkol implementations against Lean on 1,560 cases. The original 684, an exhaustive 871 within explicit size bounds, and five named controls include capture, shadowing, traversal of both application branches, and depth 24. The receipt pins the generated inputs and expected terms together.

[Typing.lean](lean-kernel/Typing.lean) gives two guarantees at different interfaces:

- `check_sound`: if the modeled term checker accepts a closed term at a type, there is a derivation of that typing judgment relative to the supplied environment.
- `target_binding_sound`: on acceptance of the modeled export, its preceding declarations build a valid environment, its independently expanded target is a portable proposition in that environment, and the selected body inhabits that proposition. The model checks transparent definitions before extending the environment. The only primitive constant available in a portable target is the independently pinned empty type.

The term model uses a proof-producing normalizer and resolved constant IDs. The Eshkol implementation uses weak-head conversion, structured names, and separate imports. Faithfulness of byte decoding, recursor admission, correspondence between the two target representations, and executable refinement remain proof obligations. The compiler, VM, Node runtime and operating system are part of the trusted execution path. [Lean export specification](https://lean-lang.org/doc/reference/latest/External-Kernels/)
