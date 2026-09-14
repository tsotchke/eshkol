# ADR-0013: One gradual type relation

- **Status:** Accepted
- **Date:** 2026-09-14
- **Decision owners:** Eshkol type-system maintainers

## Context

Eshkol has structural procedure signatures, tracked pairs and unions in addition to its nominal type graph. The checker also has gradual values (`Value` and unresolved `Invalid`), return annotations, recursive inference, and explicit `the` ascriptions. Keeping these rules in separate branch, call, return, and ascription checks caused the same types to behave differently by syntax. In particular, `if` disagreed with `cond`, signatures printed as unknown, and one branch of a return check rejected a dynamic result.

## Decision

`TypeRelation` is the sole owner of judgments over types. It provides static subtyping, gradual consistency and consistent subtyping, flow evidence, casts, join and meet, occurrence-test narrowing, inferred-slot widening, pair projections, and presentation. Its rules are defined over the interned types in `TypeEnvironment`:

- `Never` is bottom and `Value` is top for static subtyping; unresolved `Invalid` carries no evidence for a static subtype judgment.
- `Value` and unresolved `Invalid` are consistent with every type. Procedure consistency compares arrow domains and results consistently, preserving contravariant parameters and covariant results.
- Pair components are covariant. Union subtyping checks membership against an arm and requires every source arm to fit the target.
- Joins and meets recurse through unions, pairs, and arrows, then use the nominal graph. `Never` is the join identity; disjoint concrete types meet at `Never`.
- `compatibility` names whether a flow is identical, a static upcast, gradual, numeric, or rejected. Ascriptions additionally accept overlapping types.
- Recursive inference slots refuse an unrepresentable top join as a conflict, while `do` slots adopt it. The special `Boolean`/value loop shape is stated once in the `InferenceSlot` policy.

`TypeEnvironment` retains small compatibility facades for existing callers of static subtype, join, and type printing. Those facades delegate to `TypeRelation`; they do not contain independent rules. New checker code uses `TypeRelation` directly. The nominal graph remains available for operations whose contract specifically asks for registered supertype chains.

Branch-producing forms use the same join judgment. Return annotations and procedure arguments use consistent subtyping. `the` uses the relation's castability rule. Procedure diagnostics use the relation printer, so function signatures and the generic procedure type have stable readable names.

## Consequences

A concrete contradiction remains visible: `String` does not consistently subtype `Number`, and `(-> String String)` does not satisfy `(-> Number Number)`. A dynamic component remains acceptable inside a signature. Branches with no more specific representable common type produce `Value` uniformly. Typechecker policy choices such as whether an inferred slot adopts `Value` are explicit inputs to the relation rather than duplicated exception logic.

The VM and native `json-get-in` definitions both allow the optional default described in the standard-library reference. The VM surface runner must propagate a failing fixture result; a VM-only fixture's expected native compile failure is not evidence that its VM runner failed.

## Verification

The direct `type_relation_test.cpp` contract covers arrow variance, dynamic components, disjoint joins/meets, pair covariance, printing, and both widening policies. Scheme tests cover positive and negative branch, return, and arrow cases. VM parity covers both `json-get-in` arities. The type-system suite checks compile outcome and diagnostics so a printed warning alone cannot make a rejected program pass.
