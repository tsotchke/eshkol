# Depth-coverage matrix — whole-language depth-parametric testing (Pillar P6 auditor)

> Generated + gated by `scripts/check_depth_coverage.py` against
> `scripts/depth_coverage_registry.json` (the machine-readable source of truth).
> This document is the human-readable mirror; regenerate the tables with that
> script if you edit the registry. Charter: `.swarm/DEPTH_PARAMETRIC_TESTING.md`.

## Why this exists

ESH-0117 (reverse-over-nested-forward returned 0) proved that a composition can be
**correct at depth 1 and silently wrong at depth 2+**, and that our five fixed-depth
harnesses could not see it. Pillar P6 answers with depth-parametric sweeps for
*every composable construct*. This auditor is the meta-check that makes that claim
**auditable**: it enumerates the entire language construct surface from the
compiler's own AST op enum (`inc/eshkol/eshkol.h`) and records, for each construct,
whether its depth is swept parametrically (and by which pillar), only touched by a
fixed-depth harness, or **not covered at all**. No composable construct may be
silently tested only at shallow depth without showing up here as a GAP.

## How coverage is decided

Each construct is classified **composable** (it can nest within itself or other
forms, so a depth-dependent miscompile is possible) or **leaf** (an opaque-handle
runtime primitive, declaration, or predicate with no user-syntactic nesting depth —
e.g. `make-kb`, the DNC/SDNC ops, type predicates). Only composables need a depth
sweep. A composable is **covered** when it names an owning depth-sweep pillar; it is
a **GAP** when no pillar sweeps it (each GAP carries an `ESH-####` tracking task).

The depth-sweep pillars (each a `test/*-depth-parametric` branch):

- **P6a** AD depth — `test/ad-depth-parametric`
- **P6b** recursion/control depth — `test/recursion-depth-parametric`
- **P6c** syntax/data nesting depth — `test/nesting-depth-parametric`
- **P6d** numeric depth — `test/numeric-depth-parametric`
- **P6e** metaprog/module depth — `test/metaprog-depth-parametric`
- **P6f** tensor/collection/string depth — `test/tensor-collection-depth`

Fixed-depth harnesses already on master (feature-pair / shallow, *not* a depth
sweep — they are complementary, not a substitute): `tests/ad_oracle`,
`tests/differential`, `tests/edge_matrix`, `tests/stress`, `tests/vm_parity`.

## Coverage summary (auto-derived)

- **AST ops in enum:** 110 (67 composable, 43 leaf)
- **Supplemental composables** (numeric tower, collection HOFs, recursion patterns): 20
- **Total composable constructs:** 87
- **With a depth sweep (pillar-owned):** 77
- **Un-swept GAPS:** 10 -> **88.5% depth-sweep coverage**

### The explicit GAP list (composable, no depth sweep)

| GAP construct(s) | Tracking task | Reason un-swept |
|---|---|---|
| `with-region`, `owned`, `move`, `borrow`, `shared`, `weak-ref` | **ESH-0140** | Memory/ownership/region forms nest lexically but no P6 pillar owns them. |
| `:` type-annotation, `forall` | **ESH-0141** | Type expressions nest (arrow/forall) but no pillar sweeps type-expression nesting depth. |
| `unify`, `walk` | **ESH-0142** | Logic-engine term operations recurse over nested compound terms; no depth sweep for logic-term nesting. |

Everything else composable is assigned to a pillar (see full tables below). Note:
the pillar generators are landed independently on the `test/*-depth-parametric`
branches; this auditor gates *assignment* (every composable is owned by a sweep or
a tracked gap), which is the guarantee that no construct is silently shallow.

---

## AST operations (from `inc/eshkol/eshkol.h`)

| Construct | Class | Depth sweep | Note |
|---|---|---|---|
| `ESHKOL_INVALID_OP` | leaf | — | sentinel, not a real construct |
| `ESHKOL_COMPOSE_OP` | composable | **P6b** | function composition chains |
| `ESHKOL_IF_OP` | composable | **P6b** | nested conditionals |
| `ESHKOL_ADD_OP` | composable | **P6d** | arithmetic nesting |
| `ESHKOL_SUB_OP` | composable | **P6d** | arithmetic nesting |
| `ESHKOL_MUL_OP` | composable | **P6d** | arithmetic nesting |
| `ESHKOL_DIV_OP` | composable | **P6d** | arithmetic nesting (rational/complex promotion) |
| `ESHKOL_CALL_OP` | composable | **P6b** | application / recursion depth |
| `ESHKOL_DEFINE_OP` | composable | **P6c** | nested internal defines / scopes |
| `ESHKOL_SEQUENCE_OP` | composable | **P6b** | nested begin sequencing |
| `ESHKOL_EXTERN_OP` | leaf | — | FFI declaration |
| `ESHKOL_EXTERN_VAR_OP` | leaf | — | FFI variable declaration |
| `ESHKOL_LAMBDA_OP` | composable | **P6c** | nested closure capture chains |
| `ESHKOL_LET_OP` | composable | **P6c** | nested let / named-let |
| `ESHKOL_LET_STAR_OP` | composable | **P6c** | nested let* |
| `ESHKOL_LETREC_OP` | composable | **P6c** | nested letrec |
| `ESHKOL_LETREC_STAR_OP` | composable | **P6c** | nested letrec* |
| `ESHKOL_AND_OP` | composable | **P6b** | short-circuit nesting |
| `ESHKOL_OR_OP` | composable | **P6b** | short-circuit nesting |
| `ESHKOL_COND_OP` | composable | **P6b** | multi-branch nesting |
| `ESHKOL_CASE_OP` | composable | **P6b** | case nesting |
| `ESHKOL_MATCH_OP` | composable | **P6c** | nested pattern matching |
| `ESHKOL_DO_OP` | composable | **P6b** | nested iteration |
| `ESHKOL_WHEN_OP` | composable | **P6b** | one-armed conditional nesting |
| `ESHKOL_UNLESS_OP` | composable | **P6b** | negated conditional nesting |
| `ESHKOL_QUOTE_OP` | composable | **P6c** | nested quoted data |
| `ESHKOL_QUASIQUOTE_OP` | composable | **P6c** | nested quasiquote templates |
| `ESHKOL_UNQUOTE_OP` | composable | **P6c** | nested unquote escapes |
| `ESHKOL_UNQUOTE_SPLICING_OP` | composable | **P6c** | nested unquote-splicing |
| `ESHKOL_SET_OP` | composable | **P6c** | mutation of captures across nested scopes |
| `ESHKOL_DEFINE_TYPE_OP` | composable | **P6e** | type alias definitions in module scope |
| `ESHKOL_IMPORT_OP` | composable | **P6e** | legacy path import chains |
| `ESHKOL_REQUIRE_OP` | composable | **P6e** | module require chains |
| `ESHKOL_PROVIDE_OP` | composable | **P6e** | module export surface |
| `ESHKOL_WITH_REGION_OP` | composable | GAP (ESH-0140) | nested region/arena scopes — no depth sweep owns memory/ownership forms |
| `ESHKOL_OWNED_OP` | composable | GAP (ESH-0140) | linear-type wrapper nesting — un-swept |
| `ESHKOL_MOVE_OP` | composable | GAP (ESH-0140) | ownership transfer nesting — un-swept |
| `ESHKOL_BORROW_OP` | composable | GAP (ESH-0140) | borrow-scope nesting — un-swept |
| `ESHKOL_SHARED_OP` | composable | GAP (ESH-0140) | refcounted-alloc nesting — un-swept |
| `ESHKOL_WEAK_REF_OP` | composable | GAP (ESH-0140) | weak-ref nesting — un-swept |
| `ESHKOL_TENSOR_OP` | composable | **P6f** | nested tensor construction / rank depth |
| `ESHKOL_DIFF_OP` | composable | **P6a** | differentiation nesting |
| `ESHKOL_DERIVATIVE_OP` | composable | **P6a** | derivative^n |
| `ESHKOL_GRADIENT_OP` | composable | **P6a** | gradient-over-nested-forward (the ESH-0117 blind spot) |
| `ESHKOL_JACOBIAN_OP` | composable | **P6a** | jacobian nesting |
| `ESHKOL_HESSIAN_OP` | composable | **P6a** | hessian nesting |
| `ESHKOL_DIVERGENCE_OP` | composable | **P6a** | divergence nesting |
| `ESHKOL_CURL_OP` | composable | **P6a** | curl nesting |
| `ESHKOL_LAPLACIAN_OP` | composable | **P6a** | laplacian nesting |
| `ESHKOL_DIRECTIONAL_DERIV_OP` | composable | **P6a** | directional-derivative nesting |
| `ESHKOL_TYPE_ANNOTATION_OP` | composable | GAP (ESH-0141) | nested type expressions (arrow/forall) — no pillar sweeps type-expression nesting depth |
| `ESHKOL_FORALL_OP` | composable | GAP (ESH-0141) | nested polymorphic type quantifiers — un-swept |
| `ESHKOL_GUARD_OP` | composable | **P6c** | nested guard/raise |
| `ESHKOL_RAISE_OP` | composable | **P6c** | nested guard/raise |
| `ESHKOL_LET_VALUES_OP` | composable | **P6c** | nested multiple-values binding |
| `ESHKOL_LET_STAR_VALUES_OP` | composable | **P6c** | nested sequential multiple-values binding |
| `ESHKOL_VALUES_OP` | composable | **P6c** | nested values producers |
| `ESHKOL_CALL_WITH_VALUES_OP` | composable | **P6c** | producer/consumer nesting |
| `ESHKOL_DEFINE_SYNTAX_OP` | composable | **P6e** | nested macro expansion depth |
| `ESHKOL_LET_SYNTAX_OP` | composable | **P6e** | nested local macros |
| `ESHKOL_LETREC_SYNTAX_OP` | composable | **P6e** | nested recursive local macros |
| `ESHKOL_CALL_CC_OP` | composable | **P6b** | call/cc nesting |
| `ESHKOL_DYNAMIC_WIND_OP` | composable | **P6b** | dynamic-wind nesting |
| `ESHKOL_LOGIC_VAR_OP` | leaf | — | logic-variable atom |
| `ESHKOL_UNIFY_OP` | composable | GAP (ESH-0142) | unification recurses over nested compound terms — no depth sweep for logic-term nesting |
| `ESHKOL_MAKE_SUBST_OP` | leaf | — | empty substitution constructor |
| `ESHKOL_WALK_OP` | composable | GAP (ESH-0142) | walk resolves nested terms — un-swept |
| `ESHKOL_MAKE_FACT_OP` | leaf | — | fact constructor (opaque) |
| `ESHKOL_MAKE_KB_OP` | leaf | — | knowledge-base constructor (opaque handle) |
| `ESHKOL_KB_ASSERT_OP` | leaf | — | KB mutation primitive |
| `ESHKOL_KB_QUERY_OP` | leaf | — | flat query over opaque KB |
| `ESHKOL_MAKE_FACTOR_GRAPH_OP` | leaf | — | factor-graph constructor (opaque handle) |
| `ESHKOL_FG_ADD_FACTOR_OP` | leaf | — | factor-graph mutation primitive |
| `ESHKOL_FG_INFER_OP` | leaf | — | belief-propagation primitive |
| `ESHKOL_FREE_ENERGY_OP` | leaf | — | scalar functional over opaque beliefs |
| `ESHKOL_EXPECTED_FREE_ENERGY_OP` | leaf | — | scalar functional over opaque model |
| `ESHKOL_MAKE_WORKSPACE_OP` | leaf | — | workspace constructor (opaque handle) |
| `ESHKOL_WS_REGISTER_OP` | leaf | — | workspace mutation primitive |
| `ESHKOL_WS_STEP_OP` | leaf | — | workspace step primitive |
| `ESHKOL_FG_UPDATE_CPT_OP` | leaf | — | factor-graph CPT mutation primitive |
| `ESHKOL_FG_OBSERVE_OP` | leaf | — | factor-graph observation primitive |
| `ESHKOL_LOGIC_VAR_PRED_OP` | leaf | — | predicate |
| `ESHKOL_SUBSTITUTION_PRED_OP` | leaf | — | predicate |
| `ESHKOL_KB_PRED_OP` | leaf | — | predicate |
| `ESHKOL_FACT_PRED_OP` | leaf | — | predicate |
| `ESHKOL_FACTOR_GRAPH_PRED_OP` | leaf | — | predicate |
| `ESHKOL_WORKSPACE_PRED_OP` | leaf | — | predicate |
| `ESHKOL_CASE_LAMBDA_OP` | composable | **P6b** | arity-dispatched closure nesting |
| `ESHKOL_DEFINE_RECORD_TYPE_OP` | composable | **P6c** | nested record instances |
| `ESHKOL_PARAMETERIZE_OP` | composable | **P6b** | nested dynamic bindings |
| `ESHKOL_MAKE_PARAMETER_OP` | leaf | — | parameter object constructor |
| `ESHKOL_COND_EXPAND_OP` | composable | **P6e** | nested feature-conditional expansion |
| `ESHKOL_INCLUDE_OP` | composable | **P6e** | nested include chains |
| `ESHKOL_SYNTAX_ERROR_OP` | leaf | — | parse-time diagnostic form |
| `ESHKOL_KB_QUERY_PREFIX_OP` | leaf | — | prefix query over opaque KB |
| `ESHKOL_DNC_MAKE_OP` | leaf | — | DNC memory constructor (opaque handle) |
| `ESHKOL_DNC_CONTENT_ADDR_OP` | leaf | — | DNC primitive |
| `ESHKOL_DNC_LOC_ADDR_OP` | leaf | — | DNC primitive |
| `ESHKOL_DNC_READ_OP` | leaf | — | DNC primitive |
| `ESHKOL_DNC_WRITE_OP` | leaf | — | DNC primitive |
| `ESHKOL_DNC_ALLOC_WEIGHTS_OP` | leaf | — | DNC primitive |
| `ESHKOL_DNC_READ_GRAD_OP` | leaf | — | DNC primitive |
| `ESHKOL_DNC_PRED_OP` | leaf | — | predicate |
| `ESHKOL_SDNC_PROGRAM_OP` | leaf | — | SDNC program constructor (opaque handle) |
| `ESHKOL_SDNC_RUN_OP` | leaf | — | SDNC primitive |
| `ESHKOL_SDNC_WEIGHT_GRAD_OP` | leaf | — | SDNC primitive |
| `ESHKOL_SDNC_PARAMS_OP` | leaf | — | SDNC primitive |
| `ESHKOL_SDNC_SET_PARAMS_OP` | leaf | — | SDNC primitive |
| `ESHKOL_SDNC_IMPROVE_OP` | leaf | — | SDNC primitive |
| `ESHKOL_SDNC_PRED_OP` | leaf | — | predicate |

## Supplemental composables (non-op: numeric tower, collection HOFs, recursion patterns)

| Construct | Class | Depth sweep | Note |
|---|---|---|---|
| `numeric.int64` | composable | **P6d** | fixnum arithmetic depth |
| `numeric.double` | composable | **P6d** | inexact arithmetic depth |
| `numeric.bignum` | composable | **P6d** | arbitrary-precision arithmetic depth (overflow promotion) |
| `numeric.rational` | composable | **P6d** | exact rational arithmetic depth |
| `numeric.complex` | composable | **P6d** | complex arithmetic depth |
| `data.list-literal` | composable | **P6c** | nested list literals / cons chains |
| `data.vector-literal` | composable | **P6c** | nested vector literals #(...) |
| `data.pair` | composable | **P6c** | nested pairs |
| `data.string` | composable | **P6f** | string composition / length extremes |
| `hof.map` | composable | **P6f** | map over nested collections |
| `hof.filter` | composable | **P6f** | filter over nested collections |
| `hof.fold` | composable | **P6f** | fold-left/right/reduce over nested collections |
| `hof.for-each` | composable | **P6f** | for-each over nested collections |
| `hof.apply` | composable | **P6b** | variadic application depth |
| `hof.curry` | composable | **P6b** | curried-application chains |
| `recursion.named-let-loop` | composable | **P6b** | self-tail loop depth |
| `recursion.mutual` | composable | **P6b** | mutual (2- and 3-cycle) recursion depth |
| `recursion.cps-chain` | composable | **P6b** | continuation-passing chain depth |
| `recursion.through-map` | composable | **P6b** | recursion threaded through map |
| `recursion.through-eval` | composable | **P6b** | metacircular recursion-through-eval depth |
