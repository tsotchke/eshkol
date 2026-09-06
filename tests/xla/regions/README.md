# tests/xla/regions — expected region formation

Fourteen Eshkol programs and, beside each, a `<name>.expected.json` stating
what stage S5 region formation should produce for it.

The expectations were written from `docs/design/ESHKOL_S_FRAGMENT.md`,
`lib/backend/xla/builtin_classification.yaml` and
`lib/backend/xla/device_lowering_table.yaml`, before any pass existed to
compare against. They are an independent reading of the contract. Where an
expectation and the implementation disagree, that is a finding about one of
them, not a test to be edited into agreement.

## Schema

    {
      "program": "tests/xla/regions/<name>.esk",
      "intent":  "one sentence",
      "regions": [ {"ops": [...], "inputs": N, "inside_gradient": false} ],
      "breaks":  [ {"construct": "...", "reason": "...", "builtin": "..."} ]
    }

`regions` is ordered by first appearance in source order. `ops` is the multiset
of device operations in the region in evaluation order, each named by the
Eshkol builtin, operator or user function that produced it. `inputs` is the
number of distinct values the region takes from the host. `breaks` is a
multiset; `builtin` is omitted when the break is a special form.

Break reasons: `host-builtin`, `host-with-device-inner`,
`non-admitted-construct`, `host-value-domain`, `unknown-shape`, `no-lowering`.

## Reading conventions these expectations apply

The contract does not settle every question a walker has to answer. The rules
below are the ones used here, uniformly, so that a disagreement is legible
rather than diffuse. Each is a candidate for the pass to overrule; none of
them should be resolved silently.

1. **A node is reported as a break only when it is ineligible in its own
   right** — a host builtin call, a non-admitted special form, a host-domain
   literal, or a device builtin with no lowering. A node that is ineligible
   only because a descendant is is not itself reported.
2. **A host-domain literal is reported only when it is not already an argument
   of a broken host construct.** `(display "text")` is one break, not two;
   `(define TAG 'checksum)` is a `host-value-domain` break.
3. **An eligible node may take host values as inputs.** A device builtin whose
   operand is an ineligible expression is still a region root; the operand is
   outside the region, counts toward `inputs`, and is walked normally.
4. **`inputs` counts distinct free variables plus vector literals.** A vector
   literal is host data that has to be transferred; a scalar numeric literal
   becomes a device constant and is not an input. A variable referenced twice
   counts once.
5. **A top-level `define` is a container, not a node**: it is never eligible
   and never a break. When its right-hand side is eligible, the right-hand
   side is the region root. The same holds for `let`/`let*` binding positions
   and `begin`.
6. **A user function whose whole body is eligible has that body outlined as a
   region at its definition site**, and calls to it are eligible nodes that
   appear as an op named by the function. A function that is recursive, or
   whose body is not wholly eligible, is not promoted, and each call to it is a
   break; these expectations use `non-admitted-construct` for that break
   because the reason list has no entry for it.
7. **A lambda directly in the procedure argument of a
   `host-with-device-inner` builtin is not reported as a lambda break** — that
   is the sanctioned inner evaluation. A lambda handed to a plain `host`
   builtin such as `gradient` is reported, and its body is still walked.
8. **A conditional's ops are listed predicate-first, then arms in source
   order**, since a `stablehlo.case` carries both arms.
9. **A lowering is "available" if the builtin has a row in
   `device_lowering_table.yaml` or a `DeviceOpKind` in
   `inc/eshkol/backend/xla/device_lowering.h`.** The yaml table alone is not
   the whole set — it omits `tensor-add`, `matmul` and `tanh`, which do lower.

## Rulings

The expected-regions files in this directory were written from
`docs/design/ESHKOL_S_FRAGMENT.md` by someone who had not seen the pass, which
is the only arrangement under which they grade anything. On the first run the
pass and the corpus agreed on 6 of 14 programs. Every disagreement was
triaged, and the outcome was four defects in the pass (fixed, each with its
own commit) and nine questions the contract did not answer. Those nine are
settled here.

A ruling is not a licence to change an expectation to match output. It is a
statement of what the right answer is, written before the file is changed, so
that the next person can disagree with the ruling rather than with a number.
Each revised file names the rulings it was revised under in its `rulings`
field.

**R1 — a call into a host-bodied function is `host-function`, not
`host-builtin`.** A top-level function of the module whose own body leaves
the fragment is not a builtin and is not classified in
`builtin_classification.yaml`. Its breaks are reported at its definition; the
call site reports only that it cannot be inlined.

**R2 — a leaf is data, not a construct, and is never a break of its own.**
Condition 1 excludes strings and lists from the value domain, so read
literally every string literal under a `(display "x")` is a break. That names
the same boundary twice and puts a break on every argument of every host call
in the program. The construct that cannot go to the device is the `display`.

**R3 — "has a lowering" means a lowering that has been MEASURED.** The pass
admits a builtin into a region only when it appears in
`device_lowering_table.yaml`, either as a composition or in the `core_ops`
section, and every `core_ops` row is a coverage claim an already-passing
parity row makes. `reshape`, `<`, `>` and `floor` are device-LABELLED and are
not there, so they break with `no-lowering`.

The consequence is worth stating plainly rather than burying in a corpus
file: **no region can contain a conditional today**, because a conditional's
predicate is a comparison and no comparison has a measured parity row. That
is why `09_if_both_arms` forms three small regions instead of one
`stablehlo.case`. It is a gap in the measured surface, not in this pass, and
closing it is a build item: add comparison and reshape rows to
`tests/xla/op_parity_test`, then the `core_ops` entries, and the conditional
region appears with no change to the pass.

**R4 — a call into an eligible top-level function contributes its BODY's
operations to the region that calls it.** The region genuinely contains those
operations once the call is inlined, and "ops per region" is meant to say
what the device will execute. The function's own body is ALSO a region at its
definition site, because it is itself a maximal eligible subgraph; that is
not double counting, it is two different regions, one per call path.

**R5 — a lambda passed to a `host-with-device-inner` builtin is not a
break.** That label names the inner evaluation that stays eligible ("the
argument procedure's body, when that body is itself in Eshkol-S"), so the
body is outlined and the lambda is the sanctioned boundary. Under any other
callee a lambda is a closure in value position and breaks as one.

**R6 — `quote` is a value-domain break, not a control-flow one.** What it
yields is a symbol or a list.

**R7 — a special form whose keyword is a classified builtin reports that
label.** `gradient` is labelled `host` because the AD tape is a host value;
reporting it as "a construct condition 3 does not admit" is true and says
nothing about why.

**R8 — a named `let` breaks today.** Condition 3 admits a tail-recursive loop
over fragment-typed state, and this pass does not yet prove that the loop
state is fragment-typed and fixed-shape across iterations. Until it does, a
named let is reported as a break rather than assumed to be a
`stablehlo.while`. Build item: prove loop-carried state and admit the loop.

**R9 — `cond` reaches the pass as calls with a computed callee.** Its arms
are not tagged as a conditional and their operator position does not hold a
name, so each arm outlines on its own and the `cond` itself breaks. Written
as nested `if` the same program would be one region once R3's gap is closed.

**R10 — a conditional is a break until the region emitter emits
`stablehlo.case`.** Condition 3 admits `if` with both arms in the fragment,
and it does lower to `stablehlo.case` — but `region_execution.cpp` does not
emit one, so admitting a conditional would form a region that cannot be
compiled and move the failure from a reported graph break to an error inside
the emitter. It is treated as any other unlowered op and becomes eligible the
day the emitter emits the case. The programs with a conditional therefore
report TWO breaks around it: the predicate (`<` or `>`, no measured
comparison lowering, R3) and the `if` itself.

Closing R3 and R10 together is what a conditional region needs, and it is not
as cheap as R3 alone suggested: a `Compare` device op whose result is a
boolean tensor — which the f64 host-marshalling path does not carry today — a
host reference entry point for it, a parity row measuring both, and
`stablehlo.case` emission in the region emitter. Four pieces, not one.
