# tests/xla/regions — expected region formation

Eighteen Eshkol programs and, beside each, a `<name>.expected.json` stating
what stage S5 region formation should produce for it. Fourteen were written
for S5; four (15 to 18) were added by S5b, which admitted conditionals and
loops into regions, and are graded the same way.

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
   order, then `if`**, since a `stablehlo.if` carries both arms. A nested
   conditional in an arm therefore lists its own `if` before the outer one.
   **A loop's ops are the trip predicate, the carried values in binding
   order, the exit expression, then `while`.** The `if` that shapes a loop
   body is the loop's structure and is not listed; the self-call is the
   jump and is not an op.
10. **A comparison at the root of an eligible subtree is not a region.** Its
    value on the host is `#t`/`#f` and a region yields a number, so outlining
    it would change the kind of value the program sees. A comparison is a
    device op only as the predicate of a conditional or loop inside a region;
    at a root its operands are walked and any region under them still forms.
11. **A scalar region input is not a tensor on the host.** A top-level
    `(define T 1.5)`, a `(define S (tensor-sum A))`, a loop bound: each is a
    number in the program and counts as one input with shape `[]`. Generated
    code boxes it into a one-element tensor at the seam and the region runner
    builds the module over rank 0.
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
parity row makes. `reshape` and `floor` are device-LABELLED and are not
there, so they break with `no-lowering`.

The comparisons were the consequence worth stating: until S5b **no region
could contain a conditional**, because a conditional's predicate is a
comparison and no comparison had a measured parity row. S5b added the six
`Compare*` device ops, the host reference entry point
`eshkol_xla_compare_host()`, and seven rows in `tests/xla/op_parity_test`
(all six directions with a tie in the inputs, plus a tensor-against-scalar
broadcast); `<`, `<=`, `=`, `>`, `>=` then joined `core_ops`. The pass
admitted them from the table alone, as R3 said it would. A predicate is i1 on
the device and 0/1 on the host, decided once in `device_lowering.h`.

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

**R8 — a named `let` is admitted as `stablehlo.while` when its loop state is
proven fixed-shape, and breaks by name otherwise.** Condition 3 admits a
tail-recursive loop over fragment-typed state. The pass admits a named let
when all of the following hold, and reports `named-let` as a
`non-admitted-construct` break naming the condition that failed otherwise:

1. the body is `(if pred (name args...) exit)` in either arm order, the
   self-call is the only mention of the loop name, and it passes one value
   per binding — the tail-recursive shape, and nothing else;
2. every initial value, the predicate, every self-call argument and the exit
   expression are eligible, with the loop variables in scope;
3. the predicate is a comparison whose shape is statically a scalar (the
   rank-0 i1 `stablehlo.while` takes);
4. for every binding, the shape of the initial value and the shape of the
   corresponding self-call argument are both statically known and equal —
   that is the proof that the carried state keeps its shape;
5. the loop is not inside a differentiated expression (R11).

The carried tuple is the bindings in order; the exit expression is
evaluated over the final carried values after the loop. `10_named_let_loop`
and `17_while_accumulator` are one region each; a loop whose body breaks
(`02_train_step_print`'s `train`, which prints) reports its breaks where
they are and the loop stays on the host, as it did.

**R9 — `cond` reaches the pass as calls with a computed callee.** Its arms
are not tagged as a conditional and their operator position does not hold a
name, so each arm outlines on its own and the `cond` itself breaks. Written
as nested `if` the same program would be one region once R3's gap is closed.

**R10 — a conditional is admitted when its predicate is a comparison whose
shape is statically a scalar and both arms are eligible.** Condition 3
admits `if` with both arms in the fragment. The predicate has to be a
COMPARISON, not merely an eligible expression: `if` over a tensor value is
host truthiness, which is not a device question. And it has to be a scalar
statically, because `stablehlo.if` and `stablehlo.while` take a rank-0 i1
and a predicate whose rank is only settled at run time would form a region
that may not compile — which is the failure a break exists to report. A
conditional that fails either test breaks as `if`,
`non-admitted-construct`, with the reason in the text; one whose arm breaks
is not itself reported (reading convention 1), only the arm is.

The emitter lowers an admitted conditional one of two ways, and the rule is
written once, in `region_execution.cpp`:

- `stablehlo.select` when the region is inside a differentiated expression,
  or when either arm is a leaf (a variable or a literal). Both arms are
  computed; a region admits no effects, so the only cost is the arithmetic
  of the arm not taken, and a leaf arm has none to skip.
- `stablehlo.if` otherwise: both arms carry operations and one is evaluated.

Inside `gradient` it must be a select: `emitVJP` walks a flat use-def graph
and cannot enter `stablehlo.if`'s regions, while select has a VJP rule (the
cotangent goes down the arm that was taken, zero down the other). That is
how a conditional region differentiates (R11).

The four pieces this needed were the ones R10 named: the `Compare*` device
ops with an i1 result that the host path marshals as 0/1, the host reference
entry point, the parity rows (all under R3 above), and the emission. The
programs with a conditional report no break around it now; `09_if_both_arms`
is one region, predicate and arms included.

**R11 — gradient through control flow: a conditional differentiates through
select's VJP; a loop under `gradient` is a reported break.** Two constructs,
two different answers, stated here so neither is discovered inside the
emitter. A conditional inside a differentiated region is emitted as a
select whatever its arms hold, and the VJP through it is select's; the
predicate carries no gradient. A named let inside a differentiated
expression is NOT admitted (R8 condition 5): a `stablehlo.while` has no
device VJP, and unrolling it on the host tape at a recorded trip count would
need the host tape to see inside a region, which it does not — so the loop
is reported as `named-let`, `non-admitted-construct`, and its parts outline
on their own inside the gradient. `18_while_inside_gradient` records that
outcome. The while's VJP (a reverse loop over saved iterates, or the host
unroll) is a build item, not a silent gap.

A related decision that the corpus does not grade but the whole-program
parity run exercises: generated code does not call a region marked
`inside_gradient`. Calling its forward pass would hand the host AD tape a
plain tensor with no record of the ops inside, and the gradient through it
would be zero, silently. The region is still formed, reported and measured
forward-and-VJP on the device by `region_formation_test --parity`; joining
that VJP module to the host tape is its own build item.
