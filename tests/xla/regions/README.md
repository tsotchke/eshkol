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
