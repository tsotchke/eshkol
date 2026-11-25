"""Select a complete @gram function for IR review or for strict lowering."""

import re
from authority import (
    Checker,
    Contract,
    Equal,
    Gate,
    Kind,
    Level,
    Need,
    Observation,
    Permission,
    Unsupported,
    fingerprint,
)

REFERENCE = "d31dae52210cc7cdb2db17144b5f2ed51b76797a97c77aacda6876ab6cfa5c82"
GUARD = b"  %174 = icmp eq i64 %K, %170\n"
PROPOSAL = b"""  %vf.left = icmp eq ptr %tensor_ptr, %new_tensor
  %vf.right = icmp eq ptr %tensor_ptr28, %transpose_input
  %vf.skip = and i1 %vf.left, %vf.right
  br i1 %vf.skip, label %vf.join, label %vf.test

vf.test:
  %vf.eq = icmp eq i64 %K, %170
  br label %vf.join

vf.join:
  %174 = phi i1 [ true, %mm_b_merge ], [ %vf.eq, %vf.test ]
"""


def shape(candidate, ctx):
    ref = ctx["guarded"]
    if fingerprint(ref) != REFERENCE:
        raise Unsupported("unreviewed reference function")
    if candidate == ref:
        return Observation("validated", "reference retains the comparison")
    # Consumer reconstructs the guarded bytes from a complete match. The four
    # SSA identities come from the separately pinned guarded function.
    pattern = (
        rb"  %vf.left = icmp eq ptr (%\w+), (%\w+)\n"
        rb"  %vf.right = icmp eq ptr (%\w+), (%\w+)\n"
        rb"  %vf.skip = and i1 %vf.left, %vf.right\n"
        rb"  br i1 %vf.skip, label %vf.join, label %vf.test\n\nvf.test:\n"
        rb"  %vf.eq = icmp eq i64 %K, (%\d+)\n  br label %vf.join\n\nvf.join:\n"
        rb"  (%\d+) = phi i1 \[ true, %mm_b_merge \], \[ %vf.eq, %vf.test \]\n"
    )
    matches = list(re.finditer(pattern, candidate))
    if len(matches) != 1:
        return Observation("unexpected rewrite", "@gram/mm_b_merge")
    (m,) = matches
    if m.groups() != (
        b"%tensor_ptr",
        b"%new_tensor",
        b"%tensor_ptr28",
        b"%transpose_input",
        b"%170",
        b"%174",
    ):
        return Observation("wrong operand binding", "@gram/mm_b_merge")
    if candidate[: m.start()] + GUARD + candidate[m.end() :] != ref:
        return Observation("other IR changed", "@gram")
    return Observation("validated", "transpose-origin conditional omission")


def effects(candidate, ctx):
    if candidate == ctx["guarded"]:
        return Observation("retained or discharged", "guard retained")
    raise Unsupported("missing checked helper effects on the ordinary Gram path")


def gates(read):
    checks = {
        "site": Checker(Kind.REFINEMENT, Level.ARTIFACT, ("guarded", "gram-checker"), shape),
        "effects": Checker(
            Kind.TRACE, Level.ARTIFACT, ("guarded", "gram-checker", "runtime", "epoch"), effects
        ),
    }
    site = Need("site", Kind.REFINEMENT, Level.ARTIFACT, Equal("validated"))
    effect = Need("effects", Kind.TRACE, Level.ARTIFACT, Equal("retained or discharged"))
    return (
        Gate(
            Contract(
                "transpose(x) times x",
                "export function for conditional IR review",
                Permission.IR_REVIEW,
                (site,),
            ),
            checks,
            read,
        ),
        Gate(
            Contract(
                "transpose(x) times x",
                "select @gram comparison for lowering",
                Permission.LOWERING,
                (site, effect),
            ),
            checks,
            read,
        ),
        Gate(
            Contract(
                "transpose(x) times x",
                "release linked Gram",
                Permission.NATIVE_RELEASE,
                (
                    site,
                    effect,
                    Need("linked entry", Kind.REFINEMENT, Level.REFINEMENT, Equal("bound")),
                ),
            ),
            checks,
            read,
        ),
    )
