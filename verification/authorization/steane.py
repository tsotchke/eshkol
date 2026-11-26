"""Select and execute a gate-ID-bound Steane schedule under an independent policy."""

from fractions import Fraction as F
from functools import lru_cache
import json
from authority import (
    AtMost,
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
    encode,
)
from fixtures import faults, pauli, schedule

REFERENCE = "dbc3cbcb6e0b6a3c44f7ee4960bfde5f0db09b0b80ca0c65722d992308100119"


@lru_cache(maxsize=8)
def source(data):
    if fingerprint(data) != REFERENCE:
        raise Unsupported("unknown source or decoder")
    n, checks, gates, cert = json.loads(data)
    if (
        (n, checks) != pauli.CODES[0]
        or not pauli.decoder_valid(cert, n, checks)
        or not pauli.prepare_valid(n, checks + ["X" * n], gates)
    ):
        raise Unsupported("invalid preparation or decoder")
    return checks, gates, cert


def ordered(candidate, gates):
    rows = json.loads(candidate)
    if type(rows) is not list or any(
        type(row) is not list or len(row) != 2 or type(row[0]) is not int for row in rows
    ):
        raise ValueError("expected indexed gates")
    ids = [i for i, _ in rows]
    if sorted(ids) != list(range(len(gates))) or any(
        encode(g) != encode(gates[i]) for i, g in rows
    ):
        raise ValueError("gate IDs or wire roles differ")
    return ids


def column(tape, i):
    state = [int(j == i) for j in range(128)]
    for g in tape:
        state = faults.dense_gate(state, g)
    return state


def ideal(candidate, ctx):
    _, gates, _ = source(ctx["steane"])
    try:
        order = ordered(candidate, gates)
        schedule.schedule_cost(7, gates, order)
    except (ValueError, UnicodeError) as exc:
        return Observation("invalid schedule", str(exc))
    tape = [gates[i] for i in order]
    for i in range(128):
        if column(tape, i) != column(gates, i):
            return Observation("different operator", f"basis input {i}")
    return Observation("equal", "all 128 signed columns")


def law(bias):
    return encode({"fault_mass": "1/100", "event": [12, "YY"], "conditional_bias": str(bias)})


def fault_pairs(candidate, ctx):
    checks, gates, cert = source(ctx["steane"])
    order = ordered(candidate, gates)
    decode, complete = faults.decoder(7, checks, cert)

    def outcomes(ids):
        return faults.outcomes(gates, ids, decode, complete, checks)

    old, new = outcomes(range(len(gates))), outcomes(order)
    if len(old) != 225 or old.keys() != new.keys():
        raise Unsupported("incomplete gate-local fault inventory")
    return old, new


def pointwise(candidate, ctx):
    old, new = fault_pairs(candidate, ctx)
    for gate_id, fault in sorted(old):
        event = gate_id, fault
        before, after = old[event], new[event]
        if (before["syndrome"], before["logical"]) != (after["syndrome"], after["logical"]):
            return Observation(
                "changed",
                f"gate {gate_id} {fault}: {before['syndrome']}/{before['logical']} -> {after['syndrome']}/{after['logical']}",
            )
    return Observation("preserved", "225 paired syndromes and logical Paulis")


def risk(candidate, ctx):
    if ctx["noise"] not in (law(F(0)), law(F(1, 10))):
        raise Unsupported("unsupported single-fault distribution")
    bias = F(json.loads(ctx["noise"])["conditional_bias"])

    def flips(rows):
        return {k: v["logical"] in ("Z", "Y") for k, v in rows.items()}

    old, new = map(flips, fault_pairs(candidate, ctx))
    delta = sum(
        F(1, 100) * ((1 - bias) / 225 + bias * (k == (12, "YY"))) * (new[k] - old[k]) for k in old
    )
    return Observation(
        delta, f"225 paired events; gate 12 YY logical X flip {old[12, 'YY']} -> {new[12, 'YY']}"
    )


def gates(read, budget=None, *, preserve_faults=False):
    deps = ("steane", "steane-checker")
    registry = {
        "fault paths": Checker(Kind.TRACE, Level.ARTIFACT, deps, pointwise),
        "operator": Checker(Kind.EXACT, Level.ARTIFACT, deps, ideal),
        "fault increase": Checker(Kind.RISK, Level.ARTIFACT, (*deps, "noise"), risk),
    }
    needs = (Need("operator", Kind.EXACT, Level.ARTIFACT, Equal("equal")),)
    if preserve_faults:
        needs += (Need("fault paths", Kind.TRACE, Level.ARTIFACT, Equal("preserved")),)
    if budget is not None:
        needs += (Need("fault increase", Kind.RISK, Level.ARTIFACT, AtMost(budget)),)
    return Gate(
        Contract("Steane logical plus", "dispatch indexed tape", Permission.EXECUTION, needs),
        registry,
        read,
    )


def candidate(data, moved):
    _, gates, _ = source(data)
    ids = list(range(len(gates)))
    if moved:
        ids.insert(20, ids.pop(12))
    return encode([[i, gates[i]] for i in ids])


def execute(payload, data):
    _, gates, _ = source(data)
    order = ordered(payload, gates)
    state = column([gates[i] for i in order], 0)
    return {
        "depth": schedule.schedule_cost(7, gates, order)["depth"],
        "state": fingerprint(encode(state)),
    }
