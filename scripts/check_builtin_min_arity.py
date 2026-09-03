#!/usr/bin/env python3
"""check_builtin_min_arity.py — the VM builtin table and the documented
language surface must agree on how many arguments a builtin requires.

WHY THIS EXISTS. lib/backend/eshkol_vm.c's BUILTINS[] carries an `arity` that
is the number of operands emit_builtin_preamble() loads into the opcode — the
SHAPE of the call, not the caller's obligation. A builtin whose opcode is
shared with a longer sibling declares the extra slot and lets the unsupplied
local default, so its `arity` over-states its real minimum. `hash-ref` is the
case: opcode 661 is also `hash-table-ref/default`, whose third operand is the
value to return on a miss, and `(hash-ref table key)` legitimately omits it.

The compiler's under-arity refusal (vm_builtin_arity_at_index) reads the real
minimum — `min_arity` when non-zero, otherwise `arity`. If that minimum ever
exceeds the documented arity, the VM REFUSES A DOCUMENTED-LEGAL CALL at compile
time, which is what happened when the refusal first read `arity` directly: every
two-argument `hash-ref` in the test corpus stopped compiling. That direction is
a hard failure here.

The other direction — a minimum BELOW the documented arity — is a permissiveness
gap: the VM accepts a call the documentation says is too short. Those are real
but not memory-unsafe, and the known set is pinned below so a new one cannot
appear silently.

Exit 0 iff no builtin refuses a documented-legal call and no unpinned
permissiveness gap has appeared.
"""

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
VM_C = ROOT / "lib" / "backend" / "eshkol_vm.c"
SURFACE = ROOT / "tests" / "coverage" / "language_surface.json"

# Builtins whose VM minimum is BELOW the documented arity. Each entry is a
# recorded gap, not an exemption: the VM accepts a shorter call than the
# documentation describes. Shrink-only — remove a name when it is fixed, and
# never add one without a build item explaining why the two surfaces differ.
KNOWN_PERMISSIVE = {
    # `tensor` is documented as (tensor shape data) but the VM opcode loads a
    # single operand, so the VM accepts a one-argument call. Recorded in
    # .swarm/runtime/CODEX_HANDOFF.md.
    "tensor",
}

ENTRY = re.compile(r'\{\s*"([^"]+)"\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d+)\s*)?\}')


def builtins_from_source(text):
    m = re.search(r"static const BuiltinDef BUILTINS\[\] = \{(.*?)\n\};", text, re.S)
    if not m:
        raise SystemExit("check_builtin_min_arity: BUILTINS[] table not found in eshkol_vm.c")
    out = []
    for name, _native_id, arity, min_arity in ENTRY.findall(m.group(1)):
        arity = int(arity)
        minimum = int(min_arity) if min_arity else arity
        out.append((name, arity, minimum))
    return out


def main():
    if not SURFACE.exists():
        print(f"check_builtin_min_arity: {SURFACE} missing", file=sys.stderr)
        return 2
    documented = {
        e["name"]: e.get("arity")
        for e in json.loads(SURFACE.read_text())["builtins"]
    }
    entries = builtins_from_source(VM_C.read_text())
    if not entries:
        print("check_builtin_min_arity: BUILTINS[] parsed empty", file=sys.stderr)
        return 2

    refusals, permissive, unknown = [], [], []
    for name, arity, minimum in entries:
        if name not in documented:
            unknown.append(name)
            continue
        doc = documented[name]
        if doc is None:
            continue
        if minimum > doc:
            refusals.append((name, arity, minimum, doc))
        elif minimum < doc and name not in KNOWN_PERMISSIVE:
            permissive.append((name, arity, minimum, doc))

    print(f"checked {len(entries)} VM builtin entries against {SURFACE.name}")
    rc = 0
    for name in unknown:
        print(f"FAIL: {name} is in BUILTINS[] but not in the documented surface")
        rc = 1
    for name, arity, minimum, doc in refusals:
        print(
            f"FAIL: {name} would refuse a documented-legal call — VM minimum "
            f"{minimum} (arity {arity}) exceeds documented arity {doc}. Give it a "
            f"min_arity of {doc} in BUILTINS[]."
        )
        rc = 1
    for name, arity, minimum, doc in permissive:
        print(
            f"FAIL: {name} accepts a shorter call than documented — VM minimum "
            f"{minimum} (arity {arity}) below documented arity {doc}, and it is not "
            f"a pinned gap. Fix the builtin or add it to KNOWN_PERMISSIVE with a "
            f"build item."
        )
        rc = 1
    for name in sorted(KNOWN_PERMISSIVE):
        entry = next((e for e in entries if e[0] == name), None)
        if entry is None:
            print(f"FAIL: pinned permissive builtin {name} is no longer in BUILTINS[] — drop the pin")
            rc = 1
        elif documented.get(name) is not None and entry[2] >= documented[name]:
            print(f"FAIL: {name} no longer diverges — remove it from KNOWN_PERMISSIVE (shrink-only)")
            rc = 1
    if rc == 0:
        print("PASS: every VM builtin minimum agrees with the documented arity")
    return rc


if __name__ == "__main__":
    sys.exit(main())
