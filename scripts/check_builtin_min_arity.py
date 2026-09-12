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

A NEGATIVE min_arity is a third thing: the builtin is variadic in the native
lowering, so there is no minimum to enforce and the refusal must not fire.
That claim is not taken on trust. Each such name names the llvm_codegen.cpp
handler that implements it, and this script checks that the handler really does
answer a zero-argument call (`num_vars == 0`) before folding over `num_vars` —
which is what makes `(gcd)` and `(gcd 3)` legal on native. A pin nobody
verifies is how the surface and the table drift apart in the first place.

The LLVM backend used to carry its own transcription of the same numbers — a
`fixed_arity` map in lib/backend/llvm_codegen.cpp listing 34 builtins and the
arity each one takes. Two copies of one fact, kept in step by nothing but
vigilance, is how the surfaces drift apart; the map now lists only WHICH names
that dispatch checks and reads every number from BUILTINS[] at run time. This
script closes the loop: each name the backend scopes must still be a row in
BUILTINS[], or the check silently stops firing for it.

THE NUMBERS ARE DERIVED, AND THIS RE-DERIVES THEM. `min_arity` used to be
typed in by hand wherever somebody noticed, which is why 739 of 742 rows were
silently claiming that their opcode's operand count was the caller's
obligation and the VM refused `(substring s 1)`, `(read-line)` and
`(make-string 3)` — all legal, all accepted by native.
scripts/gen_builtin_min_arity.py now derives every row's minimum from the
fixed-arity macro the native dispatch expands, the arity guard the lowering
enforces, and the declarative documented signature. This gate imports THAT
module — not a transcription of it — and fails when the table disagrees with
what those sources say today, so editing a guard or a signature without
regenerating the table is a build failure rather than a new divergence.

Exit 0 iff the table matches the derivation, the manifest matches the table,
no builtin refuses a documented-legal call, no unpinned permissiveness gap has
appeared, every variadic claim is borne out by its native handler, and every
builtin the LLVM dispatch scopes is backed by the shared table.
"""

import json
import os
import pathlib
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_builtin_min_arity  # noqa: E402  — the same derivation, not a copy

ROOT = pathlib.Path(__file__).resolve().parents[1]
VM_C = ROOT / "lib" / "backend" / "eshkol_vm.c"
SURFACE = ROOT / "tests" / "coverage" / "language_surface.json"
LLVM_CPP = ROOT / "lib" / "backend" / "llvm_codegen.cpp"

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

# Builtins the table declares variadic (min_arity < 0), mapped to the
# llvm_codegen.cpp handler that makes that true. Verified, not pinned: the
# handler must contain a `num_vars == 0` case, which is the shape of a lowering
# that folds over however many operands it is given.
DECLARED_VARIADIC = {
    "gcd": "codegenGCD",
    "lcm": "codegenLCM",
}

# The min_arity field is SIGNED: `-?` matters. While this pattern read `(\d+)`
# there, a row carrying -1 matched nothing at all and the builtin dropped out of
# this gate's view entirely — the same silent-drop that the three-field pattern
# in scripts/gen_language_surface.py caused for `hash-ref`.
ENTRY = re.compile(r'\{\s*"([^"]+)"\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*(?:,\s*(-?\d+)\s*)?(?:,\s*-?\d+\s*)?\}')


def builtins_from_source(text):
    m = re.search(r"static const BuiltinDef BUILTINS\[\] = \{(.*?)\n\};", text, re.S)
    if not m:
        raise SystemExit("check_builtin_min_arity: BUILTINS[] table not found in eshkol_vm.c")
    out = []
    for name, _native_id, arity, min_arity in ENTRY.findall(m.group(1)):
        arity = int(arity)
        column = int(min_arity) if min_arity else 0
        # The min_arity encoding, as BuiltinDef states it: "" or 0 is absent
        # (the minimum IS arity), -1 declares the native lowering variadic, -2
        # declares a documented minimum of genuinely zero, and anything else is
        # the minimum itself. -2 must NOT read as "variadic": `(read-line)` is
        # a zero-argument call to a one-operand opcode, not a call to a
        # lowering that folds over however many arguments it is handed.
        if column == -2:
            minimum = 0
        elif column < 0:
            minimum = column
        else:
            minimum = column or arity
        out.append((name, arity, minimum))
    return out


def variadic_handler_answers_zero_args(text, handler):
    """True iff `handler` in llvm_codegen.cpp has a zero-argument case.

    A variadic lowering opens with the identity for an empty call and then
    folds over `num_vars`; a fixed-arity one indexes variables[0] straight
    away. Looking for the former is what turns DECLARED_VARIADIC from an
    assertion into a check.
    """
    m = re.search(r"\b%s\s*\(const eshkol_operations_t\* op\)\s*\{" % re.escape(handler), text)
    if not m:
        return None
    depth, i = 0, m.end() - 1
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    body = text[m.end():i]
    return re.search(r"num_vars\s*==\s*0", body) is not None


SCOPE_SET = re.compile(
    r"static const std::unordered_set<std::string>\s+(minimum|fixed)_arity_builtins\s*=\s*\{(.*?)\};",
    re.S)


def llvm_scoped_builtins(text):
    """Return {"minimum": {names}, "fixed": {names}} from llvm_codegen.cpp.

    Reads the two scope sets the builtin dispatch declares. They carry NAMES
    ONLY — the arity each one requires is read from BUILTINS[] at run time —
    so this is the whole of what the backend still says about arity, and it is
    what has to stay backed by the shared table.
    """
    found = {"minimum": set(), "fixed": set()}
    for kind, body in SCOPE_SET.findall(text):
        found[kind] = set(re.findall(r'"((?:[^"\\]|\\.)*)"', body))
    return found


def main():
    fixture = ('static const BuiltinDef BUILTINS[] = {\n'
               '{"error",237,1,0,1}, {"hash-ref",661,3,2}, '
               '{"gcd",45,2,-1}, {"read-line",585,1,-2}, {"sin",20,1}\n};')
    if builtins_from_source(fixture) != [
            ('error', 1, 1), ('hash-ref', 3, 2), ('gcd', 2, -1),
            ('read-line', 1, 0), ('sin', 1, 1)]:
        print('FAIL: builtin metadata parser dropped or misread a field layout')
        return 1
    if not SURFACE.exists():
        print(f"check_builtin_min_arity: {SURFACE} missing", file=sys.stderr)
        return 2
    surface = json.loads(SURFACE.read_text())["builtins"]
    documented = {e["name"]: e.get("arity") for e in surface}
    surface_minimum = {e["name"]: e.get("min_arity") for e in surface
                       if "min_arity" in e}
    entries = builtins_from_source(VM_C.read_text())
    if not entries:
        print("check_builtin_min_arity: BUILTINS[] parsed empty", file=sys.stderr)
        return 2

    derived = {r["name"]: r for r in gen_builtin_min_arity.derive(str(ROOT))}
    refusals, permissive, unknown = [], [], []
    declared_variadic = set()
    for name, arity, minimum in entries:
        if name not in documented:
            unknown.append(name)
            continue
        # A variadic row makes no minimum claim, so there is nothing to compare
        # against the documented arity — that number is the opcode's operand
        # count for this row, not an obligation on the caller.
        if minimum < 0:
            declared_variadic.add(name)
            continue
        if derived.get(name, {}).get("min_arity_column") == -2:
            # A documented minimum of zero: there is no under-arity call to
            # refuse, and `arity` below is the opcode's operand count rather
            # than an obligation on the caller.
            continue
        doc = documented[name]
        if doc is None:
            continue
        if minimum > doc:
            refusals.append((name, arity, minimum, doc))
        elif (minimum < doc and name not in KNOWN_PERMISSIVE
                and not derived.get(name, {}).get("min_arity_column")):
            # A minimum below the opcode's operand count is the NORMAL shape of
            # a builtin with a documented optional parameter, and the
            # derivation above is what makes it legitimate. One the derivation
            # does NOT claim is a row someone shortened by hand.
            permissive.append((name, arity, minimum, doc))

    print(f"checked {len(entries)} VM builtin entries against {SURFACE.name}")
    rc = 0

    # (1) THE TABLE MUST STILL BE WHAT THE SOURCES SAY. Re-derive, do not
    # re-read: a check that trusted the column it is checking would pass
    # forever after the first wrong value was written.
    drift = []
    for name, row in sorted(derived.items()):
        want = row.get("min_arity_column", row["min_arity"])
        if row["source"] == "variadic":
            continue          # -1 is a claim this module verifies separately
        if want != row["min_arity"]:
            drift.append((name, row["min_arity"], want, row["source"],
                          row["evidence"]))
    for name, have, want, source, evidence in drift:
        print(
            f"FAIL: {name} has min_arity {have} in BUILTINS[] but {source} says "
            f"{want} — re-run scripts/gen_builtin_min_arity.py --apply. "
            f"Evidence: {evidence[:120]}"
        )
        rc = 1
    if not drift:
        print(f"  derivation: {len(derived)} rows agree with their native "
              f"guard / fixed-arity macro / documented signature")

    # (2) THE MANIFEST MUST CARRY THE SAME MINIMUM. tests/coverage/
    # language_surface.json is what every exposure harness reads, so a
    # minimum that lives only in the C table is a minimum those harnesses
    # cannot test against.
    for name, arity, minimum in entries:
        if name not in surface_minimum:
            continue
        want = None if minimum < 0 else minimum
        if surface_minimum[name] != want:
            print(
                f"FAIL: {name} has caller minimum {want} in BUILTINS[] but "
                f"{surface_minimum[name]} in {SURFACE.name} — re-run "
                f"scripts/gen_language_surface.py."
            )
            rc = 1
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
    # Every variadic claim must be declared here AND borne out by its handler.
    llvm_text = LLVM_CPP.read_text() if LLVM_CPP.exists() else ""
    for name in sorted(declared_variadic - set(DECLARED_VARIADIC)):
        print(
            f"FAIL: {name} declares min_arity < 0 in BUILTINS[] but is not in "
            f"DECLARED_VARIADIC — add it with the llvm_codegen.cpp handler that "
            f"makes it variadic, so the claim can be checked."
        )
        rc = 1
    for name, handler in sorted(DECLARED_VARIADIC.items()):
        if name not in declared_variadic:
            print(
                f"FAIL: {name} is listed in DECLARED_VARIADIC but its BUILTINS[] "
                f"row no longer declares a negative min_arity — drop the entry or "
                f"restore the row."
            )
            rc = 1
            continue
        answers = variadic_handler_answers_zero_args(llvm_text, handler)
        if answers is None:
            print(f"FAIL: {name} names handler {handler}, which is not in {LLVM_CPP.name}")
            rc = 1
        elif not answers:
            print(
                f"FAIL: {name} is declared variadic but {handler} has no "
                f"`num_vars == 0` case — it does not accept a zero-argument call, "
                f"so the VM must not stop refusing short calls to it."
            )
            rc = 1
    # The LLVM dispatch names the builtins it checks; the ARITY comes from
    # BUILTINS[] via eshkol_builtin_min_arity(). A scoped name with no row is a
    # check that has quietly stopped firing.
    table_names = {name for name, _arity, _minimum in entries}
    for kind, scoped in sorted(llvm_scoped_builtins(llvm_text).items()):
        if not scoped:
            print(
                f"FAIL: could not find the {kind}_arity_builtins set in "
                f"{LLVM_CPP.name} — the LLVM dispatch's arity scope must stay "
                f"readable from here, or nothing checks that it is backed by "
                f"BUILTINS[]."
            )
            rc = 1
            continue
        for name in sorted(scoped - table_names):
            print(
                f"FAIL: {name} is in {LLVM_CPP.name}'s {kind}_arity_builtins but "
                f"has no BUILTINS[] row — eshkol_builtin_min_arity() answers -1 "
                f"for it, so the native arity check never fires. Add the row, "
                f"or drop the name and guard the builtin at its own lowering "
                f"the way exact-integer? does."
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
        print(
            f"PASS: every VM builtin minimum agrees with the documented arity "
            f"({len(declared_variadic)} declared variadic, verified against their handlers)"
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())
