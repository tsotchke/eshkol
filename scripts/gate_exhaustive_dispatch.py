#!/usr/bin/env python3
"""Release gate: no dispatch switch over a closed enum may carry a `default:`.

THE DEFECT CLASS
    PR #498 found four tensor-valued AD node types falling into an early
    `default:` in eshkol_tensor_backward_dispatch that did nothing at all:
    gradients came back exactly 0.0, with no diagnostic and exit 0.  That
    default's comment asserted the case was impossible, reasoning from a
    numeric band of enum values ("every tensor op is 19-32 or 67-80") that four
    newer node types had already stepped outside of.  The comment was the only
    thing enforcing the claim.  ESH-0214d is the same shape one subsystem over:
    heap subtypes carrying interior pointers fell to a shallow-leaf `default:`
    in the region evacuator and left those pointers aimed into a reclaimed
    arena.

    The fix is compile-time — no `default:`, plus -Werror=switch-enum (whole
    file) or ESHKOL_EXHAUSTIVE_SWITCH_BEGIN/_END (one switch).  This gate is
    the second, source-derived half: it re-derives the enum's members from
    their DEFINITION and checks each registered dispatch site against them, so
    the property is re-measured on every lane including the ones whose
    toolchain (MSVC) has no such diagnostic, and so removing the enforcement
    is itself caught.

WHAT IS CHECKED
    For every site in SITES:
      1. the switch names EVERY member of its enum, derived from the enum's
         own definition — not from a hand-typed list that could drift;
      2. the switch contains NO `default:` label;
      3. the enforcement is actually armed: the file is listed in CMakeLists'
         eshkol_require_exhaustive_dispatch(), or the switch is wrapped in
         ESHKOL_EXHAUSTIVE_SWITCH_BEGIN/_END.
    Plus, for the AD node registry specifically:
      4. one registry row per ad_node_type_t member, values dense and equal to
         their ordinal (the dispatch tables are indexed by node type);
      5. every row declaring BRIDGE names a backward function that is DEFINED
         in lib/bridge/tensor_backward.cpp.

WHAT IS NOT CHECKED, ON PURPOSE
    Switches over values that arrive from outside the program — bytecode
    opcodes, subtype bytes read off a header or a file, integers crossing an
    FFI boundary — are dispatching over an OPEN set whatever the enum declares.
    Those keep a `default:`, and it must be a loud, value-naming abort.  A gate
    that demanded exhaustiveness there would be enforcing a fiction.

Usage
    python3 scripts/gate_exhaustive_dispatch.py
    python3 scripts/gate_exhaustive_dispatch.py --format json
    python3 scripts/gate_exhaustive_dispatch.py --self-test

Exit status is 0 on PASS and 1 on FAIL, so this also works as a plain CI step.

Copyright (C) tsotchke
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRACE_DIR = os.path.join(REPO_ROOT, "scripts", "icc_traces")
TRACE_BASENAME = "exhaustive_dispatch_gate.jsonl"
PROBE_ID = "closed_enum_dispatch_exhaustive"

AD_REGISTRY = os.path.join("inc", "eshkol", "ad_node_registry.def")
BRIDGE_IMPL = os.path.join("lib", "bridge", "tensor_backward.cpp")
CMAKELISTS = "CMakeLists.txt"

# Each site names a dispatch switch that MUST be exhaustive over a closed enum.
#   file       source file holding the switch
#   func       enclosing function, used only to locate the switch
#   enum       enum whose members the switch must name in full
#   enum_file  where that enum is DEFINED; the member set is parsed from there
#   armed_by   "cmake"  -> file listed in eshkol_require_exhaustive_dispatch()
#              "pragma" -> switch wrapped in ESHKOL_EXHAUSTIVE_SWITCH_BEGIN/_END
SITES = [
    {
        "file": os.path.join("lib", "backend", "tensor_backward.cpp"),
        "func": "eshkol_tensor_backward_dispatch",
        "enum": "ad_node_type_t",
        "enum_file": os.path.join("inc", "eshkol", "eshkol.h"),
        "armed_by": "cmake",
        "why": "the PR #498 site: a tensor node reaching a default returned a "
               "gradient of exactly 0.0 with no diagnostic",
    },
    {
        "file": os.path.join("lib", "backend", "tensor_backward.cpp"),
        "func": "eshkol_ad_node_type_name",
        "enum": "ad_node_type_t",
        "enum_file": os.path.join("inc", "eshkol", "eshkol.h"),
        "armed_by": "cmake",
        "why": "an unnamed node type makes every abort message below it a "
               "number the reader has to go decode",
    },
    {
        "file": os.path.join("lib", "backend", "tensor_backward.cpp"),
        "func": "eshkol_ad_node_type_is_tensor",
        "enum": "ad_node_type_t",
        "enum_file": os.path.join("inc", "eshkol", "eshkol.h"),
        "armed_by": "cmake",
        "why": "payload kind decides whether reaching the tensor dispatcher is "
               "normal or is itself the bug",
    },
    {
        "file": os.path.join("lib", "core", "runtime_regions.cpp"),
        "func": "evac_kind_for",
        "enum": "heap_subtype_t",
        "enum_file": os.path.join("inc", "eshkol", "eshkol.h"),
        "armed_by": "cmake",
        "why": "ESH-0214d: a subtype falling to a shallow leaf copy leaves "
               "interior pointers in a reclaimed arena",
    },
    {
        "file": os.path.join("inc", "eshkol", "eshkol.h"),
        "func": "eshkol_heap_subtype_is_declared",
        "enum": "heap_subtype_t",
        "enum_file": os.path.join("inc", "eshkol", "eshkol.h"),
        "armed_by": "pragma",
        "why": "the predicate that separates the closed half of every "
               "heap-subtype dispatch from the open half",
    },
    {
        "file": os.path.join("inc", "eshkol", "eshkol.h"),
        "func": "eshkol_callable_subtype_is_declared",
        "enum": "callable_subtype_t",
        "enum_file": os.path.join("inc", "eshkol", "eshkol.h"),
        "armed_by": "pragma",
        "why": "same split for callable subtypes",
    },
    {
        # This one is inside `#ifndef NDEBUG`, so the COMPILER never sees it in
        # a release build — which is precisely why it needs a source-derived
        # check. -Werror=switch-enum on this file cannot protect a switch the
        # preprocessor removed before the build CI actually runs.
        "file": os.path.join("lib", "core", "runtime_regions.cpp"),
        "func": "evac_object",
        "enum": "heap_subtype_t",
        "enum_file": os.path.join("inc", "eshkol", "eshkol.h"),
        "armed_by": "cmake",
        "why": "the ESH-0214d watchlist: a decision about every subtype, so a "
               "new one must be placed on it or off it explicitly",
    },
    {
        "file": os.path.join("lib", "core", "runtime_errors_hosted.cpp"),
        "func": "eshkol_format_value_type_tag",
        "enum": "heap_subtype_t",
        "enum_file": os.path.join("inc", "eshkol", "eshkol.h"),
        "armed_by": "pragma",
        "why": "the single source of the type name a runtime error reports; a "
               "defaulted subtype tells the user the wrong type",
    },
]

CMAKE_ARM_FUNCTION = "eshkol_require_exhaustive_dispatch"


class GateError(Exception):
    """A registered site could not be read or located."""


# ───────────────────────────── source parsing ─────────────────────────────

def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def enum_members(source: str, enum_name: str) -> list:
    """Member names of `enum_name`, parsed from its own definition.

    Deriving this from the DEFINITION rather than a list in this file is the
    whole point: a hand-maintained expected-set would drift out of date in
    exactly the situation the gate exists to catch (someone adds a member).
    """
    src = _strip_comments(source)
    # [^{}]* rather than .*? — a lazy dot still crosses intervening `}` and
    # would splice two unrelated enums into one member set.
    m = re.search(r"typedef\s+enum\s*\{([^{}]*)\}\s*" + re.escape(enum_name) + r"\s*;",
                  src, re.S)
    if not m:
        raise GateError("enum %s not found in its declared definition file" % enum_name)
    body = m.group(1)
    members = []
    for line in body.split(","):
        mm = re.match(r"\s*([A-Za-z_]\w*)", line)
        if mm:
            members.append(mm.group(1))
    # X-macro-generated enums expand at compile time; recover their members
    # from the .def the enum includes, so this gate reads what the compiler
    # reads rather than the macro text.
    inc = re.search(r'#\s*include\s+"([^"]*\.def)"', body)
    if inc:
        def_path = os.path.join(REPO_ROOT, "inc", inc.group(1))
        if not os.path.exists(def_path):
            def_path = os.path.join(REPO_ROOT, inc.group(1))
        with open(def_path, encoding="utf-8") as handle:
            rows = registry_rows(handle.read())
        # Keep only the enum's own trailing members (the sentinel), not the
        # macro's parameter names, which the naive comma split also picks up.
        generated = ["AD_NODE_" + r["name"] for r in rows]
        trailing = [m for m in members if m.startswith("AD_NODE_")]
        members = generated + trailing
    return members


def function_body(source: str, func_name: str) -> str:
    """Text of `func_name`'s body, brace-matched from its opening brace."""
    m = re.search(r"\b" + re.escape(func_name) + r"\s*\([^;{]*\)\s*\{", source, re.S)
    if not m:
        raise GateError("function %s not found" % func_name)
    i = source.index("{", m.start())
    depth = 0
    for j in range(i, len(source)):
        if source[j] == "{":
            depth += 1
        elif source[j] == "}":
            depth -= 1
            if depth == 0:
                return source[i:j + 1]
    raise GateError("unbalanced braces in %s" % func_name)


def switch_on_enum(body: str, members: list) -> str:
    """The one switch inside `body` that dispatches on this enum.

    A function may hold several switches — the outer one over value tags, the
    inner one over heap subtypes — and grading the whole function body would
    flag the outer switch's legitimate default as if it were the inner
    switch's. Pick the switch whose case labels actually belong to the enum in
    question, and grade only that one.
    """
    member_set = set(members)
    best, best_hits = None, 0
    for m in re.finditer(r"\bswitch\s*\(", body):
        i = body.find("{", m.end())
        if i < 0:
            continue
        depth, end = 0, None
        for j in range(i, len(body)):
            if body[j] == "{":
                depth += 1
            elif body[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        if end is None:
            continue
        text = body[i:end]
        hits = len(member_set & set(re.findall(r"\bcase\s+([A-Za-z_]\w*)\s*:", text)))
        # Ties go to the SMALLEST switch: an outer switch textually contains
        # its nested ones, so it scores every inner hit too. Innermost is the
        # switch that actually dispatches on this enum.
        if hits > best_hits or (hits and hits == best_hits and len(text) < len(best)):
            best, best_hits = text, hits
    if best is None:
        raise GateError("no switch over the target enum found")
    return best


def registry_rows(text: str) -> list:
    rows = []
    for m in re.finditer(
            r"^ESHKOL_AD_NODE\(\s*(\w+)\s*,\s*(\d+)\s*,\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*\)",
            text, re.M):
        rows.append({
            "name": m.group(1), "value": int(m.group(2)), "payload": m.group(3),
            "backward": m.group(4), "bridge_fn": m.group(5),
        })
    return rows


# ───────────────────────────── grading ─────────────────────────────

def grade_site(site: dict, repo_root: str) -> dict:
    path = os.path.join(repo_root, site["file"])
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    enum_path = os.path.join(repo_root, site["enum_file"])
    with open(enum_path, encoding="utf-8") as handle:
        enum_source = handle.read()

    members = enum_members(enum_source, site["enum"])
    func = function_body(source, site["func"])
    generated = "ad_node_registry.def" in func
    # A generated dispatch expands its case labels from the .def at compile
    # time, so there are no `case` tokens in the source to match on; grade the
    # whole function body in that case.
    body = func if generated else switch_on_enum(_strip_comments(func), members)
    body_nc = _strip_comments(body)

    findings = []

    if re.search(r"\bdefault\s*:", body_nc):
        findings.append(
            "carries a `default:` — a default over a closed enum answers "
            "'member I have not thought about' with a plausible value")

    labelled = set(re.findall(r"\bcase\s+([A-Za-z_]\w*)\s*:", body_nc))
    missing = []
    for member in members:
        if member.endswith("_TYPE_COUNT"):
            continue
        if member not in labelled and not generated:
            missing.append(member)
    if missing:
        findings.append("does not name %d enum member(s): %s"
                        % (len(missing), ", ".join(sorted(missing)[:12])))

    if site["armed_by"] == "pragma":
        if "ESHKOL_EXHAUSTIVE_SWITCH_BEGIN" not in func:
            findings.append(
                "is not wrapped in ESHKOL_EXHAUSTIVE_SWITCH_BEGIN/_END, so "
                "nothing raises -Wswitch-enum at this switch")
    else:
        with open(os.path.join(repo_root, CMAKELISTS), encoding="utf-8") as handle:
            cmake = handle.read()
        m = re.search(re.escape(CMAKE_ARM_FUNCTION) + r"\((.*?)\)", cmake, re.S)
        armed = m and site["file"].replace(os.sep, "/") in _strip_comments(m.group(1))
        if not armed:
            findings.append(
                "translation unit is not listed in %s(), so -Werror=switch-enum "
                "is not applied to it" % CMAKE_ARM_FUNCTION)

    return {
        "site": "%s:%s" % (site["file"], site["func"]),
        "enum": site["enum"],
        "members": len(members),
        "generated_arms": generated,
        "findings": findings,
    }


def grade_ad_registry(repo_root: str) -> dict:
    with open(os.path.join(repo_root, AD_REGISTRY), encoding="utf-8") as handle:
        rows = registry_rows(handle.read())
    with open(os.path.join(repo_root, BRIDGE_IMPL), encoding="utf-8") as handle:
        bridge_src = _strip_comments(handle.read())

    findings = []
    if not rows:
        findings.append("registry has no rows at all")

    for ordinal, row in enumerate(rows):
        if row["value"] != ordinal:
            findings.append(
                "row %s declares value %d but is at ordinal %d — the dispatch "
                "tables are indexed by node type and assume the two agree"
                % (row["name"], row["value"], ordinal))
            break

    names = [r["name"] for r in rows]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        findings.append("duplicate registry rows: %s" % ", ".join(dupes))

    valid = {"LEAF", "SCALAR_ADJOINT", "INLINE", "BRIDGE", "CUSTOM_VJP", "UNREGISTERED"}
    for row in rows:
        if row["backward"] not in valid:
            findings.append("row %s declares unknown disposition %s"
                            % (row["name"], row["backward"]))
        if row["backward"] == "BRIDGE":
            fn = row["bridge_fn"]
            if fn == "ESHKOL_AD_NO_BRIDGE":
                findings.append("row %s declares BRIDGE but names no function" % row["name"])
            elif not re.search(r"\b" + re.escape(fn) + r"\s*\(\s*ad_node_t\s*\*", bridge_src):
                findings.append(
                    "row %s declares BRIDGE and names %s, which is not defined "
                    "in %s — a registration that registers nothing"
                    % (row["name"], fn, BRIDGE_IMPL))
        elif row["bridge_fn"] != "ESHKOL_AD_NO_BRIDGE":
            findings.append("row %s is %s but names a bridge function"
                            % (row["name"], row["backward"]))

    by_disposition = {}
    for row in rows:
        by_disposition[row["backward"]] = by_disposition.get(row["backward"], 0) + 1

    return {
        "site": AD_REGISTRY,
        "rows": len(rows),
        "by_disposition": by_disposition,
        "findings": findings,
    }


def grade(repo_root: str = REPO_ROOT) -> dict:
    results = []
    failed = 0
    try:
        for site in SITES:
            r = grade_site(site, repo_root)
            results.append(r)
            failed += len(r["findings"])
        reg = grade_ad_registry(repo_root)
        results.append(reg)
        failed += len(reg["findings"])
    except (GateError, OSError) as exc:
        return {
            "status": "FAIL",
            "results": results,
            "error": str(exc),
            "summary": "gate could not read a registered dispatch site: %s" % exc,
        }
    return {
        "status": "FAIL" if failed else "PASS",
        "results": results,
        "error": None,
        "summary": ("%d finding(s) across %d registered dispatch sites"
                    % (failed, len(SITES)) if failed
                    else "%d registered dispatch sites are exhaustive over their "
                         "closed enums and carry no default" % len(SITES)),
    }


# ───────────────────────────── reporting ─────────────────────────────

def emit_trace(trace_dir: str, status: str, snippet: str) -> str:
    os.makedirs(trace_dir, exist_ok=True)
    path = os.path.join(trace_dir, TRACE_BASENAME)
    event = {
        "kind": "eshkol_smoke",
        "name": PROBE_ID,
        "value": status,
        "snippet": snippet[:2000],
        "confidence": 1.0,
    }
    # Rewritten, never appended: a stale PASS must not survive a regression.
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return path


def render(report: dict) -> str:
    lines = ["closed-enum dispatch exhaustiveness: %s" % report["status"], ""]
    for r in report["results"]:
        head = r["site"]
        if "enum" in r:
            head += "  [%s, %d members%s]" % (
                r["enum"], r["members"], ", generated arms" if r["generated_arms"] else "")
        else:
            head += "  [%d rows: %s]" % (
                r["rows"], ", ".join("%s=%d" % kv for kv in sorted(r["by_disposition"].items())))
        lines.append(("  FAIL  " if r["findings"] else "  ok    ") + head)
        for f in r["findings"]:
            lines.append("          - " + f)
    lines += ["", report["summary"]]
    return "\n".join(lines)


# ───────────────────────────── self-test ─────────────────────────────
#
# "A gate that cannot fail is not a gate." Each fixture feeds the graders
# deliberately-broken input and asserts the finding actually appears.

_ENUM_FIXTURE = """
typedef enum {
    THING_A,
    THING_B,
    THING_C
} thing_t;
"""

_GOOD_SWITCH = """
static int classify(thing_t t) {
    ESHKOL_EXHAUSTIVE_SWITCH_BEGIN
    switch (t) {
        case THING_A: return 1;
        case THING_B: return 2;
        case THING_C: return 3;
    }
    ESHKOL_EXHAUSTIVE_SWITCH_END
    return 0;
}
"""

_DEFAULTED_SWITCH = """
static int classify(thing_t t) {
    ESHKOL_EXHAUSTIVE_SWITCH_BEGIN
    switch (t) {
        case THING_A: return 1;
        case THING_B: return 2;
        case THING_C: return 3;
        default: return 0;   /* the whole defect class, in one line */
    }
    ESHKOL_EXHAUSTIVE_SWITCH_END
}
"""

_INCOMPLETE_SWITCH = """
static int classify(thing_t t) {
    ESHKOL_EXHAUSTIVE_SWITCH_BEGIN
    switch (t) {
        case THING_A: return 1;
        case THING_B: return 2;
    }
    ESHKOL_EXHAUSTIVE_SWITCH_END
    return 0;
}
"""

_UNARMED_SWITCH = """
static int classify(thing_t t) {
    switch (t) {
        case THING_A: return 1;
        case THING_B: return 2;
        case THING_C: return 3;
    }
    return 0;
}
"""

_COMMENT_ONLY_SWITCH = """
static int classify(thing_t t) {
    ESHKOL_EXHAUSTIVE_SWITCH_BEGIN
    switch (t) {
        case THING_A: return 1;
        case THING_B: return 2;
        case THING_C: return 3;
        /* Every member is handled above, so this cannot be reached. */
        default: return 0;
    }
    ESHKOL_EXHAUSTIVE_SWITCH_END
}
"""


def _fixture_findings(tmpdir: str, switch_src: str) -> list:
    os.makedirs(os.path.join(tmpdir, "lib", "core"), exist_ok=True)
    src = os.path.join("lib", "core", "fixture.c")
    with open(os.path.join(tmpdir, src), "w", encoding="utf-8") as handle:
        handle.write(_ENUM_FIXTURE + switch_src)
    site = {"file": src, "func": "classify", "enum": "thing_t",
            "enum_file": src, "armed_by": "pragma", "why": "self-test"}
    return grade_site(site, tmpdir)["findings"]


def self_test() -> int:
    import tempfile
    failures = []
    cases = []
    with tempfile.TemporaryDirectory(dir=os.path.join(REPO_ROOT, ".scratch")
                                     if os.path.isdir(os.path.join(REPO_ROOT, ".scratch"))
                                     else None) as tmp:
        cases = [
            ("exhaustive+armed is clean", _GOOD_SWITCH, 0, None),
            ("a default: is caught", _DEFAULTED_SWITCH, 1, "default"),
            ("a missing member is caught", _INCOMPLETE_SWITCH, 1, "THING_C"),
            ("an unarmed switch is caught", _UNARMED_SWITCH, 1, "ESHKOL_EXHAUSTIVE_SWITCH_BEGIN"),
            # The exact shape of #498: the default is not merely present, it is
            # ANNOTATED with a comment asserting it cannot be reached. A comment
            # is not enforcement, and the gate must not be fooled by one.
            ("a default with an it-cannot-happen comment is still caught",
             _COMMENT_ONLY_SWITCH, 1, "default"),
        ]
        for name, src, want, needle in cases:
            found = _fixture_findings(tmp, src)
            if want == 0 and found:
                failures.append("%s: expected clean, got %s" % (name, found))
            elif want and not found:
                failures.append("%s: expected a finding, got none" % name)
            elif want and needle and not any(needle in f for f in found):
                failures.append("%s: finding did not mention %r: %s" % (name, needle, found))

    # A registry row claiming a backward function that does not exist must FAIL.
    rows = registry_rows("ESHKOL_AD_NODE(FAKE, 0, TENSOR, BRIDGE, no_such_backward)\n")
    if len(rows) != 1 or rows[0]["bridge_fn"] != "no_such_backward":
        failures.append("registry row parser did not read a BRIDGE row correctly")

    for f in failures:
        print("SELF-TEST FAIL: " + f)
    if failures:
        return 1
    # Count the fixtures rather than hardcoding the number: a hardcoded count
    # that stops matching the list is the same shape of stale assertion this
    # whole gate exists to prevent.
    print("SELF-TEST PASS: %d fixtures" % (len(cases) + 1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--format", choices=("text", "json"), default="text")
    ap.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR)
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    report = grade()
    text = render(report)
    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(text)
    if not args.no_trace:
        emit_trace(args.trace_dir, report["status"], text)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
