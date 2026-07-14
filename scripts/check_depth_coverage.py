#!/usr/bin/env python3
"""check_depth_coverage.py — depth-coverage completeness gate (Pillar P6 auditor).

The meta-check that PROVES Eshkol's depth-parametric testing covers the depth of
the ENTIRE language, not just AD. It re-derives the language construct surface
from the compiler's own AST op enum (inc/eshkol/eshkol.h) and asserts, against
scripts/depth_coverage_registry.json, that:

  1. Every op in the enum is CLASSIFIED in the registry as 'composable'
     (can nest within itself / other forms -> a depth-dependent miscompile like
     reverse-over-nested-forward is possible) or 'leaf' (opaque-handle primitive,
     declaration, or predicate with no user-syntactic nesting depth).
     A new op that nobody classified => FAIL. This forces every future language
     addition to declare whether it needs a depth sweep.

  2. The registry has no STALE op (an op removed from the enum but still listed).

  3. Every COMPOSABLE construct (ops + the supplemental non-op composables such
     as the numeric tower and collection higher-order fns) is REGISTERED with a
     depth sweep: either it names an owning pillar (P6a..P6f) OR it carries a
     gap_task acknowledging it is un-swept (an ESH-#### tracking task). A
     composable with NEITHER => FAIL. This is the ratchet: you cannot add a
     nestable construct without either sweeping its depth or filing the gap.

On success it prints a coverage summary (composables with a real pillar sweep vs
total composables) and the explicit GAP list, and writes an ICC trace event
(kind:"depth_coverage") to scripts/icc_traces/depth_coverage.jsonl so the
`depth-coverage` completion oracle can gate on it.

Usage:
  python3 scripts/check_depth_coverage.py [--repo-root DIR] [--no-trace] [--quiet]

Exit code 0 = gate green (every construct classified, no un-registered gap).
Exit code 1 = gate red (unclassified op, stale entry, or un-registered composable).
"""

import argparse
import json
import os
import re
import sys

VALID_PILLARS = {"P6a", "P6b", "P6c", "P6d", "P6e", "P6f"}
GAP_TASK_RE = re.compile(r"^ESH-\d{4}$")


def repo_root_default():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_ops_from_header(header_path):
    """Re-derive the construct surface: every ESHKOL_*_OP enumerator."""
    if not os.path.isfile(header_path):
        sys.exit(f"FATAL: op enum header not found: {header_path}")
    ops = []
    seen = set()
    with open(header_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    for m in re.finditer(r"\b(ESHKOL_[A-Z0-9_]+_OP)\b\s*,", text):
        name = m.group(1)
        if name not in seen:
            seen.add(name)
            ops.append(name)
    if not ops:
        sys.exit(f"FATAL: no ESHKOL_*_OP enumerators parsed from {header_path}")
    return ops


def load_registry(path):
    if not os.path.isfile(path):
        sys.exit(f"FATAL: registry not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_entry(name, entry, errors):
    """Validate a single construct entry. Returns (is_composable, is_swept)."""
    cls = entry.get("class")
    if cls not in ("composable", "leaf"):
        errors.append(f"{name}: class must be 'composable' or 'leaf' (got {cls!r})")
        return (False, False)
    if cls == "leaf":
        return (False, False)
    pillar = entry.get("pillar")
    gap = entry.get("gap_task")
    if pillar is not None:
        if pillar not in VALID_PILLARS:
            errors.append(f"{name}: pillar {pillar!r} not one of {sorted(VALID_PILLARS)}")
            return (True, False)
        return (True, True)
    # No pillar -> must be an acknowledged gap with a tracking task id.
    if gap is None:
        errors.append(
            f"{name}: composable with NO depth sweep — assign a pillar (P6a..P6f) "
            f"or file a gap_task (ESH-####) and record it here")
        return (True, False)
    if not GAP_TASK_RE.match(str(gap)):
        errors.append(f"{name}: gap_task {gap!r} must look like 'ESH-0123'")
    return (True, False)


def write_depth_coverage_trace(
    trace_dir, *, passed, swept, composables, pct, gap_count, error_count
):
    """Write the bounded depth-coverage runtime contract and return its path."""
    os.makedirs(trace_dir, exist_ok=True)
    trace_path = os.path.join(trace_dir, "depth_coverage.jsonl")
    rec = {
        "kind": "depth_coverage",
        "name": "depth_coverage_gate",
        "value": "PASS" if passed else "FAIL",
        "snippet": (
            f"{swept}/{composables} composables swept ({pct:.1f}%), "
            f"{gap_count} tracked gaps, {error_count} errors"
        ),
        "confidence": 0.95,
    }
    with open(trace_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
        f.write(json.dumps({
            "kind": "depth_coverage",
            "name": "depth_coverage_pct",
            "value": f"{pct:.1f}",
            "snippet": f"{swept}/{composables} composables have a pillar sweep",
            "confidence": 0.95,
        }) + "\n")
    return trace_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=repo_root_default())
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(args.repo_root)
    header = os.path.join(root, "inc", "eshkol", "eshkol.h")
    reg_path = os.path.join(root, "scripts", "depth_coverage_registry.json")

    ops = parse_ops_from_header(header)
    reg = load_registry(reg_path)
    reg_ops = reg.get("ops", {})
    supplemental = reg.get("supplemental", [])

    errors = []

    # (1) every enum op classified; (2) no stale registry op.
    enum_set = set(ops)
    reg_set = set(reg_ops.keys())
    unclassified = [o for o in ops if o not in reg_set]
    stale = sorted(reg_set - enum_set)
    for o in unclassified:
        errors.append(f"{o}: present in op enum but UNCLASSIFIED in registry "
                      f"(classify as composable+sweep/gap or leaf)")
    for o in stale:
        errors.append(f"{o}: in registry but NOT in op enum (stale — remove it)")

    # (3) validate every classified entry + supplemental.
    composables = 0
    swept = 0
    gaps = []  # (name, gap_task)
    for name in ops:
        entry = reg_ops.get(name)
        if entry is None:
            continue
        is_comp, is_swept = check_entry(name, entry, errors)
        if is_comp:
            composables += 1
            if is_swept:
                swept += 1
            else:
                gaps.append((name, entry.get("gap_task"), entry.get("note", "")))

    for i, entry in enumerate(supplemental):
        name = entry.get("name", f"<supplemental #{i}>")
        is_comp, is_swept = check_entry(name, entry, errors)
        if is_comp:
            composables += 1
            if is_swept:
                swept += 1
            else:
                gaps.append((name, entry.get("gap_task"), entry.get("note", "")))

    pct = (100.0 * swept / composables) if composables else 0.0
    passed = not errors

    if not args.quiet:
        leaf_count = sum(1 for e in reg_ops.values() if e.get("class") == "leaf")
        print("=" * 68)
        print("Depth-coverage completeness gate (Pillar P6 auditor)")
        print("=" * 68)
        print(f"AST ops in enum        : {len(ops)}")
        print(f"  classified composable: {composables - sum(1 for s in supplemental if s.get('class')=='composable')}")
        print(f"  classified leaf      : {leaf_count}")
        print(f"Supplemental composables: {sum(1 for s in supplemental if s.get('class')=='composable')}")
        print(f"TOTAL composables      : {composables}")
        print(f"  with a pillar sweep  : {swept}")
        print(f"  un-swept (gaps)      : {len(gaps)}")
        print(f"Depth-sweep coverage   : {pct:.1f}%")
        print("-" * 68)
        if gaps:
            print("GAPS (composable, no depth sweep — each tracked by a task):")
            for name, task, note in gaps:
                print(f"  - {name:<34} {task or 'UNTRACKED!':<10} {note}")
        else:
            print("No gaps: every composable construct has a depth sweep.")
        print("-" * 68)
        if errors:
            print(f"GATE FAILED ({len(errors)} problem(s)):")
            for e in errors:
                print(f"  ! {e}")
        else:
            print("GATE PASSED: every construct is classified and every composable "
                  "has a registered depth sweep or a tracked gap.")
        print("=" * 68)

    if not args.no_trace:
        trace_dir = os.path.join(root, "scripts", "icc_traces")
        write_depth_coverage_trace(
            trace_dir,
            passed=passed,
            swept=swept,
            composables=composables,
            pct=pct,
            gap_count=len(gaps),
            error_count=len(errors),
        )

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
