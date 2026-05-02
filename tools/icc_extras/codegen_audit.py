#!/usr/bin/env python3
"""Run the eshkol-codegen audit rules over the tree, a file list, or a git diff.

Three sources of findings:

  1. Regex rules in `codegen_audit_rules.json` — line-anchored patterns with
     skip_if_line_contains, skip_if_match_contains, skip_if_nearby_contains.
     Some are tagged `diff_only: true` — they fire only on lines actually added
     in the current diff (review-on-add), not on baseline code.

  2. Cross-file structural checks (in `cross_file_checks.py`) — things the
     regex layer can't see, e.g. "every parser ESHKOL_*_OP must have a case
     in findFreeVariablesImpl, or be on the documented exempt list."

  3. A baseline file `audit_baseline.json` listing accepted-as-of-{commit}
     findings. The runner subtracts it from the live findings; only NEW
     findings are reported. Use --update-baseline to refresh it after a
     human review pass.

Usage:
    # Diff-mode (recommended for CI):
    python3 tools/icc_extras/codegen_audit.py --diff origin/master

    # Whole-tree audit, ignoring baseline (manual sweep):
    python3 tools/icc_extras/codegen_audit.py --all --no-baseline

    # Refresh the baseline after a review:
    python3 tools/icc_extras/codegen_audit.py --all --update-baseline

Exit status:
    0  no findings of severity >= --severity (after baseline subtraction)
    1  one or more new findings of that severity
    2  usage / IO error
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = REPO_ROOT / "tools" / "icc_extras" / "codegen_audit_rules.json"
BASELINE_PATH = REPO_ROOT / "tools" / "icc_extras" / "audit_baseline.json"


# --------------------------------------------------------------------------- #
#  Rule loading
# --------------------------------------------------------------------------- #


def load_rules(path: Path) -> list[dict[str, Any]]:
    cfg = json.loads(path.read_text())
    rules = cfg["rules"]
    for r in rules:
        try:
            r["_compiled"] = re.compile(r["regex"], re.MULTILINE)
        except re.error as e:
            raise SystemExit(f"rule {r.get('id', '?')!r}: invalid regex: {e}")
    return rules


def globs_match(path: Path, globs: list[str]) -> bool:
    if path.is_absolute():
        try:
            rel = str(path.relative_to(REPO_ROOT))
        except ValueError:
            # Path is outside the repo — match the basename or full string,
            # whichever the caller's glob looks for.  Used so ad-hoc scans of
            # /tmp/foo.cpp still get pattern-matched against `*.cpp`.
            rel = str(path)
    else:
        rel = str(path)
    for g in globs:
        if fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(path.name, g):
            return True
    return False


def rule_applies(rule: dict[str, Any], path: Path) -> bool:
    include = rule.get("include_globs", [])
    exclude = rule.get("exclude_globs", [])
    if include and not globs_match(path, include):
        return False
    if exclude and globs_match(path, exclude):
        return False
    return True


# --------------------------------------------------------------------------- #
#  Per-file scan
# --------------------------------------------------------------------------- #


def line_skipped(line: str, rule: dict[str, Any], match: str) -> bool:
    for needle in rule.get("skip_if_line_contains", []):
        if needle in line:
            return True
    for needle in rule.get("skip_if_match_contains", []):
        if needle in match:
            return True
    return False


def nearby_skip(lines: list[str], idx: int, rule: dict[str, Any]) -> bool:
    needles = rule.get("skip_if_nearby_contains", [])
    if not needles:
        return False
    window = int(rule.get("nearby_line_window", 4))
    lo, hi = max(0, idx - window), min(len(lines), idx + window + 1)
    snippet = "\n".join(lines[lo:hi])
    return any(n in snippet for n in needles)


def scan_file(
    path: Path,
    rules: list[dict[str, Any]],
    diff_added_lines: set[int] | None,
) -> list[dict[str, Any]]:
    """If `diff_added_lines` is provided (diff-mode), rules with diff_only=True
    only fire when the matching line index appears in that set."""
    try:
        text = path.read_text(errors="replace")
    except OSError as e:
        print(f"warning: cannot read {path}: {e}", file=sys.stderr)
        return []
    lines = text.splitlines()
    findings: list[dict[str, Any]] = []
    for rule in rules:
        if not rule_applies(rule, path):
            continue
        diff_only = bool(rule.get("diff_only"))
        if diff_only and diff_added_lines is None:
            # Whole-tree scan: skip diff-only rules entirely.
            continue
        for m in rule["_compiled"].finditer(text):
            line_idx = text.count("\n", 0, m.start())
            line_no = line_idx + 1
            if diff_only and line_no not in diff_added_lines:
                continue
            line_text = lines[line_idx] if line_idx < len(lines) else ""
            if line_skipped(line_text, rule, m.group(0)):
                continue
            if nearby_skip(lines, line_idx, rule):
                continue
            try:
                file_repr = str(path.relative_to(REPO_ROOT)) if path.is_absolute() else str(path)
            except ValueError:
                file_repr = str(path)
            findings.append(
                {
                    "rule_id": rule["id"],
                    "severity": rule.get("severity", "custom"),
                    "file": file_repr,
                    "line": line_no,
                    "match": m.group(0)[:120],
                    "description": rule["description"],
                }
            )
    return findings


# --------------------------------------------------------------------------- #
#  Diff helpers
# --------------------------------------------------------------------------- #


def diff_changed_files(diff_ref: str) -> list[Path]:
    out = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "diff", "--name-only", diff_ref, "--diff-filter=AM"]
    )
    return [REPO_ROOT / line for line in out.decode().splitlines() if line.strip()]


def diff_added_lines_for_file(diff_ref: str, path: Path) -> set[int]:
    """Return the set of new (added) line numbers in `path` since `diff_ref`."""
    rel = path.relative_to(REPO_ROOT) if path.is_absolute() else path
    try:
        out = subprocess.check_output(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "diff",
                "-U0",
                diff_ref,
                "--",
                str(rel),
            ]
        )
    except subprocess.CalledProcessError:
        return set()
    added: set[int] = set()
    new_line = 0
    for line in out.decode().splitlines():
        if line.startswith("@@"):
            # @@ -L,N +L,N @@
            m = re.match(r"@@\s+-\d+(?:,\d+)?\s+\+(\d+)(?:,(\d+))?\s+@@", line)
            if m:
                new_line = int(m.group(1))
            continue
        if line.startswith("+++"):
            continue
        if line.startswith("+"):
            added.add(new_line)
            new_line += 1
        elif line.startswith("-"):
            continue  # deletion doesn't bump new_line
        else:
            new_line += 1
    return added


# --------------------------------------------------------------------------- #
#  Whole-tree file enumeration
# --------------------------------------------------------------------------- #


def all_source_files() -> list[Path]:
    out: list[Path] = []
    for d in ("lib", "exe", "scripts", "tools", "tests"):
        root = REPO_ROOT / d
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix in {".cpp", ".c", ".h", ".hpp", ".sh", ".py", ".yml"}:
                out.append(p)
    return out


# --------------------------------------------------------------------------- #
#  Cross-file checks
# --------------------------------------------------------------------------- #


# Ops that legitimately don't need findFreeVariablesImpl coverage.  Each entry
# documents WHY in inline comments — keep these deliberate, not aspirational.
FIND_FREE_VARS_EXEMPT: dict[str, str] = {
    "ESHKOL_INVALID_OP":           "sentinel; never a real AST node",
    "ESHKOL_DEFINE_SYNTAX_OP":     "macro form; expanded before free-var analysis",
    "ESHKOL_DEFINE_RECORD_TYPE_OP":"macro form; expanded to defines + lambdas before free-var analysis",
    "ESHKOL_LET_SYNTAX_OP":        "macro form; expanded before free-var analysis",
    "ESHKOL_LETREC_SYNTAX_OP":     "macro form; expanded before free-var analysis",
    "ESHKOL_INCLUDE_OP":           "top-level only; not nested inside lambdas",
    "ESHKOL_REQUIRE_OP":           "top-level only; not nested inside lambdas",
    "ESHKOL_PROVIDE_OP":           "top-level only; not nested inside lambdas",
    "ESHKOL_IMPORT_OP":            "top-level only; not nested inside lambdas",
    "ESHKOL_COND_EXPAND_OP":       "macro-like; expanded before free-var analysis",
    "ESHKOL_DEFINE_TYPE_OP":       "type declaration; no expression children to capture",
    "ESHKOL_TYPE_ANNOTATION_OP":   "type annotation; expression child handled by ESHKOL_VAR/CALL_OP path",
    "ESHKOL_EXTERN_VAR_OP":        "extern declaration; no expression children",
    "ESHKOL_SYNTAX_ERROR_OP":      "diagnostic; never reaches free-var analysis",
}


def cross_file_findFreeVariables_coverage() -> list[dict[str, Any]]:
    """Every ESHKOL_*_OP recognised by the parser must either appear as a
    `case` in findFreeVariablesImpl OR be in FIND_FREE_VARS_EXEMPT."""
    findings: list[dict[str, Any]] = []
    parser_text = (REPO_ROOT / "lib" / "frontend" / "parser.cpp").read_text()
    parser_ops = sorted(set(re.findall(r"ESHKOL_[A-Z0-9_]+_OP", parser_text)))

    cg_text = (REPO_ROOT / "lib" / "backend" / "llvm_codegen.cpp").read_text()
    cg_lines = cg_text.splitlines()

    # Bracket findFreeVariablesImpl by brace counting.
    start = next(
        (i for i, l in enumerate(cg_lines) if "void findFreeVariablesImpl(" in l),
        None,
    )
    if start is None:
        findings.append(
            {
                "rule_id": "find_free_vars_impl_missing",
                "severity": "high",
                "file": "lib/backend/llvm_codegen.cpp",
                "line": 1,
                "match": "findFreeVariablesImpl",
                "description": "Cross-file check could not locate findFreeVariablesImpl in lib/backend/llvm_codegen.cpp. Either the function moved (update this script) or it was deleted (which is a bug — free-variable analysis is load-bearing for closure capture).",
            }
        )
        return findings

    depth = 0
    in_func = False
    end = start
    for j in range(start, len(cg_lines)):
        for ch in cg_lines[j]:
            if ch == "{":
                depth += 1
                in_func = True
            elif ch == "}":
                depth -= 1
                if in_func and depth == 0:
                    end = j
                    break
        else:
            continue
        break

    body = "\n".join(cg_lines[start : end + 1])
    handled = set(re.findall(r"case (ESHKOL_[A-Z0-9_]+_OP)\b", body))

    missing = [
        op for op in parser_ops
        if op not in handled and op not in FIND_FREE_VARS_EXEMPT
    ]
    for op in sorted(missing):
        findings.append(
            {
                "rule_id": "find_free_vars_op_uncovered",
                "severity": "medium",
                "file": "lib/backend/llvm_codegen.cpp",
                "line": start + 1,
                "match": op,
                "description": (
                    f"Parser recognises {op} but findFreeVariablesImpl has no case for it. "
                    "Either add a case (recursing into the op's expression children) or, if the op is structurally exempt "
                    "(macro form, top-level only, no expression children), add it to FIND_FREE_VARS_EXEMPT in this script "
                    "with a one-line WHY note. See MEMORY.md findFreeVariablesImpl-coverage class."
                ),
            }
        )
    return findings


# --------------------------------------------------------------------------- #
#  Baseline
# --------------------------------------------------------------------------- #


def finding_key(f: dict[str, Any]) -> str:
    """Stable key: rule_id + file + match content (line drift-tolerant via match hash)."""
    h = hashlib.sha256(f["match"].encode("utf-8")).hexdigest()[:12]
    return f"{f['rule_id']}::{f['file']}::{h}"


def load_baseline(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        d = json.loads(path.read_text())
        return {entry["key"] for entry in d.get("accepted", [])}
    except (json.JSONDecodeError, KeyError):
        print(f"warning: {path} malformed; treating as empty baseline", file=sys.stderr)
        return set()


def write_baseline(path: Path, findings: list[dict[str, Any]]) -> None:
    payload = {
        "$comment": "Audit findings accepted as-of this commit. Subtract from live findings to surface only new ones. Refresh with `codegen_audit.py --all --update-baseline` after a review pass.",
        "schema_version": 1,
        "accepted": [
            {
                "key": finding_key(f),
                "rule_id": f["rule_id"],
                "file": f["file"],
                "line_at_baseline": f["line"],
                "match_excerpt": f["match"],
            }
            for f in findings
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #


SEVERITY_RANK = {"high": 3, "medium": 2, "low": 1, "custom": 0, "all": 0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", help="Specific files to scan.")
    ap.add_argument("--diff", metavar="REF", help="Scan files changed since this git ref.")
    ap.add_argument("--all", action="store_true", help="Scan all source files in the tree.")
    ap.add_argument("--rules", type=Path, default=RULES_PATH)
    ap.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    ap.add_argument("--no-baseline", action="store_true", help="Don't subtract baseline before reporting.")
    ap.add_argument("--update-baseline", action="store_true", help="Write current findings to baseline and exit 0.")
    ap.add_argument("--no-cross-file", action="store_true", help="Skip cross-file structural checks.")
    ap.add_argument("--format", choices=["text", "json"], default="text")
    ap.add_argument("--severity", choices=["high", "medium", "low", "all"], default="all")
    args = ap.parse_args()

    if args.diff and args.all:
        ap.error("--diff and --all are mutually exclusive")
    if args.files and (args.diff or args.all):
        ap.error("positional files cannot be combined with --diff/--all")

    rules = load_rules(args.rules)

    diff_added: dict[Path, set[int]] = {}
    if args.diff:
        candidates = diff_changed_files(args.diff)
        for f in candidates:
            diff_added[f] = diff_added_lines_for_file(args.diff, f)
    elif args.all:
        candidates = all_source_files()
    elif args.files:
        candidates = [Path(f).resolve() for f in args.files]
    else:
        ap.error("specify FILES, --diff REF, or --all")

    all_findings: list[dict[str, Any]] = []
    for path in candidates:
        if not path.exists():
            continue
        all_findings.extend(scan_file(path, rules, diff_added.get(path) if args.diff else None))

    if not args.no_cross_file:
        all_findings.extend(cross_file_findFreeVariables_coverage())

    # Baseline subtraction
    baseline_keys: set[str] = set()
    if not args.no_baseline and not args.update_baseline:
        baseline_keys = load_baseline(args.baseline)

    new_findings = [f for f in all_findings if finding_key(f) not in baseline_keys]

    threshold = SEVERITY_RANK[args.severity]
    filtered = [f for f in new_findings if SEVERITY_RANK.get(f["severity"], 0) >= threshold]

    if args.update_baseline:
        write_baseline(args.baseline, all_findings)
        print(f"wrote {args.baseline} with {len(all_findings)} accepted finding(s).")
        return 0

    if args.format == "json":
        print(
            json.dumps(
                {
                    "findings": filtered,
                    "scanned": len(candidates),
                    "baseline_subtracted": len(baseline_keys),
                    "total_unfiltered": len(all_findings),
                },
                indent=2,
            )
        )
    else:
        if not filtered:
            extra = (
                f" ({len(all_findings)} total, {len(baseline_keys)} baseline-suppressed)"
                if baseline_keys
                else ""
            )
            print(
                f"OK — scanned {len(candidates)} files, no NEW findings at severity "
                f">= {args.severity}{extra}."
            )
        else:
            by_sev: dict[str, list[dict[str, Any]]] = {}
            for f in filtered:
                by_sev.setdefault(f["severity"], []).append(f)
            for sev in ("high", "medium", "low", "custom"):
                items = by_sev.get(sev, [])
                if not items:
                    continue
                print(f"\n=== {sev.upper()} ({len(items)}) ===")
                for it in items:
                    print(f"  {it['file']}:{it['line']}  [{it['rule_id']}]")
                    print(f"      match: {it['match']}")
                    print(f"      why:   {it['description'][:200]}")
            print(
                f"\nTotal: {len(filtered)} new finding(s) "
                f"({len(baseline_keys)} baseline-suppressed, "
                f"{len(candidates)} files scanned)."
            )

    high = sum(1 for f in filtered if f["severity"] == "high")
    return 1 if high > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
# CreatePHI(builder, 1)
