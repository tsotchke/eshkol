#!/usr/bin/env python3
"""Inventory every fenced block on every published site page.

By default this is a source-completeness audit; it runs no documentation code.
With ``--run --eshkol-run BIN`` it uses the existing ``run_examples`` runner.
Non-Scheme fences are inventory-only, and ``illustrative``/``needs-context``
records are never reported as tested even when their process exits zero.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import extract_examples  # noqa: E402
import run_examples  # noqa: E402
import check_expected  # noqa: E402

FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})\s*([A-Za-z0-9_+.-]*)\s*$")
SCHEME = {"scheme", "eshkol", "lisp", "racket"}
INFO_LINE = re.compile(r"^(?:\[REPL\] (?:Discovered |Loaded stdlib from cached object: )|\[ESKB\] Loaded |=== Eshkol VM — running |=== Execution complete ===$)")


def assess(plain, classification, expects, checks):
    """Attach an evidence-based verdict; exit zero alone is never verification."""
    diagnostics = '\n'.join(line for line in
                            (plain["stderr"] + "\n" + plain.get("checked_stderr", "")).splitlines()
                            if line.strip() and not INFO_LINE.match(line))
    plain["classification"] = classification
    plain["executed"] = True
    plain["execution_ok"] = plain["exit"] == 0
    plain["oracle"] = "inline-annotations" if expects else None
    plain["expectations"] = checks
    plain["diagnostics"] = diagnostics
    plain["tested"] = classification == "runnable"
    plain["verified"] = bool(
        plain["tested"] and plain["execution_ok"] and expects and checks
        and plain.get("checked_exit") == 0 and len(checks) == len(expects)
        and all(c["ok"] for c in checks) and not diagnostics
    )
    return plain


def pages(root: Path):
    data = json.loads((root / "site" / "pages.json").read_text(encoding="utf-8"))
    return data, [p for p in data["pages"] if (root / p["file"]).is_file()]


def fences(root: Path, page_list):
    """Yield one record for every closed fenced block, including native code."""
    for page in page_list:
        rel = page["file"]
        lines = (root / rel).read_text(encoding="utf-8").splitlines()
        i = 0
        while i < len(lines):
            opening = FENCE_RE.match(lines[i])
            if not opening:
                i += 1
                continue
            char, width = opening.group(2)[0], len(opening.group(2))
            close = None
            for j in range(i + 1, len(lines)):
                candidate = FENCE_RE.match(lines[j])
                if (candidate and candidate.group(1) == opening.group(1)
                        and candidate.group(2)[0] == char
                        and len(candidate.group(2)) >= width
                        and not candidate.group(3)):
                    close = j
                    break
            if close is None:
                raise ValueError(f"{rel}:{i + 1}: unterminated fence")
            body = lines[i + 1:close]
            indent = opening.group(1)
            code = "\n".join(x[len(indent):] if x.startswith(indent) else x for x in body)
            yield {"file": rel, "start_line": i + 1, "end_line": close + 1,
                   "lang": opening.group(3).lower(), "code": code}
            i = close + 1


def inventory(root: Path):
    _doc, published = pages(root)
    missing = [p["file"] for p in json.loads((root / "site" / "pages.json").read_text())["pages"]
               if not (root / p["file"]).is_file()]
    records = list(fences(root, published))
    extracted = extract_examples.extract(str(root), [p["file"] for p in published])
    by_key = {(r["file"], r["start_line"]): r for r in extracted}
    for rec in records:
        linked = by_key.get((rec["file"], rec["start_line"]))
        rec["scheme"] = rec["lang"] in SCHEME
        if linked is None:
            rec["classification"] = "non-scheme" if not rec["scheme"] else "unaccounted"
        else:
            rec["classification"] = linked["klass"]
            rec["extractor"] = linked
    unaccounted = [r for r in records if r["classification"] == "unaccounted"]
    if missing or unaccounted:
        detail = {"missing_sources": missing,
                  "unaccounted_scheme": [(r["file"], r["start_line"]) for r in unaccounted]}
        raise ValueError(json.dumps(detail, indent=2))
    return records, len(published)


def run(records, root: Path, binary: str, modes, out_dir: Path):
    schemes = [r for r in records if r["scheme"]]
    support = {}
    for row in records:
        marker = (row.get("extractor") or {}).get("marker") or {}
        if marker.get("kind") == "file":
            support.setdefault(row["file"], []).append(row)
    modes = list(modes)
    invalid = sorted(set(modes) - {"jit", "aot", "vm"})
    if invalid:
        raise ValueError("unsupported mode(s): " + ", ".join(invalid))
    binary = str(Path(binary).expanduser().resolve())
    work = Path(tempfile.mkdtemp(prefix="published-examples-", dir=str(out_dir)))
    results = []
    try:
        for mode in modes:
            for rec in schemes:
                # Keep the extractor's marker/expectation fields.  The runner
                # needs a normal extractor record, not the inventory wrapper.
                trial = dict(rec.get("extractor") or rec)
                trial["code"] = rec["code"]
                trial["klass"] = rec["classification"]
                trial["support_files"] = {
                    (row.get("extractor") or {}).get("marker", {}).get("token"): row["code"]
                    for row in support.get(rec["file"], [])
                    if row["start_line"] < rec["start_line"]
                }
                plain = run_examples.run_one(trial, binary, mode, str(work), str(root))
                expects = trial.get("expects") or []
                checks = []
                if plain["exit"] == 0 and expects:
                    instrumented, attached = check_expected.instrument_block(rec["code"])
                    checked_trial = dict(trial, code=instrumented)
                    checked = run_examples.run_one(checked_trial, binary, mode, str(work), str(root),
                                                   variant="checked")
                    captured = check_expected.captured(checked["stdout"])
                    for index, exp in enumerate(attached):
                        got = captured.get(exp.get("index", index))
                        ok, shown = (check_expected.satisfied(exp["text"], got)
                                     if got is not None else (False, "<not reached>"))
                        checks.append({"expected": exp["text"], "got": shown, "ok": ok})
                    plain["checked_exit"] = checked["exit"]
                    plain["checked_stdout"] = checked["stdout"]
                    plain["checked_stderr"] = checked["stderr"]
                results.append(assess(plain, rec["classification"], expects, checks))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return results


def self_test() -> bool:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "site").mkdir()
        (root / "docs").mkdir()
        (root / "docs" / "P.md").write_text("```scheme\n(display 1)\n```\n\n```c\nint x;\n```\n")
        (root / "site" / "pages.json").write_text(json.dumps({"pages": [{"file": "docs/P.md"}], "views": {}}))
        # pages() only needs pages for this build-free completeness fixture.
        rows, count = inventory(root)
        assert count == 1 and len(rows) == 2
        assert sum(r["scheme"] for r in rows) == 1
        assert rows[1]["classification"] == "non-scheme"
        (root / "site" / "pages.json").write_text(json.dumps({"pages": [
            {"file": "docs/P.md"}, {"file": "docs/MISSING.md"}], "views": {}}))
        try:
            inventory(root)
        except ValueError as exc:
            assert "MISSING.md" in str(exc)
        else:
            raise AssertionError("missing published source was accepted")
    base = {"exit": 0, "stdout": "1\n", "stderr": "", "checked_exit": 0}
    assert assess(dict(base), "runnable", [{"text": "1"}], [{"ok": True}])["verified"]
    assert not assess(dict(base), "runnable", [{"text": "2"}], [{"ok": False}])["verified"]
    assert not assess({**base, "stderr": "WARNING: bad\n"}, "runnable",
                      [{"text": "1"}], [{"ok": True}])["verified"]
    assert not assess(dict(base), "runnable", [], [])["verified"]
    assert assess({**base, "stderr": "[REPL] Discovered 3 functions\n"}, "runnable",
                  [{"text": "1"}], [{"ok": True}])["verified"]
    assert not assess({**base, "checked_exit": 1}, "runnable",
                      [{"text": "1"}], [{"ok": True}])["verified"]
    assert not assess({**base, "checked_stderr": "WARNING: bad"}, "runnable",
                      [{"text": "1"}], [{"ok": True}])["verified"]
    assert not assess(dict(base), "runnable", [{"text": "1"}, {"text": "2"}],
                      [{"ok": True}])["verified"]
    for source in ('(begin (display "Hello") (newline) 42) ; => 42',
                   '(let ((x 42)) (display "Hello") x) ; => 42'):
        instrumented, annotations = check_expected.instrument_block(source)
        assert annotations[0]['mode'] == 'value'
        assert check_expected.VALUE_MARK in instrumented
        assert check_expected.satisfied('42', 'Hello\n' + check_expected.VALUE_MARK + '42')[0]
    _, annotations = check_expected.instrument_block('(display 42) (newline) ; => 42')
    assert annotations[0]['mode'] == 'prints'
    try:
        run([], root, "/missing/eshkol-run", ["bogus"], root)
    except ValueError as exc:
        assert "unsupported mode" in str(exc)
    else:
        raise AssertionError("invalid mode was accepted")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--manifest", help="write the complete inventory JSON")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--eshkol-run")
    ap.add_argument("--modes", default="jit,aot,vm")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        print("audit_published_examples self-test: PASS" if self_test() else "FAIL")
        return 0
    root = Path(args.root).resolve()
    try:
        records, page_count = inventory(root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"audit_published_examples: FAIL: {exc}", file=sys.stderr)
        return 1
    non_scheme = [r for r in records if not r["scheme"]]
    report = {"pages": page_count, "fences": len(records),
              "languages": dict(collections.Counter(r["lang"] or "(none)" for r in records)),
              "classifications": dict(collections.Counter(r["classification"] for r in records)),
              "native_unexecuted": {"count": len(non_scheme),
                                    "languages": dict(collections.Counter(r["lang"] or "(none)" for r in non_scheme)),
                                    "reason": "inventory only; no shell/native execution due side effects",
                                    "paths": sorted({r["file"] for r in non_scheme})},
              "records": records}
    if args.run:
        if not args.eshkol_run:
            ap.error("--run requires --eshkol-run")
        out_dir = Path(args.out_dir or (root / ".scratch" / "published-example-audit"))
        out_dir.mkdir(parents=True, exist_ok=True)
        results = run(records, root, args.eshkol_run,
                      [m for m in args.modes.split(",") if m], out_dir)
        report["runs"] = results
        report["executed"] = len(results)
        report["verified_pass"] = sum(r["verified"] for r in results)
        report["verification_fail"] = sum(r["tested"] and not r["verified"] for r in results)
        report["executed_nonzero"] = sum(r["exit"] != 0 for r in results)
        report["unverified_runs"] = sum(not r["tested"] for r in results)
    if args.manifest:
        Path(args.manifest).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
    return 1 if report.get("verification_fail", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
