#!/usr/bin/env python3
"""Release gate: every example in a gated documentation scope is executed, and
what it prints is what the page says it prints.

Motivating incident: docs/tutorials/12_LISTS.md shipped `(take 3 '(a b c d e))`
and `(sort < lst)`. Both raise when typed into the REPL, because the library
signatures are `(take lst n)` and `(sort lst less?)`. An outside contributor
found it by following the tutorial. The audit scripts in this directory could
have found it first, but they were a sweep someone ran by hand over a fixed
file list that did not include docs/tutorials, and nothing ran them in CI.

This script is the standing form of that sweep. It owns no extraction,
execution or comparison logic of its own; it composes the harness:

    extract_examples.extract        fenced blocks, their markers, their scope
    check_expected.instrument_block `;; =>` annotations -> captured output
    check_expected.compare          annotation vs captured text, as data
    check_output_blocks.collect     a block followed by a ```text fence is a
                                    program plus its exact stdout
    run_examples.run_one            one example, one engine, one time limit

WHAT IS GATED

`extract_examples.GATED_SCOPES` names the path sets under the gate. Today that
is `tutorials` (docs/tutorials). docs/guide, docs/reference and the README
samples join by adding an entry to that table and refreshing the baseline with
`--update-baseline`; this script does not name a documentation path.

WHAT AN EXAMPLE MUST DO

Every fenced scheme/eshkol block is written to a file and run on each engine
in `--engines` (default `jit,aot`: `eshkol-run -r`, which is what the REPL
runs, and a compiled binary). Within `--timeout` seconds it must exit 0 without
writing an error diagnostic, every `;; =>` / `;; prints:` annotation in it must
match (see check_expected.py for attachment and comparison rules), and the
output block pasted for it -- a ```text / ```output / bare fence directly
below, or one marked `<!-- doc-example: output stdout: ... -->` further down --
must match its stdout.

A block is tried on its own first. If that fails it is tried again after the
page's earlier examples, which is the session a reader following the page top
to bottom has. An example that only passes that way is reported as such.

WHAT MAY BE EXCLUDED, AND HOW

Only an explicit marker in the markdown excludes anything (the convention is
documented in extract_examples.py):

    <!-- doc-example: skip <reason>: <why> -->
    <!-- doc-example: run-only <reason>: <why> -->
    <!-- doc-example: known-defect <LEDGER-ID>: <what the page promises> -->

(`<!-- doc-example: file <name>: ... -->` is not an exclusion: the block is
run, and is also written next to the page's later examples as `<name>`.)

There is no heuristic skip. Every marked example is printed with its reason on
every run. The number of marked examples per file is ratcheted in
example_gate_baseline.json: a count above the baseline fails the gate, and a
count below it fails too until the baseline is lowered with
`--update-baseline`, so the recorded number can only go down. An unparseable
marker, a known-defect marker whose ledger entry is missing or closed, and a
known-defect example that now passes on every engine are all failures.

When an example and the implementation disagree, decide which is right before
touching either. A wrong example is fixed on the page. A page that states the
designed behaviour is left alone: the defect goes in the ledger and the example
gets a known-defect marker naming it.

Grading
    PASS  every unmarked example passed on every engine, no annotation is
          unattached, every marker is valid, and the marker counts equal the
          baseline.
    FAIL  anything else. The gate FAILS CLOSED: a missing compiler, a build
          without its stdlib next to the compiler, an empty scope, or an
          unreadable baseline is FAIL.

Usage
    python3 scripts/doc_audit/check_doc_examples.py --eshkol-run build/eshkol-run
    python3 scripts/doc_audit/check_doc_examples.py --eshkol-run build/eshkol-run \
        --only docs/tutorials/12_LISTS.md          # no ratchet, no trace
    python3 scripts/doc_audit/check_doc_examples.py --eshkol-run build/eshkol-run \
        --update-baseline
    python3 scripts/doc_audit/check_doc_examples.py --self-test

Exit status is 0 on PASS and 1 on FAIL.

Copyright (C) tsotchke
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import stat
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import check_expected  # noqa: E402
import check_output_blocks  # noqa: E402
import extract_examples  # noqa: E402
import run_examples  # noqa: E402

DEFAULT_BASELINE = os.path.join(HERE, "example_gate_baseline.json")
DEFAULT_TRACE_DIR = os.path.join(REPO_ROOT, "scripts", "icc_traces")
DEFAULT_ENGINES = ("jit", "aot")
DEFAULT_TIMEOUT = 120
BLOCK_SENTINEL = "@@DOCAUDIT:BLOCK@@"
BASELINE_SCHEMA = "eshkol.doc_example_gate_baseline.v1"
COUNTED_MARKERS = ("skip", "run-only", "known-defect")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def probe_id(scope: str) -> str:
    return "doc_examples_%s_clean" % scope


def trace_basename(scope: str) -> str:
    return "doc_example_gate_%s.jsonl" % scope


# ───────────────────────────── one example ─────────────────────────────

def ledger_status(repo_root: str, entry_id: str) -> str | None:
    """Return the `status:` of .icc/ledger/entries/<entry_id>.yaml, or None."""
    path = os.path.join(repo_root, ".icc", "ledger", "entries", entry_id + ".yaml")
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("status:"):
                    return line.split(":", 1)[1].strip().strip("\"'")
    except OSError:
        return None
    return ""


def support_files(records, rec):
    """Files the page created (`file` markers) before this example."""
    return {other["marker"]["token"]: other["code"] for other in records
            if other["file"] == rec["file"] and other["start_line"] < rec["start_line"]
            and (other.get("marker") or {}).get("kind") == "file"}


def page_context(records, rec):
    """The code of the earlier runnable examples on the same page."""
    parts = []
    for other in records:
        if other["file"] != rec["file"] or other["start_line"] >= rec["start_line"]:
            continue
        marker = other.get("marker")
        if marker and marker["kind"] in ("skip", "known-defect", "invalid", "file"):
            continue
        if other.get("klass") == "data-file":
            continue
        parts.append(check_expected.as_program(other["code"]))
    return "\n".join(parts)


_ERROR_LINE_RE = re.compile(r"^\s*(?:ERROR|error|Error):|Unhandled exception|fatal signal")


def error_lines(stderr, limit=3):
    """Error diagnostics an example wrote while still exiting 0. A reader who
    sees `ERROR:` scroll past has not been shown a working example, whatever
    the exit status says."""
    seen = []
    for ln in _ANSI_RE.sub("", stderr).splitlines():
        ln = ln.strip()
        if _ERROR_LINE_RE.search(ln) and ln not in seen:
            seen.append(ln[:240])
    return seen[:limit]


def _run(rec, code, engine, variant, cfg):
    trial = dict(rec)
    trial["code"] = code
    trial["support_files"] = cfg["support"].get((rec["file"], rec["start_line"]), {})
    result = run_examples.run_one(trial, cfg["eshkol_run"], engine, cfg["workroot"],
                                  cfg["repo_root"], timeout=cfg["timeout"],
                                  variant="%s_%s" % (engine, variant))
    shutil.rmtree(os.path.join(
        cfg["workroot"],
        "%s_%d_%s_%s" % (os.path.basename(rec["file"]).replace(".", "_"), rec["start_line"], engine, variant)),
        ignore_errors=True)
    return result


def _attempt(rec, engine, context, expected_stdout, cfg):
    """Run one example on one engine, alone (`context` None) or after its page
    context. Returns (problems, detail)."""
    marker = rec.get("marker") or {}
    run_only = marker.get("kind") == "run-only"
    program = check_expected.as_program(rec["code"])
    instrumented, expectations = check_expected.instrument_block(rec["code"])
    attached = [e for e in expectations if e["mode"] != "unattached"]
    prefix = ""
    if context is not None:
        prefix = context + '\n(display "%s")(newline)\n' % BLOCK_SENTINEL
    tag = "ctx" if context is not None else "alone"
    problems = []
    detail = {"expectations": len(expectations), "matched": 0}

    def tail(text):
        return text.split(BLOCK_SENTINEL + "\n", 1)[-1] if context is not None else text

    need_plain = run_only or not attached or expected_stdout is not None
    plain = None
    if need_plain:
        plain = _run(rec, prefix + program, engine, tag + "_plain", cfg)
        if plain["exit"] != 0:
            problems.append({"kind": "EXIT", "exit": plain["exit"],
                             "stderr": (plain["stderr"] + "\n" + plain["stdout"][-400:])[:cfg["stderr_tail"]]})
            return problems, detail
        noise = error_lines(plain["stderr"])
        if noise:
            problems.append({"kind": "STDERR", "exit": 0, "stderr": "\n".join(noise)})
        if expected_stdout is not None and not run_only:
            got = tail(plain["stdout"]).rstrip("\n")
            if not check_expected.output_matches(expected_stdout, got):
                problems.append({"kind": "OUTPUT", "expected": expected_stdout.rstrip("\n"), "got": got[:2000]})
    if run_only:
        return problems, detail

    for exp in expectations:
        if exp["mode"] == "unattached":
            problems.append({"kind": "UNATTACHED", "line": rec["start_line"] + 1 + exp["line"],
                             "expected": exp["text"], "why": exp["why"]})
    if attached:
        checked = _run(rec, prefix + instrumented, engine, tag + "_checked", cfg)
        printed = check_expected.captured(tail(checked["stdout"]))
        if checked["exit"] != 0:
            if plain is None:
                plain = _run(rec, prefix + program, engine, tag + "_plain", cfg)
            kind = "EXIT" if plain["exit"] != 0 else "INSTRUMENT"
            problems.append({"kind": kind, "exit": checked["exit"],
                             "stderr": (checked["stderr"] + "\n" + tail(checked["stdout"])[-400:])[:cfg["stderr_tail"]]})
        elif plain is None:
            noise = error_lines(checked["stderr"])
            if noise:
                problems.append({"kind": "STDERR", "exit": 0, "stderr": "\n".join(noise)})
        for exp in attached:
            got = printed.get(exp["index"])
            line = rec["start_line"] + 1 + exp["line"]
            if got is None:
                if checked["exit"] == 0:
                    problems.append({"kind": "NORUN", "line": line, "expected": exp["text"],
                                     "why": "the annotated form was never reached"})
            else:
                ok, shown = check_expected.satisfied(exp["text"], got)
                if ok:
                    detail["matched"] += 1
                else:
                    problems.append({"kind": "MISMATCH", "line": line, "expected": exp["text"], "got": shown[:500]})
    return problems, detail


def evaluate(rec, engine, records, expected_stdout, cfg):
    """Return the verdict record for one example on one engine."""
    problems, detail = _attempt(rec, engine, None, expected_stdout, cfg)
    used_context = False
    if any(p["kind"] in ("EXIT", "NORUN") for p in problems):
        context = page_context(records, rec)
        if context.strip():
            retry_problems, retry_detail = _attempt(rec, engine, context, expected_stdout, cfg)
            # With the page's earlier definitions in scope the example either
            # passes, or fails for its real reason rather than for a missing name.
            problems, detail, used_context = retry_problems, retry_detail, True
    return {"file": rec["file"], "start_line": rec["start_line"], "engine": engine,
            "passed": not problems, "page_context": used_context, "problems": problems, **detail}


# ───────────────────────────── one scope ───────────────────────────────

def marker_counts(records):
    counts = {}
    for rec in records:
        marker = rec.get("marker")
        if marker and marker["kind"] in COUNTED_MARKERS:
            per_file = counts.setdefault(rec["file"], {})
            per_file[marker["kind"]] = per_file.get(marker["kind"], 0) + 1
    return counts


def ratchet_errors(scope, counts, baseline):
    """Compare marked-example counts with the checked-in baseline."""
    recorded = (baseline.get("scopes") or {}).get(scope)
    if recorded is None:
        return ["scope %r has no entry in the baseline; run --update-baseline and review it" % scope]
    errors = []
    allowed = recorded.get("marked") or {}
    for path in sorted(set(counts) | set(allowed)):
        for kind in COUNTED_MARKERS:
            now = counts.get(path, {}).get(kind, 0)
            was = allowed.get(path, {}).get(kind, 0)
            if now > was:
                errors.append("%s: %d `%s` example(s), baseline allows %d -- a new exclusion needs a "
                              "reviewed baseline change" % (path, now, kind, was))
            elif now < was:
                errors.append("%s: %d `%s` example(s), baseline still says %d -- lower it with "
                              "--update-baseline so it cannot creep back" % (path, now, kind, was))
    return errors


def run_scope(scope, cfg, only=None):
    paths = extract_examples.GATED_SCOPES[scope]
    records = extract_examples.extract(cfg["repo_root"], paths)
    outputs = {(p["file"], p["code_line"]): p["expected"]
               for p in check_output_blocks.collect(cfg["repo_root"], paths)}
    selected = records
    if only:
        selected = [r for r in records
                    if r["file"] in only or "%s:%d" % (r["file"], r["start_line"]) in only]
    errors, skipped, run_only, known, todo = [], [], [], [], []
    cfg = dict(cfg, support={(r["file"], r["start_line"]): support_files(records, r) for r in records})
    provided = [r for r in selected if (r.get("marker") or {}).get("kind") == "file"]
    for rec in selected:
        if rec.get("klass") == "data-file":
            continue
        marker = rec.get("marker")
        where = "%s:%d" % (rec["file"], rec["start_line"])
        if marker and marker["kind"] == "invalid":
            errors.append("%s: malformed marker on line %d: %s" % (where, marker["line"], marker["error"]))
            continue
        if marker and marker["kind"] == "skip":
            skipped.append({"where": where, "reason": marker["token"], "text": marker["text"]})
            continue
        if marker and marker["kind"] == "known-defect":
            status = ledger_status(cfg["repo_root"], marker["token"])
            if status is None:
                errors.append("%s: known-defect marker names %s, which has no entry under "
                              ".icc/ledger/entries/" % (where, marker["token"]))
                continue
            if status != "open":
                errors.append("%s: known-defect marker names %s, whose ledger status is %r -- remove the "
                              "marker or reopen the entry" % (where, marker["token"], status))
                continue
        if marker and marker["kind"] == "run-only":
            run_only.append({"where": where, "reason": marker["token"], "text": marker["text"]})
        todo.append(rec)

    verdicts = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg["jobs"]) as pool:
        futures = [pool.submit(evaluate, rec, engine, records,
                               outputs.get((rec["file"], rec["start_line"])), cfg)
                   for rec in todo for engine in cfg["engines"]]
        for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
            verdicts.append(future.result())
            if cfg["progress"] and n % 25 == 0:
                print("... %d/%d" % (n, len(futures)), file=sys.stderr)
    verdicts.sort(key=lambda v: (v["file"], v["start_line"], v["engine"]))

    by_example = {}
    for v in verdicts:
        by_example.setdefault((v["file"], v["start_line"]), []).append(v)
    failed, passed, with_context = [], 0, 0
    for rec in todo:
        vs = by_example[(rec["file"], rec["start_line"])]
        where = "%s:%d" % (rec["file"], rec["start_line"])
        marker = rec.get("marker") or {}
        if marker.get("kind") == "known-defect":
            failing = [v["engine"] for v in vs if not v["passed"]]
            if failing:
                known.append({"where": where, "ledger": marker["token"], "text": marker["text"],
                              "failing_engines": failing})
            else:
                errors.append("%s: marked known-defect %s but passes on every engine -- the defect is "
                              "fixed; remove the marker and close the ledger entry" % (where, marker["token"]))
            continue
        if all(v["passed"] for v in vs):
            passed += 1
            with_context += any(v["page_context"] for v in vs)
        else:
            failed.extend(v for v in vs if not v["passed"])

    if not records:
        errors.append("scope %r contains no examples; a gate over nothing proves nothing" % scope)
    counts = marker_counts(records)
    return {
        "scope": scope, "paths": paths, "engines": list(cfg["engines"]),
        "files": len({r["file"] for r in records}),
        "examples": len([r for r in records if r.get("klass") != "data-file"]),
        "selected": len(selected), "executed": len(todo), "passed": passed,
        "passed_with_page_context": with_context,
        "expectations": sum(v["expectations"] for v in verdicts if v["engine"] == cfg["engines"][0]),
        "failed": failed, "skipped": skipped, "run_only": run_only, "known_defect": known,
        "provided_files": ["%s:%d -> %s" % (r["file"], r["start_line"], r["marker"]["token"]) for r in provided],
        "errors": errors, "marked": counts, "verdicts": verdicts,
    }


# ───────────────────────────── reporting ───────────────────────────────

_DIAGNOSTIC_RE = re.compile(r"error|exception|undefined|unbound|not found|mismatch|TIMEOUT|signal", re.I)


def diagnostic_lines(stderr, limit=3):
    """The few stderr lines that say why an example failed, work paths removed."""
    lines = []
    for ln in _ANSI_RE.sub("", stderr).splitlines():
        ln = re.sub(r"\S*/doc-example-gate-[^/\s]+/[^/\s]+/", "", ln).strip()
        if ln and not ln.startswith("[REPL]") and ln not in lines:
            lines.append(ln)
    picked = [ln for ln in lines if _DIAGNOSTIC_RE.search(ln)][:limit] or lines[-limit:]
    return [ln[:240] for ln in picked]


def print_report(result, out=sys.stdout):
    w = lambda text="": print(text, file=out)  # noqa: E731
    w("doc example gate: scope=%s paths=%s engines=%s" % (
        result["scope"], ",".join(result["paths"]), ",".join(result["engines"])))
    w("  files=%d examples=%d executed=%d passed=%d (of which %d needed the page's earlier examples) "
      "failed=%d" % (result["files"], result["examples"], result["executed"], result["passed"],
                     result["passed_with_page_context"],
                     len({(v["file"], v["start_line"]) for v in result["failed"]})))
    w("  annotations checked per engine=%d  marked: skip=%d run-only=%d known-defect=%d  files provided "
      "by the page=%d" % (result["expectations"], len(result["skipped"]), len(result["run_only"]),
                          len(result["known_defect"]), len(result["provided_files"])))
    for title, rows in (("NOT EXECUTED", result["skipped"]), ("EXECUTED, OUTPUT NOT COMPARED", result["run_only"])):
        if rows:
            w("  %s (%d)" % (title, len(rows)))
            for row in rows:
                w("    %s  %s: %s" % (row["where"], row["reason"], row["text"]))
    if result["known_defect"]:
        w("  KNOWN DEFECT (%d)" % len(result["known_defect"]))
        for row in result["known_defect"]:
            w("    %s  %s [fails on %s]: %s" % (row["where"], row["ledger"],
                                                ",".join(row["failing_engines"]), row["text"]))
    if result["failed"]:
        w("  FAILED")
        for v in result["failed"]:
            for p in v["problems"]:
                where = "%s:%d" % (v["file"], p.get("line", v["start_line"]))
                if p["kind"] in ("EXIT", "INSTRUMENT", "STDERR"):
                    w("    %s [%s] %s exit=%s" % (where, v["engine"], p["kind"], p["exit"]))
                    for ln in diagnostic_lines(p["stderr"]):
                        w("        " + ln)
                elif p["kind"] in ("MISMATCH", "OUTPUT"):
                    w("    %s [%s] %s expected %r got %r" % (where, v["engine"], p["kind"], p["expected"], p["got"]))
                else:
                    w("    %s [%s] %s %r: %s" % (where, v["engine"], p["kind"], p["expected"], p["why"]))
    for error in result["errors"]:
        w("  ERROR %s" % error)


def emit_trace(trace_dir, scope, status, snippet):
    os.makedirs(trace_dir, exist_ok=True)
    path = os.path.join(trace_dir, trace_basename(scope))
    event = {"kind": "eshkol_smoke", "name": probe_id(scope), "value": status,
             "snippet": snippet[:2000], "confidence": 1.0}
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return path


def load_baseline(path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("schema") != BASELINE_SCHEMA:
        raise ValueError("baseline schema is %r, expected %r" % (data.get("schema"), BASELINE_SCHEMA))
    return data


def gate(scopes, cfg, baseline_path, only=None, update_baseline=False, allow_increase=False,
         trace_dir=None, report_path=None, out=sys.stdout):
    """Run the gate over `scopes`; return True on PASS."""
    ok = True
    results = []
    for scope in scopes:
        result = run_scope(scope, cfg, only)
        if not only and not update_baseline:
            try:
                result["errors"].extend(ratchet_errors(scope, result["marked"], load_baseline(baseline_path)))
            except (OSError, ValueError) as exc:
                result["errors"].append("cannot read baseline %s: %s" % (baseline_path, exc))
        result["status"] = "FAIL" if (result["failed"] or result["errors"]) else "PASS"
        print_report(result, out)
        print("  RESULT: %s" % result["status"], file=out)
        ok = ok and result["status"] == "PASS"
        results.append(result)
        if trace_dir and not only and not update_baseline:
            snippet = ("examples=%d executed=%d passed=%d failed=%d skip=%d run_only=%d known_defect=%d "
                       "errors=%d engines=%s" % (
                           result["examples"], result["executed"], result["passed"],
                           len({(v["file"], v["start_line"]) for v in result["failed"]}),
                           len(result["skipped"]), len(result["run_only"]), len(result["known_defect"]),
                           len(result["errors"]), ",".join(result["engines"])))
            emit_trace(trace_dir, scope, result["status"], snippet)
    if update_baseline:
        try:
            previous = load_baseline(baseline_path)
        except (OSError, ValueError):
            previous = {"scopes": {}}
        for result in results:
            old = ((previous.get("scopes") or {}).get(result["scope"]) or {}).get("marked") or {}
            grew = [p for p in result["marked"] for k in COUNTED_MARKERS
                    if result["marked"][p].get(k, 0) > old.get(p, {}).get(k, 0)]
            if grew and old and not allow_increase:
                print("refusing to raise the baseline for %s (%s); pass --allow-increase if the new "
                      "exclusions are reviewed" % (result["scope"], ", ".join(sorted(set(grew)))), file=out)
                return False
        data = {"schema": BASELINE_SCHEMA,
                "note": "Marked (not fully checked) examples per file. Counts may only go down; "
                        "see scripts/doc_audit/check_doc_examples.py.",
                "scopes": dict(previous.get("scopes") or {})}
        for result in results:
            data["scopes"][result["scope"]] = {
                "marked_total": {k: sum(c.get(k, 0) for c in result["marked"].values()) for k in COUNTED_MARKERS},
                "marked": {p: result["marked"][p] for p in sorted(result["marked"])},
            }
        with open(baseline_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print("baseline written: %s" % baseline_path, file=out)
    if report_path:
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1)
    return ok


# ───────────────────────────── self-test ───────────────────────────────

# A stand-in compiler: enough of `eshkol-run -r FILE` and `eshkol-run -o OUT
# FILE` to drive the real extraction, instrumentation, execution and grading
# code without a build. It prints the literal argument of each `(display ...)`,
# honours `(newline)`, exits 1 at `(error ...)`, at `(use NAME)` with no
# `(define NAME` before it and at `(needs-file NAME)` with no such file in
# its working directory, writes `(complain "text")` to stderr, and hangs at
# `(hang)`.
_FAKE_COMPILER = r'''#!/usr/bin/env python3
import os, re, sys, time
args = sys.argv[1:]
out = args[args.index("-o") + 1] if "-o" in args else None
src = open(args[-1]).read()
text, code = [], 0
for m in re.finditer(r'\((display|newline|error|hang|use|needs-file|complain)(?=[\s()])\s*("(?:[^"\\]|\\.)*"|[^\s()]*)', src):
    op, arg = m.group(1), m.group(2)
    if op == "display":
        text.append(arg[1:-1] if arg.startswith('"') else arg)
    elif op == "newline":
        text.append("\n")
    elif op == "hang":
        time.sleep(30)
    elif op == "use" and ("(define " + arg) not in src[:m.start()]:
        sys.stderr.write("Undefined variable: %s\n" % arg); code = 1; break
    elif op == "complain":
        sys.stderr.write(arg[1:-1] + "\n")
    elif op == "needs-file" and not os.path.exists(arg):
        sys.stderr.write("Module not found: %s\n" % arg); code = 1; break
    elif op == "error":
        sys.stderr.write("error: %s\n" % arg); code = 1; break
if out:
    open(out, "w").write("#!/usr/bin/env python3\nimport sys\nsys.stdout.write(%r)\nsys.exit(%d)\n" % ("".join(text), code))
    import os; os.chmod(out, 0o755)
    sys.exit(0)
sys.stdout.write("".join(text)); sys.exit(code)
'''


def _fixture(root, body, baseline_marked=None, ledger=None):
    docs = os.path.join(root, "docs", "tutorials")
    os.makedirs(docs, exist_ok=True)
    with open(os.path.join(docs, "PAGE.md"), "w", encoding="utf-8") as fh:
        fh.write(body)
    for entry_id, status in (ledger or {}).items():
        entries = os.path.join(root, ".icc", "ledger", "entries")
        os.makedirs(entries, exist_ok=True)
        with open(os.path.join(entries, entry_id + ".yaml"), "w", encoding="utf-8") as fh:
            fh.write("id: %s\nstatus: %s\n" % (entry_id, status))
    baseline = os.path.join(root, "baseline.json")
    with open(baseline, "w", encoding="utf-8") as fh:
        json.dump({"schema": BASELINE_SCHEMA,
                   "scopes": {"tutorials": {"marked": baseline_marked or {}}}}, fh)
    return baseline


def self_test() -> bool:
    print("check_doc_examples.py self-test:")
    scratch = os.path.join(REPO_ROOT, ".scratch")
    os.makedirs(scratch, exist_ok=True)
    good = "```scheme\n7 ;; => 7\n(display \"a b\") ;; => a b\n```\n"
    page = "docs/tutorials/PAGE.md"
    cases = [
        ("clean_page_passes", good, None, None, True, ""),
        ("wrong_value_fails", "```scheme\n7 ;; => 8\n```\n", None, None, False, "MISMATCH"),
        ("raising_example_fails", "```scheme\n(error \"take: expected list\")\n```\n", None, None, False, "EXIT"),
        ("hanging_example_fails", "```scheme\n(hang)\n```\n", None, None, False, "EXIT"),
        ("stdout_block_mismatch_fails", "```scheme\n(display \"x\")\n```\n\n```text\ny\n```\n", None, None, False, "OUTPUT"),
        ("annotation_on_procedure_define_fails", "```scheme\n(define (f) 1) ;; => 1\n```\n", None, None, False, "UNATTACHED"),
        ("annotation_on_variable_define_is_its_value", "```scheme\n(define f 1) ;; => 1\n```\n", None, None, False, "MISMATCH"),
        ("transcript_output_is_checked", "```scheme\n> (display \"hi\")\nhi\n> 7\n8\n```\n", None, None, False,
         "MISMATCH expected '8' got '7'"),
        ("file_marker_with_a_path_fails",
         "<!-- doc-example: file ../x.esk: escapes the work directory -->\n```scheme\n(display 1)\n```\n",
         None, None, False, "malformed marker"),
        ("file_marker_provides_the_file_to_later_examples",
         "<!-- doc-example: file mylib.esk: the module required below -->\n```scheme\n(define square 1)\n```\n\n"
         "```scheme\n(needs-file mylib.esk)\n(display 25) ;; => 25\n```\n", None, None, True, "files provided by the page=1"),
        ("missing_file_fails", "```scheme\n(needs-file mylib.esk)\n```\n", None, None, False, "EXIT"),
        ("error_on_stderr_with_exit_0_fails", "```scheme\n(complain \"ERROR: bad storage\")\n(display 1)\n```\n",
         None, None, False, "STDERR"),
        ("marked_output_block_is_compared",
         "```scheme\n(display \"x = 5\")\n```\n\n## Expected output\n\n"
         "<!-- doc-example: output stdout: what the program prints -->\n```\nx = 6\n```\n", None, None, False, "OUTPUT"),
        ("marked_output_block_matches_by_value",
         "```scheme\n(display \"x = 5\")\n```\n\n## Expected output\n\n"
         "<!-- doc-example: output stdout: what the program prints -->\n```\nx = 5\n```\n", None, None, True, ""),
        ("later_block_uses_earlier_define",
         "```scheme\n(define helper 1)\n```\n\n```scheme\n(use helper)\n(display 3) ;; => 3\n```\n", None, None, True, ""),
        ("malformed_marker_fails",
         "<!-- doc-example: skipped -->\n```scheme\n(error \"x\")\n```\n", None, None, False, "malformed marker"),
        ("unknown_reason_fails",
         "<!-- doc-example: skip because: reasons -->\n```scheme\n(error \"x\")\n```\n", None, None, False, "unknown reason"),
        ("marked_skip_within_baseline_passes",
         "<!-- doc-example: skip pseudo-code: a shape, not a program -->\n```scheme\n(error \"x\")\n```\n" + good,
         {page: {"skip": 1}}, None, True, "NOT EXECUTED (1)"),
        ("new_skip_above_baseline_fails",
         "<!-- doc-example: skip pseudo-code: a shape, not a program -->\n```scheme\n(error \"x\")\n```\n" + good,
         None, None, False, "baseline allows 0"),
        ("stale_baseline_fails", good, {page: {"skip": 1}}, None, False, "baseline still says 1"),
        ("run_only_skips_comparison_not_execution",
         "<!-- doc-example: run-only platform-specific: prints a path -->\n```scheme\n(display \"/home/x\") ;; => /home/y\n```\n",
         {page: {"run-only": 1}}, None, True, "OUTPUT NOT COMPARED (1)"),
        ("run_only_still_fails_on_exit",
         "<!-- doc-example: run-only platform-specific: prints a path -->\n```scheme\n(error \"x\")\n```\n",
         {page: {"run-only": 1}}, None, False, "EXIT"),
        ("known_defect_open_and_failing_passes",
         "<!-- doc-example: known-defect LE-900: documented result -->\n```scheme\n7 ;; => 8\n```\n" + good,
         {page: {"known-defect": 1}}, {"LE-900": "open"}, True, "KNOWN DEFECT (1)"),
        ("known_defect_now_passing_fails",
         "<!-- doc-example: known-defect LE-900: documented result -->\n```scheme\n7 ;; => 7\n```\n",
         {page: {"known-defect": 1}}, {"LE-900": "open"}, False, "passes on every engine"),
        ("known_defect_without_ledger_entry_fails",
         "<!-- doc-example: known-defect LE-901: documented result -->\n```scheme\n7 ;; => 8\n```\n",
         {page: {"known-defect": 1}}, None, False, "no entry under"),
        ("known_defect_closed_entry_fails",
         "<!-- doc-example: known-defect LE-900: documented result -->\n```scheme\n7 ;; => 8\n```\n",
         {page: {"known-defect": 1}}, {"LE-900": "closed"}, False, "ledger status is 'closed'"),
        ("empty_scope_fails", "No examples here.\n", None, None, False, "contains no examples"),
    ]
    all_ok = True
    with tempfile.TemporaryDirectory(prefix="doc-example-gate-selftest-", dir=scratch) as temp:
        compiler = os.path.join(temp, "fake-eshkol-run")
        with open(compiler, "w", encoding="utf-8") as fh:
            fh.write(_FAKE_COMPILER)
        os.chmod(compiler, os.stat(compiler).st_mode | stat.S_IXUSR)
        for n, (name, body, marked, ledger, want_pass, want_text) in enumerate(cases):
            root = os.path.join(temp, "case%d" % n)
            os.makedirs(root)
            baseline = _fixture(root, body, marked, ledger)
            work = os.path.join(root, "work")
            os.makedirs(work)
            cfg = {"eshkol_run": compiler, "repo_root": root, "workroot": work, "timeout": 2,
                   "engines": DEFAULT_ENGINES, "jobs": 2, "progress": False, "stderr_tail": 600}

            class _Buffer(list):
                def write(self, text):
                    self.append(text)

            buf = _Buffer()
            trace_dir = os.path.join(root, "traces")
            got_pass = gate(["tutorials"], cfg, baseline, trace_dir=trace_dir, out=buf)
            text = "".join(buf)
            trace_path = os.path.join(trace_dir, trace_basename("tutorials"))
            with open(trace_path, encoding="utf-8") as fh:
                event = json.loads(fh.read())
            trace_ok = (event["kind"] == "eshkol_smoke" and event["name"] == probe_id("tutorials")
                        and event["value"] == ("PASS" if got_pass else "FAIL"))
            ok = got_pass == want_pass and want_text in text and trace_ok
            print("  %s %s" % ("ok  " if ok else "FAIL", name))
            if not ok:
                print("      wanted pass=%s text=%r trace_ok=%s; got pass=%s\n%s" % (
                    want_pass, want_text, trace_ok, got_pass, text))
            all_ok = all_ok and ok
    print("  RESULT: %s" % ("PASS" if all_ok else "FAIL"))
    return all_ok


# ───────────────────────────── entry point ─────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eshkol-run", help="path to the built eshkol-run")
    ap.add_argument("--scope", action="append", choices=sorted(extract_examples.GATED_SCOPES),
                    help="gated scope to run (repeatable; default: every gated scope)")
    ap.add_argument("--engines", default=",".join(DEFAULT_ENGINES), help="comma list of jit, aot, vm")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="per-example limit in seconds")
    ap.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 2))
    ap.add_argument("--repo-root", default=REPO_ROOT)
    ap.add_argument("--work-dir", default=None, help="scratch root (default: <repo>/.scratch)")
    ap.add_argument("--baseline", default=DEFAULT_BASELINE)
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--allow-increase", action="store_true",
                    help="with --update-baseline: accept a higher marked-example count")
    ap.add_argument("--only", default=None, help="comma list of file or file:line (no ratchet, no trace)")
    ap.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR)
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--report", default=None, help="write the full JSON result here")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1

    scopes = args.scope or sorted(extract_examples.GATED_SCOPES)
    trace_dir = None if args.no_trace else os.path.abspath(args.trace_dir)
    repo_root = os.path.abspath(args.repo_root)

    def fail_closed(message):
        print("doc example gate: FAIL: %s" % message)
        if trace_dir and not args.only:
            for scope in scopes:
                emit_trace(trace_dir, scope, "FAIL", message)
        return 1

    if not args.eshkol_run:
        return fail_closed("--eshkol-run is required")
    eshkol_run = os.path.abspath(args.eshkol_run)
    if not (os.path.isfile(eshkol_run) and os.access(eshkol_run, os.X_OK)):
        return fail_closed("compiler not found or not executable: %s" % eshkol_run)
    if not os.path.isfile(os.path.join(os.path.dirname(eshkol_run), "stdlib.o")):
        return fail_closed("no stdlib.o next to %s: an incomplete build would fall back to whatever "
                           "standard library is installed on the machine" % eshkol_run)
    engines = tuple(e for e in args.engines.split(",") if e)
    if not engines or any(e not in ("jit", "aot", "vm") for e in engines):
        return fail_closed("--engines takes a comma list of jit, aot, vm")

    scratch = os.path.abspath(args.work_dir) if args.work_dir else os.path.join(repo_root, ".scratch")
    os.makedirs(scratch, exist_ok=True)
    workroot = tempfile.mkdtemp(prefix="doc-example-gate-", dir=scratch)
    cfg = {"eshkol_run": eshkol_run, "repo_root": repo_root, "workroot": workroot,
           "timeout": args.timeout, "engines": engines, "jobs": max(1, args.jobs),
           "progress": True, "stderr_tail": 4000}
    try:
        ok = gate(scopes, cfg, args.baseline, only=set(args.only.split(",")) if args.only else None,
                  update_baseline=args.update_baseline, allow_increase=args.allow_increase,
                  trace_dir=trace_dir, report_path=args.report)
    finally:
        shutil.rmtree(workroot, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
