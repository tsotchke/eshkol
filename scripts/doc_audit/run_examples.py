#!/usr/bin/env python3
"""Execute the extracted doc examples and record the result of each.

Usage: run_examples.py <examples.json> <eshkol-run> <outdir> [--mode jit|aot|vm]
       [--only file[:line]] [--jobs N]

Writes <outdir>/results.json with one record per example:
  {file, start_line, klass, mode, exit, stdout, stderr, seconds}

VALIDATION (v1.3.5 assurance wave 2)

ICC's v1.3.4 readiness run flagged `Artifact:results-%s.json` — the file this
script writes below — as `artifact_without_test_or_trace`: produced, but
nothing validates it and nothing traces that it ran. This closes both halves
of ICC's suggested `add_artifact_validator_or_runtime_probe` action:

  * the validator: `scripts/doc_audit/check_results_schema.py` schema-checks
    the file this script just wrote (required fields, sorted order, no
    duplicate (file, start_line, mode) records, record count matches the
    examples manifest) and can be run standalone against any results file;
  * the trace: this script calls that validator on its own output before
    exiting and emits a `runtime_event`-shaped JSON-L record to
    `scripts/icc_traces/doc_examples.jsonl` naming the verdict, so ICC has
    execution evidence for this artifact rather than none at all.

Neither step changes this script's own exit behavior (validation failure is
reported, not silently promoted to a fatal error here) — the gate that FAILS
the build on a broken results file is `check_results_schema.py` itself, run
standalone in CI.
"""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile
import time

TIMEOUT = 60


def _emit_results_validation_trace(results_path, mode):
    """Validate the just-written results file and record a runtime_event
    trace of the verdict. Best-effort: a trace-emission problem must never
    fail the actual doc-example run this script exists to perform."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import check_results_schema as validator  # noqa: E402

        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        trace_dir = os.path.join(repo_root, "scripts", "icc_traces")
        data = validator._load_json(results_path)
        result = validator.check(data)
        status = "PASS" if result["passed"] else "FAIL"
        snippet = (
            f"mode={mode} records={result['record_count']} sorted={result['sorted']}"
            if result["passed"]
            else f"mode={mode} {len(result['errors'])} schema error(s): " + "; ".join(result["errors"][:3])
        )
        os.makedirs(trace_dir, exist_ok=True)
        with open(os.path.join(trace_dir, "doc_examples.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "kind": "runtime_event",
                "name": "doc_examples_results_written",
                "value": status,
                "snippet": snippet,
                "confidence": 1.0,
            }, ensure_ascii=False) + "\n")
        print("doc_examples_results_valid: %s (%s)" % (status, snippet), file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 - tracing must never break the run
        print("run_examples.py: could not emit results-validation trace: %r" % (exc,), file=sys.stderr)


def run_one(rec, eshkol_run, mode, workroot, repo):
    idx = "%s_%d" % (os.path.basename(rec["file"]).replace(".", "_"), rec["start_line"])
    d = os.path.join(workroot, idx)
    os.makedirs(d, exist_ok=True)
    src = os.path.join(d, "ex.esk")
    with open(src, "w", encoding="utf-8") as fh:
        fh.write(rec["code"])
        fh.write("\n")
    # Deliberately NOT setting ESHKOL_JIT_CACHE: the audit runs the compiler in
    # its default configuration, the one a reader of the docs has.
    env = dict(os.environ)
    if mode == "jit":
        cmd = [eshkol_run, "-r", src]
    elif mode == "vm":
        cmd = [eshkol_run, "--vm", src]
    else:
        out = os.path.join(d, "ex.out")
        cmd = [eshkol_run, "-o", out, src]
    t0 = time.time()
    try:
        p = subprocess.run(
            cmd, cwd=d, env=env, capture_output=True, text=True, timeout=TIMEOUT
        )
        rc, so, se = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        rc, so, se = -9, (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), "TIMEOUT"
    except Exception as e:  # noqa: BLE001
        rc, so, se = -99, "", "HARNESS-ERROR: %r" % (e,)
    if mode == "aot" and rc == 0:
        exe = os.path.join(d, "ex.out")
        if os.path.exists(exe):
            try:
                p2 = subprocess.run([exe], cwd=d, env=env, capture_output=True, text=True, timeout=TIMEOUT)
                rc, so, se = p2.returncode, p2.stdout, se + p2.stderr
            except subprocess.TimeoutExpired:
                rc, se = -9, se + "TIMEOUT(run)"
    return {
        "file": rec["file"],
        "start_line": rec["start_line"],
        "end_line": rec["end_line"],
        "klass": rec["klass"],
        "mode": mode,
        "exit": rc,
        "stdout": so[:8000],
        "stderr": se[:8000],
        "seconds": round(time.time() - t0, 2),
        "expects": rec.get("expects", []),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("examples")
    ap.add_argument("eshkol_run")
    ap.add_argument("outdir")
    ap.add_argument("--mode", default="jit")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--only", default=None)
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()

    recs = json.load(open(args.examples))
    if args.only:
        sel = set()
        for spec in args.only.split(","):
            sel.add(spec)
        recs = [
            r
            for r in recs
            if r["file"] in sel or ("%s:%d" % (r["file"], r["start_line"])) in sel
        ]
    os.makedirs(args.outdir, exist_ok=True)
    workroot = tempfile.mkdtemp(prefix="docaudit-", dir=args.outdir)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [
            ex.submit(run_one, r, args.eshkol_run, args.mode, workroot, args.repo)
            for r in recs
        ]
        for i, f in enumerate(concurrent.futures.as_completed(futs)):
            results.append(f.result())
            if (i + 1) % 25 == 0:
                print("... %d/%d" % (i + 1, len(futs)), file=sys.stderr)
    results.sort(key=lambda r: (r["file"], r["start_line"]))
    results_path = os.path.join(args.outdir, "results-%s.json" % args.mode)
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=1)
    ok = sum(1 for r in results if r["exit"] == 0)
    print("mode=%s ran=%d exit0=%d nonzero=%d" % (args.mode, len(results), ok, len(results) - ok))
    _emit_results_validation_trace(results_path, args.mode)


if __name__ == "__main__":
    main()
