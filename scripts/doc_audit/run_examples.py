#!/usr/bin/env python3
"""Execute the extracted doc examples and record the result of each.

Usage: run_examples.py <examples.json> <eshkol-run> <outdir> [--mode jit|aot|vm]
       [--only file[:line]] [--jobs N]

Writes <outdir>/results.json with one record per example:
  {file, start_line, klass, mode, exit, stdout, stderr, seconds}
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
    with open(os.path.join(args.outdir, "results-%s.json" % args.mode), "w") as fh:
        json.dump(results, fh, indent=1)
    ok = sum(1 for r in results if r["exit"] == 0)
    print("mode=%s ran=%d exit0=%d nonzero=%d" % (args.mode, len(results), ok, len(results) - ok))


if __name__ == "__main__":
    main()
