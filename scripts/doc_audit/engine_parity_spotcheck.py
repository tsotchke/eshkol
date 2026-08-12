#!/usr/bin/env python3
"""Engine-parity spot check for doc examples.

Takes a sample of the doc programs whose pasted output already matches on the
JIT and re-runs each one as an AOT binary and on the bytecode VM, so the docs
are shown to be true of the engines a reader might actually use — not just of
`eshkol-run -r`.

Usage: engine_parity_spotcheck.py <repo> <build-dir> <outblocks.json> [N]
"""

import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_output_blocks import collect  # noqa: E402

TIMEOUT = 120


def run(cmd, cwd):
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=TIMEOUT)
        return p.returncode, p.stdout.rstrip("\n")
    except subprocess.TimeoutExpired:
        return -9, "<timeout>"


def main():
    repo, build, results_path = sys.argv[1:4]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    run_bin = os.path.join(build, "eshkol-run")
    vm_bin = os.path.join(build, "eshkol-vm-standalone-test")
    if not os.path.exists(vm_bin):
        vm_bin = os.path.join(build, "eshkol-vm-standalone")

    ok = {(r["file"], r["code_line"]) for r in json.load(open(results_path))
          if r["verdict"] == "MATCH"}
    per_file, picked = {}, []
    for p in collect(repo):
        key = (p["file"], p["code_line"])
        if key not in ok or per_file.get(p["file"], 0) >= 2:
            continue
        per_file[p["file"]] = per_file.get(p["file"], 0) + 1
        picked.append(p)
        if len(picked) >= n:
            break

    work = tempfile.mkdtemp(prefix="doc-parity-")
    tally = {"jit": 0, "aot": 0, "vm": 0, "aot_nocompile": 0, "vm_differs": 0}
    rows = []
    for i, p in enumerate(picked):
        d = os.path.join(work, str(i))
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "ex.esk"), "w").write(p["code"] + "\n")
        exp = p["expected"].rstrip("\n")
        origin = "%s:%d" % (p["file"], p["code_line"])

        _, got = run([run_bin, "-r", "ex.esk"], d)
        jit = got == exp
        tally["jit"] += jit

        rc, _ = run([run_bin, "-o", "ex.bin", "ex.esk"], d)
        if rc == 0 and os.path.exists(os.path.join(d, "ex.bin")):
            _, got = run(["./ex.bin"], d)
            aot = got == exp
        else:
            aot, tally["aot_nocompile"] = None, tally["aot_nocompile"] + 1
        tally["aot"] += bool(aot)

        vm = None
        if os.path.exists(vm_bin):
            _, got = run([vm_bin, "ex.esk"], d)
            vm = got == exp
            tally["vm"] += vm
            tally["vm_differs"] += not vm
        rows.append((origin, jit, aot, vm))

    print("%-58s %-5s %-5s %-5s" % ("doc example", "jit", "aot", "vm"))
    for origin, jit, aot, vm in rows:
        f = lambda v: "-" if v is None else ("ok" if v else "DIFF")
        print("%-58s %-5s %-5s %-5s" % (origin, f(jit), f(aot), f(vm)))
    print("\nspot-check: %d programs | jit %d | aot %d (%d did not compile) | vm %d (%d differ)"
          % (len(rows), tally["jit"], tally["aot"], tally["aot_nocompile"],
             tally["vm"], tally["vm_differs"]))


if __name__ == "__main__":
    main()
