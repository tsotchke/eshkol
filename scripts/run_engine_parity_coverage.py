#!/usr/bin/env python3
"""run_engine_parity_coverage.py — DIFFERENTIAL language-surface coverage.

The question this answers
-------------------------
"How can the two engines possibly disagree?"  Because nothing ever required
them to agree, per construct, on a value.

What already existed:

  * scripts/language_coverage.py proves every one of the ~1091 surface
    constructs EXECUTES — but on ONE engine.  100% there means "it runs",
    never "the engines agree".
  * scripts/vm_parity_audit.py proves every codegen name is CLASSIFIED in
    PARITY.tsv — a hand-typed status string, checked against source text.
  * scripts/run_vm_parity.sh compares real output — for ~60 curated
    programs, so it covers what somebody thought to write.
  * scripts/run_surface_parity.py (added alongside this) proves a name
    RESOLVES on both engines — but not that it computes the same answer.

None of those can catch the dangerous class: a construct that runs on both
engines and returns a DIFFERENT VALUE.  Every divergence fixed in the change
that added this file was exactly that class, and every one of them passed
all four gates above:

    (if '() 'T 'F)                     native T   / VM F
    (kb-query kb '(p a ?x))            substitutions / matching facts
    (unify ?x 42 s)                    (t1 t2 subst) / (subst a b)
    (logic-var? ?x)                    #t / #f
    (kb-retract! kb (make-fact ...))   #f always / removes the fact

What this does
--------------
Both engines already carry per-construct execution instrumentation, keyed on
$ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR (native emits `P <file> <line> <col> <id>
<name>` records; the VM emits `V <vm> 0 0 <id> <name>` from
vm_language_coverage_native_dispatch / _named_call).  This runs each corpus
program under BOTH engines with tracing on, then:

  1. compares normalised stdout — a mismatch is a DIVERGENCE, reported by
     program and by the constructs that program exercised;
  2. credits a construct with DIFFERENTIAL evidence only when some program
     exercised it on both engines AND those runs agreed;
  3. reports differential_fraction = credited / surface, the number to gate
     on — the parity analogue of language_coverage.py's covered_fraction,
     which it deliberately mirrors ("[the only gated number]").

A program the VM cannot run (clean error, out-of-subset) is not a failure —
it is recorded as native-only, and every construct that only it exercises
simply earns no differential credit.  That keeps the number honest instead
of letting an unsupported subset inflate it.

Gate
----
  FAIL  any program whose engines produced different output and that is not
        in the ratchet baseline.
  FAIL  differential_fraction below the recorded floor.

Usage:
  scripts/run_engine_parity_coverage.py [--update-baseline] [--corpus GLOB]
                                        [--limit N] [--json OUT]
Exit: 0 green, 1 red, 2 misuse/environment.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = os.path.join(REPO, "tests", "vm_parity", "ENGINE_PARITY_BASELINE.json")
TRACE_DIR = os.path.join(REPO, "scripts", "icc_traces")
TRACE = os.path.join(TRACE_DIR, "engine_parity_coverage.jsonl")
SURFACE = os.path.join(REPO, "tests", "coverage", "language_surface.json")

BUILD = os.environ.get("BUILD_DIR", os.path.join(REPO, "build"))
if not os.path.isabs(BUILD):
    BUILD = os.path.join(REPO, BUILD)
ESHKOL_RUN = os.path.join(BUILD, "eshkol-run")
VM_BIN = os.path.join(BUILD, "eshkol-vm-standalone-test")
if not os.path.exists(VM_BIN):
    VM_BIN = os.path.join(BUILD, "eshkol-vm-standalone")

DEFAULT_CORPUS = [
    "tests/vm_parity/corpus/*.esk",
    "tests/logic/*.esk",
    "tests/control_flow/*.esk",
    "tests/parser/*.esk",
    "tests/v1_2_edge_cases/*.esk",
]

# Banner/diagnostic lines that are not program output. Same rule as
# run_vm_parity.sh's normalize(), kept in sync deliberately.
NOISE = re.compile(
    r"^(WARN|INFO:|DEBUG|\[ESKB\]|\[GPU\]|\[REPL\]|remark:|warning: <unknown>|"
    r"=== Eshkol VM|=== Execution complete ===|\s*\[compiled:)")


def die(msg):
    sys.stderr.write("run_engine_parity_coverage: %s\n" % msg)
    sys.exit(2)


def normalise(text):
    keep = [ln for ln in text.splitlines() if not NOISE.match(ln)
            and "NOTICE:" not in ln]
    return "".join(keep)


def read_constructs(trace_dir):
    """Construct names recorded by either engine's coverage instrumentation."""
    names = set()
    for path in glob.glob(os.path.join(trace_dir, "*")):
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    # native: P <file> <line> <col> <id> <name>
                    # vm    : V <vm>   0      0     <id> <name>
                    if len(parts) >= 6 and parts[0] in ("P", "V"):
                        if parts[5]:
                            names.add(parts[5])
        except OSError:
            continue
    return names


def run_engine(cmd, program, trace_dir, env_extra, timeout):
    """Run one engine. `trace_dir` None means PRODUCTION conditions.

    Instrumentation must never decide the answer. Setting
    ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR changes native's execution path (it is
    the same switch that made `-r` bypass the JIT object cache in ledger item
    #407), and under it tests/vm_parity/corpus/41_tensor_literals.esk raises
    where it otherwise catches. Comparing instrumented runs therefore invented
    two divergences that do not exist in production. Output is compared with
    tracing OFF; tracing is used only to attribute constructs.
    """
    env = dict(os.environ)
    env.update(env_extra)
    if trace_dir is not None:
        env["ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR"] = trace_dir
    else:
        env.pop("ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR", None)
    env["ESHKOL_LIB_DIR"] = BUILD
    try:
        r = subprocess.run(cmd + [program], capture_output=True, text=True,
                           timeout=timeout, env=env)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, "", ""
    except OSError:
        return 127, "", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--corpus", action="append", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    for path, what in ((ESHKOL_RUN, "eshkol-run"), (VM_BIN, "the VM binary")):
        if not os.path.exists(path):
            die("%s not built at %s (set BUILD_DIR)" % (what, path))

    try:
        with open(SURFACE, encoding="utf-8") as f:
            surface_blob = json.load(f)
    except (OSError, ValueError) as exc:
        die("cannot read language surface manifest %s: %s" % (SURFACE, exc))

    surface = set()
    def harvest(node):
        if isinstance(node, dict):
            if "name" in node and isinstance(node["name"], str):
                surface.add(node["name"])
            for v in node.values():
                harvest(v)
        elif isinstance(node, list):
            for v in node:
                harvest(v)
    harvest(surface_blob)
    if not surface:
        die("language surface manifest yielded no construct names")

    patterns = args.corpus or DEFAULT_CORPUS
    programs = []
    for pat in patterns:
        programs.extend(sorted(glob.glob(os.path.join(REPO, pat))))
    programs = [p for p in programs if os.path.isfile(p)]
    if args.limit:
        programs = programs[:args.limit]
    if not programs:
        die("corpus matched no programs: %s" % patterns)

    print("run_engine_parity_coverage: %d program(s), surface = %d constructs"
          % (len(programs), len(surface)))

    baseline = {"divergent_programs": [], "differential_floor": 0.0,
                "both_ran_programs": []}
    if os.path.exists(BASELINE):
        try:
            with open(BASELINE, encoding="utf-8") as f:
                baseline = json.load(f)
        except (OSError, ValueError):
            pass
    baseline_both_ran = set(baseline.get("both_ran_programs", []))

    agreed_constructs = set()
    native_only_constructs = set()
    divergences = []
    regressions = []
    both_ran_now = []
    native_only, both_ran = 0, 0

    for prog in programs:
        rel = os.path.relpath(prog, REPO)
        # Phase 1 — PRODUCTION conditions, no instrumentation: this decides
        # agreement.
        # Only STDOUT is program output. A diagnostic on stderr is not: these
        # corpus programs deliberately raise and catch, and native prints an
        # "ERROR:" line to stderr while continuing, where the VM does not.
        # Merging the streams invented two divergences whose stdout was
        # byte-identical -- which is why run_vm_parity.sh compares stdout and
        # uses stderr only to decide whether the run was valid at all.
        nrc, nout, nerr = run_engine([ESHKOL_RUN, "-r"], prog, None, {},
                                     args.timeout)
        vrc, vout, verr = run_engine([VM_BIN], prog, None,
                                     {"ESHKOL_VM_NO_DISASM": "1"},
                                     args.timeout)
        # Phase 2 — instrumented, for construct attribution only. Its output
        # is deliberately discarded.
        with tempfile.TemporaryDirectory() as nd, \
                tempfile.TemporaryDirectory() as vd:
            run_engine([ESHKOL_RUN, "-r"], prog, nd,
                       {"ESHKOL_JIT_CACHE": "0"}, args.timeout)
            run_engine([VM_BIN], prog, vd,
                       {"ESHKOL_VM_NO_DISASM": "1"}, args.timeout)
            n_constructs = read_constructs(nd)
            v_constructs = read_constructs(vd)

        if nrc != 0:
            # Native is the reference; a program native cannot run says
            # nothing about parity.
            continue

        # The VM exits 0 even on a fatal error, so stderr markers are the only
        # reliable failure signal (same rule as run_vm_parity.sh's
        # vm_stderr_clean).
        vm_failed = (vrc != 0) or ("ERROR:" in verr) or \
                    ("undefined variable" in verr) or \
                    ("FRAME OVERFLOW" in verr)
        if vm_failed:
            # "The VM cannot run this" must NEVER be a silent escape hatch.
            # A program the baseline recorded as running clean on BOTH engines
            # that now fails on the VM is a REGRESSION, not a reclassification.
            # Without this, reintroducing the '() truthiness bug made
            # 55_truthiness.esk abort on the VM and this gate quietly moved it
            # from "compared" to "native-only" and reported OK — a gate that
            # cannot fail, which is the exact defect this file exists to
            # prevent.
            native_only += 1
            native_only_constructs |= n_constructs
            if rel in baseline_both_ran:
                regressions.append({
                    "program": rel,
                    "vm": (normalise(verr) or normalise(vout))[:200] or "(no output)",
                })
            continue

        both_ran += 1
        both_ran_now.append(rel)
        if normalise(nout) != normalise(vout):
            divergences.append({
                "program": rel,
                "native": normalise(nout)[:200],
                "vm": normalise(vout)[:200],
                "constructs": sorted(n_constructs & v_constructs)[:40],
            })
            continue

        agreed_constructs |= (n_constructs & v_constructs)

    credited = agreed_constructs & surface
    fraction = (len(credited) / len(surface)) if surface else 0.0

    print("  programs where both engines ran clean : %d" % both_ran)
    print("  programs native-only (VM cannot run)  : %d" % native_only)
    print("  DIVERGENT programs (different output) : %d" % len(divergences))
    print("  REGRESSED programs (VM used to run)   : %d" % len(regressions))
    print("  constructs with DIFFERENTIAL evidence : %d / %d  (%.2f%%)"
          " [the only gated number]"
          % (len(credited), len(surface), 100.0 * fraction))

    known = set(baseline.get("divergent_programs", []))
    new_div = [d for d in divergences if d["program"] not in known]
    floor = float(baseline.get("differential_floor", 0.0))

    result = {
        "kind": "runtime_event",
        "name": "engine_semantic_parity",
        "value": ("PASS" if (not new_div and not regressions
                             and fraction >= floor) else "FAIL"),
        "snippet": ("%d/%d constructs with differential evidence (%.2f%%), "
                    "%d divergent program(s), %d new"
                    % (len(credited), len(surface), 100.0 * fraction,
                       len(divergences), len(new_div))),
        "confidence": 0.95,
    }
    os.makedirs(TRACE_DIR, exist_ok=True)
    with open(TRACE, "w", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({
                "differential_fraction": fraction,
                "credited": sorted(credited),
                "divergent": divergences,
                "native_only_constructs": sorted(
                    (native_only_constructs & surface) - credited),
            }, f, indent=2)

    if args.update_baseline:
        with open(BASELINE, "w", encoding="utf-8") as f:
            json.dump({
                "_comment": ("Ratchet for scripts/run_engine_parity_coverage.py. "
                             "divergent_programs may never grow; "
                             "differential_floor may never fall."),
                "divergent_programs": sorted(d["program"] for d in divergences),
                "both_ran_programs": sorted(both_ran_now),
                "differential_floor": round(fraction, 4),
            }, f, indent=2)
            f.write("\n")
        print("  baseline written: %d divergent program(s), floor %.4f"
              % (len(divergences), fraction))
        return 0

    rc = 0
    if regressions:
        rc = 1
        print()
        print("FAIL: %d program(s) that BOTH engines used to run now fail on "
              "the VM.\n      A VM failure is not an out-of-subset "
              "reclassification." % len(regressions))
        for r in regressions[:20]:
            print("  %s" % r["program"])
            print("      vm: %s" % r["vm"][:140])
    if new_div:
        rc = 1
        print()
        print("FAIL: %d program(s) produce DIFFERENT OUTPUT on the two engines "
              "and are not\n      in the ratchet baseline." % len(new_div))
        for d in new_div[:20]:
            print("  %s" % d["program"])
            print("      native: %s" % d["native"][:120])
            print("      vm    : %s" % d["vm"][:120])
            if d["constructs"]:
                print("      constructs exercised on both: %s"
                      % ", ".join(d["constructs"][:12]))
    if fraction < floor:
        rc = 1
        print()
        print("FAIL: differential construct coverage %.2f%% fell below the "
              "recorded floor %.2f%%." % (100.0 * fraction, 100.0 * floor))

    if rc == 0:
        print()
        print("run_engine_parity_coverage: OK — no program computes a "
              "different answer on\nthe two engines, and differential "
              "construct coverage held its floor.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
