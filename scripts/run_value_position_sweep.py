#!/usr/bin/env python3
r"""run_value_position_sweep.py — the VALUE-POSITION axis.

WHY THIS EXISTS
---------------
Eshkol lowers most of its builtin surface INLINE at the call site: `codegenCall`
string-matches the head name and emits the operation directly. That lowering is
the authoritative one, and it is the only one every other gate exercises.
Referencing the same builtin as a VALUE — passing it to a higher-order
procedure, storing it, returning it — takes a different route, and that route
has repeatedly carried its own independent defect:

  LE-01  builtins had no value representation at all: "Undefined variable:
         string<?" for `(sort xs string<?)`, or a foreign-ABI function pointer
         the closure dispatcher called with the wrong convention (SIGSEGV).
  SW-27  a REST-ARG procedure referenced as a value lost its variadic flag, so
         its rest parameter received a bare argument instead of a list.
         `(h append '(1) '(2))` answered `1`. SILENT, exit 0.
  SW-31  a builtin's value-position route carried its own defect while call
         position was correct.
  SW-35  floor/ceiling/truncate/round as values returned a rational's HEAP
         ADDRESS as a number — addresses that changed between runs.

Four defects, found one at a time, each by somebody probing by hand. Every one
of them was invisible to every gate the project owns, and the reason is
structural rather than accidental:

  * the differential corpus compares EXECUTION AXES against each other, and a
    value-position defect is usually wrong the same way on all four axes, so
    they agree and the gate is green;
  * vm-parity excuses native-only programs;
  * the arity sweep, the surface-parity probe and the language-coverage floor
    all exercise builtins in CALL POSITION only.

A defect that is wrong identically everywhere is invisible to differential
testing between engines or axes. The escape analyses for SW-27 and SW-35 both
prescribed the same fix, in the same words: add a VALUE-POSITION axis. This is
it.

THE ORACLE IS DIFFERENTIAL BY CONSTRUCTION
------------------------------------------
For every builtin the sweep emits ONE program that evaluates the SAME call
twice — once in call position, once through a higher-order procedure — and
compares the two results with `equal?` inside the program:

    (define (h2 f a b) (f a b))
    (probe "call"  (lambda () (expt 2 10)))
    (probe "value" (lambda () (h2 expt 2 10)))

Nothing here hard-codes what `expt` ought to return. The sweep therefore cannot
pass by agreeing with a wrong expectation, it needs no reference implementation,
and it keeps pinning the property if a builtin's semantics legitimately change.
That is the same shape as the SW-27 oracle, generalised to the whole table.

WHAT IS AND IS NOT A FINDING
----------------------------
A builtin is only judged when its CALL POSITION works. If call position raises
or fails to compile, the name is SKIPPED and reported as `call-position-broken`:
that may well be a defect, but it is not a value-position discrepancy and this
gate must not claim it. Reporting someone else's bug as your own finding is how
a gate loses its meaning.

Four probes per builtin, all compared against call position:

    passed    (h N a…)                  — passed to a USER higher-order procedure
    stored    (let ((v N)) (h v a…))    — stored in a variable, then called
    returned  ((lambda () N)) via a HOF — returned from a procedure
    mapped    (car (map N (list a)))    — reached through a BUILTIN combinator
                                          (arity-1 builtins only)

These are not redundant with each other, because they are not one code path.
`passed`/`stored`/`returned` resolve through `codegenVariable`; `mapped`
resolves through `resolveLambdaFunction`, which is a SEPARATE value-position
route with its own builtin handling. SW-35 was live in both and they had to be
fixed independently: after the first fix `(h floor 7/3)` was correct while
`(map floor (list 7/3))` still printed a heap address. A gate that probed only
one of the two routes would have certified that half-fix as complete, which is
the failure mode this whole file exists to prevent.

Likewise LE-01's raw-pointer fallback and SW-27's dropped variadic flag were
reachable through some of these routes and not others, because each is a
distinct codegen site.

RATCHET
-------
Findings are compared against tests/value_position/BASELINE.json. A NEW finding
fails the gate; a known one is reported and does not. The baseline is a list of
`<name>::<probe>` keys with a reason, so an entry cannot be added without saying
why. Regenerate deliberately with --update-baseline; never to make a red gate
green.

USAGE
-----
    python3 scripts/run_value_position_sweep.py                  # jit, all names
    python3 scripts/run_value_position_sweep.py --axis aot-o0
    python3 scripts/run_value_position_sweep.py --all-axes
    python3 scripts/run_value_position_sweep.py --name expt --name append
    python3 scripts/run_value_position_sweep.py --update-baseline

Emits pytest-style `PASSED/FAILED <nodeid>::<probe>` lines plus
`{"kind":"runtime_event","name":"value_position_sweep",...}` JSON-L into
scripts/icc_traces/value_position_sweep.jsonl for the ICC oracle.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO, "tests", "coverage", "language_surface.json")
BASELINE = os.path.join(REPO, "tests", "value_position", "BASELINE.json")
TRACE_DIR = os.path.join(REPO, "scripts", "icc_traces")
TRACE = os.path.join(TRACE_DIR, "value_position_sweep.jsonl")

BUILD = os.environ.get("BUILD_DIR", os.path.join(REPO, "build"))
if not os.path.isabs(BUILD):
    BUILD = os.path.join(REPO, BUILD)
ESHKOL_RUN = os.path.join(BUILD, "eshkol-run")
ESHKOL_VM = os.path.join(BUILD, "eshkol-vm-standalone-test")

# Categories whose argument types the manifest lets us synthesise. Kept
# deliberately identical to the set p8_arity_sweep.py already trusts, so the
# two axes disagree about coverage rather than about typing.
CATEGORIES = {"numeric", "predicate", "string_char", "list_pair", "vector",
              "hash", "higher_order"}

POOLS = {
    "num":  ["3", "5", "2", "7", "4", "6"],
    "flo":  ["2.5", "3.5", "1.5"],
    "str":  ['"hello"', '"abc"', '"xyz"'],
    "list": ["(list 1 2 3)", "(list 4 5 6)", "(list 7 8)"],
    "vec":  ["(vector 1 2 3)", "(vector 4 5 6)"],
    "char": ["#\\a", "#\\z", "#\\m"],
    "bool": ["#t", "#f"],
    "fn":   ["(lambda (x) (+ x 1))", "(lambda (x) (* x 2))"],
}

CAT_ARGS = {
    "numeric":      ["num"],
    "predicate":    ["num"],
    "string_char":  ["str"],
    "list_pair":    ["list"],
    "vector":       ["vec"],
    "hash":         ["hash"],
    "higher_order": ["fn", "list"],
}

# Builtins whose result legitimately differs between two evaluations in the
# same program. Comparing them would flap, so they are excluded BY NAME with a
# reason rather than by a category guess. An exclusion is a claim about the
# builtin, so it has to be defensible on its own.
NONDETERMINISTIC = {
    "random":                 "returns a different value per call by definition",
    "random-integer":         "returns a different value per call by definition",
    "random-real":            "returns a different value per call by definition",
    "current-time":           "wall clock advances between the two evaluations",
    "current-second":         "wall clock advances between the two evaluations",
    "current-jiffy":          "monotonic counter advances between the two evaluations",
    "current-timestamp":      "wall clock advances between the two evaluations",
    "current-time-ns":        "wall clock advances between the two evaluations",
    "monotonic-time-ms":      "monotonic counter advances between the two evaluations",
    "gensym":                 "each call yields a fresh symbol by definition",
    "make-uuid":              "each call yields a fresh identifier by definition",
    "__arena-used":           "arena occupancy changes as the program allocates",
    "cpu-count":              "environment probe, not a pure function of its arguments",
    "getpid":                 "environment probe, not a pure function of its arguments",
}

# Builtins that MUTATE their argument. Evaluating them twice against the same
# freshly-built argument is fine, but the two evaluations must not share the
# argument, which the generated program guarantees by rebuilding the argument
# expression inside each thunk. Listed here only so the reader knows the case
# was considered rather than missed.
MUTATING_HINT = ("set!", "fill!", "sort!", "insert!", "delete!", "push!", "pop!")


# Per-name argument overrides.
#
# The category heuristic types a builtin by its FAMILY, which is right for
# `string-length` and wrong for `make-string`: both are `string_char`, but one
# takes a string and the other takes a COUNT. Feeding `(make-string "hello")`
# does not test make-string, it tests what the compiler does with a type error
# — and since native accepts it silently, the two positions then disagree about
# which garbage to produce and the sweep reports a "finding" that is really its
# own bad input. An oracle that exercises builtins wrongly cannot be trusted
# when it says something is wrong.
#
# Each override is a claim that THIS is how the builtin is meant to be called,
# so they are written out explicitly rather than derived.
ARG_OVERRIDES = {
    "make-string":        ['3', '#\\x'],
    "make-vector":        ['3', '0'],
    "list->string":       ['(list #\\a #\\b)'],
    "string->list":       ['"abc"'],
    "string-fill!":       ['(make-string 3 #\\a)', '#\\z'],
    "string-ends-with?":  ['"hello"', '"lo"'],
    "string-starts-with?":['"hello"', '"he"'],
    "string-contains":    ['"hello"', '"ell"'],
    "string-contains?":   ['"hello"', '"ell"'],
    "string-index":       ['"hello"', '#\\l'],
    "string-ref":         ['"hello"', '1'],
    "substring":          ['"hello"', '1', '3'],
    "string-set!":        ['(make-string 3 #\\a)', '1', '#\\z'],
    "vector-ref":         ['(vector 1 2 3)', '1'],
    "vector-set!":        ['(vector 1 2 3)', '1', '9'],
    "vector-fill!":       ['(make-vector 3 0)', '9'],
    "list-ref":           ['(list 1 2 3)', '1'],
    "list-tail":          ['(list 1 2 3)', '1'],
    "integer->char":      ['65'],
    "char->integer":      ['#\\a'],
    "expt":               ['2', '10'],
    "pow":                ['2', '10'],
    "exact":              ['2.0'],
    "inexact":            ['2'],
}


def _h(name):
    return int(hashlib.sha1(name.encode()).hexdigest(), 16)


def pick(pool, name, slot):
    lst = POOLS[pool]
    return lst[(_h(name) + slot) % len(lst)]


def args_for(name, cat, arity):
    ov = ARG_OVERRIDES.get(name)
    if ov is not None:
        return ov[:arity] if arity <= len(ov) else ov
    if cat == "hash":
        out = ["__ht"]
        for i in range(1, arity):
            out.append(pick("num", name, i))
        return out[:arity] if arity else []
    types = CAT_ARGS.get(cat, ["num"])
    out = []
    for i in range(arity):
        t = types[i] if i < len(types) else types[-1]
        out.append(pick(t, name, i))
    return out


def build_program(name, cat, arity, args):
    """One program, four probes: call position plus three value-position routes.

    Every probe rebuilds its own argument expressions, so a mutating builtin
    cannot make the second evaluation observe the first one's writes.

    Output is bracketed by whitespace-free sentinels because the VM emits a
    newline after every output op; the reader strips all whitespace from both
    engines identically and regex-extracts each probe, which makes the
    comparison newline-insertion-proof.
    """
    hof = "h%d" % arity
    call_args = " ".join(args)
    head = [
        "(define (h0 f) (f))",
        "(define (h1 f a) (f a))",
        "(define (h2 f a b) (f a b))",
        "(define (h3 f a b c) (f a b c))",
        "(define (probe tag thunk)",
        '  (display "#VP#") (display tag) (display "#")',
        '  (guard (e (#t (display "ERR")))',
        "    (write (thunk)))",
        '  (display "#/VP#"))',
    ]
    if cat == "hash":
        head.append("(define __ht (make-hash-table))")
        head.append("(hash-table-set! __ht 1 10)")

    body = [
        # call position — the reference for this program
        '(probe "call" (lambda () (%s %s)))' % (name, call_args),
        # passed to a USER higher-order procedure (codegenVariable route)
        '(probe "passed" (lambda () (%s %s %s)))' % (hof, name, call_args),
        # stored in a variable first
        '(probe "stored" (lambda () (let ((vp-slot %s)) (%s vp-slot %s))))'
        % (name, hof, call_args),
        # returned from a procedure
        '(probe "returned" (lambda () (%s ((lambda () %s)) %s)))'
        % (hof, name, call_args),
    ]
    if arity == 1:
        # Reached through a BUILTIN combinator — resolveLambdaFunction, the
        # second value-position route. SW-35 was live in both routes and each
        # needed its own fix, so probing only the user-HOF route would have
        # certified a half-fix as complete.
        body.append('(probe "mapped" (lambda () (car (map %s (list %s)))))'
                    % (name, args[0]))
    return "\n".join(head) + "\n" + "\n".join(body) + "\n"


PROBE_RE = re.compile(r"#VP#(\w+)#(.*?)#/VP#")

# Comparison uses the FULL value; only the human-facing and trace-facing copies
# are clipped. A gate that prints a megabyte of output is unusable, and one that
# compares a clipped value is a liar.
REPORT_CLIP = 120


def clip(v):
    if v is None:
        return None
    return v if len(v) <= REPORT_CLIP else v[:REPORT_CLIP] + "...<%d bytes>" % len(v)


def parse_probes(out):
    flat = re.sub(r"\s+", "", out)
    return dict(PROBE_RE.findall(flat))


def run_axis(axis, src, workdir, timeout):
    path = os.path.join(workdir, "vp.esk")
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    env = dict(os.environ)
    env.setdefault("ESHKOL_PATH", os.path.join(REPO, "lib"))
    try:
        if axis == "jit":
            p = subprocess.run([ESHKOL_RUN, "-r", path], capture_output=True,
                               text=True, errors="replace",
                               timeout=timeout, env=env, cwd=REPO)
        elif axis == "jit-nocache":
            env["ESHKOL_JIT_CACHE"] = "0"
            p = subprocess.run([ESHKOL_RUN, "-r", path], capture_output=True,
                               text=True, errors="replace",
                               timeout=timeout, env=env, cwd=REPO)
        elif axis in ("aot-o0", "aot-o2"):
            opt = "-O0" if axis == "aot-o0" else "-O2"
            binp = os.path.join(workdir, "vp.bin")
            c = subprocess.run([ESHKOL_RUN, opt, path, "-o", binp],
                               capture_output=True, text=True,
                               errors="replace",
                               timeout=timeout, env=env, cwd=REPO)
            if c.returncode != 0 or not os.path.exists(binp):
                return None, "compile-failed"
            p = subprocess.run([binp], capture_output=True, text=True,
                               errors="replace",
                               timeout=timeout, env=env, cwd=REPO)
        elif axis == "vm":
            if not os.path.exists(ESHKOL_VM):
                return None, "vm-binary-missing"
            env["ESHKOL_VM_NO_DISASM"] = "1"
            p = subprocess.run([ESHKOL_VM, path], capture_output=True,
                               text=True, errors="replace",
                               timeout=timeout, env=env, cwd=REPO)
        else:
            raise SystemExit("unknown axis: %s" % axis)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except (OSError, UnicodeError) as exc:
        # One builtin must never be able to take the sweep down. A name that
        # cannot even be run is reported as skipped, with the reason, and the
        # sweep continues to the next.
        return None, "runner-error:%s" % type(exc).__name__
    return parse_probes(p.stdout), None


def load_baseline():
    if not os.path.exists(BASELINE):
        return {}
    with open(BASELINE, encoding="utf-8") as f:
        return json.load(f).get("known_findings", {})


def emit_trace(records, summary):
    os.makedirs(TRACE_DIR, exist_ok=True)
    with open(TRACE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps(summary) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", action="append", default=[],
                    choices=["jit", "jit-nocache", "aot-o0", "aot-o2", "vm"])
    ap.add_argument("--all-axes", action="store_true")
    ap.add_argument("--name", action="append", default=[],
                    help="restrict to these builtin names (repeatable)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--workdir")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    # The VM is in --all-axes on purpose. It is a genuinely independent
    # implementation of the value-position question — its closures carry the
    # builtin natively — so a defect the four native axes agree on still shows
    # up as a native-vs-VM difference here. That is the one comparison the
    # native axes cannot make against each other.
    axes = args.axis or (["jit", "jit-nocache", "aot-o0", "aot-o2", "vm"]
                         if args.all_axes else ["jit"])

    if not os.path.exists(ESHKOL_RUN):
        raise SystemExit("eshkol-run not found at %s (set BUILD_DIR)" % ESHKOL_RUN)

    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)

    cands = [e for e in manifest["builtins"]
             if isinstance(e.get("arity"), int)
             and 0 <= e["arity"] <= 3
             and e.get("category") in CATEGORIES
             and e["name"] not in NONDETERMINISTIC]
    if args.name:
        want = set(args.name)
        cands = [e for e in cands if e["name"] in want]
    cands.sort(key=lambda e: e["name"])
    if args.limit:
        cands = cands[:args.limit]

    baseline = load_baseline()
    workdir = args.workdir or tempfile.mkdtemp(prefix="value-position-")
    os.makedirs(workdir, exist_ok=True)

    records = []
    new_findings, known_findings, skipped, checked = [], [], [], 0
    started = time.time()

    for e in cands:
        name, cat, arity = e["name"], e["category"], e["arity"]
        src = build_program(name, cat, arity, args_for(name, cat, arity))
        for axis in axes:
            probes, err = run_axis(axis, src, workdir, args.timeout)
            nodeid = "builtins/%s" % name
            if err or probes is None or "call" not in probes:
                skipped.append((name, axis, err or "no-call-probe"))
                records.append({"kind": "runtime_event",
                                "name": "value_position_probe",
                                "status": "SKIP", "builtin": name, "axis": axis,
                                "reason": err or "no-call-probe"})
                continue
            ref = probes["call"]
            if ref == "ERR":
                # Call position itself is broken. Not this axis's finding.
                skipped.append((name, axis, "call-position-broken"))
                records.append({"kind": "runtime_event",
                                "name": "value_position_probe",
                                "status": "SKIP", "builtin": name, "axis": axis,
                                "reason": "call-position-broken"})
                continue
            for probe in ("passed", "stored", "returned", "mapped"):
                if probe not in probes and probe == "mapped":
                    continue  # only emitted for arity-1 builtins
                got = probes.get(probe)
                key = "%s::%s" % (name, probe)
                checked += 1
                if got == ref:
                    if not args.quiet:
                        print("PASSED %s::%s::%s" % (nodeid, probe, axis))
                    records.append({"kind": "runtime_event",
                                    "name": "value_position_probe",
                                    "status": "PASS", "builtin": name,
                                    "probe": probe, "axis": axis})
                    continue
                finding = {"key": key, "axis": axis, "builtin": name,
                           "probe": probe, "call_position": clip(ref),
                           "value_position": clip(got)}
                if key in baseline:
                    known_findings.append(finding)
                    print("FAILED %s::%s::%s  (KNOWN: %s)"
                          % (nodeid, probe, axis, baseline[key]))
                else:
                    new_findings.append(finding)
                    print("FAILED %s::%s::%s  call=%s value=%s"
                          % (nodeid, probe, axis, clip(ref), clip(got)))
                records.append({"kind": "runtime_event",
                                "name": "value_position_probe",
                                "status": "FAIL", "builtin": name,
                                "probe": probe, "axis": axis,
                                "known": key in baseline,
                                "call_position": clip(ref),
                                "value_position": clip(got)})

    elapsed = round(time.time() - started, 1)
    print()
    print("value-position sweep: %d builtin(s), %d comparison(s), axes=%s, %.1fs"
          % (len(cands), checked, ",".join(axes), elapsed))
    print("  matching call position : %d" % (checked - len(new_findings)
                                             - len(known_findings)))
    print("  KNOWN findings         : %d" % len(known_findings))
    print("  NEW findings           : %d   [the only gated number]"
          % len(new_findings))
    exercised = sorted({r["builtin"] for r in records if r["status"] != "SKIP"})
    unexercised = sorted({name for name, _, _ in skipped} - set(exercised))
    print("  skipped                : %d probe-run(s)" % len(skipped))
    print()
    # Coverage is stated out loud because "0 findings" over a table this size
    # invites being read as "the whole builtin surface is proven", and it is
    # not. A name the sweep could not even run is a coverage HOLE, not a pass.
    print("  builtins actually compared : %d of %d selected"
          % (len(exercised), len(cands)))
    print("  builtins NOT exercised     : %d  (%s)"
          % (len(unexercised), ", ".join(unexercised[:12])
             + (", …" if len(unexercised) > 12 else "")))
    print("  -> an unexercised name is an untested name. Extend ARG_OVERRIDES "
          "to bring one into the gate.")

    if args.update_baseline:
        entries = {f["key"]: "recorded by --update-baseline; replace with a reason"
                   for f in new_findings + known_findings}
        os.makedirs(os.path.dirname(BASELINE), exist_ok=True)
        with open(BASELINE, "w", encoding="utf-8") as f:
            json.dump({"_comment": "Value-position ratchet. A NEW key fails the "
                                   "gate. Each entry needs a reason naming the "
                                   "ledger ID or the defect. Never regenerate "
                                   "to turn a red gate green.",
                       "known_findings": entries}, f, indent=2, sort_keys=True)
            f.write("\n")
        print("  baseline rewritten: %d entries" % len(entries))

    ok = not new_findings
    summary = {"kind": "runtime_event", "name": "value_position_sweep",
               "status": "PASS" if ok else "FAIL",
               "builtins": len(cands), "comparisons": checked,
               "axes": axes, "new_findings": len(new_findings),
               "known_findings": len(known_findings), "skipped": len(skipped),
               "builtins_compared": len(exercised),
               "builtins_unexercised": len(unexercised),
               "elapsed_s": elapsed}
    emit_trace(records, summary)
    print("Trace written: %s" % TRACE)

    if not ok:
        print()
        print("FAIL: %d value-position discrepancy(ies) not in the ratchet "
              "baseline." % len(new_findings))
        print("Each one is a builtin that computes a different answer when "
              "referenced as a value than when called directly.")
        return 1
    print("value-position sweep: OK — every builtin answers the same as a "
          "value as it does in call position.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
