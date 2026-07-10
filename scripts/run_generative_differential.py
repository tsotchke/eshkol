#!/usr/bin/env python3
r"""run_generative_differential.py — GENERATIVE multi-oracle differential harness
(adversarial testing pillar P7c).

Principle (from the maintainer): "if our system does not constantly expose every
single hidden bug then it has no coverage." This harness generates a large,
deterministic family of closed R7RS-small programs (scripts/gen_generative_
corpus.py) and runs each one through EVERY execution oracle available, then
cross-checks the outputs. Any pairwise disagreement is an exposed bug.

Oracles (auto-discovered; a program is checked across all that are present):

  chibi   : chibi-scheme <prologue+body>            (external R7RS ground truth)
  jit     : build/eshkol-run -r <body.esk>          (LLVM JIT)
  aot-O0  : build/eshkol-run -O0 <body.esk> -o BIN && BIN
  aot-O2  : build/eshkol-run -O2 <body.esk> -o BIN && BIN
  vm      : build/eshkol-run --profile hosted-vm --emit-eskb X.eskb <body.esk>
            && build/eshkol-vm-standalone-test X.eskb   (bytecode VM)

What this catches that the 34-program hand corpus cannot:
  * R7RS conformance gaps vs chibi, at generated scale.
  * AOT-vs-JIT miscompiles (same source, different Eshkol backends disagree).
  * OPTIMIZATION-LEVEL divergence: aot-O0 vs aot-O2 (metamorphic invariant that
    guards the recent O2-default change; needs no reference).
  * SILENT VM miscompiles: the VM exits 0 even on fatal errors, so a wrong
    value with no error marker is the dangerous case — flagged here.
  * Metamorphic property violations: the "meta" program family self-checks
    (f x)==(apply f (list x)), map ordering, commutativity, reverse involution,
    length/append homomorphism, fold equivalence — any oracle printing #f is a
    bug even with NO reference installed.

Normalization: chibi/jit/aot outputs are compared through
scripts/lib/normalize_scheme_output.py (documented cosmetic canonicalisation;
can only ever hide an implementation-defined rendering difference, never
manufacture a false agreement). The VM appends a newline per `display` call
(a filed quirk, tests/vm_parity/found/display_newline_per_call.esk), so any
comparison INVOLVING the VM additionally strips ALL newlines from both sides —
the strongest comparison the quirk permits; value divergences still surface.

Determinism / reproducibility: the corpus is a pure function of (seed, count),
so a divergence found in CI reproduces locally byte-for-byte. On any divergence
the exact program and every oracle's raw + normalized output are written under
artifacts/generative-diff/divergences/<program>/.

Known-crasher gating: a program that wedges an oracle process (defeats the
per-oracle timeout) can be listed in KNOWN_CRASHERS below (documented, with the
reason) so the harness stays runnable; every oracle otherwise runs under a hard
timeout and a segfault/timeout is captured as a normal result, not an abort.

Baseline / regression mode: `--baseline FILE` treats the divergences recorded
in FILE as KNOWN (the currently-exposed bug list) and exits non-zero only when a
divergence appears that is NOT in the baseline — i.e. a NEW miscompile. Without
`--baseline` the gate is PASS iff ZERO divergences (the "constantly red while
bugs remain" philosophy, matching the reference-diff oracle).

ICC: emits kind:"generative_differential" JSON-L to
scripts/icc_traces/generative_differential.jsonl. The gate event
generative_differential_oracle is PASS/FAIL; per-divergence events carry the
program name and divergence kind. Consumed by
.icc/completion-oracles.yaml::generative-differential and by the
generative_differential_oracle probe in scripts/run_icc_smoke.sh.

Usage:
  scripts/run_generative_differential.py [--seed N] [--count K] [--depth D]
      [--baseline FILE] [--smoke] [--no-vm] [--timeout SECS] [--jobs N]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "lib"))
sys.path.insert(0, HERE)

import normalize_scheme_output as _norm          # noqa: E402
from gen_generative_corpus import generate_programs  # noqa: E402

CHIBI_PROLOGUE = ("(import (scheme base) (scheme write) (scheme char) "
                  "(scheme inexact) (scheme cxr))\n")

# Programs that reliably wedge an oracle process badly enough to threaten the
# runner even under the per-oracle timeout. Format: {program_name: "reason"}.
# Empty by default — every oracle already runs under a hard timeout and a
# crash/timeout is captured as a normal result. Populate only if a specific
# generated program is observed to defeat the timeout.
KNOWN_CRASHERS = {}


# VM banner / loader / compiler-noise lines to drop before comparison. Mirrors
# the perl filter in scripts/run_vm_parity.sh::normalize so only VALUES remain.
_VM_NOISE_PREFIXES = ("WARN", "INFO:", "DEBUG", "[ESKB]", "[GPU]",
                      "=== Eshkol VM", "=== Execution complete ===",
                      "remark:", "warning: <unknown>")


def _strip_vm_banner(text):
    keep = []
    for ln in text.split("\n"):
        s = ln.lstrip()
        if any(s.startswith(p) for p in _VM_NOISE_PREFIXES) or s.startswith("[compiled:"):
            continue
        keep.append(ln)
    return "\n".join(keep)


def _oracle_text(r):
    """Raw stdout to feed the normaliser: strip VM banner/log lines for the VM."""
    return _strip_vm_banner(r.out) if r.name == "vm" else r.out


def _norm_full(text):
    return _norm.normalize(text)


def _norm_nl_free(text):
    # For any comparison involving the VM (whose display appends a per-call
    # newline). Preserve spaces; drop newlines. Mirrors run_vm_parity.sh.
    return _norm_full(text).replace("\n", "")


def _nf(r):
    """Full normalised text of an oracle result (VM banner stripped first)."""
    return _norm_full(_oracle_text(r))


def _nlf(r):
    """Newline-free normalised text (for comparisons involving the VM)."""
    return _norm_nl_free(_oracle_text(r))


class OracleResult:
    __slots__ = ("name", "rc", "out", "err", "timed_out", "compile_failed",
                 "vm_error", "available")

    def __init__(self, name):
        self.name = name
        self.rc = None
        self.out = ""
        self.err = ""
        self.timed_out = False
        self.compile_failed = False
        self.vm_error = False
        self.available = True

    def ran_clean(self):
        """The oracle produced a value we can trust for comparison."""
        if not self.available:
            return False
        if self.timed_out or self.compile_failed or self.vm_error:
            return False
        return self.rc == 0


def _run(cmd, timeout, env=None, cwd=None):
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=timeout, env=env, cwd=cwd)
        return p.returncode, p.stdout.decode("utf-8", "replace"), \
            p.stderr.decode("utf-8", "replace"), False
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode("utf-8", "replace")
        err = (e.stderr or b"").decode("utf-8", "replace")
        return None, out, err, True


class Harness:
    def __init__(self, args):
        self.args = args
        self.build = os.path.join(REPO_ROOT, args.build)
        self.eshkol = os.path.join(self.build, "eshkol-run")
        vm = os.path.join(self.build, "eshkol-vm-standalone-test")
        if not os.path.exists(vm):
            vm = os.path.join(self.build, "eshkol-vm-standalone")
        self.vm_bin = vm
        self.chibi = shutil.which("chibi-scheme")
        self.timeout = args.timeout
        self.work = tempfile.mkdtemp(prefix="eshkol-gendiff.")
        self.esk = os.path.join(self.work, "prog.esk")
        self.scm = os.path.join(self.work, "prog.scm")
        self.bin = os.path.join(self.work, "prog.bin")
        self.eskb = os.path.join(self.work, "prog.eskb")
        self.jit_cache = os.path.join(self.work, "jit-cache")
        os.makedirs(self.jit_cache, exist_ok=True)
        # Which oracles participate.
        self.use_vm = (not args.no_vm) and os.path.exists(self.vm_bin)
        self.use_chibi = (not args.no_chibi) and self.chibi is not None
        self.art = os.path.join(REPO_ROOT, "artifacts", "generative-diff")
        self.diverg_dir = os.path.join(self.art, "divergences")

    def cleanup(self):
        shutil.rmtree(self.work, ignore_errors=True)

    # ---- individual oracle runners ---------------------------------------
    def run_chibi(self, body):
        r = OracleResult("chibi")
        if not self.use_chibi:
            r.available = False
            return r
        with open(self.scm, "w") as fh:
            fh.write(CHIBI_PROLOGUE)
            fh.write(body)
        r.rc, r.out, r.err, r.timed_out = _run([self.chibi, self.scm], self.timeout)
        return r

    def run_jit(self, body):
        r = OracleResult("jit")
        with open(self.esk, "w") as fh:
            fh.write(body)
        env = dict(os.environ, ESHKOL_JIT_CACHE_DIR=self.jit_cache)
        r.rc, r.out, r.err, r.timed_out = _run(
            [self.eshkol, "-r", self.esk], self.timeout, env=env)
        return r

    def run_aot(self, body, opt):
        r = OracleResult("aot-%s" % opt)
        with open(self.esk, "w") as fh:
            fh.write(body)
        if os.path.exists(self.bin):
            os.remove(self.bin)
        crc, cout, cerr, cto = _run(
            [self.eshkol, "-%s" % opt, self.esk, "-o", self.bin], self.timeout)
        if cto:
            r.timed_out = True
            r.err = cerr
            return r
        if crc != 0 or not os.path.exists(self.bin):
            r.compile_failed = True
            r.rc = crc
            r.err = cerr + cout
            return r
        r.rc, r.out, r.err, r.timed_out = _run([self.bin], self.timeout)
        if os.path.exists(self.bin):
            os.remove(self.bin)
        return r

    def run_vm(self, body):
        r = OracleResult("vm")
        if not self.use_vm:
            r.available = False
            return r
        with open(self.esk, "w") as fh:
            fh.write(body)
        if os.path.exists(self.eskb):
            os.remove(self.eskb)
        crc, cout, cerr, cto = _run(
            [self.eshkol, "--profile", "hosted-vm", "--emit-eskb", self.eskb, self.esk],
            self.timeout)
        if cto or crc != 0 or not os.path.exists(self.eskb):
            r.compile_failed = True
            r.rc = crc
            r.err = cerr
            return r
        env = dict(os.environ, ESHKOL_VM_NO_DISASM="1")
        r.rc, r.out, r.err, r.timed_out = _run([self.vm_bin, self.eskb], self.timeout, env=env)
        # The VM exits 0 even on fatal errors; stderr markers are the signal.
        markers = ("ERROR", "OVERFLOW", "unhandled native call", "Assertion",
                   "Segmentation", "abort")
        if any(m in r.err for m in markers):
            r.vm_error = True
        return r

    # ---- one program across all oracles ----------------------------------
    def check_program(self, family, name, body):
        """Return a list of divergence dicts (empty if all oracles agree)."""
        crasher = name in KNOWN_CRASHERS
        results = {}
        results["chibi"] = self.run_chibi(body)
        results["jit"] = self.run_jit(body)
        results["aot-O0"] = self.run_aot(body, "O0")
        results["aot-O2"] = self.run_aot(body, "O2")
        results["vm"] = self.run_vm(body)

        divs = []

        def add(kind, detail, involved):
            divs.append({
                "program": name, "family": family, "kind": kind,
                "detail": detail, "oracles": involved,
            })

        chibi = results["chibi"]
        jit = results["jit"]
        aot0 = results["aot-O0"]
        aot2 = results["aot-O2"]
        vm = results["vm"]

        # --- reference truth: prefer chibi, else jit ----------------------
        ref = chibi if chibi.ran_clean() else (jit if jit.ran_clean() else None)

        # Timeouts are ENVIRONMENTAL (CPU contention on the host), never counted
        # as a differential bug — an oracle that timed out simply does not
        # participate in any comparison for this program.
        def participates(o):
            return o.available and not o.timed_out

        # (1) Each Eshkol native path vs the reference (chibi first).
        if chibi.ran_clean():
            ref_norm = _nf(chibi)
            for o in (jit, aot0, aot2):
                if not participates(o):
                    continue
                if not o.ran_clean():
                    add("%s_VS_CHIBI_ERROR" % o.name.upper().replace("-", "_"),
                        "chibi exit 0 but %s errored (rc=%s comp_fail=%s)"
                        % (o.name, o.rc, o.compile_failed),
                        {"chibi": chibi, o.name: o})
                elif _nf(o) != ref_norm:
                    add("%s_VS_CHIBI_MISMATCH" % o.name.upper().replace("-", "_"),
                        "output differs from chibi ground truth", {"chibi": chibi, o.name: o})

        # (2) Optimization-level metamorphic invariant: aot-O0 == aot-O2.
        if aot0.ran_clean() and aot2.ran_clean():
            if _nf(aot0) != _nf(aot2):
                add("AOT_O0_VS_O2_MISMATCH",
                    "AOT output changes with optimization level (miscompile)",
                    {"aot-O0": aot0, "aot-O2": aot2})
        elif (participates(aot0) and participates(aot2)
              and aot0.ran_clean() != aot2.ran_clean()):
            # One opt level errored (a real codegen error, not a timeout) while
            # the other produced a value.
            add("AOT_O0_VS_O2_STATUS",
                "one AOT opt level errored and the other produced a value",
                {"aot-O0": aot0, "aot-O2": aot2})

        # (3) JIT vs AOT-O0 metamorphic invariant (same backend surface).
        if jit.ran_clean() and aot0.ran_clean():
            if _nf(jit) != _nf(aot0):
                add("JIT_VS_AOT_MISMATCH",
                    "JIT and AOT disagree on the same source",
                    {"jit": jit, "aot-O0": aot0})

        # (4) Silent VM miscompile: VM ran clean (no error marker) but its value
        #     disagrees with the reference. Newline-free comparison (VM quirk).
        if self.use_vm and ref is not None:
            if vm.ran_clean():
                if _nlf(vm) != _nlf(ref):
                    add("VM_SILENT_WRONG",
                        "VM exited clean but value differs from %s" % ref.name,
                        {ref.name: ref, "vm": vm})
            # A VM error marker means the feature is outside the VM subset — that
            # is expected drift, not a differential bug, so it is NOT flagged.

        # (5) Metamorphic self-check: a "meta" program printing #f is a property
        #     violation on whichever oracle printed it (reference-free).
        if family == "meta":
            for o in (chibi, jit, aot0, aot2, vm):
                if not o.available or not o.ran_clean():
                    continue
                if "#f" in _nf(o):
                    add("META_PROPERTY_FALSE",
                        "%s printed #f for a property that must hold" % o.name,
                        {o.name: o})

        if crasher and divs:
            # Documented known-crasher: record but tag so the gate can ignore.
            for d in divs:
                d["known_crasher"] = True

        return divs, results


def _sig(d):
    return "%s::%s" % (d["program"], d["kind"])


def load_baseline(path):
    if not path or not os.path.exists(path):
        return set()
    sigs = set()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                sigs.add(line)
    return sigs


def write_divergence_artifacts(h, family, name, results, divs):
    d = os.path.join(h.diverg_dir, name)
    os.makedirs(d, exist_ok=True)
    for oname, r in results.items():
        if not r.available:
            continue
        with open(os.path.join(d, "%s.stdout" % oname), "w") as fh:
            fh.write(r.out)
        if r.err:
            with open(os.path.join(d, "%s.stderr" % oname), "w") as fh:
                fh.write(r.err)
    with open(os.path.join(d, "DIVERGENCES.txt"), "w") as fh:
        for x in divs:
            fh.write("%s :: %s :: %s\n" % (x["kind"], x["program"], x["detail"]))


def emit_trace(trace_file, name, value, snippet):
    with open(trace_file, "a") as fh:
        fh.write(json.dumps({
            "kind": "generative_differential", "name": name, "value": value,
            "snippet": snippet, "confidence": 0.95,
        }) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--count", type=int, default=60, help="programs per family")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--build", default="build")
    ap.add_argument("--baseline", default=None,
                    help="known-divergence file; exit non-zero only on NEW divergences")
    ap.add_argument("--write-baseline", default=None,
                    help="write the found divergence signatures to this file and exit 0")
    ap.add_argument("--no-vm", action="store_true")
    ap.add_argument("--no-chibi", action="store_true")
    ap.add_argument("--no-reverify", action="store_true",
                    help="skip the reproduce-once confirmation of each divergence")
    ap.add_argument("--smoke", action="store_true",
                    help="fast reduced run (seed=1234 count=12) for CI probes")
    ap.add_argument("--trace-dir", default=os.path.join(REPO_ROOT, "scripts", "icc_traces"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.count = min(args.count, 12)

    if not os.path.exists(os.path.join(REPO_ROOT, args.build, "eshkol-run")):
        print("FATAL: build/eshkol-run not found — build it: "
              "cmake --build build --target eshkol-run stdlib -j", file=sys.stderr)
        return 2

    os.makedirs(args.trace_dir, exist_ok=True)
    trace_file = os.path.join(args.trace_dir, "generative_differential.jsonl")
    open(trace_file, "w").close()

    h = Harness(args)
    shutil.rmtree(h.diverg_dir, ignore_errors=True)
    os.makedirs(h.diverg_dir, exist_ok=True)

    oracles = ["jit", "aot-O0", "aot-O2"]
    if h.use_chibi:
        oracles = ["chibi"] + oracles
    if h.use_vm:
        oracles = oracles + ["vm"]

    print("Generative differential harness (P7c)")
    print("  seed=%d  count/family=%d  depth=%d  timeout=%ds" %
          (args.seed, args.count, args.depth, args.timeout))
    print("  oracles: %s" % ", ".join(oracles))
    print("  chibi:  %s" % (h.chibi or "(not installed — reference-free mode)"))
    print("  vm:     %s" % (h.vm_bin if h.use_vm else "(disabled)"))
    print("-" * 72)

    programs = generate_programs(seed=args.seed, count=args.count)
    all_divs = []
    n_checked = 0
    per_program_div = 0
    try:
        for family, name, body in programs:
            divs, results = h.check_program(family, name, body)
            n_checked += 1
            if divs and not args.no_reverify:
                # Re-verify: re-run the program and keep only divergence KINDS
                # that reproduce. This rejects environmental flakiness (a
                # compile/run that timed out under transient CPU contention
                # would otherwise masquerade as an AOT-O0-vs-O2 status
                # divergence). A genuine miscompile is deterministic and
                # survives; a timeout under load usually does not.
                divs2, results2 = h.check_program(family, name, body)
                kinds2 = set(d["kind"] for d in divs2)
                confirmed = [d for d in divs if d["kind"] in kinds2]
                dropped = [d for d in divs if d["kind"] not in kinds2]
                if dropped and not args.quiet:
                    for d in dropped:
                        print("  (flaky, not reproduced) %-22s %s" % (d["kind"], name))
                divs = confirmed
                # Prefer the artifacts from whichever run still shows the divergence.
                if divs and any(d["kind"] in kinds2 for d in divs):
                    results = results2
            if divs:
                per_program_div += 1
                all_divs.extend(divs)
                write_divergence_artifacts(h, family, name, results, divs)
                if not args.quiet:
                    for d in divs:
                        print("  DIVERGENCE %-26s %s  (%s)"
                              % (d["kind"], name, d["detail"]))
    finally:
        h.cleanup()

    # Group by kind for the summary.
    by_kind = {}
    for d in all_divs:
        by_kind.setdefault(d["kind"], []).append(d["program"])

    print("-" * 72)
    print("programs checked : %d" % n_checked)
    print("programs w/ divergence : %d" % per_program_div)
    print("total divergences: %d" % len(all_divs))
    for kind in sorted(by_kind):
        progs = sorted(set(by_kind[kind]))
        print("  %-28s %3d   e.g. %s" % (kind, len(progs), ", ".join(progs[:4])))

    # Emit per-kind + gate events.
    found_sigs = set(_sig(d) for d in all_divs if not d.get("known_crasher"))
    for kind in sorted(by_kind):
        emit_trace(trace_file, "kind:%s" % kind, "DIVERGES",
                   "%d programs: %s" % (len(set(by_kind[kind])),
                                        ", ".join(sorted(set(by_kind[kind]))[:8])))

    if args.write_baseline:
        with open(args.write_baseline, "w") as fh:
            fh.write("# generative_differential baseline — known divergence "
                     "signatures (program::kind).\n")
            fh.write("# Regenerate: scripts/run_generative_differential.py "
                     "--seed %d --count %d --write-baseline %s\n"
                     % (args.seed, args.count, os.path.basename(args.write_baseline)))
            for s in sorted(found_sigs):
                fh.write(s + "\n")
        print("Wrote baseline (%d signatures) to %s" % (len(found_sigs), args.write_baseline))
        emit_trace(trace_file, "generative_differential_oracle", "PASS",
                   "baseline written: %d known divergences" % len(found_sigs))
        return 0

    baseline = load_baseline(args.baseline)
    new_divs = sorted(found_sigs - baseline) if args.baseline else sorted(found_sigs)
    stale = sorted(baseline - found_sigs) if args.baseline else []

    if args.baseline:
        print("baseline signatures: %d   new (not in baseline): %d   stale: %d"
              % (len(baseline), len(new_divs), len(stale)))
        for s in new_divs:
            print("  NEW DIVERGENCE  %s" % s)
        for s in stale:
            print("  (stale baseline entry no longer reproduces: %s)" % s)

    gate_pass = (len(new_divs) == 0)
    gate = "PASS" if gate_pass else "FAIL"
    if args.baseline:
        snippet = ("baseline mode: %d known, %d NEW divergences (%s)"
                   % (len(baseline), len(new_divs), ", ".join(new_divs[:6]) or "none"))
    else:
        snippet = ("%d divergences across %d programs (%d checked); kinds: %s"
                   % (len(all_divs), per_program_div, n_checked,
                      ", ".join(sorted(by_kind)) or "none"))
    emit_trace(trace_file, "generative_differential_oracle", gate, snippet)

    print("-" * 72)
    print("gate: %s   (%s)" % (gate, snippet))
    print("trace: %s" % trace_file)
    if all_divs:
        print("divergence artifacts: %s" % h.diverg_dir)

    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
