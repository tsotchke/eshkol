#!/usr/bin/env python3
"""Round trip for core.certificate (ADR-0047): emit -> independent check
PASS; tamper -> FAIL; Lean export.

  run_certificate_roundtrip.py ESHKOL_RUN WORK_DIR [--aot BINARY LINK_DIR]

1. Runs tests/stdlib/certificate_test.esk (JIT with ESHKOL_RUN, or compiled
   ahead of time to BINARY against LINK_DIR and then run), with ESHKOL_CERTIFICATE_TEST_DOC and ESHKOL_CERTIFICATES set, so
   it writes the explicit certificates and the (certify ...) certificates.
2. Runs tools/certcheck/eshkol_certcheck.py on both documents: every claim
   must be accepted (the Taylor-model claim is CONDITIONAL on its premise).
3. Runs tools/certcheck/tamper_suite.py on both: every altered claim must be
   rejected.
4. Exports the explicit document to Lean (tools/certcheck/lean_export.py)
   and requires every exportable kind to be present.
Prints "RESULT: ALL PASS" on success.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools" / "certcheck"


def run(cmd, env=None):
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1200)
    return p.returncode, p.stdout + p.stderr


def main():
    if len(sys.argv) < 3:
        print("usage: run_certificate_roundtrip.py ESHKOL_RUN WORK_DIR [--aot BINARY]")
        return 2
    eshkol_run, work = sys.argv[1], Path(sys.argv[2])
    work.mkdir(parents=True, exist_ok=True)
    explicit, process = work / "explicit.json", work / "process.json"
    started = time.time() - 1

    def fresh(f):  # written by this run, not left over from an earlier one
        return f.exists() and f.stat().st_mtime >= started

    env = dict(os.environ)
    env.update({"ESHKOL_CERTIFICATE_TEST_DOC": str(explicit), "ESHKOL_CERTIFICATES": str(process),
                "ESHKOL_CERTIFICATES_PROGRAM": "certificate_test_certify_forms",
                "ESHKOL_PATH": str(REPO / "lib"), "ESHKOL_JIT_CACHE": "0"})
    if "--aot" in sys.argv:
        i = sys.argv.index("--aot")
        binary, link_dir = sys.argv[i + 1], sys.argv[i + 2]
        rc, out = run([eshkol_run, "-O", "1", "-o", binary,
                       str(REPO / "tests" / "stdlib" / "certificate_test.esk"), f"-L{link_dir}"], env)
        if rc != 0:
            print(out[-4000:])
            print("RESULT: FAIL (AOT compile)")
            return 1
        cmd = [binary]
    else:
        cmd = [eshkol_run, "-r", str(REPO / "tests" / "stdlib" / "certificate_test.esk")]
    rc, out = run(cmd, env)
    ok = rc == 0 and "PASS: certificate_test" in out and fresh(explicit) and fresh(process)
    print(f"[1] emit: {'ok' if ok else 'FAILED'} (exit {rc})")
    if not ok:
        print(out[-4000:])
        print("RESULT: FAIL")
        return 1

    rc, out = run([sys.executable, str(TOOLS / "eshkol_certcheck.py"), str(explicit), str(process)])
    print(out)
    accepted = out.count("  ACCEPTED ") + out.count("  CONDITIONAL ")
    if rc != 0 or "RESULT: ALL ACCEPTED" not in out:
        print("RESULT: FAIL")
        return 1
    print(f"[2] independent check: {accepted} claims accepted")

    rc, out = run([sys.executable, str(TOOLS / "tamper_suite.py"), str(explicit), str(process)])
    print(out[-3000:])
    if rc != 0 or "RESULT: ALL REJECTED" not in out:
        print("RESULT: FAIL")
        return 1
    print("[3] tamper suite: every altered claim rejected")

    lean = work / "CertificateTest.lean"
    rc, out = run([sys.executable, str(TOOLS / "lean_export.py"), str(explicit), "-o", str(lean)])
    text = lean.read_text(encoding="utf-8") if lean.exists() else ""
    kinds = ["determinant_", "rank_", "smith_normal_form_", "integer_gcd_", "primality_",
             "cocycle_", "cohomology_class_"]
    missing = [k for k in kinds if f"theorem {k}" not in text]
    print(out.strip())
    if rc != 0 or missing:
        print(f"lean export missing kinds: {missing}")
        print("RESULT: FAIL")
        return 1
    print(f"[4] Lean export: {text.count(chr(10) + 'theorem ')} theorems")
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
