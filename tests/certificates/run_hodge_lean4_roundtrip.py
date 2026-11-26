#!/usr/bin/env python3
"""End-to-end round-trip verification of M13/M15 certificates through:
  1. Eshkol executable emission (tests/stdlib/hodge_hitchin_lean4_test.esk)
  2. Independent Python checker (tools/certcheck/eshkol_certcheck.py --strict)
  3. Negative control: tampered JSON witness rejected by eshkol_certcheck.py
  4. Lean 4 export (tools/certcheck/lean_export.py) exporting ALL 5 claims (0 skipped)
  5. Real Lean 4 kernel verification (`lean`) proving every theorem `by decide`
  6. Negative control: tampered Lean PSD witness rejected by the Lean 4 kernel!
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def find_lean() -> str | None:
    lean = shutil.which("lean")
    if lean:
        return lean
    candidate = Path.home() / ".elan" / "bin" / "lean"
    if candidate.exists():
        return str(candidate)
    return None


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    eshkol_run = os.environ.get("ESHKOL_RUN")
    if not eshkol_run:
        print("ERROR: set ESHKOL_RUN to path of eshkol-run binary", file=sys.stderr)
        return 2

    certcheck = repo / "tools" / "certcheck" / "eshkol_certcheck.py"
    lean_export = repo / "tools" / "certcheck" / "lean_export.py"
    test_esk = repo / "tests" / "stdlib" / "hodge_hitchin_lean4_test.esk"

    with tempfile.TemporaryDirectory(prefix="eshkol_hodge_lean4_") as tmp:
        tmp_path = Path(tmp)
        cert_json = tmp_path / "hodge_hitchin_certs.json"
        tampered_json = tmp_path / "hodge_hitchin_tampered.json"
        lean_out = tmp_path / "CertHodgeHitchin.lean"
        tampered_lean = tmp_path / "CertHodgeHitchinTampered.lean"

        # 1. Run Eshkol test and emit eshkol.certificate.v1 JSON
        env = os.environ.copy()
        env["ESHKOL_CERTIFICATES"] = str(cert_json)
        r1 = subprocess.run(
            [eshkol_run, "-r", "-I", str(repo / "lib"), str(test_esk)],
            cwd=str(repo), env=env, capture_output=True, text=True
        )
        if r1.returncode != 0:
            print("FAIL: Eshkol emission failed:\n", r1.stdout, r1.stderr, file=sys.stderr)
            return 1
        print(r1.stdout.strip())
        assert cert_json.exists(), "Certificate JSON was not created"

        doc = json.loads(cert_json.read_text(encoding="utf-8"))
        assert len(doc["claims"]) == 5, f"Expected 5 claims, got {len(doc['claims'])}"
        print(f"PASS: emitted {len(doc['claims'])} eshkol.certificate.v1 claims")

        # 2. Independent Python checker (strict mode)
        r2 = subprocess.run(
            [sys.executable, str(certcheck), "--strict", str(cert_json)],
            capture_output=True, text=True
        )
        if r2.returncode != 0:
            print("FAIL: eshkol_certcheck.py rejected valid certificate:\n", r2.stdout, r2.stderr, file=sys.stderr)
            return 1
        print("PASS: eshkol_certcheck.py --strict ACCEPTED all 5 claims")

        # 3. Negative control: tamper a PSD diagonal weight in JSON -> MUST BE REJECTED
        tampered = json.loads(cert_json.read_text(encoding="utf-8"))
        for c in tampered["claims"]:
            if c["kind"] == "psd" and c["output"]["v"] is True:
                # Corrupt first diagonal weight
                d0 = c["witness"]["D"]["data"][0]
                if d0["t"] == "z":
                    d0["v"] = str(int(d0["v"]) + 1)
                elif d0["t"] == "q":
                    d0["n"] = str(int(d0["n"]) + 1)
                break
        tampered_json.write_text(json.dumps(tampered), encoding="utf-8")
        r3 = subprocess.run(
            [sys.executable, str(certcheck), "--strict", str(tampered_json)],
            capture_output=True, text=True
        )
        if r3.returncode == 0:
            print("FAIL: eshkol_certcheck.py failed to reject tampered PSD witness!", file=sys.stderr)
            return 1
        print("PASS: eshkol_certcheck.py --strict REJECTED tampered PSD witness")

        # 4. Export all 5 claims to Lean 4 (assert 0 skipped!)
        r4 = subprocess.run(
            [sys.executable, str(lean_export), str(cert_json), "-o", str(lean_out)],
            capture_output=True, text=True
        )
        if r4.returncode != 0:
            print("FAIL: lean_export.py failed:\n", r4.stdout, r4.stderr, file=sys.stderr)
            return 1
        print(r4.stdout.strip())
        lean_text = lean_out.read_text(encoding="utf-8")
        assert "Exported claims: 5; not exported: 0." in lean_text, (
            f"Expected all 5 claims exported with 0 skipped! Output:\n{r4.stdout}"
        )
        print("PASS: lean_export.py exported 5/5 claims (0 skipped)")

        # 5. Real Lean 4 kernel verification
        lean_bin = find_lean()
        if not lean_bin:
            print("FAIL: Lean 4 binary not found; kernel round-trip check required", file=sys.stderr)
            return 1
        r5 = subprocess.run([lean_bin, str(lean_out)], capture_output=True, text=True)
        if r5.returncode != 0:
            print("FAIL: Lean 4 kernel rejected exported proof:\n", r5.stdout, r5.stderr, file=sys.stderr)
            return 1
        print(f"PASS: Lean 4 kernel ({lean_bin}) verified ALL 5 theorems `by decide`!")

        # 6. Negative control: tamper exported Lean 4 theorem -> Lean kernel MUST REJECT
        assert "allPos [" in lean_text
        bad_lean = lean_text.replace("allPos [", "allPos [-1, ", 1)
        tampered_lean.write_text(bad_lean, encoding="utf-8")
        r6 = subprocess.run([lean_bin, str(tampered_lean)], capture_output=True, text=True)
        if r6.returncode == 0:
            print("FAIL: Lean 4 kernel failed to reject tampered PSD proof!", file=sys.stderr)
            return 1
        print("PASS: Lean 4 kernel REJECTED tampered PSD proof (negative control)")

    print("PASS: hodge_hitchin_lean4_roundtrip (ALL 6 GATES VERIFIED)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
