"""Replay the frozen mt pair. Requires the official Lean 4.35.0 tools on PATH."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

HERE = Path(__file__).resolve().parent
RECEIPT_SHA256 = "8d3017950bd8f8430da6db33fe9b192b78af08d7467195cf00f3cf7758b7bfda"
DIGESTS = {
    "mt.ndjson": "d4cb6b03bc24222a3c3c1608158da7aedbc466602d63bfee40ac31a41eb7d664",
    "mt-wrong-body.ndjson": "f834bcebd6b66b0d4a6350b99b3d446dc619e154c0951bec44104462b1dd5b5d",
}


def run(args, work, env):
    return subprocess.run(args, cwd=work, env=env, capture_output=True, timeout=60)


def verify(result, code=0, diagnostic=None):
    output = (result.stdout + result.stderr).decode(errors="replace")
    if result.returncode != code or (diagnostic and diagnostic not in output):
        raise ValueError(f"{result.args} exited {result.returncode}: {output[:2000]}")


def replay(candidate):
    receipt_bytes = (HERE / "R0.toml").read_bytes()
    if hashlib.sha256(receipt_bytes).hexdigest() != RECEIPT_SHA256:
        raise ValueError("Unrecognized R0 receipt or premises")
    receipt = tomllib.loads(receipt_bytes.decode())
    for name, digest in receipt["artifacts"].items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"R0 implementation or evidence hash mismatch: {name}")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        env = dict(os.environ, LEAN_PATH=str(work))
        for name, digest in DIGESTS.items():
            data = (HERE / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"Golden SHA-256 mismatch for {name}")
            (work / name).write_bytes(data)

        # Bind the original bytes and the separately authored proposition to the
        # same Lean environment. The target export is generated only in scratch.
        emitted = run(["leanexport", "Init.Core", "--", "mt"], work, env)
        verify(emitted)
        if emitted.stdout != (work / "mt.ndjson").read_bytes():
            raise ValueError("Reference export differs from the frozen original")
        shutil.copyfile(HERE / "Target.lean", work / "Target.lean")
        verify(run(["lean", "-o", "Target.olean", "Target.lean"], work, env))
        for target in ("target", "other"):
            result = run(["leanexport", "Target", "--", f"Verificatum.{target}"], work, env)
            verify(result)
            (work / f"{target}.ndjson").write_bytes(result.stdout)

        verify(run(["con-ron", "mt.ndjson"], work, env), 0, "accepted 3 declarations")
        verify(run(["con-ron", "mt-wrong-body.ndjson"], work, env), 1,
               "application type mismatch [at theorem mt")
        print("Reference PASS: original bytes reproduced, independent target type-checked,"
              " con-ron accepted mt and rejected its wrong body.")

        for file, names in [
            ("Typing.lean", ["Verificatum.Typing.check_sound", "Verificatum.Typing.target_binding_sound"]),
            ("Binders.lean", ["Verificatum.Binders.erase_beta"]),
        ]:
            audit = (HERE / file).read_text() + "".join(f"\n#print axioms {name}" for name in names)
            (work / "Audit.lean").write_text(audit + "\n")
            audit_run = run(["lean", "Audit.lean"], work, env)
            verify(audit_run)
            expected = [f"'{name}' depends on axioms: [{', '.join(receipt['proof_axioms'][name])}]"
                        for name in names]
            if audit_run.stdout.decode().splitlines() != expected:
                raise ValueError(f"Unexpected proof axiom audit: {audit_run.stdout!r}")
        corpus = run(["lean", "--run", str(HERE / "Binders.lean")], work, env)
        verify(corpus)
        if (hashlib.sha256(corpus.stdout).hexdigest() != receipt["binder_corpus"]["sha256"] or
                len(corpus.stdout.splitlines()) != receipt["binder_corpus"]["cases"]):
            raise ValueError("Binder differential corpus changed")
        compared = subprocess.run(["node", str(HERE / "binder-replay.cjs")],
                                  input=corpus.stdout, cwd=work, env=env,
                                  capture_output=True, timeout=60)
        verify(compared)
        print(compared.stdout.decode().strip())
        # Give the advertised recursor a different structured name while keeping
        # its type and the theorem intact. R0 admits the canonical False package.
        original = (work / "mt.ndjson").read_bytes()
        rec_name = b'{"in":2,"str":{"pre":1,"str":"rec"}}'
        if original.count(rec_name) != 1:
            raise ValueError("Expected exactly one False.rec name definition")
        alternate = rec_name.replace(b'"rec"', b'"unexpectedRecursor"')
        (work / "mt-other-recursor.ndjson").write_bytes(original.replace(rec_name, alternate))

        # Change Not P to P → P and the proof body to the bound argument.
        # This is a valid Lean theorem still named mt, with a different meaning.
        def changed_meaning():
            rows = [json.loads(line) for line in original.splitlines()]
            for row in rows:
                if row.get("ie") == 10:
                    row["forallE"]["body"] = 4
                if row.get("ie") == 24:
                    row["lam"]["body"] = 5
            return "".join(json.dumps(row) + "\n" for row in rows).encode()

        (work / "changed.ndjson").write_bytes(changed_meaning())
        verify(run(["con-ron", "changed.ndjson"], work, env), 0, "accepted 3 declarations")
        # Identical decoded names despite different bytes, including a
        # supplementary character exercised in both UTF-8 and surrogate form.
        def named_variant(component, alias_record):
            source = original.replace(b'"str":"Not"', b'"str":' + component)
            ref = b'{"const":{"name":6,"us":[]},"ie":13}'
            return source.replace(ref, alias_record + b"\n" + ref.replace(b'"name":6', b'"name":17'))

        for label, symbol, escape in [("alpha", "α", br"\u03b1"), ("astral", "𝛼", br"\ud835\udefc")]:
            alias = b'{"in":17,"str":{"pre":0,"str":"' + escape + b'"}}'
            (work / (label + ".ndjson")).write_bytes(named_variant(
                json.dumps(symbol, ensure_ascii=False).encode(), alias))
        (work / "component.ndjson").write_bytes(named_variant(b'"42"', b'{"in":17,"num":{"pre":0,"i":42}}'))
        (work / "duplicate.ndjson").write_bytes(original.replace(b'"in":1,', b'"in":1,"in":1,', 1))
        (work / "duplicate-index.ndjson").write_bytes(original + original.splitlines(keepends=True)[1])
        (work / "bad-unicode.ndjson").write_bytes(original.replace('"h₁"'.encode(), br'"\ud800"'))
        (work / "unknown.ndjson").write_bytes(original.replace(b'"format":{"version":"3.1.0"',
                                                                b'"format":{"version":"9.9.9"'))
        (work / "over-budget.ndjson").write_bytes(original + b" " * 32768)
        # A checked environment must account for unused declarations as well.
        missing_rec = list(map(json.loads, original.splitlines()))
        next(r["inductive"] for r in missing_rec if "inductive" in r)["recs"] = []
        (work / "missing-rec.ndjson").write_text("".join(json.dumps(r) + "\n" for r in missing_rec))
        extra_name = b'{"in":17,"str":{"pre":0,"str":"extra"}}\n'
        axiom = b'{"axiom":{"name":17,"levelParams":[],"isUnsafe":false,"type":20}}\n'
        (work / "axiom.ndjson").write_bytes(original + extra_name + axiom)
        opaque = {"name":17, "levelParams":[], "isUnsafe":False, "all":[17], "type":20, "value":28}
        (work / "opaque.ndjson").write_bytes(original + extra_name + json.dumps({"opaque":opaque}).encode() + b"\n")
        # Use the actual CLI with only Node on PATH. The candidate operates on
        # serialized inputs; reference tools are only used in the phase above.
        node = shutil.which("node")
        if not node:
            raise ValueError("Node is required to run the bundled Eshkol VM")
        tools = work / "candidate-bin"
        tools.mkdir()
        (tools / "node").symlink_to(node)
        candidate_env = {"PATH": str(tools)}
        observed = {}
        for proof, target, status, reason in (
            ("mt.ndjson", "target", "ACCEPT", "checked"),
            ("mt-wrong-body.ndjson", "target", "REJECT", "application-domain"),
            ("mt.ndjson", "other", "REJECT", "target"),
            ("changed.ndjson", "target", "REJECT", "target"),
            ("missing-rec.ndjson", "target", "REJECT", "declaration"),
            ("axiom.ndjson", "target", "REJECT", "axiom-policy"),
            ("opaque.ndjson", "target", "UNSUPPORTED", "declaration"),
            ("mt-other-recursor.ndjson", "target", "REJECT", "declaration"),
            ("alpha.ndjson", "target", "ACCEPT", "checked"),
            ("astral.ndjson", "target", "ACCEPT", "checked"),
            ("component.ndjson", "target", "REJECT", "undeclared-constant"),
            ("duplicate.ndjson", "target", "ERROR", "duplicate-json-key"),
            ("duplicate-index.ndjson", "target", "ERROR", "duplicate-index"),
            ("bad-unicode.ndjson", "target", "ERROR", "surrogate"),
            ("unknown.ndjson", "target", "UNSUPPORTED", "format-version"),
            ("over-budget.ndjson", "target", "ERROR", "byte-budget"),
        ):
            result = run([candidate, proof, "mt", f"{target}.ndjson",
                          f"Verificatum.{target}"], work, candidate_env)
            verify(result, {"ACCEPT": 0, "REJECT": 1, "UNSUPPORTED": 2, "ERROR": 3}[status])
            verdict = json.loads(result.stdout)
            if (verdict["status"], verdict["reason"], verdict["assumptions"]) != (status, reason, []):
                raise ValueError(f"{proof} against {target}: unexpected verdict {verdict}")
            if status == "ACCEPT" and verdict["closure"] != [
                {"kind": "ind", "name": ["False"]}, {"kind": "rec", "name": ["False", "rec"]},
                {"kind": "def", "name": [{"alpha.ndjson": "α", "astral.ndjson": "𝛼"}.get(proof, "Not")]}, {"kind": "thm", "name": ["mt"]},
            ]:
                raise ValueError(f"{proof}: unexpected declaration closure {verdict}")
            observed[f"{proof}/{target}"] = f"{status}/{reason}"
            print(f"Eshkol {proof} / {target}: {status} ({reason})")
        if observed != receipt["required_verdicts"]:
            raise ValueError("Required verdict set differs from the R0 receipt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, help="path to a built standalone checker")
    args = parser.parse_args()
    candidate = args.candidate
    if candidate is None:
        candidate = HERE / "check"
    try:
        replay(candidate.resolve())
    except (KeyError, OSError, ValueError, subprocess.TimeoutExpired) as error:
        sys.exit(str(error))
