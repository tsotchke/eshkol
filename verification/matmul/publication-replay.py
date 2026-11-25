"""Check the rank-two publication model and compare it with native region promotion."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = HERE / "ShapePublication.lean"
PROBE = HERE / "axis-evacuation.cpp"
EVAC = ROOT / "lib/core/runtime_regions.cpp"
THEOREMS = ("published_history", "bound_contents_at_use", "inner_guard_at_use",
            "exposed_refuses_all_headers")


def digest(path):
    with path.open("rb") as source:
        result = hashlib.sha256()
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
        return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path, help="native libeshkol-runtime.a")
    parser.add_argument("--lean", default="lean", help="Lean 4.35.0 executable")
    args = parser.parse_args()
    runtime = args.runtime.resolve(strict=True)
    lean = shutil.which(args.lean)
    if not lean:
        parser.error("Lean executable missing")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        env = dict(os.environ, LEAN_PATH=tmp)

        def run(*cmd):
            result = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, timeout=120)
            if result.returncode or result.stderr:
                raise RuntimeError(f"{cmd[0]}: {result.stdout}{result.stderr}")
            return result.stdout

        version = run(lean, "--version").strip()
        if "version 4.35.0," not in version:
            raise ValueError(version)
        shutil.copyfile(MODEL, work / MODEL.name)
        if run(lean, "-o", "ShapePublication.olean", MODEL.name):
            raise ValueError("unexpected diagnostic while checking the publication model")
        expected = run(lean, "--run", MODEL.name).splitlines()
        controls = {f"{route} {kind}" for route in ("batch", "separate")
                    for kind in ("shared-young", "shared-old", "distinct-young")}
        if len(expected) != 6 or {" ".join(row.split()[:2]) for row in expected} != controls:
            raise ValueError(f"incomplete model corpus: {expected}")
        prefix = "Verificatum.ShapePublication."
        (work / "Audit.lean").write_text("import ShapePublication\n" +
                "".join(f"#print axioms {prefix}{t}\n" for t in THEOREMS))
        axioms = run(lean, "Audit.lean").splitlines()
        for theorem, line in zip(THEOREMS, axioms, strict=True):
            allowed = ("does not depend on any axioms", "depends on axioms: [propext]")
            if line not in {f"'{prefix}{theorem}' {a}" for a in allowed}:
                raise ValueError(line)

        rule = "| relocate : Change (.published p a) (.published q a)"
        text = MODEL.read_text()
        if text.count(rule) != 1:
            raise ValueError("relocation negative control no longer identifies one rule")
        (work / "Mutant.lean").write_text(text.replace(rule, rule.replace("q a", "q b")))
        mutant = subprocess.run([lean, "Mutant.lean"], cwd=work, env=env,
                                capture_output=True, text=True, timeout=120)
        if mutant.returncode != 1 or "error: unsolved goals\ncase relocate\n" not in mutant.stdout:
            raise ValueError("weakened relocation did not fail its preservation proof")

        run("c++", "-std=c++17", "-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections",
            "-I", ROOT / "inc", "-I", ROOT / "lib/core", PROBE, EVAC, runtime,
            "-ldl", "-pthread", "-lm", "-o", "axis-evacuation")
        observed = run(work / "axis-evacuation").splitlines()
        if observed != expected:
            raise ValueError(f"model: {expected}\nnative: {observed}")

        print(json.dumps({"observations": observed, "columns": ["route", "input_ownership",
              "destination_axes_shared", "left_axes_same_address", "left_rows", "left_cols",
              "right_rows", "right_cols"], "lean": version, "axioms": axioms,
              "relocation_mutation": "preservation proof failed at relocate",
              "sha256": {p.name: digest(p) for p in (MODEL, PROBE, EVAC, runtime)},
              "native_publication_enforced": False, "guard_elision_authorized": False}, indent=2))


if __name__ == "__main__":
    main()
