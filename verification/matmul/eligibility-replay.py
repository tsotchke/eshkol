"""Replay use-site eligibility and the guarded native foreign-call witness."""
import argparse
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPECTED = {
    "published": True, "axes-copy-only": True, "other-shape-exposed": True,
    "compatible-header-changed": False, "same-header-rebound": False, "tagged-handoff": False,
    "shared-header-handoff": False, "unknown-footprint": False, "outstanding-writer": False,
    "returned-writer-old-shape": False, "retired": False, "draft-exposed": False,
    "fresh-copy-old-binding": False, "fresh-copy-new-binding": True,
}
THEOREMS = ("invocation_guard", "quarantine_persists", "unknown_revokes", "publish_fresh_snapshot",
            "crossing_respects_lifecycle", "lifecycle_at_key", "quarantine_blocks")
OBSERVER = """#include <stdint.h>
int64_t verificatum_observe_tensor(const void *tensor) { return tensor != 0 ? 73 : 0; }
"""


# Audit the effect of each embedding API in addition to the argument handoff.
# These classifications describe reviewed contracts; lowering does not consume them.
C_API = {
    "context lifetime": "init shutdown",
    "arbitrary evaluation effects": "eval eval_double eval_file",
    "scalar or copied string construction": "int64 double bool null string",
    "reachable tagged values": "cons list car cdr",
    "scalar predicate": "type to_bool is_null is_pair tensor_ndims tensor_size",
    "requires independently known numeric tag": "to_int64 to_double",
    "borrowed string storage": "to_string last_error",
    "output or error state": "display clear_error",
    "fresh axes then tagged return": "tensor_zeros tensor_from_data",
    "borrowed element storage": "tensor_data",
    "caller-owned axis copy": "tensor_shape",
}


def public_surface():
    header = ROOT / "inc/eshkol/eshkol_ffi.h"
    source = ROOT / "lib/ffi/eshkol_ffi.cpp"
    reviewed = {name for names in C_API.values() for name in names.split()}
    # Functions in the current public header have no inline bodies. Strip
    # comments before matching names, then independently check the C definitions.
    clean = re.sub(r"/\*[\s\S]*?\*/|//[^\n]*", "", header.read_text())
    declared = set(re.findall(r"\beshkol_ffi_(\w+)\s*\(", clean))
    implemented = set(re.findall(r'^extern "C"[^\n]*?\beshkol_ffi_(\w+)\s*\(', source.read_text(), re.M))
    if declared != reviewed or implemented != reviewed:
        raise ValueError(f"public C API review required: header delta {declared ^ reviewed}, "
                         f"implementation delta {implemented ^ reviewed}")
    return {kind: names.split() for kind, names in C_API.items()}


def check_foreign_call(ir):
    # Check the observed pointer comes from the tested tensor at the intervening
    # call. The guard inspector separately follows both predicates into the branch.
    calls = re.findall(r"^  %[^ ]+ = call (?:%eshkol_tagged_value|i64) "
                       r"@(audited-product|verificatum_observe_tensor)\(", ir, re.M)
    if calls != ["audited-product", "verificatum_observe_tensor", "audited-product", "audited-product"]:
        raise ValueError("unexpected order of foreign and matmul calls")
    transfer = re.search(r"^  (%[\w.]+) = extractvalue %eshkol_tagged_value (%[\w.]+), 4\n"
                         r"  (%[\w.]+) = inttoptr i64 \1 to ptr\n"
                         r"  %[\w.]+ = call i64 @verificatum_observe_tensor\(ptr \3\)", ir, re.M)
    if not transfer or not re.search(r"^  " + re.escape(transfer[2]) +
                                     r" = load %eshkol_tagged_value, ptr @eshkol_g_right,", ir, re.M):
        raise ValueError("could not bind foreign argument to the right operand")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--compiler", type=Path, required=True, help="native eshkol-run")
    cli.add_argument("--lean", default="lean", help="Lean 4.35.0")
    args = cli.parse_args()
    compiler = args.compiler.resolve(strict=True)
    lean = shutil.which(args.lean)
    if not lean:
        cli.error("Lean executable missing")
    helpers = runpy.run_path(str(HERE / "guard-replay.py"))
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        env = dict(os.environ, LEAN_PATH=tmp)

        def run(*cmd, code=0):
            return helpers["run"](list(cmd), work, env, code).decode()

        surface = public_surface()
        version = run(lean, "--version").strip()
        if "version 4.35.0," not in version:
            raise ValueError(version)
        for module in ("GuardAuthority", "ShapePublication", "Eligibility"):
            shutil.copy(HERE / (module + ".lean"), work)
            if run(lean, "-o", module + ".olean", module + ".lean"):
                raise ValueError("unexpected Lean diagnostic in " + module)
        prefix = "Verificatum.Eligibility."
        audit = "import Eligibility\n" + "".join(f"#print axioms {prefix}{t}\n" for t in THEOREMS)
        audit += ('#eval do for (label, ok) in Verificatum.Eligibility.cases do '
                  'IO.println (label ++ (if (ok : Bool) then " eligible" else " guarded"))\n')
        (work / "Audit.lean").write_text(audit)
        lines = run(lean, "Audit.lean").splitlines()
        axioms = lines[:len(THEOREMS)]
        for name, line in zip(THEOREMS, axioms, strict=True):
            if line not in {f"'{prefix}{name}' depends on axioms: [propext]",
                            f"'{prefix}{name}' does not depend on any axioms"}:
                raise ValueError(line)
        expected = [k + (" eligible" if v else " guarded") for k, v in EXPECTED.items()]
        if lines[len(THEOREMS):] != expected:
            raise ValueError(f"eligibility observations changed: {lines}")

        text = (work / "Eligibility.lean").read_text()
        rule = "| .unknown => revoke (w k)"
        if text.count(rule) != 1:
            raise ValueError("unknown-transfer control site changed")
        mutant = text.replace(rule, "| .unknown => w k")
        (work / "Mutant.lean").write_text(mutant)
        mutation = subprocess.run([lean, "Mutant.lean"], cwd=work, env=env, capture_output=True,
                                  text=True, timeout=120)
        # The theorem which promises default revocation must itself break.
        start = mutant[:mutant.index("theorem unknown_revokes")].count("\n") + 1
        end = mutant[:mutant.index("/-- The caller supplies")].count("\n") + 1
        locations = re.findall(r"Mutant\.lean:(\d+):\d+: error: unsolved goals", mutation.stdout)
        if mutation.returncode != 1 or not any(start <= int(i) < end for i in locations):
            raise ValueError("unknown_revokes did not detect the weakened transfer")

        (work / "observer.c").write_text(OBSERVER)
        run("cc", "-c", "observer.c", "-o", "observer.o")
        run("ar", "rcs", "libobserver.a", "observer.o")
        shutil.copy(HERE / "foreign-invocation.esk", work)
        observed = {}
        for opt in ("0", "2"):
            name = "site-O" + opt
            flags = ["--dump-ir"] if opt == "0" else []
            run(compiler, *flags, "-I", ROOT / "lib", "-L", work, "-l", "observer", "-O", opt,
                "-o", name, "foreign-invocation.esk")
            observed[name] = run(work / name, code=1).splitlines()
            if observed[name] != ["(2 2)", "73", "(2 2)", "Error: matmul inner dimensions mismatch (3 vs 2)"]:
                raise ValueError(f"{name}: {observed[name]}")
        ir = (work / "site-O0.ll").read_text()
        check_foreign_call(ir)
        print(json.dumps({"lean": version, "axioms": axioms, "eligibility": expected, "public_c_api": surface,
              "weakened_unknown_transfer": "rejected by unknown_revokes", "native": observed,
              "emission": helpers["inspect_site"](ir), "compiler_sha256": helpers["sha"](compiler.read_bytes()),
              "permit_consumed": False, "production_exposure_tracking": False}, indent=2))


if __name__ == "__main__":
    main()
