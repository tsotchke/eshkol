"""Check the nested-region model and optionally replay the two native BLC workloads."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from native_observations import Row, check, require


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIXTURE = HERE / "fixture"
MODELS = (
    "ArenaBump",
    "ArenaLedger",
    "NativeAllocation",
    "Reclamation",
    "NestedRegions",
)
EXPECTED = ["true", "false", "false", "false"] * 2


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recorded():
    path = HERE / "observations.json"
    require(
        sha(path) == "5c3f1990a539d6c0e047764451400ca621a4e1cab43423ea9640aa467078df0a",
        "unexpected historical receipt",
    )
    receipt = json.loads(path.read_bytes())
    for name in (
        *[n + ".lean" for n in MODELS],
        "nested_regions.esk",
        "nested_regions.cpp",
    ):
        folder = HERE / "proof" if name.endswith(".lean") else FIXTURE
        require(
            sha(folder / name) == receipt["sources"][name], f"unexpected source: {name}"
        )
    return receipt


def run(cmd, work, env=None):
    proc = subprocess.run(
        cmd, cwd=work, env=env, text=True, capture_output=True, timeout=180
    )
    if proc.returncode:
        raise RuntimeError(
            f"{cmd[0]} exited {proc.returncode}\n{proc.stdout[-1200:]}{proc.stderr[-2000:]}"
        )
    return proc.stdout


def arguments(description, native_required=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--lean", default="lean", help="Lean 4.35.0")
    parser.add_argument(
        "--release",
        type=Path,
        required=native_required,
        help="unpacked v1.3.6-evolve Linux x64 lite release",
    )
    parser.add_argument(
        "--cblas", type=Path, help="CBLAS shared library on split-BLAS hosts"
    )
    args = parser.parse_args()
    binary = shutil.which(args.lean)
    if not binary:
        parser.error("Lean 4.35.0 is required")
    args.lean = Path(binary).resolve(strict=True)
    if args.release:
        args.release = args.release.resolve(strict=True)
    if args.cblas:
        if not args.release:
            parser.error("--cblas requires --release")
        args.cblas = args.cblas.resolve(strict=True)
    return args


def build_model(work, lean):
    env = dict(os.environ, LEAN_PATH=str(work))
    version = run([lean, "--version"], work).strip()
    require(version.startswith("Lean (version 4.35.0,"), "expected Lean 4.35.0")
    for name in MODELS:
        filename = name + ".lean"
        shutil.copyfile(HERE / "proof" / filename, work / filename)
        run([lean, "-DwarningAsError=true", "-o", name + ".olean", filename], work, env)
    (work / "Axioms.lean").write_text(
        "import NestedRegions\n#print axioms Verificatum.NestedRegions.pop_layout\n"
        "#print axioms Verificatum.NestedRegions.pop_child\n"
    )
    axioms = run([lean, "-DwarningAsError=true", "Axioms.lean"], work, env).splitlines()
    require(
        len(axioms) == 2 and not any("sorryAx" in line for line in axioms),
        "unexpected axiom report",
    )
    return {"lean": version, "axioms": axioms}


def audit(work, lean, cases):
    (work / "observed.cases").write_text("\n".join(cases) + "\n")
    decisions = run(
        [lean, "--run", "NestedRegions.lean", "observed.cases"],
        work,
        dict(os.environ, LEAN_PATH=str(work)),
    ).splitlines()
    require(decisions == EXPECTED, f"unexpected binding decisions: {decisions}")
    return decisions


def compile_workload(args, work, frozen):
    release = args.release
    runner = release / "bin/eshkol-run"
    deps = {
        "compiler": runner,
        "stdlib.o": release / "lib/stdlib.o",
        "runtime": release / "lib/libeshkol-runtime.a",
        "stdlib.esk": release / "share/eshkol/stdlib.esk",
        "blc.esk": ROOT / "lib/core/blc.esk",
        "blc-fast.esk": FIXTURE / "core/blc-fast.esk",
        "arena_memory.h": ROOT / "lib/core/arena_memory.h",
        "eshkol.h": ROOT / "inc/eshkol/eshkol.h",
        "object_limits.h": ROOT / "inc/eshkol/core/object_limits.h",
    }
    for name, file in deps.items():
        require(sha(file) == frozen["inputs"][name], f"unexpected build input: {name}")
    modules = work / "modules/core"
    modules.mkdir(parents=True)
    for name in ("blc.esk", "blc-fast.esk"):
        shutil.copyfile(deps[name], modules / name)
    env = dict(
        os.environ,
        ESHKOL_LIB_DIR=str(release / "lib"),
        ESHKOL_STDLIB_PATH=str(deps["stdlib.esk"]),
    )
    if args.cblas:
        env["LD_PRELOAD"] = str(args.cblas)
    run(
        [
            runner,
            "-I",
            modules.parent,
            "-O",
            "2",
            "-c",
            "-o",
            "nested.o",
            FIXTURE / "nested_regions.esk",
        ],
        work,
        env,
    )
    return env


def link(work, args, env, output, extra=(), wrappers=()):
    cmd = [
        "c++",
        "-std=c++20",
        "-O1",
        "-I",
        ROOT,
        "-I",
        ROOT / "inc",
        FIXTURE / "nested_regions.cpp",
        *extra,
        "nested.o",
        args.release / "lib/stdlib.o",
        args.release / "lib/libeshkol-runtime.a",
        "-Wl,--wrap=arena_destroy",
        *["-Wl,--wrap=" + n for n in wrappers],
        "-Wl,--export-dynamic",
        "-Wl,-z,stack-size=536870912",
        "-lpng",
        "-ljpeg",
        "-lwebp",
    ]
    if args.cblas:
        cmd.append(args.cblas)
    run([*cmd, "-lblas", "-lcrypto", "-lm", "-ldl", "-o", output], work, env)


def check_native(work, lean, output, frozen):
    rows = []
    forms = []
    for line in output.splitlines():
        if line.startswith("REGION "):
            fields = list(map(int, line.split()[1:]))
            require(len(fields) == 73, "incomplete physical stack")
            rows.append(Row(*fields[:9], tuple(fields[9:])))
        else:
            forms.append(line)
    require(
        forms == frozen["normal_forms"] and len(rows) == 20, "incomplete BLC workload"
    )
    cases = []
    workloads = [check(rows[i : i + 10], cases) for i in (0, 10)]
    require(workloads == frozen["workloads"], "native allocation observations differ")
    raw = [" ".join(map(str, case)) for case in cases]
    decisions = audit(work, lean, raw)
    return {
        "normal_forms": forms,
        "workloads": workloads,
        "live_binding_decisions": decisions,
    }


def main():
    args = arguments(__doc__)
    frozen = recorded()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        receipt = build_model(work, args.lean)
        receipt["recorded_binding_decisions"] = audit(
            work, args.lean, frozen["binding_cases"]
        )
        if args.release:
            env = compile_workload(args, work, frozen)
            link(work, args, env, "nested")
            receipt["native"] = check_native(
                work, args.lean, run([work / "nested"], work, env), frozen
            )
            receipt["native"]["elf_sha256"] = sha(work / "nested")
            receipt["native"]["runtime_sha256"] = sha(
                args.release / "lib/libeshkol-runtime.a"
            )
        print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
