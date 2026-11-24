"""Exercise the opt-in nested-region gate at the native destruction boundary."""

import json
from pathlib import Path
import resource
import signal
import subprocess
import tempfile

from review import (
    HERE,
    ROOT,
    arguments,
    build_model,
    check_native,
    compile_workload,
    link,
    recorded,
    require,
    run,
    sha,
)


def build(work, args, env, observed, gated):
    flags = [
        "c++",
        "-std=c++20",
        "-O1",
        "-DNDEBUG",
        "-I",
        ROOT / "inc",
        "-I",
        ROOT / "lib/core",
    ]
    if gated:
        flags.append("-DESHKOL_VERIFICATUM_NESTED_STACK")
    if observed:
        flags += ["-finstrument-functions", "-fno-inline"]
    run([*flags, "-c", ROOT / "lib/core/runtime_regions.cpp", "-o", "regions.o"], work)
    link(
        work,
        args,
        env,
        "nested",
        [HERE / "gate_probe.cpp", "regions.o"],
        ["abort", "nested_boundary"],
    )


def main():
    args = arguments(__doc__, native_required=True)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    frozen = recorded()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        result = build_model(work, args.lean)
        env = compile_workload(args, work, frozen)
        env.pop("NESTED_GATE_CONTROL", None)
        builds = {}
        for name, observed, gated in (
            ("default", False, False),
            ("instrumented", True, True),
            ("ordinary_gate", False, True),
        ):
            build(work, args, env, observed, gated)
            exe = work / "nested"
            good = subprocess.run(
                [exe], cwd=work, env=env, text=True, capture_output=True, timeout=120
            )
            require(
                good.returncode == 0,
                f"{name} healthy workload failed: {good.stderr[-800:]}",
            )
            actual = check_native(work, args.lean, good.stdout, frozen)
            entries = good.stderr.count("nested destructor entered")
            if observed:
                require(
                    entries == 2, "expected one child destructor entry per workload"
                )
                actual["observed_child_destructor_entries"] = entries
            if gated:
                bad = subprocess.run(
                    [exe],
                    cwd=work,
                    env=dict(env, NESTED_GATE_CONTROL="1"),
                    text=True,
                    capture_output=True,
                    timeout=120,
                )
                require(
                    bad.returncode == -signal.SIGABRT
                    and "restored parent; inactive slot retains live child; destructor entries 0"
                    in bad.stderr
                    and "nested destructor entered" not in bad.stderr,
                    f"{name}: wrong denial boundary",
                )
                actual["denial"] = "SIGABRT at restored parent"
                if observed:
                    actual["denied_child_destructor_entries"] = 0
            actual["elf_sha256"] = sha(exe)
            builds[name] = actual
        result["builds"] = builds
        result["runtime_sha256"] = sha(args.release / "lib/libeshkol-runtime.a")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
