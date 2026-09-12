#!/usr/bin/env python3
"""Strict tensor-apply callable/AD matrix: LLVM JIT, AOT, VM and ESKB."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    build = args.build.resolve()
    native = build / "eshkol-run"
    vm = build / "eshkol-vm-standalone-test"
    for executable in (native, vm):
        if not executable.is_file():
            raise SystemExit(f"Required engine missing: {executable}")
    source = root / "tests/collections/tensor_apply_callable_test.esk"
    expected = re.findall(r'\(check "([^"]+)"', source.read_text())
    env = dict(os.environ, ESHKOL_JIT_CACHE="0")
    env["ESHKOL_AOT_MODULE_CACHE_DIR"] = str(build / "callable-module-cache")

    def run(label, command, verify=False):
        result = subprocess.run([str(x) for x in command], cwd=root, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=240)
        log = build / f"tensor-apply-{label}.log"
        log.write_text(result.stdout)
        observed = re.findall(r"^PASS: (.+)$", result.stdout, re.MULTILINE)
        if result.returncode or (verify and (observed != expected or
                                            "RESULT: PASS" not in result.stdout)):
            raise SystemExit(f"FAIL {label} (exit {result.returncode}); see {log}")
        print(f"PASS {label}" + (f" ({len(observed)} assertions)" if verify else ""), flush=True)

    common = ["-n", "-L", build, "-I", root / "lib"]
    with tempfile.TemporaryDirectory(prefix="tensor-apply-", dir=build) as directory:
        artifact = Path(directory)
        run("jit", [native, *common, "-O0", "-r", source], True)
        run("aot-compile", [native, *common, "-O2", source, "-o", artifact / "native"])
        run("aot", [artifact / "native"], True)
        run("vm", [vm, source], True)
        run("eskb-compile", [native, *common, "--profile", "hosted-vm",
                             "--emit-eskb", artifact / "matrix.eskb", source])
        run("eskb", [vm, artifact / "matrix.eskb"], True)


if __name__ == "__main__":
    main()
