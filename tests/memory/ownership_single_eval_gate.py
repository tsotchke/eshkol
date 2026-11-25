#!/usr/bin/env python3
"""Run ownership single-evaluation assertions on native and VM routes."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


EXPECTED = "RESULT: PASS 39 ownership single-evaluation assertions"
PASS_COUNT = 39


def invoke(label, command, cwd, evidence_dir):
    env = dict(os.environ)
    env["ESHKOL_VM_NO_DISASM"] = "1"
    env["ESHKOL_JIT_CACHE"] = "0"
    try:
        result = subprocess.run(
            [str(part) for part in command], cwd=cwd, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240, env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{label}: timed out; refusing incomplete evidence") from exc
    output = result.stdout or ""
    record = {"route": label, "command": [str(part) for part in command],
              "cwd": str(cwd), "returncode": result.returncode, "output": output}
    (evidence_dir / (label.lower().replace(" ", "-") + ".json")).write_text(
        json.dumps(record, indent=2) + "\n"
    )
    if result.returncode < 0:
        raise RuntimeError(f"{label}: process crashed with signal {-result.returncode}\n{output[-4000:]}")
    if result.returncode != 0:
        raise RuntimeError(f"{label}: exit {result.returncode}\n{output[-4000:]}")
    if "Segmentation fault" in output or "LLVM module verification failed" in output:
        raise RuntimeError(f"{label}: fatal compiler/runtime diagnostic\n{output[-4000:]}")
    if output.count("PASS:") != PASS_COUNT or EXPECTED not in output or "FAIL:" in output:
        raise RuntimeError(
            f"{label}: expected exactly {PASS_COUNT} assertions and final marker; "
            f"saw {output.count('PASS:')} PASS lines\n{output[-4000:]}"
        )
    print(f"PASS: {label} ({PASS_COUNT} assertions)")


def compile_aot(native, source, output, cwd, evidence_dir):
    command = [str(native), "-o", str(output), str(source), "-L", str(native.parent)]
    result = subprocess.run(
        command, cwd=cwd,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240,
    )
    (evidence_dir / "aot-compile.json").write_text(json.dumps({
        "route": "AOT compile", "command": command, "cwd": str(cwd),
        "returncode": result.returncode, "output": result.stdout or "",
    }, indent=2) + "\n")
    if result.returncode < 0:
        raise RuntimeError(f"AOT compile crashed with signal {-result.returncode}\n{result.stdout[-4000:]}")
    if result.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"AOT compile failed to produce executable (exit {result.returncode})\n{result.stdout[-4000:]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--vm", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    args = parser.parse_args()

    native = args.native.resolve()
    vm = args.vm.resolve()
    source = args.source.resolve()
    source_root = source.parents[2]
    evidence_dir = args.evidence_dir.resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    if not native.is_file() or not vm.is_file() or not source.is_file():
        raise RuntimeError("refusing to count a route whose executable or fixture is missing")

    with tempfile.TemporaryDirectory(prefix="ownership-single-eval-") as temp:
        work = Path(temp)
        invoke("native JIT", [native, "-r", source, "-L", native.parent], work, evidence_dir)

        aot = work / "ownership_single_eval_aot"
        compile_aot(native, source, aot, work, evidence_dir)
        if not aot.stat().st_mode & 0o111:
            raise RuntimeError("AOT compile produced a non-executable artifact")
        invoke("AOT executable", [aot], work, evidence_dir)

        invoke("VM source", [vm, source], source_root, evidence_dir)

        eskb = work / "ownership_single_eval.eskb"
        compiled = subprocess.run(
            [str(native), "--profile", "hosted-vm", "--emit-eskb", str(eskb),
             str(source), "-L", str(native.parent)],
            cwd=work, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=240, env={**os.environ, "ESHKOL_JIT_CACHE": "0"},
        )
        (evidence_dir / "vm-eskb-compile.json").write_text(json.dumps({
            "route": "VM ESKB compile", "command": [str(native), "--profile", "hosted-vm",
            "--emit-eskb", str(eskb), str(source), "-L", str(native.parent)],
            "cwd": str(work), "returncode": compiled.returncode,
            "output": compiled.stdout or "",
        }, indent=2) + "\n")
        if compiled.returncode < 0:
            raise RuntimeError(f"VM ESKB compile crashed with signal {-compiled.returncode}\n{compiled.stdout[-4000:]}")
        if compiled.returncode != 0:
            raise RuntimeError(f"VM ESKB compile failed (exit {compiled.returncode})\n{compiled.stdout[-4000:]}")
        if not eskb.is_file() or eskb.stat().st_size == 0:
            raise RuntimeError("VM ESKB compile returned success without a nonempty artifact")
        invoke("VM ESKB execution", [vm, eskb], work, evidence_dir)

    print("PASS: ownership single-evaluation gate (4 executions, 156 assertions)")


if __name__ == "__main__":
    main()
