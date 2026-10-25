#!/usr/bin/env python3
"""Check native cross-entropy tape connectivity and explicit loss AD refusals."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SUCCESS = "cross_entropy_gradient_test: ALL PASS"
REFUSALS = {
    "mse": (
        '(gradient (lambda (w) (mse-loss w #(0.0 1.0))) #(1.0 2.0))',
        "mse-loss: reverse-mode differentiation is not implemented",
    ),
    "bce_indirect": (
        '(define (loss w) (binary-cross-entropy-loss w #(1.0 0.0)))\n'
        '(define (run f x) (gradient f x))\n(run loss #(0.7 0.3))',
        "binary-cross-entropy-loss: reverse-mode differentiation is not implemented",
    ),
    "focal": (
        '(gradient (lambda (w) (focal-loss w #(1.0 0.0))) #(0.2 0.3))',
        "focal-loss: reverse-mode differentiation is not implemented",
    ),
    "probability_labels": (
        '(gradient (lambda (y) (cross-entropy-loss #(0.2 0.3) y)) #(0.25 0.75))',
        "cross-entropy-loss: differentiation of targets is not supported",
    ),
    "indexed_labels": (
        '(gradient (lambda (y) (cross-entropy-loss #(0.2 0.3) y)) #(0.0))',
        "cross-entropy-loss: differentiation of targets is not supported",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("jit", "aot"), required=True)
    parser.add_argument("--dense", choices=("0", "1"), required=True)
    args = parser.parse_args()
    compiler, build = args.compiler.resolve(), args.build_dir.resolve()
    env = dict(os.environ, ESHKOL_PATH=str(ROOT / "lib"), ESHKOL_JIT_CACHE="0",
               ESHKOL_DENSE_TENSOR_AD_NODES=args.dense)
    env["PATH"] = str(build) + os.pathsep + env.get("PATH", "")
    failures = []

    def invoke(command):
        result = subprocess.run(command, cwd=build, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=180, encoding="utf-8", errors="replace")
        return result.returncode, result.stdout

    with tempfile.TemporaryDirectory(prefix="ce-ad-", dir=build) as tmp:
        scratch = Path(tmp)
        # Cache isolation matters: the lowering selection is made at codegen.
        env["ESHKOL_JIT_CACHE_DIR"] = str(scratch / "cache")

        def execute(source: Path):
            if args.mode == "jit":
                return invoke([str(compiler), "-r", str(source), "-L" + str(build)])
            binary = scratch / (source.stem + (".exe" if os.name == "nt" else ""))
            code, output = invoke([str(compiler), str(source), "-o", str(binary),
                                   "-L" + str(build)])
            if code != 0 or not binary.is_file():
                raise RuntimeError(f"compilation failed for {source.name}: {code}\n{output}")
            return invoke([str(binary)])

        cases = [
            (ROOT / "tests/ad/cross_entropy_gradient_test.esk", SUCCESS),
            (ROOT / "tests/ml/loss_functions_test.esk", "PASS: smooth L1 with large diff"),
            (ROOT / "tests/vm_parity/corpus/73_cross_entropy_targets.esk", "PASS: indexed targets"),
        ]
        for source, marker in cases:
            code, output = execute(source)
            if code != 0 or marker not in output or "FAIL" in output:
                failures.append(f"{source.name}: exit={code}\n{output}")
            else:
                print(f"PASS {args.mode} dense={args.dense}: {source.name}")

        for name, (program, message) in REFUSALS.items():
            source = scratch / (name + ".esk")
            source.write_text(program + '\n(display "UNEXPECTED SUCCESS")\n', encoding="utf-8")
            code, output = execute(source)
            # A crash, verifier failure or compile failure is not a refusal.
            if code != 1 or message not in output or "UNEXPECTED SUCCESS" in output:
                failures.append(f"refusal {name}: exit={code}\n{output}")
            else:
                print(f"PASS {args.mode} dense={args.dense}: refuses {name}")

    for failure in failures:
        print(f"FAIL: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
