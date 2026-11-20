#!/usr/bin/env python3
"""Check native cross-entropy tape connectivity and explicit loss AD refusals."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SUCCESS = "cross_entropy_gradient_test: ALL PASS"
LOSS_EXPRESSIONS = {
    "mse-loss": '(mse-loss w #(1.0 0.0))',
    "mae-loss": '(mae-loss w #(1.0 0.0))',
    "bce-loss": '(bce-loss w #(1.0 0.0))',
    "binary-cross-entropy-loss": '(binary-cross-entropy-loss w #(1.0 0.0))',
    "huber-loss": '(huber-loss w #(1.0 0.0) 1.0)',
    "kl-div-loss": '(kl-div-loss w #(0.4 0.6))',
    "hinge-loss": '(hinge-loss w #(1.0 -1.0))',
    "smooth-l1-loss": '(smooth-l1-loss w #(1.0 0.0))',
    "focal-loss": '(focal-loss w #(1.0 0.0) 2.0)',
    "triplet-loss": '(triplet-loss w #(1.0 0.0) #(0.0 1.0) 1.0)',
    "contrastive-loss": '(contrastive-loss w #(1.0 0.0) #(1.0 1.0) 1.0)',
    "label-smoothing-loss": '(label-smoothing-loss w #(1.0 0.0) 2 0.1)',
    "cosine-embedding-loss": '(cosine-embedding-loss w #(1.0 0.0) 1.0 0.0)',
}
REFUSALS = {
    name: (f'(gradient (lambda (w) {expression}) #(0.7 0.3))',
           name + ': reverse-mode differentiation is not implemented')
    for name, expression in LOSS_EXPRESSIONS.items()
}
REFUSALS.update({
    "bce_indirect": (
        '(define (loss w) (binary-cross-entropy-loss w #(1.0 0.0)))\n'
        '(define (run f x) (gradient f x))\n(run loss #(0.7 0.3))',
        "binary-cross-entropy-loss: reverse-mode differentiation is not implemented",
    ),
    "probability_labels": (
        '(gradient (lambda (y) (cross-entropy-loss #(0.2 0.3) y)) #(0.25 0.75))',
        "cross-entropy-loss: differentiation of targets is not supported",
    ),
    "indexed_labels": (
        '(gradient (lambda (y) (cross-entropy-loss #(0.2 0.3) y)) #(0.0))',
        "cross-entropy-loss: differentiation of targets is not supported",
    ),
})


def positive_ok(code: int, output: str, marker: str) -> bool:
    errors = re.search(r"(^|\n)\s*(?:ERROR\b|\[ERROR\]|FAIL\b|\[FAIL\])|"
                       r"LLVM module verification failed|fatal signal|Unhandled exception|"
                       r"\b[1-9][0-9]* (?:failures|failed)\b",
                       output, re.IGNORECASE)
    return code == 0 and marker in output and errors is None


def refusal_ok(code: int, output: str, message: str) -> bool:
    return (code == 1 and message in output and "UNEXPECTED SUCCESS" not in output
            and "Unhandled Windows exception" not in output
            and "LLVM module verification failed" not in output)


def self_test() -> None:
    assert positive_ok(0, SUCCESS + "\n", SUCCESS)
    for code, output in [(0, ""), (0, "PASS: another case"), (7, SUCCESS),
                         (0, "FAIL: W1 gradient vs FD actual=0 expected=0.013\n" + SUCCESS),
                         (0, "40 checks, 8 failures\n" + SUCCESS),
                         (0, "ERROR: broken\n" + SUCCESS),
                         (0, SUCCESS + "\n[ERROR] stderr diagnostic"),
                         (0, "LLVM module verification failed\n" + SUCCESS)]:
        if positive_ok(code, output, SUCCESS):
            raise RuntimeError(f"positive validator accepted deliberate failure: {code}, {output!r}")
    message = REFUSALS["mse-loss"][1]
    assert refusal_ok(1, "Unhandled exception: " + message, message)
    for code, output in [(0, message), (1, ""), (-11, message),
                         (1, message + "\nUNEXPECTED SUCCESS"),
                         (1, message + "\nUnhandled Windows exception")]:
        if refusal_ok(code, output, message):
            raise RuntimeError(f"refusal validator accepted deliberate failure: {code}, {output!r}")
    print("cross_entropy_gradient harness self-test: PASS", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--mode", choices=("jit", "aot"))
    parser.add_argument("--dense", choices=("0", "1"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        return 0
    if not all((args.compiler, args.build_dir, args.mode, args.dense)):
        parser.error("--compiler, --build-dir, --mode and --dense are required")
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

        def execute(source: Path, no_stdlib: bool = True):
            options = ["--no-stdlib"] if no_stdlib else []
            if args.mode == "jit":
                return invoke([str(compiler), *options, "-r", str(source), "-L" + str(build)])
            binary = scratch / (source.stem + (".exe" if os.name == "nt" else ""))
            code, output = invoke([str(compiler), *options, str(source), "-o", str(binary),
                                   "-L" + str(build)])
            if code != 0 or not binary.is_file():
                raise RuntimeError(f"compilation failed for {source.name}: {code}\n{output}")
            return invoke([str(binary)])

        cases = [
            (ROOT / "tests/ad/cross_entropy_gradient_test.esk", SUCCESS, True),
            (ROOT / "tests/ml/loss_functions_test.esk", "PASS: smooth L1 with large diff", False),
            (ROOT / "tests/vm_parity/corpus/73_cross_entropy_targets.esk", "PASS: indexed targets", True),
        ]
        for source, marker, no_stdlib in cases:
            code, output = execute(source, no_stdlib)
            if not positive_ok(code, output, marker):
                failures.append(f"{source.name}: exit={code}\n{output}")
            else:
                print(f"PASS {args.mode} dense={args.dense}: {source.name}", flush=True)

        # Prove the numerical fixture itself rejects the original wrong-zero
        # failure, rather than only testing the output validator's strings.
        mutant = (ROOT / "tests/ad/cross_entropy_gradient_test.esk").read_text(encoding="utf-8")
        for name, expression in [("g1", "(gradient (lambda (w) (batch-loss w W2)) W1)"),
                                 ("g2", "(gradient (lambda (w) (batch-loss W1 w)) W2)")]:
            original = f"(define {name} {expression})"
            if mutant.count(original) != 1:
                raise RuntimeError("wrong-zero canary cannot identify the gradient definition")
            mutant = mutant.replace(original, f"(define {name} (reshape #(0.0 0.0 0.0 0.0) 2 2))")
        canary = scratch / "wrong_zero_canary.esk"
        canary.write_text(mutant, encoding="utf-8")
        code, output = execute(canary)
        if (code != 1 or "FAIL: W1 gradient vs FD" not in output
                or "FAIL: W2 gradient vs FD" not in output or SUCCESS in output):
            failures.append(f"wrong-zero canary did not fail correctly: exit={code}\n{output}")
        else:
            print(f"PASS {args.mode} dense={args.dense}: rejects wrong-zero gradients", flush=True)

        # The same well-formed calls must work outside AD. This distinguishes
        # the new refusal contract from an arity/type error or removed builtin.
        control = scratch / "numeric_loss_controls.esk"
        control.write_text("\n".join(
            f'(if (number? (let ((w #(0.7 0.3))) {expression})) '
            f'(display "PASS: numeric {name}\\n") (exit 1))'
            for name, expression in LOSS_EXPRESSIONS.items()), encoding="utf-8")
        code, output = execute(control)
        if (not positive_ok(code, output, "PASS: numeric cosine-embedding-loss")
                or any("PASS: numeric " + name not in output for name in LOSS_EXPRESSIONS)):
            failures.append(f"numeric loss controls: exit={code}\n{output}")
        else:
            print(f"PASS {args.mode} dense={args.dense}: all 13 numeric loss controls", flush=True)

        for name, (program, message) in REFUSALS.items():
            source = scratch / (name + ".esk")
            source.write_text(program + '\n(display "UNEXPECTED SUCCESS")\n', encoding="utf-8")
            code, output = execute(source)
            # A crash, verifier failure or compile failure is not a refusal.
            if not refusal_ok(code, output, message):
                failures.append(f"refusal {name}: exit={code}\n{output}")
            else:
                print(f"PASS {args.mode} dense={args.dense}: refuses {name}", flush=True)

    for failure in failures:
        print(f"FAIL: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
