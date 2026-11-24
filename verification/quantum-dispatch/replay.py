"""Check the two-wire dispatch model and replay native success, rejection and retry."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEST = Path("tests/stdlib/quantum_authority_test.esk")
PASS = "PASS: quantum_authority_test"


def run(command, cwd, env=None):
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=720,
    )
    if result.returncode:
        raise RuntimeError(
            f"{command[0]} exited {result.returncode}\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
        )
    return result.stdout


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"missing or ambiguous mutation site: {old}")
    return text.replace(old, new)


def inject_faults(tree):
    # Only the proposer and the test driver change. qr-flush!, operator checking,
    # logging, measurement and the state-vector apply function stay byte-identical.
    register = tree / "lib/core/quantum/register.esk"
    source = replace_once(
        register.read_text(),
        "(provide call-with-quantum-region",
        "(provide q1-fault! q1-corrupt! call-with-quantum-region",
    )
    helpers = """
(define q1-mode 'clean)
(define q1-held #f)
(define (q1-fault! mode) (set! q1-mode mode))
(define (q1-corrupt!)
  (if (not (and q1-held (eq? (car q1-held) 'S))) (error "FAIL: no live producer alias"))
  (set-car! q1-held 'Sdg)
  (if (not (eq? 'Sdg (car q1-held))) (error "FAIL: alias mutation did not execute")))
(define (q1-faulted-emission control target correction)
  (if (not (eq? 'S correction)) (error "FAIL: forward evidence did not synthesize S"))
  (let ((g (list 'CZ control target)))
    (set! q1-held (list correction control))
    (case q1-mode
      ((inverse) (list g (list 'Sdg control)))
      ((wire) (list g (list correction target)))
      ((global) (list g q1-held (list 'X target) (list 'Z target) (list 'X target) (list 'Z target)))
      ((inexact) (list g (list 'RZ control 1.5707963267948966)))
      (else (list g q1-held)))))
"""
    source = replace_once(
        source,
        "(define (qr-controlled-pauli-proposal",
        helpers + "\n(define (qr-controlled-pauli-proposal",
    )
    source = replace_once(
        source,
        "(list (list 'CZ control target)\n                                                    (list correction control))",
        "(q1-faulted-emission control target correction)",
    )
    register.write_text(source)
    test = tree / TEST
    extra = """
(for-each (lambda (mode)
  (q1-fault! mode)
  (q1-rejection "core.quantum: rejected emitted two-wire operator" 0)
  (display "rejected before apply and log: ") (write mode) (newline))
  '(inverse wire global inexact))
(q1-fault! 'alias)
(set! q1-after-check q1-corrupt!)
(q1-success q1-source #t '((CZ 0 1) (S 0)))
(display "executed checked snapshot after producer mutation") (newline)
"""
    test.write_text(
        replace_once(
            test.read_text(),
            ";; The release replay injects emitter faults here in a disposable source tree.",
            extra,
        )
    )


def suite(executable, tree, mutated, aot):
    env = {**os.environ, "ESHKOL_PATH": str(tree / "lib"), "ESHKOL_JIT_CACHE": "0"}
    if aot:
        binary = tree / "authority-aot"
        run(
            [executable, "-I", tree / "lib", "-O", "1", "-o", binary, tree / TEST],
            tree,
            env,
        )
        command = [binary]
    else:
        command = [executable, "-I", tree / "lib", "-r", tree / TEST]
    lines = run(command, tree, env).splitlines()
    expected = [PASS, "RESULT: PASS"]
    if mutated:
        expected = [
            f"rejected before apply and log: {name}"
            for name in ("inverse", "wire", "global", "inexact")
        ] + ["executed checked snapshot after producer mutation", *expected]
    if lines != expected:
        raise ValueError(f"unexpected output: {lines[-15:]}")
    print(
        ("AOT" if aot else "JIT"),
        ("fault controls" if mutated else "clean"),
        *lines,
        sep="\n",
        flush=True,
    )


def retry_control(tree):
    register = tree / "lib/core/quantum/register.esk"
    source = replace_once(
        register.read_text(),
        "(provide call-with-quantum-region",
        """
(define qr-test-failure (vector #f))
(define (qr-test-allocation-failure! on?) (vector-set! qr-test-failure 0 on?))
(extern void qr-test-arm i32 i64 :real eshkol_alloc_failpoint_arm)
(provide qr-test-allocation-failure! call-with-quantum-region""",
    )
    source = replace_once(
        source,
        "(stable-amps (prepare (vector-ref result 0)))",
        """
(stable-amps (begin
    (if (vector-ref qr-test-failure 0)
        (begin (if (eq? stable-log log) (error "expected promoted log"))
               (qr-test-arm 0 0)))
    (prepare (vector-ref result 0))))""",
    )
    register.write_text(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eshkol-run", default=os.environ.get("ESHKOL_RUN", "eshkol-run")
    )
    args = parser.parse_args()
    executable = shutil.which(args.eshkol_run)
    if not executable:
        parser.error("eshkol-run executable not found")
    with tempfile.TemporaryDirectory() as directory:
        tree = Path(directory)
        run_env = {**os.environ, "LEAN_PATH": str(tree)}
        for path in (
            HERE.parent / "quantum-rewrites/Context.lean",
            HERE / "Handoff.lean",
            HERE / "PhaseHandoff.lean",
        ):
            shutil.copyfile(path, tree / path.name)
            run(
                [
                    os.environ.get("LEAN", "lean"),
                    "-DwarningAsError=true",
                    "-o",
                    path.stem + ".olean",
                    path.name,
                ],
                tree,
                run_env,
            )
        print("PASS: checked dispatch and correction theorems", flush=True)
        run_native(executable, tree)


def run_native(executable, tree):
    shutil.copytree(ROOT / "lib", tree / "lib")
    (tree / TEST).parent.mkdir(parents=True)
    shutil.copyfile(ROOT / TEST, tree / TEST)
    for mutated in (False, True):
        if mutated:
            inject_faults(tree)
        for aot in (False, True):
            suite(executable, tree, mutated, aot)
    shutil.copyfile(
        ROOT / "lib/core/quantum/register.esk",
        tree / "lib/core/quantum/register.esk",
    )
    retry_control(tree)
    env = {**os.environ, "ESHKOL_PATH": str(tree / "lib"), "ESHKOL_JIT_CACHE": "0"}
    driver = tree / "retry.esk"
    shutil.copyfile(HERE / "retry.esk", driver)
    binary = tree / "retry-aot"
    for mode in ("JIT", "AOT"):
        if mode == "JIT":
            command = [executable, "-I", tree / "lib", "-r", driver]
        else:
            run(
                [executable, "-I", tree / "lib", "-O", "1", "-o", binary, driver],
                tree,
                env,
            )
            command = [binary]
        output = run(command, tree, env).strip()
        if output != "retry: two applies, one committed X":
            raise ValueError(f"{mode}: unexpected retry result: {output}")
        print(mode, output, flush=True)


if __name__ == "__main__":
    main()
