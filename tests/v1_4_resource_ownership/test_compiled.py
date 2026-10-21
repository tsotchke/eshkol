#!/usr/bin/env python3
"""Compiled contract tests for native --strict-resources.

Set ESHKOL_RUN to the task's eshkol-run binary and ESHKOL_LIB_DIR to the
matching runtime build directory. Negatives are compiled only; their binaries
are never executed and must not be emitted.
"""
import os
import argparse
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
POSITIVE = (
    "positive-borrow-scope",
    "positive-both-branches-close",
    "positive-error-branch",
    "positive-shadowed-factory",
    "positive-lexical-shadowing",
    "positive-import-alias",
    "positive-import-rename",
    "positive-letrec-star-shadow-capture",
    "positive-imported-function-declaration",
)
NEGATIVE = (
    "negative-leak",
    "negative-double-close",
    "negative-use-after-close",
    "negative-consume-while-borrowed",
    "negative-loan-escape",
    "negative-one-branch-leaks",
    "negative-shadowed-close",
    "negative-forged-error-predicate",
    "negative-closure-later-global",
    "negative-resource-copy",
    "negative-resource-arithmetic",
    "negative-resource-aggregate",
    "negative-unknown-ffi",
    "negative-factory-alias-mutation",
    "negative-predicate-alias-mutation",
    "negative-closer-alias-mutation",
    "negative-exit-before-close",
    "negative-quasiquote-acquisition",
    "negative-factory-alias-branch-mutation",
    "negative-untracked-builtin-factory",
    "negative-resource-contract-capture",
    "negative-untracked-close",
    "negative-untracked-observer",
    "negative-container-handle",
    "negative-untrusted-close-alias-collision",
    "negative-unrefined-conditional-close",
    "negative-callable-factory-via-lambda",
    "negative-imported-initializer-leak",
    "negative-imported-unknown-factory",
    "negative-two-module-export-collision",
    "negative-error-before-close",
    "negative-error-alias-before-close",
    "negative-unknown-builtin-alias-factory",
    "negative-callable-return-builtin-factory",
    "negative-re-refine-incompatible-join",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", default=os.environ.get("ESHKOL_RUN"))
    parser.add_argument("--lib-dir", default=os.environ.get("ESHKOL_LIB_DIR"))
    args = parser.parse_args()
    runner = args.runner
    lib_dir = args.lib_dir
    if not runner or not lib_dir:
        raise SystemExit("set ESHKOL_RUN and ESHKOL_LIB_DIR")
    env = os.environ.copy()
    env["ESHKOL_LIB_DIR"] = str(lib_dir)
    source_root = Path(__file__).resolve().parents[2]
    module_root = HERE / "modules"
    env["ESHKOL_PATH"] = os.pathsep.join((str(module_root), str(source_root / "lib")))
    results = []
    with tempfile.TemporaryDirectory(prefix="eshkol-strict-resources-") as temp:
        out_dir = Path(temp)
        for name in POSITIVE + NEGATIVE:
            source = FIXTURES / f"{name}.esk"
            artifact = out_dir / f"{name}.o"
            command = [runner, "--strict-resources", "-c", "-o", str(artifact), str(source), "-L", lib_dir]
            completed = subprocess.run(command, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, timeout=90)
            is_negative = name in NEGATIVE
            diagnostic = "Strict resource error:" in completed.stdout
            passed = (completed.returncode == 1 and diagnostic and not artifact.exists()) if is_negative \
                     else (completed.returncode == 0 and artifact.exists())
            results.append((name, passed, completed.returncode, diagnostic, artifact.exists(), completed.stdout))
    failures = [row for row in results if not row[1]]
    for name, passed, code, diagnostic, artifact, output in results:
        print(f"{'PASS' if passed else 'FAIL'} {name}: exit={code} resource_diagnostic={diagnostic} artifact={artifact}")
        if not passed:
            print(output[-3000:])
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
