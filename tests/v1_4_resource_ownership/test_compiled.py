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
    "positive-resource-move-to-owner",
    "positive-udp-bind-close",
    "positive-safe-scalar-conversion",
    "positive-file-borrow-scopes",
    "positive-file-call-with-port",
    "positive-file-branches",
    "positive-file-loop-drain",
    "positive-file-do-loop",
    "positive-file-iteration-scoped",
    "positive-file-for-each",
    "positive-function-handles",
    "positive-tail-recursive-drain",
    "positive-dynamic-wind-close",
    "positive-guard-handle",
    "positive-callcc-escape",
    "positive-thread-join",
    "positive-event-loop",
    "positive-tls-context",
    "positive-process",
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
    "negative-do-hidden-acquisition",
    "negative-when-hidden-acquisition",
    "negative-values-hidden-acquisition",
    "negative-raise-hidden-acquisition",
    "negative-computed-callable-result",
    "negative-letrec-callable-result",
    "negative-let-syntax-hidden-acquisition",
    "negative-letrec-syntax-hidden-acquisition",
    "negative-function-local-factory-shadow",
    "negative-discarded-resource-move",
    "negative-udp-leak",
    "negative-udp-wrong-close",
    "negative-tcp-accept-wrong-kind",
    "negative-forged-safe-scalar-extern",
    "negative-imported-safe-scalar-collision",
    "negative-file-unclosed",
    "negative-file-one-branch-unclosed",
    "negative-file-two-arm-if",
    "negative-file-double-close",
    "negative-file-use-after-close",
    "negative-file-close-while-borrowed",
    "negative-file-borrow-close-owner",
    "negative-file-borrow-escape",
    "negative-call-with-port-reuse",
    "negative-wrong-kind-close",
    "negative-loop-closes-outer",
    "negative-loop-iteration-unclosed",
    "negative-do-loop-closes-outer",
    "negative-loop-nontail",
    "negative-for-each-closes",
    "negative-function-unclosed",
    "negative-function-double-close",
    "negative-function-result-dropped",
    "negative-closure-escape",
    "negative-guard-loses-local",
    "negative-raise-before-close",
    "negative-callcc-reentry",
    "negative-callcc-open-on-escape",
    "negative-dynamic-wind-double-close",
    "negative-thread-captures-handle",
    "negative-thread-not-joined",
    "negative-mutex-not-destroyed",
    "negative-event-loop-unrefined",
    "negative-event-loop-unclosed",
    "negative-process-unclosed",
    "negative-tls-unclosed",
    "negative-tls-use-after-close",
)

# Every handle-family falsifier must fail for its own reason, not for an
# unrelated diagnostic: the expected text is part of the contract.
EXPECTED_DIAGNOSTIC = {
    "negative-file-unclosed": "resource 'in' may still be live at scope exit",
    "negative-file-one-branch-unclosed": "resource 'in' may still be live at scope exit",
    "negative-file-two-arm-if": "resource 'in' may still be live at scope exit",
    "negative-file-double-close": "duplicate close of resource 'out'",
    "negative-file-use-after-close": "use of closed or moved resource 'in'",
    "negative-file-close-while-borrowed": "cannot close borrowed resource 'p'",
    "negative-file-borrow-close-owner": "cannot consume borrowed resource 'in'",
    "negative-file-borrow-escape": "borrowed resource 'p' escapes its borrow scope",
    "negative-call-with-port-reuse": "use of closed or moved resource 'in'",
    "negative-wrong-kind-close": "wrong resource kind passed to close-input-port",
    "negative-loop-closes-outer": "leaves handle 'in' closed on its back edge",
    "negative-loop-iteration-unclosed": "acquired in an iteration of loop 'loop'",
    "negative-do-loop-closes-outer": "do loop leaves handle 'out' closed on its back edge",
    "negative-loop-nontail": "call to loop 'loop' is not in tail position",
    "negative-for-each-closes": "runs zero or more times but changes handle 'out'",
    "negative-function-unclosed": "in 'peek-first'",
    "negative-function-double-close": "duplicate close of resource 'out'",
    "negative-function-result-dropped": "factory result from 'open-log' must be bound directly",
    "negative-closure-escape": "is captured by a closure whose calls the checker cannot see",
    "negative-guard-loses-local": "is still open when an exception raised by read-line leaves its scope",
    "negative-raise-before-close": "terminal control transfer 'raise' occurs while resource 'in'",
    "negative-callcc-reentry": "continuation 'k' escapes as a value",
    "negative-callcc-open-on-escape": "is still open when continuation 'k' leaves its scope",
    "negative-dynamic-wind-double-close": "duplicate close of resource 'in'",
    "negative-thread-captures-handle": "is captured by a thread body",
    "negative-thread-not-joined": "resource 't' may still be live at scope exit",
    "negative-mutex-not-destroyed": "resource 'm' may still be live at scope exit",
    "negative-event-loop-unrefined": "must be refined with a truth test",
    "negative-event-loop-unclosed": "resource 'loop' may still be live at scope exit",
    "negative-process-unclosed": "resource 'proc' may still be live at scope exit",
    "negative-tls-unclosed": "resource 'ctx' may still be live at scope exit",
    "negative-tls-use-after-close": "use of closed or moved resource 'ctx'",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", default=os.environ.get("ESHKOL_RUN"))
    parser.add_argument("--lib-dir", default=os.environ.get("ESHKOL_LIB_DIR"))
    parser.add_argument("--case", action="append", help="run one named compiled case (repeatable)")
    args = parser.parse_args()
    runner = args.runner
    lib_dir = args.lib_dir
    if not runner or not lib_dir:
        raise SystemExit("set ESHKOL_RUN and ESHKOL_LIB_DIR")
    env = os.environ.copy()
    env["ESHKOL_LIB_DIR"] = str(lib_dir)
    source_root = Path(__file__).resolve().parents[2]
    module_root = HERE / "modules"
    env["ESHKOL_PATH"] = os.pathsep.join((str(source_root / "lib"), str(module_root)))
    available = set(POSITIVE + NEGATIVE)
    cases = args.case if args.case is not None else list(POSITIVE + NEGATIVE)
    invalid = [name for name in cases if name not in available]
    if invalid:
        parser.error("unknown --case name(s): " + ", ".join(invalid))
    if not cases:
        parser.error("at least one --case is required")
    results = []
    with tempfile.TemporaryDirectory(prefix="eshkol-strict-resources-") as temp:
        out_dir = Path(temp)
        for name in cases:
            source = FIXTURES / f"{name}.esk"
            artifact = out_dir / f"{name}.o"
            command = [runner, "--strict-resources", "-c", "-o", str(artifact), str(source), "-L", lib_dir]
            completed = subprocess.run(command, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, timeout=90)
            is_negative = name in NEGATIVE
            diagnostic = "Strict resource error:" in completed.stdout
            expected = EXPECTED_DIAGNOSTIC.get(name)
            if expected is not None:
                diagnostic = diagnostic and expected in completed.stdout
            passed = (completed.returncode == 1 and diagnostic and not artifact.exists()) if is_negative \
                     else (completed.returncode == 0 and artifact.exists())
            results.append((name, passed, completed.returncode, diagnostic, artifact.exists(), completed.stdout))
        if args.case is None:
            positive_names = [name for name in cases if name in POSITIVE]
            if positive_names:
                harmless_object = out_dir / f"{positive_names[0]}.o"
                mixed_output = out_dir / "strict-mixed-link"
                mixed = subprocess.run(
                    [runner, "--strict-resources", "-o", str(mixed_output),
                     str(FIXTURES / f"{positive_names[0]}.esk"), str(harmless_object),
                     "-L", lib_dir], env=env, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, timeout=30)
                mixed_diagnostic = "opaque .o inputs" in mixed.stdout
                results.append(("reject-mixed-opaque-object", mixed.returncode == 1 and
                                mixed_diagnostic and not mixed_output.exists(), mixed.returncode,
                                mixed_diagnostic, mixed_output.exists(), mixed.stdout))
            vm_output = out_dir / "strict-unsupported.eskb"
            vm = subprocess.run(
                [runner, "--strict-resources", "--profile", "hosted-vm", "-o", str(vm_output),
                 str(FIXTURES / f"{POSITIVE[0]}.esk")], env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30)
            vm_diagnostic = "supported only by native compilation" in vm.stdout
            results.append(("reject-vm-profile", vm.returncode == 1 and
                            vm_diagnostic and not vm_output.exists(), vm.returncode,
                            vm_diagnostic, vm_output.exists(), vm.stdout))
            wasm_output = out_dir / "strict-unsupported.wasm"
            wasm = subprocess.run(
                [runner, "--strict-resources", "--wasm", "-o", str(wasm_output),
                 str(FIXTURES / f"{POSITIVE[0]}.esk")], env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30)
            wasm_diagnostic = "native CPU artifacts only" in wasm.stdout
            results.append(("reject-wasm-profile", wasm.returncode == 1 and
                            wasm_diagnostic and not wasm_output.exists(), wasm.returncode,
                            wasm_diagnostic, wasm_output.exists(), wasm.stdout))
    failures = [row for row in results if not row[1]]
    for name, passed, code, diagnostic, artifact, output in results:
        print(f"{'PASS' if passed else 'FAIL'} {name}: exit={code} resource_diagnostic={diagnostic} artifact={artifact}")
        if not passed:
            print(output[-3000:])
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
