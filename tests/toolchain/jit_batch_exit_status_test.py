#!/usr/bin/env python3
"""Check file-mode JIT lookup failures affect the CLI status, with a good control."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def run(runner, repo, link_dir, source):
    env = os.environ.copy()
    env["ESHKOL_PATH"] = str(repo / "lib")
    # The bitcode/object artifacts are generated in the CMake binary directory;
    # source-tree `lib` contains language modules, not compiled stdlib artifacts.
    env["ESHKOL_LIB_DIR"] = str(link_dir)
    env["ESHKOL_JIT_CACHE"] = "0"
    return subprocess.run(
        [runner, "-r", str(source), "-L", str(link_dir)],
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--link-dir", type=Path, required=True)
    args = parser.parse_args()

    fixtures = args.repo_root / "tests" / "toolchain" / "fixtures"
    failed = run(args.runner, args.repo_root, args.link_dir,
                 fixtures / "jit_batch_unbound_name_failure.esk")
    failed_output = failed.stdout + failed.stderr
    if failed.returncode != 1:
        print(f"FAIL: unbound JIT entry returned {failed.returncode}, expected 1")
        print(failed_output)
        return 1
    if "jit_batch_name_that_is_never_bound" not in failed_output:
        print("FAIL: failure did not identify the deliberately unbound name")
        print(failed_output)
        return 1
    if "UNBOUND_LOOKUP_OK" in failed.stdout:
        print("FAIL: failed batch executed its marker form")
        print(failed_output)
        return 1
    print(f"PASS: unbound-name batch returned exact exit code {failed.returncode}")

    control = run(args.runner, args.repo_root, args.link_dir,
                  fixtures / "jit_batch_success_control.esk")
    if control.returncode != 0 or "JIT_BATCH_OK" not in control.stdout:
        print("FAIL: valid JIT batch did not complete successfully")
        print(control.stdout + control.stderr)
        return 1
    print(f"PASS: jit_batch_success_control.esk returned {control.returncode} and emitted JIT_BATCH_OK")

    definitions = run(args.runner, args.repo_root, args.link_dir,
                      fixtures / "jit_batch_definitions_only.esk")
    if definitions.returncode != 0:
        print("FAIL: definitions-only batch should complete successfully")
        print(definitions.stdout + definitions.stderr)
        return 1
    print(f"PASS: jit_batch_definitions_only.esk returned {definitions.returncode}")

    print("PASS: JIT batch lookup failure affects file exit status; valid and definitions-only controls pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
