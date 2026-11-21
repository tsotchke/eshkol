"""Replay BLC cost correspondence and time identical public calls on Linux x64."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CORPUS = "97bf65fc95fd4c7fe5aa306b488660e4c45af3242a20397046c0b7d0ad9051be"
BASE = "25048643f33b8b24f5e1602e4ec13d85f745bf88fda49c504dce13138cf38d6c"
CASES = [("universal", None), ("discard", None), ("church", None), ("byte-prefix", None),
         ("deep", 32), ("deep", 256), ("already-normal", None),
         ("shared-bound", 16), ("duplicate", 2), ("duplicate", 5), ("duplicate", 8)]


def run(args, work, env=None, timeout=120):
    result = subprocess.run(args, cwd=work, env=env, text=True, capture_output=True,
                            timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{args[0]} exited {result.returncode}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def check_features(runner, work, rows):
    source = ROOT / "tests/features/blc_test.esk"
    env = dict(os.environ, ESHKOL_BINDER_ROWS=str(rows))
    for name, lib in [("candidate", ROOT / "lib"), ("baseline", work / "baseline")]:
        # A stale installed module can pass the semantic tests. Using a separate
        # cwd per build keeps Eshkol's module and output caches isolated.
        cwd = work / (name + "-tests")
        cwd.mkdir()
        common = [runner, "-I", lib]
        binary = cwd / "test"
        run([*common, "-O", "2", "-o", binary, source], cwd)
        for mode, command in [("AOT", [binary]), ("JIT", [*common, "-r", source])]:
            out = run(command, cwd, env).splitlines()
            if "Binder rows passed: 1560" not in out or out[-1:] != ["RESULT: PASS"]:
                raise ValueError(f"Incomplete {name} {mode} replay: {out[-10:]}")


def check_last_contraction(runner, work):
    # Rejecting the last available contraction must fail the exact-bound
    # assertion in the real suite. A syntax or module-load error is inconclusive.
    mutant = work / "off-by-one"
    core = mutant / "core"
    core.mkdir(parents=True)
    source = (ROOT / "lib/core/blc.esk").read_text()
    guard = "((= remaining 0)"
    if source.count(guard) != 1:
        raise ValueError("unexpected bound check")
    (core / "blc.esk").write_text(source.replace(guard, "((<= remaining 1)"))
    control = subprocess.run([runner, "-I", mutant, "-r", ROOT / "tests/features/blc_test.esk"],
                             cwd=mutant, capture_output=True, text=True, timeout=120,
                             env=dict(os.environ, ESHKOL_BINDER_ROWS=""))
    if (control.returncode != 1 or
            "FAIL: three contractions succeed at bound three" not in control.stdout or
            "RESULT: FAIL" not in control.stdout):
        raise ValueError(f"Unexpected off-by-one control: {control.stdout} {control.stderr}")


def main():
    runner = shutil.which("eshkol-run")
    if not runner:
        raise SystemExit("put the v1.3.6-evolve Linux x64 lite bin directory on PATH")
    installed = Path(runner).resolve().parents[1] / "share/eshkol/lib"
    base = (installed / "core/blc.esk").read_bytes()
    if hashlib.sha256(base).hexdigest() != BASE:
        raise ValueError("installed core.blc differs from the pinned reference")
    proof = run(["lean", ROOT / "verification/blc/Costed.lean"], ROOT)
    if "'BLC.Costed.bounded_correspondence' depends on axioms: [propext, Quot.sound]" not in proof:
        raise ValueError(f"Unexpected Lean axiom report: {proof}")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        rows = gzip.decompress((ROOT / "tests/fixtures/blc/binder-rows.tsv.gz").read_bytes())
        if hashlib.sha256(rows).hexdigest() != CORPUS:
            raise ValueError("unexpected binder corpus")
        path = work / "rows"
        path.write_bytes(rows)
        (work / "baseline/core").mkdir(parents=True)
        (work / "baseline/core/blc.esk").write_bytes(base)
        check_features(runner, work, path)
        check_last_contraction(runner, work)
        if run(["node", ROOT / "benchmarks/blc_eval_vm.cjs", path], work).strip() != "1560 VM boundary pairs and shared argument bounds passed":
            raise ValueError("incomplete WASM replay")

        binaries = {}
        for name, lib in [("baseline", work / "baseline"), ("candidate", ROOT / "lib")]:
            cwd = work / (name + "-bench")
            cwd.mkdir()
            binary = cwd / "bench"
            run([runner, "-I", lib, "-O", "2", "-o", binary, ROOT / "benchmarks/blc_eval.esk"], cwd)
            binaries[name] = binary
        measurements = []
        for case, depth in CASES:
            env = dict(os.environ, BLC_CASE=case, BLC_DEPTH=str(depth or ""),
                       OPENBLAS_NUM_THREADS="1")
            trials = {key: [] for key in binaries}
            for trial in range(9):
                for key in (["baseline", "candidate"] if trial % 2 == 0 else ["candidate", "baseline"]):
                    exe = binaries[key]
                    trials[key].append(int(run([exe], exe.parent, env).strip()))
            measurements.append(dict(case=case, depth=depth, trials_ns=trials,
                                     median_ns={k: statistics.median(v) for k, v in trials.items()}))
        print(json.dumps(dict(rows=1560, modes=["AOT", "JIT", "WASM"], measurements=measurements), indent=2))


if __name__ == "__main__":
    main()
