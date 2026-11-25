"""Prove checkpoint/suffix equivalence across real native process teardown."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def verify(binary: Path) -> dict:
    def run(phase: str, cut: int, path: Path, success: bool = True) -> str:
        env = dict(os.environ, ESHKOL_CHECKPOINT_PHASE=phase,
                   ESHKOL_CHECKPOINT_CUT=str(cut), ESHKOL_CHECKPOINT_FILE=str(path))
        result = subprocess.run([str(binary)], env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        if success:
            if result.returncode or "RESULT: PASS" not in result.stdout or "FAIL:" in result.stdout:
                raise AssertionError(f"{phase}/{cut}: {result.returncode}\n{result.stdout}")
        elif result.returncode == 0 or "CHECKPOINT-FINAL:" in result.stdout:
            raise AssertionError("Corrupted checkpoint was accepted")
        return result.stdout

    with tempfile.TemporaryDirectory(prefix="eshkol-dbsp-checkpoint-") as directory:
        path = Path(directory) / "state.chk"
        reference = re.search(r"CHECKPOINT-FINAL: ([0-9a-f]{64})", run("full", 12, path))
        if not reference:
            raise AssertionError("Uninterrupted run did not report its canonical final state")
        for cut in range(13):
            run("save", cut, path)
            saved = path.read_bytes()
            resumed = re.search(r"CHECKPOINT-FINAL: ([0-9a-f]{64})", run("resume", cut, path))
            if not resumed or resumed.group(1) != reference.group(1):
                raise AssertionError(f"Restart at cut {cut} diverged")
            if path.read_bytes() != saved:
                raise AssertionError("Restore modified its checkpoint input")
        path.write_bytes(path.read_bytes()[:-1])
        run("resume", 12, path, success=False)
        return {"cuts": 13, "process_runs": 28, "corrupt_file_refused": True,
                "final_state_sha256": reference.group(1)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.binary.resolve()), sort_keys=True))
