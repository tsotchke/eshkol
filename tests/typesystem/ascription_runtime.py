"""Exercise a checked ascription through native and bytecode execution."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    compiler, vm = (Path(path).resolve() for path in sys.argv[1:])
    fixture = Path(__file__).resolve().with_suffix(".esk")
    env = dict(os.environ, ESHKOL_JIT_CACHE="0", ESHKOL_VM_NO_DISASM="1")
    with tempfile.TemporaryDirectory() as directory:
        executable = Path(directory) / "ascription"
        for mode in ([], ["--unsafe"]):
            for command in ([compiler, *mode, "-n", "-r", fixture], [vm, *mode, fixture],
                            [compiler, *mode, "-n", fixture, "-o", executable], [executable]):
                result = subprocess.run(command, cwd=directory, env=env, text=True,
                                        capture_output=True, timeout=120)
                if result.returncode or ("-o" not in command and
                                         result.stdout.splitlines() != ["3", "caught", "head", "3"]):
                    raise AssertionError(f"{command}: {result.returncode}\n"
                                         f"{result.stdout}\n{result.stderr}")


if __name__ == "__main__":
    main()
