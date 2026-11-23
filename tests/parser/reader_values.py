"""Check the bytes printed by a reader fixture on each selected engine."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


native, vm, case, engine = sys.argv[1:]
source = Path(__file__).with_name(case + ".esk")
expected = source.with_suffix(".expected").read_bytes()
root = source.parents[2]
# The working directory holds the ahead-of-time executable and module cache
# for this one run, and is removed when the check ends, whatever its outcome.
with tempfile.TemporaryDirectory(prefix=case + "-", dir=Path.cwd()) as scratch:
    work = Path(scratch)
    env = dict(os.environ, ESHKOL_VM_NO_DISASM="1",
               ESHKOL_JIT_CACHE="0",
               ESHKOL_AOT_MODULE_CACHE_DIR=str(work / "modules"))

    if engine == "jit":
        command = [native, "-r", source]
    elif engine == "vm":
        command = [vm, source]
    else:
        executable = work / "program"
        subprocess.run([native, source, "-o", executable], cwd=root, env=env,
                       timeout=120, check=True)
        command = [executable]

    result = subprocess.run(command, cwd=root, env=env, stdout=subprocess.PIPE,
                            timeout=600, check=True)
    if result.stdout != expected:
        raise SystemExit(f"{engine}: expected {expected!r}, received {result.stdout!r}")
