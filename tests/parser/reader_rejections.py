"""Check each invalid source spelling independently through a reader entry point."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


native, vm, case, engine = sys.argv[1:]
work = Path(tempfile.mkdtemp(prefix=case + "-", dir=Path.cwd()))
env = dict(os.environ, ESHKOL_VM_NO_DISASM="1",
           ESHKOL_JIT_CACHE="0",
           ESHKOL_AOT_MODULE_CACHE_DIR=str(work / "modules"))
if case == "reserved_identifiers":
    diagnostic = b"reserved syntax identifier"
    inputs = [b"'_v10.name", b"'|_v10.name|", b"'|a\\x1d;|", b"'a\x1d",
              b"'__eshkol_macro_binding_7"]
else:
    raise ValueError(case)

for index, data in enumerate(inputs):
    source = work / f"input-{index}.esk"
    source.write_bytes(data + b"\n")
    if engine == "vm":
        command = [vm, source]
    elif engine == "jit":
        command = [native, "-r", source]
    else:
        command = [native, source, "-o", work / f"program-{index}"]
    result = subprocess.run(command, env=env, capture_output=True, timeout=60)
    output = result.stdout + result.stderr
    if result.returncode <= 0 or diagnostic not in output:
        raise SystemExit(f"{engine} input {index}: status {result.returncode}, {output!r}")
