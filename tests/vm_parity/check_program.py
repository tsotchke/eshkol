"""Check a corpus program's complete transcript or its compile-time refusal."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

runner, vm, filename, work_root = sys.argv[1:]
source = Path(filename).resolve()
text = source.read_text()
expected = "".join(line + "\n" for line in re.findall(r"^;; OUTPUT: (.*)$", text, re.M))
diagnostic = re.search(r"^;; EXPECT-COMPILE: fail$", text, re.M)
env = dict(os.environ, ESHKOL_VM_NO_DISASM="1", ESHKOL_JIT_CACHE="0")
framing = re.compile(rb"^(?:\[ESKB\]|=== Eshkol VM|=== Execution complete ===|\s*\[compiled:)")


def run(args, compile_only=False, can_refuse=True):
    result = subprocess.run(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=120)
    if diagnostic and can_refuse:
        assert result.returncode == 1 and b"consumed more than once" in result.stderr, (
            args, result.returncode, result.stderr[-2000:])
        return
    assert result.returncode == 0, (args, result.returncode, result.stderr[-2000:])
    if not compile_only:
        output = b"".join(line for line in result.stdout.splitlines(keepends=True)
                          if not framing.match(line))
        assert output == expected.encode(), (args, output, expected)


with tempfile.TemporaryDirectory(prefix=source.stem + "-", dir=work_root) as work:
    artifact = Path(work) / "program"
    bytecode = Path(work) / "program.eskb"
    lib = "-L" + str(Path(runner).parent)
    run([runner, "-n", "-O0", "-r", source, lib])
    run([runner, "-n", "-O0", source, lib, "-o", artifact], compile_only=True)
    run([vm, source])
    run([runner, "--profile", "hosted-vm", "--emit-eskb", bytecode, source, lib],
        compile_only=True)
    if diagnostic:
        assert not artifact.exists() and not bytecode.exists()
    else:
        run([artifact], can_refuse=False)
        run([vm, bytecode], can_refuse=False)
