import concurrent.futures
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


runner, vm, source = sys.argv[1:]
env = dict(os.environ, ESHKOL_JIT_CACHE="0", ESHKOL_VM_NO_DISASM="1")


def draw(command):
    output = subprocess.check_output(command, text=True, env=env)
    match = re.search(r"draw=\s*([0-9.eE+-]+)", output)
    assert match and 0 <= float(match[1]) < 1, output
    return float(match[1])


with tempfile.TemporaryDirectory(dir=".") as work:
    binary = str(Path(work, "automatic").resolve())
    subprocess.run([runner, "-n", "-o", binary, source], env=env, check=True)
    for command in ([binary], [vm, source], [runner, "-n", "-r", source]):
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            first, second = pool.map(draw, [command, command])
        assert first != second, command
