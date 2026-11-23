"""Exercise XLA selection and diagnostics on a real tensor reduction."""
import os
from pathlib import Path
import re
import subprocess
import sys

runner, build, mode = sys.argv[1:]
fixture = Path(__file__).with_suffix(".esk")
lowering = "[XLA] lowering eshkol_xla_reduce"
for enabled, verbose in [(False, True), (True, False), (True, True)]:
    env = dict(os.environ, ESHKOL_ENABLE_XLA=str(int(enabled)),
               ESHKOL_XLA_VERBOSE=str(int(verbose)), ESHKOL_XLA_THRESHOLD="1",
               ESHKOL_JIT_CACHE="0", ESHKOL_LIB_DIR=build)
    binary = Path(build) / f"xla_dispatch_{mode}_{int(enabled)}_{int(verbose)}"
    flags = ["-r"] if mode == "jit" else ["-i", "-o", str(binary)]
    run = subprocess.run([runner, *flags, str(fixture), f"-L{build}"],
                         env=env, capture_output=True, text=True, check=True)
    if (lowering in run.stderr) != (enabled and verbose):
        raise RuntimeError(f"unexpected lowering trace for {enabled=}, {verbose=}: {run.stderr}")
    if mode == "aot":
        ir = Path(str(binary) + ".ll").read_text()
        call = re.search(r"call [^\n]*@eshkol_xla_reduce\(", ir)
        if bool(call) != enabled:
            raise RuntimeError(f"unexpected XLA reduce call for {enabled=}")
        run = subprocess.run([binary], env=env, capture_output=True, text=True, check=True)
    if "PASS: sum" not in run.stdout:
        raise RuntimeError(f"missing reduction result: {run.stdout}")
