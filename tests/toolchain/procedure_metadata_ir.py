"""Check how the native compiler initializes procedure source in an executable."""
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def check_ir(ir):
    functions = dict(re.findall(
        r"^define [^\n]*?@([\w.]+)\([^\n]*\) [^\n]*\{\n(.*?)^\}", ir, re.M | re.S
    ))
    registration = "call void @eshkol_lambda_registry_add("
    chunks = [body for name, body in functions.items()
              if name.startswith("__eshkol_lib_init_sexpr_chunk_")]
    if len(chunks) < 2 or any(not 1 <= body.count(registration) <= 4 for body in chunks):
        raise ValueError("expected ordered groups of up to four procedure registrations")
    main = functions["main"]
    if main.count("call void @__eshkol_lib_init_sexpr_chunk_") != len(chunks):
        raise ValueError("main must call every procedure registration group")
    if any("@eshkol_ad_home_arena(" in body for body in chunks):
        raise ValueError("the metadata cell constructor must resolve the allocation arena")


if __name__ == "__main__":
    compiler, source, build = map(Path, sys.argv[1:])
    with tempfile.TemporaryDirectory(dir=build) as directory:
        output = Path(directory) / ("procedures.exe" if os.name == "nt" else "procedures")
        subprocess.run([compiler, "--dump-ir", "-O", "1", source, "-o", output,
                        "-L" + str(build)], check=True)
        check_ir(Path(str(output) + ".ll").read_text())
        result = subprocess.run([output], capture_output=True, text=True, check=True)
        if "RESULT: PASS" not in result.stdout.splitlines() or "FAIL:" in result.stdout:
            raise ValueError(result.stdout)
