#!/usr/bin/env python3
"""CLI contract for positional source/object inputs and fail-closed suffixes."""
import os
import pathlib
import subprocess
import sys
import tempfile

run = pathlib.Path(sys.argv[1])
with tempfile.TemporaryDirectory(prefix="eshkol-input-ext-") as td:
    root = pathlib.Path(td)
    src = root / "control.esk"
    src.write_text('(display "INPUT-OK")\n', encoding="utf-8")
    env = os.environ.copy()
    env["ESHKOL_JIT_CACHE"] = "0"
    env["ESHKOL_JIT_CACHE_DIR"] = str(root / "jit-cache")

    jit = subprocess.run([str(run), "-n", "-r", str(src)], cwd=root,
                         env=env, text=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, timeout=60)
    if jit.returncode != 0 or "INPUT-OK" not in jit.stdout:
        raise SystemExit(f"valid -r source control failed ({jit.returncode}):\n{jit.stdout}")

    obj = root / "control.o"
    compile_obj = subprocess.run([str(run), "-n", "-c", "-o", str(obj), str(src)],
                                 cwd=root, env=env, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 timeout=60)
    if compile_obj.returncode != 0 or not obj.is_file() or obj.stat().st_size == 0:
        raise SystemExit(f"valid source/object control failed ({compile_obj.returncode}):\n{compile_obj.stdout}")

    linked = root / "object-control"
    link_obj = subprocess.run([str(run), "-n", "-o", str(linked), str(obj)],
                              cwd=root, env=env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              timeout=60)
    if link_obj.returncode != 0 or not linked.is_file() or not os.access(linked, os.X_OK):
        raise SystemExit(f"valid object input control failed ({link_obj.returncode}):\n{link_obj.stdout}")

    for name, args in (
        ("bytecode", [str(root / "bad.eskb")]),
        ("unknown", [str(root / "bad.txt")]),
        ("mixed", [str(src), str(root / "bad.eskb")]),
        ("run-object", ["-r", str(obj)]),
    ):
        output = root / f"{name}-must-not-exist"
        cmd = [str(run), "-n", *args, "-o", str(output)]
        result = subprocess.run(cmd, cwd=root, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=60)
        if result.returncode == 0:
            raise SystemExit(f"{name} input unexpectedly succeeded:\n{result.stdout}")
        if output.exists():
            raise SystemExit(f"{name} input created output before rejection: {output}")
