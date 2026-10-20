#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
python3 - "$root" <<'PYOUTER'
import os, pathlib, signal, subprocess, sys, tempfile, time
root = pathlib.Path(sys.argv[1])
with tempfile.TemporaryDirectory(prefix="eshkol-timeout-cancel-") as td:
    heartbeat = pathlib.Path(td) / "heartbeat"
    child = pathlib.Path(td) / "heartbeat.py"
    child.write_text("import pathlib,sys,time\np=pathlib.Path(sys.argv[1])\nwhile True:\n with p.open('a') as f: f.write('beat\\n')\n time.sleep(.05)\n")
    source = root / "scripts/lib/test_isolation.sh"
    command = f"source '{source}'; ESHKOL_TEST_TIMEOUT_FORCE_PYTHON=1 eshkol_test_timeout 30 python3 '{child}' '{heartbeat}'"
    proc = subprocess.Popen(["bash", "-c", command], start_new_session=True)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and not heartbeat.exists():
        time.sleep(.02)
    if not heartbeat.exists():
        os.killpg(proc.pid, signal.SIGKILL)
        raise SystemExit("FAIL: heartbeat child did not start")
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=4)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        raise SystemExit("FAIL: timeout wrapper did not exit after TERM")
    size = heartbeat.stat().st_size
    time.sleep(.25)
    if heartbeat.stat().st_size != size:
        raise SystemExit("FAIL: child continued after wrapper cancellation")
print("PASS: timeout wrapper forwards TERM and reaps its owned child")
PYOUTER
