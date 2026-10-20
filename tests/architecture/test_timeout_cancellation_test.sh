#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
python3 - "$root" <<'PYOUTER'
import os, pathlib, shlex, signal, subprocess, sys, tempfile, time
root = pathlib.Path(sys.argv[1])
source = root / "scripts/lib/test_isolation.sh"

def invoke(command, **kwargs):
    shell = f"trap \"\" TERM INT; source {shlex.quote(str(source))}; ESHKOL_TEST_TIMEOUT_FORCE_PYTHON=1 {command}"
    return subprocess.run(["bash", "-c", shell], **kwargs).returncode

if invoke("eshkol_test_timeout 10 python3 -c 'raise SystemExit(7)'") != 7:
    raise SystemExit("FAIL: normal exit status was not preserved")
if invoke("eshkol_test_timeout .2 python3 -c 'import time; time.sleep(10)'") != 124:
    raise SystemExit("FAIL: timeout status was not 124")

with tempfile.TemporaryDirectory(prefix="eshkol-timeout-cancel-") as td:
    heartbeat = pathlib.Path(td) / "heartbeat"
    child = pathlib.Path(td) / "heartbeat.py"
    child.write_text("import pathlib,sys,time\np=pathlib.Path(sys.argv[1])\nwhile True:\n with p.open('a') as f: f.write('beat\\n')\n time.sleep(.05)\n")
    for sig, expected in ((signal.SIGTERM, 143), (signal.SIGINT, 130)):
        heartbeat.unlink(missing_ok=True)
        command = (
            f"eshkol_test_timeout 30 python3 {shlex.quote(str(child))} "
            f"{shlex.quote(str(heartbeat))}"
        )
        shell = f"trap \"\" TERM INT; source {shlex.quote(str(source))}; ESHKOL_TEST_TIMEOUT_FORCE_PYTHON=1 {command}"
        proc = subprocess.Popen(["bash", "-c", shell], start_new_session=True)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not heartbeat.exists():
            time.sleep(.02)
        if not heartbeat.exists():
            os.killpg(proc.pid, signal.SIGKILL)
            raise SystemExit("FAIL: heartbeat child did not start")
        os.killpg(proc.pid, sig)
        try:
            status = proc.wait(timeout=4)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            raise SystemExit(f"FAIL: wrapper did not exit after {sig.name}")
        if status != expected:
            raise SystemExit(f"FAIL: {sig.name} status {status}, expected {expected}")
        size = heartbeat.stat().st_size
        time.sleep(.25)
        if heartbeat.stat().st_size != size:
            raise SystemExit(f"FAIL: descendant continued after {sig.name}")
print("PASS: timeout wrapper preserves exit/timeout/signal statuses and reaps cancelled children")
PYOUTER
