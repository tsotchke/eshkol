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
    grandchild = pathlib.Path(td) / "grandchild.py"
    leader = pathlib.Path(td) / "leader.py"
    grandchild.write_text(
        "import pathlib,signal,sys,time\np=pathlib.Path(sys.argv[1])\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        "while True:\n with p.open('a') as f: f.write('beat\\n')\n time.sleep(.03)\n"
    )
    leader.write_text(
        "import signal,subprocess,sys,time\n"
        "subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2]])\n"
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
        "signal.signal(signal.SIGINT, lambda *_: sys.exit(0))\n"
        "while True: time.sleep(.1)\n"
    )

    def run_tree(timeout, cancel_signal=None, expected=None):
        heartbeat.unlink(missing_ok=True)
        command = (
            f"eshkol_test_timeout {timeout} python3 {shlex.quote(str(leader))} "
            f"{shlex.quote(str(grandchild))} {shlex.quote(str(heartbeat))}"
        )
        shell = f"trap \"\" TERM INT; source {shlex.quote(str(source))}; ESHKOL_TEST_TIMEOUT_FORCE_PYTHON=1 {command}"
        proc = subprocess.Popen(["bash", "-c", shell], start_new_session=True)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not heartbeat.exists():
            time.sleep(.02)
        if not heartbeat.exists():
            os.killpg(proc.pid, signal.SIGKILL)
            raise SystemExit("FAIL: nested heartbeat child did not start")
        if cancel_signal is not None:
            os.killpg(proc.pid, cancel_signal)
        try:
            status = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            raise SystemExit("FAIL: wrapper did not finish bounded cleanup")
        if status != expected:
            raise SystemExit(f"FAIL: cleanup status {status}, expected {expected}")
        size = heartbeat.stat().st_size
        time.sleep(.25)
        if heartbeat.stat().st_size != size:
            raise SystemExit("FAIL: TERM-ignoring grandchild survived cleanup")

    run_tree(.3, expected=124)
    run_tree(30, signal.SIGTERM, 143)
    run_tree(30, signal.SIGINT, 130)
print("PASS: timeout wrapper preserves exit/timeout/signal statuses and reaps cancelled children")
PYOUTER
