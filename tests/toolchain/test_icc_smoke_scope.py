#!/usr/bin/env python3
"""Exercise exact probe selection through the real ICC smoke script."""

import json
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
RUNNER = r'''#!/usr/bin/env bash
set -u
printf '%s\n' "$*" >> "$FAKE_RUN_LOG"
if [ "${1:-}" = "-r" ]; then
    echo 'PASS: event_loop_test'
    exit "${FAKE_JIT_EXIT:-0}"
fi
out=''
while [ "$#" -gt 0 ]; do
    if [ "$1" = '-o' ]; then
        shift
        out="$1"
    fi
    shift
done
[ -n "$out" ] || exit 2
cat > "$out" <<'AOT'
#!/usr/bin/env bash
echo 'PASS: event_loop_test'
exit 0
AOT
chmod +x "$out"
'''


def run(script, selector, trace, env, log, extra_env=None):
    call_env = env.copy()
    call_env.update({"TRACE_DIR": str(trace), "FAKE_RUN_LOG": str(log)})
    if extra_env:
        call_env.update(extra_env)
    return subprocess.run(
        ["bash", str(script), "--only", selector],
        cwd=ROOT,
        env=call_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def smoke_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    with tempfile.TemporaryDirectory(prefix="icc-smoke-scope-") as tmp_name:
        tmp = Path(tmp_name)
        build = tmp / "build"
        scratch = tmp / "scratch"
        build.mkdir()
        scratch.mkdir()
        runner = build / "eshkol-run"
        runner.write_text(RUNNER)
        runner.chmod(0o755)
        log = tmp / "runner.log"
        script = ROOT / "scripts/run_icc_smoke.sh"
        base_env = os.environ.copy()
        base_env.update({"BUILD_DIR": str(build), "ESHKOL_SCRATCH_ROOT": str(scratch)})

        failed = run(script, "event_loop_works", tmp / "trace-fail", base_env,
                     log, {"FAKE_JIT_EXIT": "7"})
        assert failed.returncode != 0, "matching PASS output followed by exit 7 was accepted"
        assert "ICC smoke probe event_loop_works: FAIL" in failed.stdout
        assert "ICC smoke gate: PASS" not in failed.stdout
        rows = smoke_rows(tmp / "trace-fail/eshkol_smoke.jsonl")
        events = [r for r in rows if r.get("kind") == "eshkol_smoke"]
        assert len(events) == 1 and events[0].get("name") == "event_loop_works"
        assert events[0].get("value") == "FAIL", events

        log.write_text("")
        passed = run(script, "event_loop_works", tmp / "trace-pass", base_env, log)
        assert passed.returncode == 0, passed.stdout
        assert "ICC smoke probe event_loop_works: PASS (partial run; no full-smoke claim)" in passed.stdout
        assert "ICC smoke gate: PASS" not in passed.stdout
        rows = smoke_rows(tmp / "trace-pass/eshkol_smoke.jsonl")
        events = [r for r in rows if r.get("kind") == "eshkol_smoke"]
        assert len(events) == 1 and events[0].get("name") == "event_loop_works"
        assert events[0].get("value") == "PASS", events
        assert len(log.read_text().splitlines()) == 2, log.read_text()

        log.write_text("")
        unknown = run(script, "no_such_probe", tmp / "trace-unknown", base_env, log)
        assert unknown.returncode == 2, unknown.stdout
        assert "Unknown ICC smoke probe: no_such_probe" in unknown.stdout
        assert not log.read_text(), "unknown selection executed the runner"

    print("test_icc_smoke_scope.py: PASS")


if __name__ == "__main__":
    main()
