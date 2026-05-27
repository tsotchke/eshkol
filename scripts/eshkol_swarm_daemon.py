#!/usr/bin/env python3
"""Run one bounded Eshkol swarm supervision cycle."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INTERVAL_SECONDS = 900
DEFAULT_COMMAND_TIMEOUT_SECONDS = 180


@dataclass(frozen=True)
class StepResult:
    name: str
    ok: bool
    returncode: int
    seconds: float
    output_path: str
    summary: str


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fail(message: str) -> None:
    raise SystemExit(f"ESHKOL_SWARM_DAEMON_FAILED {message}")


def write_json_atomic(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="ascii").strip())
    except (FileNotFoundError, ValueError):
        return None


def unlink_owned_lock(path: Path, pid: int) -> None:
    lock_pid = read_pid(path)
    if lock_pid is None or lock_pid == pid:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def lock_file(path: Path, stale_seconds: int) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_pid = read_pid(path)
    if existing_pid is not None and process_alive(existing_pid):
        fail(f"already_running pid={existing_pid}")
    if path.exists():
        if time.time() - path.stat().st_mtime < stale_seconds:
            fail(f"recent_lock={path}")
        unlink_owned_lock(path, existing_pid or -1)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        fail(f"lock_exists={path}")
    try:
        with os.fdopen(fd, "w", encoding="ascii") as handle:
            handle.write(str(os.getpid()) + "\n")
        yield
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def redact_summary(text: str) -> str:
    text = re.sub(r"/(?:Users|home)/[^ \t\n\r\"']+", "<redacted-path>", text)
    text = re.sub(r"[A-Za-z]:\\(?:Users|Documents and Settings)\\[^ \t\n\r\"']+", "<redacted-path>", text)
    text = re.sub(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b", "<redacted-ip>", text)
    return text


def first_meaningful_line(output: str) -> str:
    for line in output.splitlines():
        stripped = line.strip()
        if stripped:
            return redact_summary(stripped)[:160]
    return ""


def run_command(name: str, argv: list[str], root: Path, output_path: Path, timeout_seconds: int) -> StepResult:
    started = time.monotonic()
    try:
        result = subprocess.run(
            argv,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        output = result.stdout + result.stderr
        returncode = result.returncode
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        output += f"\nTIMEOUT seconds={timeout_seconds}\n"
        returncode = 124
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output, encoding="utf-8")
    seconds = time.monotonic() - started
    return StepResult(
        name=name,
        ok=returncode == 0,
        returncode=returncode,
        seconds=seconds,
        output_path=str(output_path.relative_to(root)),
        summary=first_meaningful_line(output),
    )


def status_step(root: Path, runtime_dir: Path, timeout_seconds: int) -> StepResult:
    return run_command(
        "eshkol_swarm_status",
        [sys.executable, "scripts/eshkol_swarm_status.py", "--write-status"],
        root,
        runtime_dir / "latest_swarm_status.log",
        timeout_seconds,
    )


def preflight_step(root: Path, runtime_dir: Path, timeout_seconds: int) -> StepResult:
    output = runtime_dir / "latest_swarm_preflight.log"
    env = os.environ.copy()
    env.setdefault("ICC_PREFLIGHT_LOCAL_ONLY", "1")
    started = time.monotonic()
    result = subprocess.run(
        ["scripts/swarm_agent_preflight.sh"],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    data = result.stdout + result.stderr
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(data, encoding="utf-8")
    return StepResult(
        "swarm_preflight",
        result.returncode == 0,
        result.returncode,
        time.monotonic() - started,
        str(output.relative_to(root)),
        first_meaningful_line(data),
    )


def default_tsotchke_command() -> str | None:
    env_value = os.environ.get("TSOTCHKE_CHAN")
    if env_value:
        return env_value
    return shutil.which("tsotchke-chan")


def supervisor_prompt(root: Path, cycle: int) -> str:
    status_path = root / ".swarm" / "status.md"
    risks_path = root / ".swarm" / "risks.md"
    decisions_path = root / ".swarm" / "decisions.md"
    status = status_path.read_text(encoding="utf-8") if status_path.is_file() else "missing .swarm/status.md"
    risks = risks_path.read_text(encoding="utf-8") if risks_path.is_file() else "missing .swarm/risks.md"
    decisions = decisions_path.read_text(encoding="utf-8") if decisions_path.is_file() else "missing .swarm/decisions.md"
    return (
        "Eshkol swarm supervisor pass led by Tsotchke-chan. Stay inside the public Eshkol repo "
        "and its `.swarm/` ledger. Recommend exactly one next autonomous task by existing ESH or EKR id, "
        "or say BLOCKED with the required human decision. EKR tasks are platform/kernel slices and must "
        "be worked from public-master-based PR branches. Do not include private hostnames, IPs, usernames, "
        "credentials, or unsanitized machine-local evidence.\n\n"
        f"cycle={cycle}\n\n"
        "STATUS:\n"
        f"{status[-4000:]}\n\n"
        "RISKS:\n"
        f"{risks[-2000:]}\n\n"
        "DECISIONS:\n"
        f"{decisions[-2000:]}\n"
    )


def tsotchke_step(root: Path, runtime_dir: Path, timeout_seconds: int, cycle: int) -> StepResult:
    command = default_tsotchke_command()
    output_path = runtime_dir / "latest_tsotchke_chan.log"
    if command is None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("TSOTCHKE_CHAN command not found\n", encoding="utf-8")
        return StepResult("tsotchke_chan", False, 127, 0.0, str(output_path.relative_to(root)), "command not found")
    return run_command(
        "tsotchke_chan",
        [command, "talk", "--no-wait", "--json", supervisor_prompt(root, cycle)],
        root,
        output_path,
        timeout_seconds,
    )


def heartbeat_payload(
    cycle: int,
    mode: str,
    started_at: str,
    results: list[StepResult],
    interval_seconds: int,
) -> dict[str, object]:
    return {
        "schema": 1,
        "project": "eshkol",
        "mode": mode,
        "pid": os.getpid(),
        "cycle": cycle,
        "started_at": started_at,
        "updated_at": utc_now(),
        "interval_seconds": interval_seconds,
        "ok": all(result.ok for result in results),
        "checks": [
            {
                "name": result.name,
                "ok": result.ok,
                "returncode": result.returncode,
                "seconds": round(result.seconds, 3),
                "output": result.output_path,
                "summary": result.summary,
            }
            for result in results
        ],
    }


def run_cycle(args: argparse.Namespace, cycle: int) -> bool:
    root = args.root.resolve()
    runtime_dir = (args.runtime_dir or root / ".swarm" / "runtime").resolve()
    heartbeat = args.heartbeat or runtime_dir / "heartbeat.json"
    started_at = utc_now()
    results: list[StepResult] = [status_step(root, runtime_dir, args.command_timeout_seconds)]
    if args.with_preflight:
        results.append(preflight_step(root, runtime_dir, args.command_timeout_seconds))
    if args.with_tsotchke_chan:
        results.append(tsotchke_step(root, runtime_dir, args.command_timeout_seconds, cycle))

    payload = heartbeat_payload(cycle, args.mode, started_at, results, args.interval_seconds)
    write_json_atomic(heartbeat, payload)
    append_log(
        runtime_dir / "eshkol_swarm_daemon.log",
        json.dumps(
            {
                "ts": payload["updated_at"],
                "cycle": cycle,
                "ok": payload["ok"],
                "checks": [
                    {"name": result.name, "ok": result.ok, "returncode": result.returncode}
                    for result in results
                ],
            },
            sort_keys=True,
        ),
    )
    print(f"PROBE_OK eshkol_swarm_cycle={cycle}")
    print(f"PROBE_OK eshkol_swarm_cycle_ok={1 if payload['ok'] else 0}")
    return bool(payload["ok"])


def spawn(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    runtime_dir = (args.runtime_dir or root / ".swarm" / "runtime").resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    pid_file = args.pid_file or runtime_dir / "eshkol_swarm_daemon.pid"
    existing_pid = read_pid(pid_file)
    if existing_pid is not None and process_alive(existing_pid):
        fail(f"already_running pid={existing_pid}")
    child_args = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--root",
        str(root),
        "--runtime-dir",
        str(runtime_dir),
        "--pid-file",
        str(pid_file),
        "--interval-seconds",
        str(args.interval_seconds),
        "--command-timeout-seconds",
        str(args.command_timeout_seconds),
    ]
    if args.max_cycles:
        child_args.extend(["--max-cycles", str(args.max_cycles)])
    if args.with_preflight:
        child_args.append("--with-preflight")
    if args.with_tsotchke_chan:
        child_args.append("--with-tsotchke-chan")
    log_path = runtime_dir / "eshkol_swarm_daemon.stdout.log"
    with log_path.open("ab") as log:
        child = subprocess.Popen(
            child_args,
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_file.write_text(str(child.pid) + "\n", encoding="ascii")
    print(f"PROBE_OK eshkol_swarm_daemon_spawned={child.pid}")
    print(f"PROBE_OK eshkol_swarm_daemon_log={log_path.relative_to(root)}")
    return 0


def stop(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    runtime_dir = (args.runtime_dir or root / ".swarm" / "runtime").resolve()
    pid_file = args.pid_file or runtime_dir / "eshkol_swarm_daemon.pid"
    lock_path = runtime_dir / "eshkol_swarm_daemon.lock"
    pid = read_pid(pid_file)
    if pid is None:
        unlink_owned_lock(lock_path, -1)
        print("PROBE_OK eshkol_swarm_daemon_stopped=0")
        return 0
    if process_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except PermissionError:
            fail(f"stop_permission_denied pid={pid}")
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and process_alive(pid):
            time.sleep(0.1)
    if process_alive(pid):
        fail(f"stop_timeout pid={pid}")
    unlink_owned_lock(lock_path, pid)
    try:
        pid_file.unlink()
    except FileNotFoundError:
        pass
    print(f"PROBE_OK eshkol_swarm_daemon_stopped={pid}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--heartbeat", type=Path)
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--interval-seconds", type=int, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--command-timeout-seconds", type=int, default=DEFAULT_COMMAND_TIMEOUT_SECONDS)
    parser.add_argument("--with-preflight", action="store_true")
    parser.add_argument("--with-tsotchke-chan", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-cycles", type=int, default=0)
    parser.add_argument("--spawn", action="store_true")
    parser.add_argument("--stop", action="store_true")
    args = parser.parse_args()
    args.mode = "once" if args.once else "loop"

    if args.spawn:
        return spawn(args)
    if args.stop:
        return stop(args)
    if args.interval_seconds < 1:
        fail("bad_interval")
    if args.command_timeout_seconds < 1:
        fail("bad_command_timeout")

    root = args.root.resolve()
    runtime_dir = (args.runtime_dir or root / ".swarm" / "runtime").resolve()
    lock_path = runtime_dir / "eshkol_swarm_daemon.lock"
    with lock_file(lock_path, stale_seconds=max(args.command_timeout_seconds * 2, 60)):
        cycle = 1
        while True:
            ok = run_cycle(args, cycle)
            if args.once:
                return 0 if ok else 1
            if args.max_cycles and cycle >= args.max_cycles:
                return 0 if ok else 1
            cycle += 1
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
