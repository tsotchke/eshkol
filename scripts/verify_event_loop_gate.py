#!/usr/bin/env python3
"""Exact ESH-0011 log and source/build evidence checks; --self-test is offline."""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

EXPECTED = (
"(make-event-loop 64) returns a handle, not #f", "the handle is a positive integer",
"max-events of 0 raises catchably", "make-pipe returns a (read-fd . write-fd) pair",
"the read end registers for read interest", "an idle registered fd reports no events at timeout 0",
"fd-write reports 4 bytes written to the write end", "poll reports exactly one ready descriptor",
"the ready descriptor is the pipe's read end", "the reported event is readable (bit 0)",
"readiness is level-triggered, not edge-triggered", "the write end registers for write interest",
"an empty pipe's write end is reported writable", "event-loop-remove-fd! reports #t for a watched descriptor",
"event-loop-remove-fd! reports #f for an unwatched descriptor", "a removed descriptor is no longer reported",
"an idle poll returns the empty list on timeout", "an idle poll actually waited for its timeout (>= 100 ms)",
"an idle poll returned rather than hanging (< 5000 ms)", "event-loop-close reports success",
"closing an already-closed handle reports #f, without raising", "polling a closed handle raises catchably",
"adding to a closed handle raises catchably", "a fresh loop opens after the previous one was closed",
"a stale handle is still rejected after its slot is reused", "the fresh loop is usable",
"1000 open/close cycles never exhaust the descriptor table")
TERMINAL = "PASS: event_loop_test"
DIAGNOSTIC = re.compile(r"^\s*(?:\[[^\]]+\]\s*)?(?:LLVM\s+)?(?:ERROR|FAIL|SKIP)\s*:|^\s*.*\b(?:JIT session error|error\s*:)", re.I | re.M)

class GateError(Exception): pass

def verify_log(path: Path, status: int, backend: str, *, diagnostics=True) -> None:
    if status != 0: raise GateError(f"command status={status}")
    try: text = path.read_text(errors="replace")
    except OSError as e: raise GateError(f"missing/unreadable log {path}: {e}")
    if diagnostics and DIAGNOSTIC.search(text): raise GateError("error/failure/skip diagnostic present")
    if f"event-loop backend: {backend}" not in text: raise GateError("expected actual backend absent")
    lines = re.findall(r"^PASS: (.*)$", text, re.M)
    if lines.count("event_loop_test") != 1 or lines[-1:] != ["event_loop_test"]:
        raise GateError("terminal PASS sentinel missing, duplicated, or not last")
    assertions = lines[:-1]
    if len(assertions) != 27 or tuple(assertions) != EXPECTED:
        raise GateError(f"assertion labels/order/count differ (observed {len(assertions)}; require exact pinned 27)")

def verify_compile_log(path: Path, status: int) -> None:
    if status != 0: raise GateError(f"compile command status={status}")
    try: text=path.read_text(errors="replace")
    except OSError as e: raise GateError(f"missing/unreadable compile log {path}: {e}")
    if DIAGNOSTIC.search(text): raise GateError("compile ERROR/FAIL/SKIP/JIT-session diagnostic present")

def self_test() -> int:
    good = "event-loop backend: kqueue\n" + "\n".join("PASS: "+x for x in EXPECTED) + "\n"+TERMINAL+"\n"
    import tempfile
    cases = [
        ("good", good, 0, True),
        ("missing", "", 0, False),
        ("skip", good+"SKIP: event loop unavailable\n", 0, False),
        ("renamed_terminal", good.replace(TERMINAL,"PASS: renamed"), 0, False),
        ("zero_assertions", "event-loop backend: kqueue\n"+TERMINAL+"\n", 0, False),
        ("echo_only_status_zero", "PASS: event_loop_test\n", 0, False),
        ("status_zero_error_prefix", good+"  [JIT] LLVM ERROR: synthetic\n", 0, False),
        ("compile_error_prefix", "  ERROR: synthetic link failure\n", 0, False),
        ("indented_error_colon", " [JIT] LLVM ERROR: synthetic\n", 0, False),
        ("wrong_backend", good.replace("kqueue", "epoll"), 0, False),
    ]
    failed=[]
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/"fixture.log"
        for name, text, status, expected in cases:
            p.write_text(text)
            try: verify_log(p,status,"kqueue"); got=True
            except GateError: got=False
            print(f"{'PASS' if got==expected else 'FAIL'} {name}")
            if got != expected: failed.append(name)
        p.unlink()
        try: verify_log(p,0,"kqueue"); got=True
        except GateError: got=False
        print(f"{'PASS' if not got else 'FAIL'} missing_file")
        if got: failed.append("missing_file")
    print(f"event-loop verifier controls: {len(cases)+1-len(failed)}/{len(cases)+1} passed")
    return bool(failed)

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--jit-log", type=Path); ap.add_argument("--jit-status", type=int, default=0)
    ap.add_argument("--compile-log", type=Path); ap.add_argument("--compile-status", type=int, default=0)
    ap.add_argument("--aot-log", type=Path); ap.add_argument("--aot-status", type=int, default=0)
    ap.add_argument("--backend", choices=("kqueue","epoll","iocp")); ap.add_argument("--self-test",action="store_true")
    a=ap.parse_args()
    if a.self_test: return self_test()
    if not all((a.jit_log,a.compile_log,a.aot_log,a.backend)): ap.error("all three logs and --backend are required")
    for name, path, rc in (("ORC-cache0",a.jit_log,a.jit_status),("AOT-compile",a.compile_log,a.compile_status),("AOT",a.aot_log,a.aot_status)):
        try:
            if name == "AOT-compile":
                verify_compile_log(path,rc)
            else: verify_log(path,rc,a.backend)
        except (OSError,GateError) as e: raise SystemExit(f"{name}: {e}")
    print(json.dumps({"result":"PASS","backend":a.backend,"counts":{"ORC-cache0":27,"AOT":27}},sort_keys=True))
    return 0
if __name__=="__main__": sys.exit(main())
