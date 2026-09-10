#!/usr/bin/env python3
#
# Copyright (C) tsotchke
#
# SPDX-License-Identifier: MIT
#
"""Reference driver for the eshkol-repl EREPL v1 machine-mode protocol.

Stdlib-only: no third-party dependencies, no PTY. Talks to a
`eshkol-repl --machine` child process purely over its stdin/stdout/stderr
pipes -- the same mechanism on macOS, Linux, and Windows. See
docs/reference/runtime/eshkol-repl.md ("Machine mode (EREPL protocol)") for
the full wire contract this module implements: EReplClient is a direct,
literal translation of that contract into Python, so anything that changes
here should change there too (and vice versa).

Library usage:

    from erepl_client import EReplClient

    with EReplClient("/path/to/eshkol-repl") as client:
        result = client.execute("(+ 1 2)")
        # {"stdout": "", "stderr": "", "value": "3", "value_type": "integer",
        #  "error": None}

Self-test (spawns the built eshkol-repl and exercises every request type,
including an interrupted infinite loop and a structured runtime error):

    python3 tools/erepl_client.py --self-test [--binary /path/to/eshkol-repl]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import queue
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Optional


class EReplError(RuntimeError):
    """The protocol itself misbehaved (process died, timed out, a frame
    could not be parsed). A Scheme-level error from evaluated code is NOT
    this -- execute() reports that in its returned dict's "error" field
    instead of raising, exactly as the wire protocol reports it: structured,
    never by matching text."""


class EReplClient:
    """Drives one `eshkol-repl --machine` child process over plain pipes.

    One EReplClient owns one child process for its whole lifetime -- start()
    spawns it and blocks until the EREPL v1 "ready" frame arrives, and every
    other method sends exactly one JSON request line and waits for the
    response frame carrying the same request id. Safe to use from a single
    thread at a time; concurrent callers must serialize their own calls (the
    protocol itself is a strict request/response cycle, not a pipeline).
    """

    def __init__(self, binary: str, args: Optional[list] = None,
                 ready_timeout: float = 120.0):
        self._binary = binary
        self._args = list(args or [])
        self._ready_timeout = ready_timeout
        self._proc: Optional[subprocess.Popen] = None
        self._id_counter = itertools.count(1)
        self._lock = threading.Lock()
        self._stdout_buf: list[str] = []
        self._stderr_diag_buf: list[str] = []
        self._pending: dict[str, "queue.Queue[dict]"] = {}
        self._ready_event = threading.Event()
        self.ready_info: Optional[dict] = None
        self._closed = False

    # ---- lifecycle ----------------------------------------------------

    def start(self) -> dict:
        """Spawns the child and waits for its EREPL v1 ready frame. Returns
        that frame (protocol_version / pid / eshkol_version)."""
        popen_kwargs: dict[str, Any] = dict(
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
        if sys.platform == "win32":
            # Own process group so interrupt() can target this child alone
            # with CTRL_BREAK_EVENT without also signaling the driver.
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["preexec_fn"] = os.setsid  # noqa: PLW1509 -- POSIX-only, no threads yet
        self._proc = subprocess.Popen(
            [self._binary, "--machine", *self._args], **popen_kwargs
        )

        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

        if not self._ready_event.wait(self._ready_timeout):
            self._close()
            raise EReplError("timed out waiting for the EREPL v1 ready frame")
        assert self.ready_info is not None
        return self.ready_info

    def shutdown(self, timeout: float = 10.0) -> None:
        """Asks the session to exit cleanly (op=shutdown), then closes the
        pipes and waits for the process. Safe to call more than once."""
        if self._closed:
            return
        try:
            if self._proc is not None and self._proc.poll() is None:
                self._request("shutdown", {}, timeout=timeout)
        except EReplError:
            pass
        finally:
            self._close()

    def _close(self) -> None:
        self._closed = True
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=5)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass

    def __enter__(self) -> "EReplClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

    # ---- reader threads -------------------------------------------------
    #
    # Both stdout and stderr must be drained continuously and off the main
    # thread: the child flushes after every frame, and an unread pipe fills
    # up and blocks the child's write -- which would otherwise deadlock a
    # caller that is itself blocked waiting for that same write to finish.

    def _read_stdout(self) -> None:
        f = self._proc.stdout
        assert f is not None
        for line in iter(f.readline, ""):
            with self._lock:
                self._stdout_buf.append(line)

    def _read_stderr(self) -> None:
        f = self._proc.stderr
        assert f is not None
        for line in iter(f.readline, ""):
            stripped = line.rstrip("\n")
            if stripped == "EREPL READY":
                continue  # legacy sentinel; the v1 "ready" frame carries the detail
            if stripped in ("EREPL DONE", "EREPL FAIL"):
                continue  # legacy per-eval sentinel; superseded by the result frame
            if stripped.startswith("EREPL/1 "):
                self._dispatch_frame(stripped[len("EREPL/1 "):])
                continue
            # Anything else is REPL diagnostic text (stdlib load banners,
            # a runtime warning, ...), not protocol framing.
            with self._lock:
                self._stderr_diag_buf.append(line)

    def _dispatch_frame(self, body: str) -> None:
        try:
            frame = json.loads(body)
        except json.JSONDecodeError:
            return  # a malformed frame from the REPL itself; nothing to pair it to
        if frame.get("type") == "ready":
            self.ready_info = frame
            self._ready_event.set()
            return
        req_id = frame.get("id")
        with self._lock:
            q = self._pending.get(req_id)
        if q is not None:
            q.put(frame)

    # ---- request/response -----------------------------------------------

    def _next_id(self) -> str:
        return str(next(self._id_counter))

    def _request(self, op: str, fields: dict, timeout: float) -> dict:
        if self._proc is None or self._proc.poll() is not None:
            raise EReplError("the eshkol-repl process is not running")
        req_id = self._next_id()
        q: "queue.Queue[dict]" = queue.Queue(maxsize=1)
        with self._lock:
            self._pending[req_id] = q
        payload = {"id": req_id, "op": op}
        payload.update(fields)
        line = json.dumps(payload)
        try:
            assert self._proc.stdin is not None
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise EReplError(f"eshkol-repl closed its stdin: {e}") from e
        try:
            frame = q.get(timeout=timeout)
        except queue.Empty as e:
            raise EReplError(
                f"timed out waiting for a response to {op!r} (id={req_id})"
            ) from e
        finally:
            with self._lock:
                self._pending.pop(req_id, None)
        return frame

    # ---- public API -------------------------------------------------------

    def execute(self, code: str, timeout: float = 30.0) -> dict:
        """Evaluates exactly one top-level Eshkol form.

        Returns {"stdout": str, "stderr": str, "value": str | None,
        "value_type": str | None, "error": dict | None}. "value" is the
        form's own return value in R7RS `write` form (never conflated with
        anything the form printed itself, which lands in "stdout" instead).
        "error", when the form raised or failed to evaluate, is
        {"kind", "message", "line", "column", "filename", "printed",
        "irritants"} -- see the docs for the full, closed set of "kind"
        values (never wording).

        "stdout" comes straight from the response frame's own "stdout"
        field, NOT from racing the raw stdout pipe against the stderr
        response frame -- those are two independent OS pipes with no
        ordering guarantee between them, even though the server always
        writes stdout before its response frame. "stderr" is best-effort:
        any REPL diagnostic lines (not protocol frames) seen on the single,
        inherently-ordered stderr stream since the previous call returned.
        """
        with self._lock:
            stderr_start = len(self._stderr_diag_buf)
        frame = self._request("eval", {"code": code}, timeout=timeout)
        with self._lock:
            stderr = "".join(self._stderr_diag_buf[stderr_start:])
        stdout = frame.get("stdout", "")
        if frame.get("ok"):
            return {
                "stdout": stdout,
                "stderr": stderr,
                "value": frame.get("value"),
                "value_type": frame.get("value_type"),
                "error": None,
            }
        return {
            "stdout": stdout,
            "stderr": stderr,
            "value": None,
            "value_type": None,
            "error": frame.get("error"),
        }

    def interrupt(self) -> None:
        """Aborts whatever evaluation is currently in flight (a no-op if the
        session is idle). Delivered out-of-band as a signal -- SIGINT on
        POSIX, CTRL_BREAK_EVENT on Windows -- because the child's stdin
        reader is blocked inside the evaluation and cannot be reached by a
        request frame; the aborted execute() call returns normally with
        error.kind == "interrupted", and the session stays usable for the
        next request."""
        if self._proc is None or self._proc.poll() is not None:
            return
        if sys.platform == "win32":
            self._proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.kill(self._proc.pid, signal.SIGINT)

    def complete(self, prefix: str, timeout: float = 10.0) -> list:
        """Returns candidate identifier names (builtins plus this session's
        own definitions) starting with `prefix`, sorted and deduplicated."""
        frame = self._request("complete", {"prefix": prefix}, timeout=timeout)
        return frame.get("matches", [])

    def is_complete(self, code: str, timeout: float = 10.0) -> str:
        """Returns "complete", "incomplete", or "invalid"."""
        frame = self._request("is_complete", {"code": code}, timeout=timeout)
        return frame.get("status", "invalid")

    def reset(self, timeout: float = 10.0) -> bool:
        """Clears this session's tracked-definitions bookkeeping (tab
        completion, etc). Per-session JIT symbols are NOT undone -- same
        caveat as the interactive `:reset` command."""
        frame = self._request("reset", {}, timeout=timeout)
        return bool(frame.get("ok"))


# =============================================================================
# Self-test
# =============================================================================

def _default_binary() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    build_dir = os.environ.get("BUILD_DIR", "build")
    exe_name = "eshkol-repl.exe" if sys.platform == "win32" else "eshkol-repl"
    return str(repo_root / build_dir / exe_name)


class _Check:
    def __init__(self):
        self.count = 0

    def __call__(self, condition: bool, description: str, detail: str = "") -> None:
        self.count += 1
        if not condition:
            msg = f"FAIL: {description}"
            if detail:
                msg += f"\n  {detail}"
            print(msg)
            sys.exit(1)


def _self_test(binary: str) -> int:
    if not os.path.exists(binary):
        print(f"SKIP: {binary} not built")
        return 0

    check = _Check()
    client = EReplClient(binary)
    ready = client.start()
    check(ready.get("protocol_version") == 1, "ready frame announces protocol_version 1",
          repr(ready))
    check(isinstance(ready.get("pid"), int) and ready["pid"] > 0,
          "ready frame announces a pid", repr(ready))

    try:
        # -- basic eval: value distinct from stdout -----------------------
        r = client.execute("(+ 1 2)")
        check(r["error"] is None and r["value"] == "3" and r["value_type"] == "integer",
              "(+ 1 2) evaluates to value=3/integer with no error", repr(r))
        check(r["stdout"] == "", "(+ 1 2) writes nothing to stdout", repr(r))

        r = client.execute('(display (* 6 7))')
        check(r["error"] is None, "(display (* 6 7)) evaluates without error", repr(r))
        check(r["stdout"] == "42", "explicit display output lands in stdout, exactly",
              repr(r))

        # -- structured runtime error, classified by kind, not wording ----
        r = client.execute("(car (quote ()))")
        check(r["error"] is not None and r["error"]["kind"] == "type-error",
              "(car '()) fails with a structured kind=type-error payload", repr(r))
        check("message" in r["error"] and "printed" in r["error"] and
              "irritants" in r["error"],
              "the error payload carries message/printed/irritants fields", repr(r))

        # -- parse error -----------------------------------------------------
        r = client.execute(")))")
        check(r["error"] is not None and r["error"]["kind"] == "parse-error",
              "malformed source fails with kind=parse-error", repr(r))

        # -- is_complete -------------------------------------------------
        check(client.is_complete("(display 1") == "incomplete",
              "is_complete detects an unbalanced open form")
        check(client.is_complete("(display 1)") == "complete",
              "is_complete detects a balanced form")
        check(client.is_complete(")))") == "invalid",
              "is_complete detects an unmatched closing paren")

        # -- completion ----------------------------------------------------
        matches = client.complete("string-appe")
        check("string-append" in matches,
              "complete() finds a builtin by prefix", repr(matches))

        r = client.execute("(define erepl-self-test-var 42)")
        check(r["error"] is None, "defining a variable succeeds", repr(r))
        matches = client.complete("erepl-self-test-")
        check("erepl-self-test-var" in matches,
              "complete() finds a session-defined symbol", repr(matches))

        # -- reset -----------------------------------------------------------
        check(client.reset() is True, "reset() acknowledges ok")

        # -- interrupt: abort a runaway evaluation, session stays usable ----
        result_box: dict = {}

        def run_infinite_loop():
            result_box["result"] = client.execute(
                "(let loop () (loop))", timeout=60.0
            )

        t = threading.Thread(target=run_infinite_loop)
        t.start()
        t.join(timeout=2.0)  # give the loop time to actually start spinning
        client.interrupt()
        t.join(timeout=30.0)
        check(not t.is_alive(), "an interrupted infinite loop returns rather than hanging")
        r = result_box.get("result")
        check(r is not None and r["error"] is not None and
              r["error"]["kind"] == "interrupted",
              "interrupt() aborts the loop with kind=interrupted", repr(r))

        # Session must still be usable after the interrupt.
        r = client.execute("(+ 40 2)")
        check(r["error"] is None and r["value"] == "42",
              "the session evaluates normally again after an interrupt", repr(r))
    finally:
        client.shutdown()

    print(f"PASS: EREPL v1 self-test ({check.count} checks)")
    return 0


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", default=None,
                         help="Path to the eshkol-repl executable "
                              "(default: <repo>/$BUILD_DIR/eshkol-repl, BUILD_DIR defaults to 'build')")
    parser.add_argument("--self-test", action="store_true",
                         help="Run the built-in protocol self-test against --binary and exit")
    args = parser.parse_args(argv)

    binary = args.binary or _default_binary()

    if args.self_test:
        return _self_test(binary)

    parser.error("nothing to do: pass --self-test, or import EReplClient as a library")
    return 2


if __name__ == "__main__":
    sys.exit(main())
