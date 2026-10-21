#!/usr/bin/env python3
"""JIT/AOT strict-resource positives with ephemeral localhost endpoints."""
import argparse
import os
import socketserver
import subprocess
import tempfile
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = ("loopback", "borrow", "both-branches", "scalar-shadow")


class PongHandler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(5)
        chunks = []
        total = 0
        while total < 4:
            chunk = self.request.recv(4 - total)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        payload = b"".join(chunks)
        self.server.received.append(payload)
        self.request.sendall(b"pong")


def run_route(runner: Path, lib_dir: Path, env: dict, source: Path,
              mode: str, marker: str, received: list | None = None):
    with tempfile.TemporaryDirectory(prefix=f"strict-resources-{mode}-") as temp:
        executable = Path(temp) / "strict-positive-aot"
        if mode == "jit":
            command = [str(runner), "--strict-resources", "-r", str(source), "-L", str(lib_dir)]
            completed = subprocess.run(command, env=env, text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       timeout=30)
        else:
            compile_command = [str(runner), "--strict-resources", "-O0", "-o",
                               str(executable), str(source), "-L", str(lib_dir)]
            compiled = subprocess.run(compile_command, env=env, text=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      timeout=60)
            if compiled.returncode != 0:
                raise RuntimeError(f"AOT compile failed ({compiled.returncode}):\n{compiled.stdout}")
            completed = subprocess.run([str(executable)], env=env, text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       timeout=20)
        if received is not None and received != [b"ping"]:
            raise RuntimeError(f"{mode} peer received {received!r}, expected [b'ping']; "
                               f"exit={completed.returncode}, output={completed.stdout!r}")
        if completed.returncode != 0 or marker not in completed.stdout:
            raise RuntimeError(f"{mode} failed ({completed.returncode}, peer={received!r}):\n{completed.stdout}")


def run_case(case: str, runner: Path, lib_dir: Path):
    source_root = HERE.parents[1]
    module_root = HERE / "modules"
    env = os.environ.copy()
    env["ESHKOL_LIB_DIR"] = str(lib_dir)
    env["ESHKOL_PATH"] = os.pathsep.join((str(source_root / "lib"), str(module_root)))
    env["ESHKOL_JIT_CACHE"] = "0"

    if case == "loopback":
        template = (HERE / "fixtures" / "positive-loopback.esk").read_text()
        marker = "RESULT: PASS"
        for mode in ("jit", "aot"):
            server = socketserver.TCPServer(("127.0.0.1", 0), PongHandler)
            server.timeout = 5
            server.received = []
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            source_fd, source_path = tempfile.mkstemp(
                prefix=f"strict-loopback-{mode}-", suffix=".esk")
            os.close(source_fd)
            source = Path(source_path)
            source.write_text(template.replace("@PORT@", str(server.server_address[1])))
            try:
                run_route(runner, lib_dir, env, source, mode, marker, server.received)
                print(f"PASS {case} {mode} marker={marker} port={server.server_address[1]}")
            finally:
                source.unlink(missing_ok=True)
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
        return

    names = {
        "borrow": ("positive-borrow-scope", "RESULT: PASS"),
        "both-branches": ("positive-both-branches-close", "BOTH-BRANCHES: PASS"),
        "scalar-shadow": ("positive-shadowed-factory", "SHADOW-OK"),
    }
    fixture, marker = names[case]
    source = HERE / "fixtures" / f"{fixture}.esk"
    for mode in ("jit", "aot"):
        run_route(runner, lib_dir, env, source, mode, marker)
        print(f"PASS {case} {mode} marker={marker}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--lib-dir", type=Path, required=True)
    parser.add_argument("--case", action="append", choices=CASES,
                        help="run one positive case (repeatable; default: all)")
    args = parser.parse_args()
    selected = args.case if args.case is not None else CASES
    if not selected:
        parser.error("at least one --case is required")
    for case in selected:
        run_case(case, args.runner, args.lib_dir)


if __name__ == "__main__":
    main()
