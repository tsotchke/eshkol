#!/usr/bin/env python3
"""JIT/AOT strict-resource loopback check using fresh ephemeral ports."""
import argparse
import os
import socketserver
import subprocess
import tempfile
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent


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


def run_case(runner: Path, lib_dir: Path, mode: str):
    received = []
    server = socketserver.TCPServer(("127.0.0.1", 0), PongHandler)
    server.timeout = 5
    server.received = received
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    source_root = HERE.parents[1]
    module_root = HERE / "modules"
    env = os.environ.copy()
    env["ESHKOL_LIB_DIR"] = str(lib_dir)
    env["ESHKOL_PATH"] = os.pathsep.join((str(module_root), str(source_root / "lib")))
    env["ESHKOL_JIT_CACHE"] = "0"
    try:
        with tempfile.TemporaryDirectory(prefix=f"strict-resources-{mode}-") as temp:
            temp_dir = Path(temp)
            source = temp_dir / "loopback.esk"
            template = (HERE / "fixtures" / "positive-loopback.esk").read_text()
            source.write_text(template.replace("@PORT@", str(port)))
            if mode == "jit":
                command = [str(runner), "--strict-resources", "-r", str(source), "-L", str(lib_dir)]
                completed = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, timeout=25)
            else:
                executable = temp_dir / "loopback-aot"
                compile_command = [str(runner), "--strict-resources", "-O0", "-o",
                                   str(executable), str(source), "-L", str(lib_dir)]
                compiled = subprocess.run(compile_command, env=env, text=True,
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          timeout=45)
                if compiled.returncode != 0:
                    raise RuntimeError(f"AOT compile failed ({compiled.returncode}):\n{compiled.stdout}")
                completed = subprocess.run([str(executable)], env=env, text=True,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           timeout=15)
            if completed.returncode != 0 or "RESULT: PASS" not in completed.stdout:
                raise RuntimeError(f"{mode} failed ({completed.returncode}):\n{completed.stdout}")
            if received != [b"ping"]:
                raise RuntimeError(f"{mode} peer received {received!r}, expected [b'ping']")
            print(f"PASS strict-resource {mode} loopback ephemeral-port={port}")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--lib-dir", type=Path, required=True)
    args = parser.parse_args()
    run_case(args.runner, args.lib_dir, "jit")
    run_case(args.runner, args.lib_dir, "aot")


if __name__ == "__main__":
    main()
