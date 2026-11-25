import base64
import concurrent.futures
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time


server = sys.argv[1]
source = "(define (square x) (* x x))"


def check_result(reply, session):
    assert reply["success"] is True, reply
    assert reply["session_id"] == session
    wasm = base64.b64decode(reply["wasm"], validate=True)
    assert wasm.startswith(b"\0asm") and len(wasm) == reply["size"]


def request(port, session):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=45)
    try:
        payload = json.dumps({"code": source, "session_id": session}, separators=(",", ":"))
        connection.request("POST", "/compile", payload, {"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 200, response.status
        check_result(json.load(response), session)
    finally:
        connection.close()


with tempfile.TemporaryDirectory(dir=".") as tmp:
    work = Path(tmp)
    (work / "source").write_text(source)
    (work / "session").write_text("worker")
    subprocess.run([server, "--compile-worker", tmp], check=True, timeout=40)
    check_result(json.loads((work / "response").read_text()), "worker")

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = dict(os.environ, ESHKOL_SERVER_TOKEN="", TMPDIR=str(work.resolve()))
    process = subprocess.Popen([server, "--port", str(port)], env=env)
    try:
        for _ in range(200):
            if process.poll() is not None:
                raise RuntimeError("server exited during startup")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.05)
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            list(pool.map(lambda session: request(port, session), ["one", "two"]))
    finally:
        process.terminate()
        process.wait(timeout=10)
