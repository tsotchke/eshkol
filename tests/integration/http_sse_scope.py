"""Drive Scheme SSE lifetime cases with independently observed TCP responses."""

import argparse
import os
from pathlib import Path
import selectors
import socket
import subprocess
import tempfile
import time


def read_port(process):
    output = bytearray()
    deadline = time.monotonic() + 60
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        while time.monotonic() < deadline:
            if not selector.select(deadline - time.monotonic()):
                break
            chunk = os.read(process.stdout.fileno(), 65536)
            if not chunk:
                break
            output.extend(chunk)
            for line in output.split(b"\n")[:-1]:
                if line.startswith(b"SSE-PORT "):
                    return int(line.split()[1])
    raise AssertionError(f"Scheme fixture did not announce its port: {output!r}")


def exchange(port):
    family, kind, proto, _, address = socket.getaddrinfo(
        None, port, socket.AF_INET, socket.SOCK_STREAM
    )[0]
    with socket.socket(family, kind, proto) as client:
        client.settimeout(5)
        client.connect(address)
        host = address[0].encode("ascii")
        client.sendall(b"GET /events HTTP/1.1\r\nHost: " + host + b"\r\n\r\n")
        response = bytearray()
        while chunk := client.recv(512):
            response.extend(chunk)
            if len(response) > 2048:
                raise AssertionError("unexpected HTTP response length")
        return bytes(response)


def sse_body(response):
    headers, separator, body = response.partition(b"\r\n\r\n")
    assert separator and headers.startswith(b"HTTP/1.1 200 OK\r\n"), response
    fields = set(headers.lower().split(b"\r\n"))
    assert b"content-type: text/event-stream" in fields, response
    assert b"transfer-encoding: chunked" in fields, response
    return body


def check_responses(port):
    event = "1f\r\ndata: café\ndata: second line\n\n\r\n0\r\n\r\n".encode()
    assert sse_body(exchange(port)) == event
    assert sse_body(exchange(port)) == b"", "normal return sent the pending event"
    assert sse_body(exchange(port)) == b"", "exception sent the pending terminal chunk"
    assert sse_body(exchange(port)) == b"", "continuation escape sent the pending event"
    assert exchange(port) == b"", "cancellation sent the pending headers"
    response = exchange(port)
    assert response.startswith(b"HTTP/1.1 200 OK\r\n"), response
    assert response.endswith(b"\r\n\r\nordinary"), response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True)
    parser.add_argument("--link-dir", required=True)
    parser.add_argument("--mode", choices=("jit", "aot"), required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / "agent" / "http_sse_scope.esk"
    env = dict(os.environ, ESHKOL_JIT_CACHE="0")

    with tempfile.TemporaryDirectory(prefix="eshkol-sse-scope-") as directory:
        binary = Path(directory) / "scope"
        if args.mode == "aot":
            subprocess.run(
                [args.runner, "-O0", "-o", str(binary), str(source), "-L" + args.link_dir],
                check=True, timeout=180, env=env,
            )
        command = (
            [args.runner, "-r", str(source), "-L" + args.link_dir]
            if args.mode == "jit" else [str(binary)]
        )
        with subprocess.Popen(
            command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        ) as process:
            try:
                check_responses(read_port(process))
                output, _ = process.communicate(timeout=10)
                assert process.returncode == 0 and b"RESULT: PASS" in output, output
            except Exception:
                if process.poll() is None:
                    process.kill()
                output, _ = process.communicate()
                if output:
                    print(output.decode(errors="replace"), end="")
                raise
    print(f"RESULT: PASS ({args.mode} Scheme SSE scopes)")


if __name__ == "__main__":
    main()
