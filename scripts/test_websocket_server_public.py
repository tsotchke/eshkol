#!/usr/bin/env python3
"""Exercise the public Eshkol WebSocket server API with a local RFC 6455 client."""
import argparse
import os
import pathlib
import socket
import subprocess
import tempfile
import time


GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
KEY = "dGhlIHNhbXBsZSBub25jZQ=="
MESSAGE = b"jit-aot-roundtrip"
BINARY_MESSAGE = b"public-binary"


def read_exact(sock, count):
    chunks = bytearray()
    while len(chunks) < count:
        part = sock.recv(count - len(chunks))
        if not part:
            raise RuntimeError("peer closed before the frame completed")
        chunks.extend(part)
    return bytes(chunks)


def run_client(port):
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.settimeout(8)
        header = (
            "GET /ws HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\n"
            "Connection: Upgrade\r\nSec-WebSocket-Version: 13\r\n"
            f"Sec-WebSocket-Key: {KEY}\r\n\r\n"
        ).encode("ascii")
        frame = client_frame(1, MESSAGE, b"\x11\x22\x33\x44")
        sock.sendall(header + frame)
        response = bytearray()
        while not response.endswith(b"\r\n\r\n"):
            response.extend(read_exact(sock, 1))
        text = response.decode("ascii")
        expected = "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
        if "101 Switching Protocols" not in text or f"Sec-WebSocket-Accept: {expected}" not in text:
            raise RuntimeError(f"unexpected upgrade response: {text!r}")
        opcode, payload = read_server_frame(sock)
        if opcode != 1 or payload != MESSAGE:
            raise RuntimeError("public Eshkol API returned a different text payload")
        sock.sendall(client_frame(1,b"",b"\x21\x32\x43\x54"))
        opcode, payload = read_server_frame(sock)
        if opcode != 1 or payload:
            raise RuntimeError("public Eshkol API did not echo an empty text frame")
        sock.sendall(client_frame(2,BINARY_MESSAGE,b"\x31\x42\x53\x64"))
        opcode, payload = read_server_frame(sock)
        if opcode != 2 or payload != BINARY_MESSAGE:
            raise RuntimeError("public Eshkol API did not echo the binary frame")
        close_payload=b"\x03\xe8"
        sock.sendall(client_frame(8,close_payload,b"\x41\x52\x63\x74"))
        opcode, payload = read_server_frame(sock)
        if opcode != 8 or payload != close_payload:
            raise RuntimeError("public Eshkol API did not complete the close handshake")


def client_frame(opcode, payload, mask):
    if len(payload) >= 126:
        raise ValueError("test frame payload must fit a short length")
    return bytes((0x80 | opcode, 0x80 | len(payload))) + mask + bytes(
        byte ^ mask[index & 3] for index, byte in enumerate(payload)
    )


def read_server_frame(sock):
    first, second = read_exact(sock,2)
    if second & 0x80:
        raise RuntimeError("server-to-client frame was masked")
    length=second & 0x7f
    if length==126:
        length=int.from_bytes(read_exact(sock,2),"big")
    elif length==127:
        length=int.from_bytes(read_exact(sock,8),"big")
    return first & 0x0f, read_exact(sock,length)


def run_mode(mode, runner, source, build_dir, test_file, root):
    with tempfile.TemporaryDirectory(prefix=f"eshkol-ws-{mode}-") as tmp:
        tmp_path = pathlib.Path(tmp)
        port_file = tmp_path / "port"
        env = os.environ.copy()
        env["ESHKOL_WS_SERVER_PORT_FILE"] = str(port_file)
        env["ESHKOL_JIT_CACHE_DIR"] = str(tmp_path / "jit-cache")
        command = [str(runner), "-I", str(source), "-L", str(runner.parent),
                   "-L", str(build_dir)]
        executable = None
        if mode == "jit":
            command.extend(["-r", str(test_file)])
        else:
            executable = tmp_path / "websocket-server-test"
            command.extend([str(test_file), "-o", str(executable)])
        compile_or_server = subprocess.Popen(
            command, cwd=root, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True
        )
        if mode == "aot":
            output, status = compile_or_server.communicate(timeout=180)
            if status:
                raise RuntimeError(f"AOT compile failed ({status}):\n{output}")
            compile_or_server = subprocess.Popen(
                [str(executable)], cwd=root, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True
            )
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not port_file.exists():
            if compile_or_server.poll() is not None:
                output = compile_or_server.communicate()[0]
                raise RuntimeError(f"{mode} server exited before publishing its port:\n{output}")
            time.sleep(0.02)
        if not port_file.exists():
            compile_or_server.kill()
            output = compile_or_server.communicate()[0]
            raise RuntimeError(f"{mode} server did not publish a port:\n{output}")
        try:
            run_client(int(port_file.read_text().strip()))
            output, status = compile_or_server.communicate(timeout=15)
        except Exception:
            if compile_or_server.poll() is None:
                compile_or_server.kill()
                compile_or_server.communicate()
            raise
        if status or "PUBLIC-WS-ROUNDTRIP-OK" not in output:
            raise RuntimeError(f"{mode} public round-trip failed ({status}):\n{output}")
        print(f"PASS: public Eshkol {mode.upper()} WebSocket round-trip")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eshkol-run", required=True, type=pathlib.Path)
    parser.add_argument("--source", required=True, type=pathlib.Path)
    parser.add_argument("--build-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()
    source = args.source.resolve()
    run_mode("jit", args.eshkol_run.resolve(), source, args.build_dir.resolve(),
             source / "tests/agent/websocket_server_public_roundtrip.esk", source)
    run_mode("aot", args.eshkol_run.resolve(), source, args.build_dir.resolve(),
             source / "tests/agent/websocket_server_public_roundtrip.esk", source)


if __name__ == "__main__":
    main()
