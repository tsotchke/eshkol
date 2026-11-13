#!/usr/bin/env python3
"""Run the Scheme TLS client against a one-shot Python SSL loopback peer."""
import argparse
import os
from pathlib import Path
import socket
import ssl
import subprocess
import threading
import time

PAYLOAD = b"A\x00\xc3\xa9"
REPLY = b"R\x00\xc3\xa9"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--mode", choices=("jit", "aot"), required=True)
    parser.add_argument("--scenario", choices=("normal", "fatal"), default="normal")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--lib-dir", required=True)
    parser.add_argument("--link-dir", required=True)
    args = parser.parse_args()

    source = Path(args.source).resolve()
    root = source.parents[2]
    cert = root / "tests/network/fixtures/localhost-cert.pem"
    key = root / "tests/network/fixtures/localhost-key.pem"
    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["ESHKOL_PATH"] = str(Path(args.lib_dir).resolve())
    env["ESHKOL_JIT_CACHE"] = "0"
    executable = Path(args.runtime).resolve()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(10)
    port = listener.getsockname()[1]
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(certfile=cert, keyfile=key)
    failures = []

    def serve_once():
        try:
            raw, _ = listener.accept()
            with raw:
                incoming = ssl.MemoryBIO()
                outgoing = ssl.MemoryBIO()
                stream = server_context.wrap_bio(incoming, outgoing, server_side=True)

                def flush_records():
                    while True:
                        chunk = outgoing.read()
                        if not chunk:
                            return
                        raw.sendall(chunk)

                def receive_record_bytes():
                    chunk = raw.recv(16384)
                    if not chunk:
                        raise EOFError("peer closed during TLS record")
                    incoming.write(chunk)

                while True:
                    try:
                        stream.do_handshake()
                        flush_records()
                        break
                    except ssl.SSLWantReadError:
                        flush_records()
                        receive_record_bytes()
                    except ssl.SSLWantWriteError:
                        flush_records()

                if args.scenario == "fatal":
                    raw.sendall(b"\x17\x03\x03\x00\x01\xff")
                    raw.settimeout(3)
                    records = bytearray()
                    while True:
                        extra = raw.recv(4096)
                        if not extra:
                            break
                        records.extend(extra)
                    offset = 0
                    record_count = 0
                    while offset < len(records):
                        if len(records) - offset < 5:
                            raise AssertionError("client ended with a partial TLS record")
                        record_length = int.from_bytes(records[offset + 3:offset + 5], "big")
                        offset += 5 + record_length
                        if offset > len(records):
                            raise AssertionError("client ended with a truncated TLS record")
                        record_count += 1
                    # SSL_read may itself emit one fatal alert. SSL_shutdown
                    # after that fatal error would attempt an additional record.
                    if record_count > 1:
                        raise AssertionError("client performed extra TLS I/O after fatal SSL error")
                    return

                received = bytearray()
                while len(received) < len(PAYLOAD):
                    try:
                        block = stream.read(32)
                        if block:
                            received.extend(block)
                    except ssl.SSLWantReadError:
                        flush_records()
                        receive_record_bytes()
                if bytes(received) != PAYLOAD:
                    raise AssertionError(f"unexpected Scheme payload {bytes(received)!r}")
                assert stream.write(REPLY) == len(REPLY)
                encrypted_reply = bytearray()
                while True:
                    block = outgoing.read()
                    if not block:
                        break
                    encrypted_reply.extend(block)
                split = max(1, len(encrypted_reply) // 2)
                raw.sendall(encrypted_reply[:split])
                time.sleep(0.25)
                raw.sendall(encrypted_reply[split:])
        except BaseException as exc:
            failures.append(exc)
        finally:
            listener.close()

    source_text = source.read_text()
    cert_literal = '"' + str(cert).replace("\\", "\\\\").replace('"', '\\"') + '"'
    run_source = work / f"tls_public_python_{args.mode}.esk"
    run_source.write_text(source_text.replace("__TLS_PORT__", str(port)).replace(
        "__TLS_CERT__", cert_literal))
    output = work / "tls_public_python_aot"
    if args.mode == "aot":
        build = subprocess.run(
            [str(executable), "-o", str(output), str(run_source), f"-L{Path(args.link_dir).resolve()}"],
            cwd=root, env=env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=180, check=False,
        )
        if build.returncode:
            print(build.stdout, end="")
            return build.returncode
    peer = threading.Thread(target=serve_once, daemon=True)
    peer.start()
    if args.mode == "jit":
        command = [str(executable), "-r", str(run_source)]
    else:
        command = [str(output)]

    result = subprocess.run(
        command, cwd=root, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=60, check=False,
    )
    print(result.stdout, end="")
    peer.join(timeout=12)
    if peer.is_alive():
        raise TimeoutError("Python TLS peer did not finish")
    if failures:
        raise failures[0]
    if result.returncode:
        return result.returncode
    expected_marker = "TLS Python public API:" if args.scenario == "normal" else "PASS: fatal TLS errors"
    if expected_marker not in result.stdout:
        raise AssertionError("public TLS test omitted its final result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
