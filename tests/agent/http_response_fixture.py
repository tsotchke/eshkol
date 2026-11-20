"""Serve deterministic HTTP/1.x responses to a single engine test on loopback."""

import os
import socketserver
import subprocess
import sys
import threading
import time

STATUS = b"HTTP/1.1 200 OK\r\n"
CHUNKS = STATUS + b"Transfer-Encoding: Chunked\r\n\r\n"
RESPONSES = {
    "length": (b"HTTP/1.1 103 Early Hints\r\n\r\n" + STATUS
               + b"cOnTeNt-LeNgTh: 3\r\n\r\na\x00bextra"),
    "chunks": CHUNKS + b'4 ; a="b;c"\r\nWiki\r\n5\r\npedia\r\n0\r\nX-End: yes\r\n\r\n',
    "close": b"HTTP/1.0 200 OK\r\n\r\nclosed",
    "zero": STATUS + b"Content-Length: 0\r\n\r\n",
    "head": STATUS + b"Content-Length: 123456789\r\n\r\n",
    "cached": b"HTTP/1.1 304 Not Modified\r\nContent-Length: 50\r\n\r\n",
    "large": STATUS + b"Content-Length: 65537\r\n\r\n" + b"a" * 65537,
    "short": STATUS + b"Content-Length: 2\r\n\r\na",
    "invalid-size": STATUS + b"Content-Length: +1\r\n\r\n",
    "overflow": STATUS + b"Content-Length: 18446744073709551616\r\n\r\n",
    "duplicate": STATUS + b"Content-Length: 1\r\nContent-Length: 1\r\n\r\na",
    "list": STATUS + b"Content-Length: 1, 1\r\n\r\na",
    "both": STATUS + b"Content-Length: 0\r\nTransfer-Encoding: chunked\r\n\r\n",
    "coding": STATUS + b"Transfer-Encoding: gzip, chunked\r\n\r\n",
    "bad-status": b"HTTP/1.1 abc OK\r\n\r\n",
    "bad-field": STATUS + b"Bad Name: a\r\n\r\n",
    "bare-lf": b"HTTP/1.1 200 OK\n\n",
    "chunk-size": CHUNKS + b"0x0\r\n\r\n",
    "chunk-short": CHUNKS + b"3\r\nab",
    "chunk-ending": CHUNKS + b"2\r\nab!\r\n",
    "chunk-extension": CHUNKS + b'1;a="open\r\n',
    "trailer": CHUNKS + b"0\r\nContent-Length: 5\r\n\r\n",
    "trailer-eof": CHUNKS + b"0\r\nX-End: yes\r\n",
    "metadata-exact": STATUS + b"X: " + b"a" * (65536 - len(STATUS) - 7) + b"\r\n\r\nk",
    "header-limit": STATUS + b"X: " + b"a" * 65536 + b"\r\n\r\n",
    "trailer-limit": CHUNKS + b"0\r\nX: " + b"a" * 65536 + b"\r\n\r\n",
    "body-limit": STATUS + b"Content-Length: 10485761\r\n\r\n",
    "chunk-limit": CHUNKS + b"a00001\r\n",
    "close-exact": STATUS + b"\r\n",
    "close-limit": STATUS + b"\r\n",
    "close-timeout": STATUS + b"\r\npartial",
}


class Response(socketserver.StreamRequestHandler):
    def handle(self):
        path = self.rfile.readline().split()[1][1:].decode("ascii")
        while self.rfile.readline() not in (b"\r\n", b""):
            pass
        wire = RESPONSES[path]
        try:
            if path == "chunks":
                # Split CRLF pairs, extension syntax and data across writes.
                for byte in wire:
                    self.request.sendall(bytes([byte]))
                    time.sleep(0.001)
            else:
                self.request.sendall(wire)
            if path in ("close-exact", "close-limit"):
                block = b"x" * 65536
                for _ in range(160):
                    self.request.sendall(block)
                if path == "close-limit":
                    self.request.sendall(b"x")
            if path in ("zero", "chunks", "head", "close-timeout"):
                # A framed response completes while this connection stays open.
                self.request.settimeout(8)
                self.request.recv(1)
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == "__main__":
    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), Response) as server:
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        env = dict(os.environ, ESHKOL_VM_NO_DISASM="1",
                   ESHKOL_HTTP_RESPONSE_ORIGIN=f"http://127.0.0.1:{server.server_address[1]}")
        try:
            sys.exit(subprocess.run(sys.argv[1:], env=env).returncode)
        finally:
            server.shutdown()
