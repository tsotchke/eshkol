#!/usr/bin/env python3
"""Live loopback verification for agent.http bounded responses, task-local
response caching, and SHA-256 citation provenance records.
"""
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = Path(os.environ.get("BUILD_DIR", str(REPO_ROOT / "build")))
ESHKOL_RUN = BUILD_DIR / "eshkol-run"


class _State:
    def __init__(self):
        self.hits = 0


def _start_server():
    state = _State()
    small_body = b'{"dataset":"eshkol-v14","note":"caf\xc3\xa9"}'
    large_body = b"x" * 2048

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state.hits += 1
            if self.path == "/small":
                payload = small_body
            elif self.path == "/large":
                payload = large_body
            else:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port, state, small_body


class HttpBoundedCacheProvenanceTest(unittest.TestCase):
    def test_bounded_cache_and_provenance_roundtrip(self):
        self.assertTrue(ESHKOL_RUN.is_file(), f"missing {ESHKOL_RUN}")
        server, port, state, small_body = _start_server()
        expected_sha = hashlib.sha256(small_body).hexdigest()
        try:
            with tempfile.TemporaryDirectory(prefix="eshkol-http-prov-") as td:
                work = Path(td)
                src = work / "http_prov_test.esk"
                exe = work / "http_prov_test"
                src.write_text(
                    f"""(require agent.http)
(http-init)
(define url-small "http://127.0.0.1:{port}/small")
(define url-large "http://127.0.0.1:{port}/large")

;; 1. Bounded request admits payload <= max-bytes.
(define r-ok (http-get-bounded url-small 256 5000))
(unless (and (pair? r-ok) (= (car r-ok) 200))
  (display "FAIL: bounded ok") (newline) (exit 1))

;; 2. Bounded request refuses payload > max-bytes with catchable error.
(define over-refused
  (guard (e ((error-object? e)
             (string-contains? (error-object-message e) "max-bytes"))
            (#t #f))
    (http-get-bounded url-large 256 5000)
    #f))
(unless over-refused
  (display "FAIL: over-budget not refused") (newline) (exit 1))

;; 3. Task-local cache & SHA-256 provenance record.
(define cache-a (make-http-cache 8 60000))
(define cache-b (make-http-cache 8 60000))
(define p1 (http-fetch-provenance cache-a url-small 256 5000))
(define p2 (http-fetch-provenance cache-a url-small 256 5000))
(define p3 (http-fetch-provenance cache-b url-small 256 5000))

(unless (and (not (cdr (assoc "cached" p1)))
             (cdr (assoc "cached" p2))
             (not (cdr (assoc "cached" p3)))
             (= (cdr (assoc "status" p1)) 200)
             (= (cdr (assoc "bytes" p1)) {len(small_body)})
             (string=? (cdr (assoc "sha256" p1)) "{expected_sha}")
             (string=? (cdr (assoc "sha256" p2)) "{expected_sha}")
             (= (http-cache-size cache-a) 1)
             (= (http-cache-size cache-b) 1))
  (display "FAIL: provenance/cache mismatch") (newline) (exit 1))

;; 4. Cache clear invalidates entry.
(http-cache-clear! cache-a)
(unless (= (http-cache-size cache-a) 0)
  (display "FAIL: cache clear") (newline) (exit 1))
(define r-cached (http-get-cached cache-a url-small 256 5000))
(unless (and (pair? r-cached) (= (car r-cached) 200))
  (display "FAIL: http-get-cached") (newline) (exit 1))

(http-shutdown)
(display "PASS: http_bounded_cache_provenance") (newline)
""",
                    encoding="utf-8",
                )
                comp = subprocess.run(
                    [str(ESHKOL_RUN), "-I", str(REPO_ROOT / "lib"), "-L", str(BUILD_DIR), str(src), "-o", str(exe)],
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self.assertEqual(comp.returncode, 0, msg=comp.stderr + comp.stdout)
                run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
                self.assertEqual(run.returncode, 0, msg=run.stderr + run.stdout)
                self.assertIn("PASS: http_bounded_cache_provenance", run.stdout)
                # Server hits: 1 (r-ok) + 1 (large refused) + 1 (p1 miss; p2 was cache hit!)
                # + 1 (p3 isolated cache-b miss) + 1 (post-clear miss) = 5 hits total.
                self.assertEqual(state.hits, 5, f"expected 5 wire hits (p2 cached), got {state.hits}")
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
