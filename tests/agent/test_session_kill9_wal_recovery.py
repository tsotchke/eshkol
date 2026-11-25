#!/usr/bin/env python3
"""Live SIGKILL mid-WAL-commit durability verification for agent.session.

Spawns a compiled Eshkol writer performing rapid transactional session-append!
and session-checkpoint! calls against a SQLite WAL+FULL database, interrupts it
with SIGKILL mid-stream across multiple cycles, and verifies:
  1. Near-instant cold-reopen recovery time.
  2. Complete SHA-256 hash-chain and checkpoint integrity (zero torn/partial
     writes, zero lost committed sequence numbers).
  3. Clean lock release allowing immediate subsequent writes.
"""
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = Path(os.environ.get("BUILD_DIR", str(REPO_ROOT / "build")))
ESHKOL_RUN = BUILD_DIR / "eshkol-run"


WRITER_SOURCE = r"""
(require agent.session)

(define db-path (getenv "ESHKOL_WAL_DB"))
(define start-count (string->number (getenv "ESHKOL_WAL_TARGET")))

(with-session-store db-path
  (lambda (db)
    (unless (session-exists? db "wal-kill9")
      (session-create! db "wal-kill9" "{\"kind\":\"wal-durability\"}"))
    (let loop ((i 0))
      (when (< i start-count)
        (let ((seq (session-append!
                     db "wal-kill9" "worker"
                     (string-append "event-payload-" (number->string i)))))
          (when (= (modulo seq 5) 0)
            (session-checkpoint!
              db "wal-kill9"
              (string-append "{\"checkpoint_seq\":" (number->string seq) "}")))
          (display "COMMITTED:")
          (display seq)
          (newline)
          (flush-output-port (current-output-port)))
        (loop (+ i 1))))))
"""


VERIFIER_SOURCE = r"""
(require agent.session)

(define db-path (getenv "ESHKOL_WAL_DB"))

(with-session-store db-path
  (lambda (db)
    (unless (session-exists? db "wal-kill9")
      (display "FAIL: session missing after SIGKILL") (newline) (exit 1))
    (unless (session-verify! db "wal-kill9")
      (display "FAIL: session-verify! rejected chain after SIGKILL") (newline) (exit 1))
    (let* ((events (session-events db "wal-kill9"))
           (n (length events)))
      ;; Verify strictly contiguous sequence numbers 1..n.
      (let check ((rest events) (expected 1))
        (when (pair? rest)
          (let ((seq (car (car rest))))
            (unless (= seq expected)
              (display "FAIL: sequence gap") (newline) (exit 1))
            (check (cdr rest) (+ expected 1)))))
      ;; Verify database lock is completely free by appending a new event immediately.
      (let ((next-seq (session-append! db "wal-kill9" "verifier" "post-recovery-ok")))
        (unless (= next-seq (+ n 1))
          (display "FAIL: unexpected post-recovery seq") (newline) (exit 1))
        (unless (session-verify! db "wal-kill9")
          (display "FAIL: session-verify! failed after post-recovery append") (newline) (exit 1))
        (display "VERIFIED_PRE:") (display n) (newline)
        (display "VERIFIED_POST:") (display next-seq) (newline)))))
"""


class SessionKill9WalRecoveryTest(unittest.TestCase):
    def test_repeated_sigkill_mid_wal_commit_recovery(self):
        self.assertTrue(ESHKOL_RUN.is_file(), f"missing {ESHKOL_RUN}")
        with tempfile.TemporaryDirectory(prefix="eshkol-wal-kill9-") as td:
            work = Path(td)
            writer_src = work / "wal_writer.esk"
            verifier_src = work / "wal_verifier.esk"
            writer_exe = work / "wal_writer"
            verifier_exe = work / "wal_verifier"
            db_path = work / "session_wal.db"

            writer_src.write_text(WRITER_SOURCE, encoding="utf-8")
            verifier_src.write_text(VERIFIER_SOURCE, encoding="utf-8")

            for src, exe in ((writer_src, writer_exe), (verifier_src, verifier_exe)):
                comp = subprocess.run(
                    [str(ESHKOL_RUN), "-L", str(BUILD_DIR), str(src), "-o", str(exe)],
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self.assertEqual(comp.returncode, 0, msg=comp.stderr + comp.stdout)

            prev_post = 0
            max_recovery_ms = 0.0
            for cycle in range(5):
                env = dict(os.environ, ESHKOL_WAL_DB=str(db_path), ESHKOL_WAL_TARGET="2000")
                proc = subprocess.Popen(
                    [str(writer_exe)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                )
                last_acked = prev_post
                # Wait until at least 8 fresh commits have been acknowledged on stdout,
                # then send SIGKILL while the writer is actively committing the next row.
                target_ack = prev_post + 8 + cycle * 3
                assert proc.stdout is not None
                try:
                    for line in proc.stdout:
                        line = line.strip()
                        if line.startswith("COMMITTED:"):
                            last_acked = int(line.split(":", 1)[1])
                            if last_acked >= target_ack:
                                os.kill(proc.pid, signal.SIGKILL)
                                break
                    proc.wait(timeout=15)
                finally:
                    if proc.stdout:
                        proc.stdout.close()
                    if proc.stderr:
                        proc.stderr.close()
                self.assertEqual(
                    proc.returncode,
                    -signal.SIGKILL,
                    f"cycle {cycle}: expected SIGKILL termination, got {proc.returncode}",
                )

                t0 = time.perf_counter()
                ver = subprocess.run(
                    [str(verifier_exe)],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=30,
                )
                recovery_ms = (time.perf_counter() - t0) * 1000.0
                max_recovery_ms = max(max_recovery_ms, recovery_ms)
                self.assertEqual(ver.returncode, 0, msg=ver.stderr + ver.stdout)

                pre = post = None
                for line in ver.stdout.splitlines():
                    if line.startswith("VERIFIED_PRE:"):
                        pre = int(line.split(":", 1)[1])
                    elif line.startswith("VERIFIED_POST:"):
                        post = int(line.split(":", 1)[1])
                self.assertIsNotNone(pre, msg=ver.stdout)
                self.assertIsNotNone(post, msg=ver.stdout)
                # Every acknowledged commit before SIGKILL MUST survive.
                self.assertGreaterEqual(
                    pre,
                    last_acked,
                    f"cycle {cycle}: lost committed row(s): pre={pre} < acked={last_acked}",
                )
                self.assertEqual(post, pre + 1)
                prev_post = post

            # Recovery must be fast (well under 2 seconds even including process startup).
            self.assertLess(max_recovery_ms, 2000.0, f"slow recovery: {max_recovery_ms:.1f}ms")
            print(
                f"PASS: 5/5 SIGKILL WAL cycles recovered cleanly "
                f"(final_seq={prev_post}, max_recovery_ms={max_recovery_ms:.1f})"
            )


if __name__ == "__main__":
    unittest.main()
