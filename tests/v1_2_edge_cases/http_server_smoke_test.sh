#!/bin/bash
# http_server_smoke_test.sh — core HTTP server builtin round-trip (#145).
#
# Spins up the loopback HTTP server, fires a curl GET from a worker
# thread (via core.threads / make-thread = eager future), asserts
# the server sees a valid GET request and the client sees the body.
#
# Runs through the JIT (eshkol-run -r), matching the rest of the v1.2
# edge-case suite's system-builtin coverage.

set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUN="$ROOT/${BUILD_DIR:-build}/eshkol-run"

if [ ! -x "$RUN" ]; then
    echo "SKIP: $RUN not built"
    exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "SKIP: curl not on PATH"
    exit 0
fi

WORK=$(mktemp -d -t eshkol_http_server.XXXXXX)
trap 'rm -rf "$WORK"' EXIT

cat > "$WORK/http_server.esk" <<'EOF'
(require stdlib)
(require core.threads)
(require agent.subprocess)

(define passed 0)
(define failed 0)
(define (check label expected actual)
  (if (equal? expected actual)
      (begin (display "PASS: ") (display label) (newline)
             (set! passed (+ passed 1)))
      (begin (display "FAIL: ") (display label)
             (display " (expected ") (display expected)
             (display ", got ") (display actual) (display ")") (newline)
             (set! failed (+ failed 1)))))

(define (string-contains? haystack needle)
  ;; Linear scan; haystack is HTTP request text or curl stdout.
  (let ((nlen (string-length needle))
        (hlen (string-length haystack)))
    (let loop ((i 0))
      (cond
        ((> (+ i nlen) hlen) #f)
        ((string=? (substring haystack i (+ i nlen)) needle) #t)
        (else (loop (+ i 1)))))))

;; ── Server up ──────────────────────────────────────────────────────
(define (server-handle? h)
  (and (number? h) (> h 0)))

(define (create-server-with-retry attempts)
  (let loop ((remaining attempts))
    (let ((srv (http-server-create 0)))
      (if (server-handle? srv)
          srv
          (if (> remaining 0)
              (begin (sleep-ms 50) (loop (- remaining 1)))
              srv)))))

(define srv (create-server-with-retry 5))
(check "http-server-create returns positive handle" #t (server-handle? srv))

(define port (if (server-handle? srv) (http-server-port srv) #f))
(check "http-server-port returns >0"  #t (and (number? port) (> port 0)))
(check "http-server-port returns <65536" #t (and (number? port) (< port 65536)))

(define url
  (string-append "http://127.0.0.1:" (number->string port) "/health"))

;; ── Concurrent client ─────────────────────────────────────────────
(define client-thread
  (if (and (number? port) (> port 0))
      (make-thread (lambda ()
                     (run-argv-capture (list "curl" "-sS" "-m" "5" url))))
      #f))

;; ── Server-side accept ────────────────────────────────────────────
(define request
  (if client-thread
      (http-server-accept srv 4096 5000)
      #f))
(check "accept returned a string" #t (string? request))
(check "request begins with GET" #t
       (and (string? request)
            (>= (string-length request) 4)
            (string=? (substring request 0 4) "GET ")))
(check "request mentions /health" #t
       (and (string? request) (string-contains? request "/health")))

;; ── Server replies, client joins, both shut down ───────────────────
(if client-thread
    (http-server-respond srv 200 "text/plain" "OK\n")
    #f)

(define result (if client-thread (thread-join client-thread) '()))
(define stdout-pair (assq 'stdout result))
(define exit-pair   (assq 'exit-code result))
(check "client exited cleanly" 0 (and exit-pair (cdr exit-pair)))
(check "client stdout contains body" #t
       (and stdout-pair
            (string? (cdr stdout-pair))
            (string-contains? (cdr stdout-pair) "OK")))

(if (server-handle? srv) (http-server-close srv) #f)

(display "---") (newline)
(display "Passed: ") (display passed) (newline)
(display "Failed: ") (display failed) (newline)
(if (> failed 0) (exit 1) (exit 0))
EOF

ESHKOL_PATH="$ROOT" "$RUN" -r "$WORK/http_server.esk"
