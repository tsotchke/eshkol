#!/usr/bin/env bash
# http_server_smoke_test.sh — core HTTP server builtin round-trip (#145).
#
# Spins up the loopback HTTP server, forks a child that performs a core
# http-request GET, asserts the server sees a valid request and the client
# sees the body.
#
# Runs through the JIT (eshkol-run -r), matching the rest of the v1.2
# edge-case suite's system-builtin coverage.

set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
case "${BUILD_DIR:-build}" in
    /*) RUN="${BUILD_DIR}/eshkol-run" ;;
    *) RUN="$ROOT/${BUILD_DIR:-build}/eshkol-run" ;;
esac
case "${BUILD_DIR:-build}" in
    /*) BUILD_DIR_PATH="${BUILD_DIR}" ;;
    *) BUILD_DIR_PATH="$ROOT/${BUILD_DIR:-build}" ;;
esac

# ICC's source stamp is bound to its registered repo root, while --cwd only
# selects where the test command runs. Refuse a named ICC invocation if those
# roots differ, before opening sockets or producing test evidence.
if [ -n "${ICC_REPO_NAME:-}" ]; then
    ICC_BIN="${ICC_BIN:-$HOME/Desktop/infinite_context_coder/bin/icc}"
    if [ ! -x "$ICC_BIN" ]; then
        echo "FAIL: ICC receipt producer unavailable: $ICC_BIN"
        exit 3
    fi
    REGISTERED_ROOT=$("$ICC_BIN" resolve --repo "$ICC_REPO_NAME" --format json 2>/dev/null | python3 -c '
import json, os, sys
try:
    path = json.load(sys.stdin).get("repo", {}).get("path", "")
    print(os.path.realpath(path) if path else "")
except Exception:
    print("")
')
    CHECKOUT_ROOT=$(cd "$ROOT" && pwd -P)
    if [ -z "$REGISTERED_ROOT" ] || [ "$REGISTERED_ROOT" != "$CHECKOUT_ROOT" ]; then
        echo "FAIL: ICC repo $ICC_REPO_NAME resolves to '$REGISTERED_ROOT', but test checkout is '$CHECKOUT_ROOT'"
        exit 3
    fi
fi

if [ ! -x "$RUN" ]; then
    echo "FAIL: $RUN not built; HTTP server evidence is unavailable"
    exit 2
fi

# Bind the run to the exact runner digest and fail if it predates any
# build-relevant source change. The ICC test receipt separately stamps the
# clean source tree and declares this runner as measured data.
TRACE_DIR="${TRACE_DIR:-$ROOT/scripts/icc_traces}"
case "$TRACE_DIR" in
    /*) ;;
    *) TRACE_DIR="$ROOT/$TRACE_DIR" ;;
esac
mkdir -p "$TRACE_DIR"
. "$ROOT/scripts/lib/build_fingerprint.sh"
eshkol_emit_build_fingerprint_event "$TRACE_DIR" "v14_http_server_roundtrip" "$BUILD_DIR_PATH" eshkol-run
if ! python3 "$ROOT/scripts/check_build_fingerprint.py" \
    --build-dir "$BUILD_DIR_PATH" --trace-dir "$TRACE_DIR" --format json; then
    echo "FAIL: eshkol-run build fingerprint is stale or mismatched"
    exit 1
fi

WORK=$(mktemp -d -t eshkol_http_server.XXXXXX)
trap 'rm -rf "$WORK"' EXIT
RUN_TIMEOUT="${ESHKOL_HTTP_SERVER_SMOKE_TIMEOUT:-120}"

run_with_timeout() {
    local seconds="$1"
    shift

    local timeout_marker="$WORK/timeout"
    rm -f "$timeout_marker"

    "$@" &
    local cmd_pid=$!

    (
        sleep "$seconds"
        if kill -0 "$cmd_pid" 2>/dev/null; then
            touch "$timeout_marker"
            kill "$cmd_pid" 2>/dev/null || true
            sleep 1
            kill -9 "$cmd_pid" 2>/dev/null || true
        fi
    ) &
    local watchdog_pid=$!

    wait "$cmd_pid"
    local rc=$?

    kill "$watchdog_pid" 2>/dev/null || true
    wait "$watchdog_pid" 2>/dev/null || true

    if [ -f "$timeout_marker" ]; then
        return 124
    fi
    return "$rc"
}

cat > "$WORK/http_server.esk" <<'EOF'
(require stdlib)
(require core.http_server)

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
  ;; Linear scan over HTTP request text or response body.
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

(define (candidate-port attempt)
  (+ 20000 (remainder (+ (getpid) attempt) 20000)))

(define (create-server-with-retry attempts)
  (let loop ((attempt 0))
    (let ((srv (http-server-create (candidate-port attempt))))
      (if (server-handle? srv)
          srv
          (if (< attempt attempts)
              (begin (sleep-ms 50) (loop (+ attempt 1)))
              srv)))))

(define srv (create-server-with-retry 5))
(if (not (server-handle? srv))
    (begin
      (display "FAIL: http-server-create unavailable") (newline)
      (exit 1))
    #t)

(check "http-server-create returns positive handle" #t (server-handle? srv))

(define port (if (server-handle? srv) (http-server-port srv) #f))
(check "http-server-port returns >0"  #t (and (number? port) (> port 0)))
(check "http-server-port returns <65536" #t (and (number? port) (< port 65536)))

(define url
  (string-append "http://127.0.0.1:" (number->string port) "/health"))
(define marker-path
  (string-append "/tmp/eshkol-http-server-client-"
                 (number->string (getpid))
                 ".txt"))

;; ── Concurrent client ─────────────────────────────────────────────
(define client-pid
  (if (and (number? port) (> port 0))
      (fork)
      #f))

(if (and (number? client-pid) (= client-pid 0))
    (begin
      (sleep-ms 50)
      (let ((response (http-request "GET" url "" "" 5000)))
        (if (and response
                 (= (car response) 200)
                 (string? (caddr response))
                 (string-contains? (caddr response) "OK"))
            (let ((out (open-output-file marker-path)))
              (write-string "ok" out)
              (close-port out))
            #f))
      (exit 0))
    #t)

;; ── Server-side accept ────────────────────────────────────────────
(define request
  (if (and (number? client-pid) (> client-pid 0))
      (http-server-accept srv 4096 2000)
      #f))
(check "accept returned a string" #t (string? request))
(check "request begins with GET" #t
       (and (string? request)
            (>= (string-length request) 4)
            (string=? (substring request 0 4) "GET ")))
(check "request mentions /health" #t
       (and (string? request) (string-contains? request "/health")))

;; ── Server replies, client joins, both shut down ───────────────────
(if (and (number? client-pid) (> client-pid 0) request)
    (http-server-respond-response srv (http-route-request request '()))
    #f)

(if (and (number? client-pid) (> client-pid 0) (not request))
    (process-kill client-pid 15)
    #f)

(define client-status
  (if (and (number? client-pid) (> client-pid 0))
      (process-wait client-pid)
      #f))
(check "client exited cleanly" 0 client-status)
(check "client stdout contains body" #t
       (and (file-exists? marker-path)
            (= (file-size marker-path) 2)))

(if (server-handle? srv) (http-server-close srv) #f)
(if (file-exists? marker-path) (delete-file marker-path) #f)

(display "---") (newline)
(display "Passed: ") (display passed) (newline)
(display "Failed: ") (display failed) (newline)
(if (> failed 0) (exit 1) (exit 0))
EOF

ESHKOL_PATH="$ROOT" run_with_timeout "$RUN_TIMEOUT" "$RUN" -r "$WORK/http_server.esk"
rc=$?
if [ "$rc" -eq 124 ]; then
    echo "FAIL: http server smoke timed out after ${RUN_TIMEOUT}s"
fi
exit "$rc"
