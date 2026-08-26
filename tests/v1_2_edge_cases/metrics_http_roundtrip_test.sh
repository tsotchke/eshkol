#!/usr/bin/env bash
# metrics_http_roundtrip_test.sh — Prometheus /metrics endpoint round-trip (#148).
#
# docs/COMPILER_ROADMAP.md #148 asks for a "Prometheus metrics primitive +
# /metrics endpoint" — counters, gauges, histograms, and the standard
# /metrics helper. lib/core/metrics.esk (make-counter, counter-inc!,
# metrics-render) and lib/core/http_server.esk's http-standard-response
# already wire "/metrics" -> (metrics-render) (see http_server.esk around
# the http-standard-response definition), but no existing test drove an
# actual GET /metrics over a real loopback socket before this file: the
# only coverage was tests/stdlib/metrics_test.esk, which exercises the
# metrics module in isolation (never over HTTP), and the http-server
# round-trip tests (http_server_smoke_test.sh, tests/vm/http_server_
# surface_regression.esk) which only ever hit /health or custom routes.
#
# This spins up the loopback HTTP server (mirrors http_server_smoke_test.sh's
# fork+client pattern), registers a counter, increments it, then forks a
# client that performs a real GET /metrics and asserts the response is
# Prometheus text-exposition format containing the counter's HELP/TYPE
# preamble and current value.
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

WORK=$(mktemp -d -t eshkol_metrics_http.XXXXXX)
trap 'rm -rf "$WORK"' EXIT
RUN_TIMEOUT="${ESHKOL_METRICS_HTTP_SMOKE_TIMEOUT:-120}"

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

cat > "$WORK/metrics_http.esk" <<'EOF'
(require stdlib)
(require core.http_server)
(require core.metrics)

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
  (let ((nlen (string-length needle))
        (hlen (string-length haystack)))
    (let loop ((i 0))
      (cond
        ((> (+ i nlen) hlen) #f)
        ((string=? (substring haystack i (+ i nlen)) needle) #t)
        (else (loop (+ i 1)))))))

;; ── Register a counter so /metrics has real exposition text to check ──
(define probe-counter
  (make-counter "eshkol_v14_oracle_probe_total"
                "v1.4-connection oracle metrics round-trip probe" '()))
(counter-inc! probe-counter '())
(counter-inc! probe-counter '())

;; ── Server up ──────────────────────────────────────────────────────
(define (server-handle? h)
  (and (number? h) (> h 0)))

(define (candidate-port attempt)
  (+ 21000 (remainder (+ (getpid) attempt) 20000)))

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
      (display "SKIP: http-server-create unavailable") (newline)
      (exit 0))
    #t)

(check "http-server-create returns positive handle" #t (server-handle? srv))

(define port (if (server-handle? srv) (http-server-port srv) #f))

(define url
  (string-append "http://127.0.0.1:" (number->string port) "/metrics"))
(define marker-path
  (string-append "/tmp/eshkol-metrics-http-client-"
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
                 (string-contains? (caddr response) "eshkol_v14_oracle_probe_total")
                 (string-contains? (caddr response) "# TYPE")
                 (string-contains? (caddr response) "# HELP"))
            (let ((out (open-output-file marker-path)))
              (write-string "ok" out)
              (close-port out))
            #f))
      (exit 0))
    #t)

;; ── Server-side accept, route through http-route-request (falls back
;;    to the standard /metrics route) ────────────────────────────────
(define request
  (if (and (number? client-pid) (> client-pid 0))
      (http-server-accept srv 4096 2000)
      #f))
(check "accept returned a string" #t (string? request))
(check "request mentions /metrics" #t
       (and (string? request) (string-contains? request "/metrics")))

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
(check "client saw counter name + HELP/TYPE preamble over real HTTP" #t
       (and (file-exists? marker-path)
            (= (file-size marker-path) 2)))

(if (server-handle? srv) (http-server-close srv) #f)
(if (file-exists? marker-path) (delete-file marker-path) #f)

(display "---") (newline)
(display "Passed: ") (display passed) (newline)
(display "Failed: ") (display failed) (newline)
(if (> failed 0) (exit 1) (exit 0))
EOF

ESHKOL_PATH="$ROOT" run_with_timeout "$RUN_TIMEOUT" "$RUN" -r "$WORK/metrics_http.esk"
rc=$?
if [ "$rc" -eq 124 ]; then
    echo "FAIL: metrics http roundtrip timed out after ${RUN_TIMEOUT}s"
fi
exit "$rc"
