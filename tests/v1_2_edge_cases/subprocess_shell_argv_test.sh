#!/bin/bash
# subprocess_shell_argv_test.sh — explicit shell vs argv process semantics.

set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUN="$ROOT/${BUILD_DIR:-build}/eshkol-run"

if [ ! -x "$RUN" ]; then
    echo "SKIP: $RUN not built"
    exit 0
fi

if [ ! -x /bin/sh ] || [ ! -x /bin/echo ]; then
    echo "SKIP: POSIX shell tools unavailable"
    exit 0
fi

WORK=$(mktemp -d -t eshkol_subprocess_api.XXXXXX)
trap 'rm -rf "$WORK"' EXIT

cat > "$WORK/subprocess_api.esk" <<'EOF'
(require stdlib)
(load "lib/agent/subprocess.esk")

(define passed 0)
(define failed 0)

(define (check name actual expected)
  (if (equal? expected actual)
      (set! passed (+ passed 1))
      (begin
        (display "FAIL: ") (display name)
        (display " expected ") (display expected)
        (display ", got ") (display actual) (newline)
        (set! failed (+ failed 1)))))

(define shell-result
  (run-command-capture "echo out; echo err >&2; exit 42" "." 5000 4096))

(define argv-result
  (run-argv-capture (list "/bin/echo" "literal;not-shell") "." 5000 4096))

(check "shell exit code" (cdr (assoc 'exit-code shell-result)) 42)
(check "shell stdout" (cdr (assoc 'stdout shell-result)) "out\n")
(check "shell stderr" (cdr (assoc 'stderr shell-result)) "err\n")
(check "argv does not use shell" (cdr (assoc 'stdout argv-result)) "literal;not-shell\n")

(if (> failed 0)
    (exit 1)
    (begin
      (display "PASS: subprocess shell and argv contracts")
      (newline)
      (exit 0)))
EOF

ESHKOL_PATH=. "$RUN" -r "$WORK/subprocess_api.esk"
