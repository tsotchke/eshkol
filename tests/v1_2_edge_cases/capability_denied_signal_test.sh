#!/bin/bash
# ESH-0076: capability-denied runtime operations must be distinguishable from
# ordinary absent values while preserving the existing false/null return shape.

set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUN="$ROOT/${BUILD_DIR:-build}/eshkol-run"

if [ ! -x "$RUN" ]; then
    echo "SKIP: $RUN not built"
    exit 0
fi

WORK=$(mktemp -d -t eshkol_capability_denied_signal.XXXXXX)
trap 'rm -rf "$WORK"' EXIT

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

cat > "$WORK/env_read_absent.esk" <<'EOF'
(require stdlib)
(require core.capabilities)

(capability-install-policy! (list 'env-read))
(if (getenv "ESHKOL_CAPABILITY_DENIED_SIGNAL_ABSENT")
    (begin
      (display "FAIL: absent getenv returned value") (newline)
      (exit 1))
    (begin
      (display "PASS: allowed absent getenv returns false") (newline)
      (exit 0)))
EOF

cat > "$WORK/env_and_file_denied.esk" <<'EOF'
(require stdlib)
(require core.capabilities)

(capability-install-policy! '())
(define env-denied (not (getenv "ESHKOL_CAPABILITY_DENIED_SIGNAL_ABSENT")))
(define file-denied (not (file-exists? "/tmp/eshkol_capability_denied_signal_missing")))

(if (and env-denied file-denied)
    (begin
      (display "PASS: denied operations preserve false return values") (newline)
      (exit 0))
    (begin
      (display "FAIL: denied operation returned allowed value") (newline)
      (exit 1)))
EOF

ALLOWED_LOG="$WORK/env_read_absent.log"
DENIED_LOG="$WORK/env_and_file_denied.log"

if ! env LC_ALL=C LANG=C ESHKOL_PATH=. "$RUN" -r "$WORK/env_read_absent.esk" >"$ALLOWED_LOG" 2>&1; then
    sed -n '1,160p' "$ALLOWED_LOG" >&2
    fail "allowed absent getenv probe failed"
fi
grep -q "PASS: allowed absent getenv returns false" "$ALLOWED_LOG" ||
    fail "allowed absent getenv probe did not pass"
if grep -q "capability denied:" "$ALLOWED_LOG"; then
    sed -n '1,160p' "$ALLOWED_LOG" >&2
    fail "allowed absent getenv emitted a denial diagnostic"
fi

if ! env LC_ALL=C LANG=C ESHKOL_PATH=. "$RUN" -r "$WORK/env_and_file_denied.esk" >"$DENIED_LOG" 2>&1; then
    sed -n '1,160p' "$DENIED_LOG" >&2
    fail "denied capability probe failed"
fi
grep -q "PASS: denied operations preserve false return values" "$DENIED_LOG" ||
    fail "denied capability probe did not pass"
grep -q "capability denied: env-read" "$DENIED_LOG" ||
    fail "denied getenv did not emit env-read diagnostic"
grep -q "capability denied: file-read" "$DENIED_LOG" ||
    fail "denied file metadata did not emit file-read diagnostic"

cat "$ALLOWED_LOG"
cat "$DENIED_LOG"
echo "PASS: capability_denied_signal_test"
