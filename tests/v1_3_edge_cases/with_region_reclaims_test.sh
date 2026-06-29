#!/usr/bin/env bash
# ESH-0039: (with-region ...) must RECLAIM its body allocations.
#
# This harness measures PEAK RSS of two equivalent workloads:
#   (a) region:    a loop of (with-region 'j (make-vector 1000 i))
#   (b) baseline:  the same loop WITHOUT with-region
#
# Pre-fix the region loop bypassed region_pop's free and leaked every vector
# into the global arena, so (a) >= (b) and both climbed multi-GB until OOM.
# Post-fix (a) must be BOUNDED and materially lower than (b).
#
# Pass criteria:
#   - region peak RSS < REGION_CEIL_MB              (bounded, not GB-scale)
#   - region peak RSS < baseline peak RSS / 2       (materially lower)
set -euo pipefail

ESHKOL_RUN="${1:-${ESHKOL_RUN:-}}"
if [ -z "$ESHKOL_RUN" ]; then
    for cand in ./build/eshkol-run ./build-verify/eshkol-run; do
        [ -x "$cand" ] && ESHKOL_RUN="$cand" && break
    done
fi
if [ -z "${ESHKOL_RUN:-}" ] || [ ! -x "$ESHKOL_RUN" ]; then
    echo "FAIL: with_region_reclaims_test could not locate eshkol-run" >&2
    exit 1
fi

# Resolve the build dir holding stdlib.o (next to eshkol-run).
BUILD_DIR="$(cd "$(dirname "$ESHKOL_RUN")" && pwd)"

ITERS="${ESH0039_ITERS:-200000}"
REGION_CEIL_MB="${ESH0039_REGION_CEIL_MB:-512}"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

cat > "$tmpdir/region.esk" <<EOF
(define (loop i n)
  (if (< i n) (begin (with-region 'j (make-vector 1000 i)) (loop (+ i 1) n)) 'done))
(display (loop 0 $ITERS)) (newline)
EOF

cat > "$tmpdir/baseline.esk" <<EOF
(define (loop i n)
  (if (< i n) (begin (make-vector 1000 i) (loop (+ i 1) n)) 'done))
(display (loop 0 $ITERS)) (newline)
EOF

compile() {  # src out
    if ! "$ESHKOL_RUN" "$1" -o "$2" -L"$BUILD_DIR" > "$tmpdir/compile.log" 2>&1; then
        echo "FAIL: compile of $1 failed" >&2; cat "$tmpdir/compile.log" >&2; exit 1
    fi
}

compile "$tmpdir/region.esk"   "$tmpdir/region"
compile "$tmpdir/baseline.esk" "$tmpdir/baseline"

# peak_rss_mb <binary> -> echoes integer MB (portable macOS/Linux)
peak_rss_mb() {
    local bin="$1" tf="$tmpdir/time.txt" raw os
    os="$(uname -s)"
    if [ "$os" = "Darwin" ]; then
        /usr/bin/time -l "$bin" >/dev/null 2>"$tf" || true
        raw="$(grep -E 'maximum resident set size' "$tf" | awk '{print $1}')"
        echo $(( raw / 1024 / 1024 ))          # macOS reports BYTES
    else
        /usr/bin/time -v "$bin" >/dev/null 2>"$tf" || true
        raw="$(grep -E 'Maximum resident set size' "$tf" | awk -F': ' '{print $2}')"
        echo $(( raw / 1024 ))                  # Linux reports KiB
    fi
}

region_mb="$(peak_rss_mb "$tmpdir/region")"
baseline_mb="$(peak_rss_mb "$tmpdir/baseline")"

echo "ESH-0039 peak RSS over $ITERS iters: region=${region_mb}MB baseline=${baseline_mb}MB (ceil=${REGION_CEIL_MB}MB)"

rc=0
if [ "$region_mb" -ge "$REGION_CEIL_MB" ]; then
    echo "FAIL: region peak RSS ${region_mb}MB is not bounded (>= ${REGION_CEIL_MB}MB)" >&2
    rc=1
fi
if [ "$region_mb" -ge $(( baseline_mb / 2 )) ]; then
    echo "FAIL: region peak RSS ${region_mb}MB not materially below baseline ${baseline_mb}MB" >&2
    rc=1
fi

if [ "$rc" -eq 0 ]; then
    echo "PASS: with_region_reclaims_test.sh (region reclaims; bounded RSS)"
fi
exit "$rc"
