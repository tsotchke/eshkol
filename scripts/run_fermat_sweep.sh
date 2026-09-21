#!/usr/bin/env bash
# run_fermat_sweep.sh — drive examples/mathematics_fermat_hodge_classes_sweep.esk over a grid.
#
# Usage: scripts/run_fermat_sweep.sh <grid-file> <receipts.jsonl> [build-dir]
#
# The grid file has one run per line: "<degree> <dim>", e.g. "6 2" or "5 4"; lines starting with # are
# ignored. The sweep program is compiled once (AOT, -O 0) and run once per line with MATH_FERMAT_DEGREE and
# MATH_FERMAT_DIM set; every run appends one JSON receipt line to <receipts.jsonl>, wrapped with the host, the
# source commit and the wall time. Exit status is nonzero if any run failed or reported verdict FAIL.
set -u
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
GRID="${1:?grid file}"; OUT="${2:?receipts path}"; BUILD_DIR="${3:-build}"
case "$BUILD_DIR" in /*) ;; *) BUILD_DIR="$REPO_ROOT/$BUILD_DIR" ;; esac
ESHKOL_RUN="$BUILD_DIR/eshkol-run"
[ -x "$ESHKOL_RUN" ] || { echo "run_fermat_sweep.sh: $ESHKOL_RUN not found" >&2; exit 2; }
PROGRAM="$REPO_ROOT/examples/mathematics_fermat_hodge_classes_sweep.esk"
BIN="$BUILD_DIR/mathematics_fermat_hodge_classes_sweep_aot"
"$ESHKOL_RUN" -O 0 -o "$BIN" "$PROGRAM" >/dev/null 2>&1 || { echo "run_fermat_sweep.sh: compile failed" >&2; exit 2; }
HOST="$(hostname -s 2>/dev/null || echo unknown)"
SHA="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
mkdir -p "$(dirname "$OUT")"
status=0; runs=0
while read -r degree dim rest; do
    case "$degree" in ''|'#'*) continue ;; esac
    runs=$((runs+1))
    start=$(date +%s)
    line="$(MATH_FERMAT_DEGREE="$degree" MATH_FERMAT_DIM="$dim" "$BIN" 2>/dev/null | head -1)"
    rc=$?
    wall=$(( $(date +%s) - start ))
    case "$line" in '{'*) ;; *) line="{\"schema\":\"eshkol.math.fermat_hodge_classes.v1\",\"degree\":$degree,\"dim\":$dim,\"verdict\":\"FAIL\",\"error\":\"no receipt\"}"; status=1 ;; esac
    case "$line" in *'"verdict":"PASS"'*) ;; *) status=1 ;; esac
    printf '{"host":"%s","commit":"%s","wall_s":%s,"receipt":%s}\n' "$HOST" "$SHA" "$wall" "$line" >> "$OUT"
done < "$GRID"
echo "run_fermat_sweep.sh: $runs runs, receipts in $OUT, status $status"
exit $status
