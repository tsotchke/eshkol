#!/usr/bin/env bash
# run_math_sweep.sh — drive examples/mathematics_group_cohomology_sweep.esk over a grid.
#
# Usage: scripts/run_math_sweep.sh <grid-file> <receipts.jsonl> [build-dir]
#
# The grid file has one run per line: "<MATH_GROUP> <MATH_DEGREE>", e.g.
#   cyclic:6 3
#   dihedral:5 2
# Lines starting with # are ignored. The sweep program is compiled once (AOT,
# -O 0: the fast loop for these programs) and run once per line with the two
# environment variables set; every run appends one JSON receipt line to
# <receipts.jsonl>, wrapped with the host, the source commit, and the wall
# time. Exit status is nonzero if any run failed or reported verdict FAIL.
set -u
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
GRID="${1:?grid file}"; OUT="${2:?receipts path}"; BUILD_DIR="${3:-build}"
case "$BUILD_DIR" in /*) ;; *) BUILD_DIR="$REPO_ROOT/$BUILD_DIR" ;; esac
ESHKOL_RUN="$BUILD_DIR/eshkol-run"
[ -x "$ESHKOL_RUN" ] || { echo "run_math_sweep.sh: $ESHKOL_RUN not found" >&2; exit 2; }
PROGRAM="$REPO_ROOT/examples/mathematics_group_cohomology_sweep.esk"
BIN="$BUILD_DIR/mathematics_group_cohomology_sweep_aot"
"$ESHKOL_RUN" -O 0 -o "$BIN" "$PROGRAM" >/dev/null 2>&1 || { echo "run_math_sweep.sh: compile failed" >&2; exit 2; }
HOST="$(hostname -s 2>/dev/null || echo unknown)"
SHA="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
mkdir -p "$(dirname "$OUT")"
status=0; runs=0
while read -r group degree rest; do
    case "$group" in ''|'#'*) continue ;; esac
    runs=$((runs+1))
    t0=$(python3 -c 'import time;print(time.time())')
    line="$(MATH_GROUP="$group" MATH_DEGREE="${degree:-3}" "$BIN" 2>/dev/null | grep '^{')"; rc=$?
    t1=$(python3 -c 'import time;print(time.time())')
    wall=$(python3 -c "print(round($t1-$t0,2))")
    [ -n "$line" ] || { line="{\"group\":\"$group\",\"degree\":\"${degree:-3}\",\"verdict\":\"FAIL\",\"error\":\"no receipt\"}"; rc=1; }
    printf '{"host":"%s","commit":"%s","wall_s":%s,"exit":%s,"receipt":%s}\n' "$HOST" "$SHA" "$wall" "$rc" "$line" >> "$OUT"
    case "$line" in *'"verdict":"PASS"'*) echo "PASS $group degree ${degree:-3} ${wall}s" ;; *) echo "FAIL $group degree ${degree:-3}"; status=1 ;; esac
done < "$GRID"
echo "runs=$runs receipts=$OUT"
exit "$status"
