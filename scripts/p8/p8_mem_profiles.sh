#!/usr/bin/env bash
# p8_mem_profiles.sh — P8 escape-closure axis 8: workload-shaped memory
# profiles.  Extends the P4 stress pillar (scripts/run_stress.sh) with
# PARAMETRIC RSS-budget shapes.
#
# Originating escape (see .swarm/P8_ESCAPE_ANALYSIS.md): scope/region
# reclamation bugs surface as RSS that grows with the amount of WORK — an
# AD-training loop, a resident tick loop, or a region-allocating loop that
# leaks each iteration OOMs on a long run while a short smoke run looks fine.
# A fixed-size RSS ceiling is machine-specific and easy to set too loose.
#
# This harness asserts the machine-INDEPENDENT invariant that distinguishes a
# leak from bounded reclamation: peak RSS must be ~FLAT in the iteration count.
# Each workload shape is run at a small AND a large iteration count; the gate is
#     RSS(large) <= RSS(small) * RATIO_MAX     (flat -> reclamation works)
# plus a generous absolute ceiling. A leak makes RSS scale with work and trips
# the ratio regardless of the machine.
#
# Shapes x scopes (the 2026-07 workload classes):
#   ad_train      gradient-of-loss step loop        (auto-scope | with-region)
#   resident_kb   tick loop mutating a KB hash-table(auto-scope | with-region)
#   proof_churn   alloc/discard nested-structure churn (auto-scope | with-region)
#   par_batch     repeated parallel-map over a batch (auto-scope | with-region)
#
# Emits a markdown evidence table + kind=escape_matrix JSON-L events into
# scripts/icc_traces/escape_matrix.jsonl and a roll-up p8_mem_profiles_suite.
#
# Usage: scripts/p8/p8_mem_profiles.sh [--build-dir DIR] [--quick] [--table FILE]
set -u
export LC_ALL=C LC_CTYPE=C LANG=C
cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
. "$REPO_ROOT/scripts/lib/durable_work_root.sh"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
QUICK=0; TABLE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --build-dir) BUILD_DIR="$2"; shift 2;;
    --quick) QUICK=1; shift;;
    --table) TABLE="$2"; shift 2;;
    *) shift;;
  esac
done
case "$BUILD_DIR" in /*) ;; *) BUILD_DIR="$REPO_ROOT/$BUILD_DIR";; esac
ESHKOL_RUN="$BUILD_DIR/eshkol-run"
[ -x "$ESHKOL_RUN" ] || { echo "p8_mem_profiles: $ESHKOL_RUN not found" >&2; exit 2; }

TRACE_DIR="$REPO_ROOT/scripts/icc_traces"; mkdir -p "$TRACE_DIR"
TRACE_FILE="$TRACE_DIR/escape_matrix.jsonl"
if eshkol_durable_enabled; then
  WORK="$(eshkol_durable_prepare_dir p8-mem-profiles)" || exit $?
else
  WORK="$(mktemp -d "${TMPDIR:-/tmp}/p8-mem.XXXXXX")"
  trap 'rm -rf "$WORK"' EXIT
fi
export ESHKOL_JIT_CACHE_DIR="$WORK/jit"; mkdir -p "$ESHKOL_JIT_CACHE_DIR"

# small / large iteration counts (large = 10x small -> flat RSS if reclaimed).
SMALL=1000
LARGE=$([ "$QUICK" -eq 1 ] && echo 8000 || echo 40000)
RATIO_MAX="1.6"          # allow modest fixed overhead; a leak blows past this
ABS_CEIL_MB=600

emit() { python3 -c 'import json,sys;print(json.dumps({"kind":"escape_matrix","name":sys.argv[1],"value":sys.argv[2],"snippet":sys.argv[3][:200],"confidence":0.9}))' "$1" "$2" "$3" >> "$TRACE_FILE"; }

# peak RSS in MB for a run of $ESHKOL_RUN -r FILE, or empty on nonzero exit.
peak_rss_mb() { # file
  local f="$1" out rss
  if [ "$(uname)" = "Darwin" ]; then
    out=$(/usr/bin/time -l "$ESHKOL_RUN" -r "$f" 2>&1 >/dev/null)
    rss=$(printf '%s' "$out" | awk '/maximum resident set size/ {print $1}')
    [ -n "$rss" ] && echo $(( rss / 1048576 ))
  else
    out=$(/usr/bin/time -v "$ESHKOL_RUN" -r "$f" 2>&1 >/dev/null)
    rss=$(printf '%s' "$out" | awk -F: '/Maximum resident set size/ {gsub(/ /,"",$2);print $2}')
    [ -n "$rss" ] && echo $(( rss / 1024 ))
  fi
}

# Emit a workload source for (shape, scope, iters) to stdout.
gen() { # shape scope iters
  local shape="$1" scope="$2" N="$3"
  local body region_open region_close
  if [ "$scope" = "region" ]; then region_open="(with-region 'r "; region_close=")"; else region_open=""; region_close=""; fi
  case "$shape" in
    ad_train)
      cat <<EOF
(define (loss v) (+ (* (vector-ref v 0) (vector-ref v 0)) (* (vector-ref v 1) (vector-ref v 1))))
(define (step i acc)
  (if (>= i $N) acc
    (let ((g ${region_open}(gradient loss (vector (exact->inexact (modulo i 7)) 1.0))${region_close}))
      (step (+ i 1) (+ acc (vector-ref g 0))))))
(display "ad_train ") (display (step 0 0.0)) (newline)
EOF
      ;;
    resident_kb)
      cat <<EOF
(define kb (make-hash-table))
(define (tick i acc)
  (if (>= i $N) acc
    (begin (hash-table-set! kb (modulo i 64) (list i (number->string i)))
      (let ((s ${region_open}(vector-length (make-vector 32 i))${region_close}))
        (tick (+ i 1) (+ acc s))))))
(display "resident_kb ") (display (tick 0 0)) (newline)
EOF
      ;;
    proof_churn)
      cat <<EOF
(define (churn i acc)
  (if (>= i $N) acc
    (let ((n ${region_open}(length (list (list i i) (vector i i i) (number->string i) (list (list i) (list i i))))${region_close}))
      (churn (+ i 1) (+ acc n)))))
(display "proof_churn ") (display (churn 0 0)) (newline)
EOF
      ;;
    par_batch)
      cat <<EOF
(define (batch i acc)
  (if (>= i $N) acc
    (let ((r ${region_open}(length (parallel-map (lambda (x) (* x x)) (list 1 2 3 4 5 6 7 8)))${region_close}))
      (batch (+ i 1) (+ acc r)))))
(display "par_batch ") (display (batch 0 0)) (newline)
EOF
      ;;
  esac
}

SHAPES="ad_train resident_kb proof_churn par_batch"
SCOPES="auto region"
[ "$QUICK" -eq 1 ] && SHAPES="ad_train resident_kb proof_churn"   # par_batch to nightly

ROWS=""; PASS=0; FAIL=0
for shape in $SHAPES; do
  for scope in $SCOPES; do
    sf="$WORK/${shape}_${scope}_s.esk"; lf="$WORK/${shape}_${scope}_l.esk"
    gen "$shape" "$scope" "$SMALL" > "$sf"
    gen "$shape" "$scope" "$LARGE" > "$lf"
    rs=$(peak_rss_mb "$sf"); rl=$(peak_rss_mb "$lf")
    id="mem_${shape}_${scope}"
    if [ -z "$rs" ] || [ -z "$rl" ]; then
      FAIL=$((FAIL+1)); emit "$id" FAIL "run failed (s=$rs l=$rl)"
      ROWS="$ROWS| $shape | $scope | $SMALL | $LARGE | ${rs:-ERR} | ${rl:-ERR} | — | FAIL |\n"
      echo "FAILED tests/escape_matrix/mem::$id (run error)"; continue
    fi
    ratio=$(python3 -c "print('%.2f' % (($rl+1)/($rs+1)))")
    flat=$(python3 -c "print(1 if ($rl+1)/($rs+1) <= $RATIO_MAX else 0)")
    under=$(python3 -c "print(1 if $rl <= $ABS_CEIL_MB else 0)")
    if [ "$flat" -eq 1 ] && [ "$under" -eq 1 ]; then
      PASS=$((PASS+1)); emit "$id" PASS "flat rss ratio=$ratio (${rs}->${rl}MB)"
      ROWS="$ROWS| $shape | $scope | $SMALL | $LARGE | ${rs} | ${rl} | $ratio | PASS |\n"
      echo "PASSED tests/escape_matrix/mem::$id (ratio=$ratio)"
    else
      FAIL=$((FAIL+1)); emit "$id" FAIL "rss ratio=$ratio (${rs}->${rl}MB) ceil=$under"
      ROWS="$ROWS| $shape | $scope | $SMALL | $LARGE | ${rs} | ${rl} | $ratio | FAIL |\n"
      echo "FAILED tests/escape_matrix/mem::$id (ratio=$ratio > $RATIO_MAX or ${rl}MB > ${ABS_CEIL_MB})"
    fi
  done
done

TBL="| shape | scope | N_small | N_large | RSS_small(MB) | RSS_large(MB) | ratio | verdict |\n|---|---|---|---|---|---|---|---|\n$ROWS"
printf "\n== axis-8 workload memory profiles (flat-RSS invariant, RATIO_MAX=$RATIO_MAX) ==\n"
printf "$TBL" | column -t -s '|' 2>/dev/null || printf "$TBL"
[ -n "$TABLE" ] && printf "$TBL" > "$TABLE"

emit p8_mem_profiles_suite "$([ "$FAIL" -eq 0 ] && echo PASS || echo FAIL)" "pass=$PASS fail=$FAIL"
echo
echo "axis-8 mem-profiles: PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ] && { echo "axis-8 gate: PASS"; exit 0; } || { echo "axis-8 gate: FAIL"; exit 1; }
