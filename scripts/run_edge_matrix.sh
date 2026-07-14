#!/usr/bin/env bash
# run_edge_matrix.sh — run the feature-pair composition matrix (pillar P2 of
# .swarm/ADVERSARIAL_TESTING_CAMPAIGN.md) under BOTH execution paths
# (JIT `-r` and AOT binary) and classify every generated probe file:
#
#   PASS         all self-checks passed, file ran to completion
#   ASSERT-FAIL  a self-check reported a wrong VALUE (compiler bug candidate!)
#   CRASH        killed by a signal
#   COMPILE-ERR  nonzero exit without failed checks (compile or runtime error)
#   HANG         per-file timeout expired
#
# Emits ICC trace events (kind: edge_matrix) into
# scripts/icc_traces/edge_matrix.jsonl — same pattern as run_icc_smoke.sh —
# plus a human summary table.
#
# Usage:
#   scripts/run_edge_matrix.sh                  # run everything, jit+aot
#   MODES=jit scripts/run_edge_matrix.sh        # jit only
#   FILTER='pair00*' scripts/run_edge_matrix.sh # subset by glob
#   JOBS=8 scripts/run_edge_matrix.sh           # parallel workers (default 4)
#
# Regenerate the corpus first with:
#   python3 tests/edge_matrix/gen_matrix.py
#
# Known, already-triaged failures live in tests/edge_matrix/KNOWN_FAILURES.txt
# (one `<basename> <mode>` per line, '#' comments). They are reported but do
# not fail the sweep — the sweep-level trace event is PASS iff every
# non-allowlisted file+mode is PASS.
set -u
export LC_ALL=C LC_CTYPE=C LANG=C
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

GEN_DIR="$REPO_ROOT/tests/edge_matrix/generated"
TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
TRACE_FILE="$TRACE_DIR/edge_matrix.jsonl"
: "${TRACE_FILE:?TRACE_FILE not set}"
KNOWN_FAILURES="$REPO_ROOT/tests/edge_matrix/KNOWN_FAILURES.txt"
ESHKOL_RUN="$REPO_ROOT/build/eshkol-run"

MODES="${MODES:-jit aot}"
FILTER="${FILTER:-pair*.esk}"
JOBS="${JOBS:-4}"
JIT_TIMEOUT="${JIT_TIMEOUT:-60}"
AOT_COMPILE_TIMEOUT="${AOT_COMPILE_TIMEOUT:-120}"
AOT_RUN_TIMEOUT="${AOT_RUN_TIMEOUT:-60}"

if [ ! -x "$ESHKOL_RUN" ]; then
    echo "run_edge_matrix.sh: $ESHKOL_RUN not found — build first:" >&2
    echo "  cmake --build build --target eshkol-run stdlib -j" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Worker: classify one file under one mode. Writes one result line
#   <basename> <mode> <CLASS> <passed>/<expected> <detail>
# into $WORK_DIR/results/<basename>.<mode>
# ---------------------------------------------------------------------------
classify_output() {
    # args: exit_status out_file expected_checks
    local st="$1" out="$2" expected="$3"
    local fails passes
    fails=$(grep -c '^FAIL:' "$out" 2>/dev/null || true)
    passes=$(grep -c '^PASS:' "$out" 2>/dev/null || true)
    if [ "$st" -eq 142 ]; then          # SIGALRM via perl alarm
        echo "HANG $passes/$expected timeout"
    elif [ "$st" -ge 128 ]; then
        echo "CRASH $passes/$expected signal=$((st - 128))"
    elif [ "$fails" -gt 0 ]; then
        local first
        first=$(grep -m1 '^FAIL:' "$out" | tr -d '"' | cut -c1-160)
        echo "ASSERT-FAIL $passes/$expected $first"
    elif [ "$st" -ne 0 ]; then
        local err
        err=$(grep -m1 -iE 'error' "$out" | tr -d '"' | cut -c1-120)
        echo "COMPILE-ERR $passes/$expected exit=$st ${err:-no-error-line}"
    elif [ "$passes" -ne "$expected" ]; then
        echo "ASSERT-FAIL $passes/$expected truncated-output"
    else
        echo "PASS $passes/$expected ok"
    fi
}

run_one() {
    local f="$1" mode="$2"
    local base expected out st res bin
    base=$(basename "$f" .esk)
    expected=$(sed -n 's/^;; CHECKS: //p' "$f" | head -1)
    expected="${expected:-0}"
    out="$WORK_DIR/$base.$mode.out"
    if [ "$mode" = jit ]; then
        perl -e "alarm $JIT_TIMEOUT; exec @ARGV" \
            "$ESHKOL_RUN" -r "$f" > "$out" 2>&1
        st=$?
        res=$(classify_output "$st" "$out" "$expected")
    else
        bin="$WORK_DIR/$base.bin"
        perl -e "alarm $AOT_COMPILE_TIMEOUT; exec @ARGV" \
            "$ESHKOL_RUN" "$f" -o "$bin" > "$out" 2>&1
        st=$?
        if [ "$st" -ne 0 ] || [ ! -x "$bin" ]; then
            local err
            err=$(grep -m1 -iE 'error' "$out" | tr -d '"' | cut -c1-120)
            if [ "$st" -eq 142 ]; then
                res="HANG 0/$expected compile-timeout"
            else
                res="COMPILE-ERR 0/$expected exit=$st ${err:-link-or-compile-failed}"
            fi
        else
            perl -e "alarm $AOT_RUN_TIMEOUT; exec @ARGV" "$bin" > "$out" 2>&1
            st=$?
            res=$(classify_output "$st" "$out" "$expected")
        fi
        rm -f "$bin"
    fi
    echo "$base $mode $res" > "$WORK_DIR/results/$base.$mode"
}

# Self-dispatch so xargs can parallelize file×mode work items. Workers
# inherit WORK_DIR / ESHKOL_JIT_CACHE_DIR from the coordinator's env and
# must NOT create or clean up their own.
if [ "${EDGE_MATRIX_WORKER:-}" = 1 ]; then
    run_one "$1" "$2"
    exit 0
fi

mkdir -p "$TRACE_DIR"
WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/eshkol-edge-matrix.XXXXXX")
trap 'rm -rf "$WORK_DIR"' EXIT

# Isolate the JIT cache so results are reproducible.
if [ -z "${ESHKOL_JIT_CACHE_DIR:-}" ]; then
    export ESHKOL_JIT_CACHE_DIR="$WORK_DIR/jit-cache"
    mkdir -p "$ESHKOL_JIT_CACHE_DIR"
fi

mkdir -p "$WORK_DIR/results"
export WORK_DIR ESHKOL_RUN JIT_TIMEOUT AOT_COMPILE_TIMEOUT AOT_RUN_TIMEOUT

# Build the work list (deterministic order).
WORK_LIST="$WORK_DIR/worklist"
: "${WORK_LIST:?WORK_LIST not set}"
: > "$WORK_LIST"
shopt -s nullglob
for f in "$GEN_DIR"/$FILTER; do
    for mode in $MODES; do
        printf '%s\n%s\n' "$f" "$mode" >> "${WORK_LIST:?}"
    done
done
shopt -u nullglob
n_items=$(( $(wc -l < "$WORK_LIST") / 2 ))
if [ "$n_items" -eq 0 ]; then
    echo "run_edge_matrix.sh: no generated files match '$FILTER' in $GEN_DIR" >&2
    echo "  regenerate with: python3 tests/edge_matrix/gen_matrix.py" >&2
    exit 2
fi
echo "edge-matrix: $n_items file×mode items, jobs=$JOBS"

xargs -n2 -P "$JOBS" env EDGE_MATRIX_WORKER=1 bash "$0" < "$WORK_LIST"

# ---------------------------------------------------------------------------
# Aggregate, emit traces + summary
# ---------------------------------------------------------------------------
: > "${TRACE_FILE:?}"
emit_event() {   # name value snippet
    local esc
    esc=$(printf '%s' "$3" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g')
    printf '{"kind":"edge_matrix","name":"%s","value":"%s","snippet":"%s","confidence":0.95}\n' \
        "$1" "$2" "$esc" >> "${TRACE_FILE:?}"
}

is_known_failure() {  # base mode
    [ -f "$KNOWN_FAILURES" ] && \
        grep -qE "^$1[[:space:]]+$2([[:space:]]|$)" "$KNOWN_FAILURES"
}

n_pass=0; n_afail=0; n_crash=0; n_cerr=0; n_hang=0; n_known=0; unexpected=0
sorted_file="$WORK_DIR/sorted_results"
cat "$WORK_DIR"/results/* 2>/dev/null | sort > "$sorted_file"

printf '\n%-42s %-4s %-12s %-9s %s\n' FILE MODE CLASS CHECKS DETAIL
printf '%.0s-' {1..100}; printf '\n'
while IFS= read -r row; do
    [ -n "$row" ] || continue
    base=$(echo "$row" | awk '{print $1}')
    mode=$(echo "$row" | awk '{print $2}')
    cls=$(echo "$row" | awk '{print $3}')
    chk=$(echo "$row" | awk '{print $4}')
    detail=$(echo "$row" | cut -d' ' -f5-)
    case "$cls" in
        PASS) n_pass=$((n_pass+1));;
        ASSERT-FAIL) n_afail=$((n_afail+1));;
        CRASH) n_crash=$((n_crash+1));;
        COMPILE-ERR) n_cerr=$((n_cerr+1));;
        HANG) n_hang=$((n_hang+1));;
    esac
    known=""
    if [ "$cls" != PASS ]; then
        if is_known_failure "$base" "$mode"; then
            known=" (known)"
            n_known=$((n_known+1))
        else
            unexpected=$((unexpected+1))
        fi
        printf '%-42s %-4s %-12s %-9s %s%s\n' \
            "$base" "$mode" "$cls" "$chk" "$detail" "$known"
    fi
    ev_val=PASS
    [ "$cls" = PASS ] || ev_val=FAIL
    emit_event "${base}_${mode}" "$ev_val" "$cls $chk $detail$known"
done < "$sorted_file"

total=$((n_pass + n_afail + n_crash + n_cerr + n_hang))
echo
echo "edge-matrix summary: total=$total PASS=$n_pass ASSERT-FAIL=$n_afail" \
     "CRASH=$n_crash COMPILE-ERR=$n_cerr HANG=$n_hang" \
     "(known-failures=$n_known unexpected=$unexpected)"

sweep_val=PASS
[ "$unexpected" -eq 0 ] || sweep_val=FAIL
emit_event edge_matrix_sweep_clean "$sweep_val" \
    "total=$total pass=$n_pass assert_fail=$n_afail crash=$n_crash compile_err=$n_cerr hang=$n_hang known=$n_known unexpected=$unexpected"

echo "traces: $TRACE_FILE"
[ "$unexpected" -eq 0 ] && exit 0 || exit 1
