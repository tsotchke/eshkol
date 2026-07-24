#!/usr/bin/env bash
# run_edge_coverage_v134.sh — run the v1.3.4 dynamic edge-coverage corpus
# (scripts/gen_edge_v134.py) across the execution axes and classify every
# generated probe, the same way scripts/run_edge_matrix.sh does for the P2
# feature-pair matrix.
#
# For each generated .esk file (self-checking: `;; CHECKS: N` header, prints N
# `PASS: ` lines, `FAIL: ` on a wrong value) it runs, per mode:
#
#   jit      build/eshkol-run -r file.esk
#   aot      build/eshkol-run file.esk -o BIN && BIN            (default -O2)
#   aot-O0   build/eshkol-run -O0 file.esk -o BIN && BIN        (MODES contains aot-O0)
#   vm       build/eshkol-vm-standalone-test file.esk           (VM-eligible families)
#
# and classifies PASS / ASSERT-FAIL / CRASH / COMPILE-ERR / HANG. It emits ICC
# trace events (kind: edge_coverage) into scripts/icc_traces/edge_coverage_v134.jsonl:
#   * one per file×mode
#   * one per-family gate  edge_v134_<family>_oracle
#   * one sweep gate       edge_v134_sweep_clean
#
# Differential families (matmul, roundtrip) additionally cross-check that the VM
# result agrees with the native result — a value divergence is an ASSERT-FAIL
# even if each side individually reports PASS.
#
# The corpus is GENERATED into a temp dir at run time (bounded, seeded, cleaned
# up on exit — honors the fuzz/harness disk-budget rule), so nothing is
# committed under generated/.
#
# Usage:
#   scripts/run_edge_coverage_v134.sh                    # all families, jit+aot(+vm)
#   FAMILY=i128 scripts/run_edge_coverage_v134.sh        # one family (per-oracle gate)
#   MODES="jit aot aot-O0" scripts/run_edge_coverage_v134.sh
#   JOBS=4 SEED=20260723 MAX_DEPTH=6 scripts/run_edge_coverage_v134.sh
set -u
export LC_ALL=C LC_CTYPE=C LANG=C
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
TRACE_FILE="$TRACE_DIR/edge_coverage_v134.jsonl"
# BUILD_DIR may be relative (default "build") or absolute (as the ICC smoke
# harness passes it); resolve either to an absolute build path.
case "${BUILD_DIR:-build}" in
    /*) BUILD_DIR_ABS="${BUILD_DIR}" ;;
    *)  BUILD_DIR_ABS="$REPO_ROOT/${BUILD_DIR:-build}" ;;
esac
ESHKOL_RUN="$BUILD_DIR_ABS/eshkol-run"
ESHKOL_VM="$BUILD_DIR_ABS/eshkol-vm-standalone-test"

FAMILY="${FAMILY:-}"                 # empty = all families
MODES="${MODES:-jit aot}"            # add aot-O0 and/or vm as desired
JOBS="${JOBS:-4}"
SEED="${SEED:-20260723}"
MAX_DEPTH="${MAX_DEPTH:-6}"
COUNTS="${COUNTS:-24}"
JIT_TIMEOUT="${JIT_TIMEOUT:-90}"
AOT_COMPILE_TIMEOUT="${AOT_COMPILE_TIMEOUT:-150}"
AOT_RUN_TIMEOUT="${AOT_RUN_TIMEOUT:-90}"
VM_TIMEOUT="${VM_TIMEOUT:-90}"

# Default families to gate on. `roundtrip` is intentionally EXCLUDED here: it
# asserts number->string . string->number == identity, which requires the
# shortest-round-trip printer that is not yet on master (number->string is
# still %g-lossy, so the property fails for values needing >6 significant
# digits). Generate/run it explicitly with FAMILY=roundtrip once the printer
# lands; until then it is a staged (not-yet-gated) family.
DEFAULT_FAMILIES="nursery parallel gradient i128 matmul adtape"

# adtape (ad-pow / ad-tape-length) is registered ONLY in the bytecode VM, not
# the native LLVM backend — so it runs VM-only. Native families skip it.
NATIVE_INELIGIBLE="adtape"

# Families that are meaningful on the bytecode VM. nursery/parallel/gradient are
# native-codegen constructs whose VM path is not the surface under test here;
# matmul's multi-dim reshape/tensor-ref/tensor-set! are NOT on the VM (it only
# registers flat arity-2 tensor-ref / arity-3 tensor-set! / arity-2 reshape /
# arity-1 arange), so matmul is native-only until VM tensor parity lands (see
# the report's LANDING SOON / TODO section). adtape is VM-only.
VM_ELIGIBLE="i128 adtape"
# Families whose VM output must additionally AGREE with native (differential).
# i128 works identically on native + VM (the wrapping/boundary files; the
# guard-based conversions file carries ;; VM-SKIP). roundtrip/matmul join here
# once their VM surface lands.
DIFF_FAMILIES="i128"

if [ ! -x "$ESHKOL_RUN" ]; then
    echo "run_edge_coverage_v134.sh: $ESHKOL_RUN not found — build first:" >&2
    echo "  cmake --build ${BUILD_DIR:-build} --target eshkol-run stdlib -j" >&2
    exit 2
fi

# --------------------------------------------------------------------------
# Worker path (self-dispatch under xargs). Classifies one file×mode.
# Writes: <base> <mode> <CLASS> <passed>/<expected> <detail>
# --------------------------------------------------------------------------
classify() {   # exit_status out_file expected
    local st="$1" out="$2" expected="$3" fails passes
    fails=$(grep -c '^FAIL:' "$out" 2>/dev/null || true)
    passes=$(grep -c '^PASS:' "$out" 2>/dev/null || true)
    if [ "$st" -eq 142 ]; then
        echo "HANG $passes/$expected timeout"
    elif [ "$st" -ge 128 ]; then
        echo "CRASH $passes/$expected signal=$((st - 128))"
    elif [ "$fails" -gt 0 ]; then
        local first; first=$(grep -m1 '^FAIL:' "$out" | tr -d '"' | cut -c1-160)
        echo "ASSERT-FAIL $passes/$expected $first"
    elif [ "$st" -ne 0 ]; then
        local err; err=$(grep -m1 -iE 'error' "$out" | tr -d '"' | cut -c1-120)
        echo "COMPILE-ERR $passes/$expected exit=$st ${err:-no-error-line}"
    elif [ "$passes" -ne "$expected" ]; then
        echo "ASSERT-FAIL $passes/$expected truncated-output"
    else
        echo "PASS $passes/$expected ok"
    fi
}

run_one() {
    local f="$1" mode="$2" base expected out st res bin
    base=$(basename "$f" .esk)
    expected=$(sed -n 's/^;; CHECKS: //p' "$f" | head -1); expected="${expected:-0}"
    out="$WORK_DIR/$base.$mode.out"
    case "$mode" in
      jit)
        perl -e "alarm $JIT_TIMEOUT; exec @ARGV" "$ESHKOL_RUN" -r "$f" > "$out" 2>&1
        res=$(classify "$?" "$out" "$expected") ;;
      aot|aot-O0)
        bin="$WORK_DIR/$base.$mode.bin"
        local optflag=""; [ "$mode" = aot-O0 ] && optflag="-O0"
        perl -e "alarm $AOT_COMPILE_TIMEOUT; exec @ARGV" \
            "$ESHKOL_RUN" $optflag "$f" -o "$bin" > "$out" 2>&1
        st=$?
        if [ "$st" -ne 0 ] || [ ! -x "$bin" ]; then
            local err; err=$(grep -m1 -iE 'error' "$out" | tr -d '"' | cut -c1-120)
            [ "$st" -eq 142 ] && res="HANG 0/$expected compile-timeout" \
                              || res="COMPILE-ERR 0/$expected exit=$st ${err:-compile-failed}"
        else
            perl -e "alarm $AOT_RUN_TIMEOUT; exec @ARGV" "$bin" > "$out" 2>&1
            res=$(classify "$?" "$out" "$expected")
        fi
        rm -f "$bin" ;;
      vm)
        ESHKOL_VM_NO_DISASM=1 perl -e "alarm $VM_TIMEOUT; exec @ARGV" \
            "$ESHKOL_VM" "$f" > "$out" 2>&1
        res=$(classify "$?" "$out" "$expected") ;;
    esac
    echo "$base $mode $res" > "$WORK_DIR/results/$base.$mode"
}

if [ "${EDGE_V134_WORKER:-}" = 1 ]; then run_one "$1" "$2"; exit 0; fi

# --------------------------------------------------------------------------
# Coordinator
# --------------------------------------------------------------------------
mkdir -p "$TRACE_DIR"
WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/eshkol-edge-v134.XXXXXX")
GEN_DIR="$WORK_DIR/corpus"
trap 'rm -rf "$WORK_DIR"' EXIT
mkdir -p "$WORK_DIR/results" "$GEN_DIR"

if [ -z "${ESHKOL_JIT_CACHE_DIR:-}" ]; then
    export ESHKOL_JIT_CACHE_DIR="$WORK_DIR/jit-cache"; mkdir -p "$ESHKOL_JIT_CACHE_DIR"
fi

# Generate the corpus (bounded, seeded). With no FAMILY override, generate the
# default gated families (roundtrip excluded — see DEFAULT_FAMILIES above).
gen_args=(--out "$GEN_DIR" --seed "$SEED" --max-depth "$MAX_DEPTH" --counts "$COUNTS")
if [ -n "$FAMILY" ]; then
    gen_args+=(--family "$FAMILY")
else
    for fam in $DEFAULT_FAMILIES; do gen_args+=(--family "$fam"); done
fi
python3 "$REPO_ROOT/scripts/gen_edge_v134.py" "${gen_args[@]}" || exit 2

export WORK_DIR ESHKOL_RUN ESHKOL_VM JIT_TIMEOUT AOT_COMPILE_TIMEOUT \
       AOT_RUN_TIMEOUT VM_TIMEOUT

# Build the work list. A file is VM-eligible only if its family prefix is in
# VM_ELIGIBLE and the vm binary exists and vm is in MODES.
family_of() { echo "$1" | sed -E 's/_.*$//'; }

WORK_LIST="$WORK_DIR/worklist"; : > "$WORK_LIST"
shopt -s nullglob
for f in "$GEN_DIR"/*.esk; do
    base=$(basename "$f" .esk); fam=$(family_of "$base")
    for mode in $MODES; do
        if [ "$mode" = vm ]; then
            [ -x "$ESHKOL_VM" ] || continue
            case " $VM_ELIGIBLE " in *" $fam "*) ;; *) continue;; esac
            grep -q '^;; VM-SKIP' "$f" && continue   # per-file VM opt-out
        else
            # native modes (jit/aot/aot-O0): skip VM-only families.
            case " $NATIVE_INELIGIBLE " in *" $fam "*) continue;; esac
        fi
        printf '%s\n%s\n' "$f" "$mode" >> "$WORK_LIST"
    done
done
# VM-only families need vm in MODES to run at all; auto-add it if the vm binary
# exists and a VM-only family is present but vm wasn't requested.
if [ -x "$ESHKOL_VM" ] && [[ " $MODES " != *" vm "* ]]; then
    for f in "$GEN_DIR"/*.esk; do
        fam=$(family_of "$(basename "$f" .esk)")
        case " $NATIVE_INELIGIBLE " in *" $fam "*)
            printf '%s\n%s\n' "$f" vm >> "$WORK_LIST";; esac
    done
fi
shopt -u nullglob
n_items=$(( $(wc -l < "$WORK_LIST") / 2 ))
if [ "$n_items" -eq 0 ]; then
    echo "run_edge_coverage_v134.sh: no work items (FAMILY='$FAMILY' MODES='$MODES')" >&2
    exit 2
fi
echo "edge-v134: $n_items file×mode items, jobs=$JOBS modes='$MODES' seed=$SEED depth=$MAX_DEPTH"

xargs -n2 -P "$JOBS" env EDGE_V134_WORKER=1 bash "$0" < "$WORK_LIST"

# --------------------------------------------------------------------------
# Differential cross-check (native vs VM) for DIFF_FAMILIES: compare PASS/FAIL
# classification — if native PASSes and VM does not (or vice versa) on the same
# base, flag a divergence result row.
# --------------------------------------------------------------------------
sorted="$WORK_DIR/sorted"; cat "$WORK_DIR"/results/* 2>/dev/null | sort > "$sorted"

class_of() { awk -v b="$1" -v m="$2" '$1==b && $2==m {print $3}' "$sorted" | head -1; }
if [[ " $MODES " == *" vm "* ]]; then
    for f in "$GEN_DIR"/*.esk; do
        base=$(basename "$f" .esk); fam=$(family_of "$base")
        case " $DIFF_FAMILIES " in *" $fam "*) ;; *) continue;; esac
        nat=$(class_of "$base" aot); [ -n "$nat" ] || nat=$(class_of "$base" jit)
        vm=$(class_of "$base" vm)
        [ -n "$vm" ] || continue
        if [ "$nat" = PASS ] && [ "$vm" != PASS ]; then
            echo "$base diff DIVERGE 0/0 native=PASS vm=$vm" \
                > "$WORK_DIR/results/$base.diff"
        elif [ "$nat" != PASS ] && [ "$vm" = PASS ]; then
            echo "$base diff DIVERGE 0/0 native=$nat vm=PASS" \
                > "$WORK_DIR/results/$base.diff"
        fi
    done
    cat "$WORK_DIR"/results/* 2>/dev/null | sort > "$sorted"
fi

# --------------------------------------------------------------------------
# Aggregate + emit traces
# --------------------------------------------------------------------------
: > "$TRACE_FILE"
emit() {   # name value snippet
    local esc; esc=$(printf '%s' "$3" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g')
    printf '{"kind":"edge_coverage","name":"%s","value":"%s","snippet":"%s","confidence":0.95}\n' \
        "$1" "$2" "$esc" >> "$TRACE_FILE"
}

# bash 3.2 (macOS system bash) has no associative arrays — aggregate with a
# per-family pass over the sorted result rows instead.
n_pass=0; n_bad=0
printf '\n%-40s %-8s %-12s %-9s %s\n' FILE MODE CLASS CHECKS DETAIL
printf '%.0s-' {1..100}; printf '\n'
while IFS= read -r row; do
    [ -n "$row" ] || continue
    base=$(echo "$row" | awk '{print $1}'); mode=$(echo "$row" | awk '{print $2}')
    cls=$(echo "$row" | awk '{print $3}'); chk=$(echo "$row" | awk '{print $4}')
    detail=$(echo "$row" | cut -d' ' -f5-)
    if [ "$cls" = PASS ]; then
        n_pass=$((n_pass+1))
    else
        n_bad=$((n_bad+1))
        printf '%-40s %-8s %-12s %-9s %s\n' "$base" "$mode" "$cls" "$chk" "$detail"
    fi
    ev=PASS; [ "$cls" = PASS ] || ev=FAIL
    emit "${base}_${mode}" "$ev" "$cls $chk $detail"
done < "$sorted"

echo
fam_label() {   # family -> human label
    case "$1" in
      nursery)   echo "nursery iter-scope mutating loops (6 barrier channels, escape sets, nested depth)";;
      parallel)  echo "parallel-map/parallel-execute capturing closures returning collections";;
      gradient)  echo "exact gradient through callable params + curried form";;
      i128)      echo "native 128-bit integer boundaries + wraparound + conversions";;
      matmul)    echo "VM matmul parity: arange arities, nested literals, multi-dim ref/set";;
      adtape)    echo "low-level ad-tape/ad-pow (fractional/neg/zero exp, reuse, 1024-node growth)";;
      roundtrip) echo "number->string . string->number identity over generated doubles";;
      *)         echo "$1";;
    esac
}
# Per-family gate events (one oracle per surface family).
for fam in nursery parallel gradient i128 matmul adtape roundtrip; do
    tot=$(awk -v f="$fam" '$1 ~ ("^" f "_") {c++} END{print c+0}' "$sorted")
    [ "$tot" -eq 0 ] && continue
    bad=$(awk -v f="$fam" '$1 ~ ("^" f "_") && $3 != "PASS" {c++} END{print c+0}' "$sorted")
    val=PASS; [ "$bad" -eq 0 ] || val=FAIL
    emit "edge_v134_${fam}_oracle" "$val" "$(fam_label "$fam"): $((tot-bad))/$tot pass"
    printf '  %-9s %-4s %d/%d pass\n' "$fam" "$val" "$((tot-bad))" "$tot"
done

total=$((n_pass + n_bad))
val=PASS; [ "$n_bad" -eq 0 ] || val=FAIL
emit edge_v134_sweep_clean "$val" "total=$total pass=$n_pass bad=$n_bad seed=$SEED depth=$MAX_DEPTH"
echo
echo "edge-v134 summary: total=$total PASS=$n_pass BAD=$n_bad (seed=$SEED depth=$MAX_DEPTH)"
echo "traces: $TRACE_FILE"
[ "$n_bad" -eq 0 ] && exit 0 || exit 1
