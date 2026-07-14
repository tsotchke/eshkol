#!/usr/bin/env bash
# run_differential.sh — multi-path differential execution harness (adversarial P1).
#
# Eshkol has several execution paths that MUST agree on every deterministic
# program: identical program + identical input => identical (exit code,
# normalized stdout) on every axis. ANY divergence is a bug by definition —
# no external oracle needed.
#
# Axes (native, always on):
#   jit          ./build/eshkol-run -r f.esk            (run-cache default ON)
#   jit-nocache  ESHKOL_JIT_CACHE=0 ./build/eshkol-run -r f.esk
#   aot-o0       ./build/eshkol-run -O0 f.esk -o bin && ./bin
#   aot-o2       ./build/eshkol-run -O2 f.esk -o bin && ./bin
#
# Axes (VM, opt-in via --with-vm; compared ONLY against each other):
#   vm-src       ./build/eshkol-vm-standalone-test f.esk   (VM's own mini-compiler)
#   vm-eskb      ./build/eshkol-run --profile hosted-vm --emit-eskb f.eskb f.esk
#                && ./build/eshkol-vm-standalone-test f.eskb
#   The VM's display appends a trailing newline per call (native paths do
#   not), and the VM supports a language subset — so VM axes cannot be
#   byte-compared against native axes yet. See tests/differential/README.md.
#
# A corpus file PASSES an axis pair when both axes produced the same exit
# code AND byte-identical normalized stdout. Compile failure, crash, timeout
# or nonzero exit on ONE path of a pair is a divergence => FAIL naming the
# pair. A file where EVERY axis exits nonzero also FAILs (corpus programs
# must be green — a broken program hides divergences).
#
# Emits (mirroring scripts/run_sicp_smoke.sh):
#   * pytest-style lines : "PASSED tests/differential/<file>::<axisA-vs-axisB>"
#   * ICC JSON-L events  : kind=differential_smoke into
#                          scripts/icc_traces/differential_smoke.jsonl,
#                          consumed by .icc/completion-oracles.yaml::differential-clean
#
# Usage: scripts/run_differential.sh [corpus_dir] [--with-vm] [--no-aot]
set -u
export LC_ALL=C LC_CTYPE=C LANG=C
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
TRACE_FILE="$TRACE_DIR/differential_smoke.jsonl"
: "${TRACE_FILE:?TRACE_FILE not set}"
mkdir -p "$TRACE_DIR"
: > "$TRACE_FILE"

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
    /*) : ;;
    *) BUILD_DIR="$REPO_ROOT/$BUILD_DIR" ;;
esac
ESHKOL_RUN="$BUILD_DIR/eshkol-run"
VM_BIN="$BUILD_DIR/eshkol-vm-standalone-test"
[ -x "$VM_BIN" ] || VM_BIN="$BUILD_DIR/eshkol-vm-standalone"

if [ ! -x "$ESHKOL_RUN" ]; then
    echo "run_differential.sh: $ESHKOL_RUN not found — run \`cmake --build build --target eshkol-run stdlib -j\` first." >&2
    exit 2
fi

CORPUS_DIR="tests/differential/corpus"
WITH_VM=0
DO_AOT=1
for arg in "$@"; do
    case "$arg" in
        --with-vm) WITH_VM=1 ;;
        --no-aot) DO_AOT=0 ;;
        --*)
            echo "run_differential.sh: unknown flag: $arg" >&2
            exit 2 ;;
        *) CORPUS_DIR="$arg" ;;
    esac
done
case "$CORPUS_DIR" in
    /*) : ;;
    *) CORPUS_DIR="$REPO_ROOT/$CORPUS_DIR" ;;
esac
if [ ! -d "$CORPUS_DIR" ]; then
    echo "run_differential.sh: corpus dir not found: $CORPUS_DIR" >&2
    exit 2
fi
if [ "$WITH_VM" -eq 1 ] && [ ! -x "$VM_BIN" ]; then
    echo "run_differential.sh: --with-vm requested but no VM binary at $VM_BIN" >&2
    exit 2
fi

# Isolated JIT cache so runs are reproducible and the cache axis is real.
WORK="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-differential.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
export ESHKOL_JIT_CACHE_DIR="$WORK/jit-cache"
mkdir -p "$ESHKOL_JIT_CACHE_DIR"

TIMEOUT_RUN="${DIFFERENTIAL_TIMEOUT:-90}"

# macOS has no `timeout(1)`; emulate with perl alarm (exit 124 on expiry).
run_guarded() { # seconds cmd...
    perl -e 'my $s=shift; eval { local $SIG{ALRM}=sub{ exit 124 }; alarm $s; exec @ARGV or exit 127; }' \
        "$1" "${@:2}"
}

json_escape() {
    printf '%s' "$1" | perl -0pe 's/\\/\\\\/g; s/"/\\"/g; s/\n/\\n/g; s/\r/\\r/g; s/\t/\\t/g; s/([\x00-\x08\x0b\x0c\x0e-\x1f])/sprintf("\\u%04x", ord($1))/ge'
}

emit_event() { # name value snippet
    printf '{"kind":"differential_smoke","name":"%s","value":"%s","snippet":"%s","confidence":0.95}\n' \
        "$(json_escape "$1")" "$(json_escape "$2")" "$(json_escape "$3")" >> "${TRACE_FILE:?}"
}

# Normalize captured stdout: strip harness/compiler noise lines, then drop
# leading blank lines (the VM's source-mode banner emits one more "\n" than
# its ESKB-mode banner; applied uniformly to every axis so pairs stay fair).
# MUST stay in sync with normalize() in scripts/gen_differential.py.
normalize() { # infile outfile
    perl -ne 'next if
        /^WARN/ or /^INFO:/ or /^DEBUG/ or
        /^\[ESKB\]/ or /^\s*\[compiled:/ or
        /^=== Eshkol VM/ or /^=== Execution complete ===/ or
        /^remark:/ or /^warning: <unknown>/;
        next if !$seen and /^\s*$/; $seen = 1; print' "$1" > "$2"
}

# run_axis <axis> <file> <outdir>  — writes <outdir>/<axis>.out (normalized)
# and <outdir>/<axis>.rc; echoes nothing.
run_axis() {
    local axis="$1" f="$2" d="$3" rc raw="$3/$1.raw"
    case "$axis" in
        jit)
            run_guarded "$TIMEOUT_RUN" "$ESHKOL_RUN" -r "$f" >"$raw" 2>"$d/$axis.err"; rc=$? ;;
        jit-nocache)
            ESHKOL_JIT_CACHE=0 run_guarded "$TIMEOUT_RUN" "$ESHKOL_RUN" -r "$f" >"$raw" 2>"$d/$axis.err"; rc=$? ;;
        aot-o0|aot-o2)
            local olvl="-O0"; [ "$axis" = "aot-o2" ] && olvl="-O2"
            local bin="$d/$axis.bin"
            run_guarded "$TIMEOUT_RUN" "$ESHKOL_RUN" "$olvl" "$f" -o "$bin" >"$d/$axis.compile.out" 2>"$d/$axis.compile.err"
            local crc=$?
            if [ "$crc" -ne 0 ] || [ ! -x "$bin" ]; then
                printf 'COMPILE-FAIL rc=%s\n' "$crc" >"$raw"; rc=125
            else
                run_guarded "$TIMEOUT_RUN" "$bin" >"$raw" 2>"$d/$axis.err"; rc=$?
            fi ;;
        vm-src)
            ESHKOL_VM_NO_DISASM=1 run_guarded "$TIMEOUT_RUN" "$VM_BIN" "$f" >"$raw" 2>"$d/$axis.err"; rc=$? ;;
        vm-eskb)
            local eskb="$d/prog.eskb"
            run_guarded "$TIMEOUT_RUN" "$ESHKOL_RUN" --profile hosted-vm --emit-eskb "$eskb" "$f" >"$d/$axis.compile.out" 2>"$d/$axis.compile.err"
            local erc=$?
            if [ "$erc" -ne 0 ] || [ ! -f "$eskb" ]; then
                printf 'ESKB-EMIT-FAIL rc=%s\n' "$erc" >"$raw"; rc=125
            else
                ESHKOL_VM_NO_DISASM=1 run_guarded "$TIMEOUT_RUN" "$VM_BIN" "$eskb" >"$raw" 2>"$d/$axis.err"; rc=$?
            fi ;;
    esac
    normalize "$raw" "$d/$axis.out"
    printf '%s' "$rc" > "$d/$axis.rc"
}

NATIVE_AXES="jit jit-nocache"
[ "$DO_AOT" -eq 1 ] && NATIVE_AXES="jit jit-nocache aot-o0 aot-o2"

total_pairs=0; passed_pairs=0; failed_pairs=0; files_run=0; files_diverged=0
divergent_files=""

echo "Differential harness → $TRACE_FILE"
echo "Corpus: $CORPUS_DIR"
echo "Native axes: $NATIVE_AXES$([ "$WITH_VM" -eq 1 ] && printf ' | VM axes: vm-src vm-eskb')"
echo

shopt -s nullglob
corpus_files=("$CORPUS_DIR"/*.esk)
if [ "${#corpus_files[@]}" -eq 0 ]; then
    echo "run_differential.sh: no .esk files in $CORPUS_DIR" >&2
    exit 2
fi

for f in "${corpus_files[@]}"; do
    base=$(basename "$f")
    rel="tests/differential/$base"
    d="$WORK/$base.d"; mkdir -p "$d"
    files_run=$((files_run+1))
    file_diverged=0

    for axis in $NATIVE_AXES; do run_axis "$axis" "$f" "$d"; done
    if [ "$WITH_VM" -eq 1 ]; then
        run_axis vm-src "$f" "$d"
        run_axis vm-eskb "$f" "$d"
    fi

    # Pairwise comparison over native axes.
    axes_arr=($NATIVE_AXES)
    n=${#axes_arr[@]}
    all_nonzero=1
    for axis in $NATIVE_AXES; do
        [ "$(cat "$d/$axis.rc")" = "0" ] && all_nonzero=0
    done
    i=0
    while [ $i -lt $n ]; do
        j=$((i+1))
        while [ $j -lt $n ]; do
            a="${axes_arr[$i]}"; b="${axes_arr[$j]}"
            pair="$a-vs-$b"
            ra=$(cat "$d/$a.rc"); rb=$(cat "$d/$b.rc")
            total_pairs=$((total_pairs+1))
            verdict=PASS
            reason="agree rc=$ra"
            if [ "$ra" != "$rb" ]; then
                verdict=FAIL; reason="exit codes differ: $a rc=$ra vs $b rc=$rb"
            elif ! cmp -s "$d/$a.out" "$d/$b.out"; then
                verdict=FAIL; reason="stdout differs (rc=$ra both)"
            elif [ "$all_nonzero" -eq 1 ]; then
                verdict=FAIL; reason="all axes exit nonzero (rc=$ra) — corpus program must be green"
            fi
            if [ "$verdict" = PASS ]; then
                passed_pairs=$((passed_pairs+1))
                printf '  PASS  %s::%s\n' "$rel" "$pair"
                echo "PASSED $rel::$pair"
            else
                failed_pairs=$((failed_pairs+1)); file_diverged=1
                printf '  FAIL  %s::%s  (%s)\n' "$rel" "$pair" "$reason"
                echo "FAILED $rel::$pair"
                echo "    --- $a (rc=$ra) first lines:"; head -5 "$d/$a.out" | sed 's/^/    | /'
                echo "    --- $b (rc=$rb) first lines:"; head -5 "$d/$b.out" | sed 's/^/    | /'
                diff "$d/$a.out" "$d/$b.out" | head -10 | sed 's/^/    diff> /'
            fi
            emit_event "differential_${base%.esk}_${pair}" "$verdict" "$reason"
            j=$((j+1))
        done
        i=$((i+1))
    done

    # VM pair (vm-src vs vm-eskb only — see header).
    if [ "$WITH_VM" -eq 1 ]; then
        pair="vm-src-vs-vm-eskb"
        ra=$(cat "$d/vm-src.rc"); rb=$(cat "$d/vm-eskb.rc")
        total_pairs=$((total_pairs+1))
        verdict=PASS; reason="agree rc=$ra"
        if [ "$ra" != "$rb" ]; then
            verdict=FAIL; reason="exit codes differ: vm-src rc=$ra vs vm-eskb rc=$rb"
        elif ! cmp -s "$d/vm-src.out" "$d/vm-eskb.out"; then
            verdict=FAIL; reason="stdout differs (rc=$ra both)"
        fi
        if [ "$verdict" = PASS ]; then
            passed_pairs=$((passed_pairs+1))
            printf '  PASS  %s::%s\n' "$rel" "$pair"
            echo "PASSED $rel::$pair"
        else
            failed_pairs=$((failed_pairs+1)); file_diverged=1
            printf '  FAIL  %s::%s  (%s)\n' "$rel" "$pair" "$reason"
            echo "FAILED $rel::$pair"
            diff "$d/vm-src.out" "$d/vm-eskb.out" | head -10 | sed 's/^/    diff> /'
        fi
        emit_event "differential_${base%.esk}_${pair}" "$verdict" "$reason"
    fi

    if [ "$file_diverged" -eq 1 ]; then
        files_diverged=$((files_diverged+1))
        divergent_files="$divergent_files $base"
    fi
done

echo
echo "Differential summary: $passed_pairs/$total_pairs axis pairs agree across $files_run programs."
if [ "$files_diverged" -ne 0 ]; then
    echo "DIVERGENT programs ($files_diverged):$divergent_files"
    emit_event "differential_corpus_clean" "FAIL" "$files_diverged/$files_run corpus programs diverge across axes:$divergent_files"
else
    echo "No divergences: every corpus program agrees on every axis pair."
    emit_event "differential_corpus_clean" "PASS" "$files_run corpus programs agree on all axis pairs ($total_pairs pairs)"
fi
echo "Trace written: $TRACE_FILE"

[ "$failed_pairs" -eq 0 ] || exit 1
exit 0
