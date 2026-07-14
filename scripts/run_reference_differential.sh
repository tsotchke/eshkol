#!/usr/bin/env bash
# run_reference_differential.sh — reference-implementation differential oracle
# (adversarial testing campaign, pillar P7a).
#
# EXTERNAL GROUND TRUTH. Every other differential harness in this repo diffs
# Eshkol against itself (across its own JIT/AOT axes) — self-consistency only.
# This harness runs each portable R7RS-small program from
# tests/reference-diff/corpus/ on THREE engines:
#
#   ref     : the reference R7RS Scheme (chibi-scheme), run with a fixed
#             import prologue (chibi mandates (import ...); Eshkol forbids it,
#             so the prologue is the ONLY textual difference — the program body
#             is byte-identical across engines).
#   esh-jit : build/eshkol-run -r <prog.esk>
#   esh-aot : build/eshkol-run -O0 <prog.esk> -o <BIN> && <BIN>
#
# All three stdouts are canonicalised by scripts/lib/normalize_scheme_output.py
# (documented cosmetic normalisation: booleans, char-in-datum spelling,
# exact/inexact + float-precision printing) so ONLY semantic differences flag.
#
# Classification per program:
#   AGREE                     ref ok AND both Eshkol paths ok AND all three
#                             normalised stdouts identical.  <- the good case
#   ESHKOL-DIVERGES           ref ok (exit 0) but an Eshkol path errored/crashed
#                             or produced different normalised output.
#                             ==> an Eshkol conformance bug vs ground truth.
#   REFERENCE-ERROR-ESHKOL-OK ref errored but Eshkol ran clean (usually means
#                             the corpus program is non-portable — a corpus bug).
#   BOTH-ERROR                both ref and Eshkol errored (non-portable program).
#
# DISK BUDGET: a single reused temp binary is compiled, run, then deleted every
# iteration — per-program binaries are NEVER accumulated. All artifacts live
# under artifacts/reference-diff/ and are capped at DISK_CAP_MB (default 1024);
# the run aborts if the cap is exceeded. Only divergences keep logs.
#
# Emits:
#   * pytest-style lines : "PASSED tests/reference-diff/<file>::agree"  etc.
#   * ICC JSON-L events  : kind="reference_diff" into
#                          scripts/icc_traces/reference_diff.jsonl,
#                          consumed by .icc/completion-oracles.yaml::reference-diff.
#     The gate event reference_diff_gate is PASS iff every program AGREES.
#
# Usage: scripts/run_reference_differential.sh [corpus_dir]
set -u

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

CORPUS_DIR="${1:-$REPO_ROOT/tests/reference-diff/corpus}"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
ESHKOL_RUN="$BUILD_DIR/eshkol-run"
NORMALIZE="python3 $REPO_ROOT/scripts/lib/normalize_scheme_output.py"

TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
TRACE_FILE="$TRACE_DIR/reference_diff.jsonl"
mkdir -p "$TRACE_DIR"
: "${TRACE_FILE:?TRACE_FILE must be set}"
: > "$TRACE_FILE"

ART_DIR="$REPO_ROOT/artifacts/reference-diff"
DIVERG_DIR="$ART_DIR/divergences"
WORK_DIR="$ART_DIR/work"
rm -rf "$DIVERG_DIR" "$WORK_DIR"
mkdir -p "$DIVERG_DIR" "$WORK_DIR"

DISK_CAP_MB="${DISK_CAP_MB:-1024}"
TIMEOUT_SECS="${TIMEOUT_SECS:-60}"

# Single reused temp binary + scratch source paths (NEVER per-program files).
AOT_BIN="$WORK_DIR/prog.bin"
ESK_SRC="$WORK_DIR/prog.esk"
SCM_SRC="$WORK_DIR/prog.scm"
: "${AOT_BIN:?AOT_BIN must be set}"
: "${ESK_SRC:?ESK_SRC must be set}"
: "${SCM_SRC:?SCM_SRC must be set}"

# ---- cleanup trap ---------------------------------------------------------
cleanup() { rm -f "$AOT_BIN" "$ESK_SRC" "$SCM_SRC"; rm -rf "$WORK_DIR"; }
trap cleanup EXIT INT TERM

# ---- reference Scheme discovery -------------------------------------------
REF_BIN=""
REF_NAME=""
REF_PROLOGUE=""
run_reference() {  # $1 = scm file (already has prologue); prints stdout
    "$REF_BIN" "$1"
}
if command -v chibi-scheme >/dev/null 2>&1; then
    REF_BIN="$(command -v chibi-scheme)"
    REF_NAME="chibi-scheme $("$REF_BIN" -V 2>/dev/null | head -1 | tr -d '\n')"
    REF_PROLOGUE='(import (scheme base) (scheme write) (scheme char) (scheme inexact) (scheme cxr))'
elif command -v guile >/dev/null 2>&1; then
    REF_BIN="$(command -v guile)"
    REF_NAME="guile $("$REF_BIN" --version 2>/dev/null | head -1 | tr -d '\n')"
    REF_PROLOGUE='(import (scheme base) (scheme write) (scheme char) (scheme inexact) (scheme cxr))'
    run_reference() { "$REF_BIN" --r7rs -s "$1"; }
elif command -v chez >/dev/null 2>&1 || command -v scheme >/dev/null 2>&1; then
    REF_BIN="$(command -v chez 2>/dev/null || command -v scheme)"
    REF_NAME="chezscheme"
    REF_PROLOGUE='(import (scheme base) (scheme write) (scheme char) (scheme inexact) (scheme cxr))'
    run_reference() { "$REF_BIN" --script "$1"; }
else
    echo "FATAL: no reference R7RS Scheme found (tried chibi-scheme, guile, chez/scheme)." >&2
    echo "Install one, e.g.: brew install chibi-scheme" >&2
    exit 2
fi

if [ ! -x "$ESHKOL_RUN" ]; then
    echo "FATAL: eshkol-run not found at $ESHKOL_RUN (build it: cmake --build build --target eshkol-run stdlib -j)" >&2
    exit 2
fi

# ---- macOS perl-alarm timeout guard (no coreutils `timeout` on stock macOS)-
# fork()+alarm so SIGALRM is caught in the PARENT (a bare exec would replace
# perl and lose the handler). Preserves distinct exit codes: 128+signal for a
# signalled child (timeout kill -> 137, SIGSEGV -> 139) so a genuine crash is
# still distinguishable from a timeout in logs.
run_guarded() {  # run_guarded <secs> <cmd...>
    # macOS does not provide the glibc-style C.UTF-8 locale.  Perl resolves
    # the inherited locale before it executes this wrapper and otherwise
    # aborts with exit 9 before either oracle starts.  The corpus is ASCII and
    # each engine performs its own UTF-8 handling, so the portable C locale is
    # the correct deterministic environment for the timeout supervisor.
    LC_ALL=C LANG=C perl -e '
        my $t = shift;
        my $pid = fork();
        exit 127 unless defined $pid;
        if ($pid == 0) { exec @ARGV; exit 127; }
        $SIG{ALRM} = sub { kill "KILL", $pid; };
        alarm $t;
        waitpid($pid, 0);
        my $st = $?;
        alarm 0;
        if ($st & 127) { exit(128 + ($st & 127)); }
        exit($st >> 8);
    ' "$@"
}

disk_used_mb() { du -sm "$ART_DIR" 2>/dev/null | awk '{print $1}'; }
check_disk() {
    local mb; mb="$(disk_used_mb)"; mb="${mb:-0}"
    if [ "$mb" -gt "$DISK_CAP_MB" ]; then
        echo "FATAL: artifacts/reference-diff exceeded ${DISK_CAP_MB}MB (at ${mb}MB) — aborting." >&2
        exit 3
    fi
}

json_escape() { python3 -c 'import json,sys; sys.stdout.write(json.dumps(sys.stdin.read()))'; }
emit_trace() {  # emit_trace <name> <value> <snippet>
    local name="$1" value="$2" snippet="$3"
    local esnip; esnip="$(printf '%s' "$snippet" | json_escape)"
    printf '{"kind":"reference_diff","name":"%s","value":"%s","snippet":%s,"confidence":0.95}\n' \
        "$name" "$value" "$esnip" >> "${TRACE_FILE:?}"
}

# ---- run all programs -----------------------------------------------------
echo "Reference implementation : $REF_NAME"
echo "Reference invocation     : $REF_BIN <prog.scm>  (prologue: $REF_PROLOGUE)"
echo "Eshkol                   : $ESHKOL_RUN  (-r and -O0 AOT)"
echo "Corpus                   : $CORPUS_DIR"
echo "Disk cap                 : ${DISK_CAP_MB}MB   Per-run timeout: ${TIMEOUT_SECS}s"
echo "------------------------------------------------------------------------"

TOTAL=0; AGREE=0; DIVERGE=0; REFERR=0; BOTHERR=0
DIVERGENCE_LIST=""

for scm in "$CORPUS_DIR"/*.scm; do
    [ -e "$scm" ] || continue
    base="$(basename "$scm" .scm)"
    TOTAL=$((TOTAL+1))

    # Program body (strip our leading ';;' header comment lines? keep them —
    # both engines ignore comments). Build the two engine sources.
    cp "$scm" "$ESK_SRC"
    { printf '%s\n' "$REF_PROLOGUE"; cat "$scm"; } > "${SCM_SRC:?}"

    # --- reference ---
    ref_out="$(run_guarded "$TIMEOUT_SECS" "$REF_BIN" "$SCM_SRC" 2>/dev/null)"; ref_ec=$?
    ref_norm="$(printf '%s' "$ref_out" | $NORMALIZE)"

    # --- eshkol JIT ---
    jit_out="$(run_guarded "$TIMEOUT_SECS" "$ESHKOL_RUN" -r "$ESK_SRC" 2>/dev/null)"; jit_ec=$?
    jit_norm="$(printf '%s' "$jit_out" | $NORMALIZE)"

    # --- eshkol AOT (single reused binary, deleted after) ---
    rm -f -- "${AOT_BIN:?}"
    aot_comp_err="$(run_guarded "$TIMEOUT_SECS" "$ESHKOL_RUN" -O0 "$ESK_SRC" -o "$AOT_BIN" 2>&1)"; aot_comp_ec=$?
    if [ "$aot_comp_ec" -eq 0 ] && [ -x "$AOT_BIN" ]; then
        aot_out="$(run_guarded "$TIMEOUT_SECS" "$AOT_BIN" 2>/dev/null)"; aot_ec=$?
    else
        aot_out=""; aot_ec=$aot_comp_ec
    fi
    aot_norm="$(printf '%s' "$aot_out" | $NORMALIZE)"
    rm -f -- "${AOT_BIN:?}"

    ref_ok=0; [ "$ref_ec" -eq 0 ] && ref_ok=1
    jit_ok=0; [ "$jit_ec" -eq 0 ] && jit_ok=1
    aot_ok=0; [ "$aot_ec" -eq 0 ] && aot_ok=1
    esh_ok=0; [ "$jit_ok" -eq 1 ] && [ "$aot_ok" -eq 1 ] && esh_ok=1

    # --- classify ---
    class=""; note=""
    if [ "$ref_ok" -eq 1 ]; then
        if [ "$esh_ok" -eq 1 ] && [ "$jit_norm" = "$ref_norm" ] && [ "$aot_norm" = "$ref_norm" ]; then
            class="AGREE"
        else
            class="ESHKOL-DIVERGES"
            if [ "$jit_ok" -eq 0 ] || [ "$aot_ok" -eq 0 ]; then
                note="eshkol errored (jit_ec=$jit_ec aot_ec=$aot_ec) while reference exit 0"
            elif [ "$jit_norm" != "$aot_norm" ]; then
                note="eshkol JIT vs AOT also disagree"
            else
                note="eshkol output differs from reference"
            fi
        fi
    else
        if [ "$esh_ok" -eq 1 ]; then
            class="REFERENCE-ERROR-ESHKOL-OK"
            note="reference exit $ref_ec (likely non-portable corpus program)"
        else
            class="BOTH-ERROR"
            note="reference exit $ref_ec, eshkol jit_ec=$jit_ec aot_ec=$aot_ec"
        fi
    fi

    case "$class" in
        AGREE)
            AGREE=$((AGREE+1))
            echo "PASSED tests/reference-diff/$base::agree"
            emit_trace "$base" "AGREE" "$base: all three engines agree"
            ;;
        ESHKOL-DIVERGES)
            DIVERGE=$((DIVERGE+1))
            DIVERGENCE_LIST="$DIVERGENCE_LIST $base"
            echo "FAILED tests/reference-diff/$base::eshkol-diverges  ($note)"
            emit_trace "$base" "ESHKOL-DIVERGES" "$base: $note"
            d="$DIVERG_DIR/$base"; mkdir -p "$d"
            cp "$scm" "$d/program.scm"
            printf '%s' "$ref_out" > "$d/reference.stdout"
            printf '%s' "$jit_out" > "$d/eshkol-jit.stdout"
            printf '%s' "$aot_out" > "$d/eshkol-aot.stdout"
            printf '%s' "$aot_comp_err" > "$d/eshkol-aot.compile.log"
            printf 'class=%s\nnote=%s\nref_ec=%s jit_ec=%s aot_ec=%s\n--- ref_norm ---\n%s\n--- jit_norm ---\n%s\n--- aot_norm ---\n%s\n' \
                "$class" "$note" "$ref_ec" "$jit_ec" "$aot_ec" "$ref_norm" "$jit_norm" "$aot_norm" > "$d/SUMMARY.txt"
            ;;
        REFERENCE-ERROR-ESHKOL-OK)
            REFERR=$((REFERR+1))
            echo "SKIPPED tests/reference-diff/$base::reference-error  ($note)"
            emit_trace "$base" "REFERENCE-ERROR-ESHKOL-OK" "$base: $note"
            ;;
        BOTH-ERROR)
            BOTHERR=$((BOTHERR+1))
            echo "SKIPPED tests/reference-diff/$base::both-error  ($note)"
            emit_trace "$base" "BOTH-ERROR" "$base: $note"
            ;;
    esac

    check_disk
done

# ---- gate + summary -------------------------------------------------------
GATE="PASS"
[ "$AGREE" -eq "$TOTAL" ] || GATE="FAIL"
emit_trace "reference_diff_gate" "$GATE" \
    "total=$TOTAL agree=$AGREE eshkol-diverges=$DIVERGE reference-error=$REFERR both-error=$BOTHERR reference=$REF_NAME"

PEAK_MB="$(disk_used_mb)"; PEAK_MB="${PEAK_MB:-0}"
echo "------------------------------------------------------------------------"
echo "Reference     : $REF_NAME"
echo "Total         : $TOTAL"
echo "AGREE         : $AGREE"
echo "ESHKOL-DIVERGES : $DIVERGE   ->$DIVERGENCE_LIST"
echo "REFERENCE-ERROR-ESHKOL-OK : $REFERR"
echo "BOTH-ERROR    : $BOTHERR"
if [ "$TOTAL" -gt 0 ]; then
    echo "Agreement rate: $(python3 -c "print('%.1f%%' % (100.0*$AGREE/$TOTAL))")"
fi
echo "Gate          : $GATE  (PASS iff every program AGREES)"
echo "Artifacts     : ${PEAK_MB}MB / ${DISK_CAP_MB}MB cap  ($ART_DIR)"
echo "Trace         : $TRACE_FILE"

# The oracle is intended to be RED while conformance bugs remain (see
# .icc/completion-oracles.yaml). Exit 0 so this can run as a reporting step;
# consumers gate on the trace event, not the exit code.
exit 0
