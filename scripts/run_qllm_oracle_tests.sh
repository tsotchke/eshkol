#!/usr/bin/env bash
# run_qllm_oracle_tests.sh — qLLM geometric-primitive gradient oracle.
#
# Runs every exporter in tests/qllm_oracle/ under BOTH the JIT (-r) and AOT.
# Each exporter differentiates one of qLLM's geometric primitives with
# Eshkol's reverse-mode AD, self-checks the result against an in-language
# central finite difference (and, for poincare_project, against the exact
# rational forward Taylor tower), prints PASS:/FAIL: lines plus a
# Passed:/Failed: summary, and writes candidate golden-vector JSON into a
# lane-local scratch directory.
#
# The JSON in tests/qllm_oracle/golden/ is the committed independent reference:
# qLLM's fp32 C/torch/Metal tests consume it as a reference. This gate never
# overwrites that artifact. It regenerates a candidate in lane-local scratch
# and fails if the squared-distance candidate differs from the committed bytes,
# so a changed implementation cannot refresh its own oracle into green. The
# older geometric goldens remain schema-validated here; two of them are known
# to vary by host/compiler and are not rewritten by this task.
#
# Verdicts per file+mode:
#   PASS   ran to completion, no FAIL:, no nonzero Failed: summary
#   FAIL   an AD value diverged from its cross-check
#   CRASH  fatal signal, nonzero exit, or codegen/IR failure
#   HANG   exceeded the per-run timeout
#
# Usage: scripts/run_qllm_oracle_tests.sh [--no-aot] [--keep-json]
#   --no-aot     skip the AOT lane (JIT only)
#   --keep-json  write JSON from the AOT lane too (default: JIT lane only, so
#                the two lanes cannot race on the same output files)
set -u
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
GEN_DIR="$REPO_ROOT/tests/qllm_oracle"
OUT_DIR="$GEN_DIR/golden"
TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
TRACE_FILE="$TRACE_DIR/qllm_oracle.jsonl"
SCRATCH_DIR="${ORACLE_SCRATCH_DIR:-$REPO_ROOT/.scratch/qllm-oracle}"
GENERATED_DIR="$SCRATCH_DIR/generated"
AOT_BIN_DIR="$SCRATCH_DIR/aot-bin"
AOT_JSON_DIR="$SCRATCH_DIR/aot-json"
mkdir -p "$TRACE_DIR" "$OUT_DIR" "$GENERATED_DIR" "$AOT_BIN_DIR" "$AOT_JSON_DIR"
: "${TRACE_FILE:?}"; : > "$TRACE_FILE"
: "${ESHKOL_JIT_CACHE_DIR:=$SCRATCH_DIR/jit-cache}"
export ESHKOL_JIT_CACHE_DIR
mkdir -p "$ESHKOL_JIT_CACHE_DIR"

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
    /*) ESHKOL_RUN="$BUILD_DIR/eshkol-run" ;;
    *) ESHKOL_RUN="$REPO_ROOT/$BUILD_DIR/eshkol-run" ;;
esac
if [ ! -x "$ESHKOL_RUN" ]; then
    echo "run_qllm_oracle_tests.sh: $BUILD_DIR/eshkol-run not found — run \`cmake --build $BUILD_DIR --target eshkol-run stdlib\` first." >&2
    exit 2
fi

DO_AOT=1
KEEP_JSON_AOT=0
for arg in "$@"; do
    case "$arg" in
        --no-aot) DO_AOT=0 ;;
        --keep-json) KEEP_JSON_AOT=1 ;;
        *)
            echo "run_qllm_oracle_tests.sh: unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

JIT_TIMEOUT="${JIT_TIMEOUT:-240}"
AOT_COMPILE_TIMEOUT="${AOT_COMPILE_TIMEOUT:-300}"
AOT_RUN_TIMEOUT="${AOT_RUN_TIMEOUT:-120}"

# The exporters, in dependency order. qllm_oracle_lib.esk is a (load …)
# library, not a probe, so it is deliberately not listed.
EXPORTERS="
poincare_project.esk
poincare_retract.esk
sphere_ops.esk
poincare_maps.esk
sheaf_ee_step.esk
squared_distance.esk
"

# macOS has no `timeout(1)`; emulate with perl alarm (exit 142 on SIGALRM).
run_guarded() {
    perl -e 'my $seconds = shift; alarm $seconds; exec @ARGV; die "exec failed: $ARGV[0]: $!\n"' \
        "$1" "${@:2}"
}

json_escape() {
    printf '%s' "$1" | perl -0pe 's/\\/\\\\/g; s/"/\\"/g; s/\n/\\n/g; s/\r/\\r/g; s/\t/\\t/g; s/([\x00-\x08\x0b\x0c\x0e-\x1f])/sprintf("\\u%04x", ord($1))/ge'
}

emit_event() {
    local name="$1" value="$2" snippet="$3" esc_name esc_value esc_snippet
    esc_name=$(json_escape "$name")
    esc_value=$(json_escape "$value")
    esc_snippet=$(json_escape "$snippet")
    printf '{"kind":"qllm_oracle","name":"%s","value":"%s","snippet":"%s","confidence":0.95}\n' \
        "$esc_name" "$esc_value" "$esc_snippet" >> "$TRACE_FILE"
}

# args: rc out -> echoes PASS|FAIL|CRASH|HANG
verdict() {
    local rc="$1" out="$2"
    if [ "$rc" -eq 142 ]; then echo HANG; return; fi
    if [ "$rc" -ge 128 ] || printf '%s' "$out" | grep -q "fatal signal"; then
        echo CRASH; return
    fi
    if printf '%s' "$out" | grep -qE \
        "Failed to generate LLVM IR|JIT batch execution failed|LLVM module verification failed"; then
        echo CRASH; return
    fi
    if printf '%s' "$out" | grep -qE '^FAIL:|Failed:[[:space:]]+[1-9]'; then
        echo FAIL; return
    fi
    # A nonzero exit with no FAIL: line still means the exporter aborted.
    if [ "$rc" -ne 0 ]; then echo CRASH; return; fi
    if ! printf '%s' "$out" | grep -q '^Passed:'; then
        # never reached the summary — silent early death
        echo CRASH; return
    fi
    echo PASS
}

declare -i total=0 passed=0 failed=0 crashed=0 hung=0
BAD=""

count_verdict() { # verdict file mode
    local v="$1" f="$2" mode="$3" pyv
    total+=1
    emit_event "qllm_oracle_${f%.esk}_${mode}" "$v" "$f $mode -> $v"
    case "$v" in
        PASS)   passed+=1;  pyv="PASSED" ;;
        FAIL)   failed+=1;  pyv="FAILED"; BAD="$BAD $f:$mode=$v" ;;
        CRASH)  crashed+=1; pyv="FAILED"; BAD="$BAD $f:$mode=$v" ;;
        HANG)   hung+=1;    pyv="FAILED"; BAD="$BAD $f:$mode=$v" ;;
    esac
    printf '  %-6s tests/qllm_oracle/%s::%s\n' "$v" "$f" "$mode"
    echo "$pyv tests/qllm_oracle/$f::$mode"
}

echo "qLLM geometric gradient oracle -> $TRACE_FILE"
echo "golden vectors -> $OUT_DIR"
echo

for f in $EXPORTERS; do
    path="$GEN_DIR/$f"
    base="${f%.esk}"
    if [ ! -f "$path" ]; then
        echo "  MISSING tests/qllm_oracle/$f" >&2
        BAD="$BAD $f:missing"
        crashed+=1
        total+=1
        continue
    fi

    # ----- JIT (-r) : generate a candidate; never overwrite the golden -----
    rout=$(QLLM_ORACLE_OUT="$GENERATED_DIR" run_guarded "$JIT_TIMEOUT" "$ESHKOL_RUN" -r "$path" 2>&1); rrc=$?
    rv=$(verdict "$rrc" "$rout")
    count_verdict "$rv" "$f" "r"
    if [ "$rv" = "PASS" ]; then
        printf '%s\n' "$rout" | grep -E '^Passed:' | sed 's/^/         /'
    else
        printf '%s\n' "$rout" | grep -E '^FAIL:' | head -8 | sed 's/^/         /'
    fi

    # ----- AOT -----
    if [ "$DO_AOT" -eq 1 ]; then
        bin="$AOT_BIN_DIR/${base}.bin"; rm -f "$bin"
        cout=$(run_guarded "$AOT_COMPILE_TIMEOUT" "$ESHKOL_RUN" "$path" -o "$bin" 2>&1); crc=$?
        if [ "$crc" -ne 0 ] || [ ! -x "$bin" ] || printf '%s' "$cout" | grep -qE \
            "Failed to generate LLVM IR|LLVM module verification failed"; then
            if [ "$crc" -eq 142 ]; then av=HANG; else av=CRASH; fi
            printf '%s\n' "$cout" | tail -5 | sed 's/^/         /'
        else
            # By default the AOT lane writes to a scratch dir so the two lanes
            # cannot interleave writes into the committed golden/ tree.
            if [ "$KEEP_JSON_AOT" -eq 1 ]; then
                aot_out="$GENERATED_DIR"
            else
                aot_out="$AOT_JSON_DIR"
            fi
            aout=$(QLLM_ORACLE_OUT="$aot_out" run_guarded "$AOT_RUN_TIMEOUT" "$bin" 2>&1); arc=$?
            av=$(verdict "$arc" "$aout")
            if [ "$av" != "PASS" ]; then
                printf '%s\n' "$aout" | grep -E '^FAIL:' | head -8 | sed 's/^/         /'
            fi
        fi
        rm -f "$bin"
        count_verdict "$av" "$f" "aot"
    fi
done

# Every exporter must have produced parseable candidate JSON. The committed
# squared-distance reference must also match byte-for-byte; it is the external
# reference consumed by this task's bridge gate.
echo
jrc=1
if command -v python3 >/dev/null 2>&1; then
    jout=$(python3 - "$GENERATED_DIR" "$OUT_DIR" <<'PY'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
committed = pathlib.Path(sys.argv[2])
files = sorted(d.glob("*.json"))
if not files:
    print("no candidate golden JSON produced")
    sys.exit(1)
bad = 0
for f in files:
    try:
        obj = json.loads(f.read_text())
    except Exception as exc:
        print(f"  INVALID {f.name}: {exc}")
        bad += 1
        continue
    n = len(obj.get("cases", [])) if isinstance(obj, dict) else 0
    ref = committed / f.name
    if f.name == "squared_distance.json":
        if not ref.is_file() or f.read_bytes() != ref.read_bytes():
            print(f"  DRIFT {f.name}: candidate differs from committed reference")
            bad += 1
        else:
            print(f"  match {f.name}  schema_version={obj.get('schema_version')} cases={n}")
    else:
        print(f"  schema-ok {f.name}  schema_version={obj.get('schema_version')} cases={n}")
ref = committed / "squared_distance.json"
if not (d / ref.name).is_file():
    print(f"  MISSING candidate for committed reference {ref.name}")
    bad += 1
sys.exit(1 if bad else 0)
PY
) ; jrc=$?
    echo "golden JSON reference gate:"
    printf '%s\n' "$jout"
    if [ "$jrc" -ne 0 ]; then
        crashed+=1
        BAD="$BAD golden-json:reference-mismatch"
    fi
else
    echo "golden JSON validation: skipped (no python3)"
fi

echo
echo "qllm_oracle summary: total=$total passed=$passed failed=$failed crashed=$crashed hung=$hung"
[ -n "$BAD" ] && echo "qllm_oracle offenders:$BAD"

gate=PASS
if [ "$failed" -ne 0 ] || [ "$crashed" -ne 0 ] || [ "$hung" -ne 0 ] || [ "$total" -eq 0 ] || [ "$jrc" -ne 0 ]; then
    gate=FAIL
fi
emit_event "qllm_oracle_gate" "$gate" \
    "total=$total passed=$passed failed=$failed crashed=$crashed hung=$hung aot=$DO_AOT"

echo "qllm_oracle gate: $gate"
if [ "$gate" = "PASS" ]; then
    echo "PASSED tests/qllm_oracle::gate"
    exit 0
else
    echo "FAILED tests/qllm_oracle::gate"
    exit 1
fi
