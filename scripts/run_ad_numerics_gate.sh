#!/usr/bin/env bash
# Run exactly the P11 Taylor user-numerics JIT/AOT CTests and emit a scoped
# v1.4 receipt. Continuation campaign coverage remains a separate obligation.
set -euo pipefail
export LC_ALL=C LC_CTYPE=C LANG=C

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd -P)"
BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
  /*) BUILD_PATH="$(cd "$BUILD_DIR" 2>/dev/null && pwd -P || printf '%s' "$BUILD_DIR")" ;;
  *) BUILD_PATH="$REPO_ROOT/$BUILD_DIR" ;;
esac
TRACE_DIR="${TRACE_DIR:-$REPO_ROOT/scripts/icc_traces}"
. "$REPO_ROOT/scripts/lib/evidence_paths.sh"
. "$REPO_ROOT/scripts/lib/checked_write.sh"
. "$REPO_ROOT/scripts/lib/durable_work_root.sh"
. "$REPO_ROOT/scripts/lib/build_fingerprint.sh"
eshkol_evidence_abs_var TRACE_DIR "$REPO_ROOT"
mkdir -p "$TRACE_DIR"

if eshkol_durable_enabled; then
  RUN_DIR="$(eshkol_durable_prepare_dir ad-numerics-gate)"
else
  RUN_DIR="$(mktemp -d "$TRACE_DIR/ad-numerics-gate.XXXXXX")"
fi
JUNIT="$RUN_DIR/ctest.xml"
LOG="$RUN_DIR/ctest.verbose.log"
TRACE_FILE="$TRACE_DIR/ad_numerics_gate.jsonl"
eshkol_require_output_file_path "$TRACE_FILE"
: > "$TRACE_FILE"

emit_outcome() {
  local value="$1" message="$2"
  python3 - "$TRACE_FILE" "$value" "$message" "$RUN_DIR/receipt.json" <<'PY'
import json, os, sys, time
trace, value, message, receipt = sys.argv[1:]
event = {"kind":"ad_numerics", "name":"ad_taylor_p11_v14_user_numerics",
         "value":value, "snippet":message, "confidence":1.0,
         "receipt_path":receipt, "recorded_epoch":int(time.time())}
with open(trace, "a", encoding="utf-8") as stream:
    stream.write(json.dumps(event, separators=(",", ":")) + "\n")
    stream.flush(); os.fsync(stream.fileno())
PY
}
fail_infra() {
  local message="$1"
  echo "run_ad_numerics_gate.sh: INFRA: $message" >&2
  emit_outcome INFRA "$message" || true
  exit 2
}
fail_gate() {
  local message="$1"
  python3 "$REPO_ROOT/scripts/lib/ad_numerics_gate.py" --write-failure-receipt \
    "$REPO_ROOT" "$BUILD_PATH" "$RUN_DIR" "$message" || true
  echo "run_ad_numerics_gate.sh: FAIL: $message" >&2
  emit_outcome FAIL "$message" || true
  exit 1
}

if [ ! -d "$BUILD_PATH" ]; then
  fail_infra "missing configured build directory $BUILD_PATH"
fi
if ! python3 "$REPO_ROOT/scripts/lib/ad_numerics_gate.py" --preflight "$REPO_ROOT" "$BUILD_PATH"; then
  fail_infra "build configuration is absent or belongs to another source checkout"
fi
eshkol_emit_build_fingerprint_event "$RUN_DIR" run_ad_numerics_gate "$BUILD_PATH" eshkol-run

EXPECTED='^(taylor_numerics_runtime_smoke|taylor_numerics_aot_smoke)$'
if ! ctest --test-dir "$BUILD_PATH" --show-only=json-v1 > "$RUN_DIR/ctest-list.json" 2> "$RUN_DIR/ctest-list.stderr"; then
  fail_infra "CTest could not enumerate configured tests"
fi
if ! python3 "$REPO_ROOT/scripts/lib/ad_numerics_gate.py" --check-tests \
  "$RUN_DIR/ctest-list.json" "$REPO_ROOT" "$BUILD_PATH"; then
  fail_infra "configured CTest set does not include exactly the required P11 pair"
fi

set +e
ctest --test-dir "$BUILD_PATH" --parallel 1 --verbose --output-on-failure \
  --output-junit "$JUNIT" --timeout "${GATE_TIMEOUT:-900}" -R "$EXPECTED" > "$LOG" 2>&1
CTEST_STATUS=$?
set -e

eshkol_emit_build_fingerprint_event "$RUN_DIR" run_ad_numerics_gate "$BUILD_PATH" eshkol-run
if [ "$CTEST_STATUS" -ne 0 ]; then
  fail_gate "required JIT/AOT CTest run failed with exit $CTEST_STATUS"
fi
if ! python3 "$REPO_ROOT/scripts/lib/ad_numerics_gate.py" --verify-and-write \
  "$REPO_ROOT" "$BUILD_PATH" "$RUN_DIR"; then
  fail_gate "CTest evidence did not prove the exact source-bound 41/41 JIT/AOT cases"
fi

RECEIPT_SHA="$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$RUN_DIR/receipt.json")"
python3 - "$TRACE_FILE" "$RUN_DIR/receipt.json" "$RECEIPT_SHA" <<'PY'
import json, os, sys, time
trace, receipt, digest = sys.argv[1:]
event = {"kind":"ad_numerics", "name":"ad_taylor_p11_v14_user_numerics", "value":"PASS",
         "snippet":"Taylor-series ODE/root/inversion: exact JIT+AOT CTests, 41/41 checks each; source/build-bound receipt.",
         "confidence":1.0, "receipt_path":receipt, "receipt_sha256":digest,
         "recorded_epoch":int(time.time())}
with open(trace,"a",encoding="utf-8") as stream:
    stream.write(json.dumps(event,separators=(",",":"))+"\n")
    stream.flush(); os.fsync(stream.fileno())
PY

if eshkol_durable_enabled; then
  eshkol_durable_mirror_trace "$TRACE_FILE" ad_numerics_gate.jsonl
fi
echo "PASS: P11 v1.4 user numerics (JIT+AOT, 41/41 checks per case)"
echo "Receipt: $RUN_DIR/receipt.json"
echo "ICC trace: $TRACE_FILE"
