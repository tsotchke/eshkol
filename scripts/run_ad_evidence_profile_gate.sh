#!/usr/bin/env bash
# Shared launcher for immutable P11 and P6 AD evidence profiles.
set -euo pipefail
export LC_ALL=C LC_CTYPE=C LANG=C
PROFILE="${1:-p11}"
case "$PROFILE" in
  p11) FIXTURE="tests/ad/taylor_numerics_test.esk"; EXPECTED='^(taylor_numerics_runtime_smoke|taylor_numerics_aot_smoke)$'; HARNESS=run_ad_numerics_gate; TRACE_NAME=ad_numerics_gate.jsonl; CACHE_SUBDIR=jit-cache ;;
  p6) FIXTURE="tests/ad/exact_taylor_test.esk"; EXPECTED='^(exact_taylor_runtime_smoke|exact_taylor_aot_smoke|exact_taylor_vm_runtime_smoke)$'; HARNESS=run_ad_p6_gate; TRACE_NAME=ad_exact_coefficients_gate.jsonl; CACHE_SUBDIR=orc-cache ;;
  *) echo "unknown AD evidence profile: $PROFILE" >&2; exit 2 ;;
esac

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
  RUN_DIR="$(eshkol_durable_prepare_dir ad-${PROFILE}-gate)"
else
  RUN_DIR="$(mktemp -d "$TRACE_DIR/ad-${PROFILE}-gate.XXXXXX")"
fi
# Isolate all runtime caches under the owned evidence root. Force the runtime
# case through ORC rather than the persistent AOT executable cache.
export ESHKOL_JIT_CACHE_DIR="$RUN_DIR/$CACHE_SUBDIR"
export ESHKOL_JIT_CACHE=0
mkdir -p "$ESHKOL_JIT_CACHE_DIR"

JUNIT="$RUN_DIR/ctest.xml"
LOG="$RUN_DIR/ctest.verbose.log"
TRACE_FILE="$TRACE_DIR/$TRACE_NAME"
eshkol_require_output_file_path "$TRACE_FILE"
: > "$TRACE_FILE"

emit_outcome() {
  local value="$1" message="$2"
  python3 - "$TRACE_FILE" "$value" "$message" "$RUN_DIR/receipt.json" "$PROFILE" <<'PY'
import json, os, sys, time
trace, value, message, receipt, profile = sys.argv[1:]
kind, name = {"p11":("ad_numerics","ad_taylor_p11_v14_user_numerics"),
              "p6":("ad_exact","ad_taylor_p6_exact_coefficients")}[profile]
event = {"kind":kind, "name":name,
         "value":value, "snippet":message, "confidence":1.0,
         "receipt_path":receipt, "recorded_epoch":int(time.time())}
with open(trace, "a", encoding="utf-8") as stream:
    stream.write(json.dumps(event, separators=(",", ":")) + "\n")
    stream.flush(); os.fsync(stream.fileno())
PY
}
fail_infra() {
  local message="$1"
  echo "run_ad_evidence_profile_gate.sh[$PROFILE]: INFRA: $message" >&2
  emit_outcome INFRA "$message" || true
  exit 2
}
fail_gate() {
  local message="$1"
  python3 "$REPO_ROOT/scripts/lib/ad_numerics_gate.py" --write-failure-receipt \
    "$REPO_ROOT" "$BUILD_PATH" "$RUN_DIR" "$message" --profile "$PROFILE" || true
  echo "run_ad_evidence_profile_gate.sh[$PROFILE]: FAIL: $message" >&2
  emit_outcome FAIL "$message" || true
  exit 1
}

if [ ! -d "$BUILD_PATH" ]; then
  fail_infra "missing configured build directory $BUILD_PATH"
fi
if ! python3 "$REPO_ROOT/scripts/lib/ad_numerics_gate.py" --profile "$PROFILE" --preflight "$REPO_ROOT" "$BUILD_PATH"; then
  fail_infra "build configuration is absent or belongs to another source checkout"
fi
eshkol_emit_build_fingerprint_event "$RUN_DIR" "$HARNESS" "$BUILD_PATH" eshkol-run
if [ "$PROFILE" = p6 ]; then
  eshkol_emit_build_fingerprint_event "$RUN_DIR" "$HARNESS" "$BUILD_PATH" eshkol-vm-standalone-test
fi

if ! ctest --test-dir "$BUILD_PATH" --show-only=json-v1 > "$RUN_DIR/ctest-list.json" 2> "$RUN_DIR/ctest-list.stderr"; then
  fail_infra "CTest could not enumerate configured tests"
fi
if ! python3 "$REPO_ROOT/scripts/lib/ad_numerics_gate.py" --profile "$PROFILE" --check-tests \
  "$RUN_DIR/ctest-list.json" "$REPO_ROOT" "$BUILD_PATH"; then
  fail_infra "configured CTest set does not include exactly the required $PROFILE profile tests"
fi

set +e
ctest --test-dir "$BUILD_PATH" --parallel 1 --verbose --output-on-failure \
  --output-junit "$JUNIT" --timeout "${GATE_TIMEOUT:-900}" -R "$EXPECTED" > "$LOG" 2>&1
CTEST_STATUS=$?
set -e

eshkol_emit_build_fingerprint_event "$RUN_DIR" "$HARNESS" "$BUILD_PATH" eshkol-run
if [ "$PROFILE" = p6 ]; then
  eshkol_emit_build_fingerprint_event "$RUN_DIR" "$HARNESS" "$BUILD_PATH" eshkol-vm-standalone-test
fi
if [ "$CTEST_STATUS" -ne 0 ]; then
  fail_gate "required $PROFILE CTest run failed with exit $CTEST_STATUS"
fi
if ! python3 "$REPO_ROOT/scripts/lib/ad_numerics_gate.py" --profile "$PROFILE" --verify-and-write \
  "$REPO_ROOT" "$BUILD_PATH" "$RUN_DIR"; then
  fail_gate "CTest evidence did not prove the exact source-bound profile cases"
fi

RECEIPT_SHA="$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$RUN_DIR/receipt.json")"
python3 - "$TRACE_FILE" "$RUN_DIR/receipt.json" "$RECEIPT_SHA" "$PROFILE" <<'PY'
import json, os, sys, time
trace, receipt, digest, profile = sys.argv[1:]
profiles = {
  "p11": ("ad_numerics", "ad_taylor_p11_v14_user_numerics", "Taylor-series ODE/root/inversion: source/build-bound JIT+AOT CTests, 41/41 checks each."),
  "p6": ("ad_exact", "ad_taylor_p6_exact_coefficients", "Exact rational/bignum Taylor coefficient oracle: source/build-bound JIT+AOT+direct VM CTests, 192/192 checks each; uncached JIT."),
}
kind, name, snippet = profiles[profile]
event = {"kind":kind, "name":name, "value":"PASS", "snippet":snippet,
         "confidence":1.0, "receipt_path":receipt, "receipt_sha256":digest,
         "recorded_epoch":int(time.time())}
with open(trace,"a",encoding="utf-8") as stream:
    stream.write(json.dumps(event,separators=(",",":"))+"\n")
    stream.flush(); os.fsync(stream.fileno())
PY

if eshkol_durable_enabled; then
  eshkol_durable_mirror_trace "$TRACE_FILE" "$TRACE_NAME"
fi
echo "PASS: $PROFILE AD evidence profile"
echo "Receipt: $RUN_DIR/receipt.json"
echo "ICC trace: $TRACE_FILE"
