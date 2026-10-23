#!/usr/bin/env bash
# core.symbolic coefficient fields (Q and GF(p)), the series coefficient norm
# and polynomial-valued duals: runs tests/stdlib/symbolic_test.esk (the Q
# suite) and tests/stdlib/symbolic_fields_test.esk under JIT and AOT, and
# emits the ICC runtime_event `core_symbolic_fields_norm_duals` into
# $TRACE_DIR/symbolic_fields_gate.jsonl (PASS only when all four runs pass).
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
TRACE_DIR="${TRACE_DIR:-$REPO_ROOT/scripts/icc_traces}"
ESHKOL_RUN="${ESHKOL_RUN:-$BUILD_DIR/eshkol-run}"
TRACE_FILE="$TRACE_DIR/symbolic_fields_gate.jsonl"
WORK="$BUILD_DIR/symbolic-fields-gate"
mkdir -p "$TRACE_DIR" "$WORK"
: > "$TRACE_FILE"
export ESHKOL_PATH="${ESHKOL_PATH:-$REPO_ROOT/lib}" ESHKOL_LIB_DIR="${ESHKOL_LIB_DIR:-$BUILD_DIR}" ESHKOL_JIT_CACHE=0

emit() {
  python3 - "$TRACE_FILE" "$1" "$2" <<'PY'
import json, os, sys, time
trace, value, snippet = sys.argv[1:]
event = {"kind": "symbolic_exact", "name": "core_symbolic_fields_norm_duals", "value": value,
         "snippet": snippet, "confidence": 1.0, "recorded_epoch": int(time.time())}
with open(trace, "a", encoding="utf-8") as stream:
    stream.write(json.dumps(event, separators=(",", ":")) + "\n")
    stream.flush(); os.fsync(stream.fileno())
PY
}

if [ ! -x "$ESHKOL_RUN" ]; then
  emit INFRA "eshkol-run not found at $ESHKOL_RUN"; exit 2
fi

summary=""
status=0
for suite in symbolic symbolic_fields; do
  src="$REPO_ROOT/tests/stdlib/${suite}_test.esk"
  jit_out="$("$ESHKOL_RUN" -r "$src" "-L$BUILD_DIR" 2>&1)"; jit_rc=$?
  "$ESHKOL_RUN" -O 1 -o "$WORK/${suite}_aot" "$src" "-L$BUILD_DIR" > "$WORK/${suite}_aot.compile.log" 2>&1
  aot_out="$("$WORK/${suite}_aot" 2>&1)"; aot_rc=$?
  for engine in jit aot; do
    if [ "$engine" = jit ]; then out="$jit_out"; rc=$jit_rc; else out="$aot_out"; rc=$aot_rc; fi
    passed="$(printf '%s\n' "$out" | sed -n 's/^Passed: //p' | tail -1)"
    failed="$(printf '%s\n' "$out" | sed -n 's/^Failed: //p' | tail -1)"
    summary="$summary ${suite}/${engine}=${passed:-?}/${failed:-?}"
    if [ "$rc" -ne 0 ] || [ "${failed:-x}" != "0" ] || printf '%s\n' "$out" | grep -q '^FAIL:'; then
      status=1
      echo "FAIL: $suite under $engine (exit $rc, failed=${failed:-?})"
      printf '%s\n' "$out" | grep '^FAIL:' | head -20
    else
      echo "PASS: $suite under $engine (${passed} checks)"
    fi
  done
done

if [ "$status" -eq 0 ]; then
  emit PASS "core.symbolic Q and GF(p) suites, series norm and polynomial-valued duals pass under JIT and AOT (passed/failed:$summary)"
  echo "PASS: core_symbolic_fields_norm_duals"
else
  emit FAIL "core.symbolic gate (passed/failed:$summary)"
  echo "FAIL: core_symbolic_fields_norm_duals"
fi
exit "$status"
