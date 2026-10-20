#!/usr/bin/env bash
# Run the registered JIT and AOT order-8 GUW analytic oracle tests, then emit
# the P4 ad_exact event consumed by the v1.4 completion oracle.
set -u
export LC_ALL=C LC_CTYPE=C LANG=C

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
  /*) BUILD_PATH="$BUILD_DIR" ;;
  *) BUILD_PATH="$REPO_ROOT/$BUILD_DIR" ;;
esac
TRACE_DIR="${TRACE_DIR:-$REPO_ROOT/scripts/icc_traces}"
TRACE_FILE="$TRACE_DIR/ad_exact_taylor.jsonl"
emit_event() {
  python3 - "$TRACE_FILE" "$1" "$2" <<'PY'
import json, os, sys
event = {"kind": "ad_exact", "name": "ad_taylor_p4_guw_multivariate",
         "value": sys.argv[2], "snippet": sys.argv[3], "confidence": 1.0}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    f.write(json.dumps(event, separators=(",", ":")) + "\n")
    f.flush()
    os.fsync(f.fileno())
PY
}
if ! mkdir -p "$TRACE_DIR"; then
  echo "run_ad_exact_taylor_gate.sh: cannot create trace directory $TRACE_DIR" >&2
  exit 2
fi
if ! : > "$TRACE_FILE"; then
  echo "run_ad_exact_taylor_gate.sh: cannot open trace file $TRACE_FILE" >&2
  exit 2
fi
fail_gate() {
  status="$1"; shift
  if ! emit_event "$status" "$*"; then
    echo "run_ad_exact_taylor_gate.sh: could not write ICC failure event to $TRACE_FILE" >&2
    exit 2
  fi
  echo "run_ad_exact_taylor_gate.sh: $*" >&2
  exit 1
}

if [ ! -r "$BUILD_PATH/CTestTestfile.cmake" ]; then
  if ! emit_event INFRA "no CTest configuration in $BUILD_PATH; configure and build first"; then
    echo "run_ad_exact_taylor_gate.sh: could not write ICC infrastructure event to $TRACE_FILE" >&2
    exit 2
  fi
  echo "run_ad_exact_taylor_gate.sh: no CTest configuration in $BUILD_PATH; configure and build first" >&2
  exit 2
fi
if [ ! -x "$BUILD_PATH/eshkol-run" ]; then
  if ! emit_event INFRA "missing executable $BUILD_PATH/eshkol-run; build target eshkol-run first"; then
    echo "run_ad_exact_taylor_gate.sh: could not write ICC infrastructure event to $TRACE_FILE" >&2
    exit 2
  fi
  echo "run_ad_exact_taylor_gate.sh: missing executable $BUILD_PATH/eshkol-run; build target eshkol-run first" >&2
  exit 2
fi

CTEST="${CTEST:-ctest}"
GATE_TIMEOUT="${GATE_TIMEOUT:-900}"
RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ad-exact-taylor.XXXXXX")" || {
  if ! emit_event INFRA "could not create temporary CTest report directory"; then
    echo "run_ad_exact_taylor_gate.sh: could not write ICC infrastructure event to $TRACE_FILE" >&2
  fi
  exit 2
}
trap 'rm -rf -- "$RUN_DIR"' EXIT
JUNIT="$RUN_DIR/ctest.xml"
LOG="$RUN_DIR/ctest.log"
EXPECTED='ad_guw_order8_(jit|aot)_smoke'

echo "== AD exact Taylor / GUW order-8 gate =="
if ! "$CTEST" --test-dir "$BUILD_PATH" --output-on-failure \
      --output-junit "$JUNIT" --timeout "$GATE_TIMEOUT" \
      -R "^${EXPECTED}$" >"$LOG" 2>&1; then
  cat "$LOG"
  fail_gate FAIL "required GUW order-8 JIT/AOT CTest run failed or timed out"
fi
cat "$LOG"

# Do not trust CTest's overall exit alone: renamed/configured-out tests can
# produce a vacuous green run. Require exactly the two named cases, no skips,
# and zero failure/error counts in the machine-readable report.
if ! python3 - "$JUNIT" <<'PY'
import sys
import xml.etree.ElementTree as ET

expected = {"ad_guw_order8_jit_smoke", "ad_guw_order8_aot_smoke"}
try:
    root = ET.parse(sys.argv[1]).getroot()
except (OSError, ET.ParseError) as exc:
    print(f"invalid or missing CTest JUnit report: {exc}", file=sys.stderr)
    raise SystemExit(1)
cases = root.findall(".//testcase")
names = {case.attrib.get("name", "") for case in cases}
if names != expected or len(cases) != 2:
    print(f"required tests mismatch: expected={sorted(expected)} observed={[c.attrib.get('name') for c in cases]}", file=sys.stderr)
    raise SystemExit(1)
bad = [c.attrib.get("name", "?") for c in cases
       if c.find("failure") is not None or c.find("error") is not None
       or c.find("skipped") is not None]
if bad:
    print(f"failed, errored, or skipped tests: {bad}", file=sys.stderr)
    raise SystemExit(1)
print("CTest report: JIT and AOT GUW order-8 tests passed (2/2)")
PY
then
  fail_gate FAIL "CTest report did not prove both required tests (missing, skipped, or malformed report)"
fi

if ! emit_event PASS "GUW analytic mixed partial D^(3,3,2) of x^3*y^3*z^2 equals 72 with permutation symmetry; JIT+AOT CTest passed 2/2"; then
  echo "run_ad_exact_taylor_gate.sh: could not write ICC PASS event to $TRACE_FILE" >&2
  exit 2
fi
echo "PASS: GUW analytic order-8 oracle (JIT+AOT, 2/2 CTest cases)"
echo "$TRACE_FILE written"
