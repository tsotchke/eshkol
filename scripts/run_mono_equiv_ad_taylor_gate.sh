#!/usr/bin/env bash
# Execute the P2/P3 AD Taylor-tower equivalence gate under both JIT and AOT.
# Runtime evidence is written only after both execution modes prove that the
# literal-K monomorphized tower is bit-exact with the runtime tower through
# order eight (the former JET8 ceiling).
set -euo pipefail

cd "$(dirname "$0")/.."

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
  /*) ESHKOL_RUN="$BUILD_DIR/eshkol-run" ;;
  *) ESHKOL_RUN="$PWD/$BUILD_DIR/eshkol-run" ;;
esac
TEST_SOURCE="tests/ad/taylor_tower_mono_test.esk"
TRACE_DIR="${TRACE_DIR:-scripts/icc_traces}"
TRACE_FILE="$TRACE_DIR/mono_equiv.jsonl"
EXPECTED_SUMMARY="mono==runtime bit-exact tests: 441 passed, 0 failed"

if [ ! -x "$ESHKOL_RUN" ]; then
  echo "run_mono_equiv_ad_taylor_gate.sh: missing executable: $ESHKOL_RUN" >&2
  exit 2
fi

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-mono-equiv.XXXXXX")"
cleanup() {
  rm -rf -- "$tmp_dir"
}
trap cleanup EXIT

run_and_check() {
  local mode="$1"
  shift
  local output
  output="$("$@" 2>&1)"
  printf '%s\n' "$output"
  if ! printf '%s\n' "$output" | grep -Fq "$EXPECTED_SUMMARY"; then
    echo "mono-equivalence $mode gate did not produce the exact success summary" >&2
    return 1
  fi
}

echo "== AD Taylor monomorphization equivalence gate =="
run_and_check jit "$ESHKOL_RUN" -r "$TEST_SOURCE"

aot_bin="$tmp_dir/taylor_tower_mono_test"
"$ESHKOL_RUN" "$TEST_SOURCE" -o "$aot_bin"
run_and_check aot "$aot_bin"

mkdir -p "$TRACE_DIR"
cat > "$TRACE_FILE" <<'EOF'
{"kind":"mono_equiv","name":"ad_taylor_p2_monomorphization","value":"PASS","snippet":"JIT+AOT literal-K Taylor towers are bit-exact with runtime towers: 441/441 checks","confidence":1.0}
{"kind":"mono_equiv","name":"ad_taylor_p3_jet8_subsumed","value":"PASS","snippet":"JIT+AOT tagged Taylor towers preserve exact behavior through order 8: 441/441 checks","confidence":1.0}
EOF

echo "PASS: JIT+AOT monomorphized/runtime Taylor equivalence (441 checks each)"
echo "$TRACE_FILE written"
