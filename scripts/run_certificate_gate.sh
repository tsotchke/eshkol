#!/usr/bin/env bash
#
# core.certificate acceptance gate (ADR-0047): certificates for certified
# steps, checked by tools/certcheck/eshkol_certcheck.py without trusting the
# compiler, a tamper suite, and a Lean 4 export of the decidable facts.
#
# Runs, from an existing CMake build, the CTest entries that own the
# behaviour:
#   certify_form_jit / _aot / _direct_vm         the (certify ...) rewrite on every engine
#   certificate_jit / certificate_aot            tests/stdlib/certificate_test.esk
#   certificate_roundtrip_jit / _aot             emit -> check -> tamper -> Lean export
#   certificate_check_fixtures                   checker on the committed documents
#   certificate_tamper_fixtures                  every altered fixture claim rejected
#   certificate_lean_export_current              committed Lean files match the fixtures
#   transcript_jit / transcript_aot / transcript_replay_check   (core.transcript, extended)
#   polynomial_exact_jit / _aot, ring_jit / _aot, multimodular_jit / _aot,
#   exact_psd_jit / _aot, certified_solve_jit / _aot   (libraries that gained a witness)
# and, when a `lean` executable is available (LEAN, or lean on PATH), checks
# every file in tests/certificates/lean/ with it. REQUIRE_LEAN=1 makes a
# missing Lean a failure.
#
# Writes one event to $TRACE_DIR/v17_certificates.jsonl:
#   {"kind": "v1_7_synthesis", "name": "v17_certificates", "value": PASS|FAIL}
# PASS only when every expected CTest entry ran and passed. Absence of a test
# is a FAIL, never a skip.
#
#   BUILD_DIR (default build), TRACE_DIR (default scripts/icc_traces), JOBS (default 2)

set -u
set -o pipefail
export LC_ALL=C LC_CTYPE=C LANG=C
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
    /*) BUILD_DIR_PATH="$BUILD_DIR" ;;
    *)  BUILD_DIR_PATH="$REPO_ROOT/$BUILD_DIR" ;;
esac
if [ ! -x "$BUILD_DIR_PATH/eshkol-run" ]; then
    echo "run_certificate_gate.sh: $BUILD_DIR_PATH/eshkol-run not found; build first" >&2
    exit 2
fi
BUILD_DIR_PATH="$(cd "$BUILD_DIR_PATH" && pwd -P)" || exit 2

TRACE_DIR="${TRACE_DIR:-$REPO_ROOT/scripts/icc_traces}"
. "$REPO_ROOT/scripts/lib/evidence_paths.sh"
eshkol_evidence_abs_var TRACE_DIR "$REPO_ROOT" || exit $?
mkdir -p "$TRACE_DIR"
TRACE_FILE="$TRACE_DIR/v17_certificates.jsonl"

WORK="$BUILD_DIR_PATH/test-work/certificate-gate"
mkdir -p "$WORK"
export ESHKOL_PATH="$REPO_ROOT/lib${ESHKOL_PATH:+:$ESHKOL_PATH}"

TESTS='^(certify_form_jit|certify_form_aot|certify_form_direct_vm|certificate_jit|certificate_aot|certificate_roundtrip_jit|certificate_roundtrip_aot|certificate_check_fixtures|certificate_tamper_fixtures|certificate_lean_export_current|transcript_jit|transcript_aot|transcript_replay_check|polynomial_exact_jit|polynomial_exact_aot|ring_jit|ring_aot|multimodular_jit|multimodular_aot|exact_psd_jit|exact_psd_aot|certified_solve_jit|certified_solve_aot)$'
EXPECTED=23

emit_event() { # PASS|FAIL snippet
    python3 - "$TRACE_FILE" "$1" "$2" <<'PY'
import json, os, sys
path, value, snippet = sys.argv[1], sys.argv[2], sys.argv[3]
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    f.write(json.dumps({"kind": "v1_7_synthesis", "name": "v17_certificates", "value": value,
                        "snippet": snippet[:2000], "confidence": 0.95}, ensure_ascii=False) + "\n")
os.replace(tmp, path)
PY
}

status=0
ctest_log="$WORK/ctest.log"
(cd "$BUILD_DIR_PATH" && ctest -R "$TESTS" --output-on-failure -j "${JOBS:-2}") > "$ctest_log" 2>&1
ctest_status=$?
# Count the expected entries by name: CTest also runs any fixture-setup
# test they require, which must pass too (ctest exit status) but is not one
# of the EXPECTED entries.
passed=0
for name in $(echo "$TESTS" | tr -d '^$()' | tr '|' ' '); do
    if grep -qE "^ *[0-9]+/[0-9]+ Test +#[0-9]+: ${name} \.+ +Passed" "$ctest_log"; then
        passed=$((passed + 1))
    fi
done
ran=$(grep -cE '^ *[0-9]+/[0-9]+ Test +#[0-9]+: ' "$ctest_log" || true)
summary="ctest ${passed}/${EXPECTED} expected entries passed (ran ${ran} incl. fixture setup, exit ${ctest_status})"
if [ "$ctest_status" -ne 0 ] || [ "$passed" -ne "$EXPECTED" ]; then
    status=1
fi

LEAN_BIN="${LEAN:-$(command -v lean || true)}"
if [ -n "$LEAN_BIN" ] && [ -x "$LEAN_BIN" ]; then
    lean_ok=0; lean_n=0
    for f in "$REPO_ROOT"/tests/certificates/lean/*.lean; do
        lean_n=$((lean_n + 1))
        "$LEAN_BIN" "$f" > "$WORK/lean.log" 2>&1 || { lean_ok=1; cat "$WORK/lean.log"; }
    done
    lean_version="$("$LEAN_BIN" --version 2>/dev/null | head -1)"
    if [ "$lean_ok" -eq 0 ] && [ "$lean_n" -gt 0 ]; then
        summary="$summary; lean: $lean_n files checked ($lean_version)"
    else
        summary="$summary; lean: FAILED ($lean_version)"; status=1
    fi
elif [ "${REQUIRE_LEAN:-0}" = 1 ]; then
    summary="$summary; lean: not available"; status=1
else
    summary="$summary; lean: not available on this host"
fi

if [ "$status" -eq 0 ]; then
    emit_event PASS "$summary"
    echo "core.certificate gate: PASS ($summary)"
else
    emit_event FAIL "$summary"
    echo "core.certificate gate: FAIL ($summary)"
    tail -n 60 "$ctest_log"
fi
exit "$status"
