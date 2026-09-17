#!/usr/bin/env bash
# Run the v1.3.5 criteria producers omitted from the general smoke battery.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"
TRACE_DIR="${TRACE_DIR:-$REPO_ROOT/scripts/icc_traces}"
# Evidence paths are absolute before first use (scripts/lib/evidence_paths.sh).
. "$REPO_ROOT/scripts/lib/evidence_paths.sh"
eshkol_evidence_abs_var TRACE_DIR "$REPO_ROOT" || exit $?
BUILD_DIR="${BUILD_DIR:-build}"
ICC_BIN="${ICC_BIN:-icc}"
ICC_REPO="${ICC_REPO:-eshkol_lang}"
case "$BUILD_DIR" in
    /*) BUILD_DIR_PATH="$BUILD_DIR" ;;
    *) BUILD_DIR_PATH="$REPO_ROOT/$BUILD_DIR" ;;
esac
mkdir -p "$TRACE_DIR"
export TRACE_DIR ESH0103_TRACE_DIR="$TRACE_DIR" ESHKOL_TRACE_DIR="$TRACE_DIR"

. scripts/lib/harness_outcome.sh

run_test_action() {
    local name="$1" summary="$2"; shift 2
    local rc=0
    "$@" || rc=$?
    if [ "$rc" -eq 0 ]; then
        eshkol_outcome_emit_test_result "$TRACE_DIR/release_test_actions.jsonl" "$name" PASS "$summary"
    else
        eshkol_outcome_emit_test_result "$TRACE_DIR/release_test_actions.jsonl" "$name" FAIL "$summary (exit $rc)"
        return "$rc"
    fi
}

# Both NodeId criteria are emitted together by this runtime measurement.
python3 scripts/run_node_identity_gate.py

# This event uses the performance_budget kind declared by the oracle.
bash tests/perf/nested_expr_compile_time_test.sh "$BUILD_DIR/eshkol-run"

# Test-evidence criteria whose actions are gates rather than a runtime_event.
run_test_action dense_tensor_ad_gate "dense/scalarized gradients agree and dense tape ratchet holds" \
    ./scripts/run_dense_tensor_ad_gate.sh
run_test_action self_verdict_gate_self_test "self-verdict scanner detects a planted contradiction" \
    python3 scripts/check_self_verdicts.py --self-test
run_test_action test_coverage_inventory "documented test inventory matches the complete-suite runner" \
    python3 scripts/check_test_coverage.py
run_test_action release_evidence_recipe_self_test "producer ordering, CTest cardinality, archive isolation, and fingerprint fault injection" \
    python3 tests/toolchain/test_v1_3_release_evidence_recipe.py

# The release build has ESHKOL_BUILD_TESTS=ON and Python bindings enabled.
# Run only the five named release CTests and require all five in JUnit output.
ctest_junit="$TRACE_DIR/v1_3_required_ctest.junit.xml"
ctest_trace="$TRACE_DIR/v1_3_required_ctest.jsonl"
rm -f "$ctest_junit" "$ctest_trace"
ctest_rc=0
ctest --test-dir "$BUILD_DIR" --output-on-failure --output-junit "$ctest_junit" \
    -R '^(qubit_linearity_engine_parity_gate|closure_upvalue_capacity_overflow_gate|abi_layout_pin_test|v1_3_quoted_datum_kinds_runtime_smoke|python_bindings_capsule_lifetime)$' \
    || ctest_rc=$?
python3 scripts/record_release_ctest_evidence.py \
    --junit "$ctest_junit" --trace "$ctest_trace" --ctest-exit-code "$ctest_rc"

# The sanitizer script builds and uses its own ASan+UBSan tree. Keep every
# generated artifact under the readiness scratch directory, leaving `build`
# and its compiler fingerprint cohort untouched.
sanitizer_root="${ESHKOL_RELEASE_SANITIZER_ROOT:-$REPO_ROOT/.scratch/v1-3-readiness/sanitizer}"
sanitizer_report="${ESHKOL_RELEASE_SANITIZER_REPORT:-$REPO_ROOT/.scratch/v1-3-readiness/SANITIZER_FUZZ_REPORT.md}"
mkdir -p "$(dirname "$sanitizer_root")" "$(dirname "$sanitizer_report")"
run_test_action sanitizer_instrumented_build "ASan+UBSan compiler built in a separate tree using the provisioned LLVM toolchain" \
    env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:allocator_may_return_null=1 \
    LSAN_OPTIONS="suppressions=$REPO_ROOT/.icc/lsan-suppressions.txt:print_suppressions=0" \
    ESHKOL_BUILD_JOBS=4 BUILD_DIR=build-asan-ubsan \
    CMAKE_BUILD_TYPE=RelWithDebInfo scripts/build-sanitizer.sh asan+ubsan
run_test_action sanitizer_fuzz_quick "bounded ASan+UBSan corpus run with leak detection enabled" \
    env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:allocator_may_return_null=1 \
    LSAN_OPTIONS="suppressions=$REPO_ROOT/.icc/lsan-suppressions.txt:print_suppressions=0" \
    scripts/run_sanitizer_fuzz.sh --skip-build --build-dir build-asan-ubsan \
        --work-dir "$sanitizer_root" --report "$sanitizer_report" \
        --trace-file "$TRACE_DIR/sanitizer_fuzz.jsonl" --gate-limit 150

# Rosette Wire is Common Lisp/SBCL; use its pinned clone and the already-built
# release compiler. The oracle script records FAIL itself if SBCL is absent.
ESHKOL_RUN_BIN_OVERRIDE="$BUILD_DIR_PATH/eshkol-run" \
ESHKOL_ROSETTE_BUILD_DIR="$BUILD_DIR_PATH" \
TRACE_DIR="$TRACE_DIR" \
    scripts/run_rosette_oracle.sh

# Every tutorial example, run on the release compiler itself (JIT and AOT).
python3 scripts/doc_audit/check_doc_examples.py --scope tutorials \
    --eshkol-run "$BUILD_DIR_PATH/eshkol-run" \
    --work-dir "$REPO_ROOT/.scratch/v1-3-readiness/doc-example-gate" \
    --trace-dir "$TRACE_DIR"

# Build-free checks that own the rest of the required eshkol_smoke receipts.
python3 scripts/check_ledger_integrity.py --trace-dir "$TRACE_DIR"
python3 scripts/check_oracle_schema.py --trace-dir "$TRACE_DIR"
python3 scripts/audit_oracle_false_green.py --trace-dir "$TRACE_DIR"
python3 scripts/check_ps1_encoding.py --trace-dir "$TRACE_DIR"
python3 scripts/check_public_api_docs.py --trace-dir "$TRACE_DIR"
python3 scripts/gen_api_docs.py --check --trace-dir "$TRACE_DIR"
python3 scripts/check_disclosure.py --base origin/master --head HEAD --trace-dir "$TRACE_DIR"
python3 scripts/check_required_context_consistency.py --offline --trace-dir "$TRACE_DIR"
python3 scripts/check_doc_claims_residual.py --icc-bin "$ICC_BIN" --repo "$ICC_REPO" \
    --trace-dir "$TRACE_DIR" --emit-trace-dir "$TRACE_DIR"

# This must grade the fresh cohort above, including the smoke and external
# oracle receipts. Its own prior result is cleared so a failed run cannot leave
# an older PASS standing.
rm -f "$TRACE_DIR/evidence_staleness_gate.jsonl"
python3 scripts/check_evidence_staleness.py --trace-dir "$TRACE_DIR" \
    --require-trace-dir --emit-trace-dir "$TRACE_DIR"
