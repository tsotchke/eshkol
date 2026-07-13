#!/usr/bin/env bash
# Run the deterministic language-surface tracker and publish fresh ICC evidence.
# The committed policy is a one-way ratchet: callers may raise the threshold,
# but language_coverage.py will never accept a value below the policy floor.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
TRACE_DIR=${ICC_TRACE_DIR:-"$REPO_ROOT/scripts/icc_traces"}
TRACE_FILE=${LANGUAGE_COVERAGE_TRACE:-"$TRACE_DIR/language_surface_coverage.jsonl"}
RUNTIME_TRACE_DIRS=${LANGUAGE_COVERAGE_RUNTIME_TRACE_DIRS:-}
BUILD_DIR=${BUILD_DIR:-build}
QUANTUM_BUILD_DIR=${QUANTUM_BUILD_DIR:-build-quantum}
GENERATED_TRACE_ROOT=

if [ -z "$RUNTIME_TRACE_DIRS" ]; then
    ESHKOL_RUN="$REPO_ROOT/$BUILD_DIR/eshkol-run"
    ESHKOL_VM="$REPO_ROOT/$BUILD_DIR/eshkol-vm-standalone-test"
    QUANTUM_RUN="$REPO_ROOT/$QUANTUM_BUILD_DIR/eshkol-run"
    if [ ! -x "$ESHKOL_RUN" ] || [ ! -x "$ESHKOL_VM" ]; then
        echo "Missing $BUILD_DIR/eshkol-run or eshkol-vm-standalone-test." >&2
        echo "Configure with -DESHKOL_BUILD_TESTS=ON and build both targets first." >&2
        exit 2
    fi
    if [ ! -x "$QUANTUM_RUN" ]; then
        echo "Missing quantum-enabled runner: $QUANTUM_RUN" >&2
        echo "100% language coverage includes agent.quantum/agent.pqc; configure" >&2
        echo "  cmake -S . -B $QUANTUM_BUILD_DIR -DESHKOL_QUANTUM_ENABLED=ON -DESHKOL_BUILD_TESTS=ON" >&2
        exit 2
    fi

    GENERATED_TRACE_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/eshkol-language-coverage.XXXXXX")
    cleanup() {
        if [ "${KEEP_LANGUAGE_COVERAGE_TRACES:-0}" = 1 ]; then
            echo "Kept fresh language-coverage traces: $GENERATED_TRACE_ROOT"
        else
            rm -rf "$GENERATED_TRACE_ROOT"
        fi
    }
    trap cleanup EXIT
    CORE_TRACE="$GENERATED_TRACE_ROOT/core"
    QUANTUM_TRACE="$GENERATED_TRACE_ROOT/quantum"
    mkdir -p "$CORE_TRACE" "$QUANTUM_TRACE"

    echo "== Fresh core/native/VM execution evidence =="
    ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR="$CORE_TRACE" \
        BUILD_DIR="$BUILD_DIR" "$REPO_ROOT/scripts/run_all_tests.sh"

    echo "== Fresh quantum/PQC execution evidence =="
    for test in "$REPO_ROOT"/tests/quantum/*.esk; do
        echo "== ${test#"$REPO_ROOT"/} =="
        ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR="$QUANTUM_TRACE" \
            "$QUANTUM_RUN" -r "$test" "-L$REPO_ROOT/$QUANTUM_BUILD_DIR"
    done

    RUNTIME_TRACE_DIRS="$CORE_TRACE:$QUANTUM_TRACE"
fi

OLD_IFS=$IFS
IFS=:
read -r -a RUNTIME_DIR_ARRAY <<< "$RUNTIME_TRACE_DIRS"
IFS=$OLD_IFS

RUNTIME_ARGS=()
for runtime_dir in "${RUNTIME_DIR_ARRAY[@]}"; do
    if [ -z "$runtime_dir" ] || [ ! -d "$runtime_dir" ]; then
        echo "Runtime language-coverage trace directory missing: $runtime_dir" >&2
        exit 2
    fi
    RUNTIME_ARGS+=(--runtime-trace-dir "$runtime_dir")
done

cd "$REPO_ROOT"
python3 scripts/gen_language_surface.py --check
python3 scripts/test_runtime_language_coverage.py \
    --eshkol-run "$BUILD_DIR/eshkol-run" \
    --eshkol-vm "$BUILD_DIR/eshkol-vm-standalone-test"
python3 scripts/language_coverage.py \
    "${RUNTIME_ARGS[@]}" \
    --trace "$TRACE_FILE" \
    "$@"
