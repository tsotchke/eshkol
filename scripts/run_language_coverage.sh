#!/usr/bin/env bash
# Run the deterministic language-surface tracker and publish fresh ICC evidence.
# The committed policy is a one-way ratchet: callers may raise the threshold,
# but language_coverage.py will never accept a value below the policy floor.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
TRACE_DIR=${ICC_TRACE_DIR:-"$REPO_ROOT/scripts/icc_traces"}
TRACE_FILE=${LANGUAGE_COVERAGE_TRACE:-"$TRACE_DIR/language_surface_coverage.jsonl"}
RUNTIME_TRACE_DIRS=${LANGUAGE_COVERAGE_RUNTIME_TRACE_DIRS:-}
EXTRA_RUNTIME_TRACE_DIRS=${LANGUAGE_COVERAGE_EXTRA_RUNTIME_TRACE_DIRS:-}
BUILD_DIR=${BUILD_DIR:-build}
QUANTUM_BUILD_DIR=${QUANTUM_BUILD_DIR:-build-quantum}
GENERATED_TRACE_ROOT=

resolve_build_dir() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *)  printf '%s/%s\n' "$REPO_ROOT" "$1" ;;
    esac
}

BUILD_DIR_PATH=$(resolve_build_dir "$BUILD_DIR")
QUANTUM_BUILD_DIR_PATH=$(resolve_build_dir "$QUANTUM_BUILD_DIR")
ESHKOL_RUN="$BUILD_DIR_PATH/eshkol-run"
ESHKOL_VM="$BUILD_DIR_PATH/eshkol-vm-standalone-test"
QUANTUM_RUN="$QUANTUM_BUILD_DIR_PATH/eshkol-run"

# The complete-suite harness delegates to older scripts whose BUILD_DIR
# contract is repository-relative (they invoke ./$BUILD_DIR/eshkol-run).
# Accept either spelling at this boundary, but normalize an absolute path
# beneath the repository before crossing into that legacy interface.
case "$BUILD_DIR_PATH" in
    "$REPO_ROOT"/*)
        BUILD_DIR_FOR_TESTS=${BUILD_DIR_PATH#"$REPO_ROOT"/}
        ;;
    *)
        echo "BUILD_DIR must resolve inside the repository for the fresh full-suite run:" >&2
        echo "  $BUILD_DIR_PATH" >&2
        exit 2
        ;;
esac

if [ -z "$RUNTIME_TRACE_DIRS" ]; then
    if [ ! -x "$ESHKOL_RUN" ] || [ ! -x "$ESHKOL_VM" ]; then
        echo "Missing $BUILD_DIR_PATH/eshkol-run or eshkol-vm-standalone-test." >&2
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
        local rc=$?
        if [ "${KEEP_LANGUAGE_COVERAGE_TRACES:-0}" = 1 ] || [ "$rc" -ne 0 ]; then
            echo "Kept fresh language-coverage traces: $GENERATED_TRACE_ROOT"
        else
            rm -rf "$GENERATED_TRACE_ROOT"
        fi
        return "$rc"
    }
    trap cleanup EXIT
    CORE_TRACE="$GENERATED_TRACE_ROOT/core"
    QUANTUM_TRACE="$GENERATED_TRACE_ROOT/quantum"
    mkdir -p "$CORE_TRACE" "$QUANTUM_TRACE"

    echo "== Fresh core/native/VM execution evidence =="
    ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR="$CORE_TRACE" \
        BUILD_DIR="$BUILD_DIR_FOR_TESTS" "$REPO_ROOT/scripts/run_all_tests.sh"

    echo "== Fresh quantum/PQC execution evidence =="
    for test in "$REPO_ROOT"/tests/quantum/*.esk; do
        echo "== ${test#"$REPO_ROOT"/} =="
        ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR="$QUANTUM_TRACE" \
            "$QUANTUM_RUN" -r "$test" "-L$QUANTUM_BUILD_DIR_PATH"
    done

    RUNTIME_TRACE_DIRS="$CORE_TRACE:$QUANTUM_TRACE"
fi

if [ -n "$EXTRA_RUNTIME_TRACE_DIRS" ]; then
    if [ -n "$RUNTIME_TRACE_DIRS" ]; then
        RUNTIME_TRACE_DIRS="$RUNTIME_TRACE_DIRS:$EXTRA_RUNTIME_TRACE_DIRS"
    else
        RUNTIME_TRACE_DIRS="$EXTRA_RUNTIME_TRACE_DIRS"
    fi
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
    --eshkol-run "$ESHKOL_RUN" \
    --eshkol-vm "$ESHKOL_VM"
python3 scripts/language_coverage.py \
    "${RUNTIME_ARGS[@]}" \
    --trace "$TRACE_FILE" \
    "$@"
