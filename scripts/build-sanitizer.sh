#!/usr/bin/env bash
# build-sanitizer.sh — configure + build Eshkol under a sanitizer.
#
# Usage:
#   scripts/build-sanitizer.sh asan          # address sanitizer
#   scripts/build-sanitizer.sh asan+ubsan    # address + undefined-behavior
#   scripts/build-sanitizer.sh tsan          # thread sanitizer
#   scripts/build-sanitizer.sh ubsan         # UBSan only
#   scripts/build-sanitizer.sh msan          # memory sanitizer (Linux only, needs
#                                              instrumented libc++)
#
# Builds into build-<flavor>/ so the normal build/ stays intact.  Set
# BUILD_DIR to select another repository-relative or absolute output path and
# ESHKOL_BUILD_JOBS to select a positive parallel-build count.
# After a successful build the script prints the env vars you should set
# before running the binary (ASAN_OPTIONS etc.).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

FLAVOR="${1:-}"
if [[ -z "${FLAVOR}" ]]; then
    echo "usage: $0 {asan | asan+ubsan | tsan | ubsan | msan}" >&2
    exit 2
fi

ASAN=OFF; UBSAN=OFF; TSAN=OFF; MSAN=OFF

case "${FLAVOR}" in
    asan)        ASAN=ON ;;
    ubsan)       UBSAN=ON ;;
    asan+ubsan)  ASAN=ON; UBSAN=ON ;;
    tsan)        TSAN=ON ;;
    tsan+ubsan)  TSAN=ON; UBSAN=ON ;;
    msan)        MSAN=ON ;;
    *) echo "unknown flavor: ${FLAVOR}" >&2; exit 2 ;;
esac

detect_jobs() {
    local detected=""
    if command -v getconf >/dev/null 2>&1; then
        detected="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
    fi
    if ! [[ "$detected" =~ ^[1-9][0-9]*$ ]] && command -v sysctl >/dev/null 2>&1; then
        detected="$(sysctl -n hw.ncpu 2>/dev/null || true)"
    fi
    if ! [[ "$detected" =~ ^[1-9][0-9]*$ ]] && command -v nproc >/dev/null 2>&1; then
        detected="$(nproc 2>/dev/null || true)"
    fi
    if ! [[ "$detected" =~ ^[1-9][0-9]*$ ]]; then
        detected=1
    fi
    printf '%s\n' "$detected"
}

JOBS="${ESHKOL_BUILD_JOBS:-$(detect_jobs)}"
if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ESHKOL_BUILD_JOBS must be a positive integer: $JOBS" >&2
    exit 2
fi

DEFAULT_BUILD_DIR="build-${FLAVOR//+/-}"
BUILD_DIR="${BUILD_DIR:-$DEFAULT_BUILD_DIR}"
case "$BUILD_DIR" in
    /*) ;;
    *) BUILD_DIR="$REPO_ROOT/$BUILD_DIR" ;;
esac
mkdir -p "${BUILD_DIR}"
BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"
if [ "$BUILD_DIR" = "$REPO_ROOT" ]; then
    echo "refusing in-source sanitizer build: $BUILD_DIR" >&2
    exit 2
fi

# Sanitizer builds want Debug/RelWithDebInfo so stack traces have
# symbols; -O0 makes the runs unbearable on real test suites, -O1
# keeps fast symbolication. Override if you set CMAKE_BUILD_TYPE
# yourself.
: "${CMAKE_BUILD_TYPE:=RelWithDebInfo}"

cmake -S "$REPO_ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
    -DESHKOL_ENABLE_ASAN="${ASAN}" \
    -DESHKOL_ENABLE_UBSAN="${UBSAN}" \
    -DESHKOL_ENABLE_TSAN="${TSAN}" \
    -DESHKOL_ENABLE_MSAN="${MSAN}"

cmake --build "$BUILD_DIR" --target eshkol-run stdlib --parallel "$JOBS"

if [[ "$(uname -s)" == "Darwin" ]]; then
    ASAN_EXAMPLE_OPTIONS="detect_leaks=0:abort_on_error=0"
else
    ASAN_EXAMPLE_OPTIONS="detect_leaks=1:abort_on_error=0"
fi

cat <<EOF

=========================================
  Sanitizer build complete: ${BUILD_DIR}
=========================================

Run tests with:
  (cd ${BUILD_DIR} && ASAN_OPTIONS='${ASAN_EXAMPLE_OPTIONS}' \\
                     UBSAN_OPTIONS='print_stacktrace=1' \\
                     ./eshkol-run ../tests/v1_2_edge_cases/hardening_path_test.esk)

Or run the full regression suite:
  BUILD_DIR=${BUILD_DIR} bash ${SCRIPT_DIR}/run_all_tests.sh

EOF
