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
# Builds into build-<flavor>/ so the normal build/ stays intact.
# After a successful build the script prints the env vars you should set
# before running the binary (ASAN_OPTIONS etc.).

set -euo pipefail

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

BUILD_DIR="build-${FLAVOR//+/-}"
mkdir -p "${BUILD_DIR}"

# Sanitizer builds want Debug/RelWithDebInfo so stack traces have
# symbols; -O0 makes the runs unbearable on real test suites, -O1
# keeps fast symbolication. Override if you set CMAKE_BUILD_TYPE
# yourself.
: "${CMAKE_BUILD_TYPE:=RelWithDebInfo}"

cd "${BUILD_DIR}"
cmake .. \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
    -DESHKOL_ENABLE_ASAN="${ASAN}" \
    -DESHKOL_ENABLE_UBSAN="${UBSAN}" \
    -DESHKOL_ENABLE_TSAN="${TSAN}" \
    -DESHKOL_ENABLE_MSAN="${MSAN}"

make -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)" eshkol-run stdlib

cat <<EOF

=========================================
  Sanitizer build complete: ${BUILD_DIR}
=========================================

Run tests with:
  (cd ${BUILD_DIR} && ASAN_OPTIONS='detect_leaks=1:abort_on_error=0' \\
                     UBSAN_OPTIONS='print_stacktrace=1' \\
                     ./eshkol-run ../tests/v1_2_edge_cases/hardening_path_test.esk)

Or run the full regression suite:
  BUILD_DIR=${BUILD_DIR} bash scripts/run_all_tests.sh

EOF
