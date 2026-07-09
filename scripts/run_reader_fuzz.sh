#!/usr/bin/env bash
# run_reader_fuzz.sh — build (if needed) and run the seeded adversarial
# reader-hardening harness (tests/fuzz/reader_fuzz_driver.cpp) against
# the hosted S-expression reader (lib/core/runtime_reader_hosted.cpp,
# entry point eshkol_read_sexpr — the (read) builtin's implementation).
#
# Usage:
#   scripts/run_reader_fuzz.sh                 (smoke: default, ~10s)
#   scripts/run_reader_fuzz.sh --smoke         (same, explicit — wired
#                                                into run_icc_smoke.sh as
#                                                the reader_fuzz_smoke probe)
#   scripts/run_reader_fuzz.sh --full          (~10-30s: full seeded sweep,
#                                                10^5-10^6-element lists,
#                                                manual/nightly hardening pass)
#   scripts/run_reader_fuzz.sh --regression    (fixed depth-guard /
#                                                bounded-stack assertions only)
#
# DISK BUDGET (hard repo rule — prior fuzz/sanitizer harnesses have
# filled shared disks; see MEMORY.md fuzz/harness disk-budget note):
#   - Build directory: build-reader-fuzz/ at the repo root, git-ignored.
#     Only pulls in eshkol-runtime (arena/reader/symbol-intern/etc — no
#     LLVM-backed compiler frontend/backend), so a from-scratch configure
#     + build is a handful of seconds, not the full compiler's build
#     time. Left in place between runs for fast incremental rebuilds; it
#     is ordinary object-file build output, not a growing corpus.
#   - Artifact directory: a fresh mktemp -d per invocation. The driver
#     itself hard-caps total artifact bytes at 64 MB (well under the
#     repo's 200 MB corpus ceiling) and only ever writes the specific
#     inputs that triggered a crash/hang — no unbounded corpus growth,
#     no passing-case retention.
#   - trap on EXIT: a clean run (no findings) deletes the whole artifact
#     directory. A run with findings keeps the (small, capped) artifact
#     directory so the triggering input can be inspected/replayed, and
#     prints its path.
set -u
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

MODE="--smoke"
for arg in "$@"; do
    case "$arg" in
        --full) MODE="--full" ;;
        --smoke) MODE="--smoke" ;;
        --regression) MODE="--regression" ;;
        *) echo "run_reader_fuzz.sh: unknown argument '$arg'" >&2; exit 2 ;;
    esac
done

BUILD_DIR="$REPO_ROOT/build-reader-fuzz"
ARTIFACT_DIR=$(mktemp -d "${TMPDIR:-/tmp}/eshkol-reader-fuzz-artifacts.XXXXXX")

cleanup() {
    local status=$?
    if [ "$status" -eq 0 ]; then
        rm -rf "$ARTIFACT_DIR"
    else
        echo "run_reader_fuzz.sh: findings kept at $ARTIFACT_DIR" \
             "(disk-budget capped at 64 MB by the driver — see tests/fuzz/reader_fuzz_driver.cpp)" >&2
    fi
    exit "$status"
}
trap cleanup EXIT

mkdir -p "$BUILD_DIR"
if ! cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
        -DESHKOL_ENABLE_FUZZ=ON -DESHKOL_BUILD_TESTS=OFF \
        >"$BUILD_DIR.configure.log" 2>&1; then
    echo "run_reader_fuzz.sh: cmake configure failed — see $BUILD_DIR.configure.log" >&2
    exit 2
fi

NPROC=$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
if ! cmake --build "$BUILD_DIR" --target reader_fuzz -j "$NPROC" \
        >"$BUILD_DIR.build.log" 2>&1; then
    echo "run_reader_fuzz.sh: build failed — see $BUILD_DIR.build.log" >&2
    exit 2
fi

BIN="$BUILD_DIR/tests/fuzz/reader_fuzz"
if [ ! -x "$BIN" ]; then
    echo "run_reader_fuzz.sh: $BIN missing after a reported-successful build" \
         "(unexpected toolchain skip? see $BUILD_DIR.build.log)" >&2
    exit 2
fi

"$BIN" "$MODE" --artifact-dir "$ARTIFACT_DIR"
