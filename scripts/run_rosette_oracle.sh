#!/usr/bin/env bash
# run_rosette_oracle.sh — Rosette Wire external-oracle gate for Eshkol.
#
# Rosette Wire (https://github.com/RichardHoekstra/rosette-wire, Apache-2.0,
# Common Lisp/SBCL) ships a public `front-door/eshkol` adapter that lowers an
# admitted integer-program dialect through its direct evaluator, its kernel
# VM, a cache-disabled Eshkol JIT run, and a separately compiled Eshkol AOT
# artifact, then requires all four to agree. This script drives that adapter
# against a freshly built `eshkol-run` from THIS repository and records the
# result as an ICC runtime_event, so Eshkol counts as a first-class,
# independently maintained Rosette Wire backend (docs/interop.md in the
# Rosette Wire tree) rather than an untracked side effect of someone running
# its examples by hand.
#
# This is pillar P7 (external oracle): the ground truth here is not another
# Eshkol harness, it is an independent project's own four-way agreement gate.
#
# Requirements: SBCL on PATH.
#   macOS:            brew install sbcl
#   CI / Debian/Ubuntu: apt-get install -y sbcl
#
# Known macOS gap (observed, not fixed here): Rosette Wire's isolated-worker
# wraps every probed/executed command in `prlimit` (util-linux; Linux-only),
# so `eshkol-available-p`/campaign runs on a macOS host without a `prlimit`
# on PATH fail closed with a LAUNCH-ERROR at the :availability stage (visible
# as `... = :UNAVAILABLE` from examples/external-backends.lisp) even when
# eshkol-run itself is fine. This is upstream behavior in the read-only
# clone, out of scope to patch. CI (ubuntu-22.04, this repo's own workflow)
# has `prlimit` from util-linux by default, so this gap is CI-irrelevant.
#
# Hard rule: nothing here ever touches /tmp or /private/tmp. All scratch,
# clone, build and evidence output live under <repo>/.scratch/. Every path
# THIS script passes to Rosette tooling (the clone location, the evidence
# output directory) is under .scratch/. tools/test-system and
# examples/external-backends.lisp additionally hardcode their own ASDF
# fasl-cache output-translations to /tmp/rosette-contributor-cache and
# /tmp/rosette-example-cache respectively inside Rosette Wire's own source —
# that is upstream code in the read-only clone, not a path this script
# supplies, and it is out of scope to patch a vendored, unmodified clone.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=lib/harness_outcome.sh
source "${SCRIPT_DIR}/lib/harness_outcome.sh"

SCRATCH_DIR="${REPO_ROOT}/.scratch"
ROSETTE_DIR="${SCRATCH_DIR}/rosette-wire"
# Canonical upstream — this is what CI (and any machine without a private
# local mirror) clones from.
ROSETTE_UPSTREAM_URL="https://github.com/RichardHoekstra/rosette-wire.git"
# ROSETTE_LOCAL_SOURCE_OVERRIDE lets a developer point this at an existing
# local checkout instead of fetching over the network (e.g. a read-only
# contributor clone already on disk). Optional; unset in CI.
ROSETTE_LOCAL_SOURCE="${ROSETTE_LOCAL_SOURCE_OVERRIDE:-}"
#
# Pinned commit: this repo's rosette-wire dependency is frozen at the exact
# SHA that was HEAD of the upstream repository when this gate was authored.
# Bump it deliberately (re-pin + re-run this script + re-commit) rather than
# floating.
ROSETTE_PIN_SHA="bb34bbe7536f2ea9829dc302c6a578e9fdd877a0"

BUILD_DIR="${ESHKOL_ROSETTE_BUILD_DIR:-${REPO_ROOT}/build}"
NATIVE_DIR="${SCRATCH_DIR}/rosette-native"
TRACE_FILE="${REPO_ROOT}/scripts/icc_traces/rosette.jsonl"

mkdir -p "${SCRATCH_DIR}" "${NATIVE_DIR}" "$(dirname "${TRACE_FILE}")"

fail_event() { # name snippet
    local name="$1" snippet="$2"
    eshkol_outcome_emit_event "${TRACE_FILE}" rosette "${name}" FAIL "${snippet}"
    eshkol_outcome_emit_test_result "${TRACE_FILE}" "${name}" FAIL "${snippet}"
}

pass_event() { # name snippet
    local name="$1" snippet="$2"
    eshkol_outcome_emit_event "${TRACE_FILE}" rosette "${name}" PASS "${snippet}"
    eshkol_outcome_emit_test_result "${TRACE_FILE}" "${name}" PASS "${snippet}"
}

# ── 1. SBCL availability ────────────────────────────────────────────────
if ! command -v sbcl >/dev/null 2>&1; then
    echo "sbcl not found on PATH." >&2
    echo "  macOS: brew install sbcl" >&2
    echo "  CI / Debian/Ubuntu: apt-get install -y sbcl" >&2
    fail_event rosette_front_door_eshkol "sbcl not installed"
    fail_event rosette_external_backends_eshkol "sbcl not installed"
    exit 1
fi

# ── 2. Pinned clone of rosette-wire ──────────────────────────────────────
if [ ! -d "${ROSETTE_DIR}/.git" ]; then
    if [ -n "${ROSETTE_LOCAL_SOURCE}" ]; then
        if [ ! -d "${ROSETTE_LOCAL_SOURCE}" ]; then
            msg="ROSETTE_LOCAL_SOURCE_OVERRIDE=${ROSETTE_LOCAL_SOURCE} is absent"
            echo "${msg}" >&2
            fail_event rosette_front_door_eshkol "${msg}"
            fail_event rosette_external_backends_eshkol "${msg}"
            exit 1
        fi
        CLONE_FROM="${ROSETTE_LOCAL_SOURCE}"
    else
        CLONE_FROM="${ROSETTE_UPSTREAM_URL}"
    fi
    if ! git clone "${CLONE_FROM}" "${ROSETTE_DIR}"; then
        msg="git clone of rosette-wire from ${CLONE_FROM} failed"
        fail_event rosette_front_door_eshkol "${msg}"
        fail_event rosette_external_backends_eshkol "${msg}"
        exit 1
    fi
fi
if ! git -C "${ROSETTE_DIR}" checkout --quiet "${ROSETTE_PIN_SHA}"; then
    msg="git checkout of pinned rosette-wire commit ${ROSETTE_PIN_SHA} failed"
    fail_event rosette_front_door_eshkol "${msg}"
    fail_event rosette_external_backends_eshkol "${msg}"
    exit 1
fi

# ── 3. eshkol-run: build fresh, or accept an already-built binary ──────
# ESHKOL_RUN_BIN_OVERRIDE lets a caller point this gate at an already-built
# eshkol-run (e.g. to avoid a long local rebuild during development) instead
# of configuring/building this checkout. CI always builds fresh from THIS
# tree so the oracle actually exercises the commit under test.
if [ -n "${ESHKOL_RUN_BIN_OVERRIDE:-}" ]; then
    if [ ! -x "${ESHKOL_RUN_BIN_OVERRIDE}" ]; then
        msg="ESHKOL_RUN_BIN_OVERRIDE=${ESHKOL_RUN_BIN_OVERRIDE} is not an executable file"
        fail_event rosette_front_door_eshkol "${msg}"
        fail_event rosette_external_backends_eshkol "${msg}"
        exit 1
    fi
    ESHKOL_RUN_BIN="${ESHKOL_RUN_BIN_OVERRIDE}"
else
    LLVM_MAJOR="${LLVM_MAJOR:-21}"
    CMAKE_EXTRA_ARGS=()
    if [ -n "${ESHKOL_LLVM_CONFIG_EXECUTABLE:-}" ] && [ -x "${ESHKOL_LLVM_CONFIG_EXECUTABLE}" ]; then
        # CI (Linux): llvm-config-<major> is on PATH via apt, but not as the
        # bare `llvm-config` CMake looks for by default — point at it explicitly.
        CMAKE_EXTRA_ARGS+=("-DLLVM_CONFIG_EXECUTABLE=${ESHKOL_LLVM_CONFIG_EXECUTABLE}")
    elif command -v brew >/dev/null 2>&1; then
        LLVM_PREFIX="$(brew --prefix "llvm@${LLVM_MAJOR}" 2>/dev/null || true)"
        if [ -n "${LLVM_PREFIX}" ] && [ -x "${LLVM_PREFIX}/bin/llvm-config" ]; then
            export PATH="${LLVM_PREFIX}/bin:${PATH}"
            CMAKE_EXTRA_ARGS+=("-DLLVM_CONFIG_EXECUTABLE=${LLVM_PREFIX}/bin/llvm-config")
        fi
    fi

    if [ ! -d "${BUILD_DIR}" ]; then
        # NOTE: ESHKOL_BUILD_AGENT_FFI cannot be turned off here even though this
        # gate does not exercise agent FFI itself — eshkol-run's REPL JIT
        # unconditionally registers qllm_ffi_*/qllm_process_* runtime symbols
        # (lib/backend/repl_jit.cpp), so an AGENT_FFI=OFF configure fails to
        # link eshkol-run. A first configure on a machine with no prior build/
        # therefore pulls the (large, network-fetched) tree-sitter grammar set
        # this option also gates; that is unavoidable without changing the
        # compiler's own link closure, so it is accepted as one-time setup cost.
        if ! nice -n 19 cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DESHKOL_REQUIRED_LLVM_MAJOR="${LLVM_MAJOR}" \
            "${CMAKE_EXTRA_ARGS[@]}"; then
            msg="cmake configure failed"
            fail_event rosette_front_door_eshkol "${msg}"
            fail_event rosette_external_backends_eshkol "${msg}"
            exit 1
        fi
    fi
    if ! nice -n 19 cmake --build "${BUILD_DIR}" --target eshkol-run --parallel; then
        msg="eshkol-run build failed"
        fail_event rosette_front_door_eshkol "${msg}"
        fail_event rosette_external_backends_eshkol "${msg}"
        exit 1
    fi

    ESHKOL_RUN_BIN="${BUILD_DIR}/eshkol-run"
    if [ ! -x "${ESHKOL_RUN_BIN}" ]; then
        msg="built eshkol-run binary not found at ${ESHKOL_RUN_BIN}"
        fail_event rosette_front_door_eshkol "${msg}"
        fail_event rosette_external_backends_eshkol "${msg}"
        exit 1
    fi
fi

export ROSETTE_ESHKOL_BIN="${ESHKOL_RUN_BIN}"
export ROSETTE_ESHKOL_ID="$(cd "${REPO_ROOT}" && git describe --tags --always)"
# eshkol-run resolves its standard library relative to ESHKOL_PATH; without
# it a binary built into a different tree (or invoked from Rosette Wire's
# own working directory) cannot find lib/. ESHKOL_PATH_OVERRIDE lets a
# caller point this at a prebuilt tree's lib/ directory (paired with
# ESHKOL_RUN_BIN_OVERRIDE above); otherwise default to this repo's own lib/.
export ESHKOL_PATH="${ESHKOL_PATH_OVERRIDE:-${REPO_ROOT}/lib}"

ASDF_CACHE_DIR="${SCRATCH_DIR}/asdf-cache"
mkdir -p "${ASDF_CACHE_DIR}"

# ── 4. tools/test-system front-door/eshkol ──────────────────────────────
TEST_SYSTEM_OUT="${SCRATCH_DIR}/rosette-test-system.out"
TEST_SYSTEM_ERR="${SCRATCH_DIR}/rosette-test-system.err"
(
    cd "${ROSETTE_DIR}" && \
    nice -n 19 sbcl --script tools/test-system front-door/eshkol
) >"${TEST_SYSTEM_OUT}" 2>"${TEST_SYSTEM_ERR}"
TEST_SYSTEM_RC=$?

if [ "${TEST_SYSTEM_RC}" -eq 0 ]; then
    VERDICT_LINE="$(grep -Ei 'passed|success|[0-9]+ tests?, [0-9]+ failures?|ok' "${TEST_SYSTEM_OUT}" | tail -1)"
    [ -n "${VERDICT_LINE}" ] || VERDICT_LINE="$(tail -1 "${TEST_SYSTEM_OUT}")"
    pass_event rosette_front_door_eshkol "${VERDICT_LINE}"
else
    VERDICT_LINE="$(tail -5 "${TEST_SYSTEM_ERR}" "${TEST_SYSTEM_OUT}" 2>/dev/null | tr '\n' ' ')"
    fail_event rosette_front_door_eshkol "exit ${TEST_SYSTEM_RC}: ${VERDICT_LINE}"
fi

# ── 5. examples/external-backends.lisp eshkol <dir> ─────────────────────
EXT_OUT="${SCRATCH_DIR}/rosette-external-backends.out"
EXT_ERR="${SCRATCH_DIR}/rosette-external-backends.err"
(
    cd "${ROSETTE_DIR}" && \
    nice -n 19 sbcl --script examples/external-backends.lisp eshkol "${NATIVE_DIR}"
) >"${EXT_OUT}" 2>"${EXT_ERR}"
EXT_RC=$?

if [ "${EXT_RC}" -eq 0 ]; then
    VERDICT_LINE="$(grep -E '^eshkol: PASS' "${EXT_OUT}" | tail -1)"
    [ -n "${VERDICT_LINE}" ] || VERDICT_LINE="$(tail -1 "${EXT_OUT}")"
    pass_event rosette_external_backends_eshkol "${VERDICT_LINE}"
else
    VERDICT_LINE="$(grep -E 'External execution refused' "${EXT_ERR}" | tail -1)"
    [ -n "${VERDICT_LINE}" ] || VERDICT_LINE="$(tail -5 "${EXT_ERR}" "${EXT_OUT}" 2>/dev/null | tr '\n' ' ')"
    fail_event rosette_external_backends_eshkol "exit ${EXT_RC}: ${VERDICT_LINE}"
fi

echo "--- tools/test-system front-door/eshkol ---"
cat "${TEST_SYSTEM_OUT}"
echo "--- examples/external-backends.lisp eshkol ---"
cat "${EXT_OUT}"

if [ "${TEST_SYSTEM_RC}" -eq 0 ] && [ "${EXT_RC}" -eq 0 ]; then
    exit 0
fi
exit 1
