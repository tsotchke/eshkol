#!/usr/bin/env bash
# run_v14_connection_gate.sh — the v1.4-connection completion-oracle harness.
#
# .icc/completion-oracles.yaml's `v1.4-connection` target used to carry
# exactly one criterion for the whole release theme: a runtime_event nothing
# ever emitted (kind v1_4_connection / name connection_theme_deliverables,
# action `replace_with_v1_4_connection_oracle`). `icc readiness` graded that
# stub as a permanent, uninformative FAIL — it could never distinguish "v1.4
# hasn't started" from "v1.4 is half done" from "v1.4 shipped".
#
# This script is the replacement: ONE runtime_event probe per v1.4 deliverable
# (docs/COMPILER_ROADMAP.md's Networking/Concurrency/Wire-formats/Linear-types
# sections, ROADMAP.md's v1.4-connection list, docs/design/adr/0004's
# "resource-sound systems profile" exit gates, docs/design/adr/0000's Stage 3
# (v1.4.0) and Stage 4 (v1.4.1) gates, and docs/design/adr/0010's A8/A10-A13
# assurance gaps), each bound to real evidence:
#
#   * where a real, working primitive AND a real round-trip test already
#     exist, this script RUNS that evidence today and reports the actual
#     PASS/FAIL it produces (most of these were shipped ahead of the
#     roadmap's checkboxes and were simply never wired into the ICC oracle —
#     see the per-probe comments below for the executable evidence used);
#   * where this gate has no runnable or source-bound evidence for a
#     requirement, it emits an honest FAIL marked UNVERIFIED. Source presence
#     alone never earns a PASS, and an unverified status makes no claim that
#     the feature is absent;
#   * where external CI evidence is needed, the event requests a
#     source-bound CI receipt rather than inferring runner or hardware state.
#
# This script never fabricates a PASS. A criterion is PASS only when the
# thing it claims was actually exercised, this run, and produced the
# claimed result.
#
# Usage: scripts/run_v14_connection_gate.sh
#   BUILD_DIR   (env, default "build")   the CMake build directory
#   TRACE_DIR   (env, default scripts/icc_traces)

set -u
export LC_ALL=C LC_CTYPE=C LANG=C
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
    /*) BUILD_DIR_PATH="$BUILD_DIR" ;;
    *)  BUILD_DIR_PATH="$REPO_ROOT/$BUILD_DIR" ;;
esac
if [ ! -d "$BUILD_DIR_PATH" ]; then
    echo "run_v14_connection_gate.sh: build directory not found: $BUILD_DIR_PATH" >&2
    exit 2
fi
BUILD_DIR_PATH="$(cd "$BUILD_DIR_PATH" && pwd -P)" || exit 2
ESHKOL_RUN="$BUILD_DIR_PATH/eshkol-run"
VM_RUN="$BUILD_DIR_PATH/eshkol-vm-standalone-test"

TRACE_DIR="${TRACE_DIR:-$REPO_ROOT/scripts/icc_traces}"
# Evidence paths are absolute before first use (scripts/lib/evidence_paths.sh).
. "$REPO_ROOT/scripts/lib/evidence_paths.sh"
eshkol_evidence_abs_var TRACE_DIR "$REPO_ROOT" || exit $?
# shellcheck source=lib/checked_write.sh
. "$REPO_ROOT/scripts/lib/checked_write.sh"
TRACE_FILE="$TRACE_DIR/v1_4_connection.jsonl"
mkdir -p "$TRACE_DIR"
eshkol_require_output_file_path "$TRACE_FILE"
: > "$TRACE_FILE"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-v14-gate.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

PROBE_TOTAL=0
PROBE_FAILURES=0

emit_event() { # name PASS|FAIL snippet
    eshkol_require_output_file_path "$TRACE_FILE"
    python3 -c '
import json, sys
print(json.dumps({"kind": "v1_4_connection", "name": sys.argv[1], "value": sys.argv[2],
                  "snippet": sys.argv[3], "confidence": 0.95}, ensure_ascii=False))
' "$1" "$2" "$3" >> "$TRACE_FILE"
}

# A probe that actually runs a command and grades PASS/FAIL on its exit code.
probe_run() {
    local probe_id="$1" label="$2" cmd="$3"
    local out status snippet
    PROBE_TOTAL=$((PROBE_TOTAL + 1))
    out=$(eval "$cmd" 2>&1)
    status=$?
    if [ "$status" -eq 0 ] && ! printf '%s\n' "$out" | rg -qi '(^|[^[:alpha:]])(SKIP|UNVERIFIED)(:|[[:space:]]|$)'; then
        emit_event "$probe_id" PASS "${label}: OK"
        printf '  \xE2\x9C\x93 %-42s %s\n' "$probe_id" "$label"
    else
        PROBE_FAILURES=$((PROBE_FAILURES + 1))
        snippet=$(printf '%s' "$out" | tail -c 220)
        emit_event "$probe_id" FAIL "$snippet"
        printf '  \xE2\x9C\x97 %-42s %s (exit %d)\n' "$probe_id" "$label" "$status"
    fi
}

# #145 carries a second, immutable ICC receipt.  The ordinary sweep event is
# useful for the live counter, but cannot attest which source tree or runner
# binary produced it.  ICC's test-execution receipt stamps exact-clean-current
# source identity and hashes the declared executable as measurement input.
probe_http_server_source_receipt() {
    local icc_bin repo_name binding_output binding_status command_line jit_cache fingerprint_trace fingerprint_file out status snippet receipt_ok
    PROBE_TOTAL=$((PROBE_TOTAL + 1))
    icc_bin="${ICC_BIN:-$HOME/Desktop/infinite_context_coder/bin/icc}"
    if [ ! -x "$icc_bin" ]; then
        PROBE_FAILURES=$((PROBE_FAILURES + 1))
        emit_event v14_http_server_roundtrip FAIL "ICC receipt producer unavailable: $icc_bin"
        printf '  \xE2\x9C\x97 %-42s ICC receipt producer unavailable\n' v14_http_server_roundtrip
        return
    fi

    repo_name="${ICC_REPO_NAME:-eshkol}"
    binding_output=$(ICC_BIN="$icc_bin" ICC_REPO_NAME="$repo_name" \
        bash "$REPO_ROOT/tests/v1_2_edge_cases/http_server_smoke_test.sh" \
        --verify-icc-checkout-only 2>&1)
    binding_status=$?
    if [ "$binding_status" -ne 0 ]; then
        PROBE_FAILURES=$((PROBE_FAILURES + 1))
        emit_event v14_http_server_roundtrip FAIL "$binding_output"
        printf '  \xE2\x9C\x97 %-42s ICC repo checkout mismatch\n' v14_http_server_roundtrip
        return
    fi

    jit_cache="$WORK/http-server-jit-cache"
    # Keep the fingerprint receipt under the durable trace root; the rest of
    # the temporary probe state is safely removed when the sweep exits.
    fingerprint_trace="$TRACE_DIR/v14_http_server_build"
    mkdir -p "$jit_cache" "$fingerprint_trace"
    fingerprint_file="$fingerprint_trace/build_fingerprint.jsonl"
    command_line="env BUILD_DIR='$BUILD_DIR_PATH' TRACE_DIR='$fingerprint_trace' ICC_BIN='$icc_bin' ICC_REPO_NAME='$repo_name' ESHKOL_LIB_DIR='$BUILD_DIR_PATH' ESHKOL_JIT_CACHE_DIR='$jit_cache' bash ./tests/v1_2_edge_cases/http_server_smoke_test.sh"
    # The shell test invokes helper files transitively; ICC's command-seed
    # closure does not enumerate every sourced shell dependency. Disable that
    # relaxation so only an exact whole-tree source/index match is admissible;
    # any later commit retires this receipt until the round trip is rerun.
    out=$(ICC_RECEIPT_CLOSURE=0 "$icc_bin" test-execution-oracle \
        --repo "$repo_name" \
        --name v14_http_server_roundtrip \
        --cwd "$REPO_ROOT" \
        --declare-data "$ESHKOL_RUN" \
        --declare-data "$BUILD_DIR_PATH/stdlib.o" \
        --declare-data "$BUILD_DIR_PATH/stdlib.bc" \
        --declare-data "$fingerprint_file" \
        --declare-instrument scripts/run_v14_connection_gate.sh:http_server_smoke_test.sh \
        --command "$command_line" --format json 2>&1)
    status=$?
    receipt_ok=1
    if [ "$status" -eq 0 ]; then
        printf '%s' "$out" | python3 -c '
import json, re, sys
try:
    payload = json.load(sys.stdin)
    record = payload.get("record", {})
    value = record.get("value", {})
    source = value.get("source", {})
    data = (value.get("provenance", {}).get("data", {})
            if isinstance(value.get("provenance"), dict) else {})
    inputs = data.get("value") if isinstance(data, dict) else None
    binary, stdlib_object, stdlib_bitcode, fingerprint_path = sys.argv[1:]
    binary_rows = [row for row in inputs or []
                   if isinstance(row, dict) and row.get("path") == binary]
    required_paths = {stdlib_object, stdlib_bitcode, fingerprint_path}
    declared_paths = {row.get("path") for row in inputs or [] if isinstance(row, dict)}
    ok = (
        payload.get("ok") is True
        and record.get("schema") == "icc.test_execution.v1"
        and record.get("kind") == "test_result"
        and record.get("name") == "v14_http_server_roundtrip"
        and record.get("status") == "PASS"
        and source.get("dirty") is False
        and source.get("icc_fresh") is True
        and source.get("icc_source_fingerprint")
            == source.get("icc_index_source_fingerprint")
        and len(binary_rows) == 1
        and re.fullmatch(r"[0-9a-f]{64}", str(binary_rows[0].get("sha256") or ""))
        and required_paths.issubset(declared_paths)
        and all(
            re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256") or ""))
            for row in inputs or []
            if isinstance(row, dict) and row.get("path") in required_paths
        )
    )
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
' "$ESHKOL_RUN" "$BUILD_DIR_PATH/stdlib.o" "$BUILD_DIR_PATH/stdlib.bc" "$fingerprint_file" || receipt_ok=0
    else
        receipt_ok=0
    fi
    if [ "$receipt_ok" -eq 1 ]; then
        emit_event v14_http_server_roundtrip PASS "ICC test-execution receipt recorded; binary=$ESHKOL_RUN"
        printf '  \xE2\x9C\x93 %-42s source-bound ICC receipt\n' v14_http_server_roundtrip
    else
        PROBE_FAILURES=$((PROBE_FAILURES + 1))
        snippet=$(printf '%s' "$out" | tail -c 220)
        [ "$status" -eq 0 ] && snippet="ICC receipt did not bind a fresh source and runner SHA-256"
        emit_event v14_http_server_roundtrip FAIL "$snippet"
        printf '  \xE2\x9C\x97 %-42s ICC receipt failed (exit %d)\n' v14_http_server_roundtrip "$status"
    fi
}

# A requirement for which this gate has no current runnable or source-bound
# evidence. Keep it failing until a real probe or receipt can establish it;
# this status does not assert that the implementation is absent.
probe_unverified() {
    local probe_id="$1" reason="$2"
    PROBE_TOTAL=$((PROBE_TOTAL + 1))
    PROBE_FAILURES=$((PROBE_FAILURES + 1))
    emit_event "$probe_id" FAIL "UNVERIFIED: $reason"
    printf '  \xE2\x9C\x97 %-42s UNVERIFIED: %s\n' "$probe_id" "$reason"
}

# Requirements that need CI evidence stay failing until a source-bound receipt
# for the named workflow is supplied. Do not infer CI or runner availability.
probe_external_ci_evidence() {
    local probe_id="$1" workflow="$2" reason="$3"
    PROBE_TOTAL=$((PROBE_TOTAL + 1))
    PROBE_FAILURES=$((PROBE_FAILURES + 1))
    emit_event "$probe_id" FAIL "UNVERIFIED: no source-bound CI receipt supplied for $workflow — $reason"
    printf '  \xE2\x9C\x97 %-42s UNVERIFIED (CI receipt needed): %s — %s\n' "$probe_id" "$workflow" "$reason"
}

echo "Running v1.4-connection oracle probes -> $TRACE_FILE"
echo

if [ ! -x "$ESHKOL_RUN" ]; then
    echo "run_v14_connection_gate.sh: $ESHKOL_RUN not found - run \`cmake --build $BUILD_DIR_PATH\` first." >&2
    exit 2
fi

# ─────────────────────────────────────────────────────────────────
# Networking (docs/COMPILER_ROADMAP.md #145/#146/#148/#150/#161,
# ROADMAP.md v1.4-connection)
# ─────────────────────────────────────────────────────────────────

# #145 HTTP server: real primitives (lib/agent/c/agent_http_server.c
# http-server-create/-accept/-respond/-close, real bind/listen/accept/recv/
# send) + a real fork+client+server round trip over loopback, already
# CI-wired via the tests/v1_2_edge_cases/*.esk glob.
probe_http_server_source_receipt

# #148 Prometheus metrics + /metrics endpoint: lib/core/metrics.esk
# (make-counter/counter-inc!/metrics-render) is real and wired into
# http-standard-response's /metrics route (lib/core/http_server.esk). No
# existing test drove a real GET /metrics over a socket before this gate;
# tests/v1_2_edge_cases/metrics_http_roundtrip_test.sh (added alongside this
# script) does.
probe_run v14_metrics_http_roundtrip \
    'Prometheus /metrics (#148): real GET /metrics over loopback returns HELP/TYPE + counter value' \
    'BUILD_DIR="$BUILD_DIR_PATH" ./tests/v1_2_edge_cases/metrics_http_roundtrip_test.sh'

# Unix domain sockets: this probe runs the real round-trip (T1-T8) against a
# local AF_UNIX listener, including the environment-gated path.
run_unix_socket_roundtrip() {
    local sock="$WORK/v14_unix_socket_probe.sock"
    local module="$WORK/unix_socket_test.eskb"
    local out="$WORK/unix_socket_test.out"
    rm -f "$sock"
    python3 - "$sock" > "$WORK/unix_socket_listener.out" 2>&1 <<'PYEOF' &
import socket, sys, os
path = sys.argv[1]
if os.path.exists(path):
    os.remove(path)
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.settimeout(15)
s.bind(path)
s.listen(1)
conn, _ = s.accept()
data = conn.recv(256)
if data == b"vm-socket-ping":
    conn.sendall(b"vm-socket-pong")
conn.close()
s.close()
PYEOF
    local listener_pid=$!
    local waited=0
    while [ ! -S "$sock" ] && [ "$waited" -lt 50 ]; do sleep 0.1; waited=$((waited + 1)); done
    "$ESHKOL_RUN" --profile hosted-vm --emit-eskb "$module" \
        tests/vm/unix_socket_surface_regression.esk >"$WORK/unix_socket_compile.out" 2>&1 || true
    ESHKOL_VM_UNIX_SOCKET_TEST="$sock" "$VM_RUN" "$module" >"$out" 2>&1
    local rc=$?
    wait "$listener_pid" 2>/dev/null
    rm -f "$sock"
    if [ "$rc" -ne 0 ]; then return "$rc"; fi
    grep -q "^FAIL" "$out" && return 1
    grep -q "T4 socket-recv reads response string:" "$out" || return 1
    return 0
}
if [ -x "$VM_RUN" ]; then
    if run_unix_socket_roundtrip; then
        PROBE_TOTAL=$((PROBE_TOTAL + 1))
        emit_event v14_unix_socket_roundtrip PASS \
            "unix-socket-connect/socket-send/socket-recv/socket-close round-trip against a real AF_UNIX listener (T1-T8 of tests/vm/unix_socket_surface_regression.esk, driven with ESHKOL_VM_UNIX_SOCKET_TEST set for the first time by this gate)"
        printf '  \xE2\x9C\x93 %-42s %s\n' v14_unix_socket_roundtrip "Unix domain socket real round-trip"
    else
        PROBE_TOTAL=$((PROBE_TOTAL + 1)); PROBE_FAILURES=$((PROBE_FAILURES + 1))
        emit_event v14_unix_socket_roundtrip FAIL "real AF_UNIX round-trip against tests/vm/unix_socket_surface_regression.esk failed"
        printf '  \xE2\x9C\x97 %-42s FAILED\n' v14_unix_socket_roundtrip
    fi
else
    probe_unverified v14_unix_socket_roundtrip "local VM runner is unavailable; a real AF_UNIX round-trip is required"
fi

# #150 Resource limits: real enforcement (lib/core/resource_limits.cpp,
# ESHKOL_MAX_HEAP/ESHKOL_TIMEOUT_MS/etc.) with a real CTest gate
# (tests/limits/resource_limits_enforcement_gate.sh) that CMake registers
# invoke the focused enforcement gate directly to record this run's result.
if [ -x "$VM_RUN" ]; then
    probe_run v14_resource_limits_enforced \
        'Resource limits (#150): CPU/wall-time/memory/tensor/string/stack ceilings terminate with documented exit codes, and an unreached ceiling changes nothing' \
        '"$REPO_ROOT/tests/limits/resource_limits_enforcement_gate.sh" "$ESHKOL_RUN" "$VM_RUN" "$WORK/limits"'
else
    probe_unverified v14_resource_limits_enforced "local VM runner is unavailable; resource-limit enforcement evidence is required"
fi

# Numeric TCP/UDP use the public module with binary payloads, deadlines,
# endpoints and stale-handle checks. Other protocols still require their own
# implementation and verified runtime evidence.
probe_run v14_tcp_roundtrip 'public core.net TCP loopback round trip' \
    'python3 "$REPO_ROOT/scripts/run_net_gate.py" --runner "$ESHKOL_RUN"'
probe_run v14_udp_roundtrip 'public core.net UDP loopback round trip' \
    'python3 "$REPO_ROOT/scripts/run_net_gate.py" --runner "$ESHKOL_RUN"'
probe_unverified v14_tls_handshake_verified \
    'requires a verified TLS handshake test'
probe_unverified v14_websocket_server_echo \
    'requires a WebSocket server upgrade and echo round-trip test'

# ─────────────────────────────────────────────────────────────────
# Concurrency (docs/COMPILER_ROADMAP.md #156-#160, M3)
# ─────────────────────────────────────────────────────────────────

# #158 already ships (v1.3.4) and already gates release readiness under
# kind eshkol_smoke/event_loop_works (.icc/completion-oracles.yaml). This
# probe re-emits the SAME evidence under the v1_4_connection family so the
# new oracle proves its own wiring end-to-end rather than relying solely on
# a criterion authored under a different kind before this stanza existed.
if [ -x "$ESHKOL_RUN" ]; then
    probe_run v14_async_io_event_loop \
        'Async I/O event loop (#158): kqueue/epoll/IOCP pipe round-trip via event-loop-poll, timeout, close-then-use fails closed, 1000 open/close cycles' \
        '"$ESHKOL_RUN" -r tests/v1_3_edge_cases/event_loop_test.esk 2>&1 | grep -q "PASS: event_loop_test"'
fi

# #156 threads + mutex + condvar: real pthread-backed core.threads.esk
# (make-mutex/mutex-lock!/make-condvar/make-thread/thread-join), and a real
# concurrency test — 8 parallel-map workers x 1000 mutex-protected
# increments on a shared cell, asserting the exact total (a real race would
# lose increments) — already CI-wired via the tests/v1_2_edge_cases/*.esk
# glob but never fed to the ICC oracle.
probe_run v14_threads_mutex_condvar \
    'Threads + mutex + condvar (#156): 8 parallel-map workers x 1000 mutex-protected increments preserve the exact total' \
    '"$ESHKOL_RUN" -r tests/v1_2_edge_cases/threads_mutex_concurrency_test.esk 2>&1 | grep -q "PASS: mutex-protected counter is consistent across threads"'

# #157 channels: real bounded-ring-buffer CSP channels (lib/core/channels.esk)
# built on core.threads, with a real parallel-map producer/consumer test.
probe_run v14_channels_csp \
    'Channels (#157): parallel-map producer/consumer traffic over a bounded CSP channel preserves the total and drains cleanly' \
    '"$ESHKOL_RUN" -r tests/v1_2_edge_cases/channels_test.esk 2>&1 | grep -q "^Failed: 0$"'

# #160 promises/futures: future/force/force-future are real pthread-backed
# VM builtins (lib/backend/vm_region_evac.c: "thunk/result Values, plus a
# live pthread mutex+cond"); core.threads' make-thread is built directly on
# (future thunk). tests/v1_2_edge_cases/future_force_test.esk is the #222
# regression that proves force actually runs the thunk (pre-fix it silently
# fell through to "return as-is").
probe_run v14_promises_futures \
    'Promises/futures (#160): future/force run the thunk and multiple futures resolve independently (regression #222)' \
    '"$ESHKOL_RUN" -r tests/v1_2_edge_cases/future_force_test.esk 2>&1 | grep -q "^Failed: 0$"'

# Atomic ops (unlisted #-number, "Atomic ops (CAS, fetch-add)" in the
# Concurrency table): real atomic-store!/-load/-exchange!/-compare-exchange!/
# -fetch-add!/-fetch-sub!/-fetch-and!/-fetch-or!/-fetch-xor! with explicit
# memory-order args. The only test today is SEQUENTIAL (API-shape) — no
# cross-thread CAS-under-contention probe exists, so this criterion proves
# the primitive surface only, not cross-thread atomicity; the label says so.
probe_run v14_atomics_surface \
    'Atomic ops: store/load/exchange/compare-exchange/fetch-add/sub/and/or/xor pass the sequential API-surface test' \
    '"$ESHKOL_RUN" -r tests/ffi/low_level_memory_surface_test.esk 2>&1 | grep -q "Low-level FFI surface:.*0 failed"'

# #159 fibers/coroutines and semaphores/barriers require end-to-end evidence.
probe_unverified v14_fibers_coroutines \
    'requires an end-to-end fibers or coroutines behavior test'
probe_unverified v14_semaphores_barriers \
    'requires an end-to-end semaphore or synchronization-barrier behavior test'

# ─────────────────────────────────────────────────────────────────
# Wire formats (docs/COMPILER_ROADMAP.md #162/#163/#175)
# ─────────────────────────────────────────────────────────────────

# #162 MessagePack: lib/core/msgpack.esk is a real, complete encoder/decoder
# over the deterministic subset (nil/bool/signed+unsigned ints to 32-bit/
# fixstr-str8-str16/bin8-bin16/fixarray-array16/fixmap-map16 — no ext-type
# hooks, no int64/float, a documented v1.8 wire-format substrate subset).
# tests/v1_2_edge_cases/msgpack_test.esk round-trips every type against
# exact wire bytes via core.testing's run-tests.
probe_run v14_msgpack_roundtrip \
    'MessagePack (#162): exact wire bytes, UTF-8 scalar boundaries, embedded NUL, malformed-text refusal and raw binary preservation' \
    'python3 scripts/run_msgpack_gate.py --runner "$ESHKOL_RUN"'

# #175 CAS + Merkle trees: lib/core/merkle.esk (merkle-leaf/-inode/-tree/
# -leaves/-proof/-verify plus make-cas/cas-put!/cas-get/cas-has?) backed by
# lib/core/merkle.c hash primitives. tests/v1_2_edge_cases/merkle_test.esk
# self-checks tree build/determinism/order-sensitivity, inclusion-proof
# verify (valid/tampered/empty), and full CAS round-trip incl. dedup.
probe_run v14_cas_merkle_roundtrip \
    'Content-addressable storage + Merkle trees (#175): tree build, inclusion-proof verify (valid/tampered/empty), CAS put/get/dedup round-trip' \
    '"$ESHKOL_RUN" -r tests/v1_2_edge_cases/merkle_test.esk 2>&1 | grep -q "^Failed: 0$"'

# #163 Protocol Buffers requires a schema-driven proto3 encode/decode test.
probe_unverified v14_protobuf_roundtrip \
    'requires a proto3 schema parser/compiler and generated encoder/decoder round-trip test'

# ─────────────────────────────────────────────────────────────────
# Linear resource types + borrow pattern (docs/design/adr/0004-type-
# system-trajectory.md "v1.4: the resource-sound systems profile")
# ─────────────────────────────────────────────────────────────────

# The generative linear/affine kernel is real and wired into the checker:
# Context::bindLinear/useLinear/checkLinearConstraints (lib/types/
# type_checker.cpp), exercised end-to-end today on the Qubit type (which
# carries TYPE_FLAG_LINEAR, same mechanism ADR-0004 proposes reusing for
# Own/socket typestate) via a real negative compile-fail fixture: a program
# that clones a linear Qubit must FAIL to compile under --strict-types (a
# type error, not a warning -- gradual/default mode only warns, which is
# why this probe passes --strict-types explicitly).
probe_run v14_linear_consume_exactly_once_static \
    'Linear/affine kernel (ADR-0004): a program that uses a linear Qubit twice is REJECTED at compile time under --strict-types (no-cloning), proving the same mechanism v1.4 needs for Own/socket typestate' \
    '"$ESHKOL_RUN" --strict-types tests/typesystem/qubit_no_cloning_test.esk -o "$WORK/qubit_bad" >/dev/null 2>&1; rc=$?; [ "$rc" -ne 0 ] && [ ! -e "$WORK/qubit_bad" ]'

# These requirements need end-to-end compile-fail evidence for resource
# ownership diagnostics. The gate currently has no such probe attached.
probe_unverified v14_socket_leak_rejected_statically \
    'requires a source-bound compile-fail test proving leaked socket ownership is rejected'
probe_unverified v14_double_close_rejected_statically \
    'requires a source-bound compile-fail test proving a second close is rejected statically'

# Borrow-pattern enforcement needs a source-bound program test through the
# real checker, including the intended rejection/acceptance cases.
probe_unverified v14_borrow_pattern_honored \
    'requires source-bound program tests demonstrating borrow-pattern enforcement through the real checker'

# ─────────────────────────────────────────────────────────────────
# Stage 4 / v1.4.1 (docs/design/adr/0000-unified-trajectory.md Stage 4:
# "OALR ABI v2 and portable tail transfer"). NOTE: v1.4.1 is its own
# release, one stage AFTER v1.4.0-connection (which this oracle targets),
# per the ADR's own staging and the maintainer-reviewed draft ladder
# (v1.4.0 "the systems profile" [S3] vs v1.4.1 "the ABI release" [S4]).
# These two criteria are included here because the task that authored this
# oracle explicitly asked for S4 coverage; they are scoped and labelled as
# v1.4.1-stage so a v1.4.0 readiness reader does not mistake them for
# blocking the v1.4.0 cut.
# ─────────────────────────────────────────────────────────────────
probe_unverified v14_1_oalr_abi_v2_header \
    'v1.4.1-STAGE (S4): requires a source-bound ABI v2 layout and compatibility contract check'
probe_unverified v14_1_portable_tail_transfer \
    'v1.4.1-STAGE (S4): requires deep mutual-tail-recursion evidence across supported backends'

# ─────────────────────────────────────────────────────────────────
# Assurance (docs/design/adr/0010-closed-loop-assurance.md A8, A10-A13)
# ─────────────────────────────────────────────────────────────────
probe_external_ci_evidence v14_assurance_a8_tsan_lane \
    '.github/workflows/adversarial-nightly.yml' \
    'requires the applicable A8 policy evidence and a source-bound TSan CI receipt'
probe_unverified v14_assurance_a10_packaging_manifest \
    'requires a source-bound A10 packaging-manifest check and its recorded result'
probe_unverified v14_assurance_a11_diagnostic_corpus \
    'requires a source-bound diagnostic corpus check for expected diagnostic codes and spans'
probe_unverified v14_assurance_a12_lsan_lane \
    'requires source-bound leak-sanitizer and region-evacuation poison-path evidence'
probe_external_ci_evidence v14_assurance_a13_hosted_gpu_lane \
    '.github/workflows/gpu-execution-gate.yml' \
    'requires a source-bound A13 hosted-GPU CI receipt; runner availability is not inferred here'

# ─────────────────────────────────────────────────────────────────
# Distribution (W6, PENDING MAINTAINER RULING R7). These criteria need
# source-bound evidence for the v1.4.0-staged distribution requirements.
# ─────────────────────────────────────────────────────────────────
probe_unverified v14_distribution_pjrt_client_spike \
    'PENDING R7: requires source-bound evidence for the PJRT client spike requirement'
probe_unverified v14_distribution_xla_multi_device_single_host \
    'PENDING R7: requires source-bound multi-device single-host XLA execution evidence'
probe_unverified v14_distribution_native_collectives_over_sockets \
    'PENDING R7: requires source-bound native collective-over-sockets execution evidence'

echo
echo "Passed: $((PROBE_TOTAL - PROBE_FAILURES))"
echo "Failed: $PROBE_FAILURES"
echo "Total:  $PROBE_TOTAL"
echo "Trace written: $TRACE_FILE"

if [ "$PROBE_FAILURES" -eq 0 ]; then
    emit_event v14_connection_sweep_clean PASS "all $PROBE_TOTAL v1.4-connection probes green"
else
    emit_event v14_connection_sweep_clean FAIL "$PROBE_FAILURES of $PROBE_TOTAL v1.4-connection probes not satisfied (expected -- v1.4-connection has not shipped; this sweep is the honest running count, not a release gate)"
fi

exit 0
