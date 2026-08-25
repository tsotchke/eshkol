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
#     see the per-probe comments below for exactly what exists and where);
#   * where the deliverable genuinely does not exist yet, this script emits
#     an honest FAIL with a `not yet implemented` snippet pointing at the
#     grep that proves the absence, so the criterion is real and gradeable
#     the moment someone lands the feature (swap the stub body for a real
#     probe; the .icc/completion-oracles.yaml criterion does not change);
#   * where the evidence lives in scheduled CI (nightly TSan, self-hosted
#     GPU) rather than something this fast local gate can reproduce, this
#     script says so explicitly rather than faking a PASS or silently
#     skipping the criterion.
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
ESHKOL_RUN="$BUILD_DIR_PATH/eshkol-run"
VM_RUN="$BUILD_DIR_PATH/eshkol-vm-standalone-test"

TRACE_DIR="${TRACE_DIR:-$REPO_ROOT/scripts/icc_traces}"
TRACE_FILE="$TRACE_DIR/v1_4_connection.jsonl"
mkdir -p "$TRACE_DIR"
: > "$TRACE_FILE"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-v14-gate.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

PROBE_TOTAL=0
PROBE_FAILURES=0

emit_event() { # name PASS|FAIL snippet
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
    if [ "$status" -eq 0 ]; then
        emit_event "$probe_id" PASS "${label}: OK"
        printf '  \xE2\x9C\x93 %-42s %s\n' "$probe_id" "$label"
    else
        PROBE_FAILURES=$((PROBE_FAILURES + 1))
        snippet=$(printf '%s' "$out" | tail -c 220)
        emit_event "$probe_id" FAIL "$snippet"
        printf '  \xE2\x9C\x97 %-42s %s (exit %d)\n' "$probe_id" "$label" "$status"
    fi
}

# A deliverable that does not exist in the tree yet. Emits an honest FAIL
# with the evidence that it is absent, so the criterion stays real (never
# silently dropped) until someone lands the feature and swaps this stub for
# a probe_run call.
probe_not_yet_implemented() {
    local probe_id="$1" reason="$2"
    PROBE_TOTAL=$((PROBE_TOTAL + 1))
    PROBE_FAILURES=$((PROBE_FAILURES + 1))
    emit_event "$probe_id" FAIL "NOT YET IMPLEMENTED: $reason"
    printf '  \xE2\x9C\x97 %-42s NOT YET IMPLEMENTED: %s\n' "$probe_id" "$reason"
}

# A deliverable whose evidence lives in scheduled/self-hosted CI (nightly
# TSan, self-hosted GPU) that this fast local gate cannot reproduce without
# a multi-hour toolchain rebuild or hardware this box does not have. Says so
# rather than faking local evidence.
probe_external_ci_evidence() {
    local probe_id="$1" workflow="$2" reason="$3"
    PROBE_TOTAL=$((PROBE_TOTAL + 1))
    PROBE_FAILURES=$((PROBE_FAILURES + 1))
    emit_event "$probe_id" FAIL "EVIDENCE IS EXTERNAL CI, NOT LOCAL: $workflow — $reason"
    printf '  \xE2\x9C\x97 %-42s EXTERNAL CI EVIDENCE (%s): %s\n' "$probe_id" "$workflow" "$reason"
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
probe_run v14_http_server_roundtrip \
    'HTTP server (#145): real fork+client GET /health over loopback round-trips' \
    'BUILD_DIR="$BUILD_DIR_PATH" ./tests/v1_2_edge_cases/http_server_smoke_test.sh'

# #148 Prometheus metrics + /metrics endpoint: lib/core/metrics.esk
# (make-counter/counter-inc!/metrics-render) is real and wired into
# http-standard-response's /metrics route (lib/core/http_server.esk). No
# existing test drove a real GET /metrics over a socket before this gate;
# tests/v1_2_edge_cases/metrics_http_roundtrip_test.sh (added alongside this
# script) does.
probe_run v14_metrics_http_roundtrip \
    'Prometheus /metrics (#148): real GET /metrics over loopback returns HELP/TYPE + counter value' \
    'BUILD_DIR="$BUILD_DIR_PATH" ./tests/v1_2_edge_cases/metrics_http_roundtrip_test.sh'

# Unix domain sockets: unix-socket-connect/socket-send/socket-recv/
# socket-close are real VM builtins (tests/vm/unix_socket_surface_regression.esk)
# but the real round-trip (T2-T5) is gated behind ESHKOL_VM_UNIX_SOCKET_TEST,
# which no CI workflow ever sets — only the hermetic negative paths run by
# default. This probe supplies the missing half: a real AF_UNIX listener.
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
    probe_not_yet_implemented v14_unix_socket_roundtrip "eshkol-vm-standalone-test not built"
fi

# #150 Resource limits: real enforcement (lib/core/resource_limits.cpp,
# ESHKOL_MAX_HEAP/ESHKOL_TIMEOUT_MS/etc.) with a real CTest gate
# (tests/limits/resource_limits_enforcement_gate.sh) that CMake registers
# but no CI workflow ever runs via ctest. Invoke it directly.
if [ -x "$VM_RUN" ]; then
    probe_run v14_resource_limits_enforced \
        'Resource limits (#150): CPU/wall-time/memory/tensor/string/stack ceilings terminate with documented exit codes, and an unreached ceiling changes nothing' \
        '"$REPO_ROOT/tests/limits/resource_limits_enforcement_gate.sh" "$ESHKOL_RUN" "$VM_RUN" "$WORK/limits"'
else
    probe_not_yet_implemented v14_resource_limits_enforced "eshkol-vm-standalone-test not built"
fi

# TCP/UDP/TLS/WebSocket-server: confirmed absent by direct grep (tcp-connect,
# tcp-listen, udp-socket, udp-bind, tls-connect, tls-context, make-tls: zero
# hits anywhere in lib/, inc/, tests/). WebSocket: only a CLIENT exists
# (lib/core/system_builtins.c websocket-connect/-send/-receive/-close,
# performs the client-side opening handshake); there is no server-side
# accept/upgrade path (no Sec-WebSocket-Accept computation, no listen loop).
probe_not_yet_implemented v14_tcp_roundtrip \
    'no tcp-connect/tcp-listen/tcp-accept primitive exists (grep lib/ inc/ tests/ for tcp-connect|tcp-listen: no hits)'
probe_not_yet_implemented v14_udp_roundtrip \
    'no udp-socket/udp-bind primitive exists (grep lib/ inc/ tests/ for udp-socket|udp-bind: no hits)'
probe_not_yet_implemented v14_tls_handshake_verified \
    'no TLS primitive exists (grep lib/ inc/ tests/ for tls-connect|tls-context|make-tls: no hits)'
probe_not_yet_implemented v14_websocket_server_echo \
    'only a WebSocket CLIENT exists (lib/core/system_builtins.c websocket-connect/-send/-receive/-close); no server-side upgrade/accept path'

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
    'Atomic ops: store/load/exchange/compare-exchange/fetch-add/sub/and/or/xor are correct SEQUENTIALLY (no cross-thread contention probe exists yet)' \
    '"$ESHKOL_RUN" -r tests/ffi/low_level_memory_surface_test.esk 2>&1 | grep -q "Low-level FFI surface:.*0 failed"'

# #159 fibers/coroutines, and semaphores/barriers: confirmed absent.
probe_not_yet_implemented v14_fibers_coroutines \
    'no make-fiber/fiber-*/coroutine primitive exists anywhere in lib/ or tests/'
probe_not_yet_implemented v14_semaphores_barriers \
    'no make-semaphore/make-barrier primitive exists; the only "barrier" hits in the tree are the GC region write-barrier (lib/core/runtime_regions.cpp), unrelated to concurrency'

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
    'MessagePack (#162): encode/decode round-trips nil/bool/int/string/bin/array/map against exact wire bytes (deterministic subset; no ext-types, no int64/float)' \
    '"$ESHKOL_RUN" -r tests/v1_2_edge_cases/msgpack_test.esk 2>&1 | grep -q "RESULT: OK"'

# #175 CAS + Merkle trees: lib/core/merkle.esk (merkle-leaf/-inode/-tree/
# -leaves/-proof/-verify plus make-cas/cas-put!/cas-get/cas-has?) backed by
# lib/core/merkle.c hash primitives. tests/v1_2_edge_cases/merkle_test.esk
# self-checks tree build/determinism/order-sensitivity, inclusion-proof
# verify (valid/tampered/empty), and full CAS round-trip incl. dedup.
probe_run v14_cas_merkle_roundtrip \
    'Content-addressable storage + Merkle trees (#175): tree build, inclusion-proof verify (valid/tampered/empty), CAS put/get/dedup round-trip' \
    '"$ESHKOL_RUN" -r tests/v1_2_edge_cases/merkle_test.esk 2>&1 | grep -q "^Failed: 0$"'

# #163 Protocol Buffers: confirmed absent. lib/core/onnx_export.c writes
# protobuf wire bytes BY HAND for ONNX export (no .proto parser, no schema
# compiler, no generated encoders/decoders) -- that is a one-off exporter,
# not #163's "proto3 parser/compiler subset sufficient for generated
# encoders/decoders".
probe_not_yet_implemented v14_protobuf_roundtrip \
    'no .proto parser/schema compiler exists; lib/core/onnx_export.c hand-writes protobuf wire bytes for ONE fixed ONNX schema, not a general proto3 subset'

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

# The v1.4 deliverable itself -- Own/socket typestate rejecting a leaked or
# double-closed socket -- is NOT yet built: real OS resources (files,
# sockets, event-loop handles in lib/core/system_builtins.c) are plain
# runtime integers checked DYNAMICALLY ("use-after-close fails closed... a
# catchable condition", system_builtins.c) -- there is no ESHKOL_VALUE_SOCKET
# linear type, so nothing today makes a leaked socket a STATIC type error.
probe_not_yet_implemented v14_socket_leak_rejected_statically \
    'no ESHKOL_VALUE_SOCKET / linear socket type exists; sockets are plain runtime file descriptors with only DYNAMIC use-after-close guards (lib/core/system_builtins.c), not a static Own/Borrow typestate'
probe_not_yet_implemented v14_double_close_rejected_statically \
    'double-close on a real socket/file handle is a dynamic catchable error today, not a static type error -- same gap as v14_socket_leak_rejected_statically'

# BorrowChecker (Owned/Moved/Dropped/BorrowedShared/BorrowedMut states,
# move/drop/borrowShared/borrowMut) exists in lib/types/type_checker.cpp but
# is almost entirely UNWIRED: only canBorrowMut/getState are called, at one
# mutation-target check site. move()/drop()/borrowShared()/returnBorrow()
# are exercised only by isolated gtest unit tests, never by real program
# compilation -- so no .esk program can observe borrow-pattern enforcement
# today.
probe_not_yet_implemented v14_borrow_pattern_honored \
    'BorrowChecker class exists (lib/types/type_checker.cpp) but only canBorrowMut/getState are wired into the real checking pass; move/drop/borrowShared/returnBorrow are unit-tested in isolation only, never reachable by compiling an actual .esk program'

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
probe_not_yet_implemented v14_1_oalr_abi_v2_header \
    'v1.4.1-STAGE (S4): the object header is still eshkol_object_header_t, statically asserted at 8 bytes (inc/eshkol/eshkol.h); zero hits anywhere for ESHKOL_MEMORY_ABI_V2, a 32-byte header, layout descriptors, escape ledgers, or transfer capsules -- design-only per ADR-0000/0004, nothing started'
probe_not_yet_implemented v14_1_portable_tail_transfer \
    'v1.4.1-STAGE (S4): musttail is emitted conditionally on the LLVM backend only for matching self/named-let tail calls; docs/reference/language/tail-calls.md documents mutual tail recursion as a KNOWN, UNFIXED limitation (SIGILL past ~500k depth); no VM-backend musttail equivalent, no heap-owned continuation chains, no general tail-transfer dispatcher exist'

# ─────────────────────────────────────────────────────────────────
# Assurance (docs/design/adr/0010-closed-loop-assurance.md A8, A10-A13)
# ─────────────────────────────────────────────────────────────────
probe_external_ci_evidence v14_assurance_a8_tsan_lane \
    '.github/workflows/adversarial-nightly.yml (nightly cron, -DESHKOL_ENABLE_TSAN=ON, generated concurrency corpus under TSAN_OPTIONS=halt_on_error=1)' \
    'a TSan-enabled LLVM build is multi-hour; this fast local gate does not rebuild the toolchain. The lane is real and runs nightly, but "required" (branch-protection-blocking, per ADR-0010 A8''s v1.4 target) is a policy step this gate cannot itself verify -- only a real nightly CI run can feed this event PASS.'
probe_not_yet_implemented v14_assurance_a10_packaging_manifest \
    'A10 is marked "Planned" in ADR-0010''s own gap table; no toolchain_flag/type_diagnostic/package_surface manifest categories exist in .icc/architecture-model.yaml or .icc/completion-oracles.yaml. Partial: adversarial-nightly.yml does install the homebrew formula, but that is not the manifest-driven packaging LANE A10 asks for'
probe_not_yet_implemented v14_assurance_a11_diagnostic_corpus \
    'no diagnostic golden-corpus (input -> expected diagnostic code + span) directory exists anywhere under tests/'
probe_not_yet_implemented v14_assurance_a12_lsan_lane \
    'ASan lanes explicitly set detect_leaks=0 (.github/workflows/ci.yml) -- leak detection is OFF, not gated with a suppression file; lib/backend/vm_core.c has zero ESHKOL_ARENA_POISON hits, so the region_evac POISON extension to the VM route A12 asks for has not landed either'
probe_external_ci_evidence v14_assurance_a13_hosted_gpu_lane \
    '.github/workflows/gpu-execution-gate.yml (daily cron, runs-on: [self-hosted, gpu], emits a ::warning when no runner is registered)' \
    'this box has no self-hosted GPU runner registered; the lane and its staleness-WARN rule are real and already match A13''s ask, but only a real scheduled run on the registered runner can feed this event PASS'

# ─────────────────────────────────────────────────────────────────
# Distribution (W6, PENDING MAINTAINER RULING R7 -- see
# ~/Desktop/Selene/ESHKOL-ROADMAP-v135-to-v20-DRAFT.md section 5). The
# two-tier-vs-Tier-1-only scope decision has NOT been made; these three
# criteria cover only the items explicitly staged for v1.4.0 in that draft
# (PJRT client spike, XLA multi-device single-host, native collectives over
# sockets) -- NOT the v1.5.0+ Tier-2 bit-identical-allreduce mesh gate,
# which is out of v1.4 scope regardless of how R7 is ruled. This oracle
# does not commit the project to either branch of R7; it only measures
# whether the v1.4.0-staged spike work exists.
# ─────────────────────────────────────────────────────────────────
probe_not_yet_implemented v14_distribution_pjrt_client_spike \
    'PENDING R7: no "pjrt"/"PJRT" reference exists anywhere in the repo (code, headers, docs, build) -- confirmed by a whole-tree case-insensitive grep'
probe_not_yet_implemented v14_distribution_xla_multi_device_single_host \
    'PENDING R7: lib/backend/xla/ has zero hits for device_ordinal/num_replicas/num_partitions; docs/breakdown/GPU_ACCELERATION.md documents device-0-only enumeration and a single CUDA stream for all operations -- XLA execution is single-device only today'
probe_not_yet_implemented v14_distribution_native_collectives_over_sockets \
    'PENDING R7: zero hits for allreduce/all_reduce/sharding/GSPMD/replica in lib/backend/xla/ or lib/backend/gpu/; the only "distributed" module in the repo (docs/reference/stdlib/distributed.md) is pure-value Lamport/vector-clock/CRDT math, explicitly independent of sockets/async I/O'

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
