#!/usr/bin/env bash
# erepl_client_self_test.sh — EREPL v1 protocol self-test, run through the
# reference Python driver (tools/erepl_client.py --self-test) against a
# live `eshkol-repl --machine` child process over plain pipes (no PTY).
#
# Exercises every EREPL v1 request type (eval, complete, is_complete,
# reset, shutdown), a structured runtime-error payload (kind/message/
# line/column/filename/printed/irritants, never wording-matched), and an
# interrupted infinite loop that leaves the session usable afterward. See
# docs/reference/runtime/eshkol-repl.md ("Machine mode (EREPL protocol)")
# for the protocol this exercises, and the sibling
# repl_machine_mode_protocol_test.sh for the legacy bare-form framing this
# protocol stays backward compatible with.

set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON3="${ESHKOL_PYTHON3:-python3}"

if ! command -v "$PYTHON3" >/dev/null 2>&1; then
    echo "SKIP: $PYTHON3 not available"
    exit 0
fi

cd "$ROOT" || exit 1
BUILD_DIR="${BUILD_DIR:-build}" "$PYTHON3" tools/erepl_client.py --self-test
exit $?
