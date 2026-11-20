#!/usr/bin/env bash
# Proves the HTTP evidence runner rejects a registered ICC repo rooted at a
# different checkout before it can exercise loopback or emit PASS evidence.

set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ICC_REPO_NAME="${MISMATCH_ICC_REPO_NAME:-eshkol}"

if ! icc_bin=$(type -P -- "${ICC_BIN:-icc}"); then
    echo "SKIP: ICC client unavailable: ${ICC_BIN:-icc}"
    exit 0
fi

REGISTERED_ROOT=$("$icc_bin" resolve --repo "$ICC_REPO_NAME" --format json 2>/dev/null | python3 -c '
import json, os, sys
try:
    path = json.load(sys.stdin).get("repo", {}).get("path", "")
    print(os.path.realpath(path) if path else "")
except Exception:
    print("")
')
CHECKOUT_ROOT=$(cd "$ROOT" && pwd -P)

if [ -z "$REGISTERED_ROOT" ]; then
    echo "FAIL: ICC repo $ICC_REPO_NAME could not be resolved"
    exit 1
fi
if [ "$REGISTERED_ROOT" = "$CHECKOUT_ROOT" ]; then
    echo "SKIP: ICC repo $ICC_REPO_NAME already maps to this checkout"
    exit 0
fi

output=$(ICC_BIN="$icc_bin" ICC_REPO_NAME="$ICC_REPO_NAME" BUILD_DIR="$ROOT/build" \
    bash "$ROOT/tests/v1_2_edge_cases/http_server_smoke_test.sh" \
    --verify-icc-checkout-only 2>&1)
status=$?
if [ "$status" -eq 3 ] && [[ "$output" == *"ICC repo $ICC_REPO_NAME resolves to"* ]]; then
    echo "PASS: mismatched ICC checkout refused before HTTP evidence"
    exit 0
fi

echo "FAIL: expected checkout mismatch refusal (exit 3), got exit $status"
printf '%s\n' "$output"
exit 1
