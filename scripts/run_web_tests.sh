#!/bin/bash

# Eshkol Web/WASM Test Suite
# Tests WASM compilation and server functionality

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVER_BIN="$PROJECT_DIR/build/eshkol-server"
ESHKOL_RUN="$PROJECT_DIR/build/eshkol-run"
PORT=19876
SERVER_PID=""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Counters
PASS=0
FAIL=0

cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    rm -f /tmp/eshkol_web_test_*.tmp /tmp/eshkol_server_$$.log
}

trap cleanup EXIT

log_pass() {
    echo -e "${GREEN}✅ PASS${NC}: $1"
    ((PASS++)) || true
}

log_fail() {
    echo -e "${RED}❌ FAIL${NC}: $1"
    ((FAIL++)) || true
}

log_info() {
    echo -e "${YELLOW}INFO${NC}: $1"
}

echo "========================================="
echo "  Eshkol Web/WASM Test Suite"
echo "========================================="
echo ""

cd "$PROJECT_DIR"

# Check for required binaries
if [ ! -f "$ESHKOL_RUN" ]; then
    echo -e "${RED}Error: eshkol-run not found. Run cmake --build build first.${NC}"
    exit 1
fi

if [ ! -f "$SERVER_BIN" ]; then
    echo -e "${RED}Error: eshkol-server not found. Run cmake --build build first.${NC}"
    exit 1
fi

# ============================================
# Part 1: WASM Compilation Tests (.esk files)
# ============================================
echo "--- Part 1: WASM Compilation Tests ---"
echo ""

for test_file in "$PROJECT_DIR"/tests/web/*.esk; do
    if [ ! -f "$test_file" ]; then
        continue
    fi

    test_name=$(basename "$test_file")
    printf "Testing %-45s " "$test_name"

    # Compile to WASM
    output_wasm="/tmp/eshkol_web_test_$$.wasm"
    if "$ESHKOL_RUN" "$test_file" --wasm -o "$output_wasm" 2>/dev/null; then
        # Check if WASM file was created and is valid
        if [ -f "$output_wasm" ] && file "$output_wasm" | grep -q "WebAssembly"; then
            log_pass "WASM compilation"
        else
            log_fail "WASM file invalid"
        fi
        rm -f "$output_wasm"
    else
        log_fail "compilation failed"
    fi
done

echo ""

# ============================================
# Part 2: Server Tests
# ============================================
echo "--- Part 2: Server Tests ---"
echo ""

# Start server
log_info "Starting eshkol-server on port $PORT..."
"$SERVER_BIN" --port "$PORT" > /tmp/eshkol_server_$$.log 2>&1 &
SERVER_PID=$!
sleep 2

# Check if server started
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    log_fail "Server failed to start"
    cat /tmp/eshkol_server_$$.log 2>/dev/null || true
    exit 1
fi

log_info "Server running (PID: $SERVER_PID)"
echo ""

# Test: Health Check
printf "Testing %-45s " "health endpoint"
HEALTH=$(curl -s "http://localhost:$PORT/health" 2>/dev/null || echo "")
if echo "$HEALTH" | grep -q '"status":"ok"'; then
    log_pass "health check"
else
    log_fail "health check: $HEALTH"
fi

# Test: Compile simple function
printf "Testing %-45s " "compile simple function"
RESULT=$(curl -s -X POST "http://localhost:$PORT/compile" \
    -H "Content-Type: application/json" \
    -d '{"code":"(define (square x) (* x x))","session_id":"test1"}' 2>/dev/null || echo "")
if echo "$RESULT" | grep -q '"success":true'; then
    log_pass "simple compile"
else
    log_fail "simple compile: $RESULT"
fi

# Test: Compile with web externals
printf "Testing %-45s " "compile with web externals"
RESULT=$(curl -s -X POST "http://localhost:$PORT/compile" \
    -H "Content-Type: application/json" \
    -d '{"code":"(extern i32 web-get-body :real web_get_body)\n(define (test) (web-get-body))","session_id":"test2"}' 2>/dev/null || echo "")
if echo "$RESULT" | grep -q '"success":true'; then
    log_pass "web externals"
else
    log_fail "web externals: $RESULT"
fi

# Test: Compile math functions
printf "Testing %-45s " "compile math functions"
RESULT=$(curl -s -X POST "http://localhost:$PORT/compile" \
    -H "Content-Type: application/json" \
    -d '{"code":"(define (circle-area r) (* 3.14159 (* r r)))","session_id":"test3"}' 2>/dev/null || echo "")
if echo "$RESULT" | grep -q '"success":true'; then
    log_pass "math functions"
else
    log_fail "math functions: $RESULT"
fi

# Test: Invalid code error handling
printf "Testing %-45s " "error handling"
RESULT=$(curl -s -X POST "http://localhost:$PORT/compile" \
    -H "Content-Type: application/json" \
    -d '{"code":"(define incomplete","session_id":"test4"}' 2>/dev/null || echo "")
if echo "$RESULT" | grep -q '"success":false'; then
    log_pass "error handling"
else
    log_fail "expected error for invalid code"
fi

# Test: Static file serving - index.html
printf "Testing %-45s " "static: index.html"
RESULT=$(curl -s "http://localhost:$PORT/" 2>/dev/null || echo "")
if echo "$RESULT" | grep -q "Eshkol REPL"; then
    log_pass "index.html"
else
    log_fail "index.html not served"
fi

# Test: Static file serving - style.css
printf "Testing %-45s " "static: style.css"
RESULT=$(curl -s "http://localhost:$PORT/style.css" 2>/dev/null || echo "")
if echo "$RESULT" | grep -q "Eshkol REPL Styles"; then
    log_pass "style.css"
else
    log_fail "style.css not served"
fi

# Test: Static file serving - eshkol-repl.js
printf "Testing %-45s " "static: eshkol-repl.js"
RESULT=$(curl -s "http://localhost:$PORT/eshkol-repl.js" 2>/dev/null || echo "")
if echo "$RESULT" | grep -q "class EshkolRepl"; then
    log_pass "eshkol-repl.js"
else
    log_fail "eshkol-repl.js not served"
fi

# Test: WASM binary in response
printf "Testing %-45s " "WASM binary response"
RESULT=$(curl -s -X POST "http://localhost:$PORT/compile" \
    -H "Content-Type: application/json" \
    -d '{"code":"(define x 42)","session_id":"test5"}' 2>/dev/null || echo "")
if echo "$RESULT" | grep -q '"wasm":"AGFzbQ'; then
    log_pass "WASM binary (base64)"
else
    log_fail "no valid WASM in response"
fi

# Test: Multiple definitions
printf "Testing %-45s " "multiple definitions"
RESULT=$(curl -s -X POST "http://localhost:$PORT/compile" \
    -H "Content-Type: application/json" \
    -d '{"code":"(define a 1)\n(define b 2)\n(define (add x y) (+ x y))","session_id":"test6"}' 2>/dev/null || echo "")
if echo "$RESULT" | grep -q '"success":true'; then
    log_pass "multiple definitions"
else
    log_fail "multiple definitions: $RESULT"
fi

# Test: 404 handling
printf "Testing %-45s " "404 handling"
RESULT=$(curl -s "http://localhost:$PORT/nonexistent" 2>/dev/null || echo "")
if echo "$RESULT" | grep -qi "not found\|404"; then
    log_pass "404 handling"
else
    log_fail "404 not handled correctly"
fi

# Test: Session ID returned
printf "Testing %-45s " "session ID in response"
RESULT=$(curl -s -X POST "http://localhost:$PORT/compile" \
    -H "Content-Type: application/json" \
    -d '{"code":"(define y 0)","session_id":"mysession"}' 2>/dev/null || echo "")
if echo "$RESULT" | grep -q '"session_id":"mysession"'; then
    log_pass "session ID"
else
    log_fail "session ID not returned"
fi

echo ""

# ============================================
# Summary
# ============================================
echo "========================================="
echo "  Test Results Summary"
echo "========================================="
TOTAL=$((PASS + FAIL))
echo -e "Total Tests:    $TOTAL"
echo -e "${GREEN}Passed:         $PASS${NC}"
echo -e "${RED}Failed:         $FAIL${NC}"

if [ $TOTAL -gt 0 ]; then
    PASS_RATE=$((PASS * 100 / TOTAL))
    echo "Pass Rate:      ${PASS_RATE}%"
fi

echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed${NC}"
    exit 1
fi
