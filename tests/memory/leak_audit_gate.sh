#!/usr/bin/env bash
#
# Leak-audit gate (ADR-0010 gap A12, second half).
#
# scripts/check_leak_detection_selftest.sh proves the DETECTOR is armed. This
# script proves the PRODUCT is clean under it, and keeps it that way. It runs
# real workloads -- an AOT compile, the compiled program, the VM, and the REPL
# -- under ASan+LSan with the checked-in suppression file, and fails on any
# leak that the suppression file does not already name and justify.
#
# Two sections, because two different things can regress:
#
#   SECTION A -- no unsuppressed leak.
#     Any allocation site outside .icc/lsan-suppressions.txt that is still
#     unreachable at exit fails the gate. This is the part that catches a
#     newly-introduced leak in the runtime, the VM, the arena, the JIT, the
#     agent FFI or the driver.
#
#   SECTION B -- the suppressed front end must not grow FASTER.
#     The suppression rules are scoped to one function each, but within a
#     function they are total: a new leak inside parse_list() would be
#     swallowed by section A. Section B closes that by measuring, with
#     suppressions OFF, how many bytes the compiler front end retains per REPL
#     input line at two horizons, and gating on the SLOPE. The front end
#     currently retains a measured, exactly-linear 1628 bytes per line (epic
#     #182: eshkol_ast_t has no destructor). That number is pinned here: it may
#     go DOWN freely, and going up fails.
#
# ---------------------------------------------------------------------------
# IF YOU CHANGE THIS GATE: the red-proof must itself be proven
# ---------------------------------------------------------------------------
# "A gate that cannot fail is not a gate" is only half of it. The DEMONSTRATION
# that a gate can fail is itself a piece of verification, and it can be vacuous
# in exactly the same way the gate can. Two real instances from writing this
# file, both of which produced a confident green that meant nothing:
#
#   1. The first red-proof PASSED when it should have failed. The probe was
#      `void* p = malloc(4321); (void)p;` -- and at -O2 clang simply DELETED
#      the allocation, because its result never escapes. Allocation removal is
#      a legal optimisation. A leak probe must make the pointer genuinely
#      escape (store it somewhere the optimiser cannot see through, or pass it
#      to a function it cannot inline away) or you are proving nothing about a
#      release-configuration build. Verify the probe leaks by observing the
#      report BEFORE trusting that the gate caught it.
#
#   2. The first VM check invoked `eshkol-run --vm <file>`. There is no --vm
#      flag. The run printed a usage message, exited 1, emitted no LeakSanitizer
#      output -- and the gate read that as a clean VM run. It had never
#      executed the VM at all. The corrected check drives the real
#      `--profile hosted-vm --emit-eskb` route plus the standalone VM, and that
#      corrected check is what found the add_local leak the hollow one could
#      not see.
#
# Both reduce to one rule, and it generalises well beyond leaks: a check that
# reports success because its subject never ran is indistinguishable from a
# check that reports success because its subject is correct. Assert that the
# work happened -- exit codes, expected artifacts, expected output -- and not
# merely that no complaint was printed. check_workload() below does this
# deliberately; do not weaken it.
#
# ---------------------------------------------------------------------------
# REPRODUCING THIS LOCALLY -- the exact configure line
# ---------------------------------------------------------------------------
# Requires a build whose ASan runtime actually supports LeakSanitizer AT
# RUNTIME. On macOS that means Homebrew llvm@N and never Apple's system clang:
# Apple's clang compiles -fsanitize=address happily and its ASan runtime then
# refuses detect_leaks outright, which is a silent-pass failure mode
# indistinguishable from "no leaks found".
#
#   cmake -S . -B build-asan -G Ninja \
#       -DCMAKE_BUILD_TYPE=RelWithDebInfo \
#       -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm@21/bin/clang \
#       -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@21/bin/clang++ \
#       -DESHKOL_REQUIRED_LLVM_MAJOR=21 \
#       -DLLVM_CONFIG_EXECUTABLE=/opt/homebrew/opt/llvm@21/bin/llvm-config \
#       -DESHKOL_BUILD_TESTS=ON -DESHKOL_BUILD_AGENT_FFI=ON \
#       -DESHKOL_ENABLE_ASAN=ON -DESHKOL_ENABLE_UBSAN=ON
#   cmake --build build-asan --parallel
#   BUILD_DIR=build-asan ./tests/memory/leak_audit_gate.sh
#
# If the agent-FFI dependency fetches (tree-sitter and its grammars) are the
# blocker, point each one at a pre-staged source tree instead of disabling the
# whole surface -- CMake honours the override natively and performs no
# download:  -DFETCHCONTENT_SOURCE_DIR_<UPPERCASE_DEP_NAME>=/path/to/<dep>-src
#
# THE TRAP THAT MAKES THIS GATE LIE IF YOU GET IT WRONG
#
# `-r` and the AOT path link the USER's program with the host C++ driver,
# resolved from $ESHKOL_CXX_COMPILER. It must be the SAME toolchain the
# compiler itself was built with. Point it at a different clang++ (Apple's, by
# default, on a Mac) and the ASan runtime symbols do not resolve --
#
#     Undefined symbols for architecture arm64:
#       "___asan_version_mismatch_check_v8", referenced from: ...
#
# -- so every `-r`/AOT workload dies AT LINK, before running a single line of
# Eshkol. A leak gate that only greps for LeakSanitizer output then sees
# silence and reports success, having executed nothing. This script auto-
# detects a matching clang++ and asserts each workload's exit code for exactly
# that reason, but if you set ESHKOL_CXX_COMPILER by hand, set it to the
# clang++ named in the configure line above.
#
# Usage
#   BUILD_DIR=build-asan ./tests/memory/leak_audit_gate.sh
#   BUILD_DIR=build-asan ./tests/memory/leak_audit_gate.sh --no-trace
#
# Copyright (C) tsotchke
# SPDX-License-Identifier: MIT

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

BUILD_DIR="${BUILD_DIR:-build-asan}"
SUPPRESSIONS="$REPO_ROOT/.icc/lsan-suppressions.txt"
TRACE_DIR="${TRACE_DIR:-$REPO_ROOT/scripts/icc_traces}"
TRACE_FILE="$TRACE_DIR/leak_audit_gate.jsonl"
PROBE_ID="leak_audit_gate"

# Pinned front-end retention, bytes per REPL input line. Measured at 10/20/40
# lines on this base: 21171 / 37451 / 70009 bytes, i.e. 1628 B/line over both
# intervals -- linear across a 4x span.
#
# This number is expected to MOVE when the front end legitimately changes
# shape, and it did once already: the audit first measured 20392 / 35852 /
# 66772, i.e. 1546.0 B/line, and rebasing onto master raised it to 1628. The
# window contained #476 (the NodeId identity substrate), which attaches
# identity data to every parsed node -- more retained per form is the expected
# consequence, not a regression. Recording both numbers so the increase is
# accounted for rather than silently absorbed by the tolerance. Re-pin, with
# the reason, whenever a deliberate front-end change moves it; do not widen
# the tolerance instead.
#
# TOLERANCE is generous on purpose: this gate must catch a real regression (a
# new per-line retention is at minimum tens of bytes and usually far more),
# not flap on a one-node change in how a define is spelled.
PINNED_BYTES_PER_LINE="${PINNED_BYTES_PER_LINE:-1628}"
SLOPE_TOLERANCE_PCT="${SLOPE_TOLERANCE_PCT:-25}"

NO_TRACE=0
for arg in "$@"; do
    case "$arg" in
        --no-trace) NO_TRACE=1 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

if [ ! -x "$BUILD_DIR/eshkol-run" ]; then
    echo "FAIL: $BUILD_DIR/eshkol-run not found. Build an ASan configuration first:" >&2
    echo "  cmake -S . -B $BUILD_DIR -G Ninja -DESHKOL_ENABLE_ASAN=ON -DESHKOL_ENABLE_UBSAN=ON \\" >&2
    echo "        -DCMAKE_C_COMPILER=<llvm>/bin/clang -DCMAKE_CXX_COMPILER=<llvm>/bin/clang++" >&2
    exit 1
fi
if [ ! -f "$SUPPRESSIONS" ]; then
    echo "FAIL: $SUPPRESSIONS not found -- the gate fails closed rather than" >&2
    echo "      running with no suppressions (which would be red on everything)" >&2
    echo "      or with an LSAN_OPTIONS parse error (which would be green on" >&2
    echo "      everything)." >&2
    exit 1
fi

SCRATCH="$REPO_ROOT/.leak-audit-gate-$$"
mkdir -p "$SCRATCH"
cleanup() { rm -rf "$SCRATCH"; }
trap cleanup EXIT

# The `-r` and AOT paths link the user program with the host C++ driver. It
# must be the SAME toolchain the compiler was built with, or the ASan runtime
# symbols do not resolve and every workload dies at link -- which this gate
# would otherwise happily read as "no leaks reported".
if [ -z "${ESHKOL_CXX_COMPILER:-}" ]; then
    for prefix in /opt/homebrew/opt/llvm@21 /opt/homebrew/opt/llvm \
                  /usr/local/opt/llvm@21 /usr/lib/llvm-21; do
        if [ -x "$prefix/bin/clang++" ]; then
            export ESHKOL_CXX_COMPILER="$prefix/bin/clang++"
            break
        fi
    done
fi
[ -x "/opt/homebrew/opt/llvm@21/bin/llvm-symbolizer" ] && \
    export ASAN_SYMBOLIZER_PATH="${ASAN_SYMBOLIZER_PATH:-/opt/homebrew/opt/llvm@21/bin/llvm-symbolizer}"

# macOS-only process-init allocator noise (objc / CoreFoundation / CFNetwork /
# libxpc), NOT shipped in .icc/lsan-suppressions.txt. That file describes the
# LINUX ASan lane, where these frames do not exist; putting them there would
# read to a future maintainer as a Linux-side admission that they are not.
# Verified independently attributable to the system frameworks: they appear
# with identical stacks for a probe containing no Eshkol code at all.
EFFECTIVE_SUPPRESSIONS="$SUPPRESSIONS"
if [ "$(uname -s)" = "Darwin" ]; then
    cat > "$SCRATCH/darwin-noise.txt" <<'EOF'
# One-time process-init allocations made by the macOS system frameworks
# themselves, before and around main(). Reproduced with identical stacks by a
# probe containing no Eshkol code, so they are attributable to the platform.
leak:libobjc.A.dylib
leak:CoreFoundation
leak:CFNetwork
leak:Foundation
leak:libxpc.dylib
leak:libsystem_c.dylib
leak:libsystem_info.dylib
leak:Network
leak:libdyld.dylib
# Darwin's thread-local-variable machinery. __tls_init is the TLV initializer
# dyld runs the first time a `thread_local` is touched on a thread; it
# malloc()s that thread's storage once and hands it back on every later
# access. The REPL ends via std::_Exit() (it must not run static/TLS
# destructors while JIT worker threads may hold libsystem locks), so the main
# thread's TLV block is still allocated at the explicit leak check in
# repl_clean_exit(). One object per thread_local per thread, never per input
# line. The Linux ASan lane does not have this symbol -- glibc allocates TLS
# from the thread's static block -- which is why this stays out of
# .icc/lsan-suppressions.txt.
leak:__tls_init
# The C++ ABI/exception runtime and CoreGraphics allocate one-time per-process
# state of their own (libc++abi's exception-handling globals; CoreGraphics'
# lazy display init pulled in through the platform's text/terminal path).
# Neither is reached from Eshkol code that could free it.
leak:libc++abi.dylib
leak:libc++.1.dylib
leak:CoreGraphics
EOF
    EFFECTIVE_SUPPRESSIONS="$SCRATCH/merged-suppressions.txt"
    cat "$SUPPRESSIONS" "$SCRATCH/darwin-noise.txt" > "$EFFECTIVE_SUPPRESSIONS"
fi

emit_event() {
    local status="$1" snippet="$2"
    [ "$NO_TRACE" -eq 1 ] && return
    mkdir -p "$TRACE_DIR"
    python3 -c '
import json, sys
print(json.dumps({"kind": "eshkol_smoke", "name": sys.argv[1],
                  "value": sys.argv[2], "snippet": sys.argv[3],
                  "confidence": 1.0}, ensure_ascii=False))
' "$PROBE_ID" "$status" "$snippet" >> "$TRACE_FILE"
}

failures=0
reasons=""

fail() {
    failures=$((failures + 1))
    reasons="$reasons; $1"
    echo "  FAIL: $1" >&2
}

# ===========================================================================
# SECTION A -- no unsuppressed leak in any real workload
# ===========================================================================
echo "=== section A: no unsuppressed leak ==="

cat > "$SCRATCH/hello.esk" <<'EOF'
(display "leak-audit-gate")
(newline)
EOF

export ASAN_OPTIONS="detect_leaks=1:halt_on_error=1:allocator_may_return_null=1:report_objects=0"
export LSAN_OPTIONS="suppressions=$EFFECTIVE_SUPPRESSIONS:print_suppressions=0:report_objects=0"

check_workload() {
    local name="$1"; shift
    local out="$SCRATCH/$name.log"
    "$@" > "$out" 2>&1
    local rc=$?
    local bad=0

    if grep -q "ERROR: LeakSanitizer" "$out"; then
        local summary
        summary="$(grep -h 'SUMMARY: AddressSanitizer' "$out" | head -1)"
        fail "$name reported an unsuppressed leak -- $summary"
        echo "  --- first unsuppressed stack in $name ---" >&2
        sed -n '/Direct leak/,/^$/p' "$out" | head -14 >&2
        bad=1
    fi

    # A workload that died did not prove anything about leaks: a process that
    # aborts partway through simply never reaches the exit-time leak check, and
    # its silence reads identically to a clean run. This caught a real
    # regression while this gate was being written — an over-eager jit_.reset()
    # in ~ReplJITContext made `-e` abort with "recursive_mutex lock failed"
    # AFTER printing the right answer, and the leak half of this check happily
    # called it "ok". Signals (rc >= 128) are always a failure; a nonzero
    # ordinary exit is a failure unless the caller declared one is expected.
    local expected_rc="${EXPECT_RC:-0}"
    if [ "$rc" -ge 128 ]; then
        fail "$name died on signal $((rc - 128)) (exit $rc) -- a crashed workload never reaches the leak check, so its silence is not evidence of anything. Last output: $(tail -1 "$out")"
        bad=1
    elif [ "$rc" -ne "$expected_rc" ]; then
        fail "$name exited $rc, expected $expected_rc -- if the workload did not actually run, it cannot have been checked for leaks. Last output: $(tail -1 "$out")"
        bad=1
    fi

    [ "$bad" -eq 0 ] && echo "  ok: $name (exit $rc, no unsuppressed leak)"
    return "$bad"
}

check_workload aot-compile "$BUILD_DIR/eshkol-run" "$SCRATCH/hello.esk" -o "$SCRATCH/hello.bin"
if [ -x "$SCRATCH/hello.bin" ]; then
    check_workload aot-run "$SCRATCH/hello.bin"
else
    fail "aot-compile produced no binary, so aot-run could not be checked (a gate that silently skips its subject is not a gate)"
fi
# The VM route is `--profile hosted-vm --emit-eskb` to produce bytecode, then
# the standalone VM to execute it — NOT a `--vm` flag on eshkol-run, which does
# not exist. A gate that invokes a nonexistent flag gets a usage message, exit
# 1 and no leak report, and reads exactly like a clean VM run; check that the
# command really did what it claims before trusting its silence.
check_workload vm-emit-eskb "$BUILD_DIR/eshkol-run" --profile hosted-vm \
    --emit-eskb "$SCRATCH/hello.eskb" "$SCRATCH/hello.esk"
if [ -f "$SCRATCH/hello.eskb" ] && [ -x "$BUILD_DIR/eshkol-vm-standalone-test" ]; then
    check_workload vm-execute "$BUILD_DIR/eshkol-vm-standalone-test" "$SCRATCH/hello.eskb"
else
    fail "the VM route produced no .eskb (or eshkol-vm-standalone-test is not built), so the VM was never actually exercised under LSan"
fi
# The in-process JIT routes. `-e` is where this audit found the two growing
# JIT leaks (the dropped execute() result and the per-variable storage slots),
# and both scaled with the number of top-level forms — so evaluate several,
# not one, or a per-form leak hides inside the noise floor of a single form.
check_workload jit-eval "$BUILD_DIR/eshkol-run" -e \
    '(define a 1)(define b 2)(define (sq x) (* x x))(display (sq (+ a b)))'
check_workload jit-run "$BUILD_DIR/eshkol-run" -r "$SCRATCH/hello.esk"

if [ -x "$BUILD_DIR/eshkol-repl" ]; then
    check_workload repl bash -c \
        "printf '(define (g x) (* x x))\n(define k 4)\n(g k)\n' | '$BUILD_DIR/eshkol-repl'"
else
    echo "  skip: eshkol-repl not built in $BUILD_DIR"
fi

# ===========================================================================
# SECTION B -- the suppressed front end must not retain MORE per input line
# ===========================================================================
echo "=== section B: front-end per-line retention slope ==="

if [ ! -x "$BUILD_DIR/eshkol-repl" ]; then
    echo "  skip: eshkol-repl not built in $BUILD_DIR"
else
    # Suppressions OFF here on purpose: section B is measuring exactly the
    # retention section A is allowed to ignore.
    measure_repl_leak() {
        local lines="$1" i out
        out="$( { for ((i = 1; i <= lines; i++)); do
                      echo "(define (fn$i x) (+ x $i))"
                  done; } \
                | ASAN_OPTIONS="detect_leaks=1:halt_on_error=1:allocator_may_return_null=1:report_objects=0" \
                  LSAN_OPTIONS="print_suppressions=0:report_objects=0" \
                  "$BUILD_DIR/eshkol-repl" 2>&1 )"
        echo "$out" | grep -o 'SUMMARY: AddressSanitizer: [0-9]* byte' \
                    | grep -o '[0-9]*' | head -1
    }

    LOW_N=10
    HIGH_N=40
    low="$(measure_repl_leak "$LOW_N")"
    high="$(measure_repl_leak "$HIGH_N")"

    if [ -z "$low" ] || [ -z "$high" ]; then
        # No report at all means either the front end became leak-free (great,
        # and then this section retires with epic #182) or leak detection is
        # not actually running (catastrophic, and indistinguishable from the
        # good news if we just pass). Fail loudly and make a human look.
        fail "section B got no LeakSanitizer measurement from the REPL at ${LOW_N}/${HIGH_N} lines (low='$low' high='$high'). Either the front-end retention is gone -- in which case retire this section and its suppression rules -- or leak detection is not live in $BUILD_DIR, which would make section A meaningless too"
    else
        slope=$(( (high - low) / (HIGH_N - LOW_N) ))
        ceiling=$(( PINNED_BYTES_PER_LINE * (100 + SLOPE_TOLERANCE_PCT) / 100 ))
        echo "  ${LOW_N} lines: ${low} B; ${HIGH_N} lines: ${high} B; slope: ${slope} B/line (pinned ${PINNED_BYTES_PER_LINE}, ceiling ${ceiling})"
        if [ "$slope" -gt "$ceiling" ]; then
            fail "front-end retention grew to ${slope} bytes per REPL line, above the pinned ceiling of ${ceiling} (pinned ${PINNED_BYTES_PER_LINE} + ${SLOPE_TOLERANCE_PCT}%). Something now retains more per parsed form than when this gate was written"
        else
            echo "  ok: slope ${slope} B/line is at or below the pinned ceiling"
        fi
    fi
fi

echo
if [ "$failures" -eq 0 ]; then
    echo "$PROBE_ID: PASS -- no unsuppressed leak in any workload, front-end retention at or below its pinned per-line ceiling"
    emit_event "PASS" "sections A and B clean; suppressions=$(grep -c '^leak:' "$SUPPRESSIONS") rules"
    exit 0
else
    echo "$PROBE_ID: FAIL ($failures)$reasons" >&2
    emit_event "FAIL" "$failures failure(s):$reasons"
    exit 1
fi
