#!/usr/bin/env bash
#
# Copyright (C) tsotchke
#
# SPDX-License-Identifier: MIT
#
# shared_lib_export_abi_test.sh — give the `--shared-lib` export-ABI contract
# real pre-merge teeth.
#
# The contract itself is asserted by tests/toolchain/shared_lib_abi_test.sh,
# which is registered with CTest. That registration alone is not a gate: NO
# workflow lane and no mesh gate lane runs `ctest` at all, so a ctest-only test
# is checked exactly when a human remembers to run it locally. The suites that
# DO run everywhere are the shell suites, and this directory's runner globs
# `tests/v1_2_edge_cases/*.sh` — which is how the compile-error-fatality
# contract got its teeth, and it belongs to the same family: a driver/CLI
# contract that cannot be expressed as an .esk program because the test is
# *about* what the compiler emits.
#
# So this delegates rather than duplicating: exactly one implementation of the
# assertions, reachable from both the CTest registration and every gate lane
# that runs run_v1_2_edge_cases_tests.sh (CI's asan lane, run_all_tests.sh, and
# the per-platform mesh gate lanes).

set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="$ROOT/${BUILD_DIR:-build}"
RUN="$BUILD/eshkol-run"
CONTRACT="$ROOT/tests/toolchain/shared_lib_abi_test.sh"

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        # The contract test dlopen()s the library from a C harness and a ctypes
        # harness; the Windows x64 sret/by-pointer export shape is exercised by
        # its own platform lane, not from an MSYS shell.
        echo "SKIP: not applicable on Windows shells"
        exit 0
        ;;
esac

# Consistent with every sibling in this directory: no compiler built means
# there is nothing to assert against, not a failure of the contract.
if [ ! -x "$RUN" ]; then
    echo "SKIP: $RUN not built"
    exit 0
fi
if [ ! -f "$CONTRACT" ]; then
    echo "FAIL: shared_lib_export_abi_test: the contract test is missing at '$CONTRACT'"
    exit 1
fi

# The contract test owns the FAIL:/PASS: protocol this suite matches on, so its
# output and exit status pass through unchanged.
bash "$CONTRACT" "$RUN" "$BUILD" "$ROOT"
