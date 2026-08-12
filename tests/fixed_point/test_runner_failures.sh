#!/usr/bin/env bash
# Regression test: a benchmark build failure must make --bench fail.

set -u

if [ "$(basename "$0")" = "fake-cc" ]; then
    if [ "${1:-}" = "--version" ]; then
        echo "fake-cc runner regression test"
        exit 0
    fi

    output=""
    bench_source=0
    while [ "$#" -gt 0 ]; do
        case "$1" in
            *bench_dot_exact.c) bench_source=1 ;;
            -o)
                shift
                output="${1:-}"
                ;;
        esac
        shift
    done

    if [ "$bench_source" -eq 1 ]; then
        echo "intentional benchmark compile failure" >&2
        exit 42
    fi
    if [ -z "$output" ]; then
        echo "fake-cc: missing -o output" >&2
        exit 43
    fi
    # Portability: /usr/bin/true does not exist on every host (e.g. NixOS,
    # which keeps /usr/bin nearly empty). Emit a tiny POSIX-sh stub instead
    # of symlinking to an absolute path that may not exist.
    printf '#!/bin/sh\nexit 0\n' > "$output"
    chmod +x "$output"
    exit 0
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
ln -s "$HERE/test_runner_failures.sh" "$TMP/fake-cc"

set +e
output="$(CC="$TMP/fake-cc" "$HERE/run_fixed_point_tests.sh" --bench 2>&1)"
status=$?
set -e

if [ "$status" -eq 0 ]; then
    echo "FAIL: --bench returned success after the benchmark compiler failed"
    echo "$output"
    exit 1
fi
if ! printf '%s\n' "$output" | grep -q 'bench_dot_exact(build)'; then
    echo "FAIL: benchmark build failure was not included in the failure summary"
    echo "$output"
    exit 1
fi

echo "PASS: benchmark build failure propagates through --bench"
