#!/usr/bin/env bash
# Run the standard Eshkol Linux verification flow on a remote SSH host.

set -euo pipefail

HOST=""
REPO_DIR="${REPO_DIR:-/home/tyr/projects/eshkol}"
BUILD_DIR="${BUILD_DIR:-build}"
BOOTSTRAP=0
INSTALL_DEPS=0
RUN_STRESS=0
STRESS_REPEAT=1

usage() {
    cat <<'USAGE'
Usage: scripts/remote_linux_verify.sh HOST [options]

Options:
  --repo DIR            Remote repo path (default: /home/tyr/projects/eshkol)
  --build-dir DIR       Remote build directory (default: build)
  --bootstrap           Configure/build using bootstrap_linux_build_host.sh first
  --install-deps        Let bootstrap install apt dependencies; requires sudo -n
  --stress-repeat N     Run scripts/run_stress_tests.sh after CTest
  -h, --help            Show this help
USAGE
}

quote_remote() {
    printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

run_remote() {
    local command="$1"
    ssh "$HOST" "$command"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --repo)
            REPO_DIR="$2"
            shift 2
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --bootstrap)
            BOOTSTRAP=1
            shift
            ;;
        --install-deps)
            BOOTSTRAP=1
            INSTALL_DEPS=1
            shift
            ;;
        --stress-repeat)
            RUN_STRESS=1
            STRESS_REPEAT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            if [ -n "$HOST" ]; then
                echo "host already set: $HOST" >&2
                exit 2
            fi
            HOST="$1"
            shift
            ;;
    esac
done

if [ -z "$HOST" ]; then
    usage >&2
    exit 2
fi

repo_q="$(quote_remote "$REPO_DIR")"
build_q="$(quote_remote "$BUILD_DIR")"

run_remote "cd $repo_q && git status --short --branch && git rev-parse HEAD"

if [ "$BOOTSTRAP" -eq 1 ]; then
    dep_flag="--no-install-deps"
    if [ "$INSTALL_DEPS" -eq 1 ]; then
        dep_flag=""
    fi
    run_remote "cd $repo_q && BUILD_DIR=$build_q scripts/bootstrap_linux_build_host.sh $dep_flag --ctest"
else
    run_remote "cd $repo_q && cmake --build $build_q --parallel \$(getconf _NPROCESSORS_ONLN)"
    run_remote "cd $repo_q && ctest --test-dir $build_q --output-on-failure"
fi

run_remote "cd $repo_q && $build_q/eshkol-run -r tests/v1_2_edge_cases/parallel_hash_table_mutation_test.esk"

if [ "$RUN_STRESS" -eq 1 ]; then
    run_remote "cd $repo_q && BUILD_DIR=$build_q scripts/run_stress_tests.sh --repeat $STRESS_REPEAT"
fi

run_remote "cd $repo_q && git diff --check && git status --short --branch"
