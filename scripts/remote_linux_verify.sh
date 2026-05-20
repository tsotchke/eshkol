#!/usr/bin/env bash
# Run the standard Eshkol Linux verification flow on a remote SSH host.

set -euo pipefail

HOST=""
REPO_DIR="${REPO_DIR:-/home/tyr/projects/eshkol}"
BUILD_DIR="${BUILD_DIR:-build}"
LLVM_VERSION="${LLVM_VERSION:-21}"
USER_DEPS_DIR="${USER_DEPS_DIR:-}"
BOOTSTRAP=0
INSTALL_DEPS=0
USER_DEPS=0
RUN_STRESS=0
STRESS_REPEAT=1

usage() {
    cat <<'USAGE'
Usage: scripts/remote_linux_verify.sh HOST [options]

Options:
  --repo DIR            Remote repo path (default: /home/tyr/projects/eshkol)
  --build-dir DIR       Remote build directory (default: build)
  --llvm-version N      LLVM major version for --user-deps env (default: 21)
  --bootstrap           Configure/build using bootstrap_linux_build_host.sh first
  --install-deps        Let bootstrap install apt dependencies; requires sudo -n
  --user-deps           Let bootstrap extract LLVM/Clang/Ninja under ~/.local
  --user-deps-dir DIR   Remote user dependency prefix (default: ~/.local/eshkol-toolchain)
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
        --llvm-version)
            LLVM_VERSION="$2"
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
        --user-deps)
            BOOTSTRAP=1
            USER_DEPS=1
            shift
            ;;
        --user-deps-dir)
            BOOTSTRAP=1
            USER_DEPS=1
            USER_DEPS_DIR="$2"
            shift 2
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
user_env=""
if [ "$USER_DEPS" -eq 1 ]; then
    user_deps_dir_remote="\$HOME/.local/eshkol-toolchain"
    if [ -n "$USER_DEPS_DIR" ]; then
        user_deps_dir_remote="$(quote_remote "$USER_DEPS_DIR")"
    fi
    user_env="TOOL=$user_deps_dir_remote && export PATH=\"\$TOOL/usr/lib/llvm-${LLVM_VERSION}/bin:\$TOOL/usr/bin:\$PATH\" && export LD_LIBRARY_PATH=\"\$TOOL/usr/lib/llvm-${LLVM_VERSION}/lib:\$TOOL/usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH:-}\" && "
fi

run_remote "cd $repo_q && git status --short --branch && git rev-parse HEAD"

if [ "$BOOTSTRAP" -eq 1 ]; then
    dep_flag="--no-install-deps"
    if [ "$INSTALL_DEPS" -eq 1 ]; then
        dep_flag=""
    fi
    if [ "$USER_DEPS" -eq 1 ]; then
        dep_flag="--user-deps"
        if [ -n "$USER_DEPS_DIR" ]; then
            dep_flag="$dep_flag --user-deps-dir $(quote_remote "$USER_DEPS_DIR")"
        fi
    fi
    run_remote "cd $repo_q && BUILD_DIR=$build_q scripts/bootstrap_linux_build_host.sh $dep_flag --ctest"
else
    run_remote "cd $repo_q && ${user_env}cmake --build $build_q --parallel \$(getconf _NPROCESSORS_ONLN)"
    run_remote "cd $repo_q && ${user_env}ctest --test-dir $build_q --output-on-failure"
fi

run_remote "cd $repo_q/$build_q && ${user_env}ESHKOL_PATH=$repo_q ./eshkol-run -r ../tests/v1_2_edge_cases/parallel_hash_table_mutation_test.esk"

if [ "$RUN_STRESS" -eq 1 ]; then
    run_remote "cd $repo_q/$build_q && ${user_env}{ failed=0; for lap in \$(seq 1 $STRESS_REPEAT); do echo \"=== Stress lap \$lap ===\"; for h in ../tests/stress/stress_fd_exhaustion.esk ../tests/stress/stress_alloc_loop.esk ../tests/stress/stress_parallel_at_scale.esk; do printf '  %-50s ' \"\$(basename \"\$h\")\"; if ESHKOL_PATH=$repo_q ./eshkol-run -r \"\$h\" 2>/dev/null | tail -1 | grep -q 'RESULT: OK'; then echo PASS; else echo FAIL; failed=1; fi; done; done; exit \$failed; }"
fi

run_remote "cd $repo_q && git diff --check && git status --short --branch"
