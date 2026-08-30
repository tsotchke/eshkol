#!/usr/bin/env bash
set -Eeuo pipefail

echo "llvm-config: $(llvm-config --version)"
echo "clang: $(clang-21 --version | head -n 1)"
echo "lld: $(ld.lld --version | head -n 1)"
echo "gcc: $(g++ --version | head -n 1)"
echo "cmake: $(cmake --version | head -n 1)"
echo "ninja: $(ninja --version)"
echo "python: $(python3 --version)"
echo "numpy: $(python3 -c 'import numpy; print(numpy.__version__)')"
echo "jq: $(jq --version)"
echo "emcc: $(emcc --version | head -n 1)"
echo "node: $(node --version)"

if command -v ssh >/dev/null 2>&1; then
    echo "error: an SSH client is present in the CI image" >&2
    exit 1
fi

[[ "${ESHKOL_RUNNER_CONTAINER:-}" == 1 ]] || {
    echo "error: ESHKOL_RUNNER_CONTAINER is not set" >&2
    exit 1
}
[[ "${ESHKOL_RUNNER_EPHEMERAL:-}" == 1 ]] || {
    echo "error: ESHKOL_RUNNER_EPHEMERAL is not set" >&2
    exit 1
}
echo "container marker: ESHKOL_RUNNER_CONTAINER=1"
echo "ephemeral marker: ESHKOL_RUNNER_EPHEMERAL=1"
