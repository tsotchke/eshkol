#!/usr/bin/env bash

set -euo pipefail

ESHKOL_REQUIRED_LLVM_MAJOR="${ESHKOL_REQUIRED_LLVM_MAJOR:-21}"

eshkol_find_llvm_config() {
    local candidates=()

    case "$(uname -s)" in
        Darwin)
            candidates=(
                "/opt/homebrew/opt/llvm@${ESHKOL_REQUIRED_LLVM_MAJOR}/bin/llvm-config"
                "/usr/local/opt/llvm@${ESHKOL_REQUIRED_LLVM_MAJOR}/bin/llvm-config"
                "llvm-config-${ESHKOL_REQUIRED_LLVM_MAJOR}"
                "llvm-config"
            )
            ;;
        Linux)
            candidates=(
                "/usr/lib/llvm-${ESHKOL_REQUIRED_LLVM_MAJOR}/bin/llvm-config"
                "/usr/local/lib/llvm-${ESHKOL_REQUIRED_LLVM_MAJOR}/bin/llvm-config"
                "/usr/bin/llvm-config-${ESHKOL_REQUIRED_LLVM_MAJOR}"
                "/usr/local/bin/llvm-config-${ESHKOL_REQUIRED_LLVM_MAJOR}"
                "llvm-config-${ESHKOL_REQUIRED_LLVM_MAJOR}"
                "llvm-config"
            )
            ;;
        MINGW*|MSYS*|CYGWIN*)
            candidates=(
                "/mingw64/bin/llvm-config"
                "llvm-config"
            )
            ;;
        *)
            candidates=("llvm-config-${ESHKOL_REQUIRED_LLVM_MAJOR}" "llvm-config")
            ;;
    esac

    local candidate=""
    for candidate in "${candidates[@]}"; do
        if [[ "${candidate}" == */* ]]; then
            if [[ -x "${candidate}" ]]; then
                echo "${candidate}"
                return 0
            fi
        elif command -v "${candidate}" >/dev/null 2>&1; then
            command -v "${candidate}"
            return 0
        fi
    done

    echo "LLVM ${ESHKOL_REQUIRED_LLVM_MAJOR} llvm-config not found in expected locations" >&2
    return 1
}

eshkol_activate_llvm_toolchain() {
    local llvm_config="${LLVM_CONFIG_EXECUTABLE:-}"
    if [[ -z "${llvm_config}" ]]; then
        llvm_config="$(eshkol_find_llvm_config)"
    fi

    local llvm_version=""
    llvm_version="$("${llvm_config}" --version)"
    if [[ "${llvm_version%%.*}" != "${ESHKOL_REQUIRED_LLVM_MAJOR}" ]]; then
        echo "Expected LLVM ${ESHKOL_REQUIRED_LLVM_MAJOR}, got ${llvm_version} from ${llvm_config}" >&2
        return 1
    fi

    export LLVM_CONFIG_EXECUTABLE="${llvm_config}"
    export ESHKOL_LLVM_VERSION="${llvm_version}"
    export ESHKOL_LLVM_ROOT="$(cd "$(dirname "${llvm_config}")/.." && pwd)"
    export PATH="${ESHKOL_LLVM_ROOT}/bin:${PATH}"
    export CPPFLAGS="-I${ESHKOL_LLVM_ROOT}/include ${CPPFLAGS:-}"
    export LDFLAGS="-L${ESHKOL_LLVM_ROOT}/lib ${LDFLAGS:-}"

    if [[ "$(uname -s)" == "Darwin" ]]; then
        export DYLD_FALLBACK_LIBRARY_PATH="${ESHKOL_LLVM_ROOT}/lib${DYLD_FALLBACK_LIBRARY_PATH:+:${DYLD_FALLBACK_LIBRARY_PATH}}"
    fi
}

eshkol_activate_llvm21() {
    ESHKOL_REQUIRED_LLVM_MAJOR=21 eshkol_activate_llvm_toolchain
}
