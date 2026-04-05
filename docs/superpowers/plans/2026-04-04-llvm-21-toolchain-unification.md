# LLVM 21 Toolchain Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make LLVM 21 the explicit lite/native baseline across CMake, local scripts, Docker parity environments, Homebrew packaging, focused developer docs, and the native Windows smoke path.

**Architecture:** Move LLVM discovery and version enforcement into two small authorities: a dedicated CMake helper for lite/native configure-time gating and a shared shell helper for script-side environment setup. Keep bundled `STABLEHLO_ROOT` handling explicit as a separate branch, simplify backend compatibility logic around the LLVM 21 API surface, and validate the resulting contract on Linux and WSL2-driven native Windows smoke while preparing macOS and Docker paths for the same baseline.

**Tech Stack:** CMake 3.14+, Bash, Docker, Ninja, Homebrew, Visual Studio 2022, ClangCL, WSL2, PowerShell interop, LLVM 21

---

## File Structure

### New files

- `cmake/LLVMToolchain.cmake`
  Responsibility: centralize lite/native `llvm-config` discovery, version parsing, and the hard fail on non-21 toolchains.
- `scripts/lib/llvm21-env.sh`
  Responsibility: shared shell-side LLVM 21 discovery and environment export for macOS/Linux scripts.
- `scripts/test-llvm-toolchain-config.sh`
  Responsibility: one-off configure-time regression test proving lite/native builds reject fake LLVM 18 and accept fake LLVM 21.
- `tests/toolchain/fake-llvm-config-18.sh`
  Responsibility: deterministic fixture that reports LLVM 18 to CMake.
- `tests/toolchain/fake-llvm-config-21.sh`
  Responsibility: deterministic fixture that reports LLVM 21 to CMake.
- `scripts/smoke-windows-native.sh`
  Responsibility: drive a native Visual Studio 2022 + LLVM 21 configure/build/hello-world smoke from WSL2.
- `tests/toolchain/windows-smoke.esk`
  Responsibility: minimal Eshkol program compiled and executed by the Windows smoke wrapper.
- `docs/superpowers/reports/2026-04-04-llvm-21-validation.md`
  Responsibility: capture the final validation commands and outcomes for this sub-project.

### Modified files

- `CMakeLists.txt:32-209`
  Replace ad hoc lite/native LLVM discovery with the helper, preserve explicit bundled `STABLEHLO_ROOT` handling, and keep filtered flag-list usage.
- `lib/backend/llvm_codegen.cpp`
  Remove misleading `18+` assumptions and align target-triple, intrinsic, host-feature, and tail-call logic with the LLVM 21 baseline.
- `lib/backend/arithmetic_codegen.cpp`
- `lib/backend/autodiff_codegen.cpp`
- `lib/backend/complex_codegen.cpp`
- `lib/backend/tensor_codegen.cpp`
- `lib/repl/repl_jit.cpp`
  Simplify compatibility branches that no longer describe the supported baseline.
- `scripts/build-macos.sh:70-123`
  Source the shared LLVM 21 shell helper and stop hardcoding `llvm@17`.
- `scripts/run_cpp_type_tests.sh:23-84`
  Use the shared helper instead of inline path guesses and stale version fallbacks.
- `scripts/test-homebrew.sh:72-122`
  Validate `llvm@21` and reuse the shared helper for local formula simulation.
- `scripts/test-ci-locally.sh:54-320`
  Update inline Dockerfiles and macOS checks to LLVM 21, and keep this script aligned with the local toolchain contract.
- `packaging/homebrew/eshkol.rb:17-42`
  Pin the formula to `llvm@21` and reuse the matching Homebrew paths at build time.
- `docker/debian/debug/Dockerfile`
- `docker/debian/release/Dockerfile`
- `docker/ubuntu/release/Dockerfile`
- `docker/xla/Dockerfile`
- `docker/cuda/Dockerfile`
  Replace `17`-specific apt and package wiring with `21`-specific wiring.
- `README.md`
- `CONTRIBUTING.md`
- `docs/QUICKSTART.md`
- `docs/FAQ.md`
  Update the canonical user/developer install guidance to the LLVM 21 baseline.

## Task 1: Add the CMake LLVM 21 Gate and Its Regression Fixtures

**Files:**
- Create: `cmake/LLVMToolchain.cmake`
- Create: `scripts/test-llvm-toolchain-config.sh`
- Create: `tests/toolchain/fake-llvm-config-18.sh`
- Create: `tests/toolchain/fake-llvm-config-21.sh`
- Modify: `CMakeLists.txt:53-209`
- Test: `scripts/test-llvm-toolchain-config.sh`

- [ ] **Step 1: Write the failing toolchain-regression fixtures**

```bash
#!/usr/bin/env bash
# tests/toolchain/fake-llvm-config-18.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAKE_ROOT="${SCRIPT_DIR}/fake-llvm-root"

case "${1:-}" in
  --version) echo "18.1.8" ;;
  --cxxflags) echo "-I${FAKE_ROOT}/include -std=c++17 -fno-exceptions -DFAKE_LLVM=1" ;;
  --ldflags) echo "-L${FAKE_ROOT}/lib" ;;
  --libs) echo "-lLLVMCore -lLLVMSupport" ;;
  --system-libs) echo "-lpthread -ldl -lm" ;;
  --includedir) echo "${FAKE_ROOT}/include" ;;
  *) echo "unsupported arg: $*" >&2; exit 1 ;;
esac
```

```bash
#!/usr/bin/env bash
# tests/toolchain/fake-llvm-config-21.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAKE_ROOT="${SCRIPT_DIR}/fake-llvm-root"

case "${1:-}" in
  --version) echo "21.1.8" ;;
  --cxxflags) echo "-I${FAKE_ROOT}/include -std=c++20 -DFAKE_LLVM=1" ;;
  --ldflags) echo "-L${FAKE_ROOT}/lib" ;;
  --libs) echo "-lLLVMCore -lLLVMSupport" ;;
  --system-libs) echo "-lpthread -ldl -lm" ;;
  --includedir) echo "${FAKE_ROOT}/include" ;;
  *) echo "unsupported arg: $*" >&2; exit 1 ;;
esac
```

```bash
#!/usr/bin/env bash
# scripts/test-llvm-toolchain-config.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FIXTURE_ROOT="${PROJECT_ROOT}/tests/toolchain/fake-llvm-root"

mkdir -p "${FIXTURE_ROOT}/include/llvm/IR" "${FIXTURE_ROOT}/lib"
touch "${FIXTURE_ROOT}/include/llvm/IR/LLVMContext.h"

rm -rf "${PROJECT_ROOT}/build-toolchain-fake18" "${PROJECT_ROOT}/build-toolchain-fake21"

if cmake -S "${PROJECT_ROOT}" -B "${PROJECT_ROOT}/build-toolchain-fake18" -G Ninja \
    -DLLVM_CONFIG_EXECUTABLE="${PROJECT_ROOT}/tests/toolchain/fake-llvm-config-18.sh" \
    >/tmp/eshkol-cmake-fake18.log 2>&1; then
  echo "fake LLVM 18 configure unexpectedly succeeded" >&2
  exit 1
fi

cmake -S "${PROJECT_ROOT}" -B "${PROJECT_ROOT}/build-toolchain-fake21" -G Ninja \
  -DLLVM_CONFIG_EXECUTABLE="${PROJECT_ROOT}/tests/toolchain/fake-llvm-config-21.sh" \
  >/tmp/eshkol-cmake-fake21.log 2>&1
```

- [ ] **Step 2: Run the regression harness and confirm the current tree still accepts fake LLVM 18**

Run: `chmod +x tests/toolchain/fake-llvm-config-18.sh tests/toolchain/fake-llvm-config-21.sh scripts/test-llvm-toolchain-config.sh && ./scripts/test-llvm-toolchain-config.sh`

Expected: FAIL with `fake LLVM 18 configure unexpectedly succeeded`

- [ ] **Step 3: Implement the CMake helper and wire it into the lite/native path**

```cmake
# cmake/LLVMToolchain.cmake
set(ESHKOL_EXPECTED_LLVM_MAJOR 21)

function(eshkol_find_lite_llvm)
    set(_candidate_paths "")
    if(APPLE)
        list(APPEND _candidate_paths
            /opt/homebrew/opt/llvm@21/bin
            /usr/local/opt/llvm@21/bin
        )
    endif()

    if(NOT LLVM_CONFIG_EXECUTABLE)
        find_program(LLVM_CONFIG_EXECUTABLE
            NAMES llvm-config-21 llvm-config
            PATHS ${_candidate_paths}
        )
    endif()

    if(NOT LLVM_CONFIG_EXECUTABLE)
        message(FATAL_ERROR
            "LLVM 21 llvm-config not found. Install LLVM 21 and set LLVM_CONFIG_EXECUTABLE.")
    endif()

    execute_process(
        COMMAND "${LLVM_CONFIG_EXECUTABLE}" --version
        OUTPUT_VARIABLE ESHKOL_LLVM_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(REGEX MATCH "^([0-9]+)" _llvm_major_match "${ESHKOL_LLVM_VERSION}")

    if(NOT CMAKE_MATCH_1 STREQUAL "${ESHKOL_EXPECTED_LLVM_MAJOR}")
        message(FATAL_ERROR
            "Expected LLVM ${ESHKOL_EXPECTED_LLVM_MAJOR}, got ${ESHKOL_LLVM_VERSION} from ${LLVM_CONFIG_EXECUTABLE}")
    endif()

    set(LLVM_CONFIG_EXECUTABLE "${LLVM_CONFIG_EXECUTABLE}" PARENT_SCOPE)
    set(ESHKOL_LLVM_VERSION "${ESHKOL_LLVM_VERSION}" PARENT_SCOPE)
endfunction()
```

```cmake
# CMakeLists.txt
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/LLVMToolchain.cmake)
set(ESHKOL_VERSION "1.1.11")
project("Eshkol" VERSION ${ESHKOL_VERSION} LANGUAGES C CXX)

else()
    message(STATUS "=== LITE BUILD MODE: Using system LLVM ===")
    set(USING_STABLEHLO_LLVM FALSE)

    eshkol_find_lite_llvm()

    execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --cxxflags OUTPUT_VARIABLE LLVM_CXXFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --ldflags OUTPUT_VARIABLE LLVM_LDFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --libs all OUTPUT_VARIABLE LLVM_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --system-libs OUTPUT_VARIABLE LLVM_SYSTEM_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --includedir OUTPUT_VARIABLE LLVM_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(LLVM_INCLUDE_DIRS "${LLVM_INCLUDE_DIR}")
endif()
```

- [ ] **Step 4: Re-run the harness and a real configure against LLVM 21**

Run: `./scripts/test-llvm-toolchain-config.sh`

Expected: PASS with no output and both fake configure directories created

Run: `cmake -S . -B build-llvm21 -G Ninja -DLLVM_CONFIG_EXECUTABLE="$(command -v llvm-config-21)"`

Expected: PASS and CMake prints the discovered LLVM 21 executable/version

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt cmake/LLVMToolchain.cmake scripts/test-llvm-toolchain-config.sh tests/toolchain/fake-llvm-config-18.sh tests/toolchain/fake-llvm-config-21.sh
git commit -m "build: enforce LLVM 21 for lite toolchains"
```

## Task 2: Simplify Backend Compatibility Logic Around the LLVM 21 Baseline

**Files:**
- Modify: `lib/backend/llvm_codegen.cpp`
- Modify: `lib/backend/arithmetic_codegen.cpp`
- Modify: `lib/backend/autodiff_codegen.cpp`
- Modify: `lib/backend/complex_codegen.cpp`
- Modify: `lib/backend/tensor_codegen.cpp`
- Modify: `lib/repl/repl_jit.cpp`
- Test: `build-llvm21/eshkol-run`

- [ ] **Step 1: Reproduce the backend breakage under the real LLVM 21 toolchain**

Run: `cmake --build build-llvm21 --target eshkol-run --parallel 4`

Expected: FAIL in one or more backend files that still assume a broad `LLVM_VERSION_MAJOR >= 18` API split

- [ ] **Step 2: Replace vague `18+` logic with either shared code or explicit `21 vs legacy` branches**

```cpp
// lib/backend/arithmetic_codegen.cpp and similar helpers
#define ESHKOL_GET_INTRINSIC(mod, id, types) llvm::Intrinsic::getDeclaration(mod, id, types)
```

```cpp
// lib/backend/llvm_codegen.cpp
std::string target_triple_str = target_triple ? std::string(target_triple)
                                              : sys::getDefaultTargetTriple();
module->setTargetTriple(target_triple_str);

llvm::StringMap<bool> host_features;
sys::getHostCPUFeatures(host_features);

const Target* target = TargetRegistry::lookupTarget(target_triple_str, error);
std::unique_ptr<TargetMachine> target_machine(
    target->createTargetMachine(target_triple_str, g_cached_cpu_name, g_cached_features,
                                target_options, Reloc::PIC_, std::nullopt, codegen_opt));
```

```cpp
// lib/backend/llvm_codegen.cpp mutual tail-call handling
#if LLVM_VERSION_MAJOR >= 21
call->setTailCallKind(CallInst::TCK_MustTail);
#else
call->setTailCallKind(CallInst::TCK_Tail);
#endif
```

```cpp
// lib/repl/repl_jit.cpp
std::cerr << "[REPL] Module Triple: " << module->getTargetTriple() << std::endl;
```

- [ ] **Step 3: Rebuild and compile the stdlib with the LLVM 21 binaries**

Run: `cmake --build build-llvm21 --parallel 4`

Expected: PASS with `build-llvm21/eshkol-run`, `build-llvm21/eshkol-repl`, and `build-llvm21/stdlib.o` produced

Run: `./build-llvm21/eshkol-run --shared-lib -o /tmp/eshkol-stdlib-llvm21 lib/stdlib.esk`

Expected: PASS and `/tmp/eshkol-stdlib-llvm21.o` exists

- [ ] **Step 4: Commit**

```bash
git add lib/backend/llvm_codegen.cpp lib/backend/arithmetic_codegen.cpp lib/backend/autodiff_codegen.cpp lib/backend/complex_codegen.cpp lib/backend/tensor_codegen.cpp lib/repl/repl_jit.cpp
git commit -m "codegen: align lite builds with LLVM 21"
```

## Task 3: Share LLVM 21 Shell Discovery Across Local Scripts and Homebrew

**Files:**
- Create: `scripts/lib/llvm21-env.sh`
- Modify: `scripts/build-macos.sh:70-123`
- Modify: `scripts/run_cpp_type_tests.sh:23-84`
- Modify: `scripts/test-homebrew.sh:72-122`
- Modify: `packaging/homebrew/eshkol.rb:17-42`
- Test: `scripts/run_cpp_type_tests.sh`

- [ ] **Step 1: Prove the local-script layer still contains stale LLVM 17 assumptions**

Run: `rg -n "llvm@17|llvm-config-17|LLVM 17" scripts/build-macos.sh scripts/run_cpp_type_tests.sh scripts/test-homebrew.sh packaging/homebrew/eshkol.rb`

Expected: FAIL with matches in all four files

- [ ] **Step 2: Add the shared helper and update the scripts/formula to use it**

```bash
#!/usr/bin/env bash
# scripts/lib/llvm21-env.sh
set -euo pipefail

ESHKOL_EXPECTED_LLVM_MAJOR=21

eshkol_find_llvm_config() {
    local candidates=()
    case "$(uname -s)" in
        Darwin)
            candidates=(/opt/homebrew/opt/llvm@21/bin/llvm-config /usr/local/opt/llvm@21/bin/llvm-config)
            ;;
        Linux)
            candidates=(llvm-config-21 /usr/bin/llvm-config-21 /usr/local/bin/llvm-config-21 /usr/bin/llvm-config)
            ;;
        MINGW*|MSYS*|CYGWIN*)
            candidates=(/mingw64/bin/llvm-config llvm-config)
            ;;
    esac

    for candidate in "${candidates[@]}"; do
        if command -v "$candidate" >/dev/null 2>&1; then
            command -v "$candidate"
            return 0
        elif [ -x "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done

    return 1
}

eshkol_activate_llvm21() {
    local llvm_config
    llvm_config="${LLVM_CONFIG_EXECUTABLE:-$(eshkol_find_llvm_config)}"
    local llvm_version
    llvm_version="$("$llvm_config" --version)"

    if [[ "${llvm_version%%.*}" != "${ESHKOL_EXPECTED_LLVM_MAJOR}" ]]; then
        echo "Expected LLVM ${ESHKOL_EXPECTED_LLVM_MAJOR}, got ${llvm_version} from ${llvm_config}" >&2
        return 1
    fi

    export LLVM_CONFIG_EXECUTABLE="$llvm_config"
    export ESHKOL_LLVM_ROOT="$(cd "$(dirname "$llvm_config")/.." && pwd)"
    export PATH="${ESHKOL_LLVM_ROOT}/bin:${PATH}"
    export CPPFLAGS="-I${ESHKOL_LLVM_ROOT}/include ${CPPFLAGS:-}"
    export LDFLAGS="-L${ESHKOL_LLVM_ROOT}/lib ${LDFLAGS:-}"
}
```

```bash
# scripts/build-macos.sh
source "${SCRIPT_DIR}/lib/llvm21-env.sh"
eshkol_activate_llvm21
LLVM_PATH="${ESHKOL_LLVM_ROOT}"
```

```bash
# scripts/run_cpp_type_tests.sh
source "${SCRIPT_DIR}/lib/llvm21-env.sh"
eshkol_activate_llvm21
LLVM_CONFIG="${LLVM_CONFIG_EXECUTABLE}"
```

```ruby
# packaging/homebrew/eshkol.rb
depends_on "llvm@21"

def install
  llvm = Formula["llvm@21"]
  ENV["PATH"] = "#{llvm.opt_bin}:#{ENV["PATH"]}"
  ENV["LDFLAGS"] = "-L#{llvm.opt_lib} -Wl,-rpath,#{llvm.opt_lib} #{ENV["LDFLAGS"]}"
  ENV["CPPFLAGS"] = "-I#{llvm.opt_include} #{ENV["CPPFLAGS"]}"
  ENV["DYLD_FALLBACK_LIBRARY_PATH"] = llvm.opt_lib

  system "cmake", "-B", "build", "-G", "Ninja",
         "-DCMAKE_BUILD_TYPE=Release",
         "-DLLVM_DIR=#{llvm.opt_lib}/cmake/llvm",
         "-DCMAKE_INSTALL_RPATH=#{llvm.opt_lib}",
         "-DCMAKE_BUILD_RPATH=#{llvm.opt_lib}",
         "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
         "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON",
         "-DCMAKE_MACOSX_RPATH=ON",
         *std_cmake_args
end
```

- [ ] **Step 3: Re-run the string check and syntax-check the updated scripts**

Run: `rg -n "llvm@17|llvm-config-17|LLVM 17" scripts/build-macos.sh scripts/run_cpp_type_tests.sh scripts/test-homebrew.sh packaging/homebrew/eshkol.rb`

Expected: PASS with no matches

Run: `bash -n scripts/build-macos.sh scripts/run_cpp_type_tests.sh scripts/test-homebrew.sh scripts/lib/llvm21-env.sh && ruby -c packaging/homebrew/eshkol.rb`

Expected: PASS with `Syntax OK` from Ruby

- [ ] **Step 4: Run the C++ type-system smoke with the real LLVM 21 toolchain**

Run: `./scripts/run_cpp_type_tests.sh`

Expected: PASS with both `hott_types_test` and `type_checker_test` succeeding

- [ ] **Step 5: Commit**

```bash
git add scripts/lib/llvm21-env.sh scripts/build-macos.sh scripts/run_cpp_type_tests.sh scripts/test-homebrew.sh packaging/homebrew/eshkol.rb
git commit -m "scripts: standardize local toolchain setup on LLVM 21"
```

## Task 4: Add the WSL2-to-Native Windows Smoke Path

**Files:**
- Create: `scripts/smoke-windows-native.sh`
- Create: `tests/toolchain/windows-smoke.esk`
- Test: `scripts/smoke-windows-native.sh`

- [ ] **Step 1: Write the smoke input and confirm there is no Windows wrapper yet**

```scheme
; tests/toolchain/windows-smoke.esk
(display "hello from windows smoke")
```

Run: `./scripts/smoke-windows-native.sh`

Expected: FAIL with `No such file or directory`

- [ ] **Step 2: Implement the WSL2 wrapper that drives a native Visual Studio 2022 + ClangCL build**

```bash
#!/usr/bin/env bash
# scripts/smoke-windows-native.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
POWERSHELL_EXE="${POWERSHELL_EXE:-/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe}"

if [ ! -x "${POWERSHELL_EXE}" ]; then
    echo "PowerShell not found at ${POWERSHELL_EXE}" >&2
    exit 1
fi

PROJECT_ROOT_WIN="$(wslpath -w "${PROJECT_ROOT}")"
SMOKE_SRC_WIN="$(wslpath -w "${PROJECT_ROOT}/tests/toolchain/windows-smoke.esk")"

"${POWERSHELL_EXE}" -NoProfile -Command "
  Set-Location '${PROJECT_ROOT_WIN}';
  cmake -S . -B build-windows -G 'Visual Studio 17 2022' -A x64 -T ClangCL -DCMAKE_BUILD_TYPE=Release -DLLVM_DIR='C:/Program Files/LLVM/lib/cmake/llvm';
  cmake --build build-windows --config Release --parallel;
  & '.\\build-windows\\Release\\eshkol-run.exe' '${SMOKE_SRC_WIN}' -o windows-smoke.exe;
  & '.\\windows-smoke.exe'
"
```

- [ ] **Step 3: Run the Windows smoke**

Run: `chmod +x scripts/smoke-windows-native.sh && ./scripts/smoke-windows-native.sh`

Expected: PASS with the native Windows build completing and `hello from windows smoke` printed

- [ ] **Step 4: Commit**

```bash
git add scripts/smoke-windows-native.sh tests/toolchain/windows-smoke.esk
git commit -m "windows: add native LLVM 21 smoke from WSL2"
```

## Task 5: Update Docker Parity Images and the Local CI Harness

**Files:**
- Modify: `docker/debian/debug/Dockerfile`
- Modify: `docker/debian/release/Dockerfile`
- Modify: `docker/ubuntu/release/Dockerfile`
- Modify: `docker/xla/Dockerfile`
- Modify: `docker/cuda/Dockerfile`
- Modify: `scripts/test-ci-locally.sh:54-320`
- Test: `scripts/test-ci-locally.sh`

- [ ] **Step 1: Prove the Docker/local-CI layer still hardcodes LLVM 17**

Run: `rg -n "llvm-17|llvm-config-17|llvm-toolchain-.*-17|llvm@17" docker scripts/test-ci-locally.sh`

Expected: FAIL with matches in the Dockerfiles and the inline here-docs inside `scripts/test-ci-locally.sh`

- [ ] **Step 2: Switch the apt and Homebrew references to LLVM 21**

```dockerfile
# docker/debian/release/Dockerfile
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    software-properties-common \
    && wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc \
    && echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-21 main" >> /etc/apt/sources.list.d/llvm.list \
    && apt-get update

RUN apt-get install -y \
    cmake \
    ninja-build \
    g++ \
    llvm-21 \
    llvm-21-dev \
    libreadline-dev \
    libopenblas-dev \
    dpkg-dev \
    file \
    curl \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/llvm-config-21 /usr/bin/llvm-config
```

```dockerfile
# docker/xla/Dockerfile
RUN apt-get install -y \
    cmake \
    ninja-build \
    g++ \
    llvm-21 \
    llvm-21-dev \
    mlir-21-tools \
    libmlir-21-dev \
    libreadline-dev \
    libopenblas-dev \
    dpkg-dev \
    file \
    curl \
    git \
    pkg-config \
    python3 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/openxla/stablehlo.git \
    && cd stablehlo \
    && cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DMLIR_DIR=/usr/lib/llvm-21/lib/cmake/mlir \
        -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
    && cmake --build build --parallel
```

```bash
# scripts/test-ci-locally.sh
for dep in llvm@21 cmake ninja readline; do
    if brew list "$dep" &>/dev/null; then
        echo -e "  ${GREEN}[OK]${NC} $dep"
    else
        echo -e "  ${RED}[MISSING]${NC} $dep"
        DEPS_OK=false
    fi
done

export PATH="/opt/homebrew/opt/llvm@21/bin:$PATH"

echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-21 main" >> /etc/apt/sources.list.d/llvm.list
RUN apt-get install -y cmake ninja-build llvm-21 llvm-21-dev libreadline-dev dpkg-dev g++ file pkg-config
RUN ln -sf /usr/bin/llvm-config-21 /usr/bin/llvm-config
```

- [ ] **Step 3: Re-run the string check and build the parity images**

Run: `rg -n "llvm-17|llvm-config-17|llvm-toolchain-.*-17|llvm@17" docker scripts/test-ci-locally.sh`

Expected: PASS with no matches

Run: `docker build -f docker/debian/release/Dockerfile -t eshkol-debian-release-llvm21 .`

Expected: PASS

Run: `docker build -f docker/debian/debug/Dockerfile -t eshkol-debian-debug-llvm21 .`

Expected: PASS

Run: `docker build -f docker/xla/Dockerfile -t eshkol-xla-llvm21 .`

Expected: PASS, or a clear StableHLO/MLIR incompatibility that is documented as the remaining explicit exception

- [ ] **Step 4: Run the local Linux/release CI harnesses**

Run: `./scripts/test-ci-locally.sh linux`

Expected: PASS or fail only on known compiler/runtime tests, not on missing `llvm-config-17` or stale package names

Run: `ESHKOL_VERSION=1.1.11 ./scripts/test-ci-locally.sh release`

Expected: PASS or fail only after packaging/test execution, not during toolchain provisioning

- [ ] **Step 5: Commit**

```bash
git add docker/debian/debug/Dockerfile docker/debian/release/Dockerfile docker/ubuntu/release/Dockerfile docker/xla/Dockerfile docker/cuda/Dockerfile scripts/test-ci-locally.sh
git commit -m "docker: pin local parity environments to LLVM 21"
```

## Task 6: Update Canonical Docs and Record the Validation Results

**Files:**
- Modify: `README.md`
- Modify: `CONTRIBUTING.md`
- Modify: `docs/QUICKSTART.md`
- Modify: `docs/FAQ.md`
- Create: `docs/superpowers/reports/2026-04-04-llvm-21-validation.md`
- Test: `docs/superpowers/reports/2026-04-04-llvm-21-validation.md`

- [ ] **Step 1: Prove the canonical docs still tell contributors to install LLVM 17**

Run: `rg -n "LLVM 17|llvm@17|llvm-config-17" README.md CONTRIBUTING.md docs/QUICKSTART.md docs/FAQ.md`

Expected: FAIL with stale references in the docs

- [ ] **Step 2: Update the docs to the LLVM 21 contract**

```markdown
<!-- CONTRIBUTING.md / docs/FAQ.md / docs/QUICKSTART.md -->
- Version 21 required (lite/native builds and local scripts assume LLVM 21)
- On macOS: `brew install llvm@21`
- On Windows: install the official LLVM 21 SDK and point `LLVM_DIR` at its CMake package
```

```markdown
<!-- docs/FAQ.md -->
### How do I install LLVM 21?

```bash
brew install llvm@21
export PATH="/opt/homebrew/opt/llvm@21/bin:$PATH"
```
```

```markdown
<!-- README.md -->
- **Backend**: LLVM 21 with native code generation and JIT support
```

- [ ] **Step 3: Add the validation report and populate it with actual command outcomes**

```bash
LINUX_CI_RESULT="PASS"
./scripts/test-ci-locally.sh linux >/tmp/eshkol-linux-ci.log 2>&1 || LINUX_CI_RESULT="FAIL"

XLA_DOCKER_RESULT="PASS"
docker build -f docker/xla/Dockerfile -t eshkol-xla-llvm21 . >/tmp/eshkol-xla-docker.log 2>&1 || XLA_DOCKER_RESULT="FAIL"

if [ "${XLA_DOCKER_RESULT}" = "FAIL" ]; then
  XLA_DOCKER_LINE="FAIL (explicit XLA/StableHLO exception; see /tmp/eshkol-xla-docker.log)"
else
  XLA_DOCKER_LINE="PASS"
fi

cat > docs/superpowers/reports/2026-04-04-llvm-21-validation.md <<EOF
# LLVM 21 Validation Report

- `cmake -S . -B build-llvm21 -G Ninja -DLLVM_CONFIG_EXECUTABLE="$(command -v llvm-config-21)"` — PASS
- `cmake --build build-llvm21 --parallel 4` — PASS
- `./build-llvm21/eshkol-run --shared-lib -o /tmp/eshkol-stdlib-llvm21 lib/stdlib.esk` — PASS
- `./scripts/run_cpp_type_tests.sh` — PASS
- `./scripts/smoke-windows-msys2.sh` — PASS
- `./scripts/test-ci-locally.sh linux` — ${LINUX_CI_RESULT}
- `docker build -f docker/xla/Dockerfile -t eshkol-xla-llvm21 .` — ${XLA_DOCKER_LINE}
EOF
```

- [ ] **Step 4: Re-run the doc string check and write the report from real command output**

Run: `rg -n "LLVM 17|llvm@17|llvm-config-17" README.md CONTRIBUTING.md docs/QUICKSTART.md docs/FAQ.md`

Expected: PASS with no matches

Run: `cat docs/superpowers/reports/2026-04-04-llvm-21-validation.md`

Expected: PASS and the file contains the actual command results from this task sequence

- [ ] **Step 5: Commit**

```bash
git add README.md CONTRIBUTING.md docs/QUICKSTART.md docs/FAQ.md docs/superpowers/reports/2026-04-04-llvm-21-validation.md
git commit -m "docs: publish LLVM 21 toolchain baseline"
```

## Self-Review Checklist

- [ ] The plan covers every requirement in `docs/superpowers/specs/2026-04-04-llvm-21-toolchain-unification-design.md`.
- [ ] No task still assumes `LLVM 17` or vague `18+` lite/native support.
- [ ] The plan keeps `STABLEHLO_ROOT` explicit as an exception path instead of silently treating it like the lite/native baseline.
- [ ] The Windows path is validated from WSL2 into the native Visual Studio 2022 + LLVM 21 environment rather than described abstractly.
- [ ] The final validation report contains real command outcomes, not placeholders.
