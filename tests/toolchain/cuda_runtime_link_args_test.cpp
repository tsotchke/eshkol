//
// Copyright (C) tsotchke
//
// SPDX-License-Identifier: MIT
//

#include <eshkol/build_config.h>
#include <eshkol/platform_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

bool expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        return false;
    }
    return true;
}

void set_cuda_library_path(const std::string& value) {
#ifdef _WIN32
    _putenv_s("ESHKOL_CUDA_LIBRARY_PATH", value.c_str());
#else
    setenv("ESHKOL_CUDA_LIBRARY_PATH", value.c_str(), 1);
#endif
}

void clear_cuda_library_path() {
#ifdef _WIN32
    _putenv_s("ESHKOL_CUDA_LIBRARY_PATH", "");
#else
    unsetenv("ESHKOL_CUDA_LIBRARY_PATH");
#endif
}

void create_library(const fs::path& directory, const std::string& name) {
#ifdef _WIN32
    std::ofstream(directory / (name + ".lib")) << "fixture";
#else
    std::ofstream(directory / ("lib" + name + ".so")) << "fixture";
#  if defined(__linux__)
    if (ESHKOL_HOST_CUDA_MAJOR > 0) {
        std::ofstream(directory /
                      ("lib" + name + ".so." +
                       std::to_string(ESHKOL_HOST_CUDA_MAJOR))) << "fixture";
    }
#  endif
#endif
}

bool contains(const std::vector<std::string>& args, const std::string& value) {
    return std::find(args.begin(), args.end(), value) != args.end();
}

std::string expected_link_arg(const std::string& name) {
#if defined(__linux__)
    if (ESHKOL_HOST_CUDA_MAJOR > 0 && name != "cudadevrt") {
        return "-l:lib" + name + ".so." +
               std::to_string(ESHKOL_HOST_CUDA_MAJOR);
    }
#endif
    return "-l" + name;
}

}  // namespace

int main() {
    bool ok = true;
    const fs::path fixture =
        fs::temp_directory_path() / "eshkol-cuda-runtime-link-args-test";
    std::error_code ec;
    fs::remove_all(fixture, ec);
    fs::create_directories(fixture);

    for (const auto& name : {"cudart", "cublas", "cublasLt"}) {
        create_library(fixture, name);
    }
    set_cuda_library_path(fixture.string());

    const std::vector<std::string> libraries = {
        "cudart", "cublas", "cublasLt"};
    const auto args = eshkol::platform::cuda_runtime_link_args(libraries);
    const std::string canonical = fs::weakly_canonical(fixture).generic_string();

    ok &= expect(contains(args, "-L" + canonical),
                 "consumer CUDA development directory is selected");
    ok &= expect(contains(args, expected_link_arg("cudart")),
                 "CUDA runtime uses a consumer-portable linker name");
    ok &= expect(contains(args, expected_link_arg("cublas")),
                 "cuBLAS uses a consumer-portable linker name");
    ok &= expect(contains(args, expected_link_arg("cublasLt")),
                 "cuBLASLt uses a consumer-portable linker name");
#if defined(__linux__)
    ok &= expect(contains(args, "-Wl,-rpath," + canonical),
                 "consumer CUDA directory is preserved in ELF RUNPATH");
    ok &= expect(contains(args, "-Wl,-rpath-link," + canonical),
                 "consumer CUDA directory is available for transitive linking");
#endif

    for (const auto& arg : args) {
        ok &= expect(arg.find("__ESHKOL_CUDA_LIB__") == std::string::npos,
                     "internal CUDA marker never reaches the compiler driver");
    }

#ifdef _WIN32
    if (ESHKOL_HOST_CUDA_MAJOR > 0) {
        const fs::path incompatible =
            fixture / ("v" + std::to_string(ESHKOL_HOST_CUDA_MAJOR + 1)) /
            "lib" / "x64";
        fs::create_directories(incompatible);
        create_library(incompatible, "eshkolCudaVersionProbe");
        set_cuda_library_path(incompatible.string());
        const auto rejected = eshkol::platform::cuda_runtime_link_args(
            {"eshkolCudaVersionProbe"});
        const std::string incompatible_canonical =
            fs::weakly_canonical(incompatible).generic_string();
        ok &= expect(!contains(rejected, "-L" + incompatible_canonical),
                     "Windows rejects an incompatible versioned CUDA root");
        set_cuda_library_path(fixture.string());
    }
#endif

    if (ESHKOL_HOST_CUDA_MAJOR > 0) {
        const auto configured = eshkol::platform::host_runtime_link_args();
        ok &= expect(contains(configured, "-L" + canonical),
                     "configured CUDA markers resolve on the consumer");
        ok &= expect(contains(configured, expected_link_arg("cudart")),
                     "configured runtime closure resolves cudart by name");
        ok &= expect(contains(configured, expected_link_arg("cublas")),
                     "configured runtime closure resolves cuBLAS by name");
    }

    clear_cuda_library_path();
    fs::remove_all(fixture, ec);
    if (!ok) {
        return 1;
    }
    std::cout << "PASS: CUDA generated-link libraries resolve on the consumer\n";
    return 0;
}
