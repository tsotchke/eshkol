// Copyright (C) tsotchke
// SPDX-License-Identifier: MIT
#include "../../lib/backend/gpu/cublas_loader.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
using TestModule = HMODULE;
static TestModule open_test_module(const std::filesystem::path& path) {
    return LoadLibraryW(path.c_str());
}
static void* test_symbol(TestModule m, const char* n) {
    return reinterpret_cast<void*>(GetProcAddress(m, n));
}
#else
#include <dlfcn.h>
#include <unistd.h>
using TestModule = void*;
static TestModule open_test_module(const std::filesystem::path& path) {
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
}
static void* test_symbol(TestModule m, const char* n) { return dlsym(m, n); }
#endif

namespace {
bool expect(bool value, const char* description) {
    if (!value) std::cerr << "FAIL: " << description << '\n';
    return value;
}
bool marker_exists(const std::filesystem::path& path) {
    std::error_code ec;
    return std::filesystem::exists(path, ec);
}
}

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "usage: cublas_lazy_loader_test complete incomplete wrong-major property-failure create-failure stream-failure\n";
        return 2;
    }
    std::error_code ec;
    const auto test_root = std::filesystem::temp_directory_path() /
        ("eshkol-cublas-loader-" + std::to_string(
#if defined(_WIN32)
            static_cast<unsigned long>(GetCurrentProcessId())
#else
            static_cast<unsigned long>(getpid())
#endif
        ));
    std::filesystem::create_directories(test_root, ec);
    const auto marker = test_root / "load-marker";
    std::filesystem::remove(marker, ec);
#if defined(_WIN32)
    const wchar_t* root_variable = L"CUDA_PATH";
    const wchar_t* old_root = _wgetenv(root_variable);
    const bool had_root = old_root != nullptr;
    const std::wstring saved_root = old_root ? old_root : L"";
    const wchar_t* old_library_path = _wgetenv(L"ESHKOL_CUDA_LIBRARY_PATH");
    const bool had_library_path = old_library_path != nullptr;
    const std::wstring saved_library_path = old_library_path ? old_library_path : L"";
    const wchar_t* old_marker = _wgetenv(L"FAKE_CUBLAS_LOAD_MARKER");
    const bool had_marker = old_marker != nullptr;
    const std::wstring saved_marker = old_marker ? old_marker : L"";
#else
    const char* root_variable = "CUDA_HOME";
    const char* old_root = std::getenv(root_variable);
    const bool had_root = old_root != nullptr;
    const std::string saved_root = old_root ? old_root : "";
    const char* old_library_path = std::getenv("ESHKOL_CUDA_LIBRARY_PATH");
    const bool had_library_path = old_library_path != nullptr;
    const std::string saved_library_path = old_library_path ? old_library_path : "";
    const char* old_marker = std::getenv("FAKE_CUBLAS_LOAD_MARKER");
    const bool had_marker = old_marker != nullptr;
    const std::string saved_marker = old_marker ? old_marker : "";
#endif
#if defined(_WIN32)
    // Exercise native environment and DLL path handling beyond ASCII, using
    // the actual complete fixture built against the selected toolkit headers.
    const auto runtime_root = test_root / L"\u03bb\u6d4b\u8bd5-runtime";
    const auto library_dir = runtime_root / "lib" / "x64";
    const auto complete_path = library_dir / std::filesystem::u8path(argv[1]).filename();
    std::filesystem::create_directories(library_dir, ec);
    if (!ec) std::filesystem::copy_file(std::filesystem::u8path(argv[1]), complete_path,
                                       std::filesystem::copy_options::overwrite_existing, ec);
    if (ec) {
        std::cerr << "FAIL: Unicode fixture setup: " << ec.message() << '\n';
        return 2;
    }
#else
    const auto runtime_root = std::filesystem::u8path(argv[1]).parent_path().parent_path();
    const auto library_dir = std::filesystem::u8path(argv[1]).parent_path();
    const auto complete_path = std::filesystem::u8path(argv[1]);
#endif
#if defined(_WIN32)
    _wputenv_s(L"FAKE_CUBLAS_LOAD_MARKER", marker.c_str());
    _wputenv_s(root_variable, runtime_root.c_str());
    _wputenv_s(L"ESHKOL_CUDA_LIBRARY_PATH", library_dir.c_str());
#else
    setenv("FAKE_CUBLAS_LOAD_MARKER", marker.string().c_str(), 1);
    setenv(root_variable, runtime_root.string().c_str(), 1);
    setenv("ESHKOL_CUDA_LIBRARY_PATH", library_dir.string().c_str(), 1);
#endif
    bool ok = true;

    // CPU/no-GEMM startup does not call the loader and therefore leaves the
    // fake library unmapped. This invokes no CUDA or vendor API.
    eshkol::cuda::CublasLoader cpu_startup;
    ok &= expect(cpu_startup.api() == nullptr && !marker_exists(marker),
                 "CPU startup does not load the fake cuBLAS module");

    eshkol::cuda::CublasLoader complete;
    std::string diagnostic;
    std::vector<std::thread> workers;
    std::array<int, 16> results{};
    for (size_t i = 0; i < results.size(); ++i) {
        workers.emplace_back([&, i] {
            std::string worker_diagnostic;
            results[i] = complete.ensure_handle_from_discovery_for_test(
                nullptr, &worker_diagnostic) ? 1 : 0;
        });
    }
    for (auto& worker : workers) worker.join();
    for (int result : results) ok &= expect(result != 0, "concurrent complete admission succeeds");
    ok &= expect(complete.api() != nullptr && complete.handle() != nullptr,
                 "complete table and handle publish together after validation");
    ok &= expect(marker_exists(marker), "first GEMM admission opens the fake module");

    TestModule complete_module = open_test_module(complete_path);
    using Counter = int (*)();
    auto create_count = reinterpret_cast<Counter>(test_symbol(complete_module,
                                                               "fake_cublas_create_count"));
    auto property_count = reinterpret_cast<Counter>(test_symbol(complete_module,
                                                                "fake_cublas_property_count"));
    ok &= expect(create_count && create_count() == 1,
                 "concurrent first use creates exactly one shared handle");
    ok &= expect(property_count && property_count() == 1,
                 "concurrent first use queries runtime ABI exactly once");
    complete.destroy_handle();
    ok &= expect(complete.handle() == nullptr,
                 "shutdown clears the handle while retaining the validated table");
    ok &= expect(complete.ensure_handle_from_discovery_for_test(nullptr, &diagnostic),
                 "reinitialization creates a new handle from the pinned module");
    ok &= expect(create_count && property_count && create_count() == 2 && property_count() == 1,
                 "reinitialization reuses admission without loading or validating again");
    complete.destroy_handle();

    eshkol::cuda::CublasLoader incomplete;
    ok &= expect(!incomplete.ensure_handle_from_path_for_test(argv[2], nullptr, &diagnostic),
                 "missing required export rejects the whole library");
    ok &= expect(incomplete.api() == nullptr && incomplete.handle() == nullptr,
                 "incomplete library publishes no table or handle");
    ok &= expect(!incomplete.ensure_handle_from_path_for_test(argv[1], nullptr, &diagnostic) &&
                 incomplete.api() == nullptr,
                 "failed admission is terminal and cannot publish a later partial table");

    eshkol::cuda::CublasLoader missing;
    const auto missing_path = std::filesystem::u8path(argv[2]).parent_path() / "no-such-cublas.so";
    ok &= expect(!missing.ensure_handle_from_path_for_test(missing_path.string().c_str(), nullptr,
                                                            &diagnostic),
                 "missing runtime library fails closed");
    ok &= expect(missing.api() == nullptr && missing.handle() == nullptr,
                 "missing runtime library publishes nothing");

    eshkol::cuda::CublasLoader wrong_major;
    ok &= expect(!wrong_major.ensure_handle_from_path_for_test(argv[3], nullptr, &diagnostic),
                 "runtime-reported wrong major rejects an exact-name candidate");
    ok &= expect(wrong_major.api() == nullptr && wrong_major.handle() == nullptr,
                 "wrong-major library publishes no table or handle");

    eshkol::cuda::CublasLoader property_failure;
    ok &= expect(!property_failure.ensure_handle_from_path_for_test(argv[4], nullptr, &diagnostic),
                 "failed runtime property query rejects the library");
    ok &= expect(property_failure.api() == nullptr && property_failure.handle() == nullptr,
                 "failed ABI query publishes no table or handle");

    for (int index : {5, 6}) {
        eshkol::cuda::CublasLoader failed_handle;
        ok &= expect(!failed_handle.ensure_handle_from_path_for_test(argv[index], nullptr, &diagnostic),
                     "handle creation or stream binding failure rejects GEMM admission");
        ok &= expect(failed_handle.handle() == nullptr,
                     "failed GEMM admission never publishes a handle");
        TestModule module = open_test_module(std::filesystem::u8path(argv[index]));
        auto live = reinterpret_cast<Counter>(test_symbol(module, "fake_cublas_live_handles"));
        ok &= expect(live && live() == 0,
                     "failed handle admission leaves no vendor handles allocated");
#if defined(_WIN32)
        if (module) FreeLibrary(module);
#else
        if (module) dlclose(module);
#endif
    }

#if defined(_WIN32)
    if (complete_module) FreeLibrary(complete_module);
#else
    if (complete_module) dlclose(complete_module);
#endif
    std::filesystem::remove_all(test_root, ec);
#if defined(_WIN32)
    _wputenv_s(L"FAKE_CUBLAS_LOAD_MARKER", had_marker ? saved_marker.c_str() : L"");
    _wputenv_s(root_variable, had_root ? saved_root.c_str() : L"");
    _wputenv_s(L"ESHKOL_CUDA_LIBRARY_PATH", had_library_path ? saved_library_path.c_str() : L"");
#else
    if (had_marker) setenv("FAKE_CUBLAS_LOAD_MARKER", saved_marker.c_str(), 1);
    else unsetenv("FAKE_CUBLAS_LOAD_MARKER");
    if (had_root) setenv(root_variable, saved_root.c_str(), 1);
    else unsetenv(root_variable);
    if (had_library_path) setenv("ESHKOL_CUDA_LIBRARY_PATH", saved_library_path.c_str(), 1);
    else unsetenv("ESHKOL_CUDA_LIBRARY_PATH");
#endif
    return ok ? 0 : 1;
}
