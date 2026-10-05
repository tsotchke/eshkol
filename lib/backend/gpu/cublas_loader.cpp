/*
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#include "cublas_loader.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#if !defined(ESHKOL_CUBLAS_LOADER_TESTING)
#include <cuda_runtime_api.h>
#endif

namespace eshkol::cuda {
namespace {

#if defined(_WIN32)
using Module = HMODULE;
static void* symbol(Module module, const char* name) {
    return reinterpret_cast<void*>(GetProcAddress(module, name));
}
static std::string loader_error(DWORD code) {
    std::ostringstream out;
    out << "Windows loader error " << code;
    return out.str();
}
#else
using Module = void*;
static void* symbol(Module module, const char* name) { return dlsym(module, name); }
static std::string loader_error() {
    const char* error = dlerror();
    return error ? error : "unknown dynamic-loader error";
}
#endif

template <typename T>
static bool resolve(Module module, const char* name, T& out) {
    out = reinterpret_cast<T>(symbol(module, name));
    return out != nullptr;
}

#define ESHKOL_CUBLAS_STRINGIFY_INNER(symbol) #symbol
#define ESHKOL_CUBLAS_STRINGIFY(symbol) ESHKOL_CUBLAS_STRINGIFY_INNER(symbol)

#if !defined(_WIN32)
static const char* library_name() {
    return "libcublas.so.";
}
#endif

#if defined(_WIN32)
static std::vector<std::wstring> runtime_candidates() {
    const std::wstring dll = L"cublas64_" + std::to_wstring(CUBLAS_VER_MAJOR) + L".dll";
    std::vector<std::wstring> result;
    if (const wchar_t* explicit_dirs = _wgetenv(L"ESHKOL_CUDA_LIBRARY_PATH")) {
        std::wstringstream paths(explicit_dirs);
        std::wstring dir;
        while (std::getline(paths, dir, L';')) {
            if (dir.empty()) continue;
            std::filesystem::path library_dir(dir);
            if (library_dir.is_absolute()) {
                result.push_back((library_dir / dll).wstring());
                // The established Windows setting points at lib/x64 while
                // runtime DLLs live in the sibling toolkit bin directory.
                auto root = library_dir.parent_path().parent_path();
                result.push_back((root / L"bin" / dll).wstring());
            }
        }
    }
    for (const wchar_t* key : {L"CUDAToolkit_ROOT", L"CUDA_PATH", L"CUDA_HOME", L"CUDA_ROOT"}) {
        const wchar_t* root = _wgetenv(key);
        if (root && *root) {
            std::filesystem::path candidate = std::filesystem::path(root) / L"bin" / dll;
            if (candidate.is_absolute()) result.push_back(candidate.wstring());
        }
    }
    // CUDA's DLL is in the same bin directory as the cudart module. The
    // test build omits this lookup and supplies an isolated exact path.
#if !defined(ESHKOL_CUBLAS_LOADER_TESTING)
    HMODULE runtime = nullptr;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCSTR>(&cudaGetDeviceCount), &runtime)) {
        std::vector<wchar_t> path(512);
        DWORD n = 0;
        while (path.size() <= 32768) {
            n = GetModuleFileNameW(runtime, path.data(), static_cast<DWORD>(path.size()));
            if (!n) break;
            if (n < path.size()) break;
            if (path.size() == 32768) { n = 0; break; }
            path.resize(path.size() * 2);
        }
        if (n && n < path.size()) {
            std::filesystem::path runtime_path(std::wstring(path.data(), n));
            auto candidate = runtime_path.parent_path() / dll;
            if (candidate.is_absolute()) result.push_back(candidate.wstring());
        }
    }
#endif
    result.push_back(dll); // DefaultDirs lookup still requires exact major.
    return result;
}
#else
static void add_if_file(std::vector<std::string>& out, const std::string& path) {
    std::error_code ec;
    if (std::filesystem::path(path).is_absolute() && std::filesystem::is_regular_file(path, ec))
        out.push_back(path);
}

static std::vector<std::string> runtime_candidates() {
    const std::string so = std::string(library_name()) + std::to_string(CUBLAS_VER_MAJOR);
    std::vector<std::string> result;
    std::vector<std::string> roots;
    if (const char* explicit_dirs = std::getenv("ESHKOL_CUDA_LIBRARY_PATH")) {
        std::stringstream paths(explicit_dirs);
        std::string dir;
        while (std::getline(paths, dir, ':')) {
            if (!dir.empty()) add_if_file(result, (std::filesystem::path(dir) / so).string());
        }
    }
    for (const char* key : {"CUDAToolkit_ROOT", "CUDA_PATH", "CUDA_HOME", "CUDA_ROOT", "CUDA_HOME_PATH"}) {
        const char* value = std::getenv(key);
        if (value && *value) roots.emplace_back(value);
    }
#if !defined(ESHKOL_CUBLAS_LOADER_TESTING)
    // cudaGetDeviceCount is already part of the eagerly linked cudart closure;
    // dladdr inspects its owning module without issuing a CUDA runtime call.
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&cudaGetDeviceCount), &info) && info.dli_fname) {
        const std::filesystem::path runtime(info.dli_fname);
        const auto parent = runtime.parent_path();
        add_if_file(result, (parent / so).string());
        if (parent.filename() == "lib64" || parent.filename() == "lib")
            roots.push_back(parent.parent_path().string());
    }
#endif
    // nvcc on PATH supplies an installed toolkit root. This is discovery only;
    // no shell command or user-selected vendor library filename is executed.
    if (const char* path_env = std::getenv("PATH")) {
        std::stringstream paths(path_env);
        std::string entry;
        while (std::getline(paths, entry, ':')) {
            if (entry.empty()) continue;
            const std::filesystem::path nvcc = std::filesystem::path(entry) / "nvcc";
            std::error_code ec;
            if (nvcc.is_absolute() && std::filesystem::is_regular_file(nvcc, ec))
                roots.push_back(nvcc.parent_path().parent_path().string());
        }
    }
    for (const auto& root : roots) {
        for (const char* sub : {"lib64", "lib", "targets/x86_64-linux/lib", "targets/aarch64-linux/lib"})
            add_if_file(result, (std::filesystem::path(root) / sub / so).string());
    }
    // The exact major soname is intentional: the loader may use standard
    // system paths, but cannot silently bind a different CUDA/cuBLAS major.
    result.push_back(so);
    return result;
}
#endif

} // namespace

bool CublasLoader::admit_locked(const char* test_path, std::string* diagnostic) {
    if (attempted_) {
        if (diagnostic) *diagnostic = diagnostic_;
        return admitted_;
    }
    attempted_ = true;

    std::vector<std::string> candidates;
#if defined(_WIN32)
    std::vector<std::wstring> wide_candidates;
#endif
#if defined(ESHKOL_CUBLAS_LOADER_TESTING)
    if (test_path && std::strcmp(test_path, "@DISCOVERY@") == 0) {
#if defined(_WIN32)
        wide_candidates = runtime_candidates();
#else
        candidates = runtime_candidates();
#endif
    } else if (test_path && *test_path) {
#if defined(_WIN32)
        auto path = std::filesystem::u8path(test_path);
        if (!path.is_absolute()) {
            diagnostic_ = "test library path must be absolute";
            if (diagnostic) *diagnostic = diagnostic_;
            return false;
        }
        wide_candidates.emplace_back(path.wstring());
#else
        auto path = std::filesystem::u8path(test_path);
        if (!path.is_absolute()) {
            diagnostic_ = "test library path must be absolute";
            if (diagnostic) *diagnostic = diagnostic_;
            return false;
        }
        candidates.emplace_back(path.string());
#endif
    }
#else
    (void)test_path;
#if defined(_WIN32)
    wide_candidates = runtime_candidates();
#else
    candidates = runtime_candidates();
#endif
#endif

    std::ostringstream attempts;
#if defined(_WIN32)
    if (wide_candidates.empty()) {
        diagnostic_ = "no exact-major cuBLAS candidate found";
        if (diagnostic) *diagnostic = diagnostic_;
        return false;
    }
    for (const auto& path : wide_candidates) {
        const bool absolute = std::filesystem::path(path).is_absolute();
        DWORD flags = LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
        if (absolute) flags |= LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR;
        Module module = LoadLibraryExW(path.c_str(), nullptr, flags);
        const auto encoded_path = std::filesystem::path(path).u8string();
        // C++20 returns u8string here; diagnostics preserve its UTF-8 bytes
        // while Windows library admission continues to use the native path.
        const std::string display_path(
            reinterpret_cast<const char*>(encoded_path.data()), encoded_path.size());
        if (!module) {
            attempts << display_path << ": " << loader_error(GetLastError()) << "; ";
            continue;
        }
#else
    for (const auto& path : candidates) {
        dlerror();
        Module module = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!module) {
            attempts << path << ": " << loader_error() << "; ";
            continue;
        }
#endif
        CublasApi pending{};
        bool complete = resolve(module, ESHKOL_CUBLAS_STRINGIFY(cublasGetProperty), pending.get_property) &&
            resolve(module, ESHKOL_CUBLAS_STRINGIFY(cublasCreate), pending.create) &&
            resolve(module, ESHKOL_CUBLAS_STRINGIFY(cublasSetStream), pending.set_stream) &&
            resolve(module, ESHKOL_CUBLAS_STRINGIFY(cublasDestroy), pending.destroy) &&
            resolve(module, ESHKOL_CUBLAS_STRINGIFY(cublasGemmEx), pending.gemm_ex) &&
            resolve(module, ESHKOL_CUBLAS_STRINGIFY(cublasDgemm), pending.dgemm) &&
            resolve(module, ESHKOL_CUBLAS_STRINGIFY(cublasSgemm), pending.sgemm) &&
            resolve(module, ESHKOL_CUBLAS_STRINGIFY(cublasGemmStridedBatchedEx), pending.gemm_strided_batched_ex);
        if (!complete) {
            attempts <<
#if defined(_WIN32)
                display_path
#else
                path
#endif
                << ": missing one or more required cuBLAS symbols; ";
#if defined(_WIN32)
            FreeLibrary(module);
#else
            dlclose(module);
#endif
            continue;
        }
        int major = 0;
        if (pending.get_property(MAJOR_VERSION, &major) != CUBLAS_STATUS_SUCCESS ||
            major != CUBLAS_VER_MAJOR) {
            attempts <<
#if defined(_WIN32)
                display_path
#else
                path
#endif
                     << ": cuBLAS major " << major << " does not match headers "
                     << CUBLAS_VER_MAJOR << "; ";
#if defined(_WIN32)
            FreeLibrary(module);
#else
            dlclose(module);
#endif
            continue;
        }

        // Publish only after symbol and runtime-major validation. Once
        // published, the module remains pinned for process lifetime.
        module_ = module;
        api_ = pending;
        admitted_ = true;
        diagnostic_ = "loaded " +
#if defined(_WIN32)
            display_path
#else
            path
#endif
            + " (cuBLAS " + std::to_string(major) + ")";
        if (diagnostic) *diagnostic = diagnostic_;
        return true;
    }
    diagnostic_ = attempts.str().empty() ? "no exact-major cuBLAS candidate found" : attempts.str();
    if (diagnostic) *diagnostic = diagnostic_;
    return false;
}

bool CublasLoader::ensure_handle_impl(const char* test_path, cudaStream_t stream,
                                      std::string* diagnostic) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!admit_locked(test_path, diagnostic)) return false;
    if (handle_) {
        if (diagnostic) *diagnostic = diagnostic_;
        return true;
    }
    handle_status_ = api_.create(&handle_);
    if (handle_status_ == CUBLAS_STATUS_SUCCESS)
        handle_status_ = api_.set_stream(handle_, stream);
    if (handle_status_ != CUBLAS_STATUS_SUCCESS) {
        if (handle_) api_.destroy(handle_);
        handle_ = nullptr;
        diagnostic_ += "; cuBLAS handle creation/stream binding failed with status " +
                       std::to_string(static_cast<int>(handle_status_));
        if (diagnostic) *diagnostic = diagnostic_;
        return false;
    }
    if (diagnostic) *diagnostic = diagnostic_;
    return true;
}

bool CublasLoader::ensure_handle(cudaStream_t stream, std::string* diagnostic) {
    return ensure_handle_impl(nullptr, stream, diagnostic);
}

#if defined(ESHKOL_CUBLAS_LOADER_TESTING)
bool CublasLoader::ensure_handle_from_path_for_test(const char* path, cudaStream_t stream,
                                                     std::string* diagnostic) {
    return ensure_handle_impl(path, stream, diagnostic);
}

bool CublasLoader::ensure_handle_from_discovery_for_test(cudaStream_t stream,
                                                          std::string* diagnostic) {
    return ensure_handle_impl("@DISCOVERY@", stream, diagnostic);
}
#endif

void CublasLoader::destroy_handle() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (handle_) {
        api_.destroy(handle_);
        handle_ = nullptr;
        handle_status_ = CUBLAS_STATUS_NOT_INITIALIZED;
    }
}

const CublasApi* CublasLoader::api() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return admitted_ ? &api_ : nullptr;
}

cublasHandle_t CublasLoader::handle() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return handle_;
}

} // namespace eshkol::cuda
