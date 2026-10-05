/*
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 *
 * Private, lazy cuBLAS dispatch. All signatures come from the selected CUDA
 * toolkit's cublas_v2.h so the table tracks the actual header ABI.
 */
#pragma once

#include <cublas_v2.h>
#include <cstddef>
#include <mutex>
#include <string>
#include <tuple>

namespace eshkol::cuda {

#if !defined(CUBLAS_VER_MAJOR)
#error "The CUDA cuBLAS headers must define CUBLAS_VER_MAJOR"
#endif

// GemmEx APIs also expose a C++ compatibility overload using cudaDataType_t
// for compute precision. Select the modern cublasComputeType_t overload;
// taking the unqualified function address is ambiguous with real headers.
template <typename Function> struct CublasFunctionArguments;
template <typename Result, typename... Arguments>
struct CublasFunctionArguments<Result (CUBLASWINAPI *)(Arguments...)> {
    template <std::size_t Index> using Argument = std::tuple_element_t<Index, std::tuple<Arguments...>>;
};
using CublasDimension = CublasFunctionArguments<decltype(&::cublasDgemm)>::Argument<3>;
using CublasGemmEx = cublasStatus_t (CUBLASWINAPI *)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    CublasDimension, CublasDimension, CublasDimension,
    const void*, const void*, cudaDataType_t, CublasDimension,
    const void*, cudaDataType_t, CublasDimension, const void*, void*,
    cudaDataType_t, CublasDimension, cublasComputeType_t, cublasGemmAlgo_t);
using CublasGemmStridedBatchedEx = cublasStatus_t (CUBLASWINAPI *)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    CublasDimension, CublasDimension, CublasDimension,
    const void*, const void*, cudaDataType_t, CublasDimension, long long,
    const void*, cudaDataType_t, CublasDimension, long long,
    const void*, void*, cudaDataType_t, CublasDimension, long long,
    CublasDimension, cublasComputeType_t, cublasGemmAlgo_t);

struct CublasApi {
    decltype(&::cublasGetProperty) get_property = nullptr;
    decltype(&::cublasCreate) create = nullptr;
    decltype(&::cublasSetStream) set_stream = nullptr;
    decltype(&::cublasDestroy) destroy = nullptr;
    decltype(static_cast<CublasGemmEx>(&::cublasGemmEx)) gemm_ex = nullptr;
    decltype(&::cublasDgemm) dgemm = nullptr;
    decltype(&::cublasSgemm) sgemm = nullptr;
    decltype(static_cast<CublasGemmStridedBatchedEx>(&::cublasGemmStridedBatchedEx))
        gemm_strided_batched_ex = nullptr;
};

class CublasLoader {
public:
    CublasLoader() = default;
    CublasLoader(const CublasLoader&) = delete;
    CublasLoader& operator=(const CublasLoader&) = delete;

    // Terminal admission is cached, including missing/incomplete/wrong-major
    // libraries. The successful module is intentionally kept for process life.
    bool ensure_handle(cudaStream_t stream, std::string* diagnostic = nullptr);
    void destroy_handle();
    const CublasApi* api() const;
    // Runtime GEMM callers hold the outer GPU lifecycle mutex while using the
    // handle; these getters take this loader's mutex for snapshot safety.
    cublasHandle_t handle() const;

#if defined(ESHKOL_CUBLAS_LOADER_TESTING)
    // Test-only path injection. Production builds expose no environment or
    // caller-selected library override.
    bool ensure_handle_from_path_for_test(const char* path, cudaStream_t stream,
                                          std::string* diagnostic = nullptr);
    bool ensure_handle_from_discovery_for_test(cudaStream_t stream,
                                               std::string* diagnostic = nullptr);
#endif

private:
    bool ensure_handle_impl(const char* test_path, cudaStream_t stream,
                            std::string* diagnostic);
    bool admit_locked(const char* test_path, std::string* diagnostic);

    mutable std::mutex mutex_;
    bool attempted_ = false;
    bool admitted_ = false;
    CublasApi api_{};
    void* module_ = nullptr;
    cublasHandle_t handle_ = nullptr;
    cublasStatus_t handle_status_ = CUBLAS_STATUS_NOT_INITIALIZED;
    std::string diagnostic_;
};

} // namespace eshkol::cuda
