// Copyright (C) tsotchke
// SPDX-License-Identifier: MIT
#pragma once

// Narrow CUDA 12 declaration fixture for local loader-only builds on hosts
// without a CUDA toolkit. CUDA CI compiles the same probes against its real
// cublas_v2.h; this fixture is never added to product include paths.
#define CUBLAS_VER_MAJOR 12
#define CUBLAS_STATIC 1
#if defined(_WIN32)
#define CUBLASWINAPI __stdcall
#else
#define CUBLASWINAPI
#endif
#define CUBLASAPI

typedef void* cudaStream_t;
typedef struct FakeCublasHandle* cublasHandle_t;
typedef int cublasStatus_t;
typedef int cublasOperation_t;
typedef enum cudaDataType { CUDA_R_32F = 0 } cudaDataType_t;
typedef enum cublasComputeType_t { CUBLAS_COMPUTE_32F = 68 } cublasComputeType_t;
typedef int cublasGemmAlgo_t;
typedef enum libraryPropertyType_t { MAJOR_VERSION = 0 } libraryPropertyType_t;

enum { CUBLAS_STATUS_SUCCESS = 0, CUBLAS_STATUS_NOT_INITIALIZED = 1 };

extern "C" {
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetProperty(libraryPropertyType_t, int*);
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCreate_v2(cublasHandle_t*);
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetStream_v2(cublasHandle_t, cudaStream_t);
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDestroy_v2(cublasHandle_t);
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmEx(cublasHandle_t, cublasOperation_t,
    cublasOperation_t, int, int, int, const void*, const void*, cudaDataType_t, int,
    const void*, cudaDataType_t, int, const void*, void*, cudaDataType_t, int,
    cublasComputeType_t, cublasGemmAlgo_t);
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemm_v2(cublasHandle_t, cublasOperation_t,
    cublasOperation_t, int, int, int, const double*, const double*, int, const double*, int,
    const double*, double*, int);
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemm_v2(cublasHandle_t, cublasOperation_t,
    cublasOperation_t, int, int, int, const float*, const float*, int, const float*, int,
    const float*, float*, int);
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmStridedBatchedEx(cublasHandle_t,
    cublasOperation_t, cublasOperation_t, int, int, int, const void*, const void*,
    cudaDataType_t, int, long long, const void*, cudaDataType_t, int, long long,
    const void*, void*, cudaDataType_t, int, long long, int, cublasComputeType_t,
    cublasGemmAlgo_t);
}
#define cublasCreate cublasCreate_v2
#define cublasSetStream cublasSetStream_v2
#define cublasDestroy cublasDestroy_v2
#define cublasDgemm cublasDgemm_v2
#define cublasSgemm cublasSgemm_v2

// Real CUDA headers offer these C++ compatibility overloads as well as the
// modern C ABI. A bare decltype(&cublasGemmEx) must therefore fail to compile.
inline cublasStatus_t CUBLASWINAPI cublasGemmEx(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int, const void*, const void*, cudaDataType_t, int,
    const void*, cudaDataType_t, int, const void*, void*, cudaDataType_t, int,
    cudaDataType_t, cublasGemmAlgo_t) { return CUBLAS_STATUS_NOT_INITIALIZED; }
inline cublasStatus_t CUBLASWINAPI cublasGemmStridedBatchedEx(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int, const void*, const void*, cudaDataType_t, int, long long,
    const void*, cudaDataType_t, int, long long, const void*, void*,
    cudaDataType_t, int, long long, int, cudaDataType_t, cublasGemmAlgo_t) {
    return CUBLAS_STATUS_NOT_INITIALIZED;
}
