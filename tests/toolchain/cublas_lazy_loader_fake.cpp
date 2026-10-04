// Copyright (C) tsotchke
// SPDX-License-Identifier: MIT
#if defined(_WIN32)
#define CUBLAS_STATIC 1
#endif
#include <cublas_v2.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>

#if defined(_WIN32)
#define FAKE_EXPORT extern "C" __declspec(dllexport)
#else
#define FAKE_EXPORT extern "C" __attribute__((visibility("default")))
#endif

struct FakeCublasHandle { int marker; };
namespace {
std::atomic<int> create_count{0};
std::atomic<int> property_count{0};
std::atomic<int> live_handles{0};

void note_loaded() {
    const char* marker = std::getenv("FAKE_CUBLAS_LOAD_MARKER");
    if (!marker || !*marker) return;
    if (FILE* f = std::fopen(marker, "a")) {
        std::fputs("loaded\n", f);
        std::fclose(f);
    }
}

#if defined(_WIN32)
struct LoadMarker { LoadMarker() { note_loaded(); } };
LoadMarker load_marker;
#else
__attribute__((constructor)) static void mark_loaded() { note_loaded(); }
#endif
}

FAKE_EXPORT cublasStatus_t CUBLASWINAPI cublasGetProperty(libraryPropertyType_t property,
                                                           int* value) {
    ++property_count;
#if defined(FAKE_PROPERTY_QUERY_FAILURE)
    (void)property;
    (void)value;
    return CUBLAS_STATUS_NOT_INITIALIZED;
#else
    if (property != MAJOR_VERSION || !value) return CUBLAS_STATUS_NOT_INITIALIZED;
#if defined(FAKE_RUNTIME_MAJOR)
    *value = FAKE_RUNTIME_MAJOR;
#else
    *value = CUBLAS_VER_MAJOR;
#endif
    return CUBLAS_STATUS_SUCCESS;
#endif
}

FAKE_EXPORT cublasStatus_t CUBLASWINAPI cublasCreate_v2(cublasHandle_t* handle) {
    ++create_count;
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
#if defined(FAKE_HANDLE_CREATE_FAILURE)
    *handle = nullptr;
    return CUBLAS_STATUS_NOT_INITIALIZED;
#endif
    *handle = reinterpret_cast<cublasHandle_t>(new FakeCublasHandle{17});
    ++live_handles;
    return CUBLAS_STATUS_SUCCESS;
}
FAKE_EXPORT cublasStatus_t CUBLASWINAPI cublasSetStream_v2(cublasHandle_t, cudaStream_t) {
#if defined(FAKE_STREAM_BIND_FAILURE)
    return CUBLAS_STATUS_NOT_INITIALIZED;
#endif
    return CUBLAS_STATUS_SUCCESS;
}
FAKE_EXPORT cublasStatus_t CUBLASWINAPI cublasDestroy_v2(cublasHandle_t handle) {
    delete reinterpret_cast<FakeCublasHandle*>(handle);
    --live_handles;
    return CUBLAS_STATUS_SUCCESS;
}

#if !defined(FAKE_MISSING_SYMBOL)
FAKE_EXPORT cublasStatus_t CUBLASWINAPI cublasGemmEx(cublasHandle_t, cublasOperation_t,
    cublasOperation_t, int, int, int, const void*, const void*, cudaDataType_t, int,
    const void*, cudaDataType_t, int, const void*, void*, cudaDataType_t, int,
    cublasComputeType_t, cublasGemmAlgo_t) { return CUBLAS_STATUS_SUCCESS; }
FAKE_EXPORT cublasStatus_t CUBLASWINAPI cublasDgemm_v2(cublasHandle_t, cublasOperation_t,
    cublasOperation_t, int, int, int, const double*, const double*, int, const double*, int,
    const double*, double*, int) { return CUBLAS_STATUS_SUCCESS; }
FAKE_EXPORT cublasStatus_t CUBLASWINAPI cublasSgemm_v2(cublasHandle_t, cublasOperation_t,
    cublasOperation_t, int, int, int, const float*, const float*, int, const float*, int,
    const float*, float*, int) { return CUBLAS_STATUS_SUCCESS; }
FAKE_EXPORT cublasStatus_t CUBLASWINAPI cublasGemmStridedBatchedEx(cublasHandle_t,
    cublasOperation_t, cublasOperation_t, int, int, int, const void*, const void*,
    cudaDataType_t, int, long long, const void*, cudaDataType_t, int, long long,
    const void*, void*, cudaDataType_t, int, long long, int, cublasComputeType_t,
    cublasGemmAlgo_t) { return CUBLAS_STATUS_SUCCESS; }
#endif

FAKE_EXPORT int fake_cublas_create_count() { return create_count.load(); }
FAKE_EXPORT int fake_cublas_property_count() { return property_count.load(); }
FAKE_EXPORT int fake_cublas_live_handles() { return live_handles.load(); }
