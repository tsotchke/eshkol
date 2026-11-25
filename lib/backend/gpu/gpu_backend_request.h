/*
 * gpu_backend_request.h — the ESHKOL_GPU_BACKEND run-time selection.
 *
 * Shared by the host GPU backend files (gpu_memory_cuda.cpp,
 * gpu_memory_stub.cpp) so every build parses the variable the same way.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#ifndef ESHKOL_GPU_BACKEND_REQUEST_H
#define ESHKOL_GPU_BACKEND_REQUEST_H

#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Runtime backend request from ESHKOL_GPU_BACKEND (case-insensitive):
 * unset/"auto" -> AUTO; "cpu"/"none" -> CPU; "cuda"; "vulkan". Unknown values
 * are treated as AUTO. Shared by every host backend file. */
typedef enum {
    ESHKOL_GPU_REQUEST_AUTO = 0,
    ESHKOL_GPU_REQUEST_CPU = 1,
    ESHKOL_GPU_REQUEST_CUDA = 2,
    ESHKOL_GPU_REQUEST_VULKAN = 3
} EshkolGPURequest;

static inline EshkolGPURequest eshkol_gpu_backend_request(void) {
    const char* v = getenv("ESHKOL_GPU_BACKEND");
    char buf[16];
    size_t i = 0;
    if (!v || !*v) return ESHKOL_GPU_REQUEST_AUTO;
    for (; v[i] && i + 1 < sizeof(buf); i++) {
        char c = v[i];
        buf[i] = (c >= 'A' && c <= 'Z') ? (char)(c - 'A' + 'a') : c;
    }
    buf[i] = '\0';
    if (strcmp(buf, "cpu") == 0 || strcmp(buf, "none") == 0) return ESHKOL_GPU_REQUEST_CPU;
    if (strcmp(buf, "cuda") == 0) return ESHKOL_GPU_REQUEST_CUDA;
    if (strcmp(buf, "vulkan") == 0) return ESHKOL_GPU_REQUEST_VULKAN;
    return ESHKOL_GPU_REQUEST_AUTO;
}

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_GPU_BACKEND_REQUEST_H */
