#ifndef ESHKOL_RUNTIME_REGION_LEAF_LAYOUTS_H
#define ESHKOL_RUNTIME_REGION_LEAF_LAYOUTS_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
// Producer-owned opaque payload sizes. Internal linkage visibility only; no
// source builtin, JIT registration, or installed compiler/runtime interface.
#if defined(__GNUC__)
__attribute__((visibility("hidden")))
#endif
size_t eshkol_dnc_promotion_size(void);
#if defined(__GNUC__)
__attribute__((visibility("hidden")))
#endif
size_t eshkol_sdnc_promotion_size(void);
#ifdef __cplusplus
}
#endif
#endif
