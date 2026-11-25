#ifndef ESHKOL_SEED_ENTROPY_H
#define ESHKOL_SEED_ENTROPY_H

#include <stddef.h>
#include <stdlib.h>
#ifdef __EMSCRIPTEN__
#include <unistd.h>
#else
#ifdef __cplusplus
extern "C"
#endif
int eshkol_random_bytes(char*, size_t);
#endif

/* Fill a small generator seed or terminate if the host has no entropy. */
static inline void eshkol_seed_entropy(void* data, size_t size) {
#ifdef __EMSCRIPTEN__
    if (getentropy(data, size) != 0) abort();
#else
    if (eshkol_random_bytes((char*)data, size) != 0) abort();
#endif
}
#endif
