#ifndef ESHKOL_TABLE_HASH_H
#define ESHKOL_TABLE_HASH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint64_t eshkol_table_hash(const void* data, size_t length);
static inline uint64_t eshkol_table_hash_pair(uint64_t a, uint64_t b) {
    const uint64_t words[] = {a, b};
    return eshkol_table_hash(words, sizeof(words));
}

#ifdef __cplusplus
}
#endif
#endif
