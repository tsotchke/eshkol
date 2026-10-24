#ifndef ESHKOL_PRNG_STATE_H
#define ESHKOL_PRNG_STATE_H
#include <stdint.h>
/* Shared drand48-msb-v1 state operations; ownership belongs to each backend. */
static inline uint64_t eshkol_prng_state_seed(int64_t seed) {
    return (((uint64_t)seed << 16) | UINT64_C(0x330E)) & UINT64_C(0xffffffffffff);
}
static inline double eshkol_prng_state_random(uint64_t* state) {
    *state = (*state * UINT64_C(0x5DEECE66D) + UINT64_C(0xB)) & UINT64_C(0xffffffffffff);
    return (double)*state / 281474976710656.0;
}
static inline int64_t eshkol_prng_state_integer(uint64_t* state, int64_t n) {
    if (n <= 0) return 0; /* Native contract: no draw for an empty range. */
    return (int64_t)(eshkol_prng_state_random(state) * (double)n);
}
#endif
