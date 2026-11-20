/* prng.cpp — isolated PRNG state for deterministic replay and per-task isolation.
 *
 * The global drand48-like PRNG (eshkol_drand48, eshkol_srand48) in the hosted
 * runtime exports is mutex-protected — every thread shares one state.
 * For parallel-map workloads or paper artifact reproducibility, callers
 * need their own PRNG so two threads sampling concurrently get reproducible,
 * non-interleaved sequences.
 *
 * This file exposes:
 *   eshkol_prng_make(seed)              -> arena-allocated state with subtype header
 *   eshkol_prng_random(state)           -> double in [0.0, 1.0)
 *   eshkol_prng_random_integer(state,n) -> int64 in [0, n)
 *   eshkol_random_seed(seed)            -> seeds the global PRNG explicitly
 *
 * State uses the same LCG constants as drand48 so a (make-prng s) followed
 * by (prng-random p) returns the same sequence as (set-random-seed! s) +
 * (random) — useful for migrating sequential code to the isolated form.
 *
 * Lock-free: each prng_state is independent; concurrent access from multiple
 * threads to *different* states is safe. Concurrent access to the *same*
 * state is a user-side race and is documented as such.
 */

#include "../../inc/eshkol/eshkol.h"
#include <cstddef>
#include <cstdint>
#include "eshkol/core/prng_state.h"

/* Forward-declare arena_t (the full struct lives in arena_memory.cpp;
 * we only need the pointer here). */
typedef struct arena arena_t;

extern "C" {
    void* arena_allocate_with_header(arena_t* arena, size_t data_size,
                                      uint8_t subtype, uint8_t flags);
    arena_t* get_global_arena();
    /* The explicit seed helper must use the same stable runtime state as
     * generated random calls on every backend. */
    void eshkol_srand48(int64_t seed);
}

/* PRNG state on the heap is just one uint64. The arena header (subtype +
 * length) sits 8 bytes before the returned pointer per the bignum pattern,
 * so callers can validate via the standard subtype check. */
extern "C" uint64_t* eshkol_prng_make(int64_t seed) {
    arena_t* arena = get_global_arena();
    if (!arena) return nullptr;
    void* ptr = arena_allocate_with_header(arena, sizeof(uint64_t),
                                            HEAP_SUBTYPE_PRNG, 0);
    if (!ptr) return nullptr;
    uint64_t* state = static_cast<uint64_t*>(ptr);
    *state = eshkol_prng_state_seed(seed);
    return state;
}

/** Advance `state` and return the next uniform double in [0.0, 1.0). */
extern "C" double eshkol_prng_random(uint64_t* state) {
    if (!state) return 0.0;
    return eshkol_prng_state_random(state);
}

extern "C" int64_t eshkol_prng_random_integer(uint64_t* state, int64_t n) {
    if (!state || n <= 0) return 0;
    /* Multiply-then-floor preserves uniformity for the [0, n) range we
     * need; bias is < 1/2^48 which is well below user-visible. */
    return eshkol_prng_state_integer(state, n);
}

/* Explicit seed for the global PRNG. Callers also need to mark
 * __random_seeded__ so the codegen's auto-seed-from-time doesn't override —
 * that's done by the codegen for set-random-seed! (see codegenSetRandomSeed). */
extern "C" void eshkol_random_seed(int64_t seed) {
    eshkol_srand48(seed);
}
