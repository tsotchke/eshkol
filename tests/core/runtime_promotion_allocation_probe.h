#ifndef ESHKOL_TEST_PROMOTION_ALLOCATION_PROBE_H
#define ESHKOL_TEST_PROMOTION_ALLOCATION_PROBE_H
#include <cstdint>
struct PromotionAllocationProbe {
    uint64_t malloc_calls, calloc_calls, realloc_calls, aligned_alloc_calls;
    uint64_t posix_memalign_calls, new_calls, new_array_calls, exception_allocations;
};
void promotion_allocation_probe_arm() noexcept;
PromotionAllocationProbe promotion_allocation_probe_disarm() noexcept;
#endif
