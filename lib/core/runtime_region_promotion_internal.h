#ifndef ESHKOL_RUNTIME_REGION_PROMOTION_INTERNAL_H
#define ESHKOL_RUNTIME_REGION_PROMOTION_INTERNAL_H
#include "arena_memory.h"

void* eshkol_region_allocate_quiet(arena_t*, size_t size, size_t alignment) noexcept;
size_t eshkol_parameter_promotion_size() noexcept;

// Runtime-private atomic copy. Outputs may be the true destination and overlap
// the input; all promotion and native scratch cleanup precedes publication.
// Zero count permits null arrays. No-promotion copies allocate nothing.
int32_t eshkol_region_copy_tagged_checked(eshkol_tagged_value_t* out,
    const void* destination_owner, const eshkol_tagged_value_t* source,
    uint64_t count) noexcept;

#ifdef ESHKOL_PROMOTION_TESTING
void eshkol_promotion_test_emergency_reset() noexcept;
uint64_t eshkol_promotion_test_emergency_transfers() noexcept;
struct eshkol_promotion_test_stats {
    uint64_t attempts[5]; // map, worklist, span ledger, roots scratch, target
    uint64_t live_bytes[4];
    uint64_t peak_bytes[4];
    uint64_t target_bytes;
};
// Fail this and all subsequent allocations at the selected site; -1 selects
// the combined bookkeeping stream. Reset retains live accounting for old maps.
void eshkol_promotion_test_arm(int site, int64_t fail_after) noexcept;
void eshkol_promotion_test_reset() noexcept;
eshkol_promotion_test_stats eshkol_promotion_test_snapshot() noexcept;
#endif
#endif
