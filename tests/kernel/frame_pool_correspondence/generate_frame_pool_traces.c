/*
 * generate_frame_pool_traces.c
 * Purpose: run random operation traces on the native frame allocator
 * (lib/kernel/k2_mmu_frames.c) and record its complete observable state after
 * every operation, for correspondence with FramePool in EshkolKernelK2.lean.
 *
 * Correspondence:
 *   k2_frame_pool_init(base, T, K)  <->  pool_init (owners i < K kernel, generations 1)
 *   k2_frame_alloc_user -> OK idx   <->  allocUser idx (native chooses idx; the model
 *                                        must accept it)
 *   k2_frame_alloc_user -> refused  <->  no index is accepted by allocUser
 *   k2_frame_free_user(handle)      <->  freeUser idx gen, where a handle whose
 *                                        phys_addr does not match its index maps to
 *                                        an out-of-range index (the model has no
 *                                        physical address)
 *   k2_frame_alloc_kernel           <->  no model operation
 *
 * Output, one line per operation:
 *   trace <n> total <T> reserved <K>
 *   alloc_user => <STATUS> <idx> | <state>
 *   alloc_kernel => <STATUS> <idx> | <state>
 *   free <idx> <gen> <phys_consistent 0|1> => <STATUS> | <state>
 *   <state> = counts <free> <allocated> owners <o0..> generations <g0..> invariants <0|1>
 * Deterministic in the seed. Usage: <seed> <traces> <operations> <kernel_alloc 0|1>
 */
#include <eshkol/kernel/k2_mmu_frames.h>
#include <stdio.h>
#include <stdlib.h>

#define MAXIMUM_TOTAL_FRAMES 48u
#define MAXIMUM_HANDLES 256u
#define PHYSICAL_BASE 0x100000ULL

static uint64_t random_state;
static uint32_t random_below(uint32_t bound) {
    random_state ^= random_state << 13; random_state ^= random_state >> 7; random_state ^= random_state << 17;
    return (uint32_t)(random_state % bound);
}

static const char* status_name(k2_status_t s) {
    switch (s) {
    case K2_OK: return "OK";
    case K2_ERR_INVALID_ARG: return "INVALID_ARG";
    case K2_ERR_OUT_OF_FRAMES: return "OUT_OF_FRAMES";
    case K2_ERR_DOUBLE_FREE: return "DOUBLE_FREE";
    case K2_ERR_STALE_HANDLE: return "STALE_HANDLE";
    case K2_ERR_KERNEL_FRAME_LOCKED: return "KERNEL_FRAME_LOCKED";
    case K2_ERR_FRAME_STILL_MAPPED: return "FRAME_STILL_MAPPED";
    case K2_ERR_CONSERVATION_VIOLATION: return "CONSERVATION_VIOLATION";
    default: return "OTHER";
    }
}

static void print_state(const k2_frame_pool_t* pool) {
    printf(" | counts %u %u owners", pool->free_frames, pool->allocated_frames);
    for (uint32_t i = 0; i < pool->total_frames; i++) printf(" %u", (unsigned)pool->frames[i].owner);
    printf(" generations");
    for (uint32_t i = 0; i < pool->total_frames; i++) printf(" %u", pool->frames[i].generation);
    printf(" invariants %d\n", k2_frame_pool_invariants_hold(pool) ? 1 : 0);
}

int main(int argc, char** argv) {
    if (argc != 5) { fprintf(stderr, "usage: %s <seed> <traces> <operations> <kernel_alloc 0|1>\n", argv[0]); return 1; }
    random_state = strtoull(argv[1], 0, 10) | 1u;
    unsigned trace_count = (unsigned)atoi(argv[2]), operation_count = (unsigned)atoi(argv[3]);
    int kernel_alloc = atoi(argv[4]) != 0;
    static k2_frame_pool_t pool;
    static k2_page_frame_t handle[MAXIMUM_HANDLES];
    for (unsigned t = 0; t < trace_count; t++) {
        uint32_t total = 1u + random_below(MAXIMUM_TOTAL_FRAMES);
        uint32_t reserved = random_below(total + 1u);
        if (k2_frame_pool_init(&pool, PHYSICAL_BASE, total, reserved) != K2_OK) return 3;
        printf("trace %u total %u reserved %u\n", t, total, reserved);
        unsigned handle_count = 0;
        for (uint32_t i = 0; i < reserved && handle_count < MAXIMUM_HANDLES; i++)
            if (k2_frame_get_kernel_handle(&pool, i, &handle[handle_count]) == K2_OK) handle_count++;
        for (unsigned k = 0; k < operation_count; k++) {
            uint32_t choice = random_below(10);
            if (choice < 4 || handle_count == 0) {
                k2_page_frame_t f;
                k2_status_t st = k2_frame_alloc_user(&pool, &f);
                if (st == K2_OK && handle_count < MAXIMUM_HANDLES) handle[handle_count++] = f;
                printf("alloc_user => %s %u", status_name(st), st == K2_OK ? f.frame_idx : 0u);
            } else if (choice == 4 && kernel_alloc) {
                k2_page_frame_t f;
                k2_status_t st = k2_frame_alloc_kernel(&pool, &f);
                if (st == K2_OK && handle_count < MAXIMUM_HANDLES) handle[handle_count++] = f;
                printf("alloc_kernel => %s %u", status_name(st), st == K2_OK ? f.frame_idx : 0u);
            } else {
                k2_page_frame_t f = handle[random_below(handle_count)];
                uint32_t tamper = random_below(12);
                if (tamper == 0) f.generation += 1u;
                if (tamper == 1) f.generation -= 1u;
                if (tamper == 2) f.phys_addr += K2_PAGE_SIZE;
                if (tamper == 3) f.frame_idx = random_below(total + 3u);
                int phys_consistent = f.phys_addr == PHYSICAL_BASE + ((uint64_t)f.frame_idx << K2_PAGE_SHIFT);
                k2_status_t st = k2_frame_free_user(&pool, &f);
                printf("free %u %u %d => %s", f.frame_idx, f.generation, phys_consistent, status_name(st));
            }
            print_state(&pool);
        }
    }
    return 0;
}
