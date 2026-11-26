/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol OS Phase K2: Freestanding C11 Zero-Alloc Physical Frame Allocator,
 * 3-ISA Page-Table Managers (RV64 Sv39/Sv48, AArch64 4K, x86_64 4-Level),
 * TLB Shootdown Tracking, and MCU MPU Region Manager.
 *
 * Strictly zero heap allocation, zero libc/POSIX/LLVM/unwind symbols,
 * and zero 64-bit division helpers across all 5 freestanding target ISAs.
 */

#include <eshkol/kernel/k2_mmu_frames.h>

/* Compiler-barriered zeroing loop prevents -O2 lowering to external memset. */
static void k2_zero_bytes(void* dst, size_t n) {
    volatile uint8_t* p = (volatile uint8_t*)dst;
    for (size_t i = 0u; i < n; ++i) {
        p[i] = 0u;
    }
}

static int k2_bitmap_test(const k2_frame_pool_t* pool, uint32_t idx) {
    uint32_t word = idx >> 6u;
    uint32_t bit = idx & 63u;
    return (pool->alloc_bitmap[word] & (1ULL << bit)) != 0ULL;
}

static void k2_bitmap_set(k2_frame_pool_t* pool, uint32_t idx) {
    uint32_t word = idx >> 6u;
    uint32_t bit = idx & 63u;
    pool->alloc_bitmap[word] |= (1ULL << bit);
}

static void k2_bitmap_clear(k2_frame_pool_t* pool, uint32_t idx) {
    uint32_t word = idx >> 6u;
    uint32_t bit = idx & 63u;
    pool->alloc_bitmap[word] &= ~(1ULL << bit);
}

static uint32_t k2_popcount64(uint64_t x) {
    uint32_t c = 0u;
    while (x != 0ULL) {
        x &= (x - 1ULL);
        c += 1u;
    }
    return c;
}

static int k2_is_power_of_two(uint64_t v) {
    return (v != 0ULL) && ((v & (v - 1ULL)) == 0ULL);
}

static int k2_ranges_overlap(uint64_t a_base, uint64_t a_size,
                             uint64_t b_base, uint64_t b_size) {
    if (a_size == 0ULL || b_size == 0ULL) {
        return 0;
    }
    uint64_t a_end = a_base + a_size;
    uint64_t b_end = b_base + b_size;
    return (a_base < b_end) && (b_base < a_end);
}

static k2_status_t k2_check_perm_policy(uint32_t perms, uint32_t is_user_space) {
    if ((perms & K2_PERM_WRITE) != 0u && (perms & K2_PERM_EXEC) != 0u) {
        return K2_ERR_WX_VIOLATION;
    }
    if ((perms & K2_PERM_USER) != 0u && (perms & K2_PERM_KERNEL_ONLY) != 0u) {
        return K2_ERR_PRIV_VIOLATION;
    }
    if ((perms & (K2_PERM_READ | K2_PERM_WRITE | K2_PERM_EXEC)) == 0u) {
        return K2_ERR_INVALID_ARG;
    }
    /* Write without read is disallowed across Sv39/Sv48 and canonical W^X policy. */
    if ((perms & K2_PERM_WRITE) != 0u && (perms & K2_PERM_READ) == 0u) {
        return K2_ERR_INVALID_ARG;
    }
    if (is_user_space != 0u) {
        if ((perms & K2_PERM_KERNEL_ONLY) != 0u || (perms & K2_PERM_USER) == 0u) {
            return K2_ERR_PRIV_VIOLATION;
        }
    }
    return K2_OK;
}

static int k2_va_canonical_for_mode(k2_mmu_mode_t mode, uint64_t va, uint32_t is_user_space) {
    if ((va & K2_PAGE_MASK) != 0ULL) {
        return 0;
    }
    switch (mode) {
        case K2_MMU_RV64_SV39:
            if (is_user_space) {
                return va < (1ULL << 38u);
            }
            return (va < (1ULL << 38u)) || (va >= 0xFFFFFFC000000000ULL);
        case K2_MMU_RV64_SV48:
        case K2_MMU_AARCH64_4K:
        case K2_MMU_X86_64_4L:
            if (is_user_space) {
                return va < (1ULL << 47u);
            }
            return (va < (1ULL << 47u)) || (va >= 0xFFFF800000000000ULL);
        default:
            return 0;
    }
}

/* ─────────────────────────── Physical Frame Allocator ─────────────────────────── */

static void k2_meta_reset_mappings(k2_frame_meta_t* meta) {
    meta->map_refcount = 0u;
    meta->user_map_refcount = 0u;
    meta->read_refcount = 0u;
    meta->write_refcount = 0u;
    meta->exec_refcount = 0u;
    meta->mapped_perms = 0u;
    meta->mapped_asid = K2_INVALID_INDEX;
    meta->mapped_user_aspace = NULL;
}

static void k2_meta_recompute_perms(k2_frame_meta_t* meta) {
    uint32_t p = 0u;
    if (meta->read_refcount > 0u)  p |= K2_PERM_READ;
    if (meta->write_refcount > 0u) p |= K2_PERM_WRITE;
    if (meta->exec_refcount > 0u)  p |= K2_PERM_EXEC;
    meta->mapped_perms = p;
}

k2_status_t k2_frame_pool_init(k2_frame_pool_t* pool,
                               uint64_t phys_base,
                               uint32_t total_frames,
                               uint32_t kernel_reserved_count) {
    if (!pool) {
        return K2_ERR_INVALID_ARG;
    }
    if ((phys_base & K2_PAGE_MASK) != 0ULL) {
        return K2_ERR_ALIGNMENT;
    }
    if (total_frames == 0u || total_frames > K2_MAX_FRAMES) {
        return K2_ERR_INVALID_ARG;
    }
    if (kernel_reserved_count > total_frames) {
        return K2_ERR_INVALID_ARG;
    }
    uint64_t span = ((uint64_t)total_frames) << K2_PAGE_SHIFT;
    if (phys_base > (UINT64_MAX - span)) {
        return K2_ERR_INVALID_ARG;
    }

    k2_zero_bytes(pool, sizeof(*pool));
    pool->phys_base = phys_base;
    pool->total_frames = total_frames;
    pool->kernel_reserved_frames = kernel_reserved_count;
    pool->user_allocated_frames = 0u;
    pool->allocated_frames = kernel_reserved_count;
    pool->free_frames = total_frames - kernel_reserved_count;
    pool->free_head = K2_INVALID_INDEX;

    for (uint32_t i = 0u; i < kernel_reserved_count; ++i) {
        pool->frames[i].owner = K2_OWNER_KERNEL;
        pool->frames[i].generation = 1u;
        k2_meta_reset_mappings(&pool->frames[i]);
        pool->frames[i].next_free = K2_INVALID_INDEX;
        k2_bitmap_set(pool, i);
    }

    /* Push free frames in descending order so lowest free index allocates first. */
    for (uint32_t rev = total_frames; rev > kernel_reserved_count; --rev) {
        uint32_t idx = rev - 1u;
        pool->frames[idx].owner = K2_OWNER_FREE;
        pool->frames[idx].generation = 1u;
        k2_meta_reset_mappings(&pool->frames[idx]);
        pool->frames[idx].next_free = pool->free_head;
        pool->free_head = idx;
    }

    return K2_OK;
}

static k2_status_t k2_frame_alloc_internal(k2_frame_pool_t* pool,
                                           k2_frame_owner_t target_owner,
                                           k2_page_frame_t* out_frame) {
    if (!pool || !out_frame) {
        return K2_ERR_INVALID_ARG;
    }
    if (pool->free_frames == 0u || pool->free_head == K2_INVALID_INDEX) {
        return K2_ERR_OUT_OF_FRAMES;
    }

    uint32_t idx = pool->free_head;
    if (idx >= pool->total_frames || k2_bitmap_test(pool, idx) != 0) {
        return K2_ERR_CONSERVATION_VIOLATION;
    }

    k2_frame_meta_t* meta = &pool->frames[idx];
    pool->free_head = meta->next_free;
    meta->next_free = K2_INVALID_INDEX;
    meta->owner = target_owner;
    meta->generation += 1u;
    k2_meta_reset_mappings(meta);
    k2_bitmap_set(pool, idx);

    pool->free_frames -= 1u;
    pool->allocated_frames += 1u;
    if (target_owner == K2_OWNER_KERNEL) {
        pool->kernel_reserved_frames += 1u;
    } else {
        pool->user_allocated_frames += 1u;
    }

    out_frame->phys_addr = pool->phys_base + (((uint64_t)idx) << K2_PAGE_SHIFT);
    out_frame->frame_idx = idx;
    out_frame->generation = meta->generation;
    out_frame->owner = target_owner;
    out_frame->reserved = 0u;
    return K2_OK;
}

k2_status_t k2_frame_alloc_kernel(k2_frame_pool_t* pool,
                                  k2_page_frame_t* out_frame) {
    return k2_frame_alloc_internal(pool, K2_OWNER_KERNEL, out_frame);
}

k2_status_t k2_frame_alloc_user(k2_frame_pool_t* pool,
                                k2_page_frame_t* out_frame) {
    return k2_frame_alloc_internal(pool, K2_OWNER_USER, out_frame);
}

k2_status_t k2_frame_free_user(k2_frame_pool_t* pool,
                               const k2_page_frame_t* frame) {
    if (!pool || !frame) {
        return K2_ERR_INVALID_ARG;
    }
    if (frame->frame_idx >= pool->total_frames) {
        return K2_ERR_INVALID_ARG;
    }

    uint32_t idx = frame->frame_idx;
    k2_frame_meta_t* meta = &pool->frames[idx];

    if (meta->owner == K2_OWNER_KERNEL) {
        return K2_ERR_KERNEL_FRAME_LOCKED;
    }
    if (meta->owner == K2_OWNER_FREE || k2_bitmap_test(pool, idx) == 0) {
        return K2_ERR_DOUBLE_FREE;
    }
    if (frame->generation != meta->generation) {
        return K2_ERR_STALE_HANDLE;
    }
    uint64_t expected_phys = pool->phys_base + (((uint64_t)idx) << K2_PAGE_SHIFT);
    if (frame->phys_addr != expected_phys) {
        return K2_ERR_STALE_HANDLE;
    }
    if (meta->map_refcount != 0u) {
        return K2_ERR_FRAME_STILL_MAPPED;
    }

    k2_bitmap_clear(pool, idx);
    meta->owner = K2_OWNER_FREE;
    meta->generation += 1u;
    k2_meta_reset_mappings(meta);
    meta->next_free = pool->free_head;
    pool->free_head = idx;

    pool->free_frames += 1u;
    pool->allocated_frames -= 1u;
    pool->user_allocated_frames -= 1u;
    return K2_OK;
}

k2_status_t k2_frame_get_kernel_handle(const k2_frame_pool_t* pool,
                                       uint32_t frame_idx,
                                       k2_page_frame_t* out_frame) {
    if (!pool || !out_frame || frame_idx >= pool->total_frames) {
        return K2_ERR_INVALID_ARG;
    }
    const k2_frame_meta_t* meta = &pool->frames[frame_idx];
    if (meta->owner != K2_OWNER_KERNEL) {
        return K2_ERR_INVALID_ARG;
    }
    out_frame->phys_addr = pool->phys_base + (((uint64_t)frame_idx) << K2_PAGE_SHIFT);
    out_frame->frame_idx = frame_idx;
    out_frame->generation = meta->generation;
    out_frame->owner = K2_OWNER_KERNEL;
    out_frame->reserved = 0u;
    return K2_OK;
}

int k2_frame_pool_invariants_hold(const k2_frame_pool_t* pool) {
    if (!pool || pool->total_frames == 0u || pool->total_frames > K2_MAX_FRAMES) {
        return 0;
    }
    if (pool->allocated_frames != (pool->kernel_reserved_frames + pool->user_allocated_frames)) {
        return 0;
    }
    if ((pool->free_frames + pool->allocated_frames) != pool->total_frames) {
        return 0;
    }

    uint32_t pop = 0u;
    for (uint32_t w = 0u; w < K2_BITMAP_WORDS; ++w) {
        pop += k2_popcount64(pool->alloc_bitmap[w]);
    }
    if (pop != pool->allocated_frames) {
        return 0;
    }

    uint32_t counted_free = 0u;
    uint32_t cur = pool->free_head;
    while (cur != K2_INVALID_INDEX) {
        if (cur >= pool->total_frames || counted_free > pool->total_frames) {
            return 0;
        }
        if (k2_bitmap_test(pool, cur) != 0 || pool->frames[cur].owner != K2_OWNER_FREE) {
            return 0;
        }
        counted_free += 1u;
        cur = pool->frames[cur].next_free;
    }
    return counted_free == pool->free_frames;
}

/* ─────────────────────────── 3-ISA Hardware PTE Codecs ─────────────────────────── */

k2_status_t k2_encode_pte(k2_mmu_mode_t mode,
                          uint64_t phys_addr,
                          uint32_t perms,
                          uint64_t* out_raw_pte) {
    if (!out_raw_pte) {
        return K2_ERR_INVALID_ARG;
    }
    if ((phys_addr & K2_PAGE_MASK) != 0ULL) {
        return K2_ERR_ALIGNMENT;
    }
    k2_status_t pol = k2_check_perm_policy(perms, 0u);
    if (pol != K2_OK) {
        return pol;
    }

    uint64_t pte = 0ULL;
    switch (mode) {
        case K2_MMU_RV64_SV39:
        case K2_MMU_RV64_SV48: {
            uint64_t ppn = (phys_addr >> K2_PAGE_SHIFT) << 10u;
            if ((ppn & ~K2_RV64_PTE_PPN_MASK) != 0ULL) {
                return K2_ERR_INVALID_ARG;
            }
            pte = ppn | K2_RV64_PTE_V | K2_RV64_PTE_A;
            if ((perms & K2_PERM_READ) != 0u)   pte |= K2_RV64_PTE_R;
            if ((perms & K2_PERM_WRITE) != 0u)  pte |= (K2_RV64_PTE_W | K2_RV64_PTE_D);
            if ((perms & K2_PERM_EXEC) != 0u)   pte |= K2_RV64_PTE_X;
            if ((perms & K2_PERM_USER) != 0u)   pte |= K2_RV64_PTE_U;
            if ((perms & K2_PERM_GLOBAL) != 0u) pte |= K2_RV64_PTE_G;
            break;
        }
        case K2_MMU_AARCH64_4K: {
            if ((phys_addr & ~K2_AARCH64_PTE_ADDR_MASK) != 0ULL) {
                return K2_ERR_INVALID_ARG;
            }
            pte = (phys_addr & K2_AARCH64_PTE_ADDR_MASK) | K2_AARCH64_PTE_VALID | K2_AARCH64_PTE_AF;
            if ((perms & K2_PERM_USER) != 0u) {
                pte |= K2_AARCH64_PTE_AP_USER | K2_AARCH64_PTE_NG | K2_AARCH64_PTE_PXN;
                if ((perms & K2_PERM_WRITE) == 0u) {
                    pte |= K2_AARCH64_PTE_AP_RO;
                }
                if ((perms & K2_PERM_EXEC) == 0u) {
                    pte |= K2_AARCH64_PTE_UXN;
                }
            } else {
                pte |= K2_AARCH64_PTE_UXN;
                if ((perms & K2_PERM_WRITE) == 0u) {
                    pte |= K2_AARCH64_PTE_AP_RO;
                }
                if ((perms & K2_PERM_EXEC) == 0u) {
                    pte |= K2_AARCH64_PTE_PXN;
                }
            }
            break;
        }
        case K2_MMU_X86_64_4L: {
            if ((phys_addr & ~K2_X86_64_PTE_ADDR_MASK) != 0ULL) {
                return K2_ERR_INVALID_ARG;
            }
            pte = (phys_addr & K2_X86_64_PTE_ADDR_MASK) | K2_X86_64_PTE_P | K2_X86_64_PTE_A;
            if ((perms & K2_PERM_WRITE) != 0u)  pte |= (K2_X86_64_PTE_RW | K2_X86_64_PTE_D);
            if ((perms & K2_PERM_USER) != 0u)   pte |= K2_X86_64_PTE_US;
            if ((perms & K2_PERM_GLOBAL) != 0u) pte |= K2_X86_64_PTE_G;
            if ((perms & K2_PERM_EXEC) == 0u)   pte |= K2_X86_64_PTE_NX;
            break;
        }
        default:
            return K2_ERR_UNSUPPORTED_MMU;
    }

    *out_raw_pte = pte;
    return K2_OK;
}

k2_status_t k2_validate_raw_pte(k2_mmu_mode_t mode,
                                uint64_t raw_pte,
                                uint32_t is_user_space) {
    switch (mode) {
        case K2_MMU_RV64_SV39:
        case K2_MMU_RV64_SV48: {
            if ((raw_pte & K2_RV64_PTE_V) == 0ULL) {
                return K2_ERR_RAW_PTE_INVALID;
            }
            int r = (raw_pte & K2_RV64_PTE_R) != 0ULL;
            int w = (raw_pte & K2_RV64_PTE_W) != 0ULL;
            int x = (raw_pte & K2_RV64_PTE_X) != 0ULL;
            int u = (raw_pte & K2_RV64_PTE_U) != 0ULL;
            if (w && x) {
                return K2_ERR_WX_VIOLATION;
            }
            if ((w && !r) || (!r && !w && !x)) {
                return K2_ERR_RAW_PTE_INVALID;
            }
            if (is_user_space && !u) {
                return K2_ERR_PRIV_VIOLATION;
            }
            if (!is_user_space && u) {
                return K2_ERR_PRIV_VIOLATION;
            }
            return K2_OK;
        }
        case K2_MMU_AARCH64_4K: {
            if ((raw_pte & 0x3ULL) != K2_AARCH64_PTE_VALID) {
                return K2_ERR_RAW_PTE_INVALID;
            }
            int u = (raw_pte & K2_AARCH64_PTE_AP_USER) != 0ULL;
            int w = (raw_pte & K2_AARCH64_PTE_AP_RO) == 0ULL;
            int px = (raw_pte & K2_AARCH64_PTE_PXN) == 0ULL;
            int ux = (raw_pte & K2_AARCH64_PTE_UXN) == 0ULL;
            if (w && (px || ux)) {
                return K2_ERR_WX_VIOLATION;
            }
            /* User page must NEVER be privileged-executable (PXN=1 mandatory). */
            if (u && px) {
                return K2_ERR_PRIV_VIOLATION;
            }
            /* Kernel-only page must NEVER be unprivileged-executable (UXN=1 mandatory). */
            if (!u && ux) {
                return K2_ERR_PRIV_VIOLATION;
            }
            if (is_user_space && !u) {
                return K2_ERR_PRIV_VIOLATION;
            }
            if (!is_user_space && u) {
                return K2_ERR_PRIV_VIOLATION;
            }
            return K2_OK;
        }
        case K2_MMU_X86_64_4L: {
            if ((raw_pte & K2_X86_64_PTE_P) == 0ULL) {
                return K2_ERR_RAW_PTE_INVALID;
            }
            int w = (raw_pte & K2_X86_64_PTE_RW) != 0ULL;
            int x = (raw_pte & K2_X86_64_PTE_NX) == 0ULL;
            int u = (raw_pte & K2_X86_64_PTE_US) != 0ULL;
            if (w && x) {
                return K2_ERR_WX_VIOLATION;
            }
            if (is_user_space && !u) {
                return K2_ERR_PRIV_VIOLATION;
            }
            if (!is_user_space && u) {
                return K2_ERR_PRIV_VIOLATION;
            }
            return K2_OK;
        }
        default:
            return K2_ERR_UNSUPPORTED_MMU;
    }
}

k2_status_t k2_decode_pte(k2_mmu_mode_t mode,
                          uint64_t raw_pte,
                          uint64_t* out_phys_addr,
                          uint32_t* out_perms) {
    if (!out_phys_addr || !out_perms) {
        return K2_ERR_INVALID_ARG;
    }

    uint64_t phys = 0ULL;
    uint32_t perms = 0u;

    switch (mode) {
        case K2_MMU_RV64_SV39:
        case K2_MMU_RV64_SV48: {
            if ((raw_pte & K2_RV64_PTE_V) == 0ULL) {
                return K2_ERR_RAW_PTE_INVALID;
            }
            phys = ((raw_pte & K2_RV64_PTE_PPN_MASK) >> 10u) << K2_PAGE_SHIFT;
            if ((raw_pte & K2_RV64_PTE_R) != 0ULL) perms |= K2_PERM_READ;
            if ((raw_pte & K2_RV64_PTE_W) != 0ULL) perms |= K2_PERM_WRITE;
            if ((raw_pte & K2_RV64_PTE_X) != 0ULL) perms |= K2_PERM_EXEC;
            if ((raw_pte & K2_RV64_PTE_U) != 0ULL) {
                perms |= K2_PERM_USER;
            } else {
                perms |= K2_PERM_KERNEL_ONLY;
            }
            if ((raw_pte & K2_RV64_PTE_G) != 0ULL) perms |= K2_PERM_GLOBAL;
            break;
        }
        case K2_MMU_AARCH64_4K: {
            if ((raw_pte & 0x3ULL) != K2_AARCH64_PTE_VALID) {
                return K2_ERR_RAW_PTE_INVALID;
            }
            phys = raw_pte & K2_AARCH64_PTE_ADDR_MASK;
            perms |= K2_PERM_READ;
            if ((raw_pte & K2_AARCH64_PTE_AP_RO) == 0ULL) perms |= K2_PERM_WRITE;
            if ((raw_pte & K2_AARCH64_PTE_AP_USER) != 0ULL) {
                perms |= K2_PERM_USER;
                if ((raw_pte & K2_AARCH64_PTE_UXN) == 0ULL) perms |= K2_PERM_EXEC;
            } else {
                perms |= K2_PERM_KERNEL_ONLY;
                if ((raw_pte & K2_AARCH64_PTE_PXN) == 0ULL) perms |= K2_PERM_EXEC;
            }
            break;
        }
        case K2_MMU_X86_64_4L: {
            if ((raw_pte & K2_X86_64_PTE_P) == 0ULL) {
                return K2_ERR_RAW_PTE_INVALID;
            }
            phys = raw_pte & K2_X86_64_PTE_ADDR_MASK;
            perms |= K2_PERM_READ;
            if ((raw_pte & K2_X86_64_PTE_RW) != 0ULL) perms |= K2_PERM_WRITE;
            if ((raw_pte & K2_X86_64_PTE_NX) == 0ULL) perms |= K2_PERM_EXEC;
            if ((raw_pte & K2_X86_64_PTE_US) != 0ULL) {
                perms |= K2_PERM_USER;
            } else {
                perms |= K2_PERM_KERNEL_ONLY;
            }
            if ((raw_pte & K2_X86_64_PTE_G) != 0ULL) perms |= K2_PERM_GLOBAL;
            break;
        }
        default:
            return K2_ERR_UNSUPPORTED_MMU;
    }

    *out_phys_addr = phys;
    *out_perms = perms;
    return K2_OK;
}

/* ─────────────────────── Address-Space & TLB Shootdown Manager ─────────────────────── */

static uint32_t k2_tlb_slot_for_va(uint64_t va) {
    return (uint32_t)((va >> K2_PAGE_SHIFT) & (K2_TLB_ENTRIES - 1u));
}

static int k2_aspace_find_slot(const k2_aspace_t* aspace, uint64_t va) {
    for (uint32_t i = 0u; i < K2_MAX_PT_ENTRIES; ++i) {
        if (aspace->entries[i].valid != 0u && aspace->entries[i].va == va) {
            return (int)i;
        }
    }
    return -1;
}

static int k2_aspace_find_free_slot(const k2_aspace_t* aspace) {
    for (uint32_t i = 0u; i < K2_MAX_PT_ENTRIES; ++i) {
        if (aspace->entries[i].valid == 0u) {
            return (int)i;
        }
    }
    return -1;
}

k2_status_t k2_aspace_init(k2_aspace_t* aspace,
                           k2_mmu_mode_t mode,
                           uint32_t asid,
                           uint32_t is_user_space) {
    if (!aspace) {
        return K2_ERR_INVALID_ARG;
    }
    if ((uint32_t)mode >= (uint32_t)K2_MMU_MODE_COUNT) {
        return K2_ERR_UNSUPPORTED_MMU;
    }
    /* Partition ASID domains: kernel spaces use K2_KERNEL_ASID (0), user spaces use 1..K2_INVALID_INDEX-1. */
    if (is_user_space == 0u) {
        if (asid != K2_KERNEL_ASID) {
            return K2_ERR_INVALID_ARG;
        }
    } else {
        if (asid == K2_KERNEL_ASID || asid == K2_INVALID_INDEX) {
            return K2_ERR_INVALID_ARG;
        }
    }
    k2_zero_bytes(aspace, sizeof(*aspace));
    aspace->mode = mode;
    aspace->asid = asid;
    aspace->is_user_space = is_user_space ? 1u : 0u;
    return K2_OK;
}

void k2_aspace_tlb_shootdown_va(k2_aspace_t* aspace, uint64_t va) {
    if (!aspace) {
        return;
    }
    uint64_t page_va = va & ~K2_PAGE_MASK;
    aspace->tlb_shootdowns += 1ULL;
    for (uint32_t i = 0u; i < K2_TLB_ENTRIES; ++i) {
        if (aspace->tlb[i].valid != 0u &&
            aspace->tlb[i].asid == aspace->asid &&
            aspace->tlb[i].va == page_va) {
            aspace->tlb[i].valid = 0u;
            aspace->tlb[i].raw_pte = 0ULL;
            aspace->tlb_invalidated_entries += 1ULL;
        }
    }
}

void k2_aspace_tlb_shootdown_all(k2_aspace_t* aspace) {
    if (!aspace) {
        return;
    }
    aspace->tlb_shootdowns += 1ULL;
    for (uint32_t i = 0u; i < K2_TLB_ENTRIES; ++i) {
        if (aspace->tlb[i].valid != 0u) {
            aspace->tlb[i].valid = 0u;
            aspace->tlb[i].raw_pte = 0ULL;
            aspace->tlb_invalidated_entries += 1ULL;
        }
    }
}

k2_status_t k2_aspace_map(k2_aspace_t* aspace,
                          k2_frame_pool_t* pool,
                          uint64_t va,
                          const k2_page_frame_t* frame,
                          uint32_t perms) {
    if (!aspace || !pool || !frame) {
        return K2_ERR_INVALID_ARG;
    }
    if (!k2_va_canonical_for_mode(aspace->mode, va, aspace->is_user_space)) {
        return K2_ERR_ALIGNMENT;
    }
    k2_status_t pol = k2_check_perm_policy(perms, aspace->is_user_space);
    if (pol != K2_OK) {
        return pol;
    }
    if (frame->frame_idx >= pool->total_frames) {
        return K2_ERR_INVALID_ARG;
    }

    k2_frame_meta_t* meta = &pool->frames[frame->frame_idx];

    /* Hard kernel non-aliasing: kernel-reserved frames may NEVER enter any user address space. */
    if (meta->owner == K2_OWNER_KERNEL) {
        if (aspace->is_user_space != 0u) {
            return K2_ERR_KERNEL_ALIAS;
        }
    } else if (meta->owner == K2_OWNER_FREE || k2_bitmap_test(pool, frame->frame_idx) == 0) {
        return K2_ERR_STALE_HANDLE;
    }

    if (frame->generation != meta->generation) {
        return K2_ERR_STALE_HANDLE;
    }
    uint64_t expected_phys = pool->phys_base + (((uint64_t)frame->frame_idx) << K2_PAGE_SHIFT);
    if (frame->phys_addr != expected_phys) {
        return K2_ERR_STALE_HANDLE;
    }

    /* Reject conflicting-permission aliasing (W vs X across live mappings) and cross-space user aliasing. */
    if (meta->map_refcount > 0u) {
        int new_w = (perms & K2_PERM_WRITE) != 0u;
        int new_x = (perms & K2_PERM_EXEC) != 0u;
        if ((meta->write_refcount > 0u && new_x) || (meta->exec_refcount > 0u && new_w)) {
            return K2_ERR_PERM_ALIAS;
        }
    }
    if (aspace->is_user_space != 0u &&
        meta->user_map_refcount > 0u &&
        (meta->mapped_asid != aspace->asid || meta->mapped_user_aspace != aspace)) {
        return K2_ERR_PERM_ALIAS;
    }

    if (k2_aspace_find_slot(aspace, va) >= 0) {
        return K2_ERR_VA_ALREADY_MAPPED;
    }
    int free_slot = k2_aspace_find_free_slot(aspace);
    if (free_slot < 0) {
        return K2_ERR_PT_FULL;
    }

    uint64_t raw_pte = 0ULL;
    k2_status_t enc = k2_encode_pte(aspace->mode, frame->phys_addr, perms, &raw_pte);
    if (enc != K2_OK) {
        return enc;
    }
    k2_status_t val = k2_validate_raw_pte(aspace->mode, raw_pte, aspace->is_user_space);
    if (val != K2_OK) {
        return val;
    }

    k2_pt_entry_t* ent = &aspace->entries[free_slot];
    ent->valid = 1u;
    ent->frame_idx = frame->frame_idx;
    ent->va = va;
    ent->raw_pte = raw_pte;
    ent->perms = perms;
    ent->reserved = 0u;

    aspace->mapped_count += 1u;
    meta->map_refcount += 1u;
    if ((perms & K2_PERM_READ) != 0u)  meta->read_refcount += 1u;
    if ((perms & K2_PERM_WRITE) != 0u) meta->write_refcount += 1u;
    if ((perms & K2_PERM_EXEC) != 0u)  meta->exec_refcount += 1u;
    k2_meta_recompute_perms(meta);
    if (aspace->is_user_space != 0u) {
        meta->user_map_refcount += 1u;
        meta->mapped_asid = aspace->asid;
        meta->mapped_user_aspace = aspace;
    }

    k2_aspace_tlb_shootdown_va(aspace, va);
    return K2_OK;
}

k2_status_t k2_aspace_remap_perms(k2_aspace_t* aspace,
                                  k2_frame_pool_t* pool,
                                  uint64_t va,
                                  uint32_t new_perms) {
    if (!aspace || !pool) {
        return K2_ERR_INVALID_ARG;
    }
    if ((va & K2_PAGE_MASK) != 0ULL) {
        return K2_ERR_ALIGNMENT;
    }
    k2_status_t pol = k2_check_perm_policy(new_perms, aspace->is_user_space);
    if (pol != K2_OK) {
        return pol;
    }

    int slot = k2_aspace_find_slot(aspace, va);
    if (slot < 0) {
        return K2_ERR_VA_NOT_MAPPED;
    }

    k2_pt_entry_t* ent = &aspace->entries[slot];
    if (ent->frame_idx >= pool->total_frames) {
        return K2_ERR_CONSERVATION_VIOLATION;
    }
    k2_frame_meta_t* meta = &pool->frames[ent->frame_idx];

    /* Check W^X against every other active mapping of this physical frame. */
    uint32_t old_perms = ent->perms;
    uint32_t other_w = meta->write_refcount - (((old_perms & K2_PERM_WRITE) != 0u && meta->write_refcount > 0u) ? 1u : 0u);
    uint32_t other_x = meta->exec_refcount - (((old_perms & K2_PERM_EXEC) != 0u && meta->exec_refcount > 0u) ? 1u : 0u);
    int new_w = (new_perms & K2_PERM_WRITE) != 0u;
    int new_x = (new_perms & K2_PERM_EXEC) != 0u;
    if ((other_w > 0u && new_x) || (other_x > 0u && new_w)) {
        return K2_ERR_PERM_ALIAS;
    }

    uint64_t phys_addr = pool->phys_base + (((uint64_t)ent->frame_idx) << K2_PAGE_SHIFT);
    uint64_t raw_pte = 0ULL;
    k2_status_t enc = k2_encode_pte(aspace->mode, phys_addr, new_perms, &raw_pte);
    if (enc != K2_OK) {
        return enc;
    }
    k2_status_t val = k2_validate_raw_pte(aspace->mode, raw_pte, aspace->is_user_space);
    if (val != K2_OK) {
        return val;
    }

    if ((old_perms & K2_PERM_READ) != 0u && meta->read_refcount > 0u)   meta->read_refcount -= 1u;
    if ((old_perms & K2_PERM_WRITE) != 0u && meta->write_refcount > 0u) meta->write_refcount -= 1u;
    if ((old_perms & K2_PERM_EXEC) != 0u && meta->exec_refcount > 0u)   meta->exec_refcount -= 1u;
    if ((new_perms & K2_PERM_READ) != 0u)  meta->read_refcount += 1u;
    if ((new_perms & K2_PERM_WRITE) != 0u) meta->write_refcount += 1u;
    if ((new_perms & K2_PERM_EXEC) != 0u)  meta->exec_refcount += 1u;
    k2_meta_recompute_perms(meta);

    ent->raw_pte = raw_pte;
    ent->perms = new_perms;

    k2_aspace_tlb_shootdown_va(aspace, va);
    return K2_OK;
}

k2_status_t k2_aspace_unmap(k2_aspace_t* aspace,
                            k2_frame_pool_t* pool,
                            uint64_t va) {
    if (!aspace || !pool) {
        return K2_ERR_INVALID_ARG;
    }
    if ((va & K2_PAGE_MASK) != 0ULL) {
        return K2_ERR_ALIGNMENT;
    }

    int slot = k2_aspace_find_slot(aspace, va);
    if (slot < 0) {
        return K2_ERR_VA_NOT_MAPPED;
    }

    k2_pt_entry_t* ent = &aspace->entries[slot];
    uint32_t fidx = ent->frame_idx;
    if (fidx < pool->total_frames) {
        k2_frame_meta_t* meta = &pool->frames[fidx];
        if ((ent->perms & K2_PERM_READ) != 0u && meta->read_refcount > 0u)   meta->read_refcount -= 1u;
        if ((ent->perms & K2_PERM_WRITE) != 0u && meta->write_refcount > 0u) meta->write_refcount -= 1u;
        if ((ent->perms & K2_PERM_EXEC) != 0u && meta->exec_refcount > 0u)   meta->exec_refcount -= 1u;
        if (aspace->is_user_space != 0u && meta->user_map_refcount > 0u) {
            meta->user_map_refcount -= 1u;
            if (meta->user_map_refcount == 0u) {
                meta->mapped_asid = K2_INVALID_INDEX;
                meta->mapped_user_aspace = NULL;
            }
        }
        if (meta->map_refcount > 0u) {
            meta->map_refcount -= 1u;
            if (meta->map_refcount == 0u) {
                k2_meta_reset_mappings(meta);
            } else {
                k2_meta_recompute_perms(meta);
            }
        }
    }

    ent->valid = 0u;
    ent->frame_idx = K2_INVALID_INDEX;
    ent->va = 0ULL;
    ent->raw_pte = 0ULL;
    ent->perms = 0u;
    if (aspace->mapped_count > 0u) {
        aspace->mapped_count -= 1u;
    }

    k2_aspace_tlb_shootdown_va(aspace, va);
    return K2_OK;
}

k2_status_t k2_aspace_insert_raw_pte(k2_aspace_t* aspace,
                                     uint64_t va,
                                     uint64_t raw_pte) {
    if (!aspace) {
        return K2_ERR_INVALID_ARG;
    }
    if (!k2_va_canonical_for_mode(aspace->mode, va, aspace->is_user_space)) {
        return K2_ERR_ALIGNMENT;
    }
    /* Strict fail-closed validation on raw hardware PTE insertion. */
    k2_status_t val = k2_validate_raw_pte(aspace->mode, raw_pte, aspace->is_user_space);
    if (val != K2_OK) {
        return val;
    }
    if (k2_aspace_find_slot(aspace, va) >= 0) {
        return K2_ERR_VA_ALREADY_MAPPED;
    }
    int free_slot = k2_aspace_find_free_slot(aspace);
    if (free_slot < 0) {
        return K2_ERR_PT_FULL;
    }

    uint64_t phys = 0ULL;
    uint32_t perms = 0u;
    k2_status_t dec = k2_decode_pte(aspace->mode, raw_pte, &phys, &perms);
    if (dec != K2_OK) {
        return dec;
    }

    k2_pt_entry_t* ent = &aspace->entries[free_slot];
    ent->valid = 1u;
    ent->frame_idx = K2_INVALID_INDEX;
    ent->va = va;
    ent->raw_pte = raw_pte;
    ent->perms = perms;
    aspace->mapped_count += 1u;
    k2_aspace_tlb_shootdown_va(aspace, va);
    return K2_OK;
}

static k2_status_t k2_check_access_perms(uint32_t pte_perms, uint32_t access_flags) {
    if ((access_flags & K2_ACCESS_USER) != 0u && (pte_perms & K2_PERM_USER) == 0u) {
        return K2_ERR_ACCESS_DENIED;
    }
    if ((access_flags & K2_ACCESS_READ) != 0u && (pte_perms & K2_PERM_READ) == 0u) {
        return K2_ERR_ACCESS_DENIED;
    }
    if ((access_flags & K2_ACCESS_WRITE) != 0u && (pte_perms & K2_PERM_WRITE) == 0u) {
        return K2_ERR_ACCESS_DENIED;
    }
    if ((access_flags & K2_ACCESS_EXEC) != 0u && (pte_perms & K2_PERM_EXEC) == 0u) {
        return K2_ERR_ACCESS_DENIED;
    }
    return K2_OK;
}

k2_status_t k2_aspace_translate(k2_aspace_t* aspace,
                                uint64_t va,
                                uint32_t access_flags,
                                uint64_t* out_phys_addr) {
    if (!aspace || !out_phys_addr) {
        return K2_ERR_INVALID_ARG;
    }
    uint64_t page_va = va & ~K2_PAGE_MASK;
    uint64_t page_off = va & K2_PAGE_MASK;
    uint32_t tslot = k2_tlb_slot_for_va(page_va);
    k2_tlb_entry_t* tlb = &aspace->tlb[tslot];

    if (tlb->valid != 0u && tlb->asid == aspace->asid && tlb->va == page_va) {
        aspace->tlb_hits += 1ULL;
        k2_status_t val = k2_validate_raw_pte(aspace->mode, tlb->raw_pte, aspace->is_user_space);
        if (val != K2_OK) {
            return val;
        }
        k2_status_t acc = k2_check_access_perms(tlb->perms, access_flags);
        if (acc != K2_OK) {
            return acc;
        }
        *out_phys_addr = tlb->phys_addr + page_off;
        return K2_OK;
    }

    aspace->tlb_misses += 1ULL;
    int slot = k2_aspace_find_slot(aspace, page_va);
    if (slot < 0) {
        return K2_ERR_VA_NOT_MAPPED;
    }

    const k2_pt_entry_t* ent = &aspace->entries[slot];
    k2_status_t val = k2_validate_raw_pte(aspace->mode, ent->raw_pte, aspace->is_user_space);
    if (val != K2_OK) {
        return val;
    }

    uint64_t phys_base = 0ULL;
    uint32_t decoded_perms = 0u;
    k2_status_t dec = k2_decode_pte(aspace->mode, ent->raw_pte, &phys_base, &decoded_perms);
    if (dec != K2_OK) {
        return dec;
    }

    /* Fill hardware TLB cache entry only after raw PTE validation passes. */
    tlb->valid = 1u;
    tlb->asid = aspace->asid;
    tlb->va = page_va;
    tlb->raw_pte = ent->raw_pte;
    tlb->phys_addr = phys_base;
    tlb->perms = decoded_perms;

    k2_status_t acc = k2_check_access_perms(decoded_perms, access_flags);
    if (acc != K2_OK) {
        return acc;
    }

    *out_phys_addr = phys_base + page_off;
    return K2_OK;
}

/* ─────────────────────────── MCU MPU Region Manager ─────────────────────────── */

k2_status_t k2_mpu_table_init(k2_mpu_table_t* tbl,
                              uint64_t vector_table_base,
                              uint64_t vector_table_size,
                              uint64_t stack_guard_base,
                              uint64_t stack_guard_size) {
    if (!tbl) {
        return K2_ERR_INVALID_ARG;
    }
    if (vector_table_size == 0ULL || stack_guard_size == 0ULL) {
        return K2_ERR_INVALID_ARG;
    }
    if (vector_table_base > (UINT64_MAX - vector_table_size) ||
        stack_guard_base > (UINT64_MAX - stack_guard_size)) {
        return K2_ERR_INVALID_ARG;
    }
    k2_zero_bytes(tbl, sizeof(*tbl));
    tbl->vector_table_base = vector_table_base;
    tbl->vector_table_size = vector_table_size;
    tbl->stack_guard_base = stack_guard_base;
    tbl->stack_guard_size = stack_guard_size;
    return K2_OK;
}

k2_status_t k2_mpu_add_region(k2_mpu_table_t* tbl,
                              uint32_t region_id,
                              uint64_t base,
                              uint64_t size,
                              uint32_t perms) {
    if (!tbl) {
        return K2_ERR_INVALID_ARG;
    }
    if (region_id >= K2_MAX_MPU_REGIONS) {
        return K2_ERR_MPU_REGION_LIMIT;
    }
    k2_status_t pol = k2_check_perm_policy(perms, 0u);
    if (pol != K2_OK) {
        return pol;
    }
    if (size < K2_MPU_MIN_REGION_BYTES || !k2_is_power_of_two(size)) {
        return K2_ERR_MPU_ALIGNMENT;
    }
    if ((base & (size - 1ULL)) != 0ULL || base > (UINT64_MAX - size)) {
        return K2_ERR_MPU_ALIGNMENT;
    }

    /* Reject ANY region overlapping reserved hardware vector table or kernel stack guard. */
    if (k2_ranges_overlap(base, size, tbl->vector_table_base, tbl->vector_table_size) ||
        k2_ranges_overlap(base, size, tbl->stack_guard_base, tbl->stack_guard_size)) {
        return K2_ERR_MPU_RESERVED_GUARD;
    }

    /* Reject overlap with any other active MPU region. */
    for (uint32_t i = 0u; i < K2_MAX_MPU_REGIONS; ++i) {
        if (i == region_id) {
            continue;
        }
        if (tbl->regions[i].active != 0u &&
            k2_ranges_overlap(base, size, tbl->regions[i].base, tbl->regions[i].size)) {
            return K2_ERR_MPU_OVERLAP;
        }
    }

    int was_active = (tbl->regions[region_id].active != 0u);
    tbl->regions[region_id].region_id = region_id;
    tbl->regions[region_id].active = 1u;
    tbl->regions[region_id].base = base;
    tbl->regions[region_id].size = size;
    tbl->regions[region_id].perms = perms;
    if (!was_active) {
        tbl->region_count += 1u;
    }
    return K2_OK;
}

k2_status_t k2_mpu_remove_region(k2_mpu_table_t* tbl, uint32_t region_id) {
    if (!tbl || region_id >= K2_MAX_MPU_REGIONS) {
        return K2_ERR_MPU_REGION_LIMIT;
    }
    if (tbl->regions[region_id].active != 0u) {
        tbl->regions[region_id].active = 0u;
        tbl->regions[region_id].base = 0ULL;
        tbl->regions[region_id].size = 0ULL;
        tbl->regions[region_id].perms = 0u;
        if (tbl->region_count > 0u) {
            tbl->region_count -= 1u;
        }
    }
    return K2_OK;
}

k2_status_t k2_mpu_check_access(const k2_mpu_table_t* tbl,
                                uint64_t addr,
                                uint64_t len,
                                uint32_t access_flags) {
    if (!tbl || len == 0ULL || addr > (UINT64_MAX - len)) {
        return K2_ERR_INVALID_ARG;
    }
    if (k2_ranges_overlap(addr, len, tbl->vector_table_base, tbl->vector_table_size) ||
        k2_ranges_overlap(addr, len, tbl->stack_guard_base, tbl->stack_guard_size)) {
        return K2_ERR_MPU_RESERVED_GUARD;
    }
    uint64_t end = addr + len;
    for (uint32_t i = 0u; i < K2_MAX_MPU_REGIONS; ++i) {
        const k2_mpu_region_t* r = &tbl->regions[i];
        if (r->active == 0u) {
            continue;
        }
        uint64_t r_end = r->base + r->size;
        if (addr >= r->base && end <= r_end) {
            return k2_check_access_perms(r->perms, access_flags);
        }
    }
    return K2_ERR_ACCESS_DENIED;
}
