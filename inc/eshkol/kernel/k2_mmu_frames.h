/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol OS Phase K2: Verified Zero-Alloc Physical Frame Allocator,
 * 3-ISA Hardware Page-Table Managers (RV64 Sv39/Sv48, AArch64 4K,
 * x86_64 4-Level), TLB Shootdown Tracking, and MCU MPU Region Manager.
 *
 * Pure freestanding C11/C++20 ABI with zero dynamic heap allocation,
 * zero C library dependencies, and zero unwind runtime symbols.
 */

#ifndef ESHKOL_KERNEL_K2_MMU_FRAMES_H
#define ESHKOL_KERNEL_K2_MMU_FRAMES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ABI version and compile-time static capacities (zero heap allocation). */
#define K2_ABI_VERSION          1u
#define K2_PAGE_SHIFT           12u
#define K2_PAGE_SIZE            4096ULL
#define K2_PAGE_MASK            (K2_PAGE_SIZE - 1ULL)

#define K2_MAX_FRAMES           2048u
#define K2_BITMAP_WORDS         ((K2_MAX_FRAMES + 63u) / 64u)
#define K2_MAX_PT_ENTRIES       512u
#define K2_TLB_ENTRIES          64u
#define K2_MAX_MPU_REGIONS      8u
#define K2_MPU_MIN_REGION_BYTES 32ULL
#define K2_KERNEL_ASID          0u
#define K2_INVALID_INDEX        0xFFFFFFFFu

/* Canonical permission bits. */
#define K2_PERM_READ            (1u << 0)
#define K2_PERM_WRITE           (1u << 1)
#define K2_PERM_EXEC            (1u << 2)
#define K2_PERM_USER            (1u << 3)
#define K2_PERM_KERNEL_ONLY     (1u << 4)
#define K2_PERM_GLOBAL          (1u << 5)

/* Access request flags for translation and MPU checks. */
#define K2_ACCESS_READ          (1u << 0)
#define K2_ACCESS_WRITE         (1u << 1)
#define K2_ACCESS_EXEC          (1u << 2)
#define K2_ACCESS_USER          (1u << 3)

/* Hardware PTE bit constants: RISC-V 64 Sv39 / Sv48. */
#define K2_RV64_PTE_V           (1ULL << 0)
#define K2_RV64_PTE_R           (1ULL << 1)
#define K2_RV64_PTE_W           (1ULL << 2)
#define K2_RV64_PTE_X           (1ULL << 3)
#define K2_RV64_PTE_U           (1ULL << 4)
#define K2_RV64_PTE_G           (1ULL << 5)
#define K2_RV64_PTE_A           (1ULL << 6)
#define K2_RV64_PTE_D           (1ULL << 7)
#define K2_RV64_PTE_PPN_MASK    0x003FFFFFFFFFFC00ULL

/* Hardware PTE bit constants: AArch64 4 KiB L3 page descriptor. */
#define K2_AARCH64_PTE_VALID    0x3ULL
#define K2_AARCH64_PTE_AP_USER  (1ULL << 6)
#define K2_AARCH64_PTE_AP_RO    (1ULL << 7)
#define K2_AARCH64_PTE_AF       (1ULL << 10)
#define K2_AARCH64_PTE_NG       (1ULL << 11)
#define K2_AARCH64_PTE_PXN      (1ULL << 53)
#define K2_AARCH64_PTE_UXN      (1ULL << 54)
#define K2_AARCH64_PTE_ADDR_MASK 0x0000FFFFFFFFF000ULL

/* Hardware PTE bit constants: x86_64 4-level paging. */
#define K2_X86_64_PTE_P         (1ULL << 0)
#define K2_X86_64_PTE_RW        (1ULL << 1)
#define K2_X86_64_PTE_US        (1ULL << 2)
#define K2_X86_64_PTE_A         (1ULL << 5)
#define K2_X86_64_PTE_D         (1ULL << 6)
#define K2_X86_64_PTE_G         (1ULL << 8)
#define K2_X86_64_PTE_NX        (1ULL << 63)
#define K2_X86_64_PTE_ADDR_MASK 0x000FFFFFFFFFF000ULL

typedef enum k2_mmu_mode {
    K2_MMU_RV64_SV39  = 0,
    K2_MMU_RV64_SV48  = 1,
    K2_MMU_AARCH64_4K = 2,
    K2_MMU_X86_64_4L  = 3,
    K2_MMU_MODE_COUNT = 4
} k2_mmu_mode_t;

typedef enum k2_frame_owner {
    K2_OWNER_FREE   = 0,
    K2_OWNER_KERNEL = 1,
    K2_OWNER_USER   = 2
} k2_frame_owner_t;

typedef enum k2_status {
    K2_OK                         = 0,
    K2_ERR_INVALID_ARG            = 1,
    K2_ERR_UNSUPPORTED_MMU        = 2,
    K2_ERR_OUT_OF_FRAMES          = 3,
    K2_ERR_DOUBLE_FREE            = 4,
    K2_ERR_STALE_HANDLE           = 5,
    K2_ERR_KERNEL_FRAME_LOCKED    = 6,
    K2_ERR_FRAME_STILL_MAPPED     = 7,
    K2_ERR_WX_VIOLATION           = 8,
    K2_ERR_PRIV_VIOLATION         = 9,
    K2_ERR_KERNEL_ALIAS           = 10,
    K2_ERR_PERM_ALIAS             = 11,
    K2_ERR_VA_ALREADY_MAPPED      = 12,
    K2_ERR_VA_NOT_MAPPED          = 13,
    K2_ERR_PT_FULL                = 14,
    K2_ERR_ALIGNMENT              = 15,
    K2_ERR_ACCESS_DENIED          = 16,
    K2_ERR_RAW_PTE_INVALID        = 17,
    K2_ERR_MPU_OVERLAP            = 18,
    K2_ERR_MPU_RESERVED_GUARD     = 19,
    K2_ERR_MPU_REGION_LIMIT       = 20,
    K2_ERR_MPU_ALIGNMENT          = 21,
    K2_ERR_CONSERVATION_VIOLATION = 22
} k2_status_t;

/* Linear physical page frame handle. */
typedef struct k2_page_frame {
    uint64_t phys_addr;
    uint32_t frame_idx;
    uint32_t generation;
    k2_frame_owner_t owner;
    uint32_t reserved;
} k2_page_frame_t;

struct k2_aspace;

/* Per-frame internal accounting slot. */
typedef struct k2_frame_meta {
    k2_frame_owner_t owner;
    uint32_t generation;
    uint32_t map_refcount;
    uint32_t user_map_refcount;
    uint32_t read_refcount;
    uint32_t write_refcount;
    uint32_t exec_refcount;
    uint32_t mapped_perms;
    uint32_t mapped_asid;
    uint32_t next_free;
    const struct k2_aspace* mapped_user_aspace;
} k2_frame_meta_t;

/* Static boot-sized physical frame pool with O(1) free-list + bitmap guard. */
typedef struct k2_frame_pool {
    uint64_t phys_base;
    uint32_t total_frames;
    uint32_t free_frames;
    uint32_t allocated_frames;
    uint32_t kernel_reserved_frames;
    uint32_t user_allocated_frames;
    uint32_t free_head;
    uint64_t alloc_bitmap[K2_BITMAP_WORDS];
    k2_frame_meta_t frames[K2_MAX_FRAMES];
} k2_frame_pool_t;

typedef struct k2_pt_entry {
    uint32_t valid;
    uint32_t frame_idx;
    uint64_t va;
    uint64_t raw_pte;
    uint32_t perms;
    uint32_t reserved;
} k2_pt_entry_t;

typedef struct k2_tlb_entry {
    uint32_t valid;
    uint32_t asid;
    uint64_t va;
    uint64_t raw_pte;
    uint64_t phys_addr;
    uint32_t perms;
    uint32_t reserved;
} k2_tlb_entry_t;

typedef struct k2_aspace {
    k2_mmu_mode_t mode;
    uint32_t asid;
    uint32_t is_user_space;
    uint32_t mapped_count;
    uint64_t tlb_hits;
    uint64_t tlb_misses;
    uint64_t tlb_shootdowns;
    uint64_t tlb_invalidated_entries;
    k2_pt_entry_t entries[K2_MAX_PT_ENTRIES];
    k2_tlb_entry_t tlb[K2_TLB_ENTRIES];
} k2_aspace_t;

typedef struct k2_mpu_region {
    uint32_t region_id;
    uint32_t active;
    uint64_t base;
    uint64_t size;
    uint32_t perms;
    uint32_t reserved;
} k2_mpu_region_t;

typedef struct k2_mpu_table {
    uint64_t vector_table_base;
    uint64_t vector_table_size;
    uint64_t stack_guard_base;
    uint64_t stack_guard_size;
    uint32_t region_count;
    uint32_t reserved;
    k2_mpu_region_t regions[K2_MAX_MPU_REGIONS];
} k2_mpu_table_t;

/* Physical frame allocator API. */
k2_status_t k2_frame_pool_init(k2_frame_pool_t* pool,
                               uint64_t phys_base,
                               uint32_t total_frames,
                               uint32_t kernel_reserved_count);
k2_status_t k2_frame_alloc_kernel(k2_frame_pool_t* pool,
                                  k2_page_frame_t* out_frame);
k2_status_t k2_frame_alloc_user(k2_frame_pool_t* pool,
                                k2_page_frame_t* out_frame);
k2_status_t k2_frame_free_user(k2_frame_pool_t* pool,
                               const k2_page_frame_t* frame);
k2_status_t k2_frame_get_kernel_handle(const k2_frame_pool_t* pool,
                                       uint32_t frame_idx,
                                       k2_page_frame_t* out_frame);
int k2_frame_pool_invariants_hold(const k2_frame_pool_t* pool);

/* 3-ISA hardware PTE encoding, decoding, and raw bit validation. */
k2_status_t k2_encode_pte(k2_mmu_mode_t mode,
                          uint64_t phys_addr,
                          uint32_t perms,
                          uint64_t* out_raw_pte);
k2_status_t k2_decode_pte(k2_mmu_mode_t mode,
                          uint64_t raw_pte,
                          uint64_t* out_phys_addr,
                          uint32_t* out_perms);
k2_status_t k2_validate_raw_pte(k2_mmu_mode_t mode,
                                uint64_t raw_pte,
                                uint32_t is_user_space);

/* Address-space page-table manager and TLB shootdown API. */
k2_status_t k2_aspace_init(k2_aspace_t* aspace,
                           k2_mmu_mode_t mode,
                           uint32_t asid,
                           uint32_t is_user_space);
k2_status_t k2_aspace_map(k2_aspace_t* aspace,
                          k2_frame_pool_t* pool,
                          uint64_t va,
                          const k2_page_frame_t* frame,
                          uint32_t perms);
k2_status_t k2_aspace_remap_perms(k2_aspace_t* aspace,
                                  k2_frame_pool_t* pool,
                                  uint64_t va,
                                  uint32_t new_perms);
k2_status_t k2_aspace_unmap(k2_aspace_t* aspace,
                            k2_frame_pool_t* pool,
                            uint64_t va);
k2_status_t k2_aspace_insert_raw_pte(k2_aspace_t* aspace,
                                     uint64_t va,
                                     uint64_t raw_pte);
void k2_aspace_tlb_shootdown_va(k2_aspace_t* aspace, uint64_t va);
void k2_aspace_tlb_shootdown_all(k2_aspace_t* aspace);
k2_status_t k2_aspace_translate(k2_aspace_t* aspace,
                                uint64_t va,
                                uint32_t access_flags,
                                uint64_t* out_phys_addr);

/* MCU MPU region manager API (Cortex-M4 and RV32 PMP/MPU). */
k2_status_t k2_mpu_table_init(k2_mpu_table_t* tbl,
                              uint64_t vector_table_base,
                              uint64_t vector_table_size,
                              uint64_t stack_guard_base,
                              uint64_t stack_guard_size);
k2_status_t k2_mpu_add_region(k2_mpu_table_t* tbl,
                              uint32_t region_id,
                              uint64_t base,
                              uint64_t size,
                              uint32_t perms);
k2_status_t k2_mpu_remove_region(k2_mpu_table_t* tbl, uint32_t region_id);
k2_status_t k2_mpu_check_access(const k2_mpu_table_t* tbl,
                                uint64_t addr,
                                uint64_t len,
                                uint32_t access_flags);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_KERNEL_K2_MMU_FRAMES_H */
