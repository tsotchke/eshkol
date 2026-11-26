/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol OS Phase U3: OALR backing page-hook ABI.
 *
 * Allows an OS userspace runtime (or test harness) to supply zeroed,
 * page-aligned backing memory to Eshkol's OALR arena allocator without
 * depending on POSIX malloc/free/mmap.
 *
 * Contract:
 *   1. alloc_pages(num_pages, flags, user_ctx) returns either a page-aligned,
 *      zero-initialized region of (num_pages * page_size) bytes atomically, or
 *      NULL on exhaustion. Partial allocations are never permitted.
 *   2. free_pages(base, num_pages, user_ctx) releases the exact (base,
 *      num_pages) span previously returned by alloc_pages and returns 0 on
 *      success or non-zero on failure.
 *   3. When page hooks are installed, OALR never falls back to malloc/free on
 *      exhaustion or contract violation; it records the exact error status in
 *      thread-local storage (queryable via eshkol_os_oalr_last_error()) and
 *      returns NULL cleanly.
 */

#ifndef ESHKOL_CORE_OS_PAGE_HOOKS_H
#define ESHKOL_CORE_OS_PAGE_HOOKS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum eshkol_os_page_status {
    ESHKOL_OS_PAGE_OK = 0,
    ESHKOL_OS_PAGE_ERR_OOM = 1,
    ESHKOL_OS_PAGE_ERR_INVALID_ARG = 2,
    ESHKOL_OS_PAGE_ERR_ALIGNMENT = 3,
    ESHKOL_OS_PAGE_ERR_NOT_ZEROED = 4,
    ESHKOL_OS_PAGE_ERR_UNREGISTERED = 5,
    ESHKOL_OS_PAGE_ERR_FREE_FAILED = 6
} eshkol_os_page_status_t;

typedef enum eshkol_os_page_flags {
    ESHKOL_OS_PAGE_FLAG_NONE = 0u,
    ESHKOL_OS_PAGE_FLAG_ZEROED = 1u << 0,
    ESHKOL_OS_PAGE_FLAG_GUARD = 1u << 1
} eshkol_os_page_flags_t;

typedef void* (*eshkol_os_alloc_pages_fn)(size_t num_pages, uint32_t flags, void* user_ctx);
typedef int (*eshkol_os_free_pages_fn)(void* base, size_t num_pages, void* user_ctx);

typedef struct eshkol_os_page_hooks {
    size_t page_size;
    eshkol_os_alloc_pages_fn alloc_pages;
    eshkol_os_free_pages_fn free_pages;
    void* user_ctx;
    uint32_t verify_zero_on_alloc;
    uint32_t reserved;
} eshkol_os_page_hooks_t;

typedef struct eshkol_os_oalr_stats {
    uint64_t alloc_calls;
    uint64_t free_calls;
    uint64_t pages_allocated;
    uint64_t pages_freed;
    uint64_t live_pages;
    uint64_t peak_live_pages;
    uint64_t oom_failures;
    uint64_t contract_violations;
} eshkol_os_oalr_stats_t;

/* Install OS page hooks for all subsequent OALR arena allocations, or pass NULL
 * to uninstall and restore default hosted allocation. Returns ESHKOL_OS_PAGE_OK
 * on success or ESHKOL_OS_PAGE_ERR_INVALID_ARG if the descriptor is malformed. */
int eshkol_os_install_page_hooks(const eshkol_os_page_hooks_t* hooks);

/* Non-zero while OS page hooks are active. */
int eshkol_os_page_hooks_installed(void);

/* Active OS page size in bytes, or 0 when hooks are not installed. */
size_t eshkol_os_page_size(void);

/* Thread-local status from the most recent OALR page-hook operation. */
eshkol_os_page_status_t eshkol_os_oalr_last_error(void);
void eshkol_os_oalr_clear_error(void);

/* Snapshot or reset aggregate OALR page-hook accounting counters. */
void eshkol_os_oalr_get_stats(eshkol_os_oalr_stats_t* out);
void eshkol_os_oalr_reset_stats(void);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_OS_PAGE_HOOKS_H */
