/*
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 *
 * Eshkol OS Phase U3 Gate (`os_u3_oalr_page_hooks`):
 *   Structural shadow-ledger verification of OALR OS page-hook backing:
 *     1. Hook descriptor validation (page size power-of-two >= 4096, non-null fns)
 *     2. 10,000 arena lifecycle cycles (create, aligned/zeroed allocations,
 *        multi-block growth, deep nested push/pop scopes exercising both inline
 *        and overflow scope frames, reset, destroy) tracked by an independent
 *        (base, num_pages) shadow ledger asserting exact 0-page drift, 0
 *        unmatched frees, 0 double frees, flat peak live pages, and exact
 *        counter parity.
 *     3. 1,000 OALR lexical region cycles (region_create / region_push /
 *        arena_allocate / region_pop, including nested regions) with exact
 *        0-page drift.
 *     4. Hard-ceiling OOM exhaustion: explicit ESHKOL_OS_PAGE_ERR_OOM, atomic
 *        cleanup with 0 leaked pages, and zero fallback to malloc.
 *     5. Dirty-page contract violation (verify_zero_on_alloc=1): explicit
 *        ESHKOL_OS_PAGE_ERR_NOT_ZEROED and atomic release with 0 leaked pages.
 *     6. Misaligned-page contract violation: explicit ESHKOL_OS_PAGE_ERR_ALIGNMENT
 *        and atomic release with 0 leaked pages.
 *     7. Clean uninstall and restoration of hosted allocation path.
 */

#include "../../lib/core/arena_memory.h"
#include <eshkol/core/os_page_hooks.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace {

int g_checks = 0;

int fail(const char* msg) {
    std::cerr << "FAIL: " << msg << std::endl;
    return 1;
}

#define CHECK(cond, msg) do { if (!(cond)) return fail(msg); ++g_checks; } while (0)

struct ShadowPageLedger {
    size_t page_size = 4096;
    size_t max_live_pages = 0;  // 0 = unlimited
    bool inject_dirty_next = false;
    bool inject_misaligned_next = false;

    std::unordered_map<void*, size_t> live_spans;
    std::unordered_map<void*, void*> misaligned_backing;  // returned_ptr -> real_ptr
    uint64_t alloc_calls = 0;
    uint64_t free_calls = 0;
    uint64_t total_pages_alloc = 0;
    uint64_t total_pages_free = 0;
    uint64_t live_pages = 0;
    uint64_t peak_live_pages = 0;
    uint64_t unmatched_frees = 0;
    uint64_t size_mismatch_frees = 0;
    uint64_t double_frees = 0;
    uint64_t oom_rejects = 0;
    uint64_t nonzero_flag_violations = 0;

    static void* alloc_cb(size_t num_pages, uint32_t flags, void* ctx) {
        auto* self = static_cast<ShadowPageLedger*>(ctx);
        self->alloc_calls++;
        if ((flags & ESHKOL_OS_PAGE_FLAG_ZEROED) == 0) {
            self->nonzero_flag_violations++;
        }
        if (self->max_live_pages > 0 &&
            (self->live_pages + num_pages) > self->max_live_pages) {
            self->oom_rejects++;
            return nullptr;
        }

        const size_t total_bytes = num_pages * self->page_size;
        void* raw = nullptr;
        const size_t alloc_bytes = self->inject_misaligned_next
            ? (total_bytes + self->page_size)
            : total_bytes;
        if (posix_memalign(&raw, self->page_size, alloc_bytes) != 0 || !raw) {
            self->oom_rejects++;
            return nullptr;
        }
        std::memset(raw, 0, alloc_bytes);

        void* ret = raw;
        if (self->inject_misaligned_next) {
            self->inject_misaligned_next = false;
            ret = static_cast<uint8_t*>(raw) + 64;
            self->misaligned_backing[ret] = raw;
        } else if (self->inject_dirty_next) {
            self->inject_dirty_next = false;
            static_cast<uint8_t*>(ret)[total_bytes - 1] = 0xA5;
        }

        self->live_spans[ret] = num_pages;
        self->total_pages_alloc += num_pages;
        self->live_pages += num_pages;
        if (self->live_pages > self->peak_live_pages) {
            self->peak_live_pages = self->live_pages;
        }
        return ret;
    }

    static int free_cb(void* base, size_t num_pages, void* ctx) {
        auto* self = static_cast<ShadowPageLedger*>(ctx);
        self->free_calls++;
        auto it = self->live_spans.find(base);
        if (it == self->live_spans.end()) {
            self->unmatched_frees++;
            self->double_frees++;
            return -1;
        }
        if (it->second != num_pages) {
            self->size_mismatch_frees++;
            return -2;
        }
        self->live_spans.erase(it);
        self->total_pages_free += num_pages;
        self->live_pages -= num_pages;

        auto mit = self->misaligned_backing.find(base);
        if (mit != self->misaligned_backing.end()) {
            std::free(mit->second);
            self->misaligned_backing.erase(mit);
        } else {
            std::free(base);
        }
        return 0;
    }
};

struct ScopedPageHookGuard {
    ~ScopedPageHookGuard() {
        (void)eshkol_os_install_page_hooks(nullptr);
    }
};

}  // namespace

int main() {
    // 1. Validate malformed hook descriptors fail closed
    {
        eshkol_os_page_hooks_t bad{};
        bad.page_size = 2048;  // below 4096
        bad.alloc_pages = &ShadowPageLedger::alloc_cb;
        bad.free_pages = &ShadowPageLedger::free_cb;
        CHECK(eshkol_os_install_page_hooks(&bad) == ESHKOL_OS_PAGE_ERR_INVALID_ARG,
              "reject page_size < 4096");
        CHECK(eshkol_os_page_hooks_installed() == 0, "hooks remain uninstalled");

        bad.page_size = 6000;  // non-power-of-two
        CHECK(eshkol_os_install_page_hooks(&bad) == ESHKOL_OS_PAGE_ERR_INVALID_ARG,
              "reject non-power-of-two page_size");

        bad.page_size = 4096;
        bad.alloc_pages = nullptr;
        CHECK(eshkol_os_install_page_hooks(&bad) == ESHKOL_OS_PAGE_ERR_INVALID_ARG,
              "reject null alloc_pages");

        bad.alloc_pages = &ShadowPageLedger::alloc_cb;
        bad.free_pages = nullptr;
        CHECK(eshkol_os_install_page_hooks(&bad) == ESHKOL_OS_PAGE_ERR_INVALID_ARG,
              "reject null free_pages");
    }

    // 2. 10,000-cycle arena stress test with independent shadow ledger
    {
        ShadowPageLedger ledger;
        ScopedPageHookGuard guard;
        eshkol_os_oalr_reset_stats();
        eshkol_os_page_hooks_t hooks{};
        hooks.page_size = 4096;
        hooks.alloc_pages = &ShadowPageLedger::alloc_cb;
        hooks.free_pages = &ShadowPageLedger::free_cb;
        hooks.user_ctx = &ledger;
        hooks.verify_zero_on_alloc = 1;

        CHECK(eshkol_os_install_page_hooks(&hooks) == ESHKOL_OS_PAGE_OK,
              "install valid 4KiB page hooks");
        CHECK(eshkol_os_page_hooks_installed() == 1, "hooks installed");
        CHECK(eshkol_os_page_size() == 4096, "page size reported as 4096");

        for (int cycle = 0; cycle < 10000; ++cycle) {
            arena_t* a = arena_create(4096);
            CHECK(a != nullptr, "arena_create succeeds under page hooks");
            CHECK(a->page_backed, "arena marked page_backed");

            // Push scope and force multi-block growth inside scope
            arena_push_scope(a);
            void* p1 = arena_allocate_aligned(a, 2048, 64);
            void* p2 = arena_allocate_zeroed(a, 5000);  // forces new block
            CHECK(p1 != nullptr && p2 != nullptr, "multi-block allocation succeeds");
            CHECK(((uintptr_t)p1 & 63u) == 0, "64-byte alignment respected");
            std::memset(p1, 0x5A, 2048);
            std::memset(p2, 0x3C, 5000);
            arena_pop_scope(a);

            // Every 500th cycle, push 180 nested scopes to exercise BOTH
            // inline scope slots and page-backed overflow scope slabs
            if ((cycle % 500) == 0) {
                for (int d = 0; d < 180; ++d) {
                    arena_push_scope(a);
                    void* s = arena_allocate(a, 32);
                    CHECK(s != nullptr, "nested scope allocation succeeds");
                }
                for (int d = 0; d < 180; ++d) {
                    arena_pop_scope(a);
                }
            }

            // Exercise arena_reset periodically
            if ((cycle % 1000) == 1) {
                void* extra = arena_allocate(a, 8192);
                CHECK(extra != nullptr, "pre-reset allocation succeeds");
                arena_reset(a);
                void* post = arena_allocate(a, 128);
                CHECK(post != nullptr, "post-reset allocation succeeds");
            }

            arena_destroy(a);
            CHECK(ledger.live_pages == 0 && ledger.live_spans.empty(),
                  "zero live pages after every single arena_destroy cycle");
        }

        eshkol_os_oalr_stats_t stats{};
        eshkol_os_oalr_get_stats(&stats);
        CHECK(ledger.live_pages == 0, "0-page drift after 10,000 cycles");
        CHECK(ledger.live_spans.empty(), "shadow ledger empty after 10,000 cycles");
        CHECK(ledger.unmatched_frees == 0, "zero unmatched frees");
        CHECK(ledger.size_mismatch_frees == 0, "zero size-mismatch frees");
        CHECK(ledger.double_frees == 0, "zero double frees");
        CHECK(ledger.nonzero_flag_violations == 0, "ZEROED flag always passed");
        CHECK(ledger.total_pages_alloc == ledger.total_pages_free,
              "exact alloc/free page parity");
        CHECK(ledger.total_pages_alloc >= 20000,
              "every cycle backed by real page hooks");
        CHECK(ledger.peak_live_pages <= 64, "flat peak page footprint across 10,000 cycles");
        CHECK(stats.live_pages == 0, "stats.live_pages == 0");
        CHECK(stats.pages_allocated == ledger.total_pages_alloc,
              "stats.pages_allocated matches shadow ledger");
        CHECK(stats.pages_freed == ledger.total_pages_free,
              "stats.pages_freed matches shadow ledger");
        CHECK(stats.contract_violations == 0, "zero contract violations");
        CHECK(stats.oom_failures == 0, "zero OOM failures in nominal run");
    }

    // 3. 1,000-cycle OALR lexical region_create/push/pop test (including nesting)
    {
        // Ensure process-global fallback arena and thread-local OOM condition
        // reserve are initialized before measuring per-region page allocations
        // so region_push() does not lazily allocate process/thread-lifetime
        // singletons inside the test shadow ledger.
        (void)eshkol_os_install_page_hooks(nullptr);
        (void)get_global_arena();
        eshkol_reserve_allocation_failure_condition();

        ShadowPageLedger ledger;
        ScopedPageHookGuard guard;
        eshkol_os_oalr_reset_stats();
        eshkol_os_page_hooks_t hooks{};
        hooks.page_size = 4096;
        hooks.alloc_pages = &ShadowPageLedger::alloc_cb;
        hooks.free_pages = &ShadowPageLedger::free_cb;
        hooks.user_ctx = &ledger;
        hooks.verify_zero_on_alloc = 1;
        CHECK(eshkol_os_install_page_hooks(&hooks) == ESHKOL_OS_PAGE_OK,
              "install page hooks for region test");

        for (int i = 0; i < 1000; ++i) {
            eshkol_region_t* outer = region_create("u3_outer", 4096);
            CHECK(outer != nullptr, "outer region_create succeeds");
            region_push(outer);
            void* o1 = arena_allocate(outer->arena, 3000);
            CHECK(o1 != nullptr, "outer region alloc succeeds");

            eshkol_region_t* inner = region_create("u3_inner", 4096);
            CHECK(inner != nullptr, "inner region_create succeeds");
            region_push(inner);
            void* i1 = arena_allocate(inner->arena, 6000);
            CHECK(i1 != nullptr, "inner region multi-block alloc succeeds");
            region_pop();
            region_pop();
            CHECK(ledger.live_pages == 0 && ledger.live_spans.empty(),
                  "0-page drift after nested region_pop");
        }
        CHECK(ledger.total_pages_alloc == ledger.total_pages_free,
              "region cycle alloc/free parity");
        CHECK(ledger.unmatched_frees == 0 && ledger.double_frees == 0,
              "region cycle zero ledger anomalies");
    }

    // 4. Hard-ceiling OOM exhaustion: explicit error, atomic rollback, zero malloc fallback
    {
        ShadowPageLedger ledger;
        ScopedPageHookGuard guard;
        ledger.max_live_pages = 3;  // 1 arena control page + 2 block pages max
        eshkol_os_oalr_reset_stats();
        eshkol_os_page_hooks_t hooks{};
        hooks.page_size = 4096;
        hooks.alloc_pages = &ShadowPageLedger::alloc_cb;
        hooks.free_pages = &ShadowPageLedger::free_cb;
        hooks.user_ctx = &ledger;
        hooks.verify_zero_on_alloc = 1;
        CHECK(eshkol_os_install_page_hooks(&hooks) == ESHKOL_OS_PAGE_OK,
              "install OOM ceiling hooks");

        // Request an arena whose initial block needs 3 pages (+ 1 control page = 4 > 3):
        // control page must be atomically rolled back with 0 leaked pages!
        eshkol_os_oalr_clear_error();
        arena_t* too_big = arena_create(3 * 4096);
        CHECK(too_big == nullptr, "arena_create returns NULL on OOM");
        CHECK(eshkol_os_oalr_last_error() == ESHKOL_OS_PAGE_ERR_OOM,
              "OOM sets ESHKOL_OS_PAGE_ERR_OOM");
        CHECK(ledger.live_pages == 0 && ledger.live_spans.empty(),
              "atomic rollback leaves 0 leaked pages when initial block OOMs");

        // Create a 1-block arena (2 live pages: 1 control + 1 block), then attempt growth needing 2 pages (4 > 3)
        arena_t* ok = arena_create(2048);
        CHECK(ok != nullptr, "bounded-pool arena_create succeeds within ceiling");
        CHECK(ledger.live_pages == 2, "exactly 2 pages live (control + initial block)");

        eshkol_os_oalr_clear_error();
        void* overflow = arena_allocate_aligned(ok, 8192, 16);
        CHECK(overflow == nullptr, "growth returns NULL on OOM without fallback");
        CHECK(eshkol_os_oalr_last_error() == ESHKOL_OS_PAGE_ERR_OOM,
              "growth OOM sets ESHKOL_OS_PAGE_ERR_OOM");
        CHECK(ledger.live_pages == 2, "existing arena pages remain intact after OOM");

        arena_destroy(ok);
        CHECK(ledger.live_pages == 0 && ledger.live_spans.empty(),
              "0 live pages after destroying surviving arena");

        eshkol_os_oalr_stats_t stats{};
        eshkol_os_oalr_get_stats(&stats);
        CHECK(stats.oom_failures >= 2, "stats recorded OOM failures");
        CHECK(stats.live_pages == 0, "stats live_pages == 0 after OOM test");
    }

    // 5. Dirty-page contract violation: explicit ERR_NOT_ZEROED + atomic release
    {
        ShadowPageLedger ledger;
        ScopedPageHookGuard guard;
        ledger.inject_dirty_next = true;
        eshkol_os_oalr_reset_stats();
        eshkol_os_page_hooks_t hooks{};
        hooks.page_size = 4096;
        hooks.alloc_pages = &ShadowPageLedger::alloc_cb;
        hooks.free_pages = &ShadowPageLedger::free_cb;
        hooks.user_ctx = &ledger;
        hooks.verify_zero_on_alloc = 1;
        CHECK(eshkol_os_install_page_hooks(&hooks) == ESHKOL_OS_PAGE_OK,
              "install dirty-page verification hooks");

        eshkol_os_oalr_clear_error();
        arena_t* rejected = arena_create(4096);
        CHECK(rejected == nullptr, "dirty page rejected cleanly");
        CHECK(eshkol_os_oalr_last_error() == ESHKOL_OS_PAGE_ERR_NOT_ZEROED,
              "dirty page sets ESHKOL_OS_PAGE_ERR_NOT_ZEROED");
        CHECK(ledger.live_pages == 0 && ledger.live_spans.empty(),
              "dirty page atomically freed with 0 leaks");

        eshkol_os_oalr_stats_t stats{};
        eshkol_os_oalr_get_stats(&stats);
        CHECK(stats.contract_violations == 1, "dirty page counted as contract violation");
    }

    // 6. Misaligned-page contract violation: explicit ERR_ALIGNMENT + atomic release
    {
        ShadowPageLedger ledger;
        ScopedPageHookGuard guard;
        ledger.inject_misaligned_next = true;
        eshkol_os_oalr_reset_stats();
        eshkol_os_page_hooks_t hooks{};
        hooks.page_size = 4096;
        hooks.alloc_pages = &ShadowPageLedger::alloc_cb;
        hooks.free_pages = &ShadowPageLedger::free_cb;
        hooks.user_ctx = &ledger;
        hooks.verify_zero_on_alloc = 1;
        CHECK(eshkol_os_install_page_hooks(&hooks) == ESHKOL_OS_PAGE_OK,
              "install alignment verification hooks");

        eshkol_os_oalr_clear_error();
        arena_t* rejected = arena_create(4096);
        CHECK(rejected == nullptr, "misaligned page rejected cleanly");
        CHECK(eshkol_os_oalr_last_error() == ESHKOL_OS_PAGE_ERR_ALIGNMENT,
              "misaligned page sets ESHKOL_OS_PAGE_ERR_ALIGNMENT");
        CHECK(ledger.live_pages == 0 && ledger.live_spans.empty(),
              "misaligned page atomically freed with 0 leaks");

        eshkol_os_oalr_stats_t stats{};
        eshkol_os_oalr_get_stats(&stats);
        CHECK(stats.contract_violations == 1, "misaligned page counted as contract violation");
    }

    // 7. Uninstall hooks and verify hosted allocation path resumes cleanly
    {
        CHECK(eshkol_os_install_page_hooks(nullptr) == ESHKOL_OS_PAGE_OK,
              "uninstall page hooks");
        CHECK(eshkol_os_page_hooks_installed() == 0, "hooks uninstalled");
        CHECK(eshkol_os_page_size() == 0, "page size 0 when uninstalled");

        arena_t* hosted = arena_create(4096);
        CHECK(hosted != nullptr, "hosted arena_create succeeds after uninstall");
        CHECK(!hosted->page_backed, "hosted arena not page_backed");
        void* p = arena_allocate(hosted, 256);
        CHECK(p != nullptr, "hosted allocation succeeds");
        arena_destroy(hosted);
    }

    std::cout << "PASS: os_u3_oalr_page_hooks_test (" << g_checks << " checks)" << std::endl;
    return 0;
}
