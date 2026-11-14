/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Runtime region helpers: resident owners.
 *
 * This unit owns the explicit, thread-bound resident arenas (open, enter,
 * leave, hold, export, compact) that DBSP-style resident state is built on.
 * Compaction and import/export run the evacuator transaction declared in
 * runtime_regions_internal.h.
 */

#include "arena_memory.h"
#include "../../inc/eshkol/logger.h"
#include "../../inc/eshkol/exhaustive_dispatch.h"
#include "../../inc/eshkol/core/logic.h"       // eshkol_substitution_t / _fact_t / _knowledge_base_t
#include "../../inc/eshkol/core/inference.h"   // eshkol_factor_graph_t / _factor_t
#include "../../inc/eshkol/core/workspace.h"   // eshkol_workspace_t / _workspace_module_t
#include "../../inc/eshkol/core/rational.h"    // eshkol_rational_t (bignum-backed exact rationals)

#include <cstring>
#include <cstdlib>
#include <new>
#include <unordered_map>
#include <vector>
#include <utility>
#include <atomic>
#include <mutex>
#include <array>

#include "runtime_regions_internal.h"

using namespace eshkol::regions_internal;

// Explicit, thread-bound resident arenas. Only scoped library wrappers expose
// these primitives: user snapshots are generation-checked integers, never raw
// pointers into a compactable arena. Borrowed roots exist only during an active
// operation; observations are copied back to the caller's allocation domain.
namespace {
struct ResidentRoot { uint64_t token = 0; eshkol_tagged_value_t value{}; };
struct ResidentOwner {
    uint64_t token = 0;
    arena_t* arena = nullptr;
    arena_t* saved_domain = nullptr;
    arena_t* outside = nullptr;
    uint64_t mark = 0;
    uint64_t threshold = 0;
    uint64_t compactions = 0;
    uint64_t copied = 0;
    bool active = false;
    std::vector<ResidentRoot> roots;
    ~ResidentOwner() { if (arena) arena_destroy(arena); }
};
static thread_local std::array<ResidentOwner, 32> resident_owners;
static std::atomic<uint64_t> resident_owner_generation{1};
static std::atomic<uint64_t> resident_root_generation{1};
static ResidentOwner* resident_find(uint64_t token) {
    const uint64_t slot = token & 255u;
    if (!slot || slot > resident_owners.size()) return nullptr;
    auto& owner = resident_owners[slot - 1];
    return owner.arena && owner.token == token ? &owner : nullptr;
}
static ResidentRoot* resident_root(ResidentOwner* owner, uint64_t token) {
    const uint64_t slot = token & 4095u;
    if (!owner || !slot || slot > owner->roots.size()) return nullptr;
    auto& root = owner->roots[slot - 1];
    return root.token == token ? &root : nullptr;
}
static bool resident_any_active() {
    for (const auto& owner : resident_owners) if (owner.active) return true;
    return false;
}
static bool resident_copy(arena_t* target, const arena_t* source,
                          const arena_t* second, eshkol_tagged_value_t* values, uint64_t n,
                          bool import_or_export = false, const arena_t* preserved = nullptr) {
    if (source == target && !second) return true;
    try {
        EvacFwdMap forwards;
        EvacState state;
        state.target = target; state.boundary_idx = 0;
        state.span_arena = source; state.second_span_arena = second;
        state.abort_on_refusal = true; state.fwd = &forwards;
        state.resident_preserved_arena = preserved;
        state.resident_copy_mutable = import_or_export;
        if (import_or_export) {
            // Copy data even when it predates the caller's immediate nursery.
            // Otherwise an exported nested string could still alias a retained
            // row, or an imported row could borrow a caller-owned mutable value.
            state.resident_sources.push_back(get_global_arena_shared());
            state.resident_sources.push_back(eshkol_literal_arena());
            for (uint64_t i = 0; i < __region_stack_depth; ++i)
                state.resident_sources.push_back(__region_stack[i]->arena);
        }
        return evac_transaction(state, values, n).ok;
    } catch (const std::bad_alloc&) { return false; }
}
// Helpers return before a language error is raised: longjmp never skips a C++
// vector/map destructor. Failed imports and compactions publish no root changes.
static uint64_t resident_hold_impl(ResidentOwner* owner, eshkol_tagged_value_t value) {
    try {
        size_t slot = 0;
        while (slot < owner->roots.size() && owner->roots[slot].token) ++slot;
        if (slot == owner->roots.size()) {
            if (slot == 4095) return 0;
            owner->roots.emplace_back();
        }
        const uint64_t generation = resident_root_generation.fetch_add(1);
        if (generation >= (uint64_t{1} << 51)) return 0;
        const arena_t* source = eshkol_current_arena();
        if (!resident_copy(owner->arena, source, nullptr, &value, 1, true, owner->arena)) return 0;
        const uint64_t token = (generation << 12) | (slot + 1);
        owner->roots[slot].value = value; owner->roots[slot].token = token;
        return token;
    } catch (const std::bad_alloc&) { return 0; }
}
static bool resident_compact_impl(ResidentOwner* owner) {
    arena_t* fresh = arena_create(65536);
    if (!fresh) return false;
    bool ok = false;
    try {
        std::vector<eshkol_tagged_value_t> values;
        for (const auto& root : owner->roots) if (root.token) values.push_back(root.value);
        ok = resident_copy(fresh, owner->arena, nullptr, values.data(), values.size());
        if (ok) {
            size_t i = 0;
            for (auto& root : owner->roots) if (root.token) root.value = values[i++];
            arena_t* old = owner->arena; owner->arena = fresh; fresh = nullptr;
            ++owner->compactions; owner->copied += arena_get_used_memory(owner->arena);
            arena_destroy(old);
        }
    } catch (const std::bad_alloc&) { ok = false; }
    if (fresh) arena_destroy(fresh);
    return ok;
}
static void resident_error(const char* message) {
    // A refusal can itself occur while exporting an exception. Build its
    // replacement condition in the caller's domain so it cannot dangle after
    // owner compaction/closure. Public wrappers leave the borrow on every exit.
    arena_t* saved = eshkol_memctx_current()->allocation_domain;
    for (const auto& owner : resident_owners)
        if (owner.active) { eshkol_memctx_current()->allocation_domain = owner.outside; break; }
    eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR, "DBSP resident: %s", message);
    eshkol_memctx_current()->allocation_domain = saved;
}
}
extern "C" int64_t eshkol_resident_open(int64_t threshold) {
    if (threshold < 65536 || threshold > 1073741824 || resident_any_active() ||
        __ad_tape_depth || __current_ad_tape) {
        resident_error("invalid compaction threshold or active borrowed/tape context"); return 0;
    }
    for (size_t i = 0; i < resident_owners.size(); ++i) {
        auto& owner = resident_owners[i];
        if (owner.arena) continue;
        const uint64_t generation = resident_owner_generation.fetch_add(1);
        if (generation >= (uint64_t{1} << 55)) break;
        owner.arena = arena_create(65536);
        if (!owner.arena) break;
        owner.token = (generation << 8) | (i + 1); owner.threshold = threshold;
        owner.compactions = 0; owner.copied = 0; owner.roots.clear();
        return static_cast<int64_t>(owner.token);
    }
    resident_error("owner allocation failed or 32-owner limit reached"); return 0;
}
extern "C" void eshkol_resident_close(int64_t token) {
    auto* owner = resident_find(token);
    if (!owner || owner->active) { resident_error("stale owner or close during borrowed operation"); return; }
    arena_destroy(owner->arena); owner->arena = nullptr; owner->token = 0;
    std::vector<ResidentRoot>().swap(owner->roots);
}
extern "C" void eshkol_resident_enter(int64_t token) {
    auto* owner = resident_find(token);
    if (!owner || resident_any_active() || __ad_tape_depth || __current_ad_tape ||
        s_parallel_depth.load(std::memory_order_acquire) || arena_is_worker_thread()) {
        resident_error("invalid owner, reentrant borrow, parallel context or active tape"); return;
    }
    owner->outside = eshkol_current_arena();
    owner->saved_domain = eshkol_memctx_current()->allocation_domain; owner->mark = __region_stack_depth;
    owner->active = true; eshkol_memctx_current()->allocation_domain = owner->arena;
}
extern "C" void eshkol_resident_leave(int64_t token) {
    auto* owner = resident_find(token);
    if (!owner || !owner->active) { resident_error("leave without active owner"); return; }
    eshkol_region_unwind_to(owner->mark, nullptr, 0);
    eshkol_memctx_current()->allocation_domain = owner->saved_domain; owner->active = false;
}
extern "C" int64_t eshkol_resident_hold(int64_t token, const arena_tagged_cons_cell_t* holder) {
    auto* owner = resident_find(token);
    if (!owner || !holder || __ad_tape_depth || __current_ad_tape ||
        (!owner->active && resident_any_active())) {
        resident_error("invalid root import context"); return 0;
    }
    const uint64_t root = resident_hold_impl(owner, holder->car);
    if (!root) resident_error("root limit, unsupported movable value or allocation failure");
    return static_cast<int64_t>(root);
}
extern "C" void eshkol_resident_get(int64_t token, int64_t root_token, arena_tagged_cons_cell_t* holder) {
    auto* owner = resident_find(token); auto* root = resident_root(owner, root_token);
    if (!root || !holder || !owner->active) { resident_error("stale root or read outside borrowed operation"); return; }
    holder->car = root->value;
}
extern "C" void eshkol_resident_export(int64_t token, arena_tagged_cons_cell_t* holder) {
    auto* owner = resident_find(token);
    if (!owner || !owner->active || !holder) { resident_error("export outside borrowed operation"); return; }
    eshkol_tagged_value_t value = holder->car;
    const bool ok = resident_copy(owner->outside, owner->arena, eshkol_current_arena(), &value, 1, true);
    if (!ok) { resident_error("unsupported observation or export allocation failure"); return; }
    holder->car = value;
}
extern "C" void eshkol_resident_release(int64_t token, int64_t root_token) {
    auto* owner = resident_find(token); auto* root = resident_root(owner, root_token);
    if (!root) { resident_error("stale or foreign snapshot handle"); return; }
    root->token = 0; root->value = {};
}
extern "C" void eshkol_resident_compact(int64_t token, int force) {
    auto* owner = resident_find(token);
    if (!owner || resident_any_active() || __ad_tape_depth || __current_ad_tape) {
        resident_error("compaction requires an idle owner and no active tape"); return;
    }
    if (!force && arena_get_used_memory(owner->arena) <= owner->threshold) return;
    if (!resident_compact_impl(owner)) resident_error("compaction refused; retained snapshots remain valid");
}
extern "C" int64_t eshkol_resident_stat(int64_t token, int field) {
    auto* owner = resident_find(token);
    if (!owner) { resident_error("stale owner handle"); return 0; }
    switch (field) {
    case 0: { int64_t count = 0; for (const auto& root : owner->roots) if (root.token) ++count; return count; }
    case 1: return static_cast<int64_t>(arena_get_used_memory(owner->arena));
    case 2: return static_cast<int64_t>(owner->compactions);
    case 3: return static_cast<int64_t>(owner->copied);
    case 4: return static_cast<int64_t>(owner->threshold);
    default: resident_error("unknown statistic"); return 0;
    }
}
