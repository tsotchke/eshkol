/*
 * Incremental Distributed DBSP Z-Set Knowledge Graph Engine
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <eshkol/core/dbsp_zset_kg.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace {

struct TripleKey {
    uint64_t s = 0;
    uint64_t p = 0;
    uint64_t o = 0;

    bool operator==(const TripleKey& k) const {
        return s == k.s && p == k.p && o == k.o;
    }

    bool operator<(const TripleKey& k) const {
        if (s != k.s) return s < k.s;
        if (p != k.p) return p < k.p;
        return o < k.o;
    }
};

inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

struct TripleKeyHash {
    size_t operator()(const TripleKey& k) const {
        uint64_t h = splitmix64(k.s);
        h ^= splitmix64(k.p + 0x517cc1b727220a95ULL);
        h ^= splitmix64(k.o + 0x9e3779b97f4a7c15ULL);
        return static_cast<size_t>(splitmix64(h));
    }
};

inline void put_u64(std::vector<uint8_t>& out, uint64_t v) {
    for (int i = 0; i < 8; i++) {
        out.push_back(static_cast<uint8_t>((v >> (i * 8)) & 0xFFu));
    }
}

inline uint64_t fnv1a_64(const uint8_t* data, size_t len) {
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= static_cast<uint64_t>(data[i]);
        h *= 1099511628211ULL;
    }
    return h;
}

} /* anonymous namespace */

struct eshkol_zset {
    uint32_t num_shards = 1;
    std::vector<std::unordered_map<TripleKey, int64_t, TripleKeyHash>> shards;

    explicit eshkol_zset(uint32_t ns) : num_shards(ns > 0 ? ns : 1u), shards(num_shards) {}

    uint32_t shard_for(const TripleKey& k) const {
        return static_cast<uint32_t>(TripleKeyHash{}(k) % num_shards);
    }

    void add(const TripleKey& k, int64_t delta) {
        if (delta == 0) return;
        auto& mp = shards[shard_for(k)];
        auto it = mp.find(k);
        if (it == mp.end()) {
            mp.emplace(k, delta);
        } else {
            int64_t nw = it->second + delta;
            if (nw == 0) {
                mp.erase(it); /* Immediate tombstone pruning: zero zombie nodes */
            } else {
                it->second = nw;
            }
        }
    }

    void add_custom(const TripleKey& k, int64_t delta, eshkol_zset_merge_fn_t fn, void* ctx) {
        auto& mp = shards[shard_for(k)];
        auto it = mp.find(k);
        int64_t existing = (it == mp.end()) ? 0 : it->second;
        int64_t nw = fn(existing, delta, ctx);
        if (nw == 0) {
            if (it != mp.end()) mp.erase(it);
        } else if (it == mp.end()) {
            mp.emplace(k, nw);
        } else {
            it->second = nw;
        }
    }

    int64_t weight(const TripleKey& k) const {
        const auto& mp = shards[shard_for(k)];
        auto it = mp.find(k);
        return (it == mp.end()) ? 0 : it->second;
    }

    size_t active_count() const {
        size_t total = 0;
        for (const auto& mp : shards) total += mp.size();
        return total;
    }

    std::vector<std::pair<TripleKey, int64_t>> sorted_entries() const {
        std::vector<std::pair<TripleKey, int64_t>> entries;
        entries.reserve(active_count());
        for (const auto& mp : shards) {
            for (const auto& kv : mp) {
                entries.emplace_back(kv.first, kv.second);
            }
        }
        std::sort(entries.begin(), entries.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        return entries;
    }
};

namespace {

/* Equijoin accumulator: appends (a.s, join_pred, b.o) with weight w_a * w_b
 * for all pairs with a.o == b.s into `out`. */
void join_accumulate(const eshkol_zset* a,
                     const eshkol_zset* b,
                     uint64_t join_pred,
                     eshkol_zset* out) {
    if (!a || !b || !out || a->active_count() == 0 || b->active_count() == 0) return;

    /* Index smaller side on join key */
    if (b->active_count() <= a->active_count()) {
        std::unordered_map<uint64_t, std::vector<std::pair<uint64_t, int64_t>>> b_by_s;
        for (const auto& sh : b->shards) {
            for (const auto& kv : sh) {
                b_by_s[kv.first.s].emplace_back(kv.first.o, kv.second);
            }
        }
        for (const auto& sh : a->shards) {
            for (const auto& kv : sh) {
                auto it = b_by_s.find(kv.first.o);
                if (it == b_by_s.end()) continue;
                for (const auto& bo : it->second) {
                    out->add({kv.first.s, join_pred, bo.first}, kv.second * bo.second);
                }
            }
        }
    } else {
        std::unordered_map<uint64_t, std::vector<std::pair<uint64_t, int64_t>>> a_by_o;
        for (const auto& sh : a->shards) {
            for (const auto& kv : sh) {
                a_by_o[kv.first.o].emplace_back(kv.first.s, kv.second);
            }
        }
        for (const auto& sh : b->shards) {
            for (const auto& kv : sh) {
                auto it = a_by_o.find(kv.first.s);
                if (it == a_by_o.end()) continue;
                for (const auto& as : it->second) {
                    out->add({as.first, join_pred, kv.first.o}, as.second * kv.second);
                }
            }
        }
    }
}

} /* anonymous namespace */

struct eshkol_dbsp_join_view {
    uint64_t join_pred = 0;
    uint32_t num_shards = 1;
    eshkol_zset* state_a = nullptr;
    eshkol_zset* state_b = nullptr;
    eshkol_zset* state_j = nullptr;

    eshkol_dbsp_join_view(uint64_t jp, uint32_t ns)
        : join_pred(jp),
          num_shards(ns > 0 ? ns : 1u),
          state_a(new eshkol_zset(num_shards)),
          state_b(new eshkol_zset(num_shards)),
          state_j(new eshkol_zset(num_shards)) {}

    ~eshkol_dbsp_join_view() {
        delete state_a;
        delete state_b;
        delete state_j;
    }
};

extern "C" eshkol_zset_t* eshkol_zset_create(uint32_t num_shards) {
    return new (std::nothrow) eshkol_zset(num_shards > 0 ? num_shards : 1u);
}

extern "C" void eshkol_zset_destroy(eshkol_zset_t* z) {
    delete z;
}

extern "C" bool eshkol_zset_add(eshkol_zset_t* z,
                                uint64_t subject, uint64_t predicate, uint64_t object,
                                int64_t delta) {
    if (!z) return false;
    z->add({subject, predicate, object}, delta);
    return true;
}

extern "C" bool eshkol_zset_apply_delta(eshkol_zset_t* state, const eshkol_zset_t* delta) {
    if (!state || !delta) return false;
    for (const auto& sh : delta->shards) {
        for (const auto& kv : sh) {
            state->add(kv.first, kv.second);
        }
    }
    return true;
}

extern "C" bool eshkol_zset_apply_delta_custom(eshkol_zset_t* state,
                                               const eshkol_zset_t* delta,
                                               eshkol_zset_merge_fn_t merge_fn,
                                               void* user_ctx) {
    if (!state || !delta || !merge_fn) return false;
    auto sorted = delta->sorted_entries();
    for (const auto& kv : sorted) {
        state->add_custom(kv.first, kv.second, merge_fn, user_ctx);
    }
    return true;
}

extern "C" int64_t eshkol_zset_weight(const eshkol_zset_t* z,
                                      uint64_t subject, uint64_t predicate, uint64_t object) {
    if (!z) return 0;
    return z->weight({subject, predicate, object});
}

extern "C" size_t eshkol_zset_active_count(const eshkol_zset_t* z) {
    return z ? z->active_count() : 0u;
}

extern "C" uint32_t eshkol_zset_num_shards(const eshkol_zset_t* z) {
    return z ? z->num_shards : 0u;
}

extern "C" eshkol_zset_t* eshkol_zset_join_full(const eshkol_zset_t* a,
                                                const eshkol_zset_t* b,
                                                uint64_t join_predicate,
                                                uint32_t num_shards) {
    if (!a || !b) return nullptr;
    auto* out = new (std::nothrow) eshkol_zset(num_shards > 0 ? num_shards : 1u);
    if (!out) return nullptr;
    join_accumulate(a, b, join_predicate, out);
    return out;
}

extern "C" bool eshkol_zset_canonical_serialize(const eshkol_zset_t* z,
                                                uint8_t** out_buf,
                                                size_t* out_len) {
    if (!z || !out_buf || !out_len) return false;
    auto entries = z->sorted_entries();
    std::vector<uint8_t> bytes;
    bytes.reserve(8u + entries.size() * 32u);
    put_u64(bytes, static_cast<uint64_t>(entries.size()));
    for (const auto& kv : entries) {
        put_u64(bytes, kv.first.s);
        put_u64(bytes, kv.first.p);
        put_u64(bytes, kv.first.o);
        put_u64(bytes, static_cast<uint64_t>(kv.second));
    }
    auto* raw = static_cast<uint8_t*>(std::malloc(bytes.size()));
    if (!raw) return false;
    std::memcpy(raw, bytes.data(), bytes.size());
    *out_buf = raw;
    *out_len = bytes.size();
    return true;
}

extern "C" void eshkol_zset_buffer_free(uint8_t* buf) {
    std::free(buf);
}

extern "C" uint64_t eshkol_zset_canonical_hash(const eshkol_zset_t* z) {
    if (!z) return 0ULL;
    uint8_t* buf = nullptr;
    size_t len = 0;
    if (!eshkol_zset_canonical_serialize(z, &buf, &len)) return 0ULL;
    uint64_t h = fnv1a_64(buf, len);
    std::free(buf);
    return h;
}

extern "C" eshkol_dbsp_join_view_t* eshkol_dbsp_join_create(uint64_t join_predicate,
                                                            uint32_t num_shards) {
    return new (std::nothrow) eshkol_dbsp_join_view(join_predicate, num_shards);
}

extern "C" void eshkol_dbsp_join_destroy(eshkol_dbsp_join_view_t* view) {
    delete view;
}

extern "C" bool eshkol_dbsp_join_step(eshkol_dbsp_join_view_t* view,
                                      const eshkol_zset_t* delta_a,
                                      const eshkol_zset_t* delta_b,
                                      eshkol_zset_t** out_delta_j) {
    if (!view) return false;
    auto* dj = new (std::nothrow) eshkol_zset(view->num_shards);
    if (!dj) return false;

    /* Exact bilinear DBSP incremental maintenance:
     *   delta_j = (dA |><| B_old) + (A_old |><| dB) + (dA |><| dB) */
    if (delta_a) join_accumulate(delta_a, view->state_b, view->join_pred, dj);
    if (delta_b) join_accumulate(view->state_a, delta_b, view->join_pred, dj);
    if (delta_a && delta_b) join_accumulate(delta_a, delta_b, view->join_pred, dj);

    if (delta_a) eshkol_zset_apply_delta(view->state_a, delta_a);
    if (delta_b) eshkol_zset_apply_delta(view->state_b, delta_b);
    eshkol_zset_apply_delta(view->state_j, dj);

    if (out_delta_j) {
        *out_delta_j = dj;
    } else {
        delete dj;
    }
    return true;
}

extern "C" const eshkol_zset_t* eshkol_dbsp_join_state_a(const eshkol_dbsp_join_view_t* view) {
    return view ? view->state_a : nullptr;
}

extern "C" const eshkol_zset_t* eshkol_dbsp_join_state_b(const eshkol_dbsp_join_view_t* view) {
    return view ? view->state_b : nullptr;
}

extern "C" const eshkol_zset_t* eshkol_dbsp_join_result(const eshkol_dbsp_join_view_t* view) {
    return view ? view->state_j : nullptr;
}
