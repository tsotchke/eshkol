/*
 * Resident Flat-RSS Continuous-Cognition Session (Waking-to-Sleep Loop)
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <eshkol/core/forney_fg.h>
#include <eshkol/core/logic.h>
#include <eshkol/core/nars_truth.h>
#include <eshkol/core/resident_cognition.h>
#include <eshkol/core/sheaf_workspace.h>
#include <eshkol/core/vsa_hdc.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

extern "C" {
arena_t* arena_create(size_t default_block_size);
void arena_destroy(arena_t* arena);
void arena_reset(arena_t* arena);
size_t arena_get_used_memory(const arena_t* arena);
size_t arena_get_total_memory(const arena_t* arena);
const char* eshkol_intern_predicate(const char* name);
}

namespace {

inline eshkol_tagged_value_t make_int_tv(int64_t v) {
    eshkol_tagged_value_t tv;
    std::memset(&tv, 0, sizeof(tv));
    tv.type = ESHKOL_VALUE_INT64;
    tv.data.int_val = v;
    return tv;
}

inline eshkol_tagged_value_t make_var_tv(uint64_t id) {
    eshkol_tagged_value_t tv;
    std::memset(&tv, 0, sizeof(tv));
    tv.type = ESHKOL_VALUE_LOGIC_VAR;
    tv.data.int_val = static_cast<int64_t>(id);
    return tv;
}

} /* anonymous namespace */

struct eshkol_resident_cog_session {
    eshkol_resident_cog_config_t cfg{};
    eshkol_resident_cog_stats_t stats{};

    arena_t* persistent_arena = nullptr;
    arena_t* scratch_arena = nullptr;

    /* Persistent consolidated state (value-owned, zero pointers into scratch_arena) */
    uint64_t pred_percept = 0;
    uint64_t var_x = 0;
    uint64_t var_y = 0;
    eshkol_nars_truth_t consolidated_truth{0.5, 0.1};
    double last_broadcast[3] = {0.0, 0.0, 0.0};

    /* Reusable CE-2 / CE-4 / CE-5 subsystems (zero per-tick heap growth) */
    eshkol_forney_fg_t* fg = nullptr;
    eshkol_vsa_cleanup_t* cleanup = nullptr;
    eshkol_sheaf_workspace_t* ws_compat = nullptr;
    eshkol_sheaf_workspace_t* ws_twisted = nullptr;

    std::vector<double> role_vec;
    std::vector<std::vector<double>> val_vecs;
    std::vector<double> tmp_bound;
    std::vector<double> tmp_unbound;

    explicit eshkol_resident_cog_session(const eshkol_resident_cog_config_t& c) : cfg(c) {
        std::memset(&stats, 0, sizeof(stats));
    }

    ~eshkol_resident_cog_session() {
        eshkol_forney_fg_destroy(fg);
        eshkol_vsa_cleanup_destroy(cleanup);
        eshkol_sheaf_ws_destroy(ws_compat);
        eshkol_sheaf_ws_destroy(ws_twisted);
        if (scratch_arena) arena_destroy(scratch_arena);
        if (persistent_arena) arena_destroy(persistent_arena);
    }

    bool init() {
        persistent_arena = arena_create(cfg.persistent_block_bytes);
        scratch_arena = arena_create(cfg.scratch_block_bytes);
        if (!persistent_arena || !scratch_arena) return false;

        pred_percept = reinterpret_cast<uint64_t>(eshkol_intern_predicate("ce7_percept"));
        var_x = eshkol_make_logic_var("?ce7_x");
        var_y = eshkol_make_logic_var("?ce7_y");

        /* Allocate one durable anchor fact in persistent_arena */
        eshkol_tagged_value_t init_args[2] = {make_int_tv(0), make_int_tv(0)};
        if (!eshkol_make_fact(persistent_arena, pred_percept, init_args, 2)) return false;

        /* Build reusable 3-var Forney chain FG */
        const uint32_t dims[3] = {2, 2, 2};
        fg = eshkol_forney_fg_create(3, dims);
        if (!fg) return false;
        const uint32_t u0[1] = {0};
        const double p_u0[2] = {0.2, -0.2};
        const uint32_t e01[2] = {0, 1}, e12[2] = {1, 2};
        const double p_pair[4] = {0.4, -0.4, -0.4, 0.4};
        if (eshkol_forney_fg_add_factor(fg, u0, 1, p_u0, 2) == UINT32_MAX ||
            eshkol_forney_fg_add_factor(fg, e01, 2, p_pair, 4) == UINT32_MAX ||
            eshkol_forney_fg_add_factor(fg, e12, 2, p_pair, 4) == UINT32_MAX) {
            return false;
        }

        /* Build FHRR cleanup memory with 4 atoms */
        const uint32_t D = cfg.fhrr_dim;
        role_vec.resize(D);
        tmp_bound.resize(D);
        tmp_unbound.resize(D);
        eshkol_fhrr_random(role_vec.data(), D, 70001ULL);
        cleanup = eshkol_vsa_cleanup_create(ESHKOL_VSA_FHRR, D);
        if (!cleanup) return false;
        val_vecs.resize(4);
        const char* names[4] = {"s0", "s1", "s2", "s3"};
        for (uint32_t i = 0; i < 4; i++) {
            val_vecs[i].resize(D);
            eshkol_fhrr_random(val_vecs[i].data(), D, 71000ULL + i);
            eshkol_vsa_cleanup_add(cleanup, names[i], val_vecs[i].data(), D);
        }

        /* Build reusable compatible & twisted sheaf workspaces (d = 3) */
        const double I3[9] = {1, 0, 0,  0, 1, 0,  0, 0, 1};
        const double NegI3[9] = {-1, 0, 0,  0, -1, 0,  0, 0, -1};
        eshkol_sheaf_ws_config_t wcfg = eshkol_sheaf_ws_default_config();
        wcfg.refractory_steps = 1;
        wcfg.dialectic_boost = 2.0;

        ws_compat = eshkol_sheaf_ws_create(3, &wcfg);
        ws_twisted = eshkol_sheaf_ws_create(3, &wcfg);
        if (!ws_compat || !ws_twisted) return false;

        for (auto* w : {ws_compat, ws_twisted}) {
            eshkol_sheaf_ws_add_specialist(w, "percept", false);
            eshkol_sheaf_ws_add_specialist(w, "memory", false);
            eshkol_sheaf_ws_add_specialist(w, "inference", false);
            eshkol_sheaf_ws_add_specialist(w, "dialectic", true);
            eshkol_sheaf_ws_add_transition(w, 0, 1, I3, 3);
            eshkol_sheaf_ws_add_transition(w, 1, 2, I3, 3);
        }
        eshkol_sheaf_ws_add_transition(ws_compat, 2, 0, I3, 3);
        eshkol_sheaf_ws_add_transition(ws_twisted, 2, 0, NegI3, 3);
        return true;
    }

    void record_memory_watermark() {
        size_t live = arena_get_used_memory(scratch_arena) +
                      arena_get_used_memory(persistent_arena);
        size_t total = arena_get_total_memory(scratch_arena) +
                       arena_get_total_memory(persistent_arena);
        stats.live_bytes = live;
        stats.total_arena_bytes = total;
        if (live > stats.peak_bytes) stats.peak_bytes = live;

        if (stats.ticks <= cfg.warmup_ticks) {
            if (live > stats.warmup_peak_bytes) stats.warmup_peak_bytes = live;
            if (total > stats.warmup_total_arena_bytes) stats.warmup_total_arena_bytes = total;
        } else {
            if (live > stats.post_warmup_peak_bytes) stats.post_warmup_peak_bytes = live;
            stats.steady_state_growth =
                (stats.post_warmup_peak_bytes > stats.warmup_peak_bytes)
                    ? (stats.post_warmup_peak_bytes - stats.warmup_peak_bytes)
                    : 0u;
            stats.post_warmup_total_growth =
                (total > stats.warmup_total_arena_bytes)
                    ? (total - stats.warmup_total_arena_bytes)
                    : 0u;
        }
    }

    bool sleep_evacuate() {
        stats.evacuations++;
        if (cfg.poison_rotate_evacs > 0 &&
            (stats.evacuations % cfg.poison_rotate_evacs) == 0) {
            /* Full destroy + recreate actively exercises ESHKOL_ARENA_POISON=1 during soak */
            arena_destroy(scratch_arena);
            scratch_arena = arena_create(cfg.scratch_block_bytes);
            if (!scratch_arena) return false;
            stats.scratch_rotations++;
        } else {
            arena_reset(scratch_arena);
        }
        return true;
    }

    bool tick() {
        stats.ticks++;
        const uint64_t t = stats.ticks;
        const uint32_t atom_idx = static_cast<uint32_t>(t & 3u);

        /* 1. CE-0 scratch-arena logic allocations: fact + HAMT-capable substitution + unify */
        eshkol_tagged_value_t pat_args[2] = {make_var_tv(var_x), make_var_tv(var_y)};
        eshkol_tagged_value_t obs_args[2] = {make_int_tv(static_cast<int64_t>(t & 1023u)),
                                             make_int_tv(static_cast<int64_t>(atom_idx))};
        eshkol_fact_t* f_pat = eshkol_make_fact(scratch_arena, pred_percept, pat_args, 2);
        eshkol_fact_t* f_obs = eshkol_make_fact(scratch_arena, pred_percept, obs_args, 2);
        eshkol_substitution_t* s0 = eshkol_make_substitution(scratch_arena, 8);
        if (!f_pat || !f_obs || !s0) return false;

        eshkol_tagged_value_t tv_pat{}, tv_obs{};
        tv_pat.type = ESHKOL_VALUE_HEAP_PTR;
        tv_pat.data.ptr_val = reinterpret_cast<uint64_t>(f_pat);
        tv_obs.type = ESHKOL_VALUE_HEAP_PTR;
        tv_obs.data.ptr_val = reinterpret_cast<uint64_t>(f_obs);
        eshkol_substitution_t* s1 = eshkol_unify(scratch_arena, &tv_pat, &tv_obs, s0);
        if (!s1) return false;

        /* 2. CE-4 FHRR bind -> unbind -> associative cleanup verification */
        const uint32_t D = cfg.fhrr_dim;
        eshkol_fhrr_bind(role_vec.data(), val_vecs[atom_idx].data(), tmp_bound.data(), D);
        eshkol_fhrr_unbind(tmp_bound.data(), role_vec.data(), tmp_unbound.data(), D);
        eshkol_vsa_match_t match{};
        if (eshkol_vsa_cleanup_query(cleanup, tmp_unbound.data(), D, &match) &&
            match.found && match.atom_id == atom_idx) {
            stats.cleanup_hits++;
        }

        /* 3. CE-5 Sheaf Workspace step (periodic H^1 twisted episode vs normal H^0 gluing) */
        const bool inject_h1 = (cfg.h1_inject_period > 0 && (t % cfg.h1_inject_period) == 0);
        eshkol_sheaf_workspace_t* ws = inject_h1 ? ws_twisted : ws_compat;
        double phase = 0.01 * static_cast<double>(t & 255u);
        double prop[3] = {std::cos(phase), std::sin(phase), 0.5};
        eshkol_sheaf_ws_propose(ws, 0, prop, 3, 0.9);
        eshkol_sheaf_ws_propose(ws, 1, prop, 3, 0.8);
        eshkol_sheaf_ws_propose(ws, 2, prop, 3, 0.7);
        eshkol_sheaf_ws_propose(ws, 3, prop, 3, 0.2);

        eshkol_sheaf_ws_step_result_t ws_res{};
        if (!eshkol_sheaf_ws_step(ws, &ws_res)) return false;
        if (ws_res.status == ESHKOL_WS_H1_OBSTRUCTION) {
            stats.h1_obstructions++;
        }
        eshkol_sheaf_ws_get_broadcast(ws, 0, last_broadcast, 3);

        /* 4. CE-2 Forney FG fast belief update + Bethe FE (every 4 ticks to keep 100k soak crisp) */
        if ((t & 3u) == 0u) {
            double bias = 0.3 * std::cos(phase);
            double pot0[2] = {bias, -bias};
            eshkol_forney_fg_set_factor_potentials(fg, 0, pot0, 2);
            eshkol_bp_config_t bcfg = eshkol_bp_default_config();
            bcfg.schedule = ESHKOL_BP_SYNC;
            bcfg.max_iterations = 8;
            bcfg.tolerance = 1e-6;
            bcfg.damping = 1.0;
            eshkol_bp_stats_t bst{};
            eshkol_forney_fg_infer(fg, &bcfg, &bst);
            stats.last_bethe_free_energy = bst.bethe_free_energy;

            /* Bounded NARS confidence update (exponential forgetting prevents saturating at 1) */
            eshkol_nars_truth_t obs_tv{0.8, 0.25};
            eshkol_nars_truth_t decayed{consolidated_truth.frequency,
                                        consolidated_truth.confidence * 0.95};
            eshkol_nars_revision(decayed, obs_tv, ESHKOL_NARS_DEFAULT_HORIZON, &consolidated_truth);
            stats.last_nars_confidence = consolidated_truth.confidence;
        }

        /* Record peak watermark BEFORE sleep evacuation */
        record_memory_watermark();

        /* 5. Periodic waking-to-sleep scratch evacuation */
        if (cfg.evac_interval_ticks > 0 && (t % cfg.evac_interval_ticks) == 0) {
            if (!sleep_evacuate()) return false;
        }
        return true;
    }
};

extern "C" eshkol_resident_cog_config_t eshkol_resident_cog_default_config(void) {
    eshkol_resident_cog_config_t cfg;
    cfg.scratch_block_bytes = 65536u;
    cfg.persistent_block_bytes = 65536u;
    cfg.evac_interval_ticks = 16u;
    cfg.poison_rotate_evacs = 256u;
    cfg.warmup_ticks = 100u;
    cfg.h1_inject_period = 1000u;
    cfg.fhrr_dim = 64u;
    return cfg;
}

extern "C" eshkol_resident_cog_session_t* eshkol_resident_cog_create(
    const eshkol_resident_cog_config_t* cfg) {
    eshkol_resident_cog_config_t eff = cfg ? *cfg : eshkol_resident_cog_default_config();
    auto* sess = new (std::nothrow) eshkol_resident_cog_session(eff);
    if (!sess) return nullptr;
    if (!sess->init()) {
        delete sess;
        return nullptr;
    }
    return sess;
}

extern "C" void eshkol_resident_cog_destroy(eshkol_resident_cog_session_t* sess) {
    delete sess;
}

extern "C" bool eshkol_resident_cog_tick(eshkol_resident_cog_session_t* sess) {
    if (!sess) return false;
    return sess->tick();
}

extern "C" bool eshkol_resident_cog_run(eshkol_resident_cog_session_t* sess,
                                        uint64_t num_ticks,
                                        eshkol_resident_cog_stats_t* out_stats) {
    if (!sess) return false;
    for (uint64_t i = 0; i < num_ticks; i++) {
        if (!sess->tick()) return false;
    }
    if (out_stats) *out_stats = sess->stats;
    return true;
}

extern "C" void eshkol_resident_cog_get_stats(const eshkol_resident_cog_session_t* sess,
                                              eshkol_resident_cog_stats_t* out_stats) {
    if (!sess || !out_stats) return;
    *out_stats = sess->stats;
}
