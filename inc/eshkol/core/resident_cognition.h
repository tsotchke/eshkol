/*
 * Resident Flat-RSS Continuous-Cognition Session (Waking-to-Sleep Loop)
 *
 * Wires CE-0..CE-5 subsystems into a continuous resident cognitive tick loop:
 *   - CE-0 arena-backed logic facts & persistent HAMT substitutions (scratch arena)
 *   - CE-2 CSR Forney factor graph belief updates & Bethe free energy
 *   - CE-4 FHRR VSA/HDC binding, unbinding, and associative cleanup
 *   - CE-5 Sheaf-cohomological & GNWT global workspace with H^1 obstruction checks
 *   - Periodic waking-to-sleep scratch-arena evacuation & reset with exact
 *     watermark accounting (peak_bytes, live_bytes, steady_state_growth == 0)
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_CORE_RESIDENT_COGNITION_H
#define ESHKOL_CORE_RESIDENT_COGNITION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct eshkol_resident_cog_config {
    size_t scratch_block_bytes;     /* Scratch arena block size, default 65536 */
    size_t persistent_block_bytes;  /* Persistent arena block size, default 65536 */
    uint32_t evac_interval_ticks;   /* Waking-to-sleep evacuation period, default 16 */
    uint32_t poison_rotate_evacs;   /* Recreate scratch arena every N evacs for poison verification */
    uint32_t warmup_ticks;          /* Warmup tick boundary for steady-state watermark, default 100 */
    uint32_t h1_inject_period;      /* Inject twisted H^1 episode every N ticks, default 1000 */
    uint32_t fhrr_dim;              /* FHRR dimension per tick, default 64 */
} eshkol_resident_cog_config_t;

typedef struct eshkol_resident_cog_stats {
    uint64_t ticks;
    uint64_t evacuations;
    uint64_t scratch_rotations;
    uint64_t h1_obstructions;
    uint64_t cleanup_hits;
    size_t live_bytes;
    size_t peak_bytes;
    size_t warmup_peak_bytes;
    size_t post_warmup_peak_bytes;
    size_t steady_state_growth;          /* post_warmup_peak_bytes - warmup_peak_bytes (clamped >= 0) */
    size_t total_arena_bytes;
    size_t warmup_total_arena_bytes;
    size_t post_warmup_total_growth;     /* total_arena_bytes growth after warmup */
    double last_bethe_free_energy;
    double last_nars_confidence;
} eshkol_resident_cog_stats_t;

typedef struct eshkol_resident_cog_session eshkol_resident_cog_session_t;

eshkol_resident_cog_config_t eshkol_resident_cog_default_config(void);

eshkol_resident_cog_session_t* eshkol_resident_cog_create(
    const eshkol_resident_cog_config_t* cfg);

void eshkol_resident_cog_destroy(eshkol_resident_cog_session_t* sess);

/* Execute one waking tick (with automatic sleep evacuation every evac_interval_ticks). */
bool eshkol_resident_cog_tick(eshkol_resident_cog_session_t* sess);

/* Execute `num_ticks` consecutive ticks and populate `out_stats`. */
bool eshkol_resident_cog_run(eshkol_resident_cog_session_t* sess,
                             uint64_t num_ticks,
                             eshkol_resident_cog_stats_t* out_stats);

void eshkol_resident_cog_get_stats(const eshkol_resident_cog_session_t* sess,
                                   eshkol_resident_cog_stats_t* out_stats);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_RESIDENT_COGNITION_H */
