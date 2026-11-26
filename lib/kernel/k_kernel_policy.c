/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol OS Phase K5/K6 Formal Policy State Machine:
 *   - Supervisor restart policy with monotonic token generations,
 *     atomic lock release on fault (zero zombie locks), and fail-closed
 *     revoked-capability rejection.
 *   - Compositor surface spatial disjointness and read-only write trap
 *     arbitration.
 *   - Bounded real-time audio ring degradation on overrun/underrun.
 *
 * Pure freestanding C11 implementation with zero external symbol
 * dependencies across all 5 target architectures.
 */

#include <eshkol/kernel/k_kernel_policy.h>

static void k_pol_zero_bytes(void* dst, size_t n) {
    volatile uint8_t* p = (volatile uint8_t*)dst;
    for (size_t i = 0u; i < n; ++i) {
        p[i] = 0u;
    }
}

static int k_pol_intervals_overlap(uint32_t a_start, uint32_t a_len,
                                   uint32_t b_start, uint32_t b_len) {
    if (a_len == 0u || b_len == 0u) {
        return 0;
    }
    uint32_t a_end = a_start + a_len;
    uint32_t b_end = b_start + b_len;
    return (a_start < b_end) && (b_start < a_end);
}

static uint32_t k_pol_release_all_service_locks(k_policy_state_t* pol, uint32_t service_id) {
    uint32_t released = 0u;
    for (uint32_t l = 0u; l < K_POL_MAX_LOCKS; ++l) {
        if (pol->lock_owner[l] == service_id) {
            pol->lock_owner[l] = K_POL_NO_OWNER;
            released += 1u;
        }
    }
    pol->services[service_id].held_lock_mask = 0u;
    return released;
}

void k_policy_init(k_policy_state_t* pol, uint32_t audio_capacity) {
    if (!pol) {
        return;
    }
    k_pol_zero_bytes(pol, sizeof(*pol));
    pol->initialized = 1u;

    for (uint32_t i = 0u; i < K_POL_MAX_SERVICES; ++i) {
        pol->services[i].service_id = i;
        pol->services[i].state = K_POL_SVC_STOPPED;
    }
    for (uint32_t l = 0u; l < K_POL_MAX_LOCKS; ++l) {
        pol->lock_owner[l] = K_POL_NO_OWNER;
    }
    for (uint32_t s = 0u; s < K_POL_MAX_SURFACES; ++s) {
        pol->surfaces[s].surface_id = s;
    }

    uint32_t cap = audio_capacity;
    if (cap == 0u || cap > K_POL_AUDIO_RING_CAP) {
        cap = K_POL_AUDIO_RING_CAP;
    }
    pol->audio.capacity = cap;
    pol->audio.head = 0u;
    pol->audio.tail = 0u;
    pol->audio.count = 0u;
    pol->audio.supervisor_halted = 0u;
}

k_pol_status_t k_policy_service_register(k_policy_state_t* pol,
                                         uint32_t service_id,
                                         uint32_t badge,
                                         uint32_t* out_token_gen) {
    if (!pol || !pol->initialized) {
        return K_POL_ERR_INVALID_ARG;
    }
    if (service_id >= K_POL_MAX_SERVICES) {
        return K_POL_ERR_SERVICE_RANGE;
    }

    (void)k_pol_release_all_service_locks(pol, service_id);

    k_policy_service_t* svc = &pol->services[service_id];
    svc->service_id = service_id;
    svc->badge = badge;
    svc->state = K_POL_SVC_RUNNING;
    svc->root_cap_live = 1u;
    svc->token_gen = (svc->token_gen >= svc->last_faulted_gen)
        ? (svc->token_gen + 1u)
        : (svc->last_faulted_gen + 1u);
    svc->held_lock_mask = 0u;

    if (out_token_gen) {
        *out_token_gen = svc->token_gen;
    }
    return K_POL_OK;
}

k_pol_status_t k_policy_validate_token(const k_policy_state_t* pol,
                                       uint32_t service_id,
                                       uint32_t token_gen) {
    if (!pol || !pol->initialized) {
        return K_POL_ERR_INVALID_ARG;
    }
    if (service_id >= K_POL_MAX_SERVICES) {
        return K_POL_ERR_SERVICE_RANGE;
    }
    const k_policy_service_t* svc = &pol->services[service_id];
    if (svc->root_cap_live == 0u) {
        return K_POL_ERR_REVOKED_CAP;
    }
    if (token_gen == 0u || token_gen != svc->token_gen || token_gen <= svc->last_faulted_gen) {
        return K_POL_ERR_STALE_TOKEN;
    }
    if (svc->state != K_POL_SVC_RUNNING) {
        return K_POL_ERR_NOT_RUNNING;
    }
    return K_POL_OK;
}

k_pol_status_t k_policy_lock_acquire(k_policy_state_t* pol,
                                     uint32_t service_id,
                                     uint32_t token_gen,
                                     uint32_t lock_id) {
    k_pol_status_t vst = k_policy_validate_token(pol, service_id, token_gen);
    if (vst != K_POL_OK) {
        return vst;
    }
    if (lock_id >= K_POL_MAX_LOCKS) {
        return K_POL_ERR_LOCK_RANGE;
    }
    if (pol->lock_owner[lock_id] != K_POL_NO_OWNER) {
        return K_POL_ERR_LOCK_HELD;
    }
    pol->lock_owner[lock_id] = service_id;
    pol->services[service_id].held_lock_mask |= (1u << lock_id);
    return K_POL_OK;
}

k_pol_status_t k_policy_lock_release(k_policy_state_t* pol,
                                     uint32_t service_id,
                                     uint32_t token_gen,
                                     uint32_t lock_id) {
    k_pol_status_t vst = k_policy_validate_token(pol, service_id, token_gen);
    if (vst != K_POL_OK) {
        return vst;
    }
    if (lock_id >= K_POL_MAX_LOCKS) {
        return K_POL_ERR_LOCK_RANGE;
    }
    if (pol->lock_owner[lock_id] != service_id) {
        return K_POL_ERR_LOCK_HELD;
    }
    pol->lock_owner[lock_id] = K_POL_NO_OWNER;
    pol->services[service_id].held_lock_mask &= ~(1u << lock_id);
    return K_POL_OK;
}

k_pol_status_t k_policy_service_fault(k_policy_state_t* pol,
                                      uint32_t service_id,
                                      uint32_t* out_released_locks) {
    if (!pol || !pol->initialized) {
        return K_POL_ERR_INVALID_ARG;
    }
    if (service_id >= K_POL_MAX_SERVICES) {
        return K_POL_ERR_SERVICE_RANGE;
    }

    k_policy_service_t* svc = &pol->services[service_id];
    /* Atomically release every capability lock held by the faulting service (zero zombie locks). */
    uint32_t released = k_pol_release_all_service_locks(pol, service_id);
    if (out_released_locks) {
        *out_released_locks = released;
    }

    if (svc->token_gen > svc->last_faulted_gen) {
        svc->last_faulted_gen = svc->token_gen;
    }
    svc->fault_count += 1u;
    svc->state = (svc->root_cap_live != 0u) ? K_POL_SVC_FAULTED : K_POL_SVC_STOPPED;
    return K_POL_OK;
}

k_pol_status_t k_policy_revoke_service_cap(k_policy_state_t* pol,
                                           uint32_t service_id,
                                           uint32_t* out_released_locks) {
    if (!pol || !pol->initialized) {
        return K_POL_ERR_INVALID_ARG;
    }
    if (service_id >= K_POL_MAX_SERVICES) {
        return K_POL_ERR_SERVICE_RANGE;
    }

    k_policy_service_t* svc = &pol->services[service_id];
    uint32_t released = k_pol_release_all_service_locks(pol, service_id);
    if (out_released_locks) {
        *out_released_locks = released;
    }
    if (svc->token_gen > svc->last_faulted_gen) {
        svc->last_faulted_gen = svc->token_gen;
    }
    svc->root_cap_live = 0u;
    svc->state = K_POL_SVC_STOPPED;
    return K_POL_OK;
}

k_pol_status_t k_policy_service_restart(k_policy_state_t* pol,
                                        uint32_t service_id,
                                        uint32_t* out_new_token_gen) {
    if (!pol || !pol->initialized) {
        return K_POL_ERR_INVALID_ARG;
    }
    if (service_id >= K_POL_MAX_SERVICES) {
        return K_POL_ERR_SERVICE_RANGE;
    }

    k_policy_service_t* svc = &pol->services[service_id];
    /* Strict fail-closed policy: revoked root capability ALWAYS blocks restart. */
    if (svc->root_cap_live == 0u) {
        svc->state = K_POL_SVC_STOPPED;
        pol->revoked_restart_rejects += 1u;
        return K_POL_ERR_REVOKED_CAP;
    }
    if (svc->state != K_POL_SVC_FAULTED) {
        return K_POL_ERR_NOT_FAULTED;
    }

    /* Guarantee zero zombie locks before resuming. */
    (void)k_pol_release_all_service_locks(pol, service_id);

    /* Mint strictly monotonic fresh generation (> prior faulted token). */
    uint32_t next_gen = (svc->token_gen >= svc->last_faulted_gen)
        ? (svc->token_gen + 1u)
        : (svc->last_faulted_gen + 1u);
    svc->token_gen = next_gen;
    svc->state = K_POL_SVC_RUNNING;
    svc->restart_count += 1u;

    if (out_new_token_gen) {
        *out_new_token_gen = next_gen;
    }
    return K_POL_OK;
}

k_pol_status_t k_policy_surface_register(k_policy_state_t* pol,
                                         uint32_t surface_id,
                                         uint32_t owner_badge,
                                         uint32_t x_start,
                                         uint32_t x_len,
                                         uint32_t writable) {
    if (!pol || !pol->initialized) {
        return K_POL_ERR_INVALID_ARG;
    }
    if (surface_id >= K_POL_MAX_SURFACES) {
        return K_POL_ERR_SURFACE_RANGE;
    }
    if (x_len == 0u || x_start > K_POL_FB_CELLS || x_len > (K_POL_FB_CELLS - x_start)) {
        return K_POL_ERR_BOUNDS;
    }

    /* Reject any spatial overlap with another active surface. */
    for (uint32_t i = 0u; i < K_POL_MAX_SURFACES; ++i) {
        if (i == surface_id) {
            continue;
        }
        const k_policy_surface_t* other = &pol->surfaces[i];
        if (other->active != 0u &&
            k_pol_intervals_overlap(x_start, x_len, other->x_start, other->x_len)) {
            pol->conflict_reject_count += 1u;
            return K_POL_ERR_SURFACE_CONFLICT;
        }
    }

    k_policy_surface_t* s = &pol->surfaces[surface_id];
    s->surface_id = surface_id;
    s->active = 1u;
    s->owner_badge = owner_badge;
    s->x_start = x_start;
    s->x_len = x_len;
    s->writable = writable ? 1u : 0u;
    return K_POL_OK;
}

k_pol_status_t k_policy_surface_write(k_policy_state_t* pol,
                                      uint32_t surface_id,
                                      uint32_t caller_badge,
                                      uint32_t cell_offset,
                                      uint32_t len,
                                      uint32_t pixel_value) {
    if (!pol || !pol->initialized) {
        return K_POL_ERR_INVALID_ARG;
    }
    if (surface_id >= K_POL_MAX_SURFACES) {
        return K_POL_ERR_SURFACE_RANGE;
    }
    const k_policy_surface_t* s = &pol->surfaces[surface_id];
    if (s->active == 0u) {
        return K_POL_ERR_SURFACE_RANGE;
    }

    /* Cross-badge ownership conflict fails closed with framebuffer untouched. */
    if (caller_badge != s->owner_badge) {
        pol->conflict_reject_count += 1u;
        return K_POL_ERR_SURFACE_CONFLICT;
    }

    /* Read-only surface write ALWAYS traps cleanly with framebuffer untouched. */
    if (s->writable == 0u) {
        pol->ro_trap_count += 1u;
        return K_POL_ERR_RO_TRAP;
    }

    if (len == 0u || cell_offset >= s->x_len || len > (s->x_len - cell_offset)) {
        return K_POL_ERR_BOUNDS;
    }

    uint32_t abs_start = s->x_start + cell_offset;
    if (abs_start > K_POL_FB_CELLS || len > (K_POL_FB_CELLS - abs_start)) {
        return K_POL_ERR_BOUNDS;
    }

    /* Defense-in-depth spatial non-overlap verification across distinct badges. */
    for (uint32_t i = 0u; i < K_POL_MAX_SURFACES; ++i) {
        if (i == surface_id) {
            continue;
        }
        const k_policy_surface_t* other = &pol->surfaces[i];
        if (other->active != 0u && other->owner_badge != caller_badge &&
            k_pol_intervals_overlap(abs_start, len, other->x_start, other->x_len)) {
            pol->conflict_reject_count += 1u;
            return K_POL_ERR_SURFACE_CONFLICT;
        }
    }

    for (uint32_t k = 0u; k < len; ++k) {
        pol->framebuffer[abs_start + k] = pixel_value;
    }
    return K_POL_OK;
}

k_pol_status_t k_policy_audio_step(k_policy_state_t* pol,
                                   uint32_t is_produce,
                                   uint32_t sample_in,
                                   uint32_t* out_sample,
                                   uint32_t* out_degraded) {
    if (!pol || !pol->initialized) {
        return K_POL_ERR_INVALID_ARG;
    }
    k_policy_audio_t* a = &pol->audio;
    if (a->capacity == 0u || a->capacity > K_POL_AUDIO_RING_CAP) {
        return K_POL_ERR_INVALID_ARG;
    }
    if (out_degraded) {
        *out_degraded = 0u;
    }

    if (is_produce != 0u) {
        if (a->count >= a->capacity) {
            /* Bounded overrun degradation: drop incoming sample, preserve live ring, never halt. */
            a->overrun_drops += 1u;
            if (out_degraded) {
                *out_degraded = 1u;
            }
            return K_POL_OK;
        }
        a->samples[a->tail] = sample_in;
        a->tail += 1u;
        if (a->tail >= a->capacity) {
            a->tail = 0u;
        }
        a->count += 1u;
        return K_POL_OK;
    }

    /* Consume path */
    if (a->count == 0u) {
        /* Bounded underrun degradation: emit silence (0), never halt. */
        a->underrun_silences += 1u;
        if (out_sample) {
            *out_sample = 0u;
        }
        if (out_degraded) {
            *out_degraded = 1u;
        }
        return K_POL_OK;
    }

    uint32_t val = a->samples[a->head];
    a->head += 1u;
    if (a->head >= a->capacity) {
        a->head = 0u;
    }
    a->count -= 1u;
    if (out_sample) {
        *out_sample = val;
    }
    return K_POL_OK;
}

int k_policy_invariants_hold(const k_policy_state_t* pol) {
    if (!pol || !pol->initialized) {
        return 0;
    }

    /* 1. Supervisor token monotonicity, revoked-cap isolation, and zero zombie locks. */
    for (uint32_t i = 0u; i < K_POL_MAX_SERVICES; ++i) {
        const k_policy_service_t* svc = &pol->services[i];
        if (svc->state == K_POL_SVC_RUNNING) {
            if (svc->root_cap_live == 0u) {
                return 0;
            }
            if (svc->token_gen == 0u || svc->token_gen <= svc->last_faulted_gen) {
                return 0;
            }
        } else {
            if (svc->held_lock_mask != 0u) {
                return 0;
            }
        }
    }

    /* 2. Lock ownership coherence: every lock owner is RUNNING and bitmask matches. */
    for (uint32_t l = 0u; l < K_POL_MAX_LOCKS; ++l) {
        uint32_t owner = pol->lock_owner[l];
        if (owner != K_POL_NO_OWNER) {
            if (owner >= K_POL_MAX_SERVICES) {
                return 0;
            }
            if (pol->services[owner].state != K_POL_SVC_RUNNING) {
                return 0;
            }
            if ((pol->services[owner].held_lock_mask & (1u << l)) == 0u) {
                return 0;
            }
        }
    }

    /* 3. Compositor spatial disjointness across active surfaces. */
    for (uint32_t i = 0u; i < K_POL_MAX_SURFACES; ++i) {
        const k_policy_surface_t* a = &pol->surfaces[i];
        if (a->active == 0u) {
            continue;
        }
        if (a->x_len == 0u || a->x_start > K_POL_FB_CELLS || a->x_len > (K_POL_FB_CELLS - a->x_start)) {
            return 0;
        }
        for (uint32_t j = i + 1u; j < K_POL_MAX_SURFACES; ++j) {
            const k_policy_surface_t* b = &pol->surfaces[j];
            if (b->active != 0u &&
                k_pol_intervals_overlap(a->x_start, a->x_len, b->x_start, b->x_len)) {
                return 0;
            }
        }
    }

    /* 4. Audio ring boundedness and non-halting operational liveness. */
    const k_policy_audio_t* aud = &pol->audio;
    if (aud->capacity == 0u || aud->capacity > K_POL_AUDIO_RING_CAP) {
        return 0;
    }
    if (aud->count > aud->capacity || aud->head >= aud->capacity || aud->tail >= aud->capacity) {
        return 0;
    }
    if (aud->supervisor_halted != 0u) {
        return 0;
    }

    return 1;
}
