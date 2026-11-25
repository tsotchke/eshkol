/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol OS Phase K5 + K6: Freestanding zero-LLVM, zero-POSIX, zero-malloc
 * capability service supervisor, framebuffer/input/real-time audio compositor,
 * and end-to-end self-hosting reproducible builder inside K4 CAS VFS.
 *
 * External symbol closure: ZERO undefined external symbols (pure C11).
 */

#include <eshkol/core/os_k5_k6_services.h>

#include <stddef.h>
#include <stdint.h>

/* ── Freestanding byte/string helpers (volatile loops avoid libc lowering) ── */

static void k56_zero(void* dst, size_t n) {
    volatile uint8_t* p = (volatile uint8_t*)dst;
    for (size_t i = 0u; i < n; i++) {
        p[i] = 0u;
    }
}

static void k56_fill(void* dst, uint8_t val, size_t n) {
    volatile uint8_t* p = (volatile uint8_t*)dst;
    for (size_t i = 0u; i < n; i++) {
        p[i] = val;
    }
}

static void k56_copy(void* dst, const void* src, size_t n) {
    volatile uint8_t* d = (volatile uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;
    for (size_t i = 0u; i < n; i++) {
        d[i] = s[i];
    }
}

static int k56_memeq(const void* a, const void* b, size_t n) {
    const uint8_t* pa = (const uint8_t*)a;
    const uint8_t* pb = (const uint8_t*)b;
    uint8_t diff = 0u;
    for (size_t i = 0u; i < n; i++) {
        diff |= (uint8_t)(pa[i] ^ pb[i]);
    }
    return diff == 0u;
}

static size_t k56_strlen(const char* s) {
    if (!s) return 0u;
    size_t n = 0u;
    while (s[n] != '\0') n++;
    return n;
}

static uint16_t k56_rd_u16_le(const uint8_t* p) {
    return (uint16_t)(((uint16_t)p[0]) | ((uint16_t)p[1] << 8));
}

static uint32_t k56_rd_u32_le(const uint8_t* p) {
    return ((uint32_t)p[0]) |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t k56_rd_u64_le(const uint8_t* p) {
    return ((uint64_t)k56_rd_u32_le(p)) |
           ((uint64_t)k56_rd_u32_le(p + 4u) << 32);
}

static void k56_wr_u64_le(uint8_t* p, uint64_t v) {
    for (uint32_t i = 0u; i < 8u; i++) {
        p[i] = (uint8_t)((v >> (i * 8u)) & 0xFFu);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Part 1: Phase K5 Capability Service Supervisor
 * ═══════════════════════════════════════════════════════════════════════════ */

static void sup_reclaim_service_locks(eshkol_os_supervisor_t* sup, uint32_t service_id) {
    for (uint32_t i = 0u; i < ESHKOL_OS_SUP_MAX_LOCKS; i++) {
        if (sup->locks[i].held && sup->locks[i].owner_service_id == service_id) {
            sup->locks[i].held = 0u;
            sup->locks[i].owner_service_id = 0u;
            sup->locks[i].owner_generation = 0ULL;
            sup->locks[i].reclaim_count++;
            sup->total_lock_reclaims++;
        }
    }
}

eshkol_os_k56_status_t eshkol_os_supervisor_init(
    eshkol_os_supervisor_t* sup,
    eshkol_os_syscall6_fn_t syscall_fn) {
    if (!sup || !syscall_fn) return ESHKOL_K56_ERR_INVALID_ARG;
    k56_zero(sup, sizeof(*sup));
    sup->syscall_fn = syscall_fn;
    sup->next_generation = 1ULL;
    sup->initialized = 1u;
    return ESHKOL_K56_OK;
}

eshkol_os_k56_status_t eshkol_os_supervisor_register(
    eshkol_os_supervisor_t* sup,
    const char* name,
    eshkol_cap_id_t root_cap,
    uint64_t rights,
    uint32_t* out_service_id,
    eshkol_os_svc_token_t* out_token) {
    if (!sup || !sup->initialized || !name || root_cap == ESHKOL_CAP_ID_INVALID ||
        !out_service_id || !out_token) {
        return ESHKOL_K56_ERR_INVALID_ARG;
    }
    size_t nlen = k56_strlen(name);
    if (nlen == 0u || nlen > ESHKOL_OS_SUP_MAX_NAME) return ESHKOL_K56_ERR_INVALID_ARG;

    int slot = -1;
    for (uint32_t i = 0u; i < ESHKOL_OS_SUP_MAX_SERVICES; i++) {
        if (sup->services[i].state == ESHKOL_SVC_STATE_FREE) {
            slot = (int)i;
            break;
        }
    }
    if (slot < 0) return ESHKOL_K56_ERR_NO_SLOTS;

    uint64_t gen = sup->next_generation++;
    int64_t derived = sup->syscall_fn(
        ESHKOL_SYS_CAP_DERIVE, root_cap, rights, gen, 0ULL, 0ULL, 0ULL);
    if (derived == ESHKOL_OS_ERR_REVOKED ||
        derived == ESHKOL_OS_ERR_RIGHTS_DENIED ||
        derived == ESHKOL_OS_ERR_INVALID_CAP) {
        return ESHKOL_SUP_ERR_REVOKED_CAP;
    }
    if (derived <= 0) return ESHKOL_K56_ERR_SYSCALL_FAILED;

    eshkol_os_svc_record_t* rec = &sup->services[slot];
    k56_zero(rec, sizeof(*rec));
    k56_copy(rec->name, name, nlen);
    rec->state = ESHKOL_SVC_STATE_RUNNING;
    rec->generation = gen;
    rec->root_cap = root_cap;
    rec->instance_cap = (eshkol_cap_id_t)derived;
    rec->rights = rights;
    sup->service_count++;

    *out_service_id = (uint32_t)slot;
    out_token->service_id = (uint32_t)slot;
    out_token->reserved = 0u;
    out_token->generation = gen;
    out_token->instance_cap = rec->instance_cap;
    return ESHKOL_K56_OK;
}

eshkol_os_k56_status_t eshkol_os_supervisor_validate_token(
    const eshkol_os_supervisor_t* sup,
    const eshkol_os_svc_token_t* token) {
    if (!sup || !sup->initialized || !token || token->service_id >= ESHKOL_OS_SUP_MAX_SERVICES) {
        return ESHKOL_K56_ERR_INVALID_ARG;
    }
    const eshkol_os_svc_record_t* rec = &sup->services[token->service_id];
    if (rec->state == ESHKOL_SVC_STATE_FREE) return ESHKOL_SUP_ERR_STALE_TOKEN;
    if (rec->root_revoked || rec->state == ESHKOL_SVC_STATE_REVOKED) {
        return ESHKOL_SUP_ERR_REVOKED_CAP;
    }
    if (rec->state == ESHKOL_SVC_STATE_FAULTED) {
        return (token->generation != rec->generation)
                   ? ESHKOL_SUP_ERR_STALE_TOKEN
                   : ESHKOL_SUP_ERR_SERVICE_FAULTED;
    }
    if (token->generation == 0ULL ||
        token->generation != rec->generation ||
        token->instance_cap != rec->instance_cap) {
        return ESHKOL_SUP_ERR_STALE_TOKEN;
    }
    return ESHKOL_K56_OK;
}

eshkol_os_k56_status_t eshkol_os_supervisor_lock_acquire(
    eshkol_os_supervisor_t* sup,
    const eshkol_os_svc_token_t* token,
    uint32_t lock_id) {
    if (!sup || !sup->initialized || lock_id >= ESHKOL_OS_SUP_MAX_LOCKS) {
        return ESHKOL_K56_ERR_INVALID_ARG;
    }
    eshkol_os_k56_status_t st = eshkol_os_supervisor_validate_token(sup, token);
    if (st != ESHKOL_K56_OK) return st;

    eshkol_os_sup_lock_t* lk = &sup->locks[lock_id];
    if (lk->held) {
        /* Defensive check: if lock owner's generation is stale or faulted, reclaim atomically */
        const eshkol_os_svc_record_t* owner = &sup->services[lk->owner_service_id];
        if (owner->state != ESHKOL_SVC_STATE_RUNNING ||
            owner->generation != lk->owner_generation) {
            lk->held = 0u;
            lk->reclaim_count++;
            sup->total_lock_reclaims++;
        } else {
            return ESHKOL_SUP_ERR_LOCK_HELD;
        }
    }

    lk->held = 1u;
    lk->owner_service_id = token->service_id;
    lk->owner_generation = token->generation;
    lk->acquire_count++;
    return ESHKOL_K56_OK;
}

eshkol_os_k56_status_t eshkol_os_supervisor_lock_release(
    eshkol_os_supervisor_t* sup,
    const eshkol_os_svc_token_t* token,
    uint32_t lock_id) {
    if (!sup || !sup->initialized || lock_id >= ESHKOL_OS_SUP_MAX_LOCKS) {
        return ESHKOL_K56_ERR_INVALID_ARG;
    }
    eshkol_os_k56_status_t st = eshkol_os_supervisor_validate_token(sup, token);
    if (st != ESHKOL_K56_OK) return st;

    eshkol_os_sup_lock_t* lk = &sup->locks[lock_id];
    if (!lk->held || lk->owner_service_id != token->service_id ||
        lk->owner_generation != token->generation) {
        return ESHKOL_SUP_ERR_LOCK_NOT_OWNED;
    }
    lk->held = 0u;
    lk->owner_service_id = 0u;
    lk->owner_generation = 0ULL;
    return ESHKOL_K56_OK;
}

eshkol_os_k56_status_t eshkol_os_supervisor_inject_fault(
    eshkol_os_supervisor_t* sup,
    uint32_t service_id,
    eshkol_os_svc_fault_t fault) {
    if (!sup || !sup->initialized || service_id >= ESHKOL_OS_SUP_MAX_SERVICES ||
        fault == ESHKOL_SVC_FAULT_NONE) {
        return ESHKOL_K56_ERR_INVALID_ARG;
    }
    eshkol_os_svc_record_t* rec = &sup->services[service_id];
    if (rec->state == ESHKOL_SVC_STATE_FREE) return ESHKOL_K56_ERR_INVALID_ARG;

    /* Revoke the faulted instance's capability immediately */
    if (rec->instance_cap != ESHKOL_CAP_ID_INVALID) {
        (void)sup->syscall_fn(
            ESHKOL_SYS_CAP_REVOKE, rec->instance_cap, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);
        rec->instance_cap = ESHKOL_CAP_ID_INVALID;
    }

    /* Atomically release every exclusive lock held by the faulted instance */
    sup_reclaim_service_locks(sup, service_id);

    /* Advance generation so every outstanding token from this instance is immediately stale */
    rec->generation = sup->next_generation++;
    rec->last_fault = (uint32_t)fault;
    rec->fault_count++;
    sup->total_faults++;
    if (!rec->root_revoked) {
        rec->state = ESHKOL_SVC_STATE_FAULTED;
    }
    return ESHKOL_K56_OK;
}

eshkol_os_k56_status_t eshkol_os_supervisor_restart(
    eshkol_os_supervisor_t* sup,
    uint32_t service_id,
    eshkol_os_svc_token_t* out_new_token) {
    if (!sup || !sup->initialized || service_id >= ESHKOL_OS_SUP_MAX_SERVICES ||
        !out_new_token) {
        return ESHKOL_K56_ERR_INVALID_ARG;
    }
    eshkol_os_svc_record_t* rec = &sup->services[service_id];
    if (rec->state == ESHKOL_SVC_STATE_FREE) return ESHKOL_K56_ERR_INVALID_ARG;

    /* Distinct fail-closed error when the service's root capability was revoked */
    if (rec->root_revoked || rec->state == ESHKOL_SVC_STATE_REVOKED) {
        return ESHKOL_SUP_ERR_REVOKED_CAP;
    }

    /* Ensure zero residual locks or capabilities from any prior instance */
    if (rec->instance_cap != ESHKOL_CAP_ID_INVALID) {
        (void)sup->syscall_fn(
            ESHKOL_SYS_CAP_REVOKE, rec->instance_cap, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);
        rec->instance_cap = ESHKOL_CAP_ID_INVALID;
    }
    sup_reclaim_service_locks(sup, service_id);

    uint64_t new_gen = sup->next_generation++;
    int64_t derived = sup->syscall_fn(
        ESHKOL_SYS_CAP_DERIVE, rec->root_cap, rec->rights, new_gen, 0ULL, 0ULL, 0ULL);
    if (derived == ESHKOL_OS_ERR_REVOKED ||
        derived == ESHKOL_OS_ERR_RIGHTS_DENIED ||
        derived == ESHKOL_OS_ERR_INVALID_CAP) {
        rec->root_revoked = 1u;
        rec->state = ESHKOL_SVC_STATE_REVOKED;
        return ESHKOL_SUP_ERR_REVOKED_CAP;
    }
    if (derived <= 0) return ESHKOL_K56_ERR_SYSCALL_FAILED;

    rec->generation = new_gen;
    rec->instance_cap = (eshkol_cap_id_t)derived;
    rec->state = ESHKOL_SVC_STATE_RUNNING;
    rec->restart_count++;
    sup->total_restarts++;

    out_new_token->service_id = service_id;
    out_new_token->reserved = 0u;
    out_new_token->generation = new_gen;
    out_new_token->instance_cap = rec->instance_cap;
    return ESHKOL_K56_OK;
}

eshkol_os_k56_status_t eshkol_os_supervisor_revoke_root_cap(
    eshkol_os_supervisor_t* sup,
    eshkol_cap_id_t root_cap) {
    if (!sup || !sup->initialized || root_cap == ESHKOL_CAP_ID_INVALID) {
        return ESHKOL_K56_ERR_INVALID_ARG;
    }
    int64_t rc = sup->syscall_fn(
        ESHKOL_SYS_CAP_REVOKE, root_cap, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);
    if (rc < 0) return ESHKOL_K56_ERR_SYSCALL_FAILED;

    for (uint32_t i = 0u; i < ESHKOL_OS_SUP_MAX_SERVICES; i++) {
        if (sup->services[i].state != ESHKOL_SVC_STATE_FREE &&
            sup->services[i].root_cap == root_cap) {
            sup->services[i].root_revoked = 1u;
            sup->services[i].state = ESHKOL_SVC_STATE_REVOKED;
            sup->services[i].instance_cap = ESHKOL_CAP_ID_INVALID;
            sup->services[i].generation = sup->next_generation++;
            sup_reclaim_service_locks(sup, i);
        }
    }
    return ESHKOL_K56_OK;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Part 2: Phase K5 Framebuffer, Input & Real-Time Audio Compositor
 * ═══════════════════════════════════════════════════════════════════════════ */

static int rect_overlaps(uint32_t ax, uint32_t ay, uint32_t aw, uint32_t ah,
                         uint32_t bx, uint32_t by, uint32_t bw, uint32_t bh) {
    if (aw == 0u || ah == 0u || bw == 0u || bh == 0u) return 0;
    if ((uint64_t)ax + (uint64_t)aw <= (uint64_t)bx) return 0;
    if ((uint64_t)bx + (uint64_t)bw <= (uint64_t)ax) return 0;
    if ((uint64_t)ay + (uint64_t)ah <= (uint64_t)by) return 0;
    if ((uint64_t)by + (uint64_t)bh <= (uint64_t)ay) return 0;
    return 1;
}

static int rect_contains(uint32_t ox, uint32_t oy, uint32_t ow, uint32_t oh,
                         uint32_t ix, uint32_t iy, uint32_t iw, uint32_t ih) {
    if (ix < ox || iy < oy) return 0;
    if ((uint64_t)ix + (uint64_t)iw > (uint64_t)ox + (uint64_t)ow) return 0;
    if ((uint64_t)iy + (uint64_t)ih > (uint64_t)oy + (uint64_t)oh) return 0;
    return 1;
}

eshkol_os_k56_status_t eshkol_os_compositor_init(
    eshkol_os_compositor_t* comp,
    uint32_t* framebuffer,
    uint32_t width,
    uint32_t height) {
    if (!comp || !framebuffer || width == 0u || height == 0u) {
        return ESHKOL_K56_ERR_INVALID_ARG;
    }
    k56_zero(comp, sizeof(*comp));
    comp->framebuffer = framebuffer;
    comp->fb_width = width;
    comp->fb_height = height;
    comp->next_generation = 1ULL;
    comp->initialized = 1u;
    k56_zero(framebuffer, (size_t)width * (size_t)height * sizeof(uint32_t));
    return ESHKOL_K56_OK;
}

eshkol_os_k56_status_t eshkol_os_compositor_create_surface(
    eshkol_os_compositor_t* comp,
    uint32_t owner_badge,
    eshkol_cap_id_t cap_id,
    uint64_t rights,
    uint32_t x,
    uint32_t y,
    uint32_t width,
    uint32_t height,
    eshkol_os_surface_handle_t* out_handle) {
    if (!comp || !comp->initialized || owner_badge == 0u ||
        cap_id == ESHKOL_CAP_ID_INVALID || width == 0u || height == 0u || !out_handle) {
        return ESHKOL_K56_ERR_INVALID_ARG;
    }
    if ((uint64_t)x + (uint64_t)width > (uint64_t)comp->fb_width ||
        (uint64_t)y + (uint64_t)height > (uint64_t)comp->fb_height) {
        return ESHKOL_COMP_ERR_BOUNDS;
    }

    int is_writable = (rights & ESHKOL_CAP_RIGHT_WRITE) != 0ULL;
    int free_slot = -1;
    for (uint32_t i = 0u; i < ESHKOL_OS_COMP_MAX_SURFACES; i++) {
        const eshkol_os_surface_desc_t* s = &comp->surfaces[i];
        if (!s->active) {
            if (free_slot < 0) free_slot = (int)i;
            continue;
        }
        /* Disallow overlapping writable surfaces belonging to distinct capability owners */
        if (is_writable && (s->rights & ESHKOL_CAP_RIGHT_WRITE) != 0ULL &&
            s->owner_badge != owner_badge &&
            rect_overlaps(x, y, width, height, s->x, s->y, s->width, s->height)) {
            comp->overlap_rejects++;
            return ESHKOL_COMP_ERR_SURFACE_CONFLICT;
        }
    }
    if (free_slot < 0) return ESHKOL_K56_ERR_NO_SLOTS;

    uint64_t gen = comp->next_generation++;
    eshkol_os_surface_desc_t* s = &comp->surfaces[free_slot];
    s->active = 1u;
    s->owner_badge = owner_badge;
    s->x = x;
    s->y = y;
    s->width = width;
    s->height = height;
    s->generation = gen;
    s->cap_id = cap_id;
    s->rights = rights;

    out_handle->surface_id = (uint32_t)free_slot;
    out_handle->owner_badge = owner_badge;
    out_handle->generation = gen;
    out_handle->cap_id = cap_id;
    return ESHKOL_K56_OK;
}

static eshkol_os_k56_status_t comp_validate_surface(
    const eshkol_os_compositor_t* comp,
    const eshkol_os_surface_handle_t* handle) {
    if (!comp || !comp->initialized || !handle ||
        handle->surface_id >= ESHKOL_OS_COMP_MAX_SURFACES) {
        return ESHKOL_COMP_ERR_BAD_SURFACE;
    }
    const eshkol_os_surface_desc_t* s = &comp->surfaces[handle->surface_id];
    if (!s->active || s->generation != handle->generation ||
        s->owner_badge != handle->owner_badge || s->cap_id != handle->cap_id) {
        return ESHKOL_COMP_ERR_BAD_SURFACE;
    }
    return ESHKOL_K56_OK;
}

eshkol_os_k56_status_t eshkol_os_compositor_write_rect(
    eshkol_os_compositor_t* comp,
    const eshkol_os_surface_handle_t* handle,
    uint32_t abs_x,
    uint32_t abs_y,
    uint32_t width,
    uint32_t height,
    const uint32_t* pixels) {
    if (!pixels || width == 0u || height == 0u) return ESHKOL_K56_ERR_INVALID_ARG;
    eshkol_os_k56_status_t st = comp_validate_surface(comp, handle);
    if (st != ESHKOL_K56_OK) return st;

    const eshkol_os_surface_desc_t* s = &comp->surfaces[handle->surface_id];

    /* Explicit trap on read-only framebuffer capability write attempts */
    if ((s->rights & ESHKOL_CAP_RIGHT_WRITE) == 0ULL) {
        comp->ro_traps++;
        return ESHKOL_COMP_ERR_RO_TRAP;
    }
    if ((uint64_t)abs_x + (uint64_t)width > (uint64_t)comp->fb_width ||
        (uint64_t)abs_y + (uint64_t)height > (uint64_t)comp->fb_height) {
        return ESHKOL_COMP_ERR_BOUNDS;
    }

    /* Check conflict against any other service's active writable surface first */
    for (uint32_t i = 0u; i < ESHKOL_OS_COMP_MAX_SURFACES; i++) {
        const eshkol_os_surface_desc_t* other = &comp->surfaces[i];
        if (!other->active || other->owner_badge == s->owner_badge) continue;
        if ((other->rights & ESHKOL_CAP_RIGHT_WRITE) != 0ULL &&
            rect_overlaps(abs_x, abs_y, width, height,
                          other->x, other->y, other->width, other->height)) {
            comp->overlap_rejects++;
            return ESHKOL_COMP_ERR_SURFACE_CONFLICT;
        }
    }

    /* Enforce strict confinement to caller's own allocated surface rectangle */
    if (!rect_contains(s->x, s->y, s->width, s->height, abs_x, abs_y, width, height)) {
        comp->overlap_rejects++;
        return ESHKOL_COMP_ERR_SURFACE_CONFLICT;
    }

    /* Atomic full-rectangle blit (only after every capability & spatial check passed) */
    for (uint32_t row = 0u; row < height; row++) {
        uint32_t* dst_row = comp->framebuffer + (size_t)(abs_y + row) * (size_t)comp->fb_width + abs_x;
        const uint32_t* src_row = pixels + (size_t)row * (size_t)width;
        k56_copy(dst_row, src_row, (size_t)width * sizeof(uint32_t));
    }
    comp->writes_committed++;
    return ESHKOL_K56_OK;
}

eshkol_os_k56_status_t eshkol_os_compositor_read_pixel(
    const eshkol_os_compositor_t* comp,
    const eshkol_os_surface_handle_t* handle,
    uint32_t abs_x,
    uint32_t abs_y,
    uint32_t* out_pixel) {
    if (!out_pixel) return ESHKOL_K56_ERR_INVALID_ARG;
    eshkol_os_k56_status_t st = comp_validate_surface(comp, handle);
    if (st != ESHKOL_K56_OK) return st;

    const eshkol_os_surface_desc_t* s = &comp->surfaces[handle->surface_id];
    if ((s->rights & ESHKOL_CAP_RIGHT_READ) == 0ULL) return ESHKOL_COMP_ERR_BAD_SURFACE;
    if (!rect_contains(s->x, s->y, s->width, s->height, abs_x, abs_y, 1u, 1u)) {
        return ESHKOL_COMP_ERR_BOUNDS;
    }
    *out_pixel = comp->framebuffer[(size_t)abs_y * (size_t)comp->fb_width + abs_x];
    return ESHKOL_K56_OK;
}

eshkol_os_k56_status_t eshkol_os_compositor_push_input(
    eshkol_os_compositor_t* comp,
    uint32_t target_badge,
    uint32_t event_type,
    uint32_t code,
    int32_t value) {
    if (!comp || !comp->initialized || target_badge == 0u) return ESHKOL_K56_ERR_INVALID_ARG;
    if (comp->input_count >= ESHKOL_OS_COMP_MAX_INPUT_EVENTS) return ESHKOL_COMP_ERR_QUEUE_FULL;

    eshkol_os_input_event_t* ev = &comp->input_ring[comp->input_tail];
    ev->target_badge = target_badge;
    ev->event_type = event_type;
    ev->code = code;
    ev->value = value;
    ev->sequence = ++comp->input_seq;

    comp->input_tail = (comp->input_tail + 1u) % ESHKOL_OS_COMP_MAX_INPUT_EVENTS;
    comp->input_count++;
    return ESHKOL_K56_OK;
}

eshkol_os_k56_status_t eshkol_os_compositor_poll_input(
    eshkol_os_compositor_t* comp,
    uint32_t owner_badge,
    eshkol_os_input_event_t* out_event) {
    if (!comp || !comp->initialized || owner_badge == 0u || !out_event) {
        return ESHKOL_K56_ERR_INVALID_ARG;
    }
    if (comp->input_count == 0u) return ESHKOL_COMP_ERR_QUEUE_EMPTY;

    const eshkol_os_input_event_t* ev = &comp->input_ring[comp->input_head];
    if (ev->target_badge != owner_badge) return ESHKOL_COMP_ERR_QUEUE_EMPTY;

    k56_copy(out_event, ev, sizeof(*out_event));
    comp->input_head = (comp->input_head + 1u) % ESHKOL_OS_COMP_MAX_INPUT_EVENTS;
    comp->input_count--;
    return ESHKOL_K56_OK;
}

eshkol_os_k56_status_t eshkol_os_compositor_audio_write(
    eshkol_os_compositor_t* comp,
    const eshkol_os_audio_frame_t* frames,
    uint32_t frame_count,
    uint32_t* out_accepted,
    uint32_t* out_dropped) {
    if (!comp || !comp->initialized || (!frames && frame_count > 0u)) {
        return ESHKOL_K56_ERR_INVALID_ARG;
    }
    eshkol_os_audio_ring_t* ring = &comp->audio;
    uint32_t free_slots = ESHKOL_OS_AUDIO_RING_FRAMES - ring->count;
    uint32_t accept = (frame_count <= free_slots) ? frame_count : free_slots;
    uint32_t dropped = frame_count - accept;

    for (uint32_t i = 0u; i < accept; i++) {
        ring->frames[ring->write_idx].left = frames[i].left;
        ring->frames[ring->write_idx].right = frames[i].right;
        ring->write_idx = (ring->write_idx + 1u) % ESHKOL_OS_AUDIO_RING_FRAMES;
        ring->count++;
    }
    ring->frames_written += accept;
    if (dropped > 0u) {
        ring->overrun_events++;
        ring->dropped_frames += dropped;
    }

    if (out_accepted) *out_accepted = accept;
    if (out_dropped) *out_dropped = dropped;
    return ESHKOL_K56_OK;
}

eshkol_os_k56_status_t eshkol_os_compositor_audio_read(
    eshkol_os_compositor_t* comp,
    eshkol_os_audio_frame_t* out_frames,
    uint32_t requested_frames,
    uint32_t* out_from_ring,
    uint32_t* out_silence_filled) {
    if (!comp || !comp->initialized || (!out_frames && requested_frames > 0u)) {
        return ESHKOL_K56_ERR_INVALID_ARG;
    }
    eshkol_os_audio_ring_t* ring = &comp->audio;
    uint32_t avail = (requested_frames <= ring->count) ? requested_frames : ring->count;
    uint32_t silence = requested_frames - avail;

    for (uint32_t i = 0u; i < avail; i++) {
        out_frames[i].left = ring->frames[ring->read_idx].left;
        out_frames[i].right = ring->frames[ring->read_idx].right;
        ring->read_idx = (ring->read_idx + 1u) % ESHKOL_OS_AUDIO_RING_FRAMES;
        ring->count--;
    }
    for (uint32_t i = 0u; i < silence; i++) {
        out_frames[avail + i].left = 0;
        out_frames[avail + i].right = 0;
    }
    ring->frames_read += avail;
    if (silence > 0u) {
        ring->underrun_events++;
        ring->silence_filled_frames += silence;
    }

    if (out_from_ring) *out_from_ring = avail;
    if (out_silence_filled) *out_silence_filled = silence;
    return ESHKOL_K56_OK;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Part 3: Phase K6 Self-Hosting Reproducible Build Convergence in CAS VFS
 * ═══════════════════════════════════════════════════════════════════════════ */

eshkol_os_k56_status_t eshkol_os_k6_validate_wx_elf64(
    const uint8_t* image,
    size_t image_len,
    eshkol_os_target_isa_t expected_isa) {
    if (!image || image_len < 4096u + 64u) return ESHKOL_K6_ERR_BAD_ELF;
    if (image[0] != 0x7Fu || image[1] != 'E' || image[2] != 'L' || image[3] != 'F') {
        return ESHKOL_K6_ERR_BAD_ELF;
    }
    if (image[4] != 2u || image[5] != 1u || image[6] != 1u) {
        return ESHKOL_K6_ERR_BAD_ELF;
    }
    uint16_t e_type = k56_rd_u16_le(image + 16u);
    uint16_t e_machine = k56_rd_u16_le(image + 18u);
    uint64_t e_phoff = k56_rd_u64_le(image + 32u);
    uint16_t e_phentsize = k56_rd_u16_le(image + 54u);
    uint16_t e_phnum = k56_rd_u16_le(image + 56u);

    uint16_t want_mach = (expected_isa == ESHKOL_OS_TARGET_X86_64) ? 62u :
                         (expected_isa == ESHKOL_OS_TARGET_AARCH64) ? 183u :
                         (expected_isa == ESHKOL_OS_TARGET_RISCV64) ? 243u : 0u;
    if (want_mach == 0u || e_type != 2u || e_machine != want_mach ||
        e_phentsize != 56u || e_phnum == 0u ||
        e_phoff + (uint64_t)e_phnum * 56ULL > (uint64_t)image_len) {
        return ESHKOL_K6_ERR_BAD_ELF;
    }

    int saw_rx = 0;
    for (uint16_t i = 0u; i < e_phnum; i++) {
        const uint8_t* ph = image + e_phoff + (size_t)i * 56u;
        uint32_t p_type = k56_rd_u32_le(ph + 0u);
        uint32_t p_flags = k56_rd_u32_le(ph + 4u);
        uint64_t p_offset = k56_rd_u64_le(ph + 8u);
        uint64_t p_filesz = k56_rd_u64_le(ph + 32u);
        /* Only PT_LOAD allowed (zero PT_INTERP / PT_DYNAMIC) */
        if (p_type != 1u) return ESHKOL_K6_ERR_BAD_ELF;
        /* Strict W^X: never both PF_W (2) and PF_X (1) */
        if ((p_flags & 3u) == 3u) return ESHKOL_K6_ERR_BAD_ELF;
        if (p_offset + p_filesz > (uint64_t)image_len) return ESHKOL_K6_ERR_BAD_ELF;
        if (p_flags == 5u) saw_rx = 1;
    }
    return saw_rx ? ESHKOL_K56_OK : ESHKOL_K6_ERR_BAD_ELF;
}

static eshkol_os_k56_status_t k6_validate_ops(const eshkol_os_k6_builder_ops_t* ops) {
    if (!ops || !ops->vfs_read_file || !ops->vfs_put_file || !ops->sha256 ||
        !ops->compile_to_eskb || !ops->compile_to_elf64) {
        return ESHKOL_K56_ERR_INVALID_ARG;
    }
    return ESHKOL_K56_OK;
}

static eshkol_os_k56_status_t k6_compile_single(
    const eshkol_os_k6_builder_ops_t* ops,
    const eshkol_os_k6_build_request_t* req,
    const uint8_t* src_bytes,
    size_t src_len,
    void* workspace,
    size_t workspace_bytes,
    uint8_t poison,
    uint64_t salt,
    uint8_t* out_buf,
    size_t out_cap,
    size_t* out_len,
    uint8_t out_sha[ESHKOL_OS_SHA256_BYTES]) {
    k56_fill(workspace, poison, workspace_bytes);
    k56_fill(out_buf, (uint8_t)(poison ^ 0xFFu), out_cap);

    eshkol_os_compile_result_t cres;
    k56_zero(&cres, sizeof(cres));
    eshkol_os_self_host_status_t cst;
    if (req->artifact_kind == ESHKOL_K6_ARTIFACT_USER_ESKB) {
        cst = ops->compile_to_eskb(
            (const char*)src_bytes, src_len, workspace, workspace_bytes,
            out_buf, out_cap, &cres);
    } else {
        cst = ops->compile_to_elf64(
            (const char*)src_bytes, src_len, req->target_isa, workspace, workspace_bytes,
            out_buf, out_cap, &cres);
    }
    if (cst != ESHKOL_SH_OK) return ESHKOL_K6_ERR_COMPILE_FAILED;

    size_t final_len = cres.bytes_written;
    if (req->artifact_kind != ESHKOL_K6_ARTIFACT_USER_ESKB) {
        eshkol_os_k56_status_t est = eshkol_os_k6_validate_wx_elf64(
            out_buf, final_len, req->target_isa);
        if (est != ESHKOL_K56_OK) return est;
    }

    /* Optional injected non-deterministic metadata (used by negative controls) */
    if (salt != 0ULL) {
        if (req->artifact_kind == ESHKOL_K6_ARTIFACT_USER_ESKB) {
            if (final_len + 8u > out_cap) return ESHKOL_K6_ERR_BUF_TOO_SMALL;
            k56_wr_u64_le(out_buf + final_len, salt);
            final_len += 8u;
        } else {
            /* Stamp into the .data metadata segment tail (bytes 8..15 of 16-byte .data) */
            uint64_t data_off = k56_rd_u64_le(out_buf + 120u + 8u);
            uint64_t data_sz = k56_rd_u64_le(out_buf + 120u + 32u);
            if (data_sz < 16u || data_off + data_sz > (uint64_t)final_len) {
                return ESHKOL_K6_ERR_BAD_ELF;
            }
            k56_wr_u64_le(out_buf + data_off + 8u, salt);
        }
    }

    ops->sha256(out_buf, final_len, out_sha);
    *out_len = final_len;
    return ESHKOL_K56_OK;
}

eshkol_os_k56_status_t eshkol_os_k6_build_selfhost(
    eshkol_os_cas_vfs_t* vfs,
    const eshkol_os_k6_builder_ops_t* ops,
    const eshkol_os_k6_build_request_t* req,
    eshkol_os_k6_build_buffers_t* bufs,
    eshkol_os_k6_build_receipt_t* out_receipt) {
    eshkol_os_k6_build_receipt_t rec;
    k56_zero(&rec, sizeof(rec));

    if (!vfs || k6_validate_ops(ops) != ESHKOL_K56_OK || !req || !req->source_name ||
        !req->output_name || !bufs || !bufs->compiler_workspace ||
        !bufs->source_buf || !bufs->out_buf_a) {
        rec.status = ESHKOL_K56_ERR_INVALID_ARG;
        if (out_receipt) k56_copy(out_receipt, &rec, sizeof(rec));
        return rec.status;
    }

    size_t src_len = 0u;
    if (ops->vfs_read_file(vfs, req->source_name, req->source_parent_id,
                           bufs->source_buf, bufs->source_cap,
                           &src_len, rec.source_sha256) != ESHKOL_DRV_OK) {
        rec.status = ESHKOL_K6_ERR_VFS_FAILED;
        if (out_receipt) k56_copy(out_receipt, &rec, sizeof(rec));
        return rec.status;
    }
    rec.source_bytes = src_len;

    size_t art_len = 0u;
    eshkol_os_k56_status_t st = k6_compile_single(
        ops, req, bufs->source_buf, src_len,
        bufs->compiler_workspace, bufs->compiler_workspace_bytes,
        req->workspace_poison, req->nondet_salt,
        bufs->out_buf_a, bufs->out_cap_a, &art_len, rec.artifact_sha256);
    if (st != ESHKOL_K56_OK) {
        rec.status = st;
        if (out_receipt) k56_copy(out_receipt, &rec, sizeof(rec));
        return st;
    }

    uint8_t committed_sha[ESHKOL_OS_SHA256_BYTES];
    if (ops->vfs_put_file(vfs, req->output_name, req->output_parent_id,
                          bufs->out_buf_a, art_len, committed_sha) != ESHKOL_DRV_OK) {
        rec.status = ESHKOL_K6_ERR_VFS_FAILED;
        if (out_receipt) k56_copy(out_receipt, &rec, sizeof(rec));
        return rec.status;
    }
    if (!k56_memeq(committed_sha, rec.artifact_sha256, ESHKOL_OS_SHA256_BYTES)) {
        rec.status = ESHKOL_K6_ERR_VFS_FAILED;
        if (out_receipt) k56_copy(out_receipt, &rec, sizeof(rec));
        return rec.status;
    }

    rec.status = ESHKOL_K56_OK;
    rec.artifact_bytes = art_len;
    k56_copy(rec.pass2_sha256, rec.artifact_sha256, ESHKOL_OS_SHA256_BYTES);
    if (out_receipt) k56_copy(out_receipt, &rec, sizeof(rec));
    return ESHKOL_K56_OK;
}

eshkol_os_k56_status_t eshkol_os_k6_verify_reproducible(
    eshkol_os_cas_vfs_t* vfs,
    const eshkol_os_k6_builder_ops_t* ops,
    const eshkol_os_k6_build_request_t* req,
    uint8_t poison_pass_a,
    uint8_t poison_pass_b,
    uint64_t salt_pass_a,
    uint64_t salt_pass_b,
    eshkol_os_k6_build_buffers_t* bufs,
    eshkol_os_k6_build_receipt_t* out_receipt) {
    eshkol_os_k6_build_receipt_t rec;
    k56_zero(&rec, sizeof(rec));

    if (!vfs || k6_validate_ops(ops) != ESHKOL_K56_OK || !req || !req->source_name ||
        !req->output_name || !bufs || !bufs->compiler_workspace ||
        bufs->compiler_workspace_bytes < ESHKOL_OS_SH_MIN_WORKSPACE_BYTES + 16u ||
        !bufs->source_buf || !bufs->out_buf_a || !bufs->out_buf_b) {
        rec.status = ESHKOL_K56_ERR_INVALID_ARG;
        if (out_receipt) k56_copy(out_receipt, &rec, sizeof(rec));
        return rec.status;
    }

    size_t src_len = 0u;
    if (ops->vfs_read_file(vfs, req->source_name, req->source_parent_id,
                           bufs->source_buf, bufs->source_cap,
                           &src_len, rec.source_sha256) != ESHKOL_DRV_OK) {
        rec.status = ESHKOL_K6_ERR_VFS_FAILED;
        if (out_receipt) k56_copy(out_receipt, &rec, sizeof(rec));
        return rec.status;
    }
    rec.source_bytes = src_len;

    /* Pass A: poison_pass_a at offset +0 */
    size_t len_a = 0u;
    eshkol_os_k56_status_t st = k6_compile_single(
        ops, req, bufs->source_buf, src_len,
        bufs->compiler_workspace, bufs->compiler_workspace_bytes - 16u,
        poison_pass_a, salt_pass_a,
        bufs->out_buf_a, bufs->out_cap_a, &len_a, rec.artifact_sha256);
    if (st != ESHKOL_K56_OK) {
        rec.status = st;
        if (out_receipt) k56_copy(out_receipt, &rec, sizeof(rec));
        return st;
    }

    /* Pass B: poison_pass_b at shifted offset +13 */
    size_t len_b = 0u;
    st = k6_compile_single(
        ops, req, bufs->source_buf, src_len,
        ((uint8_t*)bufs->compiler_workspace) + 13u, bufs->compiler_workspace_bytes - 16u,
        poison_pass_b, salt_pass_b,
        bufs->out_buf_b, bufs->out_cap_b, &len_b, rec.pass2_sha256);
    if (st != ESHKOL_K56_OK) {
        rec.status = st;
        if (out_receipt) k56_copy(out_receipt, &rec, sizeof(rec));
        return st;
    }

    /* Fail closed on ANY digest or byte divergence */
    if (len_a != len_b ||
        !k56_memeq(rec.artifact_sha256, rec.pass2_sha256, ESHKOL_OS_SHA256_BYTES) ||
        !k56_memeq(bufs->out_buf_a, bufs->out_buf_b, len_a)) {
        rec.status = ESHKOL_K6_ERR_NONDETERMINISTIC;
        rec.artifact_bytes = len_a;
        if (out_receipt) k56_copy(out_receipt, &rec, sizeof(rec));
        return ESHKOL_K6_ERR_NONDETERMINISTIC;
    }

    /* Commit verified bit-identical artifact atomically into K4 CAS VFS */
    uint8_t committed_sha[ESHKOL_OS_SHA256_BYTES];
    if (ops->vfs_put_file(vfs, req->output_name, req->output_parent_id,
                          bufs->out_buf_a, len_a, committed_sha) != ESHKOL_DRV_OK ||
        !k56_memeq(committed_sha, rec.artifact_sha256, ESHKOL_OS_SHA256_BYTES)) {
        rec.status = ESHKOL_K6_ERR_VFS_FAILED;
        if (out_receipt) k56_copy(out_receipt, &rec, sizeof(rec));
        return rec.status;
    }

    rec.status = ESHKOL_K56_OK;
    rec.artifact_bytes = len_a;
    if (out_receipt) k56_copy(out_receipt, &rec, sizeof(rec));
    return ESHKOL_K56_OK;
}

const char* eshkol_os_k56_status_name(eshkol_os_k56_status_t status) {
    switch (status) {
    case ESHKOL_K56_OK:                    return "ESHKOL_K56_OK";
    case ESHKOL_K56_ERR_INVALID_ARG:       return "ESHKOL_K56_ERR_INVALID_ARG";
    case ESHKOL_K56_ERR_NO_SLOTS:          return "ESHKOL_K56_ERR_NO_SLOTS";
    case ESHKOL_K56_ERR_SYSCALL_FAILED:    return "ESHKOL_K56_ERR_SYSCALL_FAILED";
    case ESHKOL_SUP_ERR_STALE_TOKEN:       return "ESHKOL_SUP_ERR_STALE_TOKEN";
    case ESHKOL_SUP_ERR_REVOKED_CAP:       return "ESHKOL_SUP_ERR_REVOKED_CAP";
    case ESHKOL_SUP_ERR_SERVICE_FAULTED:   return "ESHKOL_SUP_ERR_SERVICE_FAULTED";
    case ESHKOL_SUP_ERR_LOCK_HELD:         return "ESHKOL_SUP_ERR_LOCK_HELD";
    case ESHKOL_SUP_ERR_LOCK_NOT_OWNED:    return "ESHKOL_SUP_ERR_LOCK_NOT_OWNED";
    case ESHKOL_SUP_ERR_DUAL_OWNER:        return "ESHKOL_SUP_ERR_DUAL_OWNER";
    case ESHKOL_COMP_ERR_SURFACE_CONFLICT: return "ESHKOL_COMP_ERR_SURFACE_CONFLICT";
    case ESHKOL_COMP_ERR_RO_TRAP:          return "ESHKOL_COMP_ERR_RO_TRAP";
    case ESHKOL_COMP_ERR_BOUNDS:           return "ESHKOL_COMP_ERR_BOUNDS";
    case ESHKOL_COMP_ERR_BAD_SURFACE:      return "ESHKOL_COMP_ERR_BAD_SURFACE";
    case ESHKOL_COMP_ERR_QUEUE_EMPTY:      return "ESHKOL_COMP_ERR_QUEUE_EMPTY";
    case ESHKOL_COMP_ERR_QUEUE_FULL:       return "ESHKOL_COMP_ERR_QUEUE_FULL";
    case ESHKOL_K6_ERR_COMPILE_FAILED:     return "ESHKOL_K6_ERR_COMPILE_FAILED";
    case ESHKOL_K6_ERR_VFS_FAILED:         return "ESHKOL_K6_ERR_VFS_FAILED";
    case ESHKOL_K6_ERR_NONDETERMINISTIC:   return "ESHKOL_K6_ERR_NONDETERMINISTIC";
    case ESHKOL_K6_ERR_BAD_ELF:            return "ESHKOL_K6_ERR_BAD_ELF";
    case ESHKOL_K6_ERR_BUF_TOO_SMALL:      return "ESHKOL_K6_ERR_BUF_TOO_SMALL";
    }
    return "ESHKOL_K56_UNKNOWN";
}
