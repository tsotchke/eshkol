/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol OS Phase K5/K6 Formal Policy Verification & Unified K1-K3
 * Microkernel Convergence ABI.
 *
 * Pure freestanding C11/C++20 ABI with zero dynamic heap allocation,
 * zero C library dependencies, and zero unwind runtime symbols.
 */

#ifndef ESHKOL_KERNEL_K_KERNEL_POLICY_H
#define ESHKOL_KERNEL_K_KERNEL_POLICY_H

#include <stddef.h>
#include <stdint.h>

#include <eshkol/kernel/k1_boot_traps.h>
#include <eshkol/kernel/k2_mmu_frames.h>
#include <eshkol/kernel/k3_smp_cap_ipc.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Compile-time static capacities (zero dynamic heap allocation). */
#define K_POL_ABI_VERSION       1u
#define K_POL_MAX_SERVICES      8u
#define K_POL_MAX_LOCKS         8u
#define K_POL_MAX_SURFACES      8u
#define K_POL_FB_CELLS          64u
#define K_POL_AUDIO_RING_CAP    16u
#define K_POL_NO_OWNER          0xFFFFFFFFu

typedef enum k_pol_status {
    K_POL_OK                    = 0,
    K_POL_ERR_INVALID_ARG       = 1,
    K_POL_ERR_SERVICE_RANGE     = 2,
    K_POL_ERR_LOCK_RANGE        = 3,
    K_POL_ERR_LOCK_HELD         = 4,
    K_POL_ERR_REVOKED_CAP       = 5,
    K_POL_ERR_STALE_TOKEN       = 6,
    K_POL_ERR_NOT_FAULTED       = 7,
    K_POL_ERR_NOT_RUNNING       = 8,
    K_POL_ERR_SURFACE_RANGE     = 9,
    K_POL_ERR_SURFACE_CONFLICT  = 10,
    K_POL_ERR_RO_TRAP           = 11,
    K_POL_ERR_BOUNDS            = 12,
    K_POL_ERR_CONVERGENCE_BOOT  = 13
} k_pol_status_t;

typedef enum k_pol_svc_state {
    K_POL_SVC_STOPPED = 0,
    K_POL_SVC_RUNNING = 1,
    K_POL_SVC_FAULTED = 2
} k_pol_svc_state_t;

typedef struct k_policy_service {
    uint32_t service_id;
    uint32_t badge;
    k_pol_svc_state_t state;
    uint32_t root_cap_live;
    uint32_t token_gen;
    uint32_t last_faulted_gen;
    uint32_t restart_count;
    uint32_t fault_count;
    uint32_t held_lock_mask;
    uint32_t reserved;
} k_policy_service_t;

typedef struct k_policy_surface {
    uint32_t surface_id;
    uint32_t active;
    uint32_t owner_badge;
    uint32_t x_start;
    uint32_t x_len;
    uint32_t writable;
} k_policy_surface_t;

typedef struct k_policy_audio {
    uint32_t capacity;
    uint32_t head;
    uint32_t tail;
    uint32_t count;
    uint32_t overrun_drops;
    uint32_t underrun_silences;
    uint32_t supervisor_halted;
    uint32_t reserved;
    uint32_t samples[K_POL_AUDIO_RING_CAP];
} k_policy_audio_t;

typedef struct k_policy_state {
    uint32_t initialized;
    uint32_t ro_trap_count;
    uint32_t conflict_reject_count;
    uint32_t revoked_restart_rejects;
    k_policy_service_t services[K_POL_MAX_SERVICES];
    uint32_t lock_owner[K_POL_MAX_LOCKS];
    k_policy_surface_t surfaces[K_POL_MAX_SURFACES];
    uint32_t framebuffer[K_POL_FB_CELLS];
    k_policy_audio_t audio;
} k_policy_state_t;

/* Unified K1 + K2 + K3 + Policy Microkernel Convergence Container. */
typedef struct k_microkernel {
    k1_kernel_state_t k1;
    k2_frame_pool_t k2_frames;
    k2_aspace_t k2_aspaces[K_POL_MAX_SERVICES];
    k2_mpu_table_t k2_mpu;
    k3_smp_sched_t k3_sched;
    k3_cdt_t k3_cdt;
    k3_ipc_channel_t k3_channels[K_POL_MAX_SERVICES];
    k_policy_state_t policy;
} k_microkernel_t;

/* Supervisor policy, lock arbitration, compositor surface, and audio API. */
void k_policy_init(k_policy_state_t* pol, uint32_t audio_capacity);
k_pol_status_t k_policy_service_register(k_policy_state_t* pol,
                                         uint32_t service_id,
                                         uint32_t badge,
                                         uint32_t* out_token_gen);
k_pol_status_t k_policy_validate_token(const k_policy_state_t* pol,
                                       uint32_t service_id,
                                       uint32_t token_gen);
k_pol_status_t k_policy_lock_acquire(k_policy_state_t* pol,
                                     uint32_t service_id,
                                     uint32_t token_gen,
                                     uint32_t lock_id);
k_pol_status_t k_policy_lock_release(k_policy_state_t* pol,
                                     uint32_t service_id,
                                     uint32_t token_gen,
                                     uint32_t lock_id);
k_pol_status_t k_policy_service_fault(k_policy_state_t* pol,
                                      uint32_t service_id,
                                      uint32_t* out_released_locks);
k_pol_status_t k_policy_revoke_service_cap(k_policy_state_t* pol,
                                           uint32_t service_id,
                                           uint32_t* out_released_locks);
k_pol_status_t k_policy_service_restart(k_policy_state_t* pol,
                                        uint32_t service_id,
                                        uint32_t* out_new_token_gen);

k_pol_status_t k_policy_surface_register(k_policy_state_t* pol,
                                         uint32_t surface_id,
                                         uint32_t owner_badge,
                                         uint32_t x_start,
                                         uint32_t x_len,
                                         uint32_t writable);
k_pol_status_t k_policy_surface_write(k_policy_state_t* pol,
                                      uint32_t surface_id,
                                      uint32_t caller_badge,
                                      uint32_t cell_offset,
                                      uint32_t len,
                                      uint32_t pixel_value);
k_pol_status_t k_policy_audio_step(k_policy_state_t* pol,
                                   uint32_t is_produce,
                                   uint32_t sample_in,
                                   uint32_t* out_sample,
                                   uint32_t* out_degraded);
int k_policy_invariants_hold(const k_policy_state_t* pol);

/* Unified K1+K2+K3+Policy convergence orchestrator (zero-alloc header-inline). */
static inline k_pol_status_t k_microkernel_converge_boot(
    k_microkernel_t* mk,
    const k1_boot_info_t* boot_info,
    const k1_kernel_context_t* kernel_ctx,
    k2_mmu_mode_t mmu_mode,
    uint32_t total_frames,
    uint32_t kernel_reserved_frames) {
    if (!mk || !boot_info || !kernel_ctx) {
        return K_POL_ERR_INVALID_ARG;
    }
    if (k1_kernel_boot(&mk->k1, boot_info, kernel_ctx) != K1_OK) {
        return K_POL_ERR_CONVERGENCE_BOOT;
    }
    if (k2_frame_pool_init(&mk->k2_frames, boot_info->ram_base,
                           total_frames, kernel_reserved_frames) != K2_OK) {
        return K_POL_ERR_CONVERGENCE_BOOT;
    }
    if (k2_mpu_table_init(&mk->k2_mpu, 0x00000000ULL, 0x400ULL,
                          0x20000000ULL, 0x100ULL) != K2_OK) {
        return K_POL_ERR_CONVERGENCE_BOOT;
    }
    for (uint32_t i = 0u; i < K_POL_MAX_SERVICES; ++i) {
        if (k2_aspace_init(&mk->k2_aspaces[i], mmu_mode, i + 1u, 1u) != K2_OK) {
            return K_POL_ERR_CONVERGENCE_BOOT;
        }
        if (k3_ipc_channel_init(&mk->k3_channels[i], i + 1u, i, 0u) != K3_OK) {
            return K_POL_ERR_CONVERGENCE_BOOT;
        }
    }
    if (k3_smp_sched_init(&mk->k3_sched, boot_info->cpu_count) != K3_OK) {
        return K_POL_ERR_CONVERGENCE_BOOT;
    }
    k3_cdt_init(&mk->k3_cdt);
    k_policy_init(&mk->policy, K_POL_AUDIO_RING_CAP);
    return K_POL_OK;
}

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_KERNEL_K_KERNEL_POLICY_H */
