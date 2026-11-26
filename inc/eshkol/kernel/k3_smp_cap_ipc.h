/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol OS Phase K3: Verified Zero-Alloc SMP Scheduler, Capability
 * Derivation Tree (CDT) with Immediate Atomic Revocation, and Zero-Copy
 * Session-Typed submit/complete Ring IPC.
 *
 * Pure freestanding C11/C++20 ABI with zero dynamic heap allocation,
 * zero C library dependencies, and zero unwind runtime symbols across
 * x86_64, AArch64, RISC-V 64, RISC-V 32, and Cortex-M4.
 */

#ifndef ESHKOL_KERNEL_K3_SMP_CAP_IPC_H
#define ESHKOL_KERNEL_K3_SMP_CAP_IPC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__cplusplus)
#define K3_CACHELINE_ALIGN alignas(64)
#else
#define K3_CACHELINE_ALIGN _Alignas(64)
#endif

/* ABI version and compile-time static capacities (zero dynamic allocation). */
#define K3_ABI_VERSION          1u
#define K3_CACHELINE_BYTES      64u
#define K3_MAX_CPUS             8u
#define K3_MAX_TASKS            32u
#define K3_RUNQUEUE_CAP         32u
#define K3_MAX_CAPS             256u
#define K3_MAX_CDT_DEPTH        16u
#define K3_MAX_COMPARTMENTS     16u
#define K3_IPC_RING_CAP         64u
#define K3_IPC_MAX_PAYLOAD      64u
#define K3_SHM_PAGE_BYTES       256u
#define K3_INVALID_ID           0xFFFFFFFFu

/* Capability rights bitmask. */
#define K3_RIGHT_SEND           (1u << 0)
#define K3_RIGHT_RECV           (1u << 1)
#define K3_RIGHT_DERIVE         (1u << 2)
#define K3_RIGHT_REVOKE         (1u << 3)
#define K3_RIGHT_SHM_READ       (1u << 4)
#define K3_RIGHT_SHM_WRITE      (1u << 5)
#define K3_RIGHT_GRANT          (1u << 6)
#define K3_RIGHT_ALL            0x7Fu

typedef enum k3_status {
    K3_OK                       = 0,
    K3_ERR_INVALID_ARG          = 1,
    K3_ERR_CPU_OUT_OF_RANGE     = 2,
    K3_ERR_TASK_OUT_OF_RANGE    = 3,
    K3_ERR_RUNQUEUE_FULL        = 4,
    K3_ERR_RUNQUEUE_EMPTY       = 5,
    K3_ERR_CAP_POOL_FULL        = 6,
    K3_ERR_CAP_INVALID          = 7,
    K3_ERR_CAP_REVOKED          = 8,
    K3_ERR_CAP_DENIED           = 9,
    K3_ERR_CAP_CONFINEMENT      = 10,
    K3_ERR_RIGHTS_ESCALATION    = 11,
    K3_ERR_CDT_DEPTH_OVERFLOW   = 12,
    K3_ERR_CDT_CYCLE            = 13,
    K3_ERR_IPC_RING_FULL        = 14,
    K3_ERR_IPC_RING_EMPTY       = 15,
    K3_ERR_SESSION_PROTOCOL     = 16,
    K3_ERR_BUFFER_TOO_SMALL     = 17,
    K3_ERR_SHM_BOUNDS           = 18
} k3_status_t;

typedef enum k3_task_state {
    K3_TASK_UNUSED   = 0,
    K3_TASK_RUNNABLE = 1,
    K3_TASK_RUNNING  = 2,
    K3_TASK_BLOCKED  = 3,
    K3_TASK_FAULTED  = 4,
    K3_TASK_EXITED   = 5
} k3_task_state_t;

typedef enum k3_session_state {
    K3_SESSION_INIT          = 0,
    K3_SESSION_REQUEST_SENT  = 1,
    K3_SESSION_RESPONSE_SENT = 2,
    K3_SESSION_CLOSED        = 3
} k3_session_state_t;

typedef enum k3_ipc_msg_kind {
    K3_IPC_MSG_NONE     = 0,
    K3_IPC_MSG_REQUEST  = 1,
    K3_IPC_MSG_RESPONSE = 2,
    K3_IPC_MSG_CLOSE    = 3
} k3_ipc_msg_kind_t;

/* ─────────────────────────── SMP Scheduler Types ─────────────────────────── */

typedef struct k3_task {
    uint32_t task_id;
    uint32_t compartment_id;
    uint32_t cpu_affinity;
    uint32_t priority;
    k3_task_state_t state;
    uint32_t time_slice_ticks;
    uint64_t dispatch_count;
    uint64_t fault_code;
} k3_task_t;

/* Strictly 64-byte cache-line isolated per-CPU runqueue (zero false sharing). */
typedef struct k3_cpu_runqueue {
    K3_CACHELINE_ALIGN uint32_t cpu_id;
    uint32_t lock_word;
    uint32_t head;
    uint32_t tail;
    uint32_t count;
    uint32_t current_task_id;
    uint64_t context_switches;
    uint64_t preemptions;
    uint32_t queue[K3_RUNQUEUE_CAP];
    uint32_t pad[4]; /* Pad to 192 bytes (3 x 64-byte cache lines) */
} k3_cpu_runqueue_t;

typedef struct k3_smp_sched {
    uint32_t cpu_count;
    uint32_t active_tasks;
    k3_task_t tasks[K3_MAX_TASKS];
    k3_cpu_runqueue_t runqueues[K3_MAX_CPUS];
} k3_smp_sched_t;

/* ─────────────────── Capability Derivation Tree (CDT) Types ─────────────────── */

typedef struct k3_cap_handle {
    uint32_t slot_id;
    uint32_t generation;
    uint32_t compartment_id;
    uint32_t badge;
} k3_cap_handle_t;

typedef struct k3_cdt_node {
    uint32_t active;
    uint32_t generation;
    uint32_t compartment_id;
    uint32_t resource_id;
    uint32_t rights;
    uint32_t depth;
    uint32_t parent_slot;
    uint32_t first_child;
    uint32_t next_sibling;
    uint32_t secret_badge;
} k3_cdt_node_t;

typedef struct k3_cdt {
    uint32_t rev_epoch;
    uint32_t lock_word;
    uint32_t live_count;
    uint32_t badge_seed;
    k3_cdt_node_t nodes[K3_MAX_CAPS];
} k3_cdt_t;

/* ────────────── Session-Typed submit/complete Ring IPC & SHM Types ────────────── */

typedef struct k3_ipc_sqe {
    uint32_t valid;
    uint32_t msg_kind;
    uint32_t seq_id;
    uint32_t payload_len;
    uint32_t sender_compartment;
    uint32_t receiver_compartment;
    k3_cap_handle_t sender_cap;
    uint64_t zero_copy_tag;
    uint8_t payload[K3_IPC_MAX_PAYLOAD];
} k3_ipc_sqe_t;

typedef struct k3_ipc_channel {
    uint32_t channel_id;
    uint32_t client_compartment;
    uint32_t server_compartment;
    uint32_t lock_word;
    k3_session_state_t session_state;
    k3_session_state_t pending_state;
    uint32_t head;
    uint32_t tail;
    uint32_t count;
    uint32_t next_seq;
    uint64_t submitted_total;
    uint64_t completed_total;
    uint64_t revoked_aborts;
    k3_ipc_sqe_t ring[K3_IPC_RING_CAP];
} k3_ipc_channel_t;

typedef struct k3_shm_page {
    uint32_t page_id;
    uint32_t owner_compartment;
    uint32_t peer_compartment;
    uint32_t cap_transfer_forbidden;
    uint8_t data[K3_SHM_PAGE_BYTES];
} k3_shm_page_t;

/* Static layout assertions guaranteeing cache-line isolation across cores. */
#if defined(__cplusplus)
static_assert(alignof(k3_cpu_runqueue_t) >= 64, "k3_cpu_runqueue_t must be 64-byte aligned");
static_assert((sizeof(k3_cpu_runqueue_t) % 64) == 0, "k3_cpu_runqueue_t size must be a multiple of 64 bytes");
#else
_Static_assert(_Alignof(k3_cpu_runqueue_t) >= 64, "k3_cpu_runqueue_t must be 64-byte aligned");
_Static_assert((sizeof(k3_cpu_runqueue_t) % 64) == 0, "k3_cpu_runqueue_t size must be a multiple of 64 bytes");
#endif

/* ─────────────────────────── SMP Scheduler API ─────────────────────────── */

k3_status_t k3_smp_sched_init(k3_smp_sched_t* sched, uint32_t cpu_count);
k3_status_t k3_smp_task_spawn(k3_smp_sched_t* sched,
                              uint32_t task_id,
                              uint32_t compartment_id,
                              uint32_t cpu_id,
                              uint32_t priority);
k3_status_t k3_smp_enqueue(k3_smp_sched_t* sched, uint32_t cpu_id, uint32_t task_id);
k3_status_t k3_smp_schedule_next(k3_smp_sched_t* sched, uint32_t cpu_id, uint32_t* out_task_id);
k3_status_t k3_smp_preempt_tick(k3_smp_sched_t* sched, uint32_t cpu_id, uint32_t* out_next_task);
k3_status_t k3_smp_fault_task(k3_smp_sched_t* sched, uint32_t task_id, uint64_t fault_code);

/* ─────────────────── Capability Derivation Tree (CDT) API ─────────────────── */

void k3_cdt_init(k3_cdt_t* cdt);
k3_status_t k3_cdt_mint_root(k3_cdt_t* cdt,
                             uint32_t compartment_id,
                             uint32_t resource_id,
                             uint32_t rights,
                             k3_cap_handle_t* out_handle);
k3_status_t k3_cdt_derive(k3_cdt_t* cdt,
                          const k3_cap_handle_t* parent_handle,
                          uint32_t target_compartment_id,
                          uint32_t child_rights,
                          k3_cap_handle_t* out_child);
k3_status_t k3_cdt_reparent(k3_cdt_t* cdt,
                            const k3_cap_handle_t* node_handle,
                            const k3_cap_handle_t* new_parent_handle);
k3_status_t k3_cdt_revoke_subtree(k3_cdt_t* cdt,
                                  const k3_cap_handle_t* cap_handle,
                                  uint32_t* out_revoked_count);
k3_status_t k3_cdt_validate(const k3_cdt_t* cdt,
                            const k3_cap_handle_t* handle,
                            uint32_t caller_compartment,
                            uint32_t required_rights,
                            uint32_t* out_resource_id);

/* ────────────── Zero-Copy Session-Typed Ring IPC & SHM API ────────────── */

k3_status_t k3_ipc_channel_init(k3_ipc_channel_t* ch,
                                uint32_t channel_id,
                                uint32_t client_compartment,
                                uint32_t server_compartment);
k3_status_t k3_ipc_submit(k3_ipc_channel_t* ch,
                          const k3_cdt_t* cdt,
                          const k3_cap_handle_t* sender_cap,
                          uint32_t caller_compartment,
                          k3_ipc_msg_kind_t msg_kind,
                          const uint8_t* payload,
                          size_t payload_len,
                          uint64_t zero_copy_tag,
                          uint32_t* out_seq_id);
k3_status_t k3_ipc_complete(k3_ipc_channel_t* ch,
                            const k3_cdt_t* cdt,
                            const k3_cap_handle_t* receiver_cap,
                            uint32_t caller_compartment,
                            uint8_t* out_buf,
                            size_t out_buf_cap,
                            size_t* out_len,
                            k3_ipc_msg_kind_t* out_kind,
                            uint64_t* out_zero_copy_tag);

k3_status_t k3_shm_page_init(k3_shm_page_t* page,
                             uint32_t page_id,
                             uint32_t owner_compartment,
                             uint32_t peer_compartment);
k3_status_t k3_shm_write(k3_shm_page_t* page,
                         const k3_cdt_t* cdt,
                         const k3_cap_handle_t* cap,
                         uint32_t caller_compartment,
                         size_t offset,
                         const uint8_t* src,
                         size_t len);
k3_status_t k3_shm_read(const k3_shm_page_t* page,
                        const k3_cdt_t* cdt,
                        const k3_cap_handle_t* cap,
                        uint32_t caller_compartment,
                        size_t offset,
                        uint8_t* dst,
                        size_t len);
k3_status_t k3_shm_attempt_import_cap(const k3_shm_page_t* page,
                                      const k3_cdt_t* cdt,
                                      uint32_t caller_compartment,
                                      size_t offset,
                                      k3_cap_handle_t* out_cap);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_KERNEL_K3_SMP_CAP_IPC_H */
