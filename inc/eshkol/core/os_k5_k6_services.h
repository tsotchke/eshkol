/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol OS Phase K5 + K6: Capability Service Supervisor, Framebuffer/Input/
 * Real-Time Audio Compositor, and End-to-End Self-Hosting Reproducible Build
 * Convergence inside the K4 Atomic Content-Addressed VFS.
 *
 * Pure C11/C++17 freestanding userspace library operating strictly over
 * capability handles (`eshkol_cap_id_t`), `eshkol_os_syscall6`, U6 in-OS
 * self-host compiler, and K4 atomic SHA-256 CAS VFS with ZERO LLVM, ZERO
 * POSIX/libc, ZERO dynamic heap allocation, and ZERO undefined external
 * symbols.
 */

#ifndef ESHKOL_CORE_OS_K5_K6_SERVICES_H
#define ESHKOL_CORE_OS_K5_K6_SERVICES_H

#include <stddef.h>
#include <stdint.h>

#include <eshkol/core/os_drivers_vfs.h>
#include <eshkol/core/os_self_host.h>
#include <eshkol/core/os_syscall_abi.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Maximum number of capability-sandboxed services managed by one supervisor.
 */
#define ESHKOL_OS_SUP_MAX_SERVICES 16u

/**
 * @brief Maximum number of exclusive capability locks managed by one supervisor.
 */
#define ESHKOL_OS_SUP_MAX_LOCKS 32u

/**
 * @brief Maximum length of a supervised service name (excluding trailing NUL).
 */
#define ESHKOL_OS_SUP_MAX_NAME 31u

/**
 * @brief Maximum number of spatial framebuffer surfaces in one compositor.
 */
#define ESHKOL_OS_COMP_MAX_SURFACES 16u

/**
 * @brief Capacity of the bounded compositor input event queue.
 */
#define ESHKOL_OS_COMP_MAX_INPUT_EVENTS 64u

/**
 * @brief Frame capacity of the bounded real-time stereo audio ring buffer.
 */
#define ESHKOL_OS_AUDIO_RING_FRAMES 64u

/**
 * @brief Explicit status codes for Phase K5 supervisor, compositor, and K6 builder.
 */
typedef enum eshkol_os_k56_status {
    ESHKOL_K56_OK                    = 0,
    ESHKOL_K56_ERR_INVALID_ARG       = 1,
    ESHKOL_K56_ERR_NO_SLOTS          = 2,
    ESHKOL_K56_ERR_SYSCALL_FAILED    = 3,

    ESHKOL_SUP_ERR_STALE_TOKEN       = 10,
    ESHKOL_SUP_ERR_REVOKED_CAP       = 11,
    ESHKOL_SUP_ERR_SERVICE_FAULTED   = 12,
    ESHKOL_SUP_ERR_LOCK_HELD         = 13,
    ESHKOL_SUP_ERR_LOCK_NOT_OWNED    = 14,
    ESHKOL_SUP_ERR_DUAL_OWNER        = 15,

    ESHKOL_COMP_ERR_SURFACE_CONFLICT = 20,
    ESHKOL_COMP_ERR_RO_TRAP          = 21,
    ESHKOL_COMP_ERR_BOUNDS           = 22,
    ESHKOL_COMP_ERR_BAD_SURFACE      = 23,
    ESHKOL_COMP_ERR_QUEUE_EMPTY      = 24,
    ESHKOL_COMP_ERR_QUEUE_FULL       = 25,

    ESHKOL_K6_ERR_COMPILE_FAILED     = 30,
    ESHKOL_K6_ERR_VFS_FAILED         = 31,
    ESHKOL_K6_ERR_NONDETERMINISTIC   = 32,
    ESHKOL_K6_ERR_BAD_ELF            = 33,
    ESHKOL_K6_ERR_BUF_TOO_SMALL      = 34
} eshkol_os_k56_status_t;

/* ═══════════════════════════════════════════════════════════════════════════
 * Part 1: Phase K5 Capability Service Supervisor
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Deterministic fault kinds injected into capability-sandboxed services.
 */
typedef enum eshkol_os_svc_fault {
    ESHKOL_SVC_FAULT_NONE          = 0,
    ESHKOL_SVC_FAULT_SEGV          = 1,
    ESHKOL_SVC_FAULT_KILL          = 2,
    ESHKOL_SVC_FAULT_ARENA_CORRUPT = 3
} eshkol_os_svc_fault_t;

/**
 * @brief Runtime lifecycle state of a supervised service slot.
 */
typedef enum eshkol_os_svc_state {
    ESHKOL_SVC_STATE_FREE    = 0,
    ESHKOL_SVC_STATE_RUNNING = 1,
    ESHKOL_SVC_STATE_FAULTED = 2,
    ESHKOL_SVC_STATE_REVOKED = 3
} eshkol_os_svc_state_t;

/**
 * @brief Linear capability generation token uniquely identifying one service instance.
 */
typedef struct eshkol_os_svc_token {
    uint32_t service_id;
    uint32_t reserved;
    uint64_t generation;
    eshkol_cap_id_t instance_cap;
} eshkol_os_svc_token_t;

/**
 * @brief Internal record for one capability-sandboxed service under supervision.
 */
typedef struct eshkol_os_svc_record {
    char name[32];
    uint32_t state;
    uint32_t root_revoked;
    uint32_t last_fault;
    uint32_t restart_count;
    uint32_t fault_count;
    uint32_t reserved;
    uint64_t generation;
    eshkol_cap_id_t root_cap;
    eshkol_cap_id_t instance_cap;
    uint64_t rights;
} eshkol_os_svc_record_t;

/**
 * @brief Exclusive capability lock entry tracked by the supervisor for atomic reclaim.
 */
typedef struct eshkol_os_sup_lock {
    uint32_t held;
    uint32_t owner_service_id;
    uint64_t owner_generation;
    uint64_t acquire_count;
    uint64_t reclaim_count;
} eshkol_os_sup_lock_t;

/**
 * @brief Caller-owned capability service supervisor state (zero heap allocation).
 */
typedef struct eshkol_os_supervisor {
    eshkol_os_syscall6_fn_t syscall_fn;
    uint64_t next_generation;
    uint64_t total_faults;
    uint64_t total_restarts;
    uint64_t total_lock_reclaims;
    uint32_t service_count;
    uint32_t initialized;
    eshkol_os_svc_record_t services[ESHKOL_OS_SUP_MAX_SERVICES];
    eshkol_os_sup_lock_t locks[ESHKOL_OS_SUP_MAX_LOCKS];
} eshkol_os_supervisor_t;

/**
 * @brief Initialize a userspace capability service supervisor.
 */
eshkol_os_k56_status_t eshkol_os_supervisor_init(
    eshkol_os_supervisor_t* sup,
    eshkol_os_syscall6_fn_t syscall_fn);

/**
 * @brief Register a capability-sandboxed service and mint its initial instance token.
 */
eshkol_os_k56_status_t eshkol_os_supervisor_register(
    eshkol_os_supervisor_t* sup,
    const char* name,
    eshkol_cap_id_t root_cap,
    uint64_t rights,
    uint32_t* out_service_id,
    eshkol_os_svc_token_t* out_token);

/**
 * @brief Validate that a service instance token is active and not stale or revoked.
 */
eshkol_os_k56_status_t eshkol_os_supervisor_validate_token(
    const eshkol_os_supervisor_t* sup,
    const eshkol_os_svc_token_t* token);

/**
 * @brief Acquire an exclusive capability lock on behalf of an active service instance.
 */
eshkol_os_k56_status_t eshkol_os_supervisor_lock_acquire(
    eshkol_os_supervisor_t* sup,
    const eshkol_os_svc_token_t* token,
    uint32_t lock_id);

/**
 * @brief Release an exclusive capability lock owned by an active service instance.
 */
eshkol_os_k56_status_t eshkol_os_supervisor_lock_release(
    eshkol_os_supervisor_t* sup,
    const eshkol_os_svc_token_t* token,
    uint32_t lock_id);

/**
 * @brief Inject a deterministic fault into a service, revoking its instance token and
 *        atomically releasing every exclusive lock held by the faulted instance.
 */
eshkol_os_k56_status_t eshkol_os_supervisor_inject_fault(
    eshkol_os_supervisor_t* sup,
    uint32_t service_id,
    eshkol_os_svc_fault_t fault);

/**
 * @brief Restart a faulted service with a strictly fresh capability generation token.
 *
 * Fails closed with `ESHKOL_SUP_ERR_REVOKED_CAP` if the service's root capability
 * has been revoked.
 */
eshkol_os_k56_status_t eshkol_os_supervisor_restart(
    eshkol_os_supervisor_t* sup,
    uint32_t service_id,
    eshkol_os_svc_token_t* out_new_token);

/**
 * @brief Revoke a root capability, terminating and blocking restarts for bound services.
 */
eshkol_os_k56_status_t eshkol_os_supervisor_revoke_root_cap(
    eshkol_os_supervisor_t* sup,
    eshkol_cap_id_t root_cap);

/* ═══════════════════════════════════════════════════════════════════════════
 * Part 2: Phase K5 Framebuffer, Input & Real-Time Audio Compositor
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Linear capability handle identifying one compositor framebuffer surface.
 */
typedef struct eshkol_os_surface_handle {
    uint32_t surface_id;
    uint32_t owner_badge;
    uint64_t generation;
    eshkol_cap_id_t cap_id;
} eshkol_os_surface_handle_t;

/**
 * @brief Internal spatial surface descriptor managed by the compositor.
 */
typedef struct eshkol_os_surface_desc {
    uint32_t active;
    uint32_t owner_badge;
    uint32_t x;
    uint32_t y;
    uint32_t width;
    uint32_t height;
    uint64_t generation;
    eshkol_cap_id_t cap_id;
    uint64_t rights;
} eshkol_os_surface_desc_t;

/**
 * @brief Input event routed by the compositor to a capability-badged surface owner.
 */
typedef struct eshkol_os_input_event {
    uint32_t target_badge;
    uint32_t event_type;
    uint32_t code;
    int32_t value;
    uint64_t sequence;
} eshkol_os_input_event_t;

/**
 * @brief Stereo 16-bit PCM audio frame for the bounded real-time audio ring.
 */
typedef struct eshkol_os_audio_frame {
    int16_t left;
    int16_t right;
} eshkol_os_audio_frame_t;

/**
 * @brief Bounded real-time audio ring with explicit overrun and underrun telemetry.
 */
typedef struct eshkol_os_audio_ring {
    eshkol_os_audio_frame_t frames[ESHKOL_OS_AUDIO_RING_FRAMES];
    uint32_t read_idx;
    uint32_t write_idx;
    uint32_t count;
    uint32_t reserved;
    uint64_t frames_written;
    uint64_t frames_read;
    uint64_t dropped_frames;
    uint64_t overrun_events;
    uint64_t underrun_events;
    uint64_t silence_filled_frames;
} eshkol_os_audio_ring_t;

/**
 * @brief Caller-owned capability framebuffer, input, and audio compositor state.
 */
typedef struct eshkol_os_compositor {
    uint32_t* framebuffer;
    uint32_t fb_width;
    uint32_t fb_height;
    uint32_t initialized;
    uint64_t next_generation;
    uint64_t overlap_rejects;
    uint64_t ro_traps;
    uint64_t writes_committed;
    eshkol_os_surface_desc_t surfaces[ESHKOL_OS_COMP_MAX_SURFACES];
    eshkol_os_input_event_t input_ring[ESHKOL_OS_COMP_MAX_INPUT_EVENTS];
    uint32_t input_head;
    uint32_t input_tail;
    uint32_t input_count;
    uint32_t reserved;
    uint64_t input_seq;
    eshkol_os_audio_ring_t audio;
} eshkol_os_compositor_t;

/**
 * @brief Initialize the capability framebuffer, input, and real-time audio compositor.
 */
eshkol_os_k56_status_t eshkol_os_compositor_init(
    eshkol_os_compositor_t* comp,
    uint32_t* framebuffer,
    uint32_t width,
    uint32_t height);

/**
 * @brief Allocate a capability-gated spatial surface on the compositor framebuffer.
 *
 * Rejects overlapping writable allocations from distinct owners with
 * `ESHKOL_COMP_ERR_SURFACE_CONFLICT`.
 */
eshkol_os_k56_status_t eshkol_os_compositor_create_surface(
    eshkol_os_compositor_t* comp,
    uint32_t owner_badge,
    eshkol_cap_id_t cap_id,
    uint64_t rights,
    uint32_t x,
    uint32_t y,
    uint32_t width,
    uint32_t height,
    eshkol_os_surface_handle_t* out_handle);

/**
 * @brief Arbitrated framebuffer pixel write in absolute framebuffer coordinates.
 *
 * Fails closed with `ESHKOL_COMP_ERR_RO_TRAP` if the surface lacks
 * `ESHKOL_CAP_RIGHT_WRITE`, and fails closed with `ESHKOL_COMP_ERR_SURFACE_CONFLICT`
 * if the target rectangle overlaps another owner's surface or extends outside
 * the caller's authorized surface region.
 */
eshkol_os_k56_status_t eshkol_os_compositor_write_rect(
    eshkol_os_compositor_t* comp,
    const eshkol_os_surface_handle_t* handle,
    uint32_t abs_x,
    uint32_t abs_y,
    uint32_t width,
    uint32_t height,
    const uint32_t* pixels);

/**
 * @brief Read a pixel value from a capability-authorized surface (read-only permitted).
 */
eshkol_os_k56_status_t eshkol_os_compositor_read_pixel(
    const eshkol_os_compositor_t* comp,
    const eshkol_os_surface_handle_t* handle,
    uint32_t abs_x,
    uint32_t abs_y,
    uint32_t* out_pixel);

/**
 * @brief Enqueue an input event routed to a target service badge.
 */
eshkol_os_k56_status_t eshkol_os_compositor_push_input(
    eshkol_os_compositor_t* comp,
    uint32_t target_badge,
    uint32_t event_type,
    uint32_t code,
    int32_t value);

/**
 * @brief Dequeue the next input event matching `owner_badge`.
 */
eshkol_os_k56_status_t eshkol_os_compositor_poll_input(
    eshkol_os_compositor_t* comp,
    uint32_t owner_badge,
    eshkol_os_input_event_t* out_event);

/**
 * @brief Push stereo audio frames into the real-time audio ring with graceful overrun drops.
 */
eshkol_os_k56_status_t eshkol_os_compositor_audio_write(
    eshkol_os_compositor_t* comp,
    const eshkol_os_audio_frame_t* frames,
    uint32_t frame_count,
    uint32_t* out_accepted,
    uint32_t* out_dropped);

/**
 * @brief Drain stereo audio frames from the real-time audio ring with graceful silence fill.
 */
eshkol_os_k56_status_t eshkol_os_compositor_audio_read(
    eshkol_os_compositor_t* comp,
    eshkol_os_audio_frame_t* out_frames,
    uint32_t requested_frames,
    uint32_t* out_from_ring,
    uint32_t* out_silence_filled);

/* ═══════════════════════════════════════════════════════════════════════════
 * Part 3: Phase K6 Self-Hosting Reproducible Build Convergence in CAS VFS
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Target artifact profile produced by the in-OS K6 self-hosting builder.
 */
typedef enum eshkol_os_k6_artifact_kind {
    ESHKOL_K6_ARTIFACT_USER_ESKB    = 0,
    ESHKOL_K6_ARTIFACT_USER_ELF64   = 1,
    ESHKOL_K6_ARTIFACT_KERNEL_ELF64 = 2
} eshkol_os_k6_artifact_kind_t;

/**
 * @brief Function table binding U6 in-OS compiler and K4 CAS VFS operations.
 */
typedef struct eshkol_os_k6_builder_ops {
    eshkol_os_drv_status_t (*vfs_read_file)(
        const eshkol_os_cas_vfs_t* vfs,
        const char* name,
        uint32_t parent_id,
        void* out_buf,
        size_t out_cap,
        size_t* out_len,
        uint8_t out_sha256[ESHKOL_OS_SHA256_BYTES]);
    eshkol_os_drv_status_t (*vfs_put_file)(
        eshkol_os_cas_vfs_t* vfs,
        const char* name,
        uint32_t parent_id,
        const void* data,
        size_t len,
        uint8_t out_sha256[ESHKOL_OS_SHA256_BYTES]);
    void (*sha256)(
        const void* data,
        size_t len,
        uint8_t out_digest[ESHKOL_OS_SHA256_BYTES]);
    eshkol_os_self_host_status_t (*compile_to_eskb)(
        const char* source,
        size_t source_len,
        void* workspace,
        size_t workspace_bytes,
        uint8_t* out_buf,
        size_t out_cap,
        eshkol_os_compile_result_t* out_result);
    eshkol_os_self_host_status_t (*compile_to_elf64)(
        const char* source,
        size_t source_len,
        eshkol_os_target_isa_t target,
        void* workspace,
        size_t workspace_bytes,
        uint8_t* out_buf,
        size_t out_cap,
        eshkol_os_compile_result_t* out_result);
} eshkol_os_k6_builder_ops_t;

/**
 * @brief Build request for compiling an Eshkol source file stored in K4 CAS VFS.
 */
typedef struct eshkol_os_k6_build_request {
    const char* source_name;
    uint32_t source_parent_id;
    const char* output_name;
    uint32_t output_parent_id;
    eshkol_os_k6_artifact_kind_t artifact_kind;
    eshkol_os_target_isa_t target_isa;
    uint8_t workspace_poison;
    uint8_t pad[3];
    uint64_t nondet_salt; /* 0 = deterministic canonical build; nonzero = injected metadata salt */
} eshkol_os_k6_build_request_t;

/**
 * @brief Build output and cryptographic reproducibility receipt.
 */
typedef struct eshkol_os_k6_build_receipt {
    eshkol_os_k56_status_t status;
    uint32_t reserved;
    size_t source_bytes;
    size_t artifact_bytes;
    uint8_t source_sha256[ESHKOL_OS_SHA256_BYTES];
    uint8_t artifact_sha256[ESHKOL_OS_SHA256_BYTES];
    uint8_t pass2_sha256[ESHKOL_OS_SHA256_BYTES];
} eshkol_os_k6_build_receipt_t;

/**
 * @brief Caller-supplied scratch buffers for zero-malloc K6 self-hosting builds.
 */
typedef struct eshkol_os_k6_build_buffers {
    void* compiler_workspace;
    size_t compiler_workspace_bytes;
    uint8_t* source_buf;
    size_t source_cap;
    uint8_t* out_buf_a;
    size_t out_cap_a;
    uint8_t* out_buf_b;
    size_t out_cap_b;
} eshkol_os_k6_build_buffers_t;

/**
 * @brief Compile a source file from K4 CAS VFS and commit the verified binary back to CAS VFS.
 */
eshkol_os_k56_status_t eshkol_os_k6_build_selfhost(
    eshkol_os_cas_vfs_t* vfs,
    const eshkol_os_k6_builder_ops_t* ops,
    const eshkol_os_k6_build_request_t* req,
    eshkol_os_k6_build_buffers_t* bufs,
    eshkol_os_k6_build_receipt_t* out_receipt);

/**
 * @brief Run a two-pass perturbed reproducibility build inside CAS VFS and fail closed
 *        with `ESHKOL_K6_ERR_NONDETERMINISTIC` on any SHA-256 or byte divergence.
 */
eshkol_os_k56_status_t eshkol_os_k6_verify_reproducible(
    eshkol_os_cas_vfs_t* vfs,
    const eshkol_os_k6_builder_ops_t* ops,
    const eshkol_os_k6_build_request_t* req,
    uint8_t poison_pass_a,
    uint8_t poison_pass_b,
    uint64_t salt_pass_a,
    uint64_t salt_pass_b,
    eshkol_os_k6_build_buffers_t* bufs,
    eshkol_os_k6_build_receipt_t* out_receipt);

/**
 * @brief Verify strict standalone W^X ELF64 header and segment invariants on a raw image.
 */
eshkol_os_k56_status_t eshkol_os_k6_validate_wx_elf64(
    const uint8_t* image,
    size_t image_len,
    eshkol_os_target_isa_t expected_isa);

/**
 * @brief Return a static human-readable name for a Phase K5/K6 status code.
 */
const char* eshkol_os_k56_status_name(eshkol_os_k56_status_t status);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_OS_K5_K6_SERVICES_H */
