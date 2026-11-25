/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol OS Phase U4: Native capability syscall ABI.
 *
 * Pure C11/C++17 freestanding ABI contract between Eshkol OS userspace ELF
 * processes (`--profile eshkol-os-user`) and the Eshkol OS microkernel.
 * Zero POSIX or libc dependencies.
 *
 * Architecture syscall register conventions (`eshkol_os_syscall6`):
 *
 *   x86_64 (`syscall`):
 *     sysno  : rax
 *     args   : rdi (arg0), rsi (arg1), rdx (arg2), r10 (arg3), r8 (arg4), r9 (arg5)
 *     clobber: rcx, r11, memory
 *     return : rax (signed status or non-negative value), rdx (optional secondary value)
 *
 *   AArch64 (`svc #0`):
 *     sysno  : x8
 *     args   : x0 (arg0), x1 (arg1), x2 (arg2), x3 (arg3), x4 (arg4), x5 (arg5)
 *     clobber: memory
 *     return : x0 (signed status or non-negative value), x1 (optional secondary value)
 *
 *   RISC-V 64 (`ecall`):
 *     sysno  : a7
 *     args   : a0 (arg0), a1 (arg1), a2 (arg2), a3 (arg3), a4 (arg4), a5 (arg5)
 *     clobber: memory
 *     return : a0 (signed status or non-negative value), a1 (optional secondary value)
 */

#ifndef ESHKOL_CORE_OS_SYSCALL_ABI_H
#define ESHKOL_CORE_OS_SYSCALL_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#  define ESHKOL_OS_ALIGNAS(n) _Alignas(n)
#elif defined(__cplusplus) && __cplusplus >= 201103L
#  define ESHKOL_OS_ALIGNAS(n) alignas(n)
#elif defined(__GNUC__) || defined(__clang__)
#  define ESHKOL_OS_ALIGNAS(n) __attribute__((aligned(n)))
#else
#  define ESHKOL_OS_ALIGNAS(n)
#endif

/**
 * @brief Opaque unforgeable capability handle in a userspace process CSpace.
 *
 * Zero denotes the null/invalid capability slot.
 */
typedef uint64_t eshkol_cap_id_t;

#define ESHKOL_CAP_ID_INVALID ((eshkol_cap_id_t)0ULL)

/**
 * @brief Native Eshkol OS syscall numbers.
 */
typedef enum eshkol_os_syscall_num {
    ESHKOL_SYS_CAP_DERIVE    = 1,
    ESHKOL_SYS_CAP_REVOKE    = 2,
    ESHKOL_SYS_CAP_DROP      = 3,

    ESHKOL_SYS_IPC_SUBMIT    = 10,
    ESHKOL_SYS_IPC_COMPLETE  = 11,
    ESHKOL_SYS_IPC_WAIT      = 12,

    ESHKOL_SYS_PAGE_MAP      = 20,
    ESHKOL_SYS_PAGE_UNMAP    = 21,

    ESHKOL_SYS_TASK_YIELD    = 30,
    ESHKOL_SYS_TASK_EXIT     = 31,

    ESHKOL_SYS_CONSOLE_WRITE = 40
} eshkol_os_syscall_num_t;

/**
 * @brief Signed syscall status codes (0 or positive on success, negative on rejection).
 */
typedef enum eshkol_os_syscall_status {
    ESHKOL_OS_OK                  = 0,
    ESHKOL_OS_ERR_INVALID_SYSCALL = -1,
    ESHKOL_OS_ERR_INVALID_CAP     = -2,
    ESHKOL_OS_ERR_RIGHTS_DENIED   = -3,
    ESHKOL_OS_ERR_REVOKED         = -4,
    ESHKOL_OS_ERR_INVALID_ARG     = -5,
    ESHKOL_OS_ERR_OOM             = -6,
    ESHKOL_OS_ERR_AGAIN           = -7,
    ESHKOL_OS_ERR_FAULT           = -8,
    ESHKOL_OS_ERR_QUEUE_FULL      = -9,
    ESHKOL_OS_ERR_QUEUE_EMPTY     = -10
} eshkol_os_syscall_status_t;

/**
 * @brief Capability rights bitmask constants.
 *
 * Derivation (`ESHKOL_SYS_CAP_DERIVE`) is strictly monotonic: derived rights
 * are always `(parent_rights & requested_rights)`.
 */
#define ESHKOL_CAP_RIGHT_NONE   ((uint64_t)0ULL)
#define ESHKOL_CAP_RIGHT_READ   ((uint64_t)(1ULL << 0))
#define ESHKOL_CAP_RIGHT_WRITE  ((uint64_t)(1ULL << 1))
#define ESHKOL_CAP_RIGHT_EXEC   ((uint64_t)(1ULL << 2))
#define ESHKOL_CAP_RIGHT_SEND   ((uint64_t)(1ULL << 3))
#define ESHKOL_CAP_RIGHT_RECV   ((uint64_t)(1ULL << 4))
#define ESHKOL_CAP_RIGHT_GRANT  ((uint64_t)(1ULL << 5))
#define ESHKOL_CAP_RIGHT_REVOKE ((uint64_t)(1ULL << 6))
#define ESHKOL_CAP_RIGHT_ALL    ((uint64_t)0x7FULL)

/**
 * @brief Memory mapping protection and zero-initialization flags for PAGE_MAP.
 */
#define ESHKOL_PAGE_FLAG_READ   ((uint64_t)(1ULL << 0))
#define ESHKOL_PAGE_FLAG_WRITE  ((uint64_t)(1ULL << 1))
#define ESHKOL_PAGE_FLAG_EXEC   ((uint64_t)(1ULL << 2))
#define ESHKOL_PAGE_FLAG_ZEROED ((uint64_t)(1ULL << 3))

/**
 * @brief Zero-copy IPC submission queue entry (64 bytes, 64-byte aligned).
 *
 * Fits cleanly in a single 64-byte cache line across x86_64, AArch64, and RV64
 * to eliminate false sharing across ring producers and consumers.
 */
typedef struct eshkol_ipc_sqe {
    ESHKOL_OS_ALIGNAS(64) uint32_t opcode;
    uint32_t flags;
    eshkol_cap_id_t target_cap;
    uint64_t user_data;
    uint64_t payload_addr;
    uint64_t payload_len;
    eshkol_cap_id_t transfer_cap;
    uint64_t arg0;
    uint64_t arg1;
} eshkol_ipc_sqe_t;

/**
 * @brief Zero-copy IPC completion queue entry (32 bytes, 32-byte aligned).
 */
typedef struct eshkol_ipc_cqe {
    ESHKOL_OS_ALIGNAS(32) uint64_t user_data;
    int64_t status;
    uint64_t result;
    uint32_t flags;
    uint32_t reserved;
} eshkol_ipc_cqe_t;

/**
 * @brief Structured two-register syscall return value.
 */
typedef struct eshkol_os_syscall_result {
    int64_t status;
    uint64_t value;
} eshkol_os_syscall_result_t;

/**
 * @brief Low-level 6-argument syscall trampoline supplied by the userspace
 *        architecture startup object (crt0 / arch syscall stub).
 */
int64_t eshkol_os_syscall6(uint64_t sysno,
                           uint64_t arg0,
                           uint64_t arg1,
                           uint64_t arg2,
                           uint64_t arg3,
                           uint64_t arg4,
                           uint64_t arg5);

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(eshkol_cap_id_t) == 8, "eshkol_cap_id_t must be 8 bytes");
_Static_assert(sizeof(eshkol_ipc_sqe_t) == 64, "eshkol_ipc_sqe_t must be 64 bytes");
_Static_assert(_Alignof(eshkol_ipc_sqe_t) == 64, "eshkol_ipc_sqe_t must be 64-byte aligned");
_Static_assert(sizeof(eshkol_ipc_cqe_t) == 32, "eshkol_ipc_cqe_t must be 32 bytes");
_Static_assert(_Alignof(eshkol_ipc_cqe_t) == 32, "eshkol_ipc_cqe_t must be 32-byte aligned");
_Static_assert(sizeof(eshkol_os_syscall_result_t) == 16, "eshkol_os_syscall_result_t must be 16 bytes");
#elif defined(__cplusplus) && __cplusplus >= 201103L
static_assert(sizeof(eshkol_cap_id_t) == 8, "eshkol_cap_id_t must be 8 bytes");
static_assert(sizeof(eshkol_ipc_sqe_t) == 64, "eshkol_ipc_sqe_t must be 64 bytes");
static_assert(alignof(eshkol_ipc_sqe_t) == 64, "eshkol_ipc_sqe_t must be 64-byte aligned");
static_assert(sizeof(eshkol_ipc_cqe_t) == 32, "eshkol_ipc_cqe_t must be 32 bytes");
static_assert(alignof(eshkol_ipc_cqe_t) == 32, "eshkol_ipc_cqe_t must be 32-byte aligned");
static_assert(sizeof(eshkol_os_syscall_result_t) == 16, "eshkol_os_syscall_result_t must be 16 bytes");
#endif

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_OS_SYSCALL_ABI_H */
