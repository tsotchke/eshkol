/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol OS Phase U5: Zero-LLVM, zero-POSIX in-OS ESKB bytecode VM shell ABI.
 *
 * Pure C11/C++17 freestanding runtime for executing compiled ESKB v2 bytecode
 * inside an Eshkol OS userspace shell process without LLVM, libc, POSIX, or
 * access to the kernel allocator.
 *
 * Design invariants:
 *   1. Zero dynamic host allocation: all VM state, decoded ESKB tables, and
 *      ephemeral OALR objects live strictly inside a caller-bounded workspace
 *      (optionally page-mapped and unmapped via ESHKOL_SYS_PAGE_MAP /
 *      ESHKOL_SYS_PAGE_UNMAP).
 *   2. Capability-gated host calls: desktop-native OP_NATIVE_CALL IDs
 *      (< ESHKOL_OS_VM_HOST_NATIVE_BASE) are rejected fail-closed at load and
 *      dispatch time. Every host call routes exclusively through the shell's
 *      CSpace capability table and `eshkol_os_syscall6`.
 *   3. Deterministic fuel and memory bounds: every bytecode instruction
 *      consumes one unit of fuel, and every OALR allocation enforces the
 *      configured byte ceiling. Exhaustion terminates cleanly with
 *      `ESHKOL_VM_FUEL_EXHAUSTED` or `ESHKOL_VM_ERR_OOM`, rewinds the
 *      ephemeral arena to zero live bytes, and leaves zero leaked pages.
 */

#ifndef ESHKOL_CORE_OS_VM_SHELL_H
#define ESHKOL_CORE_OS_VM_SHELL_H

#include <stddef.h>
#include <stdint.h>

#include <eshkol/core/os_syscall_abi.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Base native-call ID for OS capability host calls (matches VM ABI).
 *
 * Any OP_NATIVE_CALL with operand below this base is a desktop-native call
 * and is rejected fail-closed by the in-OS VM shell.
 */
#define ESHKOL_OS_VM_HOST_NATIVE_BASE 100000

/**
 * @brief Capability-gated host-call IDs backed exclusively by `eshkol_os_syscall6`.
 */
typedef enum eshkol_os_vm_host_call {
    ESHKOL_OS_VM_CALL_CAP_DERIVE    = ESHKOL_OS_VM_HOST_NATIVE_BASE + 0,
    ESHKOL_OS_VM_CALL_CAP_REVOKE    = ESHKOL_OS_VM_HOST_NATIVE_BASE + 1,
    ESHKOL_OS_VM_CALL_CAP_DROP      = ESHKOL_OS_VM_HOST_NATIVE_BASE + 2,
    ESHKOL_OS_VM_CALL_IPC_SUBMIT    = ESHKOL_OS_VM_HOST_NATIVE_BASE + 3,
    ESHKOL_OS_VM_CALL_IPC_COMPLETE  = ESHKOL_OS_VM_HOST_NATIVE_BASE + 4,
    ESHKOL_OS_VM_CALL_IPC_WAIT      = ESHKOL_OS_VM_HOST_NATIVE_BASE + 5,
    ESHKOL_OS_VM_CALL_PAGE_MAP      = ESHKOL_OS_VM_HOST_NATIVE_BASE + 6,
    ESHKOL_OS_VM_CALL_PAGE_UNMAP    = ESHKOL_OS_VM_HOST_NATIVE_BASE + 7,
    ESHKOL_OS_VM_CALL_TASK_YIELD    = ESHKOL_OS_VM_HOST_NATIVE_BASE + 8,
    ESHKOL_OS_VM_CALL_TASK_EXIT     = ESHKOL_OS_VM_HOST_NATIVE_BASE + 9,
    ESHKOL_OS_VM_CALL_CONSOLE_WRITE = ESHKOL_OS_VM_HOST_NATIVE_BASE + 10
} eshkol_os_vm_host_call_t;

/**
 * @brief Explicit status codes returned by the in-OS VM shell.
 */
typedef enum eshkol_os_vm_status {
    ESHKOL_VM_OK                   = 0,
    ESHKOL_VM_FUEL_EXHAUSTED       = 1,
    ESHKOL_VM_ERR_INVALID_ARG      = 2,
    ESHKOL_VM_ERR_BAD_ESKB         = 3,
    ESHKOL_VM_ERR_CRC_MISMATCH     = 4,
    ESHKOL_VM_ERR_FORBIDDEN_NATIVE = 5,
    ESHKOL_VM_ERR_CAP_DENIED       = 6,
    ESHKOL_VM_ERR_OOM              = 7,
    ESHKOL_VM_ERR_STACK_OVERFLOW   = 8,
    ESHKOL_VM_ERR_STACK_UNDERFLOW  = 9,
    ESHKOL_VM_ERR_FRAME_OVERFLOW   = 10,
    ESHKOL_VM_ERR_TYPE             = 11,
    ESHKOL_VM_ERR_DIV_ZERO         = 12,
    ESHKOL_VM_ERR_BOUNDS           = 13,
    ESHKOL_VM_ERR_BAD_OPCODE       = 14,
    ESHKOL_VM_ERR_MISSING_ENTRY    = 15,
    ESHKOL_VM_ERR_SYSCALL_FAILED   = 16
} eshkol_os_vm_status_t;

/**
 * @brief Value kind reported in `eshkol_os_vm_result_t`.
 */
typedef enum eshkol_os_vm_val_kind {
    ESHKOL_OS_VM_VAL_NIL     = 0,
    ESHKOL_OS_VM_VAL_INT     = 1,
    ESHKOL_OS_VM_VAL_FLOAT   = 2,
    ESHKOL_OS_VM_VAL_BOOL    = 3,
    ESHKOL_OS_VM_VAL_PAIR    = 4,
    ESHKOL_OS_VM_VAL_CLOSURE = 5,
    ESHKOL_OS_VM_VAL_STRING  = 6,
    ESHKOL_OS_VM_VAL_VECTOR  = 7,
    ESHKOL_OS_VM_VAL_VOID    = 8
} eshkol_os_vm_val_kind_t;

/**
 * @brief Maximum number of capability slots held in one shell CSpace table.
 */
#define ESHKOL_OS_VM_MAX_CAPS 64u

/**
 * @brief Default instruction fuel limit when config.fuel_limit == 0.
 */
#define ESHKOL_OS_VM_DEFAULT_FUEL 100000ULL

/**
 * @brief Capability grant entry in the shell's local CSpace table.
 */
typedef struct eshkol_os_vm_cap_grant {
    eshkol_cap_id_t cap_id;
    uint64_t rights;
    uint32_t active;
    uint32_t reserved;
} eshkol_os_vm_cap_grant_t;

/**
 * @brief Configuration for initializing an in-OS VM shell instance.
 */
typedef struct eshkol_os_vm_config {
    void* workspace;            /* Caller-supplied backing buffer (zero malloc) */
    size_t workspace_bytes;     /* Total byte capacity of workspace */
    eshkol_cap_id_t space_cap;  /* Optional address-space cap for PAGE_MAP/UNMAP */
    uint64_t page_size;         /* OS page size in bytes (0 -> 4096) */
    eshkol_cap_id_t console_cap;/* Optional console cap for OP_PRINT (0 = denied) */
    uint64_t console_rights;    /* Rights held on console_cap */
    uint64_t fuel_limit;        /* Max bytecode instructions per eval (0 -> default) */
    uint32_t max_heap_bytes;    /* Max ephemeral OALR heap bytes (0 -> workspace remainder) */
    uint32_t max_stack_slots;   /* Max operand stack slots (0 -> 1024, cap 1024) */
    uint32_t max_frames;        /* Max call frames (0 -> 128, cap 128) */
} eshkol_os_vm_config_t;

/**
 * @brief Evaluation result and deterministic resource telemetry.
 */
typedef struct eshkol_os_vm_result {
    eshkol_os_vm_status_t status;
    uint32_t value_kind;
    int64_t int_value;
    double float_value;
    uint32_t fault_pc;
    uint32_t reserved;
    uint64_t instructions_executed;
    uint64_t heap_bytes_peak;
    uint64_t heap_bytes_live;
    uint64_t syscalls_dispatched;
    uint64_t cap_denials;
    uint64_t mapped_pages_live;
} eshkol_os_vm_result_t;

/**
 * @brief In-OS VM shell state (caller-owned struct, zero heap allocation).
 */
typedef struct eshkol_os_vm_shell {
    uint8_t* workspace;
    size_t workspace_bytes;
    eshkol_cap_id_t space_cap;
    uint64_t page_size;
    uint64_t workspace_mapped_pages;
    uint64_t user_mapped_pages;
    eshkol_cap_id_t console_cap;
    uint64_t fuel_limit;
    uint32_t max_heap_bytes;
    uint32_t max_stack_slots;
    uint32_t max_frames;
    uint32_t initialized;
    eshkol_os_vm_cap_grant_t caps[ESHKOL_OS_VM_MAX_CAPS];
    uint64_t total_evals;
    uint64_t total_syscalls;
    uint64_t total_cap_denials;
} eshkol_os_vm_shell_t;

/**
 * @brief Initialize an in-OS VM shell over a bounded workspace.
 *
 * When `config->space_cap != 0`, registers the workspace pages through
 * `ESHKOL_SYS_PAGE_MAP` via `eshkol_os_syscall6`.
 */
eshkol_os_vm_status_t eshkol_os_vm_init(eshkol_os_vm_shell_t* shell,
                                        const eshkol_os_vm_config_t* config);

/**
 * @brief Grant or update a capability handle and rights bitmask in the shell CSpace.
 */
eshkol_os_vm_status_t eshkol_os_vm_grant_cap(eshkol_os_vm_shell_t* shell,
                                             eshkol_cap_id_t cap_id,
                                             uint64_t rights);

/**
 * @brief Revoke a capability handle in the shell CSpace so subsequent VM calls fail closed.
 */
eshkol_os_vm_status_t eshkol_os_vm_revoke_cap(eshkol_os_vm_shell_t* shell,
                                              eshkol_cap_id_t cap_id);

/**
 * @brief Query whether `cap_id` is active in the shell CSpace with all `required_rights`.
 */
int eshkol_os_vm_has_cap(const eshkol_os_vm_shell_t* shell,
                         eshkol_cap_id_t cap_id,
                         uint64_t required_rights);

/**
 * @brief Update the per-evaluation instruction fuel limit.
 */
eshkol_os_vm_status_t eshkol_os_vm_set_fuel_limit(eshkol_os_vm_shell_t* shell,
                                                  uint64_t fuel_limit);

/**
 * @brief Decode, validate, and execute an in-memory ESKB v2 module inside the shell.
 *
 * If `entry_name` is NULL, executes function 0 (`main`); otherwise executes
 * the named zero-argument function. Always rewinds the ephemeral OALR arena
 * to zero live bytes before returning, regardless of success, fuel exhaustion,
 * capability denial, or OOM.
 */
eshkol_os_vm_status_t eshkol_os_vm_eval_eskb(eshkol_os_vm_shell_t* shell,
                                             const void* eskb_data,
                                             size_t eskb_size,
                                             const char* entry_name,
                                             eshkol_os_vm_result_t* out_result);

/**
 * @brief Tear down the shell, unmapping any workspace pages mapped via `space_cap`.
 */
eshkol_os_vm_status_t eshkol_os_vm_destroy(eshkol_os_vm_shell_t* shell);

/**
 * @brief Return a static human-readable name for a shell status code.
 */
const char* eshkol_os_vm_status_name(eshkol_os_vm_status_t status);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_OS_VM_SHELL_H */
