/**
 * @file vm.h
 * @brief Public C ABI for the Eshkol bytecode VM.
 *
 * Lets external callers (e.g. qLLM) load an in-memory ESKB chunk, dispatch
 * the bytecode through the interpreter, and tear the VM down without
 * touching VM/EskbModule internals.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#ifndef ESHKOL_BACKEND_VM_H
#define ESHKOL_BACKEND_VM_H

#include <stddef.h>
#include <stdint.h>

#include "eshkol/backend/vm_limits.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque VM handle. Returned by eshkol_vm_load_chunk; freed by eshkol_vm_destroy. */
typedef struct EshkolVmHandle EshkolVmHandle;

typedef struct EshkolVmProfileLimits {
    int heap_objects;
    int stack_slots;
    int max_frames;
    int max_constants;
    int max_instructions;
} EshkolVmProfileLimits;

/* Return the VM profile limits compiled into this runtime. */
int eshkol_vm_get_profile_limits(EshkolVmProfileLimits* out);

/* Decode an ESKB chunk from an in-memory buffer and create a runnable VM.
 * Returns NULL on bad magic, version mismatch, CRC mismatch, allocation
 * failure, profile-limit violation, or NULL/zero-size input. */
EshkolVmHandle* eshkol_vm_load_chunk(const void* buffer, size_t size);

/* Run the loaded bytecode to completion. Returns 0 on success, -1 if the
 * VM raised an error or the handle is invalid. */
int eshkol_vm_run(EshkolVmHandle* h);

/* Run a named function from the loaded ESKB code section. This resets the VM
 * instruction/stack/frame state before dispatch while preserving VM heap and
 * host resources owned by the handle. Returns 0 on success, -1 for a missing
 * entry point, invalid handle, or VM error. */
int eshkol_vm_call(EshkolVmHandle* h, const char* name);

/* Return 1 if the loaded ESKB chunk contains a function with `name`, 0 if it
 * does not, and -1 for invalid inputs. */
int eshkol_vm_has_function(EshkolVmHandle* h, const char* name);

/* Destroy the VM handle and release the decoded ESKB module. NULL-safe. */
void eshkol_vm_destroy(EshkolVmHandle* h);

/* Read the top-of-stack value as int64 (also coerces float/bool). Returns 0
 * on success, -1 if the stack is empty or the top is not coercible. */
int eshkol_vm_top_int64(EshkolVmHandle* h, int64_t* out);

#define ESHKOL_VM_HOST_NATIVE_BASE 100000

#define ESHKOL_VM_NATIVE_POLICY_DESKTOP 0
#define ESHKOL_VM_NATIVE_POLICY_HOST_ONLY 1

/* Control which native-call surface OP_NATIVE_CALL can reach for this VM.
 * DESKTOP preserves the existing broad native table. HOST_ONLY rejects all
 * desktop native fids and permits only ESHKOL_VM_HOST_NATIVE_BASE + slot. */
int eshkol_vm_set_native_policy(EshkolVmHandle* h, int policy);
int eshkol_vm_get_native_policy(EshkolVmHandle* h);

/* Host-callback registry. Function pointer signature receives the opaque VM;
 * the callback reads its arguments from the VM stack and pushes results back
 * through the eshkol_vm_host_* helpers below.
 * Returns the assigned slot index on success (>=0), -1 on failure
 * (table full / duplicate name / null fn). The integer fid that bytecode
 * should encode in OP_NATIVE_CALL is ESHKOL_VM_HOST_NATIVE_BASE + slot_index. */
typedef struct VM VM;
typedef int (*eshkol_vm_host_native_fn)(VM* vm);

typedef struct EshkolVmHostNative {
    const char* name;
    eshkol_vm_host_native_fn fn;
} EshkolVmHostNative;

/* Install a deterministic host-native table. Slots map directly to the input
 * array order, so bytecode fids are ESHKOL_VM_HOST_NATIVE_BASE + index.
 * Validation is all-or-nothing: invalid names, null callbacks, duplicates, or
 * over-capacity tables leave the existing registry unchanged. */
int eshkol_vm_install_host_natives(const EshkolVmHostNative* entries, int count);
void eshkol_vm_clear_host_natives(void);
int eshkol_vm_host_native_capacity(void);
int eshkol_vm_host_native_count(void);

int eshkol_vm_register_host_native(const char* name, eshkol_vm_host_native_fn fn);

/* Release a previously-registered slot so it can be reused by a later
 * register call. The slot index remains valid (as a tombstone) so any
 * other live registrations keep their indices stable. Returns 0 on
 * success, -1 if `slot` is out of range or already free. */
int eshkol_vm_unregister_host_native(int slot);

/* Helpers callable only from a registered host-native callback. */
int eshkol_vm_host_pop_int64(VM* vm, int64_t* out);
int eshkol_vm_host_push_int64(VM* vm, int64_t value);
int eshkol_vm_host_pop_double(VM* vm, double* out);
int eshkol_vm_host_push_double(VM* vm, double value);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_BACKEND_VM_H */
