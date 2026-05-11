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

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque VM handle. Returned by eshkol_vm_load_chunk; freed by eshkol_vm_destroy. */
typedef struct EshkolVmHandle EshkolVmHandle;

/* Decode an ESKB chunk from an in-memory buffer and create a runnable VM.
 * Returns NULL on bad magic, version mismatch, CRC mismatch, allocation
 * failure, or NULL/zero-size input. */
EshkolVmHandle* eshkol_vm_load_chunk(const void* buffer, size_t size);

/* Run the loaded bytecode to completion. Returns 0 on success, -1 if the
 * VM raised an error or the handle is invalid. */
int eshkol_vm_run(EshkolVmHandle* h);

/* Destroy the VM handle and release the decoded ESKB module. NULL-safe. */
void eshkol_vm_destroy(EshkolVmHandle* h);

/* Read the top-of-stack value as int64 (also coerces float/bool). Returns 0
 * on success, -1 if the stack is empty or the top is not coercible. */
int eshkol_vm_top_int64(EshkolVmHandle* h, int64_t* out);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_BACKEND_VM_H */
