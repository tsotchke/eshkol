/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Native Windows builds use the LLVM backend / REPL path only for now.
 * Keep the CLI linkable by stubbing out ESKB emission until the bytecode VM
 * is ported separately.
 */

#include <stdio.h>

#include "eshkol/backend/vm.h"

int eshkol_vm_get_profile_limits(EshkolVmProfileLimits* out) {
    if (!out) return -1;
    out->heap_objects = ESHKOL_VM_HEAP_SIZE;
    out->stack_slots = ESHKOL_VM_STACK_SIZE;
    out->max_frames = ESHKOL_VM_MAX_FRAMES;
    out->max_constants = ESHKOL_VM_MAX_CONSTS;
    out->max_instructions = ESHKOL_VM_MAX_CODE;
    return 0;
}

int eshkol_vm_default_load_options(EshkolVmLoadOptions* out) {
    if (!out) return -1;
    out->native_policy = ESHKOL_VM_NATIVE_POLICY_DESKTOP;
    out->reject_string_constants = 0;
    out->reject_desktop_native_calls = 0;
    out->required_functions = NULL;
    out->required_function_count = 0;
    out->required_function_metadata = NULL;
    out->required_function_metadata_count = 0;
    return 0;
}

int eshkol_emit_eskb(const char* source, const char* output_path) {
    (void)source;
    (void)output_path;
    fprintf(stderr, "ESKB emission is not available in the native Windows build.\n");
    return -1;
}

int eshkol_emit_eskb_embedded(const char* source, const char* output_path) {
    (void)source;
    (void)output_path;
    fprintf(stderr, "Embedded ESKB emission is not available in the native Windows build.\n");
    return -1;
}

EshkolVmHandle* eshkol_vm_load_chunk(const void* buffer, size_t size) {
    (void)buffer;
    (void)size;
    return NULL;
}

EshkolVmHandle* eshkol_vm_load_chunk_with_options(
    const void* buffer, size_t size, const EshkolVmLoadOptions* options) {
    (void)buffer;
    (void)size;
    (void)options;
    return NULL;
}

int eshkol_vm_run(EshkolVmHandle* h) {
    (void)h;
    return -1;
}

int eshkol_vm_call(EshkolVmHandle* h, const char* name) {
    (void)h;
    (void)name;
    return -1;
}

int eshkol_vm_has_function(EshkolVmHandle* h, const char* name) {
    (void)h;
    (void)name;
    return -1;
}

int eshkol_vm_function_count(EshkolVmHandle* h) {
    (void)h;
    return -1;
}

const char* eshkol_vm_function_name(EshkolVmHandle* h, int index) {
    (void)h;
    (void)index;
    return NULL;
}

int eshkol_vm_function_info(EshkolVmHandle* h, int index,
                            EshkolVmFunctionInfo* out) {
    (void)h;
    (void)index;
    (void)out;
    return -1;
}

void eshkol_vm_destroy(EshkolVmHandle* h) {
    (void)h;
}

int eshkol_vm_top_int64(EshkolVmHandle* h, int64_t* out) {
    (void)h;
    (void)out;
    return -1;
}
