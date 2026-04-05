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

int eshkol_emit_eskb(const char* source, const char* output_path) {
    (void)source;
    (void)output_path;
    fprintf(stderr, "ESKB emission is not available in the native Windows build.\n");
    return -1;
}
