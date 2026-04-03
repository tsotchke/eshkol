/**
 * @file vm_wasm_repl.c
 * @brief WASM entry point for the Eshkol browser REPL.
 *
 * Wraps eshkol_vm.c to provide a C API callable from JavaScript via Emscripten.
 * Uses Emscripten's stdout capture for collecting output.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#define ESHKOL_VM_LIBRARY_MODE 1  /* Suppress eshkol_vm.c's main() */

#include "eshkol_vm.c"

#include <emscripten/emscripten.h>

/* Output capture buffer */
#define OUTPUT_BUF_SIZE 65536
static char output_buf[OUTPUT_BUF_SIZE];
static int output_pos = 0;
static int initialized = 0;

/* Initialize the VM (call once on page load) */
EMSCRIPTEN_KEEPALIVE
void repl_init(void) {
    output_pos = 0;
    output_buf[0] = '\0';
    initialized = 1;
}

/* Evaluate an Eshkol expression and return the output as a string. */
EMSCRIPTEN_KEEPALIVE
const char* repl_eval(const char* source) {
    if (!initialized) repl_init();

    output_pos = 0;
    output_buf[0] = '\0';

    /* compile_and_run uses printf for output.
     * Emscripten routes printf to stdout which goes to console.
     * We'll call compile_and_run directly — output goes to JS console,
     * and we return a status message. */
    compile_and_run(source);

    /* If compile_and_run produced output via printf, it went to Emscripten stdout.
     * We return empty string since output is captured by Emscripten's print handler. */
    return "";
}
