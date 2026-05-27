/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Runtime Eshkol-level frame trace for source-span stack traces in
 * runtime errors.
 *
 * Codegen pushes a frame at every Eshkol function entry; pops at exit.
 * eshkol_frame_print_trace walks the active stack and renders it
 * Python/Ruby/Julia-style:
 *
 *   Traceback (most recent call last):
 *     File "scripts/agent.esk", line 142, in (agent-handle-request)
 *     File "scripts/agent.esk", line 89, in (route-message)
 *     File "lib/core/runtime_string.cpp", in (string-append)
 *   Type error in <: expected number, got string
 *
 * Thread-safe via thread-local storage.  Capped at
 * ESHKOL_FRAME_STACK_MAX frames; overflow records a single "..."
 * marker so deep recursion doesn't drown the trace.
 */

#include <eshkol/core/runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

#define ESHKOL_FRAME_STACK_MAX 256

typedef struct {
    const char* function_name;  /* static — codegen emits a global string */
    const char* source_file;    /* static — codegen emits a global string */
    uint32_t source_line;
    uint32_t source_column;
} eshkol_frame_t;

#ifdef __clang__
#  define ESHKOL_TLS __thread
#elif defined(__GNUC__)
#  define ESHKOL_TLS __thread
#elif defined(_MSC_VER)
#  define ESHKOL_TLS __declspec(thread)
#else
#  define ESHKOL_TLS
#endif

static ESHKOL_TLS eshkol_frame_t g_frame_stack[ESHKOL_FRAME_STACK_MAX];
static ESHKOL_TLS uint32_t g_frame_stack_depth = 0;
static ESHKOL_TLS uint32_t g_frame_stack_overflowed = 0;

extern "C" {

void eshkol_frame_push(const char* function_name,
                       const char* source_file,
                       uint32_t source_line,
                       uint32_t source_column) {
    if (g_frame_stack_depth >= ESHKOL_FRAME_STACK_MAX) {
        g_frame_stack_overflowed = 1;
        return;
    }
    eshkol_frame_t* f = &g_frame_stack[g_frame_stack_depth++];
    f->function_name = function_name ? function_name : "<anonymous>";
    f->source_file = source_file;
    f->source_line = source_line;
    f->source_column = source_column;
}

void eshkol_frame_pop(void) {
    if (g_frame_stack_depth > 0) {
        g_frame_stack_depth--;
    } else if (g_frame_stack_overflowed) {
        /* If we're popping back from an overflow region, only clear the
         * marker once the depth has unwound below the cap. */
        g_frame_stack_overflowed = 0;
    }
}

uint32_t eshkol_frame_stack_depth(void) {
    return g_frame_stack_depth;
}

/*
 * Print the current Eshkol-level frame stack to `fp`.  Most-recent
 * call last (matches Python / Ruby / Julia conventions).
 *
 * Output format:
 *   Traceback (most recent call last):
 *     File "<file>", line <N>:<C>, in (<function-name>)
 *     ...
 * Followed by NO trailing newline; the caller's error message gets
 * printed by `eshkol_diagnostic_at` or similar immediately after.
 */
void eshkol_frame_print_trace(void* fp_void) {
    FILE* fp = (FILE*)fp_void;
    if (!fp) fp = stderr;
    if (g_frame_stack_depth == 0 && !g_frame_stack_overflowed) {
        return;
    }
    std::fputs("Traceback (most recent call last):\n", fp);
    if (g_frame_stack_overflowed) {
        std::fputs("  ... <stack truncated at ", fp);
        std::fprintf(fp, "%d frames>\n", ESHKOL_FRAME_STACK_MAX);
    }
    /* Frames are stored bottom-up (push index 0 first); most-recent-last
     * means we iterate forward and print as-is. */
    for (uint32_t i = 0; i < g_frame_stack_depth; i++) {
        const eshkol_frame_t* f = &g_frame_stack[i];
        const char* file = f->source_file ? f->source_file : "<unknown>";
        std::fprintf(fp, "  File \"%s\", line %u", file, f->source_line);
        if (f->source_column > 0) {
            std::fprintf(fp, ":%u", f->source_column);
        }
        std::fprintf(fp, ", in (%s)\n",
                     f->function_name ? f->function_name : "<anonymous>");
    }
}

/*
 * Convenience: print the trace to stderr.  Used by the exception
 * machinery before the actual error message lands.
 */
void eshkol_frame_print_trace_stderr(void) {
    eshkol_frame_print_trace(stderr);
}

/*
 * Clear the trace.  Called at REPL prompt boundaries so a new
 * expression doesn't inherit the stack of the previous one.
 */
void eshkol_frame_stack_reset(void) {
    g_frame_stack_depth = 0;
    g_frame_stack_overflowed = 0;
}

}  /* extern "C" */
