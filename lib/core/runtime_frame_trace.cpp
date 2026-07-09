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

/**
 * @brief Push a frame onto the current thread's Eshkol-level call stack.
 *
 * Codegen emits a call to this at every Eshkol function entry so
 * eshkol_frame_print_trace can later reconstruct a source-level backtrace.
 * `function_name` and `source_file` are expected to be static/global string
 * literals emitted by codegen, so no copy is taken. Once the stack has grown
 * past ESHKOL_FRAME_STACK_MAX entries, further pushes are dropped and the
 * thread-local overflow flag is set instead of writing out of bounds.
 *
 * @param function_name  Name of the function being entered (falls back to "<anonymous>" if null).
 * @param source_file    Source file of the call site (may be null).
 * @param source_line    Source line of the call site.
 * @param source_column  Source column of the call site (0 if unknown/unused).
 */
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

/**
 * @brief Pop the most recently pushed frame from the current thread's call stack.
 *
 * Codegen emits a call to this at every Eshkol function exit, mirroring
 * eshkol_frame_push. If the stack is already empty but the overflow marker is
 * set (meaning pushes beyond the cap were dropped), clears that marker instead
 * of underflowing the depth counter.
 */
void eshkol_frame_pop(void) {
    if (g_frame_stack_depth > 0) {
        g_frame_stack_depth--;
    } else if (g_frame_stack_overflowed) {
        /* If we're popping back from an overflow region, only clear the
         * marker once the depth has unwound below the cap. */
        g_frame_stack_overflowed = 0;
    }
}

/** @brief Return the current thread's Eshkol call-stack depth (number of active frames). */
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
