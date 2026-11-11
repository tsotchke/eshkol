/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Reader limits: one input-size cap and one per-pass traversal-depth budget,
 * shared by every reader and every entry point.  See
 * inc/eshkol/core/reader_limits.h for the contract and the configuration.
 *
 * This lives in the core runtime so that the native front end, the bytecode
 * VM's reader, the standalone VM, the REPL, the server and the WebAssembly
 * build all resolve the same number from the same place.
 */

#include "eshkol/core/reader_limits.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Resolution state.  `resolved` flips once, on first read, so a reader may
 * consult the cap from any point without an initialisation order rule; a
 * setter (the command-line option) writes the value and marks it resolved so
 * the environment is not consulted afterwards. */
static size_t g_max_source_bytes = ESHKOL_READER_DEFAULT_MAX_SOURCE_BYTES;
static int    g_max_source_bytes_resolved = 0;
static size_t g_max_pass_depth = ESHKOL_READER_DEFAULT_MAX_PASS_DEPTH;
static int    g_max_pass_depth_resolved = 0;

int eshkol_reader_parse_size(const char* text, size_t* out) {
    if (!text || !out) return 0;
    while (*text == ' ' || *text == '\t') text++;
    if (*text == '\0') return 0;
    char* end = NULL;
    unsigned long long raw = strtoull(text, &end, 10);
    if (end == text) return 0;
    unsigned long long scale = 1;
    if (end && *end) {
        switch (*end) {
            case 'k': case 'K': scale = 1024ull; end++; break;
            case 'm': case 'M': scale = 1024ull * 1024ull; end++; break;
            case 'g': case 'G': scale = 1024ull * 1024ull * 1024ull; end++; break;
            default: break;
        }
        if (*end == 'b' || *end == 'B') end++;
        while (*end == ' ' || *end == '\t') end++;
        if (*end != '\0') return 0;
    }
    if (scale != 1 && raw > (unsigned long long)((size_t)-1) / scale) return 0;
    *out = (size_t)(raw * scale);
    return 1;
}

static size_t resolve_from_env(const char* name, size_t fallback) {
    const char* text = getenv(name);
    if (!text || !*text) return fallback;
    size_t value = 0;
    if (!eshkol_reader_parse_size(text, &value)) {
        fprintf(stderr,
                "eshkol: %s is not a byte count (accepts N, NK, NM, NG); "
                "keeping the default\n", name);
        return fallback;
    }
    return value;
}

size_t eshkol_reader_max_source_bytes(void) {
    if (!g_max_source_bytes_resolved) {
        g_max_source_bytes = resolve_from_env("ESHKOL_MAX_SOURCE_BYTES",
                                              ESHKOL_READER_DEFAULT_MAX_SOURCE_BYTES);
        g_max_source_bytes_resolved = 1;
    }
    return g_max_source_bytes;
}

void eshkol_reader_set_max_source_bytes(size_t bytes) {
    g_max_source_bytes = bytes;
    g_max_source_bytes_resolved = 1;
}

size_t eshkol_reader_max_pass_depth(void) {
    if (!g_max_pass_depth_resolved) {
        g_max_pass_depth = resolve_from_env("ESHKOL_MAX_PASS_DEPTH",
                                            ESHKOL_READER_DEFAULT_MAX_PASS_DEPTH);
        g_max_pass_depth_resolved = 1;
    }
    return g_max_pass_depth;
}

void eshkol_reader_set_max_pass_depth(size_t depth) {
    g_max_pass_depth = depth;
    g_max_pass_depth_resolved = 1;
}

int eshkol_reader_source_over_cap(size_t bytes, char* message, size_t cap) {
    const size_t limit = eshkol_reader_max_source_bytes();
    if (limit == 0 || bytes <= limit) return 0;
    if (message && cap) {
        snprintf(message, cap,
                 "source text is %zu bytes, over the %zu-byte input cap "
                 "(raise it with ESHKOL_MAX_SOURCE_BYTES or "
                 "--max-source-bytes=N; 0 removes the cap)",
                 bytes, limit);
    }
    return 1;
}

void eshkol_reader_depth_message(const char* pass_name, size_t depth,
                                 char* message, size_t cap) {
    if (!message || !cap) return;
    snprintf(message, cap,
             "%s reached its %zu-level nesting budget "
             "(raise it with ESHKOL_MAX_PASS_DEPTH or --max-pass-depth=N; "
             "reading and quoted data have no depth limit)",
             pass_name ? pass_name : "this compiler pass", depth);
}
