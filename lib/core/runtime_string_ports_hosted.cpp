/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Hosted string-port runtime helpers.
 */

#include "arena_memory.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// String ports use fmemopen (input) / open_memstream (output) to create
// FILE-backed ports from in-memory strings. Existing I/O operations work
// unchanged since they operate on FILE* pointers.

#define MAX_STRING_OUTPUT_PORTS 256

static struct {
    FILE* fp;
    char* buf;
    size_t size;
} string_output_ports[MAX_STRING_OUTPUT_PORTS];

static int num_string_output_ports = 0;

extern "C" void* eshkol_open_input_string(void* arena_void, const char* str, int64_t len) {
    if (len == 0) {
        // tmpfile() is portable and returns EOF immediately on first read.
        return tmpfile();
    }
#ifdef _WIN32
    (void)arena_void;
    FILE* fp = tmpfile();
    if (!fp) {
        return nullptr;
    }
    if (fwrite(str, 1, static_cast<size_t>(len), fp) != static_cast<size_t>(len)) {
        fclose(fp);
        return nullptr;
    }
    rewind(fp);
    return fp;
#else
    auto* arena = static_cast<arena_t*>(arena_void);
    // Copy string to arena so fmemopen has a stable buffer.
    char* copy = static_cast<char*>(arena_allocate(arena, static_cast<size_t>(len) + 1));
    memcpy(copy, str, static_cast<size_t>(len));
    copy[len] = '\0';
    return fmemopen(copy, static_cast<size_t>(len), "r");
#endif
}

extern "C" void* eshkol_open_output_string(void) {
    if (num_string_output_ports >= MAX_STRING_OUTPUT_PORTS) return nullptr;
    int idx = num_string_output_ports++;
    string_output_ports[idx].buf = nullptr;
    string_output_ports[idx].size = 0;
#ifdef _WIN32
    FILE* fp = tmpfile();
#else
    FILE* fp = open_memstream(&string_output_ports[idx].buf,
                              &string_output_ports[idx].size);
#endif
    string_output_ports[idx].fp = fp;
    return fp;
}

extern "C" void* eshkol_get_output_string(void* arena_void, void* fp_void) {
    auto* arena = static_cast<arena_t*>(arena_void);
    FILE* fp = static_cast<FILE*>(fp_void);
    fflush(fp);
    for (int i = 0; i < num_string_output_ports; i++) {
        if (string_output_ports[i].fp == fp) {
#ifdef _WIN32
            long saved_pos = ftell(fp);
            if (saved_pos < 0) {
                saved_pos = 0;
            }
            if (fseek(fp, 0, SEEK_END) != 0) {
                break;
            }
            long end_pos = ftell(fp);
            if (end_pos < 0) {
                break;
            }
            rewind(fp);
            size_t len = static_cast<size_t>(end_pos);
            char* result = static_cast<char*>(arena_allocate_with_header(
                arena, len + 1, HEAP_SUBTYPE_STRING, 0));
            size_t read_len = len > 0 ? fread(result, 1, len, fp) : 0;
            result[read_len] = '\0';
            fseek(fp, saved_pos, SEEK_SET);
            return result;
#else
            size_t len = string_output_ports[i].size;
            char* result = static_cast<char*>(arena_allocate_with_header(
                arena, len + 1, HEAP_SUBTYPE_STRING, 0));
            memcpy(result, string_output_ports[i].buf, len);
            result[len] = '\0';
            return result;
#endif
        }
    }

    char* result = static_cast<char*>(arena_allocate_with_header(
        arena, 1, HEAP_SUBTYPE_STRING, 0));
    result[0] = '\0';
    return result;
}
