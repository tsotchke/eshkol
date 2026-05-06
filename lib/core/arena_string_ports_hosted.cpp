/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Hosted string-port runtime support.
 */

#include "arena_memory.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

#define MAX_STRING_OUTPUT_PORTS 256

struct string_output_port_entry {
    FILE* fp;
    char* buf;
    size_t size;
};

string_output_port_entry g_string_output_ports[MAX_STRING_OUTPUT_PORTS];
int g_num_string_output_ports = 0;

} // namespace

extern "C" void* eshkol_open_input_string(void* arena_void, const char* str, int64_t len) {
    if (len == 0) {
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
    arena_t* arena = static_cast<arena_t*>(arena_void);
    char* copy = static_cast<char*>(arena_allocate(arena, static_cast<size_t>(len) + 1));
    std::memcpy(copy, str, static_cast<size_t>(len));
    copy[len] = '\0';
    return fmemopen(copy, static_cast<size_t>(len), "r");
#endif
}

extern "C" void* eshkol_open_output_string(void) {
    if (g_num_string_output_ports >= MAX_STRING_OUTPUT_PORTS) {
        return nullptr;
    }

    const int idx = g_num_string_output_ports++;
    g_string_output_ports[idx].buf = nullptr;
    g_string_output_ports[idx].size = 0;

#ifdef _WIN32
    FILE* fp = tmpfile();
#else
    FILE* fp = open_memstream(&g_string_output_ports[idx].buf,
                              &g_string_output_ports[idx].size);
#endif

    g_string_output_ports[idx].fp = fp;
    return fp;
}

extern "C" void* eshkol_get_output_string(void* arena_void, void* fp_void) {
    arena_t* arena = static_cast<arena_t*>(arena_void);
    FILE* fp = static_cast<FILE*>(fp_void);
    fflush(fp);

    for (int i = 0; i < g_num_string_output_ports; i++) {
        if (g_string_output_ports[i].fp != fp) {
            continue;
        }

#ifdef _WIN32
        long saved_pos = ftell(fp);
        if (saved_pos < 0) {
            saved_pos = 0;
        }
        if (fseek(fp, 0, SEEK_END) != 0) {
            break;
        }

        const long end_pos = ftell(fp);
        if (end_pos < 0) {
            break;
        }

        rewind(fp);
        const size_t len = static_cast<size_t>(end_pos);
        char* result = static_cast<char*>(
            arena_allocate_with_header(arena, len + 1, HEAP_SUBTYPE_STRING, 0));
        const size_t read_len = len > 0 ? fread(result, 1, len, fp) : 0;
        result[read_len] = '\0';
        fseek(fp, saved_pos, SEEK_SET);
        return result;
#else
        const size_t len = g_string_output_ports[i].size;
        char* result = static_cast<char*>(
            arena_allocate_with_header(arena, len + 1, HEAP_SUBTYPE_STRING, 0));
        std::memcpy(result, g_string_output_ports[i].buf, len);
        result[len] = '\0';
        return result;
#endif
    }

    char* result = static_cast<char*>(
        arena_allocate_with_header(arena, 1, HEAP_SUBTYPE_STRING, 0));
    result[0] = '\0';
    return result;
}
