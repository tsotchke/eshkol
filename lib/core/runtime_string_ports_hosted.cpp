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

/**
 * @brief Open a read-only string port over `str` (R7RS `open-input-string` support).
 *
 * On Windows, copies the bytes into a tmpfile() (rewound after the write) since
 * fmemopen is unavailable there. Elsewhere, the string is first copied into a
 * NUL-terminated arena buffer (so the FILE-backed port has a stable, arena-owned
 * backing store for its lifetime) and opened read-only via fmemopen. A
 * zero-length string is special-cased to a fresh empty tmpfile() so reads hit
 * EOF immediately, on every platform, without depending on fmemopen accepting a
 * zero-size buffer.
 *
 * @param arena_void  Arena used to copy `str` (POSIX path only; ignored on Windows).
 * @param str         Source bytes to back the port (need not be NUL-terminated).
 * @param len         Number of bytes in `str`.
 * @return            A FILE* usable with ordinary stdio calls, or null on failure.
 */
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

/**
 * @brief Open a growable in-memory output port (R7RS `open-output-string` support).
 *
 * Reserves a slot in the fixed-size `string_output_ports` table (capped at
 * MAX_STRING_OUTPUT_PORTS) so `eshkol_get_output_string` can later recover the
 * written bytes by matching on the FILE* pointer. On POSIX, backed by
 * open_memstream, which owns and grows `string_output_ports[idx].buf`/`.size`
 * as data is written. On Windows (no open_memstream), backed by a tmpfile()
 * instead; the writable buffer is read back on demand in
 * `eshkol_get_output_string`.
 *
 * @return  A FILE* to write through, or null if the port table is full or the
 *          underlying stream could not be opened.
 */
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

/**
 * @brief Snapshot the bytes written so far to an output-string port as an
 * arena-allocated Eshkol string (R7RS `get-output-string` support).
 *
 * Flushes `fp` first. On POSIX, looks up the matching entry in
 * `string_output_ports` and copies its open_memstream buffer/size into a new
 * HEAP_SUBTYPE_STRING-tagged arena allocation. On Windows, since the port is a
 * plain tmpfile(), the current file position is saved, the file is measured by
 * seeking to the end, its full contents are read back into the arena copy, and
 * the original position is restored so subsequent writes continue where they
 * left off. If `fp` is not a tracked output-string port, returns an empty
 * arena-allocated string rather than failing.
 *
 * @param arena_void  Arena to allocate the resulting string from.
 * @param fp_void     The FILE* previously returned by eshkol_open_output_string.
 * @return            A NUL-terminated, arena-owned copy of the bytes written so far.
 */
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
