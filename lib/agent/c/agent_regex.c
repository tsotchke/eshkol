/*******************************************************************************
 * PCRE2 Regular Expression Wrapper for Eshkol
 *
 * Provides compile, match, match-all, and replace operations.
 * Uses opaque int64_t handles for FFI compatibility.
 *
 * Copyright (c) 2025 Eshkol Project
 ******************************************************************************/

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/*******************************************************************************
 * Handle Table
 ******************************************************************************/

#define MAX_REGEX_HANDLES 256

static pcre2_code* g_regex_handles[MAX_REGEX_HANDLES] = {0};
static int g_next_handle = 1;

static int alloc_handle(pcre2_code* code) {
    for (int i = g_next_handle; i < MAX_REGEX_HANDLES; i++) {
        if (!g_regex_handles[i]) {
            g_regex_handles[i] = code;
            g_next_handle = i + 1;
            return i;
        }
    }
    /* Wrap around */
    for (int i = 1; i < g_next_handle; i++) {
        if (!g_regex_handles[i]) {
            g_regex_handles[i] = code;
            g_next_handle = i + 1;
            return i;
        }
    }
    return -1;  /* Table full */
}

static pcre2_code* get_handle(int64_t h) {
    if (h < 1 || h >= MAX_REGEX_HANDLES) return NULL;
    return g_regex_handles[h];
}

/*******************************************************************************
 * Public API
 ******************************************************************************/

int64_t eshkol_regex_compile(const char* pattern, int flags) {
    if (!pattern) return -1;

    uint32_t options = PCRE2_UTF;
    if (flags & 1) options |= PCRE2_CASELESS;
    if (flags & 2) options |= PCRE2_MULTILINE;
    if (flags & 4) options |= PCRE2_DOTALL;

    int errnum;
    PCRE2_SIZE erroff;
    pcre2_code* code = pcre2_compile(
        (PCRE2_SPTR)pattern, PCRE2_ZERO_TERMINATED,
        options, &errnum, &erroff, NULL);

    if (!code) return -1;

    int handle = alloc_handle(code);
    if (handle < 0) {
        pcre2_code_free(code);
        return -1;
    }
    return handle;
}

int eshkol_regex_match(int64_t handle, const char* subject,
                       char* match_buf, size_t buf_size) {
    pcre2_code* code = get_handle(handle);
    if (!code || !subject) return 0;

    pcre2_match_data* md = pcre2_match_data_create_from_pattern(code, NULL);
    if (!md) return 0;

    int rc = pcre2_match(code, (PCRE2_SPTR)subject, PCRE2_ZERO_TERMINATED,
                         0, 0, md, NULL);

    if (rc < 1) {
        pcre2_match_data_free(md);
        return 0;
    }

    PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(md);
    size_t match_len = ovector[1] - ovector[0];

    if (match_buf && buf_size > 0) {
        size_t copy_len = match_len < buf_size - 1 ? match_len : buf_size - 1;
        memcpy(match_buf, subject + ovector[0], copy_len);
        match_buf[copy_len] = '\0';
    }

    pcre2_match_data_free(md);
    return 1;
}

int eshkol_regex_match_all(int64_t handle, const char* subject,
                            char* matches_buf, size_t buf_size,
                            int max_matches) {
    pcre2_code* code = get_handle(handle);
    if (!code || !subject || !matches_buf || buf_size == 0) return 0;

    pcre2_match_data* md = pcre2_match_data_create_from_pattern(code, NULL);
    if (!md) return 0;

    PCRE2_SIZE subject_len = strlen(subject);
    PCRE2_SIZE offset = 0;
    int count = 0;
    size_t buf_pos = 0;

    while (count < max_matches && offset < subject_len) {
        int rc = pcre2_match(code, (PCRE2_SPTR)subject, subject_len,
                             offset, 0, md, NULL);
        if (rc < 1) break;

        PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(md);
        size_t match_len = ovector[1] - ovector[0];

        /* Copy match to buffer, null-separated */
        if (buf_pos + match_len + 1 >= buf_size) break;
        memcpy(matches_buf + buf_pos, subject + ovector[0], match_len);
        buf_pos += match_len;
        matches_buf[buf_pos++] = '\0';

        count++;

        /* Advance past match (handle zero-length matches) */
        offset = ovector[1];
        if (ovector[0] == ovector[1]) offset++;
    }

    pcre2_match_data_free(md);
    return count;
}

int eshkol_regex_replace(int64_t handle, const char* subject,
                          const char* replacement,
                          char* output, size_t output_size) {
    pcre2_code* code = get_handle(handle);
    if (!code || !subject || !replacement || !output || output_size == 0) return -1;

    PCRE2_SIZE out_len = output_size;
    int rc = pcre2_substitute(
        code, (PCRE2_SPTR)subject, PCRE2_ZERO_TERMINATED,
        0, PCRE2_SUBSTITUTE_GLOBAL,
        NULL, NULL,
        (PCRE2_SPTR)replacement, PCRE2_ZERO_TERMINATED,
        (PCRE2_UCHAR*)output, &out_len);

    if (rc < 0) {
        /* Substitution failed — copy original */
        size_t len = strlen(subject);
        if (len >= output_size) len = output_size - 1;
        memcpy(output, subject, len);
        output[len] = '\0';
        return (int)len;
    }

    return (int)out_len;
}

void eshkol_regex_free(int64_t handle) {
    pcre2_code* code = get_handle(handle);
    if (code) {
        pcre2_code_free(code);
        g_regex_handles[handle] = NULL;
    }
}
