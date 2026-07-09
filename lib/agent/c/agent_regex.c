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

/**
 * @brief Allocates a slot in the global regex handle table for a compiled @p code.
 *
 * Scans forward from the last-used index (wrapping around once) for a free
 * slot, so repeated compile/free cycles reuse released handles instead of
 * exhausting the table.
 *
 * @param code Compiled PCRE2 pattern to store.
 * @return The new handle (>= 1), or -1 if the table is full.
 */
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

/**
 * @brief Looks up the compiled PCRE2 pattern for a given handle.
 *
 * @param h Handle previously returned by alloc_handle() / eshkol_regex_compile().
 * @return The stored pcre2_code pointer, or NULL if @p h is out of range or unused.
 */
static pcre2_code* get_handle(int64_t h) {
    if (h < 1 || h >= MAX_REGEX_HANDLES) return NULL;
    return g_regex_handles[h];
}

/*******************************************************************************
 * Public API
 ******************************************************************************/

/**
 * @brief Compiles a PCRE2 pattern and returns an opaque handle for later matching.
 *
 * Translates the low three bits of @p flags into PCRE2_CASELESS,
 * PCRE2_MULTILINE, and PCRE2_DOTALL respectively (always with PCRE2_UTF set),
 * compiles @p pattern, and stores the result in the handle table.
 *
 * @param pattern NUL-terminated UTF-8 pattern string.
 * @param flags Bit 0 = case-insensitive, bit 1 = multiline, bit 2 = dotall.
 * @return Handle (>= 1) on success, -1 on a NULL pattern, compile error, or
 *         full handle table.
 */
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

/**
 * @brief Returns a lazily-initialized, process-global PCRE2 match context with
 * ReDoS-mitigating backtrack and depth limits.
 *
 * @return The shared match_context (may be NULL if PCRE2 allocation fails).
 */
/* #195 MEDIUM: ReDoS protection. A pattern like "(a+)+$" on a long
 * subject of a's + one unmatched trailing char triggers exponential
 * backtracking and freezes the thread. Without a match_limit, the
 * regex engine will happily burn the CPU for minutes. Build a
 * process-global match_context with a conservative limit (10M
 * backtrack steps) and pass it into every pcre2_match / pcre2_
 * substitute call. 10M is well below the "noticeably slow" threshold
 * on modern hardware but easily enough for legitimate matches.
 *
 * Allocated lazily and never freed — the runtime is process-
 * lifetime anyway, and reusing a single context is cheap and
 * thread-safe per PCRE2 docs (the context is read-only during
 * matching). */
static pcre2_match_context* get_match_context(void) {
    static pcre2_match_context* s_ctx = NULL;
    if (!s_ctx) {
        s_ctx = pcre2_match_context_create(NULL);
        if (s_ctx) {
            /* Backtrack ("match") and heap limits — empirically
             * picked. 10M steps ≈ a few tens of ms on modern CPUs;
             * heap limit guards pathological alternations. */
            pcre2_set_match_limit(s_ctx, 10000000);
            pcre2_set_depth_limit(s_ctx, 100000);
        }
    }
    return s_ctx;
}

/**
 * @brief Runs a single match of a compiled pattern against @p subject.
 *
 * On a match, copies the matched substring (group 0) into @p match_buf,
 * truncating to fit @p buf_size and NUL-terminating.
 *
 * @param handle Compiled pattern handle from eshkol_regex_compile().
 * @param subject NUL-terminated subject string to search.
 * @param match_buf Optional output buffer to receive the matched text.
 * @param buf_size Size of @p match_buf in bytes.
 * @return 1 if a match was found, 0 otherwise (including invalid handle/subject).
 */
int eshkol_regex_match(int64_t handle, const char* subject,
                       char* match_buf, size_t buf_size) {
    pcre2_code* code = get_handle(handle);
    if (!code || !subject) return 0;

    pcre2_match_data* md = pcre2_match_data_create_from_pattern(code, NULL);
    if (!md) return 0;

    int rc = pcre2_match(code, (PCRE2_SPTR)subject, PCRE2_ZERO_TERMINATED,
                         0, 0, md, get_match_context());

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

/**
 * @brief Finds successive non-overlapping matches of a pattern in @p subject.
 *
 * Repeatedly matches starting after the previous match's end (advancing by
 * one byte on zero-length matches to guarantee progress), writing each
 * matched substring into @p matches_buf separated by NUL bytes, until
 * @p max_matches is reached, the subject is exhausted, or the buffer is full.
 *
 * @param handle Compiled pattern handle.
 * @param subject NUL-terminated subject string to search.
 * @param matches_buf Output buffer receiving NUL-separated matched substrings.
 * @param buf_size Size of @p matches_buf in bytes.
 * @param max_matches Maximum number of matches to collect.
 * @return The number of matches written, or 0 on invalid arguments.
 */
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
                             offset, 0, md, get_match_context());
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

/**
 * @brief Performs a global regex substitution of a pattern in @p subject.
 *
 * Delegates to pcre2_substitute() with PCRE2_SUBSTITUTE_GLOBAL. If the
 * substitution fails, falls back to copying @p subject unchanged (truncated
 * to fit @p output) rather than returning an empty result.
 *
 * @param handle Compiled pattern handle.
 * @param subject NUL-terminated subject string.
 * @param replacement Replacement text (may contain PCRE2 substitution syntax).
 * @param output Output buffer receiving the result.
 * @param output_size Size of @p output in bytes.
 * @return Length of the resulting string on success, or -1 on invalid arguments.
 */
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

/**
 * @brief Frees a compiled pattern and releases its handle slot.
 *
 * @param handle Handle previously returned by eshkol_regex_compile(). No-op if
 *        the handle is invalid or already freed.
 */
void eshkol_regex_free(int64_t handle) {
    pcre2_code* code = get_handle(handle);
    if (code) {
        pcre2_code_free(code);
        g_regex_handles[handle] = NULL;
    }
}

/*******************************************************************************
 * Capture-group extraction (Noesis #167)
 *
 * The match / match-all routines above discard everything except group 0
 * (the whole match). To support parsing workflows — JSON Schema (#172),
 * CSV/LaTeX extraction, HTTP Link headers, etc. — callers need the
 * per-group substrings. These helpers return a single flat null-separated
 * buffer so the Scheme side (lib/agent/regex.esk) can split on NUL bytes
 * into a list of strings. A dedicated count helper avoids a pre-allocation
 * guess.
 *
 * Layout of out_buf (returned by eshkol_regex_match_groups):
 *   "group0\0group1\0group2\0…\0groupN\0"
 * where groupK is the captured substring for capture K (or empty string
 * if that group didn't participate in the match).
 ******************************************************************************/

/**
 * @brief Reports how many capture groups (including group 0) a single match
 * of @p subject against the compiled pattern would produce.
 *
 * Useful for callers that want to size a buffer before calling
 * eshkol_regex_match_groups().
 *
 * @param handle Compiled pattern handle.
 * @param subject NUL-terminated subject string.
 * @return Number of groups on a match, -1 if no match, 0 on invalid input.
 */
/* Return the number of capture groups (including group 0) on success;
 * -1 on no match; 0 on invalid input. */
int eshkol_regex_match_groups_count(int64_t handle, const char* subject) {
    pcre2_code* code = get_handle(handle);
    if (!code || !subject) return 0;

    pcre2_match_data* md = pcre2_match_data_create_from_pattern(code, NULL);
    if (!md) return 0;

    int rc = pcre2_match(code, (PCRE2_SPTR)subject, PCRE2_ZERO_TERMINATED,
                         0, 0, md, get_match_context());
    pcre2_match_data_free(md);
    return rc;
}

/**
 * @brief Matches @p subject and writes every capture group's substring into
 * @p out_buf as a NUL-separated list.
 *
 * Layout of @p out_buf is "group0\0group1\0...\0groupN\0" where group 0 is
 * the whole match; a group that didn't participate in the match is emitted
 * as an empty string.
 *
 * @param handle Compiled pattern handle.
 * @param subject NUL-terminated subject string.
 * @param out_buf Output buffer receiving the NUL-separated group substrings.
 * @param buf_size Size of @p out_buf in bytes.
 * @return Number of groups written (including group 0) on success, -1 if no
 *         match, 0 on invalid input or if @p out_buf is too small.
 */
/* Fill out_buf with NUL-separated group substrings for the first match.
 * Returns the number of groups written on success (including group 0),
 * -1 if no match, 0 on invalid input or buffer overflow. */
int eshkol_regex_match_groups(int64_t handle, const char* subject,
                               char* out_buf, size_t buf_size) {
    pcre2_code* code = get_handle(handle);
    if (!code || !subject || !out_buf || buf_size == 0) return 0;

    pcre2_match_data* md = pcre2_match_data_create_from_pattern(code, NULL);
    if (!md) return 0;

    int rc = pcre2_match(code, (PCRE2_SPTR)subject, PCRE2_ZERO_TERMINATED,
                         0, 0, md, get_match_context());
    if (rc < 1) {
        pcre2_match_data_free(md);
        return -1;
    }

    PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(md);
    size_t buf_pos = 0;

    for (int i = 0; i < rc; i++) {
        PCRE2_SIZE start = ovector[2 * i];
        PCRE2_SIZE end   = ovector[2 * i + 1];

        /* Unset group: pcre2 marks with PCRE2_UNSET. Emit empty string. */
        size_t group_len = 0;
        const char* src = "";
        if (start != PCRE2_UNSET && end != PCRE2_UNSET && end >= start) {
            group_len = end - start;
            src = subject + start;
        }
        if (buf_pos + group_len + 1 > buf_size) {
            pcre2_match_data_free(md);
            return 0;  /* overflow */
        }
        memcpy(out_buf + buf_pos, src, group_len);
        buf_pos += group_len;
        out_buf[buf_pos++] = '\0';
    }

    pcre2_match_data_free(md);
    return rc;
}

/**
 * @brief Resolves a named capture group to its 1-based group number.
 *
 * @param handle Compiled pattern handle.
 * @param name Capture group name to look up.
 * @return The 1-based group number, or -1 if @p name doesn't exist in the pattern.
 */
/* Count capture groups for a named group lookup. Returns the 1-based
 * group number, or -1 if not found. */
int eshkol_regex_named_group_number(int64_t handle, const char* name) {
    pcre2_code* code = get_handle(handle);
    if (!code || !name) return -1;
    int num = pcre2_substring_number_from_name(code, (PCRE2_SPTR)name);
    if (num < 0) return -1;
    return num;
}

/* Replace with a callback. NOT implemented here — the Scheme layer can
 * compose match-groups + regex-replace as a higher-level (regex-replace-fn).
 * A dedicated pcre2_substitute_callback isn't part of PCRE2's stable API. */
