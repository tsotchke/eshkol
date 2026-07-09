/*******************************************************************************
 * Knowledge Base Persistence for Eshkol Agent
 *
 * Save/load knowledge base facts as JSON for persistent reasoning across
 * agent sessions (tool success patterns, user preferences, learned rules).
 *
 * File format: JSON array of fact objects:
 *   [{"subject":"model","predicate":"gpt2-medium","object":"layers","value":24},
 *    {"subject":"signal","predicate":"total-energy","object":"rho","value":0.3244}]
 *
 * This file does NOT depend on Eshkol's VM KB internals. It provides a
 * standalone C-level JSON serialization that the Eshkol agent can use
 * via extern declarations to persist and restore facts.
 *
 * The Eshkol wrapper calls (kb-assert!) for each loaded fact and
 * (kb-query ...) to iterate facts for saving.
 *
 * Copyright (c) 2025 Eshkol Project — tsotchke
 ******************************************************************************/

#ifdef __APPLE__
#define _DARWIN_C_SOURCE
#else
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>

/*******************************************************************************
 * Simple JSON Writer (no dependencies)
 ******************************************************************************/

typedef struct {
    char* buf;
    size_t len;
    size_t cap;
} JsonBuf;

/**
 * @brief Initializes a JsonBuf with a freshly malloc'd buffer.
 *
 * Allocates @p initial bytes for the buffer and resets the length to
 * zero. If the allocation fails, @p jb's buf is left NULL and
 * subsequent jb_append()/jb_puts() calls become no-ops.
 *
 * @param jb Buffer to initialize.
 * @param initial Initial capacity in bytes to allocate.
 */
static void jb_init(JsonBuf* jb, size_t initial) {
    jb->buf = (char*)malloc(initial);
    jb->len = 0;
    jb->cap = initial;
    if (jb->buf) jb->buf[0] = '\0';
}

/**
 * @brief Appends a raw byte span to a JsonBuf, growing the buffer as needed.
 *
 * Doubles the buffer capacity until the new content plus a NUL
 * terminator fits, then copies the bytes in. If @p jb's buffer is
 * already NULL (a prior allocation failed) or a realloc() fails here,
 * the buffer is left/marked NULL so further writes are silently
 * skipped.
 *
 * @param jb Buffer to append to.
 * @param str Bytes to append (need not be NUL-terminated).
 * @param slen Number of bytes to copy from @p str.
 */
static void jb_append(JsonBuf* jb, const char* str, size_t slen) {
    if (!jb->buf) return;
    while (jb->len + slen + 1 > jb->cap) {
        jb->cap *= 2;
        char* nb = (char*)realloc(jb->buf, jb->cap);
        if (!nb) { free(jb->buf); jb->buf = NULL; return; }
        jb->buf = nb;
    }
    memcpy(jb->buf + jb->len, str, slen);
    jb->len += slen;
    jb->buf[jb->len] = '\0';
}

/**
 * @brief Appends a NUL-terminated C string to a JsonBuf.
 *
 * Convenience wrapper around jb_append() that computes the length via
 * strlen().
 */
static void jb_puts(JsonBuf* jb, const char* str) {
    jb_append(jb, str, strlen(str));
}

/* Write a JSON-escaped string (with surrounding quotes) */
/**
 * @brief Writes a JSON string literal (quoted and escaped) into the buffer.
 *
 * Wraps @p str in double quotes and escapes control characters per the
 * JSON spec: `"`, `\`, newline, carriage return, and tab get their
 * short escapes; other bytes below 0x20 become `\u00XX`. A NULL
 * @p str is written as an empty string `""`.
 */
static void jb_string(JsonBuf* jb, const char* str) {
    jb_puts(jb, "\"");
    if (str) {
        for (const char* p = str; *p; p++) {
            switch (*p) {
                case '"':  jb_puts(jb, "\\\""); break;
                case '\\': jb_puts(jb, "\\\\"); break;
                case '\n': jb_puts(jb, "\\n"); break;
                case '\r': jb_puts(jb, "\\r"); break;
                case '\t': jb_puts(jb, "\\t"); break;
                default:
                    if ((unsigned char)*p < 0x20) {
                        char esc[8];
                        snprintf(esc, sizeof(esc), "\\u%04x", (unsigned char)*p);
                        jb_puts(jb, esc);
                    } else {
                        jb_append(jb, p, 1);
                    }
            }
        }
    }
    jb_puts(jb, "\"");
}

/**
 * @brief Frees a JsonBuf's backing buffer and resets it to an empty state.
 */
static void jb_free(JsonBuf* jb) {
    free(jb->buf);
    jb->buf = NULL;
    jb->len = 0;
    jb->cap = 0;
}

/*******************************************************************************
 * Fact Storage — Flat Array
 *
 * We store facts in a simple array since the agent typically has < 10K facts.
 * The Eshkol wrapper is responsible for bridging to/from the VM's KB.
 ******************************************************************************/

#define MAX_FACTS 16384
#define MAX_FIELD_LEN 512

typedef struct {
    char subject[MAX_FIELD_LEN];
    char predicate[MAX_FIELD_LEN];
    char object[MAX_FIELD_LEN];
    double value;           /* numeric value (0.0 if not applicable) */
    int has_value;           /* 1 if value field is meaningful */
} Fact;

static Fact g_facts[MAX_FACTS];
static int32_t g_fact_count = 0;

/*
 * Add a fact to the in-memory store.
 * Returns: fact index (>= 0), -1 if full
 */
/**
 * @brief Appends a new fact to the flat in-memory fact store.
 *
 * Copies @p subject, @p predicate, and @p object into the next free
 * slot (truncating each to MAX_FIELD_LEN - 1 characters), along with
 * @p value and the @p has_value flag. NULL string arguments are
 * stored as empty strings.
 *
 * @param subject Fact subject field, or NULL for empty.
 * @param predicate Fact predicate field, or NULL for empty.
 * @param object Fact object field, or NULL for empty.
 * @param value Numeric value associated with the fact (meaningful only if @p has_value is nonzero).
 * @param has_value Non-zero if @p value should be considered set.
 * @return The new fact's index (>= 0) on success, or -1 if the store is full (MAX_FACTS reached).
 */
int32_t eshkol_kb_fact_add(const char* subject, const char* predicate,
                             const char* object, double value, int32_t has_value) {
    if (g_fact_count >= MAX_FACTS) return -1;
    int idx = g_fact_count++;
    Fact* f = &g_facts[idx];
    strncpy(f->subject, subject ? subject : "", MAX_FIELD_LEN - 1);
    strncpy(f->predicate, predicate ? predicate : "", MAX_FIELD_LEN - 1);
    strncpy(f->object, object ? object : "", MAX_FIELD_LEN - 1);
    f->subject[MAX_FIELD_LEN - 1] = '\0';
    f->predicate[MAX_FIELD_LEN - 1] = '\0';
    f->object[MAX_FIELD_LEN - 1] = '\0';
    f->value = value;
    f->has_value = has_value;
    return idx;
}

/*
 * Clear all facts.
 */
/**
 * @brief Discards all stored facts by resetting the fact count to zero.
 *
 * Does not zero or free the underlying g_facts array; existing slots
 * are simply overwritten as new facts are added.
 */
void eshkol_kb_fact_clear(void) {
    g_fact_count = 0;
}

/*
 * Get fact count.
 */
/**
 * @brief Returns the number of facts currently stored.
 */
int32_t eshkol_kb_fact_count(void) {
    return g_fact_count;
}

/*
 * Get fact field by index.
 * field: 0=subject, 1=predicate, 2=object
 * Returns: strlen written to buf, -1 error
 */
/**
 * @brief Copies one string field of a stored fact into a caller-supplied buffer.
 *
 * Truncates to fit if the field is longer than @p buf_size - 1 and
 * always NUL-terminates @p buf.
 *
 * @param idx Fact index, must be in [0, eshkol_kb_fact_count()).
 * @param field Which field to fetch: 0=subject, 1=predicate, 2=object.
 * @param buf Destination buffer.
 * @param buf_size Size of @p buf in bytes.
 * @return Number of characters written (excluding the NUL terminator), or -1 on an invalid index, field, or NULL/zero-size buffer.
 */
int32_t eshkol_kb_fact_get_field(int32_t idx, int32_t field,
                                   char* buf, int32_t buf_size) {
    if (idx < 0 || idx >= g_fact_count || !buf || buf_size <= 0) return -1;
    const Fact* f = &g_facts[idx];
    const char* src;
    switch (field) {
        case 0: src = f->subject; break;
        case 1: src = f->predicate; break;
        case 2: src = f->object; break;
        default: return -1;
    }
    int len = (int32_t)strlen(src);
    if (len >= buf_size) len = buf_size - 1;
    memcpy(buf, src, (size_t)len);
    buf[len] = '\0';
    return len;
}

/*
 * Get fact value by index.
 * Returns: the numeric value, 0.0 if not applicable
 */
/**
 * @brief Returns the numeric value stored with a fact.
 *
 * @param idx Fact index.
 * @return The fact's value field, or 0.0 if @p idx is out of range.
 */
double eshkol_kb_fact_get_value(int32_t idx) {
    if (idx < 0 || idx >= g_fact_count) return 0.0;
    return g_facts[idx].value;
}

/**
 * @brief Reports whether a fact's numeric value field is meaningful.
 *
 * @param idx Fact index.
 * @return 1 if the fact has a valid numeric value, 0 if it does not or @p idx is out of range.
 */
int32_t eshkol_kb_fact_has_value(int32_t idx) {
    if (idx < 0 || idx >= g_fact_count) return 0;
    return g_facts[idx].has_value;
}

/*******************************************************************************
 * Save/Load to JSON file
 ******************************************************************************/

/*
 * Save all facts to a JSON file.
 * Returns: 0 success, -1 error
 */
/**
 * @brief Serializes all stored facts to a JSON array file at @p path.
 *
 * Builds the JSON text in memory (see the JsonBuf/jb_* helpers above),
 * writing each fact as an object with subject/predicate/object string
 * fields and, when the fact has a value (see eshkol_kb_fact_has_value()),
 * a numeric "value" field formatted with up to 15 significant digits.
 * The buffer is then written to @p path in one fwrite() call,
 * overwriting any existing file.
 *
 * @param path Filesystem path to write; NULL is rejected.
 * @return 0 on success, -1 if @p path is NULL, the in-memory buffer
 *   failed to allocate, the file could not be opened, or not all
 *   bytes were written.
 */
int32_t eshkol_kb_save_json(const char* path) {
    if (!path) return -1;

    JsonBuf jb;
    jb_init(&jb, 4096);
    jb_puts(&jb, "[\n");

    for (int i = 0; i < g_fact_count; i++) {
        if (i > 0) jb_puts(&jb, ",\n");
        jb_puts(&jb, "  {");
        jb_puts(&jb, "\"subject\":");  jb_string(&jb, g_facts[i].subject);
        jb_puts(&jb, ",\"predicate\":"); jb_string(&jb, g_facts[i].predicate);
        jb_puts(&jb, ",\"object\":");  jb_string(&jb, g_facts[i].object);
        if (g_facts[i].has_value) {
            char vbuf[64];
            snprintf(vbuf, sizeof(vbuf), ",\"value\":%.15g", g_facts[i].value);
            jb_puts(&jb, vbuf);
        }
        jb_puts(&jb, "}");
    }

    jb_puts(&jb, "\n]\n");

    if (!jb.buf) return -1;

    FILE* f = fopen(path, "w");
    if (!f) { jb_free(&jb); return -1; }
    size_t written = fwrite(jb.buf, 1, jb.len, f);
    fclose(f);
    jb_free(&jb);

    return written == jb.len ? 0 : -1;
}

/*
 * Load facts from a JSON file, appending to the in-memory store.
 *
 * This is a minimal JSON parser that handles the specific format we write.
 * It does NOT handle arbitrary JSON — only the array-of-objects format
 * produced by eshkol_kb_save_json.
 *
 * Returns: number of facts loaded, -1 error
 */
/**
 * @brief Loads facts from a JSON file produced by eshkol_kb_save_json(), appending them to the in-memory store.
 *
 * Reads the whole file into memory (rejecting files that are empty,
 * unreadable, or larger than 10 MiB), then runs a minimal hand-written
 * parser that only understands the specific
 * `[ {"subject":..,"predicate":..,"object":..,"value":..}, ... ]` shape
 * this module itself writes — it is not a general JSON parser.
 * Recognized string fields are unescaped for `\"`, `\\`, `\n`, `\r`,
 * `\t`; unrecognized keys/value types are skipped. Objects whose
 * subject/predicate/object all come out empty are not added.
 *
 * @param path Filesystem path to read; NULL is rejected.
 * @return Number of facts successfully appended, or -1 if @p path is
 *   NULL, the file cannot be opened, or the file size is invalid
 *   (empty or over 10 MiB).
 */
int32_t eshkol_kb_load_json(const char* path) {
    if (!path) return -1;

    FILE* f = fopen(path, "r");
    if (!f) return -1;

    /* Read entire file */
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fsize <= 0 || fsize > 10 * 1024 * 1024) { fclose(f); return -1; }

    char* content = (char*)malloc((size_t)fsize + 1);
    if (!content) { fclose(f); return -1; }
    size_t nread = fread(content, 1, (size_t)fsize, f);
    fclose(f);
    content[nread] = '\0';

    int32_t loaded = 0;
    const char* p = content;

    /* Skip to first '[' */
    while (*p && *p != '[') p++;
    if (!*p) { free(content); return -1; }
    p++;

    /* Parse each object */
    while (*p) {
        /* Skip whitespace and commas */
        while (*p && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t' || *p == ','))
            p++;
        if (*p == ']') break;
        if (*p != '{') break;
        p++;

        char subject[MAX_FIELD_LEN] = "";
        char predicate[MAX_FIELD_LEN] = "";
        char object[MAX_FIELD_LEN] = "";
        double value = 0.0;
        int has_value = 0;

        /* Parse key-value pairs within object */
        while (*p && *p != '}') {
            /* Skip whitespace and commas */
            while (*p && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t' || *p == ','))
                p++;
            if (*p == '}') break;

            /* Parse key (must be quoted string) */
            if (*p != '"') break;
            p++;
            const char* key_start = p;
            while (*p && *p != '"') p++;
            size_t key_len = (size_t)(p - key_start);
            if (*p == '"') p++;
            if (*p == ':') p++;

            /* Skip whitespace */
            while (*p && (*p == ' ' || *p == '\t')) p++;

            /* Parse value */
            if (*p == '"') {
                /* String value */
                p++;
                const char* val_start = p;
                /* Handle escaped characters */
                char val_buf[MAX_FIELD_LEN];
                int vi = 0;
                while (*p && *p != '"' && vi < MAX_FIELD_LEN - 1) {
                    if (*p == '\\' && *(p+1)) {
                        p++;
                        switch (*p) {
                            case 'n': val_buf[vi++] = '\n'; break;
                            case 'r': val_buf[vi++] = '\r'; break;
                            case 't': val_buf[vi++] = '\t'; break;
                            case '"': val_buf[vi++] = '"'; break;
                            case '\\': val_buf[vi++] = '\\'; break;
                            default: val_buf[vi++] = *p; break;
                        }
                    } else {
                        val_buf[vi++] = *p;
                    }
                    p++;
                }
                val_buf[vi] = '\0';
                if (*p == '"') p++;
                (void)val_start;

                if (key_len == 7 && memcmp(key_start, "subject", 7) == 0)
                    strncpy(subject, val_buf, MAX_FIELD_LEN - 1);
                else if (key_len == 9 && memcmp(key_start, "predicate", 9) == 0)
                    strncpy(predicate, val_buf, MAX_FIELD_LEN - 1);
                else if (key_len == 6 && memcmp(key_start, "object", 6) == 0)
                    strncpy(object, val_buf, MAX_FIELD_LEN - 1);
            } else if (*p == '-' || isdigit((unsigned char)*p)) {
                /* Numeric value */
                char* end;
                double v = strtod(p, &end);
                if (end != p) {
                    if (key_len == 5 && memcmp(key_start, "value", 5) == 0) {
                        value = v;
                        has_value = 1;
                    }
                    p = end;
                }
            } else {
                /* Unknown value type, skip */
                while (*p && *p != ',' && *p != '}') p++;
            }
        }

        if (*p == '}') p++;

        /* Add the parsed fact */
        if (subject[0] || predicate[0] || object[0]) {
            if (eshkol_kb_fact_add(subject, predicate, object, value, has_value) >= 0) {
                loaded++;
            }
        }
    }

    free(content);
    return loaded;
}
