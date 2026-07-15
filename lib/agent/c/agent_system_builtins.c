/*
 * Tagged Scheme builtin adapters for production agent capabilities.
 *
 * These entry points are called directly from LLVM-generated code.  They live
 * in eshkol-agent-ffi so compression, Tree-sitter, and Yoga use the exact same
 * implementations as their C APIs and hosted VM dispatch.  There is no
 * heuristic parser, simplified layout engine, or unavailable/no-op branch.
 */

#include <stddef.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <eshkol/agent_capabilities.h>
#include <eshkol/eshkol.h>

#include "agent_native_mutex.h"

extern void* get_global_arena(void);
extern char* arena_allocate_string_with_header(void*, size_t);
extern void* arena_allocate_cons_with_header(void*);
extern int64_t eshkol_string_byte_length(const char*);
extern void* eshkol_make_bytevector(void*, int64_t, int64_t);

typedef eshkol_tagged_value_t sv_t;

static sv_t sv_null(void) {
    sv_t v;
    memset(&v, 0, sizeof(v));
    return v;
}

static sv_t sv_bool(int value) {
    sv_t v = sv_null();
    v.type = ESHKOL_VALUE_BOOL;
    v.data.int_val = value ? 1 : 0;
    return v;
}

static sv_t sv_int(int64_t value) {
    sv_t v = sv_null();
    v.type = ESHKOL_VALUE_INT64;
    v.data.int_val = value;
    return v;
}

static sv_t sv_double(double value) {
    sv_t v = sv_null();
    v.type = ESHKOL_VALUE_DOUBLE;
    v.data.double_val = value;
    return v;
}

static uint8_t sv_subtype(const sv_t* value) {
    if (!value || (value->type & 0x0f) != ESHKOL_VALUE_HEAP_PTR ||
        value->data.ptr_val == 0)
        return UINT8_MAX;
    const eshkol_object_header_t* header =
        (const eshkol_object_header_t*)((const uint8_t*)(uintptr_t)value->data.ptr_val -
                                        sizeof(eshkol_object_header_t));
    return header->subtype;
}

static int sv_number(const sv_t* value, double* out) {
    if (!value || !out) return 0;
    uint8_t type = value->type & 0x0f;
    if (type == ESHKOL_VALUE_INT64) {
        *out = (double)value->data.int_val;
        return 1;
    }
    if (type == ESHKOL_VALUE_DOUBLE) {
        *out = value->data.double_val;
        return 1;
    }
    return 0;
}

static int sv_integer(const sv_t* value, int64_t* out) {
    double number = 0.0;
    if (!sv_number(value, &number) || !isfinite(number) ||
        number < (double)INT64_MIN || number > (double)INT64_MAX ||
        floor(number) != number)
        return 0;
    *out = (int64_t)number;
    return 1;
}

static int sv_string(const sv_t* value, const char** data, int32_t* len) {
    if (!value || !data || !len || sv_subtype(value) != HEAP_SUBTYPE_STRING)
        return 0;
    const char* ptr = (const char*)(uintptr_t)value->data.ptr_val;
    int64_t n = eshkol_string_byte_length(ptr);
    if (n < 0 || n > INT32_MAX) return 0;
    *data = ptr;
    *len = (int32_t)n;
    return 1;
}

static int sv_bytes(const sv_t* value, const char** data, int32_t* len) {
    if (sv_string(value, data, len)) return 1;
    if (!value || !data || !len || sv_subtype(value) != HEAP_SUBTYPE_BYTEVECTOR)
        return 0;
    const uint8_t* ptr = (const uint8_t*)(uintptr_t)value->data.ptr_val;
    int64_t n = 0;
    memcpy(&n, ptr, sizeof(n));
    if (n < 0 || n > INT32_MAX) return 0;
    *data = (const char*)ptr + 8;
    *len = (int32_t)n;
    return 1;
}

static sv_t sv_string_copy(const char* data, size_t len) {
    if ((!data && len != 0) || len > SIZE_MAX - 1u) return sv_bool(0);
    void* arena = get_global_arena();
    if (!arena) return sv_bool(0);
    char* copy = arena_allocate_string_with_header(arena, len);
    if (!copy) return sv_bool(0);
    if (len > 0) memcpy(copy, data, len);
    copy[len] = '\0';
    sv_t out = sv_null();
    out.type = ESHKOL_VALUE_HEAP_PTR;
    out.data.ptr_val = (uint64_t)(uintptr_t)copy;
    return out;
}

static sv_t sv_bytevector_copy(const char* data, int32_t len) {
    if (len < 0 || (!data && len != 0)) return sv_bool(0);
    void* arena = get_global_arena();
    if (!arena) return sv_bool(0);
    void* payload = eshkol_make_bytevector(arena, len, 0);
    if (!payload) return sv_bool(0);
    if (len > 0) memcpy((uint8_t*)payload + 8, data, (size_t)len);
    sv_t out = sv_null();
    out.type = ESHKOL_VALUE_HEAP_PTR;
    out.data.ptr_val = (uint64_t)(uintptr_t)payload;
    return out;
}

static sv_t sv_pair(sv_t car, sv_t cdr) {
    void* arena = get_global_arena();
    if (!arena) return sv_bool(0);
    void* cell = arena_allocate_cons_with_header(arena);
    if (!cell) return sv_bool(0);
    memcpy(cell, &car, sizeof(car));
    memcpy((uint8_t*)cell + sizeof(car), &cdr, sizeof(cdr));
    sv_t out = sv_null();
    out.type = ESHKOL_VALUE_HEAP_PTR;
    out.data.ptr_val = (uint64_t)(uintptr_t)cell;
    return out;
}

static sv_t sv_reverse(sv_t list) {
    sv_t out = sv_null();
    while (sv_subtype(&list) == HEAP_SUBTYPE_CONS) {
        sv_t car;
        sv_t cdr;
        const uint8_t* cell = (const uint8_t*)(uintptr_t)list.data.ptr_val;
        memcpy(&car, cell, sizeof(car));
        memcpy(&cdr, cell + sizeof(car), sizeof(cdr));
        out = sv_pair(car, out);
        list = cdr;
    }
    return out;
}

static sv_t sv_alist_entry(const char* key, sv_t value) {
    return sv_pair(sv_string_copy(key, strlen(key)), value);
}

/* A 256 MiB ceiling prevents decompression bombs while supporting large
 * source/model payloads.  It is a hard failure, never silent truncation. */
#define ESHKOL_BUILTIN_COMPRESSION_LIMIT (256 * 1024 * 1024)

typedef int32_t (*compression_alloc_fn)(const char*, int32_t, int32_t,
                                        char**, int32_t*);

static sv_t compression_builtin(const sv_t* input, compression_alloc_fn fn) {
    const char* data = NULL;
    int32_t data_len = 0;
    if (!fn || !sv_bytes(input, &data, &data_len)) return sv_bool(0);
    char* output = NULL;
    int32_t output_len = 0;
    if (fn(data, data_len, ESHKOL_BUILTIN_COMPRESSION_LIMIT,
           &output, &output_len) != 0)
        return sv_bool(0);
    sv_t result = sv_bytevector_copy(output, output_len);
    eshkol_compression_free(output);
    return result;
}

ESHKOL_AGENT_API void eshkol_builtin_compression_available(sv_t* out) {
    if (out) *out = sv_bool(eshkol_compression_available() == 1);
}
ESHKOL_AGENT_API void eshkol_builtin_deflate(sv_t* out, const sv_t* input) {
    if (out) *out = compression_builtin(input, eshkol_deflate_alloc);
}
ESHKOL_AGENT_API void eshkol_builtin_inflate(sv_t* out, const sv_t* input) {
    if (out) *out = compression_builtin(input, eshkol_inflate_alloc);
}
ESHKOL_AGENT_API void eshkol_builtin_gzip(sv_t* out, const sv_t* input) {
    if (out) *out = compression_builtin(input, eshkol_gzip_alloc);
}
ESHKOL_AGENT_API void eshkol_builtin_gunzip(sv_t* out, const sv_t* input) {
    if (out) *out = compression_builtin(input, eshkol_gunzip_alloc);
}

typedef struct {
    int active;
    int64_t tree;
    uint32_t start_byte;
    uint32_t end_byte;
    int root;
    char type[128];
} builtin_ts_node_t;

static int g_builtin_ts_parsers[32];
static int g_builtin_ts_trees[64];
static int g_builtin_ts_queries[32];
static builtin_ts_node_t g_builtin_ts_nodes[4096];
static eshkol_agent_mutex_t g_builtin_ts_mutex = ESHKOL_AGENT_MUTEX_INITIALIZER;

static int split_tabs(char* record, char** fields, int capacity) {
    int count = 0;
    if (!record || !fields || capacity <= 0) return 0;
    fields[count++] = record;
    for (char* p = record; *p && count < capacity; ++p) {
        if (*p == '\t') {
            *p = '\0';
            fields[count++] = p + 1;
        }
    }
    return count;
}

static int parse_node_record(char* record, builtin_ts_node_t* node) {
    char* fields[9];
    if (!node || split_tabs(record, fields, 9) != 9) return 0;
    char* end = NULL;
    unsigned long start_byte = strtoul(fields[5], &end, 10);
    if (!end || *end) return 0;
    unsigned long end_byte = strtoul(fields[6], &end, 10);
    if (!end || *end || start_byte > UINT32_MAX || end_byte > UINT32_MAX ||
        start_byte > end_byte)
        return 0;
    size_t type_len = strlen(fields[0]);
    if (type_len == 0 || type_len >= sizeof(node->type)) return 0;
    memset(node, 0, sizeof(*node));
    node->active = 1;
    node->start_byte = (uint32_t)start_byte;
    node->end_byte = (uint32_t)end_byte;
    memcpy(node->type, fields[0], type_len + 1u);
    return 1;
}

static int64_t store_node(const builtin_ts_node_t* node) {
    int64_t handle = -1;
    eshkol_agent_mutex_lock(&g_builtin_ts_mutex);
    for (int i = 1; i < (int)(sizeof(g_builtin_ts_nodes) / sizeof(g_builtin_ts_nodes[0])); ++i) {
        if (!g_builtin_ts_nodes[i].active) {
            g_builtin_ts_nodes[i] = *node;
            handle = i;
            break;
        }
    }
    eshkol_agent_mutex_unlock(&g_builtin_ts_mutex);
    return handle;
}

static int load_node(int64_t handle, builtin_ts_node_t* node) {
    int ok = 0;
    eshkol_agent_mutex_lock(&g_builtin_ts_mutex);
    if (handle > 0 && handle < (int64_t)(sizeof(g_builtin_ts_nodes) / sizeof(g_builtin_ts_nodes[0])) &&
        g_builtin_ts_nodes[handle].active) {
        *node = g_builtin_ts_nodes[handle];
        ok = 1;
    }
    eshkol_agent_mutex_unlock(&g_builtin_ts_mutex);
    return ok;
}

static void clear_tree_nodes(int64_t tree) {
    for (int i = 1; i < (int)(sizeof(g_builtin_ts_nodes) / sizeof(g_builtin_ts_nodes[0])); ++i) {
        if (g_builtin_ts_nodes[i].active && g_builtin_ts_nodes[i].tree == tree)
            memset(&g_builtin_ts_nodes[i], 0, sizeof(g_builtin_ts_nodes[i]));
    }
}

static int read_exact_serialized(int32_t (*fn)(int64_t, char*, int32_t),
                                 int64_t handle, char** out) {
    int32_t required = fn(handle, NULL, 0);
    if (required < 0 || required >= INT32_MAX) return -1;
    char* buffer = (char*)malloc((size_t)required + 1u);
    if (!buffer) return -1;
    int32_t written = fn(handle, buffer, required + 1);
    if (written != required) {
        free(buffer);
        return -1;
    }
    buffer[required] = '\0';
    *out = buffer;
    return required;
}

ESHKOL_AGENT_API void eshkol_builtin_ts_parser_new(sv_t* out, const sv_t* language_value) {
    const char* language = NULL;
    int32_t len = 0;
    if (!out) return;
    if (!sv_string(language_value, &language, &len)) { *out = sv_bool(0); return; }
    char* name = (char*)malloc((size_t)len + 1u);
    if (!name) { *out = sv_bool(0); return; }
    memcpy(name, language, (size_t)len);
    name[len] = '\0';
    int64_t handle = eshkol_ts_parser_new(name);
    free(name);
    if (handle <= 0 || handle >= (int64_t)(sizeof(g_builtin_ts_parsers) / sizeof(g_builtin_ts_parsers[0]))) {
        *out = sv_bool(0);
        return;
    }
    eshkol_agent_mutex_lock(&g_builtin_ts_mutex);
    g_builtin_ts_parsers[handle] = 1;
    eshkol_agent_mutex_unlock(&g_builtin_ts_mutex);
    *out = sv_int(handle);
}

ESHKOL_AGENT_API void eshkol_builtin_ts_parser_free(sv_t* out, const sv_t* parser_value) {
    int64_t handle = 0;
    int valid = sv_integer(parser_value, &handle);
    eshkol_agent_mutex_lock(&g_builtin_ts_mutex);
    valid = valid && handle > 0 && handle < (int64_t)(sizeof(g_builtin_ts_parsers) / sizeof(g_builtin_ts_parsers[0])) &&
            g_builtin_ts_parsers[handle];
    if (valid) g_builtin_ts_parsers[handle] = 0;
    eshkol_agent_mutex_unlock(&g_builtin_ts_mutex);
    if (valid) eshkol_ts_parser_free(handle);
    if (out) *out = sv_bool(valid);
}

ESHKOL_AGENT_API void eshkol_builtin_ts_parse(sv_t* out, const sv_t* parser_value,
                             const sv_t* source_value) {
    int64_t parser = 0;
    const char* source = NULL;
    int32_t source_len = 0;
    int valid = sv_integer(parser_value, &parser) &&
                sv_string(source_value, &source, &source_len);
    eshkol_agent_mutex_lock(&g_builtin_ts_mutex);
    valid = valid && parser > 0 && parser < (int64_t)(sizeof(g_builtin_ts_parsers) / sizeof(g_builtin_ts_parsers[0])) &&
            g_builtin_ts_parsers[parser];
    eshkol_agent_mutex_unlock(&g_builtin_ts_mutex);
    if (!out) return;
    if (!valid) { *out = sv_bool(0); return; }
    int64_t tree = eshkol_ts_parse(parser, source, source_len);
    if (tree <= 0 || tree >= (int64_t)(sizeof(g_builtin_ts_trees) / sizeof(g_builtin_ts_trees[0]))) {
        *out = sv_bool(0);
        return;
    }
    eshkol_agent_mutex_lock(&g_builtin_ts_mutex);
    g_builtin_ts_trees[tree] = 1;
    eshkol_agent_mutex_unlock(&g_builtin_ts_mutex);
    *out = sv_int(tree);
}

ESHKOL_AGENT_API void eshkol_builtin_ts_tree_free(sv_t* out, const sv_t* tree_value) {
    int64_t tree = 0;
    int valid = sv_integer(tree_value, &tree);
    eshkol_agent_mutex_lock(&g_builtin_ts_mutex);
    valid = valid && tree > 0 && tree < (int64_t)(sizeof(g_builtin_ts_trees) / sizeof(g_builtin_ts_trees[0])) &&
            g_builtin_ts_trees[tree];
    if (valid) {
        g_builtin_ts_trees[tree] = 0;
        clear_tree_nodes(tree);
    }
    eshkol_agent_mutex_unlock(&g_builtin_ts_mutex);
    if (valid) eshkol_ts_tree_free(tree);
    if (out) *out = sv_bool(valid);
}

ESHKOL_AGENT_API void eshkol_builtin_ts_tree_root(sv_t* out, const sv_t* tree_value) {
    int64_t tree = 0;
    int valid = sv_integer(tree_value, &tree);
    eshkol_agent_mutex_lock(&g_builtin_ts_mutex);
    valid = valid && tree > 0 && tree < (int64_t)(sizeof(g_builtin_ts_trees) / sizeof(g_builtin_ts_trees[0])) &&
            g_builtin_ts_trees[tree];
    eshkol_agent_mutex_unlock(&g_builtin_ts_mutex);
    if (!out) return;
    if (!valid) { *out = sv_bool(0); return; }
    char* record = NULL;
    if (read_exact_serialized(eshkol_ts_tree_root, tree, &record) < 0) {
        *out = sv_bool(0);
        return;
    }
    builtin_ts_node_t node;
    int parsed = parse_node_record(record, &node);
    free(record);
    if (!parsed) { *out = sv_bool(0); return; }
    node.tree = tree;
    node.root = 1;
    int64_t handle = store_node(&node);
    *out = handle > 0 ? sv_int(handle) : sv_bool(0);
}

ESHKOL_AGENT_API void eshkol_builtin_ts_node_type(sv_t* out, const sv_t* node_value) {
    int64_t handle = 0;
    builtin_ts_node_t node;
    if (!out) return;
    if (!sv_integer(node_value, &handle) || !load_node(handle, &node)) {
        *out = sv_bool(0);
        return;
    }
    *out = sv_string_copy(node.type, strlen(node.type));
}

ESHKOL_AGENT_API void eshkol_builtin_ts_node_text(sv_t* out, const sv_t* node_value,
                                 const sv_t* unused_source) {
    (void)unused_source;
    int64_t handle = 0;
    builtin_ts_node_t node;
    if (!out) return;
    if (!sv_integer(node_value, &handle) || !load_node(handle, &node) ||
        node.start_byte >= node.end_byte) {
        *out = sv_bool(0);
        return;
    }
    int32_t required = eshkol_ts_node_text(node.tree, node.start_byte,
                                            node.end_byte, NULL, 0);
    if (required < 0) { *out = sv_bool(0); return; }
    char* buffer = (char*)malloc((size_t)required + 1u);
    if (!buffer) { *out = sv_bool(0); return; }
    int32_t written = eshkol_ts_node_text(node.tree, node.start_byte,
                                          node.end_byte, buffer, required + 1);
    *out = written == required ? sv_string_copy(buffer, (size_t)written) : sv_bool(0);
    free(buffer);
}

ESHKOL_AGENT_API void eshkol_builtin_ts_node_children(sv_t* out, const sv_t* node_value) {
    int64_t handle = 0;
    builtin_ts_node_t node;
    if (!out) return;
    if (!sv_integer(node_value, &handle) || !load_node(handle, &node)) {
        *out = sv_bool(0);
        return;
    }
    uint32_t start = node.root ? 0u : node.start_byte;
    uint32_t end = node.root ? 0u : node.end_byte;
    int32_t count = 0;
    int32_t required = eshkol_ts_node_children(node.tree, start, end,
                                               NULL, 0, &count);
    if (required < 0) { *out = sv_bool(0); return; }
    if (count == 0) { *out = sv_null(); return; }
    char* buffer = (char*)malloc((size_t)required + 1u);
    if (!buffer) { *out = sv_bool(0); return; }
    int32_t written = eshkol_ts_node_children(node.tree, start, end,
                                              buffer, required + 1, &count);
    if (written != required) { free(buffer); *out = sv_bool(0); return; }
    sv_t list = sv_null();
    char* record = buffer;
    int ok = 1;
    for (int32_t i = 0; i < count; ++i) {
        size_t record_length = strlen(record);
        builtin_ts_node_t child;
        if (!parse_node_record(record, &child)) { ok = 0; break; }
        child.tree = node.tree;
        int64_t child_handle = store_node(&child);
        if (child_handle <= 0) { ok = 0; break; }
        list = sv_pair(sv_int(child_handle), list);
        record += record_length + 1u;
    }
    free(buffer);
    *out = ok ? sv_reverse(list) : sv_bool(0);
}

ESHKOL_AGENT_API void eshkol_builtin_ts_query_new(sv_t* out, const sv_t* language_value,
                                 const sv_t* pattern_value) {
    const char* language = NULL;
    const char* pattern = NULL;
    int32_t language_len = 0;
    int32_t pattern_len = 0;
    if (!out) return;
    if (!sv_string(language_value, &language, &language_len) ||
        !sv_string(pattern_value, &pattern, &pattern_len)) {
        *out = sv_bool(0);
        return;
    }
    char* language_copy = (char*)malloc((size_t)language_len + 1u);
    char* pattern_copy = (char*)malloc((size_t)pattern_len + 1u);
    if (!language_copy || !pattern_copy) {
        free(language_copy); free(pattern_copy); *out = sv_bool(0); return;
    }
    memcpy(language_copy, language, (size_t)language_len); language_copy[language_len] = '\0';
    memcpy(pattern_copy, pattern, (size_t)pattern_len); pattern_copy[pattern_len] = '\0';
    int64_t query = eshkol_ts_query_new(language_copy, pattern_copy);
    free(language_copy); free(pattern_copy);
    if (query <= 0 || query >= (int64_t)(sizeof(g_builtin_ts_queries) / sizeof(g_builtin_ts_queries[0]))) {
        *out = sv_bool(0); return;
    }
    eshkol_agent_mutex_lock(&g_builtin_ts_mutex);
    g_builtin_ts_queries[query] = 1;
    eshkol_agent_mutex_unlock(&g_builtin_ts_mutex);
    *out = sv_int(query);
}

ESHKOL_AGENT_API void eshkol_builtin_ts_query_free(sv_t* out, const sv_t* query_value) {
    int64_t query = 0;
    int valid = sv_integer(query_value, &query);
    eshkol_agent_mutex_lock(&g_builtin_ts_mutex);
    valid = valid && query > 0 && query < (int64_t)(sizeof(g_builtin_ts_queries) / sizeof(g_builtin_ts_queries[0])) &&
            g_builtin_ts_queries[query];
    if (valid) g_builtin_ts_queries[query] = 0;
    eshkol_agent_mutex_unlock(&g_builtin_ts_mutex);
    if (valid) eshkol_ts_query_free(query);
    if (out) *out = sv_bool(valid);
}

static sv_t query_match_value(int64_t tree, char* record) {
    char* fields[7];
    if (split_tabs(record, fields, 7) != 7) return sv_bool(0);
    char* end = NULL;
    unsigned long start = strtoul(fields[1], &end, 10);
    if (!end || *end) return sv_bool(0);
    unsigned long finish = strtoul(fields[2], &end, 10);
    if (!end || *end || start > UINT32_MAX || finish > UINT32_MAX || start >= finish)
        return sv_bool(0);

    int32_t info_len = eshkol_ts_node_info(tree, (uint32_t)start,
                                           (uint32_t)finish, NULL, 0);
    int32_t text_len = eshkol_ts_node_text(tree, (uint32_t)start,
                                           (uint32_t)finish, NULL, 0);
    if (info_len < 0 || text_len < 0) return sv_bool(0);
    char* info = (char*)malloc((size_t)info_len + 1u);
    char* text = (char*)malloc((size_t)text_len + 1u);
    if (!info || !text) { free(info); free(text); return sv_bool(0); }
    if (eshkol_ts_node_info(tree, (uint32_t)start, (uint32_t)finish,
                            info, info_len + 1) != info_len ||
        eshkol_ts_node_text(tree, (uint32_t)start, (uint32_t)finish,
                            text, text_len + 1) != text_len) {
        free(info); free(text); return sv_bool(0);
    }
    builtin_ts_node_t node;
    if (!parse_node_record(info, &node)) {
        free(info); free(text); return sv_bool(0);
    }
    sv_t match = sv_null();
    match = sv_pair(sv_alist_entry("text", sv_string_copy(text, (size_t)text_len)), match);
    match = sv_pair(sv_alist_entry("end", sv_int((int64_t)finish)), match);
    match = sv_pair(sv_alist_entry("start", sv_int((int64_t)start)), match);
    match = sv_pair(sv_alist_entry("type", sv_string_copy(node.type, strlen(node.type))), match);
    match = sv_pair(sv_alist_entry("capture", sv_string_copy(fields[0], strlen(fields[0]))), match);
    free(info); free(text);
    return match;
}

ESHKOL_AGENT_API void eshkol_builtin_ts_query_matches(sv_t* out, const sv_t* query_value,
                                     const sv_t* tree_value,
                                     const sv_t* unused_source) {
    (void)unused_source;
    int64_t query = 0;
    int64_t tree = 0;
    int valid = sv_integer(query_value, &query) && sv_integer(tree_value, &tree);
    eshkol_agent_mutex_lock(&g_builtin_ts_mutex);
    valid = valid && query > 0 && query < (int64_t)(sizeof(g_builtin_ts_queries) / sizeof(g_builtin_ts_queries[0])) &&
            tree > 0 && tree < (int64_t)(sizeof(g_builtin_ts_trees) / sizeof(g_builtin_ts_trees[0])) &&
            g_builtin_ts_queries[query] && g_builtin_ts_trees[tree];
    eshkol_agent_mutex_unlock(&g_builtin_ts_mutex);
    if (!out) return;
    if (!valid) { *out = sv_bool(0); return; }
    int32_t count = 0;
    int32_t required = eshkol_ts_query_matches(query, tree, 0, NULL, 0, &count);
    if (required < 0) { *out = sv_bool(0); return; }
    if (count == 0) { *out = sv_null(); return; }
    char* buffer = (char*)malloc((size_t)required + 1u);
    if (!buffer) { *out = sv_bool(0); return; }
    int32_t written = eshkol_ts_query_matches(query, tree, 0, buffer,
                                              required + 1, &count);
    if (written != required) { free(buffer); *out = sv_bool(0); return; }
    sv_t list = sv_null();
    char* record = buffer;
    int ok = 1;
    for (int32_t i = 0; i < count; ++i) {
        size_t record_length = strlen(record);
        sv_t match = query_match_value(tree, record);
        if ((match.type & 0x0f) == ESHKOL_VALUE_BOOL && match.data.int_val == 0) {
            ok = 0; break;
        }
        list = sv_pair(match, list);
        record += record_length + 1u;
    }
    free(buffer);
    *out = ok ? sv_reverse(list) : sv_bool(0);
}

ESHKOL_AGENT_API void eshkol_builtin_ts_available(sv_t* out) {
    if (out) *out = sv_bool(eshkol_ts_available() == 1);
}

typedef struct {
    int active;
    int parent;
    int child_count;
} builtin_yoga_node_t;

static builtin_yoga_node_t g_builtin_yoga_nodes[512];
static eshkol_agent_mutex_t g_builtin_yoga_mutex = ESHKOL_AGENT_MUTEX_INITIALIZER;

static int yoga_valid(int64_t handle) {
    return handle > 0 && handle < (int64_t)(sizeof(g_builtin_yoga_nodes) / sizeof(g_builtin_yoga_nodes[0])) &&
           g_builtin_yoga_nodes[handle].active;
}

static int string_equals(const sv_t* value, const char* expected) {
    const char* data = NULL;
    int32_t len = 0;
    size_t expected_len = strlen(expected);
    return expected_len <= INT32_MAX && sv_string(value, &data, &len) &&
           len == (int32_t)expected_len && memcmp(data, expected, expected_len) == 0;
}

static int yoga_enum_value(const sv_t* value, const char* const* names,
                           int count, int* out) {
    for (int i = 0; i < count; ++i) {
        if (string_equals(value, names[i])) { *out = i; return 1; }
    }
    return 0;
}

static void yoga_clear_subtree(int handle) {
    if (!yoga_valid(handle)) return;
    for (int i = 1; i < (int)(sizeof(g_builtin_yoga_nodes) / sizeof(g_builtin_yoga_nodes[0])); ++i) {
        if (g_builtin_yoga_nodes[i].active && g_builtin_yoga_nodes[i].parent == handle)
            yoga_clear_subtree(i);
    }
    memset(&g_builtin_yoga_nodes[handle], 0, sizeof(g_builtin_yoga_nodes[handle]));
}

ESHKOL_AGENT_API void eshkol_builtin_yoga_node_create(sv_t* out) {
    if (!out) return;
    int64_t handle = eshkol_yoga_node_create();
    if (handle <= 0 || handle >= (int64_t)(sizeof(g_builtin_yoga_nodes) / sizeof(g_builtin_yoga_nodes[0]))) {
        *out = sv_bool(0); return;
    }
    eshkol_agent_mutex_lock(&g_builtin_yoga_mutex);
    memset(&g_builtin_yoga_nodes[handle], 0, sizeof(g_builtin_yoga_nodes[handle]));
    g_builtin_yoga_nodes[handle].active = 1;
    eshkol_agent_mutex_unlock(&g_builtin_yoga_mutex);
    *out = sv_int(handle);
}

ESHKOL_AGENT_API void eshkol_builtin_yoga_node_set(sv_t* out, const sv_t* node_value,
                                  const sv_t* property,
                                  const sv_t* value) {
    int64_t handle = 0;
    double number = 0.0;
    int ok = sv_integer(node_value, &handle);
    eshkol_agent_mutex_lock(&g_builtin_yoga_mutex);
    ok = ok && yoga_valid(handle);
    eshkol_agent_mutex_unlock(&g_builtin_yoga_mutex);
    if (!out) return;
    if (!ok) { *out = sv_bool(0); return; }

    int float_prop = -1;
    int all_start = -1;
    if (string_equals(property, "width")) float_prop = 0;
    else if (string_equals(property, "height")) float_prop = 1;
    else if (string_equals(property, "min-width")) float_prop = 2;
    else if (string_equals(property, "min-height")) float_prop = 3;
    else if (string_equals(property, "max-width")) float_prop = 4;
    else if (string_equals(property, "max-height")) float_prop = 5;
    else if (string_equals(property, "flex-grow")) float_prop = 6;
    else if (string_equals(property, "flex-shrink")) float_prop = 7;
    else if (string_equals(property, "flex-basis")) float_prop = 8;
    else if (string_equals(property, "gap")) float_prop = 9;
    else if (string_equals(property, "padding")) all_start = 10;
    else if (string_equals(property, "padding-left")) float_prop = 10;
    else if (string_equals(property, "padding-right")) float_prop = 11;
    else if (string_equals(property, "padding-top")) float_prop = 12;
    else if (string_equals(property, "padding-bottom")) float_prop = 13;
    else if (string_equals(property, "margin")) all_start = 14;
    else if (string_equals(property, "margin-left")) float_prop = 14;
    else if (string_equals(property, "margin-right")) float_prop = 15;
    else if (string_equals(property, "margin-top")) float_prop = 16;
    else if (string_equals(property, "margin-bottom")) float_prop = 17;
    else if (string_equals(property, "border")) all_start = 18;
    else if (string_equals(property, "border-left")) float_prop = 18;
    else if (string_equals(property, "border-right")) float_prop = 19;
    else if (string_equals(property, "border-top")) float_prop = 20;
    else if (string_equals(property, "border-bottom")) float_prop = 21;
    if (float_prop >= 0 || all_start >= 0) {
        if (!sv_number(value, &number) || !isfinite(number)) { *out = sv_bool(0); return; }
        if (all_start >= 0) {
            for (int i = 0; i < 4; ++i)
                eshkol_yoga_node_set_float(handle, all_start + i, number);
        } else {
            eshkol_yoga_node_set_float(handle, float_prop, number);
        }
        *out = sv_bool(1);
        return;
    }

    static const char* const flex_direction[] = {"column", "column-reverse", "row", "row-reverse"};
    static const char* const justify[] = {"flex-start", "center", "flex-end", "space-between", "space-around", "space-evenly"};
    static const char* const align[] = {"auto", "flex-start", "center", "flex-end", "stretch", "baseline", "space-between", "space-around", "space-evenly"};
    static const char* const position[] = {"static", "relative", "absolute"};
    static const char* const overflow[] = {"visible", "hidden", "scroll"};
    static const char* const display[] = {"flex", "none", "contents"};
    int int_prop = -1;
    int enum_value = -1;
    if (string_equals(property, "flex-direction")) {
        int_prop = 0; ok = yoga_enum_value(value, flex_direction, 4, &enum_value);
    } else if (string_equals(property, "justify-content")) {
        int_prop = 1; ok = yoga_enum_value(value, justify, 6, &enum_value);
    } else if (string_equals(property, "align-items")) {
        int_prop = 2; ok = yoga_enum_value(value, align, 9, &enum_value);
    } else if (string_equals(property, "align-self")) {
        int_prop = 3; ok = yoga_enum_value(value, align, 9, &enum_value);
    } else if (string_equals(property, "align-content")) {
        int_prop = 4; ok = yoga_enum_value(value, align, 9, &enum_value);
    } else if (string_equals(property, "position-type")) {
        int_prop = 5; ok = yoga_enum_value(value, position, 3, &enum_value);
    } else if (string_equals(property, "overflow")) {
        int_prop = 6; ok = yoga_enum_value(value, overflow, 3, &enum_value);
    } else if (string_equals(property, "display")) {
        int_prop = 7; ok = yoga_enum_value(value, display, 3, &enum_value);
    } else {
        ok = 0;
    }
    if (!ok) { *out = sv_bool(0); return; }
    eshkol_yoga_node_set_int(handle, int_prop, enum_value);
    *out = sv_bool(1);
}

ESHKOL_AGENT_API void eshkol_builtin_yoga_node_add_child(sv_t* out, const sv_t* parent_value,
                                        const sv_t* child_value) {
    int64_t parent = 0;
    int64_t child = 0;
    int ok = sv_integer(parent_value, &parent) && sv_integer(child_value, &child);
    int index = 0;
    eshkol_agent_mutex_lock(&g_builtin_yoga_mutex);
    ok = ok && yoga_valid(parent) && yoga_valid(child) && parent != child &&
         g_builtin_yoga_nodes[child].parent == 0 &&
         g_builtin_yoga_nodes[parent].child_count < 512;
    if (ok) {
        for (int ancestor = (int)parent; ancestor != 0;
             ancestor = g_builtin_yoga_nodes[ancestor].parent) {
            if (ancestor == child) { ok = 0; break; }
        }
    }
    if (ok) {
        index = g_builtin_yoga_nodes[parent].child_count;
        g_builtin_yoga_nodes[parent].child_count++;
        g_builtin_yoga_nodes[child].parent = (int)parent;
    }
    eshkol_agent_mutex_unlock(&g_builtin_yoga_mutex);
    if (ok) eshkol_yoga_node_add_child(parent, child, index);
    if (out) *out = sv_bool(ok);
}

ESHKOL_AGENT_API void eshkol_builtin_yoga_node_calculate(sv_t* out, const sv_t* root_value,
                                        const sv_t* width_value,
                                        const sv_t* height_value) {
    int64_t root = 0;
    double width = 0.0;
    double height = 0.0;
    int ok = sv_integer(root_value, &root) && sv_number(width_value, &width) &&
             sv_number(height_value, &height) && isfinite(width) && isfinite(height) &&
             width >= 0.0 && height >= 0.0;
    eshkol_agent_mutex_lock(&g_builtin_yoga_mutex);
    ok = ok && yoga_valid(root);
    eshkol_agent_mutex_unlock(&g_builtin_yoga_mutex);
    if (ok) eshkol_yoga_node_calculate(root, width, height);
    if (out) *out = sv_bool(ok);
}

ESHKOL_AGENT_API void eshkol_builtin_yoga_node_get_computed(sv_t* out, const sv_t* node_value,
                                           const sv_t* property) {
    int64_t handle = 0;
    int prop = -1;
    int ok = sv_integer(node_value, &handle);
    if (string_equals(property, "left")) prop = 0;
    else if (string_equals(property, "top")) prop = 1;
    else if (string_equals(property, "width")) prop = 2;
    else if (string_equals(property, "height")) prop = 3;
    else if (string_equals(property, "padding-left")) prop = 4;
    else if (string_equals(property, "padding-top")) prop = 5;
    else if (string_equals(property, "padding-right")) prop = 6;
    else if (string_equals(property, "padding-bottom")) prop = 7;
    else if (string_equals(property, "margin-left")) prop = 8;
    else if (string_equals(property, "margin-top")) prop = 9;
    else if (string_equals(property, "margin-right")) prop = 10;
    else if (string_equals(property, "margin-bottom")) prop = 11;
    else if (string_equals(property, "border-left")) prop = 12;
    else if (string_equals(property, "border-top")) prop = 13;
    else if (string_equals(property, "border-right")) prop = 14;
    else if (string_equals(property, "border-bottom")) prop = 15;
    eshkol_agent_mutex_lock(&g_builtin_yoga_mutex);
    ok = ok && yoga_valid(handle) && prop >= 0;
    eshkol_agent_mutex_unlock(&g_builtin_yoga_mutex);
    if (out) *out = ok ? sv_double(eshkol_yoga_node_get_computed(handle, prop)) : sv_bool(0);
}

ESHKOL_AGENT_API void eshkol_builtin_yoga_node_free(sv_t* out, const sv_t* node_value) {
    int64_t handle = 0;
    int ok = sv_integer(node_value, &handle);
    eshkol_agent_mutex_lock(&g_builtin_yoga_mutex);
    ok = ok && yoga_valid(handle);
    if (ok) {
        int parent = g_builtin_yoga_nodes[handle].parent;
        if (yoga_valid(parent) && g_builtin_yoga_nodes[parent].child_count > 0)
            g_builtin_yoga_nodes[parent].child_count--;
        yoga_clear_subtree((int)handle);
    }
    eshkol_agent_mutex_unlock(&g_builtin_yoga_mutex);
    if (ok) eshkol_yoga_node_free(handle);
    if (out) *out = sv_bool(ok);
}
