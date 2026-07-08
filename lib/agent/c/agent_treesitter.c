/*******************************************************************************
 * Tree-sitter Integration for Eshkol Agent
 *
 * Provides structural code intelligence: parsing, AST traversal, and
 * S-expression pattern queries across 10+ programming languages.
 *
 * Enables: go-to-definition, find-references, structural rename, code
 * navigation, dead code detection — all WITHOUT external LSP servers.
 *
 * Compile with -DHAS_TREE_SITTER and link against libtree-sitter plus
 * language grammar libraries (tree-sitter-javascript, etc.).
 *
 * Without -DHAS_TREE_SITTER, all functions return graceful errors (-1 or NULL)
 * so the agent degrades to regex-based search without crashing.
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

/*******************************************************************************
 * CONDITIONAL COMPILATION: Full implementation vs graceful stubs
 ******************************************************************************/

#ifdef HAS_TREE_SITTER

#include <tree_sitter/api.h>

/*******************************************************************************
 * Language Registry
 ******************************************************************************/

/* Language grammar entry points — linked from separate .a/.so files */
extern const TSLanguage *tree_sitter_javascript(void);
extern const TSLanguage *tree_sitter_typescript(void);
extern const TSLanguage *tree_sitter_python(void);
extern const TSLanguage *tree_sitter_rust(void);
extern const TSLanguage *tree_sitter_go(void);
extern const TSLanguage *tree_sitter_c(void);
extern const TSLanguage *tree_sitter_cpp(void);
extern const TSLanguage *tree_sitter_java(void);
extern const TSLanguage *tree_sitter_ruby(void);
extern const TSLanguage *tree_sitter_bash(void);

typedef struct {
    const char* name;
    const TSLanguage* (*factory)(void);
} LangEntry;

static const LangEntry g_languages[] = {
    { "javascript",  tree_sitter_javascript },
    { "js",          tree_sitter_javascript },
    { "typescript",  tree_sitter_typescript },
    { "ts",          tree_sitter_typescript },
    { "python",      tree_sitter_python },
    { "py",          tree_sitter_python },
    { "rust",        tree_sitter_rust },
    { "rs",          tree_sitter_rust },
    { "go",          tree_sitter_go },
    { "c",           tree_sitter_c },
    { "cpp",         tree_sitter_cpp },
    { "c++",         tree_sitter_cpp },
    { "java",        tree_sitter_java },
    { "ruby",        tree_sitter_ruby },
    { "rb",          tree_sitter_ruby },
    { "bash",        tree_sitter_bash },
    { "sh",          tree_sitter_bash },
    { NULL, NULL }
};

/**
 * @brief Looks up the tree-sitter grammar factory registered under @p name in g_languages.
 *
 * @param name Language name or short alias (e.g. "javascript"/"js", "python"/"py").
 * @return The matching TSLanguage, or NULL if @p name is not registered.
 */
static const TSLanguage* find_language(const char* name) {
    for (const LangEntry* e = g_languages; e->name; e++) {
        if (strcmp(e->name, name) == 0) return e->factory();
    }
    return NULL;
}

/*******************************************************************************
 * Handle Tables
 ******************************************************************************/

#define MAX_PARSERS 32
#define MAX_TREES   64
#define MAX_QUERIES 32

static TSParser* g_parsers[MAX_PARSERS] = {0};
static TSTree*   g_trees[MAX_TREES] = {0};
static const char* g_tree_sources[MAX_TREES] = {0};  /* source text per tree */
static uint32_t g_tree_source_lens[MAX_TREES] = {0};
static TSQuery*  g_queries[MAX_QUERIES] = {0};
static const TSLanguage* g_query_langs[MAX_QUERIES] = {0};

/**
 * @brief Finds the first NULL (free) entry in a fixed-size handle table, skipping index 0 so handle 0 is never valid.
 *
 * @param table Array of @p max opaque pointers, one per handle slot.
 * @param max Number of entries in @p table.
 * @return Index of a free slot (>= 1), or -1 if the table is full.
 */
static int alloc_slot(void** table, int max) {
    for (int i = 1; i < max; i++) {
        if (!table[i]) return i;
    }
    return -1;
}

/*******************************************************************************
 * Parser Operations
 ******************************************************************************/

/*
 * Create a new parser for the given language.
 *
 * language: "javascript", "python", "rust", "go", "c", "cpp", "java",
 *           "ruby", "bash", "typescript" (and short aliases "js", "py", etc.)
 *
 * Returns: parser handle (>= 1), -1 if language unknown or slots full
 */
/**
 * @brief Creates a tree-sitter parser configured for @p language and stores it in the parser handle table.
 *
 * @param language Language name or short alias, as accepted by find_language() (e.g. "javascript", "js", "python", "rust", "go", "c", "cpp", "java", "ruby", "bash").
 * @return Parser handle (>= 1) on success, -1 if @p language is unknown, the parser slot table is full, or allocation fails.
 */
int64_t eshkol_ts_parser_new(const char* language) {
    if (!language) return -1;
    const TSLanguage* lang = find_language(language);
    if (!lang) return -1;

    TSParser* parser = ts_parser_new();
    if (!parser) return -1;
    if (!ts_parser_set_language(parser, lang)) {
        ts_parser_delete(parser);
        return -1;
    }

    int slot = alloc_slot((void**)g_parsers, MAX_PARSERS);
    if (slot < 0) { ts_parser_delete(parser); return -1; }
    g_parsers[slot] = parser;
    return (int64_t)slot;
}

/*
 * Free a parser and its slot.
 */
/**
 * @brief Deletes the parser at @p handle and clears its slot.
 *
 * @param handle Parser handle from eshkol_ts_parser_new(). No-op if out of range or already freed.
 */
void eshkol_ts_parser_free(int64_t handle) {
    if (handle < 1 || handle >= MAX_PARSERS) return;
    if (g_parsers[handle]) {
        ts_parser_delete(g_parsers[handle]);
        g_parsers[handle] = NULL;
    }
}

/*******************************************************************************
 * Parsing
 ******************************************************************************/

/*
 * Parse source code into a syntax tree.
 *
 * parser_handle: from eshkol_ts_parser_new
 * source: source code string (need not be null-terminated, length is explicit)
 * source_len: byte length of source
 *
 * Returns: tree handle (>= 1), -1 on error
 *
 * The source string is NOT copied — caller must keep it alive while the
 * tree is in use. The handle stores a reference to it.
 */
/**
 * @brief Parses @p source with the parser at @p parser_handle and stores the resulting syntax tree in the tree handle table.
 *
 * @param parser_handle Parser handle from eshkol_ts_parser_new().
 * @param source Source code bytes; need not be NUL-terminated. Not copied — must remain valid for the lifetime of the returned tree handle, since node-text extraction reads directly from it.
 * @param source_len Length of @p source in bytes.
 * @return Tree handle (>= 1) on success, -1 on invalid handle/arguments, parse failure, or if the tree slot table is full.
 */
int64_t eshkol_ts_parse(int64_t parser_handle, const char* source, int32_t source_len) {
    if (parser_handle < 1 || parser_handle >= MAX_PARSERS) return -1;
    TSParser* parser = g_parsers[parser_handle];
    if (!parser || !source || source_len <= 0) return -1;

    TSTree* tree = ts_parser_parse_string(parser, NULL, source, (uint32_t)source_len);
    if (!tree) return -1;

    int slot = alloc_slot((void**)g_trees, MAX_TREES);
    if (slot < 0) { ts_tree_delete(tree); return -1; }
    g_trees[slot] = tree;
    g_tree_sources[slot] = source;
    g_tree_source_lens[slot] = (uint32_t)source_len;
    return (int64_t)slot;
}

/*
 * Free a tree and its slot.
 */
/**
 * @brief Deletes the tree at @p handle and clears its slot, including the stored source-text reference.
 *
 * @param handle Tree handle from eshkol_ts_parse(). No-op if out of range or already freed.
 */
void eshkol_ts_tree_free(int64_t handle) {
    if (handle < 1 || handle >= MAX_TREES) return;
    if (g_trees[handle]) {
        ts_tree_delete(g_trees[handle]);
        g_trees[handle] = NULL;
        g_tree_sources[handle] = NULL;
        g_tree_source_lens[handle] = 0;
    }
}

/*******************************************************************************
 * Node Serialization Helpers
 *
 * Nodes are serialized as tab-separated fields in a buffer:
 *   "type\tstart_row\tstart_col\tend_row\tend_col\tstart_byte\tend_byte\tchild_count\tnamed"
 ******************************************************************************/

/**
 * @brief Serializes a single TSNode's type, position, byte range, and child count into a tab-separated line.
 *
 * Output format: "type\tstart_row\tstart_col\tend_row\tend_col\tstart_byte\tend_byte\tchild_count\tnamed".
 *
 * @param node Node to serialize.
 * @param buf Destination buffer.
 * @param buf_size Size of @p buf.
 * @return Value from snprintf(): the number of characters that would have been written (excluding the NUL), which may exceed @p buf_size if truncated.
 */
static int serialize_node(TSNode node, char* buf, int buf_size) {
    const char* type = ts_node_type(node);
    TSPoint start = ts_node_start_point(node);
    TSPoint end = ts_node_end_point(node);
    uint32_t start_byte = ts_node_start_byte(node);
    uint32_t end_byte = ts_node_end_byte(node);
    uint32_t child_count = ts_node_child_count(node);
    int named = ts_node_is_named(node) ? 1 : 0;

    return snprintf(buf, (size_t)buf_size,
                    "%s\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%d",
                    type ? type : "?",
                    start.row, start.column,
                    end.row, end.column,
                    start_byte, end_byte,
                    child_count, named);
}

/*******************************************************************************
 * Tree / Node Operations
 ******************************************************************************/

/*
 * Get root node info for a tree.
 *
 * Returns: strlen written to buf, -1 error
 */
/**
 * @brief Serializes the root node of tree @p tree_handle into @p buf via serialize_node().
 *
 * @param tree_handle Tree handle from eshkol_ts_parse().
 * @param buf Destination buffer for the serialized node.
 * @param buf_size Size of @p buf.
 * @return Number of bytes written to @p buf, or -1 on invalid handle/arguments or if the serialized node would not fit.
 */
int32_t eshkol_ts_tree_root(int64_t tree_handle, char* buf, int32_t buf_size) {
    if (tree_handle < 1 || tree_handle >= MAX_TREES) return -1;
    TSTree* tree = g_trees[tree_handle];
    if (!tree || !buf || buf_size <= 0) return -1;

    TSNode root = ts_tree_root_node(tree);
    int n = serialize_node(root, buf, buf_size);
    return (n >= 0 && n < buf_size) ? n : -1;
}

/*
 * Get all named children of the root, or of a node located by byte range.
 *
 * If start_byte == 0 && end_byte == 0, returns children of root.
 * Otherwise, finds the smallest named node containing [start_byte, end_byte)
 * and returns its children.
 *
 * Children are null-separated serialized nodes in buf.
 * count_out receives the number of children.
 *
 * Returns: total bytes written to buf, -1 error
 */
/**
 * @brief Serializes the named children of the root node, or of the smallest named node containing [@p start_byte, @p end_byte), as NUL-separated records in @p buf.
 *
 * When @p start_byte and @p end_byte are both 0, the children of the tree
 * root are returned. Otherwise the smallest named descendant covering that
 * byte range is located first (via
 * ts_node_named_descendant_for_byte_range()) and its named children are
 * returned instead.
 *
 * @param tree_handle Tree handle from eshkol_ts_parse().
 * @param start_byte Start of the byte range identifying the target node, or 0 with @p end_byte 0 for the root.
 * @param end_byte End of the byte range identifying the target node, or 0 with @p start_byte 0 for the root.
 * @param buf Destination buffer for NUL-separated serialized child nodes.
 * @param buf_size Size of @p buf.
 * @param count_out Output: number of children written.
 * @return Total bytes written to @p buf, or -1 on invalid handle/arguments.
 */
int32_t eshkol_ts_node_children(int64_t tree_handle,
                                  uint32_t start_byte, uint32_t end_byte,
                                  char* buf, int32_t buf_size, int32_t* count_out) {
    if (tree_handle < 1 || tree_handle >= MAX_TREES) return -1;
    TSTree* tree = g_trees[tree_handle];
    if (!tree || !buf || buf_size <= 0) return -1;

    TSNode target;
    if (start_byte == 0 && end_byte == 0) {
        target = ts_tree_root_node(tree);
    } else {
        target = ts_tree_root_node(tree);
        /* Descend to the named node containing this byte range */
        TSNode candidate = ts_node_named_descendant_for_byte_range(target, start_byte, end_byte);
        if (!ts_node_is_null(candidate)) target = candidate;
    }

    uint32_t n_children = ts_node_named_child_count(target);
    int32_t written = 0;
    int32_t count = 0;

    for (uint32_t i = 0; i < n_children && written < buf_size - 1; i++) {
        TSNode child = ts_node_named_child(target, i);
        if (ts_node_is_null(child)) continue;

        if (count > 0) {
            buf[written++] = '\0';  /* null separator */
        }

        int n = serialize_node(child, buf + written, buf_size - written);
        if (n < 0 || written + n >= buf_size) break;
        written += n;
        count++;
    }

    if (written < buf_size) buf[written] = '\0';
    if (count_out) *count_out = count;
    return written;
}

/*
 * Extract source text for a byte range from a parsed tree's source.
 *
 * Returns: strlen of extracted text, -1 error
 */
/**
 * @brief Copies the source text spanning [@p start_byte, @p end_byte) of tree @p tree_handle's original source into @p buf.
 *
 * @param tree_handle Tree handle from eshkol_ts_parse(); its stored source pointer/length are used.
 * @param start_byte Start byte offset (inclusive).
 * @param end_byte End byte offset (exclusive); must be > @p start_byte and within the source length.
 * @param buf Destination buffer, truncated to @p buf_size - 1 bytes if the range is longer.
 * @param buf_size Size of @p buf.
 * @return Number of bytes written to @p buf, or -1 on invalid handle/arguments or an out-of-range/empty byte range.
 */
int32_t eshkol_ts_node_text(int64_t tree_handle, uint32_t start_byte,
                              uint32_t end_byte, char* buf, int32_t buf_size) {
    if (tree_handle < 1 || tree_handle >= MAX_TREES) return -1;
    const char* src = g_tree_sources[tree_handle];
    uint32_t src_len = g_tree_source_lens[tree_handle];
    if (!src || !buf || buf_size <= 0) return -1;

    if (start_byte >= src_len || end_byte > src_len || start_byte >= end_byte) return -1;
    uint32_t text_len = end_byte - start_byte;
    if ((int32_t)text_len >= buf_size) text_len = (uint32_t)(buf_size - 1);

    memcpy(buf, src + start_byte, text_len);
    buf[text_len] = '\0';
    return (int32_t)text_len;
}

/*******************************************************************************
 * S-Expression Query
 ******************************************************************************/

/*
 * Create a tree-sitter query from an S-expression pattern.
 *
 * language: language name (must match parser language)
 * pattern: tree-sitter query pattern, e.g.:
 *   "(function_declaration name: (identifier) @name)"
 *   "(call_expression function: (identifier) @func)"
 *
 * Returns: query handle (>= 1), -1 on error
 */
/**
 * @brief Compiles a tree-sitter S-expression @p pattern for @p language and stores it in the query handle table.
 *
 * @param language Language name or alias the pattern is written against; must match the language used to parse any tree the query is later run on.
 * @param pattern Tree-sitter query pattern, e.g. "(function_declaration name: (identifier) @name)".
 * @return Query handle (>= 1) on success, -1 if @p language is unknown, @p pattern fails to compile, or the query slot table is full.
 */
int64_t eshkol_ts_query_new(const char* language, const char* pattern) {
    if (!language || !pattern) return -1;
    const TSLanguage* lang = find_language(language);
    if (!lang) return -1;

    uint32_t error_offset;
    TSQueryError error_type;
    TSQuery* query = ts_query_new(lang, pattern, (uint32_t)strlen(pattern),
                                   &error_offset, &error_type);
    if (!query) return -1;

    int slot = alloc_slot((void**)g_queries, MAX_QUERIES);
    if (slot < 0) { ts_query_delete(query); return -1; }
    g_queries[slot] = query;
    g_query_langs[slot] = lang;
    return (int64_t)slot;
}

/*
 * Free a query.
 */
/**
 * @brief Deletes the query at @p handle and clears its slot.
 *
 * @param handle Query handle from eshkol_ts_query_new(). No-op if out of range or already freed.
 */
void eshkol_ts_query_free(int64_t handle) {
    if (handle < 1 || handle >= MAX_QUERIES) return;
    if (g_queries[handle]) {
        ts_query_delete(g_queries[handle]);
        g_queries[handle] = NULL;
        g_query_langs[handle] = NULL;
    }
}

/*
 * Execute a query against a parsed tree and return matches.
 *
 * Each match is serialized as:
 *   "capture_name\tstart_byte\tend_byte\tstart_row\tstart_col\tend_row\tend_col"
 * Matches are null-separated in buf.
 *
 * max_matches: limit results (0 = unlimited, up to buf capacity)
 * count_out: receives number of captures written
 *
 * Returns: total bytes written to buf, -1 error
 */
/**
 * @brief Runs query @p query_handle against tree @p tree_handle and serializes each capture as a NUL-separated record in @p buf.
 *
 * Each capture is written as
 * "capture_name\tstart_byte\tend_byte\tstart_row\tstart_col\tend_row\tend_col".
 * Iterates matches via a fresh TSQueryCursor until either @p max_matches
 * matches have been processed or @p buf is full.
 *
 * @param query_handle Query handle from eshkol_ts_query_new().
 * @param tree_handle Tree handle from eshkol_ts_parse(); must have been parsed with the same language as the query.
 * @param max_matches Maximum number of matches to process, or 0 for effectively unlimited (bounded only by @p buf capacity).
 * @param buf Destination buffer for NUL-separated capture records.
 * @param buf_size Size of @p buf.
 * @param count_out Output: number of captures written.
 * @return Total bytes written to @p buf, or -1 on invalid handle/arguments.
 */
int32_t eshkol_ts_query_matches(int64_t query_handle, int64_t tree_handle,
                                  int32_t max_matches,
                                  char* buf, int32_t buf_size,
                                  int32_t* count_out) {
    if (query_handle < 1 || query_handle >= MAX_QUERIES) return -1;
    if (tree_handle < 1 || tree_handle >= MAX_TREES) return -1;
    TSQuery* query = g_queries[query_handle];
    TSTree* tree = g_trees[tree_handle];
    if (!query || !tree || !buf || buf_size <= 0) return -1;

    TSNode root = ts_tree_root_node(tree);
    TSQueryCursor* cursor = ts_query_cursor_new();
    ts_query_cursor_exec(cursor, query, root);

    int32_t written = 0;
    int32_t count = 0;
    int32_t limit = max_matches > 0 ? max_matches : 100000;

    TSQueryMatch match;
    while (count < limit && ts_query_cursor_next_match(cursor, &match)) {
        for (uint16_t i = 0; i < match.capture_count; i++) {
            TSQueryCapture capture = match.captures[i];
            TSNode node = capture.node;
            uint32_t name_len;
            const char* name = ts_query_capture_name_for_id(query, capture.index, &name_len);

            TSPoint start = ts_node_start_point(node);
            TSPoint end = ts_node_end_point(node);
            uint32_t sb = ts_node_start_byte(node);
            uint32_t eb = ts_node_end_byte(node);

            if (count > 0 && written < buf_size - 1) {
                buf[written++] = '\0';
            }

            int n = snprintf(buf + written, (size_t)(buf_size - written),
                             "%.*s\t%u\t%u\t%u\t%u\t%u\t%u",
                             (int)name_len, name,
                             sb, eb,
                             start.row, start.column,
                             end.row, end.column);
            if (n < 0 || written + n >= buf_size) goto done;
            written += n;
            count++;
        }
    }

done:
    ts_query_cursor_delete(cursor);
    if (written < buf_size) buf[written] = '\0';
    if (count_out) *count_out = count;
    return written;
}

/*
 * Get the S-expression representation of a tree (for debugging).
 *
 * Returns: strlen, -1 error
 */
/**
 * @brief Renders the S-expression form of tree @p tree_handle's root node (via ts_node_string()) into @p buf, for debugging.
 *
 * @param tree_handle Tree handle from eshkol_ts_parse().
 * @param buf Destination buffer, truncated to @p buf_size - 1 bytes if the S-expression is longer.
 * @param buf_size Size of @p buf.
 * @return Number of bytes written to @p buf, or -1 on invalid handle/arguments or allocation failure.
 */
int32_t eshkol_ts_tree_sexp(int64_t tree_handle, char* buf, int32_t buf_size) {
    if (tree_handle < 1 || tree_handle >= MAX_TREES) return -1;
    TSTree* tree = g_trees[tree_handle];
    if (!tree || !buf || buf_size <= 0) return -1;

    TSNode root = ts_tree_root_node(tree);
    char* sexp = ts_node_string(root);
    if (!sexp) return -1;

    int len = (int32_t)strlen(sexp);
    if (len >= buf_size) len = buf_size - 1;
    memcpy(buf, sexp, (size_t)len);
    buf[len] = '\0';
    free(sexp);
    return len;
}

/*
 * Check if tree-sitter is available.
 * Returns: 1 (compiled with tree-sitter support)
 */
/**
 * @brief Reports whether this build was compiled with tree-sitter support.
 *
 * @return 1 when compiled with HAS_TREE_SITTER (this definition).
 */
int32_t eshkol_ts_available(void) { return 1; }

#else /* !HAS_TREE_SITTER */

/*******************************************************************************
 * Graceful stubs when tree-sitter is not available
 ******************************************************************************/

/** @brief Stub: tree-sitter support was not compiled in, so parser creation always fails. */
int64_t eshkol_ts_parser_new(const char* language) { (void)language; return -1; }
/** @brief Stub: no-op since no parsers exist without tree-sitter support. */
void    eshkol_ts_parser_free(int64_t h) { (void)h; }
/** @brief Stub: tree-sitter support was not compiled in, so parsing always fails. */
int64_t eshkol_ts_parse(int64_t p, const char* s, int32_t l) { (void)p;(void)s;(void)l; return -1; }
/** @brief Stub: no-op since no trees exist without tree-sitter support. */
void    eshkol_ts_tree_free(int64_t h) { (void)h; }
/** @brief Stub: always fails since no trees exist without tree-sitter support. */
int32_t eshkol_ts_tree_root(int64_t h, char* b, int32_t s) { (void)h;(void)b;(void)s; return -1; }
/** @brief Stub: always fails since no trees exist without tree-sitter support. */
int32_t eshkol_ts_node_children(int64_t h, uint32_t sb, uint32_t eb, char* b, int32_t bs, int32_t* c) {
    (void)h;(void)sb;(void)eb;(void)b;(void)bs;(void)c; return -1;
}
/** @brief Stub: always fails since no trees exist without tree-sitter support. */
int32_t eshkol_ts_node_text(int64_t h, uint32_t sb, uint32_t eb, char* b, int32_t bs) {
    (void)h;(void)sb;(void)eb;(void)b;(void)bs; return -1;
}
/** @brief Stub: tree-sitter support was not compiled in, so query compilation always fails. */
int64_t eshkol_ts_query_new(const char* l, const char* p) { (void)l;(void)p; return -1; }
/** @brief Stub: no-op since no queries exist without tree-sitter support. */
void    eshkol_ts_query_free(int64_t h) { (void)h; }
/** @brief Stub: always fails since no queries or trees exist without tree-sitter support. */
int32_t eshkol_ts_query_matches(int64_t q, int64_t t, int32_t m, char* b, int32_t bs, int32_t* c) {
    (void)q;(void)t;(void)m;(void)b;(void)bs;(void)c; return -1;
}
/** @brief Stub: always fails since no trees exist without tree-sitter support. */
int32_t eshkol_ts_tree_sexp(int64_t h, char* b, int32_t s) { (void)h;(void)b;(void)s; return -1; }
/**
 * @brief Reports whether this build was compiled with tree-sitter support.
 * @return 0 (this build lacks HAS_TREE_SITTER).
 */
int32_t eshkol_ts_available(void) { return 0; }

#endif /* HAS_TREE_SITTER */
