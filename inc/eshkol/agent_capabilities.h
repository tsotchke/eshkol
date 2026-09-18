/**
 * @file agent_capabilities.h
 * @brief C ABI of the agent capability library: compression, SQLite hooks,
 *        tree-sitter parsing and queries, and Yoga flexbox layout.
 *
 * Every entry point is a plain C function over integers, doubles, byte
 * buffers and opaque integer handles, so that the same symbols can be bound
 * from compiled Eshkol code (`extern` declarations in lib/agent/*.esk), from
 * the JIT, and from the bytecode VM's native bridge.
 *
 * Conventions shared by the whole header:
 * - Handles are small positive integers indexing process-global tables.
 *   0 is never a valid handle; creation functions return -1 on failure.
 *   Passing a stale, out-of-range or never-issued handle is safe: void
 *   functions ignore it and value-returning functions report failure.
 * - Buffer-filling functions take a caller-provided buffer and its capacity
 *   in bytes and return the number of bytes produced, or -1 on error. Each
 *   function documents whether an undersized buffer is an error or a silent
 *   truncation.
 * - The prototypes carry no parameter names; the `@param` names below are the
 *   names used by the definitions and are listed in declaration order.
 * - The tree-sitter and Yoga tables are each guarded by one process-wide
 *   mutex, so those calls may be made from any thread. The compression
 *   functions keep no shared state. The SQLite hooks do not lock their handle
 *   table.
 */
#ifndef ESHKOL_AGENT_CAPABILITIES_H
#define ESHKOL_AGENT_CAPABILITIES_H

#include <stdint.h>

#if defined(_WIN32)
#  if defined(ESHKOL_AGENT_SHARED)
#    if defined(ESHKOL_AGENT_BUILD)
/**
 * @brief Symbol-visibility decoration applied to every function in this header.
 *
 * This branch (Windows, shared agent library, compiling the library itself)
 * expands to `__declspec(dllexport)`.
 */
#      define ESHKOL_AGENT_API __declspec(dllexport)
#    else
/**
 * @brief Symbol-visibility decoration applied to every function in this header.
 *
 * This branch (Windows, consuming a shared agent library) expands to
 * `__declspec(dllimport)`. Define `ESHKOL_AGENT_SHARED` only when linking
 * against a DLL build of the agent library.
 */
#      define ESHKOL_AGENT_API __declspec(dllimport)
#    endif
#  else
/**
 * @brief Symbol-visibility decoration applied to every function in this header.
 *
 * This branch (Windows, static agent library, which is the production
 * configuration) expands to nothing: static consumers must resolve ordinary
 * COFF symbols, not `__imp_*` DLL import thunks.
 */
#    define ESHKOL_AGENT_API
#  endif
#else
/**
 * @brief Symbol-visibility decoration applied to every function in this header.
 *
 * On non-Windows targets it expands to
 * `__attribute__((visibility("default")))`, which keeps the capability
 * symbols exported from executables and shared objects built with hidden
 * default visibility so that the JIT and the VM can resolve them by name.
 */
#  define ESHKOL_AGENT_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Reports whether the zlib-backed compression capability is present.
 *
 * The capability has no unavailable build: a library that defines this symbol
 * always contains the real codec. Hosts that resolve capabilities dynamically
 * treat a missing symbol as "unavailable".
 *
 * @return Always 1.
 */
ESHKOL_AGENT_API int32_t eshkol_compression_available(void);

/**
 * @brief Compresses bytes into a caller-provided buffer as a zlib stream.
 *
 * The output is the zlib container (RFC 1950: two-byte header, deflate data,
 * Adler-32 trailer) at the default compression level, not a raw RFC 1951
 * deflate stream. It is the inverse of eshkol_inflate_data().
 *
 * @param data     Input bytes. May be NULL only when @p data_len is 0.
 * @param data_len Number of input bytes; must be >= 0.
 * @param buf      Destination buffer; must not be NULL.
 * @param buf_size Capacity of @p buf in bytes; must be > 0.
 * @return Number of compressed bytes written to @p buf, or -1 when an
 *         argument is invalid or the compressed stream does not fit in
 *         @p buf_size bytes. Nothing is NUL-terminated. On -1 the contents of
 *         @p buf are unspecified; there is no partial-result mode.
 */
ESHKOL_AGENT_API int32_t eshkol_deflate(const char*, int32_t, char*, int32_t);

/**
 * @brief Decompresses a zlib stream into a caller-provided buffer.
 *
 * Accepts the RFC 1950 container produced by eshkol_deflate().
 *
 * @param data     Compressed bytes. May be NULL only when @p data_len is 0.
 * @param data_len Number of compressed bytes; must be >= 0.
 * @param buf      Destination buffer; must not be NULL.
 * @param buf_size Capacity of @p buf in bytes; must be > 0. It is also the
 *                 hard limit on the decompressed size.
 * @return Number of decompressed bytes written to @p buf, or -1 when an
 *         argument is invalid, the input is corrupt or incomplete, or the
 *         decompressed data does not fit in @p buf_size bytes. Nothing is
 *         NUL-terminated. On -1 the contents of @p buf are unspecified.
 */
ESHKOL_AGENT_API int32_t eshkol_inflate_data(const char*, int32_t, char*, int32_t);

/**
 * @brief Compresses bytes into a caller-provided buffer in gzip format.
 *
 * The output is a complete RFC 1952 gzip member (header, deflate data, CRC-32
 * and length trailer) at the default compression level, produced in a single
 * pass.
 *
 * @param data     Input bytes. May be NULL only when @p data_len is 0.
 * @param data_len Number of input bytes; must be >= 0.
 * @param buf      Destination buffer; must not be NULL.
 * @param buf_size Capacity of @p buf in bytes; must be > 0.
 * @return Number of gzip bytes written to @p buf, or -1 when an argument is
 *         invalid, the codec cannot be initialised, or the complete gzip
 *         stream does not fit in @p buf_size bytes. On -1 the contents of
 *         @p buf are unspecified.
 */
ESHKOL_AGENT_API int32_t eshkol_gzip(const char*, int32_t, char*, int32_t);

/**
 * @brief Decompresses gzip data into a caller-provided buffer.
 *
 * Accepts an RFC 1952 gzip stream (from eshkol_gzip() or any other gzip
 * producer) and decodes it in a single pass.
 *
 * @param data     Gzip bytes. May be NULL only when @p data_len is 0.
 * @param data_len Number of gzip bytes; must be >= 0.
 * @param buf      Destination buffer; must not be NULL.
 * @param buf_size Capacity of @p buf in bytes; must be > 0. It is also the
 *                 hard limit on the decompressed size.
 * @return Number of decompressed bytes written to @p buf, or -1 when an
 *         argument is invalid, the input is corrupt or truncated, or the
 *         decompressed data does not fit in @p buf_size bytes. Nothing is
 *         NUL-terminated. On -1 the contents of @p buf are unspecified.
 */
ESHKOL_AGENT_API int32_t eshkol_gunzip(const char*, int32_t, char*, int32_t);

/**
 * @brief Compresses bytes to a zlib (RFC 1950) stream in a newly allocated buffer.
 *
 * Allocating variant of eshkol_deflate(), used by the Scheme native and VM
 * bridges. The buffer is sized from the codec's worst-case bound for
 * @p data_len before compression starts.
 *
 * @param data       Input bytes. May be NULL only when @p data_len is 0.
 * @param data_len   Number of input bytes; must be >= 0.
 * @param max_output Hard output limit in bytes; must be > 0. The call fails
 *                   when the worst-case compressed size for @p data_len
 *                   exceeds this limit, even if the actual output would be
 *                   smaller.
 * @param out        Receives the allocated buffer; must not be NULL. Set to
 *                   NULL on failure.
 * @param out_len    Receives the number of valid bytes in `*out`; must not be
 *                   NULL. Set to 0 on failure.
 * @return 0 on success, -1 on invalid arguments, a limit violation, codec
 *         failure or allocation failure.
 * @note On success the caller owns `*out` and releases it with
 *       eshkol_compression_free(). The buffer is not NUL-terminated.
 */
ESHKOL_AGENT_API int32_t eshkol_deflate_alloc(const char*, int32_t, int32_t,
                                               char**, int32_t*);

/**
 * @brief Decompresses a zlib (RFC 1950) stream into a newly allocated buffer.
 *
 * Allocating variant of eshkol_inflate_data(). The output buffer starts at
 * four times the input size (at least 64 KiB, at most @p max_output) and
 * doubles until the stream ends or the limit is reached.
 *
 * @param data       Compressed bytes. May be NULL only when @p data_len is 0.
 * @param data_len   Number of compressed bytes; must be >= 0.
 * @param max_output Hard decompression-bomb limit in bytes; must be > 0. A
 *                   stream that expands beyond it fails; no partial result is
 *                   returned.
 * @param out        Receives the allocated buffer; must not be NULL. Set to
 *                   NULL on failure.
 * @param out_len    Receives the number of valid bytes in `*out`; must not be
 *                   NULL. Set to 0 on failure.
 * @return 0 on success, -1 on invalid arguments, corrupt or truncated input,
 *         output larger than @p max_output, or allocation failure.
 * @note On success the caller owns `*out` and releases it with
 *       eshkol_compression_free(). The buffer is not NUL-terminated.
 */
ESHKOL_AGENT_API int32_t eshkol_inflate_alloc(const char*, int32_t, int32_t,
                                               char**, int32_t*);

/**
 * @brief Compresses bytes to gzip (RFC 1952) format in a newly allocated buffer.
 *
 * Allocating variant of eshkol_gzip(); parameters, limit rule, return value
 * and ownership are those of eshkol_deflate_alloc().
 *
 * @param data       Input bytes. May be NULL only when @p data_len is 0.
 * @param data_len   Number of input bytes; must be >= 0.
 * @param max_output Hard output limit in bytes; must be > 0 and at least the
 *                   codec's worst-case compressed size for @p data_len.
 * @param out        Receives the allocated buffer (NULL on failure).
 * @param out_len    Receives the byte count of `*out` (0 on failure).
 * @return 0 on success, -1 on failure.
 * @note Release `*out` with eshkol_compression_free().
 */
ESHKOL_AGENT_API int32_t eshkol_gzip_alloc(const char*, int32_t, int32_t,
                                            char**, int32_t*);

/**
 * @brief Decompresses gzip (RFC 1952) data into a newly allocated buffer,
 *        subject to an output limit.
 *
 * Allocating variant of eshkol_gunzip(); growth policy, limit rule, return
 * value and ownership are those of eshkol_inflate_alloc().
 *
 * @param data       Gzip bytes. May be NULL only when @p data_len is 0.
 * @param data_len   Number of gzip bytes; must be >= 0.
 * @param max_output Hard decompression-bomb limit in bytes; must be > 0.
 * @param out        Receives the allocated buffer (NULL on failure).
 * @param out_len    Receives the byte count of `*out` (0 on failure).
 * @return 0 on success, -1 on invalid arguments, corrupt or truncated input,
 *         output larger than @p max_output, or allocation failure.
 * @note Release `*out` with eshkol_compression_free().
 */
ESHKOL_AGENT_API int32_t eshkol_gunzip_alloc(const char*, int32_t, int32_t,
                                              char**, int32_t*);

/**
 * @brief Releases a buffer returned through `out` by one of the
 *        `eshkol_*_alloc` compression entry points.
 *
 * @param ptr Buffer to release; NULL is accepted and ignored. Must not be a
 *            pointer obtained from any other allocator.
 */
ESHKOL_AGENT_API void eshkol_compression_free(void*);

/**
 * @brief Executes one or more SQL statements that return no rows.
 *
 * One of the SQLite hooks that a hosted VM resolves through this bounded
 * capability ABI. Intended for DDL and DML; result rows, if any, are
 * discarded. The SQLite error message is not retained by this call.
 *
 * @param handle Database handle issued by the agent SQLite binding
 *               (`eshkol_sqlite_open`, bound in lib/agent/sqlite.esk).
 * @param sql    NUL-terminated SQL text; may contain several statements
 *               separated by semicolons.
 * @return 0 on success, the positive SQLite result code on an SQL error, or
 *         -1 when @p handle is not an open database or @p sql is NULL.
 */
ESHKOL_AGENT_API int eshkol_sqlite_exec(int64_t handle, const char* sql);

/**
 * @brief Returns the rowid of the most recent successful INSERT on a connection.
 *
 * @param handle Database handle issued by the agent SQLite binding.
 * @return The last inserted rowid (0 when the connection has not inserted a
 *         row), or -1 when @p handle is not an open database.
 */
ESHKOL_AGENT_API int64_t eshkol_sqlite_last_insert_rowid(int64_t handle);

/**
 * @brief Returns the number of rows changed by the most recent INSERT, UPDATE
 *        or DELETE on a connection.
 *
 * @param handle Database handle issued by the agent SQLite binding.
 * @return The row-change count, which may legitimately be 0, or -1 when
 *         @p handle is not an open database.
 */
ESHKOL_AGENT_API int eshkol_sqlite_changes(int64_t handle);

/**
 * @brief Reports whether the tree-sitter capability is present.
 *
 * The capability has no fallback build: a library that defines this symbol
 * always contains the real parser and its bundled grammars.
 *
 * @return Always 1.
 */
ESHKOL_AGENT_API int32_t eshkol_ts_available(void);

/**
 * @brief Creates a tree-sitter parser for a named language.
 *
 * @param language NUL-terminated language name or alias, matched exactly:
 *                 "javascript"/"js", "typescript"/"ts", "tsx", "python"/"py",
 *                 "rust"/"rs", "go", "c", "cpp"/"c++", "java", "ruby"/"rb",
 *                 "bash"/"sh".
 * @return A parser handle (>= 1), or -1 when @p language is NULL or unknown,
 *         the parser cannot be created, or all 31 parser slots are in use.
 * @note Release the handle with eshkol_ts_parser_free(). Trees already
 *       produced by the parser stay valid after the parser is freed.
 */
ESHKOL_AGENT_API int64_t eshkol_ts_parser_new(const char*);

/**
 * @brief Releases a tree-sitter parser handle.
 *
 * @param handle Parser handle from eshkol_ts_parser_new(). Out-of-range or
 *               already released handles are ignored.
 */
ESHKOL_AGENT_API void eshkol_ts_parser_free(int64_t);

/**
 * @brief Parses source text with a tree-sitter parser handle.
 *
 * The source bytes are copied into storage owned by the returned tree, so the
 * caller may release or modify its input as soon as the call returns. The
 * copy backs eshkol_ts_node_text() and bounds the byte ranges accepted by the
 * node functions. Syntax errors do not fail the parse: they appear as ERROR
 * nodes in the tree.
 *
 * @param parser_handle Parser handle from eshkol_ts_parser_new().
 * @param source        Source bytes; need not be NUL-terminated. May be NULL
 *                      only when @p source_len is 0.
 * @param source_len    Length of @p source in bytes; must be >= 0.
 * @return A tree handle (>= 1), or -1 on an invalid handle or argument,
 *         allocation or parser failure, or when all 63 tree slots are in use.
 * @note Release the handle with eshkol_ts_tree_free().
 */
ESHKOL_AGENT_API int64_t eshkol_ts_parse(int64_t, const char*, int32_t);

/**
 * @brief Releases a tree-sitter syntax-tree handle and its copy of the source text.
 *
 * @param handle Tree handle from eshkol_ts_parse(). Out-of-range or already
 *               released handles are ignored.
 */
ESHKOL_AGENT_API void eshkol_ts_tree_free(int64_t);

/**
 * @brief Writes the root node record of a parsed tree into a caller-provided buffer.
 *
 * A node record is one line of nine tab-separated fields:
 * `type`, `start_row`, `start_col`, `end_row`, `end_col`, `start_byte`,
 * `end_byte`, `child_count`, `named` (1 for a named node, 0 for an anonymous
 * one). Rows, columns and byte offsets are zero-based; `child_count` counts
 * all children, named or not.
 *
 * @param tree_handle Tree handle from eshkol_ts_parse().
 * @param buf         Destination buffer, or NULL together with
 *                    @p buf_size 0 to query the required length.
 * @param buf_size    Capacity of @p buf in bytes, including room for the
 *                    terminating NUL; must be >= 0.
 * @return Length of the record in bytes, excluding the terminating NUL. With
 *         a non-NULL @p buf the record must fit completely
 *         (length < @p buf_size); otherwise the call returns -1 and @p buf
 *         holds a NUL-terminated truncated prefix that must not be used.
 *         Also -1 on an invalid handle or argument.
 */
ESHKOL_AGENT_API int32_t eshkol_ts_tree_root(int64_t, char*, int32_t);

/**
 * @brief Writes the record of the node covering a byte range into a
 *        caller-provided buffer.
 *
 * Selects the smallest named node whose span contains
 * [@p start_byte, @p end_byte) and serialises it in the record format
 * described at eshkol_ts_tree_root().
 *
 * @param tree_handle Tree handle from eshkol_ts_parse().
 * @param start_byte  Inclusive start offset into the parsed source.
 * @param end_byte    Exclusive end offset; must be greater than
 *                    @p start_byte and no larger than the source length.
 * @param buf         Destination buffer, or NULL together with
 *                    @p buf_size 0 to query the required length.
 * @param buf_size    Capacity of @p buf in bytes, including the NUL.
 * @return Length of the record excluding the NUL, or -1 on an invalid handle,
 *         an empty or out-of-range byte range, no matching node, or a record
 *         that does not fit completely in @p buf.
 */
ESHKOL_AGENT_API int32_t eshkol_ts_node_info(int64_t, uint32_t, uint32_t,
                                             char*, int32_t);

/**
 * @brief Writes the named children of a tree-sitter node into a
 *        caller-provided buffer.
 *
 * When @p start_byte and @p end_byte are both 0 the target is the root node;
 * otherwise it is the smallest named node containing
 * [@p start_byte, @p end_byte). Each named child is written as one node
 * record (see eshkol_ts_tree_root()); records are separated by a single NUL
 * byte and the last record is followed by a terminating NUL. Anonymous
 * children (punctuation, keywords) are not reported.
 *
 * @param tree_handle Tree handle from eshkol_ts_parse().
 * @param start_byte  Inclusive start offset, or 0 with @p end_byte 0 for the root.
 * @param end_byte    Exclusive end offset, no larger than the source length.
 * @param buf         Destination buffer, or NULL together with
 *                    @p buf_size 0 to query the required size.
 * @param buf_size    Capacity of @p buf in bytes; must exceed the returned
 *                    byte count by at least one (room for the final NUL).
 * @param count_out   Receives the number of child records; may be NULL. It is
 *                    set to 0 on every failure after argument validation.
 * @return Total bytes occupied by the records and their separators, excluding
 *         the final NUL (0 when the node has no named children), or -1 on an
 *         invalid handle, an invalid byte range, no matching node, or a buffer
 *         too small for all records. There is no truncated result: when the
 *         buffer is too small nothing is written.
 */
ESHKOL_AGENT_API int32_t eshkol_ts_node_children(int64_t, uint32_t, uint32_t,
                                                 char*, int32_t, int32_t*);

/**
 * @brief Writes the source text represented by a byte range of a parsed tree
 *        into a caller-provided buffer.
 *
 * Copies bytes [@p start_byte, @p end_byte) of the tree's own copy of the
 * source and NUL-terminates the result. Byte ranges normally come from the
 * `start_byte`/`end_byte` fields of a node record.
 *
 * @param tree_handle Tree handle from eshkol_ts_parse().
 * @param start_byte  Inclusive start offset; must be less than the source length.
 * @param end_byte    Exclusive end offset; must be greater than
 *                    @p start_byte and no larger than the source length.
 * @param buf         Destination buffer, or NULL together with
 *                    @p buf_size 0 to query the full text length.
 * @param buf_size    Capacity of @p buf in bytes including the NUL. When
 *                    @p buf is not NULL it must be at least 1.
 * @return Number of text bytes written, excluding the NUL. Text longer than
 *         @p buf_size - 1 bytes is silently truncated to that length (possibly
 *         in the middle of a UTF-8 sequence) and the truncated length is
 *         returned; compare with the size query to detect truncation. -1 on
 *         an invalid handle or argument, or an empty or out-of-range range.
 */
ESHKOL_AGENT_API int32_t eshkol_ts_node_text(int64_t, uint32_t, uint32_t,
                                             char*, int32_t);

/**
 * @brief Creates a tree-sitter query handle from a language and a query string.
 *
 * @param language NUL-terminated language name or alias, as accepted by
 *                 eshkol_ts_parser_new(). A query can only be run against
 *                 trees parsed with the same grammar.
 * @param pattern  NUL-terminated tree-sitter query in S-expression form, for
 *                 example `(function_definition name: (identifier) @name)`.
 * @return A query handle (>= 1), or -1 when an argument is NULL, the language
 *         is unknown, the pattern does not compile (the error position is not
 *         reported), or all 31 query slots are in use.
 * @note Release the handle with eshkol_ts_query_free().
 */
ESHKOL_AGENT_API int64_t eshkol_ts_query_new(const char*, const char*);

/**
 * @brief Releases a tree-sitter query handle.
 *
 * @param handle Query handle from eshkol_ts_query_new(). Out-of-range or
 *               already released handles are ignored.
 */
ESHKOL_AGENT_API void eshkol_ts_query_free(int64_t);

/**
 * @brief Writes the captures matched by a tree-sitter query against a syntax
 *        tree into a caller-provided buffer.
 *
 * Every capture of every match becomes one record of seven tab-separated
 * fields: `capture_name`, `start_byte`, `end_byte`, `start_row`, `start_col`,
 * `end_row`, `end_col`. Records are separated by a single NUL byte and the
 * last record is followed by a terminating NUL.
 *
 * @param query_handle Query handle from eshkol_ts_query_new().
 * @param tree_handle  Tree handle from eshkol_ts_parse(), parsed with the
 *                     same grammar as the query.
 * @param max_matches  Result limit; must be >= 0. 0 selects the built-in
 *                     ceiling of 100000. The limit is compared with the
 *                     number of capture records written so far before each
 *                     new match is taken, so a match with several captures
 *                     can carry the record count past the limit.
 * @param buf          Destination buffer, or NULL together with
 *                     @p buf_size 0 to query the required size.
 * @param buf_size     Capacity of @p buf in bytes; must exceed the returned
 *                     byte count by at least one.
 * @param count_out    Receives the number of capture records; may be NULL.
 * @return Total bytes occupied by the records and their separators, excluding
 *         the final NUL (0 when nothing matched), or -1 on an invalid handle
 *         or argument, a query and tree from different grammars, or a buffer
 *         too small for all records. On a too-small buffer there is no
 *         partial result: `buf[0]` is set to NUL and `*count_out` to 0.
 */
ESHKOL_AGENT_API int32_t eshkol_ts_query_matches(int64_t, int64_t, int32_t,
                                                 char*, int32_t, int32_t*);

/**
 * @brief Writes an S-expression representation of a syntax tree into a
 *        caller-provided buffer.
 *
 * The text is tree-sitter's own rendering of the root node, intended for
 * debugging and inspection.
 *
 * @param tree_handle Tree handle from eshkol_ts_parse().
 * @param buf         Destination buffer; must not be NULL (this function has
 *                    no size-query mode).
 * @param buf_size    Capacity of @p buf in bytes including the NUL; must be > 0.
 * @return Number of bytes written, excluding the NUL. An S-expression longer
 *         than @p buf_size - 1 bytes is silently truncated to that length; a
 *         return value equal to @p buf_size - 1 therefore indicates possible
 *         truncation. -1 on an invalid handle or argument, or allocation
 *         failure.
 */
ESHKOL_AGENT_API int32_t eshkol_ts_tree_sexp(int64_t, char*, int32_t);

/**
 * @brief Reports whether the Yoga flexbox layout capability is present.
 *
 * The capability has no no-op build: a library that defines this symbol
 * always contains the real layout engine.
 *
 * @return Always 1.
 */
ESHKOL_AGENT_API int32_t eshkol_yoga_available(void);

/**
 * @brief Creates an opaque Yoga layout node handle.
 *
 * The node starts with Yoga's default style and no parent.
 *
 * @return A node handle (>= 1), or -1 when the node cannot be allocated or
 *         all 511 node slots are in use.
 * @note Release the handle with eshkol_yoga_node_free().
 */
ESHKOL_AGENT_API int64_t eshkol_yoga_node_create(void);

/**
 * @brief Releases an opaque Yoga layout node handle together with its subtree.
 *
 * The node is detached from its parent, if any, and the node and all of its
 * descendants are destroyed; every handle that referred to a node of that
 * subtree becomes invalid.
 *
 * @param handle Node handle from eshkol_yoga_node_create(). Out-of-range or
 *               already released handles are ignored.
 */
ESHKOL_AGENT_API void eshkol_yoga_node_free(int64_t);

/**
 * @brief Sets a floating-point Yoga node property.
 *
 * @param handle Node handle.
 * @param prop   Property selector: 0 width, 1 height, 2 min-width,
 *               3 min-height, 4 max-width, 5 max-height, 6 flex-grow,
 *               7 flex-shrink, 8 flex-basis, 9 gap (rows and columns),
 *               10-13 padding left/right/top/bottom,
 *               14-17 margin left/right/top/bottom,
 *               18-21 border left/right/top/bottom. Lengths are in points.
 * @param value  Property value; narrowed to `float`. It must be finite, and
 *               must be >= 0 for every selector except the margins (14-17),
 *               which accept negative values.
 * @note The call is ignored, with no error reported, when the handle is
 *       invalid, the selector is unknown, or the value is rejected.
 */
ESHKOL_AGENT_API void eshkol_yoga_node_set_float(int64_t, int32_t, double);

/**
 * @brief Sets an integer (enumerated) Yoga node property.
 *
 * @param handle Node handle.
 * @param prop   Property selector: 0 flex-direction, 1 justify-content,
 *               2 align-items, 3 align-self, 4 align-content,
 *               5 position-type, 6 overflow, 7 display.
 * @param value  Numeric value of the corresponding Yoga enumeration
 *               (`YGFlexDirection`, `YGJustify`, `YGAlign`, `YGPositionType`,
 *               `YGOverflow`, `YGDisplay`); values outside the enumeration's
 *               range are rejected.
 * @note The call is ignored, with no error reported, when the handle is
 *       invalid, the selector is unknown, or the value is rejected.
 */
ESHKOL_AGENT_API void eshkol_yoga_node_set_int(int64_t, int32_t, int32_t);

/**
 * @brief Adds a child node at a requested Yoga index.
 *
 * @param parent Handle of the node that receives the child.
 * @param child  Handle of the node to insert. It must not already have a
 *               parent, must differ from @p parent, and must not be an
 *               ancestor of @p parent.
 * @param index  Zero-based insertion position; must be between 0 and the
 *               parent's current child count inclusive.
 * @note The call is ignored, with no error reported, when any of these
 *       conditions fails or a handle is invalid. The parent does not take
 *       over the child's handle: the child stays addressable, and freeing the
 *       parent invalidates it.
 */
ESHKOL_AGENT_API void eshkol_yoga_node_add_child(int64_t, int64_t, int32_t);

/**
 * @brief Calculates a Yoga node layout for the supplied dimensions.
 *
 * Runs the flexbox algorithm over the tree rooted at @p root with
 * left-to-right direction. Afterwards the results for the root and every
 * descendant are available through eshkol_yoga_node_get_computed().
 *
 * @param root   Handle of the root node of the tree to lay out.
 * @param width  Available width in points: a finite value >= 0, or NaN for
 *               "undefined" (size to content).
 * @param height Available height in points, with the same rule as @p width.
 * @note The call is ignored when the handle is invalid or a dimension is
 *       negative or infinite.
 */
ESHKOL_AGENT_API void eshkol_yoga_node_calculate(int64_t, double, double);

/**
 * @brief Reads a computed Yoga node layout value.
 *
 * Values are meaningful after eshkol_yoga_node_calculate() has run on a tree
 * containing the node. Positions are relative to the node's parent.
 *
 * @param handle Node handle.
 * @param prop   Result selector: 0 left, 1 top, 2 width, 3 height,
 *               4-7 padding left/top/right/bottom,
 *               8-11 margin left/top/right/bottom,
 *               12-15 border left/top/right/bottom. Note that the edge order
 *               differs from the setter's left/right/top/bottom order.
 * @return The computed value, or 0.0 when the handle is invalid or the
 *         selector is unknown.
 */
ESHKOL_AGENT_API double eshkol_yoga_node_get_computed(int64_t, int32_t);

#ifdef __cplusplus
}
#endif

#endif
