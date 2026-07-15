#ifndef ESHKOL_AGENT_CAPABILITIES_H
#define ESHKOL_AGENT_CAPABILITIES_H

#include <stdint.h>

#if defined(_WIN32)
#  if defined(ESHKOL_AGENT_BUILD)
#    define ESHKOL_AGENT_API __declspec(dllexport)
#  else
#    define ESHKOL_AGENT_API __declspec(dllimport)
#  endif
#else
#  define ESHKOL_AGENT_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

ESHKOL_AGENT_API int32_t eshkol_compression_available(void);
ESHKOL_AGENT_API int32_t eshkol_deflate(const char*, int32_t, char*, int32_t);
ESHKOL_AGENT_API int32_t eshkol_inflate_data(const char*, int32_t, char*, int32_t);
ESHKOL_AGENT_API int32_t eshkol_gzip(const char*, int32_t, char*, int32_t);
ESHKOL_AGENT_API int32_t eshkol_gunzip(const char*, int32_t, char*, int32_t);

/* Allocating variants used by the Scheme native/VM bridges.  The caller owns
 * *out and releases it with eshkol_compression_free.  max_output is a hard
 * decompression-bomb limit; compressed output must also fit beneath it. */
ESHKOL_AGENT_API int32_t eshkol_deflate_alloc(const char*, int32_t, int32_t,
                                               char**, int32_t*);
ESHKOL_AGENT_API int32_t eshkol_inflate_alloc(const char*, int32_t, int32_t,
                                               char**, int32_t*);
ESHKOL_AGENT_API int32_t eshkol_gzip_alloc(const char*, int32_t, int32_t,
                                            char**, int32_t*);
ESHKOL_AGENT_API int32_t eshkol_gunzip_alloc(const char*, int32_t, int32_t,
                                              char**, int32_t*);
ESHKOL_AGENT_API void eshkol_compression_free(void*);

ESHKOL_AGENT_API int32_t eshkol_ts_available(void);
ESHKOL_AGENT_API int64_t eshkol_ts_parser_new(const char*);
ESHKOL_AGENT_API void eshkol_ts_parser_free(int64_t);
ESHKOL_AGENT_API int64_t eshkol_ts_parse(int64_t, const char*, int32_t);
ESHKOL_AGENT_API void eshkol_ts_tree_free(int64_t);
ESHKOL_AGENT_API int32_t eshkol_ts_tree_root(int64_t, char*, int32_t);
ESHKOL_AGENT_API int32_t eshkol_ts_node_info(int64_t, uint32_t, uint32_t,
                                             char*, int32_t);
ESHKOL_AGENT_API int32_t eshkol_ts_node_children(int64_t, uint32_t, uint32_t,
                                                 char*, int32_t, int32_t*);
ESHKOL_AGENT_API int32_t eshkol_ts_node_text(int64_t, uint32_t, uint32_t,
                                             char*, int32_t);
ESHKOL_AGENT_API int64_t eshkol_ts_query_new(const char*, const char*);
ESHKOL_AGENT_API void eshkol_ts_query_free(int64_t);
ESHKOL_AGENT_API int32_t eshkol_ts_query_matches(int64_t, int64_t, int32_t,
                                                 char*, int32_t, int32_t*);
ESHKOL_AGENT_API int32_t eshkol_ts_tree_sexp(int64_t, char*, int32_t);

ESHKOL_AGENT_API int32_t eshkol_yoga_available(void);
ESHKOL_AGENT_API int64_t eshkol_yoga_node_create(void);
ESHKOL_AGENT_API void eshkol_yoga_node_free(int64_t);
ESHKOL_AGENT_API void eshkol_yoga_node_set_float(int64_t, int32_t, double);
ESHKOL_AGENT_API void eshkol_yoga_node_set_int(int64_t, int32_t, int32_t);
ESHKOL_AGENT_API void eshkol_yoga_node_add_child(int64_t, int64_t, int32_t);
ESHKOL_AGENT_API void eshkol_yoga_node_calculate(int64_t, double, double);
ESHKOL_AGENT_API double eshkol_yoga_node_get_computed(int64_t, int32_t);

#ifdef __cplusplus
}
#endif

#endif
