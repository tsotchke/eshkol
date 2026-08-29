#ifndef ESHKOL_AGENT_PLATFORM_H
#define ESHKOL_AGENT_PLATFORM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Compatibility metadata ABI: epoch-second timestamps and file type. */
int32_t eshkol_file_stat_fields(const char* path,
                                int64_t* out_size, int64_t* out_mtime,
                                int64_t* out_ctime, int32_t* out_mode,
                                int32_t* out_type);

/* Extended metadata ABI. The first six outputs preserve the compatibility
 * ordering; the final three add nanosecond mtime and stable identity. */
int32_t eshkol_file_stat_fields_v2(const char* path,
                                   int64_t* out_size, int64_t* out_mtime,
                                   int64_t* out_ctime, int32_t* out_mode,
                                   int32_t* out_type, int64_t* out_mtime_ns,
                                   int64_t* out_device, int64_t* out_inode);

#ifdef __cplusplus
}
#endif

#endif
