/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Stable C ABI exported by the runtime for generated programs and the REPL JIT.
 * These wrappers hide platform-specific libc differences from LLVM codegen.
 */

#ifndef ESHKOL_RUNTIME_EXPORTS_H
#define ESHKOL_RUNTIME_EXPORTS_H

#include <cstdint>
#include <cstdio>

namespace eshkol::runtime {

inline constexpr const char* stdout_stream_symbol = "eshkol_stdout_stream";
inline constexpr const char* drand48_symbol = "eshkol_drand48";
inline constexpr const char* srand48_symbol = "eshkol_srand48";
inline constexpr const char* getenv_symbol = "eshkol_getenv";
inline constexpr const char* setenv_symbol = "eshkol_setenv";
inline constexpr const char* unsetenv_symbol = "eshkol_unsetenv";
inline constexpr const char* usleep_symbol = "eshkol_usleep";
inline constexpr const char* fopen_symbol = "eshkol_fopen";
inline constexpr const char* access_symbol = "eshkol_access";
inline constexpr const char* remove_symbol = "eshkol_remove";
inline constexpr const char* rename_symbol = "eshkol_rename";
inline constexpr const char* mkdir_symbol = "eshkol_mkdir";
inline constexpr const char* rmdir_symbol = "eshkol_rmdir";
inline constexpr const char* chdir_symbol = "eshkol_chdir";
inline constexpr const char* stat_symbol = "eshkol_stat";
inline constexpr const char* opendir_symbol = "eshkol_opendir";

} // namespace eshkol::runtime

extern "C" {

FILE* eshkol_stdout_stream();
double eshkol_drand48();
void eshkol_srand48(std::int64_t seed);
char* eshkol_getenv(const char* name);
int eshkol_setenv(const char* name, const char* value, int overwrite);
int eshkol_unsetenv(const char* name);
int eshkol_usleep(std::uint32_t usec);
FILE* eshkol_fopen(const char* path, const char* mode);
int eshkol_access(const char* path, int mode);
int eshkol_remove(const char* path);
int eshkol_rename(const char* old_path, const char* new_path);
int eshkol_mkdir(const char* path, int mode);
int eshkol_rmdir(const char* path);
int eshkol_chdir(const char* path);
int eshkol_stat(const char* path, void* buf);
void* eshkol_opendir(const char* path);

}

#endif // ESHKOL_RUNTIME_EXPORTS_H
