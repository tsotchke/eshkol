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

} // namespace eshkol::runtime

extern "C" {

FILE* eshkol_stdout_stream();
double eshkol_drand48();
void eshkol_srand48(std::int64_t seed);
char* eshkol_getenv(const char* name);
int eshkol_setenv(const char* name, const char* value, int overwrite);
int eshkol_unsetenv(const char* name);
int eshkol_usleep(std::uint32_t usec);

}

#endif // ESHKOL_RUNTIME_EXPORTS_H
