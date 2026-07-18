/*
 * Native 128-bit Integer (i128) — native/JIT/AOT runtime API.
 *
 * Tagged-in / tagged-out, arena-boxing entry points implemented in
 * lib/core/i128_runtime.cpp. These are the functions the LLVM codegen lowers
 * the i128 builtins to, and the symbols the REPL/JIT must register. The pure
 * computation they delegate to lives in <eshkol/core/i128.h> (shared with the
 * bytecode VM).
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_CORE_I128_RUNTIME_H
#define ESHKOL_CORE_I128_RUNTIME_H

#include "eshkol/eshkol.h"
#include "eshkol/core/i128.h"

/* Opaque arena handle (defined in the arena allocator). Forward-declared here
 * exactly as in bignum.h so this header stands alone. */
typedef struct arena arena_t;

#ifdef __cplusplus
extern "C" {
#endif

/* (i128 x) / (int->i128 x) — construct from an exact fixnum. */
void eshkol_i128_from_int_tagged(arena_t* arena,
                                 const eshkol_tagged_value_t* x,
                                 eshkol_tagged_value_t* out);

/* (string->i128 s) — parse full signed 128-bit range, including -2^127. */
void eshkol_i128_from_string_tagged(arena_t* arena,
                                    const eshkol_tagged_value_t* s,
                                    eshkol_tagged_value_t* out);

/* (i128? x) */
void eshkol_i128_predicate_tagged(const eshkol_tagged_value_t* x,
                                  eshkol_tagged_value_t* out);

/* Binary arithmetic. op: 0=add 1=sub 2=mul 3=quotient 4=remainder. */
void eshkol_i128_binary_tagged(arena_t* arena,
                               const eshkol_tagged_value_t* a,
                               const eshkol_tagged_value_t* b,
                               int32_t op,
                               eshkol_tagged_value_t* out);

/* (i128-neg n) */
void eshkol_i128_neg_tagged(arena_t* arena,
                            const eshkol_tagged_value_t* a,
                            eshkol_tagged_value_t* out);

/* Shifts. op: 0=shl 1=ashr 2=lshr. Count is an exact fixnum in [0,127]. */
void eshkol_i128_shift_tagged(arena_t* arena,
                              const eshkol_tagged_value_t* a,
                              const eshkol_tagged_value_t* count,
                              int32_t op,
                              eshkol_tagged_value_t* out);

/* Comparisons. op: 0='=' 1='<' 2='>' 3='<=' 4='>='. */
void eshkol_i128_compare_tagged(const eshkol_tagged_value_t* a,
                                const eshkol_tagged_value_t* b,
                                int32_t op,
                                eshkol_tagged_value_t* out);

/* (i128->string n) */
void eshkol_i128_to_string_tagged(arena_t* arena,
                                  const eshkol_tagged_value_t* a,
                                  eshkol_tagged_value_t* out);

/* (i128->int n) — narrow to a fixnum, raising when out of int64 range. */
void eshkol_i128_to_int_tagged(const eshkol_tagged_value_t* a,
                               eshkol_tagged_value_t* out);

/* Render a boxed i128 payload ({lo,hi}) as decimal to a FILE* stream. */
void eshkol_i128_display(const void* payload, void* stream);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_I128_RUNTIME_H */
