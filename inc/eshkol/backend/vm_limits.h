/**
 * @file vm_limits.h
 * @brief Compile-time limits for the bytecode VM profile.
 */

#ifndef ESHKOL_BACKEND_VM_LIMITS_H
#define ESHKOL_BACKEND_VM_LIMITS_H

#include <limits.h>

/* Desktop defaults. Embedded/product profiles should override these through
 * CMake target definitions instead of editing VM sources. */
#ifndef ESHKOL_VM_HEAP_SIZE
#define ESHKOL_VM_HEAP_SIZE 65536
#endif

/* Upper bound on the growable heap object table. ESHKOL_VM_HEAP_SIZE is only
 * the INITIAL capacity — heap_alloc() doubles the table on demand up to this
 * ceiling, so a workload's live-object count (an N-parameter gradient, a large
 * literal, a long-running agent) is governed by memory rather than by a
 * compile-time pool size. Reaching the ceiling is reported by name. */
#ifndef ESHKOL_VM_HEAP_MAX_SIZE
#define ESHKOL_VM_HEAP_MAX_SIZE 16777216
#endif

#ifndef ESHKOL_VM_STACK_SIZE
#define ESHKOL_VM_STACK_SIZE 262144
#endif

#ifndef ESHKOL_VM_MAX_FRAMES
#define ESHKOL_VM_MAX_FRAMES 256
#endif

/* Initial constant-pool capacity. The pool grows on demand (see
 * vm_ensure_const_cap) up to ESHKOL_VM_MAX_CONSTS_CEILING, so a program's
 * constant count — dominated by large literals — is not a compile-time cap. */
#ifndef ESHKOL_VM_MAX_CONSTS
#define ESHKOL_VM_MAX_CONSTS 4096
#endif

#ifndef ESHKOL_VM_MAX_CONSTS_CEILING
#define ESHKOL_VM_MAX_CONSTS_CEILING 4194304
#endif

#ifndef ESHKOL_VM_MAX_CODE
#define ESHKOL_VM_MAX_CODE 1000000
#endif

/* Packed string/symbol literals carry their pack count in the native-call
 * operand.  The legacy ids 100/101 remain readable for older ESKB files, but
 * newly emitted bytecode uses these reserved ids so the runtime never has to
 * guess where a literal begins on the operand stack.  Keep the ranges below
 * the signed 32-bit instruction-operand ceiling and outside the host-native
 * range. */
#ifndef ESHKOL_VM_PACKED_STRING_MAX_BYTES
#define ESHKOL_VM_PACKED_STRING_MAX_BYTES (1024 * 1024)
#endif

#define ESHKOL_VM_PACKED_STRING_FID_BASE 2000000000
#define ESHKOL_VM_PACKED_SYMBOL_FID_BASE 2100000000
#define ESHKOL_VM_PACKED_LITERAL_MAX_PACKS \
    ((ESHKOL_VM_PACKED_STRING_MAX_BYTES + 7) / 8)
#define ESHKOL_VM_PACKED_STRING_FID_LIMIT \
    (ESHKOL_VM_PACKED_STRING_FID_BASE + ESHKOL_VM_PACKED_LITERAL_MAX_PACKS + 1)
#define ESHKOL_VM_PACKED_SYMBOL_FID_LIMIT \
    (ESHKOL_VM_PACKED_SYMBOL_FID_BASE + ESHKOL_VM_PACKED_LITERAL_MAX_PACKS + 1)
#define ESHKOL_VM_IS_PACKED_LITERAL_FID(fid) \
    (((fid) >= ESHKOL_VM_PACKED_STRING_FID_BASE && \
      (fid) < ESHKOL_VM_PACKED_STRING_FID_LIMIT) || \
     ((fid) >= ESHKOL_VM_PACKED_SYMBOL_FID_BASE && \
      (fid) < ESHKOL_VM_PACKED_SYMBOL_FID_LIMIT))

/* Compatibility name retained for embedders that inspect the VM profile.
 * Capture counts are carried by the versioned long-closure encoding and are
 * bounded only by the signed instruction/resource domain, not by a byte-sized
 * source-level limit. */
#ifndef ESHKOL_VM_MAX_CLOSURE_UPVALUES
#define ESHKOL_VM_MAX_CLOSURE_UPVALUES INT_MAX
#endif

/* Runaway-instruction guard for the bytecode interpreter: the number of
 * instructions vm_run() will execute before deciding the program is not going
 * to terminate. Unlike the capacities above this is a *default*, not a fixed
 * ceiling — `ESHKOL_VM_MAX_INSN` overrides it per run, and 0 means unlimited.
 * Enforced in lib/backend/vm_run.c by both the computed-goto and the switch
 * dispatch paths. */
#ifndef ESHKOL_VM_DEFAULT_MAX_INSN
#define ESHKOL_VM_DEFAULT_MAX_INSN 10000000ULL
#endif

#if ESHKOL_VM_HEAP_SIZE <= 0
#error "ESHKOL_VM_HEAP_SIZE must be positive"
#endif

#if ESHKOL_VM_HEAP_MAX_SIZE < ESHKOL_VM_HEAP_SIZE
#error "ESHKOL_VM_HEAP_MAX_SIZE must be >= ESHKOL_VM_HEAP_SIZE"
#endif

#if ESHKOL_VM_STACK_SIZE <= 0
#error "ESHKOL_VM_STACK_SIZE must be positive"
#endif

#if ESHKOL_VM_MAX_FRAMES <= 0
#error "ESHKOL_VM_MAX_FRAMES must be positive"
#endif

#if ESHKOL_VM_MAX_CONSTS <= 0
#error "ESHKOL_VM_MAX_CONSTS must be positive"
#endif

#if ESHKOL_VM_MAX_CONSTS_CEILING < ESHKOL_VM_MAX_CONSTS
#error "ESHKOL_VM_MAX_CONSTS_CEILING must be >= ESHKOL_VM_MAX_CONSTS"
#endif

#if ESHKOL_VM_MAX_CODE <= 0
#error "ESHKOL_VM_MAX_CODE must be positive"
#endif

#if ESHKOL_VM_PACKED_LITERAL_MAX_PACKS >= 100000000
#error "ESHKOL_VM_PACKED_STRING_MAX_BYTES leaves no safe packed-fid range"
#endif

#if ESHKOL_VM_MAX_CLOSURE_UPVALUES <= 0
#error "ESHKOL_VM_MAX_CLOSURE_UPVALUES must be positive"
#endif

/* Legacy aliases used inside the current unity-built VM components. Keep these
 * local to VM sources; new build/profile code should use the ESHKOL_VM_* names. */
#undef HEAP_SIZE
#define HEAP_SIZE ESHKOL_VM_HEAP_SIZE

#undef STACK_SIZE
#define STACK_SIZE ESHKOL_VM_STACK_SIZE

#undef MAX_FRAMES
#define MAX_FRAMES ESHKOL_VM_MAX_FRAMES

#undef MAX_CONSTS
#define MAX_CONSTS ESHKOL_VM_MAX_CONSTS

#undef MAX_CODE
#define MAX_CODE ESHKOL_VM_MAX_CODE

#endif /* ESHKOL_BACKEND_VM_LIMITS_H */
