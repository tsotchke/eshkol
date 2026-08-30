/**
 * @file vm_limits.h
 * @brief Compile-time limits for the bytecode VM profile.
 */

#ifndef ESHKOL_BACKEND_VM_LIMITS_H
#define ESHKOL_BACKEND_VM_LIMITS_H

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
#define ESHKOL_VM_STACK_SIZE 4096
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
#define ESHKOL_VM_MAX_CODE 100000
#endif

/* The bytecode closure operand stores the capture count in bits 16..23, so
 * 255 is the representable count for this bytecode format. The compiler and
 * runtime allocate capture tables per closure; this value is an encoding
 * limit, not an array capacity or a source-level closure limit.
 *
 * The former implementation used fixed compiler/runtime arrays with
 * mismatched limits. The compiler now grows its free-variable table and each
 * runtime closure owns arrays sized to its actual capture count. It no longer
 * uses a fixed source-level capture limit. */
#ifndef ESHKOL_VM_MAX_CLOSURE_UPVALUES
#define ESHKOL_VM_MAX_CLOSURE_UPVALUES 255
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

#if ESHKOL_VM_MAX_CLOSURE_UPVALUES <= 0
#error "ESHKOL_VM_MAX_CLOSURE_UPVALUES must be positive"
#endif

/* The CLOSURE instruction packs the upvalue count into an 8-bit operand
 * field (bits 16..23, see OP_CLOSURE in vm_run.c) — this must never silently
 * truncate the way the closure-array size once did. */
#if ESHKOL_VM_MAX_CLOSURE_UPVALUES > 255
#error "ESHKOL_VM_MAX_CLOSURE_UPVALUES must fit the OP_CLOSURE operand's 8-bit upvalue-count field (<= 255)"
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
