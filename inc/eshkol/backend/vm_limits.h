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

/* Per-closure upvalue capacity: the SINGLE source of truth shared by the
 * compiler (vm_parser.c's MAX_UPVALUES, which bounds how many free variables
 * one lexical scope may register) and the runtime closure representation
 * (vm_core.c's HeapObject.closure.upvalues[]/open_slots[] fixed arrays, and
 * every OP_CLOSURE/OP_GET_UPVALUE/OP_SET_UPVALUE/native-131/151 site in
 * vm_run.c and vm_native.c that indexes them).
 *
 * These two counts used to be independent literals — MAX_UPVALUES 32 at
 * compile time, a bare `16` baked into the closure arrays and every runtime
 * access at execution time. A closure needing between 17 and 32 upvalues
 * compiled cleanly (under the compiler's limit) and then, at OP_CLOSURE, had
 * its upvalue count silently clamped to 16: the runtime popped only 16 of
 * the >16 values the compiler had pushed to feed it, stranding the rest on
 * the operand stack. Nothing about that clamp was visible — no error, exit
 * 0 — so every stack slot computed at compile time for the REST OF THE
 * PROGRAM was off by the leaked count from then on, and the next top-level
 * `define` silently read back whatever stray value the leak had left in its
 * slot instead of its own closure. A single procedure with ~20 constructor
 * calls (one small closure captured per call) was enough to cross 16.
 *
 * Fixing the mismatch requires both limits to be THIS constant, everywhere,
 * so they can never diverge again; see MAX_UPVALUES below. Raising it is
 * free (an array bound, not a growth path) — 32 comfortably covers ordinary
 * programs, and vm_compile_error() below now refuses to compile silently
 * past it rather than letting a legitimately-larger closure fall through the
 * old silent-drop. */
#ifndef ESHKOL_VM_MAX_CLOSURE_UPVALUES
#define ESHKOL_VM_MAX_CLOSURE_UPVALUES 32
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

#undef MAX_UPVALUES
#define MAX_UPVALUES ESHKOL_VM_MAX_CLOSURE_UPVALUES

#endif /* ESHKOL_BACKEND_VM_LIMITS_H */
