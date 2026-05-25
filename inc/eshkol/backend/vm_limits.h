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

#ifndef ESHKOL_VM_STACK_SIZE
#define ESHKOL_VM_STACK_SIZE 4096
#endif

#ifndef ESHKOL_VM_MAX_FRAMES
#define ESHKOL_VM_MAX_FRAMES 256
#endif

#ifndef ESHKOL_VM_MAX_CONSTS
#define ESHKOL_VM_MAX_CONSTS 1024
#endif

#ifndef ESHKOL_VM_MAX_CODE
#define ESHKOL_VM_MAX_CODE 100000
#endif

#if ESHKOL_VM_HEAP_SIZE <= 0
#error "ESHKOL_VM_HEAP_SIZE must be positive"
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

#if ESHKOL_VM_MAX_CODE <= 0
#error "ESHKOL_VM_MAX_CODE must be positive"
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
