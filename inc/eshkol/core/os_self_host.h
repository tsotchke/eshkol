/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol OS Phase U6: Freestanding zero-LLVM in-OS self-hosting compiler and
 * standalone W^X ELF64 emitter ABI.
 *
 * Pure C11/C++17 freestanding compiler runtime for compiling Eshkol source
 * inside an Eshkol OS userspace process without LLVM, libc, POSIX I/O, or
 * dynamic heap allocation.
 *
 * Design invariants:
 *   1. Zero dynamic host allocation: lexer tokens, AST nodes, symbol tables,
 *      bytecode buffers, native machine-code buffers, and patch tables live
 *      strictly inside a caller-supplied bounded workspace buffer.
 *   2. Byte-identical determinism: compilation output depends solely on the
 *      source bytes and target ISA, independent of initial workspace poison
 *      contents, buffer alignment offsets, or prior compilations.
 *   3. Standalone W^X ELF64 output: native executables carry only PT_LOAD
 *      segments with strict W^X permissions (PF_R|PF_X for .text, PF_R|PF_W
 *      for .data, never PF_W|PF_X), zero PT_INTERP, zero PT_DYNAMIC, and
 *      direct architecture syscall exit via ESHKOL_SYS_TASK_EXIT.
 */

#ifndef ESHKOL_CORE_OS_SELF_HOST_H
#define ESHKOL_CORE_OS_SELF_HOST_H

#include <stddef.h>
#include <stdint.h>

#include <eshkol/core/os_syscall_abi.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Maximum S-expression nesting depth admitted by the bounded parser.
 */
#define ESHKOL_OS_SH_MAX_PARSE_DEPTH 64u

/**
 * @brief Minimum caller workspace size in bytes required by the in-OS compiler.
 */
#define ESHKOL_OS_SH_MIN_WORKSPACE_BYTES 65536u

/**
 * @brief Canonical static ELF64 base virtual address for emitted executables.
 */
#define ESHKOL_OS_SH_ELF_BASE_VADDR 0x00400000ULL

/**
 * @brief Canonical page-aligned virtual address of the executable .text segment.
 */
#define ESHKOL_OS_SH_ELF_TEXT_VADDR 0x00401000ULL

/**
 * @brief Page size used to separate W^X ELF64 load segments.
 */
#define ESHKOL_OS_SH_ELF_PAGE_SIZE 4096ULL

/**
 * @brief Supported 64-bit native target ISAs for standalone ELF64 emission.
 */
typedef enum eshkol_os_target_isa {
    ESHKOL_OS_TARGET_X86_64  = 0,
    ESHKOL_OS_TARGET_AARCH64 = 1,
    ESHKOL_OS_TARGET_RISCV64 = 2
} eshkol_os_target_isa_t;

/**
 * @brief Explicit status codes returned by the in-OS self-hosting compiler.
 */
typedef enum eshkol_os_self_host_status {
    ESHKOL_SH_OK                   = 0,
    ESHKOL_SH_ERR_INVALID_ARG      = 1,
    ESHKOL_SH_ERR_SYNTAX           = 2,
    ESHKOL_SH_ERR_DEPTH_EXCEEDED   = 3,
    ESHKOL_SH_ERR_UNBOUND_SYMBOL   = 4,
    ESHKOL_SH_ERR_ARITY            = 5,
    ESHKOL_SH_ERR_UNSUPPORTED_FORM = 6,
    ESHKOL_SH_ERR_WORKSPACE_OOM    = 7,
    ESHKOL_SH_ERR_OUTPUT_TOO_SMALL = 8,
    ESHKOL_SH_ERR_INVALID_TARGET   = 9,
    ESHKOL_SH_ERR_INVALID_SYSCALL  = 10,
    ESHKOL_SH_ERR_DUPLICATE_SYMBOL = 11,
    ESHKOL_SH_ERR_LIMIT_EXCEEDED   = 12
} eshkol_os_self_host_status_t;

/**
 * @brief Deterministic compilation result and output telemetry.
 */
typedef struct eshkol_os_compile_result {
    eshkol_os_self_host_status_t status;
    uint32_t error_line;
    uint32_t error_col;
    uint32_t crc32;
    size_t bytes_written;
    uint32_t function_count;
    uint32_t instruction_count;
    uint32_t constant_count;
    uint32_t reserved;
    uint64_t entry_vaddr;
    uint64_t text_vaddr;
    uint64_t text_bytes;
    uint64_t data_vaddr;
    uint64_t data_bytes;
    size_t workspace_peak_bytes;
} eshkol_os_compile_result_t;

/**
 * @brief Compile Eshkol source directly into CRC32-verified ESKB v2 bytecode.
 *
 * All parser, symbol, and bytecode assembly tables are allocated exclusively
 * inside the caller-supplied `workspace` buffer and rewound to zero before
 * return. Supports integer/float/boolean/string literals, arithmetic
 * (`+`, `-`, `*`, `/`, `modulo`, `abs`), comparisons (`=`, `<`, `<=`, `>`, `>=`),
 * logic (`not`, `and`, `or`), `if`, `begin`, `let`, `let*`, named `let`,
 * `define` functions with recursion and self-tail-call loops, pair primitives
 * (`cons`, `car`, `cdr`, `null?`, `pair?`), and capability syscalls via
 * `(os-syscall num cap arg0 ...)`.
 *
 * @param source           Pointer to UTF-8 Eshkol source text.
 * @param source_len       Byte length of `source`.
 * @param workspace        Caller-supplied temporary scratch memory (zero malloc).
 * @param workspace_bytes  Byte capacity of `workspace` (>= ESHKOL_OS_SH_MIN_WORKSPACE_BYTES).
 * @param out_buf          Destination buffer receiving the serialized ESKB v2 image.
 * @param out_cap          Byte capacity of `out_buf`.
 * @param out_result       Optional output struct receiving status and telemetry.
 * @return ESHKOL_SH_OK on success, or an explicit error status.
 */
eshkol_os_self_host_status_t eshkol_os_compile_to_eskb(
    const char* source,
    size_t source_len,
    void* workspace,
    size_t workspace_bytes,
    uint8_t* out_buf,
    size_t out_cap,
    eshkol_os_compile_result_t* out_result);

/**
 * @brief Compile Eshkol source directly into a standalone static W^X ELF64 binary.
 *
 * Zero-LLVM, zero-assembler, zero-linker native code generator emitting a
 * complete ET_EXEC ELF64 image for x86_64, AArch64, or RISC-V 64 with
 * page-separated PT_LOAD segments (PF_R|PF_X .text and PF_R|PF_W .data, never
 * PF_W|PF_X), zero PT_INTERP, zero PT_DYNAMIC, and an entry stub that evaluates
 * the compiled program and exits via ESHKOL_SYS_TASK_EXIT (`syscall`, `svc #0`,
 * or `ecall`).
 *
 * @param source           Pointer to UTF-8 Eshkol source text.
 * @param source_len       Byte length of `source`.
 * @param target           Target 64-bit ISA (`ESHKOL_OS_TARGET_*`).
 * @param workspace        Caller-supplied temporary scratch memory (zero malloc).
 * @param workspace_bytes  Byte capacity of `workspace` (>= ESHKOL_OS_SH_MIN_WORKSPACE_BYTES).
 * @param out_buf          Destination buffer receiving the ELF64 executable image.
 * @param out_cap          Byte capacity of `out_buf`.
 * @param out_result       Optional output struct receiving status and telemetry.
 * @return ESHKOL_SH_OK on success, or an explicit error status.
 */
eshkol_os_self_host_status_t eshkol_os_compile_to_elf64(
    const char* source,
    size_t source_len,
    eshkol_os_target_isa_t target,
    void* workspace,
    size_t workspace_bytes,
    uint8_t* out_buf,
    size_t out_cap,
    eshkol_os_compile_result_t* out_result);

/**
 * @brief Return a static human-readable name for a self-host compiler status code.
 */
const char* eshkol_os_self_host_status_name(eshkol_os_self_host_status_t status);

/**
 * @brief Return a static human-readable name for a native target ISA enum.
 */
const char* eshkol_os_target_isa_name(eshkol_os_target_isa_t target);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_OS_SELF_HOST_H */
