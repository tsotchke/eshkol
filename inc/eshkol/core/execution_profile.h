/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Execution profile model for hosted and freestanding compilation modes.
 */
#ifndef ESHKOL_CORE_EXECUTION_PROFILE_H
#define ESHKOL_CORE_EXECUTION_PROFILE_H

#include <string>
#include <string_view>

namespace eshkol::profile {

/**
 * @brief Code generation backend targeted by a given execution profile.
 */
enum class Backend {
    Native,  // Native machine code via the LLVM backend
    Wasm,    // WebAssembly output (wasm32-unknown-unknown)
    Vm       // Eshkol's portable bytecode VM (.eskb)
};

/**
 * @brief The set of supported hosted/freestanding compilation and execution modes.
 *
 * Selected via the `--profile` CLI flag (captured in Selection::requested)
 * and validated/expanded into concrete compiler behavior (compile-only,
 * no-stdlib, target triple, VM-only, etc.) by resolve(). Static metadata for
 * each value is looked up via get()/find().
 */
enum class ExecutionProfile {
    HostedNative,              // Default: native code, full host OS + stdlib support
    HostedWasm,                // Host toolchain emitting WebAssembly
    HostedVm,                  // Host OS present; execution via the bytecode VM (requires --emit-eskb)
    FreestandingKernelNative,  // No OS/stdlib; native code for kernel-level targets (requires --target)
    FreestandingMcuNative,     // No OS/stdlib; native code for microcontroller targets (requires --target)
    FreestandingVm,            // No OS/stdlib; execution via the bytecode VM (requires --emit-eskb)
    EmbeddedVm                 // No OS/stdlib, no target triple; embedded bytecode VM (requires --emit-eskb)
};

/**
 * @brief Static metadata describing one ExecutionProfile.
 *
 * One Info exists per ExecutionProfile value, in the internal profile table
 * defined in execution_profile.cpp; looked up via get()/find().
 */
struct Info {
    ExecutionProfile id;        // The profile this metadata describes
    const char* name;           // CLI-facing name, e.g. "hosted-native"
    Backend backend;            // Code generation backend for this profile
    bool hosted;                // True if a host OS/runtime is assumed
    bool freestanding;          // True if no host OS/stdlib is assumed
    bool experimental;          // True if the profile is not yet considered stable
    bool implies_compile_only;  // True if selecting this profile forces compile-only output
    bool implies_no_stdlib;     // True if selecting this profile forces --no-stdlib
    const char* target_triple;  // Default LLVM target triple, or nullptr to use the host triple
};

/**
 * @brief Raw CLI/user input describing the desired execution profile and
 * related flags, prior to validation.
 *
 * Built from parsed command-line arguments and passed to resolve(), which
 * validates the combination of profile and flags and produces a Resolution.
 */
struct Selection {
    ExecutionProfile requested = ExecutionProfile::HostedNative;  // Profile named by --profile, or the default
    bool explicit_request = false;                  // True if --profile was passed explicitly (vs. defaulted)
    const char* explicit_target_triple = nullptr;    // Value of --target, if given
    bool compile_only = false;                       // True if --compile-only (or equivalent) was passed
    bool shared_lib = false;                         // True if --shared-lib was passed
    bool wasm_flag = false;                          // True if --wasm was passed
    bool no_stdlib = false;                          // True if --no-stdlib was passed
    bool eval_mode = false;                          // True if invoked as `eshkol eval`
    bool run_mode = false;                           // True if invoked as `eshkol run`
    bool has_eskb_output = false;                     // True if --emit-eskb <path> was passed
    bool has_linked_libs = false;                     // True if --lib was passed one or more times
};

/**
 * @brief Outcome of validating a Selection against its ExecutionProfile's constraints.
 *
 * Produced by resolve(). If @c error is non-empty, the requested combination
 * of profile and flags is invalid and the caller should report @c error and
 * abort compilation; otherwise the remaining fields describe the effective
 * compilation mode to use.
 */
struct Resolution {
    const Info* profile = nullptr;       // Metadata for the resolved (requested) profile
    bool compile_only = false;           // Effective compile-only setting after profile constraints
    bool no_stdlib = false;              // Effective no-stdlib setting after profile constraints
    bool wasm_output = false;            // True if output should be WebAssembly
    bool vm_only = false;                // True if output should target the bytecode VM only
    bool embedded_vm = false;            // True specifically for the embedded-vm profile
    const char* target_triple = nullptr; // Effective target triple (explicit override, profile default, or nullptr for host)
    std::string error;                   // Non-empty if the Selection is invalid for this profile; empty on success
};

/**
 * @brief Look up profile metadata by its CLI-facing name.
 *
 * @param name Profile name, e.g. "hosted-native" (see supported_names() for the full list).
 * @return Pointer to the matching Info, or nullptr if no profile has that name.
 */
const Info* find(std::string_view name);
/**
 * @brief Look up profile metadata by ExecutionProfile enum value.
 *
 * @param profile Profile to look up.
 * @return Reference to the matching Info. Falls back to the first table
 *         entry (HostedNative) in the unexpected case that no entry matches.
 */
const Info& get(ExecutionProfile profile);
/**
 * @brief Get the CLI-facing name of a profile.
 *
 * @param profile Profile to name.
 * @return Same as get(profile).name.
 */
const char* name(ExecutionProfile profile);
/**
 * @brief Build a human-readable, comma-separated list of all supported profile names.
 *
 * Intended for help text and error messages (e.g. reporting an unrecognized `--profile` value).
 *
 * @return Comma-and-space-separated profile names, in table order.
 */
std::string supported_names();
/**
 * @brief Validate a requested profile/flag Selection and compute its effective compilation settings.
 *
 * When no `--profile` was explicitly requested (Selection::explicit_request
 * is false), only lightweight defaulting is applied (e.g. hosted-wasm
 * implies WebAssembly output) and no validation is performed. When a
 * profile was explicitly requested, each profile enforces its own
 * constraints — e.g. hosted-wasm rejects JIT eval/run and --shared-lib;
 * freestanding profiles require --target; VM-backed profiles require
 * --emit-eskb — and reports a descriptive error on the returned Resolution
 * if violated.
 *
 * @param selection User-requested profile and flags.
 * @return Resolution with effective settings, or with a non-empty
 *         Resolution::error describing why the selection is invalid.
 */
Resolution resolve(const Selection& selection);

/**
 * @brief Floating-point calling convention of an embedded target.
 */
enum class FloatAbi {
    Default,  // The LLVM target's default for the triple
    Soft,     // Floating-point values travel in integer registers
    Hard      // Floating-point values travel in FPU registers
};

/**
 * @brief Code model of an embedded target (how far code may address).
 */
enum class CodeModel {
    Default,  // The LLVM target's default (small / medlow)
    Medium    // RISC-V medany: PC-relative, any 2 GiB window (images at 0x80000000)
};

/**
 * @brief Machine description of one bare-metal (freestanding) target.
 *
 * This table is the single source the native backend consults when a
 * freestanding profile names a microcontroller or bare-metal application
 * processor with `--target`: the canonical LLVM triple, CPU, feature set,
 * ABI name and float ABI all come from here, so data layout, instruction
 * selection, ELF flags and build attributes agree for every object the
 * compiler emits for that target.
 */
struct EmbeddedTarget {
    const char* name;         // Accepted --target spelling (exact match)
    const char* llvm_triple;  // Canonical LLVM triple written into the module
    const char* cpu;          // LLVM CPU name
    const char* features;     // LLVM subtarget feature string
    const char* abi;          // ABI name (RISC-V ilp32/lp64d, ...) or "" for the default
    FloatAbi float_abi;       // Float calling convention
    CodeModel code_model;     // Addressing model for code and data references
    const char* isa;          // Human-readable ISA summary, e.g. "rv32imac"
};

/**
 * @brief Look up the embedded target description for a --target spelling.
 *
 * @param triple The spelling passed to --target, e.g. "thumbv7em-none-eabi",
 *        "riscv64gc-unknown-none-elf" or "aarch64-unknown-none".
 * @return The matching description, or nullptr when @p triple is not a
 *         bare-metal target this table describes (hosted and kernel triples
 *         such as x86_64-unknown-linux-gnu keep the generic native path).
 */
const EmbeddedTarget* find_embedded_target(std::string_view triple);

/**
 * @brief Comma-separated list of every --target spelling in the embedded table.
 */
std::string embedded_target_names();

}  // namespace eshkol::profile

#endif
