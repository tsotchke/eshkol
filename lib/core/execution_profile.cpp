/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */
#include <eshkol/core/execution_profile.h>

#include <array>

namespace eshkol::profile {

namespace {

constexpr std::array<Info, 7> kProfiles = {{
    {ExecutionProfile::HostedNative, "hosted-native", Backend::Native, true, false, false, false, false, nullptr},
    {ExecutionProfile::HostedWasm, "hosted-wasm", Backend::Wasm, true, false, false, false, false, "wasm32-unknown-unknown"},
    {ExecutionProfile::HostedVm, "hosted-vm", Backend::Vm, true, false, true, false, false, nullptr},
    {ExecutionProfile::FreestandingKernelNative, "freestanding-kernel-native", Backend::Native, false, true, true, true, true, nullptr},
    {ExecutionProfile::FreestandingMcuNative, "freestanding-mcu-native", Backend::Native, false, true, true, true, true, nullptr},
    {ExecutionProfile::FreestandingVm, "freestanding-vm", Backend::Vm, false, true, true, false, true, nullptr},
    {ExecutionProfile::EmbeddedVm, "embedded-vm", Backend::Vm, false, true, true, false, true, nullptr},
}};

}  // namespace

/** Look up an execution profile by its canonical name (e.g. "hosted-native").
 *  @return Pointer into the static profile table, or nullptr if unknown. */
const Info* find(std::string_view query) {
    for (const auto& profile : kProfiles) {
        if (query == profile.name) {
            return &profile;
        }
    }
    return nullptr;
}

/** Look up an execution profile's Info by its enum id.
 *  @return Reference into the static profile table; falls back to the
 *  first table entry if `profile` is not found (should not happen for
 *  any valid ExecutionProfile enumerator). */
const Info& get(ExecutionProfile profile) {
    for (const auto& info : kProfiles) {
        if (info.id == profile) {
            return info;
        }
    }
    return kProfiles.front();
}

/** Return the canonical string name for an execution profile. */
const char* name(ExecutionProfile profile) {
    return get(profile).name;
}

/** Build a comma-separated list of every supported profile name, for
 *  use in CLI help/error messages. */
std::string supported_names() {
    std::string names;
    for (size_t i = 0; i < kProfiles.size(); ++i) {
        if (i > 0) {
            names += ", ";
        }
        names += kProfiles[i].name;
    }
    return names;
}

/** @brief Resolve a user's CLI --profile/--wasm/--target/etc selection
 *  into a concrete build Resolution.
 *
 *  Starts from the requested profile's defaults, then validates the
 *  combination of flags for that profile (e.g. hosted-wasm rejects
 *  --shared-lib and JIT eval/run; freestanding profiles require an
 *  explicit --target; VM-only profiles require --emit-eskb). Any
 *  violation is reported via `Resolution::error` rather than thrown.
 *  When the profile wasn't explicitly requested, only the implicit
 *  wasm-output default is applied and no validation runs.
 *  @return The resolved build configuration, with `error` set (non-null)
 *  if the selection is invalid for the resolved profile. */
Resolution resolve(const Selection& selection) {
    Resolution resolved;
    resolved.profile = &get(selection.requested);
    resolved.compile_only = selection.compile_only;
    resolved.no_stdlib = selection.no_stdlib;
    resolved.wasm_output = selection.wasm_flag;
    resolved.target_triple = selection.explicit_target_triple
                                 ? selection.explicit_target_triple
                                 : resolved.profile->target_triple;

    if (!selection.explicit_request) {
        if (resolved.profile->id == ExecutionProfile::HostedWasm) {
            resolved.wasm_output = true;
        }
        return resolved;
    }

    switch (resolved.profile->id) {
    case ExecutionProfile::HostedNative:
        if (selection.wasm_flag) {
            resolved.error = "--profile hosted-native cannot be combined with --wasm";
        }
        break;

    case ExecutionProfile::HostedWasm:
        if (selection.eval_mode || selection.run_mode) {
            resolved.error = "--profile hosted-wasm does not support JIT eval/run";
            break;
        }
        if (selection.shared_lib) {
            resolved.error = "--profile hosted-wasm cannot be combined with --shared-lib";
            break;
        }
        if (selection.has_linked_libs) {
            resolved.error = "--profile hosted-wasm cannot be combined with --lib";
            break;
        }
        resolved.wasm_output = true;
        break;

    case ExecutionProfile::HostedVm:
        if (!selection.has_eskb_output) {
            resolved.error = "--profile hosted-vm requires --emit-eskb <path>";
            break;
        }
        if (selection.eval_mode || selection.run_mode) {
            resolved.error = "--profile hosted-vm does not support JIT eval/run";
            break;
        }
        if (selection.shared_lib) {
            resolved.error = "--profile hosted-vm cannot be combined with --shared-lib";
            break;
        }
        if (selection.wasm_flag) {
            resolved.error = "--profile hosted-vm cannot be combined with --wasm";
            break;
        }
        if (selection.has_linked_libs) {
            resolved.error = "--profile hosted-vm cannot be combined with --lib";
            break;
        }
        resolved.vm_only = true;
        resolved.wasm_output = false;
        resolved.target_triple = nullptr;
        break;

    case ExecutionProfile::FreestandingKernelNative:
    case ExecutionProfile::FreestandingMcuNative:
        if (selection.eval_mode || selection.run_mode) {
            resolved.error = "--profile ";
            resolved.error += resolved.profile->name;
            resolved.error += " does not support JIT eval/run";
            break;
        }
        if (selection.shared_lib) {
            resolved.error = "--profile ";
            resolved.error += resolved.profile->name;
            resolved.error += " cannot be combined with --shared-lib";
            break;
        }
        if (selection.wasm_flag) {
            resolved.error = "--profile ";
            resolved.error += resolved.profile->name;
            resolved.error += " cannot be combined with --wasm";
            break;
        }
        if (selection.has_linked_libs) {
            resolved.error = "--profile ";
            resolved.error += resolved.profile->name;
            resolved.error += " cannot be combined with --lib";
            break;
        }
        if (!selection.explicit_target_triple) {
            resolved.error = "--profile ";
            resolved.error += resolved.profile->name;
            resolved.error += " requires --target <triple>";
            break;
        }
        resolved.compile_only = true;
        resolved.no_stdlib = true;
        resolved.wasm_output = false;
        break;

    case ExecutionProfile::FreestandingVm:
        if (!selection.has_eskb_output) {
            resolved.error = "--profile freestanding-vm requires --emit-eskb <path>";
            break;
        }
        if (selection.eval_mode || selection.run_mode) {
            resolved.error = "--profile freestanding-vm does not support JIT eval/run";
            break;
        }
        if (selection.shared_lib) {
            resolved.error = "--profile freestanding-vm cannot be combined with --shared-lib";
            break;
        }
        if (selection.wasm_flag) {
            resolved.error = "--profile freestanding-vm cannot be combined with --wasm";
            break;
        }
        if (selection.has_linked_libs) {
            resolved.error = "--profile freestanding-vm cannot be combined with --lib";
            break;
        }
        resolved.no_stdlib = true;
        resolved.vm_only = true;
        resolved.wasm_output = false;
        resolved.target_triple = nullptr;
        break;

    case ExecutionProfile::EmbeddedVm:
        if (!selection.has_eskb_output) {
            resolved.error = "--profile embedded-vm requires --emit-eskb <path>";
            break;
        }
        if (selection.eval_mode || selection.run_mode) {
            resolved.error = "--profile embedded-vm does not support JIT eval/run";
            break;
        }
        if (selection.shared_lib) {
            resolved.error = "--profile embedded-vm cannot be combined with --shared-lib";
            break;
        }
        if (selection.wasm_flag) {
            resolved.error = "--profile embedded-vm cannot be combined with --wasm";
            break;
        }
        if (selection.has_linked_libs) {
            resolved.error = "--profile embedded-vm cannot be combined with --lib";
            break;
        }
        if (selection.explicit_target_triple) {
            resolved.error = "--profile embedded-vm cannot be combined with --target";
            break;
        }
        resolved.no_stdlib = true;
        resolved.vm_only = true;
        resolved.embedded_vm = true;
        resolved.wasm_output = false;
        resolved.target_triple = nullptr;
        break;
    }

    return resolved;
}

namespace {

// Feature strings mirror what clang selects for the same -mcpu/-march/-mabi,
// so objects built by eshkol-run and by clang for the freestanding runtime
// subset (lib/platform/embedded) link without attribute or ABI mismatches.
constexpr const char* kCortexM4SoftFeatures =
    "+soft-float,+dsp,+hwdiv,-fpregs,-vfp2,-vfp2sp,-vfp3,"
    "-vfp3d16,-vfp3d16sp,-vfp3sp,-fp16,-vfp4,-vfp4d16,-vfp4d16sp,-vfp4sp,"
    "-fp-armv8,-fp-armv8d16,-fp-armv8d16sp,-fp-armv8sp,-fullfp16,-fp64,-d32,"
    "-neon,-mve,-mve.fp";
constexpr const char* kCortexM4HardFeatures =
    "+dsp,+hwdiv,+vfp2sp,+vfp3d16sp,+fp16,+vfp4d16sp,-fp64,-d32,-neon,-mve,-mve.fp";
constexpr const char* kAArch64Features = "+v8a,+fp-armv8,+neon";
constexpr const char* kRv32imacFeatures = "+m,+a,+c,+zmmul,+zaamo,+zalrsc,+zca,+relax";
constexpr const char* kRv64imacFeatures = "+m,+a,+c,+zmmul,+zaamo,+zalrsc,+zca,+relax";
constexpr const char* kRv64gcFeatures =
    "+m,+a,+f,+d,+c,+zicsr,+zifencei,+zmmul,+zaamo,+zalrsc,+zca,+zcd,+relax";

constexpr std::array<EmbeddedTarget, 12> kEmbeddedTargets = {{
    // ARM Cortex-M4 (ARMv7E-M, Thumb-2), software floating point.
    {"thumbv7em-none-eabi", "thumbv7em-unknown-none-eabi", "cortex-m4",
     kCortexM4SoftFeatures, "aapcs", FloatAbi::Soft, CodeModel::Default, "armv7e-m"},
    {"thumbv7em-unknown-none-eabi", "thumbv7em-unknown-none-eabi", "cortex-m4",
     kCortexM4SoftFeatures, "aapcs", FloatAbi::Soft, CodeModel::Default, "armv7e-m"},
    // ARM Cortex-M4F: single-precision FPv4-SP-D16, hard-float calling convention.
    {"thumbv7em-none-eabihf", "thumbv7em-unknown-none-eabihf", "cortex-m4",
     kCortexM4HardFeatures, "aapcs", FloatAbi::Hard, CodeModel::Default, "armv7e-m+fpv4-sp-d16"},
    {"thumbv7em-unknown-none-eabihf", "thumbv7em-unknown-none-eabihf", "cortex-m4",
     kCortexM4HardFeatures, "aapcs", FloatAbi::Hard, CodeModel::Default, "armv7e-m+fpv4-sp-d16"},
    // AArch64 bare metal (ARMv8-A application profile, AAPCS64).
    {"aarch64-unknown-none", "aarch64-unknown-none-elf", "generic",
     kAArch64Features, "", FloatAbi::Default, CodeModel::Default, "armv8-a"},
    {"aarch64-unknown-none-elf", "aarch64-unknown-none-elf", "generic",
     kAArch64Features, "", FloatAbi::Default, CodeModel::Default, "armv8-a"},
    {"aarch64-none-elf", "aarch64-unknown-none-elf", "generic",
     kAArch64Features, "", FloatAbi::Default, CodeModel::Default, "armv8-a"},
    // RISC-V 32-bit microcontroller profile: RV32IMAC, soft-float ilp32.
    {"riscv32imac-unknown-none-elf", "riscv32-unknown-none-elf", "generic-rv32",
     kRv32imacFeatures, "ilp32", FloatAbi::Default, CodeModel::Default, "rv32imac"},
    {"riscv32-unknown-none-elf", "riscv32-unknown-none-elf", "generic-rv32",
     kRv32imacFeatures, "ilp32", FloatAbi::Default, CodeModel::Default, "rv32imac"},
    // RISC-V 64-bit application profile: RV64GC, double-float lp64d.
    {"riscv64gc-unknown-none-elf", "riscv64-unknown-none-elf", "generic-rv64",
     kRv64gcFeatures, "lp64d", FloatAbi::Default, CodeModel::Medium, "rv64gc"},
    {"riscv64-unknown-none-elf", "riscv64-unknown-none-elf", "generic-rv64",
     kRv64gcFeatures, "lp64d", FloatAbi::Default, CodeModel::Medium, "rv64gc"},
    // RISC-V 64-bit without hardware floating point: RV64IMAC, lp64.
    {"riscv64imac-unknown-none-elf", "riscv64-unknown-none-elf", "generic-rv64",
     kRv64imacFeatures, "lp64", FloatAbi::Default, CodeModel::Medium, "rv64imac"},
}};

}  // namespace

/** Look up the bare-metal target description for a --target spelling. */
const EmbeddedTarget* find_embedded_target(std::string_view triple) {
    for (const auto& target : kEmbeddedTargets) {
        if (triple == target.name) {
            return &target;
        }
    }
    return nullptr;
}

/** Comma-separated list of every --target spelling in the embedded table. */
std::string embedded_target_names() {
    std::string names;
    for (size_t i = 0; i < kEmbeddedTargets.size(); ++i) {
        if (i > 0) {
            names += ", ";
        }
        names += kEmbeddedTargets[i].name;
    }
    return names;
}

}  // namespace eshkol::profile
