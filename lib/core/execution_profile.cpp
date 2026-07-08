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

}  // namespace eshkol::profile
