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

enum class Backend {
    Native,
    Wasm,
    Vm
};

enum class ExecutionProfile {
    HostedNative,
    HostedWasm,
    HostedVm,
    FreestandingKernelNative,
    FreestandingMcuNative,
    FreestandingVm,
    EmbeddedVm
};

struct Info {
    ExecutionProfile id;
    const char* name;
    Backend backend;
    bool hosted;
    bool freestanding;
    bool experimental;
    bool implies_compile_only;
    bool implies_no_stdlib;
    const char* target_triple;
};

struct Selection {
    ExecutionProfile requested = ExecutionProfile::HostedNative;
    bool explicit_request = false;
    const char* explicit_target_triple = nullptr;
    bool compile_only = false;
    bool shared_lib = false;
    bool wasm_flag = false;
    bool no_stdlib = false;
    bool eval_mode = false;
    bool run_mode = false;
    bool has_eskb_output = false;
    bool has_linked_libs = false;
};

struct Resolution {
    const Info* profile = nullptr;
    bool compile_only = false;
    bool no_stdlib = false;
    bool wasm_output = false;
    bool vm_only = false;
    bool embedded_vm = false;
    const char* target_triple = nullptr;
    std::string error;
};

const Info* find(std::string_view name);
const Info& get(ExecutionProfile profile);
const char* name(ExecutionProfile profile);
std::string supported_names();
Resolution resolve(const Selection& selection);

}  // namespace eshkol::profile

#endif
