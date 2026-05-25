#include <eshkol/core/execution_profile.h>

#include <iostream>
#include <string>

namespace {

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << std::endl;
    return 1;
}

}  // namespace

int main() {
    using namespace eshkol::profile;

    if (const Info* info = find("hosted-native"); !info || info->id != ExecutionProfile::HostedNative) {
        return fail("failed to resolve hosted-native profile");
    }

    if (const Info* info = find("embedded-vm"); !info || info->id != ExecutionProfile::EmbeddedVm) {
        return fail("failed to resolve embedded-vm profile");
    }

    if (find("not-a-profile") != nullptr) {
        return fail("unexpectedly resolved an invalid profile name");
    }

    {
        Selection selection;
        selection.requested = ExecutionProfile::HostedWasm;
        Resolution resolved = resolve(selection);
        if (!resolved.profile || resolved.profile->id != ExecutionProfile::HostedWasm) {
            return fail("legacy wasm selection did not resolve to hosted-wasm");
        }
        if (!resolved.wasm_output || std::string(resolved.target_triple ? resolved.target_triple : "") != "wasm32-unknown-unknown") {
            return fail("legacy wasm selection did not set wasm output and target triple");
        }
    }

    {
        Selection selection;
        selection.requested = ExecutionProfile::HostedNative;
        selection.explicit_request = true;
        selection.wasm_flag = true;
        Resolution resolved = resolve(selection);
        if (resolved.error.empty()) {
            return fail("explicit hosted-native + --wasm should fail");
        }
    }

    {
        Selection selection;
        selection.requested = ExecutionProfile::HostedVm;
        selection.explicit_request = true;
        Resolution resolved = resolve(selection);
        if (resolved.error.empty()) {
            return fail("hosted-vm without --emit-eskb should fail");
        }
    }

    {
        Selection selection;
        selection.requested = ExecutionProfile::HostedVm;
        selection.explicit_request = true;
        selection.has_eskb_output = true;
        Resolution resolved = resolve(selection);
        if (!resolved.error.empty() || !resolved.vm_only) {
            return fail("hosted-vm with --emit-eskb should resolve to vm-only mode");
        }
    }

    {
        Selection selection;
        selection.requested = ExecutionProfile::FreestandingKernelNative;
        selection.explicit_request = true;
        Resolution resolved = resolve(selection);
        if (resolved.error.empty()) {
            return fail("freestanding-kernel-native without explicit target should fail");
        }
    }

    {
        Selection selection;
        selection.requested = ExecutionProfile::FreestandingKernelNative;
        selection.explicit_request = true;
        selection.explicit_target_triple = "x86_64-unknown-linux-gnu";
        Resolution resolved = resolve(selection);
        if (!resolved.error.empty()) {
            return fail("freestanding-kernel-native should resolve cleanly");
        }
        if (!resolved.compile_only || !resolved.no_stdlib) {
            return fail("freestanding-kernel-native should imply compile-only and no-stdlib");
        }
        if (std::string(resolved.target_triple ? resolved.target_triple : "") !=
            "x86_64-unknown-linux-gnu") {
            return fail("freestanding-kernel-native should preserve explicit target triple");
        }
    }

    {
        Selection selection;
        selection.requested = ExecutionProfile::FreestandingKernelNative;
        selection.explicit_request = true;
        selection.explicit_target_triple = "x86_64-unknown-linux-gnu";
        selection.shared_lib = true;
        Resolution resolved = resolve(selection);
        if (resolved.error.empty()) {
            return fail("freestanding-kernel-native + --shared-lib should fail");
        }
    }

    {
        Selection selection;
        selection.requested = ExecutionProfile::FreestandingVm;
        selection.explicit_request = true;
        selection.has_eskb_output = true;
        Resolution resolved = resolve(selection);
        if (!resolved.error.empty() || !resolved.vm_only || !resolved.no_stdlib) {
            return fail("freestanding-vm should resolve to vm-only mode and imply no-stdlib");
        }
    }

    {
        Selection selection;
        selection.requested = ExecutionProfile::EmbeddedVm;
        selection.explicit_request = true;
        Resolution resolved = resolve(selection);
        if (resolved.error.empty()) {
            return fail("embedded-vm without --emit-eskb should fail");
        }
    }

    {
        Selection selection;
        selection.requested = ExecutionProfile::EmbeddedVm;
        selection.explicit_request = true;
        selection.has_eskb_output = true;
        Resolution resolved = resolve(selection);
        if (!resolved.error.empty() || !resolved.vm_only || !resolved.no_stdlib ||
            !resolved.embedded_vm || resolved.target_triple != nullptr) {
            return fail("embedded-vm should resolve to embedded vm-only no-stdlib mode");
        }
    }

    {
        Selection selection;
        selection.requested = ExecutionProfile::EmbeddedVm;
        selection.explicit_request = true;
        selection.has_eskb_output = true;
        selection.has_linked_libs = true;
        Resolution resolved = resolve(selection);
        if (resolved.error.empty()) {
            return fail("embedded-vm + --lib should fail");
        }
    }

    {
        Selection selection;
        selection.requested = ExecutionProfile::EmbeddedVm;
        selection.explicit_request = true;
        selection.has_eskb_output = true;
        selection.explicit_target_triple = "thumbv7em-none-eabi";
        Resolution resolved = resolve(selection);
        if (resolved.error.empty()) {
            return fail("embedded-vm + --target should fail");
        }
    }

    std::cout << "PASS" << std::endl;
    return 0;
}
