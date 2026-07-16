#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::string read_file(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    std::string contents = buffer.str();
    std::string normalized;
    normalized.reserve(contents.size());
    for (char ch : contents) {
        if (ch != '\r') {
            normalized.push_back(ch);
        }
    }
    return normalized;
}

bool expect_contains(const std::string& haystack, const std::string& needle,
                     const std::string& label) {
    if (haystack.find(needle) == std::string::npos) {
        std::cerr << "missing: " << label << "\nneedle: " << needle << std::endl;
        return false;
    }
    return true;
}

bool expect_not_contains(const std::string& haystack, const std::string& needle,
                         const std::string& label) {
    if (haystack.find(needle) != std::string::npos) {
        std::cerr << "unexpected: " << label << "\nneedle: " << needle
                  << std::endl;
        return false;
    }
    return true;
}

std::size_t count_occurrences(const std::string& haystack,
                              const std::string& needle) {
    if (needle.empty()) return 0;
    std::size_t count = 0;
    std::size_t pos = 0;
    while ((pos = haystack.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << std::endl;
    return 1;
}

struct ReleaseAsset {
    const char* name;
    const char* extension;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        return fail("usage: release_workflow_surface_test <source-root>");
    }

    const fs::path source_root = argv[1];
    const fs::path workflow_path =
        source_root / ".github" / "workflows" / "release.yml";
    if (!fs::exists(workflow_path)) {
        return fail("release.yml not found under source root");
    }
    const fs::path package_verifier_path =
        source_root / "scripts" / "verify_release_package.py";
    if (!fs::exists(package_verifier_path)) {
        return fail("verify_release_package.py not found under source root");
    }
    const fs::path repl_jit_path =
        source_root / "lib" / "repl" / "repl_jit.cpp";
    const fs::path jit_target_config_path =
        source_root / "lib" / "repl" / "jit_target_config.h";
    const fs::path jit_coff_memory_manager_path =
        source_root / "lib" / "repl" / "jit_coff_memory_manager.cpp";
    const fs::path tensor_codegen_path =
        source_root / "lib" / "backend" / "tensor_codegen.cpp";
    const fs::path portable_stdlib_verifier_path =
        source_root / "scripts" / "verify_portable_stdlib.py";
    if (!fs::exists(repl_jit_path) ||
        !fs::exists(jit_target_config_path) ||
        !fs::exists(jit_coff_memory_manager_path) ||
        !fs::exists(tensor_codegen_path) ||
        !fs::exists(portable_stdlib_verifier_path)) {
        return fail("release target configuration sources not found under source root");
    }
    const fs::path runtime_def_path =
        source_root / "cmake" / "gen_runtime_def.cmake";
    const fs::path windows_export_verifier_path =
        source_root / "scripts" / "verify_windows_runtime_exports.py";
    if (!fs::exists(runtime_def_path) || !fs::exists(windows_export_verifier_path)) {
        return fail("Windows runtime export contracts not found under source root");
    }

    const std::string workflow = read_file(workflow_path);
    const std::string package_verifier = read_file(package_verifier_path);
    const std::string repl_jit = read_file(repl_jit_path);
    const std::string jit_target_config = read_file(jit_target_config_path);
    const std::string jit_coff_memory_manager =
        read_file(jit_coff_memory_manager_path);
    const std::string tensor_codegen = read_file(tensor_codegen_path);
    const std::string portable_stdlib_verifier =
        read_file(portable_stdlib_verifier_path);
    const std::string runtime_def = read_file(runtime_def_path);
    const std::string windows_export_verifier =
        read_file(windows_export_verifier_path);
    bool ok = true;

    ok = ok &&
         expect_contains(workflow, "name: Release",
                         "release workflow is named") &&
         expect_contains(workflow, "tags:\n      - 'v*'",
                         "release workflow only runs for version tags") &&
         expect_contains(workflow, "workflow_dispatch:",
                         "release workflow supports a manual dry run") &&
         expect_contains(workflow, "candidate_tag:",
                         "manual dry run requires an explicit candidate tag label") &&
         expect_contains(workflow,
                         "RELEASE_TAG: ${{ github.event_name == 'workflow_dispatch' && inputs.candidate_tag || github.ref_name }}",
                         "tag and dry-run asset naming share one release label") &&
         expect_contains(workflow, "permissions:\n  contents: write",
                         "release workflow can publish GitHub releases") &&
         expect_contains(workflow, "cancel-in-progress: false",
                         "release workflow does not cancel an in-flight tag build") &&
         expect_contains(workflow, "unix-release-matrix:",
                         "release workflow has Unix asset matrix") &&
         expect_contains(workflow, "windows-release-matrix:",
                         "release workflow has Windows asset matrix") &&
         expect_contains(workflow, "publish-release:",
                         "release workflow has publish job") &&
         expect_contains(workflow, "needs:\n      - unix-release-matrix\n      - windows-release-matrix",
                         "publish job waits for both platform matrices") &&
         expect_contains(workflow, "pattern: release-asset-*",
                         "publish job downloads all packaged assets by prefix") &&
         expect_contains(workflow, "merge-multiple: true",
                         "publish job flattens downloaded assets") &&
         expect_contains(workflow, "Validate Release Asset Set",
                         "release workflow validates the final asset set") &&
         expect_contains(workflow,
                         "asset_count=\"$(find . -maxdepth 1 -type f ! -name SHA256SUMS.txt | wc -l | tr -d ' ')\"",
                         "release workflow counts non-checksum assets") &&
         expect_contains(workflow, "Expected ${#expected[@]} release assets",
                         "release workflow rejects missing or extra assets") &&
         expect_contains(workflow, "sha256sum * > SHA256SUMS.txt",
                         "release workflow writes checksums for all assets") &&
         expect_contains(workflow, "gh release view \"$RELEASE_TAG\"",
                         "release workflow checks for existing release") &&
         expect_contains(workflow,
                         "refusing to overwrite or append assets",
                         "release workflow refuses to append to existing releases") &&
         expect_contains(workflow, "gh release create \"$RELEASE_TAG\"",
                         "release workflow creates a tag-named GitHub release") &&
         expect_contains(workflow, "--verify-tag",
                         "release workflow verifies that the release tag exists") &&
         expect_contains(workflow, "Prepare Curated Release Notes",
                         "release workflow extracts the current curated notes") &&
         expect_contains(workflow,
                         "--notes-file \"$RUNNER_TEMP/release-notes.md\"",
                         "GitHub release body uses curated release notes") &&
         expect_contains(workflow, "Upload Validated Dry-Run Assets",
                         "manual dry run preserves its validated asset set") &&
         expect_contains(workflow,
                         "if: github.event_name == 'workflow_dispatch'",
                         "dry-run artifact upload is dispatch-only") &&
         expect_contains(workflow, "dist/*",
                         "release workflow uploads the complete dist directory");

    ok = ok &&
         expect_contains(workflow,
                         "archive_root=\"eshkol-${RELEASE_TAG}-${{ matrix.name }}\"",
                         "Unix release archives include the tag and platform name") &&
         expect_contains(workflow,
                         "tar -czf \"$RUNNER_TEMP/$archive_root.tar.gz\"",
                         "Unix release matrix produces tar.gz assets") &&
         expect_contains(workflow,
                         "path: ${{ runner.temp }}/eshkol-${{ env.RELEASE_TAG }}-${{ matrix.name }}.tar.gz",
                         "Unix upload path includes the tag and platform name") &&
         expect_contains(workflow,
                         "$archiveRoot = \"eshkol-$env:RELEASE_TAG-${{ matrix.name }}\"",
                         "Windows release archives include the tag and platform name") &&
         expect_contains(workflow,
                         "$archivePath = Join-Path $env:RUNNER_TEMP \"$archiveRoot.zip\"",
                         "Windows release matrix produces zip assets") &&
         expect_contains(workflow,
                         "path: ${{ runner.temp }}\\eshkol-${{ env.RELEASE_TAG }}-${{ matrix.name }}.zip",
                         "Windows upload path includes the tag and platform name") &&
         expect_contains(workflow,
                         "for doc in README.md LICENSE CHANGELOG.md RELEASE_NOTES.md; do",
                         "Unix archives include release notes") &&
         expect_contains(workflow,
                         "foreach ($doc in @('README.md', 'LICENSE', 'CHANGELOG.md', 'RELEASE_NOTES.md'))",
                         "Windows archives include release notes") &&
         expect_contains(workflow,
                         "python3 scripts/stage_third_party_licenses.py",
                         "Unix archives stage pinned third-party licenses") &&
         expect_contains(workflow,
                         "python3 scripts/stage_linux_runtime_dependencies.py",
                         "Linux archives stage their relocatable image-codec closure") &&
         expect_contains(workflow, "apt_install() {",
                         "Linux release dependency installation is retryable") &&
         expect_contains(workflow, "apt-get install -y -o Acquire::Retries=5",
                         "Linux release dependency downloads retry transient failures") &&
         expect_contains(workflow, "-DESHKOL_STDLIB_TARGET_CPU=generic",
                         "release stdlib is compiled for the portable baseline CPU") &&
         expect_contains(workflow, "scripts/verify_portable_stdlib.py",
                         "release matrices reject builder-specific stdlib bitcode") &&
         expect_contains(workflow, "Verify Windows cache-disabled JIT exports",
                         "Windows release matrix verifies its PE runtime export table") &&
         expect_contains(workflow, "scripts/verify_windows_runtime_exports.py",
                         "Windows release matrix runs the bounded-runtime export verifier") &&
         expect_contains(workflow,
                         "python scripts/stage_third_party_licenses.py",
                         "Windows archives stage pinned third-party licenses") &&
         expect_contains(workflow,
                         "Windows Error Reporting\\LocalDumps\\eshkol-run.exe",
                         "Windows package faults produce user-mode crash dumps") &&
         expect_contains(workflow,
                         "name: release-diagnostics-${{ matrix.name }}",
                         "Windows package failures retain target-specific diagnostics") &&
         expect_contains(workflow,
                         "${{ runner.temp }}\\eshkol-crash-dumps-${{ matrix.name }}\\*.dmp",
                         "Windows failure diagnostics include crash dumps") &&
         expect_contains(workflow,
                         "${{ runner.temp }}\\eshkol-${{ env.RELEASE_TAG }}-${{ matrix.name }}\\lib\\stdlib-jit-v4-*.o",
                         "Windows failure diagnostics include the emitted stdlib object") &&
         expect_contains(workflow, "if-no-files-found: warn",
                         "partial Windows failure evidence never masks the root failure");

    ok = ok &&
         expect_contains(package_verifier,
                         "jit_env[\"ESHKOL_JIT_CACHE\"] = \"0\"",
                         "package verifier exercises cache-disabled JIT") &&
         expect_contains(package_verifier,
                         "Module not found:",
                         "package verifier rejects missing installed modules") &&
         expect_contains(package_verifier,
                         "cache-disabled package JIT reported a module-loading failure",
                         "package verifier fails closed on JIT module diagnostics");

    ok = ok &&
         expect_contains(repl_jit,
                         "static void configure_jit_target_machine_builder(",
                         "LLJIT and cached stdlib emission share one target configuration") &&
         expect_contains(repl_jit,
                         "? llvm::CodeModel::Small\n                             : llvm::CodeModel::Large",
                         "Windows ARM64 selects the SEH-correct Small code model") &&
         expect_contains(repl_jit,
                         "triple.getObjectFormat() == llvm::Triple::COFF",
                         "JIT code-model exception is restricted to COFF") &&
         expect_contains(repl_jit,
                         "builder.getOptions().FunctionSections = true;",
                         "AArch64 JIT emission isolates functions for same-section veneers") &&
         expect_contains(repl_jit,
                         "builder.getOptions().DataSections = true;",
                         "AArch64 JIT emission isolates data for safe pruning") &&
         expect_contains(repl_jit, "stdlib-jit-v4-",
                         "stdlib JIT object cache version changes with external-data lowering") &&
         expect_contains(repl_jit,
                         "prepare_jit_module_for_target(*module, jit_->getTargetTriple());",
                         "live REPL modules receive the Windows ARM64 data-reach contract") &&
         expect_contains(repl_jit,
                         "prepare_jit_module_for_target(\n                                    **emit_mod, (*emit_tm)->getTargetTriple());",
                         "cached stdlib object emission receives the data-reach contract") &&
         expect_contains(repl_jit,
                         "prepare_jit_module_for_target(\n                    *stdlib_module, jit_->getTargetTriple());",
                         "IR stdlib loading receives the data-reach contract") &&
         expect_contains(jit_target_config,
                         "global.setDLLStorageClass(llvm::GlobalValue::DLLImportStorageClass);",
                         "Windows ARM64 external data uses RuntimeDyld COFF import cells") &&
         expect_contains(jit_target_config,
                         "if (!global.isDeclaration()",
                         "JIT-owned data definitions are not rewritten as imports") &&
         expect_contains(repl_jit,
                         "std::make_unique<\n                                eshkol::CoLocatedSectionMemoryManager>()",
                         "Windows ARM64 LLJIT installs the co-located RuntimeDyld arena") &&
         expect_contains(repl_jit,
                         "setOverrideObjectFlagsWithResponsibilityFlags(true);",
                         "custom COFF object layer preserves responsibility flags") &&
         expect_contains(repl_jit,
                         "setAutoClaimResponsibilityForObjectSymbols(true);",
                         "custom COFF object layer preserves automatic symbol claims") &&
         expect_contains(jit_coff_memory_manager,
                         "needsToReserveAllocationSpace()",
                         "RuntimeDyld computes complete per-object section sizes") &&
         expect_contains(jit_coff_memory_manager,
                         "kMaximumCodeSpan",
                         "co-located arena enforces the Branch26 code bound") &&
         expect_contains(jit_coff_memory_manager,
                         "kMaximumArenaSpan",
                         "co-located arena enforces the Small-model data bound") &&
         expect_contains(tensor_codegen,
                         "std::strcmp(target_cpu, \"generic\") == 0",
                         "generic stdlib codegen does not inherit the builder CPU") &&
         expect_contains(tensor_codegen,
                         "return 2;",
                         "generic stdlib tensor vectors use the 128-bit baseline") &&
         expect_contains(portable_stdlib_verifier,
                         "fixed double vector wider than the 128-bit release baseline",
                         "portable stdlib validator rejects builder-wide fixed vectors");

    if (count_occurrences(repl_jit,
                          "configure_jit_target_machine_builder(*") != 2) {
        std::cerr << "LLJIT and cached stdlib object emission must both use the shared target configuration"
                  << std::endl;
        ok = false;
    }

    ok = ok &&
         expect_contains(repl_jit, "ADD_DATA_SYMBOL(__ad_tower_active);",
                         "JIT explicitly registers Taylor-tower active state") &&
         expect_contains(repl_jit, "ADD_DATA_SYMBOL(__ad_tower_order);",
                         "JIT explicitly registers Taylor-tower order state") &&
         expect_contains(runtime_def, "__ad_tower_(active|order)$",
                         "bounded PE export generator includes Taylor-tower data") &&
         expect_contains(windows_export_verifier, "\"__ad_tower_active\"",
                         "Windows package export verifier requires Taylor active state") &&
         expect_contains(windows_export_verifier, "\"__ad_tower_order\"",
                         "Windows package export verifier requires Taylor order state");

    const std::vector<ReleaseAsset> expected_assets = {
        {"linux-x64-lite", "tar.gz"},
        {"linux-arm64-lite", "tar.gz"},
        {"linux-x64-xla", "tar.gz"},
        {"linux-arm64-xla", "tar.gz"},
        {"linux-x64-cuda", "tar.gz"},
        {"linux-arm64-cuda", "tar.gz"},
        {"macos-arm64-lite", "tar.gz"},
        {"macos-x64-lite", "tar.gz"},
        {"macos-arm64-xla", "tar.gz"},
        {"macos-x64-xla", "tar.gz"},
        {"windows-x64-lite", "zip"},
        {"windows-arm64-lite", "zip"},
        {"windows-x64-xla", "zip"},
        {"windows-arm64-xla", "zip"},
        {"windows-x64-cuda", "zip"},
        {"windows-arm64-cuda", "zip"},
    };

    for (const ReleaseAsset& asset : expected_assets) {
        const std::string matrix_name = std::string("- name: ") + asset.name;
        const std::string expected_filename =
            std::string("\"eshkol-${RELEASE_TAG}-") + asset.name + "." +
            asset.extension + "\"";
        ok = ok &&
             expect_contains(workflow, matrix_name,
                             std::string("release matrix includes ") + asset.name) &&
             expect_contains(workflow, expected_filename,
                             std::string("validated asset list includes ") + asset.name);
    }

    const std::size_t expected_block_begin = workflow.find("          expected=(\n");
    const std::size_t expected_block_end =
        expected_block_begin == std::string::npos
            ? std::string::npos
            : workflow.find("\n          )", expected_block_begin);
    const std::string expected_block =
        expected_block_begin == std::string::npos ||
                expected_block_end == std::string::npos
            ? std::string()
            : workflow.substr(expected_block_begin,
                              expected_block_end - expected_block_begin);
    const std::size_t validated_asset_count =
        count_occurrences(expected_block, "\"eshkol-${RELEASE_TAG}-");
    if (validated_asset_count != expected_assets.size()) {
        std::cerr << "expected " << expected_assets.size()
                  << " validated release asset names, found "
                  << validated_asset_count << std::endl;
        ok = false;
    }

    ok = ok &&
         expect_not_contains(workflow, "gh release upload",
                             "release workflow should not append assets to an existing release") &&
         expect_not_contains(workflow, "--generate-notes",
                             "release workflow should not replace curated notes with generated notes") &&
         expect_not_contains(workflow, "--clobber",
                             "release workflow should not overwrite release assets") &&
         expect_not_contains(workflow, "cancel-in-progress: true",
                             "release workflow should not cancel active tag builds");

    if (count_occurrences(workflow, "if: github.event_name == 'push'") < 2) {
        std::cerr << "publish and Homebrew jobs must both be disabled during manual dry runs"
                  << std::endl;
        ok = false;
    }

    if (count_occurrences(workflow, "uses: actions/checkout@v6") < 3) {
        std::cerr << "all build matrices and the publish job must checkout the tagged source"
                  << std::endl;
        ok = false;
    }

    if (count_occurrences(workflow, "-DESHKOL_STDLIB_TARGET_CPU=generic") != 2) {
        std::cerr << "Unix and Windows release matrices must both select the generic stdlib CPU"
                  << std::endl;
        ok = false;
    }

    if (count_occurrences(workflow, "-DESHKOL_STDLIB_TARGET_FEATURES=") != 2) {
        std::cerr << "Unix and Windows release matrices must both clear builder CPU features"
                  << std::endl;
        ok = false;
    }

    if (count_occurrences(workflow, "scripts/verify_portable_stdlib.py") != 2) {
        std::cerr << "Unix and Windows release matrices must both verify stdlib portability"
                  << std::endl;
        ok = false;
    }

    if (!ok) {
        return 1;
    }

    std::cout << "PASS" << std::endl;
    return 0;
}
