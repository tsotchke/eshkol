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
    const fs::path ci_workflow_path =
        source_root / ".github" / "workflows" / "ci.yml";
    const fs::path cmake_path = source_root / "CMakeLists.txt";
    const fs::path gpu_backend_verifier_path =
        source_root / "scripts" / "verify_gpu_backend.py";
    if (!fs::exists(workflow_path) || !fs::exists(ci_workflow_path) ||
        !fs::exists(cmake_path) || !fs::exists(gpu_backend_verifier_path)) {
        return fail("release/CI GPU contracts not found under source root");
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
    const fs::path platform_runtime_path =
        source_root / "lib" / "core" / "platform_runtime.cpp";
    const fs::path portable_stdlib_verifier_path =
        source_root / "scripts" / "verify_portable_stdlib.py";
    if (!fs::exists(repl_jit_path) ||
        !fs::exists(jit_target_config_path) ||
        !fs::exists(jit_coff_memory_manager_path) ||
        !fs::exists(tensor_codegen_path) ||
        !fs::exists(platform_runtime_path) ||
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
    const std::string ci_workflow = read_file(ci_workflow_path);
    const std::string cmake = read_file(cmake_path);
    const std::string gpu_backend_verifier =
        read_file(gpu_backend_verifier_path);
    const std::string package_verifier = read_file(package_verifier_path);
    const std::string repl_jit = read_file(repl_jit_path);
    const std::string jit_target_config = read_file(jit_target_config_path);
    const std::string jit_coff_memory_manager =
        read_file(jit_coff_memory_manager_path);
    const std::string tensor_codegen = read_file(tensor_codegen_path);
    const std::string platform_runtime = read_file(platform_runtime_path);
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
                         "partial Windows failure evidence never masks the root failure") &&
         expect_contains(workflow, "CUDA_VERSION: '12.4.1'",
                         "release CUDA toolkit version is pinned") &&
         expect_contains(workflow,
                         "cuda-nvcc-${CUDA_PACKAGE_SERIES}",
                         "Linux release CUDA lanes install the NVIDIA compiler") &&
         expect_contains(workflow,
                         "libcublas-dev-${CUDA_PACKAGE_SERIES}",
                         "Linux release CUDA lanes install cuBLAS development files") &&
         expect_contains(workflow,
                         "Jimver/cuda-toolkit@3d45d157f327c09c04b50ee6ccdea2d9d017ec76",
                         "Windows x64 release CUDA setup is commit-pinned") &&
         expect_contains(workflow,
                         "sub-packages: '[\"cudart\", \"cublas\", \"cublas_dev\", \"nvcc\", \"visual_studio_integration\"]'",
                         "Windows release installs only CUDA 12.4 subpackage names") &&
         expect_contains(ci_workflow,
                         "sub-packages: '[\"cudart\", \"cublas\", \"cublas_dev\", \"nvcc\", \"visual_studio_integration\"]'",
                         "Windows CI installs only CUDA 12.4 subpackage names") &&
         expect_contains(workflow, "'-G', 'Ninja Multi-Config'",
                         "release Windows CUDA avoids the incompatible Visual Studio CUDA MSBuild path") &&
         expect_contains(ci_workflow, "'-G', 'Ninja Multi-Config'",
                         "CI Windows CUDA avoids the incompatible Visual Studio CUDA MSBuild path") &&
         expect_contains(workflow, "-vcvars_ver=14.29",
                         "release Windows CUDA uses an nvcc-supported MSVC host") &&
         expect_contains(ci_workflow, "-vcvars_ver=14.29",
                         "CI Windows CUDA uses an nvcc-supported MSVC host") &&
         expect_contains(workflow, "$cudaHostCxx.Replace('\\', '/')",
                         "release normalizes the CUDA host path before CMake forwards it to nvcc") &&
         expect_contains(ci_workflow, "$cudaHostCxx.Replace('\\', '/')",
                         "CI normalizes the CUDA host path before CMake forwards it to nvcc") &&
         expect_contains(workflow, "-DCMAKE_CUDA_HOST_COMPILER=$cudaHostCxxCMake",
                         "release Windows CUDA binds nvcc to the normalized supported host compiler") &&
         expect_contains(ci_workflow, "-DCMAKE_CUDA_HOST_COMPILER=$cudaHostCxxCMake",
                         "CI Windows CUDA binds nvcc to the normalized supported host compiler") &&
         expect_not_contains(workflow, "sub-packages: '[\"crt\",",
                             "release does not pass a newer-toolkit crt subpackage to CUDA 12.4") &&
         expect_not_contains(ci_workflow, "sub-packages: '[\"crt\",",
                             "CI does not pass a newer-toolkit crt subpackage to CUDA 12.4") &&
         expect_not_contains(workflow, "\"nvcc\", \"nvvm\",",
                             "release lets CUDA 12.4 nvcc supply its NVVM internals") &&
         expect_not_contains(ci_workflow, "\"nvcc\", \"nvvm\",",
                             "CI lets CUDA 12.4 nvcc supply its NVVM internals") &&
         expect_not_contains(workflow,
                             "$toolset = \"ClangCL,cuda=$env:CUDA_PATH\"",
                             "release does not combine ClangCL with CUDA MSBuild targets") &&
         expect_not_contains(ci_workflow,
                             "$toolset = \"ClangCL,cuda=$env:CUDA_PATH\"",
                             "CI does not combine ClangCL with CUDA MSBuild targets") &&
         expect_contains(workflow,
                         "-DESHKOL_REQUIRE_GPU_BACKEND=${{ matrix.gpu_enabled }}",
                         "release CUDA labels fail closed without a real backend") &&
         expect_contains(workflow,
                         "-DCUDAToolkit_ROOT=/usr/local/cuda-12.4",
                         "release configure passes a Bash-3-safe CUDA toolkit hint") &&
         expect_contains(ci_workflow,
                         "-DCUDAToolkit_ROOT=/usr/local/cuda-12.4",
                         "CI configure passes a Bash-3-safe CUDA toolkit hint") &&
         expect_not_contains(workflow, "CUDA_FLAGS=()",
                             "release configure avoids empty arrays under Bash nounset") &&
         expect_not_contains(ci_workflow, "CUDA_FLAGS=()",
                             "CI configure avoids empty arrays under Bash nounset") &&
         expect_contains(workflow, "scripts/verify_gpu_backend.py",
                         "release matrix verifies the resolved CUDA build graph") &&
         expect_contains(gpu_backend_verifier, "build_dir.rglob(\"*.ninja\")",
                         "CUDA verifier follows Ninja Multi-Config implementation graphs") &&
         expect_contains(ci_workflow, "- name: windows-x64-cuda",
                         "CI compiles the supported Windows x64 CUDA target") &&
         expect_not_contains(ci_workflow, "- name: windows-arm64-cuda",
                             "CI does not counterfeit unsupported Windows ARM64 CUDA") &&
         expect_not_contains(workflow, "- name: windows-arm64-cuda",
                             "release matrix omits unsupported Windows ARM64 CUDA") &&
         expect_contains(cmake, "option(ESHKOL_REQUIRE_GPU_BACKEND",
                         "CMake exposes a fail-closed GPU backend contract") &&
         expect_contains(cmake,
                         "Refusing to compile the CPU stub as a GPU-labeled build.",
                         "CMake rejects CUDA labels that resolve to the CPU stub") &&
         expect_contains(cmake,
                         "Selecting only CMAKE_CUDA_HOST_COMPILER is unsafe",
                         "CUDA rejects mixed GNU host/link toolchains") &&
         expect_contains(cmake,
                         "__ESHKOL_CUDA_LIB__/${_cuda_runtime_name}",
                         "generated links store CUDA logical names, not builder paths") &&
         expect_contains(platform_runtime,
                         "cuda_runtime_link_args(cuda_libraries)",
                         "CUDA logical names resolve against the consumer toolkit") &&
         expect_contains(platform_runtime,
                         "CUDAToolkit_ROOT\", \"CUDA_HOME\", \"CUDA_PATH",
                         "consumer CUDA resolution honors standard root overrides") &&
         expect_contains(platform_runtime,
                         "root / \"lib\" / \"x86_64-linux-gnu\"",
                         "consumer CUDA resolution supports distro multiarch layouts") &&
         expect_contains(platform_runtime,
                         "\"-l:lib\" + library + \".so.\"",
                         "Linux CUDA links require the configured ABI-major soname") &&
         expect_contains(platform_runtime,
                         "major != ESHKOL_HOST_CUDA_MAJOR",
                         "Windows rejects discovered toolkits with the wrong major") &&
         expect_contains(platform_runtime,
                         "const std::string native_directory = directory.string();",
                         "Windows CUDA resolution avoids version-coupled generic-path STL helpers") &&
         expect_not_contains(platform_runtime,
                             "const std::string native_directory = directory.generic_string();",
                             "generated links do not inherit new MSVC generic-path helper imports") &&
         expect_contains(gpu_backend_verifier, "gpu_cuda_kernels.cu",
                         "GPU verifier requires compiled CUDA kernels") &&
         expect_contains(gpu_backend_verifier,
                         "for required_arch in (\"72\", \"86\")",
                         "CUDA assets cover Xavier and RTX-class GPUs") &&
         expect_contains(gpu_backend_verifier, "gpu_memory_stub.cpp",
                         "GPU verifier rejects the fallback stub");

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
