#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

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

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << std::endl;
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        return fail("usage: windows_suite_surface_test <source-root>");
    }

    const fs::path source_root = argv[1];
    const std::string cmake = read_file(source_root / "CMakeLists.txt");
    const fs::path script_path = source_root / "scripts" / "run_all_tests.ps1";
    if (!fs::exists(script_path)) {
        return fail("run_all_tests.ps1 not found under source root");
    }

    const std::string script = read_file(script_path);
    const std::string arena_header =
        read_file(source_root / "lib" / "core" / "arena_memory.h");
    const std::string runtime_arena_core =
        read_file(source_root / "lib" / "core" / "runtime_arena_core.cpp");
    const std::string runtime_regions =
        read_file(source_root / "lib" / "core" / "runtime_regions.cpp");
    const std::string eshkol_run =
        read_file(source_root / "exe" / "eshkol-run.cpp");
    const std::string eshkol_vm_stub =
        read_file(source_root / "lib" / "backend" / "eshkol_vm_stub.c");
    const std::string llvm_codegen =
        read_file(source_root / "lib" / "backend" / "llvm_codegen.cpp");
    const std::string parallel_llvm_codegen =
        read_file(source_root / "lib" / "backend" / "parallel_llvm_codegen.cpp");
    const std::string agent_capabilities_header =
        read_file(source_root / "inc" / "eshkol" / "agent_capabilities.h");
    bool ok = true;

    ok = ok &&
         expect_contains(script, "function Invoke-JitCacheSuite",
                         "Windows suite declares persistent JIT-cache runtime coverage") &&
         expect_contains(script, "$cold.StdErr -match '\\[jit-cache\\] store '",
                         "Windows suite requires a real cold cache store") &&
         expect_contains(script, "$warm.StdErr -match '\\[jit-cache\\] hit '",
                         "Windows suite requires a real warm cache hit") &&
         expect_contains(script, "$orphanTemps.Count -eq 0",
                         "Windows suite rejects orphaned JIT-cache temporaries") &&
         expect_contains(script, "$suiteResults += Invoke-JitCacheSuite",
                         "Windows modes execute persistent JIT-cache coverage") &&
         expect_contains(script, "function Get-RegularFileItem",
                         "Windows suite declares regular-file resolver") &&
         expect_contains(script, "Get-Item -LiteralPath $Path -Force -ErrorAction Stop",
                         "Windows suite resolves file candidates without wildcard expansion") &&
         expect_contains(script, "function Test-RegularFile",
                         "Windows suite declares regular-file predicate") &&
         expect_contains(script,
                         "($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0",
                         "Windows suite rejects reparse-point file artifacts") &&
         expect_contains(script, "function Remove-RegularFileIfPresent",
                         "Windows suite declares guarded output cleanup") &&
         expect_contains(script, "Remove-Item -LiteralPath $item.FullName -Force -ErrorAction Stop",
                         "Windows suite cleanup removes only the resolved regular file") &&
         expect_contains(script, "function Test-TransientProcessStartError",
                         "Windows suite classifies transient executable sharing violations") &&
         expect_contains(script, "$current.NativeErrorCode -eq 32",
                         "Windows suite recognizes sharing-violation process start failures") &&
         expect_contains(script, "function Start-ProcessWithTransientRetry",
                         "Windows suite retries transient process start failures") &&
         expect_contains(script, "function Start-RegularFileProcess",
                         "Windows suite declares guarded process launcher") &&
         expect_contains(script, "return Start-ProcessWithTransientRetry -Start",
                         "Windows suite guarded launcher uses transient start retry") &&
         expect_contains(script, "function Join-WindowsProcessArguments",
                         "Windows suite declares Windows PowerShell argument joiner") &&
         expect_contains(script, "$psi.GetType().GetProperty(\"ArgumentList\")",
                         "Windows suite detects ProcessStartInfo.ArgumentList before use") &&
         expect_contains(script, "$psi.Arguments = Join-WindowsProcessArguments -Arguments $Arguments",
                         "Windows suite falls back to quoted Arguments on Windows PowerShell") &&
         expect_contains(script, "Refusing to start missing or non-regular executable",
                         "Windows suite guarded launcher rejects non-regular executables") &&
         expect_contains(script, "Test-RegularFile -Path (Join-Path $binDir \"eshkol-run.exe\")",
                         "Windows suite build resolver requires a regular eshkol-run.exe") &&
         expect_contains(script, "Remove-RegularFileIfPresent -Path @($exePath, $OutputBase)",
                         "Windows suite compile cleanup rejects non-regular output artifacts") &&
         expect_contains(script, "Start-ProcessWithTransientRetry -Start {\n            [void]$process.Start()\n        }",
                         "Windows suite captured process launcher retries transient sharing violations") &&
         expect_contains(script,
                         "Success  = ($captured.ExitCode -eq 0 -and (Test-RegularFile -Path $exePath))",
                         "Windows suite compile success requires a regular executable") &&
         expect_contains(script, "Test-RegularFile -Path $compile.ExePath",
                         "Windows suite expected-error path checks regular executables") &&
         expect_contains(script, "Test-RegularFile -Path $llvmCommand.Source",
                         "Windows suite PATH llvm-config fallback rejects non-regular commands") &&
         expect_contains(script, "Test-RegularFile -Path $script:EshkolServer",
                         "Windows suite server runner requires a regular executable") &&
         expect_contains(script, "Start-RegularFileProcess -FilePath $script:EshkolServer",
                         "Windows suite starts the web server through the guarded launcher") &&
         expect_contains(script, "Test-RegularFile -Path $wasmPath",
                         "Windows suite WASM validation requires a regular output file") &&
         expect_contains(script, "Remove-RegularFileIfPresent -Path @($serverStdout, $serverStderr)",
                         "Windows suite server-log cleanup rejects non-regular artifacts") &&
         expect_contains(script, "Test-RegularFile -Path $xlaBin",
                         "Windows suite XLA binary probe requires a regular executable") &&
         expect_contains(script, "Test-RegularFile -Path $path",
                         "Windows suite build-artifact probes require regular files");

    ok = ok &&
         expect_not_contains(script, "Test-Path ",
                             "Windows suite should not use plain Test-Path artifact probes") &&
         expect_not_contains(script, "Remove-Item $",
                             "Windows suite should not remove variable paths without the guard helper") &&
         expect_not_contains(script, "$server = Start-Process",
                             "Windows suite should not start the web server without the guard helper");

    ok = ok &&
         expect_contains(cmake, "if(WIN32 AND ESHKOL_USING_MSVC)",
                         "Windows native dependency branch is explicit") &&
         expect_contains(cmake,
                         "GIT_TAG bc3b39870ecb690a623a3f49149a358b95c5781d",
                         "Windows Eigen provider is pinned by immutable commit") &&
         expect_contains(cmake, "set(ESHKOL_BLAS_USAGE_TARGET Eigen3::Eigen)",
                         "Windows BLAS compilation uses Eigen's native target") &&
         expect_contains(cmake, "add_compile_definitions(ESHKOL_BLAS_EIGEN)",
                         "Windows runtime selects the Eigen CBLAS implementation") &&
         expect_contains(cmake, "function(eshkol_export_host_agent_api target_name)",
                         "hosted VM has a cross-platform agent export contract") &&
         expect_contains(cmake, "${target_name}_agent_exports.def",
                         "Windows hosted VM uses a bounded agent export table") &&
         expect_contains(cmake, "eshkol_export_host_agent_api(${_eshkol_vm_host_target})",
                         "every standalone VM host publishes the agent API") &&
         expect_not_contains(cmake,
                             "ClangCL on\n"
                             "            # Windows needs OpenBLAS",
                             "Windows must not consume a MinGW OpenBLAS archive");

    ok = ok &&
         expect_contains(agent_capabilities_header,
                         "#  if defined(ESHKOL_AGENT_SHARED)",
                         "Windows DLL decoration is explicitly opt-in") &&
         expect_contains(agent_capabilities_header,
                         "#    define ESHKOL_AGENT_API",
                         "static Windows agent consumers use ordinary COFF symbols");

    ok = ok &&
         expect_contains(arena_header, "#if defined(_WIN32)",
                         "runtime weak-linkage macro has a Windows COFF branch") &&
         expect_contains(arena_header, "#define ESHKOL_RUNTIME_WEAK __declspec(selectany)",
                         "runtime weak-linkage macro uses COFF selectany on Windows") &&
         expect_contains(arena_header, "#define ESHKOL_RUNTIME_WEAK __attribute__((weak))",
                         "runtime weak-linkage macro preserves ELF/Mach-O weak defaults") &&
         expect_contains(runtime_arena_core,
                         "ESHKOL_RUNTIME_WEAK int32_t __eshkol_argc = 0;",
                         "runtime argc default uses portable weak-linkage macro") &&
         expect_contains(runtime_arena_core,
                         "ESHKOL_RUNTIME_WEAK char** __eshkol_argv = nullptr;",
                         "runtime argv default uses portable weak-linkage macro") &&
         expect_contains(runtime_regions,
                         "ESHKOL_RUNTIME_WEAK arena_t* __global_arena = nullptr;",
                         "runtime arena default uses portable weak-linkage macro") &&
         expect_not_contains(runtime_arena_core,
                             "__attribute__((weak)) int32_t __eshkol_argc",
                             "runtime argc default should not use direct weak attribute") &&
         expect_not_contains(runtime_arena_core,
                             "__attribute__((weak)) char** __eshkol_argv",
                             "runtime argv default should not use direct weak attribute") &&
         expect_not_contains(runtime_regions,
                             "__attribute__((weak)) arena_t* __global_arena",
                             "runtime arena default should not use direct weak attribute");

    ok = ok &&
         expect_contains(eshkol_run, "#ifdef __MINGW32__",
                         "eshkol-run generated link path has a MinGW branch") &&
         expect_contains(eshkol_run,
                         "#ifdef _WIN32\n        link_args.emplace_back(\"-fuse-ld=lld\");\n#ifdef __MINGW32__",
                         "eshkol-run generated link path uses lld on Windows") &&
         expect_contains(eshkol_run, "\"-Wl,--stack,536870912\"",
                         "eshkol-run generated link path uses GNU PE stack flag on MinGW") &&
         expect_contains(eshkol_run, "\"/STACK:536870912\"",
                         "eshkol-run generated link path preserves MSVC stack flag") &&
         expect_contains(llvm_codegen, "\"-Wl,--whole-archive\"",
                         "LLVM executable link path keeps ELF whole-archive support") &&
         expect_contains(llvm_codegen,
                         "#elif defined(_WIN32)\n        link_args.emplace_back(\"-fuse-ld=lld\");\n#ifdef __MINGW32__",
                         "LLVM executable link path uses lld on Windows") &&
         expect_contains(llvm_codegen, "\"-Wl,--stack,536870912\"",
                         "LLVM executable link path uses GNU PE stack flag on MinGW") &&
         expect_contains(llvm_codegen, "\"/STACK:536870912\"",
                         "LLVM executable link path preserves MSVC stack flag") &&
         expect_contains(llvm_codegen, "link_args.emplace_back(runtime_lib_path.generic_string());",
                         "LLVM executable link path links Windows runtime archive selectively") &&
         expect_contains(llvm_codegen,
                         "arena_use_external_only = true;",
                         "LLVM codegen uses runtime-owned arena globals on Windows") &&
         expect_contains(llvm_codegen,
                         "use_external_only = true;",
                         "LLVM codegen uses runtime-owned command-line globals on Windows") &&
         expect_contains(llvm_codegen,
                         "? GlobalValue::LinkOnceODRLinkage",
                         "shared-library mode keeps COFF stdlib symbols as COMDAT definitions") &&
         expect_not_contains(llvm_codegen,
                             "COFF treats linkonce_odr as an ODR contract",
                             "shared-library mode should not emit COFF weak external aliases") &&
         expect_not_contains(llvm_codegen,
                             "\"/WHOLEARCHIVE:\" + runtime_lib_path.generic_string()",
                             "LLVM executable link path should not force-load compiler archive on Windows") &&
         expect_not_contains(cmake,
                             "foreach(_llvm_runtime_lib IN LISTS LLVM_LIBS_LIST LLVM_SYSTEM_LIBS_LIST)",
                             "generated Windows binaries should not inherit compiler LLVM link libraries");

    ok = ok &&
         expect_contains(parallel_llvm_codegen, "static llvm::GlobalValue::LinkageTypes workerInitLinkage()",
                         "parallel worker init linkage is centralized") &&
         expect_contains(parallel_llvm_codegen, "return llvm::GlobalValue::InternalLinkage;",
                         "parallel worker init uses module-local linkage on Windows") &&
         expect_contains(parallel_llvm_codegen, "init_type, workerInitLinkage(),",
                         "parallel worker init symbol uses platform-specific linkage");

    ok = ok &&
         expect_contains(eshkol_vm_stub, "int eshkol_vm_default_load_options",
                         "native Windows VM stub exports default load options") &&
         expect_contains(eshkol_vm_stub, "EshkolVmHandle* eshkol_vm_load_chunk_with_options",
                         "native Windows VM stub exports policy-aware load entry") &&
         expect_contains(eshkol_vm_stub, "int eshkol_vm_has_function",
                         "native Windows VM stub exports function lookup") &&
         expect_contains(eshkol_vm_stub, "void eshkol_vm_destroy",
                         "native Windows VM stub exports VM destruction") &&
         expect_contains(eshkol_vm_stub,
                         "out->required_function_metadata = NULL;",
                         "native Windows VM stub initializes metadata requirements") &&
         expect_contains(eshkol_vm_stub,
                         "out->required_function_metadata_count = 0;",
                         "native Windows VM stub initializes metadata requirement count") &&
         expect_contains(eshkol_vm_stub,
                         "return NULL;",
                         "native Windows VM stub keeps VM loading disabled");

    if (!ok) {
        return 1;
    }

    std::cout << "PASS" << std::endl;
    return 0;
}
