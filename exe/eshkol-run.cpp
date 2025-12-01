/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */
#include <eshkol/eshkol.h>
#include <eshkol/logger.h>

#include <eshkol/llvm_backend.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <getopt.h>
#include <unistd.h>
#include <filesystem>
#include <mach-o/dyld.h>  // For _NSGetExecutablePath on macOS

#include <string>
#include <vector>
#include <set>
#include <sstream>

static struct option long_options[] = {
    {"help", no_argument, nullptr, 'h'},
    {"debug", no_argument, nullptr, 'd'},
    {"dump-ast", no_argument, nullptr, 'a'},
    {"dump-ir", no_argument, nullptr, 'i'},
    {"output", required_argument, nullptr, 'o'},
    {"compile-only", no_argument, nullptr, 'c'},
    {"shared-lib", no_argument, nullptr, 's'},
    {"lib", required_argument, nullptr, 'l'},
    {"lib-path", required_argument, nullptr, 'L'},
    {"no-stdlib", no_argument, nullptr, 'n'},
    {0, 0, 0, 0}
};

// Set to track imported files (prevent circular imports)
static std::set<std::string> imported_files;

// Forward declarations
static void process_imports(std::vector<eshkol_ast_t>& asts, const std::string& base_dir, bool debug_mode);
static void load_file_asts(const std::string& filepath, std::vector<eshkol_ast_t>& asts, bool debug_mode);

static void print_help(int x = 0)
{
    printf(
        "Usage: eshkol-run [options] <input.esk|input.o> [input.esk|input.o]\n\n"
        "\t--help:[-h] = Print this help message.\n"
        "\t--debug:[-d] = Debugging information added inside the program.\n"
        "\t--dump-ast:[-a] = Dumps the AST into a .ast file.\n"
        "\t--dump-ir:[-i] = Dumps the IR into a .ll file.\n"
        "\t--output:[-o] = Outputs into a binary file.\n"
        "\t--compile-only:[-c] = Compiles into an intermediate object file.\n"
        "\t--shared-lib:[-s] = Compiles it into a shared library.\n"
        "\t--lib:[-l] = Links a shared library to the resulting executable.\n"
        "\t--lib-path:[-L] = Adds a directory to the library search path.\n"
        "\t--no-stdlib:[-n] = Do not auto-load the standard library.\n\n"
        "This is an early developer release (%s) of the Eshkol Compiler/Interpreter.\n",
        ESHKOL_VER
    );
    exit(x);
}

// Load ASTs from a file
static void load_file_asts(const std::string& filepath, std::vector<eshkol_ast_t>& asts, bool debug_mode)
{
    // Check file exists first (required before calling canonical)
    if (!std::filesystem::exists(filepath)) {
        eshkol_error("File not found: %s", filepath.c_str());
        return;
    }

    // Normalize path to prevent duplicate imports
    std::string normalized_path = std::filesystem::canonical(filepath).string();

    // Skip if already imported
    if (imported_files.count(normalized_path)) {
        if (debug_mode) {
            eshkol_debug("Skipping already imported file: %s", normalized_path.c_str());
        }
        return;
    }
    imported_files.insert(normalized_path);

    std::ifstream read_file(filepath);
    if (!read_file.is_open()) {
        eshkol_error("Failed to open file: %s", filepath.c_str());
        return;
    }

    if (debug_mode) {
        eshkol_info("Loading file: %s", filepath.c_str());
    }

    eshkol_ast_t ast = eshkol_parse_next_ast(read_file);
    while (ast.type != ESHKOL_INVALID) {
        if (debug_mode) {
            printf("\n=== AST Debug Output ===\n");
            eshkol_ast_pretty_print(&ast, 0);
            printf("========================\n\n");
        }
        asts.push_back(ast);
        ast = eshkol_parse_next_ast(read_file);
    }

    read_file.close();
}

// Process import statements in ASTs and load referenced files
static void process_imports(std::vector<eshkol_ast_t>& asts, const std::string& base_dir, bool debug_mode)
{
    std::vector<eshkol_ast_t> new_asts;
    std::vector<eshkol_ast_t> imported_asts;

    for (auto& ast : asts) {
        if (ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_IMPORT_OP) {
            // This is an import statement
            std::string import_path = ast.operation.import_op.path;

            // Resolve relative paths
            std::filesystem::path resolved_path;
            if (import_path[0] == '/') {
                // Absolute path
                resolved_path = import_path;
            } else {
                // Relative to base directory
                resolved_path = std::filesystem::path(base_dir) / import_path;
            }

            if (!std::filesystem::exists(resolved_path)) {
                eshkol_error("Import file not found: %s", resolved_path.c_str());
                continue;
            }

            // Load the imported file
            std::vector<eshkol_ast_t> file_asts;
            load_file_asts(resolved_path.string(), file_asts, debug_mode);

            // Recursively process imports in the loaded file
            std::string import_dir = resolved_path.parent_path().string();
            process_imports(file_asts, import_dir, debug_mode);

            // Add imported ASTs (definitions from the imported file)
            for (auto& imported_ast : file_asts) {
                imported_asts.push_back(imported_ast);
            }

            // Don't add the import statement itself to new_asts
        } else {
            // Not an import, keep the AST
            new_asts.push_back(ast);
        }
    }

    // Prepend imported ASTs before the current file's ASTs
    asts.clear();
    for (auto& ast : imported_asts) {
        asts.push_back(ast);
    }
    for (auto& ast : new_asts) {
        asts.push_back(ast);
    }
}

// Find the stdlib.esk file
static std::string find_stdlib()
{
    // Check common locations
    std::vector<std::string> stdlib_paths = {
        // Relative to current directory (development)
        "lib/stdlib.esk",
        "../lib/stdlib.esk",
        // Installation paths
        "/usr/local/share/eshkol/stdlib.esk",
        "/usr/share/eshkol/stdlib.esk",
    };

    // Also check relative to the executable
    char exe_path[4096];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len == -1) {
        // Try macOS method
        uint32_t size = sizeof(exe_path);
        if (_NSGetExecutablePath(exe_path, &size) == 0) {
            std::filesystem::path exe_dir = std::filesystem::path(exe_path).parent_path();
            stdlib_paths.insert(stdlib_paths.begin(), (exe_dir / "../lib/stdlib.esk").string());
            stdlib_paths.insert(stdlib_paths.begin(), (exe_dir / "stdlib.esk").string());
        }
    } else {
        exe_path[len] = '\0';
        std::filesystem::path exe_dir = std::filesystem::path(exe_path).parent_path();
        stdlib_paths.insert(stdlib_paths.begin(), (exe_dir / "../lib/stdlib.esk").string());
        stdlib_paths.insert(stdlib_paths.begin(), (exe_dir / "stdlib.esk").string());
    }

    for (const auto& path : stdlib_paths) {
        if (std::filesystem::exists(path)) {
            return std::filesystem::canonical(path).string();
        }
    }

    return "";  // Not found
}

int main(int argc, char **argv)
{
    int ch = 0;

    uint8_t debug_mode = 0;
    uint8_t dump_ast = 0;
    uint8_t dump_ir = 0;
    uint8_t compile_only = 0;
    uint8_t no_stdlib = 0;

    std::vector<char*> source_files;
    std::vector<char*> compiled_files;
    std::vector<char*> linked_libs;
    std::vector<char*> lib_paths;

    std::vector<eshkol_ast_t> asts;

    char *output = nullptr;

    if (argc == 1) print_help(1);

    while ((ch = getopt_long(argc, argv, "hdaio:csl:L:n", long_options, nullptr)) != -1) {
        switch (ch) {
        case 'h':
            print_help(0);
            break;
        case 'd':
            debug_mode = 1;
            eshkol_set_logger_level(ESHKOL_DEBUG);
            break;
        case 'a':
            dump_ast = 1;
            break;
        case 'i':
            dump_ir = 1;
            break;
        case 'o':
            output = optarg;
            break;
        case 'c':
            compile_only = 1;
            break;
        case 's':
            // TODO: Implement shared library support
            eshkol_warn("Shared library support not yet implemented");
            break;
        case 'l':
            linked_libs.push_back(optarg);
            break;
        case 'L':
            lib_paths.push_back(optarg);
            break;
        case 'n':
            no_stdlib = 1;
            break;
        default:
            print_help(1);
        }
    }

    if (optind == argc) print_help(1);

    // Helper function to check string suffix (C++17 compatible)
    auto ends_with = [](const std::string& str, const std::string& suffix) {
        if (suffix.size() > str.size()) return false;
        return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
    };

    for (; optind < argc; ++optind) {
        std::string tmp = (const char*) argv[optind];
        if (ends_with(tmp, ".esk"))
            source_files.push_back(argv[optind]);
        else if (ends_with(tmp, ".o"))
            compiled_files.push_back(argv[optind]);
    }

    // Load standard library first (unless --no-stdlib)
    if (!no_stdlib) {
        std::string stdlib_path = find_stdlib();
        if (!stdlib_path.empty()) {
            if (debug_mode) {
                eshkol_info("Loading standard library from: %s", stdlib_path.c_str());
            }
            load_file_asts(stdlib_path, asts, debug_mode);
        } else {
            if (debug_mode) {
                eshkol_warn("Standard library not found, proceeding without it");
            }
        }
    }

    // Load user source files
    for (const auto &source_file : source_files) {
        // Get base directory for resolving imports
        std::filesystem::path source_path(source_file);
        std::string base_dir = source_path.parent_path().string();
        if (base_dir.empty()) base_dir = ".";

        // Load the file
        load_file_asts(source_file, asts, debug_mode);

        // Process imports in the loaded ASTs
        process_imports(asts, base_dir, debug_mode);
    }

    // Handle AST dumping if requested
    if (dump_ast && !source_files.empty()) {
        std::string ast_filename;
        if (output) {
            ast_filename = std::string(output) + ".ast";
        } else {
            std::string first_source = source_files[0];
            size_t last_slash = first_source.find_last_of("/\\");
            size_t last_dot = first_source.find_last_of('.');
            std::string base_name;
            if (last_slash != std::string::npos) {
                if (last_dot != std::string::npos && last_dot > last_slash) {
                    base_name = first_source.substr(last_slash + 1, last_dot - last_slash - 1);
                } else {
                    base_name = first_source.substr(last_slash + 1);
                }
            } else {
                if (last_dot != std::string::npos) {
                    base_name = first_source.substr(0, last_dot);
                } else {
                    base_name = first_source;
                }
            }
            ast_filename = base_name + ".ast";
        }

        std::ofstream ast_file(ast_filename);
        if (ast_file.is_open()) {
            for (const auto& ast : asts) {
                ast_file << "=== AST Node ===\n";
                eshkol_ast_pretty_print(&ast, 0);
                ast_file << "=================\n\n";
            }
            ast_file.close();
            eshkol_info("AST dumped to: %s", ast_filename.c_str());
        } else {
            eshkol_error("Failed to open AST file: %s", ast_filename.c_str());
        }
    }

    // Generate LLVM IR if we have ASTs and need compilation or IR output
    // Default behavior is to compile to executable unless only AST dump is requested
    if (!asts.empty()) {
        // Determine module name from first source file or use default
        std::string module_name = "eshkol_module";
        if (!source_files.empty()) {
            std::string source_file = source_files[0];
            size_t last_slash = source_file.find_last_of("/\\");
            size_t last_dot = source_file.find_last_of('.');
            if (last_slash != std::string::npos) {
                if (last_dot != std::string::npos && last_dot > last_slash) {
                    module_name = source_file.substr(last_slash + 1, last_dot - last_slash - 1);
                } else {
                    module_name = source_file.substr(last_slash + 1);
                }
            } else {
                if (last_dot != std::string::npos) {
                    module_name = source_file.substr(0, last_dot);
                } else {
                    module_name = source_file;
                }
            }
        }
        
        eshkol_info("Generating LLVM IR for module: %s", module_name.c_str());
        
        // Generate LLVM IR
        LLVMModuleRef llvm_module = eshkol_generate_llvm_ir(
            asts.data(), 
            asts.size(), 
            module_name.c_str()
        );
        
        if (!llvm_module) {
            eshkol_error("Failed to generate LLVM IR");
            return 1;
        }
        
        // Handle different output modes
        if (dump_ir) {
            // Dump IR to file
            std::string ir_filename;
            if (output) {
                ir_filename = std::string(output) + ".ll";
            } else {
                ir_filename = module_name + ".ll";
            }
            
            eshkol_info("Dumping LLVM IR to: %s", ir_filename.c_str());
            if (eshkol_dump_llvm_ir_to_file(llvm_module, ir_filename.c_str()) != 0) {
                eshkol_error("Failed to dump LLVM IR to file");
                eshkol_dispose_llvm_module(llvm_module);
                return 1;
            }
        }
        
        if (debug_mode) {
            eshkol_info("Generated LLVM IR:");
            eshkol_print_llvm_ir(llvm_module);
        }
        
        if (compile_only) {
            // Compile to object file
            std::string obj_filename;
            if (output) {
                obj_filename = std::string(output) + ".o";
            } else {
                obj_filename = module_name + ".o";
            }
            
            eshkol_info("Compiling to object file: %s", obj_filename.c_str());
            if (eshkol_compile_llvm_ir_to_object(llvm_module, obj_filename.c_str()) != 0) {
                eshkol_error("Object file compilation failed");
                eshkol_dispose_llvm_module(llvm_module);
                return 1;
            }
        } else if (output) {
            // Compile to executable with specified output name
            eshkol_info("Compiling to executable: %s", output);
            
            // Prepare C-style arrays for library paths and libraries
            const char** lib_path_ptrs = nullptr;
            const char** linked_lib_ptrs = nullptr;
            
            if (!lib_paths.empty()) {
                lib_path_ptrs = const_cast<const char**>(lib_paths.data());
            }
            if (!linked_libs.empty()) {
                linked_lib_ptrs = const_cast<const char**>(linked_libs.data());
            }
            
            if (eshkol_compile_llvm_ir_to_executable(llvm_module, output, 
                                                   lib_path_ptrs, lib_paths.size(),
                                                   linked_lib_ptrs, linked_libs.size()) != 0) {
                eshkol_error("Executable compilation failed");
                eshkol_dispose_llvm_module(llvm_module);
                return 1;
            }
        } else if (!compile_only && !dump_ir && !dump_ast) {
            // Default behavior: compile to a.out
            eshkol_info("Compiling to executable: a.out");
            
            // Prepare C-style arrays for library paths and libraries
            const char** lib_path_ptrs = nullptr;
            const char** linked_lib_ptrs = nullptr;
            
            if (!lib_paths.empty()) {
                lib_path_ptrs = const_cast<const char**>(lib_paths.data());
            }
            if (!linked_libs.empty()) {
                linked_lib_ptrs = const_cast<const char**>(linked_libs.data());
            }
            
            if (eshkol_compile_llvm_ir_to_executable(llvm_module, "a.out",
                                                   lib_path_ptrs, lib_paths.size(),
                                                   linked_lib_ptrs, linked_libs.size()) != 0) {
                eshkol_error("Executable compilation failed");
                eshkol_dispose_llvm_module(llvm_module);
                return 1;
            }
        }
        
        // Clean up
        eshkol_dispose_llvm_module(llvm_module);
    }

    // Process compiled object files if we have them and an output target
    if (!compiled_files.empty() && output) {
        std::string link_cmd = "cc";
        
        // Add all object files
        for (const auto &compiled_file : compiled_files) {
            link_cmd += " " + std::string(compiled_file);
        }
        
        // Add library search paths
        for (const auto &lib_path : lib_paths) {
            link_cmd += " -L" + std::string(lib_path);
        }
        
        // Add linked libraries
        for (const auto &linked_lib : linked_libs) {
            link_cmd += " -l" + std::string(linked_lib);
        }
        
        // Add output
        link_cmd += " -o " + std::string(output) + " -lm";
        
        eshkol_info("Linking object files: %s", link_cmd.c_str());
        int result = system(link_cmd.c_str());
        
        if (result != 0) {
            eshkol_error("Linking failed with exit code %d", result);
            return 1;
        }
        
        eshkol_info("Successfully created executable: %s", output);
    } else if (!compiled_files.empty()) {
        eshkol_warn("Object files provided but no output specified. Use -o to specify output executable.");
        return 1;
    }

    return 0;
}
