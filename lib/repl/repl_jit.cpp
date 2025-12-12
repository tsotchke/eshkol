//
// Copyright (C) tsotchke
//
// SPDX-License-Identifier: MIT
//

#include "repl_jit.h"
#include <eshkol/eshkol.h>
#include <eshkol/llvm_backend.h>
#include "../core/arena_memory.h"  // For runtime function declarations

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IR/CFG.h>  // For predecessors()
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/DynamicLibrary.h>
#include <llvm-c/Core.h>  // For LLVMModuleRef unwrapping

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <filesystem>
#include <set>
#include <vector>
#include <cctype>

#ifdef __APPLE__
#include <mach-o/dyld.h>  // For _NSGetExecutablePath on macOS
#endif

using namespace llvm;

// Track already-loaded modules to prevent circular imports
static std::set<std::string> loaded_modules;
using namespace llvm::orc;

namespace eshkol {

// Forward declarations for static helper functions
static std::vector<eshkol_ast_t> parseAllAstsFromString(const std::string& content);
static std::string resolveModulePath(const std::string& module_name, const std::string& base_dir = ".");

ReplJITContext::ReplJITContext()
    : jit_(nullptr)
    , ts_context_(nullptr)
    , raw_context_(nullptr)
    , eval_counter_(0)
    , shared_arena_(nullptr)
{
    initializeJIT();
}

ReplJITContext::~ReplJITContext() {
    // LLJIT destructor handles cleanup
}

void ReplJITContext::initializeJIT() {
    // Initialize LLVM targets (required for JIT)
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();
    InitializeNativeTargetAsmParser();

    // Load symbols from current process (includes eshkol-static runtime)
    // This makes arena_*, printf, malloc, etc. available to JIT code
    sys::DynamicLibrary::LoadLibraryPermanently(nullptr);

    // Create thread-safe context (shared across all modules)
    auto ctx = std::make_unique<LLVMContext>();
    raw_context_ = ctx.get();  // Cache raw pointer
    ts_context_ = std::make_shared<ThreadSafeContext>(std::move(ctx));

    // Create LLJIT instance with dynamic library search
    auto jit_or_err = LLJITBuilder()
        .setNumCompileThreads(1)  // Single-threaded for simplicity
        .create();

    if (!jit_or_err) {
        auto err = jit_or_err.takeError();
        std::string err_msg;
        raw_string_ostream err_stream(err_msg);
        err_stream << err;
        std::cerr << "Failed to create LLJIT: " << err_msg << std::endl;
        std::exit(1);
    }

    jit_ = std::move(*jit_or_err);

    // Enable REPL mode in the compiler for cross-evaluation symbol persistence
    eshkol_repl_enable();

    // Add symbol resolver for current process
    // This allows JIT code to call runtime functions from eshkol-static
    auto& main_dylib = jit_->getMainJITDylib();
    auto generator = orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
        jit_->getDataLayout().getGlobalPrefix());

    if (!generator) {
        auto err = generator.takeError();
        std::string err_msg;
        raw_string_ostream err_stream(err_msg);
        err_stream << err;
        std::cerr << "Failed to create symbol generator: " << err_msg << std::endl;
        std::exit(1);
    }

    main_dylib.addGenerator(std::move(*generator));

    // Manually register runtime symbols (arena functions, etc.)
    // This is needed because macOS doesn't export symbols from static libraries
    registerRuntimeSymbols();

    // CRITICAL: Create shared arena for all REPL evaluations
    // Set the global variable that will be accessed by JIT-compiled code
    __repl_shared_arena = arena_create(8192);  // 8KB default block size
    shared_arena_ = __repl_shared_arena;  // Keep local copy for cleanup

    // REPL JIT initialized silently
}

// REMOVED: Failed IR manipulation approach - now using AST-level wrapping instead
// void ReplJITContext::modifyToDisplayResult(Module* module, Function* func) { ... }

void ReplJITContext::registerRuntimeSymbols() {
    // Build a symbol map with addresses of runtime functions
    // These are all from eshkol-static which is statically linked
    orc::SymbolMap symbols;
    auto& ES = jit_->getExecutionSession();
    auto& DL = jit_->getDataLayout();

    // Helper macro to add a callable symbol
    #define ADD_SYMBOL(name) \
        symbols[ES.intern(#name)] = { \
            orc::ExecutorAddr::fromPtr((void*)&name), \
            JITSymbolFlags::Callable | JITSymbolFlags::Exported \
        }

    // Helper macro to add a data symbol (global variable)
    #define ADD_DATA_SYMBOL(name) \
        symbols[ES.intern(#name)] = { \
            orc::ExecutorAddr::fromPtr((void*)&name), \
            JITSymbolFlags::Exported \
        }

    // ===== ARENA MEMORY MANAGEMENT =====
    ADD_SYMBOL(arena_create);
    ADD_SYMBOL(arena_destroy);
    ADD_SYMBOL(arena_allocate);
    ADD_SYMBOL(arena_allocate_aligned);
    ADD_SYMBOL(arena_allocate_zeroed);
    ADD_SYMBOL(arena_push_scope);
    ADD_SYMBOL(arena_pop_scope);
    ADD_SYMBOL(arena_reset);
    ADD_SYMBOL(arena_get_used_memory);
    ADD_SYMBOL(arena_get_total_memory);
    ADD_SYMBOL(arena_get_block_count);

    // Header-aware allocation (for consolidated HEAP_PTR/CALLABLE types)
    ADD_SYMBOL(arena_allocate_with_header);
    ADD_SYMBOL(arena_allocate_with_header_zeroed);
    ADD_SYMBOL(arena_allocate_multi_value);

    // Cons cell allocation
    ADD_SYMBOL(arena_allocate_cons_cell);
    ADD_SYMBOL(arena_allocate_tagged_cons_cell);
    ADD_SYMBOL(arena_allocate_tagged_cons_batch);
    ADD_SYMBOL(arena_allocate_cons_with_header);

    // String allocation
    ADD_SYMBOL(arena_allocate_string_with_header);

    // Vector allocation
    ADD_SYMBOL(arena_allocate_vector_with_header);

    // Tagged cons cell accessors
    ADD_SYMBOL(arena_tagged_cons_get_int64);
    ADD_SYMBOL(arena_tagged_cons_get_double);
    ADD_SYMBOL(arena_tagged_cons_get_ptr);
    ADD_SYMBOL(arena_tagged_cons_set_int64);
    ADD_SYMBOL(arena_tagged_cons_set_double);
    ADD_SYMBOL(arena_tagged_cons_set_ptr);
    ADD_SYMBOL(arena_tagged_cons_set_null);
    ADD_SYMBOL(arena_tagged_cons_get_type);
    ADD_SYMBOL(arena_tagged_cons_get_flags);
    ADD_SYMBOL(arena_tagged_cons_is_type);
    ADD_SYMBOL(arena_tagged_cons_set_tagged_value);
    ADD_SYMBOL(arena_tagged_cons_get_tagged_value);

    // Tagged cons constructors
    ADD_SYMBOL(arena_create_int64_cons);
    ADD_SYMBOL(arena_create_mixed_cons);

    // Deep equality
    ADD_SYMBOL(eshkol_deep_equal);

    // ===== EXCEPTION HANDLING =====
    ADD_SYMBOL(eshkol_raise);
    ADD_SYMBOL(eshkol_make_exception);
    ADD_SYMBOL(eshkol_make_exception_with_header);
    ADD_SYMBOL(eshkol_push_exception_handler);
    ADD_SYMBOL(eshkol_pop_exception_handler);
    ADD_SYMBOL(eshkol_exception_type_matches);
    ADD_DATA_SYMBOL(g_current_exception);
    ADD_DATA_SYMBOL(g_exception_handler_stack);

    // ===== AUTOMATIC DIFFERENTIATION =====
    ADD_SYMBOL(arena_allocate_dual_number);
    ADD_SYMBOL(arena_allocate_dual_batch);
    ADD_SYMBOL(arena_allocate_ad_node);
    ADD_SYMBOL(arena_allocate_ad_node_with_header);
    ADD_SYMBOL(arena_allocate_ad_batch);
    ADD_SYMBOL(arena_allocate_tape);
    ADD_SYMBOL(arena_tape_add_node);
    ADD_SYMBOL(arena_tape_reset);
    ADD_SYMBOL(arena_tape_get_node);
    ADD_SYMBOL(arena_tape_get_node_count);
    ADD_SYMBOL(debug_print_ad_mode);
    ADD_SYMBOL(debug_print_ptr);

    // ===== CLOSURE MEMORY MANAGEMENT =====
    ADD_SYMBOL(arena_allocate_closure_env);
    ADD_SYMBOL(arena_allocate_closure);
    ADD_SYMBOL(arena_allocate_closure_with_header);

    // ===== TENSOR MEMORY MANAGEMENT =====
    ADD_SYMBOL(arena_allocate_tensor_with_header);
    ADD_SYMBOL(arena_allocate_tensor_full);

    // ===== HASH TABLE MEMORY MANAGEMENT =====
    ADD_SYMBOL(arena_allocate_hash_table);
    ADD_SYMBOL(arena_hash_table_create);
    ADD_SYMBOL(arena_hash_table_create_with_header);
    ADD_SYMBOL(hash_table_set);
    ADD_SYMBOL(hash_table_get);
    ADD_SYMBOL(hash_table_has_key);
    ADD_SYMBOL(hash_table_remove);
    ADD_SYMBOL(hash_table_clear);
    ADD_SYMBOL(hash_table_count);
    ADD_SYMBOL(hash_table_keys);
    ADD_SYMBOL(hash_table_values);
    ADD_SYMBOL(hash_tagged_value);
    ADD_SYMBOL(hash_keys_equal);

    // ===== REGION (OALR) MEMORY MANAGEMENT =====
    ADD_SYMBOL(region_create);
    ADD_SYMBOL(region_destroy);
    ADD_SYMBOL(region_push);
    ADD_SYMBOL(region_pop);
    ADD_SYMBOL(region_current);
    ADD_SYMBOL(region_allocate);
    ADD_SYMBOL(region_allocate_aligned);
    ADD_SYMBOL(region_allocate_zeroed);
    ADD_SYMBOL(region_allocate_tagged_cons_cell);
    ADD_SYMBOL(region_get_used_memory);
    ADD_SYMBOL(region_get_total_memory);
    ADD_SYMBOL(region_get_name);
    ADD_SYMBOL(region_get_depth);

    // ===== SHARED (REF-COUNTED) MEMORY MANAGEMENT =====
    ADD_SYMBOL(shared_allocate);
    ADD_SYMBOL(shared_allocate_typed);
    ADD_SYMBOL(shared_retain);
    ADD_SYMBOL(shared_release);
    ADD_SYMBOL(shared_ref_count);
    ADD_SYMBOL(shared_get_header);
    ADD_SYMBOL(weak_ref_create);
    ADD_SYMBOL(weak_ref_upgrade);
    ADD_SYMBOL(weak_ref_release);
    ADD_SYMBOL(weak_ref_is_alive);

    // Standard library functions (printf, malloc, etc.)
    // Need to explicitly cast math functions to resolve overloading
    typedef double (*MathFunc1)(double);
    typedef double (*MathFunc2)(double, double);

    symbols[ES.intern("printf")] = {
        orc::ExecutorAddr::fromPtr((void*)&printf),
        JITSymbolFlags::Callable | JITSymbolFlags::Exported
    };
    symbols[ES.intern("malloc")] = {
        orc::ExecutorAddr::fromPtr((void*)&malloc),
        JITSymbolFlags::Callable | JITSymbolFlags::Exported
    };
    symbols[ES.intern("sin")] = {
        orc::ExecutorAddr::fromPtr((void*)(MathFunc1)&std::sin),
        JITSymbolFlags::Callable | JITSymbolFlags::Exported
    };
    symbols[ES.intern("cos")] = {
        orc::ExecutorAddr::fromPtr((void*)(MathFunc1)&std::cos),
        JITSymbolFlags::Callable | JITSymbolFlags::Exported
    };
    symbols[ES.intern("sqrt")] = {
        orc::ExecutorAddr::fromPtr((void*)(MathFunc1)&std::sqrt),
        JITSymbolFlags::Callable | JITSymbolFlags::Exported
    };
    symbols[ES.intern("pow")] = {
        orc::ExecutorAddr::fromPtr((void*)(MathFunc2)&std::pow),
        JITSymbolFlags::Callable | JITSymbolFlags::Exported
    };

    // ===== GLOBAL DATA SYMBOLS =====
    // These are shared across all JIT modules

    // AD tape pointer (for gradient/jacobian operations)
    ADD_DATA_SYMBOL(__current_ad_tape);

    // AD mode flag (CRITICAL: must be shared so lambdas see AD mode set by other modules)
    ADD_DATA_SYMBOL(__ad_mode_active);

    // Shared arena pointer (persistent across REPL evaluations)
    ADD_DATA_SYMBOL(__repl_shared_arena);

    // Region stack (for OALR memory management)
    ADD_DATA_SYMBOL(__region_stack);
    ADD_DATA_SYMBOL(__region_stack_depth);

    // Command-line arguments (for (command-line) procedure)
    ADD_DATA_SYMBOL(__eshkol_argc);
    ADD_DATA_SYMBOL(__eshkol_argv);

    // Global arena (default allocation target)
    ADD_DATA_SYMBOL(__global_arena);

    #undef ADD_SYMBOL
    #undef ADD_DATA_SYMBOL

    // Add all symbols to the main dylib
    auto& main_dylib = jit_->getMainJITDylib();
    auto err = main_dylib.define(orc::absoluteSymbols(symbols));
    if (err) {
        std::string err_msg;
        raw_string_ostream err_stream(err_msg);
        err_stream << err;
        std::cerr << "Failed to register runtime symbols: " << err_msg << std::endl;
        std::exit(1);
    }

    // Debug output disabled for cleaner REPL experience
    // std::cout << "Registered " << symbols.size() << " runtime symbols" << std::endl;
}

LLVMContext& ReplJITContext::getContext() {
    return *raw_context_;
}

std::unique_ptr<Module> ReplJITContext::createModule(const std::string& name) {
    return std::make_unique<Module>(name, getContext());
}

void ReplJITContext::addModule(std::unique_ptr<Module> module) {
    // Verify the module first
    std::string error_msg;
    raw_string_ostream error_stream(error_msg);
    if (verifyModule(*module, &error_stream)) {
        std::cerr << "Module verification failed: " << error_msg << std::endl;
        module->print(errs(), nullptr);
        throw std::runtime_error("Invalid LLVM module");
    }

    // Wrap module in ThreadSafeModule using shared context
    // All modules share the same context so they can reference each other's symbols
    auto tsm = ThreadSafeModule(std::move(module), *ts_context_);

    auto err = jit_->addIRModule(std::move(tsm));
    if (err) {
        std::string err_msg;
        raw_string_ostream err_stream(err_msg);
        err_stream << err;
        std::cerr << "Failed to add module to JIT: " << err_msg << std::endl;
        throw std::runtime_error("Failed to add module to JIT");
    }
}

uint64_t ReplJITContext::lookupSymbol(const std::string& name) {
    // First check our local symbol table
    auto it = symbol_table_.find(name);
    if (it != symbol_table_.end()) {
        return it->second;
    }

    // Look up in JIT
    auto symbol = jit_->lookup(name);
    if (!symbol) {
        // Symbol not found - this is OK, caller will handle
        consumeError(symbol.takeError());
        return 0;
    }

    uint64_t address = symbol->getValue();

    // Cache in symbol table
    symbol_table_[name] = address;

    return address;
}

bool ReplJITContext::isSymbolDefined(const std::string& name) {
    // Check our local symbol table first
    if (symbol_table_.find(name) != symbol_table_.end()) {
        return true;
    }

    // Check defined lambdas (these are pre-registered before JIT compilation)
    if (defined_lambdas_.find(name) != defined_lambdas_.end()) {
        return true;
    }

    // Check defined globals
    if (defined_globals_.find(name) != defined_globals_.end()) {
        return true;
    }

    // Try looking up in JIT (this covers symbols that might have been
    // registered via other means)
    auto symbol = jit_->lookup(name);
    if (symbol) {
        return true;
    }
    consumeError(symbol.takeError());

    return false;
}

void ReplJITContext::registerSymbol(const std::string& name, uint64_t address) {
    symbol_table_[name] = address;

    // Also register in JIT dylib so subsequent modules can link against it
    orc::SymbolMap symbols;
    auto& ES = jit_->getExecutionSession();

    // Get the platform-specific mangled name
    // On macOS/Darwin, this adds a leading underscore
    auto& DL = jit_->getDataLayout();
    std::string mangled = name;
    if (DL.getGlobalPrefix()) {
        mangled = std::string(1, DL.getGlobalPrefix()) + name;
    }

    auto mangled_symbol = ES.intern(mangled);
    symbols[mangled_symbol] = {
        orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(address)),
        JITSymbolFlags::Callable | JITSymbolFlags::Exported
    };

    auto& main_dylib = jit_->getMainJITDylib();
    auto err = main_dylib.define(orc::absoluteSymbols(symbols));
    if (err) {
        std::string err_msg;
        raw_string_ostream err_stream(err_msg);
        err_stream << err;
        std::cerr << "Warning: Failed to register symbol " << mangled << " in JIT: " << err_msg << std::endl;
    }
}

void ReplJITContext::registerLambdaVar(const std::string& var_name) {
    // Mark that this variable will hold a lambda
    // The actual lambda function name and arity will be discovered after compilation
    // For now, just mark it as pending - we'll fill in the details after seeing the module
    defined_lambdas_[var_name] = {"", 0};  // Empty name and 0 arity means "pending"
}

bool ReplJITContext::loadStdlib() {
    return loadModule("stdlib");
}

bool ReplJITContext::loadModule(const std::string& module_name) {
    std::string module_path = resolveModulePath(module_name);

    if (module_path.empty()) {
        std::cerr << "Module not found: " << module_name << std::endl;
        return false;
    }

    // Check if already loaded
    if (loaded_modules.count(module_path)) {
        return true;  // Already loaded, success
    }
    loaded_modules.insert(module_path);

    // Read the module file
    std::ifstream module_file(module_path);
    if (!module_file.is_open()) {
        std::cerr << "Cannot open module: " << module_path << std::endl;
        return false;
    }

    std::stringstream buffer;
    buffer << module_file.rdbuf();
    std::string content = buffer.str();
    module_file.close();

    // Parse all ASTs from the module
    std::vector<eshkol_ast_t> module_asts = parseAllAstsFromString(content);

    // Execute each AST in the module (silently)
    bool success = true;
    for (auto& ast_item : module_asts) {
        // Check if this is a define that creates a lambda function
        // and register it so JIT can find it later
        if (ast_item.type == ESHKOL_OP && ast_item.operation.op == ESHKOL_DEFINE_OP) {
            const char* name = ast_item.operation.define_op.name;
            bool is_lambda = ast_item.operation.define_op.is_function ||
                (ast_item.operation.define_op.value &&
                 ast_item.operation.define_op.value->type == ESHKOL_OP &&
                 ast_item.operation.define_op.value->operation.op == ESHKOL_LAMBDA_OP);
            if (name && is_lambda) {
                registerLambdaVar(name);
            }
        }

        try {
            execute(&ast_item);
        } catch (const std::exception& e) {
            // Continue loading other definitions even if one fails
            // (some internal helpers may have issues but main functions work)
        }
        eshkol_ast_clean(&ast_item);
    }

    return success;
}

void ReplJITContext::injectPreviousSymbols(Module* module) {
    // Inject external function declarations for all previously defined lambdas
    // This allows new code to reference functions defined in previous REPL evaluations

    for (const auto& [var_name, lambda_info] : defined_lambdas_) {
        const auto& [lambda_name, arity] = lambda_info;
        // Skip pending (empty) lambda names
        if (lambda_name.empty()) {
            continue;
        }

        // Check if this function already exists in the module
        Function* lambda_func = module->getFunction(lambda_name);
        if (!lambda_func) {
            // Create external declaration for the lambda function
            // All lambdas have signature: eshkol_tagged_value(*)(eshkol_tagged_value, eshkol_tagged_value, ...)
            Type* tagged_value_type = StructType::getTypeByName(module->getContext(), "eshkol_tagged_value");
            if (!tagged_value_type) {
                std::cerr << "ERROR: eshkol_tagged_value type not found - skipping lambda injection" << std::endl;
                continue;
            }

            // Create parameter types based on arity (all parameters are tagged_value)
            std::vector<Type*> param_types(arity, tagged_value_type);
            FunctionType* func_type = FunctionType::get(
                tagged_value_type,
                param_types,
                false  // NOT varargs - use exact arity
            );

            lambda_func = Function::Create(
                func_type,
                Function::ExternalLinkage,
                lambda_name,
                module
            );
        }
    }

    // Inject external global variable declarations for all previously defined globals
    // This allows new code to reference variables defined in previous REPL evaluations
    for (const auto& global_name : defined_globals_) {
        // Check if this global already exists in the module
        GlobalVariable* existing = module->getGlobalVariable(global_name);
        if (!existing) {
            // Create external declaration for the global variable
            // Use i64 as the default type (we can refine this later if needed)
            Type* global_type = Type::getInt64Ty(module->getContext());

            GlobalVariable* global_var = new GlobalVariable(
                *module,
                global_type,
                false,  // not constant
                GlobalValue::ExternalLinkage,
                nullptr,  // no initializer for external declaration
                global_name
            );
        }
    }
}

// Helper to parse all ASTs from a string content
// Returns a vector of parsed ASTs
// Note: Uses ::eshkol_parse_next_ast_from_stream from global namespace (declared in eshkol.h)
static std::vector<eshkol_ast_t> parseAllAstsFromString(const std::string& content) {
    std::vector<eshkol_ast_t> results;
    std::istringstream stream(content);

    while (stream.good() && !stream.eof()) {
        // Skip whitespace and comments
        while (stream.good()) {
            int c = stream.peek();
            if (c == EOF) break;
            if (std::isspace(c)) {
                stream.get();
            } else if (c == ';') {
                // Skip comment line
                std::string dummy;
                std::getline(stream, dummy);
            } else {
                break;
            }
        }

        if (stream.eof() || stream.peek() == EOF) break;

        // Use global namespace function (declared in eshkol.h)
        eshkol_ast_t ast = ::eshkol_parse_next_ast_from_stream(stream);
        if (ast.type == ESHKOL_INVALID) break;
        results.push_back(ast);
    }

    return results;
}

// Find the lib directory (matches eshkol-run.cpp logic)
static std::string findLibDir() {
    std::vector<std::string> lib_dirs = {
        "lib",
        "../lib",
        "/usr/local/share/eshkol/lib",
        "/usr/share/eshkol/lib",
    };

    // Check relative to executable on macOS
    #ifdef __APPLE__
    char exe_path[4096];
    uint32_t size = sizeof(exe_path);
    if (_NSGetExecutablePath(exe_path, &size) == 0) {
        std::filesystem::path exe_dir = std::filesystem::path(exe_path).parent_path();
        lib_dirs.insert(lib_dirs.begin(), (exe_dir / "../lib").string());
        lib_dirs.insert(lib_dirs.begin(), (exe_dir / "lib").string());
    }
    #endif

    for (const auto& dir : lib_dirs) {
        if (std::filesystem::exists(dir) && std::filesystem::is_directory(dir)) {
            return std::filesystem::canonical(dir).string();
        }
    }
    return "";
}

// Global lib directory cache
static std::string g_lib_dir;

// Helper to resolve module path (e.g., "core.functional.compose" -> "lib/core/functional/compose.esk")
// Matches eshkol-run.cpp module resolution logic
static std::string resolveModulePath(const std::string& module_name, const std::string& base_dir) {
    // Convert dots to path separators
    std::string path_part = module_name;
    for (char& c : path_part) {
        if (c == '.') c = '/';
    }
    path_part += ".esk";

    // Initialize lib dir if needed
    if (g_lib_dir.empty()) {
        g_lib_dir = findLibDir();
    }

    // Search order:
    // 1. Current directory (relative to base_dir)
    // 2. Library path (lib/)
    // 3. Environment variable $ESHKOL_PATH (colon-separated)

    // Try current directory first
    std::filesystem::path current_path = std::filesystem::path(base_dir) / path_part;
    if (std::filesystem::exists(current_path)) {
        return std::filesystem::canonical(current_path).string();
    }

    // Try library directory
    if (!g_lib_dir.empty()) {
        std::filesystem::path lib_path = std::filesystem::path(g_lib_dir) / path_part;
        if (std::filesystem::exists(lib_path)) {
            return std::filesystem::canonical(lib_path).string();
        }
    }

    // Try $ESHKOL_PATH
    const char* eshkol_path = std::getenv("ESHKOL_PATH");
    if (eshkol_path) {
        std::stringstream ss(eshkol_path);
        std::string search_dir;
        while (std::getline(ss, search_dir, ':')) {
            std::filesystem::path env_path = std::filesystem::path(search_dir) / path_part;
            if (std::filesystem::exists(env_path)) {
                return std::filesystem::canonical(env_path).string();
            }
        }
    }

    // Legacy fallback paths
    std::vector<std::string> fallback_paths = {
        "lib/" + path_part,
        path_part,
        "../lib/" + path_part,
    };

    for (const auto& p : fallback_paths) {
        if (std::filesystem::exists(p)) {
            return std::filesystem::canonical(p).string();
        }
    }

    return "";
}

void* ReplJITContext::execute(eshkol_ast_t* ast) {
    if (!ast) {
        throw std::runtime_error("Cannot execute null AST");
    }

    // IMPORT/REQUIRE HANDLING: Load and execute imported files
    if (ast->type == ESHKOL_OP) {
        // Handle (import "path/to/file.esk")
        if (ast->operation.op == ESHKOL_IMPORT_OP && ast->operation.import_op.path) {
            std::string import_path = ast->operation.import_op.path;

            // Resolve relative paths
            if (!std::filesystem::path(import_path).is_absolute()) {
                if (!std::filesystem::exists(import_path)) {
                    // Try relative to lib/
                    std::string lib_path = "lib/" + import_path;
                    if (std::filesystem::exists(lib_path)) {
                        import_path = lib_path;
                    }
                }
            }

            // Check if already loaded
            std::string canonical_path;
            try {
                canonical_path = std::filesystem::canonical(import_path).string();
            } catch (...) {
                std::cerr << "Import file not found: " << import_path << std::endl;
                return nullptr;
            }

            if (loaded_modules.count(canonical_path)) {
                // Already loaded, return nil
                return nullptr;
            }
            loaded_modules.insert(canonical_path);

            // Read the file
            std::ifstream file(canonical_path);
            if (!file.is_open()) {
                std::cerr << "Cannot open import file: " << canonical_path << std::endl;
                return nullptr;
            }

            std::stringstream buffer;
            buffer << file.rdbuf();
            std::string content = buffer.str();
            file.close();

            // Parse all ASTs from the file
            std::vector<eshkol_ast_t> file_asts = parseAllAstsFromString(content);
            if (file_asts.empty()) {
                // Empty file or parse failed - this is okay for modules that just define things
                return nullptr;
            }

            // Execute each AST in the file
            void* last_result = nullptr;
            for (auto& ast_item : file_asts) {
                // Check if this is a define that creates a lambda function
                // and register it so JIT can find it later
                if (ast_item.type == ESHKOL_OP && ast_item.operation.op == ESHKOL_DEFINE_OP) {
                    const char* name = ast_item.operation.define_op.name;
                    bool is_lambda = ast_item.operation.define_op.is_function ||
                        (ast_item.operation.define_op.value &&
                         ast_item.operation.define_op.value->type == ESHKOL_OP &&
                         ast_item.operation.define_op.value->operation.op == ESHKOL_LAMBDA_OP);
                    if (name && is_lambda) {
                        registerLambdaVar(name);
                    }
                }
                last_result = execute(&ast_item);
                eshkol_ast_clean(&ast_item);
            }

            return last_result;
        }

        // Handle (require module.name ...)
        if (ast->operation.op == ESHKOL_REQUIRE_OP) {
            void* last_result = nullptr;
            for (size_t i = 0; i < ast->operation.require_op.num_modules; i++) {
                std::string module_name = ast->operation.require_op.module_names[i];
                std::string module_path = resolveModulePath(module_name);

                if (module_path.empty()) {
                    std::cerr << "Module not found: " << module_name << std::endl;
                    continue;
                }

                // Check if already loaded
                if (loaded_modules.count(module_path)) {
                    continue;
                }
                loaded_modules.insert(module_path);

                // Read the module file
                std::ifstream module_file(module_path);
                if (!module_file.is_open()) {
                    std::cerr << "Cannot open module: " << module_path << std::endl;
                    continue;
                }

                std::stringstream buffer;
                buffer << module_file.rdbuf();
                std::string content = buffer.str();
                module_file.close();

                // Parse all ASTs from the module
                std::vector<eshkol_ast_t> module_asts = parseAllAstsFromString(content);

                // Execute each AST in the module
                for (auto& ast_item : module_asts) {
                    // Check if this is a define that creates a lambda function
                    // and register it so JIT can find it later
                    if (ast_item.type == ESHKOL_OP && ast_item.operation.op == ESHKOL_DEFINE_OP) {
                        const char* name = ast_item.operation.define_op.name;
                        bool is_lambda = ast_item.operation.define_op.is_function ||
                            (ast_item.operation.define_op.value &&
                             ast_item.operation.define_op.value->type == ESHKOL_OP &&
                             ast_item.operation.define_op.value->operation.op == ESHKOL_LAMBDA_OP);
                        if (name && is_lambda) {
                            registerLambdaVar(name);
                        }
                    }
                    last_result = execute(&ast_item);
                    eshkol_ast_clean(&ast_item);
                }
            }

            return last_result;
        }

        // Handle (provide ...) - just return nil, exports are implicit in REPL
        if (ast->operation.op == ESHKOL_PROVIDE_OP) {
            return nullptr;
        }
    }

    // Generate LLVM IR using the existing Eshkol compiler
    std::string module_name = "__repl_module_" + std::to_string(eval_counter_);

    // Call the existing compiler to generate LLVM IR from AST
    LLVMModuleRef c_module = eshkol_generate_llvm_ir(ast, 1, module_name.c_str());

    if (!c_module) {
        throw std::runtime_error("Failed to generate LLVM IR from AST");
    }

    // Convert LLVMModuleRef (C API) to llvm::Module* (C++ API)
    // Note: LLVMModuleRef is actually llvm::Module*, just opaque
    Module* cpp_module = llvm::unwrap(c_module);

    if (!cpp_module) {
        throw std::runtime_error("Failed to unwrap LLVM module");
    }

    // REPL SYMBOL PERSISTENCE: Inject declarations for previously-defined symbols
    // This allows the current module to reference functions/variables from previous evaluations
    injectPreviousSymbols(cpp_module);

    // REPL SYMBOL TRACKING: Extract lambda/function definitions from this module
    // Fill in names for variables that were pre-registered
    for (auto& func : cpp_module->functions()) {
        if (func.isDeclaration() || func.getName().starts_with("llvm.")) {
            continue;
        }
        std::string fname = func.getName().str();

        // Track lambda functions (they start with "lambda_")
        if (fname.find("lambda_") == 0) {
            // Fill in pending lambda variables with this lambda name and arity
            size_t arity = func.arg_size();
            for (auto& [var_name, lambda_info] : defined_lambdas_) {
                if (lambda_info.first.empty()) {
                    // This was a pending lambda registration - fill it in
                    defined_lambdas_[var_name] = {fname, arity};
                    break;  // Only fill one pending slot per lambda
                }
            }
        }
        // Track user-defined functions (e.g., squared-sum from "(define (squared-sum x) ...)")
        // These have the same name as the pre-registered variable
        else if (defined_lambdas_.find(fname) != defined_lambdas_.end()) {
            auto& lambda_info = defined_lambdas_[fname];
            if (lambda_info.first.empty()) {
                // This was a pending registration - fill it in
                // For user-defined functions, the function name IS the variable name
                size_t arity = func.arg_size();
                defined_lambdas_[fname] = {fname, arity};
            }
        }
    }

    // Debug: Print function list (disabled for cleaner output)
    // std::cout << "=== Module Functions ===" << std::endl;
    // for (auto& func : cpp_module->functions()) {
    //     std::cout << "  " << func.getName().str()
    //               << " (decl=" << func.isDeclaration()
    //               << ", local=" << func.hasLocalLinkage()
    //               << ")" << std::endl;
    // }
    // std::cout << "========================" << std::endl;

    // The compiler generates a module with potentially multiple functions
    // For a simple expression like (+ 1 2), it generates a top-level expression
    // We need to find the entry point function

    // Look for the main entry point - prioritize __top_level_expr__ or similar
    // Skip internal helper functions (prefixed with @ or containing "display", "print", etc.)
    Function* entry_func = nullptr;

    // First pass: look for explicit top-level entry points
    for (auto& func : cpp_module->functions()) {
        if (func.isDeclaration() || func.getName().starts_with("llvm.")) {
            continue;
        }
        std::string fname = func.getName().str();
        if (fname.find("__top_level") != std::string::npos ||
            fname.find("main") != std::string::npos) {
            entry_func = &func;
            // std::cout << "Found entry function: " << fname << std::endl;
            break;
        }
    }

    // Second pass: if no explicit entry point, find any non-internal function
    if (!entry_func) {
        for (auto& func : cpp_module->functions()) {
            if (func.isDeclaration() || func.getName().starts_with("llvm.")) {
                continue;
            }
            std::string fname = func.getName().str();
            // Skip internal helpers
            if (fname.find("display") != std::string::npos ||
                fname.find("print") != std::string::npos ||
                fname.find("__internal") != std::string::npos ||
                func.hasLocalLinkage()) {
                continue;
            }
            entry_func = &func;
            // std::cout << "Found candidate function: " << fname << std::endl;
            break;
        }
    }

    if (!entry_func) {
        cpp_module->print(errs(), nullptr);
        throw std::runtime_error("No entry function found in generated module");
    }

    std::string func_name = entry_func->getName().str();

    // Rename the entry function to avoid symbol conflicts across evaluations
    std::string unique_func_name = "__repl_eval_" + std::to_string(eval_counter_);
    entry_func->setName(unique_func_name);
    func_name = unique_func_name;

    // Display is now handled at AST level via eshkol_wrap_with_display()
    // No IR modification needed

    // REPL SYMBOL TRACKING: Extract global variables before adding module
    // Collect global variable names so we can register them after JIT compilation
    std::vector<std::string> global_var_names;
    for (auto& global_var : cpp_module->globals()) {
        std::string var_name = global_var.getName().str();

        if (global_var.isDeclaration() || global_var.getName().starts_with("llvm.")) {
            continue;
        }
        // Skip internal variables and function references
        if (var_name.find("__") == 0 || var_name.find("_func") != std::string::npos) {
            continue;
        }
        global_var_names.push_back(var_name);
    }

    // CRITICAL FIX: Release ownership from g_llvm_modules before JIT takes it
    // The module is stored in g_llvm_modules by eshkol_generate_llvm_ir().
    // Without this, both g_llvm_modules and JIT think they own the module,
    // causing double-free/use-after-free crashes on subsequent evaluations.
    eshkol_release_module_for_jit(c_module);

    // Add module to JIT (takes ownership)
    addModule(std::unique_ptr<Module>(cpp_module));

    // REPL MODE: Register all lambda functions from this module with global REPL context
    // This enables cross-evaluation function calls
    for (const auto& [var_name, lambda_info] : defined_lambdas_) {
        const auto& [lambda_name, arity] = lambda_info;
        if (!lambda_name.empty()) {
            // Skip if already registered
            if (registered_lambdas_.find(lambda_name) != registered_lambdas_.end()) {
                continue;
            }

            // Look up the JIT address of this lambda
            uint64_t lambda_addr = lookupSymbol(lambda_name);
            if (lambda_addr != 0) {
                // The compiler already creates var_name and var_name_func globals in the module,
                // so they're already in the JIT. Just register them in the REPL context.

                // Register in global REPL context for compiler to check (with arity)
                // IMPORTANT: Only register the actual lambda function name (e.g., "lambda_0")
                // Do NOT register var_name as a function - it's a GlobalVariable!
                eshkol_repl_register_function(lambda_name.c_str(), lambda_addr, arity);
                eshkol_repl_register_function((var_name + "_func").c_str(), lambda_addr, arity);
                // NOTE: Removed var_name registration - it's handled as a global variable below

                // Register variable -> lambda name mapping for s-expression lookup
                eshkol_repl_register_lambda_name(var_name.c_str(), lambda_name.c_str());

                // Mark this lambda as registered
                registered_lambdas_.insert(lambda_name);

                // Debug output disabled for cleaner REPL experience
                // std::cout << "REPL: Registered " << var_name << " -> " << lambda_name << " (arity " << arity << ")" << std::endl;
            }
        }
    }

    // REPL MODE: Register all global variables from this module
    // This enables cross-evaluation variable access (e.g., (define x 10) then x)
    std::vector<std::pair<std::string, uint64_t>> sexpr_globals_to_capture;  // Defer s-expression capture until after execution

    for (const auto& var_name : global_var_names) {
        // NOTE: Do NOT skip lambda variables - they need to be registered as globals too!
        // Lambda variables are GlobalVariables that store function pointers (i64)
        // Skipping them causes link failures when referenced from other modules

        // Look up the JIT address of this global variable
        // (It already exists in the JIT from the module we just compiled)
        uint64_t var_addr = lookupSymbol(var_name);
        if (var_addr != 0) {
            // Track this global for future symbol injection
            defined_globals_.insert(var_name);

            // Register in global REPL context for compiler to check
            // (No need to call registerSymbol - it's already in the JIT)
            eshkol_repl_register_symbol(var_name.c_str(), var_addr);

            // Debug output disabled for cleaner REPL experience
            // if (!var_name.starts_with(".str")) {
            //     std::cout << "REPL: Registered variable " << var_name << " @ 0x" << std::hex << var_addr << std::dec << std::endl;
            // }

            // DEFER s-expression value capture until AFTER function execution
            // (s-expressions are initialized inside the entry function)
            if (var_name.find("_sexpr") != std::string::npos) {
                sexpr_globals_to_capture.push_back({var_name, var_addr});
            }
        }
    }

    // Look up the function we just compiled
    uint64_t func_addr = lookupSymbol(func_name);
    if (func_addr == 0) {
        throw std::runtime_error("Failed to find JIT-compiled function: " + func_name);
    }

    // Increment eval counter for next evaluation
    incrementEvalCounter();

    // Cast to function pointer and call it
    // The compiler generates main as i32(), so cast appropriately
    typedef int32_t (*EvalFunc)();
    EvalFunc eval_func = reinterpret_cast<EvalFunc>(func_addr);

    int32_t result_value = eval_func();

    // CRITICAL: NOW capture s-expression values AFTER execution
    // The entry function has initialized these globals, so now they contain valid values
    for (const auto& [var_name, var_addr] : sexpr_globals_to_capture) {
        // Read the current value from the global's memory
        uint64_t* global_ptr = reinterpret_cast<uint64_t*>(var_addr);
        uint64_t sexpr_value = *global_ptr;
        // Register the s-expression value with the compiler
        eshkol_repl_register_sexpr(var_name.c_str(), sexpr_value);
    }

    // For now, return the result as a heap-allocated int64_t (promoted from i32)
    // TODO: Handle different return types (double, string, closure, etc.)
    int64_t* result_ptr = new int64_t(result_value);

    return result_ptr;
}

} // namespace eshkol
