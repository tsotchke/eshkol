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
#include <llvm/Support/MemoryBuffer.h>  // For loading object files
#include <llvm-c/Core.h>  // For LLVMModuleRef unwrapping

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <set>
#include <vector>
#include <cctype>
#include <unistd.h>

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

// Stub function called when a forward-referenced function hasn't been defined yet
static eshkol_tagged_value __repl_forward_ref_stub() {
    std::cerr << "Error: Called a forward-referenced function that was never defined" << std::endl;
    exit(1);
    return {0, 0};  // Never reached
}

void ReplJITContext::addModule(std::unique_ptr<Module> module) {
    // Collect forward references and definitions to handle
    std::vector<std::pair<std::string, std::string>> forward_ref_updates;  // (ptr_name, func_name)

    // STEP 1: Scan for globals that define previously forward-referenced functions
    // These need to be converted to external declarations, and we'll update the pointer slot later
    for (auto& gv : module->globals()) {
        if (gv.hasInitializer()) {
            std::string name = gv.getName().str();
            if (name.find("__repl_fwd_") == 0 && pending_forward_refs_.count(name)) {
                // This module defines a function that was previously forward-referenced
                // Get the function name from the initializer
                if (auto* func = dyn_cast<Function>(gv.getInitializer())) {
                    forward_ref_updates.push_back({name, func->getName().str()});
                }
                // Remove the initializer so it becomes an external declaration
                // This prevents duplicate symbol errors
                gv.setInitializer(nullptr);
                gv.setExternallyInitialized(false);
            }
        }
    }

    // STEP 2: Scan for external references to __repl_fwd_* symbols and create stubs
    for (auto& gv : module->globals()) {
        if (gv.hasExternalLinkage() && !gv.hasInitializer()) {
            std::string name = gv.getName().str();
            if (name.find("__repl_fwd_") == 0) {
                // Check if we already have this symbol
                auto symbol = jit_->lookup(name);
                if (!symbol) {
                    consumeError(symbol.takeError());

                    // Allocate actual memory for the function pointer
                    // This allows us to update it when the real function is defined
                    void** ptr_slot = new void*;
                    *ptr_slot = reinterpret_cast<void*>(&__repl_forward_ref_stub);
                    forward_ref_slots_[name] = ptr_slot;

                    // Register the pointer slot address as the symbol
                    // When the module loads __repl_fwd_X, it gets this address,
                    // and loading from it gives the function pointer
                    orc::SymbolMap stub_symbol;
                    stub_symbol[jit_->mangleAndIntern(name)] = {
                        orc::ExecutorAddr::fromPtr(ptr_slot),
                        JITSymbolFlags::Exported
                    };

                    auto& main_dylib = jit_->getMainJITDylib();
                    auto err = main_dylib.define(orc::absoluteSymbols(stub_symbol));
                    if (err) {
                        consumeError(std::move(err));
                    }

                    pending_forward_refs_.insert(name);
                }
            }
        }
    }

    // Verify the module
    std::string error_msg;
    raw_string_ostream error_stream(error_msg);
    if (verifyModule(*module, &error_stream)) {
        std::cerr << "Module verification failed: " << error_msg << std::endl;
        module->print(errs(), nullptr);
        throw std::runtime_error("Invalid LLVM module");
    }

    // Wrap module in ThreadSafeModule using shared context
    auto tsm = ThreadSafeModule(std::move(module), *ts_context_);

    auto err = jit_->addIRModule(std::move(tsm));
    if (err) {
        std::string err_msg;
        raw_string_ostream err_stream(err_msg);
        err_stream << err;
        std::cerr << "Failed to add module to JIT: " << err_msg << std::endl;
        throw std::runtime_error("Failed to add module to JIT");
    }

    // STEP 3: Update forward reference pointers to point to the real functions
    for (const auto& [ptr_name, func_name] : forward_ref_updates) {
        auto func_symbol = jit_->lookup(func_name);
        if (func_symbol) {
            // Update the pointer slot to point to the real function
            auto it = forward_ref_slots_.find(ptr_name);
            if (it != forward_ref_slots_.end()) {
                void** ptr_slot = it->second;
                *ptr_slot = func_symbol->toPtr<void*>();
                pending_forward_refs_.erase(ptr_name);
            }
        } else {
            consumeError(func_symbol.takeError());
            std::cerr << "Warning: Could not resolve forward reference to " << func_name << std::endl;
        }
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

// Find the pre-compiled stdlib.o file
static std::string findStdlibObject() {
    std::vector<std::string> stdlib_paths = {
        "stdlib.o",
        "build/stdlib.o",
        "../build/stdlib.o",
        "/usr/local/lib/eshkol/stdlib.o",
        "/usr/lib/eshkol/stdlib.o",
    };

    // Check relative to executable
    char exe_path[4096];
    bool got_exe_path = false;

#ifdef __linux__
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len != -1) {
        exe_path[len] = '\0';
        got_exe_path = true;
    }
#elif defined(__APPLE__)
    uint32_t size = sizeof(exe_path);
    if (_NSGetExecutablePath(exe_path, &size) == 0) {
        got_exe_path = true;
    }
#endif

    if (got_exe_path) {
        std::filesystem::path exe_dir = std::filesystem::path(exe_path).parent_path();
        stdlib_paths.insert(stdlib_paths.begin(), (exe_dir / "stdlib.o").string());
        stdlib_paths.insert(stdlib_paths.begin(), (exe_dir / "../lib/eshkol/stdlib.o").string());
    }

    for (const auto& path : stdlib_paths) {
        if (std::filesystem::exists(path)) {
            return std::filesystem::canonical(path).string();
        }
    }
    return "";
}

bool ReplJITContext::loadStdlib() {
    // Try to load pre-compiled stdlib.o for fast loading
    std::string stdlib_obj_path = findStdlibObject();

    if (!stdlib_obj_path.empty()) {
        // Load the object file into memory
        auto buffer_or_err = MemoryBuffer::getFile(stdlib_obj_path);
        if (buffer_or_err) {
            auto& main_dylib = jit_->getMainJITDylib();

            // Add the object file to the JIT
            auto err = jit_->addObjectFile(main_dylib, std::move(*buffer_or_err));
            if (!err) {
                // Mark stdlib modules as loaded to prevent re-loading
                loaded_modules.insert("stdlib");
                loaded_modules.insert("core.io");
                loaded_modules.insert("core.operators.arithmetic");
                loaded_modules.insert("core.operators.compare");
                loaded_modules.insert("core.logic.predicates");
                loaded_modules.insert("core.logic.types");
                loaded_modules.insert("core.logic.boolean");
                loaded_modules.insert("core.functional.compose");
                loaded_modules.insert("core.functional.curry");
                loaded_modules.insert("core.functional.flip");
                loaded_modules.insert("core.control.trampoline");
                loaded_modules.insert("core.list.compound");
                loaded_modules.insert("core.list.generate");
                loaded_modules.insert("core.list.transform");
                loaded_modules.insert("core.list.query");
                loaded_modules.insert("core.list.sort");
                loaded_modules.insert("core.list.higher_order");
                loaded_modules.insert("core.list.search");
                loaded_modules.insert("core.list.convert");
                loaded_modules.insert("core.strings");
                loaded_modules.insert("core.json");
                loaded_modules.insert("core.data.csv");
                loaded_modules.insert("core.data.base64");

                // Register ALL stdlib symbols so the REPL codegen knows they're external
                // This prevents duplicate definition errors when compiling user code
                const std::vector<std::pair<std::string, size_t>> stdlib_funcs = {
                    // Arithmetic operators
                    {"add", 2}, {"sub", 2}, {"mul", 2}, {"div", 2}, {"negate", 1},
                    // Comparison
                    {"lt", 2}, {"le", 2}, {"gt", 2}, {"ge", 2}, {"eq", 2},
                    // List operations
                    {"filter", 2}, {"map1", 2}, {"map2", 3}, {"map3", 4},
                    {"fold", 3}, {"fold-right", 3}, {"length", 1},
                    {"reverse", 1}, {"append", 2}, {"take", 2}, {"drop", 2},
                    {"member", 2}, {"member?", 2}, {"assoc", 2}, {"assq", 2}, {"assv", 2},
                    {"memq", 2}, {"memv", 2}, {"range", 2},
                    {"list-ref", 2}, {"list-tail", 2}, {"list->vector", 1}, {"vector->list", 1},
                    {"partition", 2}, {"unzip", 1}, {"zip", 2},
                    {"first", 1}, {"second", 1}, {"third", 1}, {"fourth", 1}, {"fifth", 1},
                    {"sixth", 1}, {"seventh", 1}, {"eighth", 1}, {"ninth", 1}, {"tenth", 1},
                    {"find", 2}, {"count-if", 2}, {"every", 2}, {"any", 2},
                    {"all?", 2}, {"none?", 2}, {"make-list", 2}, {"repeat", 2}, {"iota", 1},
                    {"iota-from", 2}, {"iota-step", 3}, {"for-each", 2},
                    // Car/cdr variants
                    {"cadr", 1}, {"caddr", 1}, {"cadddr", 1}, {"caar", 1}, {"cdar", 1},
                    {"cddr", 1}, {"cdddr", 1}, {"cddddr", 1}, {"caaar", 1}, {"caadr", 1},
                    {"cadar", 1}, {"caddr", 1}, {"cdaar", 1}, {"cdadr", 1}, {"cddar", 1},
                    {"caaaar", 1}, {"caaadr", 1}, {"caadar", 1}, {"caaddr", 1},
                    {"cadaar", 1}, {"cadadr", 1}, {"caddar", 1}, {"cadddr", 1},
                    {"cdaaar", 1}, {"cdaadr", 1}, {"cdadar", 1}, {"cdaddr", 1},
                    {"cddaar", 1}, {"cddadr", 1}, {"cdddar", 1}, {"cddddr", 1},
                    // Higher-order
                    {"compose", 2}, {"compose3", 3}, {"identity", 1}, {"constantly", 1},
                    {"curry2", 1}, {"curry3", 1}, {"uncurry2", 1}, {"flip", 1},
                    {"partial", 2}, {"partial1", 2}, {"partial2", 2}, {"partial3", 2},
                    // Predicates
                    {"is-zero?", 1}, {"is-positive?", 1}, {"is-negative?", 1},
                    {"is-even?", 1}, {"is-odd?", 1}, {"is-null?", 1}, {"is-pair?", 1},
                    // Trampoline
                    {"trampoline", 1}, {"bounce", 1}, {"done", 1},
                    // String functions
                    {"string-join", 2}, {"string-trim", 1}, {"string-trim-left", 1}, {"string-trim-right", 1},
                    {"string-upcase", 1}, {"string-downcase", 1}, {"string-reverse", 1},
                    {"string-replace", 3}, {"string-repeat", 2}, {"string-copy", 1},
                    {"string-contains", 2}, {"string-index", 2}, {"string-last-index", 2},
                    {"string-starts-with?", 2}, {"string-ends-with?", 2}, {"string-count", 2},
                    {"string-split-ordered", 2}, {"string->bytes", 1}, {"bytes->string", 1},
                    // Sorting
                    {"sort", 2},
                    // JSON
                    {"json-parse", 1}, {"json-stringify", 1}, {"json-get", 2}, {"json-array-ref", 2},
                    {"json-read-file", 1}, {"json-write-file", 2},
                    {"alist->json", 1}, {"alist-write-json", 2}, {"alist->hash-table", 1}, {"hash-table->alist", 1},
                    // Base64
                    {"base64-encode", 1}, {"base64-decode", 1},
                    {"base64-encode-string", 1}, {"base64-decode-string", 1},
                    {"base64-char-at", 2}, {"base64-value", 1}, {"base64-remove-padding", 1},
                    // CSV
                    {"csv-parse", 1}, {"csv-stringify", 1}, {"csv-parse-file", 1}, {"csv-write-file", 2},
                    {"csv-parse-line", 1}, {"csv-parse-lines", 1}, {"csv-split-fields", 1}, {"csv-stringify-row", 1},
                    // I/O
                    {"print", 1}, {"println", 1},
                };

                for (const auto& [name, arity] : stdlib_funcs) {
                    auto sym = jit_->lookup(name);
                    if (sym) {
                        uint64_t addr = sym->getValue();
                        eshkol_repl_register_function(name.c_str(), addr, arity);
                        defined_lambdas_[name] = {name, arity};
                        registered_lambdas_.insert(name);
                    }
                }

                // Also register all _sexpr globals so codegen doesn't try to redefine them
                const std::vector<std::string> stdlib_globals = {
                    "add_sexpr", "sub_sexpr", "mul_sexpr", "div_sexpr", "negate_sexpr",
                    "lt_sexpr", "le_sexpr", "gt_sexpr", "ge_sexpr", "eq_sexpr",
                    "filter_sexpr", "map1_sexpr", "map2_sexpr", "map3_sexpr",
                    "fold_sexpr", "fold-right_sexpr", "length_sexpr",
                    "reverse_sexpr", "append_sexpr", "take_sexpr", "drop_sexpr",
                    "partition_sexpr", "sort_sexpr", "list->vector_sexpr",
                    "all?_sexpr", "is-positive?_sexpr", "is-pair?_sexpr",
                    "compose3_sexpr", "partial3_sexpr", "flip_sexpr", "bounce_sexpr",
                    "cadadr_sexpr", "find_sexpr", "make-list_sexpr", "assoc_sexpr",
                    "string-trim-right_sexpr", "json-parse_sexpr", "csv-stringify_sexpr",
                    "base64-encode-string_sexpr", "print_sexpr",
                };

                for (const auto& name : stdlib_globals) {
                    auto sym = jit_->lookup(name);
                    if (sym) {
                        uint64_t addr = sym->getValue();
                        eshkol_repl_register_symbol(name.c_str(), addr);
                        defined_globals_.insert(name);
                    }
                }

                return true;
            } else {
                // Log the error but fall through
                std::string err_msg;
                raw_string_ostream err_stream(err_msg);
                err_stream << err;
                std::cerr << "Warning: Failed to load stdlib.o (" << err_msg << "), falling back to JIT compilation" << std::endl;
            }
        }
    }

    // Fallback: JIT compile from source (slow)
    return loadModule("stdlib");
}

bool ReplJITContext::loadModule(const std::string& module_name) {
    // Check if already loaded by NAME first (for stdlib.o preloaded modules)
    if (loaded_modules.count(module_name)) {
        return true;  // Already loaded via stdlib.o
    }

    // For stdlib or core.* modules, use precompiled stdlib.o if available
    if (module_name == "stdlib" || module_name.find("core.") == 0) {
        // Try to load via stdlib.o (which includes all core modules)
        if (loadStdlib()) {
            return true;
        }
        // If stdlib.o loading failed, fall through to JIT compilation
    }

    std::string module_path = resolveModulePath(module_name);

    if (module_path.empty()) {
        std::cerr << "Module not found: " << module_name << std::endl;
        return false;
    }

    // Check if already loaded by PATH
    if (loaded_modules.count(module_path)) {
        return true;  // Already loaded, success
    }
    loaded_modules.insert(module_path);
    loaded_modules.insert(module_name);  // Also track by name

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

    // SINGLE-PASS MODULE LOADING with deferred batch compilation:
    // - Process require/import immediately (load dependencies)
    // - Collect other ASTs for batch compilation (allows forward references)
    std::vector<eshkol_ast_t> batch_asts;
    batch_asts.reserve(module_asts.size());  // Pre-allocate for efficiency

    for (auto& ast_item : module_asts) {
        if (ast_item.type == ESHKOL_OP) {
            if (ast_item.operation.op == ESHKOL_REQUIRE_OP ||
                ast_item.operation.op == ESHKOL_IMPORT_OP) {
                // Process dependencies immediately
                try {
                    execute(&ast_item);
                } catch (const std::exception& e) {
                    // Continue even if a dependency fails
                }
                continue;
            }
            if (ast_item.operation.op == ESHKOL_PROVIDE_OP) {
                // Skip provide statements (no-op in REPL)
                continue;
            }
        }
        batch_asts.push_back(ast_item);
    }

    // Batch-compile all definitions together (allows forward references)
    if (!batch_asts.empty()) {
        try {
            executeBatch(batch_asts, true);  // silent = true for module loading
        } catch (const std::exception& e) {
            std::cerr << "     error: " << e.what() << std::endl;
        }
    }

    // Clean up ASTs
    for (auto& ast_item : module_asts) {
        eshkol_ast_clean(&ast_item);
    }

    return true;
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

    // Check relative to executable
    char exe_path[4096];
    bool got_exe_path = false;

#ifdef __linux__
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len != -1) {
        exe_path[len] = '\0';
        got_exe_path = true;
    }
#elif defined(__APPLE__)
    uint32_t size = sizeof(exe_path);
    if (_NSGetExecutablePath(exe_path, &size) == 0) {
        got_exe_path = true;
    }
#endif

    if (got_exe_path) {
        std::filesystem::path exe_dir = std::filesystem::path(exe_path).parent_path();
        lib_dirs.insert(lib_dirs.begin(), (exe_dir / "../lib").string());
        lib_dirs.insert(lib_dirs.begin(), (exe_dir / "lib").string());
    }

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

void* ReplJITContext::executeBatch(std::vector<eshkol_ast_t>& asts, bool silent) {
    if (asts.empty()) {
        return nullptr;
    }

    // Pre-register all lambda variables so they're tracked
    for (auto& ast_item : asts) {
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
    }

    // Generate LLVM IR for ALL ASTs together using the existing Eshkol compiler
    // This allows forward references between functions in the same batch
    std::string module_name = "__repl_batch_" + std::to_string(eval_counter_);

    // Call the existing compiler to generate LLVM IR from ALL ASTs at once
    // (asts vector is already contiguous, no need to copy)
    LLVMModuleRef c_module = eshkol_generate_llvm_ir(asts.data(), asts.size(), module_name.c_str());

    if (!c_module) {
        if (!silent) {
            std::cerr << "Failed to generate LLVM IR from batch" << std::endl;
        }
        return nullptr;
    }

    // Convert LLVMModuleRef (C API) to llvm::Module* (C++ API)
    Module* cpp_module = llvm::unwrap(c_module);

    if (!cpp_module) {
        if (!silent) {
            std::cerr << "Failed to unwrap LLVM module" << std::endl;
        }
        return nullptr;
    }

    // REPL SYMBOL PERSISTENCE: Inject declarations for previously-defined symbols
    injectPreviousSymbols(cpp_module);

    // REPL SYMBOL TRACKING: Extract lambda/function definitions from this module
    for (auto& func : cpp_module->functions()) {
        if (func.isDeclaration() || func.getName().starts_with("llvm.")) {
            continue;
        }
        std::string fname = func.getName().str();

        // Track lambda functions (they start with "lambda_")
        if (fname.find("lambda_") == 0) {
            size_t arity = func.arg_size();
            for (auto& [var_name, lambda_info] : defined_lambdas_) {
                if (lambda_info.first.empty()) {
                    defined_lambdas_[var_name] = {fname, arity};
                    break;
                }
            }
        }
        // Track user-defined functions
        else if (defined_lambdas_.find(fname) != defined_lambdas_.end()) {
            auto& lambda_info = defined_lambdas_[fname];
            if (lambda_info.first.empty()) {
                size_t arity = func.arg_size();
                defined_lambdas_[fname] = {fname, arity};
            }
        }
    }

    // Find entry function
    Function* entry_func = nullptr;
    for (auto& func : cpp_module->functions()) {
        if (func.isDeclaration() || func.getName().starts_with("llvm.")) {
            continue;
        }
        std::string fname = func.getName().str();
        if (fname.find("__top_level") != std::string::npos ||
            fname.find("main") != std::string::npos) {
            entry_func = &func;
            break;
        }
    }

    if (!entry_func) {
        for (auto& func : cpp_module->functions()) {
            if (func.isDeclaration() || func.getName().starts_with("llvm.")) {
                continue;
            }
            std::string fname = func.getName().str();
            if (fname.find("display") != std::string::npos ||
                fname.find("print") != std::string::npos ||
                fname.find("__internal") != std::string::npos ||
                func.hasLocalLinkage()) {
                continue;
            }
            entry_func = &func;
            break;
        }
    }

    std::string func_name;
    if (entry_func) {
        func_name = entry_func->getName().str();
        std::string unique_func_name = "__repl_batch_eval_" + std::to_string(eval_counter_);
        entry_func->setName(unique_func_name);
        func_name = unique_func_name;
    }

    // Extract global variable names before adding module
    std::vector<std::string> global_var_names;
    for (auto& global_var : cpp_module->globals()) {
        std::string var_name = global_var.getName().str();
        if (global_var.isDeclaration() || global_var.getName().starts_with("llvm.")) {
            continue;
        }
        if (var_name.find("__") == 0 || var_name.find("_func") != std::string::npos) {
            continue;
        }
        global_var_names.push_back(var_name);
    }

    // Release ownership and add module to JIT
    eshkol_release_module_for_jit(c_module);
    addModule(std::unique_ptr<Module>(cpp_module));

    // Register all lambda functions
    for (const auto& [var_name, lambda_info] : defined_lambdas_) {
        const auto& [lambda_name, arity] = lambda_info;
        if (!lambda_name.empty()) {
            if (registered_lambdas_.find(lambda_name) != registered_lambdas_.end()) {
                continue;
            }
            uint64_t lambda_addr = lookupSymbol(lambda_name);
            if (lambda_addr != 0) {
                eshkol_repl_register_function(lambda_name.c_str(), lambda_addr, arity);
                eshkol_repl_register_function((var_name + "_func").c_str(), lambda_addr, arity);
                eshkol_repl_register_lambda_name(var_name.c_str(), lambda_name.c_str());
                registered_lambdas_.insert(lambda_name);
            }
        }
    }

    // Register all global variables
    for (const auto& var_name : global_var_names) {
        uint64_t var_addr = lookupSymbol(var_name);
        if (var_addr != 0) {
            defined_globals_.insert(var_name);
            eshkol_repl_register_symbol(var_name.c_str(), var_addr);
        }
    }

    // Execute if we found an entry function
    void* result = nullptr;
    if (entry_func && !func_name.empty()) {
        uint64_t func_addr = lookupSymbol(func_name);
        if (func_addr != 0) {
            incrementEvalCounter();
            typedef int32_t (*EvalFunc)();
            EvalFunc eval_func = reinterpret_cast<EvalFunc>(func_addr);
            int32_t result_value = eval_func();
            result = new int64_t(result_value);
        }
    } else {
        // No entry function - just increment counter (defines only)
        incrementEvalCounter();
    }

    return result;
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

            // SINGLE-PASS IMPORT LOADING with deferred batch compilation
            std::vector<eshkol_ast_t> batch_asts;
            batch_asts.reserve(file_asts.size());

            for (auto& ast_item : file_asts) {
                if (ast_item.type == ESHKOL_OP) {
                    if (ast_item.operation.op == ESHKOL_REQUIRE_OP ||
                        ast_item.operation.op == ESHKOL_IMPORT_OP) {
                        try {
                            execute(&ast_item);
                        } catch (const std::exception& e) {
                            // Continue even if a dependency fails
                        }
                        continue;
                    }
                    if (ast_item.operation.op == ESHKOL_PROVIDE_OP) {
                        continue;
                    }
                }
                batch_asts.push_back(ast_item);
            }

            void* last_result = nullptr;
            if (!batch_asts.empty()) {
                try {
                    last_result = executeBatch(batch_asts, true);
                } catch (const std::exception& e) {
                    // Silently continue
                }
            }

            // Clean up ASTs
            for (auto& ast_item : file_asts) {
                eshkol_ast_clean(&ast_item);
            }

            return last_result;
        }

        // Handle (require module.name ...)
        // Use loadModule which now does proper two-pass batch loading
        if (ast->operation.op == ESHKOL_REQUIRE_OP) {
            for (size_t i = 0; i < ast->operation.require_op.num_modules; i++) {
                std::string module_name = ast->operation.require_op.module_names[i];
                loadModule(module_name);
            }
            return nullptr;
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
