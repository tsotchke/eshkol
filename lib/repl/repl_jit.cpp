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

using namespace llvm;
using namespace llvm::orc;

namespace eshkol {

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

    // Helper macro to add a symbol
    #define ADD_SYMBOL(name) \
        symbols[ES.intern(#name)] = { \
            orc::ExecutorAddr::fromPtr((void*)&name), \
            JITSymbolFlags::Callable | JITSymbolFlags::Exported \
        }

    // Arena memory management functions
    ADD_SYMBOL(arena_create);
    ADD_SYMBOL(arena_destroy);
    ADD_SYMBOL(arena_allocate);
    ADD_SYMBOL(arena_push_scope);
    ADD_SYMBOL(arena_pop_scope);
    ADD_SYMBOL(arena_allocate_cons_cell);
    ADD_SYMBOL(arena_allocate_tagged_cons_cell);
    ADD_SYMBOL(arena_tagged_cons_get_int64);
    ADD_SYMBOL(arena_tagged_cons_get_double);
    ADD_SYMBOL(arena_tagged_cons_get_ptr);
    ADD_SYMBOL(arena_tagged_cons_set_int64);
    ADD_SYMBOL(arena_tagged_cons_set_double);
    ADD_SYMBOL(arena_tagged_cons_set_ptr);
    ADD_SYMBOL(arena_tagged_cons_set_null);
    ADD_SYMBOL(arena_tagged_cons_get_type);
    ADD_SYMBOL(arena_tagged_cons_set_tagged_value);
    ADD_SYMBOL(arena_tagged_cons_get_tagged_value);
    ADD_SYMBOL(arena_allocate_tape);
    ADD_SYMBOL(arena_tape_add_node);
    ADD_SYMBOL(arena_tape_reset);
    ADD_SYMBOL(arena_allocate_ad_node);
    ADD_SYMBOL(arena_tape_get_node);
    ADD_SYMBOL(arena_tape_get_node_count);

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

    // Register global AD tape pointer (shared across all JIT modules for gradient/jacobian operations)
    // Reference global namespace symbol (defined in arena_memory.cpp, declared in arena_memory.h)
    symbols[ES.intern("__current_ad_tape")] = {
        orc::ExecutorAddr::fromPtr((void*)&::__current_ad_tape),
        JITSymbolFlags::Exported  // NOT Callable - this is a data symbol
    };

    // Register global AD mode flag (shared across all JIT modules for gradient/jacobian operations)
    // CRITICAL: This must be shared so lambdas from one module can see AD mode set by another
    symbols[ES.intern("__ad_mode_active")] = {
        orc::ExecutorAddr::fromPtr((void*)&::__ad_mode_active),
        JITSymbolFlags::Exported  // NOT Callable - this is a data symbol
    };
    std::cout << "Registered __ad_mode_active at " << (void*)&::__ad_mode_active << std::endl;

    // Register global shared arena pointer (shared across all REPL evaluations)
    symbols[ES.intern("__repl_shared_arena")] = {
        orc::ExecutorAddr::fromPtr((void*)&::__repl_shared_arena),
        JITSymbolFlags::Exported  // NOT Callable - this is a data symbol
    };

    #undef ADD_SYMBOL

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

    std::cout << "Registered " << symbols.size() << " runtime symbols" << std::endl;
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

void* ReplJITContext::execute(eshkol_ast_t* ast) {
    if (!ast) {
        throw std::runtime_error("Cannot execute null AST");
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

    // REPL SYMBOL TRACKING: Extract lambda functions defined in this module
    // Fill in lambda names for variables that were pre-registered
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

                std::cout << "REPL: Registered " << var_name << " -> " << lambda_name << " (arity " << arity << ")" << std::endl;
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

            // Only print registration message for non-string-constant globals
            if (!var_name.starts_with(".str")) {
                std::cout << "REPL: Registered variable " << var_name << " @ 0x" << std::hex << var_addr << std::dec << std::endl;
            }

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
