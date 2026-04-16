//
// Copyright (C) tsotchke
//
// SPDX-License-Identifier: MIT
//

#include "repl_jit.h"
#include <eshkol/eshkol.h>
#include <eshkol/llvm_backend.h>
#include <eshkol/platform_runtime.h>
#include <eshkol/runtime_exports.h>
#include <eshkol/model_io.h>
#include <eshkol/core/bignum.h>
#include <eshkol/core/rational.h>
#include <eshkol/types/hott_types.h>  // For TypeId decoding and BuiltinTypes
#include "../core/arena_memory.h"  // For runtime function declarations
#include <eshkol/backend/blas_backend.h>  // For BLAS runtime functions

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/CFG.h>  // For predecessors()
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/DynamicLibrary.h>
#include <llvm/Support/MemoryBuffer.h>  // For loading object files
#include <llvm/Bitcode/BitcodeReader.h>  // For loading .bc files
#include <llvm/TargetParser/SubtargetFeature.h>  // For SubtargetFeatures
#include <llvm/MC/TargetRegistry.h>              // For TargetRegistry
#include <llvm/Target/TargetMachine.h>           // For TargetMachine
#include <llvm/TargetParser/Host.h>              // For sys::getHostCPUName/Features

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#ifdef _WIN32
#include <malloc.h>           // _aligned_malloc / _aligned_free
#endif
#include <filesystem>
#include <set>
#include <vector>
#include <cctype>

static constexpr char eshkol_path_separator =
#ifdef _WIN32
    ';';
#else
    ':';
#endif

using namespace llvm;

// Forward declaration — defined in llvm_codegen.cpp (C++ linkage)
// Extracts the module's original LLVMContext and releases module ownership from g_llvm_modules.
std::unique_ptr<LLVMContext> eshkol_extract_module_context_for_jit(LLVMModuleRef module_ref);

static llvm::Module* module_from_ref(LLVMModuleRef module_ref) {
    return reinterpret_cast<llvm::Module*>(module_ref);
}

// ===== EXTERN DECLARATIONS FOR RUNTIME SYMBOLS =====
// These are defined in runtime.cpp with extern "C" linkage
extern "C" {
    void eshkol_type_error(const char* proc_name, const char* expected_type);
    void eshkol_type_error_with_value(const char* proc_name, const char* expected_type,
                                       const char* actual_type);
    int64_t eshkol_shapes_equal(const int64_t* shape_a, const int64_t* shape_b, int64_t rank);
    int64_t eshkol_broadcast_elementwise_f64(
        const double* a_data, const int64_t* a_shape, int64_t a_rank,
        const double* b_data, const int64_t* b_shape, int64_t b_rank,
        double* out_data, const int64_t* out_shape, int64_t out_rank,
        int64_t op);
    int64_t eshkol_check_recursion_depth(void);
    void eshkol_decrement_recursion_depth(void);
    int64_t eshkol_utf8_strlen(const char* s);
    int64_t eshkol_utf8_ref(const char* s, int64_t k);
    char* eshkol_utf8_substring(const char* s, int64_t start, int64_t end, void* arena);
#ifdef _WIN32
    double drand48(void);
    int clock_gettime(int clock_id, void* ts_raw);
#endif
}

// ===== PARALLEL EXECUTION RUNTIME (parallel_codegen.cpp) =====
// eshkol_tagged_value_t and arena_t are already declared via eshkol.h included above
extern "C" {
    void eshkol_parallel_map(eshkol_tagged_value_t fn,
                              eshkol_tagged_value_t list,
                              arena_t* arena,
                              eshkol_tagged_value_t* out_result);
    void eshkol_parallel_fold(eshkol_tagged_value_t fn,
                               eshkol_tagged_value_t init,
                               eshkol_tagged_value_t list,
                               arena_t* arena,
                               eshkol_tagged_value_t* out_result);
    void eshkol_parallel_filter(eshkol_tagged_value_t pred,
                                 eshkol_tagged_value_t list,
                                 arena_t* arena,
                                 eshkol_tagged_value_t* out_result);
    void eshkol_parallel_for_each(eshkol_tagged_value_t fn,
                                   eshkol_tagged_value_t list,
                                   arena_t* arena);
    int64_t eshkol_thread_pool_num_threads(void);
    void eshkol_thread_pool_print_stats(void);

    // Worker registration function (called by LLVM-generated module initializer)
    void __eshkol_register_parallel_workers(void* map_worker, void* fold_worker,
                                            void* filter_worker, void* unary_dispatcher,
                                            void* binary_dispatcher);
    bool eshkol_parallel_workers_registered(void);
}

// Track already-loaded modules to prevent circular imports
static std::set<std::string> loaded_modules;
using namespace llvm::orc;

namespace eshkol {

// Forward declarations for static helper functions
static std::vector<eshkol_ast_t> parseAllAstsFromString(const std::string& content);
static std::string resolveModulePath(const std::string& module_name, const std::string& base_dir = ".");

ReplJITContext::ReplJITContext()
    : jit_(nullptr)
    , eval_counter_(0)
    , shared_arena_(nullptr)
{
    // Enable REPL-mode codegen immediately so modules compiled before LLJIT
    // startup still emit shared runtime globals as extern declarations.
    eshkol_repl_enable();
}

ReplJITContext::~ReplJITContext() {
    // Free all forward-reference pointer slots allocated with 'new void*'
    for (auto& [name, ptr_slot] : forward_ref_slots_) {
        delete ptr_slot;
    }
    forward_ref_slots_.clear();

    // LLJIT destructor handles remaining cleanup
}

void ReplJITContext::initializeJIT() {
    // Match the batch compiler's target initialization so the Windows LLVM SDK
    // exposes registered targets to LLJIT before host detection runs.
    InitializeAllTargetInfos();
    InitializeAllTargets();
    InitializeAllTargetMCs();
    InitializeAllAsmParsers();
    InitializeAllAsmPrinters();

    // Load symbols from current process (includes eshkol-static runtime)
    // This makes arena_*, printf, malloc, etc. available to JIT code
    sys::DynamicLibrary::LoadLibraryPermanently(nullptr);

    // CRITICAL: Build a JITTargetMachineBuilder with the ACTUAL host CPU and features.
    // Without this, LLJIT's ConcurrentIRCompiler uses a default TM that may scalarize
    // struct arguments differently from stdlib.o's TM, breaking 3+ arg function calls.
    auto jtmb = orc::JITTargetMachineBuilder::detectHost();
    if (!jtmb) {
        std::cerr << "Failed to detect host for JIT: " << toString(jtmb.takeError()) << std::endl;
        std::exit(1);
    }
    // Ensure PIC relocation model (matches stdlib.o compilation)
    jtmb->setRelocationModel(Reloc::PIC_);
    // CRITICAL: Match the batch compiler's optimization level (CodeGenOptLevel::None = -O0).
    // JITTargetMachineBuilder::detectHost() defaults to CodeGenOptLevel::Default (-O2),
    // which causes LLVM to generate different struct argument stack layouts on ARM64.
    // stdlib.o is compiled at -O0, so the JIT must use the same level to ensure
    // matching ABI for {i8,i8,i16,i32,i64} tagged value arguments.
#if LLVM_VERSION_MAJOR >= 18
    jtmb->setCodeGenOptLevel(CodeGenOptLevel::None);
#else
    jtmb->setCodeGenOptLevel(CodeGenOpt::None);
#endif

    // Create LLJIT instance with explicit host-matched TM
    auto jit_or_err = LLJITBuilder()
        .setJITTargetMachineBuilder(std::move(*jtmb))
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

    // Use the runtime global arena for REPL allocations so precompiled stdlib.o
    // and JIT-generated REPL modules allocate into the same live arena.
    shared_arena_ = get_global_arena();
    __repl_shared_arena.store(static_cast<arena_t*>(shared_arena_));

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
    ADD_SYMBOL(eshkol_display_value);
    ADD_SYMBOL(eshkol_lambda_registry_init);
    ADD_SYMBOL(eshkol_lambda_registry_destroy);
    ADD_SYMBOL(eshkol_lambda_registry_add);
    ADD_SYMBOL(eshkol_lambda_registry_lookup);
    ADD_SYMBOL(eshkol_init_stack_size);
    ADD_DATA_SYMBOL(g_lambda_registry);

    // ===== EXCEPTION HANDLING =====
    ADD_SYMBOL(eshkol_raise);
    ADD_SYMBOL(eshkol_make_exception);
    ADD_SYMBOL(eshkol_make_exception_with_header);
    ADD_SYMBOL(eshkol_push_exception_handler);
    ADD_SYMBOL(eshkol_pop_exception_handler);
    ADD_SYMBOL(eshkol_exception_type_matches);
    ADD_SYMBOL(eshkol_unwind_dynamic_wind);
    ADD_SYMBOL(eshkol_check_recursion_depth);
    ADD_SYMBOL(eshkol_decrement_recursion_depth);
    ADD_DATA_SYMBOL(g_current_exception);
    ADD_DATA_SYMBOL(g_exception_handler_stack);

    // ===== PLATFORM RUNTIME EXPORTS =====
    ADD_SYMBOL(eshkol_stdout_stream);
    ADD_SYMBOL(eshkol_drand48);
    ADD_SYMBOL(eshkol_srand48);
#ifdef _WIN32
    ADD_SYMBOL(drand48);
    ADD_SYMBOL(clock_gettime);
#endif
    ADD_SYMBOL(eshkol_getenv);
    ADD_SYMBOL(eshkol_setenv);
    ADD_SYMBOL(eshkol_unsetenv);
    ADD_SYMBOL(eshkol_usleep);
    ADD_SYMBOL(eshkol_fopen);
    ADD_SYMBOL(eshkol_access);
    ADD_SYMBOL(eshkol_remove);
    ADD_SYMBOL(eshkol_rename);
    ADD_SYMBOL(eshkol_mkdir);
    ADD_SYMBOL(eshkol_rmdir);
    ADD_SYMBOL(eshkol_chdir);
    ADD_SYMBOL(eshkol_stat);
    ADD_SYMBOL(eshkol_opendir);
    symbols[ES.intern("snprintf")] = {
        orc::ExecutorAddr::fromPtr((void*)&::snprintf),
        JITSymbolFlags::Callable | JITSymbolFlags::Exported
    };

    // ===== TYPE ERRORS (R7RS Compliance) =====
    // Use global scope since these are declared extern "C" in global namespace
    symbols[ES.intern("eshkol_type_error")] = {
        orc::ExecutorAddr::fromPtr((void*)&::eshkol_type_error),
        JITSymbolFlags::Callable | JITSymbolFlags::Exported
    };
    symbols[ES.intern("eshkol_type_error_with_value")] = {
        orc::ExecutorAddr::fromPtr((void*)&::eshkol_type_error_with_value),
        JITSymbolFlags::Callable | JITSymbolFlags::Exported
    };

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
    ADD_SYMBOL(eshkol_tensor_save_tagged);
    ADD_SYMBOL(eshkol_tensor_load_tagged);
    ADD_SYMBOL(eshkol_model_save_tagged);
    ADD_SYMBOL(eshkol_model_load_tagged);
    ADD_SYMBOL(eshkol_shapes_equal);
    ADD_SYMBOL(eshkol_broadcast_elementwise_f64);

    // ===== BIGNUM / RATIONAL NUMERICS =====
    ADD_SYMBOL(eshkol_bignum_from_overflow);
    ADD_SYMBOL(eshkol_bignum_from_int64);
    ADD_SYMBOL(eshkol_is_bignum_tagged);
    ADD_SYMBOL(eshkol_bignum_binary_tagged);
    ADD_SYMBOL(eshkol_bignum_neg);
    ADD_SYMBOL(eshkol_bignum_pow_tagged);
    ADD_SYMBOL(eshkol_bignum_to_double);
    ADD_SYMBOL(eshkol_bignum_to_string);
    ADD_SYMBOL(eshkol_bignum_is_zero);
    ADD_SYMBOL(eshkol_bignum_is_even);
    ADD_SYMBOL(eshkol_bignum_is_odd);
    ADD_SYMBOL(eshkol_string_to_number_tagged);
    ADD_SYMBOL(eshkol_rational_create);
    ADD_SYMBOL(eshkol_rational_to_double);
    ADD_SYMBOL(eshkol_rational_to_string);
    ADD_SYMBOL(eshkol_is_rational_tagged_ptr);
    ADD_SYMBOL(eshkol_rational_binary_tagged_ptr);
    ADD_SYMBOL(eshkol_rational_compare_tagged_ptr);
    ADD_SYMBOL(eshkol_bignum_compare_tagged);
    ADD_SYMBOL(eshkol_utf8_strlen);
    ADD_SYMBOL(eshkol_utf8_ref);
    ADD_SYMBOL(eshkol_utf8_substring);

    // ===== BLAS ACCELERATION =====
    // Runtime matmul with automatic BLAS/scalar dispatch
    ADD_SYMBOL(eshkol_matmul_f64);
    ADD_SYMBOL(eshkol_blas_available);
    ADD_SYMBOL(eshkol_blas_get_threshold);
    ADD_SYMBOL(eshkol_blas_set_threshold);

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

    // ===== PARALLEL EXECUTION RUNTIME =====
    ADD_SYMBOL(eshkol_parallel_map);
    ADD_SYMBOL(eshkol_parallel_fold);
    ADD_SYMBOL(eshkol_parallel_filter);
    ADD_SYMBOL(eshkol_parallel_for_each);
    ADD_SYMBOL(eshkol_thread_pool_num_threads);
    ADD_SYMBOL(eshkol_thread_pool_print_stats);
    ADD_SYMBOL(__eshkol_register_parallel_workers);
    ADD_SYMBOL(eshkol_parallel_workers_registered);

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

// Stub function called when a forward-referenced function hasn't been defined yet
static eshkol_tagged_value __repl_forward_ref_stub() {
    // Raise an exception so that (guard ...) can catch it in user code
    eshkol_exception_t* exc = eshkol_make_exception(
        ESHKOL_EXCEPTION_ERROR,
        "called a forward-referenced function that was never defined");
    if (exc) {
        eshkol_raise(exc);
    }
    // If raise returns (no handler installed), fall back to returning null
    eshkol_tagged_value result = {};
    return result;
}

static bool is_repl_runtime_global(const llvm::GlobalVariable& global_var) {
    auto name = global_var.getName();
    return name == "__eshkol_argc" ||
           name == "__eshkol_argv" ||
           name == "__repl_shared_arena";
}

// REPL HOT RELOAD: derive the user-visible name from a versioned LLVM symbol
// emitted by createFunctionDeclaration. Top-level user functions are emitted
// as "<name>__rv<N>" so each redefinition is a unique JIT symbol — this helper
// recovers the original "<name>" so we can register both forms in the REPL
// registries (versioned for direct JIT lookup, unversioned for code that uses
// the function as a value, e.g. (map sq lst)). Returns "" if `vname` is not
// in versioned form.
static std::string strip_repl_version_suffix(const std::string& vname) {
    auto pos = vname.rfind("__rv");
    if (pos == std::string::npos || pos == 0) return "";
    if (pos + 4 >= vname.size()) return "";
    for (size_t i = pos + 4; i < vname.size(); i++) {
        if (vname[i] < '0' || vname[i] > '9') return "";
    }
    return vname.substr(0, pos);
}

void ReplJITContext::addModule(std::unique_ptr<Module> module, std::unique_ptr<LLVMContext> module_context) {
    if (!jit_) {
        initializeJIT();
    }

    // These globals are provided by the host process and registered into the
    // JIT dylib explicitly. Leaving definitions in each REPL module causes
    // duplicate-definition failures on Windows when the first module is added.
    for (auto& gv : module->globals()) {
        if (is_repl_runtime_global(gv) && gv.hasInitializer()) {
            gv.setInitializer(nullptr);
            gv.setLinkage(GlobalValue::ExternalLinkage);
            gv.setExternallyInitialized(false);
        }
    }

    // Collect forward references and definitions to handle
    std::vector<std::pair<std::string, std::string>> forward_ref_updates;  // (ptr_name, func_name)

    // STEP 1: Scan for __repl_fwd_<X> globals with initializers — these are
    // codegen-emitted markers that say "the slot named X should now point at
    // function Y". This is the universal hot-reload mechanism: every REPL user
    // function definition emits one of these, and we strip the IR-level global
    // so the absolute heap slot defined in STEP 2 owns the symbol. STEP 3
    // updates the slot once the new function has been materialized in the JIT.
    for (auto& gv : module->globals()) {
        if (gv.hasInitializer()) {
            std::string name = gv.getName().str();
            if (name.find("__repl_fwd_") == 0) {
                if (auto* func = dyn_cast<Function>(gv.getInitializer())) {
                    forward_ref_updates.push_back({name, func->getName().str()});
                }
                // Strip the initializer and force ExternalLinkage so the symbol
                // is resolved against the absolute heap slot, not defined inline.
                gv.setInitializer(nullptr);
                gv.setLinkage(GlobalValue::ExternalLinkage);
                gv.setExternallyInitialized(false);
            }
        }
    }

    // STEP 1B: Scan for __repl_var_<X> markers — codegen emits these whenever
    // a top-level user variable @X is declared as an external in this module.
    // For each marker we (1) ensure a 16-byte tagged_value heap slot exists for
    // X (allocated lazily on first definition; reused on redefinition), (2)
    // register `@X` as an absolute JIT symbol pointing at that slot, so all
    // present and future modules' load/store of `@X` go through shared host
    // storage, and (3) drop the marker from IR so the JIT does not materialize
    // it. The marker itself carries no data — its sole purpose is to identify
    // host-managed variable externals so we don't accidentally back random
    // unresolved externals (runtime symbols, builtin globals, etc.).
    {
        std::vector<std::string> var_markers;
        for (auto& gv : module->globals()) {
            std::string name = gv.getName().str();
            if (name.find("__repl_var_") == 0) {
                var_markers.push_back(name);
            }
        }
        for (const std::string& marker : var_markers) {
            std::string var_name = marker.substr(strlen("__repl_var_"));

            if (repl_var_storage_.count(var_name) == 0) {
                // First definition: allocate a 16-byte aligned tagged_value
                // slot, zero-initialize it, and register the variable's name
                // as an absolute symbol pointing at the slot. Subsequent
                // definitions reuse the same address, so a store from any
                // module's entry function writes to shared storage.
#ifdef _WIN32
                void* storage = _aligned_malloc(16, 16);
#else
                void* storage = nullptr;
                if (posix_memalign(&storage, 16, 16) != 0) storage = nullptr;
#endif
                if (!storage) {
                    std::cerr << "REPL: failed to allocate storage for variable " << var_name << std::endl;
                    continue;
                }
                std::memset(storage, 0, 16);
                repl_var_storage_[var_name] = storage;

                orc::SymbolMap sym;
                sym[jit_->mangleAndIntern(var_name)] = {
                    orc::ExecutorAddr::fromPtr(storage),
                    JITSymbolFlags::Exported
                };
                auto& main_dylib = jit_->getMainJITDylib();
                if (auto err = main_dylib.define(orc::absoluteSymbols(sym))) {
                    std::string err_msg;
                    raw_string_ostream err_stream(err_msg);
                    err_stream << err;
                    consumeError(std::move(err));
                    std::cerr << "REPL: failed to register absolute symbol for "
                              << var_name << ": " << err_msg << std::endl;
                }

                // Register in the codegen-visible symbol map so future
                // modules' variable read paths see this name and emit an
                // external @<name> declaration that resolves to our slot.
                defined_globals_.insert(var_name);
                eshkol_repl_register_symbol(
                    var_name.c_str(),
                    reinterpret_cast<uint64_t>(storage));
            }

            if (auto* gv = module->getNamedGlobal(marker)) {
                gv->eraseFromParent();
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

    // HOT RELOAD: Only remove user-defined symbols being redefined.
    // Skip LinkOnceODR (shared codegen like builtin_+_2arg) and internal functions.
    {
        orc::SymbolNameSet to_remove;
        for (auto& func : *module) {
            if (func.isDeclaration() || func.hasLocalLinkage()) continue;
            if (func.getName().starts_with("llvm.")) continue;
            if (func.getLinkage() == GlobalValue::LinkOnceODRLinkage) continue;
            std::string fname = func.getName().str();
            if (fname.starts_with("__repl_") || fname.starts_with("lambda_")) continue;
            if (defined_lambdas_.count(fname) == 0) continue;
            auto sym = jit_->lookup(func.getName());
            if (sym) {
                to_remove.insert(jit_->mangleAndIntern(func.getName()));
            } else {
                consumeError(sym.takeError());
            }
        }
        for (auto& gv : module->globals()) {
            if (!gv.hasInitializer() || gv.hasLocalLinkage()) continue;
            if (gv.getName().starts_with("llvm.")) continue;
            if (gv.getLinkage() == GlobalValue::LinkOnceODRLinkage) continue;
            std::string gvname = gv.getName().str();
            bool is_user_global = false;
            for (const auto& [var_name, lambda_info] : defined_lambdas_) {
                if (gvname == var_name + "_sexpr") {
                    is_user_global = true;
                    break;
                }
            }
            if (!is_user_global) continue;
            auto sym = jit_->lookup(gv.getName());
            if (sym) {
                to_remove.insert(jit_->mangleAndIntern(gv.getName()));
            } else {
                consumeError(sym.takeError());
            }
        }
        if (!to_remove.empty()) {
            if (auto err = jit_->getMainJITDylib().remove(to_remove)) {
                consumeError(std::move(err));
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

    // DEBUG: Dump module IR or DataLayout info
    if (getenv("ESHKOL_DUMP_REPL_IR")) {
        module->print(errs(), nullptr);
    }
    if (getenv("ESHKOL_DEBUG_DL")) {
        std::cerr << "[REPL] Module DataLayout: " << module->getDataLayoutStr() << std::endl;
#if LLVM_VERSION_MAJOR >= 21
        std::cerr << "[REPL] Module Triple: " << module->getTargetTriple().str() << std::endl;
#else
        std::cerr << "[REPL] Module Triple: " << module->getTargetTriple() << std::endl;
#endif
        std::cerr << "[REPL] LLJIT DataLayout: " << jit_->getDataLayout().getStringRepresentation() << std::endl;
    }

    // Wrap module with its OWN context (each module gets its own ThreadSafeContext).
    // This is LLVM's recommended ORC JIT pattern — module and context must match.
    auto ts_ctx = orc::ThreadSafeContext(std::move(module_context));
    auto tsm = ThreadSafeModule(std::move(module), ts_ctx);

    auto err = jit_->addIRModule(std::move(tsm));
    if (err) {
        std::string err_msg;
        raw_string_ostream err_stream(err_msg);
        err_stream << err;
        std::cerr << "Failed to add module to JIT: " << err_msg << std::endl;
        throw std::runtime_error("Failed to add module to JIT");
    }

    // STEP 3: Update (or create) forward reference pointer slots so they point
    // at the real function addresses we just materialized in the JIT.
    //
    // For first-time definitions, the slot doesn't exist yet — STEP 2 only
    // creates slots for *external* references. So we may need to allocate the
    // slot here and register it as an absolute symbol (so future modules that
    // reference __repl_fwd_<X> can resolve against this address).
    for (const auto& [ptr_name, func_name] : forward_ref_updates) {
        auto func_symbol = jit_->lookup(func_name);
        if (!func_symbol) {
            consumeError(func_symbol.takeError());
            std::cerr << "Warning: Could not resolve forward reference to " << func_name << std::endl;
            continue;
        }

        void* func_addr = func_symbol->toPtr<void*>();

        auto it = forward_ref_slots_.find(ptr_name);
        if (it != forward_ref_slots_.end()) {
            // Slot already exists (created earlier by STEP 2 for this or a prior
            // module). Update in place — any prior caller that loaded the slot
            // address gets the new function pointer on its next call.
            *it->second = func_addr;
        } else {
            // First definition of this REPL user function. Allocate the slot,
            // initialise it with the new function, and define the slot's address
            // as an absolute JIT symbol. This satisfies both the current module's
            // freshly-stripped external reference and any future module's
            // external reference to __repl_fwd_<X>.
            void** ptr_slot = new void*;
            *ptr_slot = func_addr;
            forward_ref_slots_[ptr_name] = ptr_slot;

            orc::SymbolMap stub_symbol;
            stub_symbol[jit_->mangleAndIntern(ptr_name)] = {
                orc::ExecutorAddr::fromPtr(ptr_slot),
                JITSymbolFlags::Exported
            };
            auto& main_dylib = jit_->getMainJITDylib();
            if (auto err = main_dylib.define(orc::absoluteSymbols(stub_symbol))) {
                consumeError(std::move(err));
            }
        }
        pending_forward_refs_.erase(ptr_name);
    }
}

uint64_t ReplJITContext::lookupSymbol(const std::string& name) {
    if (!jit_) {
        initializeJIT();
    }

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
    if (!jit_) {
        initializeJIT();
    }

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
    if (!jit_) {
        initializeJIT();
    }

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
    auto cwd = platform::current_directory();
    auto exe_dir = platform::executable_directory();

#ifdef _WIN32
    std::vector<std::filesystem::path> candidates = {
        exe_dir / "stdlib.o",
        exe_dir / "../lib/stdlib.o",
        exe_dir / "../lib/eshkol/stdlib.o",
        cwd / "stdlib.o",
        cwd / "build/stdlib.o",
        cwd.parent_path() / "build/stdlib.o",
    };
#else
    std::vector<std::filesystem::path> candidates = {
        cwd / "stdlib.o",
        cwd / "build/stdlib.o",
        cwd.parent_path() / "build/stdlib.o",
        exe_dir / "stdlib.o",
        exe_dir / "../lib/stdlib.o",
        exe_dir / "../lib/eshkol/stdlib.o",
    };
#endif

#ifndef _WIN32
    candidates.emplace_back("/usr/local/lib/eshkol/stdlib.o");
    candidates.emplace_back("/usr/lib/eshkol/stdlib.o");
#endif

    return platform::find_first_existing(candidates);
}

// Find the pre-compiled stdlib.bc bitcode file
static std::string findStdlibBitcode() {
    auto cwd = platform::current_directory();
    auto exe_dir = platform::executable_directory();

#ifdef _WIN32
    std::vector<std::filesystem::path> candidates = {
        exe_dir / "stdlib.bc",
        exe_dir / "../lib/stdlib.bc",
        exe_dir / "../lib/eshkol/stdlib.bc",
        cwd / "stdlib.bc",
        cwd / "build/stdlib.bc",
        cwd.parent_path() / "build/stdlib.bc",
    };
#else
    std::vector<std::filesystem::path> candidates = {
        cwd / "stdlib.bc",
        cwd / "build/stdlib.bc",
        cwd.parent_path() / "build/stdlib.bc",
        exe_dir / "stdlib.bc",
        exe_dir / "../lib/stdlib.bc",
        exe_dir / "../lib/eshkol/stdlib.bc",
    };
#endif

#ifndef _WIN32
    candidates.emplace_back("/usr/local/lib/eshkol/stdlib.bc");
    candidates.emplace_back("/usr/lib/eshkol/stdlib.bc");
#endif

    return platform::find_first_existing(candidates);
}

// Discover and register stdlib symbols dynamically from .bc metadata.
// No hardcoded function lists — iterates the bitcode module's IR to find
// all exported functions (names + arities) and _sexpr globals.
void ReplJITContext::registerStdlibSymbols() {
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

    // Dynamic discovery: parse .bc to find all exported functions and globals.
    // Bitcode preserves original IR types (struct params, not scalarized),
    // so F.arg_size() gives the real arity.
    std::string bc_path = findStdlibBitcode();
    if (!bc_path.empty()) {
        auto buf_or_err = MemoryBuffer::getFile(bc_path);
        if (buf_or_err) {
            auto ctx = std::make_unique<LLVMContext>();
            auto mod_or_err = parseBitcodeFile(buf_or_err->get()->getMemBufferRef(), *ctx);
            if (mod_or_err) {
                Module& mod = **mod_or_err;
                size_t func_count = 0, global_count = 0;

                for (auto& F : mod) {
                    if (F.isDeclaration()) continue;
                    if (F.hasInternalLinkage()) continue;
                    std::string name = F.getName().str();
                    // Skip internal compiler-generated names
                    if (name.rfind("__eshkol_", 0) == 0) continue;
                    if (name.rfind("lambda_", 0) == 0) continue;
                    if (name == "main") continue;

                    size_t arity = F.arg_size();
                    eshkol_repl_register_function(name.c_str(), 0, arity);
                    defined_lambdas_[name] = {name, arity};
                    registered_lambdas_.insert(name);
                    func_count++;
                }

                for (auto& G : mod.globals()) {
                    std::string name = G.getName().str();
                    // Register _sexpr globals so codegen doesn't redefine them
                    if (name.size() > 6 && name.compare(name.size() - 6, 6, "_sexpr") == 0) {
                        eshkol_repl_register_symbol(name.c_str(), 0);
                        defined_globals_.insert(name);
                        global_count++;
                    }
                }

                std::cerr << "[REPL] Discovered " << func_count << " functions, "
                          << global_count << " globals from stdlib.bc" << std::endl;
                return;
            } else {
                consumeError(mod_or_err.takeError());
            }
        }
    }

    // Fallback: if .bc not available, we can't discover symbols dynamically.
    // This means tryResolveReplFunction won't find stdlib symbols.
    std::cerr << "Warning: stdlib.bc not found — stdlib symbol discovery unavailable" << std::endl;
}

bool ReplJITContext::loadStdlib() {
    if (!jit_) {
        initializeJIT();
    }

    // Load pre-compiled stdlib.o via addObjectFile (fast, instant availability).
    std::string stdlib_obj_path = findStdlibObject();
    if (!stdlib_obj_path.empty()) {
        auto buffer_or_err = MemoryBuffer::getFile(stdlib_obj_path);
        if (buffer_or_err) {
            auto& main_dylib = jit_->getMainJITDylib();
            auto err = jit_->addObjectFile(main_dylib, std::move(*buffer_or_err));
            if (!err) {
                registerStdlibSymbols();
                std::cerr << "[REPL] Loaded stdlib from: " << stdlib_obj_path << std::endl;
                return true;
            } else {
                std::string err_msg;
                raw_string_ostream err_stream(err_msg);
                err_stream << err;
                std::cerr << "Warning: Failed to load stdlib.o (" << err_msg
                          << "), falling back to JIT compilation" << std::endl;
            }
        }
    }

    // Fallback: JIT compile from source (slowest but always correct)
    const bool was_loading_stdlib_from_source = loading_stdlib_from_source_;
    loading_stdlib_from_source_ = true;
    const bool loaded = loadModule("stdlib", false);
    loading_stdlib_from_source_ = was_loading_stdlib_from_source;
    return loaded;
}

bool ReplJITContext::loadModule(const std::string& module_name) {
    return loadModule(module_name, true);
}

bool ReplJITContext::loadModule(const std::string& module_name, bool allow_precompiled_stdlib) {
    // Check if already loaded by NAME first (for stdlib.o preloaded modules)
    if (loaded_modules.count(module_name)) {
        return true;  // Already loaded via stdlib.o
    }

    // For stdlib or core.* modules, use precompiled stdlib.o if available
    if (allow_precompiled_stdlib && !loading_stdlib_from_source_ &&
        (module_name == "stdlib" || module_name.find("core.") == 0)) {
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

    // MODULE VISIBILITY: First pass - collect exported symbols from provide statements
    std::unordered_set<std::string> exported_symbols;
    std::unordered_set<std::string> defined_symbols;
    bool has_provide = false;

    for (auto& ast_item : module_asts) {
        if (ast_item.type == ESHKOL_OP) {
            // Collect exported symbols from provide
            if (ast_item.operation.op == ESHKOL_PROVIDE_OP) {
                has_provide = true;
                for (size_t i = 0; i < ast_item.operation.provide_op.num_exports; i++) {
                    if (ast_item.operation.provide_op.export_names[i]) {
                        exported_symbols.insert(ast_item.operation.provide_op.export_names[i]);
                    }
                }
            }
            // Collect defined symbols (functions and variables)
            else if (ast_item.operation.op == ESHKOL_DEFINE_OP) {
                if (ast_item.operation.define_op.name) {
                    defined_symbols.insert(ast_item.operation.define_op.name);
                }
            }
        }
    }

    // Store module exports for visibility checking
    module_exports_[module_name] = exported_symbols;

    // Mark private symbols (defined but not exported) - only if module has provide.
    // Delay registering them with codegen until after the module finishes compiling
    // so internal forward references still work while loading the module itself.
    std::vector<std::string> private_symbols_to_register;
    if (has_provide) {
        for (const auto& sym : defined_symbols) {
            if (exported_symbols.find(sym) == exported_symbols.end()) {
                private_symbols_.insert(sym);
                private_symbols_to_register.push_back(sym);
            }
        }
    }

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
                // Already processed above, skip
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

    for (const auto& sym : private_symbols_to_register) {
        // Register after module compilation so only external accesses are blocked.
        eshkol_repl_register_private_symbol(sym.c_str());
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
    auto cwd = platform::current_directory();
    auto exe_dir = platform::executable_directory();

    std::vector<std::filesystem::path> candidates = {
        cwd / "lib",
        cwd.parent_path() / "lib",
        cwd / "share/eshkol/lib",
        exe_dir / "lib",
        exe_dir / "../lib",
        exe_dir / "../share/eshkol/lib",
    };

#ifndef _WIN32
    candidates.emplace_back("/usr/local/share/eshkol/lib");
    candidates.emplace_back("/usr/share/eshkol/lib");
#endif

    return platform::find_first_existing(candidates);
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
        while (std::getline(ss, search_dir, eshkol_path_separator)) {
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
                // HOT RELOAD: Clear old lambda registration so the new lambda can be tracked.
                // See execute() for detailed explanation.
                auto old_lambda_it = defined_lambdas_.find(name);
                if (old_lambda_it != defined_lambdas_.end()) {
                    const auto& old_lambda_name = old_lambda_it->second.first;
                    if (!old_lambda_name.empty()) {
                        registered_lambdas_.erase(old_lambda_name);
                    }
                }
                symbol_table_.erase(std::string(name));
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

    Module* cpp_module = module_from_ref(c_module);

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

    // Capture named top-level functions so later REPL evaluations can import them
    // even when the pre-registration path misses a module-loaded definition.
    std::vector<std::pair<std::string, size_t>> exported_function_infos;
    for (auto& func : cpp_module->functions()) {
        if (func.isDeclaration() || func.hasLocalLinkage() || func.getName().starts_with("llvm.")) {
            continue;
        }
        std::string fname = func.getName().str();
        if (fname.find("__") == 0 || fname.find("lambda_") == 0) {
            continue;
        }
        exported_function_infos.push_back({fname, func.arg_size()});
    }

    // Release module + extract its context for proper ThreadSafeModule pairing
    auto module_context = eshkol_extract_module_context_for_jit(c_module);
    addModule(std::unique_ptr<Module>(cpp_module), std::move(module_context));

    for (const auto& [func_name_export, arity] : exported_function_infos) {
        uint64_t func_addr_export = lookupSymbol(func_name_export);
        if (func_addr_export == 0) {
            continue;
        }

        defined_lambdas_[func_name_export] = {func_name_export, arity};
        eshkol_repl_register_function(func_name_export.c_str(), func_addr_export, arity);
        registered_lambdas_.insert(func_name_export);

        // REPL HOT RELOAD: also register under the unversioned user name so
        // function-as-value paths (e.g. (map sq lst)) can resolve "sq" to the
        // current __rv<N> definition. The lambda_names mapping tells codegen
        // which JIT symbol the user name actually points at. On redefinition
        // these are overwritten so the latest version wins.
        std::string user_name = strip_repl_version_suffix(func_name_export);
        if (!user_name.empty()) {
            eshkol_repl_register_function(user_name.c_str(), func_addr_export, arity);
            eshkol_repl_register_lambda_name(user_name.c_str(), func_name_export.c_str());
        }
    }

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
            typedef int32_t (*EvalFunc)(int32_t, char**);
            EvalFunc eval_func = reinterpret_cast<EvalFunc>(func_addr);
            int32_t result_value = eval_func(0, nullptr);
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

    // Pre-register function/lambda variables so they're tracked for REPL cross-evaluation
    // This mirrors what executeBatch does for batch compilations
    if (ast->type == ESHKOL_OP && ast->operation.op == ESHKOL_DEFINE_OP) {
        const char* name = ast->operation.define_op.name;
        bool is_lambda = ast->operation.define_op.is_function ||
            (ast->operation.define_op.value &&
             ast->operation.define_op.value->type == ESHKOL_OP &&
             ast->operation.define_op.value->operation.op == ESHKOL_LAMBDA_OP);
        if (name && is_lambda) {
            // HOT RELOAD: Clear old lambda registration so the new lambda can be tracked.
            // Each codegen invocation creates a new lambda_N, so the new definition gets
            // a fresh name. We clear the old entry from registered_lambdas_ to allow
            // the new lambda to be registered, and clear symbol_table_ cache to force
            // JIT re-lookup with the updated address maps.
            auto old_lambda_it = defined_lambdas_.find(name);
            if (old_lambda_it != defined_lambdas_.end()) {
                const auto& old_lambda_name = old_lambda_it->second.first;
                if (!old_lambda_name.empty()) {
                    registered_lambdas_.erase(old_lambda_name);
                }
            }
            symbol_table_.erase(std::string(name));
            registerLambdaVar(name);
        }
    }

    // Reserve a unique module/eval id up front so failed evaluations do not
    // reuse the same COFF init symbol names on the next attempt.
    const std::uint64_t eval_id = eval_counter_++;

    // Generate LLVM IR using the existing Eshkol compiler
    std::string module_name = "__repl_module_" + std::to_string(eval_id);

    // Call the existing compiler to generate LLVM IR from AST
    LLVMModuleRef c_module = eshkol_generate_llvm_ir(ast, 1, module_name.c_str());

    if (!c_module) {
        throw std::runtime_error("Failed to generate LLVM IR from AST");
    }

    Module* cpp_module = module_from_ref(c_module);

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
    std::string unique_func_name = "__repl_eval_" + std::to_string(eval_id);
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

    std::vector<std::pair<std::string, size_t>> exported_function_infos;
    for (auto& func : cpp_module->functions()) {
        if (func.isDeclaration() || func.hasLocalLinkage() || func.getName().starts_with("llvm.")) {
            continue;
        }
        std::string fname = func.getName().str();
        if (fname.find("__") == 0 || fname.find("lambda_") == 0) {
            continue;
        }
        exported_function_infos.push_back({fname, func.arg_size()});
    }

    // Release module + extract its context for proper ThreadSafeModule pairing
    auto module_context = eshkol_extract_module_context_for_jit(c_module);
    addModule(std::unique_ptr<Module>(cpp_module), std::move(module_context));

    for (const auto& [func_name_export, arity] : exported_function_infos) {
        uint64_t func_addr_export = lookupSymbol(func_name_export);
        if (func_addr_export == 0) {
            continue;
        }

        defined_lambdas_[func_name_export] = {func_name_export, arity};
        eshkol_repl_register_function(func_name_export.c_str(), func_addr_export, arity);
        registered_lambdas_.insert(func_name_export);

        // REPL HOT RELOAD: also register under the unversioned user name (see
        // executeBatch comment) so function-as-value paths and cross-module
        // first-class function references resolve to the latest definition.
        std::string user_name = strip_repl_version_suffix(func_name_export);
        if (!user_name.empty()) {
            eshkol_repl_register_function(user_name.c_str(), func_addr_export, arity);
            eshkol_repl_register_lambda_name(user_name.c_str(), func_name_export.c_str());
        }
    }

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

    // Cast to function pointer and call it
    // The compiler generates main as i32(i32, char**), so match the ABI
    typedef int32_t (*EvalFunc)(int32_t, char**);
    EvalFunc eval_func = reinterpret_cast<EvalFunc>(func_addr);

    int32_t result_value = eval_func(0, nullptr);

    // CRITICAL: NOW capture s-expression values AFTER execution
    // The entry function has initialized these globals, so now they contain valid values
    for (const auto& [var_name, var_addr] : sexpr_globals_to_capture) {
        // Read the current value from the global's memory
        uint64_t* global_ptr = reinterpret_cast<uint64_t*>(var_addr);
        uint64_t sexpr_value = *global_ptr;
        // Register the s-expression value with the compiler
        eshkol_repl_register_sexpr(var_name.c_str(), sexpr_value);
    }

    // Return result as heap-allocated int64_t (promoted from i32 main return).
    // For typed return value handling, use executeTagged() instead.
    int64_t* result_ptr = new int64_t(result_value);

    return result_ptr;
}

eshkol_tagged_value_t ReplJITContext::executeTagged(eshkol_ast_t* ast) {
    eshkol_tagged_value_t result;
    result.type = ESHKOL_VALUE_NULL;
    result.flags = 0;
    result.reserved = 0;
    result.data.raw_val = 0;

    if (!ast) {
        return result;
    }

    // Execute the AST and get raw result
    void* raw_result = execute(ast);

    if (!raw_result) {
        // Execution returned null - return null tagged value
        return result;
    }

    // Get the raw result value (currently JIT returns int32 promoted to int64)
    int64_t raw_val = *static_cast<int64_t*>(raw_result);
    delete static_cast<int64_t*>(raw_result);

    // Determine the result type from the AST's inferred HoTT type
    // The inferred_hott_type is packed: bits 0-15 = TypeId.id, bits 16-23 = universe, bits 24-31 = flags
    uint32_t packed_type = ast->inferred_hott_type;

    // If type wasn't inferred (value 0), fall back to analyzing AST structure
    if (packed_type == 0) {
        // Analyze the AST node type to determine result type
        switch (ast->type) {
            case ESHKOL_INT8:
            case ESHKOL_INT16:
            case ESHKOL_INT32:
            case ESHKOL_INT64:
            case ESHKOL_UINT8:
            case ESHKOL_UINT16:
            case ESHKOL_UINT32:
            case ESHKOL_UINT64:
                result.type = ESHKOL_VALUE_INT64;
                result.flags = ESHKOL_VALUE_EXACT_FLAG;
                result.data.int_val = raw_val;
                return result;

            case ESHKOL_DOUBLE:
                result.type = ESHKOL_VALUE_DOUBLE;
                result.flags = ESHKOL_VALUE_INEXACT_FLAG;
                result.data.double_val = *reinterpret_cast<double*>(&raw_val);
                return result;

            case ESHKOL_BOOL:
                result.type = ESHKOL_VALUE_BOOL;
                result.data.int_val = (raw_val != 0) ? 1 : 0;
                return result;

            case ESHKOL_CHAR:
                result.type = ESHKOL_VALUE_CHAR;
                result.data.int_val = raw_val;
                return result;

            case ESHKOL_STRING:
            case ESHKOL_BIGNUM_LITERAL:
                result.type = ESHKOL_VALUE_HEAP_PTR;
                result.data.ptr_val = static_cast<uint64_t>(raw_val);
                return result;

            case ESHKOL_CONS:
                result.type = ESHKOL_VALUE_HEAP_PTR;
                result.data.ptr_val = static_cast<uint64_t>(raw_val);
                return result;

            case ESHKOL_FUNC:
                result.type = ESHKOL_VALUE_CALLABLE;
                result.data.ptr_val = static_cast<uint64_t>(raw_val);
                return result;

            case ESHKOL_TENSOR:
                result.type = ESHKOL_VALUE_HEAP_PTR;
                result.data.ptr_val = static_cast<uint64_t>(raw_val);
                return result;

            case ESHKOL_NULL:
                result.type = ESHKOL_VALUE_NULL;
                result.data.raw_val = 0;
                return result;

            case ESHKOL_VAR:
            case ESHKOL_OP:
            default:
                // For VAR, OP, and other complex AST nodes without type info,
                // we cannot determine the type - return as null with warning
                // The caller should ensure type checking runs before eval
                result.type = ESHKOL_VALUE_NULL;
                result.data.raw_val = 0;
                return result;
        }
    }

    // Unpack the HoTT TypeId from the packed format
    // TypeId.id is in bits 0-15
    uint16_t type_id = static_cast<uint16_t>(packed_type & 0xFFFF);
    uint8_t type_flags = static_cast<uint8_t>((packed_type >> 24) & 0xFF);

    // Map HoTT TypeId to runtime value type using the BuiltinTypes constants
    using namespace eshkol::hott;

    // Numeric types
    if (type_id == BuiltinTypes::Int64.id ||
        type_id == BuiltinTypes::Integer.id ||
        type_id == BuiltinTypes::Natural.id ||
        type_id == BuiltinTypes::Number.id) {
        result.type = ESHKOL_VALUE_INT64;
        result.flags = (type_flags & TYPE_FLAG_EXACT) ? ESHKOL_VALUE_EXACT_FLAG : 0;
        result.data.int_val = raw_val;
        return result;
    }

    if (type_id == BuiltinTypes::Float64.id ||
        type_id == BuiltinTypes::Float32.id ||
        type_id == BuiltinTypes::Real.id) {
        result.type = ESHKOL_VALUE_DOUBLE;
        result.flags = ESHKOL_VALUE_INEXACT_FLAG;
        result.data.double_val = *reinterpret_cast<double*>(&raw_val);
        return result;
    }

    // Complex number types
    if (type_id == BuiltinTypes::Complex.id ||
        type_id == BuiltinTypes::Complex64.id ||
        type_id == BuiltinTypes::Complex128.id) {
        // Complex numbers are stored as heap pointers to (real, imag) pairs
        result.type = ESHKOL_VALUE_COMPLEX;
        result.data.ptr_val = static_cast<uint64_t>(raw_val);
        return result;
    }

    // Boolean type
    if (type_id == BuiltinTypes::Boolean.id) {
        result.type = ESHKOL_VALUE_BOOL;
        result.data.int_val = (raw_val != 0) ? 1 : 0;
        return result;
    }

    // Character type
    if (type_id == BuiltinTypes::Char.id) {
        result.type = ESHKOL_VALUE_CHAR;
        result.data.int_val = raw_val;
        return result;
    }

    // Text/String types
    if (type_id == BuiltinTypes::String.id ||
        type_id == BuiltinTypes::Text.id) {
        result.type = ESHKOL_VALUE_HEAP_PTR;
        result.data.ptr_val = static_cast<uint64_t>(raw_val);
        return result;
    }

    // Symbol type
    if (type_id == BuiltinTypes::Symbol.id) {
        result.type = ESHKOL_VALUE_SYMBOL;
        result.data.int_val = raw_val;
        return result;
    }

    // Null type
    if (type_id == BuiltinTypes::Null.id) {
        result.type = ESHKOL_VALUE_NULL;
        result.data.raw_val = 0;
        return result;
    }

    // Collection types (List, Vector, Pair) - heap allocated
    if (type_id == BuiltinTypes::List.id ||
        type_id == BuiltinTypes::Vector.id ||
        type_id == BuiltinTypes::Pair.id) {
        result.type = ESHKOL_VALUE_HEAP_PTR;
        result.data.ptr_val = static_cast<uint64_t>(raw_val);
        return result;
    }

    // Tensor type
    if (type_id == BuiltinTypes::Tensor.id) {
        result.type = ESHKOL_VALUE_HEAP_PTR;
        result.data.ptr_val = static_cast<uint64_t>(raw_val);
        return result;
    }

    // HashTable type
    if (type_id == BuiltinTypes::HashTable.id) {
        result.type = ESHKOL_VALUE_HEAP_PTR;
        result.data.ptr_val = static_cast<uint64_t>(raw_val);
        return result;
    }

    // Function/Closure types - callables
    if (type_id == BuiltinTypes::Function.id ||
        type_id == BuiltinTypes::Closure.id) {
        result.type = ESHKOL_VALUE_CALLABLE;
        result.data.ptr_val = static_cast<uint64_t>(raw_val);
        return result;
    }

    // Autodiff types
    if (type_id == BuiltinTypes::DualNumber.id) {
        result.type = ESHKOL_VALUE_DUAL_NUMBER;
        result.data.ptr_val = static_cast<uint64_t>(raw_val);
        return result;
    }

    if (type_id == BuiltinTypes::ADNode.id) {
        result.type = ESHKOL_VALUE_CALLABLE;
        result.data.ptr_val = static_cast<uint64_t>(raw_val);
        return result;
    }

    // Resource types (Handle, Buffer, Stream)
    if (type_id == BuiltinTypes::Handle.id) {
        result.type = ESHKOL_VALUE_HANDLE;
        result.data.ptr_val = static_cast<uint64_t>(raw_val);
        return result;
    }

    if (type_id == BuiltinTypes::Buffer.id) {
        result.type = ESHKOL_VALUE_BUFFER;
        result.data.ptr_val = static_cast<uint64_t>(raw_val);
        return result;
    }

    if (type_id == BuiltinTypes::Stream.id) {
        result.type = ESHKOL_VALUE_STREAM;
        result.data.ptr_val = static_cast<uint64_t>(raw_val);
        return result;
    }

    // Root Value type - treat as generic heap pointer
    if (type_id == BuiltinTypes::Value.id) {
        result.type = ESHKOL_VALUE_HEAP_PTR;
        result.data.ptr_val = static_cast<uint64_t>(raw_val);
        return result;
    }

    // Function types (TypeIds >= 500 are dynamically allocated function types)
    // These are created by makeFunctionType() for specific function signatures
    if (type_id >= 500) {
        result.type = ESHKOL_VALUE_CALLABLE;
        result.data.ptr_val = static_cast<uint64_t>(raw_val);
        return result;
    }

    // Universe types and proof types (runtime-erased)
    if (type_id == BuiltinTypes::TypeU0.id ||
        type_id == BuiltinTypes::TypeU1.id ||
        type_id == BuiltinTypes::TypeU2.id ||
        type_id == BuiltinTypes::Eq.id ||
        type_id == BuiltinTypes::LessThan.id ||
        type_id == BuiltinTypes::Bounded.id ||
        type_id == BuiltinTypes::Subtype.id) {
        // These are type-level values, return as null at runtime
        result.type = ESHKOL_VALUE_NULL;
        result.data.raw_val = 0;
        return result;
    }

    // Invalid or unknown type - return null
    if (type_id == BuiltinTypes::Invalid.id) {
        result.type = ESHKOL_VALUE_NULL;
        result.data.raw_val = 0;
        return result;
    }

    // Fallback for user-defined types (TypeIds 1000+) or unrecognized types
    // Treat as heap pointer since user types are typically allocated
    result.type = ESHKOL_VALUE_HEAP_PTR;
    result.data.ptr_val = static_cast<uint64_t>(raw_val);
    return result;
}

} // namespace eshkol
