//
// Copyright (C) tsotchke
//
// SPDX-License-Identifier: MIT
//

#ifndef ESHKOL_REPL_JIT_H
#define ESHKOL_REPL_JIT_H

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

// Forward declarations for LLVM types
namespace llvm {
    class LLVMContext;
    class Module;
    class Value;
    class Function;
    namespace orc {
        class LLJIT;
        class ThreadSafeContext;
    }
}

// C interface
#include <eshkol/eshkol.h>

namespace eshkol {

// Forward declarations
class CodeGenerator;

/**
 * ReplJITContext manages JIT compilation and execution for the REPL.
 *
 * This class provides a thin wrapper around LLVM's ORC JIT (LLJIT) that:
 * - Compiles AST to LLVM IR using the existing CodeGenerator
 * - JIT compiles and executes code incrementally
 * - Maintains persistent state across evaluations (variables, functions)
 * - Extracts results from JIT-executed code
 *
 * Design: This class uses the existing Eshkol compiler (parser + codegen)
 * as a backend, ensuring automatic feature parity between compiled code
 * and REPL evaluation.
 */
class ReplJITContext {
public:
    ReplJITContext();
    ~ReplJITContext();

    // Disable copy and move (JIT context is unique)
    ReplJITContext(const ReplJITContext&) = delete;
    ReplJITContext& operator=(const ReplJITContext&) = delete;
    ReplJITContext(ReplJITContext&&) = delete;
    ReplJITContext& operator=(ReplJITContext&&) = delete;

    /**
     * Compile and execute an AST node, returning the result.
     *
     * This wraps the AST in an anonymous function:
     *   define i64 @__repl_eval_N() { ... user code ... ret i64 %result }
     *
     * The function is JIT compiled, executed, and the result extracted.
     *
     * @param ast The AST to execute
     * @return Pointer to the result value (type depends on AST)
     */
    void* execute(eshkol_ast_t* ast);

    /**
     * Add a module to the JIT for execution.
     * Used for incremental compilation of REPL inputs.
     *
     * @param module The LLVM module to add (takes ownership)
     */
    void addModule(std::unique_ptr<llvm::Module> module);

    /**
     * Look up a symbol (function or global variable) by name.
     * Used to access previously defined variables/functions.
     *
     * @param name Symbol name to look up
     * @return Address of the symbol, or 0 if not found
     */
    uint64_t lookupSymbol(const std::string& name);

    /**
     * Register a symbol in the persistent environment.
     * Called after executing a (define ...) form.
     *
     * @param name Symbol name
     * @param address JIT address of the symbol
     */
    void registerSymbol(const std::string& name, uint64_t address);

    /**
     * Pre-register a lambda variable before compilation.
     * This allows the next module to know about the lambda for symbol resolution.
     *
     * @param var_name Variable name (e.g., "test-func")
     */
    void registerLambdaVar(const std::string& var_name);

    /**
     * Load the standard library (stdlib.esk) into the REPL environment.
     * This makes all stdlib functions available for use.
     *
     * @return true on success, false on error
     */
    bool loadStdlib();

    /**
     * Load a module by name (e.g., "core.functional.compose" or "stdlib")
     *
     * @param module_name Module name with dot-separated path
     * @return true on success, false on error
     */
    bool loadModule(const std::string& module_name);

    /**
     * Execute multiple ASTs as a batch (for module loading).
     * This allows forward references between functions in the same module.
     *
     * @param asts Vector of ASTs to compile and execute together
     * @param silent If true, suppress output (for module loading)
     * @return Pointer to result of last expression, or nullptr
     */
    void* executeBatch(std::vector<eshkol_ast_t>& asts, bool silent = true);

    /**
     * Check if a symbol (variable or function) is already defined in the JIT.
     * Used to detect reload scenarios where we should skip redefinition.
     *
     * @param name Symbol name to check
     * @return true if symbol exists, false otherwise
     */
    bool isSymbolDefined(const std::string& name);

    /**
     * Get the current evaluation counter.
     * Used to generate unique function names (__repl_eval_0, __repl_eval_1, etc.)
     */
    int getEvalCounter() const { return eval_counter_; }

    /**
     * Increment evaluation counter.
     * Called after each successful evaluation.
     */
    void incrementEvalCounter() { ++eval_counter_; }

    /**
     * Get the LLVM context for this REPL session.
     * Used by CodeGenerator to emit IR.
     */
    llvm::LLVMContext& getContext();

    /**
     * Create a new module for the current evaluation.
     * Each REPL input gets its own module, which is added to the JIT.
     */
    std::unique_ptr<llvm::Module> createModule(const std::string& name);

private:
    // LLVM ORC JIT instance (using LLJIT for simplicity)
    std::unique_ptr<llvm::orc::LLJIT> jit_;

    // Thread-safe context shared across all modules
    std::shared_ptr<llvm::orc::ThreadSafeContext> ts_context_;

    // Raw context pointer for convenience (points into ts_context_)
    llvm::LLVMContext* raw_context_;

    // Evaluation counter for generating unique function names
    int eval_counter_;

    // Symbol table mapping names to JIT addresses
    // Used to track user-defined variables and functions
    std::unordered_map<std::string, uint64_t> symbol_table_;

    // Track lambda functions defined across evaluations
    // Maps variable name -> (lambda function name, arity)
    // E.g., "test-func" -> ("lambda_0", 1)
    std::unordered_map<std::string, std::pair<std::string, size_t>> defined_lambdas_;

    // Track which lambdas have been registered in the JIT dylib (to avoid duplicate registration)
    std::unordered_set<std::string> registered_lambdas_;

    // Track global variables defined across evaluations
    // Maps variable name -> type (for now, just track names)
    std::unordered_set<std::string> defined_globals_;

    // SHARED ARENA: Single persistent arena shared across all evaluations
    // This ensures data allocated in one evaluation (lists, vectors, etc.)
    // is accessible in subsequent evaluations
    void* shared_arena_;

    // Initialize the JIT (called from constructor)
    void initializeJIT();

    // Register runtime symbols (arena functions, stdlib, etc.)
    void registerRuntimeSymbols();

    // Inject external declarations for previously-defined symbols
    void injectPreviousSymbols(llvm::Module* module);
};

} // namespace eshkol

#endif // ESHKOL_REPL_JIT_H
