/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * BindingCodegen - Variable binding code generation
 *
 * This module handles:
 * - define (top-level and nested variable definitions)
 * - let (local bindings with sequential evaluation)
 * - letrec (recursive bindings for mutual recursion)
 * - set! (mutation of existing bindings)
 *
 * Key design principle: ALL values are stored as tagged_value structs
 * to preserve type information across storage and retrieval.
 */
#ifndef ESHKOL_BACKEND_BINDING_CODEGEN_H
#define ESHKOL_BACKEND_BINDING_CODEGEN_H

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/backend/codegen_context.h>
#include <eshkol/backend/tagged_value_codegen.h>
#include <eshkol/eshkol.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/GlobalVariable.h>
#include <string>
#include <set>
#include <unordered_map>

namespace eshkol {

// Forward declaration for TypedValue
struct TypedValue;

/**
 * BindingCodegen handles variable binding operations.
 *
 * All bindings store values as tagged_value to preserve type information.
 * This ensures that ports, lists, lambdas, etc. can be correctly retrieved.
 */
class BindingCodegen {
public:
    /**
     * Construct BindingCodegen with context and tagged value helper.
     */
    BindingCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged);

    // === Variable Definition ===

    /**
     * Define a variable: (define name value)
     *
     * For top-level: creates GlobalVariable with tagged_value
     * For nested (inside function): creates AllocaInst with tagged_value
     *
     * @param op The define operation AST node
     * @return The stored value (or nullptr on failure)
     */
    llvm::Value* define(const eshkol_operations_t* op);

    // === Let Bindings ===

    /**
     * Let expression: (let ((x 1) (y 2)) body)
     *
     * Sequential evaluation - each binding evaluated in order,
     * but bindings are not visible to each other.
     *
     * @param op The let operation AST node
     * @return Result of body evaluation
     */
    llvm::Value* let(const eshkol_operations_t* op);

    /**
     * Letrec expression: (letrec ((f ...) (g ...)) body)
     *
     * Recursive bindings - all bindings visible to all values.
     * Used for mutually recursive function definitions.
     *
     * @param op The letrec operation AST node
     * @return Result of body evaluation
     */
    llvm::Value* letrec(const eshkol_operations_t* op);

    /**
     * Let* expression: (let* ((x 1) (y x)) body)
     *
     * Sequential evaluation - each binding visible to subsequent ones.
     *
     * @param op The let* operation AST node
     * @return Result of body evaluation
     */
    llvm::Value* letStar(const eshkol_operations_t* op);

    /**
     * Letrec* expression: (letrec* ((f ...) (g ...)) body)
     *
     * R7RS sequential recursive bindings. Like letrec, all bindings are
     * mutually visible, but they are evaluated left-to-right and each
     * binding's value is stored immediately so subsequent bindings can use it.
     *
     * @param op The letrec* operation AST node
     * @return Result of body evaluation
     */
    llvm::Value* letrecStar(const eshkol_operations_t* op);

    // === Mutation ===

    /**
     * Set! expression: (set! name value)
     *
     * Mutates an existing binding.
     *
     * @param op The set! operation AST node
     * @return The new value
     */
    llvm::Value* set(const eshkol_operations_t* op);

    // === Variable Lookup ===

    /**
     * Look up a variable by name.
     *
     * Searches local symbol table first, then global.
     *
     * @param name Variable name
     * @return The variable's storage location (alloca or global), or nullptr
     */
    llvm::Value* lookupVariable(const std::string& name);

    /**
     * Load a variable's value.
     *
     * @param name Variable name
     * @return The variable's tagged value, or nullptr
     */
    llvm::Value* loadVariable(const std::string& name);

private:
    // Shared codegen state and tagged-value helper (not owned; refs injected
    // via the constructor)
    CodegenContext& ctx_;
    TaggedValueCodegen& tagged_;

    // Callbacks for AST code generation
    using CodegenASTFunc = llvm::Value* (*)(const void* ast, void* context);
    using CodegenTypedASTFunc = void* (*)(const void* ast, void* context);
    using TypedToTaggedFunc = llvm::Value* (*)(void* typed_value, void* context);
    // Get TypedValue type (returns eshkol_value_type_t)
    using GetTypedValueTypeFunc = int (*)(void* typed_value, void* context);
    // Register function binding (for apply/call resolution)
    using RegisterFuncBindingFunc = void (*)(const char* var_name, void* typed_value, void* context);

    // Stored callback instances set via setCodegenCallbacks() (see the
    // typedefs above for each callback's signature/purpose)
    CodegenASTFunc codegen_ast_callback_ = nullptr;
    CodegenTypedASTFunc codegen_typed_ast_callback_ = nullptr;
    TypedToTaggedFunc typed_to_tagged_callback_ = nullptr;
    GetTypedValueTypeFunc get_typed_value_type_callback_ = nullptr;
    RegisterFuncBindingFunc register_func_binding_callback_ = nullptr;
    void* callback_context_ = nullptr;

    // Symbol tables (references to main codegen's tables)
    std::unordered_map<std::string, llvm::Value*>* symbol_table_ = nullptr;
    std::unordered_map<std::string, llvm::Value*>* global_symbol_table_ = nullptr;

    // Current function context
    llvm::Function** current_function_ = nullptr;

    // REPL mode flag
    bool* repl_mode_ = nullptr;

    // Lambda tracking for function bindings
    std::string* last_generated_lambda_name_ = nullptr;
    std::unordered_map<std::string, llvm::Function*>* function_table_ = nullptr;

    // LETREC REFACTOR: Set of names to exclude from free variable capture
    // Populated before generating letrec lambda bindings, cleared after
    std::set<std::string>* letrec_excluded_capture_names_ = nullptr;

    /**
     * Store a value in a binding (alloca or global).
     * Always converts to tagged_value first.
     *
     * @param name Variable name
     * @param value The value to store
     * @param value_type The Eshkol type of the value
     * @param is_global Whether to create a global variable
     * @return The storage location
     */
    llvm::Value* storeBinding(
        const std::string& name,
        llvm::Value* value,
        eshkol_value_type_t value_type,
        bool is_global
    );

    /**
     * Ensure value is a tagged_value.
     * Converts raw i64, double, etc. to tagged_value struct.
     *
     * @param value The value to convert
     * @param value_type The Eshkol type
     * @return A tagged_value
     */
    llvm::Value* ensureTaggedValue(llvm::Value* value, eshkol_value_type_t value_type);

    /**
     * Register a lambda function binding.
     * Sets up _func and _sexpr entries in symbol tables.
     *
     * @param var_name Variable name
     * @param lambda_name Lambda function name
     */
    void registerLambdaBinding(const std::string& var_name, const std::string& lambda_name);

public:
    /**
     * Set callbacks for AST code generation.
     */
    void setCodegenCallbacks(
        CodegenASTFunc codegen_ast,
        CodegenTypedASTFunc codegen_typed_ast,
        TypedToTaggedFunc typed_to_tagged,
        GetTypedValueTypeFunc get_typed_value_type,
        RegisterFuncBindingFunc register_func_binding,
        void* context
    ) {
        codegen_ast_callback_ = codegen_ast;
        codegen_typed_ast_callback_ = codegen_typed_ast;
        typed_to_tagged_callback_ = typed_to_tagged;
        get_typed_value_type_callback_ = get_typed_value_type;
        register_func_binding_callback_ = register_func_binding;
        callback_context_ = context;
    }

    /**
     * Set symbol table references.
     */
    void setSymbolTables(
        std::unordered_map<std::string, llvm::Value*>* local,
        std::unordered_map<std::string, llvm::Value*>* global
    ) {
        symbol_table_ = local;
        global_symbol_table_ = global;
    }

    /**
     * Set current function pointer reference.
     */
    void setCurrentFunction(llvm::Function** func) {
        current_function_ = func;
    }

    /**
     * Set REPL mode flag reference.
     */
    void setReplMode(bool* repl_mode) {
        repl_mode_ = repl_mode;
    }

    /**
     * Set lambda tracking references.
     */
    void setLambdaTracking(
        std::string* last_lambda_name,
        std::unordered_map<std::string, llvm::Function*>* func_table
    ) {
        last_generated_lambda_name_ = last_lambda_name;
        function_table_ = func_table;
    }

    /**
     * Set letrec excluded capture names reference.
     * Used to prevent letrec-bound lambdas from capturing themselves.
     */
    void setLetrecExcludedCaptureNames(std::set<std::string>* names) {
        letrec_excluded_capture_names_ = names;
    }

    // === Tail Call Optimization ===

    /**
     * TCO context for tracking tail-recursive function generation.
     */
    struct TailCallContext {
        std::string func_name = "";           // Name of function being compiled
        llvm::BasicBlock* loop_header = nullptr;    // Loop header for tail call transformation
        std::vector<llvm::AllocaInst*> param_allocas;  // Allocas for mutable parameters
        std::vector<std::string> param_names;    // Parameter names for lookup
        bool enabled = false;                 // Whether TCO is enabled for current lambda
        bool iter_scope = false;              // ESH-0214b: loop body runs inside a
                                              // per-iteration arena scope; the TCO
                                              // back edge must end it (pop/commit)
                                              // before jumping to the loop header

        // ESH-0214e: iter-scope PARTIAL RECLAMATION for mutating loops. When the
        // loop body is escape-safe but contains a barriered structural mutation
        // (vector-set!/vector-fill!/hash-table-set!/set-car!/set-cdr!), the loop
        // is lowered with a per-loop NURSERY REGION instead of the arena-scope
        // path: iteration allocations land in `nursery_region`'s arena, the
        // existing write barriers promote persistent-mutation escapees out of it,
        // and each back edge calls eshkol_iter_nursery_recycle (promote the
        // loop-carried out-values, then reset the nursery). `iter_nursery` and
        // `iter_scope` are mutually exclusive. `nursery_region` is the
        // region_create result; `nursery_saved_arena` is the eshkol_region_enter
        // displaced-arena token restored by eshkol_region_leave at loop exit.
        // Both are SSA values produced in the loop's setup block, which dominates
        // every back edge and the exit.
        bool iter_nursery = false;
        llvm::Value* nursery_region = nullptr;
        llvm::Value* nursery_saved_arena = nullptr;

        // --- ESH-0222: tail calls through `guard` inside a TCO loop ---
        //
        // A self-call that textually appears inside a `guard`'s protected body
        // (in tail position) or one of its handler clauses (also tail position)
        // gets rewritten by codegenTailCallFromContext into an unconditional
        // branch back to `loop_header`, bypassing whatever cleanup code
        // codegenGuard would otherwise have emitted after a *normal* return
        // from the body. Two kinds of per-iteration state must be reclaimed
        // before that branch, or they leak once per loop iteration:
        //
        //  1. `open_guard_handlers` — the number of
        //     eshkol_push_exception_handler() calls, made within the CURRENT
        //     loop body since `loop_header` was entered, that have not yet
        //     been balanced by a compile-time-emitted
        //     eshkol_pop_exception_handler() along this control-flow path.
        //     codegenGuard saves/restores this around its own body exactly
        //     like a stack; codegenTailCallFromContext drains it (emitting
        //     that many pops) immediately before branching back, so the
        //     runtime handler chain (g_exception_handler_stack) never grows
        //     across iterations and never retains a stale jmp_buf pointer.
        //
        //  2. `loop_stack_save` — the result of `llvm.stacksave()` captured
        //     right after entering `loop_header`. guard's jmp_buf is a
        //     dynamically-sized `alloca` (its size comes from a runtime call,
        //     not a compile-time constant), so LLVM cannot hoist or reuse it
        //     across iterations of a hand-rolled branch-based loop the way it
        //     would for a real recursive call's stack frame. Without an
        //     explicit `llvm.stackrestore(loop_stack_save)` before the
        //     back-edge, every iteration that passes through `guard` (or any
        //     other dynamic-alloca construct) permanently consumes more of
        //     the native stack until it overflows — this is the direct cause
        //     of the "stack overflow" SIGSEGV in long-running tick loops that
        //     wrap each iteration in a per-iteration `guard` error boundary.
        unsigned open_guard_handlers = 0;
        llvm::Value* loop_stack_save = nullptr;
    };

    /**
     * Get the current TCO context.
     */
    TailCallContext& getTCOContext() { return tco_context_; }

    /**
     * Check if TCO is currently active for a given function name.
     */
    bool isTCOActive(const std::string& func_name) const {
        return tco_context_.enabled && tco_context_.func_name == func_name;
    }

    /**
     * Set TCO callback for checking if a lambda is self-tail-recursive.
     * Signature: bool (*)(const eshkol_operations_t* lambda_op, const char* func_name, void* context)
     */
    using IsTailRecursiveFunc = bool (*)(const void* lambda_op, const char* func_name, void* context);
    void setTCOCallbacks(IsTailRecursiveFunc is_tail_recursive) {
        is_tail_recursive_callback_ = is_tail_recursive;
    }

private:
    // Backing storage for the active TCO context and the self-tail-recursion
    // detection callback (see getTCOContext()/setTCOCallbacks() above)
    TailCallContext tco_context_ = {};
    IsTailRecursiveFunc is_tail_recursive_callback_ = nullptr;
};

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
#endif // ESHKOL_BACKEND_BINDING_CODEGEN_H
