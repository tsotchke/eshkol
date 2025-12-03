/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * CodegenContext - Shared state for LLVM code generation modules
 *
 * This class provides a centralized access point for all shared state
 * needed by code generation modules. It does not own the resources;
 * it provides references to them for clean dependency injection.
 *
 * Usage:
 *   // In EshkolLLVMCodeGen constructor:
 *   ctx_ = std::make_unique<CodegenContext>(*context, *module, *builder,
 *                                            *types, *funcs, *mem);
 *
 *   // Pass to other modules:
 *   tagged_ = std::make_unique<TaggedValueCodegen>(*ctx_);
 */
#ifndef ESHKOL_BACKEND_CODEGEN_CONTEXT_H
#define ESHKOL_BACKEND_CODEGEN_CONTEXT_H

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/backend/type_system.h>
#include <eshkol/backend/function_cache.h>
#include <eshkol/backend/memory_codegen.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/GlobalVariable.h>
#include <unordered_map>
#include <string>
#include <vector>
#include <stack>

namespace eshkol {

/**
 * CodegenContext centralizes shared state for code generation.
 *
 * This class is the bridge between the main EshkolLLVMCodeGen class
 * and extracted modules like TaggedValueCodegen, ArithmeticCodegen, etc.
 *
 * Key responsibilities:
 * - Provide access to LLVM infrastructure (context, module, builder)
 * - Provide access to type system and function caches
 * - Manage symbol tables and scoping
 * - Track current function context
 * - Manage global variables (arena, AD state)
 */
class CodegenContext {
public:
    // Maximum nesting depth for gradient operations
    static constexpr size_t MAX_TAPE_DEPTH = 32;

    /**
     * Construct CodegenContext with references to LLVM infrastructure.
     * Does not take ownership of any resources.
     */
    CodegenContext(llvm::LLVMContext& llvm_ctx,
                   llvm::Module& llvm_mod,
                   llvm::IRBuilder<>& ir_builder,
                   TypeSystem& type_sys,
                   FunctionCache& func_cache,
                   MemoryCodegen& mem_codegen);

    // === LLVM Infrastructure ===
    llvm::LLVMContext& context() { return context_; }
    llvm::Module& module() { return module_; }
    llvm::IRBuilder<>& builder() { return builder_; }

    // === Type System & Caches ===
    TypeSystem& types() { return types_; }
    FunctionCache& funcs() { return funcs_; }
    MemoryCodegen& memory() { return memory_; }

    // === Commonly Used Types (convenience accessors) ===
    llvm::IntegerType* int64Type() { return types_.getInt64Type(); }
    llvm::IntegerType* int32Type() { return types_.getInt32Type(); }
    llvm::IntegerType* int16Type() { return types_.getInt16Type(); }
    llvm::IntegerType* int8Type() { return types_.getInt8Type(); }
    llvm::IntegerType* int1Type() { return types_.getInt1Type(); }
    llvm::Type* doubleType() { return types_.getDoubleType(); }
    llvm::Type* voidType() { return types_.getVoidType(); }
    llvm::PointerType* ptrType() { return types_.getPtrType(); }
    llvm::StructType* taggedValueType() { return types_.getTaggedValueType(); }
    llvm::StructType* dualNumberType() { return types_.getDualNumberType(); }
    llvm::StructType* adNodeType() { return types_.getAdNodeType(); }
    llvm::StructType* tensorType() { return types_.getTensorType(); }

    // === Symbol Table Management ===

    /** Push a new local scope */
    void pushScope();

    /** Pop the current local scope */
    void popScope();

    /** Look up a symbol in local then global scope */
    llvm::Value* lookupSymbol(const std::string& name);

    /** Define a symbol in the current scope */
    void defineSymbol(const std::string& name, llvm::Value* value);

    /** Define a symbol in global scope */
    void defineGlobalSymbol(const std::string& name, llvm::Value* value);

    /** Look up a symbol in global scope only */
    llvm::Value* lookupGlobalSymbol(const std::string& name);

    // === Function Table ===

    /** Look up a function by name */
    llvm::Function* lookupFunction(const std::string& name);

    /** Register a function */
    void defineFunction(const std::string& name, llvm::Function* func);

    /** Check if a function exists */
    bool hasFunction(const std::string& name) const;

    // === Current Function Context ===

    /** Get the current function being generated */
    llvm::Function* currentFunction() { return current_function_; }

    /** Set the current function */
    void setCurrentFunction(llvm::Function* func) { current_function_ = func; }

    /** Get the main entry block */
    llvm::BasicBlock* mainEntry() { return main_entry_; }

    /** Set the main entry block */
    void setMainEntry(llvm::BasicBlock* block) { main_entry_ = block; }

    // === Variadic Function Info ===

    /** Register a variadic function */
    void registerVariadicFunction(const std::string& name,
                                   uint64_t fixedParams,
                                   bool isVariadic);

    /** Get variadic info for a function (returns {0, false} if not found) */
    std::pair<uint64_t, bool> getVariadicInfo(const std::string& name) const;

    // === String Interning ===

    /** Get or create an interned string global */
    llvm::GlobalVariable* internString(const std::string& str);

    /** Check if a string is already interned */
    llvm::GlobalVariable* lookupInternedString(const std::string& str);

    // === Nested Function Captures ===

    /** Set captures for a nested function */
    void setFunctionCaptures(const std::string& funcName,
                             const std::vector<std::string>& captures);

    /** Get captures for a nested function */
    const std::vector<std::string>* getFunctionCaptures(const std::string& funcName) const;

    // === Lambda Return Tracking ===

    /** Register that a function returns a specific lambda */
    void setFunctionReturnsLambda(const std::string& funcName,
                                   const std::string& lambdaName);

    /** Get the lambda name a function returns (or empty string) */
    std::string getFunctionReturnsLambda(const std::string& funcName) const;

    // === Global Variables (Arena, AD State) ===

    /** Get/set the global arena variable */
    llvm::GlobalVariable* globalArena() { return global_arena_; }
    void setGlobalArena(llvm::GlobalVariable* arena) { global_arena_ = arena; }

    /** Get/set arena scope depth */
    size_t arenaScopeDepth() const { return arena_scope_depth_; }
    void setArenaScopeDepth(size_t depth) { arena_scope_depth_ = depth; }
    void incrementArenaScopeDepth() { ++arena_scope_depth_; }
    void decrementArenaScopeDepth() { if (arena_scope_depth_ > 0) --arena_scope_depth_; }

    // === AD (Automatic Differentiation) State ===

    llvm::GlobalVariable* adModeActive() { return ad_mode_active_; }
    void setAdModeActive(llvm::GlobalVariable* var) { ad_mode_active_ = var; }

    llvm::GlobalVariable* currentAdTape() { return current_ad_tape_; }
    void setCurrentAdTape(llvm::GlobalVariable* var) { current_ad_tape_ = var; }

    llvm::GlobalVariable* adTapeStack() { return ad_tape_stack_; }
    void setAdTapeStack(llvm::GlobalVariable* var) { ad_tape_stack_ = var; }

    llvm::GlobalVariable* adTapeDepth() { return ad_tape_depth_; }
    void setAdTapeDepth(llvm::GlobalVariable* var) { ad_tape_depth_ = var; }

    // Double backward support
    llvm::GlobalVariable* outerAdNodeStorage() { return outer_ad_node_storage_; }
    void setOuterAdNodeStorage(llvm::GlobalVariable* var) { outer_ad_node_storage_ = var; }

    llvm::GlobalVariable* outerAdNodeToInner() { return outer_ad_node_to_inner_; }
    void setOuterAdNodeToInner(llvm::GlobalVariable* var) { outer_ad_node_to_inner_ = var; }

    llvm::GlobalVariable* outerGradAccumulator() { return outer_grad_accumulator_; }
    void setOuterGradAccumulator(llvm::GlobalVariable* var) { outer_grad_accumulator_ = var; }

    llvm::GlobalVariable* innerVarNodePtr() { return inner_var_node_ptr_; }
    void setInnerVarNodePtr(llvm::GlobalVariable* var) { inner_var_node_ptr_ = var; }

    llvm::GlobalVariable* gradientXDegree() { return gradient_x_degree_; }
    void setGradientXDegree(llvm::GlobalVariable* var) { gradient_x_degree_ = var; }

    // N-dimensional derivatives
    llvm::GlobalVariable* outerAdNodeStack() { return outer_ad_node_stack_; }
    void setOuterAdNodeStack(llvm::GlobalVariable* var) { outer_ad_node_stack_ = var; }

    llvm::GlobalVariable* outerAdNodeDepth() { return outer_ad_node_depth_; }
    void setOuterAdNodeDepth(llvm::GlobalVariable* var) { outer_ad_node_depth_ = var; }

    // === Mode Flags ===

    bool isLibraryMode() const { return library_mode_; }
    void setLibraryMode(bool mode) { library_mode_ = mode; }

    bool isReplMode() const { return repl_mode_; }
    void setReplMode(bool mode) { repl_mode_ = mode; }

    const std::string& modulePrefix() const { return module_prefix_; }
    void setModulePrefix(const std::string& prefix) { module_prefix_ = prefix; }

    // === Builtin Function Declarations ===
    // These are set during initialization and used throughout codegen

    llvm::Function* deepEqualFunc() { return deep_equal_func_; }
    void setDeepEqualFunc(llvm::Function* func) { deep_equal_func_ = func; }

    llvm::Function* displayValueFunc() { return display_value_func_; }
    void setDisplayValueFunc(llvm::Function* func) { display_value_func_ = func; }

    llvm::Function* lambdaRegistryInitFunc() { return lambda_registry_init_func_; }
    void setLambdaRegistryInitFunc(llvm::Function* func) { lambda_registry_init_func_ = func; }

    llvm::Function* lambdaRegistryAddFunc() { return lambda_registry_add_func_; }
    void setLambdaRegistryAddFunc(llvm::Function* func) { lambda_registry_add_func_ = func; }

    llvm::Function* lambdaRegistryLookupFunc() { return lambda_registry_lookup_func_; }
    void setLambdaRegistryLookupFunc(llvm::Function* func) { lambda_registry_lookup_func_ = func; }

    // List operation helpers
    llvm::Function* lengthImplFunc() { return length_impl_func_; }
    void setLengthImplFunc(llvm::Function* func) { length_impl_func_ = func; }

    llvm::Function* appendImplFunc() { return append_impl_func_; }
    void setAppendImplFunc(llvm::Function* func) { append_impl_func_ = func; }

    llvm::Function* reverseImplFunc() { return reverse_impl_func_; }
    void setReverseImplFunc(llvm::Function* func) { reverse_impl_func_ = func; }

    llvm::Function* listRefImplFunc() { return list_ref_impl_func_; }
    void setListRefImplFunc(llvm::Function* func) { list_ref_impl_func_ = func; }

    llvm::Function* listTailImplFunc() { return list_tail_impl_func_; }
    void setListTailImplFunc(llvm::Function* func) { list_tail_impl_func_ = func; }

    llvm::Function* displayTensorRecursiveFunc() { return display_tensor_recursive_func_; }
    void setDisplayTensorRecursiveFunc(llvm::Function* func) { display_tensor_recursive_func_ = func; }

private:
    // LLVM infrastructure (references, not owned)
    llvm::LLVMContext& context_;
    llvm::Module& module_;
    llvm::IRBuilder<>& builder_;

    // Our modules (references, not owned)
    TypeSystem& types_;
    FunctionCache& funcs_;
    MemoryCodegen& memory_;

    // Symbol tables
    std::vector<std::unordered_map<std::string, llvm::Value*>> scope_stack_;
    std::unordered_map<std::string, llvm::Value*> global_symbols_;
    std::unordered_map<std::string, llvm::Function*> function_table_;

    // Function metadata
    std::unordered_map<std::string, std::pair<uint64_t, bool>> variadic_info_;
    std::unordered_map<std::string, std::vector<std::string>> function_captures_;
    std::unordered_map<std::string, std::string> functions_returning_lambda_;

    // String interning
    std::unordered_map<std::string, llvm::GlobalVariable*> interned_strings_;

    // Current function context
    llvm::Function* current_function_ = nullptr;
    llvm::BasicBlock* main_entry_ = nullptr;

    // Global variables
    llvm::GlobalVariable* global_arena_ = nullptr;
    size_t arena_scope_depth_ = 0;

    // AD state
    llvm::GlobalVariable* ad_mode_active_ = nullptr;
    llvm::GlobalVariable* current_ad_tape_ = nullptr;
    llvm::GlobalVariable* ad_tape_stack_ = nullptr;
    llvm::GlobalVariable* ad_tape_depth_ = nullptr;
    llvm::GlobalVariable* outer_ad_node_storage_ = nullptr;
    llvm::GlobalVariable* outer_ad_node_to_inner_ = nullptr;
    llvm::GlobalVariable* outer_grad_accumulator_ = nullptr;
    llvm::GlobalVariable* inner_var_node_ptr_ = nullptr;
    llvm::GlobalVariable* gradient_x_degree_ = nullptr;
    llvm::GlobalVariable* outer_ad_node_stack_ = nullptr;
    llvm::GlobalVariable* outer_ad_node_depth_ = nullptr;

    // Mode flags
    bool library_mode_ = false;
    bool repl_mode_ = false;
    std::string module_prefix_;

    // Builtin function declarations
    llvm::Function* deep_equal_func_ = nullptr;
    llvm::Function* display_value_func_ = nullptr;
    llvm::Function* lambda_registry_init_func_ = nullptr;
    llvm::Function* lambda_registry_add_func_ = nullptr;
    llvm::Function* lambda_registry_lookup_func_ = nullptr;
    llvm::Function* length_impl_func_ = nullptr;
    llvm::Function* append_impl_func_ = nullptr;
    llvm::Function* reverse_impl_func_ = nullptr;
    llvm::Function* list_ref_impl_func_ = nullptr;
    llvm::Function* list_tail_impl_func_ = nullptr;
    llvm::Function* display_tensor_recursive_func_ = nullptr;
};

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
#endif // ESHKOL_BACKEND_CODEGEN_CONTEXT_H
