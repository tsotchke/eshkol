/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * LogicWorkspaceCodegen — Code generation for the consciousness engine
 * primitives: logic variables, unification, knowledge bases, factor graphs,
 * active inference, the global workspace, and tensor / model serialization.
 *
 * Extracted from llvm_codegen.cpp during the v1.2 mechanical refactor.
 * All 23 functions in this module are thin wrappers around eshkol_*_tagged
 * runtime functions. They share the pattern:
 *
 *   1. ensureTagged() each AST argument
 *   2. allocate result slot
 *   3. declare the runtime function (signature varies)
 *   4. CreateCall + load result
 *
 * IR-identical to the prior in-class implementations; verified via the
 * pre-codegencall-split baseline.
 */
#ifndef ESHKOL_BACKEND_LOGIC_WORKSPACE_CODEGEN_H
#define ESHKOL_BACKEND_LOGIC_WORKSPACE_CODEGEN_H

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/backend/codegen_context.h>
#include <eshkol/backend/tagged_value_codegen.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/Function.h>
#include <vector>

// Forward declaration for the operations struct (lives in eshkol/eshkol.h).
struct eshkol_operation;
typedef struct eshkol_operation eshkol_operations_t;

namespace eshkol {

/**
 * LogicWorkspaceCodegen generates LLVM IR for Eshkol's consciousness-engine
 * primitives. Construction is lightweight; the heavy work (runtime function
 * declarations) happens lazily on first use via getOrDeclareRuntimeFunc().
 *
 * Several methods recurse into AST evaluation. Wire that callback via
 * setCodegenASTCallback() — the parent EshkolLLVMCodeGen passes a wrapper
 * that bounces back into its own codegenAST().
 */
class LogicWorkspaceCodegen {
public:
    /// Callback type for recursive AST code generation.
    /// Takes an eshkol_ast_t* (opaque to this module) and a context void*.
    using CodegenASTCallback = llvm::Value* (*)(const void* ast, void* context);

    /// Callback type for closure invocation. codegenWSStep needs this to
    /// invoke each registered module's process_fn against the workspace
    /// content tensor.
    using CodegenClosureCallCallback = llvm::Value* (*)(
        llvm::Value* func_result,
        const std::vector<llvm::Value*>& call_args,
        const char* caller_info,
        void* context);

    /**
     * @brief Construct a LogicWorkspaceCodegen.
     * @param ctx CodegenContext shared by all backend codegen modules.
     * @param tagged TaggedValueCodegen used to pack/unpack runtime values.
     */
    LogicWorkspaceCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged);

    /// Wire the codegenAST callback before any generator method runs.
    void setCodegenASTCallback(CodegenASTCallback cb, void* context);

    /// Wire the codegenClosureCall callback (only needed for codegenWSStep).
    void setCodegenClosureCallCallback(CodegenClosureCallCallback cb);

    // ─── Logic primitives ─────────────────────────────────────────────────

    /**
     * @brief Generate code for (logic-var 'name) — create a fresh logic variable.
     * @param op The operation AST node
     * @return Tagged logic-variable value
     */
    llvm::Value* codegenLogicVar(const eshkol_operations_t* op);

    /**
     * @brief Generate code for (unify a b subst) — unify two terms under a substitution.
     * @param op The operation AST node
     * @return Extended substitution on success, or #f on failure
     */
    llvm::Value* codegenUnify(const eshkol_operations_t* op);

    /**
     * @brief Generate code for (make-subst) — create an empty substitution.
     * @param op The operation AST node
     * @return Tagged substitution value
     */
    llvm::Value* codegenMakeSubst(const eshkol_operations_t* op);

    /**
     * @brief Generate code for (walk term subst) — resolve a term through a substitution.
     * @param op The operation AST node
     * @return The fully-walked term
     */
    llvm::Value* codegenWalk(const eshkol_operations_t* op);

    // ─── Knowledge base ───────────────────────────────────────────────────

    /**
     * @brief Generate code for (make-fact ...) — build a knowledge-base fact.
     * @param op The operation AST node
     * @return Tagged fact value
     */
    llvm::Value* codegenMakeFact(const eshkol_operations_t* op);

    /**
     * @brief Generate code for (make-kb) — create an empty knowledge base.
     * @param op The operation AST node
     * @return Tagged knowledge-base value
     */
    llvm::Value* codegenMakeKB(const eshkol_operations_t* op);

    /**
     * @brief Generate code for (kb-assert! kb fact) — add a fact to a knowledge base.
     * @param op The operation AST node
     * @return Unspecified value
     */
    llvm::Value* codegenKBAssert(const eshkol_operations_t* op);

    /**
     * @brief Generate code for (kb-query kb pattern) — query facts matching a pattern.
     * @param op The operation AST node
     * @return List of matching bindings/facts
     */
    llvm::Value* codegenKBQuery(const eshkol_operations_t* op);

    /**
     * @brief Generate code for (kb-query-prefix kb prefix) — query facts by key prefix.
     * @param op The operation AST node
     * @return List of matching bindings/facts
     */
    llvm::Value* codegenKBQueryPrefix(const eshkol_operations_t* op);

    // ─── Factor graph + active inference ──────────────────────────────────

    /**
     * @brief Generate code for (make-factor-graph) — create an empty factor graph.
     * @param op The operation AST node
     * @return Tagged factor-graph value
     */
    llvm::Value* codegenMakeFactorGraph(const eshkol_operations_t* op);

    /**
     * @brief Generate code for (fg-add-factor! graph ...) — add a factor/CPT to a factor graph.
     * @param op The operation AST node
     * @return Unspecified value
     */
    llvm::Value* codegenFGAddFactor(const eshkol_operations_t* op);

    /**
     * @brief Generate code for (fg-infer! graph) — run belief propagation / inference.
     * @param op The operation AST node
     * @return Inference result (posterior beliefs)
     */
    llvm::Value* codegenFGInfer(const eshkol_operations_t* op);

    /**
     * @brief Generate code for (free-energy graph) — compute variational free energy.
     * @param op The operation AST node
     * @return Scalar free-energy value
     */
    llvm::Value* codegenFreeEnergy(const eshkol_operations_t* op);

    /**
     * @brief Generate code for (expected-free-energy graph ...) — compute EFE for active inference.
     * @param op The operation AST node
     * @return Scalar expected free-energy value
     */
    llvm::Value* codegenEFE(const eshkol_operations_t* op);

    /**
     * @brief Generate code for (fg-update-cpt! graph ...) — update a factor's conditional probability table.
     * @param op The operation AST node
     * @return Unspecified value
     */
    llvm::Value* codegenFGUpdateCPT(const eshkol_operations_t* op);

    /**
     * @brief Generate code for observing evidence on a factor graph node.
     * @param op The operation AST node
     * @return Unspecified value
     */
    llvm::Value* codegenFGObserve(const eshkol_operations_t* op);

    // ─── Global workspace ─────────────────────────────────────────────────

    /**
     * @brief Generate code for (make-workspace) — create the global workspace.
     * @param op The operation AST node
     * @return Tagged workspace value
     */
    llvm::Value* codegenMakeWorkspace(const eshkol_operations_t* op);

    /**
     * @brief Generate code for (ws-register! workspace module) — register a processing module.
     * @param op The operation AST node
     * @return Unspecified value
     */
    llvm::Value* codegenWSRegister(const eshkol_operations_t* op);

    /**
     * @brief Generate code for (ws-step! workspace) — run one global-workspace broadcast cycle.
     *
     * Invokes each registered module's process closure against the workspace
     * content tensor via the closure-call callback.
     * @param op The operation AST node
     * @return Unspecified value
     */
    llvm::Value* codegenWSStep(const eshkol_operations_t* op);

    // ─── Tensor / model serialization ─────────────────────────────────────

    /**
     * @brief Generate code for (tensor-save tensor path) — serialize a tensor to disk.
     * @param op The operation AST node
     * @return #t on success, #f on failure
     */
    llvm::Value* codegenTensorSave(const eshkol_operations_t* op);

    /**
     * @brief Generate code for (tensor-load path) — deserialize a tensor from disk.
     * @param op The operation AST node
     * @return Loaded tensor, or #f on failure
     */
    llvm::Value* codegenTensorLoad(const eshkol_operations_t* op);

    /**
     * @brief Generate code for (model-save model path) — serialize a model (collection of tensors) to disk.
     * @param op The operation AST node
     * @return #t on success, #f on failure
     */
    llvm::Value* codegenModelSave(const eshkol_operations_t* op);

    /**
     * @brief Generate code for (model-load path) — deserialize a model from disk.
     * @param op The operation AST node
     * @return Loaded model, or #f on failure
     */
    llvm::Value* codegenModelLoad(const eshkol_operations_t* op);

private:
    // Trampolines to the parent's recursive codegen.
    llvm::Value* codegenAST(const void* ast);
    llvm::Value* codegenClosureCall(llvm::Value* func_result,
                                    const std::vector<llvm::Value*>& call_args,
                                    const char* caller_info);

    // Small utilities ported from EshkolLLVMCodeGen. Re-implementing them
    // here (rather than promoting them to CodegenContext) keeps the parent
    // class API stable for now; we can consolidate later when more modules
    // need them.
    llvm::Function* getOrDeclareRuntimeFunc(const char* name, llvm::FunctionType* ft);
    llvm::Value* loadArenaPtr();
    llvm::Value* allocaAndStore(llvm::Value* val, const char* name);
    llvm::Value* allocaResult(const char* name);
    llvm::Value* loadResult(llvm::Value* slot, const char* name);

    CodegenContext& ctx_;
    TaggedValueCodegen& tagged_;
    CodegenASTCallback ast_cb_ = nullptr;
    CodegenClosureCallCallback closure_cb_ = nullptr;
    void* cb_context_ = nullptr;  // shared parent-EshkolLLVMCodeGen pointer
};

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED

#endif // ESHKOL_BACKEND_LOGIC_WORKSPACE_CODEGEN_H
