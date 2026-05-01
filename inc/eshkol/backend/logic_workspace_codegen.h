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

    LogicWorkspaceCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged);

    /// Wire the codegenAST callback before any generator method runs.
    void setCodegenASTCallback(CodegenASTCallback cb, void* context);

    /// Wire the codegenClosureCall callback (only needed for codegenWSStep).
    void setCodegenClosureCallCallback(CodegenClosureCallCallback cb);

    // ─── Logic primitives ─────────────────────────────────────────────────
    llvm::Value* codegenLogicVar(const eshkol_operations_t* op);
    llvm::Value* codegenUnify(const eshkol_operations_t* op);
    llvm::Value* codegenMakeSubst(const eshkol_operations_t* op);
    llvm::Value* codegenWalk(const eshkol_operations_t* op);

    // ─── Knowledge base ───────────────────────────────────────────────────
    llvm::Value* codegenMakeFact(const eshkol_operations_t* op);
    llvm::Value* codegenMakeKB(const eshkol_operations_t* op);
    llvm::Value* codegenKBAssert(const eshkol_operations_t* op);
    llvm::Value* codegenKBQuery(const eshkol_operations_t* op);
    llvm::Value* codegenKBQueryPrefix(const eshkol_operations_t* op);

    // ─── Factor graph + active inference ──────────────────────────────────
    llvm::Value* codegenMakeFactorGraph(const eshkol_operations_t* op);
    llvm::Value* codegenFGAddFactor(const eshkol_operations_t* op);
    llvm::Value* codegenFGInfer(const eshkol_operations_t* op);
    llvm::Value* codegenFreeEnergy(const eshkol_operations_t* op);
    llvm::Value* codegenEFE(const eshkol_operations_t* op);
    llvm::Value* codegenFGUpdateCPT(const eshkol_operations_t* op);
    llvm::Value* codegenFGObserve(const eshkol_operations_t* op);

    // ─── Global workspace ─────────────────────────────────────────────────
    llvm::Value* codegenMakeWorkspace(const eshkol_operations_t* op);
    llvm::Value* codegenWSRegister(const eshkol_operations_t* op);
    llvm::Value* codegenWSStep(const eshkol_operations_t* op);

    // ─── Tensor / model serialization ─────────────────────────────────────
    llvm::Value* codegenTensorSave(const eshkol_operations_t* op);
    llvm::Value* codegenTensorLoad(const eshkol_operations_t* op);
    llvm::Value* codegenModelSave(const eshkol_operations_t* op);
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
