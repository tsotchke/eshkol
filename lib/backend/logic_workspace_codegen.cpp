/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * LogicWorkspaceCodegen — implementation
 *
 * Extracted from llvm_codegen.cpp during the v1.2 mechanical refactor.
 * Bodies preserved verbatim modulo helper-call rewriting:
 *
 *   ensureTaggedValue(x)             -> tagged_.ensureTagged(x)
 *   packNullToTaggedValue()          -> tagged_.packNull()
 *   packBoolToTaggedValue(b)         -> tagged_.packBool(b)
 *   packInt64ToTaggedValueWithType-
 *     AndFlags(v, t, f)              -> tagged_.packInt64WithTypeAndFlags(v, t, f)
 *   builder->...                     -> ctx_.builder()....
 *   int8_type / int32_type / ...     -> ctx_.int8Type() / ...
 *   tagged_value_type                -> ctx_.taggedValueType()
 *   loadResult / allocaResult /
 *     allocaAndStore / loadArenaPtr /
 *     getOrDeclareRuntimeFunc        -> private helpers below
 *   codegenAST / codegenClosureCall  -> private trampolines using callbacks
 *
 * IR-identical to the prior in-class implementations.
 */

#include <eshkol/backend/logic_workspace_codegen.h>
#include <eshkol/eshkol.h>
#include <eshkol/logger.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/BasicBlock.h>

namespace eshkol {

using llvm::Value;
using llvm::Function;
using llvm::FunctionType;
using llvm::ConstantInt;
using llvm::PointerType;
using llvm::Type;
using llvm::BasicBlock;
using llvm::PHINode;

LogicWorkspaceCodegen::LogicWorkspaceCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged)
    : ctx_(ctx), tagged_(tagged) {}

void LogicWorkspaceCodegen::setCodegenASTCallback(CodegenASTCallback cb, void* context) {
    ast_cb_ = cb;
    cb_context_ = context;
}

void LogicWorkspaceCodegen::setCodegenClosureCallCallback(CodegenClosureCallCallback cb) {
    closure_cb_ = cb;
}

// ---------------------------------------------------------------------------
// Trampolines + utilities
// ---------------------------------------------------------------------------

Value* LogicWorkspaceCodegen::codegenAST(const void* ast) {
    if (!ast_cb_ || !cb_context_) {
        eshkol_error("LogicWorkspaceCodegen: codegenAST callback not configured");
        return nullptr;
    }
    return ast_cb_(ast, cb_context_);
}

Value* LogicWorkspaceCodegen::codegenClosureCall(Value* func_result,
                                                 const std::vector<Value*>& call_args,
                                                 const char* caller_info) {
    if (!closure_cb_ || !cb_context_) {
        eshkol_error("LogicWorkspaceCodegen: codegenClosureCall callback not configured");
        return nullptr;
    }
    return closure_cb_(func_result, call_args, caller_info, cb_context_);
}

Function* LogicWorkspaceCodegen::getOrDeclareRuntimeFunc(const char* name, FunctionType* ft) {
    auto& module = ctx_.module();
    if (Function* existing = module.getFunction(name)) {
        return existing;
    }
    return Function::Create(ft, Function::ExternalLinkage, name, module);
}

Value* LogicWorkspaceCodegen::loadArenaPtr() {
    auto& builder = ctx_.builder();
    auto* arena_global = ctx_.globalArena();
    return builder.CreateLoad(ctx_.ptrType(), arena_global, "arena_ptr");
}

Value* LogicWorkspaceCodegen::allocaAndStore(Value* val, const char* name) {
    auto& builder = ctx_.builder();
    Value* slot = builder.CreateAlloca(ctx_.taggedValueType(), nullptr, name);
    builder.CreateStore(val, slot);
    return slot;
}

Value* LogicWorkspaceCodegen::allocaResult(const char* name) {
    auto& builder = ctx_.builder();
    return builder.CreateAlloca(ctx_.taggedValueType(), nullptr, name);
}

Value* LogicWorkspaceCodegen::loadResult(Value* slot, const char* name) {
    auto& builder = ctx_.builder();
    return builder.CreateLoad(ctx_.taggedValueType(), slot, name);
}

// ---------------------------------------------------------------------------
// Logic primitives
// ---------------------------------------------------------------------------

Value* LogicWorkspaceCodegen::codegenLogicVar(const eshkol_operations_t* op) {
    // Pack logic variable as tagged value: type=ESHKOL_VALUE_LOGIC_VAR, data=var_id
    uint64_t var_id = op->logic_var_op.var_id;
    Value* var_id_val = ConstantInt::get(ctx_.int64Type(), var_id);
    Value* type_val = ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_LOGIC_VAR);
    Value* flags_val = ConstantInt::get(ctx_.int8Type(), 0);
    return tagged_.packInt64WithTypeAndFlags(var_id_val, type_val, flags_val);
}

Value* LogicWorkspaceCodegen::codegenUnify(const eshkol_operations_t* op) {
    auto& builder = ctx_.builder();
    if (op->call_op.num_vars < 3) return tagged_.packNull();

    Value* t1 = tagged_.ensureTagged(codegenAST(&op->call_op.variables[0]));
    Value* t2 = tagged_.ensureTagged(codegenAST(&op->call_op.variables[1]));
    Value* subst = tagged_.ensureTagged(codegenAST(&op->call_op.variables[2]));

    Value* t1_a = allocaAndStore(t1, "unify_t1");
    Value* t2_a = allocaAndStore(t2, "unify_t2");
    Value* subst_a = allocaAndStore(subst, "unify_subst");
    Value* result_a = allocaResult("unify_res");

    FunctionType* fn_type = FunctionType::get(ctx_.voidType(),
        {ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType()}, false);
    Function* func = getOrDeclareRuntimeFunc("eshkol_unify_tagged", fn_type);

    builder.CreateCall(func, {loadArenaPtr(), t1_a, t2_a, subst_a, result_a});
    return loadResult(result_a, "unify_result");
}

Value* LogicWorkspaceCodegen::codegenMakeSubst(const eshkol_operations_t* /*op*/) {
    auto& builder = ctx_.builder();
    Value* result_a = allocaResult("subst_res");

    FunctionType* fn_type = FunctionType::get(ctx_.voidType(),
        {ctx_.ptrType(), ctx_.ptrType()}, false);
    Function* func = getOrDeclareRuntimeFunc("eshkol_make_substitution_tagged", fn_type);

    builder.CreateCall(func, {loadArenaPtr(), result_a});
    return loadResult(result_a, "subst_result");
}

Value* LogicWorkspaceCodegen::codegenWalk(const eshkol_operations_t* op) {
    auto& builder = ctx_.builder();
    if (op->call_op.num_vars < 2) return tagged_.packNull();

    Value* term = tagged_.ensureTagged(codegenAST(&op->call_op.variables[0]));
    Value* subst = tagged_.ensureTagged(codegenAST(&op->call_op.variables[1]));

    Value* term_a = allocaAndStore(term, "walk_term");
    Value* subst_a = allocaAndStore(subst, "walk_subst");
    Value* result_a = allocaResult("walk_res");

    FunctionType* fn_type = FunctionType::get(ctx_.voidType(),
        {ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType()}, false);
    Function* func = getOrDeclareRuntimeFunc("eshkol_walk_tagged", fn_type);

    builder.CreateCall(func, {term_a, subst_a, result_a});
    return loadResult(result_a, "walk_result");
}

// ---------------------------------------------------------------------------
// Knowledge base
// ---------------------------------------------------------------------------

Value* LogicWorkspaceCodegen::codegenMakeFact(const eshkol_operations_t* op) {
    auto& builder = ctx_.builder();
    auto& context = ctx_.context();
    if (op->call_op.num_vars < 1) return tagged_.packNull();

    Value* pred = tagged_.ensureTagged(codegenAST(&op->call_op.variables[0]));
    Value* pred_a = allocaAndStore(pred, "fact_pred");

    uint32_t arity = op->call_op.num_vars - 1;
    Value* args_a = nullptr;
    if (arity > 0) {
        args_a = builder.CreateAlloca(ctx_.taggedValueType(),
            ConstantInt::get(ctx_.int32Type(), arity), "fact_args");
        for (uint32_t i = 0; i < arity; i++) {
            Value* arg = tagged_.ensureTagged(codegenAST(&op->call_op.variables[i + 1]));
            Value* gep = builder.CreateGEP(ctx_.taggedValueType(), args_a,
                ConstantInt::get(ctx_.int32Type(), i));
            builder.CreateStore(arg, gep);
        }
    } else {
        args_a = llvm::ConstantPointerNull::get(PointerType::getUnqual(context));
    }

    Value* result_a = allocaResult("fact_res");

    FunctionType* fn_type = FunctionType::get(ctx_.voidType(),
        {ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType(), ctx_.int32Type(), ctx_.ptrType()}, false);
    Function* func = getOrDeclareRuntimeFunc("eshkol_make_fact_tagged", fn_type);

    builder.CreateCall(func, {loadArenaPtr(), pred_a, args_a,
        ConstantInt::get(ctx_.int32Type(), arity), result_a});
    return loadResult(result_a, "fact_result");
}

Value* LogicWorkspaceCodegen::codegenMakeKB(const eshkol_operations_t* /*op*/) {
    auto& builder = ctx_.builder();
    Value* result_a = allocaResult("kb_res");

    FunctionType* fn_type = FunctionType::get(ctx_.voidType(),
        {ctx_.ptrType(), ctx_.ptrType()}, false);
    Function* func = getOrDeclareRuntimeFunc("eshkol_make_kb_tagged", fn_type);

    builder.CreateCall(func, {loadArenaPtr(), result_a});
    return loadResult(result_a, "kb_result");
}

Value* LogicWorkspaceCodegen::codegenKBAssert(const eshkol_operations_t* op) {
    auto& builder = ctx_.builder();
    if (op->call_op.num_vars < 2) return tagged_.packNull();

    Value* kb = tagged_.ensureTagged(codegenAST(&op->call_op.variables[0]));
    Value* fact = tagged_.ensureTagged(codegenAST(&op->call_op.variables[1]));

    Value* kb_a = allocaAndStore(kb, "assert_kb");
    Value* fact_a = allocaAndStore(fact, "assert_fact");

    FunctionType* fn_type = FunctionType::get(ctx_.voidType(),
        {ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType()}, false);
    Function* func = getOrDeclareRuntimeFunc("eshkol_kb_assert_tagged", fn_type);

    builder.CreateCall(func, {loadArenaPtr(), kb_a, fact_a});
    return tagged_.packNull();
}

Value* LogicWorkspaceCodegen::codegenKBQuery(const eshkol_operations_t* op) {
    auto& builder = ctx_.builder();
    if (op->call_op.num_vars < 2) return tagged_.packNull();

    Value* kb = tagged_.ensureTagged(codegenAST(&op->call_op.variables[0]));
    Value* pattern = tagged_.ensureTagged(codegenAST(&op->call_op.variables[1]));

    Value* kb_a = allocaAndStore(kb, "query_kb");
    Value* pattern_a = allocaAndStore(pattern, "query_pat");
    Value* result_a = allocaResult("query_res");

    FunctionType* fn_type = FunctionType::get(ctx_.voidType(),
        {ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType()}, false);
    Function* func = getOrDeclareRuntimeFunc("eshkol_kb_query_tagged", fn_type);

    builder.CreateCall(func, {loadArenaPtr(), kb_a, pattern_a, result_a});
    return loadResult(result_a, "query_result");
}

Value* LogicWorkspaceCodegen::codegenKBQueryPrefix(const eshkol_operations_t* op) {
    auto& builder = ctx_.builder();
    if (op->call_op.num_vars < 2) return tagged_.packNull();

    Value* kb = tagged_.ensureTagged(codegenAST(&op->call_op.variables[0]));
    Value* pattern = tagged_.ensureTagged(codegenAST(&op->call_op.variables[1]));

    Value* kb_a = allocaAndStore(kb, "qprefix_kb");
    Value* pattern_a = allocaAndStore(pattern, "qprefix_pat");
    Value* result_a = allocaResult("qprefix_res");

    FunctionType* fn_type = FunctionType::get(ctx_.voidType(),
        {ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType()}, false);
    Function* func = getOrDeclareRuntimeFunc("eshkol_kb_query_prefix_tagged", fn_type);

    builder.CreateCall(func, {loadArenaPtr(), kb_a, pattern_a, result_a});
    return loadResult(result_a, "qprefix_result");
}

// ---------------------------------------------------------------------------
// Tensor / model serialization
// ---------------------------------------------------------------------------

Value* LogicWorkspaceCodegen::codegenTensorSave(const eshkol_operations_t* op) {
    auto& builder = ctx_.builder();
    if (op->call_op.num_vars < 2) return tagged_.packNull();

    Value* path = tagged_.ensureTagged(codegenAST(&op->call_op.variables[0]));
    Value* tensor = tagged_.ensureTagged(codegenAST(&op->call_op.variables[1]));
    Value* path_a = allocaAndStore(path, "tensor_save_path");
    Value* tensor_a = allocaAndStore(tensor, "tensor_save_tensor");
    Value* result_a = allocaResult("tensor_save_result");

    FunctionType* fn_type = FunctionType::get(ctx_.voidType(),
        {ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType()}, false);
    Function* func = getOrDeclareRuntimeFunc("eshkol_tensor_save_tagged", fn_type);
    builder.CreateCall(func, {loadArenaPtr(), path_a, tensor_a, result_a});
    return loadResult(result_a, "tensor_save_value");
}

Value* LogicWorkspaceCodegen::codegenTensorLoad(const eshkol_operations_t* op) {
    auto& builder = ctx_.builder();
    if (op->call_op.num_vars < 1) return tagged_.packNull();

    Value* path = tagged_.ensureTagged(codegenAST(&op->call_op.variables[0]));
    Value* path_a = allocaAndStore(path, "tensor_load_path");
    Value* result_a = allocaResult("tensor_load_result");

    FunctionType* fn_type = FunctionType::get(ctx_.voidType(),
        {ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType()}, false);
    Function* func = getOrDeclareRuntimeFunc("eshkol_tensor_load_tagged", fn_type);
    builder.CreateCall(func, {loadArenaPtr(), path_a, result_a});
    return loadResult(result_a, "tensor_load_value");
}

Value* LogicWorkspaceCodegen::codegenModelSave(const eshkol_operations_t* op) {
    auto& builder = ctx_.builder();
    if (op->call_op.num_vars < 2) return tagged_.packNull();

    Value* path = tagged_.ensureTagged(codegenAST(&op->call_op.variables[0]));
    Value* entries = tagged_.ensureTagged(codegenAST(&op->call_op.variables[1]));
    Value* path_a = allocaAndStore(path, "model_save_path");
    Value* entries_a = allocaAndStore(entries, "model_save_entries");
    Value* result_a = allocaResult("model_save_result");

    FunctionType* fn_type = FunctionType::get(ctx_.voidType(),
        {ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType()}, false);
    Function* func = getOrDeclareRuntimeFunc("eshkol_model_save_tagged", fn_type);
    builder.CreateCall(func, {loadArenaPtr(), path_a, entries_a, result_a});
    return loadResult(result_a, "model_save_value");
}

Value* LogicWorkspaceCodegen::codegenModelLoad(const eshkol_operations_t* op) {
    auto& builder = ctx_.builder();
    if (op->call_op.num_vars < 1) return tagged_.packNull();

    Value* path = tagged_.ensureTagged(codegenAST(&op->call_op.variables[0]));
    Value* path_a = allocaAndStore(path, "model_load_path");
    Value* result_a = allocaResult("model_load_result");

    FunctionType* fn_type = FunctionType::get(ctx_.voidType(),
        {ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType()}, false);
    Function* func = getOrDeclareRuntimeFunc("eshkol_model_load_tagged", fn_type);
    builder.CreateCall(func, {loadArenaPtr(), path_a, result_a});
    return loadResult(result_a, "model_load_value");
}

// ---------------------------------------------------------------------------
// Factor graph + active inference
// ---------------------------------------------------------------------------

Value* LogicWorkspaceCodegen::codegenMakeFactorGraph(const eshkol_operations_t* op) {
    auto& builder = ctx_.builder();
    if (op->call_op.num_vars < 2) return tagged_.packNull();

    Value* num_vars = tagged_.ensureTagged(codegenAST(&op->call_op.variables[0]));
    Value* dims = tagged_.ensureTagged(codegenAST(&op->call_op.variables[1]));

    Value* nv_a = allocaAndStore(num_vars, "fg_nvars");
    Value* dims_a = allocaAndStore(dims, "fg_dims");
    Value* result_a = allocaResult("fg_res");

    FunctionType* fn_type = FunctionType::get(ctx_.voidType(),
        {ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType()}, false);
    Function* func = getOrDeclareRuntimeFunc("eshkol_make_factor_graph_tagged", fn_type);

    builder.CreateCall(func, {loadArenaPtr(), nv_a, dims_a, result_a});
    return loadResult(result_a, "fg_result");
}

Value* LogicWorkspaceCodegen::codegenFGAddFactor(const eshkol_operations_t* op) {
    auto& builder = ctx_.builder();
    if (op->call_op.num_vars < 3) return tagged_.packNull();

    Value* fg = tagged_.ensureTagged(codegenAST(&op->call_op.variables[0]));
    Value* var_indices = tagged_.ensureTagged(codegenAST(&op->call_op.variables[1]));
    Value* cpt = tagged_.ensureTagged(codegenAST(&op->call_op.variables[2]));

    Value* fg_a = allocaAndStore(fg, "fgaf_fg");
    Value* vi_a = allocaAndStore(var_indices, "fgaf_vi");
    Value* cpt_a = allocaAndStore(cpt, "fgaf_cpt");
    Value* result_a = allocaResult("fgaf_res");

    FunctionType* fn_type = FunctionType::get(ctx_.voidType(),
        {ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType()}, false);
    Function* func = getOrDeclareRuntimeFunc("eshkol_fg_add_factor_tagged", fn_type);

    builder.CreateCall(func, {loadArenaPtr(), fg_a, vi_a, cpt_a, result_a});
    return tagged_.packNull();
}

Value* LogicWorkspaceCodegen::codegenFGInfer(const eshkol_operations_t* op) {
    auto& builder = ctx_.builder();
    if (op->call_op.num_vars < 2) return tagged_.packNull();

    Value* fg = tagged_.ensureTagged(codegenAST(&op->call_op.variables[0]));
    Value* max_iters = tagged_.ensureTagged(codegenAST(&op->call_op.variables[1]));

    Value* fg_a = allocaAndStore(fg, "fgi_fg");
    Value* mi_a = allocaAndStore(max_iters, "fgi_mi");
    Value* result_a = allocaResult("fgi_res");

    FunctionType* fn_type = FunctionType::get(ctx_.voidType(),
        {ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType()}, false);
    Function* func = getOrDeclareRuntimeFunc("eshkol_fg_infer_tagged", fn_type);

    builder.CreateCall(func, {loadArenaPtr(), fg_a, mi_a, result_a});
    return loadResult(result_a, "infer_result");
}

Value* LogicWorkspaceCodegen::codegenFreeEnergy(const eshkol_operations_t* op) {
    auto& builder = ctx_.builder();
    if (op->call_op.num_vars < 2) return tagged_.packNull();

    Value* fg = tagged_.ensureTagged(codegenAST(&op->call_op.variables[0]));
    Value* obs = tagged_.ensureTagged(codegenAST(&op->call_op.variables[1]));

    Value* fg_a = allocaAndStore(fg, "fe_fg");
    Value* obs_a = allocaAndStore(obs, "fe_obs");
    Value* result_a = allocaResult("fe_res");

    FunctionType* fn_type = FunctionType::get(ctx_.voidType(),
        {ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType()}, false);
    Function* func = getOrDeclareRuntimeFunc("eshkol_free_energy_tagged", fn_type);

    builder.CreateCall(func, {fg_a, obs_a, result_a});
    return loadResult(result_a, "fe_result");
}

Value* LogicWorkspaceCodegen::codegenEFE(const eshkol_operations_t* op) {
    auto& builder = ctx_.builder();
    if (op->call_op.num_vars < 3) return tagged_.packNull();

    Value* fg = tagged_.ensureTagged(codegenAST(&op->call_op.variables[0]));
    Value* action_var = tagged_.ensureTagged(codegenAST(&op->call_op.variables[1]));
    Value* action_state = tagged_.ensureTagged(codegenAST(&op->call_op.variables[2]));

    Value* fg_a = allocaAndStore(fg, "efe_fg");
    Value* av_a = allocaAndStore(action_var, "efe_av");
    Value* as_a = allocaAndStore(action_state, "efe_as");
    Value* result_a = allocaResult("efe_res");

    FunctionType* fn_type = FunctionType::get(ctx_.voidType(),
        {ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType()}, false);
    Function* func = getOrDeclareRuntimeFunc("eshkol_efe_tagged", fn_type);

    builder.CreateCall(func, {loadArenaPtr(), fg_a, av_a, as_a, result_a});
    return loadResult(result_a, "efe_result");
}

Value* LogicWorkspaceCodegen::codegenFGUpdateCPT(const eshkol_operations_t* op) {
    auto& builder = ctx_.builder();
    if (op->call_op.num_vars < 3) return tagged_.packNull();

    Value* fg = tagged_.ensureTagged(codegenAST(&op->call_op.variables[0]));
    Value* factor_idx = tagged_.ensureTagged(codegenAST(&op->call_op.variables[1]));
    Value* new_cpt = tagged_.ensureTagged(codegenAST(&op->call_op.variables[2]));

    Value* fg_a = allocaAndStore(fg, "fguc_fg");
    Value* fi_a = allocaAndStore(factor_idx, "fguc_fi");
    Value* cpt_a = allocaAndStore(new_cpt, "fguc_cpt");
    Value* result_a = allocaResult("fguc_res");

    FunctionType* fn_type = FunctionType::get(ctx_.voidType(),
        {ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType()}, false);
    Function* func = getOrDeclareRuntimeFunc("eshkol_fg_update_cpt_tagged", fn_type);

    builder.CreateCall(func, {loadArenaPtr(), fg_a, fi_a, cpt_a, result_a});
    return loadResult(result_a, "fguc_result");
}

Value* LogicWorkspaceCodegen::codegenFGObserve(const eshkol_operations_t* op) {
    auto& builder = ctx_.builder();
    if (op->call_op.num_vars < 3) return tagged_.packNull();
    Value* fg = tagged_.ensureTagged(codegenAST(&op->call_op.variables[0]));
    Value* var_id = tagged_.ensureTagged(codegenAST(&op->call_op.variables[1]));
    Value* obs_state = tagged_.ensureTagged(codegenAST(&op->call_op.variables[2]));
    Value* fg_a = allocaAndStore(fg, "fgo_fg");
    Value* vi_a = allocaAndStore(var_id, "fgo_vi");
    Value* os_a = allocaAndStore(obs_state, "fgo_os");
    Value* result_a = allocaResult("fgo_res");
    FunctionType* fn_type = FunctionType::get(ctx_.voidType(),
        {ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType()}, false);
    Function* func = getOrDeclareRuntimeFunc("eshkol_fg_observe_tagged", fn_type);
    builder.CreateCall(func, {loadArenaPtr(), fg_a, vi_a, os_a, result_a});
    return loadResult(result_a, "fgo_result");
}

// ---------------------------------------------------------------------------
// Global workspace
// ---------------------------------------------------------------------------

Value* LogicWorkspaceCodegen::codegenMakeWorkspace(const eshkol_operations_t* op) {
    auto& builder = ctx_.builder();
    if (op->call_op.num_vars < 2) return tagged_.packNull();

    Value* dim = tagged_.ensureTagged(codegenAST(&op->call_op.variables[0]));
    Value* max_mod = tagged_.ensureTagged(codegenAST(&op->call_op.variables[1]));

    Value* dim_a = allocaAndStore(dim, "ws_dim");
    Value* mm_a = allocaAndStore(max_mod, "ws_mm");
    Value* result_a = allocaResult("ws_res");

    FunctionType* fn_type = FunctionType::get(ctx_.voidType(),
        {ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType()}, false);
    Function* func = getOrDeclareRuntimeFunc("eshkol_make_workspace_tagged", fn_type);

    builder.CreateCall(func, {loadArenaPtr(), dim_a, mm_a, result_a});
    return loadResult(result_a, "ws_result");
}

Value* LogicWorkspaceCodegen::codegenWSRegister(const eshkol_operations_t* op) {
    auto& builder = ctx_.builder();
    if (op->call_op.num_vars < 3) return tagged_.packNull();

    Value* ws = tagged_.ensureTagged(codegenAST(&op->call_op.variables[0]));
    Value* name = tagged_.ensureTagged(codegenAST(&op->call_op.variables[1]));
    Value* process_fn = tagged_.ensureTagged(codegenAST(&op->call_op.variables[2]));

    Value* ws_a = allocaAndStore(ws, "wsr_ws");
    Value* name_a = allocaAndStore(name, "wsr_name");
    Value* fn_a = allocaAndStore(process_fn, "wsr_fn");

    FunctionType* fn_type = FunctionType::get(ctx_.voidType(),
        {ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType()}, false);
    Function* func = getOrDeclareRuntimeFunc("eshkol_ws_register_tagged", fn_type);

    builder.CreateCall(func, {loadArenaPtr(), ws_a, name_a, fn_a});
    return tagged_.packNull();
}

Value* LogicWorkspaceCodegen::codegenWSStep(const eshkol_operations_t* op) {
    auto& builder = ctx_.builder();
    auto& context = ctx_.context();
    if (op->call_op.num_vars < 1) return tagged_.packNull();

    Value* ws_tv = tagged_.ensureTagged(codegenAST(&op->call_op.variables[0]));

    Function* current_func = builder.GetInsertBlock()->getParent();
    Type* ptr_type = PointerType::getUnqual(context);
    Type* i32_type = Type::getInt32Ty(context);

    // Extract workspace raw pointer from tagged value data field
    Value* ws_ptr_i64 = builder.CreateExtractValue(ws_tv, {4}, "ws_data");
    Value* ws_ptr = builder.CreateIntToPtr(ws_ptr_i64, ptr_type, "ws_ptr");

    // Load num_modules (uint32_t at offset 0)
    Value* num_modules_ptr = builder.CreateGEP(ctx_.int8Type(), ws_ptr,
        ConstantInt::get(ctx_.int64Type(), 0));
    Value* num_modules = builder.CreateLoad(i32_type, num_modules_ptr, "num_modules");

    // Early exit: no modules → return workspace unchanged
    BasicBlock* has_modules_bb = BasicBlock::Create(context, "ws_has_modules", current_func);
    BasicBlock* no_modules_bb = BasicBlock::Create(context, "ws_no_modules", current_func);
    Value* has_modules = builder.CreateICmpUGT(num_modules,
        ConstantInt::get(i32_type, 0));
    builder.CreateCondBr(has_modules, has_modules_bb, no_modules_bb);

    builder.SetInsertPoint(no_modules_bb);
    // Will merge below

    builder.SetInsertPoint(has_modules_bb);

    // Load dim (uint32_t at offset 8)
    Value* dim_ptr = builder.CreateGEP(ctx_.int8Type(), ws_ptr,
        ConstantInt::get(ctx_.int64Type(), 8));
    Value* dim = builder.CreateLoad(i32_type, dim_ptr, "ws_dim");

    // Load content pointer (double* at offset 16)
    Value* content_pptr = builder.CreateGEP(ctx_.int8Type(), ws_ptr,
        ConstantInt::get(ctx_.int64Type(), 16));
    Value* content_ptr = builder.CreateLoad(ptr_type, content_pptr, "content_ptr");

    // Call eshkol_ws_make_content_tensor to wrap content as tensor tagged value
    Value* content_tv_a = allocaResult("content_tv");
    {
        FunctionType* fn_type = FunctionType::get(ctx_.voidType(),
            {ptr_type, ptr_type, i32_type, ptr_type}, false);
        Function* func = getOrDeclareRuntimeFunc("eshkol_ws_make_content_tensor", fn_type);
        builder.CreateCall(func, {loadArenaPtr(), content_ptr, dim, content_tv_a});
    }
    Value* content_tv = loadResult(content_tv_a, "content_tv");

    // Allocate results array on stack (max 16 modules)
    const uint32_t MAX_WS_MODULES = 16;
    Value* results_arr = builder.CreateAlloca(ctx_.taggedValueType(),
        ConstantInt::get(i32_type, MAX_WS_MODULES), "ws_results");

    // Modules start at ws_ptr + sizeof(eshkol_workspace_t) = ws_ptr + 24
    Value* modules_base = builder.CreateGEP(ctx_.int8Type(), ws_ptr,
        ConstantInt::get(ctx_.int64Type(), 24), "modules_base");

    // Loop: call each module's closure
    Value* i_alloca = builder.CreateAlloca(i32_type, nullptr, "ws_i");
    builder.CreateStore(ConstantInt::get(i32_type, 0), i_alloca);

    BasicBlock* loop_cond_bb = BasicBlock::Create(context, "ws_loop_cond", current_func);
    BasicBlock* loop_body_bb = BasicBlock::Create(context, "ws_loop_body", current_func);
    BasicBlock* loop_exit_bb = BasicBlock::Create(context, "ws_loop_exit", current_func);

    builder.CreateBr(loop_cond_bb);

    // Loop condition: i < num_modules
    builder.SetInsertPoint(loop_cond_bb);
    Value* i_val = builder.CreateLoad(i32_type, i_alloca, "ws_i_val");
    Value* cond = builder.CreateICmpULT(i_val, num_modules, "ws_loop_cond");
    builder.CreateCondBr(cond, loop_body_bb, loop_exit_bb);

    // Loop body
    builder.SetInsertPoint(loop_body_bb);

    // Each workspace_module_t is 32 bytes: name(8) + process_fn(16) + salience(8)
    // process_fn is at offset 8 within each module
    Value* i_ext = builder.CreateZExt(i_val, ctx_.int64Type(), "i_ext");
    Value* module_offset = builder.CreateMul(i_ext,
        ConstantInt::get(ctx_.int64Type(), 32), "mod_offset");
    Value* fn_offset = builder.CreateAdd(module_offset,
        ConstantInt::get(ctx_.int64Type(), 8), "fn_offset");
    Value* fn_ptr = builder.CreateGEP(ctx_.int8Type(), modules_base, fn_offset, "fn_gep");

    // Load the process_fn tagged value (16 bytes)
    Value* fn_tv = builder.CreateLoad(ctx_.taggedValueType(), fn_ptr, "fn_tv");

    // Call the closure with content_tv as the argument
    std::vector<Value*> closure_args = {content_tv};
    Value* result_tv = codegenClosureCall(fn_tv, closure_args, "ws-step-module");

    // Store result in results array
    Value* result_slot = builder.CreateGEP(ctx_.taggedValueType(), results_arr,
        i_val, "result_slot");
    builder.CreateStore(result_tv, result_slot);

    // Increment counter
    Value* i_next = builder.CreateAdd(i_val, ConstantInt::get(i32_type, 1), "i_next");
    builder.CreateStore(i_next, i_alloca);
    builder.CreateBr(loop_cond_bb);

    // Loop exit: call finalize
    builder.SetInsertPoint(loop_exit_bb);
    {
        FunctionType* fn_type = FunctionType::get(ctx_.voidType(),
            {ptr_type, ptr_type, i32_type}, false);
        Function* func = getOrDeclareRuntimeFunc("eshkol_ws_step_finalize", fn_type);
        builder.CreateCall(func, {ws_ptr, results_arr, num_modules});
    }

    // Merge: both paths return ws_tv
    BasicBlock* merge_bb = BasicBlock::Create(context, "ws_step_merge", current_func);
    builder.CreateBr(merge_bb);

    // Wire up the no-modules path
    builder.SetInsertPoint(no_modules_bb);
    builder.CreateBr(merge_bb);

    builder.SetInsertPoint(merge_bb);
    return ws_tv;
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
