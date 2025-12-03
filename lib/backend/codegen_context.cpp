/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * CodegenContext implementation - Shared state management
 */

#include <eshkol/backend/codegen_context.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <llvm/IR/Constants.h>

namespace eshkol {

CodegenContext::CodegenContext(llvm::LLVMContext& llvm_ctx,
                               llvm::Module& llvm_mod,
                               llvm::IRBuilder<>& ir_builder,
                               TypeSystem& type_sys,
                               FunctionCache& func_cache,
                               MemoryCodegen& mem_codegen)
    : context_(llvm_ctx)
    , module_(llvm_mod)
    , builder_(ir_builder)
    , types_(type_sys)
    , funcs_(func_cache)
    , memory_(mem_codegen) {
    // Start with one local scope
    scope_stack_.emplace_back();
}

// === Symbol Table Management ===

void CodegenContext::pushScope() {
    scope_stack_.emplace_back();
}

void CodegenContext::popScope() {
    if (scope_stack_.size() > 1) {
        scope_stack_.pop_back();
    }
}

llvm::Value* CodegenContext::lookupSymbol(const std::string& name) {
    // Search from innermost to outermost local scope
    for (auto it = scope_stack_.rbegin(); it != scope_stack_.rend(); ++it) {
        auto found = it->find(name);
        if (found != it->end()) {
            return found->second;
        }
    }
    // Fall back to global symbols
    auto global_it = global_symbols_.find(name);
    if (global_it != global_symbols_.end()) {
        return global_it->second;
    }
    return nullptr;
}

void CodegenContext::defineSymbol(const std::string& name, llvm::Value* value) {
    if (!scope_stack_.empty()) {
        scope_stack_.back()[name] = value;
    }
}

void CodegenContext::defineGlobalSymbol(const std::string& name, llvm::Value* value) {
    global_symbols_[name] = value;
}

llvm::Value* CodegenContext::lookupGlobalSymbol(const std::string& name) {
    auto it = global_symbols_.find(name);
    return (it != global_symbols_.end()) ? it->second : nullptr;
}

// === Function Table ===

llvm::Function* CodegenContext::lookupFunction(const std::string& name) {
    auto it = function_table_.find(name);
    return (it != function_table_.end()) ? it->second : nullptr;
}

void CodegenContext::defineFunction(const std::string& name, llvm::Function* func) {
    function_table_[name] = func;
}

bool CodegenContext::hasFunction(const std::string& name) const {
    return function_table_.find(name) != function_table_.end();
}

// === Variadic Function Info ===

void CodegenContext::registerVariadicFunction(const std::string& name,
                                              uint64_t fixedParams,
                                              bool isVariadic) {
    variadic_info_[name] = {fixedParams, isVariadic};
}

std::pair<uint64_t, bool> CodegenContext::getVariadicInfo(const std::string& name) const {
    auto it = variadic_info_.find(name);
    return (it != variadic_info_.end()) ? it->second : std::make_pair(0ULL, false);
}

// === String Interning ===

llvm::GlobalVariable* CodegenContext::internString(const std::string& str) {
    // Check if already interned
    auto it = interned_strings_.find(str);
    if (it != interned_strings_.end()) {
        return it->second;
    }

    // Create a new global string
    llvm::Constant* str_constant = llvm::ConstantDataArray::getString(context_, str, true);
    llvm::GlobalVariable* global_str = new llvm::GlobalVariable(
        module_,
        str_constant->getType(),
        true,  // isConstant
        llvm::GlobalValue::PrivateLinkage,
        str_constant,
        ".str"
    );
    global_str->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);

    interned_strings_[str] = global_str;
    return global_str;
}

llvm::GlobalVariable* CodegenContext::lookupInternedString(const std::string& str) {
    auto it = interned_strings_.find(str);
    return (it != interned_strings_.end()) ? it->second : nullptr;
}

// === Nested Function Captures ===

void CodegenContext::setFunctionCaptures(const std::string& funcName,
                                         const std::vector<std::string>& captures) {
    function_captures_[funcName] = captures;
}

const std::vector<std::string>* CodegenContext::getFunctionCaptures(const std::string& funcName) const {
    auto it = function_captures_.find(funcName);
    return (it != function_captures_.end()) ? &it->second : nullptr;
}

// === Lambda Return Tracking ===

void CodegenContext::setFunctionReturnsLambda(const std::string& funcName,
                                              const std::string& lambdaName) {
    functions_returning_lambda_[funcName] = lambdaName;
}

std::string CodegenContext::getFunctionReturnsLambda(const std::string& funcName) const {
    auto it = functions_returning_lambda_.find(funcName);
    return (it != functions_returning_lambda_.end()) ? it->second : "";
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
