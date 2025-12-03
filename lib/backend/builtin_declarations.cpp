/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * BuiltinDeclarations implementation - External runtime function declarations
 */

#include <eshkol/backend/builtin_declarations.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/logger.h>
#include <vector>

namespace eshkol {

BuiltinDeclarations::BuiltinDeclarations(CodegenContext& ctx)
    : ctx_(ctx)
    , deep_equal_func_(nullptr)
    , display_value_func_(nullptr)
    , lambda_registry_init_func_(nullptr)
    , lambda_registry_add_func_(nullptr)
    , lambda_registry_lookup_func_(nullptr) {

    declareDeepEqual();
    declareDisplayValue();
    declareLambdaRegistry();

    // Update context with function pointers
    ctx_.setDeepEqualFunc(deep_equal_func_);
    ctx_.setDisplayValueFunc(display_value_func_);
    ctx_.setLambdaRegistryInitFunc(lambda_registry_init_func_);
    ctx_.setLambdaRegistryAddFunc(lambda_registry_add_func_);
    ctx_.setLambdaRegistryLookupFunc(lambda_registry_lookup_func_);

    eshkol_debug("BuiltinDeclarations: declared %d runtime functions",
                 5);  // deep_equal, display_value, registry_init, registry_add, registry_lookup
}

void BuiltinDeclarations::declareDeepEqual() {
    // eshkol_deep_equal: bool eshkol_deep_equal(const eshkol_tagged_value_t* val1,
    //                                           const eshkol_tagged_value_t* val2)
    std::vector<llvm::Type*> args;
    args.push_back(ctx_.ptrType()); // const eshkol_tagged_value_t* val1
    args.push_back(ctx_.ptrType()); // const eshkol_tagged_value_t* val2

    llvm::FunctionType* func_type = llvm::FunctionType::get(
        ctx_.int1Type(), // return bool
        args,
        false  // not variadic
    );

    deep_equal_func_ = llvm::Function::Create(
        func_type,
        llvm::Function::ExternalLinkage,
        "eshkol_deep_equal",
        &ctx_.module()
    );

    // Register in context function table for backward compatibility
    ctx_.defineFunction("eshkol_deep_equal", deep_equal_func_);
}

void BuiltinDeclarations::declareDisplayValue() {
    // eshkol_display_value: void eshkol_display_value(const eshkol_tagged_value_t* value)
    std::vector<llvm::Type*> args;
    args.push_back(ctx_.ptrType()); // const eshkol_tagged_value_t* value

    llvm::FunctionType* func_type = llvm::FunctionType::get(
        ctx_.voidType(),
        args,
        false
    );

    display_value_func_ = llvm::Function::Create(
        func_type,
        llvm::Function::ExternalLinkage,
        "eshkol_display_value",
        &ctx_.module()
    );

    ctx_.defineFunction("eshkol_display_value", display_value_func_);
}

void BuiltinDeclarations::declareLambdaRegistry() {
    // eshkol_lambda_registry_init: void eshkol_lambda_registry_init(void)
    llvm::FunctionType* init_type = llvm::FunctionType::get(
        ctx_.voidType(),
        {},  // no args
        false
    );

    lambda_registry_init_func_ = llvm::Function::Create(
        init_type,
        llvm::Function::ExternalLinkage,
        "eshkol_lambda_registry_init",
        &ctx_.module()
    );

    ctx_.defineFunction("eshkol_lambda_registry_init", lambda_registry_init_func_);

    // eshkol_lambda_registry_add: void eshkol_lambda_registry_add(uint64_t func_ptr,
    //                                                             uint64_t sexpr_ptr,
    //                                                             const char* name)
    std::vector<llvm::Type*> add_args;
    add_args.push_back(ctx_.int64Type()); // uint64_t func_ptr
    add_args.push_back(ctx_.int64Type()); // uint64_t sexpr_ptr
    add_args.push_back(ctx_.ptrType());   // const char* name

    llvm::FunctionType* add_type = llvm::FunctionType::get(
        ctx_.voidType(),
        add_args,
        false
    );

    lambda_registry_add_func_ = llvm::Function::Create(
        add_type,
        llvm::Function::ExternalLinkage,
        "eshkol_lambda_registry_add",
        &ctx_.module()
    );

    ctx_.defineFunction("eshkol_lambda_registry_add", lambda_registry_add_func_);

    // eshkol_lambda_registry_lookup: uint64_t eshkol_lambda_registry_lookup(uint64_t func_ptr)
    std::vector<llvm::Type*> lookup_args;
    lookup_args.push_back(ctx_.int64Type()); // uint64_t func_ptr

    llvm::FunctionType* lookup_type = llvm::FunctionType::get(
        ctx_.int64Type(), // return uint64_t (sexpr_ptr or 0)
        lookup_args,
        false
    );

    lambda_registry_lookup_func_ = llvm::Function::Create(
        lookup_type,
        llvm::Function::ExternalLinkage,
        "eshkol_lambda_registry_lookup",
        &ctx_.module()
    );

    ctx_.defineFunction("eshkol_lambda_registry_lookup", lambda_registry_lookup_func_);
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
