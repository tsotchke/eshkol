/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 */

#include <eshkol/backend/tensorcore_codegen.h>

#include <llvm/IR/Function.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

#include <iostream>
#include <string>

namespace {

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

}  // namespace

int main() {
    llvm::LLVMContext context;
    llvm::Module module("tensorcore-adapter-gate", context);
    std::string error;

    const std::size_t declared = eshkol::declareTensorcoreAdapterAbi(
        module, llvm::Type::getInt64Ty(context), &error);
    if (declared != 34) return fail("canonical declaration count: " + error);
    if (!eshkol::verifyTensorcoreAdapterModule(module, &error)) {
        return fail(error);
    }

    llvm::Function* init = module.getFunction("eshkol_tc_init");
    llvm::Function* gemm = module.getFunction("eshkol_tc_gemm");
    llvm::Function* attention = module.getFunction("eshkol_tc_attention_forward");
    llvm::Function* capability_status =
        module.getFunction("eshkol_tc_runtime_capabilities_status");
    llvm::Function* capability_validator =
        module.getFunction("eshkol_tc_validate_runtime_capabilities");
    if (!init || !gemm || !attention || !capability_status ||
        !capability_validator) {
        return fail("required adapter symbol is missing");
    }
    if (init->arg_size() != 0 || !init->getReturnType()->isPointerTy()) {
        return fail("init calling convention drifted");
    }
    if (gemm->arg_size() != 12 || !gemm->getReturnType()->isIntegerTy(32)) {
        return fail("GEMM calling convention drifted");
    }
    if (attention->arg_size() != 12 ||
        !attention->getReturnType()->isIntegerTy(32)) {
        return fail("attention calling convention drifted");
    }
    if (capability_status->arg_size() != 2 ||
        !capability_status->getReturnType()->isIntegerTy(32) ||
        capability_validator->arg_size() != 5 ||
        !capability_validator->getReturnType()->isIntegerTy(32)) {
        return fail("capability ABI calling convention drifted");
    }

    const std::size_t functions_before = module.size();
    if (eshkol::declareTensorcoreAdapterAbi(
            module, llvm::Type::getInt64Ty(context), &error) != 34 ||
        module.size() != functions_before) {
        return fail("adapter declaration is not idempotent");
    }

    llvm::Module conflict("tensorcore-adapter-conflict", context);
    llvm::Function::Create(
        llvm::FunctionType::get(llvm::Type::getVoidTy(context), {}, false),
        llvm::Function::ExternalLinkage,
        "eshkol_tc_init",
        conflict);
    if (eshkol::declareTensorcoreAdapterAbi(
            conflict, llvm::Type::getInt64Ty(context), &error) != 0 ||
        error.find("ABI conflict") == std::string::npos) {
        return fail("mixed adapter ABI was not rejected deterministically");
    }

    std::cout << "PASS: TensorCore canonical codegen + verifyModule\n";
    return 0;
}
