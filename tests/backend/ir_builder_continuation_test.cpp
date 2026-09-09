/* Copyright (C) tsotchke. SPDX-License-Identifier: MIT */
#include <eshkol/backend/ir_builder.h>
#include <eshkol/util/continuation_task.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <iostream>
#include <stdexcept>
#include <vector>

using Builder = eshkol::CodegenIRBuilder;

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

static ContinuationTask<unsigned> operand(unsigned value) {
    co_return value;
}

static ContinuationTask<unsigned> emit(Builder& builder, llvm::Type* type,
                                       llvm::Value* array, llvm::Function* callee,
                                       llvm::Function* no_args) {
    for (unsigned i = 0; i < 3; ++i) {
        unsigned index = co_await operand(i);
        // These factory calls return ConstantInt*. The emitter must own their
        // Value* conversion before constructing a borrowed singleton ArrayRef.
        auto* gep = builder.CreateGEP(type, array, llvm::ConstantInt::get(type, index));
        auto* inbounds = builder.CreateInBoundsGEP(
            type, array, llvm::ConstantInt::get(type, index));
        auto* call = builder.CreateCall(callee, llvm::ConstantInt::get(type, index));
        auto* typed_call = builder.CreateCall(
            callee->getFunctionType(), callee, llvm::ConstantInt::get(type, index));

        llvm::Value* expected = llvm::ConstantInt::get(type, index);
        require(llvm::cast<llvm::GetElementPtrInst>(gep)->getOperand(1) == expected,
                "GEP lost its suspended scalar operand");
        require(llvm::cast<llvm::GetElementPtrInst>(inbounds)->getOperand(1) == expected,
                "inbounds GEP lost its suspended scalar operand");
        require(call->getArgOperand(0) == expected && typed_call->getArgOperand(0) == expected,
                "call lost its suspended scalar operand");

        // Preserve the existing range, initializer-list and empty-argument API.
        std::vector<llvm::Value*> indices{expected};
        auto* range_gep = builder.CreateGEP(type, array, indices);
        auto* list_call = builder.CreateCall(callee, {expected});
        auto* empty_call = builder.CreateCall(no_args, {});
        require(llvm::cast<llvm::GetElementPtrInst>(range_gep)->getOperand(1) == expected,
                "range GEP changed");
        require(list_call->getArgOperand(0) == expected && empty_call->arg_empty(),
                "range or empty call changed");
    }
    co_return 3;
}

int main() {
    llvm::LLVMContext context;
    llvm::Module module("continuation-operands", context);
    Builder builder(context);
    auto* type = llvm::Type::getInt64Ty(context);
    auto* callee = llvm::Function::Create(llvm::FunctionType::get(type, {type}, false),
        llvm::Function::ExternalLinkage, "callee", module);
    auto* no_args = llvm::Function::Create(llvm::FunctionType::get(type, false),
        llvm::Function::ExternalLinkage, "no_args", module);
    auto* function = llvm::Function::Create(llvm::FunctionType::get(type, false),
        llvm::Function::ExternalLinkage, "test", module);
    builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", function));
    auto* array = builder.CreateAlloca(type, llvm::ConstantInt::get(type, 3));
    require(emit(builder, type, array, callee, no_args).run() == 3,
            "continuation completion changed");
    builder.CreateRet(llvm::ConstantInt::get(type, 0));
    require(!llvm::verifyModule(module, &llvm::errs()), "invalid emitted module");
    std::cout << "PASS: continuation operand identity and LLVM verification\n";
}
