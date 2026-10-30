// Copyright (C) tsotchke
// SPDX-License-Identifier: MIT

#include "jit_parallel_init.h"

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#include <iostream>
#include <memory>
#include <string>

namespace {
bool expect(bool condition, const char* message) {
    if (!condition) std::cerr << "FAIL: " << message << '\n';
    return condition;
}

bool exercise_module(llvm::orc::LLJIT& jit, unsigned id, bool remove_after) {
    auto context = std::make_unique<llvm::LLVMContext>();
    const std::string suffix = std::to_string(id);
    auto module = std::make_unique<llvm::Module>("worker-init-test-" + suffix, *context);
    module->setDataLayout(jit.getDataLayout());
    module->setTargetTriple(jit.getTargetTriple());
    const std::string result_name = "worker_result_" + suffix;
    auto* i32 = llvm::Type::getInt32Ty(*context);
    auto* result = new llvm::GlobalVariable(
        *module, i32, false, llvm::GlobalValue::ExternalLinkage,
        llvm::ConstantInt::get(i32, 0), result_name);
    auto* worker = llvm::Function::Create(
        llvm::FunctionType::get(i32, false),
        llvm::GlobalValue::InternalLinkage, "private_worker", *module);
    llvm::IRBuilder<> worker_builder(
        llvm::BasicBlock::Create(*context, "entry", worker));
    worker_builder.CreateRet(llvm::ConstantInt::get(i32, id));
    auto* initializer = llvm::Function::Create(
        llvm::FunctionType::get(llvm::Type::getVoidTy(*context), false),
        llvm::GlobalValue::InternalLinkage,
        "__eshkol_init_parallel_workers", *module);
    llvm::IRBuilder<> init_builder(
        llvm::BasicBlock::Create(*context, "entry", initializer));
    init_builder.CreateStore(init_builder.CreateCall(worker), result);
    init_builder.CreateRetVoid();
    llvm::appendToGlobalCtors(*module, initializer, 65535);

    const std::string name = eshkol::expose_jit_parallel_initializer(
        *module, "__repl_worker_init_test_" + suffix);
    bool ok = expect(!name.empty(), "a defined initializer is exposed");
    ok &= expect(initializer->hasExternalLinkage(), "initializer is exported");
    ok &= expect(worker->hasLocalLinkage(), "worker retains AOT-safe linkage");
    ok &= expect(module->getNamedGlobal("llvm.global_ctors") != nullptr,
                 "existing constructor registration is retained");
    ok &= expect(!llvm::verifyModule(*module, &llvm::errs()),
                 "renaming leaves valid IR and constructor references");
    if (!ok) return false;

    auto tracker = jit.getMainJITDylib().createResourceTracker();
    if (auto error = jit.addIRModule(tracker, llvm::orc::ThreadSafeModule(
            std::move(module), std::move(context)))) {
        std::cerr << llvm::toString(std::move(error)) << '\n';
        return false;
    }
    auto address = jit.lookup(name);
    if (!address) {
        std::cerr << llvm::toString(address.takeError()) << '\n';
        return false;
    }
    reinterpret_cast<void (*)()>(address->getValue())();
    auto result_address = jit.lookup(result_name);
    if (!result_address) {
        std::cerr << llvm::toString(result_address.takeError()) << '\n';
        return false;
    }
    ok &= expect(*reinterpret_cast<unsigned*>(result_address->getValue()) == id,
                 "lookup calls this module's initializer and private worker");
    if (remove_after) {
        if (auto error = tracker->remove()) {
            std::cerr << llvm::toString(std::move(error)) << '\n';
            return false;
        }
    }
    return ok;
}
}  // namespace

int main() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::LLVMContext context;
    llvm::Module absent("absent", context);
    bool ok = expect(eshkol::expose_jit_parallel_initializer(absent, "unused").empty(),
                     "missing initializer is not synthesized");
    auto* declaration = llvm::Function::Create(
        llvm::FunctionType::get(llvm::Type::getVoidTy(context), false),
        llvm::GlobalValue::ExternalLinkage,
        "__eshkol_init_parallel_workers", absent);
    ok &= expect(eshkol::expose_jit_parallel_initializer(absent, "unused").empty(),
                 "a declaration is not mistaken for a registration body");
    ok &= expect(declaration->getName() == "__eshkol_init_parallel_workers",
                 "an external declaration remains unchanged");

    auto jit = llvm::orc::LLJITBuilder().create();
    if (!jit) {
        std::cerr << llvm::toString(jit.takeError()) << '\n';
        return 1;
    }
    // Same local helper names in live modules must not collide; a later
    // initializer must also remain callable after an earlier module unloads.
    ok &= exercise_module(**jit, 11, false);
    ok &= exercise_module(**jit, 22, true);
    ok &= exercise_module(**jit, 33, false);
    if (!ok) return 1;
    std::cout << "PASS: per-module JIT worker initialization\n";
    return 0;
}
