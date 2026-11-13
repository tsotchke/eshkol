// Copyright (C) tsotchke
// SPDX-License-Identifier: MIT

#include "jit_windows_x64_memory_manager.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <setjmpex.h>
#include <llvm/Config/llvm-config.h>

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/Mangling.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>

#include <iostream>
#include <memory>

// The generated IR uses the CRT entry point directly, including its hidden
// frame argument, rather than the C header's compiler-intrinsic macro.
extern "C" int __cdecl _setjmpex(jmp_buf, void*);

namespace {
bool expect(bool condition, const char* message) {
    if (!condition) std::cerr << "FAIL: " << message << '\n';
    return condition;
}

extern "C" __declspec(noinline) void unwind_test_jump(void* buffer) {
    longjmp(*static_cast<jmp_buf*>(buffer), 73);
}

// Deliberate failure control: normal code generation and relocation, with
// the one required registration step omitted. Never execute its longjmp.
class UnregisteredMemoryManager : public llvm::SectionMemoryManager {
    void registerEHFrames(uint8_t*, uint64_t, size_t) override {}
    void deregisterEHFrames() override {}
};

bool exercise(bool registered) {
    llvm::orc::LLJITBuilder builder;
    auto machine = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!machine) {
        std::cerr << llvm::toString(machine.takeError()) << '\n';
        return false;
    }
    // Match Eshkol's Windows x64 JIT object-generation contract.
    machine->setCodeModel(llvm::CodeModel::Large);
    machine->setRelocationModel(llvm::Reloc::PIC_);
    machine->setCodeGenOptLevel(llvm::CodeGenOptLevel::None);
    builder.setJITTargetMachineBuilder(std::move(*machine));
    auto create_layer = [registered](llvm::orc::ExecutionSession& session)
        -> llvm::Expected<std::unique_ptr<llvm::orc::ObjectLayer>> {
        auto layer = std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(
            session, [registered](const llvm::MemoryBuffer&)
                -> std::unique_ptr<llvm::RuntimeDyld::MemoryManager> {
                if (registered)
                    return std::make_unique<eshkol::WindowsX64SectionMemoryManager>();
                return std::make_unique<UnregisteredMemoryManager>();
            });
        layer->setOverrideObjectFlagsWithResponsibilityFlags(true);
        layer->setAutoClaimResponsibilityForObjectSymbols(true);
        return std::unique_ptr<llvm::orc::ObjectLayer>(std::move(layer));
    };
#if LLVM_VERSION_MAJOR >= 24
    builder.setObjectLinkingLayerCreator(
        [create_layer](llvm::orc::ExecutionSession& session,
                       llvm::jitlink::JITLinkMemoryManager&)
            -> llvm::Expected<std::unique_ptr<llvm::orc::ObjectLayer>> {
            return create_layer(session);
        });
#else
    builder.setObjectLinkingLayerCreator(create_layer);
#endif
    auto created = builder.create();
    if (!created) {
        std::cerr << llvm::toString(created.takeError()) << '\n';
        return false;
    }
    auto& jit = **created;
    llvm::orc::MangleAndInterner mangle(jit.getExecutionSession(), jit.getDataLayout());
    llvm::orc::SymbolMap symbols;
    symbols[mangle("unwind_test_jump")] = llvm::orc::ExecutorSymbolDef(
        llvm::orc::ExecutorAddr::fromPtr(&unwind_test_jump), llvm::JITSymbolFlags::Exported);
    symbols[mangle("_setjmpex")] = llvm::orc::ExecutorSymbolDef(
        llvm::orc::ExecutorAddr::fromPtr(&_setjmpex), llvm::JITSymbolFlags::Exported);
    if (auto error = jit.getMainJITDylib().define(llvm::orc::absoluteSymbols(symbols))) {
        std::cerr << llvm::toString(std::move(error)) << '\n';
        return false;
    }

    auto context = std::make_unique<llvm::LLVMContext>();
    auto module = std::make_unique<llvm::Module>("windows-unwind-test", *context);
    module->setDataLayout(jit.getDataLayout());
    module->setTargetTriple(jit.getTargetTriple());
    llvm::IRBuilder<> ir(*context);
    auto* pointer = ir.getPtrTy();
    auto* function = llvm::Function::Create(
        llvm::FunctionType::get(ir.getInt32Ty(), {pointer}, false),
        llvm::GlobalValue::ExternalLinkage, "setjmp_in_jit", *module);
    auto* jump = llvm::Function::Create(
        llvm::FunctionType::get(ir.getVoidTy(), {pointer}, false),
        llvm::GlobalValue::ExternalLinkage, "unwind_test_jump", *module);
    auto* save = llvm::Function::Create(
        llvm::FunctionType::get(ir.getInt32Ty(), {pointer, pointer}, false),
        llvm::GlobalValue::ExternalLinkage, "_setjmpex", *module);
    save->addFnAttr(llvm::Attribute::ReturnsTwice);
    ir.SetInsertPoint(llvm::BasicBlock::Create(*context, "entry", function));
    auto* frame_address = llvm::Intrinsic::getOrInsertDeclaration(
        module.get(), llvm::Intrinsic::frameaddress, {pointer});
    auto* frame = ir.CreateCall(frame_address, {ir.getInt32(0)});
    auto* result = ir.CreateCall(save, {function->getArg(0), frame});
    auto* first = llvm::BasicBlock::Create(*context, "first", function);
    auto* returned = llvm::BasicBlock::Create(*context, "returned", function);
    ir.CreateCondBr(ir.CreateICmpEQ(result, ir.getInt32(0)), first, returned);
    ir.SetInsertPoint(first);
    ir.CreateCall(jump, {function->getArg(0)});
    ir.CreateUnreachable();
    ir.SetInsertPoint(returned);
    ir.CreateRet(result);

    auto tracker = jit.getMainJITDylib().createResourceTracker();
    if (auto error = jit.addIRModule(tracker, llvm::orc::ThreadSafeModule(
            std::move(module), std::move(context)))) {
        std::cerr << llvm::toString(std::move(error)) << '\n';
        return false;
    }
    auto address = jit.lookup("setjmp_in_jit");
    if (!address) {
        std::cerr << llvm::toString(address.takeError()) << '\n';
        return false;
    }
    DWORD64 image_base = 0;
    auto* table = RtlLookupFunctionEntry(address->getValue(), &image_base, nullptr);
    bool ok = expect((table != nullptr) == registered,
                     "Windows sees exactly the registered JIT unwind metadata");
    if (registered && table) {
        jmp_buf buffer;
        auto invoke = reinterpret_cast<int (*)(void*)>(address->getValue());
        ok &= expect(invoke(&buffer) == 73,
                     "native longjmp returns to setjmp inside the JIT frame");
    }
    const auto old_address = address->getValue();
    if (auto error = tracker->remove()) {
        std::cerr << llvm::toString(std::move(error)) << '\n';
        return false;
    }
    image_base = 0;
    ok &= expect(RtlLookupFunctionEntry(old_address, &image_base, nullptr) == nullptr,
                 "module unload removes Windows unwind metadata");
    return ok;
}
}  // namespace

int main() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    if (!exercise(false) || !exercise(true) || !exercise(true)) return 1;
    std::cout << "PASS: Windows x64 JIT unwind registration and longjmp\n";
    return 0;
}
