//
// Copyright (C) tsotchke
//
// SPDX-License-Identifier: MIT
//

#include "jit_target_config.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>

#include <iostream>
#include <memory>
#include <optional>
#include <string>

namespace {

bool expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        return false;
    }
    return true;
}

std::unique_ptr<llvm::TargetMachine> make_windows_arm64_target_machine(
    const llvm::Triple& triple
) {
    std::string error;
#if LLVM_VERSION_MAJOR >= 21
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(triple, error);
#else
    const llvm::Target* target =
        llvm::TargetRegistry::lookupTarget(triple.str(), error);
#endif
    if (!target) {
        std::cerr << "FAIL: AArch64 target lookup failed: " << error << '\n';
        return nullptr;
    }

    llvm::TargetOptions options;
    options.FunctionSections = true;
    options.DataSections = true;
#if LLVM_VERSION_MAJOR >= 21
    return std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
        triple, "generic", "", options, llvm::Reloc::PIC_,
        llvm::CodeModel::Small, llvm::CodeGenOptLevel::None));
#else
    return std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
        triple.str(), "generic", "", options, llvm::Reloc::PIC_,
        llvm::CodeModel::Small, llvm::CodeGenOpt::None));
#endif
}

}  // namespace

int main() {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();

    bool ok = true;
    llvm::LLVMContext context;
    const llvm::Triple windows_arm64("aarch64-pc-windows-msvc");

    llvm::Module module("windows-arm64-jit-contract", context);
    module.setTargetTriple(windows_arm64);
    auto* i64 = llvm::Type::getInt64Ty(context);
    auto* external_data = new llvm::GlobalVariable(
        module, i64, false, llvm::GlobalValue::ExternalLinkage, nullptr,
        "external_data");
    auto* already_imported = new llvm::GlobalVariable(
        module, i64, false, llvm::GlobalValue::ExternalLinkage, nullptr,
        "already_imported");
    already_imported->setDLLStorageClass(
        llvm::GlobalValue::DLLImportStorageClass);
    auto* defined_data = new llvm::GlobalVariable(
        module, i64, false, llvm::GlobalValue::ExternalLinkage,
        llvm::ConstantInt::get(i64, 7), "defined_data");

    const std::size_t changed =
        eshkol::prepare_jit_module_for_target(module, windows_arm64);
    ok &= expect(changed == 1,
                 "only an unqualified external data declaration is changed");
    ok &= expect(external_data->getDLLStorageClass() ==
                     llvm::GlobalValue::DLLImportStorageClass,
                 "Windows ARM64 external data is lowered through __imp_");
    ok &= expect(already_imported->getDLLStorageClass() ==
                     llvm::GlobalValue::DLLImportStorageClass,
                 "existing dllimport storage is preserved");
    ok &= expect(defined_data->getDLLStorageClass() ==
                     llvm::GlobalValue::DefaultStorageClass,
                 "JIT-owned data definitions remain direct");
    ok &= expect(eshkol::prepare_jit_module_for_target(module, windows_arm64) == 0,
                 "target preparation is idempotent");

    llvm::Module linux_module("linux-arm64-jit-contract", context);
    const llvm::Triple linux_arm64("aarch64-unknown-linux-gnu");
    auto* linux_external = new llvm::GlobalVariable(
        linux_module, i64, false, llvm::GlobalValue::ExternalLinkage, nullptr,
        "linux_external_data");
    ok &= expect(eshkol::prepare_jit_module_for_target(
                     linux_module, linux_arm64) == 0,
                 "non-COFF targets are unchanged");
    ok &= expect(linux_external->getDLLStorageClass() ==
                     llvm::GlobalValue::DefaultStorageClass,
                 "Linux external data keeps its native relocation contract");

    // Force both contracts that matter in the emitted object: an external data
    // reference must use __imp_, and an 8 KiB frame must retain valid Small-
    // model SEH with a direct __chkstk branch.
    auto* load_type = llvm::FunctionType::get(i64, false);
    auto* load_function = llvm::Function::Create(
        load_type, llvm::GlobalValue::ExternalLinkage, "load_external", module);
    llvm::IRBuilder<> load_builder(
        llvm::BasicBlock::Create(context, "entry", load_function));
    load_builder.CreateRet(load_builder.CreateLoad(i64, external_data, true));

    auto* consume_type = llvm::FunctionType::get(
        llvm::Type::getVoidTy(context), {llvm::PointerType::getUnqual(context)},
        false);
    auto* consume = llvm::Function::Create(
        consume_type, llvm::GlobalValue::ExternalLinkage, "consume", module);
    auto* frame_function = llvm::Function::Create(
        llvm::FunctionType::get(llvm::Type::getVoidTy(context), false),
        llvm::GlobalValue::ExternalLinkage, "large_frame", module);
    llvm::IRBuilder<> frame_builder(
        llvm::BasicBlock::Create(context, "entry", frame_function));
    auto* buffer_type = llvm::ArrayType::get(llvm::Type::getInt8Ty(context), 8192);
    llvm::Value* buffer = frame_builder.CreateAlloca(buffer_type);
    llvm::Value* first_byte = frame_builder.CreateInBoundsGEP(
        buffer_type, buffer,
        {frame_builder.getInt64(0), frame_builder.getInt64(0)});
    frame_builder.CreateCall(consume, {first_byte});
    frame_builder.CreateRetVoid();

    auto target_machine = make_windows_arm64_target_machine(windows_arm64);
    ok &= expect(static_cast<bool>(target_machine),
                 "Windows ARM64 Small target machine is available");
    if (target_machine) {
        module.setDataLayout(target_machine->createDataLayout());
        llvm::SmallVector<char, 0> assembly_buffer;
        llvm::raw_svector_ostream assembly_stream(assembly_buffer);
        llvm::legacy::PassManager passes;
        const bool unsupported = target_machine->addPassesToEmitFile(
            passes, assembly_stream, nullptr, llvm::CodeGenFileType::AssemblyFile);
        ok &= expect(!unsupported, "target can emit Windows ARM64 assembly");
        if (!unsupported) {
            passes.run(module);
            const std::string assembly(assembly_buffer.begin(), assembly_buffer.end());
            ok &= expect(assembly.find("__imp_external_data") != std::string::npos,
                         "external data is emitted through a COFF import cell");
            ok &= expect(assembly.find("__chkstk") != std::string::npos,
                         "large frames retain Windows stack probing");
            ok &= expect(assembly.find(".seh_endprologue") != std::string::npos,
                         "large frames retain Windows unwind metadata");
        }
    }

    if (!ok) {
        return 1;
    }
    std::cout << "PASS: Windows ARM64 JIT data reach and SEH target contract\n";
    return 0;
}
