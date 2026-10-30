/* Copyright (C) tsotchke. SPDX-License-Identifier: MIT */
// Package a seed-compiled Eshkol module as a compiler/runtime object. The
// pointer thunk uses the module's actual tagged signature, never a guessed
// platform aggregate ABI. This tool performs no planning or source rewriting.
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Transforms/IPO/GlobalDCE.h>
#include <memory>
#include <string>

int main(int argc, char** argv) {
    if (argc != 5) {
        llvm::errs() << "usage: bootstrap_object INPUT.bc OUTPUT.o INIT_PREFIX ENTRY_OR_DASH\n";
        return 2;
    }
    llvm::LLVMContext context;
    llvm::SMDiagnostic diagnostic;
    auto module = llvm::parseIRFile(argv[1], diagnostic, context);
    if (!module) { diagnostic.print(argv[0], llvm::errs()); return 1; }
    for (auto& function : *module) {
        if (!function.isDeclaration() && function.getName().starts_with("__eshkol_lib_init"))
            function.setName(std::string(argv[3]) + function.getName().str());
        if (std::string(argv[4]) == "-" &&
            (function.getName() == "eshkol_incremental_state" || function.getName() == "eshkol_incremental_check"))
            function.setName("__eshkol_compiler_" + function.getName().drop_front(7).str());
    }
    if (std::string(argv[4]) != "-") {
        auto* entry = module->getFunction(argv[4]);
        auto* tagged = entry ? llvm::dyn_cast<llvm::StructType>(entry->getReturnType()) : nullptr;
        const auto& layout = module->getDataLayout();
        if (!entry || entry->isDeclaration() || entry->arg_size() != 1 || entry->isVarArg() ||
            !tagged || entry->getFunctionType()->getParamType(0) != tagged ||
            tagged->getNumElements() != 5 || layout.getTypeAllocSize(tagged) != 16 ||
            tagged->isPacked() || layout.getStructLayout(tagged)->getElementOffset(0) != 0 ||
            layout.getStructLayout(tagged)->getElementOffset(4) != 8 ||
            !tagged->getElementType(0)->isIntegerTy(8) ||
            !tagged->getElementType(1)->isIntegerTy(8) ||
            !tagged->getElementType(2)->isIntegerTy(16) ||
            !tagged->getElementType(3)->isIntegerTy(32) ||
            !tagged->getElementType(4)->isIntegerTy(64)) {
            llvm::errs() << "compiler pass has an unsupported tagged-value signature\n";
            return 1;
        }
        auto* pointer = llvm::PointerType::getUnqual(context);
        auto* type = llvm::FunctionType::get(llvm::Type::getVoidTy(context), {pointer, pointer}, false);
        if (module->getNamedValue("eshkol_incremental_plan_call")) {
            llvm::errs() << "compiler pass thunk symbol already exists\n"; return 1;
        }
        auto* thunk = llvm::Function::Create(type, llvm::GlobalValue::ExternalLinkage,
                                            "eshkol_incremental_plan_call", *module);
        llvm::IRBuilder<> builder(llvm::BasicBlock::Create(context, "entry", thunk));
        auto* argument = builder.CreateLoad(tagged, thunk->getArg(0));
        auto* result = builder.CreateCall(entry, {argument});
        builder.CreateStore(result, thunk->getArg(1));
        builder.CreateRetVoid();
    }
    // Only the narrow native gateway is link-visible. Module-local closure
    // thunks and builtin wrappers must not be interposed by a user's object.
    for (auto& function : *module) {
        if (function.isDeclaration()) continue;
        const bool exported = function.getName() == "eshkol_incremental_plan_call" ||
            function.getName() == "__eshkol_compiler_incremental_state" ||
            function.getName() == "__eshkol_compiler_incremental_check";
        function.setLinkage(exported ? llvm::GlobalValue::ExternalLinkage : llvm::GlobalValue::InternalLinkage);
        function.setComdat(nullptr);
    }
    for (auto& global : module->globals()) {
        if (!global.isDeclaration() && !global.getName().starts_with("llvm.")) {
            global.setLinkage(llvm::GlobalValue::InternalLinkage);
            global.setComdat(nullptr);
        }
    }
    llvm::ModuleAnalysisManager analyses;
    llvm::PassBuilder pass_builder;
    pass_builder.registerModuleAnalyses(analyses);
    llvm::ModulePassManager cleanup;
    cleanup.addPass(llvm::GlobalDCEPass());
    cleanup.run(*module, analyses);
    if (llvm::verifyModule(*module, &llvm::errs())) return 1;
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    std::string error;
    const auto triple = module->getTargetTriple();
    const auto* target = llvm::TargetRegistry::lookupTarget(triple, error);
    if (!target) { llvm::errs() << error << '\n'; return 1; }
    llvm::TargetOptions options;
    std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
        triple, "generic", "", options, llvm::Reloc::PIC_));
    if (!machine || machine->createDataLayout() != module->getDataLayout()) {
        llvm::errs() << "compiler pass target data layout does not match the native packager\n";
        return 1;
    }
    std::error_code ec;
    llvm::raw_fd_ostream output(argv[2], ec, llvm::sys::fs::OF_None);
    if (ec) { llvm::errs() << ec.message() << '\n'; return 1; }
    llvm::legacy::PassManager passes;
    if (machine->addPassesToEmitFile(passes, output, nullptr, llvm::CodeGenFileType::ObjectFile)) {
        llvm::errs() << "cannot emit native compiler pass object\n"; return 1;
    }
    passes.run(*module);
    output.flush();
    return output.has_error() ? 1 : 0;
}
