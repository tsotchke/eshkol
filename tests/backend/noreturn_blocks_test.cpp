/* Copyright (C) tsotchke. SPDX-License-Identifier: MIT */
// The block boundary after a no-return emission.
//
// continueAfterNoReturn() hands the caller of an unconditional no-return
// emission a fresh, predecessor-less block to keep writing in, so the module
// verifies. blocksPastUnreachable() names every block an emitter left with
// instructions behind its first `unreachable`, and changes nothing.
#include <eshkol/backend/ir_builder.h>
#include <eshkol/backend/noreturn_blocks.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include <string>

using Builder = eshkol::CodegenIRBuilder;

static int failures = 0;

static void check(bool condition, const char* what) {
    if (!condition) {
        std::cout << "FAIL: " << what << "\n";
        ++failures;
    }
}

static bool verifies(llvm::Module& module) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    return !llvm::verifyModule(module, &stream);
}

static unsigned instructionCount(const llvm::Module& module) {
    unsigned count = 0;
    for (const llvm::Function& function : module)
        for (const llvm::BasicBlock& block : function) count += block.size();
    return count;
}

struct Fixture {
    llvm::LLVMContext context;
    llvm::Module module{"noreturn-blocks", context};
    Builder builder{context};
    llvm::Type* i64 = llvm::Type::getInt64Ty(context);
    llvm::Function* raise = llvm::Function::Create(
        llvm::FunctionType::get(llvm::Type::getVoidTy(context), false),
        llvm::Function::ExternalLinkage, "raise", module);
    llvm::Function* function = llvm::Function::Create(
        llvm::FunctionType::get(i64, {i64}, false),
        llvm::Function::ExternalLinkage, "subject", module);
    llvm::BasicBlock* entry = llvm::BasicBlock::Create(context, "entry", function);

    Fixture() {
        raise->setDoesNotReturn();
        builder.SetInsertPoint(entry);
    }
};

// An unconditional no-return emission, then a caller that keeps emitting.
static void continuation_keeps_the_module_well_formed() {
    Fixture f;
    f.builder.CreateCall(f.raise, {});
    f.builder.CreateUnreachable();
    llvm::BasicBlock* continuation = eshkol::continueAfterNoReturn(f.builder, "after_raise");

    check(continuation != f.entry, "the continuation is a new block");
    check(f.builder.GetInsertBlock() == continuation, "the builder is seated on the continuation");
    check(continuation->getParent() == f.function, "the continuation belongs to the current function");
    check(llvm::pred_empty(continuation), "the continuation has no predecessors");
    check(continuation->empty(), "the continuation starts empty");
    check(continuation->getName() == "after_raise", "the continuation carries the requested name");

    // The rest of the enclosing form, as a caller would lower it.
    llvm::Value* sum = f.builder.CreateAdd(f.function->getArg(0), llvm::ConstantInt::get(f.i64, 1));
    f.builder.CreateRet(sum);

    check(f.entry->size() == 2, "nothing was appended to the ended block");
    check(eshkol::blocksPastUnreachable(f.module).empty(), "no block continues past unreachable");
    check(verifies(f.module), "the module verifies");
}

// The caller has emitted the no-return call but not its terminator yet.
static void continuation_ends_an_open_block() {
    Fixture f;
    f.builder.CreateCall(f.raise, {});
    eshkol::continueAfterNoReturn(f.builder);
    check(llvm::isa<llvm::UnreachableInst>(f.entry->back()), "an open block is ended with unreachable");
    check(f.builder.GetInsertBlock()->getName().starts_with("post_noreturn_dead"),
          "the default continuation name is used");
    f.builder.CreateRet(f.function->getArg(0));
    check(verifies(f.module), "the module verifies");
}

// Two no-return emissions in a row each get their own continuation.
static void continuations_nest() {
    Fixture f;
    f.builder.CreateCall(f.raise, {});
    llvm::BasicBlock* first = eshkol::continueAfterNoReturn(f.builder);
    f.builder.CreateCall(f.raise, {});
    llvm::BasicBlock* second = eshkol::continueAfterNoReturn(f.builder);
    check(first != second, "each emission gets its own continuation");
    f.builder.CreateRet(f.function->getArg(0));
    check(eshkol::blocksPastUnreachable(f.module).empty(), "no block continues past unreachable");
    check(verifies(f.module), "the module verifies");
}

// An emitter that returns to its caller without the continuation.
static void check_names_a_block_that_continues_past_unreachable() {
    Fixture f;
    f.builder.CreateCall(f.raise, {});
    f.builder.CreateUnreachable();
    llvm::Value* sum = f.builder.CreateAdd(f.function->getArg(0), llvm::ConstantInt::get(f.i64, 1));
    f.builder.CreateRet(sum);

    const unsigned before = instructionCount(f.module);
    const auto found = eshkol::blocksPastUnreachable(f.module);
    check(found.size() == 1, "exactly one block is reported");
    if (found.size() == 1) {
        check(found[0].function == f.function, "the report names the function");
        check(found[0].block == f.entry, "the report names the block");
        check(found[0].trailing_instructions == 2, "the report counts the trailing instructions");
    }
    check(instructionCount(f.module) == before, "the check leaves the module unchanged");
    check(!verifies(f.module), "the verifier still rejects the module");
}

// A block that continues past `ret` is the verifier's to report, not this check's.
static void check_is_specific_to_unreachable() {
    Fixture f;
    f.builder.CreateRet(f.function->getArg(0));
    f.builder.CreateRet(f.function->getArg(0));
    check(eshkol::blocksPastUnreachable(f.module).empty(), "a block past ret is not reported here");
    check(!verifies(f.module), "the verifier rejects a block that continues past ret");
}

int main() {
    continuation_keeps_the_module_well_formed();
    continuation_ends_an_open_block();
    continuations_nest();
    check_names_a_block_that_continues_past_unreachable();
    check_is_specific_to_unreachable();
    if (failures) {
        std::cout << "FAIL: no-return block continuation (" << failures << " checks)\n";
        return 1;
    }
    std::cout << "PASS: no-return block continuation\n";
    return 0;
}
