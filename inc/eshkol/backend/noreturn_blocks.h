/* Copyright (C) tsotchke. SPDX-License-Identifier: MIT */
#ifndef ESHKOL_BACKEND_NORETURN_BLOCKS_H
#define ESHKOL_BACKEND_NORETURN_BLOCKS_H

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <vector>

namespace eshkol {

// The block boundary after a call that does not return.
//
// A no-return emission (raise, error, exit, an unconditional refusal) ends its
// basic block with `unreachable`. Two shapes follow, and each has one rule:
//
//   * A guard: `condbr ok, fail`, with the no-return call in `fail`. The
//     emitter seats the builder on `ok` itself (CodegenContext::emitRaise).
//
//   * An unconditional no-return: there is no success block, yet the caller
//     that asked for the form goes on lowering the rest of the enclosing
//     expression through the same builder. continueAfterNoReturn() is the one
//     way to hand that caller a place to write: a fresh block of the current
//     function with no predecessors, with the builder seated on it. Whatever
//     the caller emits there is unreachable and stays well formed.
//
// blocksPastUnreachable() is the matching check: it lists every block that
// carries instructions behind its first `unreachable`, which is what an
// emitter of the second shape produces when it returns without calling
// continueAfterNoReturn(). It reports; it does not edit the module.

/// Seat @p builder on a fresh, predecessor-less block of the current function.
/// The current block is the one a no-return emission just ended; if the caller
/// has not written its terminator yet, `unreachable` is written first.
/// Returns the continuation block.
inline llvm::BasicBlock* continueAfterNoReturn(llvm::IRBuilderBase& builder,
                                               const llvm::Twine& name = "post_noreturn_dead") {
    llvm::BasicBlock* ended = builder.GetInsertBlock();
    if (ended->empty() || !ended->back().isTerminator()) {
        builder.CreateUnreachable();
    }
    llvm::BasicBlock* continuation =
        llvm::BasicBlock::Create(builder.getContext(), name, ended->getParent());
    builder.SetInsertPoint(continuation);
    return continuation;
}

struct BlockPastUnreachable {
    const llvm::Function* function;
    const llvm::BasicBlock* block;
    unsigned trailing_instructions;
};

/// Every block of @p module with instructions behind its first `unreachable`.
/// A block whose first terminator is anything else is left to the verifier.
inline std::vector<BlockPastUnreachable> blocksPastUnreachable(const llvm::Module& module) {
    std::vector<BlockPastUnreachable> found;
    for (const llvm::Function& function : module) {
        for (const llvm::BasicBlock& block : function) {
            const llvm::Instruction* first_unreachable = nullptr;
            for (const llvm::Instruction& instruction : block) {
                if (llvm::isa<llvm::UnreachableInst>(instruction)) {
                    first_unreachable = &instruction;
                    break;
                }
                if (instruction.isTerminator()) break;
            }
            if (!first_unreachable) continue;
            unsigned trailing = 0;
            for (const llvm::Instruction* next = first_unreachable->getNextNode(); next;
                 next = next->getNextNode()) {
                ++trailing;
            }
            if (trailing) found.push_back({&function, &block, trailing});
        }
    }
    return found;
}

}  // namespace eshkol

#endif  // ESHKOL_BACKEND_NORETURN_BLOCKS_H
