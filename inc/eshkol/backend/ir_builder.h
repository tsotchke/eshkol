/* Copyright (C) tsotchke. SPDX-License-Identifier: MIT */
#ifndef ESHKOL_BACKEND_IR_BUILDER_H
#define ESHKOL_BACKEND_IR_BUILDER_H

#include <llvm/IR/IRBuilder.h>
#include <type_traits>

namespace eshkol {

// A scalar operand belongs to the emission call. Construct LLVM's borrowed
// ArrayRef only after entering that synchronous call, from a named Value* slot.
// In particular, converting a ConstantInt* expression to ArrayRef<Value*> at a
// coroutine call site would make the view borrow an implicit pointer temporary.
// Keeping that storage here separates suspended lowering state from LLVM's
// transient operand views. Range and initializer-list calls retain LLVM's API.
class CodegenIRBuilder : public llvm::IRBuilder<> {
    using Base = llvm::IRBuilder<>;

public:
    using Base::Base;
    using Base::CreateGEP;
    using Base::CreateInBoundsGEP;
    using Base::CreateCall;

    template<class Operand>
        requires std::is_convertible_v<Operand, llvm::Value*>
    llvm::Value* CreateGEP(llvm::Type* type, llvm::Value* pointer, Operand operand,
                          const llvm::Twine& name = "",
                          llvm::GEPNoWrapFlags flags = llvm::GEPNoWrapFlags::none()) {
        llvm::Value* index = operand;
        return Base::CreateGEP(type, pointer,
                               llvm::ArrayRef<llvm::Value*>(&index, 1), name, flags);
    }

    template<class Operand>
        requires std::is_convertible_v<Operand, llvm::Value*>
    llvm::Value* CreateInBoundsGEP(llvm::Type* type, llvm::Value* pointer,
                                  Operand operand, const llvm::Twine& name = "") {
        llvm::Value* index = operand;
        return Base::CreateInBoundsGEP(type, pointer,
                                       llvm::ArrayRef<llvm::Value*>(&index, 1), name);
    }

    template<class Operand>
        requires std::is_convertible_v<Operand, llvm::Value*>
    llvm::CallInst* CreateCall(llvm::FunctionType* type, llvm::Value* callee,
                               Operand operand, const llvm::Twine& name = "",
                               llvm::MDNode* math_tag = nullptr) {
        llvm::Value* argument = operand;
        return Base::CreateCall(type, callee,
                                llvm::ArrayRef<llvm::Value*>(&argument, 1), name, math_tag);
    }

    template<class Operand>
        requires std::is_convertible_v<Operand, llvm::Value*>
    llvm::CallInst* CreateCall(llvm::FunctionCallee callee, Operand operand,
                               const llvm::Twine& name = "",
                               llvm::MDNode* math_tag = nullptr) {
        llvm::Value* argument = operand;
        return Base::CreateCall(callee,
                                llvm::ArrayRef<llvm::Value*>(&argument, 1), name, math_tag);
    }
};

} // namespace eshkol
#endif
