/**
 * @file closure_capture_scope.h
 * @brief Shared rules for supplying a statically resolved closure's captures.
 *
 * A lambda with free variables lowers to an LLVM function that takes one
 * pointer parameter per captured variable, appended after its user
 * parameters. Any site that calls such a function directly (instead of
 * dispatching on the runtime closure value) has to supply those pointers
 * itself. Two rules keep that sound:
 *
 *  1. A captured value may come only from the function being emitted: one of
 *     its own arguments or instructions, or a module-level value (global,
 *     constant, function). A symbol-table entry can still name an argument or
 *     alloca of an ENCLOSING function while a nested lambda body is being
 *     emitted; using it produces IR that refers to another function.
 *     valueUsableInFunction() is the predicate every capture site checks.
 *
 *  2. When the callee was reached through a NAME (a variable bound to a
 *     closure), its captured values belong to the closure object, not to
 *     whatever that name's free variables happen to resolve to at the call
 *     site. The call site may sit inside another function, or under a
 *     binding that shadows one of the captured names. emitClosureCaptureArguments()
 *     reads them from the closure's environment, which is exactly what the
 *     runtime closure call does.
 */

#ifndef ESHKOL_BACKEND_CLOSURE_CAPTURE_SCOPE_H
#define ESHKOL_BACKEND_CLOSURE_CAPTURE_SCOPE_H

#include <eshkol/eshkol.h>

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Value.h>

#include <string>
#include <vector>

namespace eshkol {

/**
 * @brief True when @p v may be referenced from code inside @p fn.
 *
 * Arguments and instructions are usable only inside the function that owns
 * them. Globals, constants and functions are module-level and usable
 * anywhere. A null @p v or @p fn is reported as usable so callers keep their
 * own null handling.
 */
inline bool valueUsableInFunction(const llvm::Value* v, const llvm::Function* fn) {
    if (!v || !fn) return true;
    if (const auto* arg = llvm::dyn_cast<llvm::Argument>(v)) {
        return arg->getParent() == fn;
    }
    if (const auto* inst = llvm::dyn_cast<llvm::Instruction>(v)) {
        return inst->getFunction() == fn;
    }
    return true;
}

/**
 * @brief Name of the function that owns @p v, for diagnostics.
 * @return the owner's LLVM name, or "<module>" for module-level values.
 */
inline std::string valueOwnerName(const llvm::Value* v) {
    const llvm::Function* owner = nullptr;
    if (const auto* arg = llvm::dyn_cast_or_null<llvm::Argument>(v)) {
        owner = arg->getParent();
    } else if (const auto* inst = llvm::dyn_cast_or_null<llvm::Instruction>(v)) {
        owner = inst->getFunction();
    }
    return owner ? owner->getName().str() : std::string("<module>");
}

/**
 * @brief True when @p fn takes closure-capture parameters.
 *
 * Capture parameters are the pointer parameters a lambda or local define is
 * lowered with, one per free variable (or one `captured_env` for a large
 * closure); they are the only parameters named with the `captured_` prefix.
 * A direct call to such a function must supply them.
 */
inline bool functionHasCaptureParameters(const llvm::Function* fn) {
    if (!fn) return false;
    for (const llvm::Argument& arg : fn->args()) {
        if (arg.getName().starts_with("captured_")) return true;
    }
    return false;
}

/**
 * @brief The current function's own pointer that reaches free variable
 *        @p var_name, or nullptr.
 *
 * A function reaches a variable of an enclosing scope only through its own
 * parameters: `<var>_cap` for a named-let loop's forward of an outer
 * variable, `captured_<var>` for a closure capture. Both already hold the
 * single-load convention a callee capture parameter expects.
 */
inline llvm::Value* currentFunctionCapturePointer(llvm::Function* fn,
                                                  const std::string& var_name) {
    if (!fn) return nullptr;
    const std::string named_let = var_name + "_cap";
    const std::string closure = "captured_" + var_name;
    for (llvm::Argument& arg : fn->args()) {
        if (arg.getType()->isPointerTy() && arg.getName() == named_let) return &arg;
    }
    for (llvm::Argument& arg : fn->args()) {
        if (arg.getType()->isPointerTy() && arg.getName() == closure) return &arg;
    }
    return nullptr;
}

/**
 * @brief Source location suffix for a capture diagnostic, " (line L, column C)"
 *        or empty when @p ast carries no location.
 */
inline std::string captureDiagnosticLocation(const eshkol_ast_t* ast) {
    if (!ast || ast->line == 0) return std::string();
    return " (line " + std::to_string(ast->line) + ", column " +
           std::to_string(ast->column) + ")";
}

/**
 * @brief Emit the capture-pointer arguments for a direct call to @p callee,
 *        reading them from the environment of the closure at @p closure_ptr.
 *
 * Mirrors the closure-call ABI: the closure's environment pointer sits at
 * offset 8; the environment holds a packed info word (capture count in bits
 * 0-31) followed by one tagged value per capture. A callee with at most 64
 * captures takes one pointer per capture slot; a larger one takes a single
 * `captured_env` parameter holding the environment pointer. A lone capture
 * parameter named `captured_env` is ambiguous at compile time (it is also
 * the name of a single capture of a variable called `env`), so that case is
 * decided from the environment's capture count at run time.
 *
 * @param b builder positioned where the call will be emitted.
 * @param tagged_ty the tagged value struct type.
 * @param closure_ptr pointer to the closure object (not the tagged value).
 * @param callee the function that will be called.
 * @param first_capture_param index of the callee's first capture parameter.
 * @return one pointer per capture parameter, in parameter order.
 */
inline std::vector<llvm::Value*> emitClosureCaptureArguments(
        llvm::IRBuilder<>& b, llvm::Type* tagged_ty, llvm::Value* closure_ptr,
        llvm::Function* callee, size_t first_capture_param) {
    std::vector<llvm::Value*> out;
    if (!callee || !closure_ptr) return out;
    size_t num_params = callee->getFunctionType()->getNumParams();
    if (num_params <= first_capture_param) return out;
    size_t num_capture_params = num_params - first_capture_param;

    llvm::LLVMContext& ctx = b.getContext();
    llvm::Type* i8 = llvm::Type::getInt8Ty(ctx);
    llvm::Type* i64 = llvm::Type::getInt64Ty(ctx);
    llvm::PointerType* ptr = llvm::PointerType::getUnqual(ctx);

    llvm::Value* env_addr = b.CreateGEP(i8, closure_ptr, llvm::ConstantInt::get(i64, 8));
    llvm::Value* env_ptr = b.CreateLoad(ptr, env_addr, "closure_env");
    llvm::Value* captures_base = b.CreateGEP(i8, env_ptr, llvm::ConstantInt::get(i64, 8),
                                             "closure_captures");

    if (num_capture_params == 1 &&
        callee->getArg(first_capture_param)->getName() == "captured_env") {
        llvm::Value* info = b.CreateLoad(i64, env_ptr, "closure_env_info");
        llvm::Value* count = b.CreateAnd(info, llvm::ConstantInt::get(i64, 0xFFFFFFFFULL));
        llvm::Value* large = b.CreateICmpUGT(count, llvm::ConstantInt::get(i64, 64));
        out.push_back(b.CreateSelect(large, env_ptr, captures_base, "closure_capture_arg"));
        return out;
    }

    for (size_t i = 0; i < num_capture_params; ++i) {
        out.push_back(b.CreateGEP(tagged_ty, captures_base, llvm::ConstantInt::get(i64, i),
                                  "closure_capture_slot"));
    }
    return out;
}

} // namespace eshkol

#endif // ESHKOL_BACKEND_CLOSURE_CAPTURE_SCOPE_H
