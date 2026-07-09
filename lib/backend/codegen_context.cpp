/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * CodegenContext implementation - Shared state management
 */

#include <eshkol/backend/codegen_context.h>
#include <eshkol/eshkol.h>  // HEAP_SUBTYPE_SYMBOL, etc.

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>

namespace eshkol {

// ESH-0214c: emit the region write barrier around a store of a tagged value
// into a longer-lived destination. Spills value + result to entry-block allocas
// (so the pattern is safe inside loop bodies) and calls the runtime helper.
llvm::Value* CodegenContext::emitRegionWriteBarrier(llvm::Value* dst_ptr,
                                                    llvm::Value* tagged_value) {
    if (!tagged_value || tagged_value->getType() != taggedValueType()) {
        // Only tagged values can carry a heap pointer that might dangle.
        return tagged_value;
    }

    llvm::Type* tv_ty = taggedValueType();
    llvm::PointerType* ptr_ty = ptrType();

    // Hoist the spill/result slots to the function entry block: an alloca in a
    // loop body re-adjusts the stack pointer every iteration and is only
    // reclaimed at function return.
    llvm::Function* fn = builder_.GetInsertBlock()->getParent();
    llvm::IRBuilderBase::InsertPoint saved_ip = builder_.saveIP();
    if (fn && !fn->empty()) {
        llvm::BasicBlock& entry = fn->getEntryBlock();
        builder_.SetInsertPoint(&entry, entry.begin());
    }
    llvm::AllocaInst* val_slot = builder_.CreateAlloca(tv_ty, nullptr, "wb_val_slot");
    llvm::AllocaInst* out_slot = builder_.CreateAlloca(tv_ty, nullptr, "wb_out_slot");
    builder_.restoreIP(saved_ip);

    builder_.CreateStore(tagged_value, val_slot);

    llvm::Function* wb = module_.getFunction("eshkol_region_write_barrier_into");
    if (!wb) {
        llvm::FunctionType* wb_ty = llvm::FunctionType::get(
            voidType(), {ptr_ty, ptr_ty, ptr_ty}, false);
        wb = llvm::Function::Create(wb_ty, llvm::GlobalValue::ExternalLinkage,
                                    "eshkol_region_write_barrier_into", &module_);
    }

    llvm::Value* dst_cast = dst_ptr;
    if (dst_ptr && dst_ptr->getType() != ptr_ty) {
        dst_cast = builder_.CreateBitCast(dst_ptr, ptr_ty);
    } else if (!dst_ptr) {
        dst_cast = llvm::ConstantPointerNull::get(ptr_ty);
    }

    builder_.CreateCall(wb, {out_slot, dst_cast, val_slot});
    return builder_.CreateLoad(tv_ty, out_slot, "wb_result");
}

CodegenContext::CodegenContext(llvm::LLVMContext& llvm_ctx,
                               llvm::Module& llvm_mod,
                               llvm::IRBuilder<>& ir_builder,
                               TypeSystem& type_sys,
                               FunctionCache& func_cache,
                               MemoryCodegen& mem_codegen)
    : context_(llvm_ctx)
    , module_(llvm_mod)
    , builder_(ir_builder)
    , types_(type_sys)
    , funcs_(func_cache)
    , memory_(mem_codegen) {
    // Start with one local scope
    scope_stack_.emplace_back();
}

// === Symbol Table Management ===

void CodegenContext::pushScope() {
    scope_stack_.emplace_back();
}

void CodegenContext::popScope() {
    if (scope_stack_.size() > 1) {
        scope_stack_.pop_back();
    }
}

llvm::Value* CodegenContext::lookupSymbol(const std::string& name) {
    // Search from innermost to outermost local scope
    for (auto it = scope_stack_.rbegin(); it != scope_stack_.rend(); ++it) {
        auto found = it->find(name);
        if (found != it->end()) {
            return found->second;
        }
    }
    // Fall back to global symbols
    auto global_it = global_symbols_.find(name);
    if (global_it != global_symbols_.end()) {
        return global_it->second;
    }
    return nullptr;
}

void CodegenContext::defineSymbol(const std::string& name, llvm::Value* value) {
    if (!scope_stack_.empty()) {
        scope_stack_.back()[name] = value;
    }
}

void CodegenContext::defineGlobalSymbol(const std::string& name, llvm::Value* value) {
    global_symbols_[name] = value;
}

llvm::Value* CodegenContext::lookupGlobalSymbol(const std::string& name) {
    auto it = global_symbols_.find(name);
    return (it != global_symbols_.end()) ? it->second : nullptr;
}

// === Function Table ===

llvm::Function* CodegenContext::lookupFunction(const std::string& name) {
    auto it = function_table_.find(name);
    return (it != function_table_.end()) ? it->second : nullptr;
}

void CodegenContext::defineFunction(const std::string& name, llvm::Function* func) {
    function_table_[name] = func;
}

bool CodegenContext::hasFunction(const std::string& name) const {
    return function_table_.find(name) != function_table_.end();
}

// === Variadic Function Info ===

void CodegenContext::registerVariadicFunction(const std::string& name,
                                              uint64_t fixedParams,
                                              bool isVariadic) {
    variadic_info_[name] = {fixedParams, isVariadic};
}

std::pair<uint64_t, bool> CodegenContext::getVariadicInfo(const std::string& name) const {
    auto it = variadic_info_.find(name);
    return (it != variadic_info_.end()) ? it->second : std::make_pair(static_cast<uint64_t>(0), false);
}

// === String Interning ===

llvm::GlobalVariable* CodegenContext::internString(const std::string& str) {
    // Check if already interned
    auto it = interned_strings_.find(str);
    if (it != interned_strings_.end()) {
        return it->second;
    }

    // Create a new global string
    llvm::Constant* str_constant = llvm::ConstantDataArray::getString(context_, str, true);
    llvm::GlobalVariable* global_str = new llvm::GlobalVariable(
        module_,
        str_constant->getType(),
        true,  // isConstant
        llvm::GlobalValue::PrivateLinkage,
        str_constant,
        ".str"
    );
    global_str->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);

    interned_strings_[str] = global_str;
    return global_str;
}

llvm::Value* CodegenContext::internCString(const std::string& str) {
    llvm::GlobalVariable* global_str = internString(str);
    return builder_.CreatePointerCast(global_str, llvm::PointerType::get(context_, 0));
}

llvm::GlobalVariable* CodegenContext::lookupInternedString(const std::string& str) {
    auto it = interned_strings_.find(str);
    return (it != interned_strings_.end()) ? it->second : nullptr;
}

llvm::Value* CodegenContext::internStringWithHeader(const std::string& str, uint8_t subtype) {
    // SYMBOL subtype is special: R7RS §6.5 requires every literal (quote sym)
    // to denote the same object across the whole program. A per-module
    // private constant (what the STRING path below produces) gives us a
    // fresh pointer per module, and since each top-level form in the JIT
    // compiles to its own module, `(define x 'foo) (define y 'foo)` ends
    // up with `(eq? x y)` = #f — broken. Route symbol literals through a
    // runtime helper that hashes the name into a process-global intern
    // table, so every module observes the same canonical pointer.
    //
    // The helper eshkol_intern_symbol_lookup lives in introspection.cpp
    // and shares g_interned_symbols with string->symbol / procedure-name /
    // type-of, so all four interning paths converge.
    if (subtype == HEAP_SUBTYPE_SYMBOL) {
        // Name string constant (module-local; LLVM dedups these across the
        // module naturally, so repeated references to the same symbol share
        // a single underlying `.str` global).
        llvm::Value* name_ptr = builder_.CreateGlobalStringPtr(str, ".sym_name");

        // Declare the extern runtime helper.
        llvm::Type* ptr_ty = llvm::PointerType::get(context_, 0);
        llvm::FunctionType* fn_ty = llvm::FunctionType::get(ptr_ty, {ptr_ty}, false);
        llvm::FunctionCallee fn =
            module_.getOrInsertFunction("eshkol_intern_symbol_lookup", fn_ty);

        // Emit the call. Result is the canonical symbol pointer — same
        // representation as the module-local constant the STRING path
        // produces (pointer to char data, with a HEAP_SUBTYPE_SYMBOL header
        // at ptr-8), so all downstream packPtrToTaggedValue / eq? / etc.
        // callers work unchanged.
        return builder_.CreateCall(fn, {name_ptr}, "sym");
    }

    // Check if already interned with header
    std::string key = str + "_hdr_" + std::to_string(subtype);
    auto it = headered_strings_.find(key);
    if (it != headered_strings_.end()) {
        return it->second;
    }

    // Create struct type: {i8 subtype, i8 flags, i16 ref_count, i32 size, [N x i8] data}
    // This matches eshkol_object_header_t followed by string data
    size_t str_len = str.length() + 1;  // Include null terminator

    llvm::ArrayType* str_array_type = llvm::ArrayType::get(int8Type(), str_len);
    llvm::StructType* headered_str_type = llvm::StructType::get(
        context_,
        {int8Type(), int8Type(), int16Type(), int32Type(), str_array_type},
        true  // packed for exact layout
    );

    // Create header constants
    llvm::Constant* subtype_const = llvm::ConstantInt::get(int8Type(), subtype);
    llvm::Constant* flags_const = llvm::ConstantInt::get(int8Type(), 0);
    llvm::Constant* ref_count_const = llvm::ConstantInt::get(int16Type(), 0);
    llvm::Constant* size_const = llvm::ConstantInt::get(int32Type(), (uint32_t)str_len);

    // Create string data constant
    llvm::Constant* str_data = llvm::ConstantDataArray::getString(context_, str, true);

    // Create the complete struct initializer
    llvm::Constant* struct_init = llvm::ConstantStruct::get(
        headered_str_type,
        {subtype_const, flags_const, ref_count_const, size_const, str_data}
    );

    // Create global variable
    llvm::GlobalVariable* global_struct = new llvm::GlobalVariable(
        module_,
        headered_str_type,
        true,  // isConstant
        llvm::GlobalValue::PrivateLinkage,
        struct_init,
        ".str_hdr"
    );
    global_struct->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);

    // GEP to get pointer to string data (field index 4, the array)
    llvm::Constant* zero = llvm::ConstantInt::get(int32Type(), 0);
    llvm::Constant* idx_data = llvm::ConstantInt::get(int32Type(), 4);
    llvm::Constant* indices[] = {zero, idx_data, zero};
    llvm::Constant* str_ptr = llvm::ConstantExpr::getInBoundsGetElementPtr(
        headered_str_type, global_struct, indices
    );

    headered_strings_[key] = str_ptr;
    return str_ptr;
}

// === Nested Function Captures ===

void CodegenContext::setFunctionCaptures(const std::string& funcName,
                                         const std::vector<std::string>& captures) {
    function_captures_[funcName] = captures;
}

const std::vector<std::string>* CodegenContext::getFunctionCaptures(const std::string& funcName) const {
    auto it = function_captures_.find(funcName);
    return (it != function_captures_.end()) ? &it->second : nullptr;
}

// === Lambda Return Tracking ===

void CodegenContext::setFunctionReturnsLambda(const std::string& funcName,
                                              const std::string& lambdaName) {
    functions_returning_lambda_[funcName] = lambdaName;
}

std::string CodegenContext::getFunctionReturnsLambda(const std::string& funcName) const {
    auto it = functions_returning_lambda_.find(funcName);
    return (it != functions_returning_lambda_.end()) ? it->second : "";
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
