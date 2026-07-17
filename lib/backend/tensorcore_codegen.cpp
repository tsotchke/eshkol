/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Canonical LLVM declaration and registration path for Eshkol's TensorCore
 * adapter. TensorCore compiler internals do not participate in this lowering.
 */

#include <eshkol/backend/tensorcore_codegen.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/backend/codegen_context.h>
#include <eshkol/logger.h>

#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>

#include <array>
#include <utility>

namespace eshkol {
namespace {

constexpr std::array<const char*, 34> adapter_symbols = {{
    "eshkol_tc_adapter_available",
    "eshkol_tc_adapter_status",
    "eshkol_tc_check_abi_version",
    "eshkol_tc_last_status",
    "eshkol_tc_runtime_capabilities_abi_version",
    "eshkol_tc_runtime_capabilities_status",
    "eshkol_tc_validate_runtime_capabilities",
    "eshkol_tc_known_capability_mask",
    "eshkol_tc_available_capability_mask",
    "eshkol_tc_compiled_backend_mask",
    "eshkol_tc_available_backend_mask",
    "eshkol_tc_init",
    "eshkol_tc_shutdown",
    "eshkol_tc_device_name",
    "eshkol_tc_device_family",
    "eshkol_tc_device_unified_memory",
    "eshkol_tc_device_supports_bf16",
    "eshkol_tc_device_supports_i8",
    "eshkol_tc_device_supports_tensorops_m5",
    "eshkol_tc_buffer_alloc",
    "eshkol_tc_buffer_free",
    "eshkol_tc_buffer_map",
    "eshkol_tc_buffer_size",
    "eshkol_tc_dtype_f16",
    "eshkol_tc_dtype_bf16",
    "eshkol_tc_dtype_f32",
    "eshkol_tc_dtype_i8",
    "eshkol_tc_dtype_i32",
    "eshkol_tc_gemm",
    "eshkol_tc_attention_forward",
    "eshkol_tc_last_backend_code",
    "eshkol_tc_last_backend_name",
    "eshkol_tc_version",
    "eshkol_tc_status_string",
}};

bool set_error(std::string* error, const std::string& message) {
    if (error) *error = message;
    return false;
}

}  // namespace

std::size_t declareTensorcoreAdapterAbi(llvm::Module& module,
                                        llvm::IntegerType* size_type,
                                        std::string* error) {
    if (error) error->clear();
    if (!size_type || !size_type->isIntegerTy(64)) {
        set_error(error, "tensorcore adapter ABI requires a 64-bit size type");
        return 0;
    }

    llvm::LLVMContext& llvm_context = module.getContext();
    llvm::Type* i32 = llvm::Type::getInt32Ty(llvm_context);
    llvm::Type* i64 = llvm::Type::getInt64Ty(llvm_context);
    llvm::Type* dbl = llvm::Type::getDoubleTy(llvm_context);
    llvm::Type* ptr = llvm::PointerType::get(llvm_context, 0);

    std::size_t declared = 0;
    auto add = [&](const char* name, llvm::Type* result,
                   std::initializer_list<llvm::Type*> arguments) -> bool {
        auto* function_type = llvm::FunctionType::get(result, arguments, false);
        llvm::Function* function = module.getFunction(name);
        if (function) {
            if (function->getFunctionType() != function_type) {
                return set_error(error,
                    std::string("tensorcore adapter ABI conflict for '") + name + "'");
            }
        } else {
            function = llvm::Function::Create(function_type,
                                              llvm::Function::ExternalLinkage,
                                              name,
                                              module);
        }
        ++declared;
        return true;
    };

    if (!add("eshkol_tc_adapter_available", i32, {}) ||
        !add("eshkol_tc_adapter_status", i32, {}) ||
        !add("eshkol_tc_check_abi_version", i32, {i32, i32, i32}) ||
        !add("eshkol_tc_last_status", i32, {}) ||
        !add("eshkol_tc_runtime_capabilities_abi_version", i32, {}) ||
        !add("eshkol_tc_runtime_capabilities_status", i32, {ptr, i32}) ||
        !add("eshkol_tc_validate_runtime_capabilities", i32,
             {i32, i64, i64, i64, i64}) ||
        !add("eshkol_tc_known_capability_mask", i64, {ptr}) ||
        !add("eshkol_tc_available_capability_mask", i64, {ptr}) ||
        !add("eshkol_tc_compiled_backend_mask", i64, {ptr}) ||
        !add("eshkol_tc_available_backend_mask", i64, {ptr}) ||
        !add("eshkol_tc_init", ptr, {}) ||
        !add("eshkol_tc_shutdown", i32, {ptr}) ||
        !add("eshkol_tc_device_name", ptr, {ptr}) ||
        !add("eshkol_tc_device_family", i32, {ptr}) ||
        !add("eshkol_tc_device_unified_memory", i32, {ptr}) ||
        !add("eshkol_tc_device_supports_bf16", i32, {ptr}) ||
        !add("eshkol_tc_device_supports_i8", i32, {ptr}) ||
        !add("eshkol_tc_device_supports_tensorops_m5", i32, {ptr}) ||
        !add("eshkol_tc_buffer_alloc", ptr, {ptr, size_type}) ||
        !add("eshkol_tc_buffer_free", i32, {ptr, ptr}) ||
        !add("eshkol_tc_buffer_map", ptr, {ptr}) ||
        !add("eshkol_tc_buffer_size", size_type, {ptr}) ||
        !add("eshkol_tc_dtype_f16", i32, {}) ||
        !add("eshkol_tc_dtype_bf16", i32, {}) ||
        !add("eshkol_tc_dtype_f32", i32, {}) ||
        !add("eshkol_tc_dtype_i8", i32, {}) ||
        !add("eshkol_tc_dtype_i32", i32, {}) ||
        !add("eshkol_tc_gemm", i32,
             {ptr, i32, ptr, ptr, ptr, i32, i32, i32, dbl, dbl, i32, i32}) ||
        !add("eshkol_tc_attention_forward", i32,
             {ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, dbl, i32}) ||
        !add("eshkol_tc_last_backend_code", i32, {}) ||
        !add("eshkol_tc_last_backend_name", ptr, {}) ||
        !add("eshkol_tc_version", ptr, {}) ||
        !add("eshkol_tc_status_string", ptr, {i32})) {
        return 0;
    }

    if (declared != adapter_symbols.size()) {
        set_error(error, "tensorcore adapter declaration count drift");
        return 0;
    }
    return declared;
}

std::size_t registerTensorcoreBuiltins(CodegenContext& ctx,
                                       std::string* error) {
    const std::size_t declared =
        declareTensorcoreAdapterAbi(ctx.module(), ctx.sizeType(), error);
    if (declared != adapter_symbols.size()) return 0;

    for (const char* name : adapter_symbols) {
        llvm::Function* function = ctx.module().getFunction(name);
        if (!function) {
            set_error(error,
                      std::string("tensorcore adapter declaration missing after lowering: ") + name);
            return 0;
        }
        ctx.defineFunction(name, function);
    }
    return declared;
}

bool verifyTensorcoreAdapterModule(const llvm::Module& module,
                                   std::string* error) {
    std::string verifier_output;
    llvm::raw_string_ostream stream(verifier_output);
    if (llvm::verifyModule(module, &stream)) {
        stream.flush();
        return set_error(error, "tensorcore adapter LLVM verification failed: " + verifier_output);
    }
    if (error) error->clear();
    return true;
}

}  // namespace eshkol

extern "C" int eshkol_register_tensorcore_builtins(eshkol::CodegenContext* ctx) {
    if (!ctx) return -1;
    std::string error;
    const std::size_t count = eshkol::registerTensorcoreBuiltins(*ctx, &error);
    if (count == 0) {
        eshkol_error("%s", error.empty() ?
                     "tensorcore adapter declaration failed" : error.c_str());
        return -1;
    }
    eshkol_debug("tensorcore: registered %zu canonical adapter declarations", count);
    return static_cast<int>(count);
}

#endif  /* ESHKOL_LLVM_BACKEND_ENABLED */
