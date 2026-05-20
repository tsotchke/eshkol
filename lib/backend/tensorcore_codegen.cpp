/*
 * tensorcore_codegen.cpp — Eshkol LLVM-codegen bridge for tensorcore.
 *
 * Drop-in to eshkol-platform/lib/backend/.  Models the same pattern used by
 * builtin_declarations.cpp and tensor_codegen.cpp: declares the tensorcore
 * C ABI as ExternalLinkage functions in the Eshkol LLVM module, then exposes
 * Eshkol-callable wrappers (`__tc-init`, `__tc-gemm`, `__tc-attention-forward`,
 * `__tc-attention-backward`, `__tc-buffer-alloc`, `__tc-buffer-map`,
 * `__tc-buffer-free`, `__tc-last-backend`).
 *
 * Integration steps in eshkol-platform:
 *   1. Drop this file into lib/backend/.
 *   2. Add to CMakeLists.txt:
 *        target_link_libraries(eshkol PUBLIC tensorcore::tensorcore)
 *        target_sources(eshkol PRIVATE lib/backend/tensorcore_codegen.cpp)
 *   3. Call TensorcoreDeclarations(ctx) from CodegenContext init alongside
 *      BuiltinDeclarations.
 *   4. Register the Scheme bindings:
 *        (use tensorcore)
 *      which loads eshkol/tensorcore.esk (the .esk file in this repo).
 */

#include <eshkol/backend/codegen_context.h>
#include <eshkol/logger.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

namespace eshkol {

/* ====================================================================== *
 *  Tensorcore LLVM declarations                                           *
 *  Mirror of include/tensorcore/tensorcore.h. Order matches the C ABI    *
 *  symbol layout so the linker can resolve each name unambiguously.      *
 * ====================================================================== */
class TensorcoreDeclarations {
public:
    explicit TensorcoreDeclarations(CodegenContext& ctx) : ctx_(ctx) {
        declareLifecycle();
        declareBuffers();
        declareGemm();
        declareAttention();
        declareDiagnostics();

        eshkol_debug("TensorcoreDeclarations: declared %d external functions",
                     14);
    }

    llvm::Function* tc_init           = nullptr;
    llvm::Function* tc_shutdown       = nullptr;
    llvm::Function* tc_device_info    = nullptr;
    llvm::Function* tc_buffer_alloc   = nullptr;
    llvm::Function* tc_buffer_free    = nullptr;
    llvm::Function* tc_buffer_map     = nullptr;
    llvm::Function* tc_buffer_size    = nullptr;
    llvm::Function* tc_gemm           = nullptr;
    llvm::Function* tc_attention_fwd  = nullptr;
    llvm::Function* tc_attention_bwd  = nullptr;
    llvm::Function* tc_last_backend   = nullptr;
    llvm::Function* tc_status_string  = nullptr;
    llvm::Function* tc_dtype_size     = nullptr;
    llvm::Function* tc_version        = nullptr;

private:
    CodegenContext& ctx_;

    void declareLifecycle() {
        /* tc_status_t tc_init(tc_context** out_ctx)
         * tc_status_t tc_shutdown(tc_context* ctx)
         * tc_status_t tc_device_info_get(tc_context*, tc_device_info* out) */
        {
            auto* ft = llvm::FunctionType::get(
                ctx_.int32Type(), { ctx_.ptrType() }, false);
            tc_init = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                                              "tc_init", &ctx_.module());
        }
        {
            auto* ft = llvm::FunctionType::get(
                ctx_.int32Type(), { ctx_.ptrType() }, false);
            tc_shutdown = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                                                  "tc_shutdown", &ctx_.module());
        }
        {
            auto* ft = llvm::FunctionType::get(
                ctx_.int32Type(),
                { ctx_.ptrType(), ctx_.ptrType() }, false);
            tc_device_info = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                                                     "tc_device_info_get", &ctx_.module());
        }
    }

    void declareBuffers() {
        /* tc_status_t tc_buffer_alloc(tc_context*, size_t, tc_buffer**) */
        {
            auto* ft = llvm::FunctionType::get(
                ctx_.int32Type(),
                { ctx_.ptrType(), ctx_.int64Type(), ctx_.ptrType() }, false);
            tc_buffer_alloc = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                                                      "tc_buffer_alloc", &ctx_.module());
        }
        /* tc_status_t tc_buffer_free(tc_context*, tc_buffer*) */
        {
            auto* ft = llvm::FunctionType::get(
                ctx_.int32Type(),
                { ctx_.ptrType(), ctx_.ptrType() }, false);
            tc_buffer_free = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                                                     "tc_buffer_free", &ctx_.module());
        }
        /* tc_status_t tc_buffer_map(tc_buffer*, void**) */
        {
            auto* ft = llvm::FunctionType::get(
                ctx_.int32Type(),
                { ctx_.ptrType(), ctx_.ptrType() }, false);
            tc_buffer_map = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                                                    "tc_buffer_map", &ctx_.module());
        }
        /* size_t tc_buffer_size(const tc_buffer*) */
        {
            auto* ft = llvm::FunctionType::get(
                ctx_.int64Type(), { ctx_.ptrType() }, false);
            tc_buffer_size = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                                                     "tc_buffer_size", &ctx_.module());
        }
    }

    void declareGemm() {
        /* tc_status_t tc_gemm(tc_context*, const tc_gemm_desc*,
         *                     const tc_buffer*, const tc_buffer*, tc_buffer*) */
        auto* ft = llvm::FunctionType::get(
            ctx_.int32Type(),
            { ctx_.ptrType(), ctx_.ptrType(),
              ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType() },
            false);
        tc_gemm = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                                          "tc_gemm", &ctx_.module());
    }

    void declareAttention() {
        /* tc_status_t tc_attention_forward(tc_context*, const tc_attention_desc*,
         *                                  Q, K, V, O, LSE) */
        {
            auto* ft = llvm::FunctionType::get(
                ctx_.int32Type(),
                { ctx_.ptrType(), ctx_.ptrType(),
                  ctx_.ptrType(), ctx_.ptrType(),
                  ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType() },
                false);
            tc_attention_fwd = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                                                       "tc_attention_forward", &ctx_.module());
        }
        /* tc_status_t tc_attention_backward(ctx, desc, Q,K,V,O,dO,LSE, dQ,dK,dV) */
        {
            auto* ft = llvm::FunctionType::get(
                ctx_.int32Type(),
                { ctx_.ptrType(), ctx_.ptrType(),
                  ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType(),
                  ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType(),
                  ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType() },
                false);
            tc_attention_bwd = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                                                       "tc_attention_backward", &ctx_.module());
        }
    }

    void declareDiagnostics() {
        {
            auto* ft = llvm::FunctionType::get(ctx_.int32Type(), {}, false);
            tc_last_backend = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                                                      "tc_last_backend", &ctx_.module());
        }
        {
            auto* ft = llvm::FunctionType::get(
                ctx_.ptrType(), { ctx_.int32Type() }, false);
            tc_status_string = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                                                      "tc_status_string", &ctx_.module());
        }
        {
            auto* ft = llvm::FunctionType::get(
                ctx_.int64Type(), { ctx_.int32Type() }, false);
            tc_dtype_size = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                                                   "tc_dtype_size", &ctx_.module());
        }
        {
            auto* ft = llvm::FunctionType::get(ctx_.ptrType(), {}, false);
            tc_version = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                                                 "tc_version", &ctx_.module());
        }
    }
};

/* ====================================================================== *
 *  Public entry point — call from CodegenContext init.                    *
 * ====================================================================== */
static TensorcoreDeclarations* g_tc_decls = nullptr;

extern "C" void eshkol_register_tensorcore_builtins(CodegenContext* ctx) {
    if (!ctx) return;
    if (!g_tc_decls) {
        g_tc_decls = new TensorcoreDeclarations(*ctx);
        eshkol_info("tensorcore: %d external builtins registered", 14);
    }
}

}  /* namespace eshkol */

#endif  /* ESHKOL_LLVM_BACKEND_ENABLED */
