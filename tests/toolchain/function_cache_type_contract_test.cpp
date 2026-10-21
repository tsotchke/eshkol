#include <eshkol/backend/function_cache.h>
#include <llvm/IR/GlobalVariable.h>
#include <cstdio>
#include <stdexcept>
#include <string>

int main() {
    unsigned checks = 0;
    for (bool wasm : {false, true}) {
        llvm::LLVMContext context;
        eshkol::TypeSystem types(context, wasm);
        auto* expected = llvm::FunctionType::get(types.getSizeType(), {types.getPtrType()}, false);
        llvm::Module valid("valid", context);
        auto* declared = llvm::Function::Create(expected, llvm::Function::ExternalLinkage, "strlen", valid);
        eshkol::FunctionCache good(valid, types);
        if (good.getStrlen() != declared || good.getStrlen() != declared) return 1;
        good.reset();
        if (good.getStrlen() != declared || valid.size() != 1) return 2;
        ++checks;
        llvm::FunctionType* wrong[] = {
            llvm::FunctionType::get(types.getVoidType(), {types.getPtrType()}, false),
            llvm::FunctionType::get(types.getSizeType(), {}, false),
            llvm::FunctionType::get(types.getSizeType(), {types.getPtrType()}, true),
        };
        for (auto* signature : wrong) {
            llvm::Module module("mismatch", context);
            auto* original = llvm::Function::Create(signature, llvm::Function::ExternalLinkage, "strlen", module);
            eshkol::FunctionCache cache(module, types);
            bool refused = false;
            try { cache.getStrlen(); }
            catch (const std::runtime_error& e) {
                refused = std::string(e.what()).find("strlen") != std::string::npos;
            }
            if (!refused || module.size() != 1 || module.getFunction("strlen") != original) {
                std::fprintf(stderr, "FAIL: mismatched function declaration was accepted or renamed\n");
                return 3;
            }
            ++checks;
        }
        llvm::Module collision("global-collision", context);
        auto* original = new llvm::GlobalVariable(collision, types.getInt8Type(), false,
            llvm::GlobalValue::ExternalLinkage, nullptr, "strlen");
        eshkol::FunctionCache cache(collision, types);
        bool refused = false;
        try { cache.getStrlen(); }
        catch (const std::runtime_error& e) {
            refused = std::string(e.what()).find("strlen") != std::string::npos;
        }
        if (!refused || !collision.empty() || collision.getNamedValue("strlen") != original) {
            std::fprintf(stderr, "FAIL: global/function name collision was accepted or renamed\n");
            return 4;
        }
        ++checks;
    }
    std::printf("PASS: function-cache declaration contracts %u checks\n", checks);
    return 0;
}
