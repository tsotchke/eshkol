#include <eshkol/backend/llvm_codegen.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

Function* EshkolLLVMCodeGen::tryResolveReplFunction(const std::string& func_name) {
        // Only check REPL symbols if REPL mode is enabled
        if (!eshkol::llvm_codegen_detail::replModeEnabled()) {
            return nullptr;
        }

        std::lock_guard<std::mutex> lock(eshkol::llvm_codegen_detail::replMutex());

        // MODULE VISIBILITY: Check if symbol is private (not exported from its module)
        if (eshkol::llvm_codegen_detail::replPrivateSymbols().find(func_name) != eshkol::llvm_codegen_detail::replPrivateSymbols().end()) {
            eshkol_error("Function '%s' is private (not exported from its module)", func_name.c_str());
            markFatalCodegenError();
            return nullptr;
        }

        // Check multiple name variations and track which one matched
        std::string actual_func_name;

        // 1. Exact name
        auto it = eshkol::llvm_codegen_detail::replFunctionAddresses().find(func_name);
        if (it != eshkol::llvm_codegen_detail::replFunctionAddresses().end()) {
            actual_func_name = func_name;
        } else {
            // 2. With "_func" suffix for lambda variables
            std::string func_key = func_name + "_func";
            it = eshkol::llvm_codegen_detail::replFunctionAddresses().find(func_key);
            if (it != eshkol::llvm_codegen_detail::replFunctionAddresses().end()) {
                actual_func_name = func_key;
            } else if (func_name.ends_with("_func")) {
                // 3. Try removing "_func" suffix if present
                std::string base_name = func_name.substr(0, func_name.length() - 5);
                it = eshkol::llvm_codegen_detail::replFunctionAddresses().find(base_name);
                if (it != eshkol::llvm_codegen_detail::replFunctionAddresses().end()) {
                    actual_func_name = base_name;
                }
            }
        }

        if (it == eshkol::llvm_codegen_detail::replFunctionAddresses().end()) {
            return nullptr;
        }

        // CRITICAL: If this is a lambda variable (like "square"), resolve to the actual lambda name
        // The JIT only has the lambda function (e.g., "lambda_0"), not the alias (e.g., "square_func")
        std::string jit_symbol_name = actual_func_name;
        auto lambda_it = eshkol::llvm_codegen_detail::replLambdaNames().find(func_name);
        if (lambda_it != eshkol::llvm_codegen_detail::replLambdaNames().end()) {
            // Found lambda mapping: use the actual lambda name that exists in JIT
            jit_symbol_name = lambda_it->second;
        }

        // Get the function's arity using the actual matched name
        size_t arity = 0;
        auto arity_it = eshkol::llvm_codegen_detail::replFunctionArities().find(actual_func_name);
        if (arity_it != eshkol::llvm_codegen_detail::replFunctionArities().end()) {
            arity = arity_it->second;
        }

        // Did the host explicitly mark this name as a native C function (i.e.
        // compiled by Clang/GCC, NOT by the JIT)? If so we cannot use the IR
        // struct {i8,i8,i16,i32,i64} for params/return — Clang coerces a
        // 16-byte struct-by-value into `[2 x i64]` per AAPCS/SysV, which is
        // a different register-packing than what LLVM does for the IR struct.
        // We register a *thunk* with the struct ABI (so existing call sites
        // are unchanged) that internally bitcasts to [2 x i64] and forwards.
        bool is_native_c = eshkol::llvm_codegen_detail::replNativeCFunctions().count(jit_symbol_name) > 0
                       || eshkol::llvm_codegen_detail::replNativeCFunctions().count(actual_func_name) > 0
                       || eshkol::llvm_codegen_detail::replNativeCFunctions().count(func_name) > 0;

        // Check if we already have this function in the current module.
        // For native-C functions the cache key is the thunk name, NOT the
        // raw JIT symbol (that name now refers to the [2 x i64] declaration
        // and is wrong-typed for callers who expect a tagged_value-ABI fn).
        std::string cache_key = is_native_c
            ? (jit_symbol_name + "__cabi_thunk")
            : jit_symbol_name;
        Function* extern_func = module->getFunction(cache_key);
        if (!extern_func) {
            if (is_native_c) {
                // 1. Declare the *real* C function with the [2 x i64] ABI
                //    under the original JIT symbol name. ORC will resolve
                //    that name via dlsym to the host C function (compiled
                //    by Clang/GCC, which itself coerced the 16-byte struct
                //    to [2 x i64] for AAPCS / SysV register passing).
                ArrayType* coerced_type = ArrayType::get(int64_type, 2);
                std::vector<Type*> cabi_param_types(arity, coerced_type);
                FunctionType* cabi_type = FunctionType::get(
                    coerced_type,
                    cabi_param_types,
                    false
                );
                Function* cabi_real = module->getFunction(jit_symbol_name);
                if (!cabi_real) {
                    cabi_real = Function::Create(
                        cabi_type,
                        Function::ExternalLinkage,
                        jit_symbol_name,  // ORC will dlsym this name
                        module.get()
                    );
                }

                // 2. Build a thunk with the IR struct ABI that just unpacks
                //    its struct args into [2 x i64], calls the C function,
                //    and packs the result back. The thunk has private
                //    linkage so it does not collide across modules.
                std::vector<Type*> thunk_param_types(arity, tagged_value_type);
                FunctionType* thunk_type = FunctionType::get(
                    tagged_value_type,
                    thunk_param_types,
                    false
                );
                std::string thunk_name = jit_symbol_name + "__cabi_thunk";
                Function* thunk = Function::Create(
                    thunk_type,
                    Function::PrivateLinkage,
                    thunk_name,
                    module.get()
                );
                // Save current insert point so we can return to the caller
                // mid-codegen without disturbing it.
                IRBuilder<>::InsertPoint saved_ip = builder->saveIP();
                BasicBlock* entry = BasicBlock::Create(*context, "entry", thunk);
                builder->SetInsertPoint(entry);

                // Coerce each struct arg → [2 x i64] via memory (alloca,
                // store struct, load [2 x i64]). This is exactly what Clang
                // emits for a 16-byte struct passed by value at -O0.
                std::vector<Value*> coerced_args;
                coerced_args.reserve(arity);
                for (size_t i = 0; i < arity; ++i) {
                    Value* slot = builder->CreateAlloca(
                        tagged_value_type, nullptr,
                        "cabi_arg_" + std::to_string(i));
                    builder->CreateStore(thunk->getArg(i), slot);
                    Value* loaded = builder->CreateLoad(
                        coerced_type, slot,
                        "cabi_arg_" + std::to_string(i) + "_coerced");
                    coerced_args.push_back(loaded);
                }
                Value* cabi_ret = builder->CreateCall(
                    cabi_type, cabi_real, coerced_args, "cabi_ret");

                // Coerce return [2 x i64] → struct via memory.
                Value* ret_slot = builder->CreateAlloca(
                    tagged_value_type, nullptr, "cabi_ret_slot");
                builder->CreateStore(cabi_ret, ret_slot);
                Value* ret_struct = builder->CreateLoad(
                    tagged_value_type, ret_slot, "cabi_ret_struct");
                builder->CreateRet(ret_struct);

                builder->restoreIP(saved_ip);

                extern_func = thunk;
            } else {
                // Function doesn't exist yet - create external declaration
                // All Eshkol functions return eshkol_tagged_value and take tagged_value parameters
                std::vector<Type*> param_types(arity, tagged_value_type);
                FunctionType* func_type = FunctionType::get(
                    tagged_value_type,
                    param_types,
                    false  // NOT varargs - use exact arity
                );

                // CRITICAL: Use the actual JIT symbol name that exists in the JIT, not the requested name
                extern_func = Function::Create(
                    func_type,
                    Function::ExternalLinkage,
                    jit_symbol_name,  // Use the actual JIT symbol name (e.g., "lambda_0")!
                    module.get()
                );
            }
        }

        // Register in function table so subsequent lookups find it
        // Register under BOTH the requested name and JIT symbol name for flexibility
        function_table[func_name] = extern_func;
        if (jit_symbol_name != func_name) {
            function_table[jit_symbol_name] = extern_func;
        }
        if (actual_func_name != func_name && actual_func_name != jit_symbol_name) {
            function_table[actual_func_name] = extern_func;
        }

        // Bug BB: also propagate the resolved arity into function_arity_table
        // so future variable-codegen paths that look up `func_name` in
        // function_table (line ~7212) ALSO find an arity entry and take the
        // closure-wrap branch.
        //
        // Before this line existed, the second reference to a cross-file
        // user function would:
        //   1. find the function in function_table (registered above on
        //      the first reference)
        //   2. NOT find an arity in function_arity_table
        //   3. fall through to the "raw function pointer" path at line ~7250
        //   4. produce a CALLABLE tagged value whose data field is the raw
        //      code address instead of a heap pointer to a closure header
        // and the next dispatch would dereference the code address as if it
        // were a closure → SEGV at a mangled-looking address.
        //
        // Repro:
        //   (load "lib.esk")           ; defines f
        //   (define a f) (define b f)  ; first wraps OK; second was raw
        //   (a x)  → OK
        //   (b x)  → SEGV  before this fix
        //
        // See tests/bug_BB_minimal_xfile_indirector.esk.
        if (arity > 0 || function_arity_table.find(func_name) == function_arity_table.end()) {
            function_arity_table[func_name] = arity;
            if (jit_symbol_name != func_name) {
                function_arity_table[jit_symbol_name] = arity;
            }
            if (actual_func_name != func_name && actual_func_name != jit_symbol_name) {
                function_arity_table[actual_func_name] = arity;
            }
        }

        return extern_func;
    }

#endif
