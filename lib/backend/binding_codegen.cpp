/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * BindingCodegen - Variable binding code generation
 */

#include <eshkol/backend/binding_codegen.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/logger.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>

using namespace llvm;

namespace eshkol {

BindingCodegen::BindingCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged)
    : ctx_(ctx), tagged_(tagged) {}

// Helper to get current function
static Function* getCurrentFunction(Function** ptr) {
    return ptr ? *ptr : nullptr;
}

// Helper to check REPL mode
static bool isReplMode(bool* ptr) {
    return ptr ? *ptr : false;
}

Value* BindingCodegen::ensureTaggedValue(Value* value, eshkol_value_type_t value_type) {
    if (!value) return nullptr;

    // Already a tagged_value
    if (value->getType() == ctx_.taggedValueType()) {
        return value;
    }

    // Handle by value type
    switch (value_type) {
        case ESHKOL_VALUE_DOUBLE:
            if (value->getType()->isDoubleTy()) {
                return tagged_.packDouble(value);
            }
            break;

        case ESHKOL_VALUE_BOOL:
            if (value->getType()->isIntegerTy(1)) {
                return tagged_.packBool(value);
            } else if (value->getType()->isIntegerTy()) {
                Value* as_bool = ctx_.builder().CreateTrunc(value, ctx_.builder().getInt1Ty());
                return tagged_.packBool(as_bool);
            }
            break;

        case ESHKOL_VALUE_NULL:
            return tagged_.packNull();

        case ESHKOL_VALUE_CHAR:
            return tagged_.packChar(value);

        case ESHKOL_VALUE_INT64:
            if (value->getType()->isIntegerTy(64)) {
                return tagged_.packInt64(value, true);
            } else if (value->getType()->isIntegerTy()) {
                Value* as_i64 = ctx_.builder().CreateSExtOrTrunc(value, ctx_.int64Type());
                return tagged_.packInt64(as_i64, true);
            }
            break;

        default:
            // For pointer types (CONS_PTR, STRING_PTR, ports, etc.)
            // Use packInt64WithType to preserve the type tag
            if (value->getType()->isIntegerTy(64)) {
                return tagged_.packInt64WithType(value, value_type);
            } else if (value->getType()->isPointerTy()) {
                Value* as_int = ctx_.builder().CreatePtrToInt(value, ctx_.int64Type());
                return tagged_.packInt64WithType(as_int, value_type);
            } else if (value->getType()->isIntegerTy()) {
                Value* as_i64 = ctx_.builder().CreateZExtOrTrunc(value, ctx_.int64Type());
                return tagged_.packInt64WithType(as_i64, value_type);
            }
            break;
    }

    // Fallback: pack as INT64
    eshkol_warn("ensureTaggedValue: fallback to INT64 for type %d", (int)value_type);
    if (value->getType()->isIntegerTy(64)) {
        return tagged_.packInt64(value, true);
    } else if (value->getType()->isIntegerTy()) {
        Value* as_i64 = ctx_.builder().CreateZExtOrTrunc(value, ctx_.int64Type());
        return tagged_.packInt64(as_i64, true);
    } else if (value->getType()->isPointerTy()) {
        Value* as_int = ctx_.builder().CreatePtrToInt(value, ctx_.int64Type());
        return tagged_.packInt64(as_int, true);
    }

    return value;
}

Value* BindingCodegen::storeBinding(
    const std::string& name,
    Value* value,
    eshkol_value_type_t value_type,
    bool is_global
) {
    // Convert to tagged_value
    Value* tagged_val = ensureTaggedValue(value, value_type);
    if (!tagged_val) return nullptr;

    if (is_global) {
        // Create or get GlobalVariable
        GlobalVariable* gv = ctx_.module().getNamedGlobal(name);
        if (!gv) {
            // Create zero-initialized tagged_value global
            Constant* zero_init = ConstantAggregateZero::get(ctx_.taggedValueType());
            gv = new GlobalVariable(
                ctx_.module(),
                ctx_.taggedValueType(),
                false,  // not constant
                GlobalValue::ExternalLinkage,
                zero_init,
                name
            );
            gv->setAlignment(Align(16));
        }

        // Store the value
        ctx_.builder().CreateStore(tagged_val, gv);

        // Register in symbol tables
        if (symbol_table_) (*symbol_table_)[name] = gv;
        if (global_symbol_table_) (*global_symbol_table_)[name] = gv;

        eshkol_debug("BindingCodegen: stored global %s (type=%d)", name.c_str(), (int)value_type);
        return gv;
    } else {
        // Create AllocaInst
        AllocaInst* alloca = ctx_.builder().CreateAlloca(
            ctx_.taggedValueType(),
            nullptr,
            name
        );
        alloca->setAlignment(Align(16));

        // Store the value
        ctx_.builder().CreateStore(tagged_val, alloca);

        // Register in local symbol table
        if (symbol_table_) (*symbol_table_)[name] = alloca;

        eshkol_debug("BindingCodegen: stored local %s (type=%d)", name.c_str(), (int)value_type);
        return alloca;
    }
}

void BindingCodegen::registerLambdaBinding(const std::string& var_name, const std::string& lambda_name) {
    if (!function_table_) return;

    auto it = function_table_->find(lambda_name);
    if (it == function_table_->end()) return;

    Function* lambda_func = it->second;
    std::string func_key = var_name + "_func";
    Function* current = getCurrentFunction(current_function_);

    // NESTED DEFINE SCOPING FIX: For nested defines (inside a function), only store
    // in local symbol_table with scoped key. Do NOT add unscoped key to global_symbol_table
    // because it would be overwritten by other functions with same-named nested helpers.
    if (current) {
        // Inside a function - this is a nested define
        std::string scoped_key = current->getName().str() + "." + func_key;

        // Add to local symbol table with both scoped and unscoped keys
        // The unscoped key is for direct calls within the same function
        if (symbol_table_) {
            (*symbol_table_)[func_key] = lambda_func;
            (*symbol_table_)[scoped_key] = lambda_func;
        }

        // Only add SCOPED key to global - the unscoped would conflict with other functions
        if (global_symbol_table_) {
            (*global_symbol_table_)[scoped_key] = lambda_func;
        }

        eshkol_debug("BindingCodegen: registered nested lambda %s (scoped: %s) -> %s",
                     func_key.c_str(), scoped_key.c_str(), lambda_name.c_str());
    } else {
        // Top-level define - add to both tables
        if (symbol_table_) (*symbol_table_)[func_key] = lambda_func;
        if (global_symbol_table_) (*global_symbol_table_)[func_key] = lambda_func;

        eshkol_debug("BindingCodegen: registered top-level lambda %s -> %s",
                     func_key.c_str(), lambda_name.c_str());
    }
}

Value* BindingCodegen::define(const eshkol_operations_t* op) {
    if (!op) return nullptr;

    const char* var_name = op->define_op.name;
    if (!var_name) {
        eshkol_error("define: missing variable name");
        return nullptr;
    }

    // Evaluate the value expression
    if (!op->define_op.value) {
        eshkol_error("define: missing value for %s", var_name);
        return nullptr;
    }

    // Check if this is a lambda expression
    bool is_lambda = op->define_op.value->type == ESHKOL_OP &&
                     op->define_op.value->operation.op == ESHKOL_LAMBDA_OP;

    // Get typed value from callback
    if (!codegen_typed_ast_callback_ || !typed_to_tagged_callback_) {
        eshkol_error("define: callbacks not set");
        return nullptr;
    }

    void* typed_ptr = codegen_typed_ast_callback_(op->define_op.value, callback_context_);
    if (!typed_ptr) {
        eshkol_error("define: failed to evaluate value for %s", var_name);
        return nullptr;
    }

    // Check if this is a function value (LAMBDA_SEXPR type)
    bool is_func_value = false;
    if (get_typed_value_type_callback_) {
        int value_type = get_typed_value_type_callback_(typed_ptr, callback_context_);
        is_func_value = (value_type == ESHKOL_VALUE_LAMBDA_SEXPR ||
                         value_type == ESHKOL_VALUE_CLOSURE_PTR);
        if (is_func_value) {
            eshkol_debug("BindingCodegen::define: %s is a function value (type=%d)", var_name, value_type);
        }
    }

    // Convert TypedValue to tagged_value
    Value* tagged_val = static_cast<Value*>(typed_to_tagged_callback_(typed_ptr, callback_context_));
    if (!tagged_val) {
        eshkol_error("define: failed to convert value for %s", var_name);
        return nullptr;
    }

    // Register function binding ONLY for direct lambda definitions
    // For closures returned from function calls (like (apply-twice double)),
    // we should NOT register a _func entry because:
    // 1. The value is a runtime closure, not a compile-time function pointer
    // 2. The actual function to call depends on the closure's captured environment
    // 3. Calls to such variables must go through codegenClosureCall at runtime
    if (is_lambda && is_func_value && register_func_binding_callback_) {
        register_func_binding_callback_(var_name, typed_ptr, callback_context_);
    }

    // Determine if this should be global
    Function* current = getCurrentFunction(current_function_);
    bool is_global_init = current && current->getName() == "__global_init";
    bool is_repl = isReplMode(repl_mode_);
    bool is_main = current && current->getName() == "main";

    // Check if a global already exists for this variable (pre-declared in pass 1.5)
    GlobalVariable* existing_global = ctx_.module().getNamedGlobal(var_name);

    // Debug: log what function we're in
    if (current) {
        eshkol_debug("BindingCodegen::define: in function %s (global_init=%d, repl=%d, main=%d, existing=%d)",
                     current->getName().str().c_str(), is_global_init, is_repl, is_main,
                     existing_global != nullptr);
    } else {
        eshkol_debug("BindingCodegen::define: no current function (repl=%d)", is_repl);
    }

    // Top-level defines or REPL mode: use global
    // Nested defines inside functions: use local (alloca)
    // CRITICAL: If a global already exists (pre-declared), use it even in main
    bool use_global = !current || is_global_init || is_repl || (is_main && existing_global);

    // Store the binding
    if (use_global) {
        // Create or get GlobalVariable
        GlobalVariable* gv = ctx_.module().getNamedGlobal(var_name);
        if (!gv) {
            Constant* zero_init = ConstantAggregateZero::get(ctx_.taggedValueType());
            gv = new GlobalVariable(
                ctx_.module(),
                ctx_.taggedValueType(),
                false,
                GlobalValue::ExternalLinkage,
                zero_init,
                var_name
            );
            gv->setAlignment(Align(16));
        }

        ctx_.builder().CreateStore(tagged_val, gv);

        if (symbol_table_) (*symbol_table_)[var_name] = gv;
        if (global_symbol_table_) (*global_symbol_table_)[var_name] = gv;

        eshkol_debug("BindingCodegen::define: global %s", var_name);
    } else {
        // Local define (inside a function, not __global_init)
        AllocaInst* alloca = ctx_.builder().CreateAlloca(
            ctx_.taggedValueType(),
            nullptr,
            var_name
        );
        alloca->setAlignment(Align(16));

        ctx_.builder().CreateStore(tagged_val, alloca);

        if (symbol_table_) (*symbol_table_)[var_name] = alloca;

        eshkol_debug("BindingCodegen::define: local %s", var_name);
    }

    // Register lambda if applicable
    if (is_lambda && last_generated_lambda_name_ && !last_generated_lambda_name_->empty()) {
        registerLambdaBinding(var_name, *last_generated_lambda_name_);
    }

    return tagged_val;
}

Value* BindingCodegen::let(const eshkol_operations_t* op) {
    if (!op || !op->let_op.body) {
        eshkol_error("let: missing body");
        return nullptr;
    }

    if (!symbol_table_) {
        eshkol_error("let: symbol table not set");
        return nullptr;
    }

    eshkol_debug("BindingCodegen::let: %llu bindings", (unsigned long long)op->let_op.num_bindings);

    // Save current symbol table
    std::unordered_map<std::string, Value*> saved_symbols = *symbol_table_;

    // Process bindings
    for (uint64_t i = 0; i < op->let_op.num_bindings; i++) {
        const eshkol_ast_t* binding = &op->let_op.bindings[i];

        if (binding->type != ESHKOL_CONS || !binding->cons_cell.car || !binding->cons_cell.cdr) {
            eshkol_error("let: invalid binding structure");
            continue;
        }

        // Get variable name
        const eshkol_ast_t* var_ast = binding->cons_cell.car;
        if (var_ast->type != ESHKOL_VAR || !var_ast->variable.id) {
            eshkol_error("let: binding must have variable name");
            continue;
        }
        std::string var_name = var_ast->variable.id;

        // Check if value is a lambda
        const eshkol_ast_t* val_ast = binding->cons_cell.cdr;
        bool is_lambda = val_ast && val_ast->type == ESHKOL_OP &&
                         val_ast->operation.op == ESHKOL_LAMBDA_OP;

        // Evaluate value
        void* typed_ptr = codegen_typed_ast_callback_(val_ast, callback_context_);
        if (!typed_ptr) {
            eshkol_warn("let: failed to evaluate binding for %s", var_name.c_str());
            continue;
        }

        Value* tagged_val = static_cast<Value*>(typed_to_tagged_callback_(typed_ptr, callback_context_));
        if (!tagged_val) {
            eshkol_warn("let: failed to convert binding for %s", var_name.c_str());
            continue;
        }

        // Create alloca and store
        // TCO FIX: When TCO is enabled, allocas MUST be in entry block to avoid
        // stack growth on each loop iteration. This is safe because:
        // 1. The alloca is just stack space - it doesn't capture any value
        // 2. The store happens at the current position with the correct value
        // 3. Closure captures work because we fixed codegenVariable to load from pointers
        AllocaInst* alloca = nullptr;
        if (tco_context_.enabled) {
            // TCO path: Insert alloca in entry block
            Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
            BasicBlock& entry_block = current_func->getEntryBlock();
            IRBuilderBase::InsertPoint saved_ip = ctx_.builder().saveIP();
            ctx_.builder().SetInsertPoint(&entry_block, entry_block.begin());
            alloca = ctx_.builder().CreateAlloca(
                ctx_.taggedValueType(),
                nullptr,
                var_name
            );
            alloca->setAlignment(Align(16));
            ctx_.builder().restoreIP(saved_ip);
        } else {
            // Non-TCO path: Create alloca at current position (original behavior)
            alloca = ctx_.builder().CreateAlloca(
                ctx_.taggedValueType(),
                nullptr,
                var_name
            );
            alloca->setAlignment(Align(16));
        }
        // Store happens at current position regardless of where alloca is
        ctx_.builder().CreateStore(tagged_val, alloca);

        (*symbol_table_)[var_name] = alloca;

        // Register lambda if applicable
        if (is_lambda && last_generated_lambda_name_ && !last_generated_lambda_name_->empty()) {
            registerLambdaBinding(var_name, *last_generated_lambda_name_);
        }

        eshkol_debug("let: bound %s", var_name.c_str());
    }

    // Evaluate body
    Value* body_result = nullptr;
    if (codegen_ast_callback_) {
        body_result = static_cast<Value*>(codegen_ast_callback_(op->let_op.body, callback_context_));
    }

    // Preserve _func entries and GlobalVariable captures before restoring
    // GlobalVariable captures are created by codegenLambda when a lambda captures a variable.
    // The lambda updates symbol_table[var_name] = GlobalVariable, and we must preserve this
    // so that the enclosing scope sees the modified value after set! inside the lambda.
    std::unordered_map<std::string, Value*> entries_to_preserve;
    for (auto& entry : *symbol_table_) {
        // Preserve function references (_func entries)
        if (entry.first.length() > 5 &&
            entry.first.substr(entry.first.length() - 5) == "_func") {
            entries_to_preserve[entry.first] = entry.second;
            if (global_symbol_table_) {
                (*global_symbol_table_)[entry.first] = entry.second;
            }
        }
        // MUTABLE CAPTURE FIX: Preserve GlobalVariable entries for actual variables
        // These are variables that were upgraded to GlobalVariables when captured by lambdas
        // We need to preserve them so set! mutations are visible in the enclosing scope
        // IMPORTANT: Skip entries with "_capture_" in the name - those are internal capture storage
        // (e.g., "lambda_0_capture_x") and should NOT be preserved in the outer scope
        if (entry.second && isa<GlobalVariable>(entry.second) &&
            entry.first.find("_capture_") == std::string::npos) {
            entries_to_preserve[entry.first] = entry.second;
        }
    }

    // Restore symbol table
    *symbol_table_ = saved_symbols;

    // Re-add preserved entries (function references and GlobalVariable captures)
    for (auto& entry : entries_to_preserve) {
        (*symbol_table_)[entry.first] = entry.second;
    }

    eshkol_debug("let: completed, restored scope");

    return body_result ? body_result : ConstantInt::get(ctx_.int64Type(), 0);
}

Value* BindingCodegen::letrec(const eshkol_operations_t* op) {
    if (!op || !op->let_op.body) {
        eshkol_error("letrec: missing body");
        return nullptr;
    }

    if (!symbol_table_) {
        eshkol_error("letrec: symbol table not set");
        return nullptr;
    }

    eshkol_debug("BindingCodegen::letrec: %llu bindings", (unsigned long long)op->let_op.num_bindings);

    // Save current symbol table
    std::unordered_map<std::string, Value*> saved_symbols = *symbol_table_;

    // Collect binding info for analysis
    std::vector<std::string> var_names;
    std::vector<const eshkol_ast_t*> val_asts;
    std::vector<bool> is_lambda;

    for (uint64_t i = 0; i < op->let_op.num_bindings; i++) {
        const eshkol_ast_t* binding = &op->let_op.bindings[i];
        if (binding->type != ESHKOL_CONS || !binding->cons_cell.car) continue;

        const eshkol_ast_t* var_ast = binding->cons_cell.car;
        if (var_ast->type != ESHKOL_VAR || !var_ast->variable.id) continue;

        std::string var_name = var_ast->variable.id;
        const eshkol_ast_t* val_ast = binding->cons_cell.cdr;
        bool binding_is_lambda = val_ast && val_ast->type == ESHKOL_OP &&
                                  val_ast->operation.op == ESHKOL_LAMBDA_OP;

        var_names.push_back(var_name);
        val_asts.push_back(val_ast);
        is_lambda.push_back(binding_is_lambda);
    }

    // LETREC REFACTOR: Add all names to exclusion set so findFreeVariables won't capture them
    // But we DO add them to symbol_table so recursive calls can resolve
    if (letrec_excluded_capture_names_) {
        for (const std::string& name : var_names) {
            letrec_excluded_capture_names_->insert(name);
            eshkol_debug("letrec: added %s to exclusion set", name.c_str());
        }
    }

    // Phase 1: Create GlobalVariables for all bindings
    // GlobalVariables are used because nested lambdas need cross-function access.
    // NOTE: We do NOT create forward declarations for lambda bindings.
    // Forward declarations don't work correctly when lambdas have captures (different signatures).
    // Instead, recursive calls will use codegenClosureCall which loads from the GlobalVariable.
    std::vector<GlobalVariable*> globals;
    static int letrec_global_counter = 0;

    for (size_t i = 0; i < var_names.size(); i++) {
        const std::string& var_name = var_names[i];

        // Create global variable for storing the closure
        std::string global_name = "letrec_" + var_name + "_" + std::to_string(letrec_global_counter++);
        GlobalVariable* global = new GlobalVariable(
            ctx_.module(),
            ctx_.taggedValueType(),
            false,  // not constant
            GlobalValue::InternalLinkage,
            UndefValue::get(ctx_.taggedValueType()),
            global_name
        );

        globals.push_back(global);
        (*symbol_table_)[var_name] = global;
        eshkol_debug("letrec: created global %s for %s", global_name.c_str(), var_name.c_str());

        // NOTE: We deliberately do NOT register var_name_func here.
        // This forces codegenCall to use codegenClosureCall for recursive calls,
        // which correctly handles captures.
    }

    // Phase 2: Evaluate and store values
    // CRITICAL FIX: Lambda bindings must be evaluated and stored FIRST (for mutual recursion),
    // then non-lambda bindings are evaluated (which may reference the lambdas).
    //
    // Order matters because:
    // 1. Lambda bindings may call each other (mutual recursion via GlobalVariable lookup)
    // 2. Non-lambda bindings may contain lambdas that capture/call the lambda bindings
    //    (e.g., (define result (map (lambda (i) (sibling)) list)) where sibling is a lambda)
    // 3. Non-lambda bindings need to be stored immediately so subsequent bindings can reference them

    std::vector<Value*> lambda_values;
    std::vector<std::string> lambda_names;
    std::vector<size_t> lambda_indices;

    // Phase 2a: Evaluate ALL lambda bindings first
    for (size_t i = 0; i < var_names.size(); i++) {
        if (!is_lambda[i]) continue;  // Skip non-lambda bindings for now

        const eshkol_ast_t* val_ast = val_asts[i];
        if (!val_ast) continue;

        // TCO SETUP: Check if this lambda is self-tail-recursive
        bool use_tco = false;
        if (is_tail_recursive_callback_) {
            use_tco = is_tail_recursive_callback_(&val_ast->operation, var_names[i].c_str(), callback_context_);
            if (use_tco) {
                eshkol_debug("TCO: Enabling tail call optimization for letrec lambda %s", var_names[i].c_str());
                // Set up TCO context - the main codegen will use this during lambda generation
                tco_context_.func_name = var_names[i];
                tco_context_.enabled = true;
                tco_context_.param_allocas.clear();
                tco_context_.param_names.clear();
                tco_context_.loop_header = nullptr;  // Will be set during lambda body generation
            }
        }

        // Evaluate lambda (codegen will check TCO context)
        void* typed_ptr = codegen_typed_ast_callback_(val_ast, callback_context_);

        // Clear TCO context after lambda generation
        if (use_tco) {
            tco_context_.enabled = false;
            tco_context_.func_name = "";
        }

        if (!typed_ptr) {
            eshkol_warn("letrec: failed to evaluate lambda binding for %s", var_names[i].c_str());
            continue;
        }

        Value* tagged_val = static_cast<Value*>(typed_to_tagged_callback_(typed_ptr, callback_context_));
        lambda_values.push_back(tagged_val);
        lambda_indices.push_back(i);
        if (last_generated_lambda_name_ && !last_generated_lambda_name_->empty()) {
            lambda_names.push_back(*last_generated_lambda_name_);
        } else {
            lambda_names.push_back("");
        }
    }

    // Phase 2b: Store ALL lambda values (before evaluating non-lambdas)
    // This ensures non-lambda bindings that contain lambdas calling these lambdas work correctly
    for (size_t j = 0; j < lambda_values.size(); j++) {
        size_t i = lambda_indices[j];
        ctx_.builder().CreateStore(lambda_values[j], globals[i]);
        eshkol_debug("letrec: stored lambda %s", var_names[i].c_str());
    }

    // CRITICAL: Register lambda bindings IMMEDIATELY after storing values
    // This must happen BEFORE evaluating non-lambda bindings because those bindings
    // may contain operations (like gradient/derivative) that need to resolve the lambda functions.
    // Example: (define f (lambda ...)) (define grad (gradient f point))
    // When evaluating grad, resolveLambdaFunction needs f_func to be registered.
    for (size_t j = 0; j < lambda_indices.size(); j++) {
        size_t i = lambda_indices[j];
        if (!lambda_names[j].empty()) {
            registerLambdaBinding(var_names[i], lambda_names[j]);
            eshkol_debug("letrec: registered lambda binding %s -> %s", var_names[i].c_str(), lambda_names[j].c_str());
        }
    }

    // Phase 2c: Now evaluate and store non-lambda bindings
    // They can safely reference the already-stored lambda bindings
    for (size_t i = 0; i < var_names.size(); i++) {
        if (is_lambda[i]) continue;  // Already handled in phase 2a/2b

        const eshkol_ast_t* val_ast = val_asts[i];
        if (!val_ast) continue;

        // Evaluate value
        void* typed_ptr = codegen_typed_ast_callback_(val_ast, callback_context_);
        if (!typed_ptr) {
            eshkol_warn("letrec: failed to evaluate binding for %s", var_names[i].c_str());
            continue;
        }

        Value* tagged_val = static_cast<Value*>(typed_to_tagged_callback_(typed_ptr, callback_context_));

        // Store immediately so subsequent non-lambda bindings can reference it
        ctx_.builder().CreateStore(tagged_val, globals[i]);
        eshkol_debug("letrec: stored non-lambda %s", var_names[i].c_str());
    }

    // Phase 4: Clear exclusion set
    if (letrec_excluded_capture_names_) {
        for (const std::string& name : var_names) {
            letrec_excluded_capture_names_->erase(name);
        }
    }

    // Note: Lambda bindings were already registered in Phase 2b (before non-lambda evaluation)

    // Evaluate body
    Value* body_result = nullptr;
    if (codegen_ast_callback_) {
        body_result = static_cast<Value*>(codegen_ast_callback_(op->let_op.body, callback_context_));
    }

    // Preserve _func entries and GlobalVariable captures
    std::unordered_map<std::string, Value*> entries_to_preserve;
    for (auto& entry : *symbol_table_) {
        // Preserve function references (_func entries)
        if (entry.first.length() > 5 &&
            entry.first.substr(entry.first.length() - 5) == "_func") {
            entries_to_preserve[entry.first] = entry.second;
            if (global_symbol_table_) {
                (*global_symbol_table_)[entry.first] = entry.second;
            }
        }
        // MUTABLE CAPTURE FIX: Preserve GlobalVariable entries for actual variables
        // Skip entries with "_capture_" in the name - those are internal capture storage
        if (entry.second && isa<GlobalVariable>(entry.second) &&
            entry.first.find("_capture_") == std::string::npos) {
            entries_to_preserve[entry.first] = entry.second;
        }
    }

    // Restore symbol table
    *symbol_table_ = saved_symbols;

    // Re-add preserved entries
    for (auto& entry : entries_to_preserve) {
        (*symbol_table_)[entry.first] = entry.second;
    }

    eshkol_debug("letrec: completed, restored scope");

    return body_result ? body_result : ConstantInt::get(ctx_.int64Type(), 0);
}

Value* BindingCodegen::letStar(const eshkol_operations_t* op) {
    if (!op || !op->let_op.body) {
        eshkol_error("let*: missing body");
        return nullptr;
    }

    if (!symbol_table_) {
        eshkol_error("let*: symbol table not set");
        return nullptr;
    }

    eshkol_debug("BindingCodegen::let*: %llu bindings", (unsigned long long)op->let_op.num_bindings);

    // Save current symbol table
    std::unordered_map<std::string, Value*> saved_symbols = *symbol_table_;

    // Process bindings sequentially - each visible to subsequent
    for (uint64_t i = 0; i < op->let_op.num_bindings; i++) {
        const eshkol_ast_t* binding = &op->let_op.bindings[i];

        if (binding->type != ESHKOL_CONS || !binding->cons_cell.car || !binding->cons_cell.cdr) {
            eshkol_error("let*: invalid binding structure");
            continue;
        }

        const eshkol_ast_t* var_ast = binding->cons_cell.car;
        if (var_ast->type != ESHKOL_VAR || !var_ast->variable.id) {
            eshkol_error("let*: binding must have variable name");
            continue;
        }
        std::string var_name = var_ast->variable.id;

        const eshkol_ast_t* val_ast = binding->cons_cell.cdr;
        bool is_lambda = val_ast && val_ast->type == ESHKOL_OP &&
                         val_ast->operation.op == ESHKOL_LAMBDA_OP;

        // Evaluate value (can see previous bindings)
        void* typed_ptr = codegen_typed_ast_callback_(val_ast, callback_context_);
        if (!typed_ptr) {
            eshkol_warn("let*: failed to evaluate binding for %s", var_name.c_str());
            continue;
        }

        Value* tagged_val = static_cast<Value*>(typed_to_tagged_callback_(typed_ptr, callback_context_));
        if (!tagged_val) {
            eshkol_warn("let*: failed to convert binding for %s", var_name.c_str());
            continue;
        }

        // Create alloca and store
        AllocaInst* alloca = ctx_.builder().CreateAlloca(
            ctx_.taggedValueType(),
            nullptr,
            var_name
        );
        alloca->setAlignment(Align(16));
        ctx_.builder().CreateStore(tagged_val, alloca);

        // Add to symbol table immediately (visible to subsequent bindings)
        (*symbol_table_)[var_name] = alloca;

        // Register lambda if applicable
        if (is_lambda && last_generated_lambda_name_ && !last_generated_lambda_name_->empty()) {
            registerLambdaBinding(var_name, *last_generated_lambda_name_);
        }

        eshkol_debug("let*: bound %s", var_name.c_str());
    }

    // Evaluate body
    Value* body_result = nullptr;
    if (codegen_ast_callback_) {
        body_result = static_cast<Value*>(codegen_ast_callback_(op->let_op.body, callback_context_));
    }

    // Preserve _func entries and GlobalVariable captures
    std::unordered_map<std::string, Value*> entries_to_preserve;
    for (auto& entry : *symbol_table_) {
        // Preserve function references (_func entries)
        if (entry.first.length() > 5 &&
            entry.first.substr(entry.first.length() - 5) == "_func") {
            entries_to_preserve[entry.first] = entry.second;
            if (global_symbol_table_) {
                (*global_symbol_table_)[entry.first] = entry.second;
            }
        }
        // MUTABLE CAPTURE FIX: Preserve GlobalVariable entries for actual variables
        // Skip entries with "_capture_" in the name - those are internal capture storage
        if (entry.second && isa<GlobalVariable>(entry.second) &&
            entry.first.find("_capture_") == std::string::npos) {
            entries_to_preserve[entry.first] = entry.second;
        }
    }

    // Restore symbol table
    *symbol_table_ = saved_symbols;

    // Re-add preserved entries
    for (auto& entry : entries_to_preserve) {
        (*symbol_table_)[entry.first] = entry.second;
    }

    eshkol_debug("let*: completed, restored scope");

    return body_result ? body_result : ConstantInt::get(ctx_.int64Type(), 0);
}

Value* BindingCodegen::set(const eshkol_operations_t* op) {
    if (!op) return nullptr;

    const char* var_name = op->set_op.name;
    if (!var_name) {
        eshkol_error("set!: missing variable name");
        return nullptr;
    }

    // Look up the variable
    Value* storage = lookupVariable(var_name);
    if (!storage) {
        eshkol_error("set!: undefined variable %s", var_name);
        return nullptr;
    }

    // Evaluate the new value
    if (!op->set_op.value) {
        eshkol_error("set!: missing value for %s", var_name);
        return nullptr;
    }

    void* typed_ptr = codegen_typed_ast_callback_(op->set_op.value, callback_context_);
    if (!typed_ptr) {
        eshkol_error("set!: failed to evaluate value for %s", var_name);
        return nullptr;
    }

    Value* tagged_val = static_cast<Value*>(typed_to_tagged_callback_(typed_ptr, callback_context_));
    if (!tagged_val) {
        eshkol_error("set!: failed to convert value for %s", var_name);
        return nullptr;
    }

    // Store the new value
    ctx_.builder().CreateStore(tagged_val, storage);

    eshkol_debug("set!: updated %s", var_name);

    return tagged_val;
}

Value* BindingCodegen::lookupVariable(const std::string& name) {
    // Check local symbol table first
    if (symbol_table_) {
        auto it = symbol_table_->find(name);
        if (it != symbol_table_->end()) {
            return it->second;
        }
    }

    // Check global symbol table
    if (global_symbol_table_) {
        auto it = global_symbol_table_->find(name);
        if (it != global_symbol_table_->end()) {
            return it->second;
        }
    }

    // Check module globals
    GlobalVariable* gv = ctx_.module().getNamedGlobal(name);
    if (gv) {
        return gv;
    }

    return nullptr;
}

Value* BindingCodegen::loadVariable(const std::string& name) {
    Value* storage = lookupVariable(name);
    if (!storage) return nullptr;

    // Load the tagged_value
    return ctx_.builder().CreateLoad(ctx_.taggedValueType(), storage, name + ".val");
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
