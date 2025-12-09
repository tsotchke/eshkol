/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * HashCodegen - Hash table code generation implementation
 */

#include <eshkol/backend/hash_codegen.h>
#include <eshkol/eshkol.h>
#include <eshkol/logger.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Constants.h>

namespace eshkol {

HashCodegen::HashCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged, MemoryCodegen& mem,
                         std::unordered_map<std::string, llvm::Function*>& func_table)
    : ctx_(ctx), tagged_(tagged), mem_(mem), function_table_(func_table),
      codegen_ast_cb_(nullptr), codegen_typed_ast_cb_(nullptr), callback_context_(nullptr),
      hash_table_create_func_(nullptr), hash_table_set_func_(nullptr),
      hash_table_get_func_(nullptr), hash_table_has_key_func_(nullptr),
      hash_table_remove_func_(nullptr), hash_table_keys_func_(nullptr),
      hash_table_values_func_(nullptr), hash_table_count_func_(nullptr),
      hash_table_clear_func_(nullptr)
{
    initRuntimeFunctions();
}

void HashCodegen::setCodegenCallbacks(CodegenASTCallback ast_cb, CodegenTypedASTCallback typed_cb, void* ctx) {
    codegen_ast_cb_ = ast_cb;
    codegen_typed_ast_cb_ = typed_cb;
    callback_context_ = ctx;
}

llvm::Value* HashCodegen::codegenAST(const void* ast) {
    if (!codegen_ast_cb_ || !callback_context_) {
        eshkol_error("HashCodegen: No AST callback configured");
        return nullptr;
    }
    return codegen_ast_cb_(ast, callback_context_);
}

void HashCodegen::initRuntimeFunctions() {
    auto& context = ctx_.context();
    auto& module = ctx_.module();
    auto& builder = ctx_.builder();

    llvm::Type* ptr_type = llvm::PointerType::get(context, 0);
    llvm::Type* i64_type = llvm::Type::getInt64Ty(context);
    llvm::Type* i1_type = llvm::Type::getInt1Ty(context);

    // arena_hash_table_create(arena_t* arena) -> eshkol_hash_table_t*
    {
        llvm::FunctionType* ft = llvm::FunctionType::get(ptr_type, {ptr_type}, false);
        hash_table_create_func_ = llvm::Function::Create(
            ft, llvm::Function::ExternalLinkage, "arena_hash_table_create", module);
        function_table_["arena_hash_table_create"] = hash_table_create_func_;
    }

    // hash_table_set(arena_t*, eshkol_hash_table_t*, key*, value*) -> bool
    {
        llvm::FunctionType* ft = llvm::FunctionType::get(
            i1_type, {ptr_type, ptr_type, ptr_type, ptr_type}, false);
        hash_table_set_func_ = llvm::Function::Create(
            ft, llvm::Function::ExternalLinkage, "hash_table_set", module);
        function_table_["hash_table_set"] = hash_table_set_func_;
    }

    // hash_table_get(eshkol_hash_table_t*, key*, out_value*) -> bool
    {
        llvm::FunctionType* ft = llvm::FunctionType::get(
            i1_type, {ptr_type, ptr_type, ptr_type}, false);
        hash_table_get_func_ = llvm::Function::Create(
            ft, llvm::Function::ExternalLinkage, "hash_table_get", module);
        function_table_["hash_table_get"] = hash_table_get_func_;
    }

    // hash_table_has_key(eshkol_hash_table_t*, key*) -> bool
    {
        llvm::FunctionType* ft = llvm::FunctionType::get(i1_type, {ptr_type, ptr_type}, false);
        hash_table_has_key_func_ = llvm::Function::Create(
            ft, llvm::Function::ExternalLinkage, "hash_table_has_key", module);
        function_table_["hash_table_has_key"] = hash_table_has_key_func_;
    }

    // hash_table_remove(eshkol_hash_table_t*, key*) -> bool
    {
        llvm::FunctionType* ft = llvm::FunctionType::get(i1_type, {ptr_type, ptr_type}, false);
        hash_table_remove_func_ = llvm::Function::Create(
            ft, llvm::Function::ExternalLinkage, "hash_table_remove", module);
        function_table_["hash_table_remove"] = hash_table_remove_func_;
    }

    // hash_table_keys(arena_t*, eshkol_hash_table_t*) -> arena_tagged_cons_cell_t*
    {
        llvm::FunctionType* ft = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
        hash_table_keys_func_ = llvm::Function::Create(
            ft, llvm::Function::ExternalLinkage, "hash_table_keys", module);
        function_table_["hash_table_keys"] = hash_table_keys_func_;
    }

    // hash_table_values(arena_t*, eshkol_hash_table_t*) -> arena_tagged_cons_cell_t*
    {
        llvm::FunctionType* ft = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
        hash_table_values_func_ = llvm::Function::Create(
            ft, llvm::Function::ExternalLinkage, "hash_table_values", module);
        function_table_["hash_table_values"] = hash_table_values_func_;
    }

    // hash_table_count(eshkol_hash_table_t*) -> size_t
    {
        llvm::FunctionType* ft = llvm::FunctionType::get(i64_type, {ptr_type}, false);
        hash_table_count_func_ = llvm::Function::Create(
            ft, llvm::Function::ExternalLinkage, "hash_table_count", module);
        function_table_["hash_table_count"] = hash_table_count_func_;
    }

    // hash_table_clear(eshkol_hash_table_t*) -> void
    {
        llvm::Type* void_type = llvm::Type::getVoidTy(context);
        llvm::FunctionType* ft = llvm::FunctionType::get(void_type, {ptr_type}, false);
        hash_table_clear_func_ = llvm::Function::Create(
            ft, llvm::Function::ExternalLinkage, "hash_table_clear", module);
        function_table_["hash_table_clear"] = hash_table_clear_func_;
    }

    eshkol_debug("HashCodegen: Initialized runtime function declarations");
}

llvm::Value* HashCodegen::ensureTaggedValue(llvm::Value* val, const std::string& name) {
    auto& builder = ctx_.builder();
    auto& context = ctx_.context();

    // If already a tagged value struct, return as-is
    if (val->getType() == ctx_.taggedValueType()) {
        return val;
    }

    // If it's a raw i64 (integer literal), pack it as INT64
    if (val->getType()->isIntegerTy(64)) {
        return tagged_.packInt64(val, true);  // exact integer
    }

    // If it's a raw double (float literal), pack it as DOUBLE
    if (val->getType()->isDoubleTy()) {
        return tagged_.packDouble(val);
    }

    // If it's a raw i1 (boolean), pack it as BOOL
    if (val->getType()->isIntegerTy(1)) {
        return tagged_.packBool(val);
    }

    // If it's a pointer (e.g., string), it should already be tagged
    // but just in case, pack it as a generic pointer
    if (val->getType()->isPointerTy()) {
        return tagged_.packPtr(val, ESHKOL_VALUE_CONS_PTR);  // default to cons ptr
    }

    // For any other integer types, extend to i64 and pack
    if (val->getType()->isIntegerTy()) {
        llvm::Value* extended = builder.CreateSExtOrTrunc(val, ctx_.int64Type());
        return tagged_.packInt64(extended, true);
    }

    // If it's a float, convert to double and pack
    if (val->getType()->isFloatTy()) {
        llvm::Value* as_double = builder.CreateFPExt(val, ctx_.doubleType());
        return tagged_.packDouble(as_double);
    }

    eshkol_warn("HashCodegen::ensureTaggedValue: unhandled type for %s", name.c_str());
    return val;
}

llvm::Value* HashCodegen::extractTaggedValuePtr(llvm::Value* tagged_val, const std::string& name) {
    auto& builder = ctx_.builder();
    auto& context = ctx_.context();

    // First ensure the value is a tagged value struct
    llvm::Value* ensured_val = ensureTaggedValue(tagged_val, name);

    // Allocate space for the tagged value on the stack
    llvm::Value* alloca = builder.CreateAlloca(ctx_.taggedValueType(), nullptr, name + "_alloca");

    // Store the tagged value
    builder.CreateStore(ensured_val, alloca);

    return alloca;
}

// make-hash-table: Create a new hash table
// (make-hash-table) => hash-table
llvm::Value* HashCodegen::makeHashTable(const eshkol_operations_t* op) {
    auto& builder = ctx_.builder();
    auto& context = ctx_.context();

    // Get arena pointer
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(context, 0), ctx_.globalArena());

    // Call arena_hash_table_create(arena)
    llvm::Value* table_ptr = builder.CreateCall(hash_table_create_func_, {arena_ptr});

    // Pack as HASH_PTR tagged value
    return tagged_.packPtr(table_ptr, ESHKOL_VALUE_HASH_PTR);
}

// hash-set!: Set a key-value pair in the hash table
// (hash-set! table key value) => table
llvm::Value* HashCodegen::hashSet(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 3) {
        eshkol_warn("hash-set! requires exactly 3 arguments (table key value)");
        return nullptr;
    }

    auto& builder = ctx_.builder();
    auto& context = ctx_.context();

    // Get arguments
    llvm::Value* table_arg = codegenAST(&op->call_op.variables[0]);
    llvm::Value* key_arg = codegenAST(&op->call_op.variables[1]);
    llvm::Value* value_arg = codegenAST(&op->call_op.variables[2]);
    if (!table_arg || !key_arg || !value_arg) return nullptr;

    // Extract table pointer
    llvm::Value* table_ptr = tagged_.unpackPtr(table_arg);

    // Get arena pointer
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(context, 0), ctx_.globalArena());

    // Store key and value tagged values on stack for passing pointers
    llvm::Value* key_ptr = extractTaggedValuePtr(key_arg, "hash_key");
    llvm::Value* value_ptr = extractTaggedValuePtr(value_arg, "hash_value");

    // Call hash_table_set(arena, table, &key, &value)
    builder.CreateCall(hash_table_set_func_, {arena_ptr, table_ptr, key_ptr, value_ptr});

    // Return the table (for chaining)
    return table_arg;
}

// hash-ref: Get value by key
// (hash-ref table key) => value or #f if not found
// (hash-ref table key default) => value or default if not found
llvm::Value* HashCodegen::hashRef(const eshkol_operations_t* op) {
    if (op->call_op.num_vars < 2 || op->call_op.num_vars > 3) {
        eshkol_warn("hash-ref requires 2 or 3 arguments (table key [default])");
        return nullptr;
    }

    auto& builder = ctx_.builder();
    auto& context = ctx_.context();

    // Get arguments
    llvm::Value* table_arg = codegenAST(&op->call_op.variables[0]);
    llvm::Value* key_arg = codegenAST(&op->call_op.variables[1]);
    if (!table_arg || !key_arg) return nullptr;

    // Extract table pointer
    llvm::Value* table_ptr = tagged_.unpackPtr(table_arg);

    // Store key on stack
    llvm::Value* key_ptr = extractTaggedValuePtr(key_arg, "hash_ref_key");

    // Allocate space for output value
    llvm::Value* out_value_ptr = builder.CreateAlloca(ctx_.taggedValueType(), nullptr, "hash_ref_out");

    // Call hash_table_get(table, &key, &out_value)
    llvm::Value* found = builder.CreateCall(hash_table_get_func_, {table_ptr, key_ptr, out_value_ptr});

    // Create basic blocks for branching
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* found_block = llvm::BasicBlock::Create(context, "hash_found", current_func);
    llvm::BasicBlock* not_found_block = llvm::BasicBlock::Create(context, "hash_not_found", current_func);
    llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(context, "hash_merge", current_func);

    builder.CreateCondBr(found, found_block, not_found_block);

    // Found case: return the value
    builder.SetInsertPoint(found_block);
    llvm::Value* found_val = builder.CreateLoad(ctx_.taggedValueType(), out_value_ptr);
    builder.CreateBr(merge_block);

    // Not found case: return default or #f
    builder.SetInsertPoint(not_found_block);
    llvm::Value* not_found_val;
    if (op->call_op.num_vars == 3) {
        // Use provided default - ensure it's a tagged value
        llvm::Value* default_raw = codegenAST(&op->call_op.variables[2]);
        if (!default_raw) {
            not_found_val = tagged_.packBool(
                llvm::ConstantInt::getFalse(context));
        } else {
            // Ensure the default value is a tagged value (may be raw i64/double from literal)
            not_found_val = ensureTaggedValue(default_raw, "hash_ref_default");
        }
    } else {
        // Return #f
        not_found_val = tagged_.packBool(llvm::ConstantInt::getFalse(context));
    }
    builder.CreateBr(merge_block);
    llvm::BasicBlock* not_found_exit = builder.GetInsertBlock();

    // Merge
    builder.SetInsertPoint(merge_block);
    llvm::PHINode* phi = builder.CreatePHI(ctx_.taggedValueType(), 2);
    phi->addIncoming(found_val, found_block);
    phi->addIncoming(not_found_val, not_found_exit);

    return phi;
}

// hash-has-key?: Check if key exists
// (hash-has-key? table key) => #t or #f
llvm::Value* HashCodegen::hashHasKey(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 2) {
        eshkol_warn("hash-has-key? requires exactly 2 arguments (table key)");
        return nullptr;
    }

    auto& builder = ctx_.builder();
    auto& context = ctx_.context();

    // Get arguments
    llvm::Value* table_arg = codegenAST(&op->call_op.variables[0]);
    llvm::Value* key_arg = codegenAST(&op->call_op.variables[1]);
    if (!table_arg || !key_arg) return nullptr;

    // Extract table pointer
    llvm::Value* table_ptr = tagged_.unpackPtr(table_arg);

    // Store key on stack
    llvm::Value* key_ptr = extractTaggedValuePtr(key_arg, "hash_has_key");

    // Call hash_table_has_key(table, &key)
    llvm::Value* has_key = builder.CreateCall(hash_table_has_key_func_, {table_ptr, key_ptr});

    return tagged_.packBool(has_key);
}

// hash-remove!: Remove a key from the hash table
// (hash-remove! table key) => #t if removed, #f if not found
llvm::Value* HashCodegen::hashRemove(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 2) {
        eshkol_warn("hash-remove! requires exactly 2 arguments (table key)");
        return nullptr;
    }

    auto& builder = ctx_.builder();
    auto& context = ctx_.context();

    // Get arguments
    llvm::Value* table_arg = codegenAST(&op->call_op.variables[0]);
    llvm::Value* key_arg = codegenAST(&op->call_op.variables[1]);
    if (!table_arg || !key_arg) return nullptr;

    // Extract table pointer
    llvm::Value* table_ptr = tagged_.unpackPtr(table_arg);

    // Store key on stack
    llvm::Value* key_ptr = extractTaggedValuePtr(key_arg, "hash_remove_key");

    // Call hash_table_remove(table, &key)
    llvm::Value* removed = builder.CreateCall(hash_table_remove_func_, {table_ptr, key_ptr});

    return tagged_.packBool(removed);
}

// hash-keys: Get all keys as a list
// (hash-keys table) => '(key1 key2 ...)
llvm::Value* HashCodegen::hashKeys(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) {
        eshkol_warn("hash-keys requires exactly 1 argument (table)");
        return nullptr;
    }

    auto& builder = ctx_.builder();
    auto& context = ctx_.context();

    // Get table argument
    llvm::Value* table_arg = codegenAST(&op->call_op.variables[0]);
    if (!table_arg) return nullptr;

    // Extract table pointer
    llvm::Value* table_ptr = tagged_.unpackPtr(table_arg);

    // Get arena pointer
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(context, 0), ctx_.globalArena());

    // Call hash_table_keys(arena, table)
    llvm::Value* keys_ptr = builder.CreateCall(hash_table_keys_func_, {arena_ptr, table_ptr});

    // Check if null (empty)
    llvm::Value* is_null = builder.CreateICmpEQ(keys_ptr,
        llvm::ConstantPointerNull::get(llvm::PointerType::get(context, 0)));

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* null_block = llvm::BasicBlock::Create(context, "keys_null", current_func);
    llvm::BasicBlock* valid_block = llvm::BasicBlock::Create(context, "keys_valid", current_func);
    llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(context, "keys_merge", current_func);

    builder.CreateCondBr(is_null, null_block, valid_block);

    // Null case: return null (empty list)
    builder.SetInsertPoint(null_block);
    llvm::Value* null_val = tagged_.packNull();
    builder.CreateBr(merge_block);

    // Valid case: pack as CONS_PTR
    builder.SetInsertPoint(valid_block);
    llvm::Value* list_val = tagged_.packPtr(keys_ptr, ESHKOL_VALUE_CONS_PTR);
    builder.CreateBr(merge_block);

    // Merge
    builder.SetInsertPoint(merge_block);
    llvm::PHINode* phi = builder.CreatePHI(ctx_.taggedValueType(), 2);
    phi->addIncoming(null_val, null_block);
    phi->addIncoming(list_val, valid_block);

    return phi;
}

// hash-values: Get all values as a list
// (hash-values table) => '(value1 value2 ...)
llvm::Value* HashCodegen::hashValues(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) {
        eshkol_warn("hash-values requires exactly 1 argument (table)");
        return nullptr;
    }

    auto& builder = ctx_.builder();
    auto& context = ctx_.context();

    // Get table argument
    llvm::Value* table_arg = codegenAST(&op->call_op.variables[0]);
    if (!table_arg) return nullptr;

    // Extract table pointer
    llvm::Value* table_ptr = tagged_.unpackPtr(table_arg);

    // Get arena pointer
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(context, 0), ctx_.globalArena());

    // Call hash_table_values(arena, table)
    llvm::Value* values_ptr = builder.CreateCall(hash_table_values_func_, {arena_ptr, table_ptr});

    // Check if null (empty)
    llvm::Value* is_null = builder.CreateICmpEQ(values_ptr,
        llvm::ConstantPointerNull::get(llvm::PointerType::get(context, 0)));

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* null_block = llvm::BasicBlock::Create(context, "values_null", current_func);
    llvm::BasicBlock* valid_block = llvm::BasicBlock::Create(context, "values_valid", current_func);
    llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(context, "values_merge", current_func);

    builder.CreateCondBr(is_null, null_block, valid_block);

    // Null case: return null (empty list)
    builder.SetInsertPoint(null_block);
    llvm::Value* null_val = tagged_.packNull();
    builder.CreateBr(merge_block);

    // Valid case: pack as CONS_PTR
    builder.SetInsertPoint(valid_block);
    llvm::Value* list_val = tagged_.packPtr(values_ptr, ESHKOL_VALUE_CONS_PTR);
    builder.CreateBr(merge_block);

    // Merge
    builder.SetInsertPoint(merge_block);
    llvm::PHINode* phi = builder.CreatePHI(ctx_.taggedValueType(), 2);
    phi->addIncoming(null_val, null_block);
    phi->addIncoming(list_val, valid_block);

    return phi;
}

// hash-count: Get number of entries
// (hash-count table) => integer
llvm::Value* HashCodegen::hashCount(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) {
        eshkol_warn("hash-count requires exactly 1 argument (table)");
        return nullptr;
    }

    auto& builder = ctx_.builder();
    auto& context = ctx_.context();

    // Get table argument
    llvm::Value* table_arg = codegenAST(&op->call_op.variables[0]);
    if (!table_arg) return nullptr;

    // Extract table pointer
    llvm::Value* table_ptr = tagged_.unpackPtr(table_arg);

    // Call hash_table_count(table)
    llvm::Value* count = builder.CreateCall(hash_table_count_func_, {table_ptr});

    return tagged_.packInt64(count, true);
}

// hash-clear!: Clear all entries
// (hash-clear! table) => table
llvm::Value* HashCodegen::hashClear(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) {
        eshkol_warn("hash-clear! requires exactly 1 argument (table)");
        return nullptr;
    }

    auto& builder = ctx_.builder();
    auto& context = ctx_.context();

    // Get table argument
    llvm::Value* table_arg = codegenAST(&op->call_op.variables[0]);
    if (!table_arg) return nullptr;

    // Extract table pointer
    llvm::Value* table_ptr = tagged_.unpackPtr(table_arg);

    // Call hash_table_clear(table)
    builder.CreateCall(hash_table_clear_func_, {table_ptr});

    // Return the table
    return table_arg;
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
