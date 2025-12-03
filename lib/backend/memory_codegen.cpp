/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * MemoryCodegen implementation - Arena function declarations
 */

#include <eshkol/backend/memory_codegen.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

namespace eshkol {

MemoryCodegen::MemoryCodegen(llvm::Module& mod, TypeSystem& ts)
    : module(mod), types(ts) {
    createCoreArenaFunctions();
    createConsCellFunctions();
    createClosureFunctions();
    createTaggedConsGetters();
    createTaggedConsSetters();
    createTapeFunctions();
    createAdNodeFunctions();
}

llvm::Function* MemoryCodegen::createFunc(const char* name, llvm::FunctionType* ft) {
    return llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, &module);
}

void MemoryCodegen::createCoreArenaFunctions() {
    auto ptr = types.getPtrType();
    auto i64 = types.getInt64Type();
    auto voidTy = types.getVoidType();

    // arena_create: arena_t* (size_t)
    arena_create = createFunc("arena_create",
        llvm::FunctionType::get(ptr, {i64}, false));

    // arena_destroy: void (arena_t*)
    arena_destroy = createFunc("arena_destroy",
        llvm::FunctionType::get(voidTy, {ptr}, false));

    // arena_allocate: void* (arena_t*, size_t)
    arena_allocate = createFunc("arena_allocate",
        llvm::FunctionType::get(ptr, {ptr, i64}, false));

    // arena_push_scope: void (arena_t*)
    arena_push_scope = createFunc("arena_push_scope",
        llvm::FunctionType::get(voidTy, {ptr}, false));

    // arena_pop_scope: void (arena_t*)
    arena_pop_scope = createFunc("arena_pop_scope",
        llvm::FunctionType::get(voidTy, {ptr}, false));
}

void MemoryCodegen::createConsCellFunctions() {
    auto ptr = types.getPtrType();

    // arena_allocate_cons_cell: arena_cons_cell_t* (arena_t*)
    arena_allocate_cons_cell = createFunc("arena_allocate_cons_cell",
        llvm::FunctionType::get(ptr, {ptr}, false));

    // arena_allocate_tagged_cons_cell: arena_tagged_cons_cell_t* (arena_t*)
    arena_allocate_tagged_cons_cell = createFunc("arena_allocate_tagged_cons_cell",
        llvm::FunctionType::get(ptr, {ptr}, false));
}

void MemoryCodegen::createClosureFunctions() {
    auto ptr = types.getPtrType();
    auto i64 = types.getInt64Type();

    // arena_allocate_closure: eshkol_closure_t* (arena_t*, uint64_t func_ptr, size_t num_captures, uint64_t sexpr_ptr)
    arena_allocate_closure = createFunc("arena_allocate_closure",
        llvm::FunctionType::get(ptr, {ptr, i64, i64, i64}, false));
}

void MemoryCodegen::createTaggedConsGetters() {
    auto ptr = types.getPtrType();
    auto i64 = types.getInt64Type();
    auto i8 = types.getInt8Type();
    auto i1 = types.getInt1Type();
    auto dbl = types.getDoubleType();
    auto taggedValueTy = types.getTaggedValueType();

    // arena_tagged_cons_get_int64: int64_t (cell*, bool is_cdr)
    tagged_cons_get_int64 = createFunc("arena_tagged_cons_get_int64",
        llvm::FunctionType::get(i64, {ptr, i1}, false));

    // arena_tagged_cons_get_double: double (cell*, bool is_cdr)
    tagged_cons_get_double = createFunc("arena_tagged_cons_get_double",
        llvm::FunctionType::get(dbl, {ptr, i1}, false));

    // arena_tagged_cons_get_ptr: uint64_t (cell*, bool is_cdr)
    tagged_cons_get_ptr = createFunc("arena_tagged_cons_get_ptr",
        llvm::FunctionType::get(i64, {ptr, i1}, false));

    // arena_tagged_cons_get_type: uint8_t (cell*, bool is_cdr)
    tagged_cons_get_type = createFunc("arena_tagged_cons_get_type",
        llvm::FunctionType::get(i8, {ptr, i1}, false));

    // arena_tagged_cons_get_flags: uint8_t (cell*, bool is_cdr)
    tagged_cons_get_flags = createFunc("arena_tagged_cons_get_flags",
        llvm::FunctionType::get(i8, {ptr, i1}, false));

    // arena_tagged_cons_get_tagged_value: eshkol_tagged_value_t (cell*, bool is_cdr)
    tagged_cons_get_tagged_value = createFunc("arena_tagged_cons_get_tagged_value",
        llvm::FunctionType::get(taggedValueTy, {ptr, i1}, false));
}

void MemoryCodegen::createTaggedConsSetters() {
    auto ptr = types.getPtrType();
    auto i64 = types.getInt64Type();
    auto i8 = types.getInt8Type();
    auto i1 = types.getInt1Type();
    auto dbl = types.getDoubleType();
    auto voidTy = types.getVoidType();
    auto taggedValueTy = types.getTaggedValueType();

    // arena_tagged_cons_set_int64: void (cell*, bool is_cdr, int64_t value, uint8_t type)
    tagged_cons_set_int64 = createFunc("arena_tagged_cons_set_int64",
        llvm::FunctionType::get(voidTy, {ptr, i1, i64, i8}, false));

    // arena_tagged_cons_set_double: void (cell*, bool is_cdr, double value, uint8_t type)
    tagged_cons_set_double = createFunc("arena_tagged_cons_set_double",
        llvm::FunctionType::get(voidTy, {ptr, i1, dbl, i8}, false));

    // arena_tagged_cons_set_ptr: void (cell*, bool is_cdr, uint64_t ptr, uint8_t type)
    tagged_cons_set_ptr = createFunc("arena_tagged_cons_set_ptr",
        llvm::FunctionType::get(voidTy, {ptr, i1, i64, i8}, false));

    // arena_tagged_cons_set_null: void (cell*, bool is_cdr)
    tagged_cons_set_null = createFunc("arena_tagged_cons_set_null",
        llvm::FunctionType::get(voidTy, {ptr, i1}, false));

    // arena_tagged_cons_set_tagged_value: void (cell*, bool is_cdr, const eshkol_tagged_value_t* value)
    tagged_cons_set_tagged_value = createFunc("arena_tagged_cons_set_tagged_value",
        llvm::FunctionType::get(voidTy, {ptr, i1, ptr}, false));
}

void MemoryCodegen::createTapeFunctions() {
    auto ptr = types.getPtrType();
    auto i64 = types.getInt64Type();
    auto voidTy = types.getVoidType();

    // arena_allocate_tape: ad_tape_t* (arena_t*, size_t capacity)
    arena_allocate_tape = createFunc("arena_allocate_tape",
        llvm::FunctionType::get(ptr, {ptr, i64}, false));

    // arena_tape_add_node: size_t (ad_tape_t*, ad_node_t*)
    arena_tape_add_node = createFunc("arena_tape_add_node",
        llvm::FunctionType::get(i64, {ptr, ptr}, false));

    // arena_tape_reset: void (ad_tape_t*)
    arena_tape_reset = createFunc("arena_tape_reset",
        llvm::FunctionType::get(voidTy, {ptr}, false));

    // arena_tape_get_node: ad_node_t* (ad_tape_t*, size_t index)
    arena_tape_get_node = createFunc("arena_tape_get_node",
        llvm::FunctionType::get(ptr, {ptr, i64}, false));

    // arena_tape_get_node_count: size_t (ad_tape_t*)
    arena_tape_get_node_count = createFunc("arena_tape_get_node_count",
        llvm::FunctionType::get(i64, {ptr}, false));
}

void MemoryCodegen::createAdNodeFunctions() {
    auto ptr = types.getPtrType();

    // arena_allocate_ad_node: ad_node_t* (arena_t*)
    arena_allocate_ad_node = createFunc("arena_allocate_ad_node",
        llvm::FunctionType::get(ptr, {ptr}, false));
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
