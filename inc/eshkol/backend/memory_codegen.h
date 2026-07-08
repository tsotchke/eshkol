/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * MemoryCodegen - Arena memory management function declarations
 *
 * This module provides LLVM function declarations for the arena-based
 * memory management system used by Eshkol. All arena_* functions are
 * declared here and cached for use throughout code generation.
 */
#ifndef ESHKOL_BACKEND_MEMORY_CODEGEN_H
#define ESHKOL_BACKEND_MEMORY_CODEGEN_H

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/backend/type_system.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Function.h>

namespace eshkol {

/**
 * MemoryCodegen manages arena memory function declarations.
 *
 * All functions are declared in the constructor and cached as members.
 * This class provides the LLVM Function* pointers needed to generate
 * calls to the arena memory management runtime.
 */
class MemoryCodegen {
public:
    /**
     * Construct MemoryCodegen for the given module and type system.
     * Creates all arena function declarations.
     */
    MemoryCodegen(llvm::Module& mod, TypeSystem& ts);

    // Core arena management

    /** @brief Get the declared `arena_create` runtime function. */
    llvm::Function* getArenaCreate() const { return arena_create; }
    /** @brief Get the declared `get_global_arena` runtime function (returns the process-wide arena). */
    llvm::Function* getGlobalArena() const { return get_global_arena; }
    /** @brief Get the declared `arena_destroy` runtime function. */
    llvm::Function* getArenaDestroy() const { return arena_destroy; }
    /** @brief Get the declared `arena_allocate` runtime function (raw, header-less allocation). */
    llvm::Function* getArenaAllocate() const { return arena_allocate; }
    /** @brief Get the declared `arena_allocate_with_header` runtime function. */
    llvm::Function* getArenaAllocateWithHeader() const { return arena_allocate_with_header; }
    /** @brief Get the declared `arena_push_scope` runtime function (begin a nested allocation scope). */
    llvm::Function* getArenaPushScope() const { return arena_push_scope; }
    /** @brief Get the declared `arena_pop_scope` runtime function (release a nested allocation scope). */
    llvm::Function* getArenaPopScope() const { return arena_pop_scope; }

    // Cons cell allocation (legacy - no header)

    /** @brief Get the declared `arena_allocate_cons_cell` runtime function (legacy, no object header). */
    llvm::Function* getArenaAllocateConsCell() const { return arena_allocate_cons_cell; }
    /** @brief Get the declared `arena_allocate_tagged_cons_cell` runtime function (legacy, no object header). */
    llvm::Function* getArenaAllocateTaggedConsCell() const { return arena_allocate_tagged_cons_cell; }

    // Closure allocation (legacy - no header)

    /** @brief Get the declared `arena_allocate_closure` runtime function (legacy, no object header). */
    llvm::Function* getArenaAllocateClosure() const { return arena_allocate_closure; }

    // ─── NEW: Typed allocators with object headers (for consolidated types) ───

    /** @brief Get the declared `arena_allocate_cons_with_header` runtime function (HEAP_PTR + CONS subtype). */
    llvm::Function* getArenaAllocateConsWithHeader() const { return arena_allocate_cons_with_header; }
    /** @brief Get the declared `arena_allocate_string_with_header` runtime function (HEAP_PTR + STRING subtype). */
    llvm::Function* getArenaAllocateStringWithHeader() const { return arena_allocate_string_with_header; }
    /** @brief Get the declared `arena_allocate_vector_with_header` runtime function (HEAP_PTR + VECTOR subtype). */
    llvm::Function* getArenaAllocateVectorWithHeader() const { return arena_allocate_vector_with_header; }
    /** @brief Get the declared `arena_allocate_symbol_with_header` runtime function (HEAP_PTR + SYMBOL subtype). */
    llvm::Function* getArenaAllocateSymbolWithHeader() const { return arena_allocate_symbol_with_header; }
    /** @brief Get the declared `arena_allocate_closure_with_header` runtime function (CALLABLE + CLOSURE subtype). */
    llvm::Function* getArenaAllocateClosureWithHeader() const { return arena_allocate_closure_with_header; }

    // Tagged cons cell getters

    /** @brief Get the declared `tagged_cons_get_int64` runtime function. */
    llvm::Function* getTaggedConsGetInt64() const { return tagged_cons_get_int64; }
    /** @brief Get the declared `tagged_cons_get_double` runtime function. */
    llvm::Function* getTaggedConsGetDouble() const { return tagged_cons_get_double; }
    /** @brief Get the declared `tagged_cons_get_ptr` runtime function. */
    llvm::Function* getTaggedConsGetPtr() const { return tagged_cons_get_ptr; }
    /** @brief Get the declared `tagged_cons_get_type` runtime function. */
    llvm::Function* getTaggedConsGetType() const { return tagged_cons_get_type; }
    /** @brief Get the declared `tagged_cons_get_flags` runtime function. */
    llvm::Function* getTaggedConsGetFlags() const { return tagged_cons_get_flags; }
    /** @brief Get the declared `tagged_cons_get_tagged_value` runtime function. */
    llvm::Function* getTaggedConsGetTaggedValue() const { return tagged_cons_get_tagged_value; }

    // Tagged cons cell setters

    /** @brief Get the declared `tagged_cons_set_int64` runtime function. */
    llvm::Function* getTaggedConsSetInt64() const { return tagged_cons_set_int64; }
    /** @brief Get the declared `tagged_cons_set_double` runtime function. */
    llvm::Function* getTaggedConsSetDouble() const { return tagged_cons_set_double; }
    /** @brief Get the declared `tagged_cons_set_ptr` runtime function. */
    llvm::Function* getTaggedConsSetPtr() const { return tagged_cons_set_ptr; }
    /** @brief Get the declared `tagged_cons_set_null` runtime function. */
    llvm::Function* getTaggedConsSetNull() const { return tagged_cons_set_null; }
    /** @brief Get the declared `tagged_cons_set_tagged_value` runtime function. */
    llvm::Function* getTaggedConsSetTaggedValue() const { return tagged_cons_set_tagged_value; }

    // Tape management (for reverse-mode AD)

    /** @brief Get the declared `arena_allocate_tape` runtime function (allocate a reverse-mode AD tape). */
    llvm::Function* getArenaAllocateTape() const { return arena_allocate_tape; }
    /** @brief Get the declared `arena_tape_add_node` runtime function (append a node to the AD tape). */
    llvm::Function* getArenaTapeAddNode() const { return arena_tape_add_node; }
    /** @brief Get the declared `arena_tape_reset` runtime function (clear an AD tape for reuse). */
    llvm::Function* getArenaTapeReset() const { return arena_tape_reset; }
    /** @brief Get the declared `arena_tape_get_node` runtime function (index into an AD tape). */
    llvm::Function* getArenaTapeGetNode() const { return arena_tape_get_node; }
    /** @brief Get the declared `arena_tape_get_node_count` runtime function. */
    llvm::Function* getArenaTapeGetNodeCount() const { return arena_tape_get_node_count; }

    // AD node allocation

    /** @brief Get the declared `arena_allocate_ad_node` runtime function (legacy, no object header). */
    llvm::Function* getArenaAllocateAdNode() const { return arena_allocate_ad_node; }
    /** @brief Get the declared `arena_allocate_ad_node_with_header` runtime function (CALLABLE + AD_NODE subtype). */
    llvm::Function* getArenaAllocateAdNodeWithHeader() const { return arena_allocate_ad_node_with_header; }

    // Hash table allocation (with header for HEAP_PTR type)

    /** @brief Get the declared `arena_hash_table_create_with_header` runtime function (HEAP_PTR + HASH subtype). */
    llvm::Function* getArenaHashTableCreateWithHeader() const { return arena_hash_table_create_with_header; }

    // Tensor allocation (with header for HEAP_PTR type)

    /** @brief Get the declared `arena_allocate_tensor_with_header` runtime function (HEAP_PTR + TENSOR subtype). */
    llvm::Function* getArenaAllocateTensorWithHeader() const { return arena_allocate_tensor_with_header; }
    /** @brief Get the declared `arena_allocate_tensor_full` runtime function (allocate tensor struct + backing data in one call). */
    llvm::Function* getArenaAllocateTensorFull() const { return arena_allocate_tensor_full; }

    // Exception allocation (with header for HEAP_PTR type)

    /** @brief Get the declared `eshkol_make_exception_with_header` runtime function (HEAP_PTR + EXCEPTION subtype). */
    llvm::Function* getMakeExceptionWithHeader() const { return eshkol_make_exception_with_header; }

private:
    llvm::Module& module;
    TypeSystem& types;

    // Core arena management
    llvm::Function* arena_create;
    llvm::Function* get_global_arena;
    llvm::Function* arena_destroy;
    llvm::Function* arena_allocate;
    llvm::Function* arena_allocate_with_header;
    llvm::Function* arena_push_scope;
    llvm::Function* arena_pop_scope;

    // Cons cell allocation (legacy - no header)
    llvm::Function* arena_allocate_cons_cell;
    llvm::Function* arena_allocate_tagged_cons_cell;

    // Closure allocation (legacy - no header)
    llvm::Function* arena_allocate_closure;

    // NEW: Typed allocators with object headers (for consolidated types)
    llvm::Function* arena_allocate_cons_with_header;
    llvm::Function* arena_allocate_string_with_header;
    llvm::Function* arena_allocate_vector_with_header;
    llvm::Function* arena_allocate_symbol_with_header;
    llvm::Function* arena_allocate_closure_with_header;

    // Tagged cons cell getters
    llvm::Function* tagged_cons_get_int64;
    llvm::Function* tagged_cons_get_double;
    llvm::Function* tagged_cons_get_ptr;
    llvm::Function* tagged_cons_get_type;
    llvm::Function* tagged_cons_get_flags;
    llvm::Function* tagged_cons_get_tagged_value;

    // Tagged cons cell setters
    llvm::Function* tagged_cons_set_int64;
    llvm::Function* tagged_cons_set_double;
    llvm::Function* tagged_cons_set_ptr;
    llvm::Function* tagged_cons_set_null;
    llvm::Function* tagged_cons_set_tagged_value;

    // Tape management
    llvm::Function* arena_allocate_tape;
    llvm::Function* arena_tape_add_node;
    llvm::Function* arena_tape_reset;
    llvm::Function* arena_tape_get_node;
    llvm::Function* arena_tape_get_node_count;

    // AD node allocation
    llvm::Function* arena_allocate_ad_node;
    llvm::Function* arena_allocate_ad_node_with_header;

    // Hash table allocation (with header for HEAP_PTR type)
    llvm::Function* arena_hash_table_create_with_header;

    // Tensor allocation (with header for HEAP_PTR type)
    llvm::Function* arena_allocate_tensor_with_header;
    llvm::Function* arena_allocate_tensor_full;

    // Exception allocation (with header for HEAP_PTR type)
    llvm::Function* eshkol_make_exception_with_header;

    // Helper to create function declarations
    llvm::Function* createFunc(const char* name, llvm::FunctionType* ft);

    // Create function groups
    void createCoreArenaFunctions();
    void createConsCellFunctions();
    void createClosureFunctions();
    void createTaggedConsGetters();
    void createTaggedConsSetters();
    void createTapeFunctions();
    void createAdNodeFunctions();
    void createTypedAllocatorFunctions();  // NEW: For consolidated type allocators
};

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
#endif // ESHKOL_BACKEND_MEMORY_CODEGEN_H
