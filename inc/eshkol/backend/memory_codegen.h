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
    llvm::Function* getArenaCreate() const { return arena_create; }
    llvm::Function* getArenaDestroy() const { return arena_destroy; }
    llvm::Function* getArenaAllocate() const { return arena_allocate; }
    llvm::Function* getArenaAllocateWithHeader() const { return arena_allocate_with_header; }
    llvm::Function* getArenaPushScope() const { return arena_push_scope; }
    llvm::Function* getArenaPopScope() const { return arena_pop_scope; }

    // Cons cell allocation (legacy - no header)
    llvm::Function* getArenaAllocateConsCell() const { return arena_allocate_cons_cell; }
    llvm::Function* getArenaAllocateTaggedConsCell() const { return arena_allocate_tagged_cons_cell; }

    // Closure allocation (legacy - no header)
    llvm::Function* getArenaAllocateClosure() const { return arena_allocate_closure; }

    // ─── NEW: Typed allocators with object headers (for consolidated types) ───
    llvm::Function* getArenaAllocateConsWithHeader() const { return arena_allocate_cons_with_header; }
    llvm::Function* getArenaAllocateStringWithHeader() const { return arena_allocate_string_with_header; }
    llvm::Function* getArenaAllocateVectorWithHeader() const { return arena_allocate_vector_with_header; }
    llvm::Function* getArenaAllocateClosureWithHeader() const { return arena_allocate_closure_with_header; }

    // Tagged cons cell getters
    llvm::Function* getTaggedConsGetInt64() const { return tagged_cons_get_int64; }
    llvm::Function* getTaggedConsGetDouble() const { return tagged_cons_get_double; }
    llvm::Function* getTaggedConsGetPtr() const { return tagged_cons_get_ptr; }
    llvm::Function* getTaggedConsGetType() const { return tagged_cons_get_type; }
    llvm::Function* getTaggedConsGetFlags() const { return tagged_cons_get_flags; }
    llvm::Function* getTaggedConsGetTaggedValue() const { return tagged_cons_get_tagged_value; }

    // Tagged cons cell setters
    llvm::Function* getTaggedConsSetInt64() const { return tagged_cons_set_int64; }
    llvm::Function* getTaggedConsSetDouble() const { return tagged_cons_set_double; }
    llvm::Function* getTaggedConsSetPtr() const { return tagged_cons_set_ptr; }
    llvm::Function* getTaggedConsSetNull() const { return tagged_cons_set_null; }
    llvm::Function* getTaggedConsSetTaggedValue() const { return tagged_cons_set_tagged_value; }

    // Tape management (for reverse-mode AD)
    llvm::Function* getArenaAllocateTape() const { return arena_allocate_tape; }
    llvm::Function* getArenaTapeAddNode() const { return arena_tape_add_node; }
    llvm::Function* getArenaTapeReset() const { return arena_tape_reset; }
    llvm::Function* getArenaTapeGetNode() const { return arena_tape_get_node; }
    llvm::Function* getArenaTapeGetNodeCount() const { return arena_tape_get_node_count; }

    // AD node allocation
    llvm::Function* getArenaAllocateAdNode() const { return arena_allocate_ad_node; }
    llvm::Function* getArenaAllocateAdNodeWithHeader() const { return arena_allocate_ad_node_with_header; }

    // Hash table allocation (with header for HEAP_PTR type)
    llvm::Function* getArenaHashTableCreateWithHeader() const { return arena_hash_table_create_with_header; }

    // Tensor allocation (with header for HEAP_PTR type)
    llvm::Function* getArenaAllocateTensorWithHeader() const { return arena_allocate_tensor_with_header; }
    llvm::Function* getArenaAllocateTensorFull() const { return arena_allocate_tensor_full; }

    // Exception allocation (with header for HEAP_PTR type)
    llvm::Function* getMakeExceptionWithHeader() const { return eshkol_make_exception_with_header; }

private:
    llvm::Module& module;
    TypeSystem& types;

    // Core arena management
    llvm::Function* arena_create;
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
