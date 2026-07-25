/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Freestanding-safe tensor allocation helpers.
 */

#include "arena_memory.h"
#include "../../inc/eshkol/logger.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

extern "C" {

/* Raise a catchable type error reporting the operand's actual runtime type.
 * Declared here (rather than including runtime.h) to keep this freestanding-
 * adjacent translation unit's include surface small. ABI-stable symbol. */
void eshkol_type_error_with_operand(const char* proc_name,
                                    const char* expected_type,
                                    const eshkol_tagged_value_t* actual);

/* Shared runtime fatal sink (raises a catchable Eshkol condition). Declared
 * here for the same reason as above. ABI-stable symbol. */
extern void eshkol_runtime_fatal(eshkol_exception_type_t type,
                                 const char* fmt, ...);

/*
 * Centralized, type-checked tensor-operand unpack (ESH-0069).
 *
 * Every tensor op (activations, conv/pool, reductions, shape ops, …) must route
 * its primary tensor operand through this single helper instead of blindly
 * reinterpreting the operand pointer as an eshkol_tensor_t*. Behavior:
 *
 *   (a) operand is already a tensor (HEAP_SUBTYPE_TENSOR or legacy TENSOR_PTR)
 *       -> return its data pointer unchanged (zero-copy, hot path).
 *   (b) operand is a homogeneous numeric vector (HEAP_SUBTYPE_VECTOR whose every
 *       element is an int64 or double) -> coerce to a fresh 1-D tensor.
 *   (c) operand is a proper list of numbers -> coerce to a fresh 1-D tensor,
 *       exactly as `(tensor '(1 2 3))` does. A list and a vector of the same
 *       numbers denote the same 1-D tensor, so accepting one spelling and
 *       rejecting the other would be an arbitrary distinction.
 *   (d) anything else (int, string, bool, null, improper/non-numeric list,
 *       non-numeric/heterogeneous vector, …) -> raise a clean, catchable type
 *       error via eshkol_type_error_with_operand and never touch the struct.
 *
 * This makes it structurally impossible for a tensor op to segfault on a
 * wrong-typed operand: it either gets a valid tensor or the program sees a
 * catchable condition. `op_name` is used only for the error message.
 *
 * Returns the eshkol_tensor_t* (as void*) on success; on the error path it does
 * not return (the type error raises). The trailing `return nullptr` keeps the
 * compiler happy and is never reached.
 */
void* eshkol_tensor_operand_checked(const eshkol_tagged_value_t* val,
                                    const char* op_name) {
    if (val) {
        /* Consolidated HEAP_PTR form: dispatch on the object-header subtype. */
        if (val->type == ESHKOL_VALUE_HEAP_PTR && val->data.ptr_val) {
            void* ptr = (void*)(uintptr_t)val->data.ptr_val;
            const eshkol_object_header_t* hdr = ESHKOL_GET_HEADER(ptr);
            if (hdr) {
                if (hdr->subtype == HEAP_SUBTYPE_TENSOR) {
                    return ptr;  /* already a tensor — zero-copy fast path */
                }
                if (hdr->subtype == HEAP_SUBTYPE_VECTOR) {
                    /* Coerce a *homogeneous numeric* vector to a 1-D tensor.
                     * Layout: [len:i64][eshkol_tagged_value_t elems...]. */
                    char* v = (char*)ptr;
                    int64_t len = *(int64_t*)v;
                    if (len < 0) len = 0;
                    const eshkol_tagged_value_t* elems =
                        (const eshkol_tagged_value_t*)(v + sizeof(int64_t));
                    for (int64_t i = 0; i < len; i++) {
                        uint8_t bt = (uint8_t)(elems[i].type & 0x0F);
                        if (bt != ESHKOL_VALUE_INT64 && bt != ESHKOL_VALUE_DOUBLE) {
                            /* heterogeneous / non-numeric vector — not coercible */
                            eshkol_type_error_with_operand(
                                op_name, "tensor or numeric vector", val);
                            return nullptr;  /* not reached */
                        }
                    }
                    arena_t* arena = get_global_arena();
                    eshkol_tensor_t* t =
                        arena_allocate_tensor_full(arena, 1, (uint64_t)len);
                    if (!t) return nullptr;
                    if (t->dimensions) t->dimensions[0] = (uint64_t)len;
                    for (int64_t i = 0; i < len; i++) {
                        const eshkol_tagged_value_t* e = &elems[i];
                        double d = ((e->type & 0x0F) == ESHKOL_VALUE_DOUBLE)
                                       ? e->data.double_val
                                       : (double)e->data.int_val;
                        std::memcpy(&t->elements[i], &d, sizeof(double));
                    }
                    return t;
                }
            }
        }
        /* Legacy direct TENSOR_PTR type tag. */
        if (val->type == ESHKOL_VALUE_TENSOR_PTR && val->data.ptr_val) {
            return (void*)(uintptr_t)val->data.ptr_val;
        }
        /* Proper list of numbers -> fresh 1-D tensor. Measure first (so an
         * improper or non-numeric list allocates nothing), then fill. */
        if (ESHKOL_IS_CONS_COMPAT(*val)) {
            int64_t len = 0;
            eshkol_tagged_value_t cur = *val;
            bool numeric = true;
            while (ESHKOL_IS_CONS_COMPAT(cur)) {
                const auto* cell =
                    (const arena_tagged_cons_cell_t*)(uintptr_t)cur.data.ptr_val;
                if (!cell) { numeric = false; break; }
                uint8_t bt = (uint8_t)(cell->car.type & 0x0F);
                if (bt != ESHKOL_VALUE_INT64 && bt != ESHKOL_VALUE_DOUBLE) {
                    numeric = false;
                    break;
                }
                len++;
                cur = cell->cdr;
            }
            if (numeric && cur.type == ESHKOL_VALUE_NULL) {
                arena_t* arena = get_global_arena();
                eshkol_tensor_t* t =
                    arena_allocate_tensor_full(arena, 1, (uint64_t)len);
                if (!t) return nullptr;
                if (t->dimensions) t->dimensions[0] = (uint64_t)len;
                cur = *val;
                for (int64_t i = 0; i < len && ESHKOL_IS_CONS_COMPAT(cur); i++) {
                    const auto* cell =
                        (const arena_tagged_cons_cell_t*)(uintptr_t)cur.data.ptr_val;
                    double d = ((cell->car.type & 0x0F) == ESHKOL_VALUE_DOUBLE)
                                   ? cell->car.data.double_val
                                   : (double)cell->car.data.int_val;
                    std::memcpy(&t->elements[i], &d, sizeof(double));
                    cur = cell->cdr;
                }
                return t;
            }
            eshkol_type_error_with_operand(
                op_name, "tensor, numeric vector, or list of numbers", val);
            return nullptr;  /* not reached */
        }
    }

    /* Not a tensor and not a coercible numeric collection: clean, catchable
     * error instead of a segfault from misreading the struct. */
    eshkol_type_error_with_operand(op_name, "tensor", val);
    return nullptr;  /* not reached (type error raises) */
}

/*
 * Type-checked unpack for an operand the op MUTATES IN PLACE.
 *
 * Same classification as eshkol_tensor_operand_checked, minus the coercion:
 * an in-place op's destination must be the caller's own storage, so a
 * collection is rejected rather than copied into a fresh tensor. Coercing
 * here would be *worse* than the crash it replaces — `(sgd-step params grads
 * lr)` would update a throwaway copy, return the caller's untouched vector,
 * and report success while training silently made no progress. A wrong answer
 * that looks right is the one failure mode this release will not ship.
 *
 *   (a) operand is a tensor (HEAP_SUBTYPE_TENSOR or legacy TENSOR_PTR)
 *       -> return its data pointer unchanged.
 *   (b) anything else, INCLUDING a numeric vector or list -> raise a clean,
 *       catchable type error naming the operation.
 *
 * Read-only operands of the very same op (an optimizer's `grads`, a loss's
 * `target`) still go through eshkol_tensor_operand_checked and do accept a
 * numeric collection: the distinction is whether the op writes through the
 * pointer, not which argument slot it occupies.
 */
void* eshkol_tensor_destination_checked(const eshkol_tagged_value_t* val,
                                        const char* op_name) {
    if (val) {
        if (val->type == ESHKOL_VALUE_HEAP_PTR && val->data.ptr_val) {
            void* ptr = (void*)(uintptr_t)val->data.ptr_val;
            const eshkol_object_header_t* hdr = ESHKOL_GET_HEADER(ptr);
            if (hdr && hdr->subtype == HEAP_SUBTYPE_TENSOR) {
                return ptr;
            }
        }
        if (val->type == ESHKOL_VALUE_TENSOR_PTR && val->data.ptr_val) {
            return (void*)(uintptr_t)val->data.ptr_val;
        }
    }

    eshkol_type_error_with_operand(op_name, "tensor (updated in place)", val);
    return nullptr;  /* not reached (type error raises) */
}

/*
 * Type-checked unpack for an operand that must be a MATRIX (rank >= 2).
 *
 * tensor-lu / -det / -inverse / -cholesky / -qr / -svd, tensor-solve's `A`, and
 * einsum's "ij,jk->ik" operands all document their argument as a matrix, index
 * `dims[0]` and `dims[1]`, and walk the element buffer in row-major order. None
 * of them (except tensor-qr) checked the rank, so a rank-1 operand produced a
 * fabricated answer rather than an error:
 *
 *   (tensor-det #(1 2 3 4))       =>  0                      ; not a determinant
 *   (tensor-cholesky #(4 2 2 3))  =>  #((2 0 0 0) (0 0 0 0) …) ; 4x4 from 4 numbers
 *
 * A flat collection of numbers therefore must NOT be coerced here the way it is
 * for a 1-D data operand: coercion yields a rank-1 tensor, which is precisely
 * the shape that produces those wrong answers. Nothing in the operand says how
 * to fold `n` numbers into rows and columns, so inventing a shape would be
 * guessing. Reject instead, and say what was expected.
 *
 *   (a) operand is a tensor of rank >= 2 -> return its data pointer unchanged.
 *   (b) tensor of rank < 2, numeric vector, list, or anything else -> raise a
 *       clean, catchable error naming the operation.
 *
 * A rank-1 argument that IS meaningful stays on the coercing path: tensor-solve
 * passes `A` through here and `b` through eshkol_tensor_operand_checked, so
 * `(tensor-solve A '(5 10))` keeps working.
 */
void* eshkol_tensor_matrix_operand_checked(const eshkol_tagged_value_t* val,
                                          const char* op_name) {
    const eshkol_tensor_t* t = nullptr;
    if (val) {
        if (val->type == ESHKOL_VALUE_HEAP_PTR && val->data.ptr_val) {
            void* ptr = (void*)(uintptr_t)val->data.ptr_val;
            const eshkol_object_header_t* hdr = ESHKOL_GET_HEADER(ptr);
            if (hdr && hdr->subtype == HEAP_SUBTYPE_TENSOR) {
                t = (const eshkol_tensor_t*)ptr;
            }
        } else if (val->type == ESHKOL_VALUE_TENSOR_PTR && val->data.ptr_val) {
            t = (const eshkol_tensor_t*)(uintptr_t)val->data.ptr_val;
        }
    }

    if (t && t->num_dimensions >= 2) {
        return (void*)t;
    }
    if (t) {
        eshkol_runtime_fatal(
            ESHKOL_EXCEPTION_ERROR,
            "%s: expected a matrix (rank 2 or more), got a rank-%llu tensor",
            op_name ? op_name : "matrix operation",
            (unsigned long long)t->num_dimensions);
        return nullptr;  /* not reached */
    }

    eshkol_type_error_with_operand(op_name, "matrix (tensor of rank 2 or more)",
                                   val);
    return nullptr;  /* not reached (type error raises) */
}

/*
 * Read a per-dimension repetition/count vector into `out[0 .. rank-1]`.
 *
 * `tile`'s `reps` argument is documented as "a list or vector of repetition
 * counts, one per dimension". The codegen used to peek at the operand's raw
 * bytes as a Scheme vector — `[len:i64][tagged elems...]` — which meant a
 * LIST (the first spelling in that sentence) had its cons-cell car misread as
 * the length and every count came back as 1, silently returning the tensor
 * untiled; an integer or string operand was dereferenced as a vector and
 * crashed; and a vector SHORTER than the tensor's rank was read past its end.
 *
 * Classifying the operand here instead makes all three impossible and makes
 * the documented list spelling actually work. Accepts a vector or proper list
 * of exactly `rank` integers; anything else raises a catchable error. Counts
 * are returned as-is (the caller clamps semantics, e.g. tile's "at least 1").
 */
void eshkol_tensor_counts_checked(const eshkol_tagged_value_t* val,
                                  int64_t rank,
                                  int64_t* out,
                                  const char* op_name) {
    if (!out || rank < 0) return;

    if (val && val->type == ESHKOL_VALUE_HEAP_PTR && val->data.ptr_val) {
        void* ptr = (void*)(uintptr_t)val->data.ptr_val;
        const eshkol_object_header_t* hdr = ESHKOL_GET_HEADER(ptr);
        if (hdr && hdr->subtype == HEAP_SUBTYPE_VECTOR) {
            char* v = (char*)ptr;
            int64_t len = *(int64_t*)v;
            const eshkol_tagged_value_t* elems =
                (const eshkol_tagged_value_t*)(v + sizeof(int64_t));
            if (len == rank) {
                for (int64_t i = 0; i < rank; i++) {
                    if ((uint8_t)(elems[i].type & 0x0F) != ESHKOL_VALUE_INT64) {
                        eshkol_type_error_with_operand(
                            op_name, "vector or list of integer counts", val);
                        return;  /* not reached */
                    }
                }
                for (int64_t i = 0; i < rank; i++) out[i] = elems[i].data.int_val;
                return;
            }
        } else if (hdr && hdr->subtype == HEAP_SUBTYPE_CONS) {
            /* Validate the whole list (length, every car an integer, proper
             * NULL terminator) before writing anything, so a bad list leaves
             * no half-filled count array behind. */
            int64_t len = 0;
            eshkol_tagged_value_t cur = *val;
            bool ok = true;
            while (ESHKOL_IS_CONS_COMPAT(cur)) {
                const auto* cell =
                    (const arena_tagged_cons_cell_t*)(uintptr_t)cur.data.ptr_val;
                if (!cell) { ok = false; break; }
                if ((uint8_t)(cell->car.type & 0x0F) != ESHKOL_VALUE_INT64) {
                    ok = false;
                    break;
                }
                len++;
                cur = cell->cdr;
            }
            if (ok && cur.type == ESHKOL_VALUE_NULL && len == rank) {
                cur = *val;
                for (int64_t i = 0; i < rank && ESHKOL_IS_CONS_COMPAT(cur); i++) {
                    const auto* cell =
                        (const arena_tagged_cons_cell_t*)(uintptr_t)cur.data.ptr_val;
                    out[i] = cell->car.data.int_val;
                    cur = cell->cdr;
                }
                return;
            }
        }
    }

    eshkol_type_error_with_operand(
        op_name, "vector or list of integer counts, one per dimension", val);
}

/*
 * Validate a reduction AXIS against the operand's rank.
 *
 * `(tensor-sum t axis)` / `(tensor-mean t axis)` / `(tensor-max t axis)` /
 * `(tensor-min t axis)` reduce `t` along one axis. Neither the codegen nor
 * eshkol_xla_reduce checked the axis against the rank, so an axis naming a
 * dimension the tensor does not have still produced a value:
 *
 *   (tensor-sum #(1 2 3 4) 1)  =>  ()   ; eshkol_xla_reduce returned NULL for
 *                                       ; the out-of-range axis and the NULL
 *                                       ; was packed as a heap pointer anyway
 *   (tensor-mean m23 5)        =>  ()   ; likewise
 *
 * That prints as an empty collection and compares unequal to every real number,
 * so a wrong axis read as "the tensor was empty" rather than as an error. The AD
 * path was worse: it indexes `dims[axis]` directly to size its output, so an
 * out-of-range axis read past the dimensions array before dividing by whatever
 * it found.
 *
 * The contract is exactly `0 <= axis < rank` after the caller's negative-axis
 * normalization. Note what is NOT an error: reducing the sole axis of a rank-1
 * tensor. `(tensor-sum #(1 2 3 4) 0)` is a complete reduction and yields the
 * 1-element tensor `#(10)` — the answer the VM's vm_tensor_reduce already
 * documents ("a 1D input reduces to a 1-element tensor rather than a 0-dim
 * one") and the answer numpy gives. It used to yield `#()` because `rank - 1`
 * was used unclamped as the output rank; that is fixed in eshkol_xla_reduce and
 * in the AD lowering rather than turned into an error, so both substrates and
 * both native reduce paths agree.
 *
 * Returns the validated axis so the caller can use the result directly; never
 * returns on the failure path (the error raises catchably).
 */
int64_t eshkol_tensor_axis_checked(int64_t axis, int64_t rank,
                                  const char* op_name) {
    if (axis < 0 || axis >= rank) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                             "%s: axis %lld out of range for rank-%lld tensor",
                             op_name ? op_name : "tensor axis reduction",
                             (long long)axis, (long long)rank);
        return 0;  /* not reached */
    }
    return axis;
}

/**
 * @brief Allocate an empty tensor object (header only, no dimensions/elements).
 *
 * Allocates `sizeof(eshkol_tensor_t)` bytes plus a header, 64-byte aligned,
 * tags the header with HEAP_SUBTYPE_TENSOR, and initializes the tensor to
 * the zero/empty state: no dimensions array, no elements array,
 * `num_dimensions = 0`, `total_elements = 0`, and dtype defaulted to
 * ESHKOL_TENSOR_DTYPE_F64. Callers that need a populated tensor should use
 * arena_allocate_tensor_full instead, which calls this and then allocates
 * the dimensions/elements storage. The tensor is arena-owned and lives
 * until the arena is freed/reset.
 *
 * @param arena  Arena to allocate from (must not be null).
 * @return       Newly allocated empty tensor, or nullptr on failure.
 */
eshkol_tensor_t* arena_allocate_tensor_with_header(arena_t* arena) {
    if (!arena) {
        eshkol_error("Invalid arena for tensor allocation");
        return nullptr;
    }

    const size_t data_size = sizeof(eshkol_tensor_t);
    size_t total = sizeof(eshkol_object_header_t) + data_size;
    total = (total + 63) & ~((size_t)63);

    uint8_t* mem = (uint8_t*)arena_allocate_aligned(arena, total, 64);
    if (!mem) {
        eshkol_error("Failed to allocate tensor with header");
        return nullptr;
    }

    eshkol_object_header_t* hdr = (eshkol_object_header_t*)mem;
    hdr->subtype = HEAP_SUBTYPE_TENSOR;
    hdr->flags = 0;
    hdr->ref_count = 0;
    hdr->size = (uint32_t)data_size;

    eshkol_tensor_t* tensor =
        (eshkol_tensor_t*)(mem + sizeof(eshkol_object_header_t));
    tensor->dimensions = nullptr;
    tensor->num_dimensions = 0;
    tensor->elements = nullptr;
    tensor->total_elements = 0;
    tensor->dtype = ESHKOL_TENSOR_DTYPE_F64;  // default precision

    return tensor;
}

/**
 * @brief Allocate a fully-populated tensor: header, dimensions array, and elements array.
 *
 * Calls arena_allocate_tensor_with_header for the tensor struct itself, then
 * (if `num_dims > 0`) allocates a 64-byte-aligned `uint64_t[num_dims]`
 * dimensions array, and (if `total_elements > 0`) allocates a 64-byte-aligned
 * `int64_t[total_elements]` elements array zero-initialized via memset
 * (elements store IEEE-754 double bit patterns, per eshkol_tensor_t's
 * layout note — a zeroed element therefore represents 0.0). Both allocation
 * sizes are overflow-checked before allocating. On success, sets
 * `tensor->num_dimensions` and `tensor->total_elements` to the requested
 * values. Everything allocated is arena-owned and lives until the arena is
 * freed/reset; a failure partway through leaves the already-allocated
 * pieces arena-owned garbage (not individually freed) since the arena
 * doesn't support partial frees.
 *
 * @param arena           Arena to allocate from (must not be null).
 * @param num_dims        Number of dimensions (rank); 0 allocates no dimensions array.
 * @param total_elements  Total element count (product of dimensions); 0 allocates no elements array.
 * @return                Newly allocated, populated tensor, or nullptr on failure.
 */
eshkol_tensor_t* arena_allocate_tensor_full(
    arena_t* arena, uint64_t num_dims, uint64_t total_elements) {
    if (!arena) {
        eshkol_error("Invalid arena for tensor allocation");
        return nullptr;
    }

    eshkol_tensor_t* tensor = arena_allocate_tensor_with_header(arena);
    if (!tensor) {
        return nullptr;
    }

    if (num_dims > 0) {
        if (num_dims > SIZE_MAX / sizeof(uint64_t)) {
            eshkol_error("Tensor dimensions allocation overflow (num_dims=%llu)",
                         (unsigned long long)num_dims);
            return nullptr;
        }

        tensor->dimensions = (uint64_t*)arena_allocate_aligned(
            arena, (size_t)num_dims * sizeof(uint64_t), 64);
        if (!tensor->dimensions) {
            eshkol_error("Failed to allocate tensor dimensions array");
            return nullptr;
        }
    }

    if (total_elements > 0) {
        if (total_elements > SIZE_MAX / sizeof(int64_t)) {
            eshkol_error("Tensor elements allocation overflow (total_elements=%llu)",
                         (unsigned long long)total_elements);
            return nullptr;
        }

        const size_t elem_size = (size_t)total_elements * sizeof(int64_t);
        tensor->elements = (int64_t*)arena_allocate_aligned(arena, elem_size, 64);
        if (!tensor->elements) {
            eshkol_error("Failed to allocate tensor elements array");
            return nullptr;
        }
        std::memset(tensor->elements, 0, elem_size);
    }

    tensor->num_dimensions = num_dims;
    tensor->total_elements = total_elements;

    return tensor;
}

}  // extern "C"
