/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Freestanding-safe tagged cons allocation and accessor helpers.
 */

#include "arena_memory.h"
#include "../../inc/eshkol/logger.h"

#include <cstddef>
#include <cstdint>

/**
 * @brief Allocate a single tagged cons cell with both car and cdr initialized to NULL.
 *
 * @param arena Arena to allocate the cell from.
 * @return      Newly allocated cell (16-byte aligned), or NULL if @p arena
 *              is NULL or allocation failed.
 */
arena_tagged_cons_cell_t* arena_allocate_tagged_cons_cell(arena_t* arena) {
    if (!arena) {
        eshkol_error("Cannot allocate tagged cons cell: null arena");
        return nullptr;
    }

    arena_tagged_cons_cell_t* cell = (arena_tagged_cons_cell_t*)
        arena_allocate_aligned(arena, sizeof(arena_tagged_cons_cell_t), 16);

    if (!cell) {
        eshkol_error("Failed to allocate tagged cons cell");
        return nullptr;
    }

    cell->car.type = ESHKOL_VALUE_NULL;
    cell->car.flags = 0;
    cell->car.reserved = 0;
    cell->car.data.raw_val = 0;

    cell->cdr.type = ESHKOL_VALUE_NULL;
    cell->cdr.flags = 0;
    cell->cdr.reserved = 0;
    cell->cdr.data.raw_val = 0;

    return cell;
}

/**
 * @brief Allocate a contiguous array of @p count tagged cons cells, each with car/cdr set to NULL.
 *
 * Cheaper than @p count separate calls to arena_allocate_tagged_cons_cell()
 * when a caller (e.g. list-building codegen) knows the needed length up
 * front. Checks for multiplication overflow before allocating.
 *
 * @param arena Arena to allocate the cells from.
 * @param count Number of cells to allocate; must be > 0.
 * @return      Pointer to the first cell in the contiguous array (16-byte
 *              aligned), or NULL on invalid parameters, overflow, or
 *              allocation failure.
 */
arena_tagged_cons_cell_t* arena_allocate_tagged_cons_batch(arena_t* arena, size_t count) {
    if (!arena || count == 0) {
        eshkol_error("Invalid parameters for batch tagged cons allocation");
        return nullptr;
    }

    if (count > SIZE_MAX / sizeof(arena_tagged_cons_cell_t)) {
        eshkol_error("Tagged cons batch allocation overflow (count=%zu)", count);
        return nullptr;
    }

    size_t total_size = sizeof(arena_tagged_cons_cell_t) * count;
    arena_tagged_cons_cell_t* cells = (arena_tagged_cons_cell_t*)
        arena_allocate_aligned(arena, total_size, 16);

    if (!cells) {
        eshkol_error("Failed to allocate %zu tagged cons cells", count);
        return nullptr;
    }

    for (size_t i = 0; i < count; i++) {
        cells[i].car.type = ESHKOL_VALUE_NULL;
        cells[i].car.flags = 0;
        cells[i].car.reserved = 0;
        cells[i].car.data.raw_val = 0;

        cells[i].cdr.type = ESHKOL_VALUE_NULL;
        cells[i].cdr.flags = 0;
        cells[i].cdr.reserved = 0;
        cells[i].cdr.data.raw_val = 0;
    }

    return cells;
}

/**
 * @brief Allocate a cons cell whose car and cdr are both stored as raw int64 payloads.
 *
 * Convenience constructor for the common case where both slots hold an
 * integer-storage-tagged value (e.g. INT64, BOOL, CHAR); sets each slot's
 * type tag and int_val directly without going through the generic tagged-value
 * setters.
 *
 * @param arena    Arena to allocate the cell from.
 * @param car      Raw int64 payload for the car.
 * @param car_type Type tag to store on the car.
 * @param cdr      Raw int64 payload for the cdr.
 * @param cdr_type Type tag to store on the cdr.
 * @return         Newly allocated cell, or NULL on allocation failure.
 */
arena_tagged_cons_cell_t* arena_create_int64_cons(arena_t* arena,
                                                   int64_t car, uint8_t car_type,
                                                   int64_t cdr, uint8_t cdr_type) {
    arena_tagged_cons_cell_t* cell = arena_allocate_tagged_cons_cell(arena);
    if (!cell) return nullptr;

    cell->car.type = car_type;
    cell->car.data.int_val = car;
    cell->cdr.type = cdr_type;
    cell->cdr.data.int_val = cdr;

    return cell;
}

/**
 * @brief Allocate a cons cell from raw tagged-data payloads, without validating the type tags.
 *
 * Unlike arena_create_int64_cons(), accepts any raw eshkol_tagged_data_t
 * union payload (int, double, or pointer bits) for car and cdr alongside an
 * arbitrary type tag, copying the raw bits through unchanged. Used where the
 * caller already knows the payload/type pairing is consistent.
 *
 * @param arena    Arena to allocate the cell from.
 * @param car      Raw payload for the car.
 * @param car_type Type tag to store on the car.
 * @param cdr      Raw payload for the cdr.
 * @param cdr_type Type tag to store on the cdr.
 * @return         Newly allocated cell, or NULL on allocation failure.
 */
arena_tagged_cons_cell_t* arena_create_mixed_cons(arena_t* arena,
                                                   eshkol_tagged_data_t car, uint8_t car_type,
                                                   eshkol_tagged_data_t cdr, uint8_t cdr_type) {
    arena_tagged_cons_cell_t* cell = arena_allocate_tagged_cons_cell(arena);
    if (!cell) return nullptr;

    cell->car.type = car_type;
    cell->car.data.raw_val = car.raw_val;

    cell->cdr.type = cdr_type;
    cell->cdr.data.raw_val = cdr.raw_val;

    return cell;
}

/**
 * @brief Read the car or cdr of a cons cell as an int64, validating the slot's type tag.
 *
 * @param cell   Cons cell to read from.
 * @param is_cdr false to read the car, true to read the cdr.
 * @return       The slot's int_val if its type tag is an integer-storage
 *               type (ESHKOL_IS_INT_STORAGE_TYPE); 0 (with a logged error)
 *               if @p cell is NULL or the slot does not hold an int-storage
 *               value.
 */
int64_t arena_tagged_cons_get_int64(const arena_tagged_cons_cell_t* cell, bool is_cdr) {
    if (!cell) {
        eshkol_error("Cannot get int64 from null tagged cons cell");
        return 0;
    }

    const eshkol_tagged_value_t* tv = is_cdr ? &cell->cdr : &cell->car;
    uint8_t type = tv->type;
    if (!ESHKOL_IS_INT_STORAGE_TYPE(type)) {
        eshkol_error("Attempted to get int64 from non-int-storage cell (type=%d)", type);
        return 0;
    }

    return tv->data.int_val;
}

/**
 * @brief Read the car or cdr of a cons cell as a double, validating the slot's type tag.
 *
 * @param cell   Cons cell to read from.
 * @param is_cdr false to read the car, true to read the cdr.
 * @return       The slot's double_val if its type tag is a double type
 *               (ESHKOL_IS_DOUBLE_TYPE); 0.0 (with a logged error) if
 *               @p cell is NULL or the slot does not hold a double.
 */
double arena_tagged_cons_get_double(const arena_tagged_cons_cell_t* cell, bool is_cdr) {
    if (!cell) {
        eshkol_error("Cannot get double from null tagged cons cell");
        return 0.0;
    }

    const eshkol_tagged_value_t* tv = is_cdr ? &cell->cdr : &cell->car;
    uint8_t type = tv->type;
    if (!ESHKOL_IS_DOUBLE_TYPE(type)) {
        eshkol_error("Attempted to get double from non-double cell (type=%d)", type);
        return 0.0;
    }

    return tv->data.double_val;
}

/**
 * @brief Read the car or cdr of a cons cell as a raw pointer value, validating the slot's type tag.
 *
 * @param cell   Cons cell to read from.
 * @param is_cdr false to read the car, true to read the cdr.
 * @return       0 if the slot's base type is ESHKOL_VALUE_NULL; the slot's
 *               ptr_val if its type is any pointer type
 *               (ESHKOL_IS_ANY_PTR_TYPE); 0 (with a logged error) if
 *               @p cell is NULL or the slot holds neither NULL nor a
 *               pointer type.
 */
uint64_t arena_tagged_cons_get_ptr(const arena_tagged_cons_cell_t* cell, bool is_cdr) {
    if (!cell) {
        eshkol_error("Cannot get pointer from null tagged cons cell");
        return 0;
    }

    const eshkol_tagged_value_t* tv = is_cdr ? &cell->cdr : &cell->car;
    uint8_t type = tv->type;
    uint8_t base_type = ESHKOL_GET_BASE_TYPE(type);

    if (base_type == ESHKOL_VALUE_NULL) {
        return 0;
    }

    if (!ESHKOL_IS_ANY_PTR_TYPE(type)) {
        eshkol_error("Attempted to get pointer from non-pointer cell (type=%d)", type);
        return 0;
    }

    return tv->data.ptr_val;
}

/**
 * @brief Set the car or cdr of a cons cell to an int64 value, validating @p type.
 *
 * @param cell   Cons cell to modify.
 * @param is_cdr false to set the car, true to set the cdr.
 * @param value  Integer value to store.
 * @param type   Type tag to store; must be an integer-storage type
 *               (ESHKOL_IS_INT_STORAGE_TYPE) or the call is rejected (logged
 *               error, cell left unchanged).
 */
void arena_tagged_cons_set_int64(arena_tagged_cons_cell_t* cell, bool is_cdr,
                                  int64_t value, uint8_t type) {
    if (!cell) {
        eshkol_error("Cannot set int64 on null tagged cons cell");
        return;
    }

    if (!ESHKOL_IS_INT_STORAGE_TYPE(type)) {
        eshkol_error("Invalid type for int64 storage value: %d", type);
        return;
    }

    eshkol_tagged_value_t* tv = is_cdr ? &cell->cdr : &cell->car;
    tv->type = type;
    tv->data.int_val = value;
}

/**
 * @brief Set the car or cdr of a cons cell to a double value, validating @p type.
 *
 * @param cell   Cons cell to modify.
 * @param is_cdr false to set the car, true to set the cdr.
 * @param value  Double value to store.
 * @param type   Type tag to store; must be a double type
 *               (ESHKOL_IS_DOUBLE_TYPE) or the call is rejected (logged
 *               error, cell left unchanged).
 */
void arena_tagged_cons_set_double(arena_tagged_cons_cell_t* cell, bool is_cdr,
                                   double value, uint8_t type) {
    if (!cell) {
        eshkol_error("Cannot set double on null tagged cons cell");
        return;
    }

    if (!ESHKOL_IS_DOUBLE_TYPE(type)) {
        eshkol_error("Invalid type for double value: %d", type);
        return;
    }

    eshkol_tagged_value_t* tv = is_cdr ? &cell->cdr : &cell->car;
    tv->type = type;
    tv->data.double_val = value;
}

/**
 * @brief Set the car or cdr of a cons cell to a raw pointer value, validating @p type.
 *
 * @param cell   Cons cell to modify.
 * @param is_cdr false to set the car, true to set the cdr.
 * @param value  Pointer value (as uint64_t) to store.
 * @param type   Type tag to store; must be a pointer type
 *               (ESHKOL_IS_ANY_PTR_TYPE) or the call is rejected (logged
 *               error, cell left unchanged).
 */
void arena_tagged_cons_set_ptr(arena_tagged_cons_cell_t* cell, bool is_cdr,
                                uint64_t value, uint8_t type) {
    if (!cell) {
        eshkol_error("Cannot set pointer on null tagged cons cell");
        return;
    }

    if (!ESHKOL_IS_ANY_PTR_TYPE(type)) {
        eshkol_error("Invalid type for pointer value: %d", type);
        return;
    }

    eshkol_tagged_value_t* tv = is_cdr ? &cell->cdr : &cell->car;
    tv->type = type;
    tv->data.ptr_val = value;
}

/**
 * @brief Set the car or cdr of a cons cell to the tagged empty-list ('()) value.
 *
 * @param cell   Cons cell to modify (no-op, logged error, if NULL).
 * @param is_cdr false to set the car, true to set the cdr.
 */
void arena_tagged_cons_set_null(arena_tagged_cons_cell_t* cell, bool is_cdr) {
    if (!cell) {
        eshkol_error("Cannot set null on null tagged cons cell");
        return;
    }

    eshkol_tagged_value_t* tv = is_cdr ? &cell->cdr : &cell->car;
    tv->type = ESHKOL_VALUE_NULL;
    tv->data.raw_val = 0;
}

/**
 * @brief Read the raw type tag stored on the car or cdr of a cons cell.
 *
 * @param cell   Cons cell to read from.
 * @param is_cdr false to read the car, true to read the cdr.
 * @return       The slot's type tag, or ESHKOL_VALUE_NULL (with a logged
 *               error) if @p cell is NULL.
 */
uint8_t arena_tagged_cons_get_type(const arena_tagged_cons_cell_t* cell, bool is_cdr) {
    if (!cell) {
        eshkol_error("Cannot get type from null tagged cons cell");
        return ESHKOL_VALUE_NULL;
    }

    const eshkol_tagged_value_t* tv = is_cdr ? &cell->cdr : &cell->car;
    return tv->type;
}

/**
 * @brief Read the flags byte stored on the car or cdr of a cons cell.
 *
 * @param cell   Cons cell to read from.
 * @param is_cdr false to read the car, true to read the cdr.
 * @return       The slot's flags byte, or 0 (with a logged error) if
 *               @p cell is NULL.
 */
uint8_t arena_tagged_cons_get_flags(const arena_tagged_cons_cell_t* cell, bool is_cdr) {
    if (!cell) {
        eshkol_error("Cannot get flags from null tagged cons cell");
        return 0;
    }

    const eshkol_tagged_value_t* tv = is_cdr ? &cell->cdr : &cell->car;
    return tv->flags;
}

/**
 * @brief Test whether the car or cdr of a cons cell has a given base type.
 *
 * Compares base types via ESHKOL_GET_BASE_TYPE on both sides, so flag bits
 * set on either the slot's stored type or @p type do not affect the
 * comparison.
 *
 * @param cell   Cons cell to inspect (returns false if NULL).
 * @param is_cdr false to inspect the car, true to inspect the cdr.
 * @param type   Type to compare against (base type extracted before
 *               comparing).
 * @return       true if the slot's base type matches @p type's base type.
 */
bool arena_tagged_cons_is_type(const arena_tagged_cons_cell_t* cell, bool is_cdr, uint8_t type) {
    if (!cell) return false;

    const eshkol_tagged_value_t* tv = is_cdr ? &cell->cdr : &cell->car;
    uint8_t actual_type = tv->type;
    return ESHKOL_GET_BASE_TYPE(actual_type) == ESHKOL_GET_BASE_TYPE(type);
}

/**
 * @brief Overwrite the car or cdr of a cons cell with a full tagged value (type + payload + flags).
 *
 * @param cell   Cons cell to modify (no-op, logged error, if NULL).
 * @param is_cdr false to set the car, true to set the cdr.
 * @param value  Tagged value to copy in (no-op, logged error, if NULL).
 */
void arena_tagged_cons_set_tagged_value(arena_tagged_cons_cell_t* cell,
                                         bool is_cdr,
                                         const eshkol_tagged_value_t* value) {
    if (!cell || !value) {
        eshkol_error("Cannot set tagged value: null parameter");
        return;
    }

    if (is_cdr) {
        cell->cdr = *value;
    } else {
        cell->car = *value;
    }
}

/**
 * @brief Read the car or cdr of a cons cell as a full tagged value (by copy).
 *
 * @param cell   Cons cell to read from.
 * @param is_cdr false to read the car, true to read the cdr.
 * @return       A copy of the requested slot's tagged value, or a
 *               NULL-typed tagged value (with a logged error) if @p cell is
 *               NULL.
 */
eshkol_tagged_value_t arena_tagged_cons_get_tagged_value(const arena_tagged_cons_cell_t* cell,
                                                          bool is_cdr) {
    if (!cell) {
        eshkol_error("Cannot get tagged value from null cell");
        eshkol_tagged_value_t null_val;
        null_val.type = ESHKOL_VALUE_NULL;
        null_val.flags = 0;
        null_val.reserved = 0;
        null_val.data.int_val = 0;
        return null_val;
    }

    return is_cdr ? cell->cdr : cell->car;
}
