/**
 * @file list.h
 * @brief List operations for Scheme compatibility
 * 
 * This file defines the core list operations required for Scheme compatibility,
 * including cons, car, cdr, and related functions.
 */

#ifndef ESHKOL_LIST_H
#define ESHKOL_LIST_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the list module
 * 
 * This function initializes the list module, setting up the empty list singleton
 * and any other required state.
 * 
 * @return true on success, false on failure
 */
bool eshkol_list_init(void);

/**
 * @brief Clean up the list module
 * 
 * This function cleans up the list module, freeing any resources allocated
 * during initialization.
 */
void eshkol_list_cleanup(void);

/**
 * @brief Pair structure for Scheme compatibility
 * 
 * This structure represents a cons cell, which is the basic building block
 * for lists in Scheme.
 */
typedef struct EshkolPair {
    void* car;              /**< First element of the pair */
    void* cdr;              /**< Second element of the pair */
    bool is_immutable;      /**< Whether the pair is immutable */
} EshkolPair;

/**
 * @brief Create a new pair (cons cell)
 * 
 * @param car The first element of the pair
 * @param cdr The second element of the pair
 * @return A new pair, or NULL on failure
 */
EshkolPair* eshkol_cons(void* car, void* cdr);

/**
 * @brief Get the first element of a pair
 * 
 * @param pair The pair
 * @return The first element of the pair, or NULL if pair is NULL
 */
void* eshkol_car(EshkolPair* pair);

/**
 * @brief Get the second element of a pair
 * 
 * @param pair The pair
 * @return The second element of the pair, or NULL if pair is NULL
 */
void* eshkol_cdr(EshkolPair* pair);

/**
 * @brief Set the first element of a pair
 * 
 * @param pair The pair
 * @param value The new value
 * @return true on success, false on failure (e.g., if pair is NULL or immutable)
 */
bool eshkol_set_car(EshkolPair* pair, void* value);

/**
 * @brief Set the second element of a pair
 * 
 * @param pair The pair
 * @param value The new value
 * @return true on success, false on failure (e.g., if pair is NULL or immutable)
 */
bool eshkol_set_cdr(EshkolPair* pair, void* value);

/**
 * @brief Create a list from a variable number of elements
 * 
 * @param count The number of elements
 * @param ... The elements
 * @return A new list, or NULL on failure
 */
EshkolPair* eshkol_list(size_t count, ...);

/**
 * @brief Check if an object is a pair
 * 
 * @param obj The object to check
 * @return true if the object is a pair, false otherwise
 */
bool eshkol_is_pair(void* obj);

/**
 * @brief Check if an object is the empty list
 * 
 * @param obj The object to check
 * @return true if the object is the empty list, false otherwise
 */
bool eshkol_is_null(void* obj);

/**
 * @brief Check if an object is a proper list
 * 
 * A proper list is either the empty list or a pair whose cdr is a proper list.
 * 
 * @param obj The object to check
 * @return true if the object is a proper list, false otherwise
 */
bool eshkol_is_list(void* obj);

/**
 * @brief Get the length of a list
 * 
 * @param list The list
 * @return The length of the list, or -1 if the list is not a proper list
 */
int eshkol_list_length(EshkolPair* list);

/**
 * @brief Free a pair and all pairs in its cdr chain
 * 
 * This function frees a pair and all pairs in its cdr chain, but does not
 * free the car elements. It is the caller's responsibility to free those.
 * 
 * @param pair The pair to free
 */
void eshkol_free_pair_chain(EshkolPair* pair);

/**
 * @brief The empty list singleton
 * 
 * This is the empty list singleton, which is used to represent the end of a list.
 * It is a global variable that should not be modified.
 */
extern void* ESHKOL_EMPTY_LIST;

/**
 * @brief Get the car of the car of a pair
 * 
 * @param pair The pair
 * @return The car of the car of the pair, or NULL if any operation fails
 */
void* eshkol_caar(EshkolPair* pair);

/**
 * @brief Get the car of the cdr of a pair
 * 
 * @param pair The pair
 * @return The car of the cdr of the pair, or NULL if any operation fails
 */
void* eshkol_cadr(EshkolPair* pair);

/**
 * @brief Get the cdr of the car of a pair
 * 
 * @param pair The pair
 * @return The cdr of the car of the pair, or NULL if any operation fails
 */
void* eshkol_cdar(EshkolPair* pair);

/**
 * @brief Get the cdr of the cdr of a pair
 * 
 * @param pair The pair
 * @return The cdr of the cdr of the pair, or NULL if any operation fails
 */
void* eshkol_cddr(EshkolPair* pair);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_LIST_H */
