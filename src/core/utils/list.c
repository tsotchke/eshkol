/**
 * @file list.c
 * @brief Implementation of list operations for Scheme compatibility
 */

#include "core/list.h"
#include "core/memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <assert.h>

/**
 * @brief The empty list singleton
 * 
 * This is the empty list singleton, which is used to represent the end of a list.
 * It is a global variable that should not be modified.
 */
void* ESHKOL_EMPTY_LIST = NULL;

/**
 * @brief Initialize the list module
 * 
 * This function initializes the list module, setting up the empty list singleton.
 * It should be called once at program startup.
 * 
 * @return true on success, false on failure
 */
bool eshkol_list_init(void) {
    if (ESHKOL_EMPTY_LIST == NULL) {
        // Allocate a special value for the empty list
        ESHKOL_EMPTY_LIST = malloc(1);
        if (ESHKOL_EMPTY_LIST == NULL) {
            return false;
        }
        // Set the value to a special byte pattern to identify it
        *(unsigned char*)ESHKOL_EMPTY_LIST = 0xEE;
    }
    return true;
}

/**
 * @brief Clean up the list module
 * 
 * This function cleans up the list module, freeing the empty list singleton.
 * It should be called once at program shutdown.
 */
void eshkol_list_cleanup(void) {
    if (ESHKOL_EMPTY_LIST != NULL) {
        free(ESHKOL_EMPTY_LIST);
        ESHKOL_EMPTY_LIST = NULL;
    }
}

/**
 * @brief Create a new pair (cons cell)
 */
EshkolPair* eshkol_cons(void* car, void* cdr) {
    // Initialize the list module if needed
    if (ESHKOL_EMPTY_LIST == NULL && !eshkol_list_init()) {
        return NULL;
    }
    
    // Allocate a new pair
    EshkolPair* pair = (EshkolPair*)malloc(sizeof(EshkolPair));
    if (pair == NULL) {
        return NULL;
    }
    
    // Initialize the pair
    pair->car = car;
    pair->cdr = cdr;
    pair->is_immutable = false;
    
    return pair;
}

/**
 * @brief Get the first element of a pair
 */
void* eshkol_car(EshkolPair* pair) {
    if (pair == NULL) {
        return NULL;
    }
    return pair->car;
}

/**
 * @brief Get the second element of a pair
 */
void* eshkol_cdr(EshkolPair* pair) {
    if (pair == NULL) {
        return NULL;
    }
    return pair->cdr;
}

/**
 * @brief Set the first element of a pair
 */
bool eshkol_set_car(EshkolPair* pair, void* value) {
    if (pair == NULL || pair->is_immutable) {
        return false;
    }
    pair->car = value;
    return true;
}

/**
 * @brief Set the second element of a pair
 */
bool eshkol_set_cdr(EshkolPair* pair, void* value) {
    if (pair == NULL || pair->is_immutable) {
        return false;
    }
    pair->cdr = value;
    return true;
}

/**
 * @brief Create a list from a variable number of elements
 */
EshkolPair* eshkol_list(size_t count, ...) {
    // Initialize the list module if needed
    if (ESHKOL_EMPTY_LIST == NULL && !eshkol_list_init()) {
        return NULL;
    }
    
    if (count == 0) {
        return ESHKOL_EMPTY_LIST;
    }
    
    va_list args;
    va_start(args, count);
    
    // Create the first pair
    void* first_element = va_arg(args, void*);
    EshkolPair* first_pair = eshkol_cons(first_element, ESHKOL_EMPTY_LIST);
    if (first_pair == NULL) {
        va_end(args);
        return NULL;
    }
    
    // Create the rest of the pairs
    EshkolPair* current_pair = first_pair;
    for (size_t i = 1; i < count; i++) {
        void* element = va_arg(args, void*);
        EshkolPair* new_pair = eshkol_cons(element, ESHKOL_EMPTY_LIST);
        if (new_pair == NULL) {
            // Clean up on failure
            eshkol_free_pair_chain(first_pair);
            va_end(args);
            return NULL;
        }
        current_pair->cdr = new_pair;
        current_pair = new_pair;
    }
    
    va_end(args);
    return first_pair;
}

/**
 * @brief Check if an object is a pair
 */
bool eshkol_is_pair(void* obj) {
    // Initialize the list module if needed
    if (ESHKOL_EMPTY_LIST == NULL && !eshkol_list_init()) {
        return false;
    }
    
    // Check if the object is NULL or the empty list
    if (obj == NULL || obj == ESHKOL_EMPTY_LIST) {
        return false;
    }
    
    // Check if the object is a pair
    // This is a simple pointer check, but in a real implementation,
    // we would need to check the type of the object.
    // For now, we assume that any non-NULL, non-empty-list pointer is a pair.
    return true;
}

/**
 * @brief Check if an object is the empty list
 */
bool eshkol_is_null(void* obj) {
    // Initialize the list module if needed
    if (ESHKOL_EMPTY_LIST == NULL && !eshkol_list_init()) {
        return false;
    }
    
    return obj == ESHKOL_EMPTY_LIST;
}

/**
 * @brief Check if an object is a proper list
 */
bool eshkol_is_list(void* obj) {
    // Initialize the list module if needed
    if (ESHKOL_EMPTY_LIST == NULL && !eshkol_list_init()) {
        return false;
    }
    
    // The empty list is a proper list
    if (obj == ESHKOL_EMPTY_LIST) {
        return true;
    }
    
    // Check if the object is a pair
    if (!eshkol_is_pair(obj)) {
        return false;
    }
    
    // Check if the cdr chain ends with the empty list
    EshkolPair* current = (EshkolPair*)obj;
    while (eshkol_is_pair(current->cdr)) {
        current = (EshkolPair*)current->cdr;
    }
    
    return current->cdr == ESHKOL_EMPTY_LIST;
}

/**
 * @brief Get the length of a list
 */
int eshkol_list_length(EshkolPair* list) {
    // Initialize the list module if needed
    if (ESHKOL_EMPTY_LIST == NULL && !eshkol_list_init()) {
        return -1;
    }
    
    // The empty list has length 0
    if (list == ESHKOL_EMPTY_LIST) {
        return 0;
    }
    
    // Check if the object is a pair
    if (!eshkol_is_pair(list)) {
        return -1;
    }
    
    // Count the elements in the list
    int length = 0;
    EshkolPair* current = list;
    while (current != ESHKOL_EMPTY_LIST) {
        length++;
        
        // Check if the cdr is a pair
        if (!eshkol_is_pair(current->cdr) && current->cdr != ESHKOL_EMPTY_LIST) {
            // Not a proper list
            return -1;
        }
        
        current = (EshkolPair*)current->cdr;
    }
    
    return length;
}

/**
 * @brief Free a pair and all pairs in its cdr chain
 */
void eshkol_free_pair_chain(EshkolPair* pair) {
    // Initialize the list module if needed
    if (ESHKOL_EMPTY_LIST == NULL && !eshkol_list_init()) {
        return;
    }
    
    // Nothing to free
    if (pair == NULL || pair == ESHKOL_EMPTY_LIST) {
        return;
    }
    
    // Free the cdr chain first
    if (eshkol_is_pair(pair->cdr)) {
        eshkol_free_pair_chain((EshkolPair*)pair->cdr);
    }
    
    // Free the pair itself
    free(pair);
}

/**
 * @brief Nested car/cdr operations
 * 
 * These functions implement the nested car/cdr operations from Scheme,
 * such as caar, cadr, cdar, cddr, etc.
 */

/**
 * @brief Get the car of the car of a pair
 */
void* eshkol_caar(EshkolPair* pair) {
    return eshkol_car((EshkolPair*)eshkol_car(pair));
}

/**
 * @brief Get the car of the cdr of a pair
 */
void* eshkol_cadr(EshkolPair* pair) {
    return eshkol_car((EshkolPair*)eshkol_cdr(pair));
}

/**
 * @brief Get the cdr of the car of a pair
 */
void* eshkol_cdar(EshkolPair* pair) {
    return eshkol_cdr((EshkolPair*)eshkol_car(pair));
}

/**
 * @brief Get the cdr of the cdr of a pair
 */
void* eshkol_cddr(EshkolPair* pair) {
    return eshkol_cdr((EshkolPair*)eshkol_cdr(pair));
}

/**
 * @brief Get the car of the car of the car of a pair
 */
void* eshkol_caaar(EshkolPair* pair) {
    return eshkol_car((EshkolPair*)eshkol_caar(pair));
}

/**
 * @brief Get the car of the car of the cdr of a pair
 */
void* eshkol_caadr(EshkolPair* pair) {
    return eshkol_car((EshkolPair*)eshkol_cadr(pair));
}

/**
 * @brief Get the car of the cdr of the car of a pair
 */
void* eshkol_cadar(EshkolPair* pair) {
    return eshkol_car((EshkolPair*)eshkol_cdar(pair));
}

/**
 * @brief Get the car of the cdr of the cdr of a pair
 */
void* eshkol_caddr(EshkolPair* pair) {
    return eshkol_car((EshkolPair*)eshkol_cddr(pair));
}

/**
 * @brief Get the cdr of the car of the car of a pair
 */
void* eshkol_cdaar(EshkolPair* pair) {
    return eshkol_cdr((EshkolPair*)eshkol_caar(pair));
}

/**
 * @brief Get the cdr of the car of the cdr of a pair
 */
void* eshkol_cdadr(EshkolPair* pair) {
    return eshkol_cdr((EshkolPair*)eshkol_cadr(pair));
}

/**
 * @brief Get the cdr of the cdr of the car of a pair
 */
void* eshkol_cddar(EshkolPair* pair) {
    return eshkol_cdr((EshkolPair*)eshkol_cdar(pair));
}

/**
 * @brief Get the cdr of the cdr of the cdr of a pair
 */
void* eshkol_cdddr(EshkolPair* pair) {
    return eshkol_cdr((EshkolPair*)eshkol_cddr(pair));
}
