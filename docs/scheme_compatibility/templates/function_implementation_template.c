/**
 * @scheme_function function_name
 * @description Brief description of the function
 * @signature (function-name arg1 arg2 ...)
 * @standard R7RS
 * @section X.Y.Z
 * @phase N
 * @priority PX
 * @status Planned/In Progress/Implemented/Tested
 * @implementation_notes
 *   - Note 1
 *   - Note 2
 * @edge_cases
 *   - Edge case 1
 *   - Edge case 2
 * @see_also related_function1, related_function2
 */

#include "scheme_runtime.h"

/**
 * Implementation of the Scheme function 'function-name'
 *
 * @param env The current environment
 * @param args The arguments to the function
 * @return The result of the function
 */
SchemeObject* scheme_function_name(SchemeEnvironment* env, SchemeObject* args) {
    // Validate arguments
    if (!SCHEME_IS_PAIR(args) || !SCHEME_IS_PAIR(SCHEME_CDR(args))) {
        return scheme_error("function-name: Invalid arguments");
    }

    // Extract arguments
    SchemeObject* arg1 = SCHEME_CAR(args);
    SchemeObject* arg2 = SCHEME_CAR(SCHEME_CDR(args));

    // Validate argument types
    if (!SCHEME_IS_NUMBER(arg1) || !SCHEME_IS_NUMBER(arg2)) {
        return scheme_error("function-name: Arguments must be numbers");
    }

    // Implement function logic
    double result = SCHEME_NUMBER_VALUE(arg1) + SCHEME_NUMBER_VALUE(arg2);

    // Return result
    return scheme_make_number(result);
}

/**
 * Register the function with the Scheme runtime
 */
void scheme_register_function_name(SchemeEnvironment* env) {
    scheme_define(env, "function-name", scheme_make_procedure(scheme_function_name));
}
