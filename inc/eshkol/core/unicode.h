/*
 * Unicode character classification shared by the native and VM engines.
 */
#ifndef ESHKOL_CORE_UNICODE_H
#define ESHKOL_CORE_UNICODE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int eshkol_unicode_is_alphabetic(int64_t codepoint);
int eshkol_unicode_is_numeric(int64_t codepoint);

#ifdef __cplusplus
}
#endif

#endif
