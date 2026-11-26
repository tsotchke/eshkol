// Exercise a documented GNU printf extension in a separate single-threaded process.
#define _GNU_SOURCE
#include <errno.h>
#include <printf.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <eshkol/core/resource_limits.h>

extern int __snprintf_chk(char *, size_t, int, size_t, const char *, ...);
static int rendered, described;
static bool forbid_unsigned_callback;

static int arginfo(const struct printf_info *info, size_t n, int *types, int *sizes) {
    if (info->spec != L'u' || !info->is_long_double) return -1;
    if (forbid_unsigned_callback) _Exit(90);
    ++described;
    if (n) {
        types[0] = PA_INT | PA_FLAG_LONG_LONG;
        sizes[0] = sizeof(unsigned long long);
    }
    return 1;
}

static int render(FILE *out, const struct printf_info *info, const void *const *args) {
    if (info->spec != L'u' || !info->is_long_double) return -1;
    if (forbid_unsigned_callback) _Exit(91);
    ++rendered;
    unsigned long long n = *(const unsigned long long *)args[0];
    char digits[20];
    size_t pos = sizeof digits;
    do {
        digits[--pos] = (char)('0' + n % 10);
        n /= 10;
    } while (n);
    size_t len = sizeof digits - pos;
    return fwrite(digits + pos, 1, len, out) == len ? (int)len : -1;
}

int main(int argc, char **argv) {
    if (argc != 5) return 2;
    bool registered = strcmp(argv[1], "registered") == 0;
    if (!registered && strcmp(argv[1], "default") != 0) return 2;
    eshkol_resource_limits_t limits = eshkol_get_default_limits();
    limits.enforce_hard_limits = strcmp(argv[2], "fatal") == 0;
    limits.enable_warnings = strcmp(argv[2], "warn") == 0;
    if (!limits.enforce_hard_limits && !limits.enable_warnings && strcmp(argv[2], "quiet")) return 2;
    char *end;
    errno = 0;
    if (argv[3][0] < '0' || argv[3][0] > '9') return 2;
    limits.max_execution_time_ms = strtoull(argv[3], &end, 10);
    if (errno || *end) return 2;

    static const struct {
        const char *name;
        eshkol_limit_error_t code;
    } cases[] = {
        {"timeout", ESHKOL_LIMIT_TIMEOUT}, {"heap-soft", ESHKOL_LIMIT_HEAP_SOFT},
        {"heap-hard", ESHKOL_LIMIT_HEAP_HARD}, {"stack", ESHKOL_LIMIT_STACK_OVERFLOW},
        {"tensor", ESHKOL_LIMIT_TENSOR_SIZE}, {"string", ESHKOL_LIMIT_STRING_LENGTH},
    };
    size_t i = 0;
    while (i < sizeof cases / sizeof cases[0] && strcmp(argv[4], cases[i].name)) ++i;
    if (i == sizeof cases / sizeof cases[0]) return 2;
    eshkol_set_limits(&limits);
    if (registered && register_printf_specifier('u', render, arginfo)) return 4;

    // Establish the extension is active before observing the real runtime call.
    char sample[32];
    if (__snprintf_chk(sample, sizeof sample, 1, sizeof sample, "%llums", ~0ULL) != 22 ||
        strcmp(sample, "18446744073709551615ms") ||
        rendered != (int)registered || (described > 0) != registered) return 6;
    forbid_unsigned_callback = true;
    eshkol_limit_error_t code = cases[i].code;
    eshkol_limit_enforce(code, code == ESHKOL_LIMIT_TIMEOUT ? NULL : "control");
    return eshkol_get_last_limit_error() == code ? 0 : 7;
}
