#include "eshkol/core/dtoa_shortest.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int check(double value, const char* label) {
    char text[48];
    int n = eshkol_dtoa_shortest(text, sizeof(text), value);
    if (n <= 0 || (size_t)n >= sizeof(text) || strpbrk(text, ".eE") == NULL) {
        fprintf(stderr, "FAIL: %s lacks inexact marker: %s\n", label, text);
        return 1;
    }
    char* end = NULL;
    double parsed = strtod(text, &end);
    uint64_t a, b;
    memcpy(&a, &value, sizeof(a));
    memcpy(&b, &parsed, sizeof(b));
    if (*end || a != b) {
        fprintf(stderr, "FAIL: %s roundtrip: %s\n", label, text);
        return 1;
    }
    return 0;
}

int main(void) {
    int failures = 0;
    const double samples[] = {0.0, -0.0, 1.0, -1.0, 1000.0, 1e-300,
        5e-324, 2.2250738585072014e-308, 1.7976931348623157e308,
        0.1, 9007199254740992.0};
    for (size_t i = 0; i < sizeof(samples) / sizeof(samples[0]); ++i)
        failures += check(samples[i], "boundary sample");

    uint64_t state = UINT64_C(0x6a09e667f3bcc909);
    for (unsigned i = 0; i < 50000; ++i) {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        double value;
        memcpy(&value, &state, sizeof(value));
        if ((state & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000))
            failures += check(value, "deterministic random finite");
    }

    char tiny[4] = {'x', 'x', 'x', 'x'};
    eshkol_dtoa_shortest(tiny, 1, 1.0);
    if (tiny[0] != '\0' || tiny[1] != 'x') failures++;
    eshkol_dtoa_shortest(tiny, 2, 1.0);
    if (tiny[1] != '\0' || tiny[2] != 'x') failures++;
    eshkol_dtoa_shortest(tiny, 0, 1.0);
    if (tiny[2] != 'x') failures++;

    char text[48];
    eshkol_dtoa_shortest(text, sizeof(text), -0.0);
    if (strcmp(text, "-0.0") != 0) failures++;
    eshkol_dtoa_shortest(text, sizeof(text), 0.0);
    if (strcmp(text, "0.0") != 0) failures++;
    eshkol_dtoa_shortest(text, sizeof(text), INFINITY);
    if (strcmp(text, "+inf.0") != 0) failures++;
    eshkol_dtoa_shortest(text, sizeof(text), -INFINITY);
    if (strcmp(text, "-inf.0") != 0) failures++;
    eshkol_dtoa_shortest(text, sizeof(text), NAN);
    if (strcmp(text, "+nan.0") != 0) failures++;
    if (failures) return 1;
    puts("PASS: dtoa_shortest C");
    return 0;
}
