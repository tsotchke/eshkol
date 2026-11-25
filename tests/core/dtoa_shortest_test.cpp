#include "eshkol/core/dtoa_shortest.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static bool check(double value) {
    char text[48];
    const int n = eshkol_dtoa_shortest(text, sizeof(text), value);
    if (n <= 0 || static_cast<std::size_t>(n) >= sizeof(text) || !std::strpbrk(text, ".eE"))
        return false;
    char* end = nullptr;
    const double parsed = std::strtod(text, &end);
    std::uint64_t a, b;
    std::memcpy(&a, &value, sizeof(a));
    std::memcpy(&b, &parsed, sizeof(b));
    return *end == '\0' && a == b;
}

int main() {
    const double samples[] = {0.0, -0.0, 1.0, -1.0, 1000.0, 1e-300,
        5e-324, 2.2250738585072014e-308, 1.7976931348623157e308,
        0.1, 9007199254740992.0};
    for (double value : samples) if (!check(value)) return 1;
    std::uint64_t state = UINT64_C(0xbb67ae8584caa73b);
    for (unsigned i = 0; i < 50000; ++i) {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        double value;
        std::memcpy(&value, &state, sizeof(value));
        if ((state & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000) && !check(value))
            return 1;
    }
    char tiny[4] = {'x', 'x', 'x', 'x'};
    eshkol_dtoa_shortest(tiny, 1, 1.0);
    if (tiny[0] != '\0' || tiny[1] != 'x') return 1;
    eshkol_dtoa_shortest(tiny, 2, 1.0);
    if (tiny[1] != '\0' || tiny[2] != 'x') return 1;
    eshkol_dtoa_shortest(tiny, 0, 1.0);
    if (tiny[2] != 'x') return 1;
    char text[48];
    eshkol_dtoa_shortest(text, sizeof(text), INFINITY);
    if (std::strcmp(text, "+inf.0") != 0) return 1;
    eshkol_dtoa_shortest(text, sizeof(text), -INFINITY);
    if (std::strcmp(text, "-inf.0") != 0) return 1;
    eshkol_dtoa_shortest(text, sizeof(text), NAN);
    if (std::strcmp(text, "+nan.0") != 0) return 1;
    std::puts("PASS: dtoa_shortest C++");
    return 0;
}
