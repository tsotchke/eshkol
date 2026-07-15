#include <eshkol/agent_capabilities.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

int main() {
    const std::int64_t yoga_node = eshkol_yoga_node_create();
    if (yoga_node <= 0) {
        std::fputs("agent C++ exception ABI: Yoga node creation failed\n", stderr);
        return 1;
    }

    bool caught_out_of_range = false;
    try {
        (void)std::stoll("9223372036854775808");
    } catch (const std::out_of_range&) {
        caught_out_of_range = true;
    } catch (...) {
        eshkol_yoga_node_free(yoga_node);
        std::fputs("agent C++ exception ABI: wrong exception type caught\n", stderr);
        return 1;
    }

    eshkol_yoga_node_free(yoga_node);
    if (!caught_out_of_range) {
        std::fputs("agent C++ exception ABI: out_of_range was not caught\n", stderr);
        return 1;
    }

    std::puts("agent C++ exception ABI: PASS");
    return 0;
}
