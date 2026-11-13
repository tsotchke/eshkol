/* Copyright (C) tsotchke. SPDX-License-Identifier: MIT */
#include "runtime_windows_x64_continuation.h"

#include <csetjmp>
#include <cstdio>

static int resume_with(int value) {
    jmp_buf landing;
    const int resumed = setjmp(landing);
    if (resumed == 0) eshkol::restoreWindowsX64Continuation(&landing, value);
    return resumed;
}

int main() {
    // Exercise the production helper, including Rax: STATUS_LONGJUMP supplies
    // nonvolatile registers, but setjmp's integer result is a volatile register.
    if (resume_with(37) != 37 || resume_with(0) != 1 || resume_with(-9) != -9) {
        std::fprintf(stderr, "FAIL: Windows context restore lost the setjmp result\n");
        return 1;
    }
    std::puts("PASS: Windows continuation context restore preserves return values");
    return 0;
}
