/* SPDX-License-Identifier: MIT */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int network_allowed = 1;

void eshkol_sse_test_set_network_allowed(int allowed) {
    network_allowed = allowed;
}

/* The standalone SSE executables link native HTTP without the hosted policy. */
int eshkol_capability_runtime_allows(const char* capability) {
    return network_allowed && capability && strcmp(capability, "network") == 0;
}

void eshkol_capability_enforce(const char* capability, const char* operation) {
    if (eshkol_capability_runtime_allows(capability)) return;
    fprintf(stderr, "Standalone HTTP capability denied: %s (%s)\n", capability, operation);
    abort();
}
