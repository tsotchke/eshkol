/**
 * @file simd.c
 * @brief Implementation of SIMD detection and capability utilities
 */

#include "core/simd.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/**
 * @brief SIMD capability information
 */
static SimdInfo g_simd_info = {
    .flags = SIMD_NONE,
    .name = "Unknown",
    .vector_size = 0,
    .has_fma = false,
    .has_popcnt = false
};

/**
 * @brief Flag indicating whether SIMD detection has been initialized
 */
static bool g_simd_initialized = false;

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
/**
 * @brief CPUID function for x86/x64
 * 
 * @param eax EAX register value
 * @param ebx EBX register value
 * @param ecx ECX register value
 * @param edx EDX register value
 */
static void cpuid(int* eax, int* ebx, int* ecx, int* edx) {
#if defined(_MSC_VER)
    int regs[4];
    __cpuid(regs, *eax);
    *eax = regs[0];
    *ebx = regs[1];
    *ecx = regs[2];
    *edx = regs[3];
#else
    __asm__ __volatile__(
        "cpuid"
        : "=a"(*eax), "=b"(*ebx), "=c"(*ecx), "=d"(*edx)
        : "a"(*eax), "c"(*ecx)
    );
#endif
}

/**
 * @brief CPUID function with ECX for x86/x64
 * 
 * @param eax EAX register value
 * @param ecx ECX register value
 * @param ebx EBX register value
 * @param edx EDX register value
 */
static void cpuidex(int* eax, int* ecx, int* ebx, int* edx) {
#if defined(_MSC_VER)
    int regs[4];
    __cpuidex(regs, *eax, *ecx);
    *eax = regs[0];
    *ebx = regs[1];
    *ecx = regs[2];
    *edx = regs[3];
#else
    __asm__ __volatile__(
        "cpuid"
        : "=a"(*eax), "=b"(*ebx), "=c"(*ecx), "=d"(*edx)
        : "a"(*eax), "c"(*ecx)
    );
#endif
}

/**
 * @brief Get the CPU name
 * 
 * @param name Buffer to store the CPU name
 * @param size Size of the buffer
 */
static void get_cpu_name(char* name, size_t size) {
    int regs[4];
    int i;
    
    // Check if CPU name is supported
    int eax = 0x80000000;
    int ebx = 0;
    int ecx = 0;
    int edx = 0;
    cpuid(&eax, &ebx, &ecx, &edx);
    
    if (eax < 0x80000004) {
        strncpy(name, "Unknown", size);
        return;
    }
    
    // Get CPU name
    for (i = 0; i < 3; i++) {
        eax = 0x80000002 + i;
        ecx = 0;
        cpuidex(&eax, &ecx, &ebx, &edx);
        regs[0] = eax;
        regs[1] = ebx;
        regs[2] = ecx;
        regs[3] = edx;
        memcpy(name + i * 16, regs, 16);
    }
    
    name[48] = '\0';
    
    // Remove leading spaces
    char* p = name;
    while (*p == ' ') {
        p++;
    }
    
    if (p != name) {
        memmove(name, p, strlen(p) + 1);
    }
    
    // Remove trailing spaces
    p = name + strlen(name) - 1;
    while (p >= name && *p == ' ') {
        *p-- = '\0';
    }
}

/**
 * @brief Detect SIMD capabilities for x86/x64
 */
static void detect_simd_x86(void) {
    int eax, ebx, ecx, edx;
    char cpu_name[64] = {0};
    
    // Get CPU name
    get_cpu_name(cpu_name, sizeof(cpu_name));
    g_simd_info.name = strdup(cpu_name);
    
    // Get SIMD capabilities
    eax = 1;
    ecx = 0;
    cpuidex(&eax, &ecx, &ebx, &edx);
    
    // Check SSE
    if (edx & (1 << 25)) {
        g_simd_info.flags |= SIMD_SSE;
        g_simd_info.vector_size = 16;
    }
    
    // Check SSE2
    if (edx & (1 << 26)) {
        g_simd_info.flags |= SIMD_SSE2;
    }
    
    // Check SSE3
    if (ecx & (1 << 0)) {
        g_simd_info.flags |= SIMD_SSE3;
    }
    
    // Check SSSE3
    if (ecx & (1 << 9)) {
        g_simd_info.flags |= SIMD_SSSE3;
    }
    
    // Check SSE4.1
    if (ecx & (1 << 19)) {
        g_simd_info.flags |= SIMD_SSE4_1;
    }
    
    // Check SSE4.2
    if (ecx & (1 << 20)) {
        g_simd_info.flags |= SIMD_SSE4_2;
    }
    
    // Check AVX
    if (ecx & (1 << 28)) {
        g_simd_info.flags |= SIMD_AVX;
        g_simd_info.vector_size = 32;
    }
    
    // Check POPCNT
    if (ecx & (1 << 23)) {
        g_simd_info.has_popcnt = true;
    }
    
    // Check FMA
    if (ecx & (1 << 12)) {
        g_simd_info.has_fma = true;
    }
    
    // Check AVX2 and AVX-512
    eax = 7;
    ecx = 0;
    cpuidex(&eax, &ecx, &ebx, &edx);
    
    // Check AVX2
    if (ebx & (1 << 5)) {
        g_simd_info.flags |= SIMD_AVX2;
    }
    
    // Check AVX-512 Foundation
    if (ebx & (1 << 16)) {
        g_simd_info.flags |= SIMD_AVX512F;
        g_simd_info.vector_size = 64;
    }
}

#elif defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
/**
 * @brief Detect SIMD capabilities for ARM
 */
static void detect_simd_arm(void) {
    // Set CPU name
    g_simd_info.name = strdup("ARM");
    
    // Check NEON
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    g_simd_info.flags |= SIMD_NEON;
    g_simd_info.vector_size = 16;
#endif
}
#endif

void simd_init(void) {
    if (g_simd_initialized) {
        return;
    }
    
    // Initialize SIMD detection
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    detect_simd_x86();
#elif defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
    detect_simd_arm();
#else
    g_simd_info.name = strdup("Unknown");
#endif
    
    g_simd_initialized = true;
}

const SimdInfo* simd_get_info(void) {
    if (!g_simd_initialized) {
        simd_init();
    }
    
    return &g_simd_info;
}

bool simd_is_supported(SimdFlags flags) {
    if (!g_simd_initialized) {
        simd_init();
    }
    
    return (g_simd_info.flags & flags) == flags;
}

void* simd_get_best_impl(void* generic, void* sse, void* avx, void* avx2, void* avx512, void* neon) {
    if (!g_simd_initialized) {
        simd_init();
    }
    
    // Check NEON
    if (neon && (g_simd_info.flags & SIMD_NEON)) {
        return neon;
    }
    
    // Check AVX-512
    if (avx512 && (g_simd_info.flags & SIMD_AVX512F)) {
        return avx512;
    }
    
    // Check AVX2
    if (avx2 && (g_simd_info.flags & SIMD_AVX2)) {
        return avx2;
    }
    
    // Check AVX
    if (avx && (g_simd_info.flags & SIMD_AVX)) {
        return avx;
    }
    
    // Check SSE
    if (sse && (g_simd_info.flags & SIMD_SSE)) {
        return sse;
    }
    
    // Use generic implementation
    return generic;
}

void simd_print_info(void) {
    if (!g_simd_initialized) {
        simd_init();
    }
    
    printf("SIMD Information:\n");
    printf("  CPU: %s\n", g_simd_info.name);
    printf("  Vector size: %d bytes\n", g_simd_info.vector_size);
    printf("  Instruction sets:\n");
    printf("    SSE:     %s\n", (g_simd_info.flags & SIMD_SSE) ? "Yes" : "No");
    printf("    SSE2:    %s\n", (g_simd_info.flags & SIMD_SSE2) ? "Yes" : "No");
    printf("    SSE3:    %s\n", (g_simd_info.flags & SIMD_SSE3) ? "Yes" : "No");
    printf("    SSSE3:   %s\n", (g_simd_info.flags & SIMD_SSSE3) ? "Yes" : "No");
    printf("    SSE4.1:  %s\n", (g_simd_info.flags & SIMD_SSE4_1) ? "Yes" : "No");
    printf("    SSE4.2:  %s\n", (g_simd_info.flags & SIMD_SSE4_2) ? "Yes" : "No");
    printf("    AVX:     %s\n", (g_simd_info.flags & SIMD_AVX) ? "Yes" : "No");
    printf("    AVX2:    %s\n", (g_simd_info.flags & SIMD_AVX2) ? "Yes" : "No");
    printf("    AVX-512: %s\n", (g_simd_info.flags & SIMD_AVX512F) ? "Yes" : "No");
    printf("    NEON:    %s\n", (g_simd_info.flags & SIMD_NEON) ? "Yes" : "No");
    printf("  Extensions:\n");
    printf("    FMA:     %s\n", g_simd_info.has_fma ? "Yes" : "No");
    printf("    POPCNT:  %s\n", g_simd_info.has_popcnt ? "Yes" : "No");
}
