/*
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 *
 * Reference board: `qemu-system-x86_64 -M pc -device isa-debug-exit,iobase=0xf4,iosize=0x04`,
 * booted directly as an ELF64 kernel via PVH (XEN_ELFNOTE_PHYS32_ENTRY = 18).
 *
 * Boot sequence:
 *   1. QEMU enters `eshkol_x86_64_pvh_entry` in 32-bit flat protected mode.
 *   2. Identity-map the first 1 GiB using static 2 MiB pages (PML4 -> PDPT -> PD).
 *   3. Enable PAE (CR4.PAE), set IA32_EFER.LME, enable paging (CR0.PG), load
 *      a 64-bit GDT, and far-jump into 64-bit long mode with a 16-byte-aligned
 *      stack (`__eshkol_stack_top`).
 *   4. Initialize COM1 (0x3F8) serial console and call `eshkol_embedded_start()`.
 *   5. Exit cleanly via QEMU `isa-debug-exit` at I/O port 0xF4.
 */
#include "eshkol_embedded.h"

#define COM1_BASE 0x3F8u
#define COM1_LSR  (COM1_BASE + 5u)
#define COM1_LSR_THRE 0x20u
#define ISA_DEBUG_EXIT_PORT 0xF4u

/* Static identity-map tables (4 KiB aligned in .bss) */
__attribute__((aligned(4096), used)) static uint64_t boot_pml4[512];
__attribute__((aligned(4096), used)) static uint64_t boot_pdpt[512];
__attribute__((aligned(4096), used)) static uint64_t boot_pd[512];

/* Minimal 64-bit flat GDT: null, 64-bit code (0x08), 64-bit data (0x10) */
__attribute__((aligned(16), used))
static const uint64_t boot_gdt[3] = {
    0x0000000000000000ULL,
    0x00AF9A000000FFFFULL,
    0x00CF92000000FFFFULL,
};

struct __attribute__((packed)) gdt_ptr {
    uint16_t limit;
    uint32_t base;
};

__attribute__((aligned(16), used))
static struct gdt_ptr boot_gdtr = {
    (uint16_t)(sizeof(boot_gdt) - 1),
    0 /* patched at runtime by 32-bit entry before lgdt */
};

/* PVH ELF note so `qemu-system-x86_64 -kernel <elf64>` boots directly at
 * `eshkol_x86_64_pvh_entry` in 32-bit protected mode without a bootloader. */
__asm__(
    ".pushsection .note.Xen, \"a\", @note\n\t"
    ".balign 4\n\t"
    ".long 4\n\t"                     /* namesz: "Xen\0" */
    ".long 4\n\t"                     /* descsz: 32-bit physical entry address */
    ".long 18\n\t"                    /* XEN_ELFNOTE_PHYS32_ENTRY */
    ".asciz \"Xen\"\n\t"
    ".balign 4\n\t"
    ".long eshkol_x86_64_pvh_entry\n\t"
    ".balign 4\n\t"
    ".popsection\n\t"

    ".pushsection .text.boot, \"ax\", @progbits\n\t"
    ".code32\n\t"
    ".global eshkol_x86_64_pvh_entry\n\t"
    "eshkol_x86_64_pvh_entry:\n\t"
    "cli\n\t"
    "cld\n\t"
    /* Zero boot_pml4, boot_pdpt, boot_pd (3 * 4096 bytes = 3072 dwords) */
    "xor %eax, %eax\n\t"
    "mov $boot_pml4, %edi\n\t"
    "mov $1024, %ecx\n\t"
    "rep stosl\n\t"
    "mov $boot_pdpt, %edi\n\t"
    "mov $1024, %ecx\n\t"
    "rep stosl\n\t"
    "mov $boot_pd, %edi\n\t"
    "mov $1024, %ecx\n\t"
    "rep stosl\n\t"
    /* Link PML4[0] -> PDPT, PDPT[0] -> PD */
    "mov $boot_pdpt, %eax\n\t"
    "or $0x03, %eax\n\t"
    "mov %eax, boot_pml4\n\t"
    "mov $boot_pd, %eax\n\t"
    "or $0x03, %eax\n\t"
    "mov %eax, boot_pdpt\n\t"
    /* Identity-map 512 x 2 MiB = 1 GiB via PS bit (0x83) */
    "mov $boot_pd, %edi\n\t"
    "xor %ebx, %ebx\n\t"
    "mov $512, %ecx\n\t"
    "1:\n\t"
    "mov %ebx, %eax\n\t"
    "or $0x83, %eax\n\t"
    "mov %eax, (%edi)\n\t"
    "movl $0, 4(%edi)\n\t"
    "add $0x200000, %ebx\n\t"
    "add $8, %edi\n\t"
    "dec %ecx\n\t"
    "jnz 1b\n\t"
    /* Load CR3 = boot_pml4 */
    "mov $boot_pml4, %eax\n\t"
    "mov %eax, %cr3\n\t"
    /* Enable PAE in CR4 */
    "mov %cr4, %eax\n\t"
    "or $0x20, %eax\n\t"
    "mov %eax, %cr4\n\t"
    /* Enable IA32_EFER.LME (bit 8) via MSR 0xC0000080 */
    "mov $0xC0000080, %ecx\n\t"
    "rdmsr\n\t"
    "or $0x100, %eax\n\t"
    "wrmsr\n\t"
    /* Enable paging (CR0.PG) and protected mode (CR0.PE) */
    "mov %cr0, %eax\n\t"
    "or $0x80000001, %eax\n\t"
    "mov %eax, %cr0\n\t"
    /* Patch GDT base and load GDTR */
    "mov $boot_gdt, %eax\n\t"
    "mov %eax, (boot_gdtr + 2)\n\t"
    "lgdt boot_gdtr\n\t"
    /* Far jump to 64-bit code segment 0x08 */
    "ljmp $0x08, $eshkol_x86_64_long_entry\n\t"

    ".code64\n\t"
    "eshkol_x86_64_long_entry:\n\t"
    "mov $0x10, %ax\n\t"
    "mov %ax, %ds\n\t"
    "mov %ax, %es\n\t"
    "mov %ax, %ss\n\t"
    "mov %ax, %fs\n\t"
    "mov %ax, %gs\n\t"
    "lea __eshkol_stack_top(%rip), %rsp\n\t"
    "and $-16, %rsp\n\t"
    "call eshkol_embedded_start\n\t"
    "2:\n\t"
    "cli\n\t"
    "hlt\n\t"
    "jmp 2b\n\t"
    ".popsection\n\t"
);

static inline void outb(uint16_t port, uint8_t val) {
    __asm__ volatile("outb %0, %1" :: "a"(val), "Nd"(port) : "memory");
}

static inline uint8_t inb(uint16_t port) {
    uint8_t ret;
    __asm__ volatile("inb %1, %0" : "=a"(ret) : "Nd"(port) : "memory");
    return ret;
}

void eshkol_board_init(void) {
    /* Initialize 16550 UART on COM1: 115200 8N1, FIFO enabled, IRQs off. */
    outb(COM1_BASE + 1u, 0x00u);
    outb(COM1_BASE + 3u, 0x80u);
    outb(COM1_BASE + 0u, 0x01u);
    outb(COM1_BASE + 1u, 0x00u);
    outb(COM1_BASE + 3u, 0x03u);
    outb(COM1_BASE + 2u, 0xC7u);
    outb(COM1_BASE + 4u, 0x0Bu);
}

void eshkol_board_putc(uint32_t c) {
    while ((inb(COM1_LSR) & COM1_LSR_THRE) == 0u) {
    }
    outb(COM1_BASE, (uint8_t)c);
}

/* On x86_64 PC, the console is port-mapped I/O (COM1 0x3F8), not MMIO;
 * expose a scratch byte for volatile-store! smoke tests while board_putc
 * drives COM1 directly. */
static volatile uint8_t g_x86_console_shadow;
volatile uint8_t* eshkol_board_console_register(void) {
    return &g_x86_console_shadow;
}

void eshkol_board_exit(uint32_t code) {
    /* QEMU isa-debug-exit exits with ((val << 1) | 1).
     * Write 0 for success (QEMU exits 1 -> harness maps 1 -> 0) or code+1 for failure. */
    outb(ISA_DEBUG_EXIT_PORT, (uint8_t)code);
    for (;;) {
        __asm__ volatile("cli\n\thlt" ::: "memory");
    }
}
