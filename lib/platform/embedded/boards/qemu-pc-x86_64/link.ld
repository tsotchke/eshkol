/*
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 *
 * Link script for `qemu-system-x86_64 -M pc` PVH ELF64 kernel at 1 MiB.
 */
ENTRY(eshkol_x86_64_pvh_entry)

SECTIONS {
    . = 0x00100000;

    .note.Xen : ALIGN(4) {
        KEEP(*(.note.Xen))
    }

    .text : ALIGN(4096) {
        KEEP(*(.text.boot))
        *(.text .text.*)
    }

    .rodata : ALIGN(4096) {
        *(.rodata .rodata.*)
    }

    __eshkol_data_load = .;
    .data : ALIGN(4096) {
        __eshkol_data_start = .;
        *(.data .data.*)
        __eshkol_data_end = .;
    }

    .bss (NOLOAD) : ALIGN(4096) {
        __eshkol_bss_start = .;
        *(.bss .bss.*)
        *(COMMON)
        __eshkol_bss_end = .;
    }

    . = ALIGN(16);
    . += 64K;
    __eshkol_stack_top = .;

    /DISCARD/ : {
        *(.comment)
        *(.eh_frame*)
    }
}
