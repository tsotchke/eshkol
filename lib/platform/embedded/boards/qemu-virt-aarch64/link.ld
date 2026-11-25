/* QEMU virt (AArch64): RAM from 0x40000000; the image sits above the
 * device tree QEMU places at the start of RAM. */
ENTRY(eshkol_aarch64_entry)

SECTIONS
{
  . = 0x40100000;
  .text : { KEEP(*(.text.boot)) *(.text .text.*) }
  .rodata : { *(.rodata .rodata.*) }
  .data : {
    __eshkol_data_start = .;
    *(.data .data.*)
    . = ALIGN(16);
    __eshkol_data_end = .;
  }
  __eshkol_data_load = __eshkol_data_start;
  .bss (NOLOAD) : {
    __eshkol_bss_start = .;
    *(.bss .bss.* COMMON)
    . = ALIGN(16);
    __eshkol_bss_end = .;
  }
  . = ALIGN(16);
  . += 0x10000;
  __eshkol_stack_top = .;
  /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) }
}
