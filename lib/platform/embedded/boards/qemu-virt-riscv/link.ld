/* QEMU virt (RISC-V, -bios none): RAM from 0x80000000, entry at its start. */
ENTRY(eshkol_riscv_entry)

SECTIONS
{
  . = 0x80000000;
  .text : { KEEP(*(.text.boot)) *(.text .text.*) }
  .rodata : { *(.rodata .rodata.* .srodata .srodata.*) }
  .data : {
    __eshkol_data_start = .;
    *(.data .data.*)
    __global_pointer$ = . + 0x800;
    *(.sdata .sdata.*)
    . = ALIGN(16);
    __eshkol_data_end = .;
  }
  __eshkol_data_load = __eshkol_data_start;
  .bss (NOLOAD) : {
    __eshkol_bss_start = .;
    *(.sbss .sbss.* .bss .bss.* COMMON)
    . = ALIGN(16);
    __eshkol_bss_end = .;
  }
  . = ALIGN(16);
  . += 0x10000;
  __eshkol_stack_top = .;
  /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) }
}
