/* Arm MPS2+ AN386 (Cortex-M4): code in SSRAM1 at 0x0, data in SSRAM2/3. */
ENTRY(eshkol_reset_handler)

MEMORY
{
  FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 4M
  RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 4M
}

SECTIONS
{
  .isr_vector : { KEEP(*(.isr_vector)) } > FLASH
  .text : { *(.text .text.*) } > FLASH
  .rodata : { *(.rodata .rodata.*) } > FLASH
  .ARM.exidx : { *(.ARM.exidx .ARM.exidx.*) } > FLASH

  .data : {
    __eshkol_data_start = .;
    *(.data .data.*)
    . = ALIGN(4);
    __eshkol_data_end = .;
  } > RAM AT > FLASH
  __eshkol_data_load = LOADADDR(.data);

  .bss (NOLOAD) : {
    __eshkol_bss_start = .;
    *(.bss .bss.* COMMON)
    . = ALIGN(4);
    __eshkol_bss_end = .;
  } > RAM

  __eshkol_stack_top = ORIGIN(RAM) + LENGTH(RAM);
  /DISCARD/ : { *(.comment) *(.note*) }
}
