# Embedded Targets

Eshkol compiles freestanding programs to bare-metal object code for
microcontrollers and bare-metal application processors. This page covers the
supported targets, the language subset a freestanding image uses, the
freestanding runtime subset, the reference boards, and how each target is
verified.

## Targets

`eshkol-run --profile freestanding-mcu-native --target <triple>` (or
`freestanding-kernel-native`) emits a relocatable ELF object for:

| `--target` | Machine | ABI / float | Code model | ELF facts |
|---|---|---|---|---|
| `thumbv7em-none-eabi` | Cortex-M4, ARMv7E-M Thumb-2 | AAPCS, soft float | small | ELF32, `EM_ARM`, `e_flags 0x5000200`, `Tag_CPU_arch: v7E-M`, `Tag_CPU_arch_profile: Microcontroller` |
| `thumbv7em-none-eabihf` | Cortex-M4F, FPv4-SP-D16 | AAPCS-VFP, hard float | small | ELF32, `EM_ARM`, `e_flags 0x5000400`, `Tag_ABI_VFP_args: AAPCS VFP` |
| `aarch64-unknown-none` | ARMv8-A, FP/SIMD | AAPCS64 | small | ELF64, `EM_AARCH64` |
| `riscv32imac-unknown-none-elf` | RV32IMAC | ilp32 | medlow | ELF32, `EM_RISCV`, `e_flags 0x1` (RVC, soft-float ABI) |
| `riscv64gc-unknown-none-elf` | RV64GC | lp64d | medany | ELF64, `EM_RISCV`, `e_flags 0x5` (RVC, double-float ABI) |
| `riscv64imac-unknown-none-elf` | RV64IMAC | lp64 | medany | ELF64, `EM_RISCV`, `e_flags 0x1` |

Aliases: `thumbv7em-unknown-none-eabi[hf]`, `aarch64-unknown-none-elf`,
`aarch64-none-elf`, `riscv32-unknown-none-elf` (RV32IMAC) and
`riscv64-unknown-none-elf` (RV64GC).

Each spelling resolves to one row of the embedded target table
(`eshkol::profile::find_embedded_target`, `lib/core/execution_profile.cpp`).
The row fixes the canonical LLVM triple, CPU, feature string, ABI name, float
ABI and code model, and the compiler records the row in the module
(`eshkol.embedded-target` module flag, plus `target-abi` on RISC-V). Every
target machine the compiler creates for that module, for the data layout and
for object emission, takes its configuration from the row, so the data
layout, instruction selection, ELF `e_flags` and ARM build attributes always
agree. Bare-metal objects use static relocations and one section per function
and datum, so a link script with `--gc-sections` keeps only what the image
uses. Feature strings match what clang selects for the same `-mcpu`/`-march`/
`-mabi`, so Eshkol objects link with C objects built by clang for the same
target without attribute or ABI mismatches.

Other triples (for example `x86_64-unknown-linux-gnu` for kernel-class code)
keep the generic path: the target's baseline CPU, never the build machine's.

## The freestanding language subset

A freestanding image has no hosted runtime: no garbage-collected heap, no
numeric tower, no ports, no closures that allocate. What it has:

- **Machine-typed functions.** Parameters and returns annotated with
  `u8 u16 u32 u64 i8 i16 i32 i64 usize isize ptr` (and `null` returns).
- **Machine-integer arithmetic.** `+ - * quotient remainder modulo`,
  `< > <= >= =`, `bitwise-and bitwise-or bitwise-xor bitwise-not`,
  `bit-shift-left bit-shift-right arithmetic-shift` compile to native
  instructions. The operation width and signedness come from the operands
  with a declared machine type (the widest wins; on a tie an unsigned type
  wins); integer literals and untyped values adopt it; with no typed operand
  the operation is `i64`. Results wrap modulo 2^width, division by zero traps
  (`llvm.trap`), and comparisons of unsigned types use unsigned predicates.
  `usize`/`isize` are pointer-sized for the target.
- **Memory and ordering.** `volatile-load`, `volatile-store!`, `atomic-*`,
  `compiler-fence`, `memory-fence`, `ptr-add`, `usize->ptr`, `ptr->usize`,
  `target-intrinsic`.
- **Calls into C.** `(extern <ret> name <args...> :real symbol)` calls a C
  function with its native ABI (board support, vendor HALs).
- **C-ABI exports.** An `:export-symbol` function whose parameters and return
  are all machine-typed is exported under its symbol with the platform C ABI,
  so reset handlers, vector tables and C code can call it directly. Its
  Eshkol body stays internal as `<symbol>.eshkol`. `:section`, `:align`,
  `:used`, `:weak` and `:no-return` apply to the exported entry.
- **Recursion and tail calls.** Self tail calls compile to loops.

Operations that need the hosted runtime (reals and exact rationals, strings,
lists, vectors, closures with captured state) are rejected at compile time in
a freestanding profile.

## Freestanding runtime subset

`lib/platform/embedded/` is the C side of a bare-metal image. It needs no C
library and no operating system.

| Path | Contents |
|---|---|
| `include/eshkol_embedded.h` | The image contract: `uint32_t eshkol_main(void)` from the program, the startup and memory primitives, the board hooks |
| `runtime/memory.c` | `memcpy`, `memmove`, `memset`, `memcmp` (LLVM lowers aggregate copies and zeroing to these) |
| `runtime/start.c` | `eshkol_embedded_start`: copy `.data`, zero `.bss`, `eshkol_board_init()`, `eshkol_board_exit(eshkol_main())`; `eshkol_board_puts`, `eshkol_board_put_u32` |
| `boards/<board>/` | Reset or entry code, console and exit hooks, link script |

Board hooks: `eshkol_board_init`, `eshkol_board_putc` (waits for transmit
space), `eshkol_board_console_register` (the transmit-data register, for
programs that write the console with `volatile-store!`), and
`eshkol_board_exit` (never returns).

### Reference boards

| Board | Targets | Entry | Console | Exit |
|---|---|---|---|---|
| `mps2-an386` (`qemu-system-arm -M mps2-an386`) | `thumbv7em-none-eabi`, `thumbv7em-none-eabihf` | Cortex-M vector table, reset handler (enables CP10/CP11 for hard float) | CMSDK UART0 at `0x40004000` | Arm semihosting `SYS_EXIT_EXTENDED` |
| `qemu-virt-aarch64` (`-M virt -cpu cortex-a53`) | `aarch64-unknown-none` | `eshkol_aarch64_entry` at EL1: stack, `CPACR_EL1.FPEN` | PL011 at `0x09000000` | Arm semihosting `SYS_EXIT` |
| `qemu-virt-riscv` (`-M virt -bios none`) | `riscv32imac-unknown-none-elf`, `riscv64gc-unknown-none-elf` | `eshkol_riscv_entry` in M-mode at `0x80000000`: `gp`, stack, `mstatus.FS` when the ABI uses F/D | NS16550A at `0x10000000` | SiFive test finisher at `0x100000` |

On every board the emulator's exit status is the value `eshkol_main` returned.

## Building and running an image

```sh
# 1. Program object (example: tests/embedded/fib_checksum.esk)
eshkol-run --profile freestanding-mcu-native --target riscv64gc-unknown-none-elf \
    -o fib.o tests/embedded/fib_checksum.esk

# 2. Runtime subset for every reference target (needs the LLVM toolchain's clang)
cmake --build build --target eshkol-embedded-runtime
#    -> build/embedded/<target>/libeshkol_embedded.a and link.ld

# 3. Link
ld.lld -T build/embedded/riscv64gc-unknown-none-elf/link.ld --gc-sections \
    -o fib.elf fib.o build/embedded/riscv64gc-unknown-none-elf/libeshkol_embedded.a

# 4. Run
qemu-system-riscv64 -M virt -bios none -display none -monitor none \
    -serial stdio -kernel fib.elf
# eshkol bare-metal fib checksum 429027928      (exit status 0)
```

The equivalent QEMU lines for the other boards are in
`tests/embedded/embedded_targets.sh`.

`tests/embedded/fib_checksum.esk` exercises the subset end to end: u32
modular arithmetic (Fibonacci numbers 0..63 overflow 32 bits), tail-call
loops, decimal formatting with `quotient`/`remainder`, console output through
`volatile-store!` to the board's transmit register, and a C-ABI
`eshkol_main` whose result becomes the exit status.

## Verification

| CTest | What it checks | Needs |
|---|---|---|
| `embedded_targets_objects` | For each of the five reference targets: the program compiles with no foreign CPU/feature, the object references only board hooks and memory primitives, the image links with no undefined symbols, `eshkol_main` is a global text symbol, ELF class, machine, `e_flags` and build attributes match the table above | clang, ld.lld, llvm-ar, llvm-readelf, llvm-nm from the LLVM toolchain |
| `embedded_targets_qemu` | Everything above, then boots each image and requires the console line and exit status 0 | additionally `qemu-system-arm`, `-aarch64`, `-riscv32`, `-riscv64` |
| `freestanding_object_smoke_test`, `freestanding_lowlevel_object_test`, `freestanding_elf_link_smoke_test`, `freestanding_reject_hosted_builtin_test`, `execution_profile_test` | Profile resolution, hosted-symbol exclusion, low-level forms | eshkol-run |

Both embedded tests exit 77 (reported as skipped) when their tools are not
installed. The driver also runs by hand:

```sh
tests/embedded/embedded_targets.sh check build/eshkol-run . build/embedded-check        # objects
tests/embedded/embedded_targets.sh check build/eshkol-run . build/embedded-qemu qemu    # + QEMU
```

## Planned

- Planned for v1.8.1: more boards (STM32 Cortex-M4 parts, an RV32 MCU such as
  the GD32VF103 / ESP32-C3 class) with flashable `.bin`/`.hex` output.
- Planned for v1.8.1: interrupt handlers written in Eshkol (`:section` plus
  vector-table generation from `:export-symbol` entries).
- Planned for v1.8.2: a freestanding numeric subset (fixed-point and
  soft-float reals) so real arithmetic is available without the hosted tower.
