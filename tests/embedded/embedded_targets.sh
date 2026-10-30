#!/usr/bin/env bash
# Copyright (C) tsotchke
# SPDX-License-Identifier: MIT
#
# Bare-metal target matrix for Eshkol's freestanding profiles.
#
#   embedded_targets.sh runtime <source-dir> <out-dir>
#       Cross-compile the freestanding runtime subset (lib/platform/embedded)
#       for every reference target into <out-dir>/<target>/libeshkol_embedded.a.
#
#   embedded_targets.sh check <eshkol-run> <source-dir> <work-dir> [qemu]
#       For every reference target: compile tests/embedded/fib_checksum.esk
#       with --profile freestanding-mcu-native, link it with the runtime
#       subset and the board link script, and verify the ELF (class, machine,
#       e_flags, build attributes, entry point, no undefined symbols).  With
#       the trailing `qemu` argument, also boot each image under QEMU and
#       require the expected console line and exit status 0.
#
# Tools come from ESHKOL_EMBEDDED_LLVM_BIN (a directory holding clang,
# ld.lld, llvm-ar, llvm-readelf and llvm-nm) or PATH; QEMU from PATH.
# Exit 77 (CTest skip) when a required tool is missing.  Never writes
# outside <out-dir> / <work-dir>.

set -u

SKIP=77
EXPECTED_LINE="eshkol bare-metal fib checksum 429027928"

# target | board | clang flags | expected readelf -h class | machine | e_flags | attribute needles (;-separated) | qemu command
TARGETS=(
  "thumbv7em-none-eabi|mps2-an386|--target=thumbv7em-none-eabi -mcpu=cortex-m4 -mfloat-abi=soft -mthumb|ELF32|ARM|0x5000200|Tag_CPU_arch: v7E-M;Tag_CPU_arch_profile: Microcontroller;Tag_THUMB_ISA_use: Thumb-2|qemu-system-arm -M mps2-an386 -semihosting-config enable=on,target=native"
  "thumbv7em-none-eabihf|mps2-an386|--target=thumbv7em-none-eabihf -mcpu=cortex-m4 -mfloat-abi=hard -mfpu=fpv4-sp-d16 -mthumb|ELF32|ARM|0x5000400|Tag_CPU_arch: v7E-M;Tag_CPU_arch_profile: Microcontroller;Tag_ABI_VFP_args: AAPCS VFP|qemu-system-arm -M mps2-an386 -semihosting-config enable=on,target=native"
  "aarch64-unknown-none|qemu-virt-aarch64|--target=aarch64-none-elf|ELF64|AArch64|0x0||qemu-system-aarch64 -M virt -cpu cortex-a53 -semihosting"
  "riscv32imac-unknown-none-elf|qemu-virt-riscv|--target=riscv32-unknown-elf -march=rv32imac -mabi=ilp32 -mcmodel=medany|ELF32|RISC-V|0x1|Tag_RISCV_arch: rv32i2p1_m2p0_a2p1_c2p0|qemu-system-riscv32 -M virt -bios none"
  "riscv64gc-unknown-none-elf|qemu-virt-riscv|--target=riscv64-unknown-elf -march=rv64gc -mabi=lp64d -mcmodel=medany|ELF64|RISC-V|0x5|Tag_RISCV_arch: rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0|qemu-system-riscv64 -M virt -bios none"
)

tool() {
  local name=$1
  if [ -n "${ESHKOL_EMBEDDED_LLVM_BIN:-}" ] && [ -x "$ESHKOL_EMBEDDED_LLVM_BIN/$name" ]; then
    echo "$ESHKOL_EMBEDDED_LLVM_BIN/$name"
    return 0
  fi
  command -v "$name" 2>/dev/null
}

require_tools() {
  CLANG=$(tool clang) || { echo "SKIP: clang not found"; exit $SKIP; }
  LLD=$(tool ld.lld) || { echo "SKIP: ld.lld not found"; exit $SKIP; }
  AR=$(tool llvm-ar) || { echo "SKIP: llvm-ar not found"; exit $SKIP; }
  READELF=$(tool llvm-readelf) || { echo "SKIP: llvm-readelf not found"; exit $SKIP; }
  NM=$(tool llvm-nm) || { echo "SKIP: llvm-nm not found"; exit $SKIP; }
}

fail() { echo "FAIL: $*"; exit 1; }

field() { echo "$1" | cut -d'|' -f"$2"; }

# build_runtime <source-dir> <out-dir> <target> <board> <clang flags>
build_runtime() {
  local src=$1 out=$2 target=$3 board=$4 flags=$5
  local dir="$out/$target"
  mkdir -p "$dir/obj"
  local objs=()
  local f
  for f in "$src"/lib/platform/embedded/runtime/*.c "$src"/lib/platform/embedded/boards/"$board"/*.c; do
    local o="$dir/obj/$(basename "$f" .c).o"
    # shellcheck disable=SC2086
    "$CLANG" $flags -ffreestanding -fno-builtin -nostdlib -O2 -g0 \
      -ffunction-sections -fdata-sections -Wall -Werror \
      -I "$src/lib/platform/embedded/include" -c "$f" -o "$o" ||
      fail "$target: compiling $(basename "$f")"
    objs+=("$o")
  done
  rm -f "$dir/libeshkol_embedded.a"
  "$AR" rcs "$dir/libeshkol_embedded.a" "${objs[@]}" || fail "$target: archiving runtime"
  cp "$src/lib/platform/embedded/boards/$board/link.ld" "$dir/link.ld"
}

cmd_runtime() {
  local src=$1 out=$2
  require_tools
  local row
  for row in "${TARGETS[@]}"; do
    build_runtime "$src" "$out" "$(field "$row" 1)" "$(field "$row" 2)" "$(field "$row" 3)"
    echo "runtime $(field "$row" 1): $out/$(field "$row" 1)/libeshkol_embedded.a"
  done
}

cmd_check() {
  local eshkol_run=$1 src=$2 work=$3 mode=${4:-}
  require_tools
  [ -x "$eshkol_run" ] || fail "eshkol-run not found: $eshkol_run"
  mkdir -p "$work"
  local passed=0 ran_qemu=0 row
  for row in "${TARGETS[@]}"; do
    local target board flags class machine flags_hex attrs qemu
    target=$(field "$row" 1); board=$(field "$row" 2); flags=$(field "$row" 3)
    class=$(field "$row" 4); machine=$(field "$row" 5); flags_hex=$(field "$row" 6)
    attrs=$(field "$row" 7); qemu=$(field "$row" 8)
    local dir="$work/$target"
    build_runtime "$src" "$work/runtime" "$target" "$board" "$flags"
    mkdir -p "$dir"

    local obj="$dir/fib_checksum.o" elf="$dir/fib_checksum.elf"
    (cd "$dir" && "$eshkol_run" --profile freestanding-mcu-native --target "$target" \
        -o "$obj" "$src/tests/embedded/fib_checksum.esk") > "$dir/compile.log" 2>&1 ||
      { cat "$dir/compile.log"; fail "$target: eshkol-run compile"; }
    if grep -q "is not a recognized" "$dir/compile.log"; then
      cat "$dir/compile.log"; fail "$target: compiler used a CPU/feature foreign to the target"
    fi

    # The program object may only reference the board and memory primitives.
    local undefined
    undefined=$("$NM" -u "$obj" | awk '{print $NF}' | sort -u | tr '\n' ' ')
    local sym
    for sym in $undefined; do
      case "$sym" in
        eshkol_board_*|memcpy|memmove|memset|memcmp) ;;
        *) fail "$target: object references hosted symbol '$sym' (undefined: $undefined)" ;;
      esac
    done

    "$LLD" -T "$work/runtime/$target/link.ld" --gc-sections -o "$elf" \
      "$obj" "$work/runtime/$target/libeshkol_embedded.a" > "$dir/link.log" 2>&1 ||
      { cat "$dir/link.log"; fail "$target: link"; }

    local header attributes
    header=$("$READELF" -h "$elf")
    echo "$header" | grep -q "Class:[[:space:]]*$class" || fail "$target: ELF class is not $class"
    echo "$header" | grep -q "Machine:[[:space:]]*$machine" || fail "$target: e_machine is not $machine"
    echo "$header" | grep -Eq "Flags:[[:space:]]*$flags_hex(,|$|[[:space:]])" ||
      { echo "$header" | grep Flags; fail "$target: e_flags are not $flags_hex"; }
    [ -z "$("$NM" -u "$elf")" ] || fail "$target: linked image has undefined symbols"
    "$NM" "$elf" | grep -q " T eshkol_main$" || fail "$target: eshkol_main is not a global text symbol"
    if [ -n "$attrs" ]; then
      attributes=$("$READELF" -A "$elf" 2>&1)
      local needle
      IFS=';' read -r -a needles <<< "$attrs"
      for needle in "${needles[@]}"; do
        echo "$attributes" | grep -q "$needle" ||
          { echo "$attributes"; fail "$target: build attribute '$needle' missing"; }
      done
    fi
    echo "PASS objects $target ($class $machine flags $flags_hex)"

    if [ "$mode" = qemu ]; then
      local qbin=${qemu%% *}
      if ! command -v "$qbin" > /dev/null 2>&1; then
        echo "SKIP qemu $target: $qbin not found"
      else
        local out rc
        # shellcheck disable=SC2086
        out=$(perl -e 'alarm shift; exec @ARGV' 60 $qemu -display none -monitor none \
                -serial stdio -kernel "$elf" 2>&1)
        rc=$?
        echo "$out" > "$dir/qemu.log"
        echo "$out" | grep -q "$EXPECTED_LINE" ||
          { echo "$out"; fail "$target: console line missing under $qbin"; }
        [ "$rc" -eq 0 ] || { echo "$out"; fail "$target: exit status $rc under $qbin"; }
        echo "PASS qemu $target: '$EXPECTED_LINE' exit 0"
        ran_qemu=$((ran_qemu + 1))
      fi
    fi
    passed=$((passed + 1))
  done
  if [ "$mode" = qemu ] && [ "$ran_qemu" -eq 0 ]; then
    echo "SKIP: no qemu-system-* emulator found"
    exit $SKIP
  fi
  echo "PASS $passed/${#TARGETS[@]} targets"
}

case "${1:-}" in
  runtime) shift; [ $# -eq 2 ] || fail "usage: $0 runtime <source-dir> <out-dir>"; cmd_runtime "$@" ;;
  check) shift; [ $# -ge 3 ] || fail "usage: $0 check <eshkol-run> <source-dir> <work-dir> [qemu]"; cmd_check "$@" ;;
  *) fail "usage: $0 runtime|check ..." ;;
esac
