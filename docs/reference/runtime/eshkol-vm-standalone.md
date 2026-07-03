# `eshkol-vm-standalone` — Bytecode VM & ESKB Format

`eshkol-vm-standalone` is the standalone bytecode virtual machine. It runs
compiled Eshkol bytecode (`.eskb`) files, and can also compile-and-run `.esk`
source directly. It is the execution engine for the VM profiles
(`hosted-vm`, `freestanding-vm`, `embedded-vm`) that `eshkol-run --emit-eskb`
targets.

## Invoking the VM

It uses a hand-rolled argv parser (there is **no** `--help`):

```
eshkol-vm-standalone [--trace] [--emit-eskb PATH] <input.eskb | input.esk> [program args...]
```

| Token | Effect |
|-------|--------|
| `<file>.eskb` | Load and run the bytecode chunk |
| `<file>.esk` | Compile the source and run it |
| `--trace` | Enable per-instruction VM tracing |
| `--emit-eskb PATH` | Emit bytecode to `PATH` |
| `--` | Treat the rest of argv as positional |
| *(no args)* | Run the built-in VM self-tests |

Everything from the input file onward is passed to the program as its command
line. Set `ESHKOL_VM_NO_DISASM=1` to suppress the disassembly dump.

### Verified example

```sh
# Emit bytecode from source (VM profile required):
$ eshkol-run --profile hosted-vm -B out.eskb prog.esk
[ESKB] Wrote 17130 bytes to out.eskb (3 functions, 4198 instructions, 735 constants)
[ESKB] Emitted bytecode to out.eskb

# Run it on the standalone VM:
$ eshkol-vm-standalone out.eskb
[ESKB] Loaded out.eskb: 4141 instructions, 735 constants
=== Eshkol VM — running out.eskb ===
3
=== Execution complete ===
```

## Emitting ESKB with `eshkol-run`

Bytecode emission is done by `eshkol-run` under a VM profile:

```
eshkol-run --profile <hosted-vm|freestanding-vm|embedded-vm> --emit-eskb OUT.eskb INPUT.esk
```

`--emit-eskb` (short form `-B`) requires a VM profile. VM profiles set the
backend to the VM and **forbid** JIT eval/run, `--shared-lib`, `--wasm`, and
`--lib`. `embedded-vm` additionally forbids `--target` and hardens admission:
host-only native policy, and it rejects string constants and desktop native
calls (so the bytecode is portable to a constrained embedded VM).

### Entry-point admission gates

After emission you can assert that the produced bytecode exposes named entry
points; on failure the emitted file is deleted and the command errors.

| Flag | Meaning |
|------|---------|
| `--require-vm-entry NAME` | Re-load the emitted ESKB and require a function named `NAME` exists |
| `--require-vm-entry-zero-arg NAME` | Require `NAME` exists as a **zero-argument** entry with no upvalues |

Both require a VM profile plus `--emit-eskb`.

## ESKB binary format

ESKB ("Eshkol Bytecode") is a section-based binary using LEB128 variable-length
encoding. Defined in `lib/backend/eskb_format.h`.

- **Magic** `0x45534B42` = ASCII `"ESKB"`; **version** `1`.
- **16-byte header**: `magic`, `version`, `flags`, `checksum` (CRC32, polynomial
  `0xEDB88320`, over everything after the header).
- **Flags**: `LITTLE_ENDIAN` (0x01), `DEBUG_INFO` (0x02).
- **Sections** (`{u8 id, u32 size}` descriptors): `CONST` (0), `CODE` (1),
  `META`/debug (2), `SYMB`/export table (3).
- **Constant tags**: NIL 0, INT64 1, F64 2, BOOL 3, STRING 6.

Two emit variants exist: `eshkol_emit_eskb` (desktop — keeps the desktop builtin
preamble) and `eshkol_emit_eskb_embedded` (omits the preamble and rejects
bytecode that still reaches the desktop native table). `--profile embedded-vm`
selects the embedded variant.

> Note: the same `"ESKB"` 4-character code is also used by an unrelated
> knowledge-base persistence file format (v2) in `lib/core/kb_persistence.cpp`;
> that is not VM bytecode.

## Related

- [`eshkol-run`](eshkol-run.md) — the `--profile` / `--emit-eskb` /
  `--require-vm-entry` flags.
- [Platform target matrix](../../platform/TARGET_SUPPORT_MATRIX.md) — artifact
  expectations for the VM profiles.
- `ESHKOL_VM_MAX_INSN` in [environment variables](environment-variables.md) —
  the VM runaway-instruction guard.
