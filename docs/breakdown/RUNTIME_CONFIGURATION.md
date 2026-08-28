# Runtime Configuration

**Status:** Production (v1.3.5-evolve)
**Applies to:** Eshkol compiler v1.2.0-scale and later

---

## Overview

Eshkol programs can be configured through a layered system that resolves settings from four sources, in order of increasing priority:

1. **Default values** (lowest priority) -- built into the compiler
2. **Config file** -- `.eshkol.toml` (project-local), `~/.config/eshkol/config.toml` (XDG), or `~/.eshkol/config.toml`
3. **Environment variables** -- `ESHKOL_*` prefix
4. **Command-line flags** (highest priority) -- passed to `eshkol-run`

The configuration system is defined in `inc/eshkol/core/config.h` (unified config) and `inc/eshkol/core/resource_limits.h` (runtime limits).

---

## Environment Variables

### Runtime Limits

| Variable | Default | Description |
|----------|---------|-------------|
| `ESHKOL_MAX_HEAP` | 1 GB | Maximum heap allocation in bytes. Supports `K`, `M`, `G` suffixes. |
| `ESHKOL_TIMEOUT_MS` | 30000 | Execution timeout in milliseconds. Set to `0` for unlimited. |
| `ESHKOL_MAX_STACK` | 100000 | Maximum recursion depth (number of stack frames). |
| `ESHKOL_MAX_TENSOR_ELEMS` | 1,000,000,000 | Maximum number of elements in a single tensor. |
| `ESHKOL_MAX_STRING_LEN` | 100 MB | Maximum string length in bytes. |
| `ESHKOL_VM_MAX_INSN` | 10,000,000 | Bytecode-VM runaway-instruction guard. Set to `0` for unlimited. |
| `ESHKOL_ENFORCE_LIMITS` | `true` | When `true`, hard limit violations terminate the process. When `false`, the breach is recorded and warned about and execution continues. |
| `ESHKOL_LIMIT_WARNINGS` | `true` | When `true`, log warnings when soft limits are approached. |

Malformed runtime-limit values are ignored and leave the documented default in
place. Size variables accept optional `K`, `M`, or `G` suffixes, with an optional
trailing `B`.

Each ceiling is **opt-in**: it binds a run only when that run sets the
variable. The defaults above are the values a limit takes when you turn it on,
not ceilings every program is silently held to — see
[environment-variables.md](../reference/runtime/environment-variables.md#limits-are-opt-in).

Exceeding an active hard limit under the default `ESHKOL_ENFORCE_LIMITS=true` flushes
pending output, prints one `eshkol: fatal: …` line to stderr naming the limit,
the ceiling and the variable that set it, and exits with a status specific to
that limit:

| Limit | Exit status |
|-------|-------------|
| `ESHKOL_MAX_HEAP` | 120 |
| `ESHKOL_MAX_STACK` | 121 |
| `ESHKOL_MAX_TENSOR_ELEMS` | 122 |
| `ESHKOL_MAX_STRING_LEN` | 123 |
| `ESHKOL_TIMEOUT_MS` | 124 |
| `ESHKOL_VM_MAX_INSN` | 125 |

`124` matches GNU coreutils `timeout(1)`, the convention this project already
uses for `run-command` / `run-argv` subprocess timeouts. The full set is defined
as `ESHKOL_EXIT_LIMIT_*` in `inc/eshkol/core/resource_limits.h`. A program that
stays under its ceilings is unaffected: the checks sit on the arena's
block-acquisition path, on object creation, and on periodic VM/loop
checkpoints, and none of them reads or writes a program value.

### Bytecode VM Region Reclamation

Added in v1.3.5 (SW-14, `lib/backend/vm_region_evac.c`): `with-region`
reclaims on the bytecode VM the same way it does on native codegen, via a
Stage-1 mark-and-sweep evacuator over the VM's index-addressed heap. See
[memory model](../reference/runtime/memory-model.md#which-engine-reclaims)
for the full mechanism and
[environment-variables.md](../reference/runtime/environment-variables.md)
for the complete descriptions; the variables themselves:

| Variable | Default | Description |
|----------|---------|-------------|
| `ESHKOL_VM_REGION_EVAC` | on | `0` disables VM region reclamation entirely and restores the pre-Stage-1 pass-through (for A/B measurement). |
| `ESHKOL_VM_REGION_VERIFY` | off | After each region pop, independently audits the object table for any surviving reference to a retired index. Implied by `ESHKOL_ARENA_POISON`. |
| `ESHKOL_VM_REGION_VERIFY_FATAL` | off | Makes that audit exit nonzero, so a CI lane can gate on it. |
| `ESHKOL_VM_REGION_COMPACT` | on | `0` stops a surviving object's header from being copied out of the dying region (diagnostic only; keeps addresses stable). Forced off under `ESHKOL_ARENA_POISON`. |
| `ESHKOL_VM_REGION_RECYCLE` | on | `0` stops retired heap indices from being reused; a stale reference then reads as invalid forever instead of aliasing a new object. Forced off under `ESHKOL_ARENA_POISON`. |
| `ESHKOL_VM_REGION_QUIET` | off | Suppresses the one-time stderr note that a VM `region-close` (the handle surface, still bookkeeping-only in Stage-1) reclaims no heap. |
| `ESHKOL_VM_HEAP_BUDGET_MB` | 1024 | VM arena size past which a diagnostic names the growth and its cause — for allocation that happens *outside* a region, which the VM still never reclaims. `0` disables the watchdog. |
| `ESHKOL_VM_HEAP_BUDGET_FATAL` | off | Makes crossing the heap budget exit nonzero instead of advisory. |

Re-verified for this documentation wave, run directly against a
from-source build of commit `487c2a62` (`#461` merged onto `694c3179`):

```
$ bash tests/memory/vm_region_flat_rss_test.sh
iterations=1000  peak RSS=25 MB  answer=120000
iterations=4000  peak RSS=26 MB  answer=480000
iterations=16000 peak RSS=27 MB  answer=1920000
unwrapped control (begin instead of with-region): peak RSS=704 MB
at 16000 iterations: with-region+evacuator=27 MB, evacuator disabled=793 MB
vm-region-flat-rss: 6 passed, 0 failed  -- PASS

$ bash tests/memory/vm_region_evac_subtype_coverage_test.sh
peak RSS: default=80 MB, poison=130 MB, reclaim-off=129 MB
vm-region-evac-subtype-coverage: 8 passed, 0 failed  -- PASS

$ bash tests/memory/vm_region_growth_watchdog_test.sh
vm-region-watchdog: 10 passed, 0 failed  -- PASS
```

The exact peak-RSS figures move a megabyte or two run to run (25-27 MB
flat rather than a single fixed number); the CHANGELOG's own numbers from
the same fixture (26/26/26 MB, 796 MB disabled) are consistent with this
run within that noise band. What is gated and does not move: the curve is
flat with reclamation on, an order of magnitude (or more) larger with it
off, and the returned answer is identical either way.

### Stack Size

| Variable | Default | Description |
|----------|---------|-------------|
| `ESHKOL_STACK_SIZE` | 512 MB | OS-level stack size in bytes. Minimum 1 MB. Affects deep recursion capacity. |

The stack size is set at process startup by `eshkol_init_stack_size()` in `lib/core/runtime_stack_hosted.cpp`. On macOS, the main thread stack is also set at link time via `-Wl,-stack_size`. The `ESHKOL_STACK_SIZE` environment variable overrides the default for both the main thread (via `setrlimit`) and spawned threads.

The maximum recursion depth (`ESHKOL_MAX_STACK` / `ESHKOL_DEFAULT_MAX_STACK_DEPTH`) is a separate software limit tracked by `eshkol_stack_push()` / `eshkol_stack_pop()`. With the default 512 MB OS stack, approximately 80,000+ frames are supported; the software default of 100,000 frames provides a safety margin.

### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `ESHKOL_LOG_LEVEL` | `WARN` | Minimum log level: `DEBUG`, `INFO`, `WARN`, `ERROR`, `NONE`. |
| `ESHKOL_LOG_FORMAT` | `TEXT` | Log output format: `TEXT` (human-readable) or `JSON` (structured). |
| `ESHKOL_LOG_FILE` | (stderr) | Path to log file. If unset, logs go to stderr. |

### Optimization and Acceleration

| Variable | Default | Description |
|----------|---------|-------------|
| `ESHKOL_OPT_LEVEL` | 2 | LLVM optimization level (0-3). |
| `ESHKOL_ENABLE_SIMD` | `true` | Enable SIMD vectorization in tensor operations. |
| `ESHKOL_ENABLE_XLA` | `false` | Enable XLA backend for tensor operations. |
| `ESHKOL_ENABLE_GPU` | `false` | Enable GPU acceleration (Metal on macOS, CUDA on Linux). |

### GPU-Specific Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ESHKOL_GPU_MATMUL_THRESHOLD` | 100000 | Element count threshold for GPU matmul dispatch. Set to `1` to force all matmul through GPU. |
| `ESHKOL_GPU_PRECISION` | `exact` | GPU precision mode: `exact` (sf64, 53-bit), `high` (df64, ~48-bit), `fast` (f32, 24-bit). |
| `ESHKOL_SF64_KERNEL` | `v2` | Software float64 kernel version: `v1` (original) or `v2` (deferred rounding). |

### Debug and Diagnostics

| Variable | Default | Description |
|----------|---------|-------------|
| `ESHKOL_DEBUG` | (unset) | Enable debug output from the compiler and runtime. |
| `ESHKOL_DUMP_REPL_IR` | (unset) | Set to `1` to print JIT-compiled LLVM IR to stderr in the REPL. |
| `ESHKOL_DEBUG_DL` | (unset) | Set to `1` to print DataLayout and target triple information for ABI debugging. |

### Library Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `ESHKOL_LIB_PATH` | (unset) | Colon-separated list of directories to search for libraries and precompiled modules. |

---

## Config File

Eshkol searches for a TOML configuration file in these locations (first found wins):

1. `./.eshkol.toml` -- project-local configuration
2. `./eshkol.toml` -- project-local, no leading dot
3. `~/.config/eshkol/config.toml` -- XDG standard location
4. `~/.eshkol/config.toml` -- home directory fallback

### Example Config File

```toml
# .eshkol.toml

[runtime]
max_heap = "2G"
timeout_ms = 60000
max_stack = 200000

[logging]
level = "info"
format = "text"

[optimization]
llvm_opt_level = 2
enable_simd = true
enable_gpu = false

[debug]
dump_ast = false
dump_ir = false

[features]
strict_mode = false
enable_warnings = true
color_output = true
```

`[types] strict` / `[types] unsafe` are documented as configuration but are not
yet read by `apply_config_section` in `lib/core/config.cpp`, which handles
`runtime`, `logging`, `optimization`, `debug` and `features` only; today
`--strict-types` and the unsafe flag are CLI-only. Wiring the section is an
open build item.

---

## Resource Limits

The resource limits system provides runtime enforcement of memory, time, and structural constraints. It is defined in `inc/eshkol/core/resource_limits.h`.

### Heap Memory

- **Hard limit:** Maximum total heap allocation (default 1 GB).
- **Soft limit:** Warning threshold at 80% of the hard limit.
- **Tracking:** Every arena allocation calls `eshkol_track_allocation()` to check against limits.
- **Near-limit check:** `eshkol_is_near_memory_limit()` returns true when within 10% of the hard limit.

```c
bool eshkol_track_allocation(size_t bytes);   // returns false if limit exceeded
size_t eshkol_get_heap_usage(void);           // current total
size_t eshkol_get_peak_heap_usage(void);      // high-water mark
```

### Execution Timeout

The timeout watchdog monitors execution time and can terminate long-running operations:

```c
void eshkol_start_timer(uint64_t timeout_ms);  // 0 = use configured limit
void eshkol_stop_timer(void);
bool eshkol_is_timed_out(void);
uint64_t eshkol_get_remaining_time_ms(void);
```

The execution timeout is opt-in: with `ESHKOL_TIMEOUT_MS` unset there is no
timeout at all. Setting the variable arms the watchdog; `ESHKOL_TIMEOUT_MS=0`
arms it with no ceiling, and any other value sets the ceiling in milliseconds
(30000 when the variable is set but unparseable).

### Stack Depth

Stack depth is tracked in software independently of the OS stack size:

```c
bool   eshkol_stack_push(void);       // returns false on overflow
void   eshkol_stack_pop(void);
size_t eshkol_get_stack_depth(void);
```

### Data Structure Limits

```c
bool eshkol_check_tensor_size(size_t num_elements);  // default: 1 billion elements
bool eshkol_check_string_length(size_t length);       // default: 100 MB
```

### Error Reporting

```c
typedef enum {
    ESHKOL_LIMIT_OK = 0,
    ESHKOL_LIMIT_HEAP_SOFT,       // Soft heap limit (warning only)
    ESHKOL_LIMIT_HEAP_HARD,       // Hard heap limit exceeded
    ESHKOL_LIMIT_TIMEOUT,         // Execution timeout
    ESHKOL_LIMIT_STACK_OVERFLOW,  // Stack depth exceeded
    ESHKOL_LIMIT_TENSOR_SIZE,     // Tensor too large
    ESHKOL_LIMIT_STRING_LENGTH    // String too long
} eshkol_limit_error_t;

eshkol_limit_error_t eshkol_get_last_limit_error(void);
const char* eshkol_limit_error_message(eshkol_limit_error_t error);
```

### Diagnostics

```c
void eshkol_print_resource_stats(void);       // print usage report
void eshkol_reset_resource_tracking(void);    // reset all counters
```

---

## C++ RAII Helpers

For C++ code that integrates with the Eshkol runtime, RAII guard classes are provided:

### StackFrameGuard

Automatically pushes/pops a stack frame. Useful in codegen helpers.

```cpp
{
    eshkol::StackFrameGuard guard;
    if (!guard) {
        // stack overflow -- handle gracefully
        return;
    }
    // ... function body ...
}  // automatic stack_pop on scope exit
```

### TimerGuard

Starts and stops the execution timer on scope entry/exit:

```cpp
{
    eshkol::TimerGuard timer(5000);  // 5 second timeout
    // ... long operation ...
    if (timer.isTimedOut()) {
        // timed out
    }
}  // automatic timer stop
```

### Macros

```cpp
ESHKOL_STACK_GUARD()                 // early-return void on overflow
ESHKOL_STACK_GUARD_WITH_VALUE(val)   // early-return val on overflow
```

---

## Implementation Files

| File | Purpose |
|------|---------|
| `inc/eshkol/core/config.h` | Unified configuration structure and API |
| `inc/eshkol/core/resource_limits.h` | Resource limit definitions and tracking |
| `lib/core/resource_limits.cpp` | Limit parsing, tracking and check implementations |
| `lib/core/runtime_stack_hosted.cpp` | Stack size initialization (`eshkol_init_stack_size`) |

---

## See Also

- [Command-Line Reference](COMMAND_LINE_REFERENCE.md) -- All compiler flags
- [Developer Tools](DEVELOPER_TOOLS.md) -- Debug flags and REPL IR dumps
- [Memory Management](MEMORY_MANAGEMENT.md) -- Arena allocation internals
- [Benchmarking](BENCHMARKING.md) -- Performance measurement and tuning
