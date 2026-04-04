# Runtime Configuration

**Status:** Production (v1.1.11)
**Applies to:** Eshkol compiler v1.1-accelerate and later

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
| `ESHKOL_ENFORCE_LIMITS` | `true` | When `true`, hard limit violations terminate the process. When `false`, errors are returned. |
| `ESHKOL_LIMIT_WARNINGS` | `true` | When `true`, log warnings when soft limits are approached. |

### Stack Size

| Variable | Default | Description |
|----------|---------|-------------|
| `ESHKOL_STACK_SIZE` | 512 MB | OS-level stack size in bytes. Minimum 1 MB. Affects deep recursion capacity. |

The stack size is set at process startup by `eshkol_init_stack_size()` in `lib/core/arena_memory.cpp`. On macOS, the main thread stack is also set at link time via `-Wl,-stack_size`. The `ESHKOL_STACK_SIZE` environment variable overrides the default for both the main thread (via `setrlimit`) and spawned threads.

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
2. `~/.config/eshkol/config.toml` -- XDG standard location
3. `~/.eshkol/config.toml` -- home directory fallback

### Example Config File

```toml
# .eshkol.toml

[runtime]
max_heap = "2G"
timeout_ms = 60000
max_stack_depth = 200000

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

[types]
strict = false
unsafe = false
```

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

The default timeout is 30 seconds. Set `ESHKOL_TIMEOUT_MS=0` for unlimited execution.

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
| `lib/core/arena_memory.cpp` | Stack size initialization (`eshkol_init_stack_size`) |

---

## See Also

- [Command-Line Reference](COMMAND_LINE_REFERENCE.md) -- All compiler flags
- [Developer Tools](DEVELOPER_TOOLS.md) -- Debug flags and REPL IR dumps
- [Memory Management](MEMORY_MANAGEMENT.md) -- Arena allocation internals
- [Benchmarking](BENCHMARKING.md) -- Performance measurement and tuning
