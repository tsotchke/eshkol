---
kind: explanation
status: current
owner-area: platform
since: v1.8.0
sources:
  - inc/eshkol/platform/platform.h
  - inc/eshkol/platform/event.h
  - inc/eshkol/platform/window.h
  - inc/eshkol/platform/audio.h
  - inc/eshkol/platform/midi.h
  - lib/platform/
  - lib/backend/gpu/gpu_vulkan.c
  - lib/backend/gpu/gpu_memory_stub.cpp
  - lib/backend/gpu/gpu_memory_cuda.cpp
  - lib/core/execution_profile.cpp
  - lib/platform/embedded/
  - cmake/EshkolPlatform.cmake
---
# ADR-0053: The v1.8.0 platform layer

**Status:** Accepted
**Scope:** Windows and events, real-time audio, MIDI, Vulkan compute in the
ordinary GPU dispatch, and embedded cross-compilation (ARM and RISC-V bare
metal).

## Context

The v1.8.0-platform rung asks for cross-platform windowing, an event system,
callback-based real-time audio, MIDI, Vulkan compute beyond Metal and CUDA,
and embedded cross-compilation. The multimedia architecture
(docs/future/MULTIMEDIA_SYSTEM_ARCHITECTURE.md, section 8) gives the C
interface for windows and audio. The runtime already had a generation-checked
handle registry (`inc/eshkol/core/handle.h`), the capability policy, the
`extern` FFI used by `core.net` and `core.threads`, a GPU dispatch with an
`ESHKOL_GPU_VULKAN` slot reserved, and freestanding execution profiles with
typed low-level forms.

Three constraints shape the design. Build machines and CI runners mostly have
no display, no sound card and no MIDI devices, yet the code paths must be
tested. The runtime archive is linked into every AOT program, so a new link
dependency would reach every user. And several SDKs (Wayland, Win32, WASAPI,
MoltenVK) are not present on the machines this rung is built and verified on.

## Decision

1. **One C ABI per subsystem, one object model.** `inc/eshkol/platform/`
   declares windows (with an XRGB8888 surface and a per-window event queue),
   events (type numbering from the architecture document, platform-independent
   key codes), audio devices and MIDI ports. A backend is a table of function
   pointers selected per subsystem; the common code owns validation, queues,
   rings, counters and the Eshkol bridge, so every backend is a thin
   translation layer.

2. **Every native backend binds its system library at run time.** X11
   (`libX11.so.6`), ALSA (`libasound.so.2`), AppKit, AudioToolbox, CoreAudio,
   CoreMIDI and the Vulkan loader are opened with the dynamic loader on first
   use. ALSA and Vulkan entry points are declared from their stable C ABIs (the
   ALSA sequencer event layout is checked against the kernel header where it
   exists; the Vulkan declarations are checked against the Khronos header by a
   test), so neither needs development headers to build. AppKit classes are
   resolved by name and the view and delegate classes are registered with the
   Objective-C runtime, so the macOS link line is unchanged. A program that
   never opens a device is unaffected, and a binary runs on hosts without the
   libraries.

3. **Each subsystem has a headless backend that runs the same paths.**
   Offscreen windows emit the same geometry events a window system reports;
   the null audio backend is a real-time clock thread driving the same period
   processing as a device; MIDI loopback delivers through the same parser and
   ring as the OS backends. `auto` selects the native backend whose library
   loads, otherwise the headless one; an explicitly named backend that is not
   available is an error, never a silent substitution.

4. **Real-time paths are lock-free and allocation-free.** The audio thread
   reads the monotonic clock, runs the device callback, and updates counters
   with atomics; queue mode moves frames through a single-producer/
   single-consumer ring whose writes are staged and published with one release
   store, so the audio thread never observes a partial frame. Underruns,
   overruns, device xruns, callback durations, period jitter and latency are
   counted on the audio thread. MIDI input crosses from the backend thread
   through a lock-free ring of timestamped records.

5. **The Eshkol surface uses the handle registry.** `lib/platform/*.esk`
   call `eshkol_plat_*` bridge functions with generation-checked integer
   handles (new registry kinds: window, audio device, MIDI port), and each
   platform object carries a kind tag checked on every borrow. A closed or
   wrong-kind handle is refused, never dereferenced. Opening a device requires
   the `display`, `audio` or `midi` capability under an installed policy.

6. **Vulkan is a backend of the ordinary GPU dispatch.** `gpu_vulkan.c`
   implements the `eshkol_gpu_*` operations whose f64 meaning Vulkan compute
   reproduces exactly (elementwise arithmetic, full reductions, matmul) with
   embedded SPIR-V generated from GLSL sources by a checked script. The CUDA
   and CPU-only builds route Vulkan buffers to it when no CUDA device
   initialises or `ESHKOL_GPU_BACKEND=vulkan` is set; every other operation
   returns nonzero, which callers already treat as "use the CPU path". Results
   are cross-checked bit for bit against the CPU reference on every reachable
   driver.

7. **Embedded targets extend the freestanding profiles.** The existing
   `freestanding-mcu-native` and `freestanding-kernel-native` profiles gain one
   target-description table (CPU, features, ABI per triple) for Cortex-M4
   Thumb, AArch64, RV32IMAC and RV64GC bare metal, plus a freestanding runtime
   subset (`lib/platform/embedded/`: memory primitives, start-up code, console
   and exit hooks, linker scripts for the QEMU reference boards). See
   docs/platform/EMBEDDED_TARGETS.md.

## Consequences

- Windows, audio and MIDI are usable from Eshkol and C on Linux and macOS,
  and every layer is exercised in CI on any POSIX host without devices.
- The display backend list stays open: Wayland and Win32 are additional
  backend tables against the same window object, as are WASAPI and WinMM.
  These are Planned for v1.8.1 together with compiling the headless backends
  on Windows; this rung's build machines carry none of those SDKs.
- Cocoa windows must be driven from the main thread (an AppKit rule the API
  checks and reports).
- Vulkan transcendental operations in f64, Apple Vulkan through MoltenVK, and
  further kernels (softmax, axis reductions, backward passes) remain on the
  CPU path until implemented as shaders.
- Touch is part of the event record; no v1.8.0 backend produces touch events
  from hardware yet.
