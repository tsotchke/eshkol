# Platform Layer: Windows, Events, Real-Time Audio, MIDI

**Status:** v1.8.0-platform. Design record: [ADR-0053](../design/adr/0053-platform-layer.md).
Blueprint: [Multimedia System Architecture](../future/MULTIMEDIA_SYSTEM_ARCHITECTURE.md) section 8.

The platform layer gives Eshkol programs and C embedders one interface for
windows with pixel surfaces, keyboard/mouse/touch/window events, real-time
audio with callback I/O, and MIDI input/output, on every POSIX host. Each
subsystem has native backends and a headless backend; the headless backends
run the same public code paths and are what the test suites use on build
machines without displays or sound cards.

Companion pages: [Vulkan compute](VULKAN_COMPUTE.md) and
[embedded targets](EMBEDDED_TARGETS.md) complete the v1.8.0-platform surface.

## Backends

| Subsystem | Backend | Hosts | Binding | Selected by `auto` when |
|---|---|---|---|---|
| display | `offscreen` | all POSIX | built in | no native backend is usable |
| display | `x11` | Linux, BSD | `libX11.so.6` at run time; compiled where the Xlib headers exist | `libX11` loads and `$DISPLAY` is set |
| display | `cocoa` | macOS | AppKit at run time; classes resolved by name | a window-server session exists |
| audio | `null` | all POSIX | built in: a real-time clock thread | no native library loads |
| audio | `alsa` | Linux | `libasound.so.2` at run time; no development headers needed | `libasound` loads |
| audio | `coreaudio` | macOS | AudioToolbox and CoreAudio at run time | always on macOS |
| MIDI | `loopback` | all POSIX | built in, in-process | no native backend is usable |
| MIDI | `alsa` (sequencer) | Linux | `libasound.so.2` at run time | `libasound` loads and `/dev/snd/seq` is readable and writable |
| MIDI | `coremidi` | macOS | CoreMIDI at run time | always on macOS |

Planned for v1.8.1: Wayland and Win32 display backends, WASAPI audio, WinMM
MIDI, a PulseAudio/PipeWire audio backend, named (non-default) CoreAudio
devices, and compiling the headless backends on Windows. Each depends on an
SDK that the v1.8.0 build machines do not carry.

Because every native backend binds its system library when first used, the
runtime archive and every AOT binary carry no X11, ALSA, AppKit, AudioToolbox
or CoreMIDI link dependency: a program that never opens a window or device
behaves exactly as before, and a binary built on one host runs on a host
without those libraries, where `auto` selects the headless backend.

### Choosing a backend

```sh
ESHKOL_PLATFORM_DISPLAY=offscreen|x11|cocoa|auto
ESHKOL_PLATFORM_AUDIO=null|alsa|coreaudio|auto
ESHKOL_PLATFORM_MIDI=loopback|alsa|coremidi|auto
```

From Eshkol: `(platform-select-backend! 'display "offscreen")`, and
`(platform-backend 'audio)` names the current choice. From C:
`eshkol_platform_select_backend(ESHKOL_PLATFORM_AUDIO, ESHKOL_BACKEND_ALSA)`.
An explicitly named backend that is not available is reported as
"backend unavailable"; selection never silently substitutes another backend.

## Windows and events

```scheme
(require platform.window)

(define w (window-create "demo" 320 200 'resizable))
(window-clear! w (rgb 20 20 40))
(window-fill-rect! w 10 10 100 50 (rgb 255 128 0))
(window-present! w)

(let loop ()
  (let ((e (window-wait-event w 16)))
    (cond ((and e (eq? (event-type e) 'window-close)) (window-destroy! w))
          ((and e (key-pressed? e) (= (key-code e) (key-code-of 'escape)))
           (window-destroy! w))
          (else
           (when (and e (eq? (event-type e) 'mouse-down))
             (window-set-pixel! w (mouse-x e) (mouse-y e) (rgb 255 255 255))
             (window-present! w))
           (loop)))))
```

- **Surface.** Every window owns an XRGB8888 surface (`0xRRGGBB` integers).
  `window-present!` copies it to the display: an `XPutImage` of a client-side
  image on X11, a `CGImage` drawn by the content view on Cocoa, a frame copy
  offscreen. `window-presented-pixel` reads back what the display received
  (`XGetImage`, the view's rendered backing, or the copied frame), so the
  present path is checked end to end. Cocoa tags the image with the window's
  own colour space, so presenting performs no colour conversion.
- **Events.** Each window has a bounded (1024), thread-safe event queue.
  `window-poll-event` and `window-wait-event` pump the backend, translate OS
  events, and pop the next one; `window-push-event!` (C:
  `eshkol_window_push_event`) may be called from any thread. A full queue
  drops the new event and counts it (`window-dropped-events`).
- **Event types** follow the numbering of the multimedia architecture:
  `key-down key-up key-repeat text-input mouse-move mouse-down mouse-up
  mouse-scroll mouse-enter mouse-leave touch-begin touch-move touch-end
  touch-cancel window-resize window-move window-close window-focus
  window-blur window-show window-hide window-expose quit timer user`.
- **Key codes** are platform independent: a printable key is its unshifted
  US-layout character (`a`..`z`, digits, punctuation), control keys are their
  ASCII control value (return 13, escape 27, ...), and arrows, navigation,
  F1-F12 and modifier keys live at 0x1000 and above (`key-code-of` names
  them). `key-unicode` carries the character the key produced.
  X11 reports auto-repeat as `key-repeat` (the release/press pair with equal
  timestamps is folded); Cocoa uses `isARepeat`.
- **Geometry.** `window-set-size!` and `window-set-position!` produce the same
  `window-resize` / `window-move` events a user-driven change would; the
  surface is reallocated to the new size, keeping the overlapping pixels.
- **Close.** The window manager's close button produces `window-close` and sets
  `window-should-close?`; the program decides when to `window-destroy!`.
- **Threads.** X11 windows each own their display connection, so a window may
  live on any thread as long as one thread uses it. Cocoa requires every
  window call on the process main thread (an AppKit rule); a call from another
  thread is refused with "invalid state".

Touch events are part of the event record and the push API; no v1.8.0
backend generates them from hardware. Touch input from X11 XInput2 and Cocoa
trackpad touches is Planned for v1.8.1.

## Real-time audio

```scheme
(require platform.audio)

(define out (audio-open-output 48000 2 256))      ; rate, channels, period frames
(audio-write-samples! out tone-samples)            ; interleaved, in [-1, 1]
(audio-start! out)
...
(audio-stats out)    ; => ((callbacks . 412) (frames . 105472) (underruns . 0) ...)
(audio-latency-ms out)
(audio-close! out)
```

**The real-time contract.** The backend's audio thread (the null clock
thread, the ALSA I/O thread, or CoreAudio's own I/O thread) calls the device
callback once per period. The callback path performs no allocation, takes no
lock and never blocks: it reads the monotonic clock, runs the callback and
updates counters with atomic operations.

- **Callback mode (C).** `eshkol_audio_open_output(&config, callback, userdata)`
  runs `callback(userdata, buffer, frames)` on the audio thread with the
  interleaved buffer in the configured format (`u8 s16 s24 s32 f32`).
- **Queue mode (C and Eshkol).** With no callback, the built-in callback moves
  frames between the device and a lock-free single-producer/single-consumer
  ring. Output: Eshkol stages samples with `audio-write!` and publishes them
  atomically with `audio-commit!` (`audio-write-samples!` does both), so the
  audio thread never sees a half-written frame. A short period is padded with
  silence and counted as an underrun. Input: captured frames enter the ring
  and are read with `audio-read`/`audio-read-samples`; a full ring drops the
  newest frames and counts an overrun.

**Counters** (`audio-stats`, C `eshkol_audio_get_stats`): callbacks, frames,
underruns, underrun-frames, overruns, overrun-frames, device-xruns (ALSA
`-EPIPE` recoveries, CoreAudio render failures), last and maximum callback
duration, maximum period jitter (largest deviation of period start spacing
from `period / rate`), device latency in frames (ALSA `snd_pcm_delay`, CoreAudio
device latency + safety offset + buffer size, null: one period), queued
frames, negotiated period and rate. `audio-latency-ms` adds device latency
and queued frames.

Backend specifics:

- **null** runs one period every `period / rate` seconds against absolute
  monotonic deadlines (no drift accumulates), discarding output and delivering
  silence as input. It requests `SCHED_FIFO` where the process is allowed it.
- **ALSA** opens the PCM in blocking interleaved mode with four periods of
  buffering, adopts the period size ALSA grants, and recovers xruns with
  `snd_pcm_recover`. `device` names any ALSA PCM (`default`, `hw:0,0`, `null`).
- **CoreAudio** uses the default-output unit for output and the HAL unit on the
  default input device for capture, with the requested buffer frame size.
  Capture requires the user's microphone permission.

## MIDI

```scheme
(require platform.midi)

(define out (midi-open-output "eshkol-synth"))     ; a virtual source others can connect to
(define in  (midi-open-input "Keystation"))       ; connect to a source by name
(midi-note-on! out 0 60 100)
(midi-receive in 100)          ; => (144 60 100) or #f after 100 ms
(midi-last-timestamp in)       ; arrival time in microseconds
(midi-sources)                 ; names visible on the backend
```

- Output ports publish a named, subscribable source (CoreMIDI virtual source,
  ALSA sequencer port with read/subscribe capability, or a loopback source).
  Input ports connect to the source whose name matches exactly, otherwise the
  first containing the name.
- Messages are complete MIDI 1.0 messages; `midi-send!` refuses malformed
  ones (a length that does not match the message, data byte with the high bit, unterminated SysEx,
  undefined status). Incoming byte streams pass through one parser that
  expands running status, delivers interleaved real-time bytes immediately and
  reassembles SysEx up to 1024 bytes.
- Received messages cross from the backend thread to the reader through a
  64 KiB lock-free ring with arrival timestamps; a full ring drops and counts
  whole messages (`midi-stats`: sent, received, dropped, malformed).

## Capabilities

Under an installed capability policy, opening a window requires `'display`,
an audio device `'audio`, and a MIDI port `'midi`
([capabilities.md](../reference/stdlib/capabilities.md)).

## C ABI

Headers: `inc/eshkol/platform/{platform,event,window,audio,midi}.h`. The
lifecycle functions follow section 8 of the multimedia architecture
(`eshkol_window_create`, `eshkol_window_get_surface`, `eshkol_window_present`,
`eshkol_window_poll_event`, `eshkol_audio_open_output`, `eshkol_audio_queue`,
...), and `_ex` variants take an explicit backend. The `eshkol_plat_*`
functions are the Eshkol language bridge: they take generation-checked
integer handles from the runtime handle registry
(`inc/eshkol/core/handle.h`, kinds window/audio-device/MIDI-port), so a closed
handle, or a handle of another kind, is refused instead of being dereferenced.

## Verification

| Check | Command | Where it runs |
|---|---|---|
| headless layer | `ctest -R platform_headless` | every POSIX build |
| X11 backend | `ctest -R platform_x11` (under `xvfb-run` when present) | Linux with Xvfb |
| ALSA PCM and sequencer | `ctest -R platform_alsa` (alsa-lib `null` PCM; sequencer when accessible) | Linux with libasound |
| Eshkol surface, JIT and AOT | `ctest -R 'platform_(window\|audio\|midi)_esk'` | every POSIX build |
| Cocoa, CoreAudio, CoreMIDI | `platform_test macos` | a Mac with a logged-in session |

`platform_test macos` creates a hidden Cocoa window, checks the present path
pixel for pixel, posts synthetic AppKit key and mouse events and checks their
translation, runs CoreAudio output for 200 ms (period count, underruns,
latency) and loops MIDI through a CoreMIDI virtual source. It is a manual
check because it needs a window-server session and an audio device.

Manual checks for devices a build machine lacks:

1. **CoreAudio capture:** `ESHKOL_PLATFORM_AUDIO=coreaudio` and
   `(audio-open-input 48000 1 256)`, start, speak, and confirm non-zero samples
   from `audio-read-samples` (grant microphone access when macOS asks).
2. **ALSA hardware:** `(audio-open-output 48000 2 256)` with the user in the
   `audio` group, a 440 Hz tone for one second, `device-xruns` stays 0.
3. **ALSA sequencer:** with `/dev/snd/seq` accessible, `platform_test alsa`
   runs the sequencer loopback checks; `aconnect -l` lists the `eshkol` client
   and its ports while they are open.
4. **Visible windows:** run the example above under X11 or Cocoa; click to
   draw, press Escape or the close button to exit.
