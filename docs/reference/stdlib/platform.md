# platform.window, platform.audio, platform.midi

Windows with pixel surfaces and events, real-time audio, and MIDI I/O.
Architecture, backends and the real-time contract:
[docs/platform/PLATFORM_LAYER.md](../../platform/PLATFORM_LAYER.md).
Design record: [ADR-0053](../../design/adr/0053-platform-layer.md).

Each module is required individually and refuses to open a device when a
capability policy is active and lacks `'display`, `'audio` or `'midi`
respectively. Errors from the runtime (unavailable backend, closed handle,
invalid argument, device error) are raised with `error`; `raise`/`guard`
handle them as usual. Backends are chosen with `ESHKOL_PLATFORM_DISPLAY`,
`ESHKOL_PLATFORM_AUDIO` and `ESHKOL_PLATFORM_MIDI` or
`platform-select-backend!`.

## platform.window

| Procedure | Result |
|---|---|
| `(window-create title width height flag ...)` | a window; flags `'resizable 'borderless 'fullscreen 'hidden 'always-on-top` |
| `(window-destroy! w)` | closes the window; later use raises |
| `(window? x)` `(window-backend w)` | predicate; `"offscreen"`, `"x11"` or `"cocoa"` |
| `(window-width w)` `(window-height w)` `(window-size w)` `(window-position w)` | integers; pairs |
| `(window-set-size! w width height)` `(window-set-position! w x y)` `(window-set-title! w s)` | queue `window-resize` / `window-move` events |
| `(window-should-close? w)` `(window-request-close! w)` | close state; queues `window-close` |
| `(window-fill-rect! w x y width height color)` `(window-set-pixel! w x y color)` `(window-clear! w color)` | draw on the surface (clipped) |
| `(window-pixel w x y)` | surface pixel `0xRRGGBB` |
| `(window-present! w)` `(window-present-count w)` | copy the surface to the display; count |
| `(window-presented-pixel w x y)` | the pixel the display received |
| `(window-poll-event w)` | next event or `#f` |
| `(window-wait-event w)` `(window-wait-event w timeout-ms)` | next event; `#f` on timeout |
| `(window-push-event! w type a b c)` | queue an event (key: keycode modifiers unicode; mouse: x y button; resize: width height; move: x y; user: code data1 data2) |
| `(window-dropped-events w)` | events dropped by a full queue |
| `(rgb r g b)` `(rgb-red c)` `(rgb-green c)` `(rgb-blue c)` | colour packing |
| `(event? e)` `(event-type e)` `(event-timestamp e)` `(event-window e)` | event type symbol, microseconds, window id |
| `(key-event? e)` `(key-pressed? e)` `(key-code e)` `(key-scancode e)` `(key-modifiers e)` `(key-unicode e)` | keyboard fields |
| `(mouse-event? e)` `(mouse-x e)` `(mouse-y e)` `(mouse-dx e)` `(mouse-dy e)` `(mouse-button e)` `(mouse-buttons e)` `(scroll-x e)` `(scroll-y e)` | pointer fields |
| `(window-event? e)` `(event-width e)` `(event-height e)` `(event-x e)` `(event-y e)` | window geometry fields |
| `(user-event-code e)` `(user-event-data1 e)` `(user-event-data2 e)` | user event payload |
| `(key-code-of key)` | key code of a character or `'return 'escape 'left 'f1`-style name |
| `modifier-shift` `modifier-ctrl` `modifier-alt` `modifier-meta` `button-left` `button-middle` `button-right` | constants |
| `(platform-backend 'display)` `(platform-select-backend! 'display name)` | backend selection (also `'audio`, `'midi`) |
| `(platform-error? x)` | recognises platform error vectors |

Event type symbols: `key-down key-up key-repeat text-input mouse-move
mouse-down mouse-up mouse-scroll mouse-enter mouse-leave touch-begin
touch-move touch-end touch-cancel window-resize window-move window-close
window-focus window-blur window-show window-hide window-expose quit timer user`.

## platform.audio

| Procedure | Result |
|---|---|
| `(audio-open direction rate channels format period-frames ring-frames)` | a queue-mode device; direction `'output`/`'input`, format `'u8 's16 's24 's32 'f32`, ring 0 = 8 periods |
| `(audio-open-output rate channels period-frames)` `(audio-open-input rate channels period-frames)` | `f32` shorthands |
| `(audio-start! d)` `(audio-stop! d)` `(audio-pause! d)` `(audio-close! d)` | stream control |
| `(audio-write! d sample)` | stage one interleaved sample in [-1, 1]; `#f` when the ring is full |
| `(audio-commit! d)` | publish staged samples to the audio thread; frames published |
| `(audio-write-samples! d list-or-vector)` | stage and commit; samples accepted |
| `(audio-available d)` `(audio-read d)` `(audio-read-samples d n)` | captured frames; next sample or `#f`; a vector of up to n samples |
| `(audio-stats d)` `(audio-stat d name)` | every counter as an alist; one counter |
| `(audio-latency-ms d)` | device latency plus queued frames, in ms |
| `(audio-backend d)` `(audio-device? x)` `(audio-format name)` | `"null"`, `"alsa"` or `"coreaudio"`; predicate; format code |

Counters: `callbacks frames underruns underrun-frames overruns overrun-frames
device-xruns last-callback-ns max-callback-ns max-jitter-ns
device-latency-frames queued-frames period-frames sample-rate`.

## platform.midi

| Procedure | Result |
|---|---|
| `(midi-open-output name)` | a port publishing the source `name` |
| `(midi-open-input source-name)` | a port connected to that source |
| `(midi-close! p)` `(midi-port? x)` `(midi-backend p)` | close; predicate; `"loopback"`, `"alsa"` or `"coremidi"` |
| `(midi-send! p bytes)` | send one complete message (list or vector of bytes); malformed messages raise |
| `(midi-note-on! p ch note vel)` `(midi-note-off! p ch note vel)` `(midi-control-change! p ch cc v)` `(midi-program-change! p ch prog)` `(midi-pitch-bend! p ch value)` | channel messages (pitch bend 0..16383, 8192 centre) |
| `(midi-receive p)` `(midi-receive p timeout-ms)` | next message as a byte list, or `#f` |
| `(midi-last-timestamp p)` | arrival time of the last received message, microseconds |
| `(midi-sources)` | source names on the selected backend |
| `(midi-stats p)` | `((sent . n) (received . n) (dropped . n) (malformed . n))` |
