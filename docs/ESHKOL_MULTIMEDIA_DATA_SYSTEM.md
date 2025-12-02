# Eshkol Multimedia and Data System

## Technical Specification

**Version:** 1.0
**Date:** 2025-12-01
**Status:** Design Specification

---

## Table of Contents

1. [Overview](#1-overview)
2. [Design Principles](#2-design-principles)
3. [Type Universe Structure](#3-type-universe-structure)
4. [Core Type Families](#4-core-type-families)
   - 4.1 [Handle Types](#41-handle-types)
   - 4.2 [Buffer Types](#42-buffer-types)
   - 4.3 [Stream Types](#43-stream-types)
   - 4.4 [Event Types](#44-event-types)
5. [Linear Resource Management](#5-linear-resource-management)
   - 5.1 [The Linear Modality](#51-the-linear-modality)
   - 5.2 [Resource Lifecycle](#52-resource-lifecycle)
   - 5.3 [Borrowing and Lending](#53-borrowing-and-lending)
   - 5.4 [Escape Hatches](#54-escape-hatches)
6. [Dependent Dimensions](#6-dependent-dimensions)
   - 6.1 [Static Dimension Tracking](#61-static-dimension-tracking)
   - 6.2 [Dimension Proofs](#62-dimension-proofs)
   - 6.3 [Dynamic Dimensions](#63-dynamic-dimensions)
7. [Runtime Representation](#7-runtime-representation)
   - 7.1 [Tagged Value Extensions](#71-tagged-value-extensions)
   - 7.2 [Buffer Memory Layout](#72-buffer-memory-layout)
   - 7.3 [Handle Structures](#73-handle-structures)
   - 7.4 [Proof Erasure](#74-proof-erasure)
8. [Primitive Operations](#8-primitive-operations)
   - 8.1 [Window Operations](#81-window-operations)
   - 8.2 [Graphics Operations](#82-graphics-operations)
   - 8.3 [Audio Operations](#83-audio-operations)
   - 8.4 [Network Operations](#84-network-operations)
   - 8.5 [Buffer Operations](#85-buffer-operations)
9. [Implementation Architecture](#9-implementation-architecture)
   - 9.1 [Compiler Layer (C++/LLVM)](#91-compiler-layer-cllvm)
   - 9.2 [Platform Layer (C)](#92-platform-layer-c)
   - 9.3 [Library Layer (Eshkol)](#93-library-layer-eshkol)
10. [Integration with Eshkol Core](#10-integration-with-eshkol-core)
11. [Example Programs](#11-example-programs)

---

## 1. Overview

The Eshkol Multimedia and Data System provides a type-safe foundation for handling:

- **Graphics**: Windows, surfaces, pixel buffers, drawing operations
- **Audio**: Playback, recording, synthesis, streaming
- **Networking**: TCP/UDP sockets, data transmission
- **Raw Data**: Byte buffers, typed arrays, binary protocols
- **Events**: Input handling, system notifications, asynchronous signals

The system is built on three pillars:

1. **HoTT Type Foundations**: Media types are organized in a universe hierarchy with dependent types for dimensions and formats

2. **Linear Resource Management**: Hardware resources (windows, sockets, audio devices) use linear types ensuring exactly-once usage

3. **Proof Erasure**: All type-level information (dimensions, proofs, linearity) exists only at compile time—runtime code has zero overhead

---

## 2. Design Principles

### 2.1 Type Safety Without Runtime Cost

Every operation is statically verified:
- Buffer bounds are checked at compile time via dependent types
- Resource lifecycles are tracked via linear types
- Format compatibility is verified via type indices

At runtime, none of this information exists—only efficient machine code remains.

### 2.2 Explicit Resource Management

Resources are never implicitly acquired or released:
- Acquisition is explicit: `(window-create ...)`
- Release is explicit: `(window-destroy handle)`
- The type system ensures release happens exactly once

### 2.3 Composable Abstractions

Low-level primitives compose into high-level abstractions:
- Primitives implemented in C for platform portability
- Composition patterns implemented in Eshkol
- No "magic"—all abstractions reduce to explicit operations

### 2.4 Platform Independence

The type system is platform-independent:
- Same types across all platforms
- Platform-specific code isolated in C layer
- Eshkol code is write-once, run-anywhere

---

## 3. Type Universe Structure

Media types integrate into Eshkol's HoTT universe hierarchy:

```
𝒰₀ : Base Kinds and Values
────────────────────────────────────────────────────────────
ℕ                   Natural numbers (dimensions, sizes, indices)
Bool                Boolean values
Int8, Int16, ...    Fixed-width integers
Float32, Float64    Floating-point numbers

ResourceKind        Enumeration of resource types
  = Window
  | SocketTCP
  | SocketUDP
  | AudioOutput
  | AudioInput
  | FileHandle

ElementKind         Enumeration of buffer element types
  = Byte
  | Int16
  | Int32
  | Float32
  | Float64
  | PixelRGBA8
  | PixelRGB8
  | PixelGray8
  | PixelFloat32
  | SampleInt16
  | SampleFloat32

EventKind           Enumeration of event types
  = KeyEvent
  | MouseEvent
  | WindowEvent
  | NetworkEvent
  | CustomEvent


𝒰₁ : Type Constructors (Parameterized Types)
────────────────────────────────────────────────────────────
Handle : ResourceKind → 𝒰₁
    Managed resource handle indexed by resource kind

Buffer : ElementKind → ℕ → 𝒰₁
    Fixed-size buffer indexed by element type and length

Stream : ElementKind → 𝒰₁
    Potentially infinite sequence of elements

Event : EventKind → 𝒰₁
    Event data indexed by event kind

Linear : 𝒰₁ → 𝒰₁
    Linearity modality (must use exactly once)

Unrestricted : 𝒰₁ → 𝒰₁  (also written !)
    Unrestricted modality (can copy/drop freely)

Borrow : 𝒰₁ → 𝒰₁
    Borrowed reference (temporary access without ownership)


𝒰₂ : Type-Level Propositions
────────────────────────────────────────────────────────────
_<_ : ℕ → ℕ → 𝒰₂
    Less-than relation for bounds proofs

_≤_ : ℕ → ℕ → 𝒰₂
    Less-than-or-equal relation

_+_≤_ : ℕ → ℕ → ℕ → 𝒰₂
    Combined addition and comparison

Compatible : ElementKind → ElementKind → 𝒰₂
    Element type compatibility for operations

IsOpen : Handle k → 𝒰₂
    Proposition that a handle is in open state
```

---

## 4. Core Type Families

### 4.1 Handle Types

Handles represent managed resources with explicit lifecycles:

```
Handle : ResourceKind → 𝒰₁
```

**Inhabitants by Kind:**

| Kind | Type | Represents |
|------|------|------------|
| `Window` | `Handle Window` | GUI window with drawing surface |
| `SocketTCP` | `Handle SocketTCP` | TCP network connection |
| `SocketUDP` | `Handle SocketUDP` | UDP socket |
| `AudioOutput` | `Handle AudioOutput` | Audio playback device |
| `AudioInput` | `Handle AudioInput` | Audio recording device |
| `FileHandle` | `Handle FileHandle` | Open file descriptor |

**Key Property:** Handles are always linear by default:

```scheme
;; Acquisition returns a linear handle
(: window-create (-> String Int Int (IO (Linear (Handle Window)))))

;; Release consumes the linear handle
(: window-destroy (-> (Linear (Handle Window)) (IO Unit)))
```

### 4.2 Buffer Types

Buffers are contiguous memory regions with known element types and sizes:

```
Buffer : ElementKind → ℕ → 𝒰₁
```

**Interpretation:** `Buffer e n` is a buffer containing `n` elements of kind `e`.

**Examples:**

```scheme
;; 1024 raw bytes
(: raw-data (Buffer Byte 1024))

;; 800×600 RGBA pixels (stored linearly)
(: framebuffer (Buffer PixelRGBA8 480000))

;; 44100 stereo audio samples
(: audio-buffer (Buffer SampleFloat32 88200))

;; 256 32-bit integers
(: int-array (Buffer Int32 256))
```

**Dimension Aliases for Clarity:**

```scheme
;; 2D pixel buffer with named dimensions
(define-type (PixelBuffer2D w h fmt)
  (Buffer (Pixel fmt) (* w h)))

;; Audio buffer with sample rate and channels
(define-type (AudioBuffer rate channels samples fmt)
  (Buffer (Sample fmt) (* channels samples)))
```

### 4.3 Stream Types

Streams represent potentially infinite sequences:

```
Stream : ElementKind → 𝒰₁
```

**Interpretation:** `Stream e` is a lazy, potentially infinite sequence of elements of kind `e`.

**Key Operations:**

```scheme
;; Consume next element (if available)
(: stream-next (-> (Stream e) (IO (Maybe (e × (Stream e))))))

;; Transform elements
(: stream-map (-> (-> a b) (Stream a) (Stream b)))

;; Combine streams
(: stream-zip (-> (Stream a) (Stream b) (Stream (a × b))))

;; Take finite prefix into buffer
(: stream-take (-> (n : ℕ) (Stream e) (IO (Buffer e n × Stream e))))

;; Fold over stream (for finite streams or with early termination)
(: stream-fold-until (-> (-> acc e (Either acc result)) acc (Stream e) (IO result)))
```

**Stream Sources:**

```scheme
;; Audio input as stream
(: audio-input-stream (-> (Handle AudioInput) (Stream SampleFloat32)))

;; Network data as byte stream
(: socket-stream (-> (Handle SocketTCP) (Stream Byte)))

;; File contents as byte stream
(: file-stream (-> (Handle FileHandle) (Stream Byte)))
```

### 4.4 Event Types

Events are discriminated unions indexed by event kind:

```
Event : EventKind → 𝒰₁
```

**Event Data Structures:**

```scheme
;; Keyboard event
(record KeyEventData
  (key-code    : Int)
  (scan-code   : Int)
  (modifiers   : Int)
  (pressed     : Bool)
  (repeat      : Bool))

;; Mouse motion event
(record MouseMoveData
  (x           : Int)
  (y           : Int)
  (dx          : Int)
  (dy          : Int))

;; Mouse button event
(record MouseButtonData
  (x           : Int)
  (y           : Int)
  (button      : Int)
  (pressed     : Bool))

;; Window event
(record WindowEventData
  (event-type  : WindowEventType)  ;; Resize, Close, Focus, etc.
  (width       : Int)
  (height      : Int))

;; Network event
(record NetworkEventData
  (socket      : Handle SocketTCP)
  (event-type  : NetworkEventType)  ;; Data, Closed, Error
  (bytes-ready : Int))
```

**Event Handling:**

```scheme
;; Poll for event (non-blocking)
(: poll-event (-> (Handle Window) (IO (Maybe AnyEvent))))

;; Wait for event (blocking)
(: wait-event (-> (Handle Window) (IO AnyEvent)))

;; Pattern match on event type
(define (handle-event event)
  (match event
    ((KeyEvent data)
     (printf "Key: %d %s\n"
             (key-event-code data)
             (if (key-event-pressed data) "pressed" "released")))
    ((MouseMove data)
     (printf "Mouse: %d, %d\n"
             (mouse-move-x data)
             (mouse-move-y data)))
    ((WindowEvent data)
     (when (eq? (window-event-type data) 'close)
       (request-quit)))
    (_ (pure unit))))
```

---

## 5. Linear Resource Management

### 5.1 The Linear Modality

Linear types ensure values are used **exactly once**—no more, no less.

```
Linear : 𝒰₁ → 𝒰₁
```

**Typing Rules:**

```
                  Γ, x : Linear A ⊢ e : B    (x used exactly once in e)
Introduction:     ─────────────────────────────────────────────────────
                  Γ ⊢ e : B


                  Γ ⊢ e₁ : Linear A    Δ, x : A ⊢ e₂ : B    (x used exactly once)
Elimination:      ──────────────────────────────────────────────────────────────────
                  Γ, Δ ⊢ let-linear x = e₁ in e₂ : B
```

**What "exactly once" means:**

- The value must be passed to a function that consumes it, OR
- The value must be destructured (for products), OR
- The value must be pattern matched (for sums)

**What is NOT allowed:**

```scheme
;; ERROR: Linear value not consumed (weakening)
(define (leak-handle)
  (let ((h (window-create "Test" 800 600)))
    42))  ;; h is never used!

;; ERROR: Linear value used twice (contraction)
(define (double-use h)
  (window-present h)
  (window-present h))  ;; h already consumed!

;; ERROR: Linear value copied
(define (copy-handle h)
  (let ((h2 h))  ;; This would be copying
    (list h h2)))
```

### 5.2 Resource Lifecycle

Every managed resource follows a lifecycle:

```
    ┌─────────────┐
    │  Acquired   │ ← window-create, socket-connect, etc.
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │    Open     │ ← Can be borrowed for operations
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │  Released   │ ← window-destroy, socket-close, etc.
    └─────────────┘
```

**Acquisition produces linear handle:**

```scheme
(: window-create (-> String Int Int (IO (Linear (Handle Window)))))
(: socket-connect (-> String Int (IO (Linear (Handle SocketTCP)))))
(: audio-open-output (-> AudioConfig (IO (Linear (Handle AudioOutput)))))
```

**Release consumes linear handle:**

```scheme
(: window-destroy (-> (Linear (Handle Window)) (IO Unit)))
(: socket-close (-> (Linear (Handle SocketTCP)) (IO Unit)))
(: audio-close (-> (Linear (Handle AudioOutput)) (IO Unit)))
```

**The type system guarantees:**
1. Every acquired resource is eventually released (no leaks)
2. No resource is released twice (no double-free)
3. No resource is used after release (no use-after-free)

### 5.3 Borrowing and Lending

For operations that don't consume the resource, we use **borrowing**:

```
Borrow : 𝒰₁ → 𝒰₁
```

**Borrowing Rule:**

```scheme
;; Borrow a linear value for temporary use
(: borrow (-> (Linear a) (-> (Borrow a) b) (b × Linear a)))

;; Usage:
(let ((handle (window-create "Test" 800 600)))
  (let-values (((result handle') (borrow handle
                                   (lambda (h)
                                     (window-get-size h)))))
    ;; handle' is the same handle, still linear
    ;; result is the size
    (window-destroy handle')
    result))
```

**Simplified Syntax:**

```scheme
;; with-borrow macro for cleaner code
(with-borrow handle h
  (window-set-title h "New Title")
  (window-get-size h))
;; handle still available after with-borrow
```

**Operations on Borrowed References:**

```scheme
;; These operations take borrowed references, not ownership
(: window-get-size (-> (Borrow (Handle Window)) (IO (Int × Int))))
(: window-set-title (-> (Borrow (Handle Window)) String (IO Unit)))
(: window-get-surface (-> (Borrow (Handle Window)) (IO Surface)))
(: window-present (-> (Borrow (Handle Window)) (IO Unit)))

;; Read from socket (doesn't close it)
(: socket-recv (-> (Borrow (Handle SocketTCP)) Int (IO (Buffer Byte n))))

;; Write to socket (doesn't close it)
(: socket-send (-> (Borrow (Handle SocketTCP)) (Buffer Byte n) (IO Int)))
```

### 5.4 Escape Hatches

For cases where linearity is too restrictive, explicit escape hatches exist:

#### 5.4.1 Explicit Discard

```scheme
;; Discard consumes the linear value without meaningful action
(: discard (-> (Linear a) (IO Unit)))

;; Usage (should be rare and justified):
(let ((handle (window-create "Test" 800 600)))
  (if (initialization-failed?)
      (begin
        (log-error "Init failed, discarding window")
        (discard handle))  ;; Explicit acknowledgment of "leak"
      (run-app handle)))
```

#### 5.4.2 Bracket Pattern

```scheme
;; Bracket ensures cleanup even on exceptions
(: with-window (-> String Int Int
                   (-> (Borrow (Handle Window)) (IO a))
                   (IO a)))

(define (with-window title w h body)
  (let ((handle (window-create title w h)))
    (let ((result (try (borrow handle body)
                       (catch (e)
                         (window-destroy (snd (borrow handle (const unit))))
                         (raise e)))))
      (window-destroy (snd result))
      (fst result))))

;; Usage:
(with-window "My App" 800 600
  (lambda (win)
    (window-set-title win "Running...")
    (run-main-loop win)))
;; Window automatically destroyed
```

#### 5.4.3 Unrestricted Promotion

```scheme
;; Some values can be promoted to unrestricted
(: promote-if-safe (-> (Linear a) (Maybe (! a))))

;; Or explicitly marked unrestricted at creation
(: make-shared-buffer (-> Int (! (Buffer Byte n))))
;; Shared buffer can be copied, but requires explicit memory management
```

---

## 6. Dependent Dimensions

### 6.1 Static Dimension Tracking

Buffer dimensions are part of the type:

```scheme
;; Buffer with statically known size
(: make-buffer (-> (e : ElementKind) (n : ℕ) (Buffer e n)))

;; Example: create a 1024-byte buffer
(define buf (make-buffer Byte 1024))
;; buf : Buffer Byte 1024
```

**Dimension-Preserving Operations:**

```scheme
;; Map preserves dimensions
(: buffer-map (-> (-> a b) (Buffer a n) (Buffer b n)))

;; Zip requires matching dimensions
(: buffer-zip (-> (Buffer a n) (Buffer b n) (Buffer (a × b) n)))

;; Split produces two buffers with summed dimensions
(: buffer-split (-> (Buffer e (+ m n)) (Buffer e m × Buffer e n)))
```

**Dimension-Changing Operations:**

```scheme
;; Take prefix
(: buffer-take (-> (m : ℕ) (Buffer e n) (proof : m ≤ n) (Buffer e m)))

;; Drop prefix
(: buffer-drop (-> (m : ℕ) (Buffer e n) (proof : m ≤ n) (Buffer e (n - m))))

;; Concatenate
(: buffer-append (-> (Buffer e m) (Buffer e n) (Buffer e (+ m n))))

;; Reshape (total elements must match)
(: buffer-reshape (-> (Buffer e (* m n)) (m : ℕ) (n : ℕ) (Buffer2D e m n)))
```

### 6.2 Dimension Proofs

Operations requiring bounds produce proof obligations:

```scheme
;; Index access requires proof that index < length
(: buffer-ref (-> {n : ℕ} (Buffer e n) (i : ℕ) (proof : i < n) e))

;; Slice requires proof that range is valid
(: buffer-slice (-> {n : ℕ} (Buffer e n)
                    (start : ℕ) (len : ℕ)
                    (proof : start + len ≤ n)
                    (Buffer e len)))
```

**How Proofs Work:**

For compile-time constants, the compiler automatically verifies:

```scheme
(define buf (make-buffer Int32 100))

;; Valid: 50 < 100 is trivially true
(buffer-ref buf 50 refl)

;; Invalid: 150 < 100 is false
(buffer-ref buf 150 ???)  ;; COMPILE ERROR: Cannot construct proof
```

For computed indices, proofs must be constructed:

```scheme
(define (safe-access buf i)
  (if (< i (buffer-length buf))
      ;; The 'if' branch provides proof that i < n
      (Some (buffer-ref buf i (if-true-implies-< i (buffer-length buf))))
      None))
```

### 6.3 Dynamic Dimensions

For runtime-determined dimensions, use existential types:

```scheme
;; Existentially quantified buffer (dimension unknown statically)
(define-type DynBuffer e
  (exists (n : ℕ) (Buffer e n)))

;; Create from runtime value
(: make-dyn-buffer (-> (e : ElementKind) Int (DynBuffer e)))

;; Work with dynamic buffer via pattern matching
(define (process-dyn-buffer db)
  (match db
    ((pack n buf)
     ;; Inside here, we know buf : Buffer e n
     ;; n is abstract but consistent
     (buffer-map process-element buf))))
```

**Checked Operations for Dynamic Indices:**

```scheme
;; Returns Maybe instead of requiring proof
(: buffer-ref-checked (-> (Buffer e n) Int (Maybe e)))

;; Returns Either with error info
(: buffer-slice-checked (-> (Buffer e n) Int Int (Either Error (DynBuffer e))))
```

---

## 7. Runtime Representation

### 7.1 Tagged Value Extensions

The 16-byte tagged value structure extends for media types:

```c
typedef struct eshkol_tagged_value {
    uint8_t  type;      // Primary type tag
    uint8_t  subtype;   // Subtype/kind tag
    uint16_t flags;     // Additional flags
    uint32_t aux;       // Auxiliary data (small ints, etc.)
    uint64_t data;      // Main payload (pointer or value)
} eshkol_tagged_value_t;

// Type tags for media types
#define ESHKOL_TYPE_HANDLE   16
#define ESHKOL_TYPE_BUFFER   17
#define ESHKOL_TYPE_STREAM   18
#define ESHKOL_TYPE_EVENT    19

// Handle subtypes
#define ESHKOL_HANDLE_WINDOW      0
#define ESHKOL_HANDLE_SOCKET_TCP  1
#define ESHKOL_HANDLE_SOCKET_UDP  2
#define ESHKOL_HANDLE_AUDIO_OUT   3
#define ESHKOL_HANDLE_AUDIO_IN    4
#define ESHKOL_HANDLE_FILE        5

// Buffer subtypes (element kinds)
#define ESHKOL_BUFFER_BYTE          0
#define ESHKOL_BUFFER_INT16         1
#define ESHKOL_BUFFER_INT32         2
#define ESHKOL_BUFFER_FLOAT32       3
#define ESHKOL_BUFFER_FLOAT64       4
#define ESHKOL_BUFFER_PIXEL_RGBA8   5
#define ESHKOL_BUFFER_PIXEL_RGB8    6
#define ESHKOL_BUFFER_PIXEL_GRAY8   7
#define ESHKOL_BUFFER_SAMPLE_I16    8
#define ESHKOL_BUFFER_SAMPLE_F32    9
```

### 7.2 Buffer Memory Layout

```c
// Buffer header (precedes data in memory)
typedef struct eshkol_buffer_header {
    uint32_t length;        // Number of elements
    uint32_t capacity;      // Allocated capacity
    uint32_t element_size;  // Bytes per element
    uint32_t flags;         // Ownership, alignment flags
    // Optional extended header for 2D/3D:
    uint32_t dimensions[4]; // width, height, depth, channels
    uint32_t strides[4];    // Byte strides for each dimension
} eshkol_buffer_header_t;

// Buffer structure
typedef struct eshkol_buffer {
    eshkol_buffer_header_t* header;  // Points to header
    uint8_t* data;                    // Points to element data
} eshkol_buffer_t;

// Memory layout:
// [header (32-64 bytes)] [padding for alignment] [element data...]
//                                                 ^
//                                                 data pointer

// Alignment: data is always 64-byte aligned (cache line)
```

**Element Sizes:**

| Element Kind | Size (bytes) |
|--------------|--------------|
| Byte | 1 |
| Int16 | 2 |
| Int32 | 4 |
| Float32 | 4 |
| Float64 | 8 |
| PixelRGBA8 | 4 |
| PixelRGB8 | 3 (padded to 4) |
| PixelGray8 | 1 |
| SampleInt16 | 2 |
| SampleFloat32 | 4 |

### 7.3 Handle Structures

```c
// Generic handle wrapper
typedef struct eshkol_handle {
    void*    platform_handle;  // Platform-specific handle
    uint32_t handle_type;      // ResourceKind
    uint32_t state;            // Open, closed, error
    void     (*destructor)(void*);  // Cleanup function
} eshkol_handle_t;

// Window handle (extended)
typedef struct eshkol_window_handle {
    eshkol_handle_t base;
    uint32_t width;
    uint32_t height;
    uint32_t flags;           // Fullscreen, resizable, etc.
    eshkol_buffer_t* surface; // Drawing surface
} eshkol_window_handle_t;

// Socket handle (extended)
typedef struct eshkol_socket_handle {
    eshkol_handle_t base;
    int      fd;              // File descriptor
    uint32_t protocol;        // TCP or UDP
    uint32_t state;           // Connected, listening, etc.
} eshkol_socket_handle_t;

// Audio handle (extended)
typedef struct eshkol_audio_handle {
    eshkol_handle_t base;
    uint32_t sample_rate;
    uint16_t channels;
    uint16_t format;          // Sample format
    uint32_t buffer_frames;   // Frames per buffer
    void*    callback_data;   // For streaming
} eshkol_audio_handle_t;
```

### 7.4 Proof Erasure

**Compile Time vs Runtime:**

```
Source Code                          Compiled Code
─────────────────────────────────────────────────────────────────

(: buf (Buffer Int32 100))           void* buf;

(buffer-ref buf 50                   int32_t val =
  (proof-that-50<100))                 ((int32_t*)buf->data)[50];
                                     // No bounds check!

(buffer-slice buf 10 20              void* slice =
  (proof-that-10+20≤100))              buf->data + 10 * sizeof(int32_t);
                                     // Direct pointer arithmetic!

(Linear (Handle Window))             eshkol_window_handle_t*
                                     // No wrapper, no linearity tag!
```

**What Gets Erased:**

| Compile-Time Construct | Runtime Representation |
|------------------------|------------------------|
| `Buffer e n` | Pointer to buffer struct |
| `Linear a` | Same as `a` |
| `Borrow a` | Same as `a` |
| `(proof : P)` | Nothing (erased) |
| `{n : ℕ}` (implicit) | Nothing (erased) |
| Element kind | Subtype tag (if needed) |

**What Remains:**

| Construct | Runtime Representation |
|-----------|------------------------|
| Type tag | 1 byte in tagged value |
| Subtype | 1 byte in tagged value |
| Buffer length | In buffer header |
| Buffer data | Pointer to memory |
| Handle | Pointer to handle struct |

---

## 8. Primitive Operations

### 8.1 Window Operations

```scheme
;; Creation and destruction
(: window-create (-> String Int Int (IO (Linear (Handle Window)))))
(: window-destroy (-> (Linear (Handle Window)) (IO Unit)))

;; Properties (borrowed access)
(: window-get-size (-> (Borrow (Handle Window)) (IO (Int × Int))))
(: window-set-size (-> (Borrow (Handle Window)) Int Int (IO Unit)))
(: window-get-title (-> (Borrow (Handle Window)) (IO String)))
(: window-set-title (-> (Borrow (Handle Window)) String (IO Unit)))
(: window-get-position (-> (Borrow (Handle Window)) (IO (Int × Int))))
(: window-set-position (-> (Borrow (Handle Window)) Int Int (IO Unit)))

;; Surface access
(: window-get-surface (-> (Borrow (Handle Window)) (IO (Borrow Surface))))
(: window-present (-> (Borrow (Handle Window)) (IO Unit)))

;; State queries
(: window-should-close (-> (Borrow (Handle Window)) (IO Bool)))
(: window-is-focused (-> (Borrow (Handle Window)) (IO Bool)))
(: window-is-minimized (-> (Borrow (Handle Window)) (IO Bool)))

;; Event handling
(: window-poll-event (-> (Borrow (Handle Window)) (IO (Maybe Event))))
(: window-wait-event (-> (Borrow (Handle Window)) (IO Event)))
```

### 8.2 Graphics Operations

```scheme
;; Surface operations
(: surface-width (-> (Borrow Surface) Int))
(: surface-height (-> (Borrow Surface) Int))
(: surface-format (-> (Borrow Surface) PixelFormat))

;; Pixel access
(: surface-get-pixel (-> (Borrow Surface) Int Int (IO Color)))
(: surface-set-pixel (-> (Borrow Surface) Int Int Color (IO Unit)))

;; Drawing primitives
(: surface-clear (-> (Borrow Surface) Color (IO Unit)))
(: surface-fill-rect (-> (Borrow Surface) Int Int Int Int Color (IO Unit)))
(: surface-draw-line (-> (Borrow Surface) Int Int Int Int Color (IO Unit)))
(: surface-draw-rect (-> (Borrow Surface) Int Int Int Int Color (IO Unit)))
(: surface-draw-circle (-> (Borrow Surface) Int Int Int Color (IO Unit)))
(: surface-fill-circle (-> (Borrow Surface) Int Int Int Color (IO Unit)))

;; Blitting
(: surface-blit (-> (Borrow Surface)     ;; destination
                    Int Int               ;; dest x, y
                    (Borrow Surface)      ;; source
                    Int Int Int Int       ;; src x, y, w, h
                    (IO Unit)))

;; Buffer-based operations
(: buffer-to-surface (-> (Buffer PixelRGBA8 n) Int Int (Surface)))
(: surface-to-buffer (-> (Borrow Surface) (IO (Buffer PixelRGBA8 n))))
```

### 8.3 Audio Operations

```scheme
;; Device management
(: audio-open-output (-> AudioConfig (IO (Linear (Handle AudioOutput)))))
(: audio-open-input (-> AudioConfig (IO (Linear (Handle AudioInput)))))
(: audio-close (-> (Linear (Handle AudioOutput)) (IO Unit)))

;; Configuration
(record AudioConfig
  (sample-rate : Int)      ;; 44100, 48000, etc.
  (channels    : Int)      ;; 1 = mono, 2 = stereo
  (format      : SampleFormat)
  (buffer-size : Int))     ;; Frames per buffer

;; Playback control
(: audio-start (-> (Borrow (Handle AudioOutput)) (IO Unit)))
(: audio-stop (-> (Borrow (Handle AudioOutput)) (IO Unit)))
(: audio-pause (-> (Borrow (Handle AudioOutput)) (IO Unit)))
(: audio-resume (-> (Borrow (Handle AudioOutput)) (IO Unit)))

;; Queue-based output
(: audio-queue (-> (Borrow (Handle AudioOutput))
                   (Buffer SampleFloat32 n)
                   (IO Bool)))  ;; True if queued, false if full
(: audio-queued-frames (-> (Borrow (Handle AudioOutput)) (IO Int)))

;; Input (recording)
(: audio-read (-> (Borrow (Handle AudioInput))
                  (Buffer SampleFloat32 n)
                  (IO Int)))  ;; Frames actually read

;; Streaming
(: audio-output-stream (-> (Handle AudioOutput) (Stream SampleFloat32)))
(: audio-input-stream (-> (Handle AudioInput) (Stream SampleFloat32)))
```

### 8.4 Network Operations

```scheme
;; TCP operations
(: tcp-connect (-> String Int (IO (Linear (Handle SocketTCP)))))
(: tcp-listen (-> Int (IO (Linear (Handle SocketTCP)))))
(: tcp-accept (-> (Borrow (Handle SocketTCP))
                  (IO (Linear (Handle SocketTCP)))))
(: tcp-close (-> (Linear (Handle SocketTCP)) (IO Unit)))

;; UDP operations
(: udp-create (-> (IO (Linear (Handle SocketUDP)))))
(: udp-bind (-> (Borrow (Handle SocketUDP)) Int (IO Bool)))
(: udp-close (-> (Linear (Handle SocketUDP)) (IO Unit)))

;; Data transfer
(: socket-send (-> (Borrow (Handle SocketTCP))
                   (Buffer Byte n)
                   (IO Int)))  ;; Bytes sent

(: socket-recv (-> (Borrow (Handle SocketTCP))
                   Int         ;; Max bytes
                   (IO (DynBuffer Byte))))  ;; Received data

(: socket-sendto (-> (Borrow (Handle SocketUDP))
                     (Buffer Byte n)
                     String Int  ;; Host, port
                     (IO Int)))

(: socket-recvfrom (-> (Borrow (Handle SocketUDP))
                       Int
                       (IO (DynBuffer Byte × String × Int))))

;; Socket options
(: socket-set-blocking (-> (Borrow (Handle SocketTCP)) Bool (IO Unit)))
(: socket-set-timeout (-> (Borrow (Handle SocketTCP)) Int (IO Unit)))

;; Polling
(: socket-poll (-> (List (Handle SocketTCP)) Int (IO (List Bool))))
```

### 8.5 Buffer Operations

```scheme
;; Creation
(: make-buffer (-> (e : ElementKind) (n : ℕ) (Buffer e n)))
(: make-buffer-init (-> (e : ElementKind) (n : ℕ) e (Buffer e n)))

;; Access (with proofs)
(: buffer-ref (-> {n : ℕ} (Buffer e n) (i : ℕ) (i < n) e))
(: buffer-set! (-> {n : ℕ} (Buffer e n) (i : ℕ) (i < n) e (IO Unit)))

;; Access (checked, no proofs)
(: buffer-ref-checked (-> (Buffer e n) Int (Maybe e)))
(: buffer-set-checked! (-> (Buffer e n) Int e (IO Bool)))

;; Properties
(: buffer-length (-> (Buffer e n) ℕ))
(: buffer-element-size (-> (Buffer e n) Int))
(: buffer-byte-length (-> (Buffer e n) Int))

;; Transformations
(: buffer-map (-> (-> a b) (Buffer a n) (Buffer b n)))
(: buffer-map! (-> (-> a a) (Buffer a n) (IO Unit)))
(: buffer-fold (-> (-> acc a acc) acc (Buffer a n) acc))
(: buffer-zip (-> (Buffer a n) (Buffer b n) (Buffer (a × b) n)))
(: buffer-zip-with (-> (-> a b c) (Buffer a n) (Buffer b n) (Buffer c n)))

;; Slicing
(: buffer-take (-> (m : ℕ) (Buffer e n) (m ≤ n) (Buffer e m)))
(: buffer-drop (-> (m : ℕ) (Buffer e n) (m ≤ n) (Buffer e (- n m))))
(: buffer-slice (-> (start : ℕ) (len : ℕ) (Buffer e n)
                    ((+ start len) ≤ n) (Buffer e len)))

;; Combining
(: buffer-append (-> (Buffer e m) (Buffer e n) (Buffer e (+ m n))))
(: buffer-concat (-> (List (DynBuffer e)) (DynBuffer e)))

;; Copying
(: buffer-copy (-> (Buffer e n) (Buffer e n)))
(: buffer-copy-to! (-> (Buffer e n) (Buffer e n) (IO Unit)))
(: buffer-fill! (-> (Buffer e n) e (IO Unit)))
```

---

## 9. Implementation Architecture

### 9.1 Compiler Layer (C++/LLVM)

**Responsibilities:**
- Type checking with linearity and dimension verification
- Proof obligation generation and checking
- LLVM IR generation for media operations
- Proof erasure during code generation

**Key Components:**

```cpp
// Linearity checker
class LinearityChecker {
public:
    void enterScope();
    void exitScope();  // Verify all linear vars consumed
    void bindLinear(const std::string& name, const Type& type);
    void useLinear(const std::string& name);  // Error if already used
    bool isConsumed(const std::string& name) const;
};

// Dimension checker
class DimensionChecker {
public:
    // Attempt to prove inequality at compile time
    bool canProve(const Expr& lhs, Relation rel, const Expr& rhs);

    // Generate proof obligation for runtime
    ProofObligation requireProof(const Expr& lhs, Relation rel, const Expr& rhs);
};

// Media type codegen
class MediaCodegen {
public:
    // Generate buffer operations
    llvm::Value* codegenBufferCreate(ElementKind kind, uint64_t size);
    llvm::Value* codegenBufferRef(llvm::Value* buf, llvm::Value* index);
    llvm::Value* codegenBufferSet(llvm::Value* buf, llvm::Value* index, llvm::Value* val);

    // Generate handle operations
    llvm::Value* codegenHandleCreate(ResourceKind kind, llvm::ArrayRef<llvm::Value*> args);
    void codegenHandleDestroy(llvm::Value* handle);
};
```

**Files:**
- `lib/backend/linearity_checker.cpp`
- `lib/backend/dimension_checker.cpp`
- `lib/backend/media_codegen.cpp`

### 9.2 Platform Layer (C)

**Responsibilities:**
- Platform-specific implementations
- Memory management for buffers
- System call wrappers

**Directory Structure:**

```
lib/platform/
├── platform.h              # Common definitions
├── memory.c                # Buffer allocation (arena-based)
├── window/
│   ├── window.h           # Window interface
│   ├── window_x11.c       # X11 implementation
│   ├── window_wayland.c   # Wayland implementation
│   ├── window_win32.c     # Windows implementation
│   ├── window_cocoa.m     # macOS implementation
│   └── window_null.c      # Headless/testing
├── graphics/
│   ├── graphics.h         # Graphics interface
│   └── graphics_sw.c      # Software renderer
├── audio/
│   ├── audio.h            # Audio interface
│   ├── audio_alsa.c       # Linux ALSA
│   ├── audio_pulse.c      # PulseAudio
│   ├── audio_coreaudio.c  # macOS
│   └── audio_wasapi.c     # Windows
├── network/
│   ├── network.h          # Network interface
│   └── network_bsd.c      # BSD sockets (portable)
└── event/
    ├── event.h            # Event interface
    └── event.c            # Event queue
```

**Interface Example:**

```c
// lib/platform/window/window.h

#ifndef ESHKOL_WINDOW_H
#define ESHKOL_WINDOW_H

#include <stdint.h>
#include <stdbool.h>

typedef struct eshkol_window eshkol_window_t;
typedef struct eshkol_surface eshkol_surface_t;
typedef struct eshkol_event eshkol_event_t;

// Lifecycle
eshkol_window_t* eshkol_window_create(const char* title, int w, int h);
void eshkol_window_destroy(eshkol_window_t* window);

// Properties
void eshkol_window_get_size(eshkol_window_t* window, int* w, int* h);
void eshkol_window_set_size(eshkol_window_t* window, int w, int h);
void eshkol_window_set_title(eshkol_window_t* window, const char* title);
bool eshkol_window_should_close(eshkol_window_t* window);

// Surface
eshkol_surface_t* eshkol_window_get_surface(eshkol_window_t* window);
void eshkol_window_present(eshkol_window_t* window);

// Events
bool eshkol_window_poll_event(eshkol_window_t* window, eshkol_event_t* event);
void eshkol_window_wait_event(eshkol_window_t* window, eshkol_event_t* event);

#endif
```

### 9.3 Library Layer (Eshkol)

**Responsibilities:**
- High-level abstractions
- Convenience functions
- Algorithm implementations

**Files:**

```
lib/media/
├── media.esk           # Main module (re-exports)
├── window.esk          # Window utilities
├── graphics.esk        # Drawing algorithms
├── color.esk           # Color manipulation
├── audio.esk           # Audio synthesis/processing
├── network.esk         # Network utilities
├── buffer.esk          # Buffer utilities
├── event.esk           # Event handling patterns
└── stream.esk          # Stream combinators
```

**Example: Color Module**

```scheme
;; lib/media/color.esk

;; Color representation: 32-bit RGBA
(define-type Color Int32)

;; Construction
(define (rgba r g b a)
  (bitwise-ior
    (bitwise-and r #xFF)
    (arithmetic-shift (bitwise-and g #xFF) 8)
    (arithmetic-shift (bitwise-and b #xFF) 16)
    (arithmetic-shift (bitwise-and a #xFF) 24)))

(define (rgb r g b) (rgba r g b 255))

;; Extraction
(define (color-r c) (bitwise-and c #xFF))
(define (color-g c) (bitwise-and (arithmetic-shift c -8) #xFF))
(define (color-b c) (bitwise-and (arithmetic-shift c -16) #xFF))
(define (color-a c) (bitwise-and (arithmetic-shift c -24) #xFF))

;; Blending
(define (color-blend c1 c2 t)
  (rgba (lerp (color-r c1) (color-r c2) t)
        (lerp (color-g c1) (color-g c2) t)
        (lerp (color-b c1) (color-b c2) t)
        (lerp (color-a c1) (color-a c2) t)))

;; Predefined colors
(define color-black   (rgb 0 0 0))
(define color-white   (rgb 255 255 255))
(define color-red     (rgb 255 0 0))
(define color-green   (rgb 0 255 0))
(define color-blue    (rgb 0 0 255))
(define color-yellow  (rgb 255 255 0))
(define color-cyan    (rgb 0 255 255))
(define color-magenta (rgb 255 0 255))
```

---

## 10. Integration with Eshkol Core

### 10.1 Arena Memory

Media buffers are allocated from Eshkol's arena:

```scheme
;; Buffers allocated from current arena
(define buf (make-buffer Float32 1000))
;; buf lives as long as current arena scope

;; For persistent buffers, use explicit allocation
(define persistent-buf (make-persistent-buffer Float32 1000))
;; Must be explicitly freed: (free-buffer persistent-buf)
```

### 10.2 Tensor Integration

Buffers interoperate with Eshkol tensors:

```scheme
;; Convert buffer to tensor
(: buffer->tensor (-> (Buffer Float32 n) (Tensor Float [n])))

;; Convert tensor to buffer
(: tensor->buffer (-> (Tensor Float [n]) (Buffer Float32 n)))

;; Zero-copy view (when layout matches)
(: buffer-as-tensor (-> (Buffer Float32 (* m n)) (m : ℕ) (n : ℕ)
                        (Tensor Float [m n])))
```

### 10.3 Autodiff on Buffers

Buffer operations participate in automatic differentiation:

```scheme
;; Image loss function (differentiable)
(define (image-mse predicted target)
  (let ((diff (buffer-zip-with - predicted target)))
    (buffer-fold + 0.0 (buffer-map square diff))))

;; Compute gradient
(define grad (gradient image-mse predicted-image target-image))
```

### 10.4 FFI with C

Media types are FFI-compatible:

```scheme
;; Declare external C function
(extern void process_image uint8_t* int int)

;; Call with buffer
(with-buffer-ptr buf ptr
  (process_image ptr (buffer-length buf) 1))
```

---

## 11. Example Programs

### 11.1 Simple Window

```scheme
(define (main)
  (with-window "Hello Eshkol" 800 600
    (lambda (win)
      (let loop ()
        (let ((event (window-poll-event win)))
          (match event
            ((Some (WindowEvent (Close)))
             (pure unit))
            (_
             (let ((surface (window-get-surface win)))
               (surface-clear surface color-blue)
               (surface-fill-rect surface 100 100 200 150 color-red)
               (window-present win)
               (loop)))))))))
```

### 11.2 Audio Playback

```scheme
(define (main)
  (let ((config (audio-config 44100 2 'float32 1024)))
    (with-audio-output config
      (lambda (audio)
        (audio-start audio)
        (let ((samples (generate-sine-wave 440.0 44100 2.0)))
          (audio-queue audio samples)
          (sleep 2.0)
          (audio-stop audio))))))

(define (generate-sine-wave freq sample-rate duration)
  (let* ((num-samples (floor (* sample-rate duration)))
         (buf (make-buffer SampleFloat32 (* 2 num-samples))))
    (buffer-generate! buf
      (lambda (i)
        (let ((t (/ (quotient i 2) sample-rate))
              (sample (sin (* 2.0 pi freq t))))
          sample)))
    buf))
```

### 11.3 TCP Echo Server

```scheme
(define (main)
  (with-tcp-listener 8080
    (lambda (listener)
      (printf "Listening on port 8080\n")
      (let accept-loop ()
        (let ((client (tcp-accept listener)))
          (spawn (lambda () (handle-client client)))
          (accept-loop))))))

(define (handle-client client)
  (let loop ()
    (let ((data (socket-recv client 1024)))
      (when (> (buffer-length data) 0)
        (socket-send client data)
        (loop))))
  (socket-close client))
```

### 11.4 Image Processing Pipeline

```scheme
(define (grayscale-pipeline input-file output-file)
  (let ((img (load-image input-file)))
    (let ((gray (buffer-map rgb-to-gray img)))
      (let ((blurred (gaussian-blur gray 3)))
        (let ((edges (sobel-edge-detect blurred)))
          (save-image output-file edges))))))

(define (rgb-to-gray pixel)
  (let ((r (pixel-r pixel))
        (g (pixel-g pixel))
        (b (pixel-b pixel)))
    (floor (+ (* 0.299 r) (* 0.587 g) (* 0.114 b)))))

(define (gaussian-blur buf radius)
  (let ((kernel (make-gaussian-kernel radius)))
    (convolve-2d buf kernel)))

(define (sobel-edge-detect buf)
  (let ((gx (convolve-2d buf sobel-x-kernel))
        (gy (convolve-2d buf sobel-y-kernel)))
    (buffer-zip-with
      (lambda (x y) (sqrt (+ (* x x) (* y y))))
      gx gy)))
```

---

## Appendix A: Type Signature Reference

```scheme
;; === Handles ===
(: window-create     (-> String Int Int (IO (Linear (Handle Window)))))
(: window-destroy    (-> (Linear (Handle Window)) (IO Unit)))
(: tcp-connect       (-> String Int (IO (Linear (Handle SocketTCP)))))
(: tcp-close         (-> (Linear (Handle SocketTCP)) (IO Unit)))
(: audio-open-output (-> AudioConfig (IO (Linear (Handle AudioOutput)))))
(: audio-close       (-> (Linear (Handle AudioOutput)) (IO Unit)))

;; === Buffers ===
(: make-buffer       (-> (e : ElementKind) (n : ℕ) (Buffer e n)))
(: buffer-ref        (-> {n : ℕ} (Buffer e n) (i : ℕ) (i < n) e))
(: buffer-set!       (-> {n : ℕ} (Buffer e n) (i : ℕ) (i < n) e (IO Unit)))
(: buffer-length     (-> (Buffer e n) ℕ))
(: buffer-map        (-> (-> a b) (Buffer a n) (Buffer b n)))
(: buffer-fold       (-> (-> acc a acc) acc (Buffer a n) acc))
(: buffer-slice      (-> (s : ℕ) (l : ℕ) (Buffer e n) ((+ s l) ≤ n) (Buffer e l)))
(: buffer-append     (-> (Buffer e m) (Buffer e n) (Buffer e (+ m n))))

;; === Streams ===
(: stream-next       (-> (Stream e) (IO (Maybe (e × Stream e)))))
(: stream-map        (-> (-> a b) (Stream a) (Stream b)))
(: stream-take       (-> (n : ℕ) (Stream e) (IO (Buffer e n × Stream e))))

;; === Linear Utilities ===
(: borrow            (-> (Linear a) (-> (Borrow a) b) (b × Linear a)))
(: discard           (-> (Linear a) (IO Unit)))
(: with-resource     (-> (IO (Linear a)) (-> (Linear a) (IO Unit))
                         (-> (Borrow a) (IO b)) (IO b)))
```

---

*End of Specification*
