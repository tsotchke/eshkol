# Eshkol Web Platform

**Status**: Stable — v1.1-accelerate
**Module**: `web` / `web.http`
**Source**: `lib/web/http.esk`
**Compiler flag**: `--wasm` (`-w`)

---

## 1. Overview

Eshkol's web platform enables compiling Eshkol programs to WebAssembly for
direct execution in browsers or WASM runtimes. The library provides 73
functions covering DOM manipulation, events, canvas drawing, timers, fetch,
local storage, and browser window/location APIs — all accessible from
Eshkol's functional S-expression syntax.

The design prioritizes simplicity at the language boundary: every browser
object is represented as a 32-bit integer handle, every call crosses the
WASM/JS boundary via host-provided C functions, and there is no hidden
garbage collection or object graph on the Eshkol side. Programs remain
purely functional at the macro level, with mutation performed explicitly
through the imperative web API.

---

## 2. Architecture

### 2.1 WASM Compilation Pipeline

The `--wasm` (`-w`) flag routes compilation through a dedicated code path in
`eshkol-run.cpp` (`exe/eshkol-run.cpp:2709`):

```
Eshkol source (.esk)
        |
        v
   LLVM IR (via llvm_codegen.cpp)
        |
        v
   LLVM optimization passes
        |
        v
   WASM target machine (wasm32-unknown-unknown)
        |
        v
   .wasm binary
```

Internally, `eshkol_compile_llvm_ir_to_wasm` (`llvm_codegen.cpp:34796`)
initializes the WebAssembly LLVM backend, sets the target triple to
`wasm32-unknown-unknown`, applies standard optimization passes, and emits
a binary WASM module via `PassManager::run`. The output is a self-contained
`.wasm` file ready for loading in a browser or WASM host.

WASM target availability is guarded by a compile-time flag:

```cpp
// System LLVM 21 builds include WebAssembly; XLA/StableHLO builds may not.
#define ESHKOL_HAS_WASM_TARGET 1   // default system build
#define ESHKOL_HAS_WASM_TARGET 0   // ESHKOL_XLA_FULL_MLIR builds
```

If the target is unavailable, `eshkol-run` emits a clear error at compile
time. The `wasm32` relocation model is `PIC_` — position-independent — as
required by the WASM memory model.

### 2.2 FFI via `(extern ...)`

Every web function is declared with the `extern` form, which instructs the
Eshkol compiler to emit a WASM import rather than a native call:

```scheme
(extern i32 web-create-element ptr :real web_create_element)
```

The `:real` keyword provides the C symbol name that the WASM runtime must
supply as a host import. The browser-side JavaScript glue layer (or any
compliant WASM host) must export these symbols into the WASM module's import
namespace before instantiation. All 73 functions follow this pattern.

Supported FFI types in the web API:

| FFI type | C/WASM type | Description |
|----------|-------------|-------------|
| `i32`    | `int32_t`   | Integer, boolean, or handle |
| `ptr`    | `const char*` | Null-terminated string from WASM linear memory |
| `double` | `f64`       | IEEE 754 double-precision float |
| `void`   | `void`      | No return value |

### 2.3 The Integer Handle System

JavaScript objects — DOM elements, events, canvas contexts, timers — cannot
be passed directly across the WASM/JS boundary as arbitrary pointers. The
web platform solves this with a **handle table**: a JavaScript-side map from
integer IDs to live JS objects. Eshkol code holds only the integer handle;
all JS-side operations are dispatched through the host import functions.

```
Eshkol (WASM)              Host (JavaScript)
─────────────────          ──────────────────────────────────────
  (define btn 42)  <->     handleTable[42] = document.createElement("div")
  (web-append-child        web_append_child(3, 42)
    (web-get-body)         -> handleTable[3].appendChild(handleTable[42])
    btn)
```

Three handles are permanently reserved and never require dynamic allocation:

| Constant handle | Object | Eshkol accessor |
|-----------------|--------|-----------------|
| `1` | `document` | `(web-get-document)` |
| `2` | `window`   | `(web-get-window)` |
| `3` | `document.body` | `(web-get-body)` |

All other handles are allocated by the host on creation (e.g.,
`web-create-element`, `web-get-context-2d`) and must be explicitly released
with `web-release-handle` when the Eshkol program is done with them.
Failing to call `web-release-handle` causes the handle table to grow
indefinitely, leaking memory on the JavaScript side.

---

## 3. Getting Started

### 3.1 Compiling to WASM

```bash
# Compile an Eshkol web program to a .wasm binary
eshkol-run myapp.esk --wasm -o myapp

# Output: myapp.wasm
# (The compiler appends .wasm if the extension is absent)
```

### 3.2 Minimal Example

```scheme
;;; hello-web.esk
(require web)

;; Create a heading element
(define h1 (web-create-element "h1"))
(web-set-text-content h1 "Hello from Eshkol!")

;; Style it
(web-add-class h1 "hero-title")
(web-set-style h1 "color" "#3B82F6")

;; Insert into DOM
(web-append-child (web-get-body) h1)
```

```bash
eshkol-run hello-web.esk --wasm -o hello-web
# Produces: hello-web.wasm
```

Loading in HTML:

```html
<script type="module">
  const imports = buildEshkolWebImports(); // host glue
  const { instance } = await WebAssembly.instantiateStreaming(
    fetch("hello-web.wasm"), imports
  );
  instance.exports._eshkol_main();
</script>
```

### 3.3 Module Import

```scheme
;; Import the entire web module
(require web)

;; Or import the specific sub-module
(require web.http)
```

---

## 4. Handle Lifecycle

### 4.1 Acquiring Handles

Handles are returned by creation and query functions:

```scheme
(define div   (web-create-element "div"))        ; new handle
(define txt   (web-create-text-node "foo"))      ; new handle
(define btn   (web-get-element-by-id "submit"))  ; new handle, or 0 if not found
(define ctx   (web-get-context-2d canvas))       ; new handle
```

Always check for `0` before using a handle returned by a query function.
A zero handle indicates the object was not found and must not be passed to
other web functions.

```scheme
(define el (web-get-element-by-id "nonexistent"))
(when (not (= el 0))
  (web-set-text-content el "found it"))
```

### 4.2 Releasing Handles

```scheme
;; Release a single handle
(web-release-handle div)

;; Pattern: create, use, release
(let ((tmp (web-create-element "span")))
  (web-set-text-content tmp "temporary")
  (web-append-child container tmp)
  (web-release-handle tmp))  ; release after appending
```

Note: releasing a handle does not remove the DOM node. The DOM tree holds its
own reference to the underlying JS object. `web-release-handle` only frees
the entry in the host handle table that Eshkol was using to reference it.

### 4.3 Permanent Handles

The document (1), window (2), and body (3) handles are permanent. Calling
`web-release-handle` on them is a no-op in a correct host implementation.

---

## 5. Complete API Reference

### 5.1 Special Handles (3 functions)

```scheme
(web-get-document) -> i32   ; Always handle 1 (document)
(web-get-window)   -> i32   ; Always handle 2 (window)
(web-get-body)     -> i32   ; Always handle 3 (document.body)
```

These functions involve no dynamic allocation and return fixed constants.
They exist for consistency: code that receives a node handle as a parameter
can call these functions to obtain root references without requiring ambient
global state.

---

### 5.2 Document Methods (5 functions)

```scheme
(web-create-element    tag-string) -> i32
(web-create-text-node  text-string) -> i32
(web-get-element-by-id id-string)  -> i32   ; 0 if not found
(web-query-selector    css-selector) -> i32  ; 0 if no match
(web-query-selector-all css-selector) -> i32 ; NodeList handle
```

`web-create-element` and `web-create-text-node` always return a valid handle
(assuming host memory is available). `web-get-element-by-id` and
`web-query-selector` return `0` when the query matches nothing.
`web-query-selector-all` returns a handle to a NodeList object; traverse it
using `web-get-children-count` and `web-get-child-at`.

Example:

```scheme
(define section (web-create-element "section"))
(web-set-attribute section "id" "main-content")

(define existing (web-get-element-by-id "header"))
(when (not (= existing 0))
  (web-append-child existing section))
```

---

### 5.3 Node Tree Manipulation (12 functions)

```scheme
(web-append-child   parent child)          -> i32  ; 1 ok, 0 fail
(web-remove-child   parent child)          -> i32
(web-insert-before  parent new-node ref)   -> i32
(web-replace-child  parent new-node old)   -> i32
(web-clone-node     node deep)             -> i32  ; deep=1 for deep clone
(web-get-parent     node)                  -> i32  ; 0 if no parent
(web-get-first-child node)                 -> i32  ; 0 if no children
(web-get-last-child  node)                 -> i32
(web-get-next-sibling node)                -> i32  ; 0 if none
(web-get-prev-sibling node)                -> i32
(web-get-children-count node)              -> i32
(web-get-child-at   node index)            -> i32  ; 0 if out of range
```

`web-clone-node` with `deep=1` performs a recursive deep clone of the subtree;
`deep=0` clones only the element itself without children. The returned handle
references a detached node not yet inserted into any document.

Traversal example:

```scheme
;; Iterate over all children of a container
(define n (web-get-children-count container))
(let loop ((i 0))
  (when (< i n)
    (let ((child (web-get-child-at container i)))
      (web-set-style child "opacity" "0.5")
      (loop (+ i 1)))))
```

---

### 5.4 Attributes (4 functions)

```scheme
(web-set-attribute    element name value) -> i32
(web-get-attribute    element name buf buflen) -> i32  ; length written
(web-remove-attribute element name)       -> i32
(web-has-attribute    element name)       -> i32  ; 1=yes, 0=no
```

`web-get-attribute` writes the attribute value as a null-terminated string
into the caller-provided buffer `buf` of capacity `buflen`. It returns the
number of bytes written (excluding the null terminator), or `-1` if the
attribute does not exist or the buffer is too small.

```scheme
(web-set-attribute link "href" "https://example.com")
(web-set-attribute link "target" "_blank")
(web-set-attribute img "src" "/images/photo.jpg")
(web-set-attribute img "alt" "A photograph")

;; Remove disabled attribute from a button
(web-remove-attribute submit-btn "disabled")
```

---

### 5.5 Content (4 functions)

```scheme
(web-set-inner-html   element html-string) -> i32
(web-get-inner-html   element buf buflen)  -> i32
(web-set-text-content element text-string) -> i32
(web-get-text-content element buf buflen)  -> i32
```

Prefer `web-set-text-content` over `web-set-inner-html` whenever the content
is plain text — it avoids XSS injection and is faster. Use `web-set-inner-html`
only when constructing markup programmatically from trusted data.

```scheme
;; Safe: text content
(web-set-text-content label (format "Score: ~a" score))

;; Markup: structured HTML
(web-set-inner-html card
  "<div class='title'>Report</div><div class='body'>...</div>")
```

---

### 5.6 CSS Classes (4 functions)

```scheme
(web-add-class    element class-name) -> i32
(web-remove-class element class-name) -> i32
(web-toggle-class element class-name) -> i32  ; 1=added, 0=removed
(web-has-class    element class-name) -> i32  ; 1=yes, 0=no
```

```scheme
;; Toggle active state
(define active (web-toggle-class tab "active"))
(if (= active 1)
  (web-console-log "tab is now active")
  (web-console-log "tab is now inactive"))

;; Conditional class
(when (> error-count 0)
  (web-add-class status-bar "error")
  (web-remove-class status-bar "success"))
```

---

### 5.7 Inline Styles (2 functions)

```scheme
(web-set-style element property value)         -> i32
(web-get-style element property buf buflen)    -> i32
```

Property names use camelCase as in the DOM `style` object:

```scheme
(web-set-style panel "backgroundColor" "#1E293B")
(web-set-style panel "borderRadius" "8px")
(web-set-style panel "display" "flex")
(web-set-style panel "padding" "16px 24px")
```

---

### 5.8 Form Elements (6 functions)

```scheme
(web-get-value   element buf buflen)  -> i32  ; length written
(web-set-value   element value-string) -> i32
(web-get-checked element)             -> i32  ; 1=checked, 0=unchecked
(web-set-checked element state)       -> i32  ; state: 1 or 0
(web-focus       element)             -> i32
(web-blur        element)             -> i32
```

```scheme
;; Read text input
(define input (web-get-element-by-id "search"))
;; In practice: allocate a buffer, call web-get-value into it

;; Set a select value
(web-set-value (web-get-element-by-id "color-picker") "blue")

;; Check/uncheck a box
(web-set-checked (web-get-element-by-id "agree") 1)
(define is-checked (web-get-checked (web-get-element-by-id "agree")))

;; Focus the first field
(web-focus (web-get-element-by-id "username"))
```

---

### 5.9 Events (9 functions)

```scheme
(web-add-event-listener      element event-name callback-handle) -> i32  ; callback-id
(web-remove-event-listener   callback-id)                        -> i32

(web-event-prevent-default   event-handle) -> i32
(web-event-stop-propagation  event-handle) -> i32
(web-event-get-target        event-handle) -> i32  ; element handle

(web-event-get-key           event-handle buf buflen) -> i32  ; key name
(web-event-get-key-code      event-handle)            -> i32  ; numeric keyCode
(web-event-get-mouse-x       event-handle)            -> i32  ; clientX
(web-event-get-mouse-y       event-handle)            -> i32  ; clientY
```

`web-add-event-listener` takes an integer `callback-handle`, which in the
WASM context is a function table index. The host glue layer dispatches the
callback by invoking the referenced WASM function with the event handle as
its sole argument. The return value is a `callback-id` that can be passed to
`web-remove-event-listener` to deregister the listener.

See Section 7 for the full event programming model and examples.

---

### 5.10 Timers (6 functions)

```scheme
(web-set-timeout           callback-handle delay-ms) -> i32  ; timer-id
(web-set-interval          callback-handle delay-ms) -> i32  ; timer-id
(web-clear-timeout         timer-id)                 -> void
(web-clear-interval        timer-id)                 -> void
(web-request-animation-frame callback-handle)        -> i32  ; request-id
(web-cancel-animation-frame  request-id)             -> void
```

Timers accept a callback handle (WASM function table index) and a delay in
milliseconds. `web-request-animation-frame` schedules the callback before the
next browser repaint; it is the correct mechanism for animations (see
Section 6.2 for a canvas animation example).

```scheme
;; One-shot delay
(define timer-id (web-set-timeout my-callback 2000))

;; Repeating interval — store ID to cancel later
(define interval-id (web-set-interval poll-callback 500))

;; Cancel when done
(web-clear-interval interval-id)
```

---

### 5.11 Console (3 functions)

```scheme
(web-console-log   message-string) -> void
(web-console-warn  message-string) -> void
(web-console-error message-string) -> void
```

These write to the browser developer console. They accept only string
arguments (the WASM linear memory pointer). Use Eshkol's `format` or
`number->string` to construct the string before logging:

```scheme
(web-console-log "Eshkol web platform initialised")
(web-console-warn "Deprecated API called")
(web-console-error "Fatal: DOM node not found")
```

---

### 5.12 Window (8 functions)

```scheme
(web-alert             message-string) -> void
(web-confirm           message-string) -> i32   ; 1=OK, 0=Cancel
(web-prompt            message default-string buf buflen) -> i32  ; length

(web-get-window-width)  -> i32
(web-get-window-height) -> i32

(web-get-scroll-x)      -> i32
(web-get-scroll-y)      -> i32
(web-scroll-to          x y) -> void
```

`web-confirm` blocks execution and returns `1` if the user clicks OK, `0` for
Cancel. `web-prompt` writes the user's input into the provided buffer and
returns its length. The `default-string` is the pre-filled text.

```scheme
;; Confirm before destructive action
(when (= (web-confirm "Delete all data?") 1)
  (web-storage-clear)
  (web-set-inner-html (web-get-body) ""))

;; Scroll to top of page
(web-scroll-to 0 0)

;; Responsive layout check
(define width  (web-get-window-width))
(define height (web-get-window-height))
(when (< width 768)
  (web-add-class app-root "mobile"))
```

---

### 5.13 Location (4 functions)

```scheme
(web-get-href  buf buflen) -> i32   ; writes current URL
(web-set-href  url-string) -> void  ; navigate
(web-get-hash  buf buflen) -> i32   ; writes hash fragment
(web-set-hash  hash-string) -> void ; set hash without navigation
```

`web-set-href` triggers a full navigation; `web-set-hash` updates only the
fragment portion of the URL without reloading the page, making it suitable
for single-page application routing.

```scheme
;; Client-side routing by hash
(define (navigate-to route)
  (web-set-hash route))

(navigate-to "settings")   ; URL becomes #settings
(navigate-to "dashboard")  ; URL becomes #dashboard
```

---

### 5.14 Local Storage (4 functions)

```scheme
(web-storage-get    key buf buflen) -> i32  ; length of value, -1 if missing
(web-storage-set    key value)      -> i32
(web-storage-remove key)            -> i32
(web-storage-clear)                 -> i32
```

`web-storage-get` returns `-1` when the key is absent. The value is written
into the caller-supplied buffer; check the return value to detect missing
keys before using the buffer contents.

```scheme
(web-storage-set "theme" "dark")
(web-storage-set "lang"  "en-US")

;; Check for stored preference
(define len (web-storage-get "theme" buf 64))
(when (>= len 0)
  (web-console-log "Found stored theme"))

;; Reset all user preferences
(web-storage-clear)
```

---

### 5.15 Fetch API (1 function)

```scheme
(web-fetch url method body) -> i32  ; promise-handle
```

`web-fetch` initiates an asynchronous HTTP request and returns a promise
handle. The actual response processing depends on the host glue layer's
promise resolution mechanism. The `body` argument is a string (used for POST
requests); pass an empty string `""` for GET requests.

```scheme
;; GET request
(define req (web-fetch "https://api.example.com/data" "GET" ""))

;; POST request with JSON body
(define req2 (web-fetch "https://api.example.com/create" "POST"
                        "{\"name\":\"Eshkol\",\"version\":\"1.1\"}"))
```

Note: Promise handles require host-side cooperative scheduling. The host
glue layer must arrange for the WASM module to be re-invoked with the
response data when the network request completes.

---

### 5.16 Canvas 2D API (21 functions)

See Section 6 for a comprehensive treatment with a full drawing example.

```scheme
;; Context
(web-get-context-2d canvas-handle) -> i32   ; context-handle

;; Rectangle drawing
(web-canvas-fill-rect   ctx x y w h)  -> void
(web-canvas-stroke-rect ctx x y w h)  -> void
(web-canvas-clear-rect  ctx x y w h)  -> void

;; Fill and stroke style
(web-canvas-fill-style   ctx color-string) -> void
(web-canvas-stroke-style ctx color-string) -> void
(web-canvas-line-width   ctx width)        -> void

;; Path building
(web-canvas-begin-path ctx)           -> void
(web-canvas-close-path ctx)           -> void
(web-canvas-move-to    ctx x y)       -> void
(web-canvas-line-to    ctx x y)       -> void
(web-canvas-arc        ctx x y r start end) -> void
(web-canvas-fill       ctx)           -> void
(web-canvas-stroke     ctx)           -> void

;; Text
(web-canvas-fill-text ctx text x y) -> void
(web-canvas-font      ctx font-spec) -> void

;; Transforms
(web-canvas-save      ctx) -> void
(web-canvas-restore   ctx) -> void
(web-canvas-translate ctx x y)    -> void
(web-canvas-rotate    ctx angle)  -> void
(web-canvas-scale     ctx sx sy)  -> void
```

---

### 5.17 Handle Management (1 function)

```scheme
(web-release-handle handle) -> void
```

Release the host handle table entry for the given handle. Must be called for
every handle acquired by a creation or query function when that handle is no
longer needed. See Section 4 for full lifecycle guidance.

---

## 6. Canvas 2D Drawing

### 6.1 Obtaining a Context

The Canvas 2D API operates on a context handle obtained from a `<canvas>`
element. The canvas element must exist in the DOM; for WASM applications the
element is typically defined in the HTML file and referenced by ID.

```scheme
(define canvas (web-get-element-by-id "game-canvas"))
(define ctx    (web-get-context-2d canvas))
```

### 6.2 Drawing Primitives

```scheme
;;; draw-scene.esk — Canvas 2D drawing example
(require web)

(define canvas (web-get-element-by-id "canvas"))
(define ctx    (web-get-context-2d canvas))

;; Clear background
(web-canvas-fill-style ctx "#0F172A")
(web-canvas-fill-rect  ctx 0.0 0.0 800.0 600.0)

;; Draw a filled circle
(web-canvas-fill-style ctx "#3B82F6")
(web-canvas-begin-path ctx)
(web-canvas-arc        ctx 400.0 300.0 80.0 0.0 6.283185307)  ; full circle
(web-canvas-fill       ctx)

;; Draw a triangle using path
(web-canvas-stroke-style ctx "#F59E0B")
(web-canvas-line-width   ctx 3.0)
(web-canvas-begin-path   ctx)
(web-canvas-move-to      ctx 200.0 100.0)
(web-canvas-line-to      ctx 300.0 280.0)
(web-canvas-line-to      ctx 100.0 280.0)
(web-canvas-close-path   ctx)
(web-canvas-stroke       ctx)

;; Render text label
(web-canvas-fill-style ctx "#F8FAFC")
(web-canvas-font       ctx "bold 24px sans-serif")
(web-canvas-fill-text  ctx "Eshkol Canvas" 320.0 560.0)
```

### 6.3 Transforms and State Save/Restore

```scheme
;;; Draw a rotated square using transform stack
(define (draw-rotated-square ctx cx cy size angle color)
  (web-canvas-save      ctx)             ; push state
  (web-canvas-translate ctx cx cy)       ; move origin
  (web-canvas-rotate    ctx angle)       ; rotate around origin
  (web-canvas-fill-style ctx color)
  (web-canvas-fill-rect  ctx
    (- (/ size 2.0)) (- (/ size 2.0))   ; centered
    size size)
  (web-canvas-restore   ctx))            ; pop state

(draw-rotated-square ctx 400.0 300.0 120.0 0.785398 "#10B981")  ; 45 degrees
(draw-rotated-square ctx 400.0 300.0  80.0 1.570796 "#6366F1")  ; 90 degrees
```

### 6.4 Animation Loop

Use `web-request-animation-frame` for smooth 60 fps animation. The callback
handle must reference a WASM-exported function.

```scheme
;;; animation.esk
(require web)

(define canvas (web-get-element-by-id "canvas"))
(define ctx    (web-get-context-2d canvas))

;; Mutable angle state — use a vector as a mutable cell
(define state (vector 0.0))

(define (render _frame-time)
  (let ((angle (vector-ref state 0)))
    ;; Clear
    (web-canvas-fill-style ctx "#0F172A")
    (web-canvas-fill-rect  ctx 0.0 0.0 400.0 400.0)

    ;; Orbiting circle
    (web-canvas-fill-style ctx "#F59E0B")
    (web-canvas-begin-path ctx)
    (web-canvas-arc ctx
      (+ 200.0 (* 120.0 (cos angle)))
      (+ 200.0 (* 120.0 (sin angle)))
      20.0 0.0 6.283185307)
    (web-canvas-fill ctx)

    ;; Advance angle
    (vector-set! state 0 (+ angle 0.04))

    ;; Schedule next frame
    (web-request-animation-frame render)))

;; Start animation
(web-request-animation-frame render)
```

---

## 7. Event System

### 7.1 Registering Listeners

`web-add-event-listener` binds a named DOM event to a WASM function. The
third argument is a function table index in the WASM module; the host glue
layer must translate DOM event objects to event handles before invoking it.

```scheme
(define (on-click event)
  (web-event-prevent-default event)
  (let ((target (web-event-get-target event)))
    (web-toggle-class target "selected")))

;; Register listener, save callback-id for later removal
(define click-id
  (web-add-event-listener btn "click" on-click))

;; Deregister when no longer needed
(web-remove-event-listener click-id)
```

### 7.2 Keyboard Events

```scheme
(define (on-keydown event)
  (let ((code (web-event-get-key-code event)))
    (cond
      ((= code 13)   ; Enter
       (submit-form))
      ((= code 27)   ; Escape
       (close-modal))
      (else
       (web-console-log "other key")))))

(web-add-event-listener
  (web-get-document)
  "keydown"
  on-keydown)
```

### 7.3 Mouse Events

```scheme
(define (on-mousemove event)
  (let ((mx (web-event-get-mouse-x event))
        (my (web-event-get-mouse-y event)))
    (update-cursor-position mx my)))

(web-add-event-listener canvas "mousemove" on-mousemove)
```

### 7.4 Event Propagation Control

```scheme
(define (on-dropdown-click event)
  ;; Prevent the click from bubbling to document close handler
  (web-event-stop-propagation event)
  (web-toggle-class dropdown-menu "open"))

(web-add-event-listener dropdown-btn "click" on-dropdown-click)
```

---

## 8. Complete Application Example

The following example demonstrates a self-contained interactive application
combining DOM construction, event handling, local storage persistence, and
canvas visualization.

```scheme
;;; counter-app.esk — Interactive counter with canvas visualisation
(require web)

;;; ─── State ────────────────────────────────────────────────────────────────

(define state (vector 0))   ; count

(define (get-count) (vector-ref state 0))
(define (set-count! n) (vector-set! state 0 n))

;;; ─── Canvas bar chart ─────────────────────────────────────────────────────

(define canvas #f)
(define ctx    #f)

(define (draw-chart)
  (let* ((count (get-count))
         (bar-w  200.0)
         (bar-h  (* (min count 20) 10.0))
         (cx     50.0)
         (cy     (- 200.0 bar-h)))
    ;; Background
    (web-canvas-fill-style ctx "#1E293B")
    (web-canvas-fill-rect  ctx 0.0 0.0 400.0 220.0)
    ;; Bar
    (web-canvas-fill-style ctx "#3B82F6")
    (web-canvas-fill-rect  ctx cx cy bar-w bar-h)
    ;; Label
    (web-canvas-fill-style ctx "#F8FAFC")
    (web-canvas-font       ctx "16px monospace")
    (web-canvas-fill-text  ctx
      (string-append "count=" (number->string count))
      (+ cx 10.0) (- cy 8.0))))

;;; ─── Persistence ──────────────────────────────────────────────────────────

(define (save-state!)
  (web-storage-set "count" (number->string (get-count))))

;;; ─── DOM construction ─────────────────────────────────────────────────────

(define (build-ui)
  (let ((body (web-get-body)))
    ;; Outer container
    (define app (web-create-element "div"))
    (web-add-class app "app-container")
    (web-set-style app "fontFamily" "sans-serif")
    (web-set-style app "padding" "32px")

    ;; Title
    (define title (web-create-element "h1"))
    (web-set-text-content title "Eshkol Counter")
    (web-append-child app title)

    ;; Canvas
    (set! canvas (web-create-element "canvas"))
    (web-set-attribute canvas "width" "400")
    (web-set-attribute canvas "height" "220")
    (web-append-child app canvas)
    (set! ctx (web-get-context-2d canvas))

    ;; Button row
    (define row (web-create-element "div"))
    (web-set-style row "marginTop" "16px")

    (define dec-btn (web-create-element "button"))
    (web-set-text-content dec-btn "- Decrement")
    (web-add-event-listener dec-btn "click"
      (lambda (_ev)
        (when (> (get-count) 0)
          (set-count! (- (get-count) 1))
          (save-state!)
          (draw-chart))))
    (web-append-child row dec-btn)

    (define inc-btn (web-create-element "button"))
    (web-set-text-content inc-btn "+ Increment")
    (web-set-style inc-btn "marginLeft" "8px")
    (web-add-event-listener inc-btn "click"
      (lambda (_ev)
        (set-count! (+ (get-count) 1))
        (save-state!)
        (draw-chart)))
    (web-append-child row inc-btn)

    (define reset-btn (web-create-element "button"))
    (web-set-text-content reset-btn "Reset")
    (web-set-style reset-btn "marginLeft" "8px")
    (web-add-event-listener reset-btn "click"
      (lambda (_ev)
        (set-count! 0)
        (save-state!)
        (draw-chart)))
    (web-append-child row reset-btn)

    (web-append-child app row)
    (web-append-child body app)

    ;; Release temporary element handles (still live in DOM via JS ref)
    (web-release-handle title)
    (web-release-handle row)
    (web-release-handle dec-btn)
    (web-release-handle inc-btn)
    (web-release-handle reset-btn)
    (web-release-handle app)))

;;; ─── Startup ──────────────────────────────────────────────────────────────

;; Restore persisted count if present
;; (buffer read pattern omitted for brevity — see web-storage-get docs)

(build-ui)
(draw-chart)
```

Build and load:

```bash
eshkol-run counter-app.esk --wasm -o counter-app
# Output: counter-app.wasm
```

```html
<canvas id="canvas"></canvas>
<script type="module">
  const { instance } = await WebAssembly.instantiateStreaming(
    fetch("counter-app.wasm"),
    buildEshkolWebImports()
  );
  instance.exports._eshkol_main();
</script>
```

---

## 9. Performance Considerations

### 9.1 Handle Table Pressure

Each call to a creation or query function allocates a slot in the JavaScript
handle table. For DOM-heavy applications that create hundreds of elements in a
tight loop, failing to call `web-release-handle` after each temporary node is
inserted into the DOM will cause the table to grow without bound.

The recommended pattern is: create, mutate, insert, release.

```scheme
;; Efficient: release after insertion
(define (append-list-items parent items)
  (for-each
    (lambda (text)
      (let ((li (web-create-element "li")))
        (web-set-text-content li text)
        (web-append-child parent li)
        (web-release-handle li)))   ; release immediately
    items))
```

### 9.2 DOM Batching

DOM reads and writes that interleave cause browser layout thrashing (forced
synchronous layout). Batch all reads first, then perform all writes:

```scheme
;; Thrashing: alternates read/write per element
(for-each
  (lambda (el)
    (let ((h (web-get-window-height)))  ; forces layout
      (web-set-style el "top" (number->string (/ h 2)))))
  elements)

;; Better: read once, then write all
(let ((h (web-get-window-height)))
  (for-each
    (lambda (el)
      (web-set-style el "top" (number->string (/ h 2))))
    elements))
```

### 9.3 Animation Frames vs Intervals

Use `web-request-animation-frame` instead of `web-set-interval` for visual
updates. The browser synchronizes RAF callbacks with the display refresh
cycle (typically 60 fps or the display's native rate) and automatically
suspends them when the tab is not visible, reducing CPU and battery usage.

### 9.4 Event Listener Cleanup

Long-lived single-page applications that add event listeners dynamically
(e.g., per-modal, per-dialog) must call `web-remove-event-listener` with the
returned callback ID when the component is torn down. Accumulated listeners
on detached or replaced DOM nodes will not fire (since the nodes are gone)
but still consume memory in the host dispatch table.

```scheme
;; Store listener IDs alongside component state
(define modal-listeners '())

(define (open-modal)
  (let ((id1 (web-add-event-listener close-btn "click" on-close))
        (id2 (web-add-event-listener overlay "click" on-close)))
    (set! modal-listeners (list id1 id2))))

(define (close-modal-cleanup)
  (for-each web-remove-event-listener modal-listeners)
  (set! modal-listeners '()))
```

---

## 10. Host Glue Layer

For the web platform to function, the host (browser JavaScript or a WASM
runtime) must provide all 73 C import functions under the namespace expected
by the WASM module. The canonical host implementation maps each C symbol
(e.g., `web_create_element`) to the corresponding browser DOM API:

| C symbol | Browser API |
|----------|-------------|
| `web_get_document` | `-> 1` (reserved) |
| `web_get_window` | `-> 2` (reserved) |
| `web_get_body` | `-> 3` (reserved) |
| `web_create_element` | `document.createElement(tag)` + allocate handle |
| `web_append_child` | `handles[p].appendChild(handles[c])` |
| `web_add_event_listener` | `handles[el].addEventListener(name, ...)` |
| `web_canvas_fill_rect` | `handles[ctx].fillRect(x,y,w,h)` |
| `web_storage_get` | `localStorage.getItem(key)` + copy to WASM memory |
| ... | ... |

Strings passed as `ptr` arguments are null-terminated UTF-8 in WASM linear
memory. The host must read them with `TextDecoder` or equivalent. Strings
returned from the browser (e.g., `web-get-attribute`, `web-storage-get`) are
written back into caller-provided WASM linear memory buffers by the host.

---

## 11. See Also

- `lib/web/http.esk` — Full source for all 73 extern declarations
- `exe/eshkol-run.cpp` — `--wasm` flag handling and WASM file emission
- `lib/backend/llvm_codegen.cpp` (`eshkol_compile_llvm_ir_to_wasm`) — LLVM
  WebAssembly backend integration
- [Compilation Guide](COMPILATION_GUIDE.md) — General compiler flags and
  build pipeline
- [Getting Started](GETTING_STARTED.md) — Installation and first programs
- [FFI and Extern Declarations](COMPILER_ARCHITECTURE.md) — The `extern`
  form and FFI type system
