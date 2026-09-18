---
kind: reference
status: current
owner-area: web
since: v1.3.5
sources:
  - lib/web/web.esk
  - web/eshkol-repl.js
---

# `web.web` — DOM manipulation and browser APIs

**Source**: [`lib/web/web.esk`](../../../lib/web/web.esk)
**Require**: `(require web)` — **must be required individually**; there is no `(require web)` line in `lib/stdlib.esk`, so `(require stdlib)` alone does not load it. `(require web.web)` resolves to the same file (the dotted name expands directly to `web/web.esk`; the bare `web` form resolves through the directory-as-module rule that tries `<lib>/web/web.esk` before `<lib>/web/index.esk` — see `inc/eshkol/platform_runtime.h`). Both forms are equivalent; examples below use `(require web)` to match the tutorial and `docs/breakdown/WEB_PLATFORM.md`.

## Overview

`web.web` is a browser bridge, not a portable library: every one of its 97 exports is declared with `(extern ... :real <c-symbol>)`, which the compiler turns into a WebAssembly **import** rather than a linkable native symbol. The C-symbol side (`web_get_document`, `web_create_element`, …) exists only as JavaScript functions supplied by a WASM host's import object — in this repository, the glue in [`web/eshkol-repl.js`](../../../web/eshkol-repl.js) (an older, non-matching set of DOM bindings also lives in `site/static/eshkol-runtime.js`; do not use it as a reference for this module — see Known issues). A program that `(require web)` and calls any of these procedures therefore only *runs* inside that hosted setting: a `.wasm` file loaded by a page that supplies the `env.web_*` imports.

### Native JIT and AOT cannot link this module

Compiling code that requires `web.web` with the native engines fails at the link step, not at compile time, because `extern ... :real` on a native target becomes an undefined native symbol reference:

```
$ eshkol-run -r program.esk
Undefined symbols for architecture arm64:
  "_web_console_log", referenced from:
      _main in ...
ld: symbol(s) not found for architecture arm64
clang++: error: linker command failed with exit code 1 (use -v to see invocation)
ERROR: Linking failed with exit code 1
ERROR: -r: native link of '.../program.esk' failed; refusing to fall back to a reduced in-process run.
```

AOT (`eshkol-run program.esk -o program`) fails with the identical `Undefined symbols ... "_web_console_log" ... ld: symbol(s) not found` error at the same link stage — one missing `web_*` symbol is enough to fail the whole build; a program that mixes `web.web` calls with ordinary logic cannot be partially linked native. The only way to build a program that requires `web.web` is `eshkol-run --wasm` (`-w`), which never runs a native linker: it emits WASM directly from LLVM IR and leaves every `web_*` reference as an unresolved import for the JS host to fill in at `WebAssembly.instantiate()` time.

### The handle model

Every DOM/browser object crossing the WASM/JS boundary is represented as a plain `i32` **handle** — an opaque key into a host-side JavaScript `Map`, never a real pointer into WASM linear memory. Handles `1`, `2` and `3` are pre-registered and permanent: `(web-get-document)`, `(web-get-window)` and `(web-get-body)` always return exactly `1`, `2` and `3`. Every other handle is allocated on demand by a creation or query call (`web-create-element`, `web-get-element-by-id`, `web-get-context-2d`, event-target lookups, `web-fetch`'s promise, …) and stays live in the host table until explicitly released with `(web-release-handle h)`. Releasing a handle only drops the host-side table entry — it does not detach the underlying DOM node, which the document may still reference; calling `web-release-handle` on handles `1`–`3` is a no-op. A handle value of `0` always means "not found" / "no object" and must not be passed to a function expecting a live handle — every accessor returns `0` for a stale, invalid, or absent handle rather than trapping. An event handle passed into an event-listener callback (see Events, below) is valid only for the duration of that one callback invocation; the host releases it immediately afterward.

### Buffer-return convention

Every function that reads a string out of the browser (`web-get-attribute`, `web-get-inner-html`, `web-get-text-content`, `web-get-style`, `web-get-value`, `web-event-get-key`, `web-prompt`, `web-get-href`, `web-get-hash`, `web-storage-get`) takes a caller-owned `buf` pointer and a `buflen` capacity, writes a NUL-terminated copy truncated to at most `buflen - 1` bytes into `buf`, and returns the **full untruncated length** of the source string as its `i32` result — not the number of bytes actually written. Comparing the return value against `buflen` is how a caller detects truncation. On a missing element, missing attribute, or invalid handle, these functions write nothing and return `0`; none of them ever returns a negative sentinel (`docs/breakdown/WEB_PLATFORM.md` describes `-1` return values for `web-get-attribute` and `web-storage-get` — that does not match `web/eshkol-repl.js`; see Known issues).

---

## Special Handles

| Procedure | Signature | Returns | Notes |
|-----------|-----------|---------|-------|
| <a id="web-get-document"></a>`web-get-document` | `(web-get-document)` | `i32` handle, always `1` | No allocation; safe to call any number of times |
| <a id="web-get-window"></a>`web-get-window` | `(web-get-window)` | `i32` handle, always `2` | |
| <a id="web-get-body"></a>`web-get-body` | `(web-get-body)` | `i32` handle, always `3` | `document.body` |

## Document Methods

| Procedure | Signature | Returns | Notes |
|-----------|-----------|---------|-------|
| <a id="web-create-element"></a>`web-create-element` | `(web-create-element tag)` | new `i32` element handle | `tag` is a tag name string, e.g. `"div"` |
| <a id="web-create-text-node"></a>`web-create-text-node` | `(web-create-text-node text)` | new `i32` node handle | |
| <a id="web-get-element-by-id"></a>`web-get-element-by-id` | `(web-get-element-by-id id)` | `i32` element handle, or `0` if not found | |
| <a id="web-query-selector"></a>`web-query-selector` | `(web-query-selector css-selector)` | `i32` element handle, or `0` if no match | First match only |
| <a id="web-query-selector-all"></a>`web-query-selector-all` | `(web-query-selector-all css-selector)` | `i32` handle wrapping a `NodeList` | See Known issues: this module's generic tree-walkers (`web-get-children-count`/`web-get-child-at`) do not traverse a `NodeList` handle |

## Node Tree Manipulation

| Procedure | Signature | Returns | Notes |
|-----------|-----------|---------|-------|
| <a id="web-append-child"></a>`web-append-child` | `(web-append-child parent child)` | `i32`: `1` ok, `0` if either handle is invalid | |
| <a id="web-remove-child"></a>`web-remove-child` | `(web-remove-child parent child)` | `i32`: `1` ok, `0` if either handle is invalid | Throws (uncaught) if `child` is not actually a child of `parent` |
| <a id="web-insert-before"></a>`web-insert-before` | `(web-insert-before parent new-node ref-node)` | `i32`: `1` ok, `0` if `parent`/`new-node` invalid | `ref-node` may be `0`, which inserts `new-node` as the last child (`insertBefore(node, null)`) |
| <a id="web-replace-child"></a>`web-replace-child` | `(web-replace-child parent new-child old-child)` | `i32`: `1` ok, `0` if any handle is invalid | Throws (uncaught) if `old-child` is not a child of `parent` |
| <a id="web-clone-node"></a>`web-clone-node` | `(web-clone-node node deep)` | new `i32` handle for a detached clone, or `0` | `deep` nonzero performs a recursive deep clone; `0` clones only the node itself |
| <a id="web-get-parent"></a>`web-get-parent` | `(web-get-parent node)` | `i32` handle, or `0` if no parent | |
| <a id="web-get-first-child"></a>`web-get-first-child` | `(web-get-first-child node)` | `i32` handle, or `0` | Includes text nodes (`Node.firstChild`) |
| <a id="web-get-last-child"></a>`web-get-last-child` | `(web-get-last-child node)` | `i32` handle, or `0` | Includes text nodes (`Node.lastChild`) |
| <a id="web-get-next-sibling"></a>`web-get-next-sibling` | `(web-get-next-sibling node)` | `i32` handle, or `0` | |
| <a id="web-get-prev-sibling"></a>`web-get-prev-sibling` | `(web-get-prev-sibling node)` | `i32` handle, or `0` | |
| <a id="web-get-children-count"></a>`web-get-children-count` | `(web-get-children-count node)` | `i32` count | Element children only (`Node.children`, an `HTMLCollection`) — text nodes are not counted, and the result is `0` for any handle without a `.children` property (e.g. a `NodeList` from `web-query-selector-all`) |
| <a id="web-get-child-at"></a>`web-get-child-at` | `(web-get-child-at node index)` | `i32` handle, or `0` if `index` is out of range | Indexes `Node.children` (elements only), same caveat as above |

## Attributes

| Procedure | Signature | Returns | Notes |
|-----------|-----------|---------|-------|
| <a id="web-set-attribute"></a>`web-set-attribute` | `(web-set-attribute el name value)` | `i32`: `1` ok, `0` if `el` invalid | |
| <a id="web-get-attribute"></a>`web-get-attribute` | `(web-get-attribute el name buf buflen)` | `i32` full attribute length (buffer-return convention) | `0` if `el` is invalid or the attribute is absent |
| <a id="web-remove-attribute"></a>`web-remove-attribute` | `(web-remove-attribute el name)` | `i32`: `1` ok, `0` if `el` invalid | |
| <a id="web-has-attribute"></a>`web-has-attribute` | `(web-has-attribute el name)` | `i32`: `1` yes, `0` no | |

## Content

| Procedure | Signature | Returns | Notes |
|-----------|-----------|---------|-------|
| <a id="web-set-inner-html"></a>`web-set-inner-html` | `(web-set-inner-html el html)` | `i32`: `1` ok, `0` if `el` invalid | No sanitization — `html` is assigned to `innerHTML` verbatim |
| <a id="web-get-inner-html"></a>`web-get-inner-html` | `(web-get-inner-html el buf buflen)` | `i32` full length (buffer-return convention) | `0` if `el` invalid |
| <a id="web-set-text-content"></a>`web-set-text-content` | `(web-set-text-content el text)` | `i32`: `1` ok, `0` if `el` invalid | Prefer this over `web-set-inner-html` for plain text — no markup injection |
| <a id="web-get-text-content"></a>`web-get-text-content` | `(web-get-text-content el buf buflen)` | `i32` full length (buffer-return convention) | `0` if `el` invalid |

## CSS Classes

| Procedure | Signature | Returns | Notes |
|-----------|-----------|---------|-------|
| <a id="web-add-class"></a>`web-add-class` | `(web-add-class el class-name)` | `i32`: `1` ok, `0` if `el` has no `classList` | |
| <a id="web-remove-class"></a>`web-remove-class` | `(web-remove-class el class-name)` | `i32`: `1` ok, `0` if `el` has no `classList` | |
| <a id="web-toggle-class"></a>`web-toggle-class` | `(web-toggle-class el class-name)` | `i32`: `1` if the class is now present, `0` if now absent | Also `0` if `el` has no `classList` |
| <a id="web-has-class"></a>`web-has-class` | `(web-has-class el class-name)` | `i32`: `1` yes, `0` no | |

## Styles

| Procedure | Signature | Returns | Notes |
|-----------|-----------|---------|-------|
| <a id="web-set-style"></a>`web-set-style` | `(web-set-style el property value)` | `i32`: `1` ok, `0` if `el` has no `style` object | `property` is the camelCase CSSOM name, e.g. `"backgroundColor"`, not the hyphenated CSS name |
| <a id="web-get-style"></a>`web-get-style` | `(web-get-style el property buf buflen)` | `i32` full length (buffer-return convention) | Reads only the element's own inline `style` attribute, never computed/cascaded style; `0` if unset |

## Form Elements

| Procedure | Signature | Returns | Notes |
|-----------|-----------|---------|-------|
| <a id="web-get-value"></a>`web-get-value` | `(web-get-value el buf buflen)` | `i32` full length (buffer-return convention) | `0` if `el` has no `value` property |
| <a id="web-set-value"></a>`web-set-value` | `(web-set-value el value)` | `i32`: `1` ok, `0` if `el` has no `value` property | |
| <a id="web-get-checked"></a>`web-get-checked` | `(web-get-checked el)` | `i32`: `1` checked, `0` unchecked or no `checked` property | |
| <a id="web-set-checked"></a>`web-set-checked` | `(web-set-checked el state)` | `i32`: `1` ok, `0` if `el` has no `checked` property | `state` uses JS truthiness — any nonzero value checks the box |

## Focus

| Procedure | Signature | Returns | Notes |
|-----------|-----------|---------|-------|
| <a id="web-focus"></a>`web-focus` | `(web-focus el)` | `i32`: `1` ok, `0` if `el` has no `focus` method | |
| <a id="web-blur"></a>`web-blur` | `(web-blur el)` | `i32`: `1` ok, `0` if `el` has no `blur` method | |

## Events

| Procedure | Signature | Returns | Notes |
|-----------|-----------|---------|-------|
| <a id="web-add-event-listener"></a>`web-add-event-listener` | `(web-add-event-listener el event-name callback)` | `i32` nonzero callback-id, or `0` on failure | `callback` is a function value referring to a top-level WASM-exported function; the host invokes it with one argument, an event handle valid only for the duration of that single call |
| <a id="web-remove-event-listener"></a>`web-remove-event-listener` | `(web-remove-event-listener callback-id)` | `i32`: `1` ok, `0` if `callback-id` is unknown | |
| <a id="web-event-prevent-default"></a>`web-event-prevent-default` | `(web-event-prevent-default event)` | `i32`: `1` ok, `0` if `event` invalid | |
| <a id="web-event-stop-propagation"></a>`web-event-stop-propagation` | `(web-event-stop-propagation event)` | `i32`: `1` ok, `0` if `event` invalid | |
| <a id="web-event-get-target"></a>`web-event-get-target` | `(web-event-get-target event)` | `i32` element handle, or `0` if the event has no target | |
| <a id="web-event-get-key"></a>`web-event-get-key` | `(web-event-get-key event buf buflen)` | `i32` full length (buffer-return convention) | `0` if the event is not a keyboard event |
| <a id="web-event-get-key-code"></a>`web-event-get-key-code` | `(web-event-get-key-code event)` | `i32` numeric `keyCode`, or `0` | `KeyboardEvent.keyCode` is a legacy DOM property; some browsers report `0` for keys it never assigned |
| <a id="web-event-get-mouse-x"></a>`web-event-get-mouse-x` | `(web-event-get-mouse-x event)` | `i32` `clientX`, or `0` if not a mouse event | |
| <a id="web-event-get-mouse-y"></a>`web-event-get-mouse-y` | `(web-event-get-mouse-y event)` | `i32` `clientY`, or `0` if not a mouse event | |

## Timers

| Procedure | Signature | Returns | Notes |
|-----------|-----------|---------|-------|
| <a id="web-set-timeout"></a>`web-set-timeout` | `(web-set-timeout callback delay-ms)` | `i32` timer id (the `setTimeout` id) | `callback` is invoked with no arguments |
| <a id="web-set-interval"></a>`web-set-interval` | `(web-set-interval callback delay-ms)` | `i32` timer id (the `setInterval` id) | `callback` is invoked with no arguments, repeatedly |
| <a id="web-clear-timeout"></a>`web-clear-timeout` | `(web-clear-timeout id)` | none (`void`) | |
| <a id="web-clear-interval"></a>`web-clear-interval` | `(web-clear-interval id)` | none (`void`) | |
| <a id="web-request-animation-frame"></a>`web-request-animation-frame` | `(web-request-animation-frame callback)` | `i32` request id | `callback` is invoked with one argument, the frame's `DOMHighResTimeStamp` (not independently exercised in this build — see report) |
| <a id="web-cancel-animation-frame"></a>`web-cancel-animation-frame` | `(web-cancel-animation-frame id)` | none (`void`) | |

## Console

| Procedure | Signature | Returns | Notes |
|-----------|-----------|---------|-------|
| <a id="web-console-log"></a>`web-console-log` | `(web-console-log message)` | none (`void`) | `console.log` |
| <a id="web-console-warn"></a>`web-console-warn` | `(web-console-warn message)` | none (`void`) | `console.warn` |
| <a id="web-console-error"></a>`web-console-error` | `(web-console-error message)` | none (`void`) | `console.error` |

## Window / Dialogs / Scroll

| Procedure | Signature | Returns | Notes |
|-----------|-----------|---------|-------|
| <a id="web-alert"></a>`web-alert` | `(web-alert message)` | none (`void`) | Blocks on `window.alert` until dismissed |
| <a id="web-confirm"></a>`web-confirm` | `(web-confirm message)` | `i32`: `1` OK, `0` Cancel | Blocks on `window.confirm` |
| <a id="web-prompt"></a>`web-prompt` | `(web-prompt message default buf buflen)` | `i32` full length of the entered text (buffer-return convention) | Blocks on `window.prompt`; if the user cancels, `prompt()` returns `null`, treated as `""` (length `0`) |
| <a id="web-get-window-width"></a>`web-get-window-width` | `(web-get-window-width)` | `i32` `window.innerWidth` | |
| <a id="web-get-window-height"></a>`web-get-window-height` | `(web-get-window-height)` | `i32` `window.innerHeight` | |
| <a id="web-get-scroll-x"></a>`web-get-scroll-x` | `(web-get-scroll-x)` | `i32` `window.scrollX`, truncated to integer | |
| <a id="web-get-scroll-y"></a>`web-get-scroll-y` | `(web-get-scroll-y)` | `i32` `window.scrollY`, truncated to integer | |
| <a id="web-scroll-to"></a>`web-scroll-to` | `(web-scroll-to x y)` | none (`void`) | `window.scrollTo(x, y)` |

## Location

| Procedure | Signature | Returns | Notes |
|-----------|-----------|---------|-------|
| <a id="web-get-href"></a>`web-get-href` | `(web-get-href buf buflen)` | `i32` full length (buffer-return convention) | `location.href` |
| <a id="web-set-href"></a>`web-set-href` | `(web-set-href url)` | none (`void`) | Triggers a full page navigation |
| <a id="web-get-hash"></a>`web-get-hash` | `(web-get-hash buf buflen)` | `i32` full length (buffer-return convention) | `location.hash`, including the leading `#`; empty string if there is no fragment |
| <a id="web-set-hash"></a>`web-set-hash` | `(web-set-hash hash)` | none (`void`) | Updates only the fragment — no reload; the basis for hash-based client-side routing |

## Storage

| Procedure | Signature | Returns | Notes |
|-----------|-----------|---------|-------|
| <a id="web-storage-get"></a>`web-storage-get` | `(web-storage-get key buf buflen)` | `i32` full length (buffer-return convention) | `0` if `key` is absent from `localStorage` — never `-1` |
| <a id="web-storage-set"></a>`web-storage-set` | `(web-storage-set key value)` | `i32`, always `1` | `localStorage.setItem` throws (uncaught) rather than returning failure, e.g. on quota exceeded |
| <a id="web-storage-remove"></a>`web-storage-remove` | `(web-storage-remove key)` | `i32`, always `1` | |
| <a id="web-storage-clear"></a>`web-storage-clear` | `(web-storage-clear)` | `i32`, always `1` | Clears all of `localStorage` for the page's origin, not just keys this program wrote |

## Fetch

| Procedure | Signature | Returns | Notes |
|-----------|-----------|---------|-------|
| <a id="web-fetch"></a>`web-fetch` | `(web-fetch url method body)` | `i32` handle wrapping the `Promise` returned by `fetch()` | `method` empty/`"GET"` sends no body; any other method sends `body` with header `Content-Type: application/json`. This module has no function to await or read the promise's eventual value — resolving it is entirely up to the host glue's own scheduling (see report) |

## Canvas 2D

Every Canvas 2D procedure takes a context handle from `web-get-context-2d` as its first argument.

| Procedure | Signature | Returns | Notes |
|-----------|-----------|---------|-------|
| <a id="web-get-context-2d"></a>`web-get-context-2d` | `(web-get-context-2d canvas)` | `i32` context handle, or `0` if `canvas` has no `getContext` method | `canvas.getContext("2d")` |
| <a id="web-canvas-fill-rect"></a>`web-canvas-fill-rect` | `(web-canvas-fill-rect ctx x y w h)` | none (`void`) | `x y w h` are `double` |
| <a id="web-canvas-stroke-rect"></a>`web-canvas-stroke-rect` | `(web-canvas-stroke-rect ctx x y w h)` | none (`void`) | |
| <a id="web-canvas-clear-rect"></a>`web-canvas-clear-rect` | `(web-canvas-clear-rect ctx x y w h)` | none (`void`) | |
| <a id="web-canvas-fill-style"></a>`web-canvas-fill-style` | `(web-canvas-fill-style ctx color)` | none (`void`) | `color` is any CSS color string |
| <a id="web-canvas-stroke-style"></a>`web-canvas-stroke-style` | `(web-canvas-stroke-style ctx color)` | none (`void`) | |
| <a id="web-canvas-line-width"></a>`web-canvas-line-width` | `(web-canvas-line-width ctx width)` | none (`void`) | `width` is `double` |
| <a id="web-canvas-begin-path"></a>`web-canvas-begin-path` | `(web-canvas-begin-path ctx)` | none (`void`) | |
| <a id="web-canvas-close-path"></a>`web-canvas-close-path` | `(web-canvas-close-path ctx)` | none (`void`) | |
| <a id="web-canvas-move-to"></a>`web-canvas-move-to` | `(web-canvas-move-to ctx x y)` | none (`void`) | |
| <a id="web-canvas-line-to"></a>`web-canvas-line-to` | `(web-canvas-line-to ctx x y)` | none (`void`) | |
| <a id="web-canvas-arc"></a>`web-canvas-arc` | `(web-canvas-arc ctx x y r start end)` | none (`void`) | `start`/`end` are angles in radians |
| <a id="web-canvas-fill"></a>`web-canvas-fill` | `(web-canvas-fill ctx)` | none (`void`) | Fills the current path |
| <a id="web-canvas-stroke"></a>`web-canvas-stroke` | `(web-canvas-stroke ctx)` | none (`void`) | Strokes the current path |
| <a id="web-canvas-fill-text"></a>`web-canvas-fill-text` | `(web-canvas-fill-text ctx text x y)` | none (`void`) | |
| <a id="web-canvas-font"></a>`web-canvas-font` | `(web-canvas-font ctx font)` | none (`void`) | `font` is a CSS font shorthand, e.g. `"bold 24px sans-serif"` |
| <a id="web-canvas-save"></a>`web-canvas-save` | `(web-canvas-save ctx)` | none (`void`) | Pushes the drawing state stack |
| <a id="web-canvas-restore"></a>`web-canvas-restore` | `(web-canvas-restore ctx)` | none (`void`) | Pops the drawing state stack |
| <a id="web-canvas-translate"></a>`web-canvas-translate` | `(web-canvas-translate ctx x y)` | none (`void`) | |
| <a id="web-canvas-rotate"></a>`web-canvas-rotate` | `(web-canvas-rotate ctx angle)` | none (`void`) | `angle` in radians |
| <a id="web-canvas-scale"></a>`web-canvas-scale` | `(web-canvas-scale ctx sx sy)` | none (`void`) | |

## Handle Management

| Procedure | Signature | Returns | Notes |
|-----------|-----------|---------|-------|
| <a id="web-release-handle"></a>`web-release-handle` | `(web-release-handle handle)` | none (`void`) | Frees the host-side table entry only; no-op on handles `1`–`3`; does not detach the DOM node |

---

## Example

The following program touches every group above at least once — element creation, attributes, content, classes, styles, form values, node-tree assembly, event listeners, a timer, console output, local storage, a location hash update, canvas 2D drawing, and handle release. It requires a hosted WASM runtime to execute (see the native-failure block above), so this only demonstrates that it **compiles**:

```scheme
;;; web-demo.esk - representative spread of the web.web API surface
(require web)

(define (build-ui)
  (let* ((body    (web-get-body))
         (panel   (web-create-element "div"))
         (heading (web-create-element "h1"))
         (input   (web-create-element "input"))
         (button  (web-create-element "button"))
         (canvas  (web-create-element "canvas")))

    ;; Content + attributes + classes + styles
    (web-set-text-content heading "Eshkol Web Demo")
    (web-add-class panel "demo-panel")
    (web-set-style panel "padding" "12px")
    (web-set-attribute input "placeholder" "type a message")
    (web-set-value input "hello")
    (web-set-attribute canvas "id" "demo-canvas")

    ;; Assemble the tree
    (web-append-child panel heading)
    (web-append-child panel input)
    (web-append-child panel button)
    (web-append-child panel canvas)
    (web-append-child body panel)

    ;; Event handling
    (web-add-event-listener button "click" on-click)

    ;; Local storage + location + window
    (web-storage-set "last-run" "web-demo")
    (web-console-log "web-demo: UI built")
    (web-set-hash "demo")

    ;; Canvas 2D drawing
    (let ((ctx (web-get-context-2d canvas)))
      (web-canvas-fill-style ctx "#3B82F6")
      (web-canvas-fill-rect  ctx 0.0 0.0 100.0 50.0)
      (web-canvas-begin-path ctx)
      (web-canvas-arc ctx 50.0 25.0 20.0 0.0 6.283185307)
      (web-canvas-stroke ctx)
      (web-release-handle ctx))

    ;; Timer-driven follow-up
    (web-set-timeout on-timeout 1000)

    ;; Release the transient handles this function created directly
    (web-release-handle heading)
    (web-release-handle input)
    (web-release-handle button)
    (web-release-handle canvas)
    (web-release-handle panel)))

(define (on-click event)
  (web-event-prevent-default event)
  (web-console-log "button clicked")
  (web-toggle-class (web-event-get-target event) "active"))

(define (on-timeout)
  (web-console-warn "one second elapsed")
  (web-alert "demo timer fired"))

(build-ui)
```

```
$ eshkol-run --wasm web-demo.esk -o web-demo
$ ls -la web-demo.wasm
-rw-r--r--  1 user  staff  192793 web-demo.wasm
$ file web-demo.wasm
web-demo.wasm: WebAssembly (wasm) binary module version 0x1 (MVP)
```

The module exports a zero-argument `main` entry point; a hosting page calls it after `WebAssembly.instantiate()` supplies the `env.web_*` imports (`docs/breakdown/WEB_PLATFORM.md` names the export `_eshkol_main` — that does not match this build; see report).

## Known issues

### `web-query-selector-all` handles are not walked by the generic tree functions

`web-query-selector-all` wraps a `NodeList` in a handle, but `web-get-children-count`/`web-get-child-at` read `Node.children` (an `HTMLCollection`, defined on `Element`/`Document`/`DocumentFragment`), which `NodeList` does not have. Calling either on a `web-query-selector-all` result reads `undefined` and returns `0` every time — it is not possible to enumerate a `NodeList` handle through this module's exports. `docs/breakdown/WEB_PLATFORM.md` §5.3 suggests exactly this traversal ("traverse it using `web-get-children-count` and `web-get-child-at`"); that guidance does not work against the current glue.

### Buffer-returning calls never return `-1`

`docs/breakdown/WEB_PLATFORM.md` documents `-1` as the "attribute does not exist / buffer too small" sentinel for `web-get-attribute`, and as the "key absent" sentinel for `web-storage-get`. Neither `web_get_attribute` nor `web_storage_get` in `web/eshkol-repl.js` ever returns a negative number: a missing attribute, key, or invalid handle returns `0` (empty string written), and a buffer smaller than the value's length still returns the full source length (not `-1`) with the copy silently truncated. The same pattern (return `0` on absence/invalidity, full length on success) holds for every other buffer-returning function in this module.

### Two non-matching sets of `web_*` JavaScript bindings exist in the tree

`web/eshkol-repl.js` implements all 97 `web_*` imports and matches `lib/web/web.esk` exactly (verified by symbol-name and arity diff against every `(extern ...)` form). `site/static/eshkol-runtime.js` implements a different, smaller, older set (`web_canvas_get_context`, `web_canvas_set_fill_style`, `web_event_client_x`, `web_load_content`, …) that does not match the current `web.esk` export names at all. A WASM module built against current `web.web` would fail to instantiate against `site/static/eshkol-runtime.js`'s import object.
