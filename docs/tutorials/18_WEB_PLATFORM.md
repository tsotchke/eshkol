# Tutorial 18: Web Platform and WebAssembly

Eshkol compiles to WebAssembly for browser deployment. The eshkol.ai
website itself is written in Eshkol and compiled to WASM.

---

## Compiling to WASM

```bash
# Compile an Eshkol program to WebAssembly
$ eshkol-run program.esk -w -o program.wasm
```

The `-w` flag targets the wasm32 backend. The output `.wasm` file can be
loaded by any JavaScript runtime that supports WebAssembly.

---

## The Browser REPL

The browser REPL at [eshkol.ai](https://eshkol.ai) runs the bytecode VM
compiled to WASM via Emscripten. It supports:

- All Scheme primitives (lists, strings, numbers, booleans)
- Exact arithmetic (bignums, rationals)
- Forward-mode autodiff (dual numbers)
- Complex numbers
- Higher-order functions (map, fold, filter)
- Closures and first-class continuations
- call/cc, dynamic-wind, guard/raise

Type any expression and press Enter:

```scheme
(define (factorial n)
  (if (= n 0) 1 (* n (factorial (- n 1)))))
(display (factorial 20))
;; => 2432902008176640000

(display (derivative (lambda (x) (* x x x)) 2.0))
;; => 12.0
```

---

## DOM API (80+ Functions)

When compiling a full web application, Eshkol provides DOM bindings via
`extern` declarations:

```scheme
;; Declare external DOM functions
(extern i32 web-create-element ptr :real web_create_element)
(extern i32 web-set-text-content i32 ptr :real web_set_text_content)
(extern i32 web-append-child i32 i32 :real web_append_child)
(extern i32 web-set-inner-html i32 ptr :real web_set_inner_html)
(extern i32 web-add-class i32 ptr :real web_add_class)
(extern i32 web-set-attribute i32 ptr ptr :real web_set_attribute)
(extern i32 web-add-event-listener i32 ptr ptr :real web_add_event_listener)

;; Create a heading
(define h1 (web-create-element "h1"))
(web-set-text-content h1 "Hello from Eshkol!")
(web-append-child 0 h1)  ;; 0 = document.body handle
```

The eshkol.ai website uses this API to build the entire UI — navigation,
code blocks, the REPL, documentation viewer, download tables — all in
Eshkol.

---

## Event Handling

```scheme
(extern i32 web-add-event-listener i32 ptr ptr :real web_add_event_listener)

;; Add click handler
(web-add-event-listener button "click"
  (lambda (event)
    (web-set-text-content output "Button clicked!")))

;; Keyboard events
(web-add-event-listener input "keydown"
  (lambda (event)
    (let ((key (web-event-key event)))
      (if (string=? key "Enter")
          (process-input)))))
```

---

## Styling

```scheme
;; Set inline styles via a helper
(define (style! el prop val)
  (web-set-style-property el prop val))

(style! div "backgroundColor" "#0d0d15")
(style! div "padding" "20px")
(style! div "borderRadius" "12px")
```

---

## Architecture

```
Eshkol source (.esk)
  → LLVM IR (wasm32 target)
  → .wasm binary
  → loaded by eshkol-runtime.js
  → DOM bindings bridge JS ↔ WASM
```

The runtime (`eshkol-runtime.js`) provides:
- Memory management (bump allocator for arena)
- DOM API bridging (element creation, events, styles)
- String marshalling (UTF-8 ↔ WASM linear memory)
- The bytecode VM REPL (separate `eshkol-vm.js`)
