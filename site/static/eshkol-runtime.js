/**
 * Eshkol WASM Runtime — DOM bridge for Eshkol programs compiled to WebAssembly.
 *
 * Extracted from web/eshkol-repl.js (the browser REPL runtime).
 * Provides the handle-based DOM API that maps Eshkol's extern declarations
 * to JavaScript DOM operations.
 *
 * Handle system: integer handles → JavaScript objects
 *   0 = null, 1 = document, 2 = window, 3 = document.body
 */
class EshkolRuntime {
    constructor() {
        // Handle system
        this.handles = new Map();
        this.nextHandle = 1;
        this.documentHandle = this.createHandle(document);       // 1
        this.windowHandle = this.createHandle(window);           // 2
        this.bodyHandle = this.createHandle(document.body);      // 3

        // WASM instance (set after instantiation)
        this.instance = null;
        this.memory = null;
    }

    setInstance(instance) {
        this.instance = instance;
        this.memory = instance.exports.memory;
    }

    // === Handle System ===

    createHandle(obj) {
        if (obj === null || obj === undefined) return 0;
        const handle = this.nextHandle++;
        this.handles.set(handle, obj);
        return handle;
    }

    getHandle(handle) {
        if (handle === 0) return null;
        return this.handles.get(handle) || null;
    }

    releaseHandle(handle) {
        if (handle > 3) this.handles.delete(handle);
    }

    // === String Helpers ===

    readString(ptr) {
        if (!ptr || !this.memory) return '';
        const bytes = new Uint8Array(this.memory.buffer);
        let end = ptr;
        while (end < bytes.length && bytes[end] !== 0) end++;
        return new TextDecoder().decode(bytes.slice(ptr, end));
    }

    writeString(ptr, str, maxLen) {
        if (!ptr || !this.memory) return 0;
        const bytes = new TextEncoder().encode(str);
        const view = new Uint8Array(this.memory.buffer);
        const len = Math.min(bytes.length, maxLen - 1);
        view.set(bytes.subarray(0, len), ptr);
        view[ptr + len] = 0;
        return len;
    }

    // === WASM Imports ===

    createImports() {
        const rt = this;
        return {
            env: {
                // Memory
                __linear_memory: new WebAssembly.Memory({ initial: 256, maximum: 1024 }),

                // Arena stubs (minimal — use static strings in Eshkol code)
                arena_create: (size) => 1,
                arena_create_threadsafe: (size) => 1,
                arena_destroy: (arena) => {},
                arena_allocate: (arena, size) => {
                    // Bump allocator in WASM memory
                    if (!rt._bumpPtr) rt._bumpPtr = 65536; // Start after 64KB
                    const ptr = rt._bumpPtr;
                    rt._bumpPtr += ((size + 7) & ~7); // 8-byte aligned
                    return ptr;
                },
                arena_allocate_with_header: (arena, size, type, subtype) => {
                    if (!rt._bumpPtr) rt._bumpPtr = 65536;
                    const ptr = rt._bumpPtr;
                    rt._bumpPtr += ((size + 15) & ~7);
                    return ptr + 8; // Skip header
                },
                arena_push_scope: (arena) => {},
                arena_pop_scope: (arena) => {},

                // Cons cells (for lists)
                arena_allocate_cons_cell: (arena) => {
                    return rt.createImports().env.arena_allocate(arena, 32);
                },
                arena_allocate_cons_with_header: (arena) => {
                    return rt.createImports().env.arena_allocate_with_header(arena, 24, 8, 0);
                },
                arena_allocate_tagged_cons_cell: (arena) => {
                    return rt.createImports().env.arena_allocate(arena, 48);
                },
                eshkol_tagged_cons_set_tagged_value: () => {},

                // Tensor stubs
                arena_allocate_tensor_with_header: (arena) => {
                    return rt.createImports().env.arena_allocate(arena, 64);
                },
                arena_allocate_tensor_full: (arena, ndim, total) => {
                    return rt.createImports().env.arena_allocate(arena, 32 + total * 8);
                },
                arena_allocate_vector_with_header: (arena, n) => {
                    return rt.createImports().env.arena_allocate(arena, 8 + n * 16);
                },

                // AD stubs
                arena_allocate_ad_node: (arena) => rt.createImports().env.arena_allocate(arena, 128),
                arena_allocate_ad_node_with_header: (arena) => rt.createImports().env.arena_allocate_with_header(arena, 128, 9, 2),
                arena_allocate_tape: (arena, cap) => rt.createImports().env.arena_allocate(arena, 64),
                arena_tape_add_node: () => 0,
                arena_tape_reset: () => {},
                arena_tape_get_node: () => 0,
                arena_tape_get_node_count: () => 0,

                // String stubs
                arena_allocate_string_with_header: (arena, len) => {
                    return rt.createImports().env.arena_allocate_with_header(arena, len + 1, 8, 1);
                },

                // Closure stubs
                arena_allocate_closure_with_header: (arena, funcPtr, packed, sexpr, retType, name) => {
                    return rt.createImports().env.arena_allocate(arena, 64);
                },

                // Runtime functions
                eshkol_init_global_arena: () => {},
                eshkol_init_stack_size: () => {},
                eshkol_display_value: () => {},
                eshkol_deep_equal: () => 0,
                eshkol_lambda_registry_init: () => {},
                eshkol_lambda_registry_add: () => {},
                eshkol_lambda_registry_lookup: () => 0,
                eshkol_bignum_binary_tagged: () => 0,
                eshkol_bignum_compare_tagged: () => 0,
                eshkol_is_bignum_tagged: () => 0,
                eshkol_rational_compare_tagged_ptr: () => 0,

                // I/O
                printf: (fmt, ...args) => { console.log('printf:', rt.readString(fmt)); return 0; },
                fprintf: (stream, fmt, ...args) => { console.log(rt.readString(fmt)); return 0; },
                puts: (s) => { console.log(rt.readString(s)); return 0; },
                putchar: (c) => { const ch = String.fromCharCode(c); process?.stdout?.write?.(ch) ?? console.log(ch); return c; },
                fflush: () => 0,
                snprintf: () => 0,
                abort: () => { throw new Error('abort called'); },
                exit: (code) => { console.log('exit:', code); },
                pow: Math.pow,
                fmod: (a, b) => a % b,
                strlen: (ptr) => { let len = 0; const b = new Uint8Array(rt.memory?.buffer || rt.createImports().env.__linear_memory.buffer); while (b[ptr + len]) len++; return len; },
                memcpy: (dst, src, n) => { const b = new Uint8Array(rt.memory?.buffer || new ArrayBuffer(0)); b.copyWithin(dst, src, src + n); return dst; },
                memset: (ptr, val, n) => { const b = new Uint8Array(rt.memory?.buffer || new ArrayBuffer(0)); b.fill(val, ptr, ptr + n); return ptr; },

                // Math
                sin: Math.sin, cos: Math.cos, tan: Math.tan,
                asin: Math.asin, acos: Math.acos, atan: Math.atan, atan2: Math.atan2,
                sinh: Math.sinh, cosh: Math.cosh, tanh: Math.tanh,
                exp: Math.exp, log: Math.log, log2: Math.log2, log10: Math.log10,
                sqrt: Math.sqrt, cbrt: Math.cbrt, fabs: Math.abs,
                floor: Math.floor, ceil: Math.ceil, round: Math.round, trunc: Math.trunc,
                fmin: Math.min, fmax: Math.max,
                exp2: (x) => Math.pow(2, x),

                // Globals
                __global_arena: new WebAssembly.Global({ value: 'i32', mutable: false }, 0),
                __stdinp: new WebAssembly.Global({ value: 'i32', mutable: false }, 0),
                __stdoutp: new WebAssembly.Global({ value: 'i32', mutable: false }, 0),
                __stderrp: new WebAssembly.Global({ value: 'i32', mutable: false }, 0),

                // === DOM API ===

                web_get_body: () => rt.bodyHandle,
                web_get_document: () => rt.documentHandle,
                web_get_window: () => rt.windowHandle,

                web_create_element: (tagPtr) => {
                    const tag = rt.readString(tagPtr);
                    return rt.createHandle(document.createElement(tag));
                },

                web_create_text_node: (textPtr) => {
                    return rt.createHandle(document.createTextNode(rt.readString(textPtr)));
                },

                web_append_child: (parentHandle, childHandle) => {
                    const parent = rt.getHandle(parentHandle);
                    const child = rt.getHandle(childHandle);
                    if (parent && child) parent.appendChild(child);
                    return childHandle;
                },

                web_remove_child: (parentHandle, childHandle) => {
                    const parent = rt.getHandle(parentHandle);
                    const child = rt.getHandle(childHandle);
                    if (parent && child) parent.removeChild(child);
                    return 0;
                },

                web_set_text_content: (elHandle, textPtr) => {
                    const el = rt.getHandle(elHandle);
                    if (el) el.textContent = rt.readString(textPtr);
                    return 0;
                },

                web_set_inner_html: (elHandle, htmlPtr) => {
                    const el = rt.getHandle(elHandle);
                    if (el) el.innerHTML = rt.readString(htmlPtr);
                    return 0;
                },

                web_set_attribute: (elHandle, namePtr, valuePtr) => {
                    const el = rt.getHandle(elHandle);
                    if (el) el.setAttribute(rt.readString(namePtr), rt.readString(valuePtr));
                    return 0;
                },

                web_get_attribute: (elHandle, namePtr, bufPtr, bufLen) => {
                    const el = rt.getHandle(elHandle);
                    if (!el) return 0;
                    const val = el.getAttribute(rt.readString(namePtr)) || '';
                    return rt.writeString(bufPtr, val, bufLen);
                },

                web_set_style: (elHandle, propPtr, valuePtr) => {
                    const el = rt.getHandle(elHandle);
                    if (el) el.style[rt.readString(propPtr)] = rt.readString(valuePtr);
                    return 0;
                },

                web_add_class: (elHandle, classPtr) => {
                    const el = rt.getHandle(elHandle);
                    if (el) el.classList.add(rt.readString(classPtr));
                    return 0;
                },

                web_remove_class: (elHandle, classPtr) => {
                    const el = rt.getHandle(elHandle);
                    if (el) el.classList.remove(rt.readString(classPtr));
                    return 0;
                },

                web_get_element_by_id: (idPtr) => {
                    const el = document.getElementById(rt.readString(idPtr));
                    return el ? rt.createHandle(el) : 0;
                },

                web_query_selector: (selectorPtr) => {
                    const el = document.querySelector(rt.readString(selectorPtr));
                    return el ? rt.createHandle(el) : 0;
                },

                web_query_selector_all: (selectorPtr) => {
                    const els = document.querySelectorAll(rt.readString(selectorPtr));
                    return rt.createHandle(Array.from(els));
                },

                web_add_event_listener: (elHandle, eventPtr, callbackFuncPtr) => {
                    const el = rt.getHandle(elHandle);
                    if (!el || !rt.instance) return 0;
                    const eventName = rt.readString(eventPtr);
                    const fn = rt.instance.exports.__indirect_function_table.get(callbackFuncPtr);
                    if (!fn) return 0;
                    const handler = (e) => {
                        const eventHandle = rt.createHandle(e);
                        try { fn(eventHandle); } catch(err) { console.error('Event handler error:', err); }
                    };
                    el.addEventListener(eventName, handler);
                    return 1;
                },

                web_prevent_default: (eventHandle) => {
                    const e = rt.getHandle(eventHandle);
                    if (e && e.preventDefault) e.preventDefault();
                    return 0;
                },

                web_console_log: (msgPtr) => {
                    console.log(rt.readString(msgPtr));
                    return 0;
                },

                // Canvas 2D
                web_canvas_get_context: (canvasHandle, ctxTypePtr) => {
                    const canvas = rt.getHandle(canvasHandle);
                    if (!canvas) return 0;
                    return rt.createHandle(canvas.getContext(rt.readString(ctxTypePtr)));
                },

                web_canvas_fill_rect: (ctxHandle, x, y, w, h) => {
                    const ctx = rt.getHandle(ctxHandle);
                    if (ctx) ctx.fillRect(x, y, w, h);
                    return 0;
                },

                web_canvas_clear_rect: (ctxHandle, x, y, w, h) => {
                    const ctx = rt.getHandle(ctxHandle);
                    if (ctx) ctx.clearRect(x, y, w, h);
                    return 0;
                },

                web_canvas_set_fill_style: (ctxHandle, colorPtr) => {
                    const ctx = rt.getHandle(ctxHandle);
                    if (ctx) ctx.fillStyle = rt.readString(colorPtr);
                    return 0;
                },

                web_canvas_set_stroke_style: (ctxHandle, colorPtr) => {
                    const ctx = rt.getHandle(ctxHandle);
                    if (ctx) ctx.strokeStyle = rt.readString(colorPtr);
                    return 0;
                },

                web_canvas_set_line_width: (ctxHandle, width) => {
                    const ctx = rt.getHandle(ctxHandle);
                    if (ctx) ctx.lineWidth = width;
                    return 0;
                },

                web_canvas_begin_path: (ctxHandle) => {
                    const ctx = rt.getHandle(ctxHandle);
                    if (ctx) ctx.beginPath();
                    return 0;
                },

                web_canvas_move_to: (ctxHandle, x, y) => {
                    const ctx = rt.getHandle(ctxHandle);
                    if (ctx) ctx.moveTo(x, y);
                    return 0;
                },

                web_canvas_line_to: (ctxHandle, x, y) => {
                    const ctx = rt.getHandle(ctxHandle);
                    if (ctx) ctx.lineTo(x, y);
                    return 0;
                },

                web_canvas_stroke: (ctxHandle) => {
                    const ctx = rt.getHandle(ctxHandle);
                    if (ctx) ctx.stroke();
                    return 0;
                },

                web_canvas_fill: (ctxHandle) => {
                    const ctx = rt.getHandle(ctxHandle);
                    if (ctx) ctx.fill();
                    return 0;
                },

                web_canvas_fill_text: (ctxHandle, textPtr, x, y) => {
                    const ctx = rt.getHandle(ctxHandle);
                    if (ctx) ctx.fillText(rt.readString(textPtr), x, y);
                    return 0;
                },

                web_canvas_set_font: (ctxHandle, fontPtr) => {
                    const ctx = rt.getHandle(ctxHandle);
                    if (ctx) ctx.font = rt.readString(fontPtr);
                    return 0;
                },

                web_canvas_arc: (ctxHandle, x, y, r, start, end) => {
                    const ctx = rt.getHandle(ctxHandle);
                    if (ctx) ctx.arc(x, y, r, start, end);
                    return 0;
                },

                // Timers & Animation
                web_set_timeout: (callbackFuncPtr, ms) => {
                    if (!rt.instance) return 0;
                    const fn = rt.instance.exports.__indirect_function_table.get(callbackFuncPtr);
                    return setTimeout(() => { try { fn(0); } catch(e) { console.error(e); } }, ms);
                },

                web_set_interval: (callbackFuncPtr, ms) => {
                    if (!rt.instance) return 0;
                    const fn = rt.instance.exports.__indirect_function_table.get(callbackFuncPtr);
                    return setInterval(() => { try { fn(0); } catch(e) { console.error(e); } }, ms);
                },

                web_clear_interval: (id) => { clearInterval(id); return 0; },
                web_clear_timeout: (id) => { clearTimeout(id); return 0; },

                web_request_animation_frame: (callbackFuncPtr) => {
                    if (!rt.instance) return 0;
                    const fn = rt.instance.exports.__indirect_function_table.get(callbackFuncPtr);
                    return requestAnimationFrame((ts) => { try { fn(ts | 0); } catch(e) { console.error(e); } });
                },

                // Fetch API
                web_fetch: (urlPtr, callbackFuncPtr) => {
                    const url = rt.readString(urlPtr);
                    fetch(url).then(r => r.text()).then(text => {
                        if (rt.instance) {
                            const fn = rt.instance.exports.__indirect_function_table.get(callbackFuncPtr);
                            // Write response to WASM memory
                            const ptr = rt.createImports().env.arena_allocate(0, text.length + 1);
                            rt.writeString(ptr, text, text.length + 1);
                            try { fn(ptr); } catch(e) { console.error(e); }
                        }
                    }).catch(e => console.error('fetch error:', e));
                    return 0;
                },

                // Location
                web_get_hash: (bufPtr, bufLen) => {
                    return rt.writeString(bufPtr, window.location.hash, bufLen);
                },

                web_set_hash: (hashPtr) => {
                    window.location.hash = rt.readString(hashPtr);
                    return 0;
                },

                web_get_href: (bufPtr, bufLen) => {
                    return rt.writeString(bufPtr, window.location.href, bufLen);
                },

                // Window
                web_get_window_width: () => window.innerWidth,
                web_get_window_height: () => window.innerHeight,
                web_get_scroll_y: () => window.scrollY | 0,

                // Performance
                web_performance_now: () => performance.now() | 0,

                // Storage
                web_local_storage_set: (keyPtr, valuePtr) => {
                    localStorage.setItem(rt.readString(keyPtr), rt.readString(valuePtr));
                    return 0;
                },

                web_local_storage_get: (keyPtr, bufPtr, bufLen) => {
                    const val = localStorage.getItem(rt.readString(keyPtr)) || '';
                    return rt.writeString(bufPtr, val, bufLen);
                },

                // Event properties
                web_event_target: (eventHandle) => {
                    const e = rt.getHandle(eventHandle);
                    return e?.target ? rt.createHandle(e.target) : 0;
                },

                web_event_key: (eventHandle, bufPtr, bufLen) => {
                    const e = rt.getHandle(eventHandle);
                    return rt.writeString(bufPtr, e?.key || '', bufLen);
                },

                web_event_client_x: (eventHandle) => {
                    const e = rt.getHandle(eventHandle);
                    return e?.clientX || 0;
                },

                web_event_client_y: (eventHandle) => {
                    const e = rt.getHandle(eventHandle);
                    return e?.clientY || 0;
                },

                // Input elements
                web_get_value: (elHandle, bufPtr, bufLen) => {
                    const el = rt.getHandle(elHandle);
                    return rt.writeString(bufPtr, el?.value || '', bufLen);
                },

                web_set_value: (elHandle, valuePtr) => {
                    const el = rt.getHandle(elHandle);
                    if (el) el.value = rt.readString(valuePtr);
                    return 0;
                },
            }
        };
    }
}
