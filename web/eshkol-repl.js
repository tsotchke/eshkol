/**
 * Eshkol REPL Client Library
 *
 * Communicates with the Eshkol WASM compilation server and manages
 * WebAssembly module instantiation in the browser.
 *
 * Eshkol is a Lisp-family language with first-class automatic differentiation.
 */

class EshkolRepl {
    constructor(serverUrl = 'http://localhost:8080') {
        this.serverUrl = serverUrl;
        this.sessionId = null;
        this.modules = [];      // Compiled WASM modules
        this.instances = [];    // Instantiated modules
        this.symbols = new Map(); // Symbol table: name -> { func, arity, moduleIndex }
        this.memory = null;     // Shared WebAssembly.Memory (for future use)

        // DOM handle system - maps integer handles to JS objects
        this.handles = new Map();
        this.nextHandle = 1;  // 0 = null/invalid

        // Pre-register special handles
        this.documentHandle = this.createHandle(document);
        this.windowHandle = this.createHandle(window);
        this.bodyHandle = this.createHandle(document.body);

        // Callback registry for event handlers
        this.callbacks = new Map();
        this.nextCallbackId = 1;
    }

    /**
     * Create a handle for a JS object
     * @param {any} obj - JavaScript object
     * @returns {number} - Integer handle
     */
    createHandle(obj) {
        if (obj === null || obj === undefined) return 0;
        const handle = this.nextHandle++;
        this.handles.set(handle, obj);
        return handle;
    }

    /**
     * Get JS object from handle
     * @param {number} handle - Integer handle
     * @returns {any} - JavaScript object or null
     */
    getHandle(handle) {
        if (handle === 0) return null;
        return this.handles.get(handle) || null;
    }

    /**
     * Release a handle
     * @param {number} handle - Integer handle
     */
    releaseHandle(handle) {
        if (handle > 3) {  // Don't release document, window, body
            this.handles.delete(handle);
        }
    }

    /**
     * Read a null-terminated string from WASM memory
     * @param {number} ptr - Pointer to string in WASM memory
     * @returns {string} - JavaScript string
     */
    readString(ptr) {
        if (!this.memory || ptr === 0) return '';
        const view = new Uint8Array(this.memory.buffer);
        let end = ptr;
        while (view[end] !== 0) end++;
        const bytes = view.slice(ptr, end);
        return new TextDecoder().decode(bytes);
    }

    /**
     * Write a string to WASM memory (caller must ensure space)
     * @param {string} str - JavaScript string
     * @param {number} ptr - Pointer to write location
     * @returns {number} - Number of bytes written (including null terminator)
     */
    writeString(str, ptr) {
        if (!this.memory) return 0;
        const bytes = new TextEncoder().encode(str);
        const view = new Uint8Array(this.memory.buffer);
        view.set(bytes, ptr);
        view[ptr + bytes.length] = 0;  // Null terminator
        return bytes.length + 1;
    }

    /**
     * Check if the server is available
     */
    async checkHealth() {
        try {
            const response = await fetch(`${this.serverUrl}/health`, {
                method: 'GET',
                mode: 'cors'
            });
            const data = await response.json();
            return data.status === 'ok';
        } catch (e) {
            console.error('Health check failed:', e);
            return false;
        }
    }

    /**
     * Compile Eshkol code to WASM
     * @param {string} code - Eshkol Scheme source code
     * @returns {Object} - { success, wasm, size, error, session_id }
     */
    async compile(code) {
        try {
            const response = await fetch(`${this.serverUrl}/compile`, {
                method: 'POST',
                mode: 'cors',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    code: code,
                    session_id: this.sessionId
                })
            });

            const result = await response.json();

            if (result.success) {
                this.sessionId = result.session_id;

                // Decode base64 WASM
                const wasmBinary = this.base64ToArrayBuffer(result.wasm);

                // Compile to WebAssembly.Module
                const module = await WebAssembly.compile(wasmBinary);
                this.modules.push({
                    module: module,
                    binary: wasmBinary,
                    size: result.size,
                    timestamp: Date.now()
                });

                return {
                    success: true,
                    size: result.size,
                    moduleIndex: this.modules.length - 1
                };
            } else {
                return {
                    success: false,
                    error: result.error
                };
            }
        } catch (e) {
            return {
                success: false,
                error: e.message
            };
        }
    }

    /**
     * Instantiate the latest compiled WASM module
     * @returns {Object} - WebAssembly instance with exports
     */
    async instantiateLatest() {
        if (this.modules.length === 0) {
            throw new Error('No modules compiled');
        }

        const moduleInfo = this.modules[this.modules.length - 1];
        return await this.instantiate(moduleInfo.module);
    }

    /**
     * Instantiate a WebAssembly module with the runtime imports
     * @param {WebAssembly.Module} module - Compiled WASM module
     * @returns {Object} - { instance, exports }
     */
    async instantiate(module) {
        // Create import object with runtime functions
        const imports = this.createImports();

        try {
            const instance = await WebAssembly.instantiate(module, imports);
            this.instances.push(instance);

            // Register exports as symbols
            for (const [name, value] of Object.entries(instance.exports)) {
                if (typeof value === 'function') {
                    this.symbols.set(name, {
                        func: value,
                        moduleIndex: this.instances.length - 1
                    });
                }
            }

            return {
                instance: instance,
                exports: instance.exports
            };
        } catch (e) {
            console.error('Instantiation failed:', e);
            throw e;
        }
    }

    /**
     * Create WebAssembly import object with runtime functions
     * @returns {Object} - Import object for WebAssembly.instantiate
     */
    createImports() {
        // Create memory if not exists
        if (!this.memory) {
            this.memory = new WebAssembly.Memory({ initial: 256, maximum: 4096 });
        }

        return {
            env: {
                // Memory
                __linear_memory: this.memory,
                __stack_pointer: new WebAssembly.Global({ value: 'i32', mutable: true }, 65536),
                __memory_base: 0,

                // Console I/O
                printf: (fmt, ...args) => {
                    console.log('printf called:', fmt, args);
                    return 0;
                },

                // Arena memory management (stubs for now)
                arena_create: (size) => {
                    console.log('arena_create:', size);
                    return 0;
                },
                arena_destroy: (arena) => {
                    console.log('arena_destroy:', arena);
                },
                arena_allocate: (arena, size) => {
                    console.log('arena_allocate:', arena, size);
                    return 0;
                },
                arena_allocate_with_header: (arena, size, type, subtype) => 0,
                arena_allocate_cons_cell: (arena) => 0,
                arena_allocate_tagged_cons_cell: (arena) => 0,
                arena_allocate_closure: (arena, a, b, c, d) => 0,
                arena_allocate_vector_with_header: (arena, size) => 0,
                arena_allocate_tensor_with_header: (arena) => 0,
                arena_allocate_tensor_full: (arena, a, b) => 0,
                arena_allocate_ad_node: (arena) => 0,
                arena_allocate_ad_node_with_header: (arena) => 0,
                arena_allocate_cons_with_header: (arena) => 0,
                arena_allocate_string_with_header: (arena, size) => 0,
                arena_allocate_closure_with_header: (arena, a, b, c, d) => 0,
                arena_allocate_tape: (arena, size) => 0,
                arena_hash_table_create: (arena) => 0,
                arena_hash_table_create_with_header: (arena) => 0,

                // Tagged cons operations
                arena_tagged_cons_get_int64: (cell, iscar) => 0n,
                arena_tagged_cons_get_double: (cell, iscar) => 0.0,
                arena_tagged_cons_get_ptr: (cell, iscar) => 0n,
                arena_tagged_cons_get_type: (cell, iscar) => 0,
                arena_tagged_cons_get_flags: (cell, iscar) => 0,
                arena_tagged_cons_get_tagged_value: (cell, iscar) => {},
                arena_tagged_cons_set_int64: (cell, iscar, val, type) => {},
                arena_tagged_cons_set_double: (cell, iscar, val, type) => {},
                arena_tagged_cons_set_ptr: (cell, iscar, val, type) => {},
                arena_tagged_cons_set_null: (cell, iscar) => {},
                arena_tagged_cons_set_tagged_value: (cell, iscar, val) => {},

                // Tape operations (autodiff)
                arena_tape_add_node: (tape, node) => 0n,
                arena_tape_reset: (tape) => {},
                arena_tape_get_node: (tape, idx) => 0,
                arena_tape_get_node_count: (tape) => 0n,

                // Hash table
                hash_table_set: (ht, arena, key, val) => false,
                hash_table_get: (ht, key, out) => false,
                hash_table_has_key: (ht, key) => false,
                hash_table_remove: (ht, key) => false,
                hash_table_keys: (ht, arena) => 0,
                hash_table_values: (ht, arena) => 0,
                hash_table_count: (ht) => 0n,
                hash_table_clear: (ht) => {},

                // Exception handling
                eshkol_make_exception_with_header: (code, msg) => 0,
                eshkol_deep_equal: (a, b) => false,
                eshkol_display_value: (val) => {},

                // Lambda registry
                eshkol_lambda_registry_init: () => {},
                eshkol_lambda_registry_add: (a, b, c) => {},
                eshkol_lambda_registry_lookup: (a) => 0n,

                // Math functions
                sin: Math.sin,
                cos: Math.cos,
                tan: Math.tan,
                asin: Math.asin,
                acos: Math.acos,
                atan: Math.atan,
                atan2: Math.atan2,
                sinh: Math.sinh,
                cosh: Math.cosh,
                tanh: Math.tanh,
                asinh: Math.asinh,
                acosh: Math.acosh,
                atanh: Math.atanh,
                exp: Math.exp,
                exp2: (x) => Math.pow(2, x),
                log: Math.log,
                log10: Math.log10,
                log2: Math.log2,
                pow: Math.pow,
                sqrt: Math.sqrt,
                cbrt: Math.cbrt,
                fabs: Math.abs,
                floor: Math.floor,
                ceil: Math.ceil,
                round: Math.round,
                trunc: Math.trunc,
                fmod: (x, y) => x % y,
                remainder: (x, y) => x - Math.round(x / y) * y,
                fmin: Math.min,
                fmax: Math.max,

                // System functions (stubs)
                exit: (code) => { throw new Error(`exit(${code})`); },
                fopen: () => 0,
                fclose: () => 0,
                fgets: () => 0,
                feof: () => 1,
                fputs: () => 0,
                fputc: () => 0,
                strlen: () => 0n,
                drand48: Math.random,
                srand48: () => {},
                time: () => BigInt(Math.floor(Date.now() / 1000)),
                getenv: () => 0,
                setenv: () => 0,
                unsetenv: () => 0,
                system: () => -1,
                usleep: () => 0,
                access: () => -1,
                remove: () => -1,
                rename: () => -1,
                mkdir: () => -1,
                rmdir: () => -1,
                getcwd: () => 0,
                chdir: () => -1,
                stat: () => -1,
                opendir: () => 0,
                readdir: () => 0,
                closedir: () => 0,
                fseek: () => -1,
                ftell: () => -1n,
                fread: () => 0n,
                fwrite: () => 0n,

                // QRNG (use regular random)
                eshkol_qrng_double: Math.random,
                eshkol_qrng_uint64: () => BigInt(Math.floor(Math.random() * Number.MAX_SAFE_INTEGER)),
                eshkol_qrng_range: (min, max) => BigInt(Math.floor(Math.random() * Number(max - min)) + Number(min)),

                // ============================================
                // DOM API - Make Eshkol a Web Language
                // ============================================

                // Get special handles
                web_get_document: () => this.documentHandle,
                web_get_window: () => this.windowHandle,
                web_get_body: () => this.bodyHandle,

                // Document methods
                web_create_element: (tagPtr) => {
                    const tag = this.readString(tagPtr);
                    const el = document.createElement(tag);
                    return this.createHandle(el);
                },
                web_create_text_node: (textPtr) => {
                    const text = this.readString(textPtr);
                    const node = document.createTextNode(text);
                    return this.createHandle(node);
                },
                web_get_element_by_id: (idPtr) => {
                    const id = this.readString(idPtr);
                    const el = document.getElementById(id);
                    return this.createHandle(el);
                },
                web_query_selector: (selectorPtr) => {
                    const selector = this.readString(selectorPtr);
                    const el = document.querySelector(selector);
                    return this.createHandle(el);
                },
                web_query_selector_all: (selectorPtr) => {
                    const selector = this.readString(selectorPtr);
                    const els = document.querySelectorAll(selector);
                    // Return handle to NodeList (can iterate with web_nodelist_*)
                    return this.createHandle(els);
                },

                // Element methods
                web_append_child: (parentHandle, childHandle) => {
                    const parent = this.getHandle(parentHandle);
                    const child = this.getHandle(childHandle);
                    if (parent && child) {
                        parent.appendChild(child);
                        return 1;
                    }
                    return 0;
                },
                web_remove_child: (parentHandle, childHandle) => {
                    const parent = this.getHandle(parentHandle);
                    const child = this.getHandle(childHandle);
                    if (parent && child) {
                        parent.removeChild(child);
                        return 1;
                    }
                    return 0;
                },
                web_insert_before: (parentHandle, newNodeHandle, refNodeHandle) => {
                    const parent = this.getHandle(parentHandle);
                    const newNode = this.getHandle(newNodeHandle);
                    const refNode = this.getHandle(refNodeHandle);
                    if (parent && newNode) {
                        parent.insertBefore(newNode, refNode);
                        return 1;
                    }
                    return 0;
                },
                web_replace_child: (parentHandle, newChildHandle, oldChildHandle) => {
                    const parent = this.getHandle(parentHandle);
                    const newChild = this.getHandle(newChildHandle);
                    const oldChild = this.getHandle(oldChildHandle);
                    if (parent && newChild && oldChild) {
                        parent.replaceChild(newChild, oldChild);
                        return 1;
                    }
                    return 0;
                },
                web_clone_node: (nodeHandle, deep) => {
                    const node = this.getHandle(nodeHandle);
                    if (node) {
                        return this.createHandle(node.cloneNode(!!deep));
                    }
                    return 0;
                },
                web_get_parent: (nodeHandle) => {
                    const node = this.getHandle(nodeHandle);
                    if (node && node.parentNode) {
                        return this.createHandle(node.parentNode);
                    }
                    return 0;
                },
                web_get_first_child: (nodeHandle) => {
                    const node = this.getHandle(nodeHandle);
                    if (node && node.firstChild) {
                        return this.createHandle(node.firstChild);
                    }
                    return 0;
                },
                web_get_last_child: (nodeHandle) => {
                    const node = this.getHandle(nodeHandle);
                    if (node && node.lastChild) {
                        return this.createHandle(node.lastChild);
                    }
                    return 0;
                },
                web_get_next_sibling: (nodeHandle) => {
                    const node = this.getHandle(nodeHandle);
                    if (node && node.nextSibling) {
                        return this.createHandle(node.nextSibling);
                    }
                    return 0;
                },
                web_get_prev_sibling: (nodeHandle) => {
                    const node = this.getHandle(nodeHandle);
                    if (node && node.previousSibling) {
                        return this.createHandle(node.previousSibling);
                    }
                    return 0;
                },
                web_get_children_count: (nodeHandle) => {
                    const node = this.getHandle(nodeHandle);
                    if (node && node.children) {
                        return node.children.length;
                    }
                    return 0;
                },
                web_get_child_at: (nodeHandle, index) => {
                    const node = this.getHandle(nodeHandle);
                    if (node && node.children && index < node.children.length) {
                        return this.createHandle(node.children[index]);
                    }
                    return 0;
                },

                // Attributes
                web_set_attribute: (elHandle, namePtr, valuePtr) => {
                    const el = this.getHandle(elHandle);
                    const name = this.readString(namePtr);
                    const value = this.readString(valuePtr);
                    if (el && el.setAttribute) {
                        el.setAttribute(name, value);
                        return 1;
                    }
                    return 0;
                },
                web_get_attribute: (elHandle, namePtr, bufPtr, bufSize) => {
                    const el = this.getHandle(elHandle);
                    const name = this.readString(namePtr);
                    if (el && el.getAttribute) {
                        const value = el.getAttribute(name) || '';
                        this.writeString(value.slice(0, bufSize - 1), bufPtr);
                        return value.length;
                    }
                    return 0;
                },
                web_remove_attribute: (elHandle, namePtr) => {
                    const el = this.getHandle(elHandle);
                    const name = this.readString(namePtr);
                    if (el && el.removeAttribute) {
                        el.removeAttribute(name);
                        return 1;
                    }
                    return 0;
                },
                web_has_attribute: (elHandle, namePtr) => {
                    const el = this.getHandle(elHandle);
                    const name = this.readString(namePtr);
                    if (el && el.hasAttribute) {
                        return el.hasAttribute(name) ? 1 : 0;
                    }
                    return 0;
                },

                // Inner HTML / Text
                web_set_inner_html: (elHandle, htmlPtr) => {
                    const el = this.getHandle(elHandle);
                    const html = this.readString(htmlPtr);
                    if (el) {
                        el.innerHTML = html;
                        return 1;
                    }
                    return 0;
                },
                web_get_inner_html: (elHandle, bufPtr, bufSize) => {
                    const el = this.getHandle(elHandle);
                    if (el) {
                        const html = el.innerHTML || '';
                        this.writeString(html.slice(0, bufSize - 1), bufPtr);
                        return html.length;
                    }
                    return 0;
                },
                web_set_text_content: (elHandle, textPtr) => {
                    const el = this.getHandle(elHandle);
                    const text = this.readString(textPtr);
                    if (el) {
                        el.textContent = text;
                        return 1;
                    }
                    return 0;
                },
                web_get_text_content: (elHandle, bufPtr, bufSize) => {
                    const el = this.getHandle(elHandle);
                    if (el) {
                        const text = el.textContent || '';
                        this.writeString(text.slice(0, bufSize - 1), bufPtr);
                        return text.length;
                    }
                    return 0;
                },

                // CSS Classes
                web_add_class: (elHandle, classPtr) => {
                    const el = this.getHandle(elHandle);
                    const cls = this.readString(classPtr);
                    if (el && el.classList) {
                        el.classList.add(cls);
                        return 1;
                    }
                    return 0;
                },
                web_remove_class: (elHandle, classPtr) => {
                    const el = this.getHandle(elHandle);
                    const cls = this.readString(classPtr);
                    if (el && el.classList) {
                        el.classList.remove(cls);
                        return 1;
                    }
                    return 0;
                },
                web_toggle_class: (elHandle, classPtr) => {
                    const el = this.getHandle(elHandle);
                    const cls = this.readString(classPtr);
                    if (el && el.classList) {
                        return el.classList.toggle(cls) ? 1 : 0;
                    }
                    return 0;
                },
                web_has_class: (elHandle, classPtr) => {
                    const el = this.getHandle(elHandle);
                    const cls = this.readString(classPtr);
                    if (el && el.classList) {
                        return el.classList.contains(cls) ? 1 : 0;
                    }
                    return 0;
                },

                // Inline Styles
                web_set_style: (elHandle, propPtr, valuePtr) => {
                    const el = this.getHandle(elHandle);
                    const prop = this.readString(propPtr);
                    const value = this.readString(valuePtr);
                    if (el && el.style) {
                        el.style[prop] = value;
                        return 1;
                    }
                    return 0;
                },
                web_get_style: (elHandle, propPtr, bufPtr, bufSize) => {
                    const el = this.getHandle(elHandle);
                    const prop = this.readString(propPtr);
                    if (el && el.style) {
                        const value = el.style[prop] || '';
                        this.writeString(value.slice(0, bufSize - 1), bufPtr);
                        return value.length;
                    }
                    return 0;
                },

                // Form elements
                web_get_value: (elHandle, bufPtr, bufSize) => {
                    const el = this.getHandle(elHandle);
                    if (el && 'value' in el) {
                        const value = el.value || '';
                        this.writeString(value.slice(0, bufSize - 1), bufPtr);
                        return value.length;
                    }
                    return 0;
                },
                web_set_value: (elHandle, valuePtr) => {
                    const el = this.getHandle(elHandle);
                    const value = this.readString(valuePtr);
                    if (el && 'value' in el) {
                        el.value = value;
                        return 1;
                    }
                    return 0;
                },
                web_get_checked: (elHandle) => {
                    const el = this.getHandle(elHandle);
                    if (el && 'checked' in el) {
                        return el.checked ? 1 : 0;
                    }
                    return 0;
                },
                web_set_checked: (elHandle, checked) => {
                    const el = this.getHandle(elHandle);
                    if (el && 'checked' in el) {
                        el.checked = !!checked;
                        return 1;
                    }
                    return 0;
                },

                // Focus
                web_focus: (elHandle) => {
                    const el = this.getHandle(elHandle);
                    if (el && el.focus) {
                        el.focus();
                        return 1;
                    }
                    return 0;
                },
                web_blur: (elHandle) => {
                    const el = this.getHandle(elHandle);
                    if (el && el.blur) {
                        el.blur();
                        return 1;
                    }
                    return 0;
                },

                // Events
                web_add_event_listener: (elHandle, eventPtr, callbackFuncPtr) => {
                    const el = this.getHandle(elHandle);
                    const event = this.readString(eventPtr);
                    if (el && el.addEventListener && callbackFuncPtr) {
                        const callbackId = this.nextCallbackId++;
                        const callback = (e) => {
                            // Store event data for access from WASM
                            const eventHandle = this.createHandle(e);
                            try {
                                // Call the WASM function
                                const fn = this.instances[this.instances.length - 1]?.exports;
                                if (fn && fn.__indirect_function_table) {
                                    fn.__indirect_function_table.get(callbackFuncPtr)(eventHandle);
                                }
                            } finally {
                                this.releaseHandle(eventHandle);
                            }
                        };
                        el.addEventListener(event, callback);
                        this.callbacks.set(callbackId, { el, event, callback });
                        return callbackId;
                    }
                    return 0;
                },
                web_remove_event_listener: (callbackId) => {
                    const entry = this.callbacks.get(callbackId);
                    if (entry) {
                        entry.el.removeEventListener(entry.event, entry.callback);
                        this.callbacks.delete(callbackId);
                        return 1;
                    }
                    return 0;
                },

                // Event data access
                web_event_prevent_default: (eventHandle) => {
                    const e = this.getHandle(eventHandle);
                    if (e && e.preventDefault) {
                        e.preventDefault();
                        return 1;
                    }
                    return 0;
                },
                web_event_stop_propagation: (eventHandle) => {
                    const e = this.getHandle(eventHandle);
                    if (e && e.stopPropagation) {
                        e.stopPropagation();
                        return 1;
                    }
                    return 0;
                },
                web_event_get_target: (eventHandle) => {
                    const e = this.getHandle(eventHandle);
                    if (e && e.target) {
                        return this.createHandle(e.target);
                    }
                    return 0;
                },
                web_event_get_key: (eventHandle, bufPtr, bufSize) => {
                    const e = this.getHandle(eventHandle);
                    if (e && e.key) {
                        this.writeString(e.key.slice(0, bufSize - 1), bufPtr);
                        return e.key.length;
                    }
                    return 0;
                },
                web_event_get_key_code: (eventHandle) => {
                    const e = this.getHandle(eventHandle);
                    if (e) return e.keyCode || 0;
                    return 0;
                },
                web_event_get_mouse_x: (eventHandle) => {
                    const e = this.getHandle(eventHandle);
                    if (e) return e.clientX || 0;
                    return 0;
                },
                web_event_get_mouse_y: (eventHandle) => {
                    const e = this.getHandle(eventHandle);
                    if (e) return e.clientY || 0;
                    return 0;
                },

                // Timers
                web_set_timeout: (callbackFuncPtr, delayMs) => {
                    const id = setTimeout(() => {
                        try {
                            const fn = this.instances[this.instances.length - 1]?.exports;
                            if (fn && fn.__indirect_function_table) {
                                fn.__indirect_function_table.get(callbackFuncPtr)();
                            }
                        } catch (e) {
                            console.error('Timeout callback error:', e);
                        }
                    }, delayMs);
                    return id;
                },
                web_set_interval: (callbackFuncPtr, delayMs) => {
                    const id = setInterval(() => {
                        try {
                            const fn = this.instances[this.instances.length - 1]?.exports;
                            if (fn && fn.__indirect_function_table) {
                                fn.__indirect_function_table.get(callbackFuncPtr)();
                            }
                        } catch (e) {
                            console.error('Interval callback error:', e);
                        }
                    }, delayMs);
                    return id;
                },
                web_clear_timeout: (id) => {
                    clearTimeout(id);
                },
                web_clear_interval: (id) => {
                    clearInterval(id);
                },
                web_request_animation_frame: (callbackFuncPtr) => {
                    return requestAnimationFrame((timestamp) => {
                        try {
                            const fn = this.instances[this.instances.length - 1]?.exports;
                            if (fn && fn.__indirect_function_table) {
                                fn.__indirect_function_table.get(callbackFuncPtr)(timestamp);
                            }
                        } catch (e) {
                            console.error('RAF callback error:', e);
                        }
                    });
                },
                web_cancel_animation_frame: (id) => {
                    cancelAnimationFrame(id);
                },

                // Console
                web_console_log: (msgPtr) => {
                    console.log(this.readString(msgPtr));
                },
                web_console_warn: (msgPtr) => {
                    console.warn(this.readString(msgPtr));
                },
                web_console_error: (msgPtr) => {
                    console.error(this.readString(msgPtr));
                },

                // Window
                web_alert: (msgPtr) => {
                    alert(this.readString(msgPtr));
                },
                web_confirm: (msgPtr) => {
                    return confirm(this.readString(msgPtr)) ? 1 : 0;
                },
                web_prompt: (msgPtr, defaultPtr, bufPtr, bufSize) => {
                    const msg = this.readString(msgPtr);
                    const def = this.readString(defaultPtr);
                    const result = prompt(msg, def) || '';
                    this.writeString(result.slice(0, bufSize - 1), bufPtr);
                    return result.length;
                },
                web_get_window_width: () => window.innerWidth,
                web_get_window_height: () => window.innerHeight,
                web_get_scroll_x: () => window.scrollX,
                web_get_scroll_y: () => window.scrollY,
                web_scroll_to: (x, y) => {
                    window.scrollTo(x, y);
                },

                // Location
                web_get_href: (bufPtr, bufSize) => {
                    const href = window.location.href;
                    this.writeString(href.slice(0, bufSize - 1), bufPtr);
                    return href.length;
                },
                web_set_href: (urlPtr) => {
                    window.location.href = this.readString(urlPtr);
                },
                web_get_hash: (bufPtr, bufSize) => {
                    const hash = window.location.hash;
                    this.writeString(hash.slice(0, bufSize - 1), bufPtr);
                    return hash.length;
                },
                web_set_hash: (hashPtr) => {
                    window.location.hash = this.readString(hashPtr);
                },

                // Local Storage
                web_storage_get: (keyPtr, bufPtr, bufSize) => {
                    const key = this.readString(keyPtr);
                    const value = localStorage.getItem(key) || '';
                    this.writeString(value.slice(0, bufSize - 1), bufPtr);
                    return value.length;
                },
                web_storage_set: (keyPtr, valuePtr) => {
                    const key = this.readString(keyPtr);
                    const value = this.readString(valuePtr);
                    localStorage.setItem(key, value);
                    return 1;
                },
                web_storage_remove: (keyPtr) => {
                    const key = this.readString(keyPtr);
                    localStorage.removeItem(key);
                    return 1;
                },
                web_storage_clear: () => {
                    localStorage.clear();
                    return 1;
                },

                // Fetch API (async - returns promise handle)
                web_fetch: (urlPtr, methodPtr, bodyPtr) => {
                    const url = this.readString(urlPtr);
                    const method = this.readString(methodPtr) || 'GET';
                    const body = bodyPtr ? this.readString(bodyPtr) : null;

                    const promise = fetch(url, {
                        method,
                        body: method !== 'GET' ? body : undefined,
                        headers: body ? { 'Content-Type': 'application/json' } : {}
                    });
                    return this.createHandle(promise);
                },

                // Canvas 2D API
                web_get_context_2d: (canvasHandle) => {
                    const canvas = this.getHandle(canvasHandle);
                    if (canvas && canvas.getContext) {
                        const ctx = canvas.getContext('2d');
                        return this.createHandle(ctx);
                    }
                    return 0;
                },
                web_canvas_fill_rect: (ctxHandle, x, y, w, h) => {
                    const ctx = this.getHandle(ctxHandle);
                    if (ctx) ctx.fillRect(x, y, w, h);
                },
                web_canvas_stroke_rect: (ctxHandle, x, y, w, h) => {
                    const ctx = this.getHandle(ctxHandle);
                    if (ctx) ctx.strokeRect(x, y, w, h);
                },
                web_canvas_clear_rect: (ctxHandle, x, y, w, h) => {
                    const ctx = this.getHandle(ctxHandle);
                    if (ctx) ctx.clearRect(x, y, w, h);
                },
                web_canvas_fill_style: (ctxHandle, colorPtr) => {
                    const ctx = this.getHandle(ctxHandle);
                    const color = this.readString(colorPtr);
                    if (ctx) ctx.fillStyle = color;
                },
                web_canvas_stroke_style: (ctxHandle, colorPtr) => {
                    const ctx = this.getHandle(ctxHandle);
                    const color = this.readString(colorPtr);
                    if (ctx) ctx.strokeStyle = color;
                },
                web_canvas_line_width: (ctxHandle, width) => {
                    const ctx = this.getHandle(ctxHandle);
                    if (ctx) ctx.lineWidth = width;
                },
                web_canvas_begin_path: (ctxHandle) => {
                    const ctx = this.getHandle(ctxHandle);
                    if (ctx) ctx.beginPath();
                },
                web_canvas_close_path: (ctxHandle) => {
                    const ctx = this.getHandle(ctxHandle);
                    if (ctx) ctx.closePath();
                },
                web_canvas_move_to: (ctxHandle, x, y) => {
                    const ctx = this.getHandle(ctxHandle);
                    if (ctx) ctx.moveTo(x, y);
                },
                web_canvas_line_to: (ctxHandle, x, y) => {
                    const ctx = this.getHandle(ctxHandle);
                    if (ctx) ctx.lineTo(x, y);
                },
                web_canvas_arc: (ctxHandle, x, y, r, start, end) => {
                    const ctx = this.getHandle(ctxHandle);
                    if (ctx) ctx.arc(x, y, r, start, end);
                },
                web_canvas_fill: (ctxHandle) => {
                    const ctx = this.getHandle(ctxHandle);
                    if (ctx) ctx.fill();
                },
                web_canvas_stroke: (ctxHandle) => {
                    const ctx = this.getHandle(ctxHandle);
                    if (ctx) ctx.stroke();
                },
                web_canvas_fill_text: (ctxHandle, textPtr, x, y) => {
                    const ctx = this.getHandle(ctxHandle);
                    const text = this.readString(textPtr);
                    if (ctx) ctx.fillText(text, x, y);
                },
                web_canvas_font: (ctxHandle, fontPtr) => {
                    const ctx = this.getHandle(ctxHandle);
                    const font = this.readString(fontPtr);
                    if (ctx) ctx.font = font;
                },
                web_canvas_save: (ctxHandle) => {
                    const ctx = this.getHandle(ctxHandle);
                    if (ctx) ctx.save();
                },
                web_canvas_restore: (ctxHandle) => {
                    const ctx = this.getHandle(ctxHandle);
                    if (ctx) ctx.restore();
                },
                web_canvas_translate: (ctxHandle, x, y) => {
                    const ctx = this.getHandle(ctxHandle);
                    if (ctx) ctx.translate(x, y);
                },
                web_canvas_rotate: (ctxHandle, angle) => {
                    const ctx = this.getHandle(ctxHandle);
                    if (ctx) ctx.rotate(angle);
                },
                web_canvas_scale: (ctxHandle, x, y) => {
                    const ctx = this.getHandle(ctxHandle);
                    if (ctx) ctx.scale(x, y);
                },

                // Handle management
                web_release_handle: (handle) => {
                    this.releaseHandle(handle);
                },

                // Provide previously defined symbols
                ...this.getSymbolImports()
            },

            // Global variables
            'GOT.mem': {
                __global_arena: new WebAssembly.Global({ value: 'i32', mutable: false }, 0),
                square_sexpr: new WebAssembly.Global({ value: 'i32', mutable: false }, 0)
            },

            'GOT.func': {
                square: new WebAssembly.Global({ value: 'i32', mutable: false }, 0)
            }
        };
    }

    /**
     * Get imports for previously defined symbols
     * @returns {Object} - Symbol imports
     */
    getSymbolImports() {
        const imports = {};
        for (const [name, info] of this.symbols) {
            imports[name] = info.func;
        }
        return imports;
    }

    /**
     * Call a function exported by a WASM module
     * @param {string} name - Function name
     * @param  {...any} args - Function arguments
     * @returns {any} - Function result
     */
    call(name, ...args) {
        const symbol = this.symbols.get(name);
        if (!symbol) {
            throw new Error(`Symbol not found: ${name}`);
        }
        return symbol.func(...args);
    }

    /**
     * List all available symbols
     * @returns {string[]} - Symbol names
     */
    listSymbols() {
        return Array.from(this.symbols.keys());
    }

    /**
     * Convert base64 string to ArrayBuffer
     * @param {string} base64 - Base64 encoded string
     * @returns {ArrayBuffer} - Decoded binary data
     */
    base64ToArrayBuffer(base64) {
        const binaryString = atob(base64);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        return bytes.buffer;
    }

    /**
     * Reset the REPL state
     */
    reset() {
        this.sessionId = null;
        this.modules = [];
        this.instances = [];
        this.symbols.clear();
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EshkolRepl;
}
