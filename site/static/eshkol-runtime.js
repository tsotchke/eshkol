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

    // Bump allocator for arena stubs
    _bump(size) {
        if (!this._bumpPtr) this._bumpPtr = 131072; // Start at 128KB
        const ptr = this._bumpPtr;
        this._bumpPtr += ((size + 7) & ~7); // 8-byte aligned
        return ptr;
    }

    setInstance(instance) {
        this.instance = instance;
        this.memory = instance.exports.memory || this._importedMemory;
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
        const mem = this.memory || this._importedMemory;
        if (!ptr || !mem) return '';
        const bytes = new Uint8Array(mem.buffer);
        let end = ptr;
        while (end < bytes.length && bytes[end] !== 0) end++;
        return new TextDecoder().decode(bytes.slice(ptr, end));
    }

    writeString(ptr, str, maxLen) {
        const mem = this.memory || this._importedMemory;
        if (!ptr || !mem) return 0;
        const bytes = new TextEncoder().encode(str);
        const view = new Uint8Array(mem.buffer);
        const len = Math.min(bytes.length, maxLen - 1);
        view.set(bytes.subarray(0, len), ptr);
        view[ptr + len] = 0;
        return len;
    }

    // === Markdown Renderer ===
    // Lightweight markdown-to-HTML converter (no external dependencies)

    // Single-pass Scheme/Eshkol tokenizer used by the markdown highlighter.
    //
    // History note: an earlier version did syntax highlighting by chaining
    // .replace() calls — first wrapping comments in a span, then strings,
    // then numbers, etc. That was self-corrupting because each later regex
    // happily matched substrings the earlier passes had ALREADY inserted
    // into the output. The string regex matched the literal "color:#606078"
    // attribute the comment pass had just added, and the number regex then
    // matched "606078" inside *that* attribute, producing
    //
    //     <span style=<span ...>"color:#<span ...>606078</span>"</span>>...
    //
    // which the browser then displayed as raw text, leaving things like
    //     "color:#606078">; Integers are exact
    // visible in the rendered docs.
    //
    // The fix is to walk the source ONCE, recognise each token (comment,
    // string, number, hash literal, keyword, identifier, other), and emit
    // its HTML in one shot. Once a region of the source has been emitted
    // as a span, it cannot be re-tokenised, so the cross-corruption is
    // architecturally impossible.
    //
    // Returns an HTML string with all `<`, `>`, `&` properly escaped.
    static highlightScheme(source) {
        const KEYWORDS = new Set([
            'define','lambda','let','let*','letrec','letrec*','if','cond','begin',
            'set!','when','unless','do','case','match','and','or','not','quote',
            'quasiquote','unquote','unquote-splicing','require','provide','extern',
            'gradient','derivative','jacobian','hessian','divergence','curl','laplacian',
        ]);
        const COL_COMMENT = '#606078';
        const COL_KEYWORD = '#c084fc';
        const COL_STRING  = '#a78bfa';
        const COL_NUMBER  = '#00ff88';
        const escape = (s) =>
            s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        const span = (color, text) =>
            `<span style="color:${color}">${escape(text)}</span>`;
        const isDigit  = (c) => c >= '0' && c <= '9';
        const isIdent  = (c) =>
            (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') ||
            '!$%&*+-./:<=>?@^_~'.indexOf(c) !== -1;
        const isIdentStart = (c) =>
            (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            '!$%&*+-./:<=>?@^_~'.indexOf(c) !== -1;
        let out = '';
        let i = 0;
        while (i < source.length) {
            const c = source[i];

            // Line comment: ; ... \n
            if (c === ';') {
                let j = i;
                while (j < source.length && source[j] !== '\n') j++;
                out += span(COL_COMMENT, source.slice(i, j));
                i = j;
                continue;
            }

            // String literal: "...", honoring \" escapes
            if (c === '"') {
                let j = i + 1;
                while (j < source.length) {
                    if (source[j] === '\\' && j + 1 < source.length) { j += 2; continue; }
                    if (source[j] === '"') { j++; break; }
                    j++;
                }
                out += span(COL_STRING, source.slice(i, j));
                i = j;
                continue;
            }

            // Hash literal: #t, #f, #\char
            if (c === '#' && i + 1 < source.length) {
                const n = source[i + 1];
                if (n === 't' || n === 'f') {
                    out += span(COL_NUMBER, source.slice(i, i + 2));
                    i += 2;
                    continue;
                }
                if (n === '\\' && i + 2 < source.length) {
                    out += span(COL_NUMBER, source.slice(i, i + 3));
                    i += 3;
                    continue;
                }
            }

            // Number literal: digits with optional fraction
            // Treats a leading '-' or '+' as numeric only when followed by a digit
            // and the previous emitted character is whitespace or an opening paren,
            // so things like (- x 1) keep the '-' as an identifier/operator.
            if (isDigit(c) || ((c === '-' || c === '+') && i + 1 < source.length && isDigit(source[i + 1]))) {
                let j = i + 1;
                let sawDot = false;
                while (j < source.length) {
                    const cj = source[j];
                    if (isDigit(cj)) { j++; continue; }
                    if (cj === '.' && !sawDot) { sawDot = true; j++; continue; }
                    break;
                }
                out += span(COL_NUMBER, source.slice(i, j));
                i = j;
                continue;
            }

            // Identifier or keyword
            if (isIdentStart(c)) {
                let j = i + 1;
                while (j < source.length && isIdent(source[j])) j++;
                const word = source.slice(i, j);
                if (KEYWORDS.has(word)) {
                    out += span(COL_KEYWORD, word);
                } else {
                    out += escape(word);
                }
                i = j;
                continue;
            }

            // Anything else (parens, whitespace, punctuation) — just escape
            out += escape(c);
            i++;
        }
        return out;
    }

    renderMarkdown(md) {
        let html = md;
        // Extract code blocks FIRST to protect them from markdown transforms
        const codeBlocks = [];
        const SCHEME_LANGS = new Set(['scheme', 'eshkol', 'lisp', 'scm', '']);
        html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
            let body;
            if (SCHEME_LANGS.has(lang)) {
                // Single-pass tokeniser handles its own escaping.
                body = EshkolRuntime.highlightScheme(code);
            } else {
                // Other languages — just escape, no highlighting.
                body = code.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            }
            const block = `<pre style="background:#0a0a14;border:1px solid #27272a;border-radius:8px;padding:16px 20px;overflow-x:auto;margin:1em 0"><code style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;line-height:1.6;color:#d4d4d8">${body}</code></pre>`;
            codeBlocks.push(block);
            return `\x00CODEBLOCK${codeBlocks.length - 1}\x00`;
        });
        // Inline code
        html = html.replace(/`([^`]+)`/g, '<code style="background:#0d0d15;padding:2px 6px;border-radius:4px;font-size:0.85em;color:#a78bfa">$1</code>');
        // Headers
        html = html.replace(/^######\s+(.+)$/gm, '<h6 style="margin-top:1.5em;margin-bottom:0.3em;font-size:0.85rem;color:#a0a0b8">$1</h6>');
        html = html.replace(/^#####\s+(.+)$/gm, '<h5 style="margin-top:1.5em;margin-bottom:0.3em;font-size:0.9rem;color:#a0a0b8">$1</h5>');
        html = html.replace(/^####\s+(.+)$/gm, '<h4 style="margin-top:1.8em;margin-bottom:0.4em;font-size:1rem;color:#a0a0b8">$1</h4>');
        html = html.replace(/^###\s+(.+)$/gm, '<h3 style="margin-top:2em;margin-bottom:0.5em;font-size:1.2rem;color:#e8e8f0">$1</h3>');
        html = html.replace(/^##\s+(.+)$/gm, '<h2 style="margin-top:2.5em;margin-bottom:0.5em;font-size:1.5rem;color:#e8e8f0;border-bottom:1px solid #2a2a3a;padding-bottom:0.3em">$1</h2>');
        html = html.replace(/^#\s+(.+)$/gm, '<h1 style="margin-top:2em;margin-bottom:0.5em;font-size:2rem;color:#e8e8f0;border-bottom:1px solid #2a2a3a;padding-bottom:0.3em">$1</h1>');
        // Bold and italic
        html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
        // Links
        html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" style="color:#a78bfa">$1</a>');
        // Images (just show as links)
        html = html.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<em>[$1]</em>');
        // Blockquotes
        html = html.replace(/^>\s+(.+)$/gm, '<blockquote style="border-left:3px solid #7c3aed;padding-left:16px;margin:1em 0;color:#a0a0b8">$1</blockquote>');
        // Horizontal rules
        html = html.replace(/^---+$/gm, '<hr style="border:none;border-top:1px solid #2a2a3a;margin:2em 0">');
        // Tables
        html = html.replace(/^\|(.+)\|$/gm, (line) => {
            const cells = line.split('|').filter(c => c.trim());
            if (cells.every(c => /^[-:\s]+$/.test(c))) return ''; // separator row
            const tag = 'td';
            const cellHtml = cells.map(c => `<${tag} style="border:1px solid #2a2a3a;padding:8px 12px;color:#a0a0b8">${c.trim()}</${tag}>`).join('');
            return `<tr>${cellHtml}</tr>`;
        });
        html = html.replace(/(<tr>[\s\S]*?<\/tr>\n?)+/g, m => `<table style="width:100%;border-collapse:collapse;margin:1em 0">${m}</table>`);
        // Unordered lists
        html = html.replace(/^[\s]*[-*]\s+(.+)$/gm, '<li style="color:#a0a0b8;margin:0.2em 0">$1</li>');
        html = html.replace(/(<li[\s\S]*?<\/li>\n?)+/g, m => `<ul style="margin:0.8em 0;padding-left:1.5em">${m}</ul>`);
        // Ordered lists
        html = html.replace(/^\d+\.\s+(.+)$/gm, '<li style="color:#a0a0b8;margin:0.2em 0">$1</li>');
        // Paragraphs (lines not already tagged — skip code block placeholders)
        html = html.replace(/^(?!<[huplbtdoa]|<\/|<hr|<code|<str|<em>|\x00|$)(.+)$/gm, '<p style="margin:0.6em 0;color:#a0a0b8;line-height:1.7">$1</p>');
        // Restore code blocks from placeholders
        html = html.replace(/\x00CODEBLOCK(\d+)\x00/g, (_, idx) => codeBlocks[parseInt(idx)]);
        // Clean up double-spaced lines in pre blocks (ASCII art, etc.)
        html = html.replace(/<\/p>\n<p/g, '</p><p');
        return html;
    }

    // === WASM Imports ===

    createImports() {
        const rt = this;
        return {
            env: {
                // Memory
                __linear_memory: rt._importedMemory = new WebAssembly.Memory({ initial: 256, maximum: 1024 }),
                // WASM linker globals (required by LLVM static relocation)
                __stack_pointer: new WebAssembly.Global({ value: 'i32', mutable: true }, 1048576), // 1MB stack
                __indirect_function_table: new WebAssembly.Table({ initial: 256, element: 'anyfunc' }),

                // Bump allocator (all arena functions use this)
                // All params may be BigInt (i64) — convert with Number()
                arena_create: () => 1,
                arena_create_threadsafe: () => 1,
                arena_destroy: () => {},
                arena_allocate: (arena, size) => { return rt._bump(Number(size)); },
                arena_allocate_with_header: (arena, size) => { return rt._bump(Number(size) + 8) + 8; },
                arena_push_scope: () => {},
                arena_pop_scope: () => {},
                arena_allocate_cons_cell: () => rt._bump(32),
                arena_allocate_cons_with_header: () => rt._bump(40) + 8,
                arena_allocate_tagged_cons_cell: () => rt._bump(48),
                eshkol_tagged_cons_set_tagged_value: () => {},
                arena_tagged_cons_set_ptr: () => {},
                arena_tagged_cons_set_null: () => {},
                arena_tagged_cons_set_int64: () => {},
                arena_allocate_tensor_with_header: () => rt._bump(72) + 8,
                arena_allocate_tensor_full: (arena, ndim, total) => rt._bump(32 + Number(total) * 8),
                arena_allocate_vector_with_header: (arena, n) => rt._bump(8 + Number(n) * 16),
                arena_allocate_ad_node: () => rt._bump(128),
                arena_allocate_ad_node_with_header: () => rt._bump(136) + 8,
                arena_allocate_tape: () => rt._bump(64),
                arena_tape_add_node: () => 0,
                arena_tape_reset: () => {},
                arena_tape_get_node: () => 0,
                arena_tape_get_node_count: () => 0,
                arena_allocate_string_with_header: (arena, len) => rt._bump(Number(len) + 9) + 8,
                arena_allocate_closure_with_header: () => rt._bump(64),

                // Runtime functions
                __eshkol_register_parallel_workers: () => {},
                eshkol_init_global_arena: () => {},
                eshkol_init_stack_size: () => {},
                eshkol_check_recursion_depth: () => 0,  // returns i32 (size_t)
                eshkol_decrement_recursion_depth: () => {},
                eshkol_make_exception_with_header: () => 0,
                eshkol_raise: (exc) => { console.error('Eshkol exception raised'); },
                eshkol_display_value: () => {},
                eshkol_deep_equal: () => 0,
                eshkol_lambda_registry_init: () => {},
                eshkol_lambda_registry_add: () => {},
                eshkol_lambda_registry_lookup: () => 0,
                eshkol_bignum_binary_tagged: () => 0,
                eshkol_bignum_compare_tagged: () => 0,
                eshkol_is_bignum_tagged: () => 0,
                eshkol_rational_compare_tagged_ptr: () => 0,
                eshkol_is_rational_tagged_ptr: () => 0,

                // C library
                strcmp: (a, b) => {
                    const mem = new Uint8Array(rt._importedMemory?.buffer || rt.instance?.exports?.memory?.buffer);
                    let i = 0;
                    while (true) {
                        const ca = mem[a + i], cb = mem[b + i];
                        if (ca !== cb) return ca < cb ? -1 : 1;
                        if (ca === 0) return 0;
                        i++;
                    }
                },
                strlen: (s) => {
                    const mem = new Uint8Array(rt._importedMemory?.buffer || rt.instance?.exports?.memory?.buffer);
                    let i = 0;
                    while (mem[s + i] !== 0) i++;
                    return i;
                },
                memcpy: (dst, src, n) => {
                    const mem = new Uint8Array(rt._importedMemory?.buffer || rt.instance?.exports?.memory?.buffer);
                    mem.copyWithin(dst, src, src + Number(n));
                    return dst;
                },
                memset: (dst, val, n) => {
                    const mem = new Uint8Array(rt._importedMemory?.buffer || rt.instance?.exports?.memory?.buffer);
                    mem.fill(val, dst, dst + Number(n));
                    return dst;
                },

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
                    if (el && el.innerHTML !== undefined) el.innerHTML = rt.readString(htmlPtr);
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

                // Location (pathname-based routing — clean URLs, no hash)
                web_get_hash: (bufPtr, bufLen) => {
                    var path = window.location.pathname.replace(/\/$/, '') || '/';
                    return rt.writeString(bufPtr, path, bufLen);
                },

                web_set_hash: (hashPtr) => {
                    var path = rt.readString(hashPtr);
                    history.pushState(null, '', path);
                    return 0;
                },

                web_get_href: (bufPtr, bufLen) => {
                    return rt.writeString(bufPtr, window.location.href, bufLen);
                },

                // REPL evaluation (uses eshkol-vm.wasm — the full bytecode VM)
                web_repl_eval: (sourcePtr, resultHandle) => {
                    const source = rt.readString(sourcePtr);
                    const target = rt.handles.get(resultHandle);
                    if (target && rt._eshkolVM) {
                        try {
                            const evalFn = rt._eshkolVM.cwrap('repl_eval', 'string', ['string']);
                            const result = evalFn(source);
                            target.textContent += result;
                        } catch (e) {
                            target.textContent += 'Error: ' + e.message + '\n';
                        }
                    }
                    return 0;
                },

                // Content loading (fetch URL, render as markdown into target element)
                web_load_content: (urlPtr, targetHandle) => {
                    const url = rt.readString(urlPtr);
                    const target = rt.handles.get(targetHandle);
                    if (target) {
                        target.innerHTML = '<p style="color:#606078;font-style:italic">Loading...</p>';
                        fetch(url).then(r => {
                            if (!r.ok) throw new Error(`HTTP ${r.status}`);
                            return r.text();
                        }).then(text => {
                            // Render markdown to HTML with syntax highlighting
                            target.innerHTML = rt.renderMarkdown(text);
                        }).catch(e => {
                            target.innerHTML = '<p style="color:#ff4444">Failed to load: ' + e.message + '</p>';
                        });
                    }
                    return 0;
                },

                // Window
                web_get_window_width: () => window.innerWidth,
                web_get_window_height: () => window.innerHeight,
                web_get_scroll_y: () => window.scrollY | 0,
                web_scroll_to: (x, y) => { window.scrollTo(Number(x), Number(y)); return 0; },

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

                // Runnable code block — code block + Run ▶ button + output area
                web_create_runnable_code: (rawCodePtr, htmlPtr, parentHandle) => {
                    const rawCode = rt.readString(rawCodePtr);
                    const html = rt.readString(htmlPtr);
                    const parent = rt.getHandle(parentHandle);
                    if (!parent) return 0;

                    const wrapper = document.createElement('div');
                    wrapper.className = 'runnable-code';
                    wrapper.style.cssText = 'background:#0d0d15;border:1px solid #2a2a3a;border-radius:12px;overflow:hidden;margin-bottom:0';

                    // Code display
                    const codeDiv = document.createElement('div');
                    codeDiv.style.cssText = 'padding:16px 20px;overflow-x:auto;overflow-y:visible';
                    codeDiv.innerHTML = html;
                    wrapper.appendChild(codeDiv);

                    // Toolbar
                    const toolbar = document.createElement('div');
                    toolbar.style.cssText = 'display:flex;align-items:center;justify-content:flex-end;padding:6px 16px;border-top:1px solid #1a1a28;background:#0a0a0f';

                    const btn = document.createElement('button');
                    btn.textContent = 'Run \u25b6';
                    btn.dataset.runCode = rawCode;
                    btn.style.cssText = 'background:#7c3aed;color:#fff;border:none;padding:4px 14px;border-radius:5px;font-size:0.78rem;font-weight:600;cursor:pointer;font-family:JetBrains Mono,monospace;letter-spacing:0.5px';
                    btn.addEventListener('mouseenter', () => { btn.style.background = '#6d28d9'; });
                    btn.addEventListener('mouseleave', () => { btn.style.background = '#7c3aed'; });
                    toolbar.appendChild(btn);
                    wrapper.appendChild(toolbar);

                    // Output area (hidden until first run)
                    const output = document.createElement('pre');
                    output.setAttribute('data-run-output', '1');
                    output.style.cssText = 'display:none;margin:0;padding:10px 28px;color:#00ff88;font-family:JetBrains Mono,monospace;font-size:0.82rem;line-height:1.5;background:#050510;border-top:1px solid #1a1a28;white-space:pre-wrap';
                    wrapper.appendChild(output);

                    parent.appendChild(wrapper);
                    return rt.createHandle(wrapper);
                },
            }
        };
    }
}
