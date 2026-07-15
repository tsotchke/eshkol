/* Heap pointer validation — prevent OOB heap access from untrusted values.
 * Used before any vm->heap.objects[val.as.ptr] dereference. */
#define VM_VALIDATE_HEAP(vm, val) \
    (is_valid_heap_ptr((vm), (val).as.ptr) ? \
     (vm)->heap.objects[(val).as.ptr] : \
     (fprintf(stderr, "HEAP ACCESS: invalid ptr %d (max %d)\n", \
              (val).as.ptr, (vm)->heap.next_free), \
      (vm)->error = 1, (HeapObject*)NULL))

/** @brief Allocate a cons pair of two raw int64 values (as INT_VALs). */
static Value vm_int_pair(VM* vm, int64_t car, int64_t cdr) {
    int32_t ptr = heap_alloc(&vm->heap);
    if (ptr < 0) {
        vm->error = 1;
        return NIL_VAL;
    }
    vm->heap.objects[ptr]->type = HEAP_CONS;
    vm->heap.objects[ptr]->cons.car = INT_VAL(car);
    vm->heap.objects[ptr]->cons.cdr = INT_VAL(cdr);
    return PAIR_VAL(ptr);
}

/** @brief Allocate a cons pair from two already-tagged Values. */
static Value vm_cons_value(VM* vm, Value car, Value cdr) {
    int32_t ptr = heap_alloc(&vm->heap);
    if (ptr < 0) {
        vm->error = 1;
        return NIL_VAL;
    }
    vm->heap.objects[ptr]->type = HEAP_CONS;
    vm->heap.objects[ptr]->cons.car = car;
    vm->heap.objects[ptr]->cons.cdr = cdr;
    return PAIR_VAL(ptr);
}

/** @brief Copy a C string (or explicit-length buffer) into a new heap VmString Value. */
static Value vm_string_value(VM* vm, const char* data, int64_t len) {
    if (!data) return NIL_VAL;
    if (len < 0) len = (int64_t)strlen(data);
    VmString* s = vm_string_new(&vm->heap.regions, data, len);
    if (!s) return NIL_VAL;
    int32_t ptr = heap_alloc(&vm->heap);
    if (ptr < 0) {
        vm->error = 1;
        return NIL_VAL;
    }
    vm->heap.objects[ptr]->type = HEAP_STRING;
    vm->heap.objects[ptr]->opaque.ptr = s;
    return (Value){.type = VAL_STRING, .as.ptr = ptr};
}

/** @brief Build a `(key . value)` pair for use as one entry of an association list. */
static Value vm_alist_entry(VM* vm, const char* key, Value value) {
    return vm_cons_value(vm, vm_string_value(vm, key, -1), value);
}

/** @brief Extract a process id from a process handle Value, which may be a raw
 *         number or a `(pid . fd)` pair as used by PTY-backed subprocesses. */
static int64_t vm_process_pid_from_value(VM* vm, Value value) {
    if (value.type == VAL_PAIR && is_valid_heap_ptr(vm, value.as.ptr)) {
        HeapObject* obj = vm->heap.objects[value.as.ptr];
        if (obj && obj->type == HEAP_CONS)
            return (int64_t)as_number(obj->cons.car);
    }
    return (int64_t)as_number(value);
}

/** @brief Extract a file descriptor from a process handle Value: prefers the
 *         `(pid . fd)` pair form, otherwise looks up the fd by pid in the
 *         VM's tracked PTY handle table, falling back to the raw number. */
static int vm_process_fd_from_value(VM* vm, Value value) {
    if (value.type == VAL_PAIR && is_valid_heap_ptr(vm, value.as.ptr)) {
        HeapObject* obj = vm->heap.objects[value.as.ptr];
        if (obj && obj->type == HEAP_CONS)
            return (int)as_number(obj->cons.cdr);
    }

    int64_t pid_or_fd = (int64_t)as_number(value);
    for (int i = 0; i < vm->n_pty_handles; i++) {
        if (vm->pty_handles[i].pid == pid_or_fd)
            return vm->pty_handles[i].fd;
    }
    return (int)pid_or_fd;
}

/** @brief Record (or update) the fd associated with a spawned PTY process's pid
 *         in the VM's fixed-size pty_handles table, so it can later be resolved
 *         back to a file descriptor by vm_process_fd_from_value(). */
static void vm_process_track_pty(VM* vm, int64_t pid, int fd) {
    if (!vm || pid <= 0 || fd < 0) return;
    for (int i = 0; i < vm->n_pty_handles; i++) {
        if (vm->pty_handles[i].pid == pid) {
            vm->pty_handles[i].fd = fd;
            return;
        }
    }
    if (vm->n_pty_handles < (int)(sizeof(vm->pty_handles) / sizeof(vm->pty_handles[0]))) {
        vm->pty_handles[vm->n_pty_handles].pid = pid;
        vm->pty_handles[vm->n_pty_handles].fd = fd;
        vm->n_pty_handles++;
    }
}

/** @brief Remove a pid's entry from the PTY handle table (swap-with-last), optionally
 *         closing its fd first. No-op on POSIX-less builds beyond bookkeeping. */
static void vm_process_forget_pty(VM* vm, int64_t pid, int close_fd) {
    if (!vm || pid <= 0) return;
    for (int i = 0; i < vm->n_pty_handles; i++) {
        if (vm->pty_handles[i].pid == pid) {
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
            if (close_fd && vm->pty_handles[i].fd >= 0) close(vm->pty_handles[i].fd);
#else
            (void)close_fd;
#endif
            vm->pty_handles[i] = vm->pty_handles[vm->n_pty_handles - 1];
            vm->n_pty_handles--;
            return;
        }
    }
}

/** @brief Safely unwrap a heap-backed string or symbol spelling.
 *
 * Symbols and strings have distinct Scheme value tags, but intentionally
 * share the immutable VmString byte representation.  Existing hosted APIs
 * that historically accepted the VM's old string-as-symbol representation
 * therefore remain source-compatible while `string?`, `symbol?`, and
 * `write` can now distinguish the language types correctly.
 */
static VmString* vm_value_as_string(VM* vm, Value value) {
    if (!vm || (value.type != VAL_STRING && value.type != VAL_SYMBOL) ||
        !is_valid_heap_ptr(vm, value.as.ptr))
        return NULL;
    HeapObject* obj = vm->heap.objects[value.as.ptr];
    if (!obj || obj->type != HEAP_STRING)
        return NULL;
    return (VmString*)obj->opaque.ptr;
}

/** Return a UTF-8 string view for either a Scheme string or character.
 * Character-accepting string helpers use this instead of collapsing VAL_CHAR
 * to NULL through vm_value_as_string().  The one-codepoint string is arena
 * owned, matching every other VM string temporary. */
static VmString* vm_value_as_string_or_char(VM* vm, Value value) {
    VmString* string_value = vm_value_as_string(vm, value);
    if (string_value || value.type != VAL_CHAR) return string_value;
    char utf8[4];
    int byte_len = vm_utf8_encode((int)value.as.i, utf8);
    return vm_string_new(&vm->heap.regions, utf8, byte_len);
}

/** @brief Safely unwrap a VAL_PORT Value to its underlying VmPort*, resolving the
 *         well-known stdin/stdout/stderr opaque pointers to the live singleton
 *         port objects (re-initializing them if needed). Returns NULL on any
 *         type/heap mismatch. */
static VmPort* vm_value_as_port(VM* vm, Value value) {
    if (!vm || value.type != VAL_PORT || !is_valid_heap_ptr(vm, value.as.ptr))
        return NULL;
    HeapObject* obj = vm->heap.objects[value.as.ptr];
    if (!obj || obj->type != HEAP_PORT || !obj->opaque.ptr)
        return NULL;

    void* opaque = obj->opaque.ptr;
    if (opaque == (void*)stdin) return vm_port_current_input();
    if (opaque == (void*)stdout) return vm_port_current_output();
    if (opaque == (void*)stderr) return vm_port_current_error();

    VmPort* port = (VmPort*)opaque;
    if (port == &vm_stdin_port || port == &vm_stdout_port || port == &vm_stderr_port)
        vm_io_init_std_ports();
    return port;
}

/** @brief Safely unwrap a VAL_BYTEVECTOR Value to its underlying VmBytevector*,
 *         or NULL if the value isn't a valid heap-backed bytevector. */
static VmBytevector* vm_value_as_bytevector(VM* vm, Value value) {
    if (!vm || value.type != VAL_BYTEVECTOR || !is_valid_heap_ptr(vm, value.as.ptr))
        return NULL;
    HeapObject* obj = vm->heap.objects[value.as.ptr];
    if (!obj || obj->type != HEAP_BYTEVECTOR)
        return NULL;
    return (VmBytevector*)obj->opaque.ptr;
}

typedef int32_t (*VmCompressionAllocFn)(const char*, int32_t, int32_t,
                                        char**, int32_t*);
typedef void (*VmCompressionFreeFn)(void*);
typedef int (*VmSqliteExecFn)(int64_t, const char*);
typedef int64_t (*VmSqliteLastInsertIdFn)(int64_t);
typedef int (*VmSqliteChangesFn)(int64_t);

#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
static volatile sig_atomic_t vm_last_signal = 0;
static volatile sig_atomic_t vm_signal_count = 0;

/** @brief POSIX signal handler: async-signal-safe — just latches the signal
 *         number and bumps a counter for later polling by VM opcodes. */
static void vm_signal_handler(int sig) {
    vm_last_signal = sig;
    vm_signal_count++;
}
#endif

/** @brief True if stdout is attached to a terminal (always false on WASM/Windows). */
static int vm_term_stdout_is_tty(void) {
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
    return isatty(STDOUT_FILENO);
#else
    return 0;
#endif
}

/** @brief True if stdin is attached to a terminal (always false on WASM/Windows). */
static int vm_term_stdin_is_tty(void) {
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
    return isatty(STDIN_FILENO);
#else
    return 0;
#endif
}

/** @brief Write a raw string straight to stdout, but only if stdout is a tty
 *         (used for control/escape sequences that would corrupt piped output). */
static void vm_term_write_tty(const char* s) {
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
    if (s && vm_term_stdout_is_tty()) {
        fputs(s, stdout);
        fflush(stdout);
    }
#else
    (void)s;
#endif
}

/** @brief printf-style two-int formatted write to stdout, gated on tty detection
 *         like vm_term_write_tty(). */
static void vm_term_printf_tty(const char* fmt, int a, int b) {
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
    if (fmt && vm_term_stdout_is_tty()) {
        printf(fmt, a, b);
        fflush(stdout);
    }
#else
    (void)fmt; (void)a; (void)b;
#endif
}

/** @brief Write `data` to the system clipboard via the OSC52 terminal escape
 *         sequence, base64-encoding it inline through a local encode table.
 *         No-op if stdout isn't a tty. */
static void vm_term_write_osc52_tty(const char* data, size_t len) {
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
    static const char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    if (!data || !vm_term_stdout_is_tty()) return;

    fputs("\033]52;c;", stdout);
    for (size_t i = 0; i < len; i += 3) {
        unsigned b0 = (unsigned char)data[i];
        unsigned b1 = (i + 1 < len) ? (unsigned char)data[i + 1] : 0;
        unsigned b2 = (i + 2 < len) ? (unsigned char)data[i + 2] : 0;
        fputc(table[(b0 >> 2) & 0x3F], stdout);
        fputc(table[((b0 << 4) | (b1 >> 4)) & 0x3F], stdout);
        fputc(i + 1 < len ? table[((b1 << 2) | (b2 >> 6)) & 0x3F] : '=', stdout);
        fputc(i + 2 < len ? table[b2 & 0x3F] : '=', stdout);
    }
    fputc('\a', stdout);
    fflush(stdout);
#else
    (void)data; (void)len;
#endif
}

/** @brief Parse an xterm SGR mouse escape sequence (`\033[<btn;x;y;M|m`) from
 *         `buf` into a Scheme list `(button x y modifiers phase)`, or #f if no
 *         such sequence is found. */
static Value vm_term_mouse_event_value(VM* vm, const char* buf) {
    const char* start = buf ? strstr(buf, "\033[<") : NULL;
    int raw = 0, x = 0, y = 0;
    char suffix = '\0';
    if (!start || sscanf(start, "\033[<%d;%d;%d%c", &raw, &x, &y, &suffix) != 4)
        return BOOL_VAL(0);

    int modifiers = 0;
    if (raw & 4) modifiers |= 1;   /* shift */
    if (raw & 8) modifiers |= 2;   /* meta */
    if (raw & 16) modifiers |= 4;  /* control */

    Value result = NIL_VAL;
    result = vm_cons_value(vm, vm_string_value(vm, suffix == 'm' ? "release" : "press", -1), result);
    result = vm_cons_value(vm, INT_VAL((int64_t)modifiers), result);
    result = vm_cons_value(vm, INT_VAL((int64_t)y), result);
    result = vm_cons_value(vm, INT_VAL((int64_t)x), result);
    result = vm_cons_value(vm, INT_VAL((int64_t)(raw & 3)), result);
    return result;
}

/** @brief Detect terminal capabilities from environment variables (TERM,
 *         COLORTERM, TERM_PROGRAM, locale) and tty status, returning an
 *         association list of color depth, unicode support, and tty flags
 *         for use by Scheme-level terminal-capability queries. */
static Value vm_term_detect_capabilities(VM* vm) {
    const char* term = getenv("TERM");
    const char* colorterm = getenv("COLORTERM");
    const char* term_program = getenv("TERM_PROGRAM");
    const char* locale = getenv("LC_ALL");
    if (!locale || !*locale) locale = getenv("LC_CTYPE");
    if (!locale || !*locale) locale = getenv("LANG");

    int color_depth = 4;
    if (colorterm && (strstr(colorterm, "truecolor") || strstr(colorterm, "24bit")))
        color_depth = 24;
    else if (term && strstr(term, "256color"))
        color_depth = 8;
    else if (term && strcmp(term, "dumb") == 0)
        color_depth = 0;

    int unicode = locale && (strstr(locale, "UTF-8") || strstr(locale, "utf8") || strstr(locale, "UTF8"));
    int tty = vm_term_stdout_is_tty();

    Value alist = NIL_VAL;
    alist = vm_cons_value(vm, vm_alist_entry(vm, "clipboard", BOOL_VAL(tty)), alist);
    alist = vm_cons_value(vm, vm_alist_entry(vm, "mouse", BOOL_VAL(tty)), alist);
    alist = vm_cons_value(vm, vm_alist_entry(vm, "alternate-screen", BOOL_VAL(tty)), alist);
    alist = vm_cons_value(vm, vm_alist_entry(vm, "unicode", BOOL_VAL(unicode)), alist);
    alist = vm_cons_value(vm, vm_alist_entry(vm, "color-depth", INT_VAL((int64_t)color_depth)), alist);
    alist = vm_cons_value(vm, vm_alist_entry(vm, "tty?", BOOL_VAL(tty)), alist);
    if (term_program && *term_program)
        alist = vm_cons_value(vm, vm_alist_entry(vm, "term-program", vm_string_value(vm, term_program, -1)), alist);
    if (term && *term)
        alist = vm_cons_value(vm, vm_alist_entry(vm, "term", vm_string_value(vm, term, -1)), alist);
    return alist;
}

/** @brief Stat `path` and report existence, modification time (nanoseconds),
 *         and size — used to detect file-watch changes by comparing snapshots
 *         over time. All output params are zeroed first; no-op on WASM. */
static void vm_file_watch_signature(const char* path, int* exists, int64_t* mtime_ns, int64_t* size) {
    if (exists) *exists = 0;
    if (mtime_ns) *mtime_ns = 0;
    if (size) *size = 0;
    if (!path || !*path) return;
#if defined(_WIN32) && !defined(ESHKOL_VM_WASM)
    struct _stat64 st;
    if (_stat64(path, &st) == 0) {
        if (exists) *exists = 1;
        if (mtime_ns) *mtime_ns = (int64_t)st.st_mtime * 1000000000LL;
        if (size) *size = (int64_t)st.st_size;
    }
#elif !defined(ESHKOL_VM_WASM)
    struct stat st;
    if (stat(path, &st) == 0) {
        if (exists) *exists = 1;
#ifdef __APPLE__
        if (mtime_ns) *mtime_ns = (int64_t)st.st_mtimespec.tv_sec * 1000000000LL +
                                  (int64_t)st.st_mtimespec.tv_nsec;
#else
        if (mtime_ns) *mtime_ns = (int64_t)st.st_mtim.tv_sec * 1000000000LL +
                                  (int64_t)st.st_mtim.tv_nsec;
#endif
        if (size) *size = (int64_t)st.st_size;
    }
#endif
}

/** @brief Allocate a free slot in the VM's fixed-size fs_watchers table and
 *         record `path`'s initial existence/mtime/size signature so future
 *         polls can detect changes. Returns the slot index, or -1 if no
 *         slot is free or the path is invalid. */
static int vm_file_watch_start(VM* vm, const char* path, int recursive) {
    if (!vm || !path || !*path) return -1;
    int slot = -1;
    for (int i = 1; i < (int)(sizeof(vm->fs_watchers) / sizeof(vm->fs_watchers[0])); i++) {
        if (!vm->fs_watchers[i].active) {
            slot = i;
            break;
        }
    }
    if (slot < 0) return -1;

    int exists = 0;
    int64_t mtime_ns = 0;
    int64_t size = 0;
    vm_file_watch_signature(path, &exists, &mtime_ns, &size);
    vm->fs_watchers[slot].active = 1;
    vm->fs_watchers[slot].recursive = recursive ? 1 : 0;
    vm->fs_watchers[slot].exists = exists;
    vm->fs_watchers[slot].mtime_ns = mtime_ns;
    vm->fs_watchers[slot].size = size;
    snprintf(vm->fs_watchers[slot].path, sizeof(vm->fs_watchers[slot].path), "%s", path);
    return slot;
}

/** @brief Resolve an executable name to its full path: on Windows via
 *         SearchPathA, on POSIX by checking for a literal slash (direct
 *         access() check) or scanning $PATH for an executable match.
 *         Returns #f if not found (and always #f on WASM). */
static Value vm_executable_path_value(VM* vm, const char* name) {
    if (!vm || !name || !*name) return BOOL_VAL(0);
#if defined(ESHKOL_VM_WASM)
    (void)name;
    return BOOL_VAL(0);
#elif defined(_WIN32)
    char result[MAX_PATH];
    DWORD n = SearchPathA(NULL, name, ".exe", MAX_PATH, result, NULL);
    if (n > 0 && n < MAX_PATH) return vm_string_value(vm, result, (int64_t)n);
    n = SearchPathA(NULL, name, NULL, MAX_PATH, result, NULL);
    if (n > 0 && n < MAX_PATH) return vm_string_value(vm, result, (int64_t)n);
    return BOOL_VAL(0);
#else
    if (strchr(name, '/')) {
        if (access(name, X_OK) == 0) return vm_string_value(vm, name, -1);
        return BOOL_VAL(0);
    }

    const char* path_env = getenv("PATH");
    if (!path_env || !*path_env) return BOOL_VAL(0);
    char* path_copy = strdup(path_env);
    if (!path_copy) return BOOL_VAL(0);
    Value result = BOOL_VAL(0);
    char* save = NULL;
    for (char* dir = strtok_r(path_copy, ":", &save); dir; dir = strtok_r(NULL, ":", &save)) {
        if (!*dir) dir = ".";
        char full[4096];
        int n = snprintf(full, sizeof(full), "%s/%s", dir, name);
        if (n > 0 && n < (int)sizeof(full) && access(full, X_OK) == 0) {
            result = vm_string_value(vm, full, n);
            break;
        }
    }
    free(path_copy);
    return result;
#endif
}

/** @brief Portable monotonic clock in milliseconds (CLOCK_MONOTONIC on POSIX,
 *         GetTickCount64 on Windows, always 0 on WASM). */
static int64_t vm_monotonic_time_ms(void) {
#if defined(ESHKOL_VM_WASM)
    return 0;
#elif defined(_WIN32)
    return (int64_t)GetTickCount64();
#else
    struct timespec ts;
#ifdef CLOCK_MONOTONIC
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0)
        return (int64_t)ts.tv_sec * 1000 + (int64_t)ts.tv_nsec / 1000000;
#endif
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000 + (int64_t)tv.tv_usec / 1000;
#endif
}

/** @brief Return the platform temp directory: GetTempPathA on Windows (falling
 *         back to %TEMP%), or the first of $TMPDIR/$TMP/$TEMP/"/tmp" on POSIX. */
static const char* vm_temp_directory_path(void) {
#if defined(_WIN32)
    static char buf[MAX_PATH];
    DWORD n = GetTempPathA(MAX_PATH, buf);
    if (n > 0 && n < MAX_PATH) {
        while (n > 1 && (buf[n - 1] == '\\' || buf[n - 1] == '/')) buf[--n] = '\0';
        return buf;
    }
    const char* tmp = getenv("TEMP");
    if (tmp && *tmp) return tmp;
    return ".";
#else
    const char* tmp = getenv("TMPDIR");
    if (!tmp || !*tmp) tmp = getenv("TMP");
    if (!tmp || !*tmp) tmp = getenv("TEMP");
    if (!tmp || !*tmp) tmp = "/tmp";
    return tmp;
#endif
}

/** @brief Register a new sleep-inhibitor handle in the VM's fixed-size table
 *         and, on Windows, call SetThreadExecutionState to actually prevent
 *         system/display sleep. Returns a positive handle, or 0 if the table
 *         is full. */
static int64_t vm_prevent_sleep_start(VM* vm) {
    if (!vm) return 0;
    int slot = -1;
    for (int i = 1; i < (int)(sizeof(vm->sleep_inhibitors) / sizeof(vm->sleep_inhibitors[0])); i++) {
        if (!vm->sleep_inhibitors[i].active) {
            slot = i;
            break;
        }
    }
    if (slot < 0) return 0;
    if (vm->next_sleep_inhibitor <= 0) vm->next_sleep_inhibitor = 1;
    int64_t handle = vm->next_sleep_inhibitor++;
    vm->sleep_inhibitors[slot].active = 1;
    vm->sleep_inhibitors[slot].handle = handle;
#if defined(_WIN32) && !defined(ESHKOL_VM_WASM)
    SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED);
#endif
    return handle;
}

/** @brief Release a sleep-inhibitor handle previously returned by
 *         vm_prevent_sleep_start(); on Windows, restores normal sleep once no
 *         inhibitors remain active. Returns 1 if the handle was found. */
static int vm_prevent_sleep_stop(VM* vm, int64_t handle) {
    if (!vm || handle <= 0) return 0;
    for (int i = 1; i < (int)(sizeof(vm->sleep_inhibitors) / sizeof(vm->sleep_inhibitors[0])); i++) {
        if (vm->sleep_inhibitors[i].active && vm->sleep_inhibitors[i].handle == handle) {
            memset(&vm->sleep_inhibitors[i], 0, sizeof(vm->sleep_inhibitors[i]));
#if defined(_WIN32) && !defined(ESHKOL_VM_WASM)
            int still_active = 0;
            for (int j = 1; j < (int)(sizeof(vm->sleep_inhibitors) / sizeof(vm->sleep_inhibitors[0])); j++)
                if (vm->sleep_inhibitors[j].active) still_active = 1;
            if (!still_active) SetThreadExecutionState(ES_CONTINUOUS);
#endif
            return 1;
        }
    }
    return 0;
}

/** @brief Return the byte length of the ANSI/C1 escape sequence starting at
 *         `s[pos]` (CSI, OSC, DCS/APC/PM/SOS, or a 2-3 byte ESC-prefixed
 *         sequence), or 0 if `s[pos]` doesn't begin an escape sequence.
 *         Used to skip over escapes when measuring or stripping display text. */
static int vm_ansi_escape_len(const char* s, int len, int pos) {
    if (!s || pos < 0 || pos >= len) return 0;
    unsigned char c = (unsigned char)s[pos];

    if (c == 0x9B) { /* C1 CSI */
        int i = pos + 1;
        while (i < len) {
            unsigned char ch = (unsigned char)s[i++];
            if (ch >= 0x40 && ch <= 0x7E) return i - pos;
        }
        return len - pos;
    }
    if (c == 0x9D) { /* C1 OSC */
        int i = pos + 1;
        while (i < len) {
            unsigned char ch = (unsigned char)s[i];
            if (ch == 0x07) return i + 1 - pos;
            if (ch == 0x1B && i + 1 < len && s[i + 1] == '\\') return i + 2 - pos;
            i++;
        }
        return len - pos;
    }
    if (c != 0x1B) return 0;
    if (pos + 1 >= len) return 1;

    unsigned char next = (unsigned char)s[pos + 1];
    if (next == '[') { /* CSI */
        int i = pos + 2;
        while (i < len) {
            unsigned char ch = (unsigned char)s[i++];
            if (ch >= 0x40 && ch <= 0x7E) return i - pos;
        }
        return len - pos;
    }
    if (next == ']' || next == 'P' || next == '^' || next == '_' || next == 'X') {
        int i = pos + 2;
        while (i < len) {
            unsigned char ch = (unsigned char)s[i];
            if (ch == 0x07) return i + 1 - pos;
            if (ch == 0x1B && i + 1 < len && s[i + 1] == '\\') return i + 2 - pos;
            i++;
        }
        return len - pos;
    }
    if (strchr("()*+-./", next)) return (pos + 2 < len) ? 3 : 2;
    return 2;
}

/** @brief True if Unicode codepoint `cp` renders as double-width in a terminal
 *         (CJK ideographs, Hangul, fullwidth forms, most emoji, etc). */
static int vm_display_is_wide_char(uint32_t cp) {
    if (cp >= 0x4E00 && cp <= 0x9FFF) return 1;
    if (cp >= 0x3400 && cp <= 0x4DBF) return 1;
    if (cp >= 0x20000 && cp <= 0x2A6DF) return 1;
    if (cp >= 0xF900 && cp <= 0xFAFF) return 1;
    if (cp >= 0xAC00 && cp <= 0xD7AF) return 1;
    if (cp >= 0xFF01 && cp <= 0xFF60) return 1;
    if (cp >= 0xFFE0 && cp <= 0xFFE6) return 1;
    if (cp >= 0x2E80 && cp <= 0x303E) return 1;
    if (cp >= 0x3040 && cp <= 0x30FF) return 1;
    if (cp >= 0x31F0 && cp <= 0x31FF) return 1;
    if (cp >= 0x1F300 && cp <= 0x1F9FF) return 1;
    if (cp >= 0x1FA00 && cp <= 0x1FAFF) return 1;
    if (cp >= 0x2600 && cp <= 0x27BF) return 1;
    return 0;
}

/** @brief True if Unicode codepoint `cp` is zero-width for display purposes
 *         (combining marks, variation selectors, joiners/format chars). */
static int vm_display_is_zero_width(uint32_t cp) {
    if (cp >= 0x0300 && cp <= 0x036F) return 1;
    if (cp >= 0x1AB0 && cp <= 0x1AFF) return 1;
    if (cp >= 0x1DC0 && cp <= 0x1DFF) return 1;
    if (cp >= 0x20D0 && cp <= 0x20FF) return 1;
    if (cp >= 0xFE20 && cp <= 0xFE2F) return 1;
    if (cp == 0x200B || cp == 0x200C || cp == 0x200D ||
        cp == 0x200E || cp == 0x200F || cp == 0xFEFF) return 1;
    if (cp >= 0xFE00 && cp <= 0xFE0F) return 1;
    if (cp >= 0xE0100 && cp <= 0xE01EF) return 1;
    return 0;
}

/** @brief Compute the terminal display width (in columns) of a UTF-8 byte
 *         buffer, decoding codepoints and skipping ANSI escapes, zero-width
 *         combining marks (counted as 0), and counting wide CJK/emoji
 *         codepoints as 2 columns. */
static int64_t vm_string_display_width_bytes(const char* data, int len) {
    if (!data || len <= 0) return 0;
    int64_t width = 0;
    int pos = 0;
    while (pos < len) {
        int skip = vm_ansi_escape_len(data, len, pos);
        if (skip > 0) {
            pos += skip;
            continue;
        }
        int before = pos;
        int cp = vm_utf8_decode(data, len, &pos);
        if (pos <= before) pos = before + 1;
        if (vm_display_is_zero_width((uint32_t)cp)) continue;
        width += vm_display_is_wide_char((uint32_t)cp) ? 2 : 1;
    }
    return width;
}

/** @brief Find the byte length of the longest prefix of a UTF-8 buffer whose
 *         display width (per vm_string_display_width_bytes rules) does not
 *         exceed `max_cols`. Used to truncate strings at a column budget. */
static int vm_display_prefix_byte_len(const char* data, int len, int64_t max_cols) {
    if (!data || len <= 0 || max_cols < 0) return 0;
    int64_t width = 0;
    int pos = 0;
    int end = 0;
    while (pos < len) {
        int skip = vm_ansi_escape_len(data, len, pos);
        if (skip > 0) {
            pos += skip;
            end = pos;
            continue;
        }
        int before = pos;
        int cp = vm_utf8_decode(data, len, &pos);
        if (pos <= before) pos = before + 1;
        int char_width = 0;
        if (!vm_display_is_zero_width((uint32_t)cp))
            char_width = vm_display_is_wide_char((uint32_t)cp) ? 2 : 1;
        if (width + char_width > max_cols) break;
        width += char_width;
        end = pos;
    }
    return end;
}

/** @brief Return a copy of `input` with all ANSI/C1 escape sequences removed. */
static Value vm_ansi_strip_value(VM* vm, VmString* input) {
    if (!vm || !input || !input->data) return BOOL_VAL(0);
    int len = input->byte_len;
    char* out = (char*)malloc((size_t)len + 1);
    if (!out) return BOOL_VAL(0);

    int i = 0;
    int j = 0;
    while (i < len) {
        int skip = vm_ansi_escape_len(input->data, len, i);
        if (skip > 0) {
            i += skip;
            continue;
        }
        out[j++] = input->data[i++];
    }
    out[j] = '\0';
    Value result = vm_string_value(vm, out, j);
    free(out);
    return result;
}

/** @brief Truncate `input` to at most `max_cols` display columns, appending
 *         `suffix` (e.g. an ellipsis) when truncation occurs. If `suffix`
 *         alone is wider than `max_cols`, only a prefix of the suffix is used.
 *         Returns `input` unchanged if it already fits. */
static Value vm_string_truncate_display_value(VM* vm,
                                              VmString* input,
                                              int64_t max_cols,
                                              VmString* suffix) {
    if (!vm || !input || !input->data) return vm_string_value(vm, "", 0);
    if (max_cols < 0) max_cols = 0;

    int64_t full_width = vm_string_display_width_bytes(input->data, input->byte_len);
    if (full_width <= max_cols)
        return vm_string_value(vm, input->data, input->byte_len);

    const char* suffix_data = (suffix && suffix->data) ? suffix->data : "";
    int suffix_len = (suffix && suffix->data) ? suffix->byte_len : 0;
    int64_t suffix_width = vm_string_display_width_bytes(suffix_data, suffix_len);

    int prefix_len = 0;
    int append_suffix_len = suffix_len;
    if (suffix_width <= max_cols) {
        prefix_len = vm_display_prefix_byte_len(input->data, input->byte_len,
                                                max_cols - suffix_width);
    } else {
        append_suffix_len = vm_display_prefix_byte_len(suffix_data, suffix_len, max_cols);
    }

    size_t out_len = (size_t)prefix_len + (size_t)append_suffix_len;
    char* out = (char*)malloc(out_len + 1);
    if (!out) return BOOL_VAL(0);
    if (prefix_len > 0) memcpy(out, input->data, (size_t)prefix_len);
    if (append_suffix_len > 0)
        memcpy(out + prefix_len, suffix_data, (size_t)append_suffix_len);
    out[out_len] = '\0';
    Value result = vm_string_value(vm, out, (int64_t)out_len);
    free(out);
    return result;
}

/** @brief True if `c` is an RFC 3986 URL-unreserved character (alnum, `-_.~`)
 *         that does not need percent-encoding. */
static int vm_url_is_unreserved(unsigned char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
           (c >= '0' && c <= '9') || c == '-' || c == '_' ||
           c == '.' || c == '~';
}

/** @brief Decode a single hex digit character to its 0-15 value, or -1 if not
 *         a hex digit. */
static int vm_url_hex_value(unsigned char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

/** @brief Percent-encode `input` per RFC 3986 (unreserved chars pass through
 *         literally, everything else becomes `%XX`). */
static Value vm_url_encode_value(VM* vm, VmString* input) {
    if (!vm || !input || !input->data) return BOOL_VAL(0);
    static const char hex[] = "0123456789ABCDEF";
    size_t cap = (size_t)input->byte_len * 3 + 1;
    char* out = (char*)malloc(cap);
    if (!out) return BOOL_VAL(0);
    size_t pos = 0;
    for (int i = 0; i < input->byte_len; i++) {
        unsigned char c = (unsigned char)input->data[i];
        if (vm_url_is_unreserved(c)) {
            out[pos++] = (char)c;
        } else {
            out[pos++] = '%';
            out[pos++] = hex[(c >> 4) & 0x0F];
            out[pos++] = hex[c & 0x0F];
        }
    }
    out[pos] = '\0';
    Value result = vm_string_value(vm, out, (int64_t)pos);
    free(out);
    return result;
}

/** @brief Percent-decode `input` (`%XX` sequences and `+` as space, form-encoding style). */
static Value vm_url_decode_value(VM* vm, VmString* input) {
    if (!vm || !input || !input->data) return BOOL_VAL(0);
    char* out = (char*)malloc((size_t)input->byte_len + 1);
    if (!out) return BOOL_VAL(0);
    int pos = 0;
    for (int i = 0; i < input->byte_len; i++) {
        unsigned char c = (unsigned char)input->data[i];
        if (c == '%' && i + 2 < input->byte_len) {
            int hi = vm_url_hex_value((unsigned char)input->data[i + 1]);
            int lo = vm_url_hex_value((unsigned char)input->data[i + 2]);
            if (hi >= 0 && lo >= 0) {
                out[pos++] = (char)((hi << 4) | lo);
                i += 2;
                continue;
            }
        }
        out[pos++] = (c == '+') ? ' ' : (char)c;
    }
    out[pos] = '\0';
    Value result = vm_string_value(vm, out, pos);
    free(out);
    return result;
}

/** @brief Scan forward from `p` for the first URL authority-terminating
 *         character (`/`, `?`, or `#`), returning `end` if none is found. */
static const char* vm_find_url_sep(const char* p, const char* end) {
    while (p < end) {
        if (*p == '/' || *p == '?' || *p == '#') return p;
        p++;
    }
    return end;
}

/** @brief Compare a length-delimited byte span `p[0..len)` against a C string
 *         for exact equality. */
static int vm_span_eq_cstr(const char* p, int len, const char* s) {
    return p && s && len == (int)strlen(s) && memcmp(p, s, (size_t)len) == 0;
}

/** @brief Parse a `scheme://host[:port][/path][?query][#fragment]` URL into
 *         an association list of its components. Applies default ports for
 *         http/https when none is given. Returns #f for malformed input
 *         (missing scheme separator or empty authority/host). */
static Value vm_url_parse_value(VM* vm, VmString* input) {
    if (!vm || !input || !input->data || input->byte_len <= 0) return BOOL_VAL(0);
    const char* begin = input->data;
    const char* end = input->data + input->byte_len;
    const char* scheme_end = NULL;
    for (const char* p = begin; p + 2 < end; p++) {
        if (p[0] == ':' && p[1] == '/' && p[2] == '/') {
            scheme_end = p;
            break;
        }
    }
    if (!scheme_end || scheme_end == begin) return BOOL_VAL(0);

    const char* authority = scheme_end + 3;
    const char* authority_end = vm_find_url_sep(authority, end);
    if (authority == authority_end) return BOOL_VAL(0);

    const char* host_start = authority;
    const char* host_end = authority_end;
    int64_t port = 0;
    const char* colon = NULL;
    for (const char* p = authority_end; p > authority; p--) {
        if (*(p - 1) == ':') {
            colon = p - 1;
            break;
        }
    }
    if (colon && colon + 1 < authority_end) {
        int all_digits = 1;
        int64_t parsed = 0;
        for (const char* p = colon + 1; p < authority_end; p++) {
            if (!isdigit((unsigned char)*p)) { all_digits = 0; break; }
            parsed = parsed * 10 + (*p - '0');
            if (parsed > 65535) { all_digits = 0; break; }
        }
        if (all_digits) {
            host_end = colon;
            port = parsed;
        }
    }
    if (host_start == host_end) return BOOL_VAL(0);
    int scheme_len = (int)(scheme_end - begin);
    if (port == 0) {
        if (vm_span_eq_cstr(begin, scheme_len, "https")) port = 443;
        else if (vm_span_eq_cstr(begin, scheme_len, "http")) port = 80;
    }

    const char* path_start = authority_end;
    const char* path_end = authority_end;
    if (path_start < end && *path_start == '/') {
        path_end = path_start;
        while (path_end < end && *path_end != '?' && *path_end != '#') path_end++;
    }

    const char* query_start = NULL;
    const char* query_end = NULL;
    const char* fragment_start = NULL;
    for (const char* p = authority_end; p < end; p++) {
        if (*p == '?' && !query_start && !fragment_start) {
            query_start = p + 1;
            query_end = end;
        } else if (*p == '#') {
            fragment_start = p + 1;
            if (query_start) query_end = p;
            break;
        }
    }

    Value result = NIL_VAL;
    if (fragment_start)
        result = vm_cons_value(vm, vm_alist_entry(vm, "fragment",
                              vm_string_value(vm, fragment_start, (int64_t)(end - fragment_start))), result);
    if (query_start)
        result = vm_cons_value(vm, vm_alist_entry(vm, "query",
                              vm_string_value(vm, query_start, (int64_t)(query_end - query_start))), result);
    if (path_start < end && *path_start == '/')
        result = vm_cons_value(vm, vm_alist_entry(vm, "path",
                              vm_string_value(vm, path_start, (int64_t)(path_end - path_start))), result);
    else
        result = vm_cons_value(vm, vm_alist_entry(vm, "path", vm_string_value(vm, "/", 1)), result);
    if (port > 0)
        result = vm_cons_value(vm, vm_alist_entry(vm, "port", INT_VAL(port)), result);
    result = vm_cons_value(vm, vm_alist_entry(vm, "host",
                          vm_string_value(vm, host_start, (int64_t)(host_end - host_start))), result);
    result = vm_cons_value(vm, vm_alist_entry(vm, "scheme",
                          vm_string_value(vm, begin, (int64_t)scheme_len)), result);
    return result;
}

/** @brief Extract a raw byte pointer and length from a Value that is either a
 *         string or a bytevector, for use by encoding/hashing routines that
 *         operate on either representation uniformly. Returns 0 if neither. */
static int vm_bytes_from_value(VM* vm, Value value, const unsigned char** data, int64_t* len) {
    VmString* s = vm_value_as_string(vm, value);
    if (s && s->data) {
        if (data) *data = (const unsigned char*)s->data;
        if (len) *len = s->byte_len;
        return 1;
    }
    VmBytevector* bv = vm_value_as_bytevector(vm, value);
    if (bv && bv->data) {
        if (data) *data = (const unsigned char*)bv->data;
        if (len) *len = bv->len;
        return 1;
    }
    return 0;
}

/** @brief Base64url-encode (RFC 4648 sec 5, unpadded) the bytes of a string
 *         or bytevector Value. */
static Value vm_base64url_encode_value(VM* vm, Value input) {
    static const char table[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    const unsigned char* data = NULL;
    int64_t len = 0;
    if (!vm_bytes_from_value(vm, input, &data, &len) || len < 0) return BOOL_VAL(0);
    size_t out_len = ((size_t)len / 3) * 4;
    if (len % 3 == 1) out_len += 2;
    else if (len % 3 == 2) out_len += 3;
    char* out = (char*)malloc(out_len + 1);
    if (!out) return BOOL_VAL(0);
    size_t pos = 0;
    int64_t i = 0;
    for (; i + 2 < len; i += 3) {
        unsigned int n = ((unsigned int)data[i] << 16) |
                         ((unsigned int)data[i + 1] << 8) |
                         (unsigned int)data[i + 2];
        out[pos++] = table[(n >> 18) & 63];
        out[pos++] = table[(n >> 12) & 63];
        out[pos++] = table[(n >> 6) & 63];
        out[pos++] = table[n & 63];
    }
    if (i < len) {
        unsigned int n = (unsigned int)data[i] << 16;
        if (i + 1 < len) n |= (unsigned int)data[i + 1] << 8;
        out[pos++] = table[(n >> 18) & 63];
        out[pos++] = table[(n >> 12) & 63];
        if (i + 1 < len) out[pos++] = table[(n >> 6) & 63];
    }
    out[pos] = '\0';
    Value result = vm_string_value(vm, out, (int64_t)pos);
    free(out);
    return result;
}

/** @brief Decode a single base64url alphabet character to its 0-63 value, or
 *         -1 if not part of the alphabet. */
static int vm_base64url_value(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '-') return 62;
    if (c == '_') return 63;
    return -1;
}

/** @brief Base64url-decode (RFC 4648 sec 5, tolerating trailing `=` padding)
 *         the bytes of a string or bytevector Value into a new string. */
static Value vm_base64url_decode_value(VM* vm, Value input) {
    const unsigned char* data = NULL;
    int64_t len = 0;
    if (!vm_bytes_from_value(vm, input, &data, &len) || len < 0) return BOOL_VAL(0);
    while (len > 0 && data[len - 1] == '=') len--;
    if ((len % 4) == 1) return BOOL_VAL(0);
    size_t out_cap = ((size_t)len * 6) / 8 + 1;
    char* out = (char*)malloc(out_cap);
    if (!out) return BOOL_VAL(0);
    int acc = 0;
    int bits = 0;
    size_t pos = 0;
    for (int64_t i = 0; i < len; i++) {
        int v = vm_base64url_value(data[i]);
        if (v < 0) {
            free(out);
            return BOOL_VAL(0);
        }
        acc = (acc << 6) | v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out[pos++] = (char)((acc >> bits) & 0xFF);
        }
    }
    out[pos] = '\0';
    Value result = vm_string_value(vm, out, (int64_t)pos);
    free(out);
    return result;
}

/** @brief Base64-encode bytes using the padded RFC 4648 section 4 alphabet. */
static Value vm_base64_encode_value(VM* vm, Value input) {
    static const char table[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const unsigned char* data = NULL;
    int64_t len = 0;
    if (!vm_bytes_from_value(vm, input, &data, &len) || len < 0)
        return BOOL_VAL(0);
    const size_t groups = ((size_t)len + 2u) / 3u;
    if (groups > (SIZE_MAX - 1u) / 4u)
        return BOOL_VAL(0);

    size_t out_len = 4u * groups;
    char* out = (char*)malloc(out_len + 1u);
    if (!out) return BOOL_VAL(0);

    size_t pos = 0;
    int64_t i = 0;
    while (i + 2 < len) {
        uint32_t word = ((uint32_t)data[i] << 16) |
                        ((uint32_t)data[i + 1] << 8) |
                        (uint32_t)data[i + 2];
        out[pos++] = table[(word >> 18) & 63u];
        out[pos++] = table[(word >> 12) & 63u];
        out[pos++] = table[(word >> 6) & 63u];
        out[pos++] = table[word & 63u];
        i += 3;
    }
    if (i < len) {
        uint32_t word = (uint32_t)data[i] << 16;
        const int have_second = (i + 1 < len);
        if (have_second) word |= (uint32_t)data[i + 1] << 8;
        out[pos++] = table[(word >> 18) & 63u];
        out[pos++] = table[(word >> 12) & 63u];
        out[pos++] = have_second ? table[(word >> 6) & 63u] : '=';
        out[pos++] = '=';
    }
    out[pos] = '\0';
    Value result = vm_string_value(vm, out, (int64_t)pos);
    free(out);
    return result;
}

/** @brief Decode one RFC 4648 section 4 alphabet character. */
static int vm_base64_value(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

/** @brief Strictly decode padded RFC 4648 section 4 Base64 into a string.
 *
 * Rejects misplaced padding and non-canonical non-zero trailing bits rather
 * than accepting multiple encodings of the same byte sequence. */
static Value vm_base64_decode_value(VM* vm, Value input) {
    const unsigned char* data = NULL;
    int64_t len = 0;
    if (!vm_bytes_from_value(vm, input, &data, &len) || len < 0)
        return BOOL_VAL(0);
    if (len == 0) return vm_string_value(vm, "", 0);
    if ((len % 4) != 0) return BOOL_VAL(0);

    int padding = 0;
    if (data[len - 1] == '=') padding++;
    if (len >= 2 && data[len - 2] == '=') padding++;
    for (int64_t i = 0; i < len - padding; ++i) {
        if (data[i] == '=' || vm_base64_value(data[i]) < 0)
            return BOOL_VAL(0);
    }
    for (int64_t i = len - padding; i < len; ++i) {
        if (data[i] != '=') return BOOL_VAL(0);
    }

    size_t out_len = ((size_t)len / 4u) * 3u - (size_t)padding;
    char* out = (char*)malloc(out_len + 1u);
    if (!out) return BOOL_VAL(0);
    size_t pos = 0;

    for (int64_t i = 0; i < len; i += 4) {
        const int last = (i + 4 == len);
        int a = vm_base64_value(data[i]);
        int b = vm_base64_value(data[i + 1]);
        int c = data[i + 2] == '=' ? 0 : vm_base64_value(data[i + 2]);
        int d = data[i + 3] == '=' ? 0 : vm_base64_value(data[i + 3]);
        if (a < 0 || b < 0 || c < 0 || d < 0 ||
            (!last && (data[i + 2] == '=' || data[i + 3] == '='))) {
            free(out);
            return BOOL_VAL(0);
        }
        if (last && padding == 2 && (b & 0x0f) != 0) {
            free(out);
            return BOOL_VAL(0);
        }
        if (last && padding == 1 && (c & 0x03) != 0) {
            free(out);
            return BOOL_VAL(0);
        }

        uint32_t word = ((uint32_t)a << 18) | ((uint32_t)b << 12) |
                        ((uint32_t)c << 6) | (uint32_t)d;
        if (pos < out_len) out[pos++] = (char)((word >> 16) & 0xffu);
        if (pos < out_len) out[pos++] = (char)((word >> 8) & 0xffu);
        if (pos < out_len) out[pos++] = (char)(word & 0xffu);
    }
    out[pos] = '\0';
    Value result = vm_string_value(vm, out, (int64_t)pos);
    free(out);
    return result;
}

/** @brief Fill `out` with `len` random bytes, preferring /dev/urandom on POSIX
 *         and falling back to a seeded rand() stream (not cryptographically
 *         strong) if the device is unavailable or on other platforms. */
static int vm_random_bytes(unsigned char* out, size_t len) {
    if (!out) return 0;
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd >= 0) {
        size_t pos = 0;
        while (pos < len) {
            ssize_t n = read(fd, out + pos, len - pos);
            if (n <= 0) break;
            pos += (size_t)n;
        }
        close(fd);
        if (pos == len) return 1;
    }
#endif
    srand((unsigned)time(NULL) ^ (unsigned)(uintptr_t)out);
    for (size_t i = 0; i < len; i++) out[i] = (unsigned char)(rand() & 0xFF);
    return 1;
}

/** @brief Generate a random RFC 4122 version-4 UUID string, using
 *         vm_random_bytes() and setting the version/variant bits per spec. */
static Value vm_uuid_v4_value(VM* vm) {
    unsigned char bytes[16];
    if (!vm_random_bytes(bytes, sizeof(bytes))) return BOOL_VAL(0);
    bytes[6] = (bytes[6] & 0x0F) | 0x40;
    bytes[8] = (bytes[8] & 0x3F) | 0x80;
    char buf[37];
    snprintf(buf, sizeof(buf),
             "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             bytes[0], bytes[1], bytes[2], bytes[3],
             bytes[4], bytes[5], bytes[6], bytes[7],
             bytes[8], bytes[9], bytes[10], bytes[11],
             bytes[12], bytes[13], bytes[14], bytes[15]);
    return vm_string_value(vm, buf, 36);
}

/** @brief Compare two string/bytevector Values for equality in constant time
 *         (branchless OR-accumulated XOR over the full max length), to avoid
 *         timing side channels when comparing secrets like tokens or MACs. */
static Value vm_constant_time_equal_value(VM* vm, Value a_val, Value b_val) {
    const unsigned char* a = NULL;
    const unsigned char* b = NULL;
    int64_t a_len = 0;
    int64_t b_len = 0;
    if (!vm_bytes_from_value(vm, a_val, &a, &a_len) ||
        !vm_bytes_from_value(vm, b_val, &b, &b_len))
        return BOOL_VAL(0);
    int64_t max_len = a_len > b_len ? a_len : b_len;
    volatile unsigned char diff = (unsigned char)(a_len ^ b_len);
    for (int64_t i = 0; i < max_len; i++) {
        unsigned char ac = i < a_len ? a[i] : 0;
        unsigned char bc = i < b_len ? b[i] : 0;
        diff |= (unsigned char)(ac ^ bc);
    }
    return BOOL_VAL(diff == 0);
}

typedef struct {
    uint32_t h[8];
    uint64_t bit_len;
    unsigned char buf[64];
    size_t buf_len;
} VmSha256;

/** @brief Right-rotate a 32-bit word by `n` bits (SHA-256 primitive). */
static uint32_t vm_sha256_rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

/** @brief Process one 64-byte block through the SHA-256 compression function,
 *         updating `ctx->h` in place (standard FIPS 180-4 message schedule
 *         and 64-round compression loop). */
static void vm_sha256_transform(VmSha256* ctx, const unsigned char block[64]) {
    static const uint32_t k[64] = {
        0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
        0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
        0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
        0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
        0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
        0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
        0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
        0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U
    };
    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i * 4] << 24) |
               ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) |
               (uint32_t)block[i * 4 + 3];
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = vm_sha256_rotr(w[i - 15], 7) ^ vm_sha256_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = vm_sha256_rotr(w[i - 2], 17) ^ vm_sha256_rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = ctx->h[0], b = ctx->h[1], c = ctx->h[2], d = ctx->h[3];
    uint32_t e = ctx->h[4], f = ctx->h[5], g = ctx->h[6], h = ctx->h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = vm_sha256_rotr(e, 6) ^ vm_sha256_rotr(e, 11) ^ vm_sha256_rotr(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + S1 + ch + k[i] + w[i];
        uint32_t S0 = vm_sha256_rotr(a, 2) ^ vm_sha256_rotr(a, 13) ^ vm_sha256_rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;
        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }
    ctx->h[0] += a; ctx->h[1] += b; ctx->h[2] += c; ctx->h[3] += d;
    ctx->h[4] += e; ctx->h[5] += f; ctx->h[6] += g; ctx->h[7] += h;
}

/** @brief Initialize a SHA-256 context with the standard FIPS 180-4 initial hash values. */
static void vm_sha256_init(VmSha256* ctx) {
    static const uint32_t init[8] = {
        0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,
        0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U
    };
    memcpy(ctx->h, init, sizeof(init));
    ctx->bit_len = 0;
    ctx->buf_len = 0;
}

/** @brief Feed `len` bytes of `data` into a SHA-256 context, buffering
 *         partial blocks and transforming complete 64-byte blocks as they fill. */
static void vm_sha256_update(VmSha256* ctx, const unsigned char* data, size_t len) {
    ctx->bit_len += (uint64_t)len * 8U;
    while (len > 0) {
        size_t n = 64 - ctx->buf_len;
        if (n > len) n = len;
        memcpy(ctx->buf + ctx->buf_len, data, n);
        ctx->buf_len += n;
        data += n;
        len -= n;
        if (ctx->buf_len == 64) {
            vm_sha256_transform(ctx, ctx->buf);
            ctx->buf_len = 0;
        }
    }
}

/** @brief Apply SHA-256 padding (0x80, zero bytes, 64-bit big-endian bit
 *         length) to the context's final partial block and write the 32-byte
 *         digest to `out`. */
static void vm_sha256_final(VmSha256* ctx, unsigned char out[32]) {
    ctx->buf[ctx->buf_len++] = 0x80;
    if (ctx->buf_len > 56) {
        while (ctx->buf_len < 64) ctx->buf[ctx->buf_len++] = 0;
        vm_sha256_transform(ctx, ctx->buf);
        ctx->buf_len = 0;
    }
    while (ctx->buf_len < 56) ctx->buf[ctx->buf_len++] = 0;
    for (int i = 7; i >= 0; i--) {
        ctx->buf[ctx->buf_len++] = (unsigned char)((ctx->bit_len >> (i * 8)) & 0xFF);
    }
    vm_sha256_transform(ctx, ctx->buf);
    for (int i = 0; i < 8; i++) {
        out[i * 4] = (unsigned char)(ctx->h[i] >> 24);
        out[i * 4 + 1] = (unsigned char)(ctx->h[i] >> 16);
        out[i * 4 + 2] = (unsigned char)(ctx->h[i] >> 8);
        out[i * 4 + 3] = (unsigned char)ctx->h[i];
    }
}

/** @brief Stream-hash a file's contents with SHA-256 and return the digest as
 *         a lowercase hex string. Returns #f if the file can't be opened or
 *         a read error occurs. */
static Value vm_sha256_file_value(VM* vm, VmString* path) {
    if (!vm || !path || !path->data) return BOOL_VAL(0);
    FILE* f = fopen(path->data, "rb");
    if (!f) return BOOL_VAL(0);
    VmSha256 ctx;
    vm_sha256_init(&ctx);
    unsigned char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        vm_sha256_update(&ctx, buf, n);
    }
    int ok = !ferror(f);
    fclose(f);
    if (!ok) return BOOL_VAL(0);
    unsigned char digest[32];
    vm_sha256_final(&ctx, digest);
    static const char hex[] = "0123456789abcdef";
    char out[65];
    for (int i = 0; i < 32; i++) {
        out[i * 2] = hex[(digest[i] >> 4) & 0x0F];
        out[i * 2 + 1] = hex[digest[i] & 0x0F];
    }
    out[64] = '\0';
    return vm_string_value(vm, out, 64);
}

/** @brief Compile an extended POSIX regex and store it in a free slot of the
 *         VM's regex_handles table. Returns the slot index (usable as an
 *         opaque handle Value) or -1 on failure/no free slot; always -1 on
 *         WASM/Windows where POSIX regex.h isn't available. */
static int vm_regex_compile_handle(VM* vm, VmString* pattern) {
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
    if (!vm || !pattern || !pattern->data) return -1;
    int slot = -1;
    for (int i = 1; i < (int)(sizeof(vm->regex_handles) / sizeof(vm->regex_handles[0])); i++) {
        if (!vm->regex_handles[i].active) { slot = i; break; }
    }
    if (slot < 0) return -1;
    regex_t* re = (regex_t*)calloc(1, sizeof(regex_t));
    if (!re) return -1;
    int rc = regcomp(re, pattern->data, REG_EXTENDED);
    if (rc != 0) {
        free(re);
        return -1;
    }
    vm->regex_handles[slot].active = 1;
    vm->regex_handles[slot].regex = re;
    return slot;
#else
    (void)vm; (void)pattern;
    return -1;
#endif
}

/** @brief Resolve a regex handle Value back to its compiled regex_t*, or NULL
 *         if the handle is out of range or not active. */
static void* vm_regex_get(VM* vm, Value handle_val) {
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
    if (!vm) return NULL;
    int handle = (int)as_number(handle_val);
    if (handle <= 0 || handle >= (int)(sizeof(vm->regex_handles) / sizeof(vm->regex_handles[0])) ||
        !vm->regex_handles[handle].active)
        return NULL;
    return vm->regex_handles[handle].regex;
#else
    (void)vm; (void)handle_val;
    return NULL;
#endif
}

/** @brief Free the compiled regex behind a handle Value and clear its slot. */
static int vm_regex_free_handle(VM* vm, Value handle_val) {
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
    if (!vm) return 0;
    int handle = (int)as_number(handle_val);
    if (handle <= 0 || handle >= (int)(sizeof(vm->regex_handles) / sizeof(vm->regex_handles[0])) ||
        !vm->regex_handles[handle].active)
        return 0;
    regex_t* re = (regex_t*)vm->regex_handles[handle].regex;
    if (re) {
        regfree(re);
        free(re);
    }
    memset(&vm->regex_handles[handle], 0, sizeof(vm->regex_handles[handle]));
    return 1;
#else
    (void)vm; (void)handle_val;
    return 0;
#endif
}

/** @brief Free every still-active compiled regex owned by this VM (called on VM teardown). */
static void vm_regex_free_all(VM* vm) {
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
    if (!vm) return;
    for (int i = 1; i < (int)(sizeof(vm->regex_handles) / sizeof(vm->regex_handles[0])); i++) {
        if (!vm->regex_handles[i].active) continue;
        regex_t* re = (regex_t*)vm->regex_handles[i].regex;
        if (re) {
            regfree(re);
            free(re);
        }
        memset(&vm->regex_handles[i], 0, sizeof(vm->regex_handles[i]));
    }
#else
    (void)vm;
#endif
}

/** @brief Run a compiled regex against `subject`. If `boolean_only`, returns
 *         #t/#f for whether it matched; otherwise returns the matched
 *         substring, or #f if there's no match. */
static Value vm_regex_match_value(VM* vm, void* re_ptr, VmString* subject, int boolean_only) {
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
    regex_t* re = (regex_t*)re_ptr;
    if (!vm || !re || !subject || !subject->data) return BOOL_VAL(0);
    regmatch_t m[1];
    int rc = regexec(re, subject->data, 1, m, 0);
    if (rc != 0 || m[0].rm_so < 0) return BOOL_VAL(0);
    if (boolean_only) return BOOL_VAL(1);
    return vm_string_value(vm, subject->data + m[0].rm_so, (int64_t)(m[0].rm_eo - m[0].rm_so));
#else
    (void)vm; (void)re_ptr; (void)subject; (void)boolean_only;
    return BOOL_VAL(0);
#endif
}

/** @brief Run a compiled regex against `subject` and return an association
 *         list with the full match ("full"), its start/end byte offsets, and
 *         the list of capture group strings ("groups"), or #f on no match. */
static Value vm_regex_match_groups_value(VM* vm, void* re_ptr, VmString* subject) {
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
    regex_t* re = (regex_t*)re_ptr;
    if (!vm || !re || !subject || !subject->data) return BOOL_VAL(0);
    size_t nmatch = re->re_nsub + 1;
    if (nmatch > 32) nmatch = 32;
    regmatch_t matches[32];
    int rc = regexec(re, subject->data, nmatch, matches, 0);
    if (rc != 0 || matches[0].rm_so < 0) return BOOL_VAL(0);
    Value groups = NIL_VAL;
    for (size_t i = nmatch; i > 1; i--) {
        regmatch_t m = matches[i - 1];
        Value item = (m.rm_so >= 0)
            ? vm_string_value(vm, subject->data + m.rm_so, (int64_t)(m.rm_eo - m.rm_so))
            : vm_string_value(vm, "", 0);
        groups = vm_cons_value(vm, item, groups);
    }
    Value result = NIL_VAL;
    result = vm_cons_value(vm, vm_alist_entry(vm, "end", INT_VAL((int64_t)matches[0].rm_eo)), result);
    result = vm_cons_value(vm, vm_alist_entry(vm, "start", INT_VAL((int64_t)matches[0].rm_so)), result);
    result = vm_cons_value(vm, vm_alist_entry(vm, "groups", groups), result);
    result = vm_cons_value(vm, vm_alist_entry(vm, "full",
                          vm_string_value(vm, subject->data + matches[0].rm_so,
                                          (int64_t)(matches[0].rm_eo - matches[0].rm_so))), result);
    return result;
#else
    (void)vm; (void)re_ptr; (void)subject;
    return BOOL_VAL(0);
#endif
}

/** @brief Split `subject` into a list of substrings at each successive match
 *         of the compiled regex (up to 1024 splits), similar to Perl/JS
 *         string.split(regex). */
static Value vm_regex_split_value(VM* vm, void* re_ptr, VmString* subject) {
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
    regex_t* re = (regex_t*)re_ptr;
    if (!vm || !re || !subject || !subject->data) return BOOL_VAL(0);
    const char* base = subject->data;
    int offset = 0;
    Value rev = NIL_VAL;
    int parts = 0;
    while (offset <= subject->byte_len && parts < 1024) {
        regmatch_t m[1];
        int rc = regexec(re, base + offset, 1, m, 0);
        if (rc != 0 || m[0].rm_so < 0) break;
        int start = offset + (int)m[0].rm_so;
        int end = offset + (int)m[0].rm_eo;
        rev = vm_cons_value(vm, vm_string_value(vm, base + offset, start - offset), rev);
        parts++;
        if (end <= offset) break;
        offset = end;
    }
    rev = vm_cons_value(vm, vm_string_value(vm, base + offset, subject->byte_len - offset), rev);
    Value out = NIL_VAL;
    while (rev.type == VAL_PAIR) {
        Value car = vm->heap.objects[rev.as.ptr]->cons.car;
        out = vm_cons_value(vm, car, out);
        rev = vm->heap.objects[rev.as.ptr]->cons.cdr;
    }
    return out;
#else
    (void)vm; (void)re_ptr; (void)subject;
    return BOOL_VAL(0);
#endif
}

/** @brief Gregorian leap-year test. */
static int vm_time_is_leap_year(int year) {
    return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
}

/** @brief Number of days in a given month/year (accounting for leap years). */
static int vm_time_days_in_month(int year, int month) {
    static const int days[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (month < 1 || month > 12) return 0;
    if (month == 2 && vm_time_is_leap_year(year)) return 29;
    return days[month - 1];
}

/** @brief Portable replacement for timegm(): convert a UTC broken-down date/time
 *         to Unix seconds without depending on a non-standard libc function. */
static int64_t vm_timegm_portable(int year, int month, int day,
                                  int hour, int minute, int second) {
    static const int days_before_month[] = {
        0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334
    };
    int64_t days = 0;
    if (year >= 1970) {
        for (int y = 1970; y < year; y++)
            days += vm_time_is_leap_year(y) ? 366 : 365;
    } else {
        for (int y = year; y < 1970; y++)
            days -= vm_time_is_leap_year(y) ? 366 : 365;
    }
    days += days_before_month[month - 1];
    if (month > 2 && vm_time_is_leap_year(year)) days++;
    days += day - 1;
    return days * 86400LL + (int64_t)hour * 3600LL +
           (int64_t)minute * 60LL + (int64_t)second;
}

/** @brief Portable wall-clock time in nanoseconds since the Unix epoch
 *         (FILETIME on Windows, clock_gettime/gettimeofday on POSIX, 0 on WASM). */
static int64_t vm_current_time_ns_now(void) {
#if defined(ESHKOL_VM_WASM)
    return 0;
#elif defined(_WIN32)
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);
    uint64_t v = ((uint64_t)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
    v -= 116444736000000000ULL;
    return (int64_t)(v * 100ULL);
#else
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) == 0)
        return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
    struct timeval tv;
    if (gettimeofday(&tv, NULL) == 0)
        return (int64_t)tv.tv_sec * 1000000000LL + (int64_t)tv.tv_usec * 1000LL;
    return 0;
#endif
}

/** @brief Format a nanosecond (or, if the magnitude implies it, millisecond)
 *         epoch timestamp Value as an ISO-8601 UTC string with millisecond
 *         precision, e.g. "2026-07-08T12:00:00.000Z". */
static Value vm_format_iso8601_value(VM* vm, Value ns_val) {
    int64_t ns;
    if (ns_val.type == VAL_INT) {
        ns = ns_val.as.i;
    } else {
        double d = as_number(ns_val);
        if (fabs(d) >= 1000000000000000.0)
            ns = (int64_t)llround(d / 1000000.0) * 1000000LL;
        else
            ns = (int64_t)d;
    }
    time_t secs = (time_t)(ns / 1000000000LL);
    long ms = (long)((ns / 1000000LL) % 1000LL);
    if (ms < 0) {
        ms += 1000;
        secs -= 1;
    }

    struct tm tmv;
#if defined(_WIN32) && !defined(ESHKOL_VM_WASM)
    if (gmtime_s(&tmv, &secs) != 0) return BOOL_VAL(0);
#elif !defined(ESHKOL_VM_WASM)
    if (!gmtime_r(&secs, &tmv)) return BOOL_VAL(0);
#else
    memset(&tmv, 0, sizeof(tmv));
    tmv.tm_year = 70;
    tmv.tm_mday = 1;
#endif

    char buf[40];
    int n = snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d.%03ldZ",
                     tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday,
                     tmv.tm_hour, tmv.tm_min, tmv.tm_sec, ms);
    if (n <= 0 || n >= (int)sizeof(buf)) return BOOL_VAL(0);
    return vm_string_value(vm, buf, n);
}

/** @brief Parse an ISO-8601 date-time string (with optional fractional
 *         seconds and 'Z' or `+HH:MM`/`-HH:MM` timezone offset) into a
 *         nanosecond epoch timestamp Value, or #f if malformed or out of range. */
static Value vm_parse_iso8601_value(VM* vm, VmString* str) {
    (void)vm;
    if (!str || !str->data) return BOOL_VAL(0);
    const char* s = str->data;
    int year = 0, month = 0, day = 0, hour = 0, minute = 0, second = 0;
    int ms = 0;
    int n = 0;
    if (sscanf(s, "%4d-%2d-%2d%*[T ]%2d:%2d:%2d%n",
               &year, &month, &day, &hour, &minute, &second, &n) != 6)
        return BOOL_VAL(0);
    if (month < 1 || month > 12 || day < 1 ||
        day > vm_time_days_in_month(year, month) ||
        hour < 0 || hour > 23 || minute < 0 || minute > 59 ||
        second < 0 || second > 59)
        return BOOL_VAL(0);

    const char* p = s + n;
    if (*p == '.') {
        p++;
        int digits = 0;
        int frac = 0;
        while (*p >= '0' && *p <= '9' && digits < 9) {
            frac = frac * 10 + (*p - '0');
            digits++;
            p++;
        }
        if (digits == 0) return BOOL_VAL(0);
        while (digits < 3) { frac *= 10; digits++; }
        while (digits > 3) { frac /= 10; digits--; }
        ms = frac;
    }

    char tz_sign = 0;
    int tz_hour = 0, tz_minute = 0;
    if (*p == 'Z') {
        p++;
    } else if (*p == '+' || *p == '-') {
        tz_sign = *p++;
        int consumed = 0;
        if (sscanf(p, "%2d:%2d%n", &tz_hour, &tz_minute, &consumed) != 2)
            return BOOL_VAL(0);
        if (tz_hour < 0 || tz_hour > 23 || tz_minute < 0 || tz_minute > 59)
            return BOOL_VAL(0);
        p += consumed;
    }
    if (*p != '\0') return BOOL_VAL(0);

    int64_t secs = vm_timegm_portable(year, month, day, hour, minute, second);
    if (tz_sign == '+') secs -= (int64_t)tz_hour * 3600LL + (int64_t)tz_minute * 60LL;
    if (tz_sign == '-') secs += (int64_t)tz_hour * 3600LL + (int64_t)tz_minute * 60LL;
    return INT_VAL(secs * 1000000000LL + (int64_t)ms * 1000000LL);
}

/** @brief Format an elapsed-time duration as a short human-readable relative
 *         string ("5s ago", "3m ago", "2h ago", "1d ago"). */
static Value vm_format_relative_value(VM* vm, int64_t seconds_ago) {
    if (seconds_ago < 0) seconds_ago = 0;
    char buf[32];
    if (seconds_ago < 60)
        snprintf(buf, sizeof(buf), "%llds ago", (long long)seconds_ago);
    else if (seconds_ago < 3600)
        snprintf(buf, sizeof(buf), "%lldm ago", (long long)(seconds_ago / 60));
    else if (seconds_ago < 86400)
        snprintf(buf, sizeof(buf), "%lldh ago", (long long)(seconds_ago / 3600));
    else
        snprintf(buf, sizeof(buf), "%lldd ago", (long long)(seconds_ago / 86400));
    return vm_string_value(vm, buf, -1);
}

/** @brief Compute the local timezone's current UTC offset in seconds by
 *         comparing localtime() and gmtime() of "now" (0 on WASM). */
static int64_t vm_local_timezone_offset_seconds(void) {
#if defined(ESHKOL_VM_WASM)
    return 0;
#else
    time_t now = time(NULL);
    struct tm local_tm;
    struct tm utc_tm;
#if defined(_WIN32)
    if (localtime_s(&local_tm, &now) != 0 || gmtime_s(&utc_tm, &now) != 0)
        return 0;
#else
    if (!localtime_r(&now, &local_tm) || !gmtime_r(&now, &utc_tm))
        return 0;
#endif
    return (int64_t)difftime(mktime(&local_tm), mktime(&utc_tm));
#endif
}

typedef struct {
    const char* data;
    int len;
} VmLineSlice;

/** @brief Split a string into newline-delimited line slices (borrowed
 *         pointers into `s`'s data, no copying), up to `cap` lines. Returns
 *         the line count, or -1 if the line count would exceed `cap`. */
static int vm_split_lines(VmString* s, VmLineSlice* out, int cap) {
    if (!s || !s->data || cap <= 0) return -1;
    if (s->byte_len == 0) return 0;
    int count = 0;
    int start = 0;
    for (int i = 0; i <= s->byte_len; i++) {
        if (i == s->byte_len || s->data[i] == '\n') {
            if (count >= cap) return -1;
            out[count].data = s->data + start;
            out[count].len = i - start;
            count++;
            start = i + 1;
        }
    }
    return count;
}

/** @brief Byte-for-byte equality of two line slices. */
static int vm_line_equal(VmLineSlice a, VmLineSlice b) {
    return a.len == b.len && memcmp(a.data, b.data, (size_t)a.len) == 0;
}

/** @brief Build a new string consisting of `prefix` followed by `line`'s text
 *         (used to render diff output lines with a leading `+`/`-`/` `). */
static Value vm_prefixed_line_value(VM* vm, char prefix, VmLineSlice line) {
    char* buf = (char*)malloc((size_t)line.len + 2);
    if (!buf) return BOOL_VAL(0);
    buf[0] = prefix;
    if (line.len > 0) memcpy(buf + 1, line.data, (size_t)line.len);
    buf[line.len + 1] = '\0';
    Value v = vm_string_value(vm, buf, (int64_t)line.len + 1);
    free(buf);
    return v;
}

/** @brief Compute a line-based diff between `old_s` and `new_s` using classic
 *         dynamic-programming LCS (longest common subsequence) over up to
 *         MAX_LINES=256 lines each, returning a list of prefixed lines
 *         (`+`/`-`/` ` per vm_prefixed_line_value) reconstructed by
 *         backtracking through the DP table. Returns #f if either input
 *         exceeds the line cap. */
static Value vm_diff_lines_value(VM* vm, VmString* old_s, VmString* new_s) {
    enum { MAX_LINES = 256 };
    VmLineSlice old_lines[MAX_LINES];
    VmLineSlice new_lines[MAX_LINES];
    int n_old = vm_split_lines(old_s, old_lines, MAX_LINES);
    int n_new = vm_split_lines(new_s, new_lines, MAX_LINES);
    if (n_old < 0 || n_new < 0) return BOOL_VAL(0);

    int cols = n_new + 1;
    int cells = (n_old + 1) * (n_new + 1);
    int* dp = (int*)calloc((size_t)cells, sizeof(int));
    if (!dp) return BOOL_VAL(0);
    for (int i = 1; i <= n_old; i++) {
        for (int j = 1; j <= n_new; j++) {
            if (vm_line_equal(old_lines[i - 1], new_lines[j - 1]))
                dp[i * cols + j] = dp[(i - 1) * cols + (j - 1)] + 1;
            else {
                int a = dp[(i - 1) * cols + j];
                int b = dp[i * cols + (j - 1)];
                dp[i * cols + j] = a > b ? a : b;
            }
        }
    }

    int i = n_old;
    int j = n_new;
    Value rev = NIL_VAL;
    while (i > 0 || j > 0) {
        if (i > 0 && j > 0 && vm_line_equal(old_lines[i - 1], new_lines[j - 1])) {
            rev = vm_cons_value(vm, vm_prefixed_line_value(vm, '=', old_lines[i - 1]), rev);
            i--;
            j--;
        } else if (j > 0 && (i == 0 || dp[i * cols + (j - 1)] >= dp[(i - 1) * cols + j])) {
            rev = vm_cons_value(vm, vm_prefixed_line_value(vm, '+', new_lines[j - 1]), rev);
            j--;
        } else {
            rev = vm_cons_value(vm, vm_prefixed_line_value(vm, '-', old_lines[i - 1]), rev);
            i--;
        }
    }
    free(dp);
    return rev;
}

/** @brief Case-insensitive single-character equality. */
static int vm_fuzzy_char_equal(char a, char b) {
    return tolower((unsigned char)a) == tolower((unsigned char)b);
}

/** @brief Fuzzy-match `pattern` as a subsequence of `candidate` (fzf-style),
 *         scoring consecutive matches and matches at word boundaries (after
 *         `-`/`_`/space/`/` or at a lower-to-upper camelCase transition)
 *         higher, and penalizing gaps between matched characters. Returns
 *         a non-negative score, or -1 if `pattern` isn't a subsequence. */
static int vm_fuzzy_score(const char* pattern, const char* candidate) {
    if (!pattern || !candidate) return -1;
    if (!*pattern) return 0;
    int score = 0;
    int pi = 0;
    int last_match = -2;
    for (int ci = 0; candidate[ci] && pattern[pi]; ci++) {
        if (!vm_fuzzy_char_equal(pattern[pi], candidate[ci])) continue;
        score += 10;
        if (ci == last_match + 1) score += 5;
        if (ci == 0 || candidate[ci - 1] == '-' || candidate[ci - 1] == '_' ||
            candidate[ci - 1] == ' ' || candidate[ci - 1] == '/' ||
            (islower((unsigned char)candidate[ci - 1]) && isupper((unsigned char)candidate[ci])))
            score += 3;
        score -= ci - last_match - 1;
        last_match = ci;
        pi++;
    }
    return pattern[pi] ? -1 : score;
}

typedef struct {
    int score;
    Value candidate;
} VmFuzzyResult;

/** @brief qsort comparator for VmFuzzyResult, sorting by descending score. */
static int vm_fuzzy_result_cmp(const void* a, const void* b) {
    const VmFuzzyResult* ra = (const VmFuzzyResult*)a;
    const VmFuzzyResult* rb = (const VmFuzzyResult*)b;
    return rb->score - ra->score;
}

/** @brief Fuzzy-match `pattern` against a list of string `candidates` (up to
 *         1024), scoring each with vm_fuzzy_score(), sorting by descending
 *         score, and returning the top `max_results` as a list of
 *         `(score . candidate)` pairs. `key_fn` is currently unused. */
static Value vm_fuzzy_match_value(VM* vm, VmString* pattern, Value candidates, Value key_fn, int max_results) {
    (void)key_fn;
    if (!vm || !pattern || !pattern->data || max_results <= 0) return NIL_VAL;
    VmFuzzyResult results[1024];
    int n = 0;
    while (candidates.type == VAL_PAIR && n < (int)(sizeof(results) / sizeof(results[0]))) {
        HeapObject* pair = vm->heap.objects[candidates.as.ptr];
        Value candidate = pair->cons.car;
        VmString* s = vm_value_as_string(vm, candidate);
        if (s && s->data) {
            int score = vm_fuzzy_score(pattern->data, s->data);
            if (score >= 0) {
                results[n].score = score;
                results[n].candidate = candidate;
                n++;
            }
        }
        candidates = pair->cons.cdr;
    }
    qsort(results, (size_t)n, sizeof(results[0]), vm_fuzzy_result_cmp);
    if (max_results > n) max_results = n;
    Value out = NIL_VAL;
    for (int i = max_results - 1; i >= 0; i--) {
        Value entry = vm_cons_value(vm, INT_VAL((int64_t)results[i].score), results[i].candidate);
        out = vm_cons_value(vm, entry, out);
    }
    return out;
}

typedef struct {
    int major;
    int minor;
    int patch;
    char prerelease[128];
    char build[128];
} VmSemver;

/** @brief Parse a run of decimal digits at `*p` into `*out`, advancing `*p`.
 *         Returns 0 (no advance) if there's no leading digit. */
static int vm_parse_uint_component(const char** p, int* out) {
    if (!p || !*p || !isdigit((unsigned char)**p)) return 0;
    int value = 0;
    while (isdigit((unsigned char)**p)) {
        value = value * 10 + (**p - '0');
        (*p)++;
    }
    *out = value;
    return 1;
}

/** @brief Parse a semantic version string ("vMAJOR.MINOR.PATCH[-prerelease][+build]")
 *         into a VmSemver struct. Returns 0 on any malformed input. */
static int vm_semver_parse_cstr(const char* s, VmSemver* out) {
    if (!s || !out) return 0;
    memset(out, 0, sizeof(*out));
    const char* p = s;
    if (*p == 'v' || *p == 'V') p++;
    if (!vm_parse_uint_component(&p, &out->major) || *p++ != '.') return 0;
    if (!vm_parse_uint_component(&p, &out->minor) || *p++ != '.') return 0;
    if (!vm_parse_uint_component(&p, &out->patch)) return 0;
    if (*p == '-') {
        p++;
        int n = 0;
        while (*p && *p != '+' && n < (int)sizeof(out->prerelease) - 1) {
            out->prerelease[n++] = *p++;
        }
        out->prerelease[n] = '\0';
        if (n == 0) return 0;
    }
    if (*p == '+') {
        p++;
        int n = 0;
        while (*p && n < (int)sizeof(out->build) - 1) {
            out->build[n++] = *p++;
        }
        out->build[n] = '\0';
        if (n == 0) return 0;
    }
    return *p == '\0';
}

/** @brief True if a dot-separated prerelease identifier span is all digits
 *         (numeric identifiers sort before alphanumeric ones in semver). */
static int vm_semver_identifier_numeric(const char* s, int len) {
    if (len <= 0) return 0;
    for (int i = 0; i < len; i++)
        if (!isdigit((unsigned char)s[i])) return 0;
    return 1;
}

/** @brief Compare two semver prerelease strings per the SemVer 2.0 precedence
 *         rules: dot-separated identifiers compared left to right, numeric
 *         identifiers compared numerically and sorting before alphanumeric
 *         ones, a shorter identifier list with equal leading fields sorts
 *         lower, and no prerelease (empty string) outranks any prerelease. */
static int vm_semver_compare_prerelease(const char* a, const char* b) {
    if (!a[0] && !b[0]) return 0;
    if (!a[0]) return 1;
    if (!b[0]) return -1;
    const char* pa = a;
    const char* pb = b;
    while (*pa || *pb) {
        const char* ea = strchr(pa, '.');
        const char* eb = strchr(pb, '.');
        int la = ea ? (int)(ea - pa) : (int)strlen(pa);
        int lb = eb ? (int)(eb - pb) : (int)strlen(pb);
        if (la == 0 && lb == 0) return 0;
        if (la == 0) return -1;
        if (lb == 0) return 1;
        int na = vm_semver_identifier_numeric(pa, la);
        int nb = vm_semver_identifier_numeric(pb, lb);
        if (na && nb) {
            long va = 0, vb = 0;
            for (int i = 0; i < la; i++) va = va * 10 + (pa[i] - '0');
            for (int i = 0; i < lb; i++) vb = vb * 10 + (pb[i] - '0');
            if (va < vb) return -1;
            if (va > vb) return 1;
        } else if (na != nb) {
            return na ? -1 : 1;
        } else {
            int cmp = strncmp(pa, pb, (size_t)(la < lb ? la : lb));
            if (cmp < 0) return -1;
            if (cmp > 0) return 1;
            if (la < lb) return -1;
            if (la > lb) return 1;
        }
        pa = ea ? ea + 1 : pa + la;
        pb = eb ? eb + 1 : pb + lb;
    }
    return 0;
}

/** @brief Full semver comparison: major, then minor, then patch, then prerelease
 *         precedence. Returns -1/0/1. */
static int vm_semver_compare_structs(const VmSemver* a, const VmSemver* b) {
    if (a->major != b->major) return a->major < b->major ? -1 : 1;
    if (a->minor != b->minor) return a->minor < b->minor ? -1 : 1;
    if (a->patch != b->patch) return a->patch < b->patch ? -1 : 1;
    return vm_semver_compare_prerelease(a->prerelease, b->prerelease);
}

/** @brief Parse a semver string and return it as an association list of
 *         major/minor/patch/prerelease/build fields, or #f if malformed. */
static Value vm_semver_parse_value(VM* vm, VmString* s) {
    VmSemver v;
    if (!s || !s->data || !vm_semver_parse_cstr(s->data, &v)) return BOOL_VAL(0);
    Value result = NIL_VAL;
    result = vm_cons_value(vm, vm_alist_entry(vm, "build", vm_string_value(vm, v.build, -1)), result);
    result = vm_cons_value(vm, vm_alist_entry(vm, "prerelease", vm_string_value(vm, v.prerelease, -1)), result);
    result = vm_cons_value(vm, vm_alist_entry(vm, "patch", INT_VAL((int64_t)v.patch)), result);
    result = vm_cons_value(vm, vm_alist_entry(vm, "minor", INT_VAL((int64_t)v.minor)), result);
    result = vm_cons_value(vm, vm_alist_entry(vm, "major", INT_VAL((int64_t)v.major)), result);
    return result;
}

/** @brief Parse and compare two semver strings, returning -1/0/1 (or #f if
 *         either fails to parse). */
static Value vm_semver_compare_value(VM* vm, VmString* a_s, VmString* b_s) {
    (void)vm;
    VmSemver a, b;
    if (!a_s || !b_s || !vm_semver_parse_cstr(a_s->data, &a) ||
        !vm_semver_parse_cstr(b_s->data, &b))
        return BOOL_VAL(0);
    int cmp = vm_semver_compare_structs(&a, &b);
    if (cmp < 0) return INT_VAL(-1);
    if (cmp > 0) return INT_VAL(1);
    return INT_VAL(0);
}

/** @brief Test whether `version` satisfies a single npm-style semver range
 *         expression: comparison operators (`=`,`>`,`>=`,`<`,`<=`), caret
 *         ranges (`^x.y.z` — compatible within the same major version), or
 *         tilde ranges (`~x.y.z` — compatible within the same minor version). */
static int vm_semver_satisfies_range(const VmSemver* version, const char* range) {
    while (*range && isspace((unsigned char)*range)) range++;
    char op[3] = "";
    if ((range[0] == '>' || range[0] == '<' || range[0] == '=') && range[1] == '=') {
        op[0] = range[0];
        op[1] = '=';
        range += 2;
    } else if (*range == '>' || *range == '<' || *range == '=') {
        op[0] = *range++;
    } else if (*range == '^' || *range == '~') {
        op[0] = *range++;
    }
    while (*range && isspace((unsigned char)*range)) range++;

    VmSemver target;
    if (!vm_semver_parse_cstr(range, &target)) return 0;
    int cmp = vm_semver_compare_structs(version, &target);
    if (!op[0] || op[0] == '=') return cmp == 0;
    if (strcmp(op, ">=") == 0) return cmp >= 0;
    if (strcmp(op, "<=") == 0) return cmp <= 0;
    if (strcmp(op, ">") == 0) return cmp > 0;
    if (strcmp(op, "<") == 0) return cmp < 0;
    if (op[0] == '^') {
        if (cmp < 0) return 0;
        VmSemver upper = target;
        upper.major++;
        upper.minor = 0;
        upper.patch = 0;
        upper.prerelease[0] = '\0';
        return vm_semver_compare_structs(version, &upper) < 0;
    }
    if (op[0] == '~') {
        if (cmp < 0) return 0;
        VmSemver upper = target;
        upper.minor++;
        upper.patch = 0;
        upper.prerelease[0] = '\0';
        return vm_semver_compare_structs(version, &upper) < 0;
    }
    return 0;
}

/** @brief Parse `version_s` and test it against range expression `range_s`,
 *         returning #t/#f (or #f if `version_s` fails to parse). */
static Value vm_semver_satisfies_value(VM* vm, VmString* version_s, VmString* range_s) {
    (void)vm;
    VmSemver version;
    if (!version_s || !range_s || !vm_semver_parse_cstr(version_s->data, &version))
        return BOOL_VAL(0);
    return BOOL_VAL(vm_semver_satisfies_range(&version, range_s->data));
}

/** @brief Allocate a free slot in the VM's line_readers table for buffered
 *         async line-at-a-time reading of file descriptor `fd`. Returns the
 *         slot index (handle), or -1 if no slot is free. */
static int vm_line_reader_start(VM* vm, int fd) {
    if (!vm || fd < 0) return -1;
    for (int i = 1; i < (int)(sizeof(vm->line_readers) / sizeof(vm->line_readers[0])); i++) {
        if (vm->line_readers[i].active) continue;
        vm->line_readers[i].active = 1;
        vm->line_readers[i].fd = fd;
        vm->line_readers[i].len = 0;
        return i;
    }
    return -1;
}

/** @brief Non-blocking poll of a line reader: returns the next complete line
 *         already buffered, otherwise attempts a non-blocking read() to pull
 *         in more data and re-checks for a newline; if the internal buffer
 *         fills without a newline, flushes it as a line anyway. Returns #f
 *         if no line is available yet. */
static Value vm_line_reader_poll_value(VM* vm, int handle) {
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
    if (!vm || handle <= 0 ||
        handle >= (int)(sizeof(vm->line_readers) / sizeof(vm->line_readers[0])) ||
        !vm->line_readers[handle].active)
        return BOOL_VAL(0);
    int len = vm->line_readers[handle].len;
    for (int i = 0; i < len; i++) {
        if (vm->line_readers[handle].buffer[i] == '\n') {
            Value line = vm_string_value(vm, vm->line_readers[handle].buffer, i);
            int remaining = len - i - 1;
            if (remaining > 0)
                memmove(vm->line_readers[handle].buffer,
                        vm->line_readers[handle].buffer + i + 1,
                        (size_t)remaining);
            vm->line_readers[handle].len = remaining;
            return line;
        }
    }

    int fd = vm->line_readers[handle].fd;
    int old_flags = fcntl(fd, F_GETFL, 0);
    if (old_flags < 0) return BOOL_VAL(0);
    if (fcntl(fd, F_SETFL, old_flags | O_NONBLOCK) != 0) return BOOL_VAL(0);
    char chunk[512];
    ssize_t n = read(fd, chunk, sizeof(chunk));
    (void)fcntl(fd, F_SETFL, old_flags);
    if (n <= 0) return BOOL_VAL(0);
    int space = (int)sizeof(vm->line_readers[handle].buffer) - len;
    if (n > space) n = space;
    if (n > 0) {
        memcpy(vm->line_readers[handle].buffer + len, chunk, (size_t)n);
        len += (int)n;
        vm->line_readers[handle].len = len;
    }
    for (int i = 0; i < len; i++) {
        if (vm->line_readers[handle].buffer[i] == '\n') {
            Value line = vm_string_value(vm, vm->line_readers[handle].buffer, i);
            int remaining = len - i - 1;
            if (remaining > 0)
                memmove(vm->line_readers[handle].buffer,
                        vm->line_readers[handle].buffer + i + 1,
                        (size_t)remaining);
            vm->line_readers[handle].len = remaining;
            return line;
        }
    }
    if (len >= (int)sizeof(vm->line_readers[handle].buffer)) {
        Value line = vm_string_value(vm, vm->line_readers[handle].buffer, len);
        vm->line_readers[handle].len = 0;
        return line;
    }
#else
    (void)vm; (void)handle;
#endif
    return BOOL_VAL(0);
}

static int vm_values_equal_deep(VM* vm, Value a, Value b, int depth);

/** @brief True if `handle` refers to an active LRU cache in the VM's table. */
static int vm_lru_cache_valid(VM* vm, int handle) {
    return vm && handle > 0 &&
           handle < (int)(sizeof(vm->lru_caches) / sizeof(vm->lru_caches[0])) &&
           vm->lru_caches[handle].active;
}

/** @brief Linear-search a cache's entries for one matching `key` (deep
 *         equality). Returns the entry index, or -1 if not found. */
static int vm_lru_find_entry(VM* vm, int handle, Value key) {
    if (!vm_lru_cache_valid(vm, handle)) return -1;
    for (int i = 0; i < (int)(sizeof(vm->lru_caches[handle].entries) /
                              sizeof(vm->lru_caches[handle].entries[0])); i++) {
        if (vm->lru_caches[handle].entries[i].active &&
            vm_values_equal_deep(vm, vm->lru_caches[handle].entries[i].key, key, 0))
            return i;
    }
    return -1;
}

/** @brief Find a free entry slot in the cache, or if full, evict and return
 *         the slot with the oldest (smallest) access tick — the LRU victim. */
static int vm_lru_alloc_entry(VM* vm, int handle) {
    if (!vm_lru_cache_valid(vm, handle)) return -1;
    for (int i = 0; i < vm->lru_caches[handle].max_size; i++) {
        if (!vm->lru_caches[handle].entries[i].active)
            return i;
    }
    int victim = 0;
    int64_t oldest = vm->lru_caches[handle].entries[0].tick;
    for (int i = 1; i < vm->lru_caches[handle].max_size; i++) {
        if (vm->lru_caches[handle].entries[i].tick < oldest) {
            oldest = vm->lru_caches[handle].entries[i].tick;
            victim = i;
        }
    }
    return victim;
}

/** @brief Allocate a new LRU cache (capacity clamped to [1,64]) in a free
 *         slot of the VM's lru_caches table. Returns the handle, or -1 if
 *         no slot is free. */
static int vm_lru_create(VM* vm, int max_size) {
    if (!vm) return -1;
    if (max_size < 1) max_size = 1;
    if (max_size > 64) max_size = 64;
    for (int i = 1; i < (int)(sizeof(vm->lru_caches) / sizeof(vm->lru_caches[0])); i++) {
        if (vm->lru_caches[i].active) continue;
        memset(&vm->lru_caches[i], 0, sizeof(vm->lru_caches[i]));
        vm->lru_caches[i].active = 1;
        vm->lru_caches[i].max_size = max_size;
        vm->lru_caches[i].tick = 1;
        return i;
    }
    return -1;
}

/** @brief Look up `key` in the cache, bumping its access tick (marking it
 *         most-recently-used) on a hit. Returns #f on a miss. */
static Value vm_lru_get_value(VM* vm, int handle, Value key) {
    int idx = vm_lru_find_entry(vm, handle, key);
    if (idx < 0) return BOOL_VAL(0);
    vm->lru_caches[handle].entries[idx].tick = ++vm->lru_caches[handle].tick;
    return vm->lru_caches[handle].entries[idx].value;
}

/** @brief Insert or update `key`→`value` in the cache, evicting the LRU
 *         entry if the cache is full and `key` is new. Returns 1 on success,
 *         0 if the handle is invalid. */
static int vm_lru_set_value(VM* vm, int handle, Value key, Value value) {
    if (!vm_lru_cache_valid(vm, handle)) return 0;
    int idx = vm_lru_find_entry(vm, handle, key);
    if (idx < 0) {
        idx = vm_lru_alloc_entry(vm, handle);
        if (idx < 0) return 0;
        if (!vm->lru_caches[handle].entries[idx].active)
            vm->lru_caches[handle].size++;
    }
    vm->lru_caches[handle].entries[idx].active = 1;
    vm->lru_caches[handle].entries[idx].tick = ++vm->lru_caches[handle].tick;
    vm->lru_caches[handle].entries[idx].key = key;
    vm->lru_caches[handle].entries[idx].value = value;
    return 1;
}

/** @brief True if `handle` refers to an active event emitter in the VM's table. */
static int vm_event_emitter_valid(VM* vm, int handle) {
    return vm && handle > 0 &&
           handle < (int)(sizeof(vm->event_emitters) / sizeof(vm->event_emitters[0])) &&
           vm->event_emitters[handle].active;
}

/** @brief Allocate a new event emitter in a free slot of the VM's
 *         event_emitters table. Returns the handle, or -1 if none free. */
static int vm_event_emitter_create(VM* vm) {
    if (!vm) return -1;
    for (int i = 1; i < (int)(sizeof(vm->event_emitters) / sizeof(vm->event_emitters[0])); i++) {
        if (vm->event_emitters[i].active) continue;
        memset(&vm->event_emitters[i], 0, sizeof(vm->event_emitters[i]));
        vm->event_emitters[i].active = 1;
        return i;
    }
    return -1;
}

/** @brief Register `handler` (a closure) as a listener for `event` on the
 *         given emitter, optionally auto-removing after the first invocation
 *         (`once`). Returns 1 on success, 0 if the handle/handler is invalid
 *         or the listener table is full. */
static int vm_event_listener_add(VM* vm, int handle, Value event, Value handler, int once) {
    if (!vm_event_emitter_valid(vm, handle) || handler.type != VAL_CLOSURE)
        return 0;
    for (int i = 0; i < (int)(sizeof(vm->event_emitters[handle].listeners) /
                              sizeof(vm->event_emitters[handle].listeners[0])); i++) {
        if (vm->event_emitters[handle].listeners[i].active) continue;
        vm->event_emitters[handle].listeners[i].active = 1;
        vm->event_emitters[handle].listeners[i].once = once ? 1 : 0;
        vm->event_emitters[handle].listeners[i].event = event;
        vm->event_emitters[handle].listeners[i].handler = handler;
        return 1;
    }
    return 0;
}

/** @brief Remove all listeners on `handle` that match both `event` and
 *         `handler` (deep equality). Returns the number removed. */
static int vm_event_listener_remove(VM* vm, int handle, Value event, Value handler) {
    if (!vm_event_emitter_valid(vm, handle)) return 0;
    int removed = 0;
    for (int i = 0; i < (int)(sizeof(vm->event_emitters[handle].listeners) /
                              sizeof(vm->event_emitters[handle].listeners[0])); i++) {
        if (!vm->event_emitters[handle].listeners[i].active) continue;
        if (vm_values_equal_deep(vm, vm->event_emitters[handle].listeners[i].event, event, 0) &&
            vm_values_equal_deep(vm, vm->event_emitters[handle].listeners[i].handler, handler, 0)) {
            memset(&vm->event_emitters[handle].listeners[i], 0,
                   sizeof(vm->event_emitters[handle].listeners[i]));
            removed++;
        }
    }
    return removed;
}

/** @brief Copy up to `max_args` elements of a Scheme list into a flat C
 *         array, for passing as native call arguments. Returns the count copied. */
static int vm_event_args_from_list(VM* vm, Value args_list, Value* args, int max_args) {
    int argc = 0;
    Value cur = args_list;
    while (cur.type == VAL_PAIR && argc < max_args && is_valid_heap_ptr(vm, cur.as.ptr)) {
        HeapObject* node = vm->heap.objects[cur.as.ptr];
        if (!node || node->type != HEAP_CONS) break;
        args[argc++] = node->cons.car;
        cur = node->cons.cdr;
    }
    return argc;
}

/** @brief Synchronously invoke every listener registered for `event` on
 *         `handle`, passing `args_list`'s elements as call arguments, and
 *         removing any that were registered with `once`. Returns the number
 *         of listeners invoked, or -1 for an invalid handle. */
static int vm_event_emit(VM* vm, int handle, Value event, Value args_list) {
    if (!vm_event_emitter_valid(vm, handle)) return -1;
    Value args[16];
    int argc = vm_event_args_from_list(vm, args_list, args, 16);
    int invoked = 0;

    for (int i = 0; i < (int)(sizeof(vm->event_emitters[handle].listeners) /
                              sizeof(vm->event_emitters[handle].listeners[0])); i++) {
        if (!vm->event_emitters[handle].listeners[i].active) continue;
        if (!vm_values_equal_deep(vm, vm->event_emitters[handle].listeners[i].event, event, 0))
            continue;

        Value handler = vm->event_emitters[handle].listeners[i].handler;
        int once = vm->event_emitters[handle].listeners[i].once;
        if (once)
            memset(&vm->event_emitters[handle].listeners[i], 0,
                   sizeof(vm->event_emitters[handle].listeners[i]));
        (void)vm_call_closure_from_native(vm, handler, args, argc);
        invoked++;
    }
    return invoked;
}

/** @brief True if `handle` refers to an active bounded channel in the VM's table. */
static int vm_channel_valid(VM* vm, int handle) {
    return vm && handle > 0 &&
           handle < (int)(sizeof(vm->channels) / sizeof(vm->channels[0])) &&
           vm->channels[handle].active;
}

/** @brief Allocate a new fixed-capacity ring-buffer channel (capacity clamped
 *         to [1,64]) in a free slot of the VM's channels table. */
static int vm_channel_create(VM* vm, int capacity) {
    if (!vm) return -1;
    if (capacity < 1) capacity = 1;
    if (capacity > 64) capacity = 64;
    for (int i = 1; i < (int)(sizeof(vm->channels) / sizeof(vm->channels[0])); i++) {
        if (vm->channels[i].active) continue;
        memset(&vm->channels[i], 0, sizeof(vm->channels[i]));
        vm->channels[i].active = 1;
        vm->channels[i].capacity = capacity;
        return i;
    }
    return -1;
}

/** @brief Non-blocking send: push `value` onto the channel's ring buffer.
 *         Returns 0 (without blocking) if the channel is closed or full. */
static int vm_channel_send_value(VM* vm, int handle, Value value) {
    if (!vm_channel_valid(vm, handle) || vm->channels[handle].closed)
        return 0;
    if (vm->channels[handle].count >= vm->channels[handle].capacity)
        return 0;
    vm->channels[handle].buffer[vm->channels[handle].tail] = value;
    vm->channels[handle].tail =
        (vm->channels[handle].tail + 1) % vm->channels[handle].capacity;
    vm->channels[handle].count++;
    return 1;
}

/** @brief Non-blocking receive: pop the oldest value from the channel's ring
 *         buffer. Returns #f if empty. */
static Value vm_channel_receive_value(VM* vm, int handle) {
    if (!vm_channel_valid(vm, handle) || vm->channels[handle].count <= 0)
        return BOOL_VAL(0);
    Value value = vm->channels[handle].buffer[vm->channels[handle].head];
    vm->channels[handle].buffer[vm->channels[handle].head] = NIL_VAL;
    vm->channels[handle].head =
        (vm->channels[handle].head + 1) % vm->channels[handle].capacity;
    vm->channels[handle].count--;
    return value;
}

/** @brief True if `handle` refers to an active mutex in the VM's table. */
static int vm_mutex_valid(VM* vm, int handle) {
    return vm && handle > 0 &&
           handle < (int)(sizeof(vm->mutexes) / sizeof(vm->mutexes[0])) &&
           vm->mutexes[handle].active;
}

/** @brief Allocate a new cooperative (non-OS-level) recursive mutex in a free
 *         slot of the VM's mutexes table. Returns the handle, or -1 if none free. */
static int vm_mutex_create(VM* vm) {
    if (!vm) return -1;
    for (int i = 1; i < (int)(sizeof(vm->mutexes) / sizeof(vm->mutexes[0])); i++) {
        if (vm->mutexes[i].active) continue;
        memset(&vm->mutexes[i], 0, sizeof(vm->mutexes[i]));
        vm->mutexes[i].active = 1;
        return i;
    }
    return -1;
}

/** @brief Acquire the (cooperative, recursion-counted) mutex; always succeeds
 *         immediately since this VM is single-threaded per call. */
static int vm_mutex_lock_handle(VM* vm, int handle) {
    if (!vm_mutex_valid(vm, handle)) return 0;
    vm->mutexes[handle].locked = 1;
    vm->mutexes[handle].recursion++;
    return 1;
}

/** @brief Release one recursive acquisition of the mutex; clears the locked
 *         flag once the recursion count reaches zero. Returns 0 if not held. */
static int vm_mutex_unlock_handle(VM* vm, int handle) {
    if (!vm_mutex_valid(vm, handle) || vm->mutexes[handle].recursion <= 0)
        return 0;
    vm->mutexes[handle].recursion--;
    if (vm->mutexes[handle].recursion == 0)
        vm->mutexes[handle].locked = 0;
    return 1;
}

/** @brief True if `handle` refers to an active condition variable in the VM's table. */
static int vm_condvar_valid(VM* vm, int handle) {
    return vm && handle > 0 &&
           handle < (int)(sizeof(vm->condvars) / sizeof(vm->condvars[0])) &&
           vm->condvars[handle].active;
}

/** @brief Allocate a new condition variable in a free slot of the VM's
 *         condvars table. Returns the handle, or -1 if none free. */
static int vm_condvar_create(VM* vm) {
    if (!vm) return -1;
    for (int i = 1; i < (int)(sizeof(vm->condvars) / sizeof(vm->condvars[0])); i++) {
        if (vm->condvars[i].active) continue;
        memset(&vm->condvars[i], 0, sizeof(vm->condvars[i]));
        vm->condvars[i].active = 1;
        return i;
    }
    return -1;
}

/** @brief True if `handle` refers to an allocated timer in the VM's table. */
static int vm_timer_valid(VM* vm, int handle) {
    return vm && handle > 0 &&
           handle < (int)(sizeof(vm->timers) / sizeof(vm->timers[0])) &&
           vm->timers[handle].allocated;
}

/** @brief Register a one-shot or repeating timer that invokes `callback`
 *         after `delay_ms` (polled cooperatively by vm_timers_poll_due(),
 *         not driven by OS timer interrupts). Returns the handle, or -1 on
 *         invalid args / no free slot. */
static int vm_timer_create(VM* vm, int64_t delay_ms, Value callback, int repeating) {
    if (!vm || callback.type != VAL_CLOSURE) return -1;
    if (delay_ms < 0) delay_ms = 0;
    if (repeating && delay_ms <= 0) delay_ms = 1;
    for (int i = 1; i < (int)(sizeof(vm->timers) / sizeof(vm->timers[0])); i++) {
        if (vm->timers[i].allocated) continue;
        memset(&vm->timers[i], 0, sizeof(vm->timers[i]));
        vm->timers[i].allocated = 1;
        vm->timers[i].active = 1;
        vm->timers[i].repeating = repeating ? 1 : 0;
        vm->timers[i].interval_ms = delay_ms;
        vm->timers[i].next_due_ms = vm_monotonic_time_ms() + delay_ms;
        vm->timers[i].callback = callback;
        return i;
    }
    return -1;
}

/** @brief Cancel and free a timer's slot. */
static int vm_timer_cancel(VM* vm, int handle) {
    if (!vm_timer_valid(vm, handle)) return 0;
    memset(&vm->timers[handle], 0, sizeof(vm->timers[handle]));
    return 1;
}

/** @brief Scan all active timers and synchronously invoke the callback of
 *         any whose due time has passed, rescheduling repeating timers or
 *         deactivating one-shot ones. Re-entrancy-guarded via
 *         `vm->polling_timers` since callbacks may themselves trigger polling. */
static void vm_timers_poll_due(VM* vm) {
    if (!vm || vm->polling_timers) return;
    vm->polling_timers = 1;
    int64_t now = vm_monotonic_time_ms();
    for (int i = 1; i < (int)(sizeof(vm->timers) / sizeof(vm->timers[0])); i++) {
        if (!vm->timers[i].allocated || !vm->timers[i].active)
            continue;
        if (now < vm->timers[i].next_due_ms)
            continue;
        Value callback = vm->timers[i].callback;
        if (vm->timers[i].fired_count < INT32_MAX)
            vm->timers[i].fired_count++;
        if (vm->timers[i].repeating) {
            vm->timers[i].next_due_ms = now + vm->timers[i].interval_ms;
        } else {
            vm->timers[i].active = 0;
        }
        if (callback.type == VAL_CLOSURE)
            (void)vm_call_closure_from_native(vm, callback, NULL, 0);
    }
    vm->polling_timers = 0;
}

/** @brief Register a zero-argument closure to run on VM exit (like C's
 *         atexit()). Returns 0 if the handler table is full or `thunk` isn't
 *         a closure. */
static int vm_exit_handler_add(VM* vm, Value thunk) {
    if (!vm || thunk.type != VAL_CLOSURE ||
        vm->n_exit_handlers >= (int)(sizeof(vm->exit_handlers) / sizeof(vm->exit_handlers[0])))
        return 0;
    vm->exit_handlers[vm->n_exit_handlers++] = thunk;
    return 1;
}

/** @brief Run all registered exit handlers in LIFO order, exactly once (guarded
 *         by `exit_handlers_drained`). */
static void vm_run_exit_handlers(VM* vm) {
    if (!vm || vm->exit_handlers_drained) return;
    vm->exit_handlers_drained = 1;
    for (int i = vm->n_exit_handlers - 1; i >= 0; i--) {
        if (vm->exit_handlers[i].type == VAL_CLOSURE)
            (void)vm_call_closure_from_native(vm, vm->exit_handlers[i], NULL, 0);
    }
    vm->n_exit_handlers = 0;
}

/** @brief Look up a symbol by name in the running process's dynamic symbol
 *         table. POSIX uses dlsym(RTLD_DEFAULT); native Windows resolves the
 *         explicitly exported agent ABI from the executable module. */
static void* vm_runtime_symbol(const char* name) {
#if defined(ESHKOL_VM_WASM)
    (void)name;
    return NULL;
#elif defined(_WIN32)
    if (!name) return NULL;
    HMODULE self = GetModuleHandleW(NULL);
    return self ? (void*)(uintptr_t)GetProcAddress(self, name) : NULL;
#else
    if (!name) return NULL;
#ifdef RTLD_DEFAULT
    return dlsym(RTLD_DEFAULT, name);
#else
    static void* self_handle = NULL;
    if (!self_handle) self_handle = dlopen(NULL, RTLD_LAZY);
    return self_handle ? dlsym(self_handle, name) : NULL;
#endif
#endif
}

/** @brief Resolve the optional `eshkol_sqlite_exec` runtime hook, if the host
 *         binary links a SQLite bridge. */
static VmSqliteExecFn vm_sqlite_exec_symbol(void) {
    return (VmSqliteExecFn)(uintptr_t)vm_runtime_symbol("eshkol_sqlite_exec");
}

/** @brief Resolve the optional `eshkol_sqlite_last_insert_rowid` runtime hook. */
static VmSqliteLastInsertIdFn vm_sqlite_last_insert_id_symbol(void) {
    return (VmSqliteLastInsertIdFn)(uintptr_t)
        vm_runtime_symbol("eshkol_sqlite_last_insert_rowid");
}

/** @brief Resolve the optional `eshkol_sqlite_changes` runtime hook. */
static VmSqliteChangesFn vm_sqlite_changes_symbol(void) {
    return (VmSqliteChangesFn)(uintptr_t)vm_runtime_symbol("eshkol_sqlite_changes");
}

/** @brief True if `handle` refers to an active, non-NULL loaded dynamic library handle. */
static int vm_dlopen_valid(VM* vm, int handle) {
    return vm && handle > 0 &&
           handle < (int)(sizeof(vm->dynamic_libraries) / sizeof(vm->dynamic_libraries[0])) &&
           vm->dynamic_libraries[handle].active &&
           vm->dynamic_libraries[handle].handle;
}

/** @brief Store a native dlopen() handle in a free slot of the VM's
 *         dynamic_libraries table, returning its VM-level handle index. If
 *         the table is full, closes `handle` immediately and returns -1. */
static int vm_dlopen_store(VM* vm, void* handle) {
    if (!vm || !handle) return -1;
    for (int i = 1; i < (int)(sizeof(vm->dynamic_libraries) /
                              sizeof(vm->dynamic_libraries[0])); i++) {
        if (vm->dynamic_libraries[i].active) continue;
        vm->dynamic_libraries[i].active = 1;
        vm->dynamic_libraries[i].handle = handle;
        return i;
    }
#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM)
    dlclose(handle);
#endif
    return -1;
}

/** @brief dlclose() every still-active dynamic library owned by this VM
 *         (called on VM teardown). */
static void vm_dlopen_close_all(VM* vm) {
    if (!vm) return;
#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM)
    for (int i = 1; i < (int)(sizeof(vm->dynamic_libraries) /
                              sizeof(vm->dynamic_libraries[0])); i++) {
        if (vm->dynamic_libraries[i].active && vm->dynamic_libraries[i].handle)
            dlclose(vm->dynamic_libraries[i].handle);
        vm->dynamic_libraries[i].active = 0;
        vm->dynamic_libraries[i].handle = NULL;
    }
#else
    (void)vm;
#endif
}

/** @brief Append `len` bytes of `s` to the fixed-capacity buffer `out` at
 *         `*pos`, NUL-terminating, and advance `*pos`. Returns 0 (no write)
 *         if it would overflow `cap`. */
static int vm_format_append(char* out, size_t cap, size_t* pos, const char* s, size_t len) {
    if (!out || !pos || !s) return 0;
    if (*pos + len >= cap) return 0;
    memcpy(out + *pos, s, len);
    *pos += len;
    out[*pos] = '\0';
    return 1;
}

/** @brief vm_format_append() convenience wrapper for a NUL-terminated C string. */
static int vm_format_append_cstr(char* out, size_t cap, size_t* pos, const char* s) {
    return vm_format_append(out, cap, pos, s ? s : "", s ? strlen(s) : 0);
}

/** @brief Format a single Value according to a `~`-format directive character
 *         (`d` decimal, `x` hex, `f` float, `s` write-style quoted string,
 *         `a`/default display-style) and append the result to `out`. */
static int vm_format_append_value(VM* vm, char* out, size_t cap, size_t* pos,
                                  Value value, char directive) {
    char buf[128];
    switch (directive) {
    case 'd':
        snprintf(buf, sizeof(buf), "%lld", (long long)as_number(value));
        return vm_format_append_cstr(out, cap, pos, buf);
    case 'x':
        snprintf(buf, sizeof(buf), "%llx", (unsigned long long)(int64_t)as_number(value));
        return vm_format_append_cstr(out, cap, pos, buf);
    case 'f':
        snprintf(buf, sizeof(buf), "%.6g", as_number(value));
        return vm_format_append_cstr(out, cap, pos, buf);
    case 's':
        if (value.type == VAL_STRING) {
            VmString* s = vm_value_as_string(vm, value);
            if (!vm_format_append_cstr(out, cap, pos, "\"")) return 0;
            if (s && s->data &&
                !vm_format_append(out, cap, pos, s->data, (size_t)s->byte_len))
                return 0;
            return vm_format_append_cstr(out, cap, pos, "\"");
        }
        /* fall through: non-strings print as display values */
    case 'a':
    default:
        switch ((int)value.type) {
        case VAL_STRING:
        case VAL_SYMBOL: {
            VmString* s = vm_value_as_string(vm, value);
            return s && s->data &&
                   vm_format_append(out, cap, pos, s->data, (size_t)s->byte_len);
        }
        case VAL_INT:
            snprintf(buf, sizeof(buf), "%lld", (long long)value.as.i);
            return vm_format_append_cstr(out, cap, pos, buf);
        case VAL_FLOAT:
            snprintf(buf, sizeof(buf), "%.6g", value.as.f);
            return vm_format_append_cstr(out, cap, pos, buf);
        case VAL_BOOL:
            return vm_format_append_cstr(out, cap, pos, value.as.b ? "#t" : "#f");
        case VAL_NIL:
            return vm_format_append_cstr(out, cap, pos, "()");
        default:
            snprintf(buf, sizeof(buf), "#<value:%d>", (int)value.type);
            return vm_format_append_cstr(out, cap, pos, buf);
        }
    }
}

/** @brief Pop the next argument off a Scheme argument list, advancing `*args`
 *         to its tail. Returns 0 if the list is exhausted or malformed. */
static int vm_format_next_arg(VM* vm, Value* args, Value* out) {
    if (!vm || !args || !out || args->type != VAL_PAIR || !is_valid_heap_ptr(vm, args->as.ptr))
        return 0;
    HeapObject* cell = vm->heap.objects[args->as.ptr];
    if (!cell || cell->type != HEAP_CONS) return 0;
    *out = cell->cons.car;
    *args = cell->cons.cdr;
    return 1;
}

/** @brief Common Lisp/Scheme `format`-style directive interpreter: expands
 *         `~a`/`~s`/`~d`/`~x`/`~f` placeholders in `fmt` by consuming
 *         successive values from `args`, `~%` as a newline, and `~~` as a
 *         literal tilde, into a 16KB-capped output buffer. Returns #f on
 *         malformed format strings or buffer overflow. */
static Value vm_format_list_value(VM* vm, VmString* fmt, Value args) {
    if (!fmt || !fmt->data) return BOOL_VAL(0);
    char out[16384];
    size_t pos = 0;
    out[0] = '\0';
    for (int64_t i = 0; i < fmt->byte_len; i++) {
        char ch = fmt->data[i];
        if (ch != '~') {
            if (!vm_format_append(out, sizeof(out), &pos, &ch, 1)) return BOOL_VAL(0);
            continue;
        }
        if (++i >= fmt->byte_len) {
            if (!vm_format_append_cstr(out, sizeof(out), &pos, "~")) return BOOL_VAL(0);
            break;
        }
        char directive = fmt->data[i];
        if (directive == '~') {
            if (!vm_format_append_cstr(out, sizeof(out), &pos, "~")) return BOOL_VAL(0);
        } else if (directive == '%') {
            if (!vm_format_append_cstr(out, sizeof(out), &pos, "\n")) return BOOL_VAL(0);
        } else if (directive == 'a' || directive == 's' || directive == 'd' ||
                   directive == 'x' || directive == 'f') {
            Value arg = NIL_VAL;
            if (!vm_format_next_arg(vm, &args, &arg)) return BOOL_VAL(0);
            if (!vm_format_append_value(vm, out, sizeof(out), &pos, arg, directive))
                return BOOL_VAL(0);
        } else {
            if (!vm_format_append_cstr(out, sizeof(out), &pos, "~") ||
                !vm_format_append(out, sizeof(out), &pos, &directive, 1))
                return BOOL_VAL(0);
        }
    }
    return vm_string_value(vm, out, (int64_t)pos);
}

/** @brief True if `handle` refers to an active Yoga-style flexbox layout node. */
static int vm_yoga_valid(VM* vm, int handle) {
    return vm && handle > 0 &&
           handle < (int)(sizeof(vm->yoga_nodes) / sizeof(vm->yoga_nodes[0])) &&
           vm->yoga_nodes[handle].active;
}

/** @brief Compare a VmString against a C string literal for exact equality. */
static int vm_yoga_string_eq(VmString* s, const char* lit) {
    return s && s->data && lit && s->byte_len == (int64_t)strlen(lit) &&
           memcmp(s->data, lit, (size_t)s->byte_len) == 0;
}

typedef int64_t (*VmYogaCreateFn)(void);
typedef void (*VmYogaFreeFn)(int64_t);
typedef void (*VmYogaSetFloatFn)(int64_t, int32_t, double);
typedef void (*VmYogaSetIntFn)(int64_t, int32_t, int32_t);
typedef void (*VmYogaAddChildFn)(int64_t, int64_t, int32_t);
typedef void (*VmYogaCalculateFn)(int64_t, double, double);
typedef double (*VmYogaGetComputedFn)(int64_t, int32_t);

static int vm_yoga_enum_value(VmString* value, const char* const* names,
                              int count, int* out) {
    for (int i = 0; i < count; ++i) {
        if (vm_yoga_string_eq(value, names[i])) { *out = i; return 1; }
    }
    return 0;
}

static void vm_yoga_clear_subtree(VM* vm, int handle) {
    if (!vm_yoga_valid(vm, handle)) return;
    for (int i = 1; i < (int)(sizeof(vm->yoga_nodes) / sizeof(vm->yoga_nodes[0])); ++i) {
        if (vm->yoga_nodes[i].active && vm->yoga_nodes[i].parent == handle)
            vm_yoga_clear_subtree(vm, i);
    }
    memset(&vm->yoga_nodes[handle], 0, sizeof(vm->yoga_nodes[handle]));
}

/** @brief True if `handle` refers to an active listening HTTP server socket. */
static int vm_http_server_valid(VM* vm, int handle) {
    return vm && handle > 0 &&
           handle < (int)(sizeof(vm->http_servers) / sizeof(vm->http_servers[0])) &&
           vm->http_servers[handle].active;
}

/** @brief Register a listening socket fd as an HTTP server in a free slot of
 *         the VM's http_servers table (client_fd starts unset). Returns the
 *         handle, or -1 if none free. */
static int vm_http_server_store(VM* vm, int listen_fd, int port) {
    for (int i = 1; i < (int)(sizeof(vm->http_servers) / sizeof(vm->http_servers[0])); i++) {
        if (!vm->http_servers[i].active) {
            vm->http_servers[i].active = 1;
            vm->http_servers[i].listen_fd = listen_fd;
            vm->http_servers[i].client_fd = -1;
            vm->http_servers[i].port = port;
            return i;
        }
    }
    return -1;
}

/** @brief Parse a minimal `http://host[:port][/path]` URL into separate host,
 *         port (default 80), and path (default "/") buffers. Returns 0 on
 *         any malformed input or buffer-capacity overflow. */
static int vm_http_parse_url(VmString* url, char* host, size_t host_cap,
                             int* port, char* path, size_t path_cap) {
    if (!url || !url->data || !host || !port || !path || host_cap == 0 || path_cap == 0)
        return 0;
    const char* s = url->data;
    const char* prefix = "http://";
    size_t prefix_len = strlen(prefix);
    if ((size_t)url->byte_len <= prefix_len || memcmp(s, prefix, prefix_len) != 0)
        return 0;
    const char* host_start = s + prefix_len;
    const char* end = s + url->byte_len;
    const char* path_start = host_start;
    while (path_start < end && *path_start != '/') path_start++;
    const char* colon = NULL;
    for (const char* p = host_start; p < path_start; p++) {
        if (*p == ':') colon = p;
    }
    size_t host_len = (size_t)((colon ? colon : path_start) - host_start);
    if (host_len == 0 || host_len >= host_cap) return 0;
    memcpy(host, host_start, host_len);
    host[host_len] = '\0';
    *port = 80;
    if (colon) {
        int parsed_port = atoi(colon + 1);
        if (parsed_port <= 0 || parsed_port > 65535) return 0;
        *port = parsed_port;
    }
    size_t path_len = (path_start < end) ? (size_t)(end - path_start) : 1;
    if (path_len >= path_cap) return 0;
    if (path_start < end) {
        memcpy(path, path_start, path_len);
        path[path_len] = '\0';
    } else {
        snprintf(path, path_cap, "/");
    }
    return 1;
}

/** @brief Resolve `host`/`port` via getaddrinfo and open a blocking TCP
 *         connection with a send/receive timeout, trying each resolved
 *         address in turn. Returns the connected fd, or -1 on failure
 *         (always -1 on WASM/Windows). */
static int vm_http_connect(const char* host, int port, int timeout_ms) {
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
    char port_buf[16];
    snprintf(port_buf, sizeof(port_buf), "%d", port);
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo* res = NULL;
    if (getaddrinfo(host, port_buf, &hints, &res) != 0) return -1;
    int fd = -1;
    for (struct addrinfo* ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        struct timeval tv;
        tv.tv_sec = timeout_ms > 0 ? timeout_ms / 1000 : 30;
        tv.tv_usec = timeout_ms > 0 ? (timeout_ms % 1000) * 1000 : 0;
        (void)setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, (socklen_t)sizeof(tv));
        (void)setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, (socklen_t)sizeof(tv));
        if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    return fd;
#else
    (void)host; (void)port; (void)timeout_ms;
    return -1;
#endif
}

/** @brief Build a `(status headers body)` list Value representing a parsed
 *         HTTP response. Returns #f if either string allocation fails. */
static Value vm_http_response_value(VM* vm, int status,
                                    const char* headers, int64_t headers_len,
                                    const char* body, int64_t body_len) {
    Value body_value = vm_string_value(vm, body ? body : "", body_len >= 0 ? body_len : 0);
    Value headers_value = vm_string_value(vm, headers ? headers : "", headers_len >= 0 ? headers_len : 0);
    if (body_value.type != VAL_STRING || headers_value.type != VAL_STRING)
        return BOOL_VAL(0);
    Value result = NIL_VAL;
    result = vm_cons_value(vm, body_value, result);
    result = vm_cons_value(vm, headers_value, result);
    result = vm_cons_value(vm, INT_VAL((int64_t)status), result);
    return result;
}

/** @brief Parse a `ws://host[:port][/path]` URL by rewriting it to an
 *         equivalent `http://` URL and delegating to vm_http_parse_url(). */
static int vm_ws_parse_url(VmString* url, char* host, size_t host_cap,
                           int* port, char* path, size_t path_cap) {
    if (!url || !url->data) return 0;
    const char* s = url->data;
    const char* prefix = "ws://";
    size_t prefix_len = strlen(prefix);
    if ((size_t)url->byte_len <= prefix_len || memcmp(s, prefix, prefix_len) != 0)
        return 0;
    VmString tmp = *url;
    tmp.data = (char*)url->data;
    tmp.byte_len = url->byte_len;
    char http_url[4096];
    int n = snprintf(http_url, sizeof(http_url), "http://%.*s",
                     (int)(url->byte_len - (int64_t)prefix_len), s + prefix_len);
    if (n <= 0 || n >= (int)sizeof(http_url)) return 0;
    tmp.data = http_url;
    tmp.byte_len = n;
    return vm_http_parse_url(&tmp, host, host_cap, port, path, path_cap);
}

/** @brief True if `handle` refers to an active, not-yet-closed WebSocket client. */
static int vm_ws_valid(VM* vm, int handle) {
    return vm && handle > 0 &&
           handle < (int)(sizeof(vm->websocket_clients) / sizeof(vm->websocket_clients[0])) &&
           vm->websocket_clients[handle].active &&
           !vm->websocket_clients[handle].closed;
}

/** @brief Register a connected socket fd as a WebSocket client in a free slot
 *         of the VM's websocket_clients table. Returns the handle, or -1 if
 *         none free. */
static int vm_ws_store(VM* vm, int fd) {
    for (int i = 1; i < (int)(sizeof(vm->websocket_clients) / sizeof(vm->websocket_clients[0])); i++) {
        if (!vm->websocket_clients[i].active) {
            vm->websocket_clients[i].active = 1;
            vm->websocket_clients[i].fd = fd;
            vm->websocket_clients[i].closed = 0;
            return i;
        }
    }
    return -1;
}

/** @brief Loop send() until all `len` bytes are written to `fd`. Returns 0 on
 *         any send failure. */
static int vm_ws_send_all(int fd, const void* data, size_t len) {
    const char* p = (const char*)data;
    size_t sent_total = 0;
    while (sent_total < len) {
        ssize_t n = send(fd, p + sent_total, len - sent_total, 0);
        if (n <= 0) return 0;
        sent_total += (size_t)n;
    }
    return 1;
}

/** @brief Encode and send one RFC 6455 WebSocket frame (FIN bit set, given
 *         opcode, RFC-mandated masked payload length encoding with a
 *         zero mask key) followed by the payload. */
static int vm_ws_send_frame(int fd, int opcode, const void* data, size_t len) {
    unsigned char header[14];
    int hlen = 0;
    header[hlen++] = (unsigned char)(0x80 | (opcode & 0x0f));
    if (len < 126) {
        header[hlen++] = (unsigned char)(0x80 | len);
    } else if (len <= 65535) {
        header[hlen++] = 0x80 | 126;
        header[hlen++] = (unsigned char)((len >> 8) & 0xff);
        header[hlen++] = (unsigned char)(len & 0xff);
    } else {
        header[hlen++] = 0x80 | 127;
        for (int i = 7; i >= 0; i--)
            header[hlen++] = (unsigned char)((len >> (i * 8)) & 0xff);
    }
    unsigned char mask[4] = {0, 0, 0, 0};
    memcpy(header + hlen, mask, sizeof(mask));
    hlen += 4;
    return vm_ws_send_all(fd, header, (size_t)hlen) &&
           (len == 0 || vm_ws_send_all(fd, data, len));
}

typedef int32_t (*VmTsAvailableFn)(void);
typedef int64_t (*VmTsParserNewFn)(const char*);
typedef void (*VmTsParserFreeFn)(int64_t);
typedef int64_t (*VmTsParseFn)(int64_t, const char*, int32_t);
typedef void (*VmTsTreeFreeFn)(int64_t);
typedef int32_t (*VmTsTreeRootFn)(int64_t, char*, int32_t);
typedef int32_t (*VmTsNodeInfoFn)(int64_t, uint32_t, uint32_t, char*, int32_t);
typedef int32_t (*VmTsNodeChildrenFn)(int64_t, uint32_t, uint32_t,
                                     char*, int32_t, int32_t*);
typedef int32_t (*VmTsNodeTextFn)(int64_t, uint32_t, uint32_t, char*, int32_t);
typedef int64_t (*VmTsQueryNewFn)(const char*, const char*);
typedef void (*VmTsQueryFreeFn)(int64_t);
typedef int32_t (*VmTsQueryMatchesFn)(int64_t, int64_t, int32_t,
                                     char*, int32_t, int32_t*);

/** The VM deliberately resolves the same pinned native Tree-sitter bridge as
 * LLVM/JIT and AOT.  There is no line-oriented approximation: if the real
 * capability is not linked, ts-available reports false and constructors fail. */
static void* vm_ts_symbol(const char* name) {
    return vm_runtime_symbol(name);
}

static char* vm_ts_string_copy(const VmString* string) {
    if (!string || !string->data || string->byte_len < 0 ||
        string->byte_len > INT32_MAX)
        return NULL;
    char* copy = (char*)malloc((size_t)string->byte_len + 1u);
    if (!copy) return NULL;
    if (string->byte_len > 0)
        memcpy(copy, string->data, (size_t)string->byte_len);
    copy[string->byte_len] = '\0';
    return copy;
}

static int vm_ts_split_tabs(char* record, char** fields, int capacity) {
    int count = 0;
    if (!record || !fields || capacity <= 0) return 0;
    fields[count++] = record;
    for (char* cursor = record; *cursor && count < capacity; ++cursor) {
        if (*cursor == '\t') {
            *cursor = '\0';
            fields[count++] = cursor + 1;
        }
    }
    return count;
}

static int vm_ts_parse_node_record(char* record, int tree, int root,
                                   uint32_t* start_out, uint32_t* end_out,
                                   char* type_out, size_t type_capacity) {
    char* fields[9];
    (void)tree;
    (void)root;
    if (!record || !start_out || !end_out || !type_out || type_capacity == 0 ||
        vm_ts_split_tabs(record, fields, 9) != 9)
        return 0;
    char* endptr = NULL;
    unsigned long start = strtoul(fields[5], &endptr, 10);
    if (!endptr || *endptr) return 0;
    unsigned long finish = strtoul(fields[6], &endptr, 10);
    size_t type_len = strlen(fields[0]);
    if (!endptr || *endptr || start > UINT32_MAX || finish > UINT32_MAX ||
        start > finish || type_len == 0 || type_len >= type_capacity)
        return 0;
    *start_out = (uint32_t)start;
    *end_out = (uint32_t)finish;
    memcpy(type_out, fields[0], type_len + 1u);
    return 1;
}

static int vm_ts_store_node(VM* vm, int tree, int root, char* record) {
    if (!vm || tree <= 0 || tree >= (int)(sizeof(vm->ts_trees) / sizeof(vm->ts_trees[0])) ||
        !vm->ts_trees[tree].active)
        return -1;
    uint32_t start = 0;
    uint32_t end = 0;
    char type[128];
    if (!vm_ts_parse_node_record(record, tree, root, &start, &end,
                                 type, sizeof(type)))
        return -1;
    for (int i = 1; i < (int)(sizeof(vm->ts_nodes) / sizeof(vm->ts_nodes[0])); ++i) {
        if (!vm->ts_nodes[i].active) {
            vm->ts_nodes[i].active = 1;
            vm->ts_nodes[i].tree = tree;
            vm->ts_nodes[i].start = start;
            vm->ts_nodes[i].end = end;
            vm->ts_nodes[i].root = root;
            memcpy(vm->ts_nodes[i].type, type, strlen(type) + 1u);
            return i;
        }
    }
    return -1;
}

/** @brief Reverse a proper Scheme list, allocating a fresh chain of cons cells. */
static Value vm_reverse_list(VM* vm, Value list) {
    Value out = NIL_VAL;
    Value cur = list;
    while (cur.type == VAL_PAIR && is_valid_heap_ptr(vm, cur.as.ptr)) {
        HeapObject* cell = vm->heap.objects[cur.as.ptr];
        if (!cell || cell->type != HEAP_CONS) break;
        out = vm_cons_value(vm, cell->cons.car, out);
        cur = cell->cons.cdr;
    }
    return out;
}

/** Return the named children reported by the real pinned Tree-sitter runtime. */
static Value vm_ts_node_children_value(VM* vm, int node_handle) {
    if (!vm || node_handle <= 0 ||
        node_handle >= (int)(sizeof(vm->ts_nodes) / sizeof(vm->ts_nodes[0])) ||
        !vm->ts_nodes[node_handle].active)
        return BOOL_VAL(0);

    int tree_handle = vm->ts_nodes[node_handle].tree;
    if (tree_handle <= 0 ||
        tree_handle >= (int)(sizeof(vm->ts_trees) / sizeof(vm->ts_trees[0])) ||
        !vm->ts_trees[tree_handle].active)
        return BOOL_VAL(0);

    VmTsNodeChildrenFn children_fn = (VmTsNodeChildrenFn)(uintptr_t)
        vm_ts_symbol("eshkol_ts_node_children");
    if (!children_fn) return BOOL_VAL(0);
    uint32_t start = vm->ts_nodes[node_handle].root ? 0u : vm->ts_nodes[node_handle].start;
    uint32_t end = vm->ts_nodes[node_handle].root ? 0u : vm->ts_nodes[node_handle].end;
    int32_t count = 0;
    int32_t required = children_fn(tree_handle, start, end, NULL, 0, &count);
    if (required < 0 || count < 0) return BOOL_VAL(0);
    if (count == 0) return NIL_VAL;
    char* buffer = (char*)malloc((size_t)required + 1u);
    int* allocated = (int*)calloc((size_t)count, sizeof(int));
    if (!buffer || !allocated) {
        free(buffer);
        free(allocated);
        return BOOL_VAL(0);
    }
    int32_t written = children_fn(tree_handle, start, end, buffer,
                                  required + 1, &count);
    if (written != required) {
        free(buffer);
        free(allocated);
        return BOOL_VAL(0);
    }
    Value result = NIL_VAL;
    char* record = buffer;
    int ok = 1;
    int32_t stored = 0;
    for (int32_t i = 0; i < count; ++i) {
        size_t record_length = strlen(record);
        int child = vm_ts_store_node(vm, tree_handle, 0, record);
        if (child <= 0) { ok = 0; break; }
        allocated[stored++] = child;
        result = vm_cons_value(vm, INT_VAL((int64_t)child), result);
        if (vm->error) { ok = 0; break; }
        record += record_length + 1u;
    }
    if (!ok) {
        for (int32_t i = 0; i < stored; ++i)
            memset(&vm->ts_nodes[allocated[i]], 0, sizeof(vm->ts_nodes[allocated[i]]));
    }
    free(buffer);
    free(allocated);
    return ok ? vm_reverse_list(vm, result) : BOOL_VAL(0);
}

/** Build an association list for one exact Tree-sitter query capture. */
static Value vm_ts_match_value(VM* vm, int tree, char* record) {
    char* fields[7];
    if (vm_ts_split_tabs(record, fields, 7) != 7) return BOOL_VAL(0);
    char* endptr = NULL;
    unsigned long start_ul = strtoul(fields[1], &endptr, 10);
    if (!endptr || *endptr) return BOOL_VAL(0);
    unsigned long end_ul = strtoul(fields[2], &endptr, 10);
    if (!endptr || *endptr || start_ul > UINT32_MAX || end_ul > UINT32_MAX ||
        start_ul >= end_ul)
        return BOOL_VAL(0);
    uint32_t start = (uint32_t)start_ul;
    uint32_t end = (uint32_t)end_ul;
    VmTsNodeInfoFn info_fn = (VmTsNodeInfoFn)(uintptr_t)
        vm_ts_symbol("eshkol_ts_node_info");
    VmTsNodeTextFn text_fn = (VmTsNodeTextFn)(uintptr_t)
        vm_ts_symbol("eshkol_ts_node_text");
    if (!info_fn || !text_fn) return BOOL_VAL(0);

    int32_t info_required = info_fn(tree, start, end, NULL, 0);
    int32_t text_required = text_fn(tree, start, end, NULL, 0);
    if (info_required < 0 || text_required < 0) return BOOL_VAL(0);
    char* info = (char*)malloc((size_t)info_required + 1u);
    char* text = (char*)malloc((size_t)text_required + 1u);
    if (!info || !text) {
        free(info);
        free(text);
        return BOOL_VAL(0);
    }
    if (info_fn(tree, start, end, info, info_required + 1) != info_required ||
        text_fn(tree, start, end, text, text_required + 1) != text_required) {
        free(info);
        free(text);
        return BOOL_VAL(0);
    }
    uint32_t node_start = 0;
    uint32_t node_end = 0;
    char type[128];
    if (!vm_ts_parse_node_record(info, tree, 0, &node_start, &node_end,
                                 type, sizeof(type))) {
        free(info);
        free(text);
        return BOOL_VAL(0);
    }
    Value match = NIL_VAL;
    match = vm_cons_value(vm, vm_alist_entry(vm, "text", vm_string_value(vm, text, text_required)), match);
    match = vm_cons_value(vm, vm_alist_entry(vm, "end", INT_VAL((int64_t)end)), match);
    match = vm_cons_value(vm, vm_alist_entry(vm, "start", INT_VAL((int64_t)start)), match);
    match = vm_cons_value(vm, vm_alist_entry(vm, "type", vm_string_value(vm, type, -1)), match);
    match = vm_cons_value(vm, vm_alist_entry(vm, "capture", vm_string_value(vm, fields[0], -1)), match);
    free(info);
    free(text);
    return match;
}

static Value vm_ts_query_matches_value(VM* vm, int query, int tree) {
    if (!vm || query <= 0 ||
        query >= (int)(sizeof(vm->ts_queries) / sizeof(vm->ts_queries[0])) ||
        tree <= 0 || tree >= (int)(sizeof(vm->ts_trees) / sizeof(vm->ts_trees[0])) ||
        !vm->ts_queries[query].active || !vm->ts_trees[tree].active)
        return BOOL_VAL(0);
    VmTsQueryMatchesFn matches_fn = (VmTsQueryMatchesFn)(uintptr_t)
        vm_ts_symbol("eshkol_ts_query_matches");
    if (!matches_fn) return BOOL_VAL(0);
    int32_t count = 0;
    int32_t required = matches_fn(query, tree, 0, NULL, 0, &count);
    if (required < 0 || count < 0) return BOOL_VAL(0);
    if (count == 0) return NIL_VAL;
    char* buffer = (char*)malloc((size_t)required + 1u);
    if (!buffer) return BOOL_VAL(0);
    int32_t written = matches_fn(query, tree, 0, buffer, required + 1, &count);
    if (written != required) {
        free(buffer);
        return BOOL_VAL(0);
    }
    Value result = NIL_VAL;
    char* record = buffer;
    int ok = 1;
    for (int32_t i = 0; i < count; ++i) {
        size_t record_length = strlen(record);
        Value match = vm_ts_match_value(vm, tree, record);
        if (match.type == VAL_BOOL && !match.as.b) { ok = 0; break; }
        result = vm_cons_value(vm, match, result);
        if (vm->error) { ok = 0; break; }
        record += record_length + 1u;
    }
    free(buffer);
    return ok ? vm_reverse_list(vm, result) : BOOL_VAL(0);
}

/** @brief Look up `key` (by deep equality) in an association-list-shaped
 *         Value, writing its value into `*out`. Returns 1 on a hit. */
static int vm_json_alist_ref(VM* vm, Value obj, Value key, Value* out) {
    Value cur = obj;
    while (cur.type == VAL_PAIR && is_valid_heap_ptr(vm, cur.as.ptr)) {
        HeapObject* node = vm->heap.objects[cur.as.ptr];
        if (!node || node->type != HEAP_CONS) break;
        Value entry = node->cons.car;
        if (entry.type == VAL_PAIR && is_valid_heap_ptr(vm, entry.as.ptr)) {
            HeapObject* pair = vm->heap.objects[entry.as.ptr];
            if (pair && pair->type == HEAP_CONS &&
                vm_values_equal_deep(vm, pair->cons.car, key, 0)) {
                if (out) *out = pair->cons.cdr;
                return 1;
            }
        }
        cur = node->cons.cdr;
    }
    return 0;
}

/** @brief Look up the element at `index` in a proper list Value, writing it
 *         into `*out`. Returns 1 on a hit. */
static int vm_json_list_ref(VM* vm, Value obj, int index, Value* out) {
    if (index < 0) return 0;
    Value cur = obj;
    int i = 0;
    while (cur.type == VAL_PAIR && is_valid_heap_ptr(vm, cur.as.ptr)) {
        HeapObject* node = vm->heap.objects[cur.as.ptr];
        if (!node || node->type != HEAP_CONS) break;
        if (i == index) {
            if (out) *out = node->cons.car;
            return 1;
        }
        i++;
        cur = node->cons.cdr;
    }
    return 0;
}

/** @brief Walk a JSON-like Value (nested alists/lists/vectors) along a `path`
 *         list of keys/indices, returning the nested value found, or
 *         `default_value` if any step of the path doesn't resolve. */
static Value vm_json_get_in_value(VM* vm, Value obj, Value path, Value default_value) {
    Value current = obj;
    Value cur_path = path;
    while (cur_path.type == VAL_PAIR && is_valid_heap_ptr(vm, cur_path.as.ptr)) {
        HeapObject* node = vm->heap.objects[cur_path.as.ptr];
        if (!node || node->type != HEAP_CONS) return default_value;
        Value key = node->cons.car;
        Value next = NIL_VAL;
        int found = 0;

        if (current.type == VAL_PAIR) {
            if (key.type == VAL_INT || key.type == VAL_FLOAT)
                found = vm_json_list_ref(vm, current, (int)as_number(key), &next);
            if (!found)
                found = vm_json_alist_ref(vm, current, key, &next);
        } else if (current.type == VAL_VECTOR && is_valid_heap_ptr(vm, current.as.ptr) &&
                   (key.type == VAL_INT || key.type == VAL_FLOAT)) {
            VmVector* vec = (VmVector*)vm->heap.objects[current.as.ptr]->opaque.ptr;
            int idx = (int)as_number(key);
            if (vec && idx >= 0 && idx < vec->len) {
                next = vec->items[idx];
                found = 1;
            }
        }

        if (!found) return default_value;
        current = next;
        cur_path = node->cons.cdr;
    }
    return current;
}

/** @brief True if `key` is present as a key in association-list `alist`. */
static int vm_json_key_in_alist(VM* vm, Value alist, Value key) {
    Value ignored = NIL_VAL;
    return vm_json_alist_ref(vm, alist, key, &ignored);
}

/** @brief Shallow-merge two association lists: entries of `a` whose key
 *         appears in `b` are dropped, then all of `b`'s entries are
 *         appended, so `b` wins on key conflicts (up to 128 total entries). */
static Value vm_json_merge_value(VM* vm, Value a, Value b) {
    Value entries[128];
    int n = 0;
    Value cur = a;
    while (cur.type == VAL_PAIR && n < 128 && is_valid_heap_ptr(vm, cur.as.ptr)) {
        HeapObject* node = vm->heap.objects[cur.as.ptr];
        if (!node || node->type != HEAP_CONS) break;
        Value entry = node->cons.car;
        int overridden = 0;
        if (entry.type == VAL_PAIR && is_valid_heap_ptr(vm, entry.as.ptr)) {
            HeapObject* pair = vm->heap.objects[entry.as.ptr];
            if (pair && pair->type == HEAP_CONS)
                overridden = vm_json_key_in_alist(vm, b, pair->cons.car);
        }
        if (!overridden) entries[n++] = entry;
        cur = node->cons.cdr;
    }
    cur = b;
    while (cur.type == VAL_PAIR && n < 128 && is_valid_heap_ptr(vm, cur.as.ptr)) {
        HeapObject* node = vm->heap.objects[cur.as.ptr];
        if (!node || node->type != HEAP_CONS) break;
        entries[n++] = node->cons.car;
        cur = node->cons.cdr;
    }
    Value result = NIL_VAL;
    for (int i = n - 1; i >= 0; i--)
        result = vm_cons_value(vm, entries[i], result);
    return result;
}

typedef struct {
    char* data;
    size_t cap;
    size_t pos;
    int ok;
} VmJsonBuffer;

/** @brief Append a NUL-terminated string to a fixed-capacity VmJsonBuffer,
 *         marking `out->ok = 0` on overflow instead of writing past the end. */
static void vm_json_append(VmJsonBuffer* out, const char* s) {
    if (!out || !out->ok || !s) return;
    size_t len = strlen(s);
    if (out->pos + len >= out->cap) {
        out->ok = 0;
        return;
    }
    memcpy(out->data + out->pos, s, len);
    out->pos += len;
    out->data[out->pos] = '\0';
}

/** @brief Append one character to a VmJsonBuffer, marking `out->ok = 0` on overflow. */
static void vm_json_append_char(VmJsonBuffer* out, char c) {
    if (!out || !out->ok || out->pos + 1 >= out->cap) {
        if (out) out->ok = 0;
        return;
    }
    out->data[out->pos++] = c;
    out->data[out->pos] = '\0';
}

/** @brief Emit `level * indent` space characters for pretty-printed JSON indentation. */
static void vm_json_indent(VmJsonBuffer* out, int level, int indent) {
    for (int i = 0; i < level * indent; i++)
        vm_json_append_char(out, ' ');
}

static void vm_json_write_value(VM* vm, VmJsonBuffer* out, Value value, int level, int indent);

/** @brief Heuristic for JSON serialization: a non-empty proper list whose
 *         every element is a `(string-key . value)` pair is treated as a
 *         JSON object; otherwise (or if empty) it's serialized as an array. */
static int vm_json_list_is_object(VM* vm, Value list) {
    int saw = 0;
    Value cur = list;
    while (cur.type == VAL_PAIR && is_valid_heap_ptr(vm, cur.as.ptr)) {
        HeapObject* node = vm->heap.objects[cur.as.ptr];
        if (!node || node->type != HEAP_CONS) return 0;
        Value entry = node->cons.car;
        if (entry.type != VAL_PAIR || !is_valid_heap_ptr(vm, entry.as.ptr)) return 0;
        HeapObject* pair = vm->heap.objects[entry.as.ptr];
        if (!pair || pair->type != HEAP_CONS || pair->cons.car.type != VAL_STRING)
            return 0;
        saw = 1;
        cur = node->cons.cdr;
    }
    return saw && cur.type == VAL_NIL;
}

/** @brief Write a Scheme string Value as a JSON string literal, escaping
 *         `"`, `\`, and control characters (`\n`/`\r`/`\t` as short escapes,
 *         other control chars as `\u00XX`). */
static void vm_json_write_string(VM* vm, VmJsonBuffer* out, Value value) {
    VmString* s = vm_value_as_string(vm, value);
    vm_json_append_char(out, '"');
    if (s && s->data) {
        for (int64_t i = 0; i < s->byte_len; i++) {
            unsigned char c = (unsigned char)s->data[i];
            char tmp[8];
            switch (c) {
            case '"': vm_json_append(out, "\\\""); break;
            case '\\': vm_json_append(out, "\\\\"); break;
            case '\n': vm_json_append(out, "\\n"); break;
            case '\r': vm_json_append(out, "\\r"); break;
            case '\t': vm_json_append(out, "\\t"); break;
            default:
                if (c < 0x20) {
                    snprintf(tmp, sizeof(tmp), "\\u%04x", c);
                    vm_json_append(out, tmp);
                } else {
                    vm_json_append_char(out, (char)c);
                }
                break;
            }
        }
    }
    vm_json_append_char(out, '"');
}

/** @brief Serialize a Scheme list as a JSON array, recursing into
 *         vm_json_write_value() for each element and applying pretty-print
 *         indentation when `indent > 0`. */
static void vm_json_write_array(VM* vm, VmJsonBuffer* out, Value list, int level, int indent) {
    vm_json_append_char(out, '[');
    Value cur = list;
    int first = 1;
    while (cur.type == VAL_PAIR && is_valid_heap_ptr(vm, cur.as.ptr)) {
        HeapObject* node = vm->heap.objects[cur.as.ptr];
        if (!node || node->type != HEAP_CONS) break;
        if (!first) vm_json_append_char(out, ',');
        if (indent > 0) {
            vm_json_append_char(out, '\n');
            vm_json_indent(out, level + 1, indent);
        }
        vm_json_write_value(vm, out, node->cons.car, level + 1, indent);
        first = 0;
        cur = node->cons.cdr;
    }
    if (!first && indent > 0) {
        vm_json_append_char(out, '\n');
        vm_json_indent(out, level, indent);
    }
    vm_json_append_char(out, ']');
}

/** @brief Serialize an association list as a JSON object, writing each
 *         `(string-key . value)` pair as `"key": value`, with pretty-print
 *         indentation when `indent > 0`. */
static void vm_json_write_object(VM* vm, VmJsonBuffer* out, Value alist, int level, int indent) {
    vm_json_append_char(out, '{');
    Value cur = alist;
    int first = 1;
    while (cur.type == VAL_PAIR && is_valid_heap_ptr(vm, cur.as.ptr)) {
        HeapObject* node = vm->heap.objects[cur.as.ptr];
        if (!node || node->type != HEAP_CONS) break;
        Value entry = node->cons.car;
        if (entry.type == VAL_PAIR && is_valid_heap_ptr(vm, entry.as.ptr)) {
            HeapObject* pair = vm->heap.objects[entry.as.ptr];
            if (pair && pair->type == HEAP_CONS) {
                if (!first) vm_json_append_char(out, ',');
                if (indent > 0) {
                    vm_json_append_char(out, '\n');
                    vm_json_indent(out, level + 1, indent);
                }
                vm_json_write_string(vm, out, pair->cons.car);
                vm_json_append(out, indent > 0 ? ": " : ":");
                vm_json_write_value(vm, out, pair->cons.cdr, level + 1, indent);
                first = 0;
            }
        }
        cur = node->cons.cdr;
    }
    if (!first && indent > 0) {
        vm_json_append_char(out, '\n');
        vm_json_indent(out, level, indent);
    }
    vm_json_append_char(out, '}');
}

/** @brief Central JSON serialization dispatcher: maps nil→null, booleans,
 *         integers, floats (%.17g round-trip precision), strings (escaped),
 *         lists (object or array per vm_json_list_is_object()), and vectors
 *         (always arrays) to their JSON text form; anything else becomes "null". */
static void vm_json_write_value(VM* vm, VmJsonBuffer* out, Value value, int level, int indent) {
    char num[64];
    switch ((int)value.type) {
    case VAL_NIL:
        vm_json_append(out, "null");
        break;
    case VAL_BOOL:
        vm_json_append(out, value.as.b ? "true" : "false");
        break;
    case VAL_INT:
        snprintf(num, sizeof(num), "%lld", (long long)value.as.i);
        vm_json_append(out, num);
        break;
    case VAL_FLOAT:
        snprintf(num, sizeof(num), "%.17g", value.as.f);
        vm_json_append(out, num);
        break;
    case VAL_STRING:
    case VAL_SYMBOL:
        vm_json_write_string(vm, out, value);
        break;
    case VAL_PAIR:
        if (vm_json_list_is_object(vm, value))
            vm_json_write_object(vm, out, value, level, indent);
        else
            vm_json_write_array(vm, out, value, level, indent);
        break;
    case VAL_VECTOR: {
        if (!is_valid_heap_ptr(vm, value.as.ptr)) {
            vm_json_append(out, "null");
            break;
        }
        VmVector* vec = (VmVector*)vm->heap.objects[value.as.ptr]->opaque.ptr;
        vm_json_append_char(out, '[');
        for (int i = 0; vec && i < vec->len; i++) {
            if (i > 0) vm_json_append_char(out, ',');
            if (indent > 0) {
                vm_json_append_char(out, '\n');
                vm_json_indent(out, level + 1, indent);
            }
            vm_json_write_value(vm, out, vec->items[i], level + 1, indent);
        }
        if (vec && vec->len > 0 && indent > 0) {
            vm_json_append_char(out, '\n');
            vm_json_indent(out, level, indent);
        }
        vm_json_append_char(out, ']');
        break;
    }
    default:
        vm_json_append(out, "null");
        break;
    }
}

/** @brief Serialize `value` to a JSON string (indent clamped to [0,8] spaces
 *         per level; 0 means compact single-line output), into a 16KB-capped
 *         buffer. Returns #f on overflow. */
static Value vm_json_stringify_pretty_value(VM* vm, Value value, int indent) {
    char buf[16384];
    VmJsonBuffer out = {buf, sizeof(buf), 0, 1};
    buf[0] = '\0';
    if (indent < 0) indent = 0;
    if (indent > 8) indent = 8;
    vm_json_write_value(vm, &out, value, 0, indent);
    if (!out.ok) return BOOL_VAL(0);
    return vm_string_value(vm, out.data, (int64_t)out.pos);
}

/** @brief Resolve a named optional compression runtime hook (e.g. a
 *         gzip/zlib bridge), if the host binary links one. */
static VmCompressionAllocFn vm_compression_symbol(const char* name) {
    return (VmCompressionAllocFn)(uintptr_t)vm_runtime_symbol(name);
}

/** @brief True if the optional `eshkol_compression_available` runtime hook is
 *         linked and reports support. */
static int vm_compression_available(void) {
    typedef int32_t (*VmCompressionAvailableFn)(void);
    VmCompressionAvailableFn fn =
        (VmCompressionAvailableFn)(uintptr_t)
        vm_runtime_symbol("eshkol_compression_available");
    return fn && fn();
}

/** @brief Extract a raw byte pointer/length (bounded to int32_t) from a
 *         string or bytevector Value for passing to a compression FFI hook. */
static int vm_compression_input(VM* vm, Value value, const char** data, int32_t* len) {
    VmString* s = vm_value_as_string(vm, value);
    if (s && s->data && s->byte_len >= 0 && s->byte_len <= INT32_MAX) {
        *data = s->data;
        *len = (int32_t)s->byte_len;
        return 1;
    }
    VmBytevector* bv = vm_value_as_bytevector(vm, value);
    if (bv && bv->data && bv->len >= 0 && bv->len <= INT32_MAX) {
        *data = (const char*)bv->data;
        *len = (int32_t)bv->len;
        return 1;
    }
    return 0;
}

/** @brief Invoke the production allocating compression hook. The native zlib
 *         implementation grows decompression storage exactly as required and
 *         enforces the same 256 MiB bomb limit as the LLVM builtin adapter. */
static Value vm_compression_call(VM* vm, Value input,
                                 VmCompressionAllocFn fn) {
    const char* data = NULL;
    int32_t len = 0;
    VmCompressionFreeFn free_fn = (VmCompressionFreeFn)(uintptr_t)
        vm_runtime_symbol("eshkol_compression_free");
    if (!fn || !vm_compression_available() ||
        !free_fn || !vm_compression_input(vm, input, &data, &len))
        return BOOL_VAL(0);
    char* buf = NULL;
    int32_t n = 0;
    if (fn(data, len, 256 * 1024 * 1024, &buf, &n) != 0 || n < 0 || !buf) {
        if (buf) free_fn(buf);
        return BOOL_VAL(0);
    }
    VmBytevector* bv = vm_bv_make(&vm->heap.regions, n, 0);
    if (!bv) {
        free_fn(buf);
        return BOOL_VAL(0);
    }
    if (n > 0) memcpy(bv->data, buf, (size_t)n);
    free_fn(buf);
    int32_t ptr = heap_alloc(&vm->heap);
    if (ptr < 0) {
        vm->error = 1;
        return BOOL_VAL(0);
    }
    vm->heap.objects[ptr]->type = HEAP_BYTEVECTOR;
    vm->heap.objects[ptr]->opaque.ptr = bv;
    return (Value){.type = VAL_BYTEVECTOR, .as.ptr = ptr};
}

/** @brief Byte-level suffix test: does `s` end with `suffix`. */
static int vm_string_ends_with_bytes(VmString* s, VmString* suffix) {
    if (!s || !suffix || !s->data || !suffix->data) return 0;
    if (suffix->byte_len > s->byte_len) return 0;
    return memcmp(s->data + s->byte_len - suffix->byte_len,
                  suffix->data,
                  (size_t)suffix->byte_len) == 0;
}

/** @brief Find the character index of the first occurrence of `sub` in `s`
 *         at or after character index `start_idx`, decoding UTF-8 as it
 *         scans. Returns #f if not found (or if `start_idx` is out of range). */
static Value vm_string_index_of_value(VmString* s, VmString* sub, int64_t start_idx) {
    if (!s || !sub || !s->data || !sub->data) return BOOL_VAL(0);
    if (start_idx < 0) start_idx = 0;
    if (start_idx > s->char_len) return BOOL_VAL(0);
    if (sub->byte_len == 0) return INT_VAL(start_idx);
    int start_byte = vm_utf8_byte_offset(s->data, s->byte_len, (int)start_idx);
    if (start_byte < 0) start_byte = s->byte_len;
    for (int char_i = (int)start_idx, byte_i = start_byte;
         byte_i + sub->byte_len <= s->byte_len && char_i <= s->char_len;
         char_i++) {
        if (memcmp(s->data + byte_i, sub->data, (size_t)sub->byte_len) == 0)
            return INT_VAL(char_i);
        int next = byte_i;
        vm_utf8_decode(s->data, s->byte_len, &next);
        if (next <= byte_i) next = byte_i + 1;
        byte_i = next;
    }
    return BOOL_VAL(0);
}

/** @brief Pad `s` to `width` characters (clamped to 1M) by inserting the
 *         codepoint `cp` (default space if invalid) on the left or right per
 *         `left`. Returns `s` unchanged if it's already at least `width` chars. */
static Value vm_string_pad_value(VM* vm, VmString* s, int64_t width, int cp, int left) {
    if (!vm || !s || !s->data) return BOOL_VAL(0);
    if (width <= s->char_len) return vm_string_value(vm, s->data, s->byte_len);
    if (width > 1000000) width = 1000000;
    char enc[4];
    int enc_len = vm_utf8_encode(cp, enc);
    if (enc_len <= 0) {
        enc[0] = ' ';
        enc_len = 1;
    }
    int64_t pad_count = width - s->char_len;
    size_t out_len = (size_t)s->byte_len + (size_t)pad_count * (size_t)enc_len;
    char* out = (char*)malloc(out_len + 1);
    if (!out) return BOOL_VAL(0);
    char* p = out;
    if (!left) {
        memcpy(p, s->data, (size_t)s->byte_len);
        p += s->byte_len;
    }
    for (int64_t i = 0; i < pad_count; i++) {
        memcpy(p, enc, (size_t)enc_len);
        p += enc_len;
    }
    if (left) {
        memcpy(p, s->data, (size_t)s->byte_len);
        p += s->byte_len;
    }
    *p = '\0';
    Value result = vm_string_value(vm, out, (int64_t)out_len);
    free(out);
    return result;
}

#if defined(_WIN32) && !defined(ESHKOL_VM_WASM)
/** @brief Windows-only: append one argument to a CreateProcess command line
 *         buffer, quoting it and escaping embedded backslashes/quotes per
 *         the MSVCRT argv-parsing convention. Returns 0 on buffer overflow. */
static int vm_win_append_process_arg(char* out, size_t out_size, size_t* pos, const char* arg) {
    if (!out || !pos || !arg) return 0;

    if (*pos + 1 >= out_size) return 0;
    if (*pos > 0) out[(*pos)++] = ' ';

    if (*pos + 1 >= out_size) return 0;
    out[(*pos)++] = '"';

    size_t backslashes = 0;
    for (const char* p = arg; *p; ++p) {
        if (*p == '\\') {
            backslashes++;
            continue;
        }
        if (*p == '"') {
            while (backslashes > 0) {
                backslashes--;
                if (*pos + 2 >= out_size) return 0;
                out[(*pos)++] = '\\';
                out[(*pos)++] = '\\';
            }
            if (*pos + 2 >= out_size) return 0;
            out[(*pos)++] = '\\';
            out[(*pos)++] = '"';
            continue;
        }
        while (backslashes > 0) {
            backslashes--;
            if (*pos + 1 >= out_size) return 0;
            out[(*pos)++] = '\\';
        }
        if (*pos + 1 >= out_size) return 0;
        out[(*pos)++] = *p;
    }

    while (backslashes > 0) {
        backslashes--;
        if (*pos + 2 >= out_size) return 0;
        out[(*pos)++] = '\\';
        out[(*pos)++] = '\\';
    }

    if (*pos + 2 > out_size) return 0;
    out[(*pos)++] = '"';
    out[*pos] = '\0';
    return 1;
}

/** @brief Windows-only: build a full space-separated, quoted CreateProcess
 *         command line string from an argv array. */
static int vm_win_build_process_command_line(char* out, size_t out_size, char** argv, int argc) {
    if (!out || out_size == 0 || !argv || argc <= 0) return 0;
    size_t pos = 0;
    out[0] = '\0';
    for (int i = 0; i < argc; ++i) {
        if (!vm_win_append_process_arg(out, out_size, &pos, argv[i])) return 0;
    }
    return 1;
}
#endif

/** @brief Structural (deep) equality of two Values, recursing into pairs
 *         (with depth cap 128 against cycles/deep structures), comparing
 *         string contents byte-for-byte, treating int/float as
 *         numerically comparable, and unwrapping HEAP_FACT wrappers so a
 *         fact and its underlying value compare equal. Non-pair/string heap
 *         objects fall back to pointer identity. */
static int vm_values_equal_deep(VM* vm, Value a, Value b, int depth) {
    if (depth > 128) return 0;
    if (a.type != b.type) {
        if ((a.type == VAL_INT || a.type == VAL_FLOAT) &&
            (b.type == VAL_INT || b.type == VAL_FLOAT))
            return as_number(a) == as_number(b);
        return 0;
    }

    switch ((int)a.type) {
    case VAL_NIL:
        return 1;
    case VAL_INT:
        return a.as.i == b.as.i;
    case VAL_FLOAT:
        return a.as.f == b.as.f;
    case VAL_BOOL:
        return a.as.b == b.as.b;
    case VAL_STRING:
    case VAL_SYMBOL: {
        if (!is_valid_heap_ptr(vm, a.as.ptr) || !is_valid_heap_ptr(vm, b.as.ptr)) return 0;
        if (vm->heap.objects[a.as.ptr]->type != HEAP_STRING ||
            vm->heap.objects[b.as.ptr]->type != HEAP_STRING) return 0;
        VmString* sa = (VmString*)vm->heap.objects[a.as.ptr]->opaque.ptr;
        VmString* sb = (VmString*)vm->heap.objects[b.as.ptr]->opaque.ptr;
        return sa && sb && sa->byte_len == sb->byte_len &&
               memcmp(sa->data, sb->data, (size_t)sa->byte_len) == 0;
    }
    case VAL_PAIR: {
        if (!is_valid_heap_ptr(vm, a.as.ptr) || !is_valid_heap_ptr(vm, b.as.ptr)) return 0;
        HeapObject* oa = vm->heap.objects[a.as.ptr];
        HeapObject* ob = vm->heap.objects[b.as.ptr];
        if (oa->type == HEAP_FACT)
            return vm_values_equal_deep(vm, oa->cons.car, b, depth + 1);
        if (ob->type == HEAP_FACT)
            return vm_values_equal_deep(vm, a, ob->cons.car, depth + 1);
        if (oa->type != HEAP_CONS || ob->type != HEAP_CONS) return a.as.ptr == b.as.ptr;
        return vm_values_equal_deep(vm, oa->cons.car, ob->cons.car, depth + 1) &&
               vm_values_equal_deep(vm, oa->cons.cdr, ob->cons.cdr, depth + 1);
    }
    default:
        return a.as.ptr == b.as.ptr;
    }
}

#ifndef ESHKOL_VM_WASM
/** @brief Safety guard for recursive directory deletion: resolves `path` to
 *         its canonical form and rejects it if it matches a hardcoded list
 *         of critical system directories (`/`, `/usr`, `/etc`, `/Users`, etc). */
static int vm_directory_delete_forbidden_root(const char* path) {
    if (!path || !*path) return 1;

    char resolved[4096];
    const char* p = path;
    if (realpath(path, resolved)) p = resolved;

    static const char* forbidden[] = {
        "/", "/usr", "/bin", "/sbin", "/etc", "/var", "/home", "/Users",
        "/System", "/Library", "/Applications", "/private", "/private/tmp",
        NULL
    };

    for (int i = 0; forbidden[i]; i++) {
        if (strcmp(p, forbidden[i]) == 0) return 1;
    }
    return 0;
}

/** @brief POSIX recursive rm -rf: for a directory, recurses into every
 *         non-`.`/`..` entry (via lstat, so symlinks aren't followed into)
 *         then rmdir()s it; for anything else, unlink()s it directly.
 *         Depth-capped at 128 to guard against symlink cycles. */
static int vm_directory_delete_recursive_posix(const char* path, int depth) {
    if (!path || depth > 128) return 0;

    struct stat st;
    if (lstat(path, &st) != 0) return 0;

    if (S_ISDIR(st.st_mode) && !S_ISLNK(st.st_mode)) {
        DIR* dir = opendir(path);
        if (!dir) return 0;

        int ok = 1;
        struct dirent* ent;
        while ((ent = readdir(dir)) != NULL) {
            if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
                continue;

            char child[4096];
            int n = snprintf(child, sizeof(child), "%s/%s", path, ent->d_name);
            if (n <= 0 || n >= (int)sizeof(child)) {
                ok = 0;
                continue;
            }
            if (!vm_directory_delete_recursive_posix(child, depth + 1))
                ok = 0;
        }
        closedir(dir);

        if (rmdir(path) != 0) ok = 0;
        return ok;
    }

    return unlink(path) == 0;
}
#endif

/** @brief True if `c` is a path separator for the target platform (`/` or
 *         `\` on Windows, `/` only elsewhere). */
static int vm_path_is_separator(char c) {
#ifdef _WIN32
    return c == '/' || c == '\\';
#else
    return c == '/';
#endif
}

/** @brief The platform's canonical path separator character. */
static char vm_path_separator(void) {
#ifdef _WIN32
    return '\\';
#else
    return '/';
#endif
}

/** @brief Find the last path-separator character in `path`, or NULL if none. */
static const char* vm_path_last_separator(const char* path) {
    const char* last = NULL;
    if (!path) return NULL;
    for (const char* p = path; *p; ++p) {
        if (vm_path_is_separator(*p)) last = p;
    }
    return last;
}

/** @brief True if `path` is absolute in the platform's native sense (leading
 *         `/` on POSIX; drive-letter `C:\` or UNC `\\` prefix on Windows). */
static int vm_path_is_absolute_native(const char* path) {
    if (!path || path[0] == '\0') return 0;
#ifdef _WIN32
    size_t len = strlen(path);
    if (len >= 3 && isalpha((unsigned char)path[0]) &&
        path[1] == ':' && vm_path_is_separator(path[2])) {
        return 1;
    }
    return len >= 2 && vm_path_is_separator(path[0]) && vm_path_is_separator(path[1]);
#else
    return path[0] == '/';
#endif
}

/** @brief Read a byte range `[offset, offset+len)` of a file into a new
 *         bytevector via a memory-mapped read (mmap on POSIX,
 *         CreateFileMapping/MapViewOfFile on Windows, aligning the mapping to
 *         the platform's allocation granularity), avoiding a full-file
 *         buffered read for large files. `len < 0` means "to end of file".
 *         Returns NULL on any I/O error, out-of-range offset, or on WASM
 *         (unsupported). */
static VmBytevector* vm_file_mmap_copy_to_bytevector(VM* vm,
                                                     const char* path,
                                                     int64_t offset,
                                                     int64_t len) {
    if (!vm || !path || offset < 0) return NULL;

#if defined(ESHKOL_VM_WASM)
    (void)vm;
    (void)path;
    (void)offset;
    (void)len;
    return NULL;
#elif defined(_WIN32)
    HANDLE file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file == INVALID_HANDLE_VALUE) return NULL;

    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(file, &file_size) || file_size.QuadPart < 0 ||
        offset > file_size.QuadPart) {
        CloseHandle(file);
        return NULL;
    }

    int64_t available = file_size.QuadPart - offset;
    if (len < 0 || len > available) len = available;
    if (len < 0 || len > INT32_MAX) {
        CloseHandle(file);
        return NULL;
    }

    if (len == 0) {
        CloseHandle(file);
        return vm_bv_alloc(&vm->heap.regions, 0);
    }

    HANDLE mapping = CreateFileMappingA(file, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mapping) {
        CloseHandle(file);
        return NULL;
    }

    SYSTEM_INFO system_info;
    GetSystemInfo(&system_info);
    uint64_t granularity = system_info.dwAllocationGranularity
        ? (uint64_t)system_info.dwAllocationGranularity
        : 65536u;
    uint64_t offset_u = (uint64_t)offset;
    uint64_t aligned = offset_u - (offset_u % granularity);
    size_t delta = (size_t)(offset_u - aligned);
    uint64_t map_len_u = (uint64_t)delta + (uint64_t)len;
    if (map_len_u > (uint64_t)SIZE_MAX) {
        CloseHandle(mapping);
        CloseHandle(file);
        return NULL;
    }

    void* mapped = MapViewOfFile(mapping, FILE_MAP_READ,
                                 (DWORD)(aligned >> 32),
                                 (DWORD)(aligned & 0xffffffffu),
                                 (SIZE_T)map_len_u);
    if (!mapped) {
        CloseHandle(mapping);
        CloseHandle(file);
        return NULL;
    }

    VmBytevector* bv = vm_bv_alloc(&vm->heap.regions, (int)len);
    if (bv) {
        memcpy(bv->data, (const uint8_t*)mapped + delta, (size_t)len);
    }
    UnmapViewOfFile(mapped);
    CloseHandle(mapping);
    CloseHandle(file);
    return bv;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;

    struct stat st;
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) ||
        offset > (int64_t)st.st_size) {
        close(fd);
        return NULL;
    }

    int64_t available = (int64_t)st.st_size - offset;
    if (len < 0 || len > available) len = available;
    if (len < 0 || len > INT32_MAX) {
        close(fd);
        return NULL;
    }

    if (len == 0) {
        close(fd);
        return vm_bv_alloc(&vm->heap.regions, 0);
    }

    long page = sysconf(_SC_PAGE_SIZE);
    if (page <= 0) page = 4096;
    int64_t aligned = offset - (offset % page);
    size_t delta = (size_t)(offset - aligned);
    size_t map_len = delta + (size_t)len;

    void* mapped = mmap(NULL, map_len, PROT_READ, MAP_PRIVATE, fd, (off_t)aligned);
    close(fd);
    if (mapped == MAP_FAILED) return NULL;

    VmBytevector* bv = vm_bv_alloc(&vm->heap.regions, (int)len);
    if (bv) {
        memcpy(bv->data, (const uint8_t*)mapped + delta, (size_t)len);
    }
    munmap(mapped, map_len);
    return bv;
#endif
}

/** @brief Bounds-checked strcpy: copies `src` into `dst` only if it
 *         (including NUL) fits within `dst_len`. Returns 0 on overflow. */
static int vm_path_copy_cstr(char* dst, size_t dst_len, const char* src) {
    if (!dst || dst_len == 0 || !src) return 0;
    size_t len = strlen(src);
    if (len >= dst_len) return 0;
    memcpy(dst, src, len + 1);
    return 1;
}

/** @brief Normalize a filesystem path: splits on separators, resolves `.`
 *         and `..` components (dropping a preceding segment for each `..`),
 *         preserves platform-specific absolute-path prefixes (drive letters
 *         and UNC `\\` prefixes on Windows, leading `/` on POSIX), and
 *         rejoins with the platform separator. Falls back to "." for an
 *         empty result. Returns 0 on any buffer-capacity overflow. */
static int vm_path_normalize_cstr(const char* input, char* result, size_t result_len) {
    if (!input || !result || result_len == 0) return 0;

    char buf[4096];
    if (!vm_path_copy_cstr(buf, sizeof(buf), input)) return 0;

    char* parts[256];
    int nparts = 0;
    char prefix[8] = "";
    char* scan = buf;
    int absolute = vm_path_is_absolute_native(buf);

#ifdef _WIN32
    size_t buf_len = strlen(buf);
    if (buf_len >= 2 && isalpha((unsigned char)buf[0]) && buf[1] == ':') {
        prefix[0] = buf[0];
        prefix[1] = ':';
        prefix[2] = 0;
        scan = buf + 2;
        if (vm_path_is_separator(*scan)) scan++;
    } else if (buf_len >= 2 && vm_path_is_separator(buf[0]) && vm_path_is_separator(buf[1])) {
        prefix[0] = vm_path_separator();
        prefix[1] = vm_path_separator();
        prefix[2] = 0;
        scan = buf + 2;
    } else if (vm_path_is_separator(buf[0])) {
        scan = buf + 1;
    }
    char* tok = strtok(scan, "/\\");
#else
    if (buf[0] == '/') scan = buf + 1;
    char* tok = strtok(scan, "/");
#endif

    while (tok && nparts < 256) {
        if (strcmp(tok, ".") == 0 || strcmp(tok, "") == 0) {
            /* skip */
        } else if (strcmp(tok, "..") == 0) {
            if (nparts > 0) nparts--;
        } else {
            parts[nparts++] = tok;
        }
#ifdef _WIN32
        tok = strtok(NULL, "/\\");
#else
        tok = strtok(NULL, "/");
#endif
    }

    size_t pos = 0;
    if (prefix[0]) {
        size_t prefix_len = strlen(prefix);
        if (prefix_len >= result_len) return 0;
        memcpy(result + pos, prefix, prefix_len);
        pos += prefix_len;
    }
    if (absolute &&
        !(prefix[0] == vm_path_separator() &&
          prefix[1] == vm_path_separator() &&
          prefix[2] == 0)) {
        if (pos >= result_len - 1) return 0;
        result[pos++] = vm_path_separator();
    }
    for (int i = 0; i < nparts; i++) {
        if (i > 0) {
            if (pos >= result_len - 1) return 0;
            result[pos++] = vm_path_separator();
        }
        size_t len = strlen(parts[i]);
        if (pos + len >= result_len) return 0;
        memcpy(result + pos, parts[i], len);
        pos += len;
    }

    if (pos == 0) {
        if (result_len < 2) return 0;
        result[pos++] = '.';
    }
    result[pos] = 0;
    return 1;
}

/** @brief Split `path` in place (via strtok, destroying separators) into up
 *         to `max_parts` non-empty component pointers. Returns the count. */
static int vm_path_split_mut(char* path, char** parts, int max_parts) {
    int nparts = 0;
#ifdef _WIN32
    char* tok = strtok(path, "/\\");
#else
    char* tok = strtok(path, "/");
#endif
    while (tok && nparts < max_parts) {
        if (*tok) parts[nparts++] = tok;
#ifdef _WIN32
        tok = strtok(NULL, "/\\");
#else
        tok = strtok(NULL, "/");
#endif
    }
    return nparts;
}

/** @brief Compute the relative path from `from` to `to`: normalizes both,
 *         falls back to `to`'s absolute form if their absoluteness or
 *         (on Windows) drive/UNC-server differ, otherwise finds the longest
 *         common path-component prefix and emits one `..` per remaining
 *         `from` component followed by `to`'s remaining components. */
static int vm_path_relative_cstr(const char* from, const char* to, char* result, size_t result_len) {
    if (!from || !to || !result || result_len == 0) return 0;

    char from_norm[4096];
    char to_norm[4096];
    if (!vm_path_normalize_cstr(from, from_norm, sizeof(from_norm)) ||
        !vm_path_normalize_cstr(to, to_norm, sizeof(to_norm))) {
        return 0;
    }

    int from_abs = vm_path_is_absolute_native(from_norm);
    int to_abs = vm_path_is_absolute_native(to_norm);
    if (from_abs != to_abs) return vm_path_copy_cstr(result, result_len, to_norm);

    char from_buf[4096];
    char to_buf[4096];
    if (!vm_path_copy_cstr(from_buf, sizeof(from_buf), from_norm) ||
        !vm_path_copy_cstr(to_buf, sizeof(to_buf), to_norm)) {
        return 0;
    }

    char* from_parts[256];
    char* to_parts[256];
    int n_from = vm_path_split_mut(from_buf, from_parts, 256);
    int n_to = vm_path_split_mut(to_buf, to_parts, 256);

#ifdef _WIN32
    if (from_abs && to_abs) {
        if (n_from > 0 && n_to > 0 &&
            strchr(from_parts[0], ':') && strchr(to_parts[0], ':') &&
            _stricmp(from_parts[0], to_parts[0]) != 0) {
            return vm_path_copy_cstr(result, result_len, to_norm);
        }
        if (from_norm[0] == '\\' && from_norm[1] == '\\' &&
            to_norm[0] == '\\' && to_norm[1] == '\\' &&
            (n_from < 2 || n_to < 2 ||
             _stricmp(from_parts[0], to_parts[0]) != 0 ||
             _stricmp(from_parts[1], to_parts[1]) != 0)) {
            return vm_path_copy_cstr(result, result_len, to_norm);
        }
    }
#endif

    int common = 0;
    while (common < n_from && common < n_to &&
#ifdef _WIN32
           _stricmp(from_parts[common], to_parts[common]) == 0
#else
           strcmp(from_parts[common], to_parts[common]) == 0
#endif
    ) {
        common++;
    }

    size_t pos = 0;
    for (int i = common; i < n_from; i++) {
        if (pos > 0) {
            if (pos >= result_len - 1) return 0;
            result[pos++] = vm_path_separator();
        }
        if (pos + 2 >= result_len) return 0;
        result[pos++] = '.';
        result[pos++] = '.';
    }
    for (int i = common; i < n_to; i++) {
        if (pos > 0) {
            if (pos >= result_len - 1) return 0;
            result[pos++] = vm_path_separator();
        }
        size_t len = strlen(to_parts[i]);
        if (pos + len >= result_len) return 0;
        memcpy(result + pos, to_parts[i], len);
        pos += len;
    }

    if (pos == 0) {
        if (result_len < 2) return 0;
        result[pos++] = '.';
    }
    result[pos] = 0;
    return 1;
}

/** @brief Extract the underlying datum from a knowledge-base fact Value: for
 *         a HEAP_FACT wrapper, unwraps to its stored car; for a plain cons,
 *         returns it as-is. Returns 0 for anything else. */
static int vm_kb_extract_fact_datum(VM* vm, Value fact_val, Value* out) {
    if (!out) return 0;
    if (fact_val.type == VAL_PAIR && is_valid_heap_ptr(vm, fact_val.as.ptr)) {
        HeapObject* obj = vm->heap.objects[fact_val.as.ptr];
        if (obj->type == HEAP_FACT) {
            *out = obj->cons.car;
            return 1;
        }
        if (obj->type == HEAP_CONS) {
            *out = fact_val;
            return 1;
        }
    }
    return 0;
}

/** @brief Extract the datum Value stored in an internal VmFact struct, if present. */
static int vm_kb_stored_fact_datum(VM* vm, VmFact* fact, Value* out) {
    if (!fact || !fact->has_datum || !out) return 0;
    if (!is_valid_heap_ptr(vm, fact->datum_ptr)) return 0;
    *out = PAIR_VAL(fact->datum_ptr);
    return 1;
}

/** @brief True if a fact's datum is a `(predicate . args)` pair whose car
 *         deep-equals `predicate`. */
static int vm_kb_fact_predicate_matches(VM* vm, VmFact* fact, Value predicate) {
    Value datum;
    if (!vm_kb_stored_fact_datum(vm, fact, &datum)) return 0;
    if (datum.type != VAL_PAIR || !is_valid_heap_ptr(vm, datum.as.ptr)) return 0;
    HeapObject* obj = vm->heap.objects[datum.as.ptr];
    if (obj->type != HEAP_CONS) return 0;
    return vm_values_equal_deep(vm, obj->cons.car, predicate, 0);
}

/** @brief True if `value` is a string starting with `?`, the knowledge-base
 *         query pattern convention for an unbound logic variable. */
static int vm_kb_pattern_is_logic_var(VM* vm, Value value) {
    if (!vm || value.type != VAL_STRING || !is_valid_heap_ptr(vm, value.as.ptr))
        return 0;
    HeapObject* obj = vm->heap.objects[value.as.ptr];
    if (!obj || obj->type != HEAP_STRING || !obj->opaque.ptr) return 0;
    VmString* s = (VmString*)obj->opaque.ptr;
    return s && s->byte_len > 0 && s->data && s->data[0] == '?';
}

/** @brief Test whether a query `pattern` list matches a stored `fact` list
 *         element-by-element, treating `?`-prefixed pattern elements
 *         (vm_kb_pattern_is_logic_var) as wildcards that match anything,
 *         and requiring exact deep-equality elsewhere. Falls back to plain
 *         deep equality for non-list patterns/facts. */
static int vm_kb_datums_match(VM* vm, Value pattern, Value fact) {
    if (pattern.type == VAL_NIL) return 1;

    if (pattern.type == VAL_PAIR && fact.type == VAL_PAIR) {
        Value pc = pattern;
        Value fc = fact;
        while (pc.type == VAL_PAIR && fc.type == VAL_PAIR) {
            if (!is_valid_heap_ptr(vm, pc.as.ptr) || !is_valid_heap_ptr(vm, fc.as.ptr))
                return 0;
            HeapObject* po = vm->heap.objects[pc.as.ptr];
            HeapObject* fo = vm->heap.objects[fc.as.ptr];
            if (!po || !fo || po->type != HEAP_CONS || fo->type != HEAP_CONS)
                return 0;

            Value pe = po->cons.car;
            Value fe = fo->cons.car;
            if (!vm_kb_pattern_is_logic_var(vm, pe) &&
                !vm_values_equal_deep(vm, pe, fe, 0))
                return 0;

            pc = po->cons.cdr;
            fc = fo->cons.cdr;
        }
        return pc.type == VAL_NIL && fc.type == VAL_NIL;
    }

    return vm_values_equal_deep(vm, pattern, fact, 0);
}

/** @brief Query the terminal for the cursor's current row/column via the
 *         DSR (`\033[6n`) escape sequence: temporarily switches stdin to raw
 *         non-blocking mode, writes the query, polls (100ms timeout) for the
 *         `\033[row;colR` reply, and restores the terminal mode. Returns 0
 *         (with row/col left at 0) if stdin/stdout aren't ttys, the terminal
 *         doesn't respond, or on WASM/Windows. */
static int vm_query_terminal_cursor(int* row, int* col) {
    if (row) *row = 0;
    if (col) *col = 0;
#if defined(ESHKOL_VM_WASM) || defined(_WIN32)
    return 0;
#else
    if (!row || !col) return 0;
    if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) return 0;

    struct termios orig;
    int restore = 0;
    if (tcgetattr(STDIN_FILENO, &orig) == 0) {
        struct termios raw = orig;
        raw.c_lflag &= ~(ICANON | ECHO);
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 0;
        if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) == 0) restore = 1;
    }

    int ok = 0;
    char buf[32];
    int i = 0;
    if (write(STDOUT_FILENO, "\033[6n", 4) == 4) {
        while (i < (int)sizeof(buf) - 1) {
            struct pollfd pfd = { .fd = STDIN_FILENO, .events = POLLIN };
            if (poll(&pfd, 1, 100) <= 0) break;
            if (read(STDIN_FILENO, &buf[i], 1) != 1) break;
            if (buf[i++] == 'R') break;
        }
        buf[i] = '\0';
        ok = (sscanf(buf, "\033[%d;%dR", row, col) == 2) ||
             (sscanf(buf, "\033[%d;%d", row, col) == 2);
    }

    if (restore) tcsetattr(STDIN_FILENO, TCSANOW, &orig);
    if (!ok) {
        *row = 0;
        *col = 0;
    }
    return ok;
#endif
}

/** @brief Safely unwrap a VAL_AD_TAPE Value to its underlying AdTape*, or
 *         NULL if the value isn't a valid heap-backed AD tape. */
static AdTape* vm_ad_tape_from_value(VM* vm, Value tape_val) {
    if (tape_val.type != VAL_AD_TAPE) return NULL;
    if (!is_heap_type(vm, tape_val, HEAP_AD_TAPE)) return NULL;
    return (AdTape*)vm->heap.objects[tape_val.as.ptr]->opaque.ptr;
}

/* ESH-0226: centralized, type-checked tensor-operand unpack.
 *
 * Every tensor op below must route its tensor operand through this helper
 * instead of blindly reinterpreting the heap pointer as a VmTensor* (the
 * naive `(VmTensor*)vm->heap.objects[v.as.ptr]->opaque.ptr` pattern segfaults
 * whenever the operand is actually some other heap-allocated type). Mirrors
 * the LLVM-path invariant already enforced natively via
 * eshkol_tensor_operand_checked() (lib/core/runtime_tensor_alloc.cpp,
 * ESH-0069): a bare vector literal such as #(1.0 2.0 3.0 4.0) is a
 * first-class argument to tensor ops (matmul, reshape, ...) in documented
 * Eshkol programs, so it must be coerced to a fresh 1-D tensor rather than
 * rejected or, worse, misread as one.
 *
 *   (a) v is already a tensor -> return its VmTensor* unchanged (zero-copy).
 *   (b) v is a homogeneous numeric vector (every element int or float) ->
 *       coerce to a fresh 1-D tensor.
 *   (c) anything else -> clean VM error (vm->error = 1), no fabricated
 *       value, no crash.
 */
static VmTensor* vm_tensor_operand(VM* vm, Value v, const char* op_name) {
    if (v.type == VAL_TENSOR) {
        if (!is_valid_heap_ptr(vm, v.as.ptr)) return NULL;
        return (VmTensor*)vm->heap.objects[v.as.ptr]->opaque.ptr;
    }
    if (v.type == VAL_VECTOR) {
        if (!is_valid_heap_ptr(vm, v.as.ptr)) return NULL;
        VmVector* vec = (VmVector*)vm->heap.objects[v.as.ptr]->opaque.ptr;
        if (!vec) return NULL;
        for (int i = 0; i < vec->len; i++) {
            if (vec->items[i].type != VAL_INT && vec->items[i].type != VAL_FLOAT) {
                fprintf(stderr, "ERROR: %s: vector operand must be numeric (element %d is not a number)\n",
                        op_name ? op_name : "tensor-op", i);
                vm->error = 1;
                return NULL;
            }
        }
        int64_t shape1[1] = { vec->len };
        VmTensor* t = vm_tensor_new(&vm->heap.regions, shape1, 1);
        if (!t) return NULL;
        for (int i = 0; i < vec->len; i++) t->data[i] = as_number(vec->items[i]);
        return t;
    }
    fprintf(stderr, "ERROR: %s: expected a tensor or numeric vector operand\n",
            op_name ? op_name : "tensor-op");
    vm->error = 1;
    return NULL;
}

/* quantum-random / -int / -range (dispatch cases 1860-1862 below) used to be
 * backed by a VM-only xorshift64* PRNG that was numerically divergent from
 * the eshkol_qrng_* generator the LLVM AOT/JIT backend used for the same
 * builtins (docs/design/MOONLAB_INTEGRATION.md Section 3.2 flagged this as a
 * cross-backend honesty problem: same builtin name, different numbers,
 * neither one real quantum entropy). Fixed by routing the VM dispatch
 * through the SAME eshkol_qrng_uint64() / eshkol_qrng_double() entry points
 * (lib/quantum/quantum_rng_wrapper.c, included via eshkol_vm.c) that the
 * LLVM backend calls - see that file's honesty notice for what source is
 * actually active (classical fallback by default and on WASM; Moonlab's
 * Bell-verified QRNG when built with -DESHKOL_QUANTUM_ENABLED=ON). */

/* Runtime host-native registry. Compile-time fids stay in the static switch
 * below; fids >= ESHKOL_VM_HOST_NATIVE_BASE are looked up here by slot index
 * (slot = fid - ESHKOL_VM_HOST_NATIVE_BASE). */
#ifndef ESHKOL_VM_HOST_NATIVE_BASE
#define ESHKOL_VM_HOST_NATIVE_BASE 100000
#endif

#ifndef ESHKOL_VM_NATIVE_POLICY_DESKTOP
#define ESHKOL_VM_NATIVE_POLICY_DESKTOP 0
#endif

#ifndef ESHKOL_VM_NATIVE_POLICY_HOST_ONLY
#define ESHKOL_VM_NATIVE_POLICY_HOST_ONLY 1
#endif

#define VM_HOST_NATIVE_MAX 64
#define VM_HOST_NATIVE_NAME_MAX 128

#ifndef ESHKOL_BACKEND_VM_H
typedef int (*eshkol_vm_host_native_fn)(VM* vm);
typedef struct EshkolVmHostNative {
    const char* name;
    eshkol_vm_host_native_fn fn;
} EshkolVmHostNative;
#endif

static eshkol_vm_host_native_fn g_host_natives[VM_HOST_NATIVE_MAX];
static char g_host_native_names[VM_HOST_NATIVE_MAX][VM_HOST_NATIVE_NAME_MAX];
static int g_host_native_count = 0;

/** @brief True if `name` is non-NULL, non-empty, and fits within the fixed
 *         VM_HOST_NATIVE_NAME_MAX buffer. */
static int vm_host_native_name_is_valid(const char* name) {
    if (!name) return 0;
    size_t name_len = strlen(name);
    return name_len > 0 && name_len < VM_HOST_NATIVE_NAME_MAX;
}

/** @brief Tombstone a host-native registry slot (clear function pointer and name). */
static void vm_host_native_clear_slot(int slot) {
    g_host_natives[slot] = NULL;
    g_host_native_names[slot][0] = '\0';
}

/** @brief Populate a host-native registry slot with a function pointer and its name. */
static void vm_host_native_copy_slot(int slot,
                                     const char* name,
                                     eshkol_vm_host_native_fn fn) {
    size_t name_len = strlen(name);
    g_host_natives[slot] = fn;
    memcpy(g_host_native_names[slot], name, name_len + 1);
}

/** @brief Replace the entire host-native function registry with `entries`
 *         (rejecting the call on any duplicate name, NULL fn, or invalid
 *         name, or if `count` exceeds VM_HOST_NATIVE_MAX). Used by embedders
 *         to expose custom native functions to Eshkol bytecode via fids
 *         >= ESHKOL_VM_HOST_NATIVE_BASE. Returns 0 on success, -1 on
 *         validation failure (leaving the existing registry untouched). */
int eshkol_vm_install_host_natives(const EshkolVmHostNative* entries, int count) {
    if (count < 0 || count > VM_HOST_NATIVE_MAX) return -1;
    if (count > 0 && !entries) return -1;

    for (int i = 0; i < count; i++) {
        if (!entries[i].fn || !vm_host_native_name_is_valid(entries[i].name)) return -1;
        for (int j = 0; j < i; j++) {
            if (strcmp(entries[i].name, entries[j].name) == 0) return -1;
        }
    }

    for (int i = 0; i < VM_HOST_NATIVE_MAX; i++) {
        vm_host_native_clear_slot(i);
    }
    for (int i = 0; i < count; i++) {
        vm_host_native_copy_slot(i, entries[i].name, entries[i].fn);
    }
    g_host_native_count = count;
    return 0;
}

/** @brief Remove all registered host-native functions. */
void eshkol_vm_clear_host_natives(void) {
    for (int i = 0; i < VM_HOST_NATIVE_MAX; i++) {
        vm_host_native_clear_slot(i);
    }
    g_host_native_count = 0;
}

/** @brief Maximum number of host-native functions that can be registered at once. */
int eshkol_vm_host_native_capacity(void) {
    return VM_HOST_NATIVE_MAX;
}

/** @brief Current number of registered host-native functions (including any tombstoned slots below it). */
int eshkol_vm_host_native_count(void) {
    return g_host_native_count;
}

/** @brief Register a single named host-native function, reusing a
 *         tombstoned slot if available to preserve dense/stable slot
 *         indices (bytecode fids encode the slot directly). Returns the
 *         assigned slot index, or -1 on a duplicate name, invalid name, or
 *         if the registry is full. */
int eshkol_vm_register_host_native(const char* name, eshkol_vm_host_native_fn fn) {
    if (!fn || !vm_host_native_name_is_valid(name)) return -1;
    /* Tombstone-aware lookup: a tombstoned slot has fn == NULL and an empty
     * name; live slots cannot duplicate `name`. */
    for (int i = 0; i < g_host_native_count; i++) {
        if (g_host_natives[i] && strcmp(g_host_native_names[i], name) == 0) return -1;
    }
    /* Prefer a tombstoned slot to preserve dense indexing and keep the table
     * within capacity. Stable slot indices are essential because bytecode
     * encodes the fid as ESHKOL_VM_HOST_NATIVE_BASE + slot. */
    int slot = -1;
    for (int i = 0; i < g_host_native_count; i++) {
        if (g_host_natives[i] == NULL) { slot = i; break; }
    }
    if (slot < 0) {
        if (g_host_native_count >= VM_HOST_NATIVE_MAX) return -1;
        slot = g_host_native_count++;
    }
    vm_host_native_copy_slot(slot, name, fn);
    return slot;
}

/** @brief Tombstone a previously-registered host-native slot. Returns 0 on
 *         success, -1 if the slot is out of range or already empty. */
int eshkol_vm_unregister_host_native(int slot) {
    if (slot < 0 || slot >= g_host_native_count) return -1;
    if (g_host_natives[slot] == NULL) return -1;
    vm_host_native_clear_slot(slot);
    return 0;
}

/** @brief Host-native FFI helper: pop the top VM stack value and coerce it to
 *         an int64 (accepting VAL_INT, VAL_FLOAT truncation, or VAL_BOOL as
 *         0/1). Returns -1 on a VM error or unsupported type. */
int eshkol_vm_host_pop_int64(VM* vm, int64_t* out) {
    if (!vm || !out) return -1;
    Value v = vm_pop(vm);
    if (vm->error) return -1;
    if (v.type == VAL_INT) { *out = v.as.i; return 0; }
    if (v.type == VAL_FLOAT) { *out = (int64_t)v.as.f; return 0; }
    if (v.type == VAL_BOOL) { *out = v.as.b ? 1 : 0; return 0; }
    return -1;
}

/** @brief Host-native FFI helper: push an int64 onto the VM stack as an
 *         INT_VAL. Returns -1 if the push failed to grow the stack by exactly one. */
int eshkol_vm_host_push_int64(VM* vm, int64_t value) {
    if (!vm) return -1;
    int32_t before = vm->sp;
    vm_push(vm, INT_VAL(value));
    return (!vm->error && vm->sp == before + 1) ? 0 : -1;
}

/** @brief Host-native FFI helper: pop the top VM stack value and coerce it to
 *         a double (accepting VAL_FLOAT, VAL_INT, or VAL_BOOL as 0.0/1.0).
 *         Returns -1 on a VM error or unsupported type. */
int eshkol_vm_host_pop_double(VM* vm, double* out) {
    if (!vm || !out) return -1;
    Value v = vm_pop(vm);
    if (vm->error) return -1;
    if (v.type == VAL_FLOAT) { *out = v.as.f; return 0; }
    if (v.type == VAL_INT)   { *out = (double)v.as.i; return 0; }
    if (v.type == VAL_BOOL)  { *out = v.as.b ? 1.0 : 0.0; return 0; }
    return -1;
}

/** @brief Host-native FFI helper: push a double onto the VM stack as a FLOAT_VAL. */
int eshkol_vm_host_push_double(VM* vm, double value) {
    if (!vm) return -1;
    int32_t before = vm->sp;
    vm_push(vm, FLOAT_VAL(value));
    return (!vm->error && vm->sp == before + 1) ? 0 : -1;
}

/** @brief Invoke a user closure as part of an AD operation and account for
 *         the actual primal evaluation performed by this VM backend. */
static Value vm_ad_call_closure(VM* vm, Value closure, Value* args, int argc) {
    if (vm) vm->ad_primal_calls++;
    return vm_call_closure_from_native(vm, closure, args, argc);
}

/** @brief Peek into a closure's bytecode to find the native function id (fid)
 *         it's a thin wrapper around, by scanning its first few instructions
 *         for an OP_NATIVE_CALL before an OP_RETURN. Used to recognize
 *         first-class references to builtins (e.g. for AD dual-number
 *         dispatch). Returns -1 if `fn` isn't such a wrapper closure. */
static int vm_closure_native_id(VM* vm, Value fn) {
    if (!vm || fn.type != VAL_CLOSURE || fn.as.ptr < 0 || fn.as.ptr >= vm->heap.capacity) return -1;
    HeapObject* cl = vm->heap.objects[fn.as.ptr];
    if (!cl || cl->type != HEAP_CLOSURE) return -1;
    int pc = cl->closure.func_pc;
    if (pc < 0 || pc >= vm->code_len) return -1;

    for (int i = 0; i < 8 && pc + i < vm->code_len; i++) {
        Instr ins = vm->code[pc + i];
        if (ins.op == OP_NATIVE_CALL) return ins.operand;
        if (ins.op == OP_RETURN) break;
    }
    return -1;
}

/**
 * @brief Central native-function dispatcher for the bytecode VM: given a
 *        native function id (fid) popped from an OP_NATIVE_CALL instruction,
 *        pops its arguments off the VM stack, performs the operation, and
 *        pushes the result. This single function is the implementation of
 *        essentially the entire R7RS-plus-extensions builtin surface — math,
 *        predicates, pairs/lists, strings, vectors, bytevectors, I/O, ports,
 *        hash tables, records, continuations/dynamic-wind, exceptions,
 *        parameters, promises, tensors/AD, complex/rational/bignum numerics,
 *        processes, sockets, regex, JSON, and every helper documented above
 *        in this file — organized as one giant `switch (fid)` over numbered
 *        ranges (see the banner comments before each range for their
 *        purpose). fids >= ESHKOL_VM_HOST_NATIVE_BASE are instead routed to
 *        embedder-registered host-native functions (see
 *        eshkol_vm_register_host_native()). Desktop-only fids are rejected
 *        under ESHKOL_VM_NATIVE_POLICY_HOST_ONLY. Also drains any due
 *        timers on every call via vm_timers_poll_due(). Errors are reported
 *        via `vm->error = 1` rather than by return value.
 */
/**
 * @brief Round @p x to the nearest integer using round-half-to-even (banker's
 *        rounding), as required by R7RS `round`. The C library `round()`
 *        rounds halves away from zero (2.5 → 3), which silently disagreed with
 *        the native/reference path (2.5 → 2, 3.5 → 4).
 */
static double vm_round_half_even(double x) {
    double f = floor(x);
    double diff = x - f;
    double r;
    if (diff < 0.5) r = f;
    else if (diff > 0.5) r = f + 1.0;
    else r = (fmod(f, 2.0) == 0.0) ? f : f + 1.0; /* exactly halfway: round to even */
    /* Preserve the sign of zero so (round -0.5) yields -0.0, matching the
     * reference path, rather than +0.0. */
    if (r == 0.0) return copysign(0.0, x);
    return r;
}

static VmParameter* vm_parameter_from_value(VM* vm, Value value) {
    if (!vm || !is_heap_type(vm, value, HEAP_PARAMETER)) return NULL;
    return (VmParameter*)vm->heap.objects[value.as.ptr]->opaque.ptr;
}

static Value vm_parameter_converter(VmParameter* parameter) {
    Value converter = NIL_VAL;
    vm_param_converter_ref(parameter, &converter);
    return converter;
}

static Value vm_parameter_apply_converter(VM* vm, VmParameter* parameter,
                                          Value value) {
    Value converter = vm_parameter_converter(parameter);
    if (converter.type == VAL_CLOSURE) {
        return vm_call_closure_from_native(vm, converter, &value, 1);
    }
    return value;
}

static void vm_unwind_parameter_bindings(VM* vm, int target_depth) {
    if (!vm) return;
    if (target_depth < 0) target_depth = 0;
    while (vm->n_parameter_bindings > target_depth) {
        Value parameter_value = vm->parameter_bindings[--vm->n_parameter_bindings];
        VmParameter* parameter = vm_parameter_from_value(vm, parameter_value);
        if (parameter) vm_param_pop(parameter);
    }
}

static int vm_record_parameter_binding(VM* vm, Value parameter_value) {
    if (!vm) return 0;
    if (vm->n_parameter_bindings >=
        (int)(sizeof(vm->parameter_bindings) / sizeof(vm->parameter_bindings[0]))) {
        fprintf(stderr, "ERROR: parameterize binding stack overflow\n");
        vm->error = 1;
        return 0;
    }
    vm->parameter_bindings[vm->n_parameter_bindings++] = parameter_value;
    return 1;
}

static void vm_pop_recorded_parameter_binding(VM* vm, Value parameter_value) {
    if (!vm || vm->n_parameter_bindings <= 0) return;
    int top = vm->n_parameter_bindings - 1;
    if (vm->parameter_bindings[top].type == parameter_value.type &&
        vm->parameter_bindings[top].as.ptr == parameter_value.as.ptr) {
        vm->n_parameter_bindings = top;
    }
}

/* Parameters are entered in the VM's dynamic-wind stack as cleanup entries.
 * That makes an outer dynamic-wind's after thunk observe the restored
 * parameter value on exceptions and continuation escapes, just as the
 * compiled native dynamic-wind lowering does. */
static void vm_pop_parameter_binding(VM* vm, Value parameter_value) {
    VmParameter* parameter = vm_parameter_from_value(vm, parameter_value);
    if (parameter) vm_param_pop(parameter);
    vm_pop_recorded_parameter_binding(vm, parameter_value);
}

static void vm_run_wind_after(VM* vm, Value after) {
    if (after.type == VAL_CLOSURE) {
        vm_call_closure_from_native(vm, after, NULL, 0);
    } else if (after.type == VAL_PARAMETER_OBJ) {
        vm_pop_parameter_binding(vm, after);
    }
}

/* Procedure-call semantics for VAL_PARAMETER_OBJ.  The bytecode interpreter
 * calls this from OP_CALL/OP_TAIL_CALL; a zero-arity call reads the dynamic
 * stack and a one-or-more-arity call updates its current slot with the first
 * converted argument, matching the hosted native path's legacy setter API. */
static Value vm_parameter_invoke(VM* vm, Value parameter_value,
                                 const Value* args, int argc) {
    VmParameter* parameter = vm_parameter_from_value(vm, parameter_value);
    if (!parameter) {
        if (vm) vm->error = 1;
        return NIL_VAL;
    }
    if (argc == 0) {
        Value result;
        vm_param_ref(parameter, &result);
        return result;
    }
    Value converted = vm_parameter_apply_converter(vm, parameter, args[0]);
    vm_param_set(parameter, &converted);
    return converted;
}

/**
 * @brief VM identity equality shared by `eq?`, `eqv?`, and `memq`.
 *
 * Symbols have a dedicated VAL_SYMBOL tag but share the immutable VmString
 * payload representation.  Quoted occurrences are materialised independently,
 * so compare symbol spellings to provide the R7RS canonical-symbol guarantee.
 * `eq?` on distinct mutable strings is unspecified; retaining deterministic
 * content equality for VAL_STRING preserves the VM's established behavior.
 */
static int vm_identity_equal(VM* vm, Value a, Value b) {
    if (a.type != b.type) return 0;
    switch ((int)a.type) {
        case VAL_NIL:   return 1;
        case VAL_BOOL:  return a.as.b == b.as.b;
        case VAL_INT:   return a.as.i == b.as.i;
        case VAL_CHAR:  return a.as.i == b.as.i;
        case VAL_FLOAT: return a.as.f == b.as.f;
        case VAL_STRING: {
            VmString* as = vm_value_as_string(vm, a);
            VmString* bs = vm_value_as_string(vm, b);
            return as && bs && as->byte_len == bs->byte_len &&
                   memcmp(as->data, bs->data, (size_t)as->byte_len) == 0;
        }
        case VAL_SYMBOL: {
            VmString* as = vm_value_as_string(vm, a);
            VmString* bs = vm_value_as_string(vm, b);
            return as && bs && as->byte_len == bs->byte_len &&
                   memcmp(as->data, bs->data, (size_t)as->byte_len) == 0;
        }
        default: return a.as.ptr == b.as.ptr;
    }
}

/**
 * @brief Deep structural equality (`equal?` per R7RS): recurses through pairs
 *        and vectors, compares strings by content and numbers by value.
 *        Numbers keep exactness distinctions (an exact int is not `equal?` to
 *        an inexact float), matching the native/reference path. The prior
 *        implementation compared pairs and vectors by heap identity, so
 *        freshly-built equal structures reported `#f`.
 */
static int vm_deep_equal(VM* vm, Value a, Value b) {
    if (a.type != b.type) {
        /* Native #(...) literals use the tensor representation while
         * (vector ...) uses tagged vector slots. Preserve R7RS equal?
         * semantics across that internal boundary. */
        if ((a.type == VAL_VECTOR && b.type == VAL_TENSOR) ||
            (a.type == VAL_TENSOR && b.type == VAL_VECTOR)) {
            Value vv = a.type == VAL_VECTOR ? a : b;
            Value tv = a.type == VAL_TENSOR ? a : b;
            HeapObject* vo = vm->heap.objects[vv.as.ptr];
            HeapObject* to = vm->heap.objects[tv.as.ptr];
            VmVector* vector = (vo && vo->opaque.ptr) ? (VmVector*)vo->opaque.ptr : NULL;
            VmTensor* tensor = (to && to->opaque.ptr) ? (VmTensor*)to->opaque.ptr : NULL;
            if (!vector || !tensor || tensor->n_dims != 1 || vector->len != tensor->total)
                return 0;
            for (int i = 0; i < vector->len; ++i) {
                Value item = vector->items[i];
                double numeric;
                if (item.type == VAL_INT) numeric = (double)item.as.i;
                else if (item.type == VAL_FLOAT) numeric = item.as.f;
                else return 0;
                if (numeric != tensor->data[i]) return 0;
            }
            return 1;
        }
        return 0;
    }
    switch ((int)a.type) {
        case VAL_NIL:   return 1;
        case VAL_BOOL:  return a.as.b == b.as.b;
        case VAL_INT:   return a.as.i == b.as.i;
        case VAL_CHAR:  return a.as.i == b.as.i;
        case VAL_FLOAT: return a.as.f == b.as.f;
        case VAL_STRING: {
            VmString* as = vm_value_as_string(vm, a);
            VmString* bs = vm_value_as_string(vm, b);
            return as && bs && as->byte_len == bs->byte_len &&
                   memcmp(as->data, bs->data, (size_t)as->byte_len) == 0;
        }
        case VAL_SYMBOL: {
            VmString* as = vm_value_as_string(vm, a);
            VmString* bs = vm_value_as_string(vm, b);
            return as && bs && as->byte_len == bs->byte_len &&
                   memcmp(as->data, bs->data, (size_t)as->byte_len) == 0;
        }
        case VAL_PAIR: {
            /* Iterate the spine to avoid deep C recursion on long lists;
             * recurse only into each car. */
            while (a.type == VAL_PAIR && b.type == VAL_PAIR) {
                HeapObject* ao = vm->heap.objects[a.as.ptr];
                HeapObject* bo = vm->heap.objects[b.as.ptr];
                if (!ao || !bo) return 0;
                if (!vm_deep_equal(vm, ao->cons.car, bo->cons.car)) return 0;
                a = ao->cons.cdr;
                b = bo->cons.cdr;
            }
            return vm_deep_equal(vm, a, b); /* compare tails (incl. proper-list NIL) */
        }
        case VAL_VECTOR: {
            HeapObject* ao = vm->heap.objects[a.as.ptr];
            HeapObject* bo = vm->heap.objects[b.as.ptr];
            VmVector* av = (ao && ao->opaque.ptr) ? (VmVector*)ao->opaque.ptr : NULL;
            VmVector* bv = (bo && bo->opaque.ptr) ? (VmVector*)bo->opaque.ptr : NULL;
            if (!av || !bv) return av == bv;
            if (av->len != bv->len) return 0;
            for (int i = 0; i < av->len; i++)
                if (!vm_deep_equal(vm, av->items[i], bv->items[i])) return 0;
            return 1;
        }
        case VAL_BIGNUM: {
            HeapObject* ao = vm->heap.objects[a.as.ptr];
            HeapObject* bo = vm->heap.objects[b.as.ptr];
            VmBignum* ab = (ao && ao->opaque.ptr) ? (VmBignum*)ao->opaque.ptr : NULL;
            VmBignum* bb = (bo && bo->opaque.ptr) ? (VmBignum*)bo->opaque.ptr : NULL;
            return ab && bb && bignum_compare(ab, bb) == 0;
        }
        case VAL_RATIONAL: {
            HeapObject* ao = vm->heap.objects[a.as.ptr];
            HeapObject* bo = vm->heap.objects[b.as.ptr];
            VmRational* ar = (ao && ao->opaque.ptr) ? (VmRational*)ao->opaque.ptr : NULL;
            VmRational* br = (bo && bo->opaque.ptr) ? (VmRational*)bo->opaque.ptr : NULL;
            return ar && br && ar->num == br->num && ar->denom == br->denom;
        }
        case VAL_TENSOR: {
            /* Numeric vector literals compile to tensors in this VM, so
             * structural tensor equality is needed for equal? on such
             * "vectors" to match the native path. */
            HeapObject* ao = vm->heap.objects[a.as.ptr];
            HeapObject* bo = vm->heap.objects[b.as.ptr];
            VmTensor* at = (ao && ao->opaque.ptr) ? (VmTensor*)ao->opaque.ptr : NULL;
            VmTensor* bt = (bo && bo->opaque.ptr) ? (VmTensor*)bo->opaque.ptr : NULL;
            if (!at || !bt) return at == bt;
            if (at->n_dims != bt->n_dims || at->total != bt->total) return 0;
            for (int i = 0; i < at->n_dims; i++)
                if (at->shape[i] != bt->shape[i]) return 0;
            for (int64_t i = 0; i < at->total; i++)
                if (at->data[i] != bt->data[i]) return 0;
            return 1;
        }
        default:
            return a.as.ptr == b.as.ptr; /* eq? fallback for opaque types */
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Bignum-aware arithmetic helpers
 *
 * The base VM's arithmetic/comparison opcodes and native builtins coerced
 * operands through as_number(), which returns 0.0 for a VAL_BIGNUM. Any
 * expression whose exact result overflows int64 (e.g. (+ (expt 7 41) 13),
 * (modulo (expt 10 26) 3)) therefore silently produced a wrong small value on
 * the VM path while the native/JIT/AOT paths stayed correct. These helpers let
 * the ops promote to the arena bignum runtime (vm_bignum.c) whenever a bignum
 * operand appears or an int64 op overflows.
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief True when either operand is a heap-boxed bignum. */
static inline int vm_either_bignum(Value a, Value b) {
    return a.type == VAL_BIGNUM || b.type == VAL_BIGNUM;
}

/** @brief Coerce an integer-ish Value to a VmBignum*. VAL_BIGNUM returns its
 *         heap payload directly; VAL_INT/VAL_CHAR (and anything else, best
 *         effort) is materialised as a fresh bignum. NULL only on alloc
 *         failure. */
static VmBignum* vm_coerce_bignum(VM* vm, Value v) {
    if (v.type == VAL_BIGNUM)
        return (VmBignum*)vm->heap.objects[v.as.ptr]->opaque.ptr;
    int64_t iv = (v.type == VAL_INT || v.type == VAL_CHAR)
        ? v.as.i : (int64_t)as_number(v);
    return bignum_from_int64(&vm->heap.regions, iv);
}

/** @brief Push a bignum result, demoting it back to a fixnum (VAL_INT) when it
 *         fits in int64 so downstream ops and eqv?/= keep matching the native
 *         fixnum path. Sets vm->error if @p b is NULL. */
static void vm_push_bignum_norm(VM* vm, VmBignum* b) {
    if (!b) { vm->error = 1; return; }
    int ov = 0;
    int64_t iv = bignum_to_int64(b, &ov);
    if (!ov) vm_push(vm, INT_VAL(iv));
    else VM_PUSH_HEAP_OPAQUE(vm, HEAP_BIGNUM, VAL_BIGNUM, b);
}

/** @brief Compute (a op b) exactly in the bignum domain and push the
 *         normalized result. @p op is one of '+','-','*','q' (quotient),
 *         'r' (remainder), 'm' (R7RS modulo). A float operand on +,-,* falls
 *         back to inexact float (R7RS contagion). Division by zero sets
 *         vm->error. */
static void vm_bignum_arith(VM* vm, Value a, Value b, char op) {
    /* Bignum mixed with an inexact float → inexact float. */
    if ((op == '+' || op == '-' || op == '*') &&
        (a.type == VAL_FLOAT || b.type == VAL_FLOAT)) {
        double x = (a.type == VAL_BIGNUM)
            ? bignum_to_double((VmBignum*)vm->heap.objects[a.as.ptr]->opaque.ptr)
            : as_number(a);
        double y = (b.type == VAL_BIGNUM)
            ? bignum_to_double((VmBignum*)vm->heap.objects[b.as.ptr]->opaque.ptr)
            : as_number(b);
        double r = (op == '+') ? x + y : (op == '-') ? x - y : x * y;
        vm_push(vm, FLOAT_VAL(r));
        return;
    }
    VmRegionStack* rs = &vm->heap.regions;
    VmBignum* ab = vm_coerce_bignum(vm, a);
    VmBignum* bb = vm_coerce_bignum(vm, b);
    if (!ab || !bb) { vm->error = 1; return; }
    VmBignum* r = NULL;
    switch (op) {
    case '+': r = bignum_add(rs, ab, bb); break;
    case '-': r = bignum_sub(rs, ab, bb); break;
    case '*': r = bignum_mul(rs, ab, bb); break;
    case 'q': if (bignum_is_zero(bb)) { vm->error = 1; return; }
              r = bignum_div(rs, ab, bb); break;
    case 'r': if (bignum_is_zero(bb)) { vm->error = 1; return; }
              r = bignum_mod(rs, ab, bb); break;
    case 'm': {
        /* R7RS modulo: result carries the sign of the divisor. bignum_mod
         * returns the truncated remainder (sign of the dividend), so adjust
         * by adding the divisor when the signs disagree. */
        if (bignum_is_zero(bb)) { vm->error = 1; return; }
        r = bignum_mod(rs, ab, bb);
        if (r && !bignum_is_zero(r) && bignum_sign(r) != bignum_sign(bb))
            r = bignum_add(rs, r, bb);
        break;
    }
    default: vm->error = 1; return;
    }
    vm_push_bignum_norm(vm, r);
}

/** @brief Three-way compare of two integer-ish values through the bignum
 *         domain (-1/0/1). A float operand compares via double. */
static int vm_bignum_compare_vals(VM* vm, Value a, Value b) {
    if (a.type == VAL_FLOAT || b.type == VAL_FLOAT) {
        double x = (a.type == VAL_BIGNUM)
            ? bignum_to_double((VmBignum*)vm->heap.objects[a.as.ptr]->opaque.ptr)
            : as_number(a);
        double y = (b.type == VAL_BIGNUM)
            ? bignum_to_double((VmBignum*)vm->heap.objects[b.as.ptr]->opaque.ptr)
            : as_number(b);
        return (x < y) ? -1 : (x > y) ? 1 : 0;
    }
    VmBignum* ab = vm_coerce_bignum(vm, a);
    VmBignum* bb = vm_coerce_bignum(vm, b);
    if (!ab || !bb) return 0;
    return bignum_compare(ab, bb);
}

/** Write one value to an arbitrary VM output port using either display or
 * R7RS external-representation syntax.  Keeping this port-aware serializer
 * beside native dispatch lets `(write value port)` work for file and string
 * ports instead of silently falling back to stdout. */
static void vm_write_value_port(VM* vm, Value value, VmPort* port,
                                int write_syntax) {
    char number[160];
    if (!vm || !port) return;
    switch ((int)value.type) {
    case VAL_NIL: vm_port_write_cstr(port, "()"); break;
    case VAL_INT:
        snprintf(number, sizeof(number), "%lld", (long long)value.as.i);
        vm_port_write_cstr(port, number);
        break;
    case VAL_FLOAT:
        snprintf(number, sizeof(number), "%.17g", value.as.f);
        vm_port_write_cstr(port, number);
        break;
    case VAL_BOOL: vm_port_write_cstr(port, value.as.b ? "#t" : "#f"); break;
    case VAL_CHAR: {
        if (write_syntax) {
            if (value.as.i == ' ') { vm_port_write_cstr(port, "#\\space"); break; }
            if (value.as.i == '\n') { vm_port_write_cstr(port, "#\\newline"); break; }
            if (value.as.i == '\t') { vm_port_write_cstr(port, "#\\tab"); break; }
            vm_port_write_cstr(port, "#\\");
        }
        char bytes[4];
        int count = vm_utf8_encode((int)value.as.i, bytes);
        if (count > 0) vm_port_write_bytes(port, bytes, count);
        break;
    }
    case VAL_STRING:
    case VAL_SYMBOL: {
        VmString* string = vm_value_as_string(vm, value);
        int quote = write_syntax && value.type == VAL_STRING;
        if (quote) vm_port_write_cstr(port, "\"");
        if (string && string->data) {
            for (int i = 0; i < string->byte_len; ++i) {
                unsigned char ch = (unsigned char)string->data[i];
                if (quote && ch == '"') vm_port_write_cstr(port, "\\\"");
                else if (quote && ch == '\\') vm_port_write_cstr(port, "\\\\");
                else if (quote && ch == '\n') vm_port_write_cstr(port, "\\n");
                else if (quote && ch == '\r') vm_port_write_cstr(port, "\\r");
                else if (quote && ch == '\t') vm_port_write_cstr(port, "\\t");
                else vm_port_write_bytes(port, (const char*)&string->data[i], 1);
            }
        }
        if (quote) vm_port_write_cstr(port, "\"");
        break;
    }
    case VAL_PAIR: {
        vm_port_write_cstr(port, "(");
        Value cursor = value;
        int first = 1;
        while (cursor.type == VAL_PAIR) {
            if (!first) vm_port_write_cstr(port, " ");
            first = 0;
            HeapObject* cell = vm->heap.objects[cursor.as.ptr];
            vm_write_value_port(vm, cell->cons.car, port, write_syntax);
            cursor = cell->cons.cdr;
        }
        if (cursor.type != VAL_NIL) {
            vm_port_write_cstr(port, " . ");
            vm_write_value_port(vm, cursor, port, write_syntax);
        }
        vm_port_write_cstr(port, ")");
        break;
    }
    case VAL_VECTOR: {
        vm_port_write_cstr(port, "#(");
        HeapObject* object = vm->heap.objects[value.as.ptr];
        VmVector* vector = object ? (VmVector*)object->opaque.ptr : NULL;
        for (int i = 0; vector && i < vector->len; ++i) {
            if (i) vm_port_write_cstr(port, " ");
            vm_write_value_port(vm, vector->items[i], port, write_syntax);
        }
        vm_port_write_cstr(port, ")");
        break;
    }
    case VAL_RATIONAL: {
        VmRational* rational = (VmRational*)vm->heap.objects[value.as.ptr]->opaque.ptr;
        if (!rational) { vm_port_write_cstr(port, "#<rational>"); break; }
        snprintf(number, sizeof(number), "%lld/%lld", (long long)rational->num,
                 (long long)rational->denom);
        vm_port_write_cstr(port, number);
        break;
    }
    case VAL_BIGNUM: {
        VmBignum* bignum = (VmBignum*)vm->heap.objects[value.as.ptr]->opaque.ptr;
        char* text = bignum ? bignum_to_string(&vm->heap.regions, bignum) : NULL;
        vm_port_write_cstr(port, text ? text : "#<bignum>");
        break;
    }
    case VAL_EOF: vm_port_write_cstr(port, "#<eof>"); break;
    case VAL_VOID: break;
    default:
        snprintf(number, sizeof(number), "#<value:%d>", (int)value.type);
        vm_port_write_cstr(port, number);
        break;
    }
}

/* ── Runtime datum reader -------------------------------------------------
 *
 * The source VM used to implement `(read)` as a single getchar(), returning
 * an integer and ignoring explicit ports.  Keep this reader intentionally
 * self-contained over VmPort so source execution, cached ESKB execution, and
 * WASM share the same bounded R7RS datum semantics.  Flat lists are built
 * iteratively; only syntactic nesting consumes C recursion, guarded by the
 * same conservative depth bound on every platform.
 */
#define VM_READER_MAX_DEPTH 512
#define VM_READER_MAX_TOKEN (1024 * 1024)

static Value vm_reader_datum(VM* vm, VmPort* port, int depth, int* ok, int* eof);
static Value vm_reader_token(VM* vm, VmPort* port, int first, int* ok);

static int vm_reader_delimiter(int ch) {
    return ch < 0 || isspace((unsigned char)ch) || ch == '(' || ch == ')' ||
           ch == '"' || ch == ';' || ch == '\'' || ch == '`' || ch == ',';
}

static int vm_reader_skip_space(VmPort* port) {
    for (;;) {
        int ch = vm_port_peek_byte(port);
        if (ch < 0) return -1;
        if (isspace((unsigned char)ch)) {
            (void)vm_port_read_byte(port);
            continue;
        }
        if (ch == ';') {
            do { ch = vm_port_read_byte(port); } while (ch >= 0 && ch != '\n');
            continue;
        }
        return ch;
    }
}

static Value vm_reader_string_value(VM* vm, const char* bytes, size_t len,
                                    int symbol) {
    if (!vm || !bytes || len > INT32_MAX) return NIL_VAL;
    VmString* string = vm_string_new(&vm->heap.regions, bytes, (int)len);
    int32_t ptr = string ? heap_alloc(&vm->heap) : -1;
    if (!string || ptr < 0) {
        vm->error = 1;
        return NIL_VAL;
    }
    vm->heap.objects[ptr]->type = HEAP_STRING;
    vm->heap.objects[ptr]->opaque.ptr = string;
    return (Value){.type = symbol ? VAL_SYMBOL : VAL_STRING, .as.ptr = ptr};
}

static Value vm_reader_cons(VM* vm, Value car, Value cdr, int* ok) {
    int32_t ptr = heap_alloc(&vm->heap);
    if (ptr < 0) {
        if (ok) *ok = 0;
        vm->error = 1;
        return NIL_VAL;
    }
    vm->heap.objects[ptr]->type = HEAP_CONS;
    vm->heap.objects[ptr]->cons.car = car;
    vm->heap.objects[ptr]->cons.cdr = cdr;
    return PAIR_VAL(ptr);
}

static Value vm_reader_prefixed(VM* vm, const char* name, Value datum, int* ok) {
    Value symbol = vm_reader_string_value(vm, name, strlen(name), 1);
    Value tail = vm_reader_cons(vm, datum, NIL_VAL, ok);
    return *ok ? vm_reader_cons(vm, symbol, tail, ok) : NIL_VAL;
}

static Value vm_reader_list(VM* vm, VmPort* port, int depth, int* ok) {
    Value head = NIL_VAL;
    Value tail = NIL_VAL;
    for (;;) {
        int ch = vm_reader_skip_space(port);
        if (ch < 0) { *ok = 0; vm->error = 1; return NIL_VAL; }
        if (ch == ')') {
            (void)vm_port_read_byte(port);
            return head;
        }
        Value item;
        int datum_eof = 0;
        if (ch == '.') {
            (void)vm_port_read_byte(port);
            int next = vm_port_peek_byte(port);
            if (vm_reader_delimiter(next)) {
                if (head.type != VAL_PAIR) {
                    *ok = 0; vm->error = 1; return NIL_VAL;
                }
                Value cdr = vm_reader_datum(vm, port, depth + 1, ok, &datum_eof);
                if (!*ok || datum_eof || vm_reader_skip_space(port) != ')') {
                    *ok = 0; vm->error = 1; return NIL_VAL;
                }
                (void)vm_port_read_byte(port);
                vm->heap.objects[tail.as.ptr]->cons.cdr = cdr;
                return head;
            }
            /* A dot followed by a non-delimiter is a peculiar identifier
             * (for example `.foo`), not dotted-pair punctuation.  The dot
             * has already been consumed, so seed the ordinary token parser
             * with it and let that parser consume the remainder. */
            item = vm_reader_token(vm, port, '.', ok);
        } else {
            item = vm_reader_datum(vm, port, depth + 1, ok, &datum_eof);
        }
        if (!*ok || datum_eof) { *ok = 0; vm->error = 1; return NIL_VAL; }
        Value cell = vm_reader_cons(vm, item, NIL_VAL, ok);
        if (!*ok) return NIL_VAL;
        if (head.type == VAL_NIL) head = cell;
        else vm->heap.objects[tail.as.ptr]->cons.cdr = cell;
        tail = cell;
    }
}

static Value vm_reader_vector(VM* vm, VmPort* port, int depth, int* ok) {
    Value list = vm_reader_list(vm, port, depth, ok);
    if (!*ok) return NIL_VAL;
    int count = 0;
    for (Value cur = list; cur.type == VAL_PAIR;
         cur = vm->heap.objects[cur.as.ptr]->cons.cdr) ++count;
    int32_t ptr = heap_alloc(&vm->heap);
    VmVector* vector = ptr >= 0
        ? (VmVector*)vm_alloc(&vm->heap.regions, sizeof(VmVector)) : NULL;
    if (ptr < 0 || !vector) { *ok = 0; vm->error = 1; return NIL_VAL; }
    vector->len = count;
    vector->cap = count;
    vector->items = count ? (Value*)vm_alloc(&vm->heap.regions,
                                              (size_t)count * sizeof(Value)) : NULL;
    if (count && !vector->items) { *ok = 0; vm->error = 1; return NIL_VAL; }
    int index = 0;
    for (Value cur = list; cur.type == VAL_PAIR;
         cur = vm->heap.objects[cur.as.ptr]->cons.cdr)
        vector->items[index++] = vm->heap.objects[cur.as.ptr]->cons.car;
    vm->heap.objects[ptr]->type = HEAP_VECTOR;
    vm->heap.objects[ptr]->opaque.ptr = vector;
    return (Value){.type = VAL_VECTOR, .as.ptr = ptr};
}

static Value vm_reader_string(VM* vm, VmPort* port, int* ok) {
    size_t cap = 64, len = 0;
    char* bytes = (char*)malloc(cap);
    if (!bytes) { *ok = 0; vm->error = 1; return NIL_VAL; }
    for (;;) {
        int ch = vm_port_read_byte(port);
        if (ch < 0) { free(bytes); *ok = 0; vm->error = 1; return NIL_VAL; }
        if (ch == '"') break;
        if (ch == '\\') {
            ch = vm_port_read_byte(port);
            if (ch < 0) { free(bytes); *ok = 0; vm->error = 1; return NIL_VAL; }
            if (ch == 'n') ch = '\n';
            else if (ch == 'r') ch = '\r';
            else if (ch == 't') ch = '\t';
            else if (ch == '\n') continue;
        }
        if (len >= VM_READER_MAX_TOKEN) {
            free(bytes); *ok = 0; vm->error = 1; return NIL_VAL;
        }
        if (len == cap) {
            size_t next_cap = cap * 2;
            char* grown = (char*)realloc(bytes, next_cap);
            if (!grown) { free(bytes); *ok = 0; vm->error = 1; return NIL_VAL; }
            bytes = grown;
            cap = next_cap;
        }
        bytes[len++] = (char)ch;
    }
    Value result = vm_reader_string_value(vm, bytes, len, 0);
    free(bytes);
    return result;
}

static Value vm_reader_token(VM* vm, VmPort* port, int first, int* ok) {
    size_t cap = 64, len = 0;
    char* token = (char*)malloc(cap);
    if (!token) { *ok = 0; vm->error = 1; return NIL_VAL; }
    token[len++] = (char)first;
    while (!vm_reader_delimiter(vm_port_peek_byte(port))) {
        int ch = vm_port_read_byte(port);
        if (len >= VM_READER_MAX_TOKEN) {
            free(token); *ok = 0; vm->error = 1; return NIL_VAL;
        }
        if (len + 1 >= cap) {
            size_t next_cap = cap * 2;
            char* grown = (char*)realloc(token, next_cap);
            if (!grown) { free(token); *ok = 0; vm->error = 1; return NIL_VAL; }
            token = grown;
            cap = next_cap;
        }
        token[len++] = (char)ch;
    }
    token[len] = '\0';

    if (!strcmp(token, "#t") || !strcmp(token, "#true")) {
        free(token); return BOOL_VAL(1);
    }
    if (!strcmp(token, "#f") || !strcmp(token, "#false")) {
        free(token); return BOOL_VAL(0);
    }
    if (!strncmp(token, "#\\", 2)) {
        const char* spelling = token + 2;
        int64_t cp = !strcmp(spelling, "space") ? ' ' :
                     !strcmp(spelling, "newline") ? '\n' :
                     !strcmp(spelling, "tab") ? '\t' :
                     !strcmp(spelling, "return") ? '\r' :
                     !strcmp(spelling, "null") ? 0 :
                     (unsigned char)spelling[0];
        if (spelling[0] == 'x' && spelling[1] != '\0') {
            char* hex_end = NULL;
            errno = 0;
            long parsed = strtol(spelling + 1, &hex_end, 16);
            if (errno == 0 && hex_end && *hex_end == '\0' &&
                parsed >= 0 && parsed <= 0x10ffffL)
                cp = (int64_t)parsed;
        }
        free(token);
        return (Value){.type = VAL_CHAR, .as.i = cp};
    }

    char* end = NULL;
    errno = 0;
    long long integer = strtoll(token, &end, 10);
    if (end && *end == '\0' && end != token && errno != ERANGE) {
        free(token); return INT_VAL((int64_t)integer);
    }
    /* Preserve exactness beyond int64 instead of silently turning a decimal
     * integer into a symbol.  bignum_from_string is arena-backed and accepts
     * the same optional leading sign as strtoll. */
    int decimal_integer = token[0] != '\0';
    size_t digit_index = (token[0] == '+' || token[0] == '-') ? 1u : 0u;
    if (token[digit_index] == '\0') decimal_integer = 0;
    for (size_t i = digit_index; decimal_integer && token[i] != '\0'; ++i)
        if (!isdigit((unsigned char)token[i])) decimal_integer = 0;
    if (decimal_integer && errno == ERANGE) {
        VmBignum* bignum = bignum_from_string(&vm->heap.regions, token);
        int32_t ptr = bignum ? heap_alloc(&vm->heap) : -1;
        if (!bignum || ptr < 0) {
            free(token); *ok = 0; vm->error = 1; return NIL_VAL;
        }
        vm->heap.objects[ptr]->type = HEAP_BIGNUM;
        vm->heap.objects[ptr]->opaque.ptr = bignum;
        free(token);
        return (Value){.type = VAL_BIGNUM, .as.ptr = ptr};
    }
    /* Exact rationals are normalized by the production numeric tower and
     * collapse to an integer when the reduced denominator is one. */
    char* slash = strchr(token, '/');
    if (slash && slash != token && slash[1] != '\0' && !strchr(slash + 1, '/')) {
        *slash = '\0';
        char *num_end = NULL, *den_end = NULL;
        errno = 0;
        long long numerator = strtoll(token, &num_end, 10);
        int numerator_ok = errno != ERANGE && num_end && *num_end == '\0';
        errno = 0;
        long long denominator = strtoll(slash + 1, &den_end, 10);
        int denominator_ok = errno != ERANGE && den_end && *den_end == '\0' &&
                             denominator != 0;
        *slash = '/';
        if (numerator_ok && denominator_ok) {
            VmRational* rational = vm_rational_make(
                vm_active_arena(&vm->heap.regions),
                (int64_t)numerator, (int64_t)denominator);
            if (!rational) {
                free(token); *ok = 0; vm->error = 1; return NIL_VAL;
            }
            if (rational->denom == 1) {
                int64_t value = rational->num;
                free(token);
                return INT_VAL(value);
            }
            int32_t ptr = heap_alloc(&vm->heap);
            if (ptr < 0) {
                free(token); *ok = 0; vm->error = 1; return NIL_VAL;
            }
            vm->heap.objects[ptr]->type = HEAP_RATIONAL;
            vm->heap.objects[ptr]->opaque.ptr = rational;
            free(token);
            return (Value){.type = VAL_RATIONAL, .as.ptr = ptr};
        }
    }
    errno = 0;
    end = NULL;
    double real = strtod(token, &end);
    if (end && *end == '\0' && end != token && errno != ERANGE &&
        (strchr(token, '.') || strchr(token, 'e') || strchr(token, 'E'))) {
        free(token); return FLOAT_VAL(real);
    }
    Value result = vm_reader_string_value(vm, token, len, 1);
    free(token);
    return result;
}

static Value vm_reader_datum(VM* vm, VmPort* port, int depth, int* ok, int* eof) {
    if (eof) *eof = 0;
    if (!vm || !port || depth > VM_READER_MAX_DEPTH) {
        if (ok) *ok = 0;
        if (vm) vm->error = 1;
        return NIL_VAL;
    }
    int ch = vm_reader_skip_space(port);
    if (ch < 0) {
        if (eof) *eof = 1;
        return (Value){.type = VAL_EOF};
    }
    ch = vm_port_read_byte(port);
    if (ch == '(') return vm_reader_list(vm, port, depth, ok);
    if (ch == ')') { *ok = 0; vm->error = 1; return NIL_VAL; }
    if (ch == '"') return vm_reader_string(vm, port, ok);
    if (ch == '#' && vm_port_peek_byte(port) == '(') {
        (void)vm_port_read_byte(port);
        return vm_reader_vector(vm, port, depth, ok);
    }
    if (ch == '\'' || ch == '`' || ch == ',') {
        const char* prefix = ch == '\'' ? "quote" : ch == '`' ? "quasiquote" : "unquote";
        if (ch == ',' && vm_port_peek_byte(port) == '@') {
            (void)vm_port_read_byte(port);
            prefix = "unquote-splicing";
        }
        int datum_eof = 0;
        Value datum = vm_reader_datum(vm, port, depth + 1, ok, &datum_eof);
        if (!*ok || datum_eof) { *ok = 0; vm->error = 1; return NIL_VAL; }
        return vm_reader_prefixed(vm, prefix, datum, ok);
    }
    return vm_reader_token(vm, port, ch, ok);
}

static Value vm_reader_from_port(VM* vm, VmPort* port) {
    int ok = 1, eof = 0;
    Value result = vm_reader_datum(vm, port, 0, &ok, &eof);
    return ok ? result : NIL_VAL;
}

static int vm_is_promise_value(VM* vm, Value value, VmVector** out) {
    if (value.type != VAL_VECTOR || !is_valid_heap_ptr(vm, value.as.ptr)) return 0;
    HeapObject* object = vm->heap.objects[value.as.ptr];
    if (!object || object->type != HEAP_PROMISE || !object->opaque.ptr) return 0;
    VmVector* promise = (VmVector*)object->opaque.ptr;
    if (promise->len != 3) return 0;
    if (out) *out = promise;
    return 1;
}

static int vm_promise_state(const VmVector* promise) {
    if (!promise || promise->len != 3) return -1;
    if (promise->items[0].type == VAL_INT)
        return (int)promise->items[0].as.i;
    if (promise->items[0].type == VAL_BOOL)
        return promise->items[0].as.b ? 1 : 0;
    return -1;
}

static int vm_promise_mark_equal(Value a, Value b) {
    if (a.type != b.type) return 0;
    switch (a.type) {
    case VAL_NIL:
        return 1;
    case VAL_VECTOR:
        return a.as.ptr == b.as.ptr;
    default:
        return 0;
    }
}

static int vm_promise_mark_reachable(VM* vm, Value mark) {
    Value cursor = vm->promise_eval_head;
    while (!vm_promise_mark_equal(cursor, mark)) {
        VmVector* promise = NULL;
        if (!vm_is_promise_value(vm, cursor, &promise)) return 0;
        int state = vm_promise_state(promise);
        if (state != 3 && state != 4) return 0;
        cursor = promise->items[2];
    }
    return 1;
}

static int vm_promise_eval_begin(VM* vm, Value value, int original_state) {
    VmVector* promise = NULL;
    if ((original_state != 0 && original_state != 2) ||
        !vm_is_promise_value(vm, value, &promise)) {
        vm->error = 1;
        return 0;
    }
    promise->items[2] = vm->promise_eval_head;
    promise->items[0] = INT_VAL(original_state == 0 ? 3 : 4);
    vm->promise_eval_head = value;
    return 1;
}

static int vm_promise_eval_commit_one(VM* vm, Value value, Value result) {
    VmVector* promise = NULL;
    if (!vm_promise_mark_equal(vm->promise_eval_head, value) ||
        !vm_is_promise_value(vm, value, &promise)) {
        vm->error = 1;
        return 0;
    }
    const int state = vm_promise_state(promise);
    if (state != 3 && state != 4) {
        vm->error = 1;
        return 0;
    }
    Value previous = promise->items[2];
    promise->items[1] = NIL_VAL;
    promise->items[2] = result;
    promise->items[0] = INT_VAL(1);
    vm->promise_eval_head = previous;
    return 1;
}

static int vm_promise_eval_commit_to(VM* vm, Value mark, Value result) {
    if (!vm_promise_mark_reachable(vm, mark)) {
        vm->error = 1;
        return 0;
    }
    while (!vm_promise_mark_equal(vm->promise_eval_head, mark)) {
        Value promise_value = vm->promise_eval_head;
        if (!vm_promise_eval_commit_one(vm, promise_value, result)) return 0;
    }
    return 1;
}

static void vm_promise_eval_unwind_to(VM* vm, Value mark) {
    /* A continuation captured inside a force whose evaluation has since
     * completed cannot recreate that expired dynamic extent.  Match the
     * hosted runtime: only roll back when the saved mark is still active. */
    if (!vm_promise_mark_reachable(vm, mark)) return;
    while (!vm_promise_mark_equal(vm->promise_eval_head, mark)) {
        Value promise_value = vm->promise_eval_head;
        VmVector* promise = NULL;
        if (!vm_is_promise_value(vm, promise_value, &promise)) return;
        const int state = vm_promise_state(promise);
        Value previous = promise->items[2];
        promise->items[0] = INT_VAL(state == 4 ? 2 : 0);
        promise->items[2] = NIL_VAL;
        vm->promise_eval_head = previous;
    }
}

static void vm_escape_native_control(VM* vm) {
    if (vm->native_call_depth > 0 && vm->native_escape_ready) {
        longjmp(vm->native_escape_jmp, 1);
    }
}

/* R7RS delay-force trampoline.  Both ordinary and delay-force evaluations
 * join the VM-wide intrusive chain so exceptions and continuations can roll
 * them back in O(1) auxiliary memory.  Successful delay-force chains are
 * path-compressed to the final value. */
static Value vm_force_promise_value(VM* vm, Value initial) {
    const Value mark = vm->promise_eval_head;
    Value current = initial;
    Value final = initial;

    for (;;) {
        VmVector* promise = NULL;
        if (!vm_is_promise_value(vm, current, &promise)) {
            final = current;
            break;
        }

        const int state = vm_promise_state(promise);
        if (state == 1) {
            final = promise->items[2];
            break;
        }
        if (state == 3 || state == 4) {
            fprintf(stderr, "ERROR: re-entrant force of an evaluating promise\n");
            vm->error = 1;
            final = NIL_VAL;
            break;
        }
        if (state != 0 && state != 2) {
            fprintf(stderr, "ERROR: invalid promise state %d\n", state);
            vm->error = 1;
            final = NIL_VAL;
            break;
        }

        Value thunk = promise->items[1];
        if (thunk.type != VAL_CLOSURE) {
            fprintf(stderr, "ERROR: unforced promise has no callable thunk\n");
            vm->error = 1;
            final = NIL_VAL;
            break;
        }

        if (!vm_promise_eval_begin(vm, current, state)) {
            final = NIL_VAL;
            break;
        }
        Value result = vm_call_closure_from_native(vm, thunk, NULL, 0);
        if (vm->error) {
            final = NIL_VAL;
            break;
        }

        if (state == 0) {
            if (!vm_promise_eval_commit_one(vm, current, result)) {
                final = NIL_VAL;
                break;
            }
            final = result;
            break;
        }

        current = result;
    }

    if (!vm->error) {
        (void)vm_promise_eval_commit_to(vm, mark, final);
    } else {
        vm_promise_eval_unwind_to(vm, mark);
    }
    return final;
}

static void vm_dispatch_native(VM* vm, int fid) {
    vm_timers_poll_due(vm);
    if (fid >= ESHKOL_VM_HOST_NATIVE_BASE) {
        int slot = fid - ESHKOL_VM_HOST_NATIVE_BASE;
        if (slot >= 0 && slot < g_host_native_count && g_host_natives[slot]) {
            if (g_host_natives[slot](vm) != 0) vm->error = 1;
        } else {
            vm->error = 1;
        }
        return;
    }
    if (vm->native_policy == ESHKOL_VM_NATIVE_POLICY_HOST_ONLY) {
        fprintf(stderr, "VM native policy rejected desktop native fid %d\n", fid);
        vm->error = 1;
        return;
    }
    switch (fid) {
    /* ══════════════════════════════════════════════════════════════════════
     * Math functions (20-35)
     * ══════════════════════════════════════════════════════════════════════ */
    case 20: { Value a = vm_pop(vm); if (a.type==VAL_DUAL) { vm_push(vm,a); vm_dispatch_native(vm,377); } else vm_push(vm, FLOAT_VAL(sin(as_number(a)))); break; }
    case 21: { Value a = vm_pop(vm); if (a.type==VAL_DUAL) { vm_push(vm,a); vm_dispatch_native(vm,378); } else vm_push(vm, FLOAT_VAL(cos(as_number(a)))); break; }
    case 22: { Value a = vm_pop(vm); if (a.type==VAL_DUAL) { /* tan = sin/cos */ vm_push(vm,a); vm_dispatch_native(vm,377); Value s=vm_pop(vm); vm_push(vm,a); vm_dispatch_native(vm,378); Value c=vm_pop(vm); vm_push(vm,s); vm_push(vm,c); vm_dispatch_native(vm,376); } else vm_push(vm, FLOAT_VAL(tan(as_number(a)))); break; }
    case 23: { Value a = vm_pop(vm); if (a.type==VAL_DUAL) { vm_push(vm,a); vm_dispatch_native(vm,379); } else vm_push(vm, FLOAT_VAL(exp(as_number(a)))); break; }
    case 24: { Value a = vm_pop(vm); if (a.type==VAL_DUAL) { vm_push(vm,a); vm_dispatch_native(vm,380); } else vm_push(vm, FLOAT_VAL(log(as_number(a)))); break; }
    case 25: { Value a = vm_pop(vm); if (a.type==VAL_DUAL) { vm_push(vm,a); vm_dispatch_native(vm,381); } else vm_push(vm, FLOAT_VAL(sqrt(as_number(a)))); break; }
    case 26: { Value a = vm_pop(vm); vm_push(vm, number_val(floor(as_number_vm(vm,a)))); break; }
    case 27: { Value a = vm_pop(vm); vm_push(vm, number_val(ceil(as_number_vm(vm,a)))); break; }
    case 28: { Value a = vm_pop(vm); vm_push(vm, number_val(vm_round_half_even(as_number_vm(vm,a)))); break; }
    case 29: { Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(asin(as_number_vm(vm,a)))); break; }
    case 30: { Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(acos(as_number_vm(vm,a)))); break; }
    case 31: { Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(atan(as_number_vm(vm,a)))); break; }
    case 32: { Value b = vm_pop(vm); Value a = vm_pop(vm);
        if (a.type==VAL_DUAL||b.type==VAL_DUAL) { vm_push(vm,a); vm_push(vm,b); vm_dispatch_native(vm,385); break; }
        /* Exact integer base with a non-negative integer exponent → exact
         * result: an int64 while it fits, promoting to a bignum on overflow
         * (matching the native path). Previously always used pow() and
         * returned an inexact float, so (expt 2 40) printed 1.09951e+12. */
        if ((a.type==VAL_INT || a.type==VAL_BIGNUM) && b.type==VAL_INT && b.as.i >= 0) {
            VmRegionStack* bn_rs = &vm->heap.regions;
            VmBignum* base_bn = (a.type==VAL_BIGNUM)
                ? (VmBignum*)vm->heap.objects[a.as.ptr]->opaque.ptr
                : bignum_from_int64(bn_rs, a.as.i);
            VmBignum* result = base_bn ? bignum_pow(bn_rs, base_bn, (uint64_t)b.as.i) : NULL;
            if (result) {
                int ov = 0; int64_t iv = bignum_to_int64(result, &ov);
                if (!ov) vm_push(vm, INT_VAL(iv));
                else VM_PUSH_HEAP_OPAQUE(vm, HEAP_BIGNUM, VAL_BIGNUM, result);
                break;
            }
        }
        vm_push(vm, FLOAT_VAL(pow(as_number(a), as_number(b)))); break; }
    case 33: { Value b = vm_pop(vm); Value a = vm_pop(vm);
        if (vm_either_bignum(a,b)) { vm_push(vm, vm_bignum_compare_vals(vm,a,b) <= 0 ? a : b); break; }
        double da=as_number_vm(vm,a),db=as_number_vm(vm,b); vm_push(vm, number_val(da<db?da:db)); break; }
    case 34: { Value b = vm_pop(vm); Value a = vm_pop(vm);
        if (vm_either_bignum(a,b)) { vm_push(vm, vm_bignum_compare_vals(vm,a,b) >= 0 ? a : b); break; }
        double da=as_number_vm(vm,a),db=as_number_vm(vm,b); vm_push(vm, number_val(da>db?da:db)); break; }
    case 35: { Value a = vm_pop(vm); if (a.type==VAL_DUAL) { vm_push(vm,a); vm_dispatch_native(vm,383); }
        else if (a.type==VAL_BIGNUM) { vm_push_bignum_norm(vm, bignum_abs_val(&vm->heap.regions, (VmBignum*)vm->heap.objects[a.as.ptr]->opaque.ptr)); }
        else vm_push(vm, number_val(fabs(as_number(a)))); break; }
    /* modulo, remainder, quotient — first-class closure versions */
    case 36: { Value b = vm_pop(vm); Value a = vm_pop(vm);
        if (vm_either_bignum(a,b)) { vm_bignum_arith(vm,a,b,'m'); break; }
        int64_t ia=(int64_t)as_number(a), ib=(int64_t)as_number(b);
        if (ib==0){vm->error=1;break;}
        int64_t r=ia%ib; if(r!=0&&((r^ib)<0)) r+=ib;
        vm_push(vm, INT_VAL(r)); break; }
    case 37: { Value b = vm_pop(vm); Value a = vm_pop(vm);
        if (vm_either_bignum(a,b)) { vm_bignum_arith(vm,a,b,'r'); break; }
        int64_t ia=(int64_t)as_number(a), ib=(int64_t)as_number(b);
        if (ib==0){vm->error=1;break;}
        vm_push(vm, INT_VAL(ia%ib)); break; }
    case 38: { Value b = vm_pop(vm); Value a = vm_pop(vm);
        if (vm_either_bignum(a,b)) { vm_bignum_arith(vm,a,b,'q'); break; }
        int64_t ia=(int64_t)as_number(a), ib=(int64_t)as_number(b);
        if (ib==0){vm->error=1;break;}
        vm_push(vm, INT_VAL(ia/ib)); break; }

    /* ══════════════════════════════════════════════════════════════════════
     * Predicates (40-50)
     * ══════════════════════════════════════════════════════════════════════ */
    case 40: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(as_number(a) > 0)); break; }
    case 41: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(as_number(a) < 0)); break; }
    case 42: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL((int64_t)as_number(a) % 2 != 0)); break; }
    case 43: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL((int64_t)as_number(a) % 2 == 0)); break; }
    case 44: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(as_number(a) == 0)); break; }
    case 45: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_PAIR)); break; }
    case 46: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_INT || a.type == VAL_FLOAT)); break; }
    case 47: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_STRING)); break; }
    case 48: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_BOOL)); break; }
    case 49: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_CLOSURE)); break; }
    case 50: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_VECTOR)); break; }

    /* ══════════════════════════════════════════════════════════════════════
     * String operations (51-56) — legacy IDs
     * ══════════════════════════════════════════════════════════════════════ */
    case 51: { /* number->string (n, radix) — radix defaults to 10 via prelude wrapper */
        Value radix_val = vm_pop(vm);
        Value a = vm_pop(vm);
        int radix = (radix_val.type == VAL_INT) ? (int)radix_val.as.i : 10;
        char buf[128];
        if (radix == 10 || radix <= 1 || radix > 36) {
            if (a.type == VAL_INT) snprintf(buf, sizeof(buf), "%lld", (long long)a.as.i);
            else snprintf(buf, sizeof(buf), "%.15g", as_number(a));
        } else {
            int64_t n = (a.type == VAL_INT) ? a.as.i : (int64_t)as_number(a);
            if (n == 0) { buf[0] = '0'; buf[1] = '\0'; }
            else {
                static const char digits[] = "0123456789abcdefghijklmnopqrstuvwxyz";
                char tmp[128]; int pos = 0, neg = (n < 0);
                uint64_t un = neg ? (uint64_t)(-(n + 1)) + 1 : (uint64_t)n;
                while (un > 0) { tmp[pos++] = digits[un % (uint64_t)radix]; un /= (uint64_t)radix; }
                if (neg) tmp[pos++] = '-';
                for (int i = 0; i < pos; i++) buf[i] = tmp[pos - 1 - i];
                buf[pos] = '\0';
            }
        }
        VmString* s = vm_string_from_cstr(&vm->heap.regions, buf);
        if (s) {
            int32_t ptr = heap_alloc(&vm->heap);
            if (ptr >= 0) {
                vm->heap.objects[ptr]->type = HEAP_STRING;
                vm->heap.objects[ptr]->opaque.ptr = s;
                vm_push(vm, (Value){.type = VAL_STRING, .as.ptr = ptr});
                break;
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 55: { Value b = vm_pop(vm); Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(as_number(a) == as_number(b))); break; }

    /* ══════════════════════════════════════════════════════════════════════
     * I/O (60-61)
     * ══════════════════════════════════════════════════════════════════════ */
    case 60: printf("\n"); fflush(stdout); vm_push(vm, (Value){.type = VAL_VOID}); break;
    case 61: { Value v = vm_pop(vm); print_value(vm, v); fflush(stdout); vm_push(vm, (Value){.type = VAL_VOID}); break; }

    /* ══════════════════════════════════════════════════════════════════════
     * List/apply (70-73)
     * ══════════════════════════════════════════════════════════════════════ */
    case 70: { /* apply */
        Value args_list = vm_pop(vm);
        Value func = vm_pop(vm);
        if (func.type != VAL_CLOSURE) { vm->error = 1; break; }
        /* Push closure below args (OP_CALL convention: closure at fp-1) */
        vm_push(vm, func);
        int argc = 0;
        Value cur = args_list;
        while (cur.type == VAL_PAIR && argc < 16) {
            vm_push(vm, vm->heap.objects[cur.as.ptr]->cons.car);
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
            argc++;
        }
        HeapObject* cl70 = vm->heap.objects[func.as.ptr];
        if (vm->frame_count >= MAX_FRAMES) { vm->error = 1; break; }
        vm->frames[vm->frame_count].return_pc = vm->pc;
        vm->frames[vm->frame_count].return_fp = vm->fp;
        vm->frames[vm->frame_count].func_pc = cl70->closure.func_pc;
        vm->frame_count++;
        vm->fp = vm->sp - argc;
        vm->pc = cl70->closure.func_pc;
        break;
    }
    case 71: { /* length */
        Value lst = vm_pop(vm);
        int len = 0;
        while (lst.type == VAL_PAIR) { len++; if (len > 1000000) { vm->error = 1; break; } lst = vm->heap.objects[lst.as.ptr]->cons.cdr; }
        if (vm->error) break;
        vm_push(vm, INT_VAL(len));
        break;
    }
    case 72: { /* reverse */
        Value lst = vm_pop(vm);
        Value result = NIL_VAL;
        while (lst.type == VAL_PAIR) {
            Value car = vm->heap.objects[lst.as.ptr]->cons.car;
            int32_t ptr = heap_alloc(&vm->heap);
            if (ptr < 0) { vm->error = 1; break; }
            vm->heap.objects[ptr]->type = HEAP_CONS;
            vm->heap.objects[ptr]->cons.car = car;
            vm->heap.objects[ptr]->cons.cdr = result;
            result = PAIR_VAL(ptr);
            lst = vm->heap.objects[lst.as.ptr]->cons.cdr;
        }
        vm_push(vm, result);
        break;
    }
    case 73: { /* append */
        Value b = vm_pop(vm), a = vm_pop(vm);
        if (a.type == VAL_NIL) { vm_push(vm, b); break; }
        Value rev = NIL_VAL;
        Value cur2 = a;
        while (cur2.type == VAL_PAIR) {
            int32_t p = heap_alloc(&vm->heap);
            if (p < 0) { vm->error = 1; break; }
            vm->heap.objects[p]->type = HEAP_CONS;
            vm->heap.objects[p]->cons.car = vm->heap.objects[cur2.as.ptr]->cons.car;
            vm->heap.objects[p]->cons.cdr = rev;
            rev = PAIR_VAL(p);
            cur2 = vm->heap.objects[cur2.as.ptr]->cons.cdr;
        }
        Value result2 = b;
        while (rev.type == VAL_PAIR) {
            int32_t p = heap_alloc(&vm->heap);
            if (p < 0) { vm->error = 1; break; }
            vm->heap.objects[p]->type = HEAP_CONS;
            vm->heap.objects[p]->cons.car = vm->heap.objects[rev.as.ptr]->cons.car;
            vm->heap.objects[p]->cons.cdr = result2;
            result2 = PAIR_VAL(p);
            rev = vm->heap.objects[rev.as.ptr]->cons.cdr;
        }
        vm_push(vm, result2);
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * List search: member (137), assoc (138)
     * ══════════════════════════════════════════════════════════════════════ */
    case 137: { /* member: (member obj list) — returns sublist starting from obj, or #f */
        Value lst = vm_pop(vm), obj = vm_pop(vm);
        while (lst.type == VAL_PAIR) {
            Value car = vm->heap.objects[lst.as.ptr]->cons.car;
            if (vm_deep_equal(vm, car, obj)) {
                vm_push(vm, lst); break;
            }
            lst = vm->heap.objects[lst.as.ptr]->cons.cdr;
        }
        if (lst.type != VAL_PAIR) vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 138: { /* assoc: (assoc key alist) — returns (key . val) pair, or #f */
        Value alist = vm_pop(vm), key = vm_pop(vm);
        int found = 0;
        while (alist.type == VAL_PAIR) {
            Value pair = vm->heap.objects[alist.as.ptr]->cons.car;
            if (pair.type == VAL_PAIR) {
                Value car = vm->heap.objects[pair.as.ptr]->cons.car;
                if (vm_deep_equal(vm, car, key)) {
                    vm_push(vm, pair); found = 1; break;
                }
            }
            alist = vm->heap.objects[alist.as.ptr]->cons.cdr;
        }
        if (!found) vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 139: { /* memq: (memq obj list) — identity-based search (eq?) */
        Value lst = vm_pop(vm), obj = vm_pop(vm);
        while (lst.type == VAL_PAIR) {
            Value car = vm->heap.objects[lst.as.ptr]->cons.car;
            if (vm_identity_equal(vm, car, obj)) {
                vm_push(vm, lst); break;
            }
            lst = vm->heap.objects[lst.as.ptr]->cons.cdr;
        }
        if (lst.type != VAL_PAIR) vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 140: { /* vector->list: preserve element order */
        Value vec_val = vm_pop(vm);
        if (vec_val.type != VAL_VECTOR ||
            !is_valid_heap_ptr(vm, vec_val.as.ptr) ||
            vm->heap.objects[vec_val.as.ptr]->type != HEAP_VECTOR) {
            fprintf(stderr, "ERROR: vector->list expects a vector\n");
            vm->error = 1;
            break;
        }
        VmVector* vec = (VmVector*)vm->heap.objects[vec_val.as.ptr]->opaque.ptr;
        Value result = NIL_VAL;
        if (vec) {
            for (int i = vec->len - 1; i >= 0; i--) {
                int32_t p = heap_alloc(&vm->heap);
                if (p < 0) { vm->error = 1; break; }
                vm->heap.objects[p]->type = HEAP_CONS;
                vm->heap.objects[p]->cons.car = vec->items[i];
                vm->heap.objects[p]->cons.cdr = result;
                result = PAIR_VAL(p);
            }
        }
        if (!vm->error) vm_push(vm, result);
        break;
    }

    case 227: { /* list->vector: (list->vector lst) → #(elts...)
                 * Native id 227: previously the name table aliased
                 * list->vector to id 139 (memq), so it silently returned #f. */
        Value lst = vm_pop(vm);
        int n = 0;
        for (Value cur = lst; cur.type == VAL_PAIR; cur = vm->heap.objects[cur.as.ptr]->cons.cdr) n++;
        int32_t p = heap_alloc(&vm->heap); if (p < 0) { vm->error = 1; break; }
        vm->heap.objects[p]->type = HEAP_VECTOR;
        VmVector* v = (VmVector*)vm_alloc(&vm->heap.regions, sizeof(VmVector));
        if (!v) { vm->error = 1; break; }
        v->len = n; v->cap = n;
        v->items = n ? (Value*)vm_alloc(&vm->heap.regions, (size_t)n * sizeof(Value)) : NULL;
        if (n && !v->items) { vm->error = 1; break; }
        int i = 0;
        for (Value cur = lst; cur.type == VAL_PAIR; cur = vm->heap.objects[cur.as.ptr]->cons.cdr)
            v->items[i++] = vm->heap.objects[cur.as.ptr]->cons.car;
        vm->heap.objects[p]->opaque.ptr = v;
        vm_push(vm, (Value){VAL_VECTOR, {.ptr = p}});
        break;
    }

    case 228: { /* char-from-int: tag an integer codepoint as a VAL_CHAR.
                 * Emitted by the compiler for #\x literals so display/char?
                 * distinguish a character from its integer code point. */
        Value a = vm_pop(vm);
        vm_push(vm, (Value){.type = VAL_CHAR, .as.i = (int64_t)as_number(a)});
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Make-vector (260)
     * ══════════════════════════════════════════════════════════════════════ */
    case 260: {
        Value fill = vm_pop(vm);
        Value n_val = vm_pop(vm);
        (void)fill;
        int n = (int)as_number(n_val);
        if (n < 0) n = 0;
        if (n > 256) n = 256;
        int32_t ptr = heap_alloc(&vm->heap);
        if (ptr < 0) { vm->error = 1; break; }
        vm->heap.objects[ptr]->type = HEAP_VECTOR;
        vm_push(vm, (Value){.type = VAL_VECTOR, .as.ptr = ptr});
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Complex Number Operations (300-319)
     * ══════════════════════════════════════════════════════════════════════ */
    case 300: case 301: case 302: case 303: case 304: case 305: case 306:
    case 307: case 308: case 309: case 310: case 311: case 312: case 313:
    case 314: case 315: case 316: case 317: case 318: case 319: {
        if (fid == 300) { /* make-rectangular */
            Value imag = vm_pop(vm), real = vm_pop(vm);
            VmComplex* z = vm_complex_new(&vm->heap.regions, as_number(real), as_number(imag));
            int32_t ptr = heap_alloc(&vm->heap);
            if (ptr < 0 || !z) { vm->error = 1; break; }
            vm->heap.objects[ptr]->type = HEAP_COMPLEX;
            vm->heap.objects[ptr]->opaque.ptr = z;
            vm_push(vm, (Value){.type = VAL_COMPLEX, .as.ptr = ptr});
        } else if (fid == 301) { /* make-polar */
            /* Arguments are pushed left-to-right, so angle is on top.  This
             * must be handled before the unary complex-operation path: that
             * path used to pop angle as the magnitude and then magnitude as
             * the angle, silently constructing the wrong complex number. */
            Value angle = vm_pop(vm), magnitude = vm_pop(vm);
            VmComplex* z = vm_make_polar(&vm->heap.regions,
                                          as_number(magnitude),
                                          as_number(angle));
            int32_t ptr = heap_alloc(&vm->heap);
            if (ptr < 0 || !z) { vm->error = 1; break; }
            vm->heap.objects[ptr]->type = HEAP_COMPLEX;
            vm->heap.objects[ptr]->opaque.ptr = z;
            vm_push(vm, (Value){.type = VAL_COMPLEX, .as.ptr = ptr});
        } else if (fid == 302) { /* real-part */
            Value z_val = vm_pop(vm);
            if (z_val.type == VAL_COMPLEX) {
                VmComplex* z = (VmComplex*)vm->heap.objects[z_val.as.ptr]->opaque.ptr;
                vm_push(vm, FLOAT_VAL(z->real));
            } else { vm_push(vm, FLOAT_VAL(as_number(z_val))); }
        } else if (fid == 303) { /* imag-part */
            Value z_val = vm_pop(vm);
            if (z_val.type == VAL_COMPLEX) {
                VmComplex* z = (VmComplex*)vm->heap.objects[z_val.as.ptr]->opaque.ptr;
                vm_push(vm, FLOAT_VAL(z->imag));
            } else { vm_push(vm, FLOAT_VAL(0.0)); }
        } else if (fid == 304) { /* magnitude */
            Value z_val = vm_pop(vm);
            if (z_val.type == VAL_COMPLEX) {
                VmComplex* z = (VmComplex*)vm->heap.objects[z_val.as.ptr]->opaque.ptr;
                vm_push(vm, FLOAT_VAL(vm_complex_magnitude(z)));
            } else { vm_push(vm, FLOAT_VAL(fabs(as_number(z_val)))); }
        } else if (fid == 317) { /* complex? */
            Value v = vm_pop(vm);
            vm_push(vm, BOOL_VAL(v.type == VAL_COMPLEX));
        } else {
            int is_binary = (fid >= 307 && fid <= 310) || fid == 318 || fid == 319;
            if (is_binary) {
                Value b_val = vm_pop(vm), a_val = vm_pop(vm);
                VmComplex a_z = {as_number(a_val), 0}, b_z = {as_number(b_val), 0};
                if (a_val.type == VAL_COMPLEX) a_z = *(VmComplex*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
                if (b_val.type == VAL_COMPLEX) b_z = *(VmComplex*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
                VmComplex* result = NULL;
                switch (fid) {
                    case 307: result = vm_complex_add(&vm->heap.regions, &a_z, &b_z); break;
                    case 308: result = vm_complex_sub(&vm->heap.regions, &a_z, &b_z); break;
                    case 309: result = vm_complex_mul(&vm->heap.regions, &a_z, &b_z); break;
                    case 310: result = vm_complex_div(&vm->heap.regions, &a_z, &b_z); break;
                    case 318: result = vm_complex_expt(&vm->heap.regions, &a_z, &b_z); break;
                    case 319: vm_push(vm, BOOL_VAL(a_z.real == b_z.real && a_z.imag == b_z.imag)); break;
                }
                if (fid != 319) {
                    if (!result) { vm->error = 1; break; }
                    int32_t ptr = heap_alloc(&vm->heap);
                    if (ptr < 0) { vm->error = 1; break; }
                    vm->heap.objects[ptr]->type = HEAP_COMPLEX;
                    vm->heap.objects[ptr]->opaque.ptr = result;
                    vm_push(vm, (Value){.type = VAL_COMPLEX, .as.ptr = ptr});
                }
            } else {
                Value a_val = vm_pop(vm);
                VmComplex a_z = {as_number(a_val), 0};
                if (a_val.type == VAL_COMPLEX) a_z = *(VmComplex*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
                VmComplex* result = NULL;
                switch (fid) {
                    case 305: vm_push(vm, FLOAT_VAL(vm_complex_angle(&a_z))); break;
                    case 306: result = vm_complex_conjugate(&vm->heap.regions, &a_z); break;
                    case 311: result = vm_complex_sqrt(&vm->heap.regions, &a_z); break;
                    case 312: result = vm_complex_exp(&vm->heap.regions, &a_z); break;
                    case 313: result = vm_complex_log(&vm->heap.regions, &a_z); break;
                    case 314: result = vm_complex_sin(&vm->heap.regions, &a_z); break;
                    case 315: result = vm_complex_cos(&vm->heap.regions, &a_z); break;
                    case 316: result = vm_complex_tan(&vm->heap.regions, &a_z); break;
                }
                if (fid != 305) {
                    if (!result) { vm->error = 1; break; }
                    int32_t ptr = heap_alloc(&vm->heap);
                    if (ptr < 0) { vm->error = 1; break; }
                    vm->heap.objects[ptr]->type = HEAP_COMPLEX;
                    vm->heap.objects[ptr]->opaque.ptr = result;
                    vm_push(vm, (Value){.type = VAL_COMPLEX, .as.ptr = ptr});
                }
            }
        }
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Rational Number Operations (330-349)
     * ══════════════════════════════════════════════════════════════════════ */
    case 330: case 331: case 332: case 333: case 334: case 335: case 336:
    case 337: case 338: case 339: case 340: case 341: case 342: case 343:
    case 344: case 345: case 346: case 347: case 348: case 349: {
        VmArena* rat_arena = vm_active_arena(&vm->heap.regions);
        switch (fid) {
        case 330: { Value denom = vm_pop(vm), num = vm_pop(vm);
            VmRational* r = vm_rational_make(rat_arena, (int64_t)as_number(num), (int64_t)as_number(denom));
            if (!r) { vm_push(vm, NIL_VAL); break; }
            int32_t ptr = heap_alloc(&vm->heap); if (ptr < 0) { vm->error = 1; break; }
            vm->heap.objects[ptr]->type = HEAP_RATIONAL; vm->heap.objects[ptr]->opaque.ptr = r;
            vm_push(vm, (Value){.type = VAL_RATIONAL, .as.ptr = ptr}); break; }
        case 331: case 332: case 333: case 334: {
            Value b_val = vm_pop(vm), a_val = vm_pop(vm);
            VmRational a_r = {(int64_t)as_number(a_val), 1}, b_r = {(int64_t)as_number(b_val), 1};
            if (a_val.type == VAL_RATIONAL) a_r = *(VmRational*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
            if (b_val.type == VAL_RATIONAL) b_r = *(VmRational*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
            VmRational* result = NULL;
            switch (fid) {
                case 331: result = vm_rational_add(rat_arena, &a_r, &b_r); break;
                case 332: result = vm_rational_sub(rat_arena, &a_r, &b_r); break;
                case 333: result = vm_rational_mul(rat_arena, &a_r, &b_r); break;
                case 334: result = vm_rational_div(rat_arena, &a_r, &b_r); break;
            }
            if (!result) { vm_push(vm, FLOAT_VAL((double)a_r.num/(double)a_r.denom + (double)b_r.num/(double)b_r.denom)); break; }
            if (result->denom == 1) { vm_push(vm, INT_VAL(result->num)); break; }
            int32_t ptr = heap_alloc(&vm->heap); if (ptr < 0) { vm->error = 1; break; }
            vm->heap.objects[ptr]->type = HEAP_RATIONAL; vm->heap.objects[ptr]->opaque.ptr = result;
            vm_push(vm, (Value){.type = VAL_RATIONAL, .as.ptr = ptr}); break; }
        case 335: { Value v = vm_pop(vm);
            if (v.type == VAL_RATIONAL) {
                VmRational* r = (VmRational*)vm->heap.objects[v.as.ptr]->opaque.ptr;
                VmRational* res = vm_rational_neg(rat_arena, r);
                if (!res) { vm_push(vm, NIL_VAL); break; }
                int32_t ptr = heap_alloc(&vm->heap); if (ptr < 0) { vm->error = 1; break; }
                vm->heap.objects[ptr]->type = HEAP_RATIONAL; vm->heap.objects[ptr]->opaque.ptr = res;
                vm_push(vm, (Value){.type = VAL_RATIONAL, .as.ptr = ptr});
            } else vm_push(vm, number_val(-as_number(v))); break; }
        case 336: { Value v = vm_pop(vm);
            if (v.type == VAL_RATIONAL) {
                VmRational* r = (VmRational*)vm->heap.objects[v.as.ptr]->opaque.ptr;
                VmRational* res = vm_rational_abs(rat_arena, r);
                if (!res) { vm_push(vm, NIL_VAL); break; }
                int32_t ptr = heap_alloc(&vm->heap); if (ptr < 0) { vm->error = 1; break; }
                vm->heap.objects[ptr]->type = HEAP_RATIONAL; vm->heap.objects[ptr]->opaque.ptr = res;
                vm_push(vm, (Value){.type = VAL_RATIONAL, .as.ptr = ptr});
            } else vm_push(vm, number_val(fabs(as_number(v)))); break; }
        case 337: { Value v = vm_pop(vm);
            if (v.type == VAL_RATIONAL) {
                VmRational* r = (VmRational*)vm->heap.objects[v.as.ptr]->opaque.ptr;
                VmRational* res = vm_rational_inv(rat_arena, r);
                if (!res) { vm_push(vm, NIL_VAL); break; }
                int32_t ptr = heap_alloc(&vm->heap); if (ptr < 0) { vm->error = 1; break; }
                vm->heap.objects[ptr]->type = HEAP_RATIONAL; vm->heap.objects[ptr]->opaque.ptr = res;
                vm_push(vm, (Value){.type = VAL_RATIONAL, .as.ptr = ptr});
            } else vm_push(vm, FLOAT_VAL(1.0 / as_number(v))); break; }
        case 338: { Value b_val = vm_pop(vm), a_val = vm_pop(vm);
            VmRational a_r = {(int64_t)as_number(a_val), 1}, b_r = {(int64_t)as_number(b_val), 1};
            if (a_val.type == VAL_RATIONAL) a_r = *(VmRational*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
            if (b_val.type == VAL_RATIONAL) b_r = *(VmRational*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
            vm_push(vm, INT_VAL(vm_rational_compare(&a_r, &b_r))); break; }
        case 339: { Value b_val = vm_pop(vm), a_val = vm_pop(vm);
            VmRational a_r = {(int64_t)as_number(a_val), 1}, b_r = {(int64_t)as_number(b_val), 1};
            if (a_val.type == VAL_RATIONAL) a_r = *(VmRational*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
            if (b_val.type == VAL_RATIONAL) b_r = *(VmRational*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
            vm_push(vm, BOOL_VAL(vm_rational_equal(&a_r, &b_r))); break; }
        case 340: { Value v = vm_pop(vm);
            if (v.type == VAL_RATIONAL) { VmRational* r = (VmRational*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, FLOAT_VAL(vm_rational_to_double(r))); }
            else vm_push(vm, FLOAT_VAL(as_number(v))); break; }
        case 341: { Value v = vm_pop(vm);
            VmRational* r = vm_rational_from_int(rat_arena, (int64_t)as_number(v));
            if (!r) { vm_push(vm, NIL_VAL); break; }
            int32_t ptr = heap_alloc(&vm->heap); if (ptr < 0) { vm->error = 1; break; }
            vm->heap.objects[ptr]->type = HEAP_RATIONAL; vm->heap.objects[ptr]->opaque.ptr = r;
            vm_push(vm, (Value){.type = VAL_RATIONAL, .as.ptr = ptr}); break; }
        case 342: { Value v = vm_pop(vm);
            if (v.type == VAL_RATIONAL) { VmRational* r = (VmRational*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, INT_VAL(vm_rational_floor(r))); }
            else vm_push(vm, INT_VAL((int64_t)floor(as_number(v)))); break; }
        case 343: { Value v = vm_pop(vm);
            if (v.type == VAL_RATIONAL) { VmRational* r = (VmRational*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, INT_VAL(vm_rational_ceil(r))); }
            else vm_push(vm, INT_VAL((int64_t)ceil(as_number(v)))); break; }
        case 344: { Value v = vm_pop(vm);
            if (v.type == VAL_RATIONAL) { VmRational* r = (VmRational*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, INT_VAL(vm_rational_truncate(r))); }
            else vm_push(vm, INT_VAL((int64_t)as_number(v))); break; }
        case 345: { Value v = vm_pop(vm);
            if (v.type == VAL_RATIONAL) { VmRational* r = (VmRational*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, INT_VAL(vm_rational_round(r))); }
            else vm_push(vm, INT_VAL((int64_t)vm_round_half_even(as_number(v)))); break; }
        case 346: { Value v = vm_pop(vm);
            if (v.type == VAL_RATIONAL) { VmRational* r = (VmRational*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, INT_VAL(vm_rational_numerator(r))); }
            else vm_push(vm, INT_VAL((int64_t)as_number(v))); break; }
        case 347: { Value v = vm_pop(vm);
            if (v.type == VAL_RATIONAL) { VmRational* r = (VmRational*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, INT_VAL(vm_rational_denominator(r))); }
            else vm_push(vm, INT_VAL(1)); break; }
        case 348: { Value b_v = vm_pop(vm), a_v = vm_pop(vm);
            vm_push(vm, INT_VAL(vm_rational_gcd((int64_t)as_number(a_v), (int64_t)as_number(b_v)))); break; }
        case 349: { Value tol = vm_pop(vm), x = vm_pop(vm);
            VmRational* r = vm_rationalize(rat_arena, as_number(x), as_number(tol));
            if (!r) { vm_push(vm, NIL_VAL); break; }
            if (r->denom == 1) { vm_push(vm, INT_VAL(r->num)); break; }
            int32_t ptr = heap_alloc(&vm->heap); if (ptr < 0) { vm->error = 1; break; }
            vm->heap.objects[ptr]->type = HEAP_RATIONAL; vm->heap.objects[ptr]->opaque.ptr = r;
            vm_push(vm, (Value){.type = VAL_RATIONAL, .as.ptr = ptr}); break; }
        default: vm_push(vm, NIL_VAL); break;
        }
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Bignum Operations (350-369)
     * ══════════════════════════════════════════════════════════════════════ */
    case 350: case 351: case 352: case 353: case 354: case 355: case 356:
    case 357: case 358: case 359: case 360: case 361: case 362: case 363:
    case 364: case 365: case 366: case 367: case 368: case 369: {
        VmRegionStack* bn_rs = &vm->heap.regions;
        switch (fid) {
        case 350: { Value v = vm_pop(vm); VmBignum* b = bignum_from_int64(bn_rs, (int64_t)as_number(v));
            if (!b) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_BIGNUM, VAL_BIGNUM, b); break; }
        case 351: { Value v = vm_pop(vm);
            const char* s = (v.type == VAL_STRING && vm->heap.objects[v.as.ptr]->opaque.ptr) ? (const char*)vm->heap.objects[v.as.ptr]->opaque.ptr : "0";
            VmBignum* b = bignum_from_string(bn_rs, s);
            if (!b) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_BIGNUM, VAL_BIGNUM, b); break; }
        case 352: { Value v = vm_pop(vm);
            if (v.type == VAL_BIGNUM) {
                VmBignum* b = (VmBignum*)vm->heap.objects[v.as.ptr]->opaque.ptr;
                char* s = bignum_to_string(bn_rs, b);
                if (!s) { vm->error = 1; break; }
                VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s);
            } else {
                char buf[32]; snprintf(buf, sizeof(buf), "%lld", (long long)(int64_t)as_number(v));
                char* s = (char*)vm_alloc(bn_rs, strlen(buf)+1); if (s) strcpy(s, buf);
                if (!s) { vm->error = 1; break; }
                VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s);
            } break; }
        case 353: { Value v = vm_pop(vm);
            if (v.type == VAL_BIGNUM) { VmBignum* b = (VmBignum*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, FLOAT_VAL(bignum_to_double(b))); }
            else vm_push(vm, FLOAT_VAL(as_number(v))); break; }
        case 354: { Value v = vm_pop(vm);
            if (v.type == VAL_BIGNUM) { VmBignum* b = (VmBignum*)vm->heap.objects[v.as.ptr]->opaque.ptr; int ov=0; vm_push(vm, INT_VAL(bignum_to_int64(b, &ov))); }
            else vm_push(vm, INT_VAL((int64_t)as_number(v))); break; }
        case 355: case 356: case 357: case 358: case 359: {
            Value b_val = vm_pop(vm), a_val = vm_pop(vm);
            VmBignum* a_bn = (a_val.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[a_val.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(a_val));
            VmBignum* b_bn = (b_val.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[b_val.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(b_val));
            if (!a_bn || !b_bn) { vm_push(vm, NIL_VAL); break; }
            VmBignum* result = NULL;
            switch (fid) { case 355: result=bignum_add(bn_rs,a_bn,b_bn); break; case 356: result=bignum_sub(bn_rs,a_bn,b_bn); break;
                case 357: result=bignum_mul(bn_rs,a_bn,b_bn); break; case 358: result=bignum_div(bn_rs,a_bn,b_bn); break; case 359: result=bignum_mod(bn_rs,a_bn,b_bn); break; }
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_BIGNUM, VAL_BIGNUM, result); break; }
        case 360: { Value v = vm_pop(vm);
            VmBignum* b = (v.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[v.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(v));
            if (!b) { vm_push(vm, NIL_VAL); break; }
            VmBignum* result = bignum_neg(bn_rs, b);
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_BIGNUM, VAL_BIGNUM, result); break; }
        case 361: { Value v = vm_pop(vm);
            VmBignum* b = (v.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[v.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(v));
            if (!b) { vm_push(vm, NIL_VAL); break; }
            VmBignum* result = bignum_abs_val(bn_rs, b);
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_BIGNUM, VAL_BIGNUM, result); break; }
        case 362: { Value exp_val = vm_pop(vm), base_val = vm_pop(vm);
            VmBignum* base_bn = (base_val.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[base_val.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(base_val));
            if (!base_bn) { vm_push(vm, NIL_VAL); break; }
            VmBignum* result = bignum_pow(bn_rs, base_bn, (uint64_t)(int64_t)as_number(exp_val));
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_BIGNUM, VAL_BIGNUM, result); break; }
        case 363: { Value b_val = vm_pop(vm), a_val = vm_pop(vm);
            VmBignum* a_bn = (a_val.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[a_val.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(a_val));
            VmBignum* b_bn = (b_val.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[b_val.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(b_val));
            if (!a_bn || !b_bn) { vm_push(vm, NIL_VAL); break; }
            vm_push(vm, INT_VAL(bignum_compare(a_bn, b_bn))); break; }
        case 364: { Value b_val = vm_pop(vm), a_val = vm_pop(vm);
            VmBignum* a_bn = (a_val.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[a_val.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(a_val));
            VmBignum* b_bn = (b_val.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[b_val.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(b_val));
            if (!a_bn || !b_bn) { vm_push(vm, NIL_VAL); break; }
            VmBignum* result = bignum_gcd(bn_rs, a_bn, b_bn);
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_BIGNUM, VAL_BIGNUM, result); break; }
        case 365: case 366: case 367: {
            Value b_val = vm_pop(vm), a_val = vm_pop(vm);
            VmBignum* a_bn = (a_val.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[a_val.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(a_val));
            VmBignum* b_bn = (b_val.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[b_val.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(b_val));
            if (!a_bn || !b_bn) { vm_push(vm, NIL_VAL); break; }
            VmBignum* result = NULL;
            switch (fid) { case 365: result=bignum_bitwise_and(bn_rs,a_bn,b_bn); break; case 366: result=bignum_bitwise_or(bn_rs,a_bn,b_bn); break; case 367: result=bignum_bitwise_xor(bn_rs,a_bn,b_bn); break; }
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_BIGNUM, VAL_BIGNUM, result); break; }
        case 368: { Value v = vm_pop(vm);
            VmBignum* b = (v.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[v.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(v));
            if (!b) { vm_push(vm, NIL_VAL); break; }
            VmBignum* result = bignum_bitwise_not(bn_rs, b);
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_BIGNUM, VAL_BIGNUM, result); break; }
        case 369: { Value shift_val = vm_pop(vm), v = vm_pop(vm);
            VmBignum* b = (v.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[v.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(v));
            if (!b) { vm_push(vm, NIL_VAL); break; }
            int shift = (int)as_number(shift_val);
            VmBignum* result = (shift >= 0) ? bignum_shift_left(bn_rs, b, shift) : bignum_shift_right(bn_rs, b, -shift);
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_BIGNUM, VAL_BIGNUM, result); break; }
        default: vm_push(vm, NIL_VAL); break;
        }
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Dual Number Operations (370-389)
     * ══════════════════════════════════════════════════════════════════════ */
    case 370: case 371: case 372: case 373: case 374: case 375: case 376:
    case 377: case 378: case 379: case 380: case 381: case 382: case 383:
    case 384: case 385: case 386: case 387: case 388: case 389: {
        VmRegionStack* dual_rs = &vm->heap.regions;
        switch (fid) {
        case 370: { Value tangent = vm_pop(vm), primal = vm_pop(vm);
            VmDual* d = vm_dual_make(dual_rs, as_number(primal), as_number(tangent));
            if (!d) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_DUAL, VAL_DUAL, d); break; }
        case 371: { Value v = vm_pop(vm);
            if (v.type == VAL_DUAL) { VmDual* d = (VmDual*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, FLOAT_VAL(d->primal)); }
            else vm_push(vm, FLOAT_VAL(as_number(v))); break; }
        case 372: { Value v = vm_pop(vm);
            if (v.type == VAL_DUAL) { VmDual* d = (VmDual*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, FLOAT_VAL(d->tangent)); }
            else vm_push(vm, FLOAT_VAL(0.0)); break; }
        case 373: case 374: case 375: case 376: {
            Value b_val = vm_pop(vm), a_val = vm_pop(vm);
            VmDual a_d = {as_number(a_val), 0.0}, b_d = {as_number(b_val), 0.0};
            if (a_val.type == VAL_DUAL) a_d = *(VmDual*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
            if (b_val.type == VAL_DUAL) b_d = *(VmDual*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
            VmDual* result = NULL;
            switch (fid) { case 373: result=vm_dual_add(dual_rs,&a_d,&b_d); break; case 374: result=vm_dual_sub(dual_rs,&a_d,&b_d); break;
                case 375: result=vm_dual_mul(dual_rs,&a_d,&b_d); break; case 376: result=vm_dual_div(dual_rs,&a_d,&b_d); break; }
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_DUAL, VAL_DUAL, result); break; }
        case 377: case 378: case 379: case 380: case 381:
        case 383: case 384: case 385: case 386: case 387: {
            Value v = vm_pop(vm);
            VmDual a_d = {as_number(v), 0.0};
            if (v.type == VAL_DUAL) a_d = *(VmDual*)vm->heap.objects[v.as.ptr]->opaque.ptr;
            VmDual* result = NULL;
            switch (fid) { case 377: result=vm_dual_sin(dual_rs,&a_d); break; case 378: result=vm_dual_cos(dual_rs,&a_d); break;
                case 379: result=vm_dual_exp(dual_rs,&a_d); break; case 380: result=vm_dual_log(dual_rs,&a_d); break;
                case 381: result=vm_dual_sqrt(dual_rs,&a_d); break; case 383: result=vm_dual_abs(dual_rs,&a_d); break;
                case 384: result=vm_dual_neg(dual_rs,&a_d); break; case 385: result=vm_dual_relu(dual_rs,&a_d); break;
                case 386: result=vm_dual_sigmoid(dual_rs,&a_d); break; case 387: result=vm_dual_tanh(dual_rs,&a_d); break; }
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_DUAL, VAL_DUAL, result); break; }
        case 382: { Value exp_val = vm_pop(vm), base_val = vm_pop(vm);
            VmDual a_d = {as_number(base_val), 0.0};
            if (base_val.type == VAL_DUAL) a_d = *(VmDual*)vm->heap.objects[base_val.as.ptr]->opaque.ptr;
            VmDual* result = vm_dual_pow(dual_rs, &a_d, as_number(exp_val));
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_DUAL, VAL_DUAL, result); break; }
        case 388: { Value v = vm_pop(vm);
            VmDual* d = vm_dual_from_double(dual_rs, as_number(v));
            if (!d) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_DUAL, VAL_DUAL, d); break; }
        case 389: { Value dual_val = vm_pop(vm), scalar_val = vm_pop(vm);
            VmDual a_d = {as_number(dual_val), 0.0};
            if (dual_val.type == VAL_DUAL) a_d = *(VmDual*)vm->heap.objects[dual_val.as.ptr]->opaque.ptr;
            VmDual* result = vm_dual_scale(dual_rs, as_number(scalar_val), &a_d);
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_DUAL, VAL_DUAL, result); break; }
        default: vm_push(vm, NIL_VAL); break;
        }
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Hyper-Dual Numbers (1900-1921) — exact second derivatives
     * ══════════════════════════════════════════════════════════════════════ */
    case 1900: case 1901: case 1902: case 1903: case 1904:
    case 1905: case 1906: case 1907: case 1908: case 1909:
    case 1910: case 1911: case 1912: case 1913: case 1914:
    case 1915: case 1916: case 1917: case 1918: case 1919:
    case 1920: case 1921: {
        VmRegionStack* hd_rs = &vm->heap.regions;
        switch (fid) {
        case 1900: { /* make-hyper-dual(f, f1, f2, f12) */
            Value f12v = vm_pop(vm), f2v = vm_pop(vm), f1v = vm_pop(vm), fv = vm_pop(vm);
            VmHyperDual* h = vm_hd_make(hd_rs, as_number(fv), as_number(f1v), as_number(f2v), as_number(f12v));
            if (!h) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_HYPER_DUAL, VAL_HYPER_DUAL, h); break; }
        case 1901: { Value v = vm_pop(vm); /* f component */
            if (v.type == VAL_HYPER_DUAL) { VmHyperDual* h = (VmHyperDual*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, FLOAT_VAL(h->f)); }
            else vm_push(vm, FLOAT_VAL(as_number(v))); break; }
        case 1902: { Value v = vm_pop(vm); /* f1 component */
            if (v.type == VAL_HYPER_DUAL) { VmHyperDual* h = (VmHyperDual*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, FLOAT_VAL(h->f1)); }
            else vm_push(vm, FLOAT_VAL(0.0)); break; }
        case 1903: { Value v = vm_pop(vm); /* f2 component */
            if (v.type == VAL_HYPER_DUAL) { VmHyperDual* h = (VmHyperDual*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, FLOAT_VAL(h->f2)); }
            else vm_push(vm, FLOAT_VAL(0.0)); break; }
        case 1904: { Value v = vm_pop(vm); /* f12 — the second derivative */
            if (v.type == VAL_HYPER_DUAL) { VmHyperDual* h = (VmHyperDual*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, FLOAT_VAL(h->f12)); }
            else vm_push(vm, FLOAT_VAL(0.0)); break; }
        case 1905: case 1906: case 1907: case 1908: { /* binary: add/sub/mul/div */
            Value b_val = vm_pop(vm), a_val = vm_pop(vm);
            VmHyperDual a_h = {as_number(a_val), 0, 0, 0}, b_h = {as_number(b_val), 0, 0, 0};
            if (a_val.type == VAL_HYPER_DUAL) a_h = *(VmHyperDual*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
            if (b_val.type == VAL_HYPER_DUAL) b_h = *(VmHyperDual*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
            VmHyperDual* result = NULL;
            switch (fid) {
                case 1905: result = vm_hd_add(hd_rs, &a_h, &b_h); break;
                case 1906: result = vm_hd_sub(hd_rs, &a_h, &b_h); break;
                case 1907: result = vm_hd_mul(hd_rs, &a_h, &b_h); break;
                case 1908: result = vm_hd_div(hd_rs, &a_h, &b_h); break;
            }
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_HYPER_DUAL, VAL_HYPER_DUAL, result); break; }
        case 1909: case 1910: case 1911: case 1912: case 1913:
        case 1914: case 1916: case 1917: case 1918: case 1919: { /* unary */
            Value v = vm_pop(vm);
            VmHyperDual a_h = {as_number(v), 0, 0, 0};
            if (v.type == VAL_HYPER_DUAL) a_h = *(VmHyperDual*)vm->heap.objects[v.as.ptr]->opaque.ptr;
            VmHyperDual* result = NULL;
            switch (fid) {
                case 1909: result = vm_hd_neg(hd_rs, &a_h); break;
                case 1910: result = vm_hd_sin(hd_rs, &a_h); break;
                case 1911: result = vm_hd_cos(hd_rs, &a_h); break;
                case 1912: result = vm_hd_exp(hd_rs, &a_h); break;
                case 1913: result = vm_hd_log(hd_rs, &a_h); break;
                case 1914: result = vm_hd_sqrt(hd_rs, &a_h); break;
                case 1916: result = vm_hd_abs(hd_rs, &a_h); break;
                case 1917: result = vm_hd_relu(hd_rs, &a_h); break;
                case 1918: result = vm_hd_sigmoid(hd_rs, &a_h); break;
                case 1919: result = vm_hd_tanh(hd_rs, &a_h); break;
            }
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_HYPER_DUAL, VAL_HYPER_DUAL, result); break; }
        case 1915: { /* pow(base, exp) */
            Value exp_val = vm_pop(vm), base_val = vm_pop(vm);
            VmHyperDual a_h = {as_number(base_val), 0, 0, 0};
            if (base_val.type == VAL_HYPER_DUAL) a_h = *(VmHyperDual*)vm->heap.objects[base_val.as.ptr]->opaque.ptr;
            VmHyperDual* result = vm_hd_pow(hd_rs, &a_h, as_number(exp_val));
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_HYPER_DUAL, VAL_HYPER_DUAL, result); break; }
        case 1920: { /* from-double */
            Value v = vm_pop(vm);
            VmHyperDual* h = vm_hd_from_double(hd_rs, as_number(v));
            if (!h) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_HYPER_DUAL, VAL_HYPER_DUAL, h); break; }
        case 1921: { /* scale(scalar, hd) */
            Value hd_val = vm_pop(vm), scalar_val = vm_pop(vm);
            VmHyperDual a_h = {as_number(hd_val), 0, 0, 0};
            if (hd_val.type == VAL_HYPER_DUAL) a_h = *(VmHyperDual*)vm->heap.objects[hd_val.as.ptr]->opaque.ptr;
            VmHyperDual* result = vm_hd_scale(hd_rs, as_number(scalar_val), &a_h);
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_HYPER_DUAL, VAL_HYPER_DUAL, result); break; }
        default: vm_push(vm, NIL_VAL); break;
        }
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * AD Operations (390-409) — reverse-mode tape + forward-mode derivative
     * ══════════════════════════════════════════════════════════════════════ */
    case 2082: { /* ad-reset-counters! */
        vm->ad_primal_calls = 0;
        vm->ad_reverse_passes = 0;
        vm->ad_tape_allocations = 0;
        vm->ad_tape_nodes = 0;
        vm->ad_finite_difference_evals = 0;
        vm_push(vm, NIL_VAL);
        break;
    }
    case 2083: { vm_push(vm, INT_VAL((int64_t)vm->ad_primal_calls)); break; }
    case 2084: { vm_push(vm, INT_VAL((int64_t)vm->ad_reverse_passes)); break; }
    case 2085: { vm_push(vm, INT_VAL((int64_t)vm->ad_tape_allocations)); break; }
    case 2086: { vm_push(vm, INT_VAL((int64_t)vm->ad_finite_difference_evals)); break; }
    case 2087: { /* ad-counters → ordered five-entry association list */
        Value result = NIL_VAL;
        result = vm_cons_value(vm, vm_alist_entry(vm, "finite-difference-evals",
                               INT_VAL((int64_t)vm->ad_finite_difference_evals)), result);
        result = vm_cons_value(vm, vm_alist_entry(vm, "tape-nodes",
                               INT_VAL((int64_t)vm->ad_tape_nodes)), result);
        result = vm_cons_value(vm, vm_alist_entry(vm, "tape-allocations",
                               INT_VAL((int64_t)vm->ad_tape_allocations)), result);
        result = vm_cons_value(vm, vm_alist_entry(vm, "reverse-passes",
                               INT_VAL((int64_t)vm->ad_reverse_passes)), result);
        result = vm_cons_value(vm, vm_alist_entry(vm, "primal-calls",
                               INT_VAL((int64_t)vm->ad_primal_calls)), result);
        vm_push(vm, result);
        break;
    }
    case 390: { /* ad-tape-new */
        AdTape* tape = ad_tape_new(&vm->heap.regions);
        if (!tape) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_AD_TAPE, VAL_AD_TAPE, tape);
        break;
    }
    case 391: { /* ad-const(tape, value) */
        Value val = vm_pop(vm), tape_val = vm_pop(vm);
        AdTape* tape = vm_ad_tape_from_value(vm, tape_val);
        if (!tape) { vm_push(vm, NIL_VAL); break; }
        vm_push(vm, INT_VAL(ad_const(tape, as_number(val))));
        break;
    }
    case 392: { /* ad-var(tape, value) */
        Value val = vm_pop(vm), tape_val = vm_pop(vm);
        AdTape* tape = vm_ad_tape_from_value(vm, tape_val);
        if (!tape) { vm_push(vm, NIL_VAL); break; }
        vm_push(vm, INT_VAL(ad_var(tape, as_number(val))));
        break;
    }
    case 393: { /* derivative: (derivative f x) → f'(x) using forward-mode dual numbers */
        Value x_val = vm_pop(vm), f_val = vm_pop(vm);
        /* Create dual number: x + 1ε */
        VmDual* d = vm_dual_make(&vm->heap.regions, as_number(x_val), 1.0);
        if (!d) { vm_push(vm, FLOAT_VAL(0)); break; }
        int32_t dptr = heap_alloc(&vm->heap);
        if (dptr < 0) { vm->error = 1; break; }
        vm->heap.objects[dptr]->type = HEAP_DUAL;
        vm->heap.objects[dptr]->opaque.ptr = d;
        Value dual_arg = (Value){.type = VAL_DUAL, .as.ptr = dptr};
        /* Call f(dual) via closure bridge */
        Value result = vm_ad_call_closure(vm, f_val, &dual_arg, 1);
        /* Extract tangent = derivative */
        if (result.type == VAL_DUAL && result.as.ptr >= 0) {
            VmDual* rd = (VmDual*)vm->heap.objects[result.as.ptr]->opaque.ptr;
            vm_push(vm, FLOAT_VAL(rd ? rd->tangent : 0));
        } else {
            vm_push(vm, FLOAT_VAL(0)); /* non-dual result = constant function */
        }
        break;
    }
    case 394: case 395: case 396: case 397: { /* ad-add, ad-sub, ad-mul, ad-div(tape, left, right) */
        Value right = vm_pop(vm), left = vm_pop(vm), tape_val = vm_pop(vm);
        AdTape* tape = vm_ad_tape_from_value(vm, tape_val);
        if (!tape) { vm_push(vm, NIL_VAL); break; }
        int result = -1;
        switch (fid) {
            case 394: result = ad_add(tape, (int)left.as.i, (int)right.as.i); break;
            case 395: result = ad_sub(tape, (int)left.as.i, (int)right.as.i); break;
            case 396: result = ad_mul(tape, (int)left.as.i, (int)right.as.i); break;
            case 397: result = ad_div(tape, (int)left.as.i, (int)right.as.i); break;
        }
        vm_push(vm, INT_VAL(result));
        break;
    }
    case 398: case 399: case 400: case 401: case 402:
    case 403: case 404: case 405: case 406: case 407: { /* ad unary ops(tape, node) */
        Value node = vm_pop(vm), tape_val = vm_pop(vm);
        AdTape* tape = vm_ad_tape_from_value(vm, tape_val);
        if (!tape) { vm_push(vm, NIL_VAL); break; }
        int result = -1;
        int idx = (int)node.as.i;
        switch (fid) {
            case 398: result = ad_sin(tape, idx); break;
            case 399: result = ad_cos(tape, idx); break;
            case 400: result = ad_exp(tape, idx); break;
            case 401: result = ad_log(tape, idx); break;
            case 402: result = ad_sqrt(tape, idx); break;
            case 403: result = ad_neg(tape, idx); break;
            case 404: result = ad_abs(tape, idx); break;
            case 405: result = ad_relu(tape, idx); break;
            case 406: result = ad_sigmoid(tape, idx); break;
            case 407: result = ad_tanh(tape, idx); break;
        }
        vm_push(vm, INT_VAL(result));
        break;
    }
    case 408: { /* ad-backward(tape, output_node) */
        Value node = vm_pop(vm), tape_val = vm_pop(vm);
        AdTape* tape = vm_ad_tape_from_value(vm, tape_val);
        if (!tape) { vm_push(vm, NIL_VAL); break; }
        ad_backward(tape, (int)node.as.i);
        vm_push(vm, NIL_VAL); /* backward is side-effectful */
        break;
    }
    case 409: { /* ad-gradient(tape, node) → gradient value */
        Value node = vm_pop(vm), tape_val = vm_pop(vm);
        AdTape* tape = vm_ad_tape_from_value(vm, tape_val);
        if (!tape) { vm_push(vm, FLOAT_VAL(0.0)); break; }
        int idx = (int)node.as.i;
        if (idx >= 0 && idx < tape->len) {
            vm_push(vm, FLOAT_VAL(tape->nodes[idx].gradient));
        } else {
            vm_push(vm, FLOAT_VAL(0.0));
        }
        break;
    }
    case 1841: { /* ad-tape-release(tape) - VM arena-backed logical release */
        Value tape_val = vm_pop(vm);
        if (tape_val.type == VAL_AD_TAPE && is_heap_type(vm, tape_val, HEAP_AD_TAPE)) {
            vm->heap.objects[tape_val.as.ptr]->opaque.ptr = NULL;
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 1842: { /* ad-node-value/ad-value(tape, node) → forward value */
        Value node = vm_pop(vm), tape_val = vm_pop(vm);
        AdTape* tape = vm_ad_tape_from_value(vm, tape_val);
        if (!tape) { vm_push(vm, FLOAT_VAL(0.0)); break; }
        vm_push(vm, FLOAT_VAL(ad_get_value(tape, (int)node.as.i)));
        break;
    }
    case 1843: { /* ad-tape-length(tape) */
        Value tape_val = vm_pop(vm);
        AdTape* tape = vm_ad_tape_from_value(vm, tape_val);
        vm_push(vm, INT_VAL(tape ? tape->len : 0));
        break;
    }
    case 1844: { /* ad-pow(tape, base_node, exponent_node) */
        Value exponent = vm_pop(vm), base = vm_pop(vm), tape_val = vm_pop(vm);
        AdTape* tape = vm_ad_tape_from_value(vm, tape_val);
        if (!tape) { vm_push(vm, NIL_VAL); break; }
        vm_push(vm, INT_VAL(ad_pow(tape, (int)base.as.i, (int)exponent.as.i)));
        break;
    }
    case 1849: { /* dual? */
        Value value = vm_pop(vm);
        vm_push(vm, BOOL_VAL(value.type == VAL_DUAL));
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Tensor Core Operations (410-420)
     * ══════════════════════════════════════════════════════════════════════ */
    case 410: { /* make-tensor(shape, fill) */
        Value fill = vm_pop(vm), shape_val = vm_pop(vm);
        int64_t shape[8]; int n_dims = vm_extract_shape(vm, shape_val, shape, 8);
        if (n_dims == 0) { vm_push(vm, NIL_VAL); break; }
        VmTensor* t = vm_tensor_fill(&vm->heap.regions, shape, n_dims, as_number(fill));
        if (!t) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, t);
        break;
    }
    case 411: { /* tensor-ref(tensor, indices) — flat or multi-dim access */
        Value idx_val = vm_pop(vm), t_val = vm_pop(vm);
        VmTensor* t = vm_tensor_operand(vm, t_val, "tensor-ref");
        if (!t) { vm_push(vm, FLOAT_VAL(0)); break; }
        /* Single int/float index: flat access; list: multi-dim */
        if (idx_val.type == VAL_INT || idx_val.type == VAL_FLOAT) {
            int64_t flat = (int64_t)as_number(idx_val);
            if (flat >= 0 && flat < t->total)
                vm_push(vm, FLOAT_VAL(t->data[flat]));
            else vm_push(vm, FLOAT_VAL(0));
        } else if (idx_val.type == VAL_PAIR) {
            /* Multi-dim: walk list to get indices */
            int64_t indices[8]; int nd = 0;
            Value cur = idx_val;
            while (cur.type == VAL_PAIR && nd < 8) {
                indices[nd++] = (int64_t)as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
            }
            double val = vm_tensor_ref(t, indices, nd);
            vm_push(vm, FLOAT_VAL(val));
        } else {
            vm_push(vm, FLOAT_VAL(0));
        }
        break;
    }
    case 412: { /* tensor-set!(tensor, indices, value) — mutates in place, so the
                 * operand must already be a real tensor: coercing a vector here
                 * would silently mutate a throwaway copy and drop the write. */
        Value val = vm_pop(vm), idx_val = vm_pop(vm), t_val = vm_pop(vm);
        if (t_val.type != VAL_TENSOR) {
            fprintf(stderr, "ERROR: tensor-set!: expected a tensor operand\n");
            vm->error = 1; vm_push(vm, NIL_VAL); break;
        }
        VmTensor* t = vm_tensor_operand(vm, t_val, "tensor-set!");
        if (!t) { vm_push(vm, NIL_VAL); break; }
        int64_t indices[8]; int n = vm_extract_shape(vm, idx_val, indices, 8);
        if (n == 0) { indices[0] = (int64_t)as_number(idx_val); n = 1; }
        vm_tensor_set(t, indices, n, as_number(val));
        vm_push(vm, NIL_VAL);
        break;
    }
    case 413: { /* tensor-shape → list */
        Value t_val = vm_pop(vm);
        VmTensor* t = vm_tensor_operand(vm, t_val, "tensor-shape");
        if (!t) { vm_push(vm, NIL_VAL); break; }
        Value result = NIL_VAL;
        for (int i = t->n_dims - 1; i >= 0; i--) {
            int32_t p = heap_alloc(&vm->heap);
            if (p < 0) { vm->error = 1; break; }
            vm->heap.objects[p]->type = HEAP_CONS;
            vm->heap.objects[p]->cons.car = INT_VAL(t->shape[i]);
            vm->heap.objects[p]->cons.cdr = result;
            result = PAIR_VAL(p);
        }
        vm_push(vm, result);
        break;
    }
    case 414: { /* tensor-data → flat list (for small tensors) */
        Value t_val = vm_pop(vm);
        VmTensor* t = vm_tensor_operand(vm, t_val, "tensor-data");
        if (!t) { vm_push(vm, NIL_VAL); break; }
        Value result = NIL_VAL;
        int64_t limit = t->total > 1024 ? 1024 : t->total;
        for (int64_t i = limit - 1; i >= 0; i--) {
            int32_t p = heap_alloc(&vm->heap);
            if (p < 0) { vm->error = 1; break; }
            vm->heap.objects[p]->type = HEAP_CONS;
            vm->heap.objects[p]->cons.car = FLOAT_VAL(t->data[i]);
            vm->heap.objects[p]->cons.cdr = result;
            result = PAIR_VAL(p);
        }
        vm_push(vm, result);
        break;
    }
    case 415: { /* reshape(tensor, new_shape) */
        Value shape_val = vm_pop(vm), t_val = vm_pop(vm);
        VmTensor* t = vm_tensor_operand(vm, t_val, "reshape");
        if (!t) { vm_push(vm, NIL_VAL); break; }
        int64_t shape[8]; int n = vm_extract_shape(vm, shape_val, shape, 8);
        VmTensor* out = vm_tensor_reshape(&vm->heap.regions, t, shape, n);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 416: { /* transpose */
        Value t_val = vm_pop(vm);
        VmTensor* t = vm_tensor_operand(vm, t_val, "transpose");
        if (!t) { vm_push(vm, NIL_VAL); break; }
        VmTensor* out = vm_gpu_try_transpose(&vm->heap.regions, t);
        if (!out) out = vm_tensor_transpose(&vm->heap.regions, t);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 417: { /* zeros(shape) */
        Value shape_val = vm_pop(vm);
        int64_t shape[8]; int n = vm_extract_shape(vm, shape_val, shape, 8);
        if (n == 0) { vm_push(vm, NIL_VAL); break; }
        VmTensor* t = vm_tensor_zeros(&vm->heap.regions, shape, n);
        if (!t) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, t);
        break;
    }
    case 418: { /* ones(shape) */
        Value shape_val = vm_pop(vm);
        int64_t shape[8]; int n = vm_extract_shape(vm, shape_val, shape, 8);
        if (n == 0) { vm_push(vm, NIL_VAL); break; }
        VmTensor* t = vm_tensor_ones(&vm->heap.regions, shape, n);
        if (!t) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, t);
        break;
    }
    case 419: { /* arange(start, stop, step) */
        Value step = vm_pop(vm), stop = vm_pop(vm), start = vm_pop(vm);
        VmTensor* t = vm_tensor_arange(&vm->heap.regions, as_number(start), as_number(stop), as_number(step));
        if (!t) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, t);
        break;
    }
    case 420: { /* flatten */
        Value t_val = vm_pop(vm);
        VmTensor* t = vm_tensor_operand(vm, t_val, "flatten");
        if (!t) { vm_push(vm, NIL_VAL); break; }
        VmTensor* out = vm_tensor_flatten(&vm->heap.regions, t);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 421: { /* tensor-dtype(tensor) -> symbol name */
        Value t_val = vm_pop(vm);
        if (!is_heap_type(vm, t_val, HEAP_TENSOR)) { vm_push(vm, NIL_VAL); break; }
        VmTensor* t = (VmTensor*)vm->heap.objects[t_val.as.ptr]->opaque.ptr;
        if (!t) { vm_push(vm, NIL_VAL); break; }
        vm_push(vm, vm_string_value(vm, vm_tensor_dtype_name(t->dtype), -1));
        break;
    }
    case 422: { /* tensor-cast(tensor, dtype) */
        Value dtype_val = vm_pop(vm), t_val = vm_pop(vm);
        if (!is_heap_type(vm, t_val, HEAP_TENSOR)) { vm_push(vm, NIL_VAL); break; }
        VmTensor* t = (VmTensor*)vm->heap.objects[t_val.as.ptr]->opaque.ptr;
        VmString* dtype_name = vm_value_as_string(vm, dtype_val);
        if (!t || !dtype_name) { vm_push(vm, NIL_VAL); break; }
        VmTensor* out = vm_tensor_cast_dtype(&vm->heap.regions, t,
                                             vm_tensor_dtype_from_name(dtype_name->data));
        if (!out) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 423: { /* make-tensor(shape, fill, dtype) */
        Value dtype_val = vm_pop(vm), fill = vm_pop(vm), shape_val = vm_pop(vm);
        int64_t shape[8]; int n_dims = vm_extract_shape(vm, shape_val, shape, 8);
        VmString* dtype_name = vm_value_as_string(vm, dtype_val);
        if (n_dims == 0 || !dtype_name) { vm_push(vm, NIL_VAL); break; }
        VmTensor* t = vm_tensor_fill(&vm->heap.regions, shape, n_dims, as_number(fill));
        if (!t) { vm_push(vm, NIL_VAL); break; }
        t->dtype = vm_tensor_dtype_from_name(dtype_name->data);
        for (int64_t i = 0; i < t->total; i++) {
            t->data[i] = vm_tensor_quantize_value(t->data[i], t->dtype);
        }
        VM_PUSH_TENSOR(vm, t);
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Tensor Operations (440-469)
     * ══════════════════════════════════════════════════════════════════════ */
    case 440: { /* matmul — GPU dispatch if tensor is large enough */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        VmTensor* a = vm_tensor_operand(vm, a_val, "matmul");
        VmTensor* b = vm_tensor_operand(vm, b_val, "matmul");
        if (!a || !b) { vm_push(vm, NIL_VAL); break; }
        /* Try GPU first, fall through to CPU */
        VmTensor* out = vm_gpu_try_matmul(&vm->heap.regions, a, b);
        if (!out) out = vm_tensor_matmul(&vm->heap.regions, a, b);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        out->dtype = vm_tensor_promote_dtype(a, b);
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 441: case 442: case 443: case 444: case 445: case 446: case 447: { /* tensor binary: +,-,*,/,pow,max,min */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        VmTensor* a = vm_tensor_operand(vm, a_val, "tensor-binary-op");
        VmTensor* b = vm_tensor_operand(vm, b_val, "tensor-binary-op");
        if (!a || !b) { vm_push(vm, NIL_VAL); break; }
        /* GPU dispatch for add/sub/mul/div (ops 0-3) */
        VmTensor* out = NULL;
        static const int gpu_binary_ops[] = {0,1,2,3,-1,-1,-1}; /* add,sub,mul,div,pow,max,min */
        int gpu_op = gpu_binary_ops[fid - 441];
        if (gpu_op >= 0) out = vm_gpu_try_binary(&vm->heap.regions, a, b, gpu_op);
        if (!out) switch (fid) {
            case 441: out = vm_tensor_add(&vm->heap.regions, a, b); break;
            case 442: out = vm_tensor_sub(&vm->heap.regions, a, b); break;
            case 443: out = vm_tensor_mul(&vm->heap.regions, a, b); break;
            case 444: out = vm_tensor_div(&vm->heap.regions, a, b); break;
            case 445: out = vm_tensor_pow(&vm->heap.regions, a, b); break;
            case 446: out = vm_tensor_maximum(&vm->heap.regions, a, b); break;
            case 447: out = vm_tensor_minimum(&vm->heap.regions, a, b); break;
        }
        if (!out) { vm_push(vm, NIL_VAL); break; }
        out->dtype = vm_tensor_promote_dtype(a, b);
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 448: { /* batch-matmul */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        VmTensor* a = vm_tensor_operand(vm, a_val, "batch-matmul");
        VmTensor* b = vm_tensor_operand(vm, b_val, "batch-matmul");
        if (!a || !b) { vm_push(vm, NIL_VAL); break; }
        VmTensor* out = vm_tensor_batch_matmul(&vm->heap.regions, a, b);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 449: { /* dot */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        VmTensor* a = vm_tensor_operand(vm, a_val, "tensor-dot");
        VmTensor* b = vm_tensor_operand(vm, b_val, "tensor-dot");
        if (!a || !b) { vm_push(vm, FLOAT_VAL(0.0)); break; }
        vm_push(vm, FLOAT_VAL(vm_tensor_dot(a, b)));
        break;
    }
    case 450: case 451: case 452: case 453: case 454: case 455: { /* tensor unary: neg,abs,sqrt,exp,log,sin,cos */
        Value t_val = vm_pop(vm);
        VmTensor* t = vm_tensor_operand(vm, t_val, "tensor-unary-op");
        if (!t) { vm_push(vm, NIL_VAL); break; }
        VmTensor* out = NULL;
        switch (fid) {
            case 450: out = vm_tensor_neg(&vm->heap.regions, t); break;
            case 451: out = vm_tensor_abs(&vm->heap.regions, t); break;
            case 452: out = vm_tensor_sqrt_op(&vm->heap.regions, t); break;
            case 453: out = vm_tensor_exp_op(&vm->heap.regions, t); break;
            case 454: out = vm_tensor_log_op(&vm->heap.regions, t); break;
            case 455: out = vm_tensor_sin_op(&vm->heap.regions, t); break;
        }
        if (!out) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 456: { /* scale(tensor, scalar) */
        Value scalar = vm_pop(vm), t_val = vm_pop(vm);
        VmTensor* t = vm_tensor_operand(vm, t_val, "tensor-scale");
        if (!t) { vm_push(vm, NIL_VAL); break; }
        VmTensor* out = vm_tensor_scale(&vm->heap.regions, t, as_number(scalar));
        if (!out) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 457: case 458: case 459: case 460: { /* reduce: sum,mean,max,min (tensor, axis) */
        Value axis_val = vm_pop(vm), t_val = vm_pop(vm);
        VmTensor* t = vm_tensor_operand(vm, t_val, "tensor-reduce");
        if (!t) { vm_push(vm, NIL_VAL); break; }
        int axis = (int)as_number(axis_val);
        /* GPU dispatch for full-tensor reductions (axis=-1 or axis covers all) */
        VmTensor* out = NULL;
        if (axis < 0 || t->n_dims == 1) {
            static const int gpu_reduce_ops[] = {0, 4, 3, 2}; /* sum=0, mean=4, max=3, min=2 */
            double gpu_result = vm_gpu_try_reduce(t, gpu_reduce_ops[fid - 457]);
            if (!isnan(gpu_result)) {
                int64_t shape[1] = {1};
                out = vm_tensor_zeros(&vm->heap.regions, shape, 1);
                if (out) {
                    out->data[0] = gpu_result;
                    out->dtype = t->dtype;
                }
            }
        }
        if (!out) switch (fid) {
            case 457: out = vm_tensor_sum(&vm->heap.regions, t, axis); break;
            case 458: out = vm_tensor_mean(&vm->heap.regions, t, axis); break;
            case 459: out = vm_tensor_max(&vm->heap.regions, t, axis); break;
            case 460: out = vm_tensor_min(&vm->heap.regions, t, axis); break;
        }
        if (!out) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 461: { /* cos tensor */
        Value t_val = vm_pop(vm);
        VmTensor* t = vm_tensor_operand(vm, t_val, "tensor-cos");
        if (!t) { vm_push(vm, NIL_VAL); break; }
        VmTensor* out = vm_tensor_cos_op(&vm->heap.regions, t);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 462: case 463: case 464: case 465: case 466: case 467: case 468: { /* activations: relu,sigmoid,tanh,leaky_relu,elu,gelu,swish */
        Value t_val = vm_pop(vm);
        /* Scalar fallback: tensors now have dedicated VAL_TENSOR type.
         * Plain VAL_INT and VAL_FLOAT are genuine scalars. */
        int is_tensor_or_vector = (t_val.type == VAL_TENSOR || t_val.type == VAL_VECTOR);
        if (!is_tensor_or_vector && (t_val.type == VAL_INT || t_val.type == VAL_FLOAT)) {
            double x = as_number(t_val);
            double r;
            switch (fid) {
                case 462: r = x > 0 ? x : 0; break;            /* relu */
                case 464: r = 1.0 / (1.0 + exp(-x)); break;    /* sigmoid */
                case 465: r = x > 0 ? x : 0.01 * x; break;     /* leaky_relu */
                case 463: case 466: case 467: case 468:
                default:  r = x; break;
            }
            vm_push(vm, (Value){.type = VAL_FLOAT, .as.f = r});
            break;
        }
        if (!is_tensor_or_vector) { vm_push(vm, NIL_VAL); break; }
        VmTensor* t = vm_tensor_operand(vm, t_val, "tensor-activation");
        if (!t) { vm_push(vm, NIL_VAL); break; }
        VmTensor* out = NULL;
        /* GPU dispatch for softmax */
        if (fid == 463) out = vm_gpu_try_softmax(&vm->heap.regions, t);
        if (!out) switch (fid) {
            case 462: out = vm_tensor_relu(&vm->heap.regions, t); break;
            case 463: out = vm_tensor_softmax(&vm->heap.regions, t, t->n_dims - 1); break;
            case 464: out = vm_tensor_sigmoid(&vm->heap.regions, t); break;
            case 465: out = vm_tensor_tanh_act(&vm->heap.regions, t); break;
            case 466: out = vm_tensor_leaky_relu(&vm->heap.regions, t); break;
            case 467: out = vm_tensor_elu(&vm->heap.regions, t); break;
            case 468: out = vm_tensor_gelu(&vm->heap.regions, t); break;
        }
        if (!out) { vm_push(vm, NIL_VAL); break; }
        out->dtype = t->dtype;
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 469: { /* swish */
        Value t_val = vm_pop(vm);
        VmTensor* t = vm_tensor_operand(vm, t_val, "swish");
        if (!t) { vm_push(vm, NIL_VAL); break; }
        VmTensor* out = vm_tensor_swish(&vm->heap.regions, t);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 470: { /* gpu-elementwise(fn, a, b) */
        Value b = vm_pop(vm), a = vm_pop(vm), fn = vm_pop(vm);
        int native_id = vm_closure_native_id(vm, fn);
        int target = -1;
        switch (native_id) {
            case 142: case 441: target = 441; break;
            case 143: case 442: target = 442; break;
            case 144: case 443: target = 443; break;
            case 145: case 444: target = 444; break;
        }
        if (target < 0) { vm_push(vm, NIL_VAL); break; }
        vm_push(vm, a);
        vm_push(vm, b);
        vm_dispatch_native(vm, target);
        break;
    }
    case 471: { /* gpu-reduce(fn, tensor) */
        Value t = vm_pop(vm), fn = vm_pop(vm);
        int native_id = vm_closure_native_id(vm, fn);
        int target = -1;
        switch (native_id) {
            case 457: target = 457; break;
            case 458: target = 458; break;
            case 459: target = 459; break;
            case 460: target = 460; break;
        }
        if (target < 0) { vm_push(vm, NIL_VAL); break; }
        vm_push(vm, t);
        vm_push(vm, INT_VAL(-1));
        vm_dispatch_native(vm, target);
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Model I/O (800-803)
     * ══════════════════════════════════════════════════════════════════════ */
    case 800: { /* model-save(path, entries) */
        Value entries = vm_pop(vm), path = vm_pop(vm);
        vm_push(vm, BOOL_VAL(vm_model_save_model_file(vm, path, entries)));
        break;
    }
    case 801: { /* model-load(path) */
        vm_model_model_load(vm);
        break;
    }
    case 802: { /* tensor-save(path, tensor) */
        Value tensor = vm_pop(vm), path = vm_pop(vm);
        vm_push(vm, BOOL_VAL(vm_model_save_tensor_file(vm, path, tensor)));
        break;
    }
    case 803: { /* tensor-load(path) */
        vm_model_tensor_load(vm);
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Logic Operations (500-512)
     * ══════════════════════════════════════════════════════════════════════ */
    case 500: { /* make-logic-var(name) — name as int (id) */
        Value name_val = vm_pop(vm), dummy = vm_pop(vm); (void)dummy; (void)name_val;
        /* For simplicity, use the integer as var ID */
        vm_push(vm, INT_VAL((int64_t)vm_make_logic_var("?auto")));
        break;
    }
    case 501: { /* logic-var? — check heap type for LOGIC_VAR */
        Value v = vm_pop(vm);
        int is_lv = (is_heap_type(vm, v, HEAP_LOGIC_VAR));
        vm_push(vm, BOOL_VAL(is_lv));
        break;
    }
    case 502: { /* unify(subst, a, b) — bind a to b in substitution (copy-on-extend) */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm), s_val = vm_pop(vm);
        if (is_heap_type(vm, s_val, HEAP_SUBST)) {
            VmSubstitution* subst = (VmSubstitution*)vm->heap.objects[s_val.as.ptr]->opaque.ptr;
            if (subst && a_val.type == VAL_INT) {
                /* Treat a_val as logic var ID, bind it to b_val */
                VmValue term;
                if (b_val.type == VAL_INT) { term.type = VM_VAL_INT64; term.data.int_val = b_val.as.i; }
                else if (b_val.type == VAL_FLOAT) { term.type = VM_VAL_DOUBLE; term.data.double_val = b_val.as.f; }
                else if (b_val.type == VAL_BOOL) { term.type = VM_VAL_BOOL; term.data.int_val = b_val.as.b; }
                else { term.type = VM_VAL_INT64; term.data.int_val = b_val.as.i; }
                VmSubstitution* extended = vm_subst_extend(&vm->heap.regions, subst, (uint64_t)a_val.as.i, &term);
                if (extended) {
                    VM_PUSH_HEAP_OPAQUE(vm, HEAP_SUBST, VAL_SUBST, extended);
                    break;
                }
            }
        }
        vm_push(vm, NIL_VAL); /* unification failed */
        break;
    }
    case 505: { /* make-substitution */
        VmSubstitution* s = vm_make_substitution(&vm->heap.regions, 16);
        if (!s) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_SUBST, VAL_SUBST, s);
        break;
    }
    case 506: { /* substitution? */
        Value v = vm_pop(vm);
        int is_subst = (is_heap_type(vm, v, HEAP_SUBST));
        vm_push(vm, BOOL_VAL(is_subst));
        break;
    }
    case 509: { /* make-kb */
        VmKnowledgeBase* kb = vm_make_kb(&vm->heap.regions);
        if (!kb) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_KB, VAL_KB, kb);
        break;
    }
    case 510: { /* kb? */
        Value v = vm_pop(vm);
        int is_kb = (is_heap_type(vm, v, HEAP_KB));
        vm_push(vm, BOOL_VAL(is_kb));
        break;
    }
    case 503: { /* walk(term, subst) */
        Value subst_val = vm_pop(vm), term_val = vm_pop(vm);
        /* Walk resolves variables through substitution chains */
        if (is_heap_type(vm, subst_val, HEAP_SUBST)) {
            VmSubstitution* s = (VmSubstitution*)vm->heap.objects[subst_val.as.ptr]->opaque.ptr;
            if (s && term_val.type == VAL_INT) {
                /* Check if term is a var_id that has a binding */
                for (int i = 0; i < s->n_bindings; i++) {
                    if (s->var_ids[i] == (uint64_t)term_val.as.i) {
                        /* Convert VmValue to VM Value */
                        VmValue bound = s->terms[i];
                        if (bound.type == VM_VAL_INT64)
                            vm_push(vm, INT_VAL(bound.data.int_val));
                        else if (bound.type == VM_VAL_DOUBLE)
                            vm_push(vm, FLOAT_VAL(bound.data.double_val));
                        else if (bound.type == VM_VAL_BOOL)
                            vm_push(vm, BOOL_VAL((int)bound.data.int_val));
                        else
                            vm_push(vm, INT_VAL(bound.data.int_val));
                        goto walk_done;
                    }
                }
            }
        }
        vm_push(vm, term_val); /* unbound — return as-is */
        walk_done: break;
    }
    case 504: { /* walk-deep — recursive walk through substitution chains */
        Value subst_val = vm_pop(vm), term_val = vm_pop(vm);
        /* Walk repeatedly until term no longer resolves */
        if (is_heap_type(vm, subst_val, HEAP_SUBST)) {
            VmSubstitution* s = (VmSubstitution*)vm->heap.objects[subst_val.as.ptr]->opaque.ptr;
            Value resolved = term_val;
            int depth = 0;
            while (s && resolved.type == VAL_INT && depth < 100) {
                int found = 0;
                for (int i = 0; i < s->n_bindings; i++) {
                    if (s->var_ids[i] == (uint64_t)resolved.as.i) {
                        VmValue bound = s->terms[i];
                        if (bound.type == VM_VAL_INT64) resolved = INT_VAL(bound.data.int_val);
                        else if (bound.type == VM_VAL_DOUBLE) resolved = FLOAT_VAL(bound.data.double_val);
                        else if (bound.type == VM_VAL_BOOL) resolved = BOOL_VAL((int)bound.data.int_val);
                        else resolved = INT_VAL(bound.data.int_val);
                        found = 1;
                        break;
                    }
                }
                if (!found) break;
                depth++;
            }
            vm_push(vm, resolved);
        } else {
            vm_push(vm, term_val);
        }
        break;
    }
    case 507: { /* make-fact(datum) — store a list/value as a fact */
        Value datum = vm_pop(vm);
        int32_t ptr = heap_alloc(&vm->heap);
        if (ptr < 0) { vm->error = 1; break; }
        vm->heap.objects[ptr]->type = HEAP_FACT;
        vm->heap.objects[ptr]->cons.car = datum;
        vm->heap.objects[ptr]->cons.cdr = NIL_VAL;
        vm_push(vm, (Value){.type = VAL_PAIR, .as.ptr = ptr});
        break;
    }
    case 508: { /* fact? */
        Value v = vm_pop(vm);
        int is_fact = (is_heap_type(vm, v, HEAP_FACT));
        vm_push(vm, BOOL_VAL(is_fact));
        break;
    }
    case 511: { /* kb-assert!(kb, fact) — store fact in the KB */
        Value fact_val = vm_pop(vm), kb_val = vm_pop(vm);
        if (is_heap_type(vm, kb_val, HEAP_KB)) {
            VmKnowledgeBase* kb_obj = (VmKnowledgeBase*)vm->heap.objects[kb_val.as.ptr]->opaque.ptr;
            if (kb_obj && kb_obj->n_facts < kb_obj->capacity) {
                VmFact* f = (VmFact*)vm_alloc(&vm->heap.regions, sizeof(VmFact));
                if (f) {
                    memset(f, 0, sizeof(VmFact));
                    Value datum;
                    if (vm_kb_extract_fact_datum(vm, fact_val, &datum) &&
                        datum.type == VAL_PAIR && is_valid_heap_ptr(vm, datum.as.ptr)) {
                        f->has_datum = 1;
                        f->datum_ptr = datum.as.ptr;
                        kb_obj->facts[kb_obj->n_facts++] = f;
                    }
                }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 512: { /* kb-query(kb, pattern) → list of matching facts */
        Value pattern = vm_pop(vm), kb_val = vm_pop(vm);
        Value pattern_datum;
        int has_pattern = vm_kb_extract_fact_datum(vm, pattern, &pattern_datum);
        if (!has_pattern) pattern_datum = pattern;
        if (is_heap_type(vm, kb_val, HEAP_KB)) {
            VmKnowledgeBase* kb_obj = (VmKnowledgeBase*)vm->heap.objects[kb_val.as.ptr]->opaque.ptr;
            if (kb_obj) {
                Value result = NIL_VAL;
                for (int i = kb_obj->n_facts - 1; i >= 0; i--) {
                    VmFact* f = kb_obj->facts[i];
                    Value fact_datum;
                    if (!vm_kb_stored_fact_datum(vm, f, &fact_datum)) continue;
                    if (vm_kb_datums_match(vm, pattern_datum, fact_datum)) {
                        int32_t p = heap_alloc(&vm->heap);
                        if (p < 0) break;
                        vm->heap.objects[p]->type = HEAP_CONS;
                        vm->heap.objects[p]->cons.car = fact_datum;
                        vm->heap.objects[p]->cons.cdr = result;
                        result = PAIR_VAL(p);
                    }
                }
                vm_push(vm, result);
                break;
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Inference Operations (520-526)
     * ══════════════════════════════════════════════════════════════════════ */
    case 520: { /* make-factor-graph(num_vars, var_dims) */
        Value dims_val = vm_pop(vm), nvars_val = vm_pop(vm);
        int nv = (int)as_number(nvars_val);
        int var_dims[64]; int nd = 0;
        /* var_dims may be a `#(2 2 2)` tensor (canonical Eshkol API,
         * matching the native LLVM runtime), a `(2 2 2)` list, or a
         * bare scalar.  The VM previously only read the list/scalar
         * forms, so a tensor collapsed to nd=1 and num_vars was clamped
         * down to 1 — every multi-variable factor graph silently became
         * single-variable, leaving inference and free-energy degenerate. */
        if (dims_val.as.ptr >= 0 && dims_val.type != VAL_PAIR &&
            vm->heap.objects[dims_val.as.ptr] &&
            vm->heap.objects[dims_val.as.ptr]->type == HEAP_TENSOR) {
            VmTensor* dt = (VmTensor*)vm->heap.objects[dims_val.as.ptr]->opaque.ptr;
            if (dt && dt->data) {
                int dn = (int)dt->total;
                for (int i = 0; i < dn && nd < 64; i++)
                    var_dims[nd++] = (int)dt->data[i];
            }
        } else if (dims_val.type == VAL_PAIR) {
            Value cur = dims_val;
            while (cur.type == VAL_PAIR && nd < 64) {
                var_dims[nd++] = (int)as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
            }
        } else { var_dims[0] = (int)as_number(dims_val); nd = 1; }
        if (nd < nv) nv = nd;
        VmFactorGraph* fg = vm_make_factor_graph(&vm->heap.regions, nv, var_dims);
        if (!fg) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_FACTOR_GRAPH, VAL_FACTOR_GRAPH, fg);
        break;
    }
    case 521: { /* factor-graph? */
        Value v = vm_pop(vm);
        vm_push(vm, BOOL_VAL(is_heap_type(vm, v, HEAP_FACTOR_GRAPH)));
        break;
    }
    case 522: { /* fg-add-factor!(fg, var_indices, cpt) */
        Value cpt = vm_pop(vm), vars = vm_pop(vm), fg_val = vm_pop(vm);
        if (is_heap_type(vm, fg_val, HEAP_FACTOR_GRAPH)) {
            VmFactorGraph* fg = (VmFactorGraph*)vm->heap.objects[fg_val.as.ptr]->opaque.ptr;
            if (fg) {
                /* Extract var_indices from either a tensor `#(0 1)` or a
                 * list `(0 1)`.  The canonical Eshkol API (and the native
                 * LLVM runtime) passes var indices as a `#(...)` tensor;
                 * the VM previously only read the list form, so factors
                 * added with tensor indices were silently dropped
                 * (n_vars stayed 0), leaving beliefs uniform and
                 * free-energy at ~0. */
                int var_idx[8], n_vars = 0;
                if (vars.as.ptr >= 0 && vars.type != VAL_PAIR &&
                    vm->heap.objects[vars.as.ptr] &&
                    vm->heap.objects[vars.as.ptr]->type == HEAP_TENSOR) {
                    VmTensor* vt = (VmTensor*)vm->heap.objects[vars.as.ptr]->opaque.ptr;
                    if (vt && vt->data) {
                        int vn = (int)vt->total;
                        for (int i = 0; i < vn && n_vars < 8; i++)
                            var_idx[n_vars++] = (int)vt->data[i];
                    }
                } else {
                    Value cur = vars;
                    while (cur.type == VAL_PAIR && n_vars < 8) {
                        var_idx[n_vars++] = (int)as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                        cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
                    }
                }
                /* Extract CPT data from tensor or list */
                double* cpt_data = NULL;
                if (is_heap_type(vm, cpt, HEAP_TENSOR)) {
                    VmTensor* t = (VmTensor*)vm->heap.objects[cpt.as.ptr]->opaque.ptr;
                    if (t) cpt_data = t->data;
                }
                if (n_vars > 0 && cpt_data) {
                    /* Build dims array from factor graph's var_dims */
                    int dims[8];
                    for (int i = 0; i < n_vars; i++) {
                        int vi = var_idx[i];
                        dims[i] = (vi >= 0 && vi < fg->num_vars) ? fg->var_dims[vi] : 2;
                    }
                    VmFactor* factor = vm_make_factor(&vm->heap.regions, var_idx, n_vars, cpt_data, dims);
                    if (factor) vm_fg_add_factor(&vm->heap.regions, fg, factor);
                }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 523: { /* fg-infer!(fg, max_iters, tolerance) */
        Value tol = vm_pop(vm), iters = vm_pop(vm), fg_val = vm_pop(vm);
        if (is_heap_type(vm, fg_val, HEAP_FACTOR_GRAPH)) {
            VmFactorGraph* fg = (VmFactorGraph*)vm->heap.objects[fg_val.as.ptr]->opaque.ptr;
            if (fg) {
                int converged = vm_fg_infer(&vm->heap.regions, fg, (int)as_number(iters), as_number(tol));
                vm_push(vm, BOOL_VAL(converged));
                break;
            }
        }
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 524: { /* fg-update-cpt!(fg, factor_idx, new_cpt_tensor) */
        Value cpt = vm_pop(vm), idx = vm_pop(vm), fg_val = vm_pop(vm);
        if (is_heap_type(vm, fg_val, HEAP_FACTOR_GRAPH)) {
            VmFactorGraph* fg = (VmFactorGraph*)vm->heap.objects[fg_val.as.ptr]->opaque.ptr;
            if (fg && is_heap_type(vm, cpt, HEAP_TENSOR)) {
                VmTensor* t = (VmTensor*)vm->heap.objects[cpt.as.ptr]->opaque.ptr;
                if (t) vm_fg_update_cpt(fg, (int)as_number(idx), t->data, (int)t->total);
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 525: { /* free-energy(fg, observations) */
        Value obs = vm_pop(vm), fg_val = vm_pop(vm);
        if (is_heap_type(vm, fg_val, HEAP_FACTOR_GRAPH)) {
            VmFactorGraph* fg = (VmFactorGraph*)vm->heap.objects[fg_val.as.ptr]->opaque.ptr;
            if (fg) {
                /* Parse observations.  Canonical Eshkol form is a flat
                 * `#(var state)` tensor (one observation) or an even-length
                 * `#(v0 s0 v1 s1 ...)` tensor (several), matching the native
                 * LLVM runtime; `#()` means no evidence.  A list of
                 * (var . state) pairs is still accepted for back-compat. */
                int obs_pairs[32][2], n_obs = 0;
                if (obs.as.ptr >= 0 && obs.type != VAL_PAIR &&
                    vm->heap.objects[obs.as.ptr] &&
                    vm->heap.objects[obs.as.ptr]->type == HEAP_TENSOR) {
                    VmTensor* ot = (VmTensor*)vm->heap.objects[obs.as.ptr]->opaque.ptr;
                    if (ot && ot->data) {
                        int total = (int)ot->total;
                        for (int i = 0; i + 1 < total && n_obs < 32; i += 2) {
                            obs_pairs[n_obs][0] = (int)ot->data[i];
                            obs_pairs[n_obs][1] = (int)ot->data[i + 1];
                            n_obs++;
                        }
                    }
                } else {
                    Value cur = obs;
                    while (cur.type == VAL_PAIR && n_obs < 32) {
                        Value pair = vm->heap.objects[cur.as.ptr]->cons.car;
                        if (pair.type == VAL_PAIR) {
                            obs_pairs[n_obs][0] = (int)as_number(vm->heap.objects[pair.as.ptr]->cons.car);
                            obs_pairs[n_obs][1] = (int)as_number(vm->heap.objects[vm->heap.objects[pair.as.ptr]->cons.cdr.as.ptr]->cons.car);
                            n_obs++;
                        }
                        cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
                    }
                }
                double fe = vm_free_energy(fg, (const double*)obs_pairs, n_obs);
                vm_push(vm, FLOAT_VAL(fe));
                break;
            }
        }
        vm_push(vm, FLOAT_VAL(0.0));
        break;
    }
    case 526: { /* expected-free-energy(fg, action_var, action_state) */
        Value state = vm_pop(vm), var = vm_pop(vm), fg_val = vm_pop(vm);
        if (is_heap_type(vm, fg_val, HEAP_FACTOR_GRAPH)) {
            VmFactorGraph* fg = (VmFactorGraph*)vm->heap.objects[fg_val.as.ptr]->opaque.ptr;
            if (fg) {
                double efe = vm_expected_free_energy(&vm->heap.regions, fg, (int)as_number(var), (int)as_number(state));
                vm_push(vm, FLOAT_VAL(efe));
                break;
            }
        }
        vm_push(vm, FLOAT_VAL(0.0));
        break;
    }

    case 527: { /* fg-observe!(fg, var_id, observed_state)
                 * Clamps a variable to an observed state for evidence injection.
                 * After observing, re-run fg-infer! to propagate the evidence.
                 * Ref: Standard factor graph evidence clamping (Kschischang et al. 2001). */
        Value state_val = vm_pop(vm), var_val = vm_pop(vm), fg_val = vm_pop(vm);
        if (is_heap_type(vm, fg_val, HEAP_FACTOR_GRAPH)) {
            VmFactorGraph* fg = (VmFactorGraph*)vm->heap.objects[fg_val.as.ptr]->opaque.ptr;
            if (fg) {
                int var_id = (int)as_number(var_val);
                int obs_state = (int)as_number(state_val);
                if (var_id >= 0 && var_id < fg->num_vars &&
                    obs_state >= 0 && obs_state < fg->var_dims[var_id]) {
                    /* Clamp beliefs: set observed state to probability 1, others to 0 */
                    int dim = fg->var_dims[var_id];
                    for (int s = 0; s < dim; s++) {
                        fg->beliefs[var_id][s] = (s == obs_state) ? 0.0 : -1e30;
                    }
                    /* Mark variable as observed (skip during belief update in fg-infer!).
                     * Allocated via VM arena — lives as long as the factor graph. */
                    if (!fg->observed) {
                        fg->observed = (bool*)vm_alloc(&vm->heap.regions, fg->num_vars * sizeof(bool));
                        if (fg->observed) memset(fg->observed, 0, fg->num_vars * sizeof(bool));
                    }
                    if (fg->observed) {
                        fg->observed[var_id] = true;
                    }
                    vm_push(vm, BOOL_VAL(true));
                    break;
                }
            }
        }
        vm_push(vm, BOOL_VAL(false));
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Workspace Operations (540-547)
     * ══════════════════════════════════════════════════════════════════════ */
    case 540: { /* make-workspace(dim, max_modules) */
        Value max_m = vm_pop(vm), dim_val = vm_pop(vm);
        VmWorkspace* ws = vm_ws_new(&vm->heap.regions, (int)as_number(dim_val), (int)as_number(max_m));
        if (!ws) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_WORKSPACE, VAL_WORKSPACE, ws);
        break;
    }
    case 541: { /* workspace? */
        Value v = vm_pop(vm);
        vm_push(vm, BOOL_VAL(is_heap_type(vm, v, HEAP_WORKSPACE)));
        break;
    }
    case 542: { /* ws-register!(ws, name, closure) */
        Value closure = vm_pop(vm), name_val = vm_pop(vm), ws_val = vm_pop(vm);
        if (is_heap_type(vm, ws_val, HEAP_WORKSPACE)) {
            VmWorkspace* ws = (VmWorkspace*)vm->heap.objects[ws_val.as.ptr]->opaque.ptr;
            if (ws) {
                const char* name = "module";
                if (name_val.type == VAL_STRING && vm->heap.objects[name_val.as.ptr]->opaque.ptr) {
                    VmString* ns = (VmString*)vm->heap.objects[name_val.as.ptr]->opaque.ptr;
                    name = ns->data;
                }
                /* Allocate stable Value on arena so pointer survives past this scope */
                Value* stable_closure = (Value*)vm_alloc(&vm->heap.regions, sizeof(Value));
                if (stable_closure) {
                    *stable_closure = closure;
                    vm_ws_register(ws, name, stable_closure);
                }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 543: { /* ws-step!(ws) — invoke module closures via closure bridge */
        Value ws_val = vm_pop(vm);
        if (is_heap_type(vm, ws_val, HEAP_WORKSPACE)) {
            VmWorkspace* ws = (VmWorkspace*)vm->heap.objects[ws_val.as.ptr]->opaque.ptr;
            if (ws && ws->content) {
                /* Build content tensor */
                int64_t shape[1] = {ws->dim};
                VmTensor* ct = vm_tensor_from_data(&vm->heap.regions, ws->content, shape, 1);
                int32_t tptr = heap_alloc(&vm->heap);
                if (tptr < 0 || !ct) { vm_push(vm, NIL_VAL); break; }
                vm->heap.objects[tptr]->type = HEAP_TENSOR;
                vm->heap.objects[tptr]->opaque.ptr = ct;
                Value content_val = (Value){.type = VAL_TENSOR, .as.ptr = tptr};

                /* Call each module's closure, collect salience + proposal */
                int n_mod = ws->n_modules;
                if (n_mod > 256) n_mod = 256;
                double* saliences = (double*)vm_alloc(&vm->heap.regions, n_mod * sizeof(double));
                Value* proposals = (Value*)vm_alloc(&vm->heap.regions, n_mod * sizeof(Value));
                if (!saliences || !proposals) { vm_push(vm, NIL_VAL); break; }
                memset(saliences, 0, n_mod * sizeof(double));
                memset(proposals, 0, n_mod * sizeof(Value));
                for (int i = 0; i < n_mod; i++) {
                    Value* closure_ptr = (Value*)ws->modules[i].process_fn;
                    if (closure_ptr && closure_ptr->type == VAL_CLOSURE) {
                        Value result = vm_call_closure_from_native(vm, *closure_ptr, &content_val, 1);
                        if (result.type == VAL_PAIR) {
                            saliences[i] = as_number(vm->heap.objects[result.as.ptr]->cons.car);
                            proposals[i] = vm->heap.objects[result.as.ptr]->cons.cdr;
                        } else {
                            saliences[i] = as_number(result);
                            proposals[i] = content_val;
                        }
                    } else {
                        saliences[i] = 0;
                        proposals[i] = content_val;
                    }
                }
                /* Softmax competition: highest salience wins */
                int winner = 0;
                for (int i = 1; i < n_mod; i++)
                    if (saliences[i] > saliences[winner]) winner = i;
                ws->step_count++;
                Value winner_proposal = proposals[winner];
                vm_push(vm, winner_proposal);
                break;
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 544: { /* ws-get-content */
        Value ws_val = vm_pop(vm);
        if (is_heap_type(vm, ws_val, HEAP_WORKSPACE)) {
            VmWorkspace* ws = (VmWorkspace*)vm->heap.objects[ws_val.as.ptr]->opaque.ptr;
            if (ws && ws->content) {
                /* Return content as tensor */
                int64_t shape[1] = {ws->dim};
                VmTensor* t = vm_tensor_from_data(&vm->heap.regions, ws->content, shape, 1);
                if (t) {
                    int32_t ptr = heap_alloc(&vm->heap);
                    if (ptr >= 0) {
                        vm->heap.objects[ptr]->type = HEAP_TENSOR;
                        vm->heap.objects[ptr]->opaque.ptr = t;
                        vm_push(vm, (Value){.type = VAL_TENSOR, .as.ptr = ptr});
                        break;
                    }
                }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 545: { /* ws-set-content!(ws, tensor) */
        Value tensor_val = vm_pop(vm), ws_val = vm_pop(vm);
        if (is_heap_type(vm, ws_val, HEAP_WORKSPACE) &&
            is_heap_type(vm, tensor_val, HEAP_TENSOR)) {
            VmWorkspace* ws = (VmWorkspace*)vm->heap.objects[ws_val.as.ptr]->opaque.ptr;
            VmTensor* t = (VmTensor*)vm->heap.objects[tensor_val.as.ptr]->opaque.ptr;
            if (ws && t && t->data) {
                int copy_dim = (t->total < ws->dim) ? (int)t->total : ws->dim;
                if (copy_dim > 0) memcpy(ws->content, t->data, (size_t)copy_dim * sizeof(double));
                for (int i = copy_dim; i < ws->dim; i++) ws->content[i] = 0.0;
                vm_push(vm, NIL_VAL);
                break;
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 546: { /* ws-get-dim */
        Value ws_val = vm_pop(vm);
        if (is_heap_type(vm, ws_val, HEAP_WORKSPACE)) {
            VmWorkspace* ws = (VmWorkspace*)vm->heap.objects[ws_val.as.ptr]->opaque.ptr;
            vm_push(vm, INT_VAL(ws ? ws->dim : 0));
        } else {
            vm_push(vm, INT_VAL(0));
        }
        break;
    }
    case 547: { /* ws-get-step-count */
        Value ws_val = vm_pop(vm);
        if (is_heap_type(vm, ws_val, HEAP_WORKSPACE)) {
            VmWorkspace* ws = (VmWorkspace*)vm->heap.objects[ws_val.as.ptr]->opaque.ptr;
            vm_push(vm, INT_VAL(ws ? ws->step_count : 0));
        } else {
            vm_push(vm, INT_VAL(0));
        }
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * String Operations (550-570) — real VmString dispatch
     * ══════════════════════════════════════════════════════════════════════ */
    case 550: { /* string-length */
        Value s_val = vm_pop(vm);
        if (s_val.type == VAL_STRING && vm->heap.objects[s_val.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr;
            vm_push(vm, INT_VAL(vm_string_length(s)));
        } else vm_push(vm, INT_VAL(0));
        break;
    }
    case 551: { /* string-ref(str, idx) */
        Value idx = vm_pop(vm), s_val = vm_pop(vm);
        if (s_val.type == VAL_STRING && vm->heap.objects[s_val.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr;
            int cp = vm_string_ref(s, (int)as_number(idx));
            vm_push(vm, INT_VAL(cp >= 0 ? cp : 0));
        } else vm_push(vm, INT_VAL(0));
        break;
    }
    case 552: { /* string-set!(str, idx, char) → new string */
        Value ch = vm_pop(vm), idx = vm_pop(vm), s_val = vm_pop(vm);
        if (s_val.type == VAL_STRING && vm->heap.objects[s_val.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr;
            VmString* result = vm_string_set(&vm->heap.regions, s, (int)as_number(idx), (int)as_number(ch));
            if (result) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, result); }
            else vm_push(vm, NIL_VAL);
        } else vm_push(vm, NIL_VAL);
        break;
    }
    case 553: { /* substring(str, start, end) */
        Value end = vm_pop(vm), start = vm_pop(vm), s_val = vm_pop(vm);
        if (s_val.type == VAL_STRING && vm->heap.objects[s_val.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr;
            VmString* result = vm_string_substring(&vm->heap.regions, s, (int)as_number(start), (int)as_number(end));
            if (result) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, result); }
            else vm_push(vm, NIL_VAL);
        } else vm_push(vm, NIL_VAL);
        break;
    }
    case 554: { /* string-append */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        VmString* a = (a_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[a_val.as.ptr]->opaque.ptr : NULL;
        VmString* b = (b_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[b_val.as.ptr]->opaque.ptr : NULL;
        if (a && b) {
            VmString* result = vm_string_append(&vm->heap.regions, a, b);
            if (result) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, result); break; }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 555: { /* string-contains(str, substr) */
        Value sub_val = vm_pop(vm), s_val = vm_pop(vm);
        VmString* s = (s_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr : NULL;
        VmString* sub = (sub_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[sub_val.as.ptr]->opaque.ptr : NULL;
        vm_push(vm, INT_VAL(vm_string_contains(s, sub)));
        break;
    }
    case 556: { /* make-string(n, char) */
        Value ch = vm_pop(vm), n = vm_pop(vm);
        VmString* result = vm_string_make(&vm->heap.regions, (int)as_number(n), (int)as_number(ch));
        if (result) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, result); }
        else vm_push(vm, NIL_VAL);
        break;
    }
    case 557: { /* string-upcase */
        Value s_val = vm_pop(vm);
        VmString* s = (s_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr : NULL;
        if (s) { VmString* r = vm_string_upcase(&vm->heap.regions, s); if (r) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, r); break; } }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 558: { /* string-downcase */
        Value s_val = vm_pop(vm);
        VmString* s = (s_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr : NULL;
        if (s) { VmString* r = vm_string_downcase(&vm->heap.regions, s); if (r) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, r); break; } }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 559: { /* string-contains (duplicate of 555 for compat) */
        Value sub_val = vm_pop(vm), s_val = vm_pop(vm);
        VmString* s = (s_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr : NULL;
        VmString* sub = (sub_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[sub_val.as.ptr]->opaque.ptr : NULL;
        vm_push(vm, INT_VAL(vm_string_contains(s, sub)));
        break;
    }
    case 560: { /* string=? */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        VmString* a = (a_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[a_val.as.ptr]->opaque.ptr : NULL;
        VmString* b = (b_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[b_val.as.ptr]->opaque.ptr : NULL;
        vm_push(vm, BOOL_VAL(vm_string_eq(a, b)));
        break;
    }
    case 561: { /* string<? */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        VmString* a = (a_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[a_val.as.ptr]->opaque.ptr : NULL;
        VmString* b = (b_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[b_val.as.ptr]->opaque.ptr : NULL;
        vm_push(vm, BOOL_VAL(vm_string_lt(a, b)));
        break;
    }
    case 562: { /* string-ci=? */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        VmString* a = (a_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[a_val.as.ptr]->opaque.ptr : NULL;
        VmString* b = (b_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[b_val.as.ptr]->opaque.ptr : NULL;
        vm_push(vm, BOOL_VAL(vm_string_ci_eq(a, b)));
        break;
    }
    case 563: case 564: { /* string->number / number->string */
        if (fid == 563) { /* string->number */
            Value s_val = vm_pop(vm);
            VmString* s = (s_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr : NULL;
            double d = vm_string_to_number(s);
            if (isnan(d)) vm_push(vm, BOOL_VAL(0)); /* #f on parse failure */
            else vm_push(vm, number_val(d));
        } else { /* number->string */
            Value n = vm_pop(vm);
            VmString* r = vm_number_to_string(&vm->heap.regions, as_number(n));
            if (r) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, r); }
            else vm_push(vm, NIL_VAL);
        }
        break;
    }
    case 565: { /* string->list — convert string to list of character codepoints */
        Value s_val = vm_pop(vm);
        if (s_val.type == VAL_STRING && vm->heap.objects[s_val.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr;
            int len = vm_string_length(s);
            Value result = NIL_VAL;
            for (int i = len - 1; i >= 0; i--) {
                int cp = vm_string_ref(s, i);
                int32_t p = heap_alloc(&vm->heap);
                if (p < 0) break;
                vm->heap.objects[p]->type = HEAP_CONS;
                vm->heap.objects[p]->cons.car = (Value){.type = VAL_CHAR, .as.i = cp >= 0 ? cp : 0};
                vm->heap.objects[p]->cons.cdr = result;
                result = PAIR_VAL(p);
            }
            vm_push(vm, result);
        } else {
            vm_push(vm, NIL_VAL);
        }
        break;
    }
    case 566: { /* string-copy */
        Value s_val = vm_pop(vm);
        VmString* s = (s_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr : NULL;
        if (s) { VmString* r = vm_string_copy(&vm->heap.regions, s); if (r) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, r); break; } }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 567: { /* list->string — convert list of character codepoints to string */
        Value lst = vm_pop(vm);
        /* Count characters */
        int len = 0;
        Value cur = lst;
        while (cur.type == VAL_PAIR && len < 4096) {
            len++;
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
        }
        char* buf = (char*)vm_alloc(&vm->heap.regions, (size_t)(len + 1));
        if (buf) {
            cur = lst;
            int idx = 0;
            while (cur.type == VAL_PAIR && idx < len) {
                int cp = (int)as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                buf[idx++] = (cp >= 0 && cp < 128) ? (char)cp : '?';
                cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
            }
            buf[idx] = '\0';
            VmString* s = vm_string_new(&vm->heap.regions, buf, idx);
            if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 568: { /* string->number (alt ID) */
        Value s_val = vm_pop(vm);
        VmString* s = (s_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr : NULL;
        double d = vm_string_to_number(s);
        if (isnan(d)) vm_push(vm, BOOL_VAL(0));
        else vm_push(vm, number_val(d));
        break;
    }
    case 569: { /* number->string (alt ID) */
        Value n = vm_pop(vm);
        VmString* r = vm_number_to_string(&vm->heap.regions, as_number(n));
        if (r) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, r); }
        else vm_push(vm, NIL_VAL);
        break;
    }
    case 570: { /* string-hash */
        Value s_val = vm_pop(vm);
        VmString* s = (s_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr : NULL;
        vm_push(vm, INT_VAL(s ? (int64_t)vm_string_hash(s) : 0));
        break;
    }
    case 571: { /* string-byte-length */
        Value s_val = vm_pop(vm);
        VmString* s = (s_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr : NULL;
        vm_push(vm, INT_VAL(s ? (int64_t)s->byte_len : 0));
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * I/O Operations (580-602) — port-based I/O
     * ══════════════════════════════════════════════════════════════════════ */
    case 580: { /* open-input-file(path) */
        Value path = vm_pop(vm);
        if (path.type == VAL_STRING && vm->heap.objects[path.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[path.as.ptr]->opaque.ptr;
            VmPort* p = vm_port_open_input_file(&vm->heap.regions, s->data);
            if (p) {
                int32_t ptr = heap_alloc(&vm->heap);
                if (ptr >= 0) {
                    vm->heap.objects[ptr]->type = HEAP_PORT;
                    vm->heap.objects[ptr]->opaque.ptr = p;
                    vm_push(vm, (Value){.type = VAL_PORT, .as.ptr = ptr});
                    break;
                }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 581: { /* open-output-file(path) */
        Value path = vm_pop(vm);
        if (path.type == VAL_STRING && vm->heap.objects[path.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[path.as.ptr]->opaque.ptr;
            VmPort* p = vm_port_open_output_file(&vm->heap.regions, s->data);
            if (p) {
                int32_t ptr = heap_alloc(&vm->heap);
                if (ptr >= 0) {
                    vm->heap.objects[ptr]->type = HEAP_PORT;
                    vm->heap.objects[ptr]->opaque.ptr = p;
                    vm_push(vm, (Value){.type = VAL_PORT, .as.ptr = ptr});
                    break;
                }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 582: { /* close-port(port) */
        Value port_val = vm_pop(vm);
        VmPort* port = vm_value_as_port(vm, port_val);
        if (port) {
            vm_port_close(port);
        }
        vm_push(vm, (Value){.type = VAL_VOID});
        break;
    }
    case 583: { /* read-char(port) */
        Value port_val = vm_pop(vm);
        VmPort* port = vm_value_as_port(vm, port_val);
        if (!port) port = vm_port_current_input();
        int ch = vm_port_read_char(port);
        vm_push(vm, ch == EOF ? NIL_VAL : INT_VAL(ch));
        break;
    }
    case 584: { /* write-char(char, port) */
        Value port = vm_pop(vm), ch = vm_pop(vm); (void)port;
        putchar((int)as_number(ch));
        vm_push(vm, NIL_VAL);
        break;
    }
    case 585: { /* read-line(port) */
        Value port_val = vm_pop(vm);
        VmPort* port = vm_value_as_port(vm, port_val);
        if (!port) port = vm_port_current_input();
        VmString* line = vm_port_read_line(&vm->heap.regions, port);
        if (line) {
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, line);
        } else {
            vm_push(vm, NIL_VAL);
        }
        break;
    }
    case 586: { /* write-char(char, port) — write to stdout if no port */
        Value ch = vm_pop(vm);
        putchar((int)as_number(ch));
        fflush(stdout);
        vm_push(vm, (Value){.type = VAL_VOID});
        break;
    }
    case 587: { /* write-string(str, port) */
        Value port_val = vm_pop(vm);
        Value str_val = vm_pop(vm);
        VmString* str = (str_val.type == VAL_STRING &&
                         str_val.as.ptr >= 0 && str_val.as.ptr < vm->heap.next_free &&
                         vm->heap.objects[str_val.as.ptr]->type == HEAP_STRING)
            ? (VmString*)vm->heap.objects[str_val.as.ptr]->opaque.ptr
            : NULL;
        VmPort* port = vm_value_as_port(vm, port_val);
        if (!port) port = vm_port_current_output();
        vm_port_write_string(port, str);
        vm_push(vm, (Value){.type = VAL_VOID});
        break;
    }
    case 588: { /* _read0 — complete datum reader over current input */
        vm_push(vm, vm_reader_from_port(vm, vm_port_current_input()));
        break;
    }
    case 619: { /* _read1(port) — complete datum reader over explicit port */
        Value port_value = vm_pop(vm);
        VmPort* port = vm_value_as_port(vm, port_value);
        if (!port || port->dir != VM_PORT_INPUT) {
            fprintf(stderr, "ERROR: read requires an open input port\n");
            vm->error = 1;
            vm_push(vm, NIL_VAL);
            break;
        }
        vm_push(vm, vm_reader_from_port(vm, port));
        break;
    }
    case 589: { /* write(datum) */
        Value v = vm_pop(vm);
        vm_write_value_port(vm, v, vm_port_current_output(), 1);
        vm_push(vm, NIL_VAL);
        break;
    }
    case 590: { /* display(datum) — print without quoting (same as write in this VM) */
        Value v = vm_pop(vm);
        print_value(vm, v);
        vm_push(vm, NIL_VAL);
        break;
    }
    case 591: { /* newline */
        printf("\n"); vm_push(vm, NIL_VAL); break;
    }
    case 592: { /* eof-object? */
        Value v = vm_pop(vm);
        vm_push(vm, BOOL_VAL(v.type == VAL_EOF));
        break;
    }
    case 593: { /* current-input-port */
        vm_push(vm, NIL_VAL); break;
    }
    case 594: { /* current-output-port */
        vm_push(vm, NIL_VAL); break;
    }
    case 595: { /* current-error-port → just return a sentinel */
        vm_push(vm, INT_VAL(-3)); /* stderr sentinel */
        break;
    }
    case 596: { /* open-input-string */
        Value str = vm_pop(vm);
        VmString* src = (str.type == VAL_STRING && vm->heap.objects[str.as.ptr]->opaque.ptr)
            ? (VmString*)vm->heap.objects[str.as.ptr]->opaque.ptr : NULL;
        VmPort* p = vm_port_open_input_string(&vm->heap.regions, src);
        if (p) {
            int32_t ptr = heap_alloc(&vm->heap);
            if (ptr >= 0) {
                vm->heap.objects[ptr]->type = HEAP_PORT;
                vm->heap.objects[ptr]->opaque.ptr = p;
                vm_push(vm, (Value){.type = VAL_PORT, .as.ptr = ptr});
                break;
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 597: { /* open-output-string */
        VmPort* p = vm_port_open_output_string(&vm->heap.regions);
        if (p) {
            int32_t ptr = heap_alloc(&vm->heap);
            if (ptr >= 0) {
                vm->heap.objects[ptr]->type = HEAP_PORT;
                vm->heap.objects[ptr]->opaque.ptr = p;
                vm_push(vm, (Value){.type = VAL_PORT, .as.ptr = ptr});
                break;
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 598: { /* get-output-string */
        Value port_val = vm_pop(vm);
        VmPort* p = vm_value_as_port(vm, port_val);
        if (p) {
            VmString* s = vm_port_get_output_string(&vm->heap.regions, p);
            if (s) {
                int32_t ptr = heap_alloc(&vm->heap);
                if (ptr >= 0) {
                    vm->heap.objects[ptr]->type = HEAP_STRING;
                    vm->heap.objects[ptr]->opaque.ptr = s;
                    vm_push(vm, (Value){.type = VAL_STRING, .as.ptr = ptr});
                    break;
                }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 599: { /* file-exists? */
        Value path = vm_pop(vm);
        int exists = 0;
        if (path.type == VAL_STRING && vm->heap.objects[path.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[path.as.ptr]->opaque.ptr;
            exists = vm_port_file_exists(s->data);
        }
        vm_push(vm, BOOL_VAL(exists));
        break;
    }
    case 600: { /* delete-file */
        Value path = vm_pop(vm);
        if (path.type == VAL_STRING && vm->heap.objects[path.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[path.as.ptr]->opaque.ptr;
            vm_port_delete_file(s->data);
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 601: { /* directory-entries(path) → list of filename strings */
        Value path_val = vm_pop(vm);
#ifdef ESHKOL_VM_WASM
        vm_push(vm, NIL_VAL); break;
#else
        if (path_val.type != VAL_STRING) { vm_push(vm, NIL_VAL); break; }
        VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
        if (!ps) { vm_push(vm, NIL_VAL); break; }
        DIR* dir = opendir(ps->data);
        if (!dir) { vm_push(vm, NIL_VAL); break; }
        Value result601 = NIL_VAL;
        struct dirent* ent;
        while ((ent = readdir(dir)) != NULL) {
            if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) continue;
            VmString* s = vm_string_from_cstr(&vm->heap.regions, ent->d_name);
            if (!s) continue;
            int32_t sp601 = heap_alloc(&vm->heap); if (sp601 < 0) continue;
            vm->heap.objects[sp601]->type = HEAP_STRING;
            vm->heap.objects[sp601]->opaque.ptr = s;
            int32_t cp601 = heap_alloc(&vm->heap); if (cp601 < 0) continue;
            vm->heap.objects[cp601]->type = HEAP_CONS;
            vm->heap.objects[cp601]->cons.car = (Value){.type = VAL_STRING, .as.ptr = sp601};
            vm->heap.objects[cp601]->cons.cdr = result601;
            result601 = PAIR_VAL(cp601);
        }
        closedir(dir);
        /* reverse to get alphabetical order */
        Value rev601 = NIL_VAL;
        while (result601.type == VAL_PAIR) {
            Value car601 = vm->heap.objects[result601.as.ptr]->cons.car;
            int32_t rp601 = heap_alloc(&vm->heap); if (rp601 < 0) break;
            vm->heap.objects[rp601]->type = HEAP_CONS;
            vm->heap.objects[rp601]->cons.car = car601;
            vm->heap.objects[rp601]->cons.cdr = rev601;
            rev601 = PAIR_VAL(rp601);
            result601 = vm->heap.objects[result601.as.ptr]->cons.cdr;
        }
        vm_push(vm, rev601);
        break;
#endif
    }
    case 602: { /* command-line → list of argv strings */
        Value result602 = NIL_VAL;
        for (int i = g_vm_argc - 1; i >= 0; i--) {
            const char* arg = (g_vm_argv && g_vm_argv[i]) ? g_vm_argv[i] : "";
            VmString* s = vm_string_from_cstr(&vm->heap.regions, arg);
            if (!s) continue;
            int32_t sp602 = heap_alloc(&vm->heap); if (sp602 < 0) continue;
            vm->heap.objects[sp602]->type = HEAP_STRING;
            vm->heap.objects[sp602]->opaque.ptr = s;
            int32_t cp602 = heap_alloc(&vm->heap); if (cp602 < 0) continue;
            vm->heap.objects[cp602]->type = HEAP_CONS;
            vm->heap.objects[cp602]->cons.car = (Value){.type = VAL_STRING, .as.ptr = sp602};
            vm->heap.objects[cp602]->cons.cdr = result602;
            result602 = PAIR_VAL(cp602);
        }
        vm_push(vm, result602);
        break;
    }
    case 603: { /* term-cursor-pos → (row . col), or (0 . 0) off-TTY */
        int row = 0;
        int col = 0;
        (void)vm_query_terminal_cursor(&row, &col);
        vm_push(vm, vm_int_pair(vm, row, col));
        break;
    }
    case 1930: { /* term-set-scroll-region(top, bottom) → bool */
        Value bottom_val = vm_pop(vm), top_val = vm_pop(vm);
        int top = (int)as_number(top_val);
        int bottom = (int)as_number(bottom_val);
        if (top < 1) top = 1;
        if (bottom >= top) {
            vm_term_printf_tty("\033[%d;%dr", top, bottom);
            vm_push(vm, BOOL_VAL(1));
        } else {
            vm_push(vm, BOOL_VAL(0));
        }
        break;
    }
    case 1931: { /* term-reset-scroll-region() → bool */
        vm_term_write_tty("\033[r");
        vm_push(vm, BOOL_VAL(1));
        break;
    }
    case 1932: { /* term-enable-mouse() → bool */
        vm_term_write_tty("\033[?1000h\033[?1006h");
        vm_push(vm, BOOL_VAL(1));
        break;
    }
    case 1933: { /* term-disable-mouse() → bool */
        vm_term_write_tty("\033[?1006l\033[?1000l");
        vm_push(vm, BOOL_VAL(1));
        break;
    }
    case 1934: { /* term-read-mouse-event(timeout-ms) → (button x y modifiers type) or #f */
        Value timeout_val = vm_pop(vm);
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
        int timeout_ms = (int)as_number(timeout_val);
        if (timeout_ms < 0) timeout_ms = 0;
        if (vm_term_stdin_is_tty()) {
            struct pollfd pfd;
            pfd.fd = STDIN_FILENO;
            pfd.events = POLLIN;
            pfd.revents = 0;
            if (poll(&pfd, 1, timeout_ms) > 0 && (pfd.revents & POLLIN)) {
                char buf[64];
                int old_flags = fcntl(STDIN_FILENO, F_GETFL, 0);
                if (old_flags >= 0 && fcntl(STDIN_FILENO, F_SETFL, old_flags | O_NONBLOCK) == 0) {
                    ssize_t n = read(STDIN_FILENO, buf, sizeof(buf) - 1);
                    (void)fcntl(STDIN_FILENO, F_SETFL, old_flags);
                    if (n > 0) {
                        buf[n] = '\0';
                        int b = 0, x = 0, y = 0;
                        char kind = '\0';
                        if (sscanf(buf, "\033[<%d;%d;%d%c", &b, &x, &y, &kind) == 4) {
                            const char* type = (kind == 'm') ? "release" : "press";
                            int button = b & 3;
                            int modifiers = b & (4 | 8 | 16);
                            Value result = NIL_VAL;
                            result = vm_cons_value(vm, vm_string_value(vm, type, -1), result);
                            result = vm_cons_value(vm, INT_VAL(modifiers), result);
                            result = vm_cons_value(vm, INT_VAL(y), result);
                            result = vm_cons_value(vm, INT_VAL(x), result);
                            result = vm_cons_value(vm, INT_VAL(button), result);
                            vm_push(vm, result);
                            break;
                        }
                    }
                }
            }
        }
#else
        (void)timeout_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 1935: { /* term-enable-alternate-screen() → bool */
        vm_term_write_tty("\033[?1049h");
        vm_push(vm, BOOL_VAL(1));
        break;
    }
    case 1936: { /* term-disable-alternate-screen() → bool */
        vm_term_write_tty("\033[?1049l");
        vm_push(vm, BOOL_VAL(1));
        break;
    }
    case 1937: { /* term-clipboard-write(text) → bool */
        Value text_val = vm_pop(vm);
        VmString* text = vm_value_as_string(vm, text_val);
        if (!text || !text->data) {
            vm_push(vm, BOOL_VAL(0));
            break;
        }
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
        if (vm_term_stdout_is_tty() && text->byte_len <= 4096) {
            static const char table[] =
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            char encoded[5500];
            int pos = 0;
            for (int i = 0; i < text->byte_len; i += 3) {
                unsigned int a = (unsigned char)text->data[i];
                unsigned int b = (i + 1 < text->byte_len) ? (unsigned char)text->data[i + 1] : 0;
                unsigned int c = (i + 2 < text->byte_len) ? (unsigned char)text->data[i + 2] : 0;
                encoded[pos++] = table[(a >> 2) & 63];
                encoded[pos++] = table[((a & 3) << 4) | ((b >> 4) & 15)];
                encoded[pos++] = (i + 1 < text->byte_len) ? table[((b & 15) << 2) | ((c >> 6) & 3)] : '=';
                encoded[pos++] = (i + 2 < text->byte_len) ? table[c & 63] : '=';
            }
            encoded[pos] = '\0';
            fputs("\033]52;c;", stdout);
            fputs(encoded, stdout);
            fputs("\a", stdout);
            fflush(stdout);
        }
#endif
        vm_push(vm, BOOL_VAL(1));
        break;
    }
    case 1938: { /* term-clipboard-read() → string or #f */
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 1939: { /* term-hyperlink(url, text) → string */
        Value text_val = vm_pop(vm), url_val = vm_pop(vm);
        VmString* url = vm_value_as_string(vm, url_val);
        VmString* text = vm_value_as_string(vm, text_val);
        if (url && url->data && text && text->data) {
            char buf[8192];
            int n = snprintf(buf, sizeof(buf), "\033]8;;%s\033\\%s\033]8;;\033\\",
                             url->data, text->data);
            if (n >= 0 && n < (int)sizeof(buf)) {
                vm_push(vm, vm_string_value(vm, buf, n));
                break;
            }
        }
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 1940: { /* term-detect-capabilities() → alist */
        int color_depth = 8;
        int unicode = 0;
        const char* colorterm = getenv("COLORTERM");
        const char* term = getenv("TERM");
        const char* lang = getenv("LANG");
        if (colorterm && (strstr(colorterm, "truecolor") || strstr(colorterm, "24bit")))
            color_depth = 24;
        else if (term && strstr(term, "256color"))
            color_depth = 8;
        if ((lang && (strstr(lang, "UTF-8") || strstr(lang, "utf8"))) ||
            (term && strstr(term, "utf")))
            unicode = 1;
        Value result = NIL_VAL;
        result = vm_cons_value(vm, vm_alist_entry(vm, "tty", BOOL_VAL(vm_term_stdout_is_tty())), result);
        result = vm_cons_value(vm, vm_alist_entry(vm, "unicode", BOOL_VAL(unicode)), result);
        result = vm_cons_value(vm, vm_alist_entry(vm, "color-depth", INT_VAL(color_depth)), result);
        vm_push(vm, result);
        break;
    }
    case 1941: { /* term-bell() → bool */
        vm_term_write_tty("\a");
        vm_push(vm, BOOL_VAL(1));
        break;
    }
    case 1942: { /* fs-watch-native(path, callback) → watcher or #f */
        Value callback_val = vm_pop(vm), path_val = vm_pop(vm);
        (void)callback_val;
        VmString* path = vm_value_as_string(vm, path_val);
        int handle = path ? vm_file_watch_start(vm, path->data, 0) : -1;
        if (handle > 0) vm_push(vm, INT_VAL((int64_t)handle));
        else vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 1943: { /* fs-watch-recursive(path, callback) → watcher or #f */
        Value callback_val = vm_pop(vm), path_val = vm_pop(vm);
        (void)callback_val;
        VmString* path = vm_value_as_string(vm, path_val);
        int handle = path ? vm_file_watch_start(vm, path->data, 1) : -1;
        if (handle > 0) vm_push(vm, INT_VAL((int64_t)handle));
        else vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 1944: { /* fs-watch-poll(watcher) → "event\tpath" or #f */
        Value handle_val = vm_pop(vm);
        int handle = (int)as_number(handle_val);
        if (handle > 0 && handle < (int)(sizeof(vm->fs_watchers) / sizeof(vm->fs_watchers[0])) &&
            vm->fs_watchers[handle].active) {
            int exists = 0;
            int64_t mtime_ns = 0;
            int64_t size = 0;
            vm_file_watch_signature(vm->fs_watchers[handle].path, &exists, &mtime_ns, &size);
            const char* event = NULL;
            if (vm->fs_watchers[handle].exists && !exists)
                event = "delete";
            else if (!vm->fs_watchers[handle].exists && exists)
                event = "create";
            else if (exists && (mtime_ns != vm->fs_watchers[handle].mtime_ns ||
                               size != vm->fs_watchers[handle].size))
                event = "change";

            vm->fs_watchers[handle].exists = exists;
            vm->fs_watchers[handle].mtime_ns = mtime_ns;
            vm->fs_watchers[handle].size = size;
            if (event) {
                char buf[1200];
                int n = snprintf(buf, sizeof(buf), "%s\t%s", event, vm->fs_watchers[handle].path);
                if (n > 0 && n < (int)sizeof(buf)) {
                    vm_push(vm, vm_string_value(vm, buf, n));
                    break;
                }
            }
        }
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 1945: { /* fs-unwatch(watcher) → bool */
        Value handle_val = vm_pop(vm);
        int handle = (int)as_number(handle_val);
        if (handle > 0 && handle < (int)(sizeof(vm->fs_watchers) / sizeof(vm->fs_watchers[0])) &&
            vm->fs_watchers[handle].active) {
            memset(&vm->fs_watchers[handle], 0, sizeof(vm->fs_watchers[handle]));
            vm_push(vm, BOOL_VAL(1));
            break;
        }
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 1946: { /* ansi-strip(str) → string or #f */
        Value str_val = vm_pop(vm);
        VmString* s = vm_value_as_string(vm, str_val);
        vm_push(vm, vm_ansi_strip_value(vm, s));
        break;
    }
    case 1947: { /* string-display-width(str) → integer */
        Value str_val = vm_pop(vm);
        VmString* s = vm_value_as_string(vm, str_val);
        vm_push(vm, INT_VAL(s ? vm_string_display_width_bytes(s->data, s->byte_len) : 0));
        break;
    }
    case 1948: { /* string-truncate-display(str, max, suffix) → string */
        Value suffix_val = vm_pop(vm);
        Value max_val = vm_pop(vm);
        Value str_val = vm_pop(vm);
        VmString* s = vm_value_as_string(vm, str_val);
        VmString* suffix = vm_value_as_string(vm, suffix_val);
        vm_push(vm, vm_string_truncate_display_value(vm, s, (int64_t)as_number(max_val), suffix));
        break;
    }
    case 1949: { /* executable-path(name) → string or #f */
        Value name_val = vm_pop(vm);
        VmString* name = vm_value_as_string(vm, name_val);
        vm_push(vm, vm_executable_path_value(vm, name ? name->data : NULL));
        break;
    }
    case 1950: { /* monotonic-time-ms() → integer */
        vm_push(vm, INT_VAL(vm_monotonic_time_ms()));
        break;
    }
    case 1951: { /* temp-directory() → string */
        const char* tmp = vm_temp_directory_path();
        vm_push(vm, vm_string_value(vm, tmp, -1));
        break;
    }
    case 1952: { /* prevent-sleep(reason) → handle or #f */
        Value reason_val = vm_pop(vm);
        (void)reason_val;
        int64_t handle = vm_prevent_sleep_start(vm);
        if (handle > 0) vm_push(vm, INT_VAL(handle));
        else vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 1953: { /* allow-sleep(handle) → bool */
        Value handle_val = vm_pop(vm);
        vm_push(vm, BOOL_VAL(vm_prevent_sleep_stop(vm, (int64_t)as_number(handle_val))));
        break;
    }
    case 1954: { /* url-encode(str) → string */
        Value str_val = vm_pop(vm);
        VmString* s = vm_value_as_string(vm, str_val);
        vm_push(vm, vm_url_encode_value(vm, s));
        break;
    }
    case 1955: { /* url-decode(str) → string */
        Value str_val = vm_pop(vm);
        VmString* s = vm_value_as_string(vm, str_val);
        vm_push(vm, vm_url_decode_value(vm, s));
        break;
    }
    case 1960: { /* url-parse(str) → alist or #f */
        Value str_val = vm_pop(vm);
        VmString* s = vm_value_as_string(vm, str_val);
        vm_push(vm, vm_url_parse_value(vm, s));
        break;
    }
    case 1961: { /* base64url-encode(data) → string */
        Value data_val = vm_pop(vm);
        vm_push(vm, vm_base64url_encode_value(vm, data_val));
        break;
    }
    case 1962: { /* base64url-decode(data) → string or #f */
        Value data_val = vm_pop(vm);
        vm_push(vm, vm_base64url_decode_value(vm, data_val));
        break;
    }
    case 2080: { /* base64-encode-string(data) → padded RFC 4648 string */
        Value data_val = vm_pop(vm);
        vm_push(vm, vm_base64_encode_value(vm, data_val));
        break;
    }
    case 2081: { /* base64-decode-string(data) → decoded string or #f */
        Value data_val = vm_pop(vm);
        vm_push(vm, vm_base64_decode_value(vm, data_val));
        break;
    }
    case 1963: { /* uuid-v4() → string */
        vm_push(vm, vm_uuid_v4_value(vm));
        break;
    }
    case 1964: { /* constant-time-equal?(a, b) → bool */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        vm_push(vm, vm_constant_time_equal_value(vm, a_val, b_val));
        break;
    }
    case 1965: { /* sha256-file(path) → string or #f */
        Value path_val = vm_pop(vm);
        vm_push(vm, vm_sha256_file_value(vm, vm_value_as_string(vm, path_val)));
        break;
    }
    case 1966: { /* regex-compile(pattern) → handle or #f */
        Value pattern_val = vm_pop(vm);
        int handle = vm_regex_compile_handle(vm, vm_value_as_string(vm, pattern_val));
        if (handle > 0) vm_push(vm, INT_VAL((int64_t)handle));
        else vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 1967: { /* regex-free(handle) → bool */
        Value handle_val = vm_pop(vm);
        vm_push(vm, BOOL_VAL(vm_regex_free_handle(vm, handle_val)));
        break;
    }
    case 1968: { /* regex-match(handle, subject) → string or #f */
        Value subject_val = vm_pop(vm), handle_val = vm_pop(vm);
        vm_push(vm, vm_regex_match_value(vm, vm_regex_get(vm, handle_val),
                                         vm_value_as_string(vm, subject_val), 0));
        break;
    }
    case 1969: { /* regex-match?(handle, subject) → bool */
        Value subject_val = vm_pop(vm), handle_val = vm_pop(vm);
        vm_push(vm, vm_regex_match_value(vm, vm_regex_get(vm, handle_val),
                                         vm_value_as_string(vm, subject_val), 1));
        break;
    }
    case 1970: { /* regex-match-groups(handle, subject) → alist or #f */
        Value subject_val = vm_pop(vm), handle_val = vm_pop(vm);
        vm_push(vm, vm_regex_match_groups_value(vm, vm_regex_get(vm, handle_val),
                                                vm_value_as_string(vm, subject_val)));
        break;
    }
    case 1971: { /* regex-split(handle, subject) → list or #f */
        Value subject_val = vm_pop(vm), handle_val = vm_pop(vm);
        vm_push(vm, vm_regex_split_value(vm, vm_regex_get(vm, handle_val),
                                         vm_value_as_string(vm, subject_val)));
        break;
    }
    case 1972: { /* current-timestamp() → seconds since epoch */
        vm_push(vm, FLOAT_VAL((double)vm_current_time_ns_now() / 1000000000.0));
        break;
    }
    case 1973: { /* current-time-ns() → integer nanoseconds since epoch */
        vm_push(vm, INT_VAL(vm_current_time_ns_now()));
        break;
    }
    case 1974: { /* format-iso8601(ns) → string */
        Value ns_val = vm_pop(vm);
        vm_push(vm, vm_format_iso8601_value(vm, ns_val));
        break;
    }
    case 1975: { /* parse-iso8601(str) → integer nanoseconds or #f */
        Value str_val = vm_pop(vm);
        vm_push(vm, vm_parse_iso8601_value(vm, vm_value_as_string(vm, str_val)));
        break;
    }
    case 1976: { /* format-relative(seconds-ago) → string */
        Value seconds_val = vm_pop(vm);
        vm_push(vm, vm_format_relative_value(vm, (int64_t)as_number(seconds_val)));
        break;
    }
    case 1977: { /* local-timezone-offset() → seconds east of UTC */
        vm_push(vm, INT_VAL(vm_local_timezone_offset_seconds()));
        break;
    }
    case 1978: { /* diff-lines(old, new) → list of prefixed lines */
        Value new_val = vm_pop(vm), old_val = vm_pop(vm);
        vm_push(vm, vm_diff_lines_value(vm, vm_value_as_string(vm, old_val),
                                        vm_value_as_string(vm, new_val)));
        break;
    }
    case 1979: { /* fuzzy-match(pattern, candidates, key-fn, max) → ((score . candidate) ...) */
        Value max_val = vm_pop(vm), key_fn_val = vm_pop(vm);
        Value candidates_val = vm_pop(vm), pattern_val = vm_pop(vm);
        vm_push(vm, vm_fuzzy_match_value(vm, vm_value_as_string(vm, pattern_val),
                                         candidates_val, key_fn_val,
                                         (int)as_number(max_val)));
        break;
    }
    case 1980: { /* semver-parse(str) → alist or #f */
        Value str_val = vm_pop(vm);
        vm_push(vm, vm_semver_parse_value(vm, vm_value_as_string(vm, str_val)));
        break;
    }
    case 1981: { /* semver-compare(a, b) → -1, 0, 1, or #f */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        vm_push(vm, vm_semver_compare_value(vm, vm_value_as_string(vm, a_val),
                                            vm_value_as_string(vm, b_val)));
        break;
    }
    case 1982: { /* semver-satisfies?(version, range) → bool */
        Value range_val = vm_pop(vm), version_val = vm_pop(vm);
        vm_push(vm, vm_semver_satisfies_value(vm, vm_value_as_string(vm, version_val),
                                              vm_value_as_string(vm, range_val)));
        break;
    }
    case 1983: { /* make-pipe() → (read-fd . write-fd) or #f */
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
        int fds[2];
        if (pipe(fds) == 0) {
            Value pair = vm_cons_value(vm, INT_VAL((int64_t)fds[0]), INT_VAL((int64_t)fds[1]));
            vm_push(vm, pair);
            break;
        }
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 1984: { /* fd-write(fd, string-or-bytevector) → bytes or #f */
        Value data_val = vm_pop(vm), fd_val = vm_pop(vm);
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
        int fd = (int)as_number(fd_val);
        const void* data = NULL;
        size_t len = 0;
        VmString* s = vm_value_as_string(vm, data_val);
        if (s && s->data) {
            data = s->data;
            len = (size_t)s->byte_len;
        } else {
            VmBytevector* bv = vm_value_as_bytevector(vm, data_val);
            if (bv && bv->data) {
                data = bv->data;
                len = (size_t)bv->len;
            }
        }
        if (fd >= 0 && data) {
            ssize_t n = write(fd, data, len);
            if (n >= 0) {
                vm_push(vm, INT_VAL((int64_t)n));
                break;
            }
        }
#else
        (void)data_val; (void)fd_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 1985: { /* make-line-reader(fd, callback) → reader or #f */
        Value callback_val = vm_pop(vm), fd_val = vm_pop(vm);
        (void)callback_val;
        int handle = vm_line_reader_start(vm, (int)as_number(fd_val));
        if (handle > 0) vm_push(vm, INT_VAL((int64_t)handle));
        else vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 1986: { /* line-reader-poll(reader) → line string or #f */
        Value handle_val = vm_pop(vm);
        vm_push(vm, vm_line_reader_poll_value(vm, (int)as_number(handle_val)));
        break;
    }
    case 1987: { /* line-reader-close(reader) → bool */
        Value handle_val = vm_pop(vm);
        int handle = (int)as_number(handle_val);
        if (handle > 0 && handle < (int)(sizeof(vm->line_readers) / sizeof(vm->line_readers[0])) &&
            vm->line_readers[handle].active) {
            memset(&vm->line_readers[handle], 0, sizeof(vm->line_readers[handle]));
            vm_push(vm, BOOL_VAL(1));
            break;
        }
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 1988: { /* fd-close(fd) → bool */
        Value fd_val = vm_pop(vm);
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
        int fd = (int)as_number(fd_val);
        if (fd >= 0 && close(fd) == 0) {
            vm_push(vm, BOOL_VAL(1));
            break;
        }
#else
        (void)fd_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 1989: { /* make-lru-cache(max-size) → cache or #f */
        Value max_val = vm_pop(vm);
        int handle = vm_lru_create(vm, (int)as_number(max_val));
        if (handle > 0) vm_push(vm, INT_VAL((int64_t)handle));
        else vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 1990: { /* lru-get(cache, key) → value or #f */
        Value key_val = vm_pop(vm), cache_val = vm_pop(vm);
        vm_push(vm, vm_lru_get_value(vm, (int)as_number(cache_val), key_val));
        break;
    }
    case 1991: { /* lru-set!(cache, key, value) → bool */
        Value value_val = vm_pop(vm), key_val = vm_pop(vm), cache_val = vm_pop(vm);
        vm_push(vm, BOOL_VAL(vm_lru_set_value(vm, (int)as_number(cache_val), key_val, value_val)));
        break;
    }
    case 1992: { /* lru-has?(cache, key) → bool */
        Value key_val = vm_pop(vm), cache_val = vm_pop(vm);
        vm_push(vm, BOOL_VAL(vm_lru_find_entry(vm, (int)as_number(cache_val), key_val) >= 0));
        break;
    }
    case 1993: { /* lru-delete!(cache, key) → bool */
        Value key_val = vm_pop(vm), cache_val = vm_pop(vm);
        int handle = (int)as_number(cache_val);
        int idx = vm_lru_find_entry(vm, handle, key_val);
        if (idx >= 0) {
            memset(&vm->lru_caches[handle].entries[idx], 0, sizeof(vm->lru_caches[handle].entries[idx]));
            if (vm->lru_caches[handle].size > 0) vm->lru_caches[handle].size--;
            vm_push(vm, BOOL_VAL(1));
        } else {
            vm_push(vm, BOOL_VAL(0));
        }
        break;
    }
    case 1994: { /* lru-clear!(cache) → bool */
        Value cache_val = vm_pop(vm);
        int handle = (int)as_number(cache_val);
        if (vm_lru_cache_valid(vm, handle)) {
            memset(vm->lru_caches[handle].entries, 0, sizeof(vm->lru_caches[handle].entries));
            vm->lru_caches[handle].size = 0;
            vm_push(vm, BOOL_VAL(1));
        } else {
            vm_push(vm, BOOL_VAL(0));
        }
        break;
    }
    case 1995: { /* lru-size(cache) → integer */
        Value cache_val = vm_pop(vm);
        int handle = (int)as_number(cache_val);
        if (vm_lru_cache_valid(vm, handle))
            vm_push(vm, INT_VAL((int64_t)vm->lru_caches[handle].size));
        else
            vm_push(vm, INT_VAL(0));
        break;
    }
    case 1996: { /* _emit-event(emitter, event, args-list) → invoked-count or #f */
        Value args_val = vm_pop(vm), event_val = vm_pop(vm), emitter_val = vm_pop(vm);
        int invoked = vm_event_emit(vm, (int)as_number(emitter_val), event_val, args_val);
        if (invoked >= 0) vm_push(vm, INT_VAL((int64_t)invoked));
        else vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 1997: { /* make-event-emitter() → emitter or #f */
        int handle = vm_event_emitter_create(vm);
        if (handle > 0) vm_push(vm, INT_VAL((int64_t)handle));
        else vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 1998: { /* on!(emitter, event, handler) → bool */
        Value handler_val = vm_pop(vm), event_val = vm_pop(vm), emitter_val = vm_pop(vm);
        vm_push(vm, BOOL_VAL(vm_event_listener_add(vm, (int)as_number(emitter_val),
                                                   event_val, handler_val, 0)));
        break;
    }
    case 1999: { /* once!(emitter, event, handler) → bool */
        Value handler_val = vm_pop(vm), event_val = vm_pop(vm), emitter_val = vm_pop(vm);
        vm_push(vm, BOOL_VAL(vm_event_listener_add(vm, (int)as_number(emitter_val),
                                                   event_val, handler_val, 1)));
        break;
    }
    case 2000: { /* off!(emitter, event, handler) → removed-count */
        Value handler_val = vm_pop(vm), event_val = vm_pop(vm), emitter_val = vm_pop(vm);
        vm_push(vm, INT_VAL((int64_t)vm_event_listener_remove(vm, (int)as_number(emitter_val),
                                                              event_val, handler_val)));
        break;
    }
    case 2001: { /* make-channel(capacity) → channel or #f */
        Value capacity_val = vm_pop(vm);
        int handle = vm_channel_create(vm, (int)as_number(capacity_val));
        if (handle > 0) vm_push(vm, INT_VAL((int64_t)handle));
        else vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2002: { /* channel-send!(channel, value) → bool */
        Value value_val = vm_pop(vm), channel_val = vm_pop(vm);
        vm_push(vm, BOOL_VAL(vm_channel_send_value(vm, (int)as_number(channel_val), value_val)));
        break;
    }
    case 2003: { /* channel-receive(channel, timeout-ms) → value or #f */
        Value timeout_val = vm_pop(vm), channel_val = vm_pop(vm);
        (void)timeout_val;
        vm_push(vm, vm_channel_receive_value(vm, (int)as_number(channel_val)));
        break;
    }
    case 2004: { /* channel-try-receive(channel) → value or #f */
        Value channel_val = vm_pop(vm);
        vm_push(vm, vm_channel_receive_value(vm, (int)as_number(channel_val)));
        break;
    }
    case 2005: { /* channel-close!(channel) → bool */
        Value channel_val = vm_pop(vm);
        int handle = (int)as_number(channel_val);
        if (vm_channel_valid(vm, handle)) {
            vm->channels[handle].closed = 1;
            vm_push(vm, BOOL_VAL(1));
        } else {
            vm_push(vm, BOOL_VAL(0));
        }
        break;
    }
    case 2006: { /* make-mutex() → mutex or #f */
        int handle = vm_mutex_create(vm);
        if (handle > 0) vm_push(vm, INT_VAL((int64_t)handle));
        else vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2007: { /* mutex-lock!(mutex) → bool */
        Value mutex_val = vm_pop(vm);
        vm_push(vm, BOOL_VAL(vm_mutex_lock_handle(vm, (int)as_number(mutex_val))));
        break;
    }
    case 2008: { /* mutex-unlock!(mutex) → bool */
        Value mutex_val = vm_pop(vm);
        vm_push(vm, BOOL_VAL(vm_mutex_unlock_handle(vm, (int)as_number(mutex_val))));
        break;
    }
    case 2009: { /* with-mutex(mutex, thunk) → value or #f */
        Value thunk_val = vm_pop(vm), mutex_val = vm_pop(vm);
        int handle = (int)as_number(mutex_val);
        if (thunk_val.type == VAL_CLOSURE && vm_mutex_lock_handle(vm, handle)) {
            Value result = vm_call_closure_from_native(vm, thunk_val, NULL, 0);
            (void)vm_mutex_unlock_handle(vm, handle);
            vm_push(vm, result);
        } else {
            vm_push(vm, BOOL_VAL(0));
        }
        break;
    }
    case 2010: { /* make-condition-variable() → cv or #f */
        int handle = vm_condvar_create(vm);
        if (handle > 0) vm_push(vm, INT_VAL((int64_t)handle));
        else vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2011: { /* condition-wait(cv, mutex) → bool */
        Value mutex_val = vm_pop(vm), cv_val = vm_pop(vm);
        int cv = (int)as_number(cv_val);
        int mutex = (int)as_number(mutex_val);
        if (vm_condvar_valid(vm, cv) && vm_mutex_valid(vm, mutex)) {
            (void)vm_mutex_unlock_handle(vm, mutex);
            (void)vm_mutex_lock_handle(vm, mutex);
            vm_push(vm, BOOL_VAL(1));
        } else {
            vm_push(vm, BOOL_VAL(0));
        }
        break;
    }
    case 2012: { /* condition-signal(cv) → bool */
        Value cv_val = vm_pop(vm);
        int cv = (int)as_number(cv_val);
        if (vm_condvar_valid(vm, cv)) {
            vm->condvars[cv].signals++;
            vm_push(vm, BOOL_VAL(1));
        } else {
            vm_push(vm, BOOL_VAL(0));
        }
        break;
    }
    case 2013: { /* condition-broadcast(cv) → bool */
        Value cv_val = vm_pop(vm);
        int cv = (int)as_number(cv_val);
        if (vm_condvar_valid(vm, cv)) {
            vm->condvars[cv].signals++;
            vm_push(vm, BOOL_VAL(1));
        } else {
            vm_push(vm, BOOL_VAL(0));
        }
        break;
    }
    case 2022: { /* make-timer(delay-ms, callback) → timer or #f */
        Value callback_val = vm_pop(vm), delay_val = vm_pop(vm);
        int handle = vm_timer_create(vm, (int64_t)as_number(delay_val), callback_val, 0);
        if (handle > 0) vm_push(vm, INT_VAL((int64_t)handle));
        else vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2023: { /* timer-cancel!(timer) → bool */
        Value timer_val = vm_pop(vm);
        vm_push(vm, BOOL_VAL(vm_timer_cancel(vm, (int)as_number(timer_val))));
        break;
    }
    case 2024: { /* make-interval(interval-ms, callback) → interval or #f */
        Value callback_val = vm_pop(vm), interval_val = vm_pop(vm);
        int handle = vm_timer_create(vm, (int64_t)as_number(interval_val), callback_val, 1);
        if (handle > 0) vm_push(vm, INT_VAL((int64_t)handle));
        else vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2025: { /* interval-cancel!(interval) → bool */
        Value interval_val = vm_pop(vm);
        vm_push(vm, BOOL_VAL(vm_timer_cancel(vm, (int)as_number(interval_val))));
        break;
    }
    case 2026: { /* timer-check(timer) → bool */
        Value timer_val = vm_pop(vm);
        int handle = (int)as_number(timer_val);
        if (vm_timer_valid(vm, handle) && vm->timers[handle].fired_count > 0) {
            vm->timers[handle].fired_count--;
            vm_push(vm, BOOL_VAL(1));
        } else {
            vm_push(vm, BOOL_VAL(0));
        }
        break;
    }
    case 2027: { /* db-transaction(db, thunk) → result or #f */
        Value thunk_val = vm_pop(vm), db_val = vm_pop(vm);
        VmSqliteExecFn exec_fn = vm_sqlite_exec_symbol();
        int64_t db = (int64_t)as_number(db_val);
        if (!exec_fn || thunk_val.type != VAL_CLOSURE || db < 0 ||
            exec_fn(db, "BEGIN") != 0) {
            vm_push(vm, BOOL_VAL(0));
            break;
        }
        Value result = vm_call_closure_from_native(vm, thunk_val, NULL, 0);
        if (exec_fn(db, "COMMIT") != 0) {
            (void)exec_fn(db, "ROLLBACK");
            vm_push(vm, BOOL_VAL(0));
        } else {
            vm_push(vm, result);
        }
        break;
    }
    case 2028: { /* db-busy-timeout(db, ms) → bool */
        Value ms_val = vm_pop(vm), db_val = vm_pop(vm);
        VmSqliteExecFn exec_fn = vm_sqlite_exec_symbol();
        int64_t db = (int64_t)as_number(db_val);
        int64_t ms = (int64_t)as_number(ms_val);
        if (ms < 0) ms = 0;
        if (exec_fn && db >= 0) {
            char sql[96];
            snprintf(sql, sizeof(sql), "PRAGMA busy_timeout=%lld", (long long)ms);
            vm_push(vm, BOOL_VAL(exec_fn(db, sql) == 0));
        } else {
            vm_push(vm, BOOL_VAL(0));
        }
        break;
    }
    case 2029: { /* db-last-insert-id(db) → integer or #f */
        Value db_val = vm_pop(vm);
        VmSqliteLastInsertIdFn fn = vm_sqlite_last_insert_id_symbol();
        if (fn) {
            int64_t rowid = fn((int64_t)as_number(db_val));
            if (rowid >= 0) {
                vm_push(vm, INT_VAL(rowid));
                break;
            }
        }
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2030: { /* db-changes(db) → integer or #f */
        Value db_val = vm_pop(vm);
        VmSqliteChangesFn fn = vm_sqlite_changes_symbol();
        if (fn) {
            int changes = fn((int64_t)as_number(db_val));
            if (changes >= 0) {
                vm_push(vm, INT_VAL((int64_t)changes));
                break;
            }
        }
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2031: { /* at-exit(thunk) → bool */
        Value thunk_val = vm_pop(vm);
        vm_push(vm, BOOL_VAL(vm_exit_handler_add(vm, thunk_val)));
        break;
    }
    case 2032: { /* dlopen(path) → handle or #f */
        Value path_val = vm_pop(vm);
#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM)
        VmString* path = vm_value_as_string(vm, path_val);
        if (path && path->data) {
            const char* cpath = (path->byte_len == 0) ? NULL : path->data;
            void* handle = dlopen(cpath, RTLD_LAZY | RTLD_LOCAL);
            int slot = vm_dlopen_store(vm, handle);
            if (slot > 0) {
                vm_push(vm, INT_VAL((int64_t)slot));
                break;
            }
        }
#else
        (void)path_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2033: { /* dlsym(handle, name) → pointer-integer or #f */
        Value name_val = vm_pop(vm), handle_val = vm_pop(vm);
#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM)
        int handle = (int)as_number(handle_val);
        VmString* name = vm_value_as_string(vm, name_val);
        if (vm_dlopen_valid(vm, handle) && name && name->data && name->byte_len > 0) {
            void* ptr = dlsym(vm->dynamic_libraries[handle].handle, name->data);
            if (ptr) {
                vm_push(vm, INT_VAL((int64_t)(intptr_t)ptr));
                break;
            }
        }
#else
        (void)name_val; (void)handle_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2034: { /* dlclose(handle) → bool */
        Value handle_val = vm_pop(vm);
#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM)
        int handle = (int)as_number(handle_val);
        if (vm_dlopen_valid(vm, handle) &&
            dlclose(vm->dynamic_libraries[handle].handle) == 0) {
            vm->dynamic_libraries[handle].active = 0;
            vm->dynamic_libraries[handle].handle = NULL;
            vm_push(vm, BOOL_VAL(1));
            break;
        }
#else
        (void)handle_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2035: { /* _format-list(fmt, args) → string or #f */
        Value args_val = vm_pop(vm), fmt_val = vm_pop(vm);
        vm_push(vm, vm_format_list_value(vm, vm_value_as_string(vm, fmt_val), args_val));
        break;
    }
    case 2036: { /* yoga-node-create() → node or #f */
        VmYogaCreateFn fn = (VmYogaCreateFn)(uintptr_t)
            vm_runtime_symbol("eshkol_yoga_node_create");
        int64_t handle = fn ? fn() : -1;
        if (handle > 0 && handle < (int64_t)(sizeof(vm->yoga_nodes) / sizeof(vm->yoga_nodes[0]))) {
            memset(&vm->yoga_nodes[handle], 0, sizeof(vm->yoga_nodes[handle]));
            vm->yoga_nodes[handle].active = 1;
            vm_push(vm, INT_VAL(handle));
        } else {
            vm_push(vm, BOOL_VAL(0));
        }
        break;
    }
    case 2037: { /* yoga-node-set!(node, prop, value) → bool */
        Value value_val = vm_pop(vm), prop_val = vm_pop(vm), node_val = vm_pop(vm);
        int handle = (int)as_number(node_val);
        VmString* prop = vm_value_as_string(vm, prop_val);
        VmYogaSetFloatFn set_float = (VmYogaSetFloatFn)(uintptr_t)
            vm_runtime_symbol("eshkol_yoga_node_set_float");
        VmYogaSetIntFn set_int = (VmYogaSetIntFn)(uintptr_t)
            vm_runtime_symbol("eshkol_yoga_node_set_int");
        int ok = vm_yoga_valid(vm, handle) && prop && prop->data && set_float && set_int;
        int float_prop = -1;
        int all_start = -1;
        if (ok && vm_yoga_string_eq(prop, "width")) float_prop = 0;
        else if (ok && vm_yoga_string_eq(prop, "height")) float_prop = 1;
        else if (ok && vm_yoga_string_eq(prop, "min-width")) float_prop = 2;
        else if (ok && vm_yoga_string_eq(prop, "min-height")) float_prop = 3;
        else if (ok && vm_yoga_string_eq(prop, "max-width")) float_prop = 4;
        else if (ok && vm_yoga_string_eq(prop, "max-height")) float_prop = 5;
        else if (ok && vm_yoga_string_eq(prop, "flex-grow")) float_prop = 6;
        else if (ok && vm_yoga_string_eq(prop, "flex-shrink")) float_prop = 7;
        else if (ok && vm_yoga_string_eq(prop, "flex-basis")) float_prop = 8;
        else if (ok && vm_yoga_string_eq(prop, "gap")) float_prop = 9;
        else if (ok && vm_yoga_string_eq(prop, "padding")) all_start = 10;
        else if (ok && vm_yoga_string_eq(prop, "padding-left")) float_prop = 10;
        else if (ok && vm_yoga_string_eq(prop, "padding-right")) float_prop = 11;
        else if (ok && vm_yoga_string_eq(prop, "padding-top")) float_prop = 12;
        else if (ok && vm_yoga_string_eq(prop, "padding-bottom")) float_prop = 13;
        else if (ok && vm_yoga_string_eq(prop, "margin")) all_start = 14;
        else if (ok && vm_yoga_string_eq(prop, "margin-left")) float_prop = 14;
        else if (ok && vm_yoga_string_eq(prop, "margin-right")) float_prop = 15;
        else if (ok && vm_yoga_string_eq(prop, "margin-top")) float_prop = 16;
        else if (ok && vm_yoga_string_eq(prop, "margin-bottom")) float_prop = 17;
        else if (ok && vm_yoga_string_eq(prop, "border")) all_start = 18;
        else if (ok && vm_yoga_string_eq(prop, "border-left")) float_prop = 18;
        else if (ok && vm_yoga_string_eq(prop, "border-right")) float_prop = 19;
        else if (ok && vm_yoga_string_eq(prop, "border-top")) float_prop = 20;
        else if (ok && vm_yoga_string_eq(prop, "border-bottom")) float_prop = 21;
        if (float_prop >= 0 || all_start >= 0) {
            double number = as_number(value_val);
            ok = isfinite(number);
            if (ok && all_start >= 0) {
                for (int i = 0; i < 4; ++i) set_float(handle, all_start + i, number);
            } else if (ok) {
                set_float(handle, float_prop, number);
            }
            vm_push(vm, BOOL_VAL(ok));
            break;
        }
        static const char* const flex_direction[] = {"column", "column-reverse", "row", "row-reverse"};
        static const char* const justify[] = {"flex-start", "center", "flex-end", "space-between", "space-around", "space-evenly"};
        static const char* const align[] = {"auto", "flex-start", "center", "flex-end", "stretch", "baseline", "space-between", "space-around", "space-evenly"};
        static const char* const position[] = {"static", "relative", "absolute"};
        static const char* const overflow[] = {"visible", "hidden", "scroll"};
        static const char* const display[] = {"flex", "none", "contents"};
        VmString* enum_string = vm_value_as_string(vm, value_val);
        int int_prop = -1;
        int enum_value = -1;
        if (ok && vm_yoga_string_eq(prop, "flex-direction")) {
            int_prop = 0; ok = vm_yoga_enum_value(enum_string, flex_direction, 4, &enum_value);
        } else if (ok && vm_yoga_string_eq(prop, "justify-content")) {
            int_prop = 1; ok = vm_yoga_enum_value(enum_string, justify, 6, &enum_value);
        } else if (ok && vm_yoga_string_eq(prop, "align-items")) {
            int_prop = 2; ok = vm_yoga_enum_value(enum_string, align, 9, &enum_value);
        } else if (ok && vm_yoga_string_eq(prop, "align-self")) {
            int_prop = 3; ok = vm_yoga_enum_value(enum_string, align, 9, &enum_value);
        } else if (ok && vm_yoga_string_eq(prop, "align-content")) {
            int_prop = 4; ok = vm_yoga_enum_value(enum_string, align, 9, &enum_value);
        } else if (ok && vm_yoga_string_eq(prop, "position-type")) {
            int_prop = 5; ok = vm_yoga_enum_value(enum_string, position, 3, &enum_value);
        } else if (ok && vm_yoga_string_eq(prop, "overflow")) {
            int_prop = 6; ok = vm_yoga_enum_value(enum_string, overflow, 3, &enum_value);
        } else if (ok && vm_yoga_string_eq(prop, "display")) {
            int_prop = 7; ok = vm_yoga_enum_value(enum_string, display, 3, &enum_value);
        } else {
            ok = 0;
        }
        if (ok) set_int(handle, int_prop, enum_value);
        vm_push(vm, BOOL_VAL(ok));
        break;
    }
    case 2038: { /* yoga-node-add-child!(parent, child) → bool */
        Value child_val = vm_pop(vm), parent_val = vm_pop(vm);
        int parent = (int)as_number(parent_val);
        int child = (int)as_number(child_val);
        VmYogaAddChildFn fn = (VmYogaAddChildFn)(uintptr_t)
            vm_runtime_symbol("eshkol_yoga_node_add_child");
        int ok = fn && vm_yoga_valid(vm, parent) && vm_yoga_valid(vm, child) &&
                 parent != child && vm->yoga_nodes[child].parent == 0 &&
                 vm->yoga_nodes[parent].child_count < 511;
        if (ok) {
            for (int ancestor = parent; ancestor != 0;
                 ancestor = vm->yoga_nodes[ancestor].parent) {
                if (ancestor == child) { ok = 0; break; }
            }
        }
        if (ok) {
            int index = vm->yoga_nodes[parent].child_count;
            fn(parent, child, index);
            vm->yoga_nodes[parent].child_count++;
            vm->yoga_nodes[child].parent = parent;
        }
        vm_push(vm, BOOL_VAL(ok));
        break;
    }
    case 2039: { /* yoga-node-calculate!(root, width, height) → bool */
        Value height_val = vm_pop(vm), width_val = vm_pop(vm), root_val = vm_pop(vm);
        int root = (int)as_number(root_val);
        VmYogaCalculateFn fn = (VmYogaCalculateFn)(uintptr_t)
            vm_runtime_symbol("eshkol_yoga_node_calculate");
        double width = as_number(width_val);
        double height = as_number(height_val);
        int ok = fn && vm_yoga_valid(vm, root) && isfinite(width) && isfinite(height) &&
                 width >= 0.0 && height >= 0.0;
        if (ok) fn(root, width, height);
        vm_push(vm, BOOL_VAL(ok));
        break;
    }
    case 2040: { /* yoga-node-get-computed(node, prop) → number or #f */
        Value prop_val = vm_pop(vm), node_val = vm_pop(vm);
        int handle = (int)as_number(node_val);
        VmString* prop = vm_value_as_string(vm, prop_val);
        VmYogaGetComputedFn fn = (VmYogaGetComputedFn)(uintptr_t)
            vm_runtime_symbol("eshkol_yoga_node_get_computed");
        int computed_prop = -1;
        if (vm_yoga_string_eq(prop, "left")) computed_prop = 0;
        else if (vm_yoga_string_eq(prop, "top")) computed_prop = 1;
        else if (vm_yoga_string_eq(prop, "width")) computed_prop = 2;
        else if (vm_yoga_string_eq(prop, "height")) computed_prop = 3;
        else if (vm_yoga_string_eq(prop, "padding-left")) computed_prop = 4;
        else if (vm_yoga_string_eq(prop, "padding-top")) computed_prop = 5;
        else if (vm_yoga_string_eq(prop, "padding-right")) computed_prop = 6;
        else if (vm_yoga_string_eq(prop, "padding-bottom")) computed_prop = 7;
        else if (vm_yoga_string_eq(prop, "margin-left")) computed_prop = 8;
        else if (vm_yoga_string_eq(prop, "margin-top")) computed_prop = 9;
        else if (vm_yoga_string_eq(prop, "margin-right")) computed_prop = 10;
        else if (vm_yoga_string_eq(prop, "margin-bottom")) computed_prop = 11;
        else if (vm_yoga_string_eq(prop, "border-left")) computed_prop = 12;
        else if (vm_yoga_string_eq(prop, "border-top")) computed_prop = 13;
        else if (vm_yoga_string_eq(prop, "border-right")) computed_prop = 14;
        else if (vm_yoga_string_eq(prop, "border-bottom")) computed_prop = 15;
        if (fn && vm_yoga_valid(vm, handle) && computed_prop >= 0)
            vm_push(vm, FLOAT_VAL(fn(handle, computed_prop)));
        else
            vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2041: { /* yoga-node-free!(node) → bool */
        Value node_val = vm_pop(vm);
        int handle = (int)as_number(node_val);
        VmYogaFreeFn fn = (VmYogaFreeFn)(uintptr_t)
            vm_runtime_symbol("eshkol_yoga_node_free");
        if (fn && vm_yoga_valid(vm, handle)) {
            int parent = vm->yoga_nodes[handle].parent;
            if (vm_yoga_valid(vm, parent) && vm->yoga_nodes[parent].child_count > 0)
                vm->yoga_nodes[parent].child_count--;
            vm_yoga_clear_subtree(vm, handle);
            fn(handle);
            vm_push(vm, BOOL_VAL(1));
            break;
        }
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2042: { /* http-server-create(port) → server or #f */
        Value port_val = vm_pop(vm);
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
        int requested_port = (int)as_number(port_val);
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd >= 0) {
            int opt = 1;
            (void)setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, (socklen_t)sizeof(opt));
            struct sockaddr_in addr;
            memset(&addr, 0, sizeof(addr));
            addr.sin_family = AF_INET;
            addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            addr.sin_port = htons((uint16_t)requested_port);
            if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) == 0 &&
                listen(fd, 8) == 0) {
                socklen_t addr_len = (socklen_t)sizeof(addr);
                if (getsockname(fd, (struct sockaddr*)&addr, &addr_len) == 0) {
                    int handle = vm_http_server_store(vm, fd, (int)ntohs(addr.sin_port));
                    if (handle > 0) {
                        vm_push(vm, INT_VAL((int64_t)handle));
                        break;
                    }
                }
            }
            close(fd);
        }
#else
        (void)port_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2043: { /* http-server-port(server) → integer or #f */
        Value server_val = vm_pop(vm);
        int handle = (int)as_number(server_val);
        if (vm_http_server_valid(vm, handle)) {
            vm_push(vm, INT_VAL((int64_t)vm->http_servers[handle].port));
            break;
        }
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2044: { /* http-server-accept(server, buffer-size, timeout-ms) → request string or #f */
        Value timeout_val = vm_pop(vm), buffer_val = vm_pop(vm), server_val = vm_pop(vm);
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
        int handle = (int)as_number(server_val);
        int buffer_size = (int)as_number(buffer_val);
        int timeout_ms = (int)as_number(timeout_val);
        if (buffer_size < 128) buffer_size = 128;
        if (buffer_size > 65536) buffer_size = 65536;
        if (timeout_ms < 0) timeout_ms = 0;
        if (vm_http_server_valid(vm, handle)) {
            struct pollfd pfd;
            pfd.fd = vm->http_servers[handle].listen_fd;
            pfd.events = POLLIN;
            pfd.revents = 0;
            if (poll(&pfd, 1, timeout_ms) > 0 && (pfd.revents & POLLIN)) {
                int client_fd = accept(vm->http_servers[handle].listen_fd, NULL, NULL);
                if (client_fd >= 0) {
                    if (vm->http_servers[handle].client_fd >= 0)
                        close(vm->http_servers[handle].client_fd);
                    vm->http_servers[handle].client_fd = client_fd;
                    char* buf = (char*)malloc((size_t)buffer_size + 1);
                    if (buf) {
                        int total = 0;
                        while (total < buffer_size) {
                            ssize_t n = recv(client_fd, buf + total, (size_t)(buffer_size - total), 0);
                            if (n <= 0) break;
                            total += (int)n;
                            buf[total] = '\0';
                            if (strstr(buf, "\r\n\r\n") || strstr(buf, "\n\n")) break;
                        }
                        Value request = total > 0 ? vm_string_value(vm, buf, total) : BOOL_VAL(0);
                        free(buf);
                        vm_push(vm, request);
                        break;
                    }
                }
            }
        }
#else
        (void)timeout_val; (void)buffer_val; (void)server_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2045: { /* http-server-respond(server, status, content-type, body) → bool */
        Value body_val = vm_pop(vm), type_val = vm_pop(vm), status_val = vm_pop(vm), server_val = vm_pop(vm);
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
        int handle = (int)as_number(server_val);
        int status = (int)as_number(status_val);
        VmString* content_type = vm_value_as_string(vm, type_val);
        VmString* body = vm_value_as_string(vm, body_val);
        if (vm_http_server_valid(vm, handle) && vm->http_servers[handle].client_fd >= 0 &&
            body && body->data) {
            if (status <= 0) status = 200;
            const char* reason = status == 404 ? "Not Found" :
                                 status == 500 ? "Internal Server Error" :
                                 status == 400 ? "Bad Request" : "OK";
            const char* ctype = (content_type && content_type->data && content_type->byte_len > 0)
                ? content_type->data : "text/plain";
            char header[512];
            int hlen = snprintf(header, sizeof(header),
                                "HTTP/1.1 %d %s\r\n"
                                "Content-Type: %s\r\n"
                                "Content-Length: %lld\r\n"
                                "Connection: close\r\n\r\n",
                                status, reason, ctype, (long long)body->byte_len);
            if (hlen > 0 && hlen < (int)sizeof(header) &&
                send(vm->http_servers[handle].client_fd, header, (size_t)hlen, 0) == hlen &&
                send(vm->http_servers[handle].client_fd, body->data, (size_t)body->byte_len, 0) == body->byte_len) {
                close(vm->http_servers[handle].client_fd);
                vm->http_servers[handle].client_fd = -1;
                vm_push(vm, BOOL_VAL(1));
                break;
            }
        }
#else
        (void)body_val; (void)type_val; (void)status_val; (void)server_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2046: { /* http-server-close(server) → bool */
        Value server_val = vm_pop(vm);
        int handle = (int)as_number(server_val);
        if (vm_http_server_valid(vm, handle)) {
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
            if (vm->http_servers[handle].client_fd >= 0) close(vm->http_servers[handle].client_fd);
            if (vm->http_servers[handle].listen_fd >= 0) close(vm->http_servers[handle].listen_fd);
#endif
            memset(&vm->http_servers[handle], 0, sizeof(vm->http_servers[handle]));
            vm_push(vm, BOOL_VAL(1));
            break;
        }
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2047: { /* http-request(method, url, headers, body, timeout-ms) → (status headers body) or #f */
        Value timeout_val = vm_pop(vm), body_val = vm_pop(vm), headers_val = vm_pop(vm);
        Value url_val = vm_pop(vm), method_val = vm_pop(vm);
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
        VmString* method = vm_value_as_string(vm, method_val);
        VmString* url = vm_value_as_string(vm, url_val);
        VmString* headers = vm_value_as_string(vm, headers_val);
        VmString* body = vm_value_as_string(vm, body_val);
        char host[256];
        char path[2048];
        int port = 80;
        int timeout_ms = (int)as_number(timeout_val);
        if (timeout_ms <= 0) timeout_ms = 30000;
        if (method && method->data && method->byte_len > 0 &&
            vm_http_parse_url(url, host, sizeof(host), &port, path, sizeof(path))) {
            int fd = vm_http_connect(host, port, timeout_ms);
            if (fd >= 0) {
                const char* body_data = (body && body->data) ? body->data : "";
                int64_t body_len = (body && body->data) ? body->byte_len : 0;
                const char* extra_headers = (headers && headers->data) ? headers->data : "";
                char header[8192];
                int hlen = snprintf(header, sizeof(header),
                                    "%.*s %s HTTP/1.1\r\n"
                                    "Host: %s\r\n"
                                    "Connection: close\r\n"
                                    "%s%s"
                                    "Content-Length: %lld\r\n\r\n",
                                    (int)method->byte_len, method->data,
                                    path, host,
                                    extra_headers,
                                    extra_headers[0] ? (strstr(extra_headers, "\n") ? "" : "\r\n") : "",
                                    (long long)body_len);
                int ok = hlen > 0 && hlen < (int)sizeof(header);
                if (ok) {
                    ssize_t sent = send(fd, header, (size_t)hlen, 0);
                    ok = sent == hlen;
                }
                if (ok && body_len > 0) {
                    ssize_t sent = send(fd, body_data, (size_t)body_len, 0);
                    ok = sent == body_len;
                }
                if (ok) {
                    int cap = 65536;
                    char* response = (char*)malloc((size_t)cap + 1);
                    if (response) {
                        int total = 0;
                        while (total < cap) {
                            ssize_t n = recv(fd, response + total, (size_t)(cap - total), 0);
                            if (n <= 0) break;
                            total += (int)n;
                        }
                        response[total] = '\0';
                        close(fd);
                        int status = 0;
                        (void)sscanf(response, "HTTP/%*s %d", &status);
                        char* body_start = strstr(response, "\r\n\r\n");
                        int header_len = total;
                        int body_offset = total;
                        if (body_start) {
                            body_start += 4;
                            body_offset = (int)(body_start - response);
                            header_len = body_offset - 4;
                        } else {
                            body_start = response + total;
                        }
                        Value result = vm_http_response_value(vm, status,
                                                              response, header_len,
                                                              body_start,
                                                              total - body_offset);
                        free(response);
                        vm_push(vm, result);
                        break;
                    }
                }
                close(fd);
            }
        }
#else
        (void)timeout_val; (void)body_val; (void)headers_val; (void)url_val; (void)method_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2048: { /* websocket-connect(url, headers) → ws or #f */
        Value headers_val = vm_pop(vm), url_val = vm_pop(vm);
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
        VmString* url = vm_value_as_string(vm, url_val);
        VmString* headers = vm_value_as_string(vm, headers_val);
        char host[256];
        char path[2048];
        int port = 80;
        if (vm_ws_parse_url(url, host, sizeof(host), &port, path, sizeof(path))) {
            int fd = vm_http_connect(host, port, 5000);
            if (fd >= 0) {
                const char* extra_headers = (headers && headers->data) ? headers->data : "";
                char request[4096];
                int n = snprintf(request, sizeof(request),
                                 "GET %s HTTP/1.1\r\n"
                                 "Host: %s\r\n"
                                 "Upgrade: websocket\r\n"
                                 "Connection: Upgrade\r\n"
                                 "Sec-WebSocket-Version: 13\r\n"
                                 "Sec-WebSocket-Key: ZXNoa29sLXZtLXdlYnNvY2tldA==\r\n"
                                 "%s%s"
                                 "\r\n",
                                 path, host, extra_headers,
                                 extra_headers[0] ? (strstr(extra_headers, "\n") ? "" : "\r\n") : "");
                if (n > 0 && n < (int)sizeof(request) &&
                    vm_ws_send_all(fd, request, (size_t)n)) {
                    char response[4096];
                    ssize_t r = recv(fd, response, sizeof(response) - 1, 0);
                    if (r > 0) {
                        response[r] = '\0';
                        if (strstr(response, " 101 ") || strstr(response, " 101\r") ||
                            strstr(response, " 101\n")) {
                            int handle = vm_ws_store(vm, fd);
                            if (handle > 0) {
                                vm_push(vm, INT_VAL((int64_t)handle));
                                break;
                            }
                        }
                    }
                }
                close(fd);
            }
        }
#else
        (void)headers_val; (void)url_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2049: { /* websocket-send(ws, data) → bool */
        Value data_val = vm_pop(vm), ws_val = vm_pop(vm);
        int handle = (int)as_number(ws_val);
        VmString* data = vm_value_as_string(vm, data_val);
        if (vm_ws_valid(vm, handle) && data && data->data) {
            vm_push(vm, BOOL_VAL(vm_ws_send_frame(vm->websocket_clients[handle].fd, 1,
                                                  data->data, (size_t)data->byte_len)));
            break;
        }
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2050: { /* websocket-send-binary(ws, data) → bool */
        Value data_val = vm_pop(vm), ws_val = vm_pop(vm);
        int handle = (int)as_number(ws_val);
        const void* data = NULL;
        size_t len = 0;
        VmString* s = vm_value_as_string(vm, data_val);
        if (s && s->data) {
            data = s->data;
            len = (size_t)s->byte_len;
        } else {
            VmBytevector* bv = vm_value_as_bytevector(vm, data_val);
            if (bv && bv->data) {
                data = bv->data;
                len = (size_t)bv->len;
            }
        }
        if (vm_ws_valid(vm, handle) && data) {
            vm_push(vm, BOOL_VAL(vm_ws_send_frame(vm->websocket_clients[handle].fd, 2, data, len)));
            break;
        }
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2051: { /* websocket-receive(ws, timeout-ms) → (type . data) or #f */
        Value timeout_val = vm_pop(vm), ws_val = vm_pop(vm);
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
        int handle = (int)as_number(ws_val);
        int timeout_ms = (int)as_number(timeout_val);
        if (timeout_ms < 0) timeout_ms = 0;
        if (vm_ws_valid(vm, handle)) {
            int fd = vm->websocket_clients[handle].fd;
            struct pollfd pfd;
            pfd.fd = fd;
            pfd.events = POLLIN;
            pfd.revents = 0;
            if (poll(&pfd, 1, timeout_ms) > 0 && (pfd.revents & POLLIN)) {
                unsigned char hdr[2];
                if (recv(fd, hdr, 2, MSG_WAITALL) == 2) {
                    int opcode = hdr[0] & 0x0f;
                    int masked = hdr[1] & 0x80;
                    uint64_t len = hdr[1] & 0x7f;
                    if (len == 126) {
                        unsigned char ext[2];
                        if (recv(fd, ext, 2, MSG_WAITALL) != 2) len = 65537;
                        else len = ((uint64_t)ext[0] << 8) | ext[1];
                    } else if (len == 127) {
                        unsigned char ext[8];
                        if (recv(fd, ext, 8, MSG_WAITALL) != 8) len = 65537;
                        else {
                            len = 0;
                            for (int i = 0; i < 8; i++) len = (len << 8) | ext[i];
                        }
                    }
                    unsigned char mask[4] = {0, 0, 0, 0};
                    if (masked && recv(fd, mask, 4, MSG_WAITALL) != 4) len = 65537;
                    if (len <= 65536) {
                        char* payload = (char*)malloc((size_t)len + 1);
                        if (payload) {
                            if (recv(fd, payload, (size_t)len, MSG_WAITALL) == (ssize_t)len) {
                                if (masked) {
                                    for (uint64_t i = 0; i < len; i++) payload[i] ^= (char)mask[i % 4];
                                }
                                payload[len] = '\0';
                                const char* type = opcode == 1 ? "text" :
                                                   opcode == 2 ? "binary" :
                                                   opcode == 9 ? "ping" :
                                                   opcode == 8 ? "close" : "unknown";
                                if (opcode == 8) vm->websocket_clients[handle].closed = 1;
                                Value result = vm_cons_value(vm,
                                    vm_string_value(vm, type, -1),
                                    vm_string_value(vm, payload, (int64_t)len));
                                free(payload);
                                vm_push(vm, result);
                                break;
                            }
                            free(payload);
                        }
                    }
                }
            }
        }
#else
        (void)timeout_val; (void)ws_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2052: { /* websocket-close(ws) → bool */
        Value ws_val = vm_pop(vm);
        int handle = (int)as_number(ws_val);
        if (vm_ws_valid(vm, handle)) {
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
            (void)vm_ws_send_frame(vm->websocket_clients[handle].fd, 8, "", 0);
            close(vm->websocket_clients[handle].fd);
#endif
            memset(&vm->websocket_clients[handle], 0, sizeof(vm->websocket_clients[handle]));
            vm_push(vm, BOOL_VAL(1));
            break;
        }
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2053: { /* ts-parser-new(language) → parser or #f */
        Value language_val = vm_pop(vm);
        VmString* language = vm_value_as_string(vm, language_val);
        VmTsParserNewFn new_fn = (VmTsParserNewFn)(uintptr_t)
            vm_ts_symbol("eshkol_ts_parser_new");
        char* language_copy = vm_ts_string_copy(language);
        if (new_fn && language_copy) {
            int64_t handle = new_fn(language_copy);
            free(language_copy);
            if (handle > 0 &&
                handle < (int64_t)(sizeof(vm->ts_parsers) / sizeof(vm->ts_parsers[0]))) {
                vm->ts_parsers[handle].active = 1;
                vm_push(vm, INT_VAL(handle));
                break;
            }
            if (handle > 0) {
                VmTsParserFreeFn free_fn = (VmTsParserFreeFn)(uintptr_t)
                    vm_ts_symbol("eshkol_ts_parser_free");
                if (free_fn) free_fn(handle);
            }
        } else {
            free(language_copy);
        }
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2054: { /* ts-parser-free(parser) → bool */
        Value parser_val = vm_pop(vm);
        int handle = (int)as_number(parser_val);
        if (handle > 0 && handle < (int)(sizeof(vm->ts_parsers) / sizeof(vm->ts_parsers[0])) &&
            vm->ts_parsers[handle].active) {
            VmTsParserFreeFn free_fn = (VmTsParserFreeFn)(uintptr_t)
                vm_ts_symbol("eshkol_ts_parser_free");
            if (!free_fn) { vm_push(vm, BOOL_VAL(0)); break; }
            free_fn(handle);
            memset(&vm->ts_parsers[handle], 0, sizeof(vm->ts_parsers[handle]));
            vm_push(vm, BOOL_VAL(1));
            break;
        }
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2055: { /* ts-parse(parser, source) → tree or #f */
        Value source_val = vm_pop(vm), parser_val = vm_pop(vm);
        int parser = (int)as_number(parser_val);
        VmString* source = vm_value_as_string(vm, source_val);
        if (parser > 0 && parser < (int)(sizeof(vm->ts_parsers) / sizeof(vm->ts_parsers[0])) &&
            vm->ts_parsers[parser].active && source && source->data) {
            VmTsParseFn parse_fn = (VmTsParseFn)(uintptr_t)
                vm_ts_symbol("eshkol_ts_parse");
            if (parse_fn && source->byte_len <= INT32_MAX) {
                int64_t tree = parse_fn(parser, source->data, (int32_t)source->byte_len);
                if (tree > 0 &&
                    tree < (int64_t)(sizeof(vm->ts_trees) / sizeof(vm->ts_trees[0]))) {
                    vm->ts_trees[tree].active = 1;
                    vm_push(vm, INT_VAL(tree));
                    break;
                }
                if (tree > 0) {
                    VmTsTreeFreeFn free_fn = (VmTsTreeFreeFn)(uintptr_t)
                        vm_ts_symbol("eshkol_ts_tree_free");
                    if (free_fn) free_fn(tree);
                }
            }
        }
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2056: { /* ts-tree-free(tree) → bool */
        Value tree_val = vm_pop(vm);
        int handle = (int)as_number(tree_val);
        if (handle > 0 && handle < (int)(sizeof(vm->ts_trees) / sizeof(vm->ts_trees[0])) &&
            vm->ts_trees[handle].active) {
            VmTsTreeFreeFn free_fn = (VmTsTreeFreeFn)(uintptr_t)
                vm_ts_symbol("eshkol_ts_tree_free");
            if (!free_fn) { vm_push(vm, BOOL_VAL(0)); break; }
            free_fn(handle);
            for (int i = 0; i < (int)(sizeof(vm->ts_nodes) / sizeof(vm->ts_nodes[0])); i++) {
                if (vm->ts_nodes[i].active && vm->ts_nodes[i].tree == handle)
                    memset(&vm->ts_nodes[i], 0, sizeof(vm->ts_nodes[i]));
            }
            memset(&vm->ts_trees[handle], 0, sizeof(vm->ts_trees[handle]));
            vm_push(vm, BOOL_VAL(1));
            break;
        }
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2057: { /* ts-node-type(node) → string or #f */
        Value node_val = vm_pop(vm);
        int node = (int)as_number(node_val);
        if (node > 0 && node < (int)(sizeof(vm->ts_nodes) / sizeof(vm->ts_nodes[0])) &&
            vm->ts_nodes[node].active) {
            vm_push(vm, vm_string_value(vm, vm->ts_nodes[node].type, -1));
            break;
        }
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2058: { /* ts-node-text(node, source) → string or #f */
        Value source_val = vm_pop(vm), node_val = vm_pop(vm);
        (void)source_val;
        int node = (int)as_number(node_val);
        if (node > 0 && node < (int)(sizeof(vm->ts_nodes) / sizeof(vm->ts_nodes[0])) &&
            vm->ts_nodes[node].active) {
            VmTsNodeTextFn text_fn = (VmTsNodeTextFn)(uintptr_t)
                vm_ts_symbol("eshkol_ts_node_text");
            uint32_t start = vm->ts_nodes[node].start;
            uint32_t end = vm->ts_nodes[node].end;
            int tree = vm->ts_nodes[node].tree;
            int32_t required = text_fn ? text_fn(tree, start, end, NULL, 0) : -1;
            if (required >= 0) {
                char* buffer = (char*)malloc((size_t)required + 1u);
                if (buffer && text_fn(tree, start, end, buffer, required + 1) == required) {
                    Value text_value = vm_string_value(vm, buffer, required);
                    free(buffer);
                    vm_push(vm, text_value);
                    break;
                }
                free(buffer);
            }
        }
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2059: { /* ts-node-children(node) → list or #f */
        Value node_val = vm_pop(vm);
        vm_push(vm, vm_ts_node_children_value(vm, (int)as_number(node_val)));
        break;
    }
    case 2060: { /* ts-query-new(language, pattern) → query or #f */
        Value pattern_val = vm_pop(vm), language_val = vm_pop(vm);
        VmString* language = vm_value_as_string(vm, language_val);
        VmString* pattern = vm_value_as_string(vm, pattern_val);
        VmTsQueryNewFn new_fn = (VmTsQueryNewFn)(uintptr_t)
            vm_ts_symbol("eshkol_ts_query_new");
        char* language_copy = vm_ts_string_copy(language);
        char* pattern_copy = vm_ts_string_copy(pattern);
        if (new_fn && language_copy && pattern_copy) {
            int64_t query = new_fn(language_copy, pattern_copy);
            free(language_copy);
            free(pattern_copy);
            if (query > 0 &&
                query < (int64_t)(sizeof(vm->ts_queries) / sizeof(vm->ts_queries[0]))) {
                vm->ts_queries[query].active = 1;
                vm_push(vm, INT_VAL(query));
                break;
            }
            if (query > 0) {
                VmTsQueryFreeFn free_fn = (VmTsQueryFreeFn)(uintptr_t)
                    vm_ts_symbol("eshkol_ts_query_free");
                if (free_fn) free_fn(query);
            }
        } else {
            free(language_copy);
            free(pattern_copy);
        }
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2061: { /* ts-query-matches(query, tree, source) → list or #f */
        Value source_val = vm_pop(vm), tree_val = vm_pop(vm), query_val = vm_pop(vm);
        (void)source_val;
        int query = (int)as_number(query_val);
        int tree = (int)as_number(tree_val);
        vm_push(vm, vm_ts_query_matches_value(vm, query, tree));
        break;
    }
    case 2062: { /* ts-query-free(query) → bool */
        Value query_val = vm_pop(vm);
        int handle = (int)as_number(query_val);
        if (handle > 0 && handle < (int)(sizeof(vm->ts_queries) / sizeof(vm->ts_queries[0])) &&
            vm->ts_queries[handle].active) {
            VmTsQueryFreeFn free_fn = (VmTsQueryFreeFn)(uintptr_t)
                vm_ts_symbol("eshkol_ts_query_free");
            if (!free_fn) { vm_push(vm, BOOL_VAL(0)); break; }
            free_fn(handle);
            memset(&vm->ts_queries[handle], 0, sizeof(vm->ts_queries[handle]));
            vm_push(vm, BOOL_VAL(1));
            break;
        }
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2063: { /* ts-available() → bool */
        VmTsAvailableFn available_fn = (VmTsAvailableFn)(uintptr_t)
            vm_ts_symbol("eshkol_ts_available");
        vm_push(vm, BOOL_VAL(available_fn && available_fn() != 0));
        break;
    }
    case 2064: { /* ts-tree-root(tree) → node or #f */
        Value tree_val = vm_pop(vm);
        int tree = (int)as_number(tree_val);
        if (tree > 0 && tree < (int)(sizeof(vm->ts_trees) / sizeof(vm->ts_trees[0])) &&
            vm->ts_trees[tree].active) {
            VmTsTreeRootFn root_fn = (VmTsTreeRootFn)(uintptr_t)
                vm_ts_symbol("eshkol_ts_tree_root");
            int32_t required = root_fn ? root_fn(tree, NULL, 0) : -1;
            if (required >= 0) {
                char* record = (char*)malloc((size_t)required + 1u);
                if (record && root_fn(tree, record, required + 1) == required) {
                    int node = vm_ts_store_node(vm, tree, 1, record);
                    free(record);
                    if (node > 0) {
                        vm_push(vm, INT_VAL((int64_t)node));
                        break;
                    }
                } else {
                    free(record);
                }
            }
        }
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2065: { /* http-set-proxy(url) → bool */
        Value proxy_val = vm_pop(vm);
        VmString* proxy = vm_value_as_string(vm, proxy_val);
        if (proxy && proxy->data && proxy->byte_len < (int64_t)sizeof(vm->http_proxy_url)) {
            snprintf(vm->http_proxy_url, sizeof(vm->http_proxy_url),
                     "%.*s", (int)proxy->byte_len, proxy->data);
            vm_push(vm, BOOL_VAL(1));
            break;
        }
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2066: { /* http-set-tls-client-cert(cert, key, ca) → bool */
        Value ca_val = vm_pop(vm), key_val = vm_pop(vm), cert_val = vm_pop(vm);
        VmString* cert = vm_value_as_string(vm, cert_val);
        VmString* key = vm_value_as_string(vm, key_val);
        VmString* ca = vm_value_as_string(vm, ca_val);
        if (cert && key && ca && cert->data && key->data && ca->data &&
            cert->byte_len < (int64_t)sizeof(vm->http_tls_cert) &&
            key->byte_len < (int64_t)sizeof(vm->http_tls_key) &&
            ca->byte_len < (int64_t)sizeof(vm->http_tls_ca)) {
            snprintf(vm->http_tls_cert, sizeof(vm->http_tls_cert),
                     "%.*s", (int)cert->byte_len, cert->data);
            snprintf(vm->http_tls_key, sizeof(vm->http_tls_key),
                     "%.*s", (int)key->byte_len, key->data);
            snprintf(vm->http_tls_ca, sizeof(vm->http_tls_ca),
                     "%.*s", (int)ca->byte_len, ca->data);
            vm_push(vm, BOOL_VAL(1));
            break;
        }
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2067: { /* display-error(str) → bool */
        Value str_val = vm_pop(vm);
        VmString* str = vm_value_as_string(vm, str_val);
        if (str && str->data) {
            fwrite(str->data, 1, (size_t)str->byte_len, stderr);
            fflush(stderr);
            vm_push(vm, BOOL_VAL(1));
            break;
        }
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2068: { /* open-binary-input-file(path) → port or #f */
        Value path_val = vm_pop(vm);
        VmString* path = vm_value_as_string(vm, path_val);
        VmPort* p = (path && path->data)
            ? vm_port_open_binary_input_file(&vm->heap.regions, path->data)
            : NULL;
        if (p) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_PORT, VAL_PORT, p); break; }
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2069: { /* open-binary-output-file(path) → port or #f */
        Value path_val = vm_pop(vm);
        VmString* path = vm_value_as_string(vm, path_val);
        VmPort* p = (path && path->data)
            ? vm_port_open_binary_output_file(&vm->heap.regions, path->data)
            : NULL;
        if (p) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_PORT, VAL_PORT, p); break; }
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 2070: { /* read-u8(port) → byte or eof */
        Value port_val = vm_pop(vm);
        VmPort* p = vm_value_as_port(vm, port_val);
        int b = vm_port_read_u8(p);
        vm_push(vm, b < 0 ? NIL_VAL : INT_VAL(b));
        break;
    }
    case 2071: { /* write-u8(byte, port) → void */
        Value port_val = vm_pop(vm), byte_val = vm_pop(vm);
        VmPort* p = vm_value_as_port(vm, port_val);
        vm_port_write_u8(p, (int)as_number(byte_val));
        vm_push(vm, NIL_VAL);
        break;
    }
    case 2072: { /* read-bytevector(k, port) → bytevector or eof */
        Value port_val = vm_pop(vm), k_val = vm_pop(vm);
        VmPort* p = vm_value_as_port(vm, port_val);
        int k = (int)as_number(k_val);
        if (!p || k < 0) { vm_push(vm, NIL_VAL); break; }
        VmBytevector* bv = vm_bv_make(&vm->heap.regions, k, 0);
        if (!bv) { vm_push(vm, NIL_VAL); break; }
        int n = 0;
        for (; n < k; n++) {
            int b = vm_port_read_u8(p);
            if (b < 0) break;
            bv->data[n] = (uint8_t)b;
        }
        if (n == 0 && k > 0) { vm_push(vm, NIL_VAL); break; }
        bv->len = n;
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_BYTEVECTOR, VAL_BYTEVECTOR, bv);
        break;
    }
    case 2073: { /* write-bytevector(bv, port) → void */
        Value port_val = vm_pop(vm), bv_val = vm_pop(vm);
        VmPort* p = vm_value_as_port(vm, port_val);
        VmBytevector* bv = vm_value_as_bytevector(vm, bv_val);
        if (p && bv) vm_port_write_bytevector(p, (const char*)bv->data, bv->len);
        vm_push(vm, NIL_VAL);
        break;
    }
    case 2014: { /* json-get-in(obj, path, default) → value */
        Value default_val = vm_pop(vm), path_val = vm_pop(vm), obj_val = vm_pop(vm);
        vm_push(vm, vm_json_get_in_value(vm, obj_val, path_val, default_val));
        break;
    }
    case 2015: { /* json-stringify-pretty(obj, indent) → string or #f */
        Value indent_val = vm_pop(vm), obj_val = vm_pop(vm);
        vm_push(vm, vm_json_stringify_pretty_value(vm, obj_val, (int)as_number(indent_val)));
        break;
    }
    case 2016: { /* json-merge(a, b) → merged alist */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        vm_push(vm, vm_json_merge_value(vm, a_val, b_val));
        break;
    }
    case 2017: { /* compression-available() → bool */
        vm_push(vm, BOOL_VAL(vm_compression_available()));
        break;
    }
    case 2018: { /* deflate(data) → bytevector or #f */
        Value input_val = vm_pop(vm);
        vm_push(vm, vm_compression_call(vm, input_val,
                                        vm_compression_symbol("eshkol_deflate_alloc")));
        break;
    }
    case 2019: { /* inflate(data) → bytevector or #f */
        Value input_val = vm_pop(vm);
        vm_push(vm, vm_compression_call(vm, input_val,
                                        vm_compression_symbol("eshkol_inflate_alloc")));
        break;
    }
    case 2020: { /* gzip(data) → bytevector or #f */
        Value input_val = vm_pop(vm);
        vm_push(vm, vm_compression_call(vm, input_val,
                                        vm_compression_symbol("eshkol_gzip_alloc")));
        break;
    }
    case 2021: { /* gunzip(data) → bytevector or #f */
        Value input_val = vm_pop(vm);
        vm_push(vm, vm_compression_call(vm, input_val,
                                        vm_compression_symbol("eshkol_gunzip_alloc")));
        break;
    }
    case 1956: { /* string-ends-with?(str, suffix) → bool */
        Value suffix_val = vm_pop(vm);
        Value str_val = vm_pop(vm);
        vm_push(vm, BOOL_VAL(vm_string_ends_with_bytes(vm_value_as_string(vm, str_val),
                                                       vm_value_as_string_or_char(vm, suffix_val))));
        break;
    }
    case 1957: { /* string-index-of(str, substr, start) → integer or #f */
        Value start_val = vm_pop(vm);
        Value sub_val = vm_pop(vm);
        Value str_val = vm_pop(vm);
        vm_push(vm, vm_string_index_of_value(vm_value_as_string(vm, str_val),
                                             vm_value_as_string_or_char(vm, sub_val),
                                             (int64_t)as_number(start_val)));
        break;
    }
    case 1958: { /* string-pad-left(str, width, ch) → string */
        Value ch_val = vm_pop(vm);
        Value width_val = vm_pop(vm);
        Value str_val = vm_pop(vm);
        vm_push(vm, vm_string_pad_value(vm, vm_value_as_string(vm, str_val),
                                        (int64_t)as_number(width_val),
                                        (int)as_number(ch_val), 1));
        break;
    }
    case 1959: { /* string-pad-right(str, width, ch) → string */
        Value ch_val = vm_pop(vm);
        Value width_val = vm_pop(vm);
        Value str_val = vm_pop(vm);
        vm_push(vm, vm_string_pad_value(vm, vm_value_as_string(vm, str_val),
                                        (int64_t)as_number(width_val),
                                        (int)as_number(ch_val), 0));
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Parallel (620-628) — pool-backed scheduling with serialized VM closure calls
     * ══════════════════════════════════════════════════════════════════════ */
    case 620: { /* parallel-map(fn, list) — parallel via thread pool when available */
        Value list = vm_pop(vm), fn = vm_pop(vm);

        /* Count elements */
        int n = 0;
        Value cur = list;
        while (cur.type == VAL_PAIR && n < 4096) {
            n++;
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
        }

        if (n == 0) { vm_push(vm, NIL_VAL); break; }

        /* Extract elements into array */
        Value* elems = (Value*)vm_alloc(&vm->heap.regions, (size_t)n * sizeof(Value));
        if (!elems) { vm_push(vm, NIL_VAL); break; }
        cur = list;
        for (int i = 0; i < n; i++) {
            elems[i] = vm->heap.objects[cur.as.ptr]->cons.car;
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
        }

        /* Allocate results array */
        Value* results = (Value*)vm_alloc(&vm->heap.regions, (size_t)n * sizeof(Value));
        if (!results) { vm_push(vm, NIL_VAL); break; }

        /* Use thread pool if available and list is large enough.
         * WASM target has no pthread → vm_parallel.c is excluded; always sequential. */
#ifndef ESHKOL_VM_WASM
        VmThreadPool* pool = vm_parallel_ensure_pool();
        if (pool && n >= 4) {
            VmParMapTask* tasks = (VmParMapTask*)vm_alloc(&vm->heap.regions,
                                      (size_t)n * sizeof(VmParMapTask));
            if (tasks) {
                for (int i = 0; i < n; i++) {
                    tasks[i].main_vm = vm;
                    tasks[i].closure = fn;
                    tasks[i].input = elems[i];
                    tasks[i].output = NIL_VAL;
                }
                for (int i = 0; i < n; i++) {
                    vm_pool_submit(pool, vm_parmap_task_fn, &tasks[i], NULL);
                }
                vm_pool_wait_all(pool);
                for (int i = 0; i < n; i++) {
                    results[i] = tasks[i].output;
                }
            } else {
                /* Fallback: sequential */
                for (int i = 0; i < n; i++) {
                    results[i] = vm_call_closure_from_native(vm, fn, &elems[i], 1);
                }
            }
        } else
#endif
        {
            /* Sequential: small list, no pool, or WASM target */
            for (int i = 0; i < n; i++) {
                results[i] = vm_call_closure_from_native(vm, fn, &elems[i], 1);
            }
        }

        /* Build result list from array (reverse order → cons → correct order) */
        Value result = NIL_VAL;
        for (int i = n - 1; i >= 0; i--) {
            int32_t p = heap_alloc(&vm->heap);
            if (p < 0) break;
            vm->heap.objects[p]->type = HEAP_CONS;
            vm->heap.objects[p]->cons.car = results[i];
            vm->heap.objects[p]->cons.cdr = result;
            result = PAIR_VAL(p);
        }
        vm_push(vm, result);
        break;
    }
    case 621: { /* parallel-filter(pred, list) — parallel predicate evaluation */
        Value list = vm_pop(vm), pred = vm_pop(vm);

        /* Count and extract elements */
        int n = 0;
        Value cur = list;
        while (cur.type == VAL_PAIR && n < 4096) { n++; cur = vm->heap.objects[cur.as.ptr]->cons.cdr; }
        if (n == 0) { vm_push(vm, NIL_VAL); break; }

        Value* elems = (Value*)vm_alloc(&vm->heap.regions, (size_t)n * sizeof(Value));
        Value* preds = (Value*)vm_alloc(&vm->heap.regions, (size_t)n * sizeof(Value));
        if (!elems || !preds) { vm_push(vm, NIL_VAL); break; }
        cur = list;
        for (int i = 0; i < n; i++) {
            elems[i] = vm->heap.objects[cur.as.ptr]->cons.car;
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
        }

        /* Evaluate predicates (parallel via pool if available; WASM is sequential). */
#ifndef ESHKOL_VM_WASM
        VmThreadPool* pool = vm_parallel_ensure_pool();
        if (pool && n >= 4) {
            VmParMapTask* tasks = (VmParMapTask*)vm_alloc(&vm->heap.regions,
                                      (size_t)n * sizeof(VmParMapTask));
            if (tasks) {
                for (int i = 0; i < n; i++) {
                    tasks[i].main_vm = vm; tasks[i].closure = pred;
                    tasks[i].input = elems[i]; tasks[i].output = NIL_VAL;
                }
                for (int i = 0; i < n; i++)
                    vm_pool_submit(pool, vm_parmap_task_fn, &tasks[i], NULL);
                vm_pool_wait_all(pool);
                for (int i = 0; i < n; i++) preds[i] = tasks[i].output;
            } else {
                for (int i = 0; i < n; i++)
                    preds[i] = vm_call_closure_from_native(vm, pred, &elems[i], 1);
            }
        } else
#endif
        {
            for (int i = 0; i < n; i++)
                preds[i] = vm_call_closure_from_native(vm, pred, &elems[i], 1);
        }

        /* Build filtered list (correct order) */
        Value result = NIL_VAL;
        for (int i = n - 1; i >= 0; i--) {
            if (is_truthy(preds[i])) {
                int32_t p = heap_alloc(&vm->heap);
                if (p < 0) break;
                vm->heap.objects[p]->type = HEAP_CONS;
                vm->heap.objects[p]->cons.car = elems[i];
                vm->heap.objects[p]->cons.cdr = result;
                result = PAIR_VAL(p);
            }
        }
        vm_push(vm, result);
        break;
    }
    case 622: { /* parallel-fold(fn, init, list) — sequential via closure bridge */
        Value list = vm_pop(vm), init = vm_pop(vm), fn = vm_pop(vm);
        Value acc = init;
        Value cur = list;
        while (cur.type == VAL_PAIR) {
            Value elem = vm->heap.objects[cur.as.ptr]->cons.car;
            Value args[2] = {acc, elem};
            acc = vm_call_closure_from_native(vm, fn, args, 2);
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
        }
        vm_push(vm, acc);
        break;
    }
    case 623: { /* parallel-for-each(fn, list) — parallel side-effect execution */
        Value list = vm_pop(vm), fn = vm_pop(vm);

        int n = 0;
        Value cur = list;
        while (cur.type == VAL_PAIR && n < 4096) { n++; cur = vm->heap.objects[cur.as.ptr]->cons.cdr; }
        if (n == 0) { vm_push(vm, NIL_VAL); break; }

        Value* elems = (Value*)vm_alloc(&vm->heap.regions, (size_t)n * sizeof(Value));
        if (!elems) { vm_push(vm, NIL_VAL); break; }
        cur = list;
        for (int i = 0; i < n; i++) {
            elems[i] = vm->heap.objects[cur.as.ptr]->cons.car;
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
        }

        /* parallel-for-each: parallel via pool if available; WASM is sequential. */
#ifndef ESHKOL_VM_WASM
        VmThreadPool* pool = vm_parallel_ensure_pool();
        if (pool && n >= 4) {
            VmParMapTask* tasks = (VmParMapTask*)vm_alloc(&vm->heap.regions,
                                      (size_t)n * sizeof(VmParMapTask));
            if (tasks) {
                for (int i = 0; i < n; i++) {
                    tasks[i].main_vm = vm; tasks[i].closure = fn;
                    tasks[i].input = elems[i]; tasks[i].output = NIL_VAL;
                }
                for (int i = 0; i < n; i++)
                    vm_pool_submit(pool, vm_parmap_task_fn, &tasks[i], NULL);
                vm_pool_wait_all(pool);
            } else {
                for (int i = 0; i < n; i++)
                    vm_call_closure_from_native(vm, fn, &elems[i], 1);
            }
        } else
#endif
        {
            for (int i = 0; i < n; i++)
                vm_call_closure_from_native(vm, fn, &elems[i], 1);
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 624: { /* parallel-execute(thunks-list) */
        Value thunks = vm_pop(vm);

        if (thunks.type == VAL_CLOSURE) {
            Value result = vm_call_closure_from_native(vm, thunks, NULL, 0);
            vm_push(vm, result);
            break;
        }

        int n = 0;
        Value cur = thunks;
        while (cur.type == VAL_PAIR && n < 4096) {
            n++;
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
        }
        if (n == 0) { vm_push(vm, NIL_VAL); break; }

        Value* closures = (Value*)vm_alloc(&vm->heap.regions, (size_t)n * sizeof(Value));
        Value* results = (Value*)vm_alloc(&vm->heap.regions, (size_t)n * sizeof(Value));
        if (!closures || !results) { vm_push(vm, NIL_VAL); break; }

        cur = thunks;
        for (int i = 0; i < n; i++) {
            closures[i] = vm->heap.objects[cur.as.ptr]->cons.car;
            results[i] = NIL_VAL;
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
        }

#ifndef ESHKOL_VM_WASM
        VmThreadPool* pool = vm_parallel_ensure_pool();
        if (pool && n >= 2) {
            VmParThunkTask* tasks = (VmParThunkTask*)vm_alloc(&vm->heap.regions,
                                        (size_t)n * sizeof(VmParThunkTask));
            if (tasks) {
                for (int i = 0; i < n; i++) {
                    tasks[i].main_vm = vm;
                    tasks[i].closure = closures[i];
                    tasks[i].output = NIL_VAL;
                }
                for (int i = 0; i < n; i++)
                    vm_pool_submit(pool, vm_parthunk_task_fn, &tasks[i], NULL);
                vm_pool_wait_all(pool);
                for (int i = 0; i < n; i++) results[i] = tasks[i].output;
            } else {
                for (int i = 0; i < n; i++)
                    results[i] = vm_call_closure_from_native(vm, closures[i], NULL, 0);
            }
        } else
#endif
        {
            for (int i = 0; i < n; i++)
                results[i] = vm_call_closure_from_native(vm, closures[i], NULL, 0);
        }

        Value result = NIL_VAL;
        for (int i = n - 1; i >= 0; i--) {
            int32_t p = heap_alloc(&vm->heap);
            if (p < 0) break;
            vm->heap.objects[p]->type = HEAP_CONS;
            vm->heap.objects[p]->cons.car = results[i];
            vm->heap.objects[p]->cons.cdr = result;
            result = PAIR_VAL(p);
        }
        vm_push(vm, result);
        break;
    }
    case 625: { /* future(thunk-or-value) — async handle when pool is available */
        Value thunk_or_value = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        VmFuture* fut = vm_future_create(vm, thunk_or_value);
        if (fut) {
            int32_t fut_ptr = heap_alloc(&vm->heap);
            if (fut_ptr < 0) {
                if (thunk_or_value.type == VAL_CLOSURE)
                    thunk_or_value = vm_call_closure_from_native(vm, thunk_or_value, NULL, 0);
                vm_push(vm, thunk_or_value);
                break;
            }
            vm->heap.objects[fut_ptr]->type = HEAP_FUTURE;
            vm->heap.objects[fut_ptr]->opaque.ptr = fut;
            Value fut_value = (Value){.type = VAL_FUTURE, .as.ptr = fut_ptr};

            VmThreadPool* pool = vm_parallel_ensure_pool();
            if (pool && thunk_or_value.type == VAL_CLOSURE &&
                vm_pool_submit(pool, vm_future_task_fn, fut, NULL) == 0) {
                vm_push(vm, fut_value);
                break;
            }
            Value result = thunk_or_value;
            if (thunk_or_value.type == VAL_CLOSURE)
                result = vm_call_closure_from_native(vm, thunk_or_value, NULL, 0);
            vm_future_mark_ready(fut, result);
            vm_push(vm, fut_value);
            break;
        }
#endif
        if (thunk_or_value.type == VAL_CLOSURE)
            thunk_or_value = vm_call_closure_from_native(vm, thunk_or_value, NULL, 0);
        vm_push(vm, thunk_or_value);
        break;
    }
    case 626: { /* force-future */
        Value fut = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (fut.type == VAL_FUTURE && fut.as.ptr >= 0 && fut.as.ptr < vm->heap.next_free &&
            vm->heap.objects[fut.as.ptr]->type == HEAP_FUTURE) {
            VmFuture* handle = (VmFuture*)vm->heap.objects[fut.as.ptr]->opaque.ptr;
            vm_push(vm, vm_future_force(handle));
            break;
        }
#endif
        vm_push(vm, fut); /* compatibility: forcing a plain value returns it */
        break;
    }
    case 627: { /* future-ready? */
        Value fut = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (fut.type == VAL_FUTURE && fut.as.ptr >= 0 && fut.as.ptr < vm->heap.next_free &&
            vm->heap.objects[fut.as.ptr]->type == HEAP_FUTURE) {
            VmFuture* handle = (VmFuture*)vm->heap.objects[fut.as.ptr]->opaque.ptr;
            vm_push(vm, BOOL_VAL(vm_future_is_ready(handle)));
            break;
        }
#endif
        vm_push(vm, BOOL_VAL(1));
        break;
    }
    case 628: { /* thread-pool-info */
#ifndef ESHKOL_VM_WASM
        VmThreadPool* pool = vm_parallel_ensure_pool();
        vm_push(vm, INT_VAL(pool ? vm_pool_thread_count(pool) : 1));
#else
        vm_push(vm, INT_VAL(1));
#endif
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * MultiValue Operations (650-654)
     * ══════════════════════════════════════════════════════════════════════ */
    case 650: { /* pack-values: value... count -> opaque multi-value packet */
        Value count_value = vm_pop(vm);
        int count = (int)as_number(count_value);
        if (count < 0 || count > vm->sp) {
            fprintf(stderr, "ERROR: invalid multiple-values count %d\n", count);
            vm->error = 1;
            break;
        }
        int32_t ptr = heap_alloc(&vm->heap);
        if (ptr < 0) { vm->error = 1; break; }
        VmVector* values = (VmVector*)vm_alloc(&vm->heap.regions, sizeof(VmVector));
        if (!values) { vm->error = 1; break; }
        values->len = count;
        values->cap = count;
        values->items = (Value*)vm_alloc(&vm->heap.regions,
                                         (size_t)(count > 0 ? count : 1) * sizeof(Value));
        if (!values->items) { vm->error = 1; break; }
        for (int i = count - 1; i >= 0; i--) values->items[i] = vm_pop(vm);
        vm->heap.objects[ptr]->type = HEAP_MULTI_VALUE;
        vm->heap.objects[ptr]->opaque.ptr = values;
        vm_push(vm, (Value){.type = VAL_MULTI_VALUE, .as.ptr = ptr});
        break;
    }
    case 651: { /* multi-value-ref(mv, idx) — extract from multi-value container */
        Value idx = vm_pop(vm), mv = vm_pop(vm);
        if (mv.type == VAL_MULTI_VALUE && is_heap_type(vm, mv, HEAP_MULTI_VALUE)) {
            VmVector* mvobj = (VmVector*)vm->heap.objects[mv.as.ptr]->opaque.ptr;
            int i = (int)as_number(idx);
            if (mvobj && i >= 0 && i < mvobj->len) {
                vm_push(vm, mvobj->items[i]);
                break;
            }
        }
        /* Single value: index 0 returns the value itself */
        if ((int)as_number(idx) == 0) { vm_push(vm, mv); }
        else { vm_push(vm, NIL_VAL); }
        break;
    }
    case 652: { /* multi-value-count(mv) — number of values in container */
        Value mv = vm_pop(vm);
        if (mv.type == VAL_MULTI_VALUE && is_heap_type(vm, mv, HEAP_MULTI_VALUE)) {
            VmVector* mvobj = (VmVector*)vm->heap.objects[mv.as.ptr]->opaque.ptr;
            vm_push(vm, INT_VAL(mvobj ? mvobj->len : 0));
        } else {
            vm_push(vm, INT_VAL(1)); /* single value counts as 1 */
        }
        break;
    }
    case 653: { /* multi-value? — check if value is a multi-value container */
        Value v = vm_pop(vm);
        int is_mv = (v.type == VAL_MULTI_VALUE &&
                     is_heap_type(vm, v, HEAP_MULTI_VALUE));
        vm_push(vm, BOOL_VAL(is_mv));
        break;
    }
    case 654: { /* call-with-values(producer, consumer) — via closure bridge */
        Value consumer = vm_pop(vm), producer = vm_pop(vm);
        /* Call producer with 0 args */
        Value produced = vm_call_closure_from_native(vm, producer, NULL, 0);
        Value result;
        if (produced.type == VAL_MULTI_VALUE &&
            is_heap_type(vm, produced, HEAP_MULTI_VALUE)) {
            VmVector* values = (VmVector*)vm->heap.objects[produced.as.ptr]->opaque.ptr;
            result = vm_call_closure_from_native(vm, consumer,
                                                 values ? values->items : NULL,
                                                 values ? values->len : 0);
        } else {
            result = vm_call_closure_from_native(vm, consumer, &produced, 1);
        }
        vm_push(vm, result);
        break;
    }
    case 655: { /* multi-value-rest: packet start -> proper list of remaining values */
        Value start_value = vm_pop(vm), packet = vm_pop(vm);
        int start = (int)as_number(start_value);
        int count = 1;
        Value* items = &packet;
        if (packet.type == VAL_MULTI_VALUE &&
            is_heap_type(vm, packet, HEAP_MULTI_VALUE)) {
            VmVector* values = (VmVector*)vm->heap.objects[packet.as.ptr]->opaque.ptr;
            count = values ? values->len : 0;
            items = values ? values->items : NULL;
        }
        if (start < 0 || start > count) {
            fprintf(stderr, "ERROR: multiple-values rest index %d out of range [0,%d]\n",
                    start, count);
            vm->error = 1;
            break;
        }
        Value result = NIL_VAL;
        for (int i = count - 1; i >= start; i--)
            result = vm_cons_value(vm, items[i], result);
        vm_push(vm, result);
        break;
    }
    case 656: { /* validate-values-arity: packet min max(-1 means unbounded) */
        Value max_value = vm_pop(vm), min_value = vm_pop(vm), packet = vm_pop(vm);
        int min_count = (int)as_number(min_value);
        int max_count = (int)as_number(max_value);
        int count = 1;
        if (packet.type == VAL_MULTI_VALUE &&
            is_heap_type(vm, packet, HEAP_MULTI_VALUE)) {
            VmVector* values = (VmVector*)vm->heap.objects[packet.as.ptr]->opaque.ptr;
            count = values ? values->len : 0;
        }
        if (count < min_count || (max_count >= 0 && count > max_count)) {
            if (max_count < 0)
                fprintf(stderr, "ERROR: expected at least %d values, received %d\n",
                        min_count, count);
            else
                fprintf(stderr, "ERROR: expected %d values, received %d\n",
                        min_count, count);
            vm->error = 1;
            break;
        }
        vm_push(vm, packet);
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Hash Table Operations (660-670)
     * ══════════════════════════════════════════════════════════════════════ */
    case 660: { /* make-hash-table */
        VmHashTable* ht = vm_ht_make(&vm->heap.regions);
        if (!ht) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_HASH, VAL_HASH, ht);
        break;
    }
    case 661: { /* hash-ref(ht, key, default) */
        Value dflt = vm_pop(vm), key = vm_pop(vm), ht_val = vm_pop(vm);
        if (is_heap_type(vm, ht_val, HEAP_HASH)) {
            VmHashTable* ht = (VmHashTable*)vm->heap.objects[ht_val.as.ptr]->opaque.ptr;
            void* result = vm_ht_ref(ht, (void*)(uintptr_t)key.as.i, (void*)(uintptr_t)dflt.as.i);
            vm_push(vm, INT_VAL((int64_t)(intptr_t)result));
        } else vm_push(vm, dflt);
        break;
    }
    case 662: { /* hash-set!(ht, key, value) */
        Value val = vm_pop(vm), key = vm_pop(vm), ht_val = vm_pop(vm);
        if (is_heap_type(vm, ht_val, HEAP_HASH)) {
            VmHashTable* ht = (VmHashTable*)vm->heap.objects[ht_val.as.ptr]->opaque.ptr;
            vm_ht_set(&vm->heap.regions, ht, (void*)(uintptr_t)key.as.i, (void*)(uintptr_t)val.as.i);
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 663: { /* hash-delete!(ht, key) */
        Value key = vm_pop(vm), ht_val = vm_pop(vm);
        if (is_heap_type(vm, ht_val, HEAP_HASH)) {
            VmHashTable* ht = (VmHashTable*)vm->heap.objects[ht_val.as.ptr]->opaque.ptr;
            vm_ht_remove(ht, (void*)(uintptr_t)key.as.i);
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 664: { /* hash-has-key?(ht, key) */
        Value key = vm_pop(vm), ht_val = vm_pop(vm);
        if (is_heap_type(vm, ht_val, HEAP_HASH)) {
            VmHashTable* ht = (VmHashTable*)vm->heap.objects[ht_val.as.ptr]->opaque.ptr;
            vm_push(vm, BOOL_VAL(vm_ht_has_key(ht, (void*)(uintptr_t)key.as.i)));
        } else vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 665: { /* hash-keys */
        Value ht_val = vm_pop(vm);
        if (is_heap_type(vm, ht_val, HEAP_HASH)) {
            VmHashTable* ht = (VmHashTable*)vm->heap.objects[ht_val.as.ptr]->opaque.ptr;
            if (ht) {
                Value result = NIL_VAL;
                for (int i = ht->capacity - 1; i >= 0; i--) {
                    if (ht->keys[i]) {
                        int32_t p = heap_alloc(&vm->heap);
                        if (p < 0) break;
                        vm->heap.objects[p]->type = HEAP_CONS;
                        vm->heap.objects[p]->cons.car = *(Value*)ht->keys[i];
                        vm->heap.objects[p]->cons.cdr = result;
                        result = PAIR_VAL(p);
                    }
                }
                vm_push(vm, result);
                break;
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 666: { /* hash-values */
        Value ht_val = vm_pop(vm);
        if (is_heap_type(vm, ht_val, HEAP_HASH)) {
            VmHashTable* ht = (VmHashTable*)vm->heap.objects[ht_val.as.ptr]->opaque.ptr;
            if (ht) {
                Value result = NIL_VAL;
                for (int i = ht->capacity - 1; i >= 0; i--) {
                    if (ht->keys[i]) {
                        int32_t p = heap_alloc(&vm->heap);
                        if (p < 0) break;
                        vm->heap.objects[p]->type = HEAP_CONS;
                        vm->heap.objects[p]->cons.car = *(Value*)ht->values[i];
                        vm->heap.objects[p]->cons.cdr = result;
                        result = PAIR_VAL(p);
                    }
                }
                vm_push(vm, result);
                break;
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 667: { /* hash-count */
        Value ht_val = vm_pop(vm);
        if (is_heap_type(vm, ht_val, HEAP_HASH)) {
            VmHashTable* ht = (VmHashTable*)vm->heap.objects[ht_val.as.ptr]->opaque.ptr;
            vm_push(vm, INT_VAL(ht ? vm_ht_count(ht) : 0));
        } else vm_push(vm, INT_VAL(0));
        break;
    }
    case 668: { /* hash-table-copy(ht) — shallow copy of hash table */
        Value ht_val = vm_pop(vm);
        if (is_heap_type(vm, ht_val, HEAP_HASH)) {
            VmHashTable* ht = (VmHashTable*)vm->heap.objects[ht_val.as.ptr]->opaque.ptr;
            if (ht) {
                VmHashTable* copy = vm_ht_make(&vm->heap.regions);
                if (copy) {
                    VM_PUSH_HEAP_OPAQUE(vm, HEAP_HASH, VAL_HASH, copy);
                    break;
                }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 669: { /* hash-clear! */
        Value ht_val = vm_pop(vm);
        if (is_heap_type(vm, ht_val, HEAP_HASH)) {
            VmHashTable* ht = (VmHashTable*)vm->heap.objects[ht_val.as.ptr]->opaque.ptr;
            if (ht) vm_ht_clear(ht);
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 670: { /* hash-table? */
        Value v = vm_pop(vm);
        int is_ht = (is_heap_type(vm, v, HEAP_HASH));
        vm_push(vm, BOOL_VAL(is_ht));
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Bytevector Operations (680-689)
     * ══════════════════════════════════════════════════════════════════════ */
    case 680: { /* make-bytevector(n, fill) */
        Value fill = vm_pop(vm), n = vm_pop(vm);
        VmBytevector* bv = vm_bv_make(&vm->heap.regions, (int)as_number(n), (int)as_number(fill));
        if (!bv) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_BYTEVECTOR, VAL_BYTEVECTOR, bv);
        break;
    }
    case 681: { /* bytevector-length */
        Value bv_val = vm_pop(vm);
        if (is_heap_type(vm, bv_val, HEAP_BYTEVECTOR)) {
            VmBytevector* bv = (VmBytevector*)vm->heap.objects[bv_val.as.ptr]->opaque.ptr;
            vm_push(vm, INT_VAL(vm_bv_length(bv)));
        } else vm_push(vm, INT_VAL(0));
        break;
    }
    case 682: { /* bytevector-u8-ref(bv, k) */
        Value k = vm_pop(vm), bv_val = vm_pop(vm);
        if (is_heap_type(vm, bv_val, HEAP_BYTEVECTOR)) {
            VmBytevector* bv = (VmBytevector*)vm->heap.objects[bv_val.as.ptr]->opaque.ptr;
            vm_push(vm, INT_VAL(vm_bv_u8_ref(bv, (int)as_number(k))));
        } else vm_push(vm, INT_VAL(0));
        break;
    }
    case 683: { /* bytevector-u8-set!(bv, k, byte) */
        Value byte = vm_pop(vm), k = vm_pop(vm), bv_val = vm_pop(vm);
        if (is_heap_type(vm, bv_val, HEAP_BYTEVECTOR)) {
            VmBytevector* bv = (VmBytevector*)vm->heap.objects[bv_val.as.ptr]->opaque.ptr;
            vm_bv_u8_set(bv, (int)as_number(k), (int)as_number(byte));
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 684: { /* bytevector-append(a, b) */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        VmBytevector* a = (is_heap_type(vm, a_val, HEAP_BYTEVECTOR))
            ? (VmBytevector*)vm->heap.objects[a_val.as.ptr]->opaque.ptr : NULL;
        VmBytevector* b = (is_heap_type(vm, b_val, HEAP_BYTEVECTOR))
            ? (VmBytevector*)vm->heap.objects[b_val.as.ptr]->opaque.ptr : NULL;
        if (a && b) {
            VmBytevector* r = vm_bv_append(&vm->heap.regions, a, b);
            if (r) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_BYTEVECTOR, VAL_BYTEVECTOR, r); break; }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 685: { /* bytevector-copy!(to, at, from) — copy bytes from src into dst */
        Value from_val = vm_pop(vm), at_val = vm_pop(vm), to_val = vm_pop(vm);
        if (is_heap_type(vm, to_val, HEAP_BYTEVECTOR) &&
            is_heap_type(vm, from_val, HEAP_BYTEVECTOR)) {
            VmBytevector* to_bv = (VmBytevector*)vm->heap.objects[to_val.as.ptr]->opaque.ptr;
            VmBytevector* from_bv = (VmBytevector*)vm->heap.objects[from_val.as.ptr]->opaque.ptr;
            int at = (int)as_number(at_val);
            if (to_bv && from_bv && at >= 0) {
                int copy_len = from_bv->len;
                if (at + copy_len > to_bv->len) copy_len = to_bv->len - at;
                if (copy_len > 0) memcpy(to_bv->data + at, from_bv->data, (size_t)copy_len);
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 686: { /* bytevector? */
        Value v = vm_pop(vm);
        int is_bv = (is_heap_type(vm, v, HEAP_BYTEVECTOR));
        vm_push(vm, BOOL_VAL(is_bv));
        break;
    }
    case 687: { /* bytevector-copy(bv) */
        Value bv_val = vm_pop(vm);
        if (is_heap_type(vm, bv_val, HEAP_BYTEVECTOR)) {
            VmBytevector* bv = (VmBytevector*)vm->heap.objects[bv_val.as.ptr]->opaque.ptr;
            VmBytevector* r = vm_bv_copy(&vm->heap.regions, bv, 0, bv->len);
            if (r) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_BYTEVECTOR, VAL_BYTEVECTOR, r); break; }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 688: { /* utf8->string(bv) */
        Value bv_val = vm_pop(vm);
        if (is_heap_type(vm, bv_val, HEAP_BYTEVECTOR)) {
            VmBytevector* bv = (VmBytevector*)vm->heap.objects[bv_val.as.ptr]->opaque.ptr;
            if (bv) {
                VmString* s = vm_string_new(&vm->heap.regions, (const char*)bv->data, bv->len);
                if (s) {
                    int32_t ptr = heap_alloc(&vm->heap);
                    if (ptr >= 0) {
                        vm->heap.objects[ptr]->type = HEAP_STRING;
                        vm->heap.objects[ptr]->opaque.ptr = s;
                        vm_push(vm, (Value){.type = VAL_STRING, .as.ptr = ptr});
                        break;
                    }
                }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 689: { /* string->utf8(str) */
        Value str_val = vm_pop(vm);
        if (str_val.type == VAL_STRING && vm->heap.objects[str_val.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[str_val.as.ptr]->opaque.ptr;
            VmBytevector* bv = vm_bv_make(&vm->heap.regions, s->byte_len, 0);
            if (bv) {
                memcpy(bv->data, s->data, s->byte_len);
                int32_t ptr = heap_alloc(&vm->heap);
                if (ptr >= 0) {
                    vm->heap.objects[ptr]->type = HEAP_BYTEVECTOR;
                    vm->heap.objects[ptr]->opaque.ptr = bv;
                    vm_push(vm, (Value){.type = VAL_BYTEVECTOR, .as.ptr = ptr});
                    break;
                }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Parameter Operations (700-705)
     * ══════════════════════════════════════════════════════════════════════ */
    case 700: { /* make-parameter(default, converter) */
        Value conv = vm_pop(vm), dflt = vm_pop(vm);
        /* R7RS: the converter is applied to the default during creation. */
        if (conv.type == VAL_CLOSURE) {
            dflt = vm_call_closure_from_native(vm, conv, &dflt, 1);
        }
        VmParameter* p = vm_param_make(&vm->heap.regions, &dflt, &conv);
        if (!p) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_PARAMETER, VAL_PARAMETER_OBJ, p);
        break;
    }
    case 701: { /* parameter-ref */
        Value p_val = vm_pop(vm);
        VmParameter* p = vm_parameter_from_value(vm, p_val);
        if (p) {
            Value value;
            vm_param_ref(p, &value);
            vm_push(vm, value);
        } else vm_push(vm, NIL_VAL);
        break;
    }
    case 702: { /* parameterize-push(param, converted-value) */
        Value val = vm_pop(vm), p_val = vm_pop(vm);
        VmParameter* p = vm_parameter_from_value(vm, p_val);
        if (!p) {
            vm->error = 1;
        } else if (vm->n_winds >=
                   (int)(sizeof(vm->wind_stack) / sizeof(vm->wind_stack[0]))) {
            fprintf(stderr, "ERROR: parameterize dynamic stack overflow\n");
            vm->error = 1;
        } else {
            vm_param_push(&vm->heap.regions, p, &val);
            if (vm_record_parameter_binding(vm, p_val)) {
                vm->wind_stack[vm->n_winds].before = NIL_VAL;
                vm->wind_stack[vm->n_winds].after = p_val;
                vm->n_winds++;
            } else {
                vm_param_pop(p);
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 703: { /* parameterize-pop(param) */
        Value p_val = vm_pop(vm);
        if (vm->n_winds > 0 &&
            vm->wind_stack[vm->n_winds - 1].after.type == p_val.type &&
            vm->wind_stack[vm->n_winds - 1].after.as.ptr == p_val.as.ptr) {
            vm->n_winds--;
            vm_pop_parameter_binding(vm, p_val);
        } else {
            /* A malformed bytecode stream must not silently leave a binding
             * active; retain the old direct cleanup behavior and flag it. */
            vm->error = 1;
            vm_pop_parameter_binding(vm, p_val);
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 704: { /* parameter? */
        Value v = vm_pop(vm);
        int is_param = (is_heap_type(vm, v, HEAP_PARAMETER));
        vm_push(vm, BOOL_VAL(is_param));
        break;
    }
    case 705: { /* parameter-convert(param, value) */
        Value value = vm_pop(vm), p_val = vm_pop(vm);
        VmParameter* p = vm_parameter_from_value(vm, p_val);
        vm_push(vm, p ? vm_parameter_apply_converter(vm, p, value) : NIL_VAL);
        if (!p) vm->error = 1;
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Error Operations (710-714)
     * ══════════════════════════════════════════════════════════════════════ */
    case 710: { /* error(type, message) */
        Value msg = vm_pop(vm), type = vm_pop(vm);
        (void)type; (void)msg;
        VmError* e = vm_error_make(&vm->heap.regions, "error", "runtime error", NULL, 0);
        if (!e) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_ERROR, VAL_ERROR_OBJ, e);
        break;
    }
    case 711: { /* error-message */
        Value e_val = vm_pop(vm);
        if (is_heap_type(vm, e_val, HEAP_ERROR)) {
            VmError* e = (VmError*)vm->heap.objects[e_val.as.ptr]->opaque.ptr;
            VmString* s = vm_string_from_cstr(&vm->heap.regions, e ? e->message : "");
            if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 712: { /* error-type */
        Value e_val = vm_pop(vm);
        if (is_heap_type(vm, e_val, HEAP_ERROR)) {
            VmError* e = (VmError*)vm->heap.objects[e_val.as.ptr]->opaque.ptr;
            VmString* s = vm_string_from_cstr(&vm->heap.regions, e ? e->type : "");
            if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 713: { /* error-irritants — returns nil (no irritant list support in Value system) */
        Value e_val = vm_pop(vm); (void)e_val;
        vm_push(vm, NIL_VAL);
        break;
    }
    case 714: { /* error? */
        Value v = vm_pop(vm);
        int is_err = (is_heap_type(vm, v, HEAP_ERROR));
        vm_push(vm, BOOL_VAL(is_err));
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * High-level AD: gradient, jacobian, hessian, divergence, curl,
     *                laplacian, directional-derivative (750-756)
     *
     * Architecture:
     *   Forward-mode AD via dual numbers (VmDual). For multi-variable
     *   functions f(x1,...,xn), we perform N forward passes, each seeding
     *   one variable with tangent=1 and the rest with tangent=0. The
     *   tangent of the result gives the partial derivative ∂f/∂xi.
     *
     *   Hessian uses central difference of exact gradients:
     *     f''(x) ≈ (f'(x+h) - f'(x-h)) / (2h)
     *   where f' is computed exactly via dual numbers, giving O(h²)
     *   accuracy with exact first derivatives.
     *
     * Helper: vm_ad_make_dual_val — allocate a dual on the heap, return Value
     * Helper: vm_ad_extract_point — read point from scalar/list/tensor
     * Helper: vm_ad_partial — compute ∂f/∂xi at a multi-variable point
     * Helper: vm_ad_eval_component — call f, extract i-th output component
     * ══════════════════════════════════════════════════════════════════════ */

#define VM_AD_MAX_VARS 64

    /* --- Helper: create a dual number Value on the heap --- */
#define VM_AD_MAKE_DUAL(vm, primal_val, tangent_val, out_val) do { \
    VmDual* _d = vm_dual_new(&(vm)->heap.regions, (primal_val), (tangent_val)); \
    if (!_d) { (out_val) = FLOAT_VAL(0); break; } \
    int32_t _dp = heap_alloc(&(vm)->heap); \
    if (_dp < 0) { (vm)->error = 1; (out_val) = FLOAT_VAL(0); break; } \
    (vm)->heap.objects[_dp]->type = HEAP_DUAL; \
    (vm)->heap.objects[_dp]->opaque.ptr = _d; \
    (out_val) = (Value){.type = VAL_DUAL, .as.ptr = _dp}; \
} while(0)

    case 750: { /* gradient(f, point) → scalar or tensor of partial derivatives
                 * Scalar point: returns f'(x) (same as derivative)
                 * List/tensor point: returns #(∂f/∂x1 ∂f/∂x2 ... ∂f/∂xn) */
        Value x_val = vm_pop(vm), f_val = vm_pop(vm);

        /* Extract point values */
        double point[VM_AD_MAX_VARS];
        int n = 0;

        if (x_val.type == VAL_PAIR) {
            /* List of values: (list x1 x2 ... xn) */
            Value cur = x_val;
            while (cur.type == VAL_PAIR && n < VM_AD_MAX_VARS) {
                point[n++] = as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
            }
        } else if (x_val.type == VAL_TENSOR && x_val.as.ptr >= 0) {
            /* Tensor of values: #(x1 x2 ... xn) */
            VmTensor* t = (VmTensor*)vm->heap.objects[x_val.as.ptr]->opaque.ptr;
            if (t && t->data) {
                n = (int)(t->total < VM_AD_MAX_VARS ? t->total : VM_AD_MAX_VARS);
                for (int i = 0; i < n; i++) point[i] = t->data[i];
            }
        } else {
            /* Scalar: single-variable derivative */
            point[0] = as_number(x_val);
            n = 1;
        }

        if (n == 0) { vm_push(vm, FLOAT_VAL(0)); break; }

        if (n == 1) {
            /* Scalar case: f'(x) via single dual pass */
            Value dual_arg;
            VM_AD_MAKE_DUAL(vm, point[0], 1.0, dual_arg);
            Value result = vm_ad_call_closure(vm, f_val, &dual_arg, 1);
            if (result.type == VAL_DUAL && result.as.ptr >= 0) {
                VmDual* rd = (VmDual*)vm->heap.objects[result.as.ptr]->opaque.ptr;
                vm_push(vm, FLOAT_VAL(rd ? rd->tangent : 0));
            } else {
                vm_push(vm, FLOAT_VAL(0)); /* constant function */
            }
        } else {
            /* Multi-variable: N forward passes, seed each variable in turn */
            double partials[VM_AD_MAX_VARS];
            for (int i = 0; i < n; i++) {
                Value args[VM_AD_MAX_VARS];
                for (int j = 0; j < n; j++) {
                    VM_AD_MAKE_DUAL(vm, point[j], (j == i) ? 1.0 : 0.0, args[j]);
                }
                Value result = vm_ad_call_closure(vm, f_val, args, n);
                if (result.type == VAL_DUAL && result.as.ptr >= 0) {
                    VmDual* rd = (VmDual*)vm->heap.objects[result.as.ptr]->opaque.ptr;
                    partials[i] = rd ? rd->tangent : 0;
                } else {
                    partials[i] = 0;
                }
            }
            /* Return as tensor */
            int64_t shape[1] = { n };
            VmTensor* t = vm_tensor_from_data(&vm->heap.regions, partials, shape, 1);
            if (t) {
                VM_PUSH_HEAP_OPAQUE(vm, HEAP_TENSOR, VAL_TENSOR, t);
            } else {
                vm_push(vm, NIL_VAL);
            }
        }
        break;
    }

    case 751: { /* jacobian(f, point) → matrix of partial derivatives
                 * For f: R^n → R^m, returns m×n tensor J[i][j] = ∂fi/∂xj
                 * Scalar→scalar: returns f'(x)
                 * Multi-var→scalar: returns gradient (1×n)
                 * Multi-var→vector: returns m×n Jacobian matrix */
        Value x_val = vm_pop(vm), f_val = vm_pop(vm);

        /* Extract point */
        double point[VM_AD_MAX_VARS];
        int n = 0;
        if (x_val.type == VAL_PAIR) {
            Value cur = x_val;
            while (cur.type == VAL_PAIR && n < VM_AD_MAX_VARS) {
                point[n++] = as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
            }
        } else if (x_val.type == VAL_TENSOR && x_val.as.ptr >= 0) {
            VmTensor* t = (VmTensor*)vm->heap.objects[x_val.as.ptr]->opaque.ptr;
            if (t && t->data) {
                n = (int)(t->total < VM_AD_MAX_VARS ? t->total : VM_AD_MAX_VARS);
                for (int i = 0; i < n; i++) point[i] = t->data[i];
            }
        } else {
            point[0] = as_number(x_val);
            n = 1;
        }

        if (n == 0) { vm_push(vm, FLOAT_VAL(0)); break; }

        /* First pass with variable 0 seeded to determine output dimension m */
        Value probe_args[VM_AD_MAX_VARS];
        for (int j = 0; j < n; j++) {
            VM_AD_MAKE_DUAL(vm, point[j], (j == 0) ? 1.0 : 0.0, probe_args[j]);
        }
        Value probe_result = vm_ad_call_closure(vm, f_val, probe_args, n);

        /* Determine output dimension: scalar (m=1) or tensor (m = tensor size) */
        int m = 1;
        if (probe_result.type == VAL_TENSOR && probe_result.as.ptr >= 0) {
            VmTensor* rt = (VmTensor*)vm->heap.objects[probe_result.as.ptr]->opaque.ptr;
            if (rt) m = (int)rt->total;
        }

        if (n == 1 && m == 1) {
            /* Scalar → scalar: just the derivative */
            if (probe_result.type == VAL_DUAL && probe_result.as.ptr >= 0) {
                VmDual* rd = (VmDual*)vm->heap.objects[probe_result.as.ptr]->opaque.ptr;
                vm_push(vm, FLOAT_VAL(rd ? rd->tangent : 0));
            } else {
                vm_push(vm, FLOAT_VAL(0));
            }
        } else {
            /* Build m×n Jacobian matrix */
            double* jac_data = (double*)vm_alloc(&vm->heap.regions,
                                                  (size_t)(m * n) * sizeof(double));
            if (!jac_data) { vm_push(vm, NIL_VAL); break; }
            memset(jac_data, 0, (size_t)(m * n) * sizeof(double));

            for (int i = 0; i < n; i++) {
                Value args[VM_AD_MAX_VARS];
                for (int j = 0; j < n; j++) {
                    VM_AD_MAKE_DUAL(vm, point[j], (j == i) ? 1.0 : 0.0, args[j]);
                }
                Value result = vm_ad_call_closure(vm, f_val, args, n);

                if (m == 1) {
                    /* Scalar output */
                    if (result.type == VAL_DUAL && result.as.ptr >= 0) {
                        VmDual* rd = (VmDual*)vm->heap.objects[result.as.ptr]->opaque.ptr;
                        jac_data[i] = rd ? rd->tangent : 0; /* row 0, col i */
                    }
                } else if (result.type == VAL_TENSOR && result.as.ptr >= 0) {
                    /* Vector output: each element is a dual or scalar */
                    VmTensor* rt = (VmTensor*)vm->heap.objects[result.as.ptr]->opaque.ptr;
                    if (rt && rt->data) {
                        /* Tensor of doubles — tangents already extracted by AD */
                        for (int k = 0; k < m && k < (int)rt->total; k++) {
                            jac_data[k * n + i] = rt->data[k]; /* J[k][i] */
                        }
                    }
                } else if (result.type == VAL_DUAL && result.as.ptr >= 0) {
                    /* Single dual result for m=1 case */
                    VmDual* rd = (VmDual*)vm->heap.objects[result.as.ptr]->opaque.ptr;
                    jac_data[i] = rd ? rd->tangent : 0;
                }
            }

            if (m == 1) {
                /* 1×n → return as 1D tensor (gradient) */
                int64_t shape[1] = { n };
                VmTensor* t = vm_tensor_from_data(&vm->heap.regions, jac_data, shape, 1);
                if (t) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_TENSOR, VAL_TENSOR, t); }
                else { vm_push(vm, NIL_VAL); }
            } else {
                /* m×n Jacobian matrix */
                int64_t shape[2] = { m, n };
                VmTensor* t = vm_tensor_from_data(&vm->heap.regions, jac_data, shape, 2);
                if (t) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_TENSOR, VAL_TENSOR, t); }
                else { vm_push(vm, NIL_VAL); }
            }
        }
        break;
    }

    case 752: { /* hessian(f, point) → EXACT second derivative via hyper-dual numbers
                 * Scalar: seed x = (x,1,1,0) → f₁₂ = f''(x)
                 * Multi-var: H[i][j] = ∂²f/∂xᵢ∂xⱼ via hyper-dual seeding
                 * NO finite differences — mathematically exact. */
#define VM_HD_MAKE(vm, fv, f1v, f2v, f12v, out) do { \
    VmHyperDual* _h = vm_hd_make(&(vm)->heap.regions, (fv), (f1v), (f2v), (f12v)); \
    if (!_h) { (out) = FLOAT_VAL(0); break; } \
    int32_t _hp = heap_alloc(&(vm)->heap); \
    if (_hp < 0) { (vm)->error = 1; (out) = FLOAT_VAL(0); break; } \
    (vm)->heap.objects[_hp]->type = HEAP_HYPER_DUAL; \
    (vm)->heap.objects[_hp]->opaque.ptr = _h; \
    (out) = (Value){.type = VAL_HYPER_DUAL, .as.ptr = _hp}; \
} while(0)
        Value x_val = vm_pop(vm), f_val = vm_pop(vm);

        double point[VM_AD_MAX_VARS];
        int n = 0;
        if (x_val.type == VAL_PAIR) {
            Value cur = x_val;
            while (cur.type == VAL_PAIR && n < VM_AD_MAX_VARS) {
                point[n++] = as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
            }
        } else if (x_val.type == VAL_TENSOR && x_val.as.ptr >= 0) {
            VmTensor* t = (VmTensor*)vm->heap.objects[x_val.as.ptr]->opaque.ptr;
            if (t && t->data) {
                n = (int)(t->total < VM_AD_MAX_VARS ? t->total : VM_AD_MAX_VARS);
                for (int i = 0; i < n; i++) point[i] = t->data[i];
            }
        } else {
            point[0] = as_number(x_val);
            n = 1;
        }

        if (n == 0) { vm_push(vm, FLOAT_VAL(0)); break; }

        if (n == 1) {
            /* Scalar hessian via hyper-dual: seed (x, 1, 1, 0) → f₁₂ = f''(x) */
            Value hd_arg;
            VM_HD_MAKE(vm, point[0], 1.0, 1.0, 0.0, hd_arg);
            Value result = vm_ad_call_closure(vm, f_val, &hd_arg, 1);
            if (result.type == VAL_HYPER_DUAL && result.as.ptr >= 0) {
                VmHyperDual* rh = (VmHyperDual*)vm->heap.objects[result.as.ptr]->opaque.ptr;
                vm_push(vm, FLOAT_VAL(rh ? rh->f12 : 0.0));
            } else {
                vm_push(vm, FLOAT_VAL(0.0));
            }
        } else {
            /* Multi-variable Hessian via hyper-dual: H[i][j] = ∂²f/∂xᵢ∂xⱼ
             * Seed xₖ = (point[k], δₖᵢ, δₖⱼ, 0) → result.f12 = H[i][j] */
            double* hess_data = (double*)vm_alloc(&vm->heap.regions,
                                                   (size_t)(n * n) * sizeof(double));
            if (!hess_data) { vm_push(vm, NIL_VAL); break; }

            for (int i = 0; i < n; i++) {
                for (int j = i; j < n; j++) {
                    Value args[VM_AD_MAX_VARS];
                    for (int k = 0; k < n; k++) {
                        VM_HD_MAKE(vm, point[k], (k==i)?1.0:0.0, (k==j)?1.0:0.0, 0.0, args[k]);
                    }
                    Value r = vm_ad_call_closure(vm, f_val, args, n);
                    double h_ij = 0.0;
                    if (r.type == VAL_HYPER_DUAL && r.as.ptr >= 0) {
                        VmHyperDual* rh = (VmHyperDual*)vm->heap.objects[r.as.ptr]->opaque.ptr;
                        if (rh) h_ij = rh->f12;
                    }
                    hess_data[i * n + j] = h_ij;
                    hess_data[j * n + i] = h_ij;
                }
            }

            int64_t shape[2] = { n, n };
            VmTensor* t = vm_tensor_from_data(&vm->heap.regions, hess_data, shape, 2);
            if (t) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_TENSOR, VAL_TENSOR, t); }
            else { vm_push(vm, NIL_VAL); }
        }
#undef VM_HD_MAKE
        break;
    }

    case 753: { /* divergence(F, point) → scalar
                 * div(F) = ∂F1/∂x1 + ∂F2/∂x2 + ... + ∂Fn/∂xn
                 * F: R^n → R^n (vector field), point: list or tensor */
        Value x_val = vm_pop(vm), f_val = vm_pop(vm);

        double point[VM_AD_MAX_VARS];
        int n = 0;
        if (x_val.type == VAL_PAIR) {
            Value cur = x_val;
            while (cur.type == VAL_PAIR && n < VM_AD_MAX_VARS) {
                point[n++] = as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
            }
        } else if (x_val.type == VAL_TENSOR && x_val.as.ptr >= 0) {
            VmTensor* t = (VmTensor*)vm->heap.objects[x_val.as.ptr]->opaque.ptr;
            if (t && t->data) {
                n = (int)(t->total < VM_AD_MAX_VARS ? t->total : VM_AD_MAX_VARS);
                for (int i = 0; i < n; i++) point[i] = t->data[i];
            }
        } else {
            point[0] = as_number(x_val);
            n = 1;
        }

        if (n == 0) { vm_push(vm, FLOAT_VAL(0)); break; }

        /* Sum of ∂Fi/∂xi: for each i, seed variable i, extract component i */
        double div = 0;
        for (int i = 0; i < n; i++) {
            Value args[VM_AD_MAX_VARS];
            for (int j = 0; j < n; j++) {
                VM_AD_MAKE_DUAL(vm, point[j], (j == i) ? 1.0 : 0.0, args[j]);
            }
            Value result = vm_ad_call_closure(vm, f_val, args, n);

            /* Extract the i-th component's tangent */
            if (result.type == VAL_TENSOR && result.as.ptr >= 0) {
                /* F returns a tensor — we need the tangent of element i.
                 * Since the tensor contains primals (doubles), the tangent
                 * information is lost. We need to use a different approach:
                 * call F component-wise. But if F returns a tensor of duals,
                 * we can extract directly. For tensor-returning functions,
                 * use finite differences as fallback. */
                VmTensor* rt = (VmTensor*)vm->heap.objects[result.as.ptr]->opaque.ptr;
                if (rt && rt->data && i < (int)rt->total) {
                    /* Tensor element — use finite difference for this component */
                    double fplus, fminus;
                    double pt_plus[VM_AD_MAX_VARS], pt_minus[VM_AD_MAX_VARS];
                    double h = 1e-7;
                    for (int k = 0; k < n; k++) {
                        pt_plus[k] = point[k] + ((k == i) ? h : 0);
                        pt_minus[k] = point[k] - ((k == i) ? h : 0);
                    }
                    /* F(x + h*ei)[i] */
                    Value ap[VM_AD_MAX_VARS];
                    for (int k = 0; k < n; k++) ap[k] = FLOAT_VAL(pt_plus[k]);
                    Value rp = vm_ad_call_closure(vm, f_val, ap, n);
                    fplus = 0;
                    if (rp.type == VAL_TENSOR && rp.as.ptr >= 0) {
                        VmTensor* tp = (VmTensor*)vm->heap.objects[rp.as.ptr]->opaque.ptr;
                        if (tp && tp->data && i < (int)tp->total) fplus = tp->data[i];
                    }
                    /* F(x - h*ei)[i] */
                    Value am[VM_AD_MAX_VARS];
                    for (int k = 0; k < n; k++) am[k] = FLOAT_VAL(pt_minus[k]);
                    Value rm = vm_ad_call_closure(vm, f_val, am, n);
                    vm->ad_finite_difference_evals += 2;
                    fminus = 0;
                    if (rm.type == VAL_TENSOR && rm.as.ptr >= 0) {
                        VmTensor* tm = (VmTensor*)vm->heap.objects[rm.as.ptr]->opaque.ptr;
                        if (tm && tm->data && i < (int)tm->total) fminus = tm->data[i];
                    }
                    div += (fplus - fminus) / (2.0 * h);
                }
            } else if (result.type == VAL_PAIR) {
                /* F returns a list — walk to i-th element, extract tangent */
                Value cur = result;
                for (int k = 0; k < i && cur.type == VAL_PAIR; k++) {
                    cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
                }
                if (cur.type == VAL_PAIR) {
                    Value elem = vm->heap.objects[cur.as.ptr]->cons.car;
                    if (elem.type == VAL_DUAL && elem.as.ptr >= 0) {
                        VmDual* rd = (VmDual*)vm->heap.objects[elem.as.ptr]->opaque.ptr;
                        if (rd) div += rd->tangent;
                    }
                }
            } else if (n == 1 && result.type == VAL_DUAL && result.as.ptr >= 0) {
                /* Scalar function */
                VmDual* rd = (VmDual*)vm->heap.objects[result.as.ptr]->opaque.ptr;
                if (rd) div += rd->tangent;
            }
        }
        vm_push(vm, FLOAT_VAL(div));
        break;
    }

    case 754: { /* curl(F, point) → 3D vector
                 * curl(F) = (∂F3/∂y - ∂F2/∂z, ∂F1/∂z - ∂F3/∂x, ∂F2/∂x - ∂F1/∂y)
                 * F: R^3 → R^3, point must have exactly 3 components */
        Value x_val = vm_pop(vm), f_val = vm_pop(vm);

        double point[3];
        int n = 0;
        if (x_val.type == VAL_PAIR) {
            Value cur = x_val;
            while (cur.type == VAL_PAIR && n < 3) {
                point[n++] = as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
            }
        } else if (x_val.type == VAL_TENSOR && x_val.as.ptr >= 0) {
            VmTensor* t = (VmTensor*)vm->heap.objects[x_val.as.ptr]->opaque.ptr;
            if (t && t->data) {
                n = (int)(t->total < 3 ? t->total : 3);
                for (int i = 0; i < n; i++) point[i] = t->data[i];
            }
        }

        if (n != 3) {
            /* Curl requires exactly 3 dimensions */
            vm_push(vm, NIL_VAL);
            break;
        }

        /* Compute the 3×3 Jacobian via central differences:
         * J[i][j] = ∂Fi/∂xj
         * curl = (J[2][1] - J[1][2], J[0][2] - J[2][0], J[1][0] - J[0][1]) */
        double jac[3][3];
        double h = 1e-7;

        for (int j = 0; j < 3; j++) {
            /* ∂F/∂xj via central difference */
            Value ap[3], am[3];
            for (int k = 0; k < 3; k++) {
                ap[k] = FLOAT_VAL(point[k] + ((k == j) ? h : 0));
                am[k] = FLOAT_VAL(point[k] - ((k == j) ? h : 0));
            }
            Value rp = vm_ad_call_closure(vm, f_val, ap, 3);
            Value rm = vm_ad_call_closure(vm, f_val, am, 3);
            vm->ad_finite_difference_evals += 2;

            /* Extract 3 components from each result */
            double fp[3] = {0,0,0}, fm[3] = {0,0,0};
            if (rp.type == VAL_TENSOR && rp.as.ptr >= 0) {
                VmTensor* tp = (VmTensor*)vm->heap.objects[rp.as.ptr]->opaque.ptr;
                if (tp && tp->data) for (int i = 0; i < 3 && i < (int)tp->total; i++) fp[i] = tp->data[i];
            } else if (rp.type == VAL_PAIR) {
                Value cur = rp; int idx = 0;
                while (cur.type == VAL_PAIR && idx < 3) {
                    fp[idx++] = as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                    cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
                }
            }
            if (rm.type == VAL_TENSOR && rm.as.ptr >= 0) {
                VmTensor* tm = (VmTensor*)vm->heap.objects[rm.as.ptr]->opaque.ptr;
                if (tm && tm->data) for (int i = 0; i < 3 && i < (int)tm->total; i++) fm[i] = tm->data[i];
            } else if (rm.type == VAL_PAIR) {
                Value cur = rm; int idx = 0;
                while (cur.type == VAL_PAIR && idx < 3) {
                    fm[idx++] = as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                    cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
                }
            }

            for (int i = 0; i < 3; i++) {
                jac[i][j] = (fp[i] - fm[i]) / (2.0 * h);
            }
        }

        /* curl = (∂F3/∂y - ∂F2/∂z, ∂F1/∂z - ∂F3/∂x, ∂F2/∂x - ∂F1/∂y) */
        double curl_data[3] = {
            jac[2][1] - jac[1][2],  /* ∂F3/∂y - ∂F2/∂z */
            jac[0][2] - jac[2][0],  /* ∂F1/∂z - ∂F3/∂x */
            jac[1][0] - jac[0][1]   /* ∂F2/∂x - ∂F1/∂y */
        };
        int64_t shape[1] = { 3 };
        VmTensor* t = vm_tensor_from_data(&vm->heap.regions, curl_data, shape, 1);
        if (t) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_TENSOR, VAL_TENSOR, t); }
        else { vm_push(vm, NIL_VAL); }
        break;
    }

    case 755: { /* laplacian(f, point) → scalar
                 * ∇²f = ∂²f/∂x1² + ∂²f/∂x2² + … + ∂²f/∂xn²
                 *     = trace of the Hessian matrix
                 *
                 * Exact via hyper-dual numbers — NO finite differences.
                 * For each i, seed xi with (value, 1, 1, 0) and every
                 * other xk with (value, 0, 0, 0); the returned
                 * hyper-dual's f12 component is ∂²f/∂xi². Sum over i. */
#define VM_HD_MAKE_L(vm, fv, f1v, f2v, f12v, out) do { \
    VmHyperDual* _h = vm_hd_make(&(vm)->heap.regions, (fv), (f1v), (f2v), (f12v)); \
    if (!_h) { (out) = FLOAT_VAL(0); break; } \
    int32_t _hp = heap_alloc(&(vm)->heap); \
    if (_hp < 0) { (vm)->error = 1; (out) = FLOAT_VAL(0); break; } \
    (vm)->heap.objects[_hp]->type = HEAP_HYPER_DUAL; \
    (vm)->heap.objects[_hp]->opaque.ptr = _h; \
    (out) = (Value){.type = VAL_HYPER_DUAL, .as.ptr = _hp}; \
} while(0)
        Value x_val = vm_pop(vm), f_val = vm_pop(vm);

        double point[VM_AD_MAX_VARS];
        int n = 0;
        if (x_val.type == VAL_PAIR) {
            Value cur = x_val;
            while (cur.type == VAL_PAIR && n < VM_AD_MAX_VARS) {
                point[n++] = as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
            }
        } else if (x_val.type == VAL_TENSOR && x_val.as.ptr >= 0) {
            VmTensor* t = (VmTensor*)vm->heap.objects[x_val.as.ptr]->opaque.ptr;
            if (t && t->data) {
                n = (int)(t->total < VM_AD_MAX_VARS ? t->total : VM_AD_MAX_VARS);
                for (int i = 0; i < n; i++) point[i] = t->data[i];
            }
        } else {
            point[0] = as_number(x_val);
            n = 1;
        }

        if (n == 0) { vm_push(vm, FLOAT_VAL(0)); break; }

        double laplacian = 0;
        for (int i = 0; i < n; i++) {
            Value args[VM_AD_MAX_VARS];
            for (int k = 0; k < n; k++) {
                VM_HD_MAKE_L(vm, point[k],
                             (k == i) ? 1.0 : 0.0,
                             (k == i) ? 1.0 : 0.0,
                             0.0,
                             args[k]);
            }
            Value r = vm_ad_call_closure(vm, f_val, args, n);
            if (r.type == VAL_HYPER_DUAL && r.as.ptr >= 0) {
                VmHyperDual* rh = (VmHyperDual*)vm->heap.objects[r.as.ptr]->opaque.ptr;
                if (rh) laplacian += rh->f12;
            }
        }
#undef VM_HD_MAKE_L

        vm_push(vm, FLOAT_VAL(laplacian));
        break;
    }

    case 756: { /* directional-derivative(f, point, direction) → scalar
                 * D_v(f) = ∇f · v = Σ (∂f/∂xi * vi)
                 * Uses a single forward pass with tangent = direction vector */
        Value dir_val = vm_pop(vm), x_val = vm_pop(vm), f_val = vm_pop(vm);

        double point[VM_AD_MAX_VARS], dir[VM_AD_MAX_VARS];
        int n = 0, nd = 0;

        /* Extract point */
        if (x_val.type == VAL_PAIR) {
            Value cur = x_val;
            while (cur.type == VAL_PAIR && n < VM_AD_MAX_VARS) {
                point[n++] = as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
            }
        } else if (x_val.type == VAL_TENSOR && x_val.as.ptr >= 0) {
            VmTensor* t = (VmTensor*)vm->heap.objects[x_val.as.ptr]->opaque.ptr;
            if (t && t->data) {
                n = (int)(t->total < VM_AD_MAX_VARS ? t->total : VM_AD_MAX_VARS);
                for (int i = 0; i < n; i++) point[i] = t->data[i];
            }
        } else {
            point[0] = as_number(x_val);
            n = 1;
        }

        /* Extract direction */
        if (dir_val.type == VAL_PAIR) {
            Value cur = dir_val;
            while (cur.type == VAL_PAIR && nd < VM_AD_MAX_VARS) {
                dir[nd++] = as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
            }
        } else if (dir_val.type == VAL_TENSOR && dir_val.as.ptr >= 0) {
            VmTensor* t = (VmTensor*)vm->heap.objects[dir_val.as.ptr]->opaque.ptr;
            if (t && t->data) {
                nd = (int)(t->total < VM_AD_MAX_VARS ? t->total : VM_AD_MAX_VARS);
                for (int i = 0; i < nd; i++) dir[i] = t->data[i];
            }
        } else {
            dir[0] = as_number(dir_val);
            nd = 1;
        }

        if (n == 0 || nd != n) {
            vm_push(vm, FLOAT_VAL(0));
            break;
        }

        /* Single forward pass: seed tangent = direction vector
         * D_v(f)(x) = Σ vi * ∂f/∂xi = tangent when all tangents are vi
         * This is the efficient approach — one pass instead of n+1 */
        Value args[VM_AD_MAX_VARS];
        for (int j = 0; j < n; j++) {
            VM_AD_MAKE_DUAL(vm, point[j], dir[j], args[j]);
        }
        Value result = vm_ad_call_closure(vm, f_val, args, n);
        if (result.type == VAL_DUAL && result.as.ptr >= 0) {
            VmDual* rd = (VmDual*)vm->heap.objects[result.as.ptr]->opaque.ptr;
            vm_push(vm, FLOAT_VAL(rd ? rd->tangent : 0));
        } else {
            vm_push(vm, FLOAT_VAL(0));
        }
        break;
    }

#undef VM_AD_MAKE_DUAL
#undef VM_AD_MAX_VARS


    /* ══════════════════════════════════════════════════════════════════════
     * Compiler-required native functions (merged from eshkol_compiler.c)
     * ══════════════════════════════════════════════════════════════════════ */

    case 100: /* build-string-from-packed: TOS has packed chars, below that length */
    case 101: { /* build-symbol-from-packed: same payload, distinct Scheme type */
        /* Stack: [len, pack0, pack1, ..., packN-1] where N = (len+7)/8 */
        /* Peek down to find the length */
        int n_packs_guess = 0;
        int slen = 0;
        /* The compiler pushes: CONST(len), CONST(pack0), ..., CONST(packN-1), NATIVE_CALL 100 */
        /* So TOS-N = len, TOS-(N-1) through TOS-0 = packs, where N = (len+7)/8 */
        /* We need to scan backwards from TOS to find the length */
        for (int try_n = 0; try_n < 64; try_n++) {
            int len_pos = vm->sp - try_n - 1;
            if (len_pos >= 0 && vm->stack[len_pos].type == VAL_INT) {
                int candidate = (int)vm->stack[len_pos].as.i;
                int expected_packs = (candidate + 7) / 8;
                if (expected_packs == try_n && candidate >= 0 && candidate < 256) {
                    slen = candidate;
                    n_packs_guess = try_n;
                    break;
                }
            }
        }
        char buf[256];
        for (int p = n_packs_guess - 1; p >= 0; p--) {
            Value pack_v = vm_pop(vm);
            int64_t pack = pack_v.as.i;
            for (int b = 0; b < 8 && p * 8 + b < slen; b++)
                buf[p * 8 + b] = (char)((pack >> (b * 8)) & 0xFF);
        }
        vm_pop(vm); /* pop length */
        buf[slen] = 0;
        VmString* s = vm_string_from_cstr(&vm->heap.regions, buf);
        if (s) {
            int32_t ptr = heap_alloc(&vm->heap);
            if (ptr >= 0) {
                vm->heap.objects[ptr]->type = HEAP_STRING;
                vm->heap.objects[ptr]->opaque.ptr = s;
                vm_push(vm, (Value){.type = fid == 101 ? VAL_SYMBOL : VAL_STRING,
                                    .as.ptr = ptr});
                break;
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    case 131: { /* open_upvalues(closure, count, base_slot) — letrec binding */
        Value base_v = vm_pop(vm), count_v = vm_pop(vm), cl_val = vm_pop(vm);
        /* For letrec: patch closure upvalues to point to current stack values */
        if (cl_val.type == VAL_CLOSURE) {
            HeapObject* cl = vm->heap.objects[cl_val.as.ptr];
            int count = (int)as_number(count_v);
            int base = (int)as_number(base_v);
            for (int i = 0; i < cl->closure.n_upvalues && i < count; i++) {
                int slot = base + i;
                if (slot >= 0 && slot < vm->sp)
                    cl->closure.upvalues[i] = vm->stack[vm->fp + slot];
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    case 133: { /* eq?: identity equality */
        Value b = vm_pop(vm), a = vm_pop(vm);
        vm_push(vm, BOOL_VAL(vm_identity_equal(vm, a, b)));
        break;
    }

    case 134: { /* equal?: deep structural equality (pairs, vectors, strings, numbers) */
        Value b = vm_pop(vm), a = vm_pop(vm);
        vm_push(vm, BOOL_VAL(vm_deep_equal(vm, a, b)));
        break;
    }

    case 142: { /* add2 — complex-aware */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        if (a_val.type == VAL_COMPLEX || b_val.type == VAL_COMPLEX) {
            VmComplex a_z = {as_number(a_val), 0}, b_z = {as_number(b_val), 0};
            if (a_val.type == VAL_COMPLEX) a_z = *(VmComplex*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
            if (b_val.type == VAL_COMPLEX) b_z = *(VmComplex*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
            VmComplex* r = vm_complex_add(&vm->heap.regions, &a_z, &b_z);
            if (!r) { vm->error = 1; break; }
            int32_t p = heap_alloc(&vm->heap); if (p < 0) { vm->error = 1; break; }
            vm->heap.objects[p]->type = HEAP_COMPLEX; vm->heap.objects[p]->opaque.ptr = r;
            vm_push(vm, (Value){.type = VAL_COMPLEX, .as.ptr = p});
        } else if (vm_either_bignum(a_val,b_val)) { vm_bignum_arith(vm,a_val,b_val,'+'); }
        else if (a_val.type==VAL_INT && b_val.type==VAL_INT) {
            int64_t r; if (__builtin_add_overflow(a_val.as.i,b_val.as.i,&r)) vm_bignum_arith(vm,a_val,b_val,'+'); else vm_push(vm, INT_VAL(r));
        } else { vm_push(vm, number_val(as_number(a_val) + as_number(b_val))); }
        break; }
    case 143: { /* sub2 — complex-aware */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        if (a_val.type == VAL_COMPLEX || b_val.type == VAL_COMPLEX) {
            VmComplex a_z = {as_number(a_val), 0}, b_z = {as_number(b_val), 0};
            if (a_val.type == VAL_COMPLEX) a_z = *(VmComplex*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
            if (b_val.type == VAL_COMPLEX) b_z = *(VmComplex*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
            VmComplex* r = vm_complex_sub(&vm->heap.regions, &a_z, &b_z);
            if (!r) { vm->error = 1; break; }
            int32_t p = heap_alloc(&vm->heap); if (p < 0) { vm->error = 1; break; }
            vm->heap.objects[p]->type = HEAP_COMPLEX; vm->heap.objects[p]->opaque.ptr = r;
            vm_push(vm, (Value){.type = VAL_COMPLEX, .as.ptr = p});
        } else if (vm_either_bignum(a_val,b_val)) { vm_bignum_arith(vm,a_val,b_val,'-'); }
        else if (a_val.type==VAL_INT && b_val.type==VAL_INT) {
            int64_t r; if (__builtin_sub_overflow(a_val.as.i,b_val.as.i,&r)) vm_bignum_arith(vm,a_val,b_val,'-'); else vm_push(vm, INT_VAL(r));
        } else { vm_push(vm, number_val(as_number(a_val) - as_number(b_val))); }
        break; }
    case 144: { /* mul2 — complex-aware */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        if (a_val.type == VAL_COMPLEX || b_val.type == VAL_COMPLEX) {
            VmComplex a_z = {as_number(a_val), 0}, b_z = {as_number(b_val), 0};
            if (a_val.type == VAL_COMPLEX) a_z = *(VmComplex*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
            if (b_val.type == VAL_COMPLEX) b_z = *(VmComplex*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
            VmComplex* r = vm_complex_mul(&vm->heap.regions, &a_z, &b_z);
            if (!r) { vm->error = 1; break; }
            int32_t p = heap_alloc(&vm->heap); if (p < 0) { vm->error = 1; break; }
            vm->heap.objects[p]->type = HEAP_COMPLEX; vm->heap.objects[p]->opaque.ptr = r;
            vm_push(vm, (Value){.type = VAL_COMPLEX, .as.ptr = p});
        } else if (vm_either_bignum(a_val,b_val)) { vm_bignum_arith(vm,a_val,b_val,'*'); }
        else if (a_val.type==VAL_INT && b_val.type==VAL_INT) {
            int64_t r; if (__builtin_mul_overflow(a_val.as.i,b_val.as.i,&r)) vm_bignum_arith(vm,a_val,b_val,'*'); else vm_push(vm, INT_VAL(r));
        } else { vm_push(vm, number_val(as_number(a_val) * as_number(b_val))); }
        break; }
    case 145: { /* div2 — complex- and rational-aware.
                 * The prelude's variadic `/` folds with div2, so this is the
                 * real path for (/ 1 3). Exact/exact division yields an exact
                 * rational (or an integer when it divides), matching the native
                 * path; previously it always produced an inexact float. */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        if (a_val.type == VAL_COMPLEX || b_val.type == VAL_COMPLEX) {
            VmComplex a_z = {as_number(a_val), 0}, b_z = {as_number(b_val), 0};
            if (a_val.type == VAL_COMPLEX) a_z = *(VmComplex*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
            if (b_val.type == VAL_COMPLEX) b_z = *(VmComplex*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
            VmComplex* r = vm_complex_div(&vm->heap.regions, &a_z, &b_z);
            if (!r) { vm->error = 1; break; }
            int32_t p = heap_alloc(&vm->heap); if (p < 0) { vm->error = 1; break; }
            vm->heap.objects[p]->type = HEAP_COMPLEX; vm->heap.objects[p]->opaque.ptr = r;
            vm_push(vm, (Value){.type = VAL_COMPLEX, .as.ptr = p});
        } else if (a_val.type == VAL_RATIONAL || b_val.type == VAL_RATIONAL ||
                   (a_val.type == VAL_INT && b_val.type == VAL_INT)) {
            if ((a_val.type == VAL_INT && b_val.type == VAL_INT && b_val.as.i == 0)) { vm->error = 1; break; }
            vm_push(vm, a_val); vm_push(vm, b_val); vm_dispatch_native(vm, 334); /* rational div: reduces, collapses denom==1 to int */
        } else { vm_push(vm, number_val(as_number(a_val) / as_number(b_val))); }
        break; }
    /* Comparison operators as first-class functions (for sort, map, fold, etc.) */
    case 146: { Value b = vm_pop(vm), a = vm_pop(vm); if (vm_either_bignum(a,b)) { vm_push(vm, BOOL_VAL(vm_bignum_compare_vals(vm,a,b) <  0)); break; } vm_push(vm, BOOL_VAL(as_number(a) < as_number(b))); break; }  /* < */
    case 147: { Value b = vm_pop(vm), a = vm_pop(vm); if (vm_either_bignum(a,b)) { vm_push(vm, BOOL_VAL(vm_bignum_compare_vals(vm,a,b) >  0)); break; } vm_push(vm, BOOL_VAL(as_number(a) > as_number(b))); break; }  /* > */
    case 148: { Value b = vm_pop(vm), a = vm_pop(vm); if (vm_either_bignum(a,b)) { vm_push(vm, BOOL_VAL(vm_bignum_compare_vals(vm,a,b) <= 0)); break; } vm_push(vm, BOOL_VAL(as_number(a) <= as_number(b))); break; } /* <= */
    case 149: { Value b = vm_pop(vm), a = vm_pop(vm); if (vm_either_bignum(a,b)) { vm_push(vm, BOOL_VAL(vm_bignum_compare_vals(vm,a,b) >= 0)); break; } vm_push(vm, BOOL_VAL(as_number(a) >= as_number(b))); break; } /* >= */
    case 150: { Value b = vm_pop(vm), a = vm_pop(vm); if (vm_either_bignum(a,b)) { vm_push(vm, BOOL_VAL(vm_bignum_compare_vals(vm,a,b) == 0)); break; } vm_push(vm, BOOL_VAL(as_number(a) == as_number(b))); break; } /* = */

    /* Core operations as first-class native functions (IDs 200-226) */
    case 200: { Value a = vm_pop(vm); /* car */
        if (a.type == VAL_PAIR) { HeapObject* o = vm->heap.objects[a.as.ptr]; vm_push(vm, o->cons.car); }
        else { fprintf(stderr, "CAR on non-pair\n"); vm_push(vm, NIL_VAL); } break; }
    case 201: { Value a = vm_pop(vm); /* cdr */
        if (a.type == VAL_PAIR) { HeapObject* o = vm->heap.objects[a.as.ptr]; vm_push(vm, o->cons.cdr); }
        else { fprintf(stderr, "CDR on non-pair\n"); vm_push(vm, NIL_VAL); } break; }
    case 202: { Value b = vm_pop(vm), a = vm_pop(vm); /* cons */
        int32_t p = heap_alloc(&vm->heap); if (p < 0) { vm->error = 1; break; }
        vm->heap.objects[p]->type = HEAP_CONS;
        vm->heap.objects[p]->cons.car = a; vm->heap.objects[p]->cons.cdr = b;
        vm_push(vm, (Value){VAL_PAIR, {.ptr = p}}); break; }
    case 203: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_NIL)); break; }  /* null? */
    case 204: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_PAIR)); break; } /* pair? */
    case 205: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(!is_truthy(a))); break; }      /* not */
    case 206: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_INT || a.type == VAL_FLOAT)); break; } /* number? */
    case 207: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_STRING)); break; } /* string? */
    case 208: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_BOOL)); break; }   /* boolean? */
    case 209: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_CLOSURE)); break; }/* procedure? */
    case 210: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_VECTOR)); break; } /* vector? */
    case 211: { Value a = vm_pop(vm); print_value(vm, a); fflush(stdout); vm_push(vm, (Value){.type = VAL_VOID}); break; } /* display */
    case 212: { /* _write1(value) */
        Value a = vm_pop(vm);
        vm_write_value_port(vm, a, vm_port_current_output(), 1);
        fflush(stdout);
        vm_push(vm, (Value){.type = VAL_VOID});
        break;
    }
    case 618: { /* _write2(value, port) */
        Value port_value = vm_pop(vm);
        Value value = vm_pop(vm);
        VmPort* port = vm_value_as_port(vm, port_value);
        if (!port || port->dir != VM_PORT_OUTPUT) {
            fprintf(stderr, "ERROR: write requires an open output port\n");
            vm->error = 1;
        } else {
            vm_write_value_port(vm, value, port, 1);
        }
        vm_push(vm, (Value){.type = VAL_VOID});
        break;
    }
    case 213: { Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(as_number_vm(vm, a))); break; }  /* exact->inexact */
    case 214: { Value a = vm_pop(vm); vm_push(vm, INT_VAL((int64_t)as_number_vm(vm, a))); break; } /* inexact->exact */
    case 215: { /* string->number — handles #x/#b/#o/#d prefixes */
        Value a = vm_pop(vm);
        if (a.type != VAL_STRING) { vm_push(vm, BOOL_VAL(0)); break; }
        VmString* s215 = (VmString*)vm->heap.objects[a.as.ptr]->opaque.ptr;
        if (!s215 || !s215->data || s215->data[0] == '\0') { vm_push(vm, BOOL_VAL(0)); break; }
        const char* p215 = s215->data;
        int radix215 = 10;
        if (p215[0] == '#') {
            char pfx = (char)(p215[1] | 32); /* lowercase */
            if      (pfx == 'x') { radix215 = 16; p215 += 2; }
            else if (pfx == 'b') { radix215 = 2;  p215 += 2; }
            else if (pfx == 'o') { radix215 = 8;  p215 += 2; }
            else if (pfx == 'd') { radix215 = 10; p215 += 2; }
            else { vm_push(vm, BOOL_VAL(0)); break; }
        }
        char* end215 = NULL;
        if (radix215 == 10) {
            /* Try integer first; fall back to float */
            long long iv = strtoll(p215, &end215, 10);
            if (*end215 == '\0' && end215 != p215) { vm_push(vm, INT_VAL((int64_t)iv)); break; }
            double dv = strtod(p215, &end215);
            if (*end215 == '\0' && end215 != p215) { vm_push(vm, FLOAT_VAL(dv)); break; }
            vm_push(vm, BOOL_VAL(0));
        } else {
            long long iv = strtoll(p215, &end215, radix215);
            if (*end215 == '\0' && end215 != p215) { vm_push(vm, INT_VAL((int64_t)iv)); break; }
            vm_push(vm, BOOL_VAL(0));
        }
        break; }
    case 216: { Value a = vm_pop(vm); vm_push(vm, INT_VAL((int64_t)as_number(a))); break; } /* char->integer */
    case 217: { Value a = vm_pop(vm); vm_push(vm, (Value){.type = VAL_CHAR, .as.i = (int64_t)as_number(a)}); break; } /* integer->char */
    case 218: { /* make-vector */
        Value fill = vm_pop(vm), size_v = vm_pop(vm);
        int sz = (int)as_number(size_v);
        int32_t p = heap_alloc(&vm->heap); if (p < 0) { vm->error = 1; break; }
        vm->heap.objects[p]->type = HEAP_VECTOR;
        VmVector* v = (VmVector*)vm_alloc(&vm->heap.regions, sizeof(VmVector));
        v->len = sz; v->cap = sz;
        v->items = (Value*)vm_alloc(&vm->heap.regions, sz * sizeof(Value));
        for (int i = 0; i < sz; i++) v->items[i] = fill;
        vm->heap.objects[p]->opaque.ptr = v;
        vm_push(vm, (Value){VAL_VECTOR, {.ptr = p}}); break; }
    case 219: { /* vector-ref */
        Value idx_v = vm_pop(vm), vec_v = vm_pop(vm);
        if (vec_v.type == VAL_VECTOR) {
            VmVector* v = (VmVector*)vm->heap.objects[vec_v.as.ptr]->opaque.ptr;
            int idx = (int)as_number(idx_v);
            if (v && idx >= 0 && idx < v->len) vm_push(vm, v->items[idx]);
            else vm_push(vm, NIL_VAL);
        } else vm_push(vm, NIL_VAL); break; }
    case 220: { /* vector-set! */
        Value val = vm_pop(vm), idx_v = vm_pop(vm), vec_v = vm_pop(vm);
        if (vec_v.type == VAL_VECTOR) {
            VmVector* v = (VmVector*)vm->heap.objects[vec_v.as.ptr]->opaque.ptr;
            int idx = (int)as_number(idx_v);
            if (v && idx >= 0 && idx < v->len) v->items[idx] = val;
        } vm_push(vm, NIL_VAL); break; }
    case 221: { /* vector-length */
        Value v = vm_pop(vm);
        if (v.type == VAL_VECTOR) {
            VmVector* vec = (VmVector*)vm->heap.objects[v.as.ptr]->opaque.ptr;
            vm_push(vm, INT_VAL(vec ? vec->len : 0));
        } else vm_push(vm, INT_VAL(0)); break; }
    case 222: { /* string->list */
        Value s_val = vm_pop(vm);
        Value result = NIL_VAL;
        if (s_val.type == VAL_STRING) {
            VmString* s = (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr;
            if (s) for (int i = s->char_len - 1; i >= 0; i--) {
                int cp = vm_string_ref(s, i);
                int32_t p = heap_alloc(&vm->heap); if (p < 0) break;
                vm->heap.objects[p]->type = HEAP_CONS;
                vm->heap.objects[p]->cons.car = (Value){.type = VAL_CHAR, .as.i = cp};
                vm->heap.objects[p]->cons.cdr = result;
                result = PAIR_VAL(p);
            }
        }
        vm_push(vm, result); break;
    }
    case 223: { /* list->string */
        Value lst = vm_pop(vm);
        char buf[4096]; int len = 0;
        Value cur = lst;
        while (cur.type == VAL_PAIR && len < 4095) {
            int cp = (int)as_number(vm->heap.objects[cur.as.ptr]->cons.car);
            if (cp >= 0 && cp < 128) buf[len++] = (char)cp;
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
        }
        buf[len] = 0;
        VmString* s = vm_string_from_cstr(&vm->heap.regions, buf);
        if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); }
        else vm_push(vm, NIL_VAL);
        break;
    }
    case 224: { /* gcd */
        Value b = vm_pop(vm), a = vm_pop(vm);
        int64_t x = llabs((int64_t)as_number(a)), y = llabs((int64_t)as_number(b));
        while (y != 0) { int64_t t = y; y = x % y; x = t; }
        vm_push(vm, INT_VAL(x)); break;
    }
    case 225: { /* lcm */
        Value b = vm_pop(vm), a = vm_pop(vm);
        int64_t x = llabs((int64_t)as_number(a)), y = llabs((int64_t)as_number(b));
        if (x == 0 || y == 0) { vm_push(vm, INT_VAL(0)); break; }
        int64_t g = x, h = y;
        while (h != 0) { int64_t t = h; h = g % h; g = t; }
        vm_push(vm, INT_VAL(x / g * y)); break;
    }
    case 226: { /* make-string(n, char) */
        Value ch = vm_pop(vm), n = vm_pop(vm);
        int sz = (int)as_number(n), c = (int)as_number(ch);
        if (sz < 0) sz = 0; if (sz > 65536) sz = 65536;
        char* buf = (char*)vm_alloc(&vm->heap.regions, (size_t)(sz + 1));
        if (buf) { memset(buf, c > 0 && c < 128 ? c : ' ', sz); buf[sz] = 0;
            VmString* s = vm_string_from_cstr(&vm->heap.regions, buf);
            if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
        }
        vm_push(vm, NIL_VAL); break;
    }

    case 151: { /* direct open slot: set closure upvalue to reference a stack slot */
        Value slot_v = vm_pop(vm), uv_idx_v = vm_pop(vm), cl_val = vm_pop(vm);
        if (cl_val.type == VAL_CLOSURE) {
            HeapObject* cl = vm->heap.objects[cl_val.as.ptr];
            int uv_idx = (int)as_number(uv_idx_v);
            int slot = (int)as_number(slot_v);
            int absolute_slot = vm->fp + slot;
            if (uv_idx >= 0 && uv_idx < cl->closure.n_upvalues &&
                absolute_slot >= 0 && absolute_slot < vm->sp)
                cl->closure.open_slots[uv_idx] = absolute_slot;
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    case 130: { /* raise: dispatch to handler or error */
        Value exn = vm_pop(vm);
        vm->current_exception = exn;
        if (vm->n_handlers > 0) {
            vm->n_handlers--;
            /* Unwind dynamic-wind after-thunks */
            int target_winds = vm->handler_stack[vm->n_handlers].n_winds;
            while (vm->n_winds > target_winds) {
                vm->n_winds--;
                Value after = vm->wind_stack[vm->n_winds].after;
                vm_run_wind_after(vm, after);
            }
            vm_unwind_parameter_bindings(
                vm, vm->handler_stack[vm->n_handlers].n_parameter_bindings);
            vm_promise_eval_unwind_to(
                vm, vm->handler_stack[vm->n_handlers].promise_mark);
            vm->sp = vm->handler_stack[vm->n_handlers].sp;
            vm->fp = vm->handler_stack[vm->n_handlers].fp;
            vm->frame_count = vm->handler_stack[vm->n_handlers].frame_count;
            vm->pc = vm->handler_stack[vm->n_handlers].pc;
            vm_escape_native_control(vm);
        } else {
            fprintf(stderr, "ERROR: unhandled exception: ");
            print_value(vm, exn);
            fprintf(stderr, "\n");
            vm->error = 1;
        }
        break;
    }

    case 132: { /* force: force a promise (thunk memoization) */
        Value promise = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (promise.type == VAL_FUTURE && promise.as.ptr >= 0 &&
            promise.as.ptr < vm->heap.next_free &&
            vm->heap.objects[promise.as.ptr]->type == HEAP_FUTURE) {
            VmFuture* handle = (VmFuture*)vm->heap.objects[promise.as.ptr]->opaque.ptr;
            vm_push(vm, vm_future_force(handle));
            break;
        }
#endif
        vm_push(vm, vm_force_promise_value(vm, promise));
        break;
    }

    case VM_NATIVE_PROMISE_CREATE: {
        Value promise = vm_pop(vm);
        if (promise.type != VAL_VECTOR ||
            !is_valid_heap_ptr(vm, promise.as.ptr) ||
            !vm->heap.objects[promise.as.ptr] ||
            vm->heap.objects[promise.as.ptr]->type != HEAP_VECTOR) {
            fprintf(stderr, "ERROR: promise constructor received invalid storage\n");
            vm->error = 1;
            vm_push(vm, NIL_VAL);
            break;
        }
        VmVector* slots =
            (VmVector*)vm->heap.objects[promise.as.ptr]->opaque.ptr;
        if (!slots || slots->len != 3) {
            fprintf(stderr, "ERROR: promise constructor requires three slots\n");
            vm->error = 1;
            vm_push(vm, NIL_VAL);
            break;
        }
        vm->heap.objects[promise.as.ptr]->type = HEAP_PROMISE;
        vm_push(vm, promise);
        break;
    }

    case VM_NATIVE_PROMISE_P: {
        Value value = vm_pop(vm);
        vm_push(vm, BOOL_VAL(vm_is_promise_value(vm, value, NULL)));
        break;
    }

    case 135: { /* append */
        Value b = vm_pop(vm), a = vm_pop(vm);
        if (a.type == VAL_NIL) { vm_push(vm, b); break; }
        if (a.type != VAL_PAIR) { vm_push(vm, b); break; }
        /* Copy list a, set last cdr to b */
        Value head = NIL_VAL, tail = NIL_VAL;
        Value cur = a;
        while (cur.type == VAL_PAIR) {
            int32_t p = heap_alloc(&vm->heap);
            if (p < 0) { vm->error = 1; break; }
            vm->heap.objects[p]->type = HEAP_CONS;
            vm->heap.objects[p]->cons.car = vm->heap.objects[cur.as.ptr]->cons.car;
            vm->heap.objects[p]->cons.cdr = NIL_VAL;
            Value node = PAIR_VAL(p);
            if (head.type == VAL_NIL) head = node;
            else vm->heap.objects[tail.as.ptr]->cons.cdr = node;
            tail = node;
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
        }
        if (tail.type == VAL_PAIR) vm->heap.objects[tail.as.ptr]->cons.cdr = b;
        vm_push(vm, head);
        break;
    }
    case 136: { /* reverse */
        Value lst = vm_pop(vm);
        Value result = NIL_VAL;
        while (lst.type == VAL_PAIR) {
            int32_t p = heap_alloc(&vm->heap);
            if (p < 0) { vm->error = 1; break; }
            vm->heap.objects[p]->type = HEAP_CONS;
            vm->heap.objects[p]->cons.car = vm->heap.objects[lst.as.ptr]->cons.car;
            vm->heap.objects[p]->cons.cdr = result;
            result = PAIR_VAL(p);
            lst = vm->heap.objects[lst.as.ptr]->cons.cdr;
        }
        vm_push(vm, result);
        break;
    }

    case 240: { /* display (native) */
        Value v = vm_pop(vm);
        print_value(vm, v);
        vm_push(vm, NIL_VAL);
        break;
    }
    case 241: { /* write (native) */
        Value v = vm_pop(vm);
        vm_write_value_port(vm, v, vm_port_current_output(), 1);
        vm_push(vm, NIL_VAL);
        break;
    }
    case 250: { /* atan2 */
        Value x = vm_pop(vm), y = vm_pop(vm);
        vm_push(vm, FLOAT_VAL(atan2(as_number(y), as_number(x))));
        break;
    }
    case 251: { /* call-with-values-apply: unpack multi-value result */
        Value consumer = vm_pop(vm), result = vm_pop(vm);
        if (consumer.type == VAL_CLOSURE) {
            /* Only an opaque multiple-values packet is unpacked.  An ordinary
             * Scheme vector is one value and must be passed as one argument. */
            if (result.type == VAL_MULTI_VALUE &&
                is_heap_type(vm, result, HEAP_MULTI_VALUE)) {
                VmVector* mv = (VmVector*)vm->heap.objects[result.as.ptr]->opaque.ptr;
                if (mv && mv->len > 0) {
                    Value r = vm_call_closure_from_native(vm, consumer, mv->items, mv->len);
                    vm_push(vm, r);
                } else {
                    Value r = vm_call_closure_from_native(vm, consumer, NULL, 0);
                    vm_push(vm, r);
                }
            } else {
                Value args[1] = {result};
                Value r = vm_call_closure_from_native(vm, consumer, args, 1);
                vm_push(vm, r);
            }
        } else vm_push(vm, result);
        break;
    }
    case 252: { /* propagate a parent's open upvalue slot into a child closure */
        Value slot_v = vm_pop(vm), uv_idx_v = vm_pop(vm), cl_val = vm_pop(vm);
        if (cl_val.type == VAL_CLOSURE) {
            HeapObject* cl = vm->heap.objects[cl_val.as.ptr];
            int uv_idx = (int)as_number(uv_idx_v);
            int slot = (int)as_number(slot_v);
            if (vm->fp > 0) {
                Value parent_val = vm->stack[vm->fp - 1];
                if (parent_val.type == VAL_CLOSURE) {
                    HeapObject* parent_cl = vm->heap.objects[parent_val.as.ptr];
                    if (uv_idx >= 0 && uv_idx < cl->closure.n_upvalues &&
                        slot >= 0 && slot < parent_cl->closure.n_upvalues) {
                        int32_t parent_open = parent_cl->closure.open_slots[slot];
                        if (parent_open >= 0)
                            cl->closure.open_slots[uv_idx] = parent_open;
                        else
                            cl->closure.upvalues[uv_idx] = parent_cl->closure.upvalues[slot];
                    }
                }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    case 237: { /* error */
        Value msg = vm_pop(vm);
        fprintf(stderr, "ERROR: ");
        print_value(vm, msg);
        fprintf(stderr, "\n");
        vm->error = 1;
        break;
    }
    case 238: { /* void */
        vm_push(vm, NIL_VAL);
        break;
    }
    case 235: { /* not */
        Value v = vm_pop(vm);
        vm_push(vm, BOOL_VAL(!is_truthy(v)));
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Character operations (1680-1691)
     * ══════════════════════════════════════════════════════════════════════ */
    case 1680: { Value a = vm_pop(vm); int c = (int)as_number(a); vm_push(vm, BOOL_VAL(isalpha(c))); break; }
    case 1681: { Value a = vm_pop(vm); int c = (int)as_number(a); vm_push(vm, BOOL_VAL(isdigit(c))); break; }
    case 1682: { Value a = vm_pop(vm); int c = (int)as_number(a); vm_push(vm, BOOL_VAL(isspace(c))); break; }
    case 1683: { Value a = vm_pop(vm); int c = (int)as_number(a); vm_push(vm, BOOL_VAL(isupper(c))); break; }
    case 1684: { Value a = vm_pop(vm); int c = (int)as_number(a); vm_push(vm, BOOL_VAL(islower(c))); break; }
    case 1685: { Value a = vm_pop(vm); int c = (int)as_number(a); vm_push(vm, (Value){.type = VAL_CHAR, .as.i = toupper(c)}); break; }
    case 1686: { Value a = vm_pop(vm); int c = (int)as_number(a); vm_push(vm, (Value){.type = VAL_CHAR, .as.i = tolower(c)}); break; }
    case 1687: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, BOOL_VAL((int)as_number(a) == (int)as_number(b))); break; }
    case 1688: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, BOOL_VAL((int)as_number(a) < (int)as_number(b))); break; }
    case 1689: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, BOOL_VAL((int)as_number(a) > (int)as_number(b))); break; }

    case 184: { /* symbol->string: preserve immutable spelling, change Scheme tag */
        Value a = vm_pop(vm);
        if (a.type == VAL_SYMBOL && vm_value_as_string(vm, a)) a.type = VAL_STRING;
        else a = BOOL_VAL(0);
        vm_push(vm, a);
        break;
    }
    case 185: { /* string->symbol: preserve immutable spelling, change Scheme tag */
        Value a = vm_pop(vm);
        if (a.type == VAL_STRING && vm_value_as_string(vm, a)) a.type = VAL_SYMBOL;
        else a = BOOL_VAL(0);
        vm_push(vm, a);
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Bitwise operations (1692-1696)
     * ══════════════════════════════════════════════════════════════════════ */
    case 1692: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, INT_VAL((int64_t)as_number(a) & (int64_t)as_number(b))); break; }
    case 1693: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, INT_VAL((int64_t)as_number(a) | (int64_t)as_number(b))); break; }
    case 1694: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, INT_VAL((int64_t)as_number(a) ^ (int64_t)as_number(b))); break; }
    case 1695: { Value a = vm_pop(vm); vm_push(vm, INT_VAL(~(int64_t)as_number(a))); break; }
    case 1696: { Value b = vm_pop(vm), a = vm_pop(vm);
        int64_t val = (int64_t)as_number(a), shift = (int64_t)as_number(b);
        vm_push(vm, INT_VAL(shift >= 0 ? val << shift : val >> (-shift))); break; }

    /* ══════════════════════════════════════════════════════════════════════
     * Type predicates (1697-1699)
     * ══════════════════════════════════════════════════════════════════════ */
    case 1697: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_INT || a.type == VAL_FLOAT || a.type == VAL_RATIONAL)); break; }
    case 1698: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_RATIONAL)); break; }
    case 1699: { Value a = vm_pop(vm);
        vm_push(vm, BOOL_VAL(a.type == VAL_TENSOR)); break; }

    /* ══════════════════════════════════════════════════════════════════════
     * Additional predicates (160-166)
     * ══════════════════════════════════════════════════════════════════════ */
    case 160: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_SYMBOL)); break; } /* symbol? */
    case 161: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_CHAR)); break; } /* char? */
    case 162: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_INT || a.type == VAL_RATIONAL)); break; } /* exact? */
    case 163: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_FLOAT)); break; } /* inexact? */
    case 164: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_FLOAT && isnan(a.as.f))); break; } /* nan? */
    case 165: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_FLOAT && isinf(a.as.f))); break; } /* infinite? */
    case 166: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type != VAL_FLOAT || isfinite(a.as.f))); break; } /* finite? */

    /* ══════════════════════════════════════════════════════════════════════
     * Additional list ops (186-189)
     * ══════════════════════════════════════════════════════════════════════ */
    case 186: { /* list-ref */
        Value idx = vm_pop(vm), lst = vm_pop(vm);
        int n = (int)as_number(idx);
        while (n > 0 && lst.type == VAL_PAIR) { lst = vm->heap.objects[lst.as.ptr]->cons.cdr; n--; }
        vm_push(vm, (lst.type == VAL_PAIR) ? vm->heap.objects[lst.as.ptr]->cons.car : NIL_VAL);
        break; }
    case 187: { /* list-tail */
        Value idx = vm_pop(vm), lst = vm_pop(vm);
        int n = (int)as_number(idx);
        while (n > 0 && lst.type == VAL_PAIR) { lst = vm->heap.objects[lst.as.ptr]->cons.cdr; n--; }
        vm_push(vm, lst); break; }
    case 188: { /* last-pair */
        Value lst = vm_pop(vm);
        if (lst.type != VAL_PAIR) { vm_push(vm, lst); break; }
        while (vm->heap.objects[lst.as.ptr]->cons.cdr.type == VAL_PAIR)
            lst = vm->heap.objects[lst.as.ptr]->cons.cdr;
        vm_push(vm, lst); break; }
    case 189: { /* list? */
        Value lst = vm_pop(vm);
        int is_list = (lst.type == VAL_NIL);
        if (!is_list) {
            Value cur = lst; int limit = 10000;
            while (cur.type == VAL_PAIR && limit-- > 0) cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
            is_list = (cur.type == VAL_NIL);
        }
        vm_push(vm, BOOL_VAL(is_list)); break; }

    /* Complex ops 300-319 already implemented above (line ~218) */

    /* ══════════════════════════════════════════════════════════════════════
     * Math extensions (720-746)
     * ══════════════════════════════════════════════════════════════════════ */
    case 720: { Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(cosh(as_number(a)))); break; }
    case 721: { Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(sinh(as_number(a)))); break; }
    case 722: { Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(tanh(as_number(a)))); break; }
    case 726: { /* write-line */
        Value s = vm_pop(vm);
        if (s.type == VAL_STRING) { VmString* vs = (VmString*)vm->heap.objects[s.as.ptr]->opaque.ptr;
            if (vs) { printf("%.*s\n", vs->byte_len, vs->data); fflush(stdout); } }
        vm_push(vm, NIL_VAL); break; }
    case 728: { /* input-port? */
        Value a = vm_pop(vm);
        VmPort* p = vm_value_as_port(vm, a);
        int is_ip = (p && p->dir == VM_PORT_INPUT);
        vm_push(vm, BOOL_VAL(is_ip)); break; }
    case 729: { /* output-port? */
        Value a = vm_pop(vm);
        VmPort* p = vm_value_as_port(vm, a);
        int is_op = (p && p->dir == VM_PORT_OUTPUT);
        vm_push(vm, BOOL_VAL(is_op)); break; }
    case 730: { /* port? */
        Value a = vm_pop(vm);
        vm_push(vm, BOOL_VAL(vm_value_as_port(vm, a) != NULL)); break; }
    case 740: { /* type-of */
        Value a = vm_pop(vm);
        const char* t = "unknown";
        switch ((int)a.type) {
            case VAL_NIL: t = "nil"; break; case VAL_INT: t = "integer"; break;
            case VAL_FLOAT: t = "float"; break; case VAL_BOOL: t = "boolean"; break;
            case VAL_PAIR: t = "pair"; break; case VAL_CLOSURE: t = "procedure"; break;
            case VAL_STRING: t = "string"; break; case VAL_SYMBOL: t = "symbol"; break;
            case VAL_VECTOR: t = "vector"; break;
            case VAL_COMPLEX: t = "complex"; break; case VAL_RATIONAL: t = "rational"; break;
            case VAL_FUTURE: t = "future"; break;
        }
        VmString* s = vm_string_from_cstr(&vm->heap.regions, t);
        if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); }
        else vm_push(vm, NIL_VAL); break; }
    case 743: { Value a = vm_pop(vm); double v = as_number(a);
        vm_push(vm, INT_VAL(v > 0 ? 1 : (v < 0 ? -1 : 0))); break; }
    case 745: { /* eye(n) — identity matrix as n×n tensor */
        Value n_val = vm_pop(vm);
        int n = (int)as_number(n_val);
        if (n <= 0 || n > 1024) { vm_push(vm, NIL_VAL); break; }
        int64_t shape[2] = {n, n};
        VmTensor* t = vm_tensor_zeros(&vm->heap.regions, shape, 2);
        if (!t) { vm_push(vm, NIL_VAL); break; }
        for (int i = 0; i < n; i++) t->data[i * n + i] = 1.0;
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_TENSOR, VAL_TENSOR, t);
        break; }
    case 746: { /* linspace(start, stop, n) — n evenly spaced points */
        Value n_val = vm_pop(vm), stop_val = vm_pop(vm), start_val = vm_pop(vm);
        double s = as_number(start_val), e = as_number(stop_val);
        int n = (int)as_number(n_val);
        if (n <= 0 || n > 100000) { vm_push(vm, NIL_VAL); break; }
        int64_t shape[1] = {n};
        VmTensor* t = vm_tensor_zeros(&vm->heap.regions, shape, 1);
        if (!t) { vm_push(vm, NIL_VAL); break; }
        double step = (n > 1) ? (e - s) / (n - 1) : 0;
        for (int i = 0; i < n; i++) t->data[i] = s + i * step;
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_TENSOR, VAL_TENSOR, t);
        break; }

    /* ══════════════════════════════════════════════════════════════════════
     * Higher-order native accelerated (900-910)
     * ══════════════════════════════════════════════════════════════════════ */
    case 900: { /* any(pred, list) */
        Value lst = vm_pop(vm), pred = vm_pop(vm);
        int found = 0;
        while (lst.type == VAL_PAIR) {
            Value car = vm->heap.objects[lst.as.ptr]->cons.car;
            Value r = vm_call_closure_from_native(vm, pred, &car, 1);
            if (is_truthy(r)) { found = 1; break; }
            lst = vm->heap.objects[lst.as.ptr]->cons.cdr;
        }
        vm_push(vm, BOOL_VAL(found)); break; }
    case 901: { /* every(pred, list) */
        Value lst = vm_pop(vm), pred = vm_pop(vm);
        int all = 1;
        while (lst.type == VAL_PAIR) {
            Value car = vm->heap.objects[lst.as.ptr]->cons.car;
            Value r = vm_call_closure_from_native(vm, pred, &car, 1);
            if (!is_truthy(r)) { all = 0; break; }
            lst = vm->heap.objects[lst.as.ptr]->cons.cdr;
        }
        vm_push(vm, BOOL_VAL(all)); break; }
    case 902: { /* find(pred, list) */
        Value lst = vm_pop(vm), pred = vm_pop(vm);
        while (lst.type == VAL_PAIR) {
            Value car = vm->heap.objects[lst.as.ptr]->cons.car;
            Value r = vm_call_closure_from_native(vm, pred, &car, 1);
            if (is_truthy(r)) { vm_push(vm, car); goto done_find; }
            lst = vm->heap.objects[lst.as.ptr]->cons.cdr;
        }
        vm_push(vm, BOOL_VAL(0));
        done_find: break; }
    case 903: { /* take(n, list) */
        Value lst = vm_pop(vm), n_val = vm_pop(vm);
        int n = (int)as_number(n_val);
        Value result = NIL_VAL;
        while (n > 0 && lst.type == VAL_PAIR) {
            Value car = vm->heap.objects[lst.as.ptr]->cons.car;
            int32_t p = heap_alloc(&vm->heap); if (p < 0) break;
            vm->heap.objects[p]->type = HEAP_CONS;
            vm->heap.objects[p]->cons.car = car; vm->heap.objects[p]->cons.cdr = result;
            result = PAIR_VAL(p);
            lst = vm->heap.objects[lst.as.ptr]->cons.cdr; n--;
        }
        /* reverse */
        Value rev = NIL_VAL;
        while (result.type == VAL_PAIR) {
            Value car = vm->heap.objects[result.as.ptr]->cons.car;
            int32_t rp = heap_alloc(&vm->heap); if (rp < 0) break;
            vm->heap.objects[rp]->type = HEAP_CONS;
            vm->heap.objects[rp]->cons.car = car; vm->heap.objects[rp]->cons.cdr = rev;
            rev = PAIR_VAL(rp);
            result = vm->heap.objects[result.as.ptr]->cons.cdr;
        }
        vm_push(vm, rev); break; }
    case 904: { /* drop(n, list) */
        Value lst = vm_pop(vm), n_val = vm_pop(vm);
        int n = (int)as_number(n_val);
        while (n > 0 && lst.type == VAL_PAIR) { lst = vm->heap.objects[lst.as.ptr]->cons.cdr; n--; }
        vm_push(vm, lst); break; }
    case 905: { /* string-reverse */
        Value s_val = vm_pop(vm);
        if (s_val.type == VAL_STRING) {
            VmString* s = (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr;
            if (s && s->byte_len > 0) {
                char* buf = (char*)vm_alloc(&vm->heap.regions, (size_t)(s->byte_len + 1));
                if (buf) { for (int i = 0; i < s->byte_len; i++) buf[i] = s->data[s->byte_len - 1 - i]; buf[s->byte_len] = 0;
                    VmString* r = vm_string_from_cstr(&vm->heap.regions, buf);
                    if (r) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, r); break; } } }
        }
        vm_push(vm, s_val); break; }
    case 906: { /* string-repeat(str, n) */
        Value n_val = vm_pop(vm), s_val = vm_pop(vm);
        if (s_val.type == VAL_STRING) {
            VmString* s = (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr;
            int n = (int)as_number(n_val);
            if (s && n > 0 && n < 10000 && s->byte_len > 0) {
                /* P1: byte_len * n overflows int32 for a large string even with
                   n < 10000 (byte_len can be ~2^31), giving an undersized alloc
                   then a memcpy overflow. Compute in int64 and bound to 64 MiB. */
                int64_t total64 = (int64_t)s->byte_len * (int64_t)n;
                if (total64 > 0 && total64 <= (int64_t)64 * 1024 * 1024) {
                    int total = (int)total64;
                    char* buf = (char*)vm_alloc(&vm->heap.regions, (size_t)(total + 1));
                    if (buf) { for (int i = 0; i < n; i++) memcpy(buf + i * s->byte_len, s->data, s->byte_len); buf[total] = 0;
                        VmString* r = vm_string_from_cstr(&vm->heap.regions, buf);
                        if (r) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, r); break; } }
                }
            }
        }
        vm_push(vm, s_val); break; }
    case 907: { /* string-trim */
        Value s_val = vm_pop(vm);
        if (s_val.type == VAL_STRING) {
            VmString* s = (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr;
            if (s) { int start = 0, end = s->byte_len;
                while (start < end && isspace((unsigned char)s->data[start])) start++;
                while (end > start && isspace((unsigned char)s->data[end-1])) end--;
                VmString* r = vm_string_new(&vm->heap.regions, s->data + start, end - start);
                if (r) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, r); break; } } }
        vm_push(vm, s_val); break; }
    case 908: { /* string-split(str, delim) */
        Value d_val = vm_pop(vm), s_val = vm_pop(vm);
        Value result = NIL_VAL;
        if (s_val.type == VAL_STRING && d_val.type == VAL_STRING) {
            VmString* s = (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr;
            VmString* d = (VmString*)vm->heap.objects[d_val.as.ptr]->opaque.ptr;
            if (s && d && d->byte_len > 0) {
                const char* p = s->data; int slen = s->byte_len, dlen = d->byte_len;
                while (1) {
                    const char* found = (slen >= dlen) ? strstr(p, d->data) : NULL;
                    int seg_len = found ? (int)(found - p) : (int)(s->data + slen - p);
                    VmString* seg = vm_string_new(&vm->heap.regions, p, seg_len);
                    if (seg) { int32_t sp2 = heap_alloc(&vm->heap); if (sp2 >= 0) {
                        vm->heap.objects[sp2]->type = HEAP_STRING; vm->heap.objects[sp2]->opaque.ptr = seg;
                        int32_t cp2 = heap_alloc(&vm->heap); if (cp2 >= 0) {
                            vm->heap.objects[cp2]->type = HEAP_CONS;
                            vm->heap.objects[cp2]->cons.car = (Value){.type = VAL_STRING, .as.ptr = sp2};
                            vm->heap.objects[cp2]->cons.cdr = result; result = PAIR_VAL(cp2); } } }
                    if (!found) break;
                    p = found + dlen;
                }
                /* reverse */
                Value rev2 = NIL_VAL;
                while (result.type == VAL_PAIR) {
                    Value car2 = vm->heap.objects[result.as.ptr]->cons.car;
                    int32_t rp2 = heap_alloc(&vm->heap); if (rp2 < 0) break;
                    vm->heap.objects[rp2]->type = HEAP_CONS;
                    vm->heap.objects[rp2]->cons.car = car2; vm->heap.objects[rp2]->cons.cdr = rev2;
                    rev2 = PAIR_VAL(rp2); result = vm->heap.objects[result.as.ptr]->cons.cdr;
                }
                vm_push(vm, rev2); break;
            }
        }
        vm_push(vm, NIL_VAL); break; }
    case 909: { /* string-join(list, delim) */
        Value d_val = vm_pop(vm), lst = vm_pop(vm);
        char buf[8192]; int pos = 0; int first = 1;
        const char* delim = "";  int dlen = 0;
        if (d_val.type == VAL_STRING) { VmString* ds = (VmString*)vm->heap.objects[d_val.as.ptr]->opaque.ptr;
            if (ds) { delim = ds->data; dlen = ds->byte_len; } }
        while (lst.type == VAL_PAIR && pos < 8000) {
            Value car = vm->heap.objects[lst.as.ptr]->cons.car;
            if (!first && dlen > 0 && pos + dlen < 8000) { memcpy(buf + pos, delim, dlen); pos += dlen; }
            first = 0;
            if (car.type == VAL_STRING) { VmString* cs = (VmString*)vm->heap.objects[car.as.ptr]->opaque.ptr;
                if (cs && pos + cs->byte_len < 8000) { memcpy(buf + pos, cs->data, cs->byte_len); pos += cs->byte_len; } }
            lst = vm->heap.objects[lst.as.ptr]->cons.cdr;
        }
        buf[pos] = 0;
        VmString* r = vm_string_from_cstr(&vm->heap.regions, buf);
        if (r) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, r); }
        else vm_push(vm, NIL_VAL); break; }

    /* ══════════════════════════════════════════════════════════════════════
     * System Information (1700-1719)
     * ══════════════════════════════════════════════════════════════════════ */

    case 1700: { /* os-type → "darwin", "linux", "windows", etc. */
#ifdef __APPLE__
        const char* os = "darwin";
#elif defined(_WIN32)
        const char* os = "windows";
#elif defined(__linux__)
        const char* os = "linux";
#elif defined(__FreeBSD__)
        const char* os = "freebsd";
#else
        const char* os = "unknown";
#endif
        VmString* s = vm_string_from_cstr(&vm->heap.regions, os);
        if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); }
        else { vm_push(vm, NIL_VAL); }
        break;
    }

    case 1701: { /* os-arch → "arm64", "x86_64", etc. */
#if defined(__aarch64__) || defined(_M_ARM64)
        const char* arch = "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
        const char* arch = "x86_64";
#elif defined(__i386__) || defined(_M_IX86)
        const char* arch = "x86";
#elif defined(__riscv)
        const char* arch = "riscv64";
#else
        const char* arch = "unknown";
#endif
        VmString* s = vm_string_from_cstr(&vm->heap.regions, arch);
        if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); }
        else { vm_push(vm, NIL_VAL); }
        break;
    }

    case 1702: { /* home-directory → "/Users/foo" or "/home/foo" */
#ifndef ESHKOL_VM_WASM
        const char* home = getenv("HOME");
#ifdef _WIN32
        if (!home) home = getenv("USERPROFILE");
#endif
        if (home) {
            VmString* s = vm_string_from_cstr(&vm->heap.regions, home);
            if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
        }
#endif
        vm_push(vm, NIL_VAL);
        break;
    }

    case 1703: { /* current-directory → cwd string */
#ifndef ESHKOL_VM_WASM
        char cwd[4096];
        if (getcwd(cwd, sizeof(cwd))) {
            VmString* s = vm_string_from_cstr(&vm->heap.regions, cwd);
            if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
        }
#endif
        vm_push(vm, NIL_VAL);
        break;
    }

    case 1704: { /* set-current-directory!(path) → #t or #f */
#ifndef ESHKOL_VM_WASM
        Value path_val = vm_pop(vm);
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps && chdir(ps->data) == 0) { vm_push(vm, BOOL_VAL(1)); break; }
        }
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1705: { /* hostname → string */
#ifndef ESHKOL_VM_WASM
        char hostname[256];
        if (gethostname(hostname, sizeof(hostname)) == 0) {
            VmString* s = vm_string_from_cstr(&vm->heap.regions, hostname);
            if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
        }
#endif
        vm_push(vm, NIL_VAL);
        break;
    }

    case 1706: { /* username → string */
#ifndef ESHKOL_VM_WASM
        const char* user = getenv("USER");
#ifdef _WIN32
        if (!user) user = getenv("USERNAME");
#endif
        if (user) {
            VmString* s = vm_string_from_cstr(&vm->heap.regions, user);
            if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
        }
#endif
        vm_push(vm, NIL_VAL);
        break;
    }

    case 1707: { /* cpu-count → integer */
#ifndef ESHKOL_VM_WASM
#ifdef _WIN32
        SYSTEM_INFO si; GetSystemInfo(&si);
        vm_push(vm, INT_VAL(si.dwNumberOfProcessors));
#elif defined(_SC_NPROCESSORS_ONLN)
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        vm_push(vm, INT_VAL(n > 0 ? n : 1));
#else
        vm_push(vm, INT_VAL(1));
#endif
#else
        vm_push(vm, INT_VAL(1));
#endif
        break;
    }

    case 1708: { /* executable-exists?(name) → #t or #f */
        Value name_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (name_val.type == VAL_STRING) {
            VmString* ns = (VmString*)vm->heap.objects[name_val.as.ptr]->opaque.ptr;
            if (ns) {
                /* Search PATH for the executable */
                const char* path_env = getenv("PATH");
                if (path_env) {
                    char buf[4096];
                    strncpy(buf, path_env, sizeof(buf) - 1); buf[sizeof(buf) - 1] = 0;
                    char* dir = strtok(buf, ":");
                    while (dir) {
                        char full[4096];
                        snprintf(full, sizeof(full), "%s/%s", dir, ns->data);
                        if (access(full, X_OK) == 0) {
                            vm_push(vm, BOOL_VAL(1)); goto done_1708;
                        }
                        dir = strtok(NULL, ":");
                    }
                }
            }
        }
#else
        (void)name_val;
#endif
        vm_push(vm, BOOL_VAL(0));
#ifndef ESHKOL_VM_WASM
        done_1708:
#endif
        break;
    }

    case 1709: { /* current-time-ms → integer (milliseconds since epoch) */
#ifndef ESHKOL_VM_WASM
        struct timeval tv;
        gettimeofday(&tv, NULL);
        int64_t ms = (int64_t)tv.tv_sec * 1000 + (int64_t)tv.tv_usec / 1000;
        vm_push(vm, INT_VAL(ms));
#else
        vm_push(vm, INT_VAL(0));
#endif
        break;
    }

    case 1710: { /* getpid → integer */
#ifndef ESHKOL_VM_WASM
        vm_push(vm, INT_VAL((int64_t)getpid()));
#else
        vm_push(vm, INT_VAL(0));
#endif
        break;
    }

    case 1711: { /* sleep-ms(milliseconds) → void */
        Value ms_val = vm_pop(vm);
        /* ESH-0228: type-check the argument. as_number() silently returns 0.0
         * for non-numeric values, so a bad argument used to no-op instead of
         * signalling an error; match the AOT/JIT path and reject non-numbers
         * cleanly (the VM's runtime-error idiom: message + vm->error = 1). */
        if (ms_val.type != VAL_INT && ms_val.type != VAL_FLOAT) {
            fprintf(stderr, "ERROR: sleep-ms: expected a number\n");
            vm->error = 1;
            break;
        }
#ifndef ESHKOL_VM_WASM
        int64_t ms = (int64_t)as_number(ms_val);
        if (ms > 0) {
            struct timespec ts;
            ts.tv_sec = ms / 1000;
            ts.tv_nsec = (ms % 1000) * 1000000L;
            nanosleep(&ts, NULL);
        }
#else
        (void)ms_val;
#endif
        vm_push(vm, NIL_VAL);
        break;
    }

    case 1712: { /* setenv(name, value) → #t or #f */
        Value val_v = vm_pop(vm), name_v = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (name_v.type == VAL_STRING && val_v.type == VAL_STRING) {
            VmString* ns = (VmString*)vm->heap.objects[name_v.as.ptr]->opaque.ptr;
            VmString* vs = (VmString*)vm->heap.objects[val_v.as.ptr]->opaque.ptr;
            if (ns && vs && setenv(ns->data, vs->data, 1) == 0) {
                vm_push(vm, BOOL_VAL(1)); break;
            }
        }
#else
        (void)val_v; (void)name_v;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1713: { /* unsetenv(name) → #t or #f */
        Value name_v = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (name_v.type == VAL_STRING) {
            VmString* ns = (VmString*)vm->heap.objects[name_v.as.ptr]->opaque.ptr;
            if (ns && unsetenv(ns->data) == 0) {
                vm_push(vm, BOOL_VAL(1)); break;
            }
        }
#else
        (void)name_v;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1714: { /* current-error-port → port for stderr */
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_PORT, VAL_PORT, vm_port_current_error());
        break;
    }

    case 1715: { /* get-environment-variable(name) → string or #f */
        Value name_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (name_val.type == VAL_STRING) {
            VmString* ns = (VmString*)vm->heap.objects[name_val.as.ptr]->opaque.ptr;
            if (ns) {
                const char* val = getenv(ns->data);
                if (val) {
                    VmString* s = vm_string_from_cstr(&vm->heap.regions, val);
                    if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
                }
            }
        }
#else
        (void)name_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1716: { /* delete-file(path) — alias for 600 but with proper registration */
        Value path_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps && unlink(ps->data) == 0) { vm_push(vm, BOOL_VAL(1)); break; }
        }
#else
        (void)path_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Path Manipulation (1720-1739)
     * ══════════════════════════════════════════════════════════════════════ */

    case 1720: { /* path-join(a, b) → string */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        if (a_val.type == VAL_STRING && b_val.type == VAL_STRING) {
            VmString* as = (VmString*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
            VmString* bs = (VmString*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
            if (as && bs) {
                char buf[4096];
                int alen = (int)strlen(as->data);
                if (alen > 0 && vm_path_is_separator(as->data[alen - 1]))
                    snprintf(buf, sizeof(buf), "%s%s", as->data, bs->data);
                else
                    snprintf(buf, sizeof(buf), "%s%c%s", as->data, vm_path_separator(), bs->data);
                VmString* s = vm_string_from_cstr(&vm->heap.regions, buf);
                if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    case 1721: { /* path-dirname(path) → string */
        Value path_val = vm_pop(vm);
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps) {
                char buf[4096];
                strncpy(buf, ps->data, sizeof(buf) - 1); buf[sizeof(buf) - 1] = 0;
                /* Find last path separator. */
                char* last_slash = (char*)vm_path_last_separator(buf);
                if (last_slash) {
#ifdef _WIN32
                    if (last_slash == buf ||
                        (last_slash == buf + 2 && isalpha((unsigned char)buf[0]) && buf[1] == ':')) {
                        last_slash[1] = 0; /* root "\\" or "C:\\" */
                    } else {
                        *last_slash = 0;
                    }
#else
                    if (last_slash == buf) buf[1] = 0; /* root "/" */
                    else *last_slash = 0;
#endif
                } else {
                    strcpy(buf, ".");
                }
                VmString* s = vm_string_from_cstr(&vm->heap.regions, buf);
                if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    case 1722: { /* path-basename(path) → string */
        Value path_val = vm_pop(vm);
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps) {
                const char* last_slash = vm_path_last_separator(ps->data);
                const char* base = last_slash ? last_slash + 1 : ps->data;
                VmString* s = vm_string_from_cstr(&vm->heap.regions, base);
                if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    case 1723: { /* path-extname(path) → string (e.g., ".txt") or "" */
        Value path_val = vm_pop(vm);
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps) {
                const char* base = vm_path_last_separator(ps->data);
                if (!base) base = ps->data; else base++;
                const char* dot = strrchr(base, '.');
                const char* ext = (dot && dot != base) ? dot : "";
                VmString* s = vm_string_from_cstr(&vm->heap.regions, ext);
                if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    case 1724: { /* path-is-absolute?(path) → #t or #f */
        Value path_val = vm_pop(vm);
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps && vm_path_is_absolute_native(ps->data)) {
                vm_push(vm, BOOL_VAL(1)); break;
            }
        }
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1725: { /* path-normalize(path) → string with resolved . and .. */
        Value path_val = vm_pop(vm);
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps) {
                char result[4096];
                if (vm_path_normalize_cstr(ps->data, result, sizeof(result))) {
                    VmString* s = vm_string_from_cstr(&vm->heap.regions, result);
                    if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
                }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    case 1726: { /* realpath(path) → resolved absolute path string */
        Value path_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps) {
#ifdef _WIN32
                char resolved[4096];
                DWORD written = GetFullPathNameA(ps->data, (DWORD)sizeof(resolved), resolved, NULL);
                if (written > 0 && written < (DWORD)sizeof(resolved)) {
                    VmString* s = vm_string_from_cstr(&vm->heap.regions, resolved);
                    if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
                }
#else
                char resolved[4096];
                if (realpath(ps->data, resolved)) {
                    VmString* s = vm_string_from_cstr(&vm->heap.regions, resolved);
                    if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
                }
#endif
            }
        }
#else
        (void)path_val;
#endif
        vm_push(vm, NIL_VAL);
        break;
    }

    case 1727: { /* path-relative(from, to) → string */
        Value to_val = vm_pop(vm), from_val = vm_pop(vm);
        if (from_val.type == VAL_STRING && to_val.type == VAL_STRING) {
            VmString* fs = (VmString*)vm->heap.objects[from_val.as.ptr]->opaque.ptr;
            VmString* ts = (VmString*)vm->heap.objects[to_val.as.ptr]->opaque.ptr;
            if (fs && ts) {
                char result[4096];
                if (vm_path_relative_cstr(fs->data, ts->data, result, sizeof(result))) {
                    VmString* s = vm_string_from_cstr(&vm->heap.regions, result);
                    if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
                }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    case 1728: { /* path-resolve(base, rel) → normalized absolute/relative path */
        Value rel_val = vm_pop(vm), base_val = vm_pop(vm);
        if (base_val.type == VAL_STRING && rel_val.type == VAL_STRING) {
            VmString* bs = (VmString*)vm->heap.objects[base_val.as.ptr]->opaque.ptr;
            VmString* rs = (VmString*)vm->heap.objects[rel_val.as.ptr]->opaque.ptr;
            if (bs && rs) {
                char joined[4096];
                char result[4096];
                const char* input = rs->data;
                if (!vm_path_is_absolute_native(rs->data)) {
                    int n = snprintf(joined, sizeof(joined), "%s%c%s",
                                     bs->data, vm_path_separator(), rs->data);
                    if (n <= 0 || n >= (int)sizeof(joined)) {
                        vm_push(vm, NIL_VAL);
                        break;
                    }
                    input = joined;
                }
                if (vm_path_normalize_cstr(input, result, sizeof(result))) {
                    VmString* s = vm_string_from_cstr(&vm->heap.regions, result);
                    if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
                }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Filesystem Operations (1740-1769)
     * ══════════════════════════════════════════════════════════════════════ */

    case 1740: { /* file-size(path) → integer (bytes) or #f */
        Value path_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps) {
                struct stat st;
                if (stat(ps->data, &st) == 0) {
                    vm_push(vm, INT_VAL((int64_t)st.st_size));
                    break;
                }
            }
        }
#else
        (void)path_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1741: { /* file-stat(path) → list: (size mtime-sec type-char) or #f
                  * type-char: 'f' = regular, 'd' = directory, 'l' = symlink, '?' = other */
        Value path_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps) {
                struct stat st;
                if (lstat(ps->data, &st) == 0) {
                    char type_ch = '?';
                    if (S_ISREG(st.st_mode)) type_ch = 'f';
                    else if (S_ISDIR(st.st_mode)) type_ch = 'd';
                    else if (S_ISLNK(st.st_mode)) type_ch = 'l';

                    /* Build list: (size mtime type-string) */
                    char type_str[2] = { type_ch, 0 };
                    VmString* ts = vm_string_from_cstr(&vm->heap.regions, type_str);
                    if (!ts) { vm_push(vm, BOOL_VAL(0)); break; }

                    /* Build cons cells: (type . nil) → (mtime . ...) → (size . ...) */
                    int32_t t_sp = heap_alloc(&vm->heap); if (t_sp < 0) { vm_push(vm, BOOL_VAL(0)); break; }
                    vm->heap.objects[t_sp]->type = HEAP_STRING;
                    vm->heap.objects[t_sp]->opaque.ptr = ts;

                    int32_t c3 = heap_alloc(&vm->heap); if (c3 < 0) { vm_push(vm, BOOL_VAL(0)); break; }
                    vm->heap.objects[c3]->type = HEAP_CONS;
                    vm->heap.objects[c3]->cons.car = (Value){.type = VAL_STRING, .as.ptr = t_sp};
                    vm->heap.objects[c3]->cons.cdr = NIL_VAL;

                    int32_t c2 = heap_alloc(&vm->heap); if (c2 < 0) { vm_push(vm, BOOL_VAL(0)); break; }
                    vm->heap.objects[c2]->type = HEAP_CONS;
                    vm->heap.objects[c2]->cons.car = INT_VAL((int64_t)st.st_mtime);
                    vm->heap.objects[c2]->cons.cdr = PAIR_VAL(c3);

                    int32_t c1 = heap_alloc(&vm->heap); if (c1 < 0) { vm_push(vm, BOOL_VAL(0)); break; }
                    vm->heap.objects[c1]->type = HEAP_CONS;
                    vm->heap.objects[c1]->cons.car = INT_VAL((int64_t)st.st_size);
                    vm->heap.objects[c1]->cons.cdr = PAIR_VAL(c2);

                    vm_push(vm, PAIR_VAL(c1));
                    break;
                }
            }
        }
#else
        (void)path_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1742: { /* file-rename(old, new) → #t or #f */
        Value new_val = vm_pop(vm), old_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (old_val.type == VAL_STRING && new_val.type == VAL_STRING) {
            VmString* os = (VmString*)vm->heap.objects[old_val.as.ptr]->opaque.ptr;
            VmString* ns = (VmString*)vm->heap.objects[new_val.as.ptr]->opaque.ptr;
            if (os && ns) {
#ifdef _WIN32
                if (MoveFileExA(os->data, ns->data, MOVEFILE_REPLACE_EXISTING | MOVEFILE_COPY_ALLOWED)) {
                    vm_push(vm, BOOL_VAL(1)); break;
                }
#else
                if (rename(os->data, ns->data) == 0) {
                    vm_push(vm, BOOL_VAL(1)); break;
                }
#endif
            }
        }
#else
        (void)new_val; (void)old_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1743: { /* file-copy(src, dst) → #t or #f */
        Value dst_val = vm_pop(vm), src_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (src_val.type == VAL_STRING && dst_val.type == VAL_STRING) {
            VmString* ss = (VmString*)vm->heap.objects[src_val.as.ptr]->opaque.ptr;
            VmString* ds = (VmString*)vm->heap.objects[dst_val.as.ptr]->opaque.ptr;
            if (ss && ds) {
                FILE* fin = fopen(ss->data, "rb");
                FILE* fout = fin ? fopen(ds->data, "wb") : NULL;
                if (fin && fout) {
                    char cbuf[8192];
                    size_t n;
                    while ((n = fread(cbuf, 1, sizeof(cbuf), fin)) > 0)
                        fwrite(cbuf, 1, n, fout);
                    fclose(fin); fclose(fout);
                    vm_push(vm, BOOL_VAL(1)); break;
                }
                if (fin) fclose(fin);
                if (fout) fclose(fout);
            }
        }
#else
        (void)dst_val; (void)src_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1744: { /* mkdir-recursive(path) → #t or #f */
        Value path_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps) {
                char buf[4096];
                strncpy(buf, ps->data, sizeof(buf) - 1); buf[sizeof(buf) - 1] = 0;
                /* Create each component */
                int ok = 1;
                for (char* p = buf + 1; *p; p++) {
                    if (*p == '/') {
                        *p = 0;
                        if (mkdir(buf, 0755) != 0 && errno != EEXIST) { ok = 0; break; }
                        *p = '/';
                    }
                }
                if (ok && mkdir(buf, 0755) != 0 && errno != EEXIST) ok = 0;
                if (ok || errno == EEXIST) { vm_push(vm, BOOL_VAL(1)); break; }
            }
        }
#else
        (void)path_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1745: { /* file-chmod(path, mode) → #t or #f (mode is integer, e.g. 0o755 = 493) */
        Value mode_val = vm_pop(vm), path_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps && chmod(ps->data, (mode_t)as_number(mode_val)) == 0) {
                vm_push(vm, BOOL_VAL(1)); break;
            }
        }
#else
        (void)mode_val; (void)path_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1746: { /* symlink-create(target, linkpath) → #t or #f */
        Value link_val = vm_pop(vm), target_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (target_val.type == VAL_STRING && link_val.type == VAL_STRING) {
            VmString* ts = (VmString*)vm->heap.objects[target_val.as.ptr]->opaque.ptr;
            VmString* ls = (VmString*)vm->heap.objects[link_val.as.ptr]->opaque.ptr;
            if (ts && ls && symlink(ts->data, ls->data) == 0) {
                vm_push(vm, BOOL_VAL(1)); break;
            }
        }
#else
        (void)link_val; (void)target_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1747: { /* symlink-read(linkpath) → target string or #f */
        Value path_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps) {
                char buf[4096];
                ssize_t len = readlink(ps->data, buf, sizeof(buf) - 1);
                if (len > 0) {
                    buf[len] = 0;
                    VmString* s = vm_string_from_cstr(&vm->heap.regions, buf);
                    if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
                }
            }
        }
#else
        (void)path_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1748: { /* directory-walk(path) → flat list of all file paths (recursive) */
        Value path_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps) {
                /* BFS using a simple stack of directories to visit */
                Value result = NIL_VAL;
                char dirs[256][4096];
                int dir_count = 0;
                strncpy(dirs[0], ps->data, 4095); dirs[0][4095] = 0;
                dir_count = 1;
                int dir_idx = 0;
                while (dir_idx < dir_count && dir_count < 256) {
                    DIR* d = opendir(dirs[dir_idx]);
                    dir_idx++;
                    if (!d) continue;
                    struct dirent* ent;
                    while ((ent = readdir(d)) != NULL) {
                        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) continue;
                        char full[4096];
                        snprintf(full, sizeof(full), "%s/%s", dirs[dir_idx - 1], ent->d_name);
                        struct stat st;
                        if (stat(full, &st) == 0 && S_ISDIR(st.st_mode) && dir_count < 256) {
                            strncpy(dirs[dir_count], full, 4095);
                            dirs[dir_count][4095] = 0;
                            dir_count++;
                        }
                        /* Add to result list */
                        VmString* s = vm_string_from_cstr(&vm->heap.regions, full);
                        if (!s) continue;
                        int32_t sp = heap_alloc(&vm->heap); if (sp < 0) continue;
                        vm->heap.objects[sp]->type = HEAP_STRING;
                        vm->heap.objects[sp]->opaque.ptr = s;
                        int32_t cp = heap_alloc(&vm->heap); if (cp < 0) continue;
                        vm->heap.objects[cp]->type = HEAP_CONS;
                        vm->heap.objects[cp]->cons.car = (Value){.type = VAL_STRING, .as.ptr = sp};
                        vm->heap.objects[cp]->cons.cdr = result;
                        result = PAIR_VAL(cp);
                    }
                    closedir(d);
                }
                vm_push(vm, result);
                break;
            }
        }
#else
        (void)path_val;
#endif
        vm_push(vm, NIL_VAL);
        break;
    }

    case 1749: { /* directory-delete-recursive(path) → #t or #f */
        Value path_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps) {
                if (!vm_directory_delete_forbidden_root(ps->data) &&
                    vm_directory_delete_recursive_posix(ps->data, 0)) {
                    vm_push(vm, BOOL_VAL(1));
                    break;
                }
            }
        }
#else
        (void)path_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1750: { /* mkstemp(template) → (fd . path) or #f
                  * Template should end with XXXXXX */
        Value tmpl_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (tmpl_val.type == VAL_STRING) {
            VmString* ts = (VmString*)vm->heap.objects[tmpl_val.as.ptr]->opaque.ptr;
            if (ts) {
                char buf[4096];
                strncpy(buf, ts->data, sizeof(buf) - 1); buf[sizeof(buf) - 1] = 0;
                int fd = mkstemp(buf);
                if (fd >= 0) {
                    VmString* ps = vm_string_from_cstr(&vm->heap.regions, buf);
                    if (ps) {
                        int32_t sp = heap_alloc(&vm->heap); if (sp >= 0) {
                            vm->heap.objects[sp]->type = HEAP_STRING;
                            vm->heap.objects[sp]->opaque.ptr = ps;
                            int32_t cp = heap_alloc(&vm->heap); if (cp >= 0) {
                                vm->heap.objects[cp]->type = HEAP_CONS;
                                vm->heap.objects[cp]->cons.car = INT_VAL(fd);
                                vm->heap.objects[cp]->cons.cdr = (Value){.type = VAL_STRING, .as.ptr = sp};
                                vm_push(vm, PAIR_VAL(cp)); break;
                            }
                        }
                    }
                    close(fd);
                }
            }
        }
#else
        (void)tmpl_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1751: { /* mkdtemp(template) → path string or #f */
        Value tmpl_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (tmpl_val.type == VAL_STRING) {
            VmString* ts = (VmString*)vm->heap.objects[tmpl_val.as.ptr]->opaque.ptr;
            if (ts) {
                char buf[4096];
                strncpy(buf, ts->data, sizeof(buf) - 1); buf[sizeof(buf) - 1] = 0;
                if (mkdtemp(buf)) {
                    VmString* s = vm_string_from_cstr(&vm->heap.regions, buf);
                    if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
                }
            }
        }
#else
        (void)tmpl_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1752: { /* file-mtime(path) → integer seconds or #f */
        Value path_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps) {
                struct stat st;
                if (stat(ps->data, &st) == 0) {
                    vm_push(vm, INT_VAL((int64_t)st.st_mtime));
                    break;
                }
            }
        }
#else
        (void)path_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1753: { /* file-atime(path) → integer seconds or #f */
        Value path_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps) {
                struct stat st;
                if (stat(ps->data, &st) == 0) {
                    vm_push(vm, INT_VAL((int64_t)st.st_atime));
                    break;
                }
            }
        }
#else
        (void)path_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1754: { /* file-lock(fd) → #t or #f */
        Value fd_val = vm_pop(vm);
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
        struct flock fl;
        memset(&fl, 0, sizeof(fl));
        fl.l_type = F_WRLCK;
        fl.l_whence = SEEK_SET;
        if (fcntl((int)as_number(fd_val), F_SETLK, &fl) != -1) {
            vm_push(vm, BOOL_VAL(1));
            break;
        }
#else
        (void)fd_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1755: { /* file-unlock(fd) → #t or #f */
        Value fd_val = vm_pop(vm);
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
        struct flock fl;
        memset(&fl, 0, sizeof(fl));
        fl.l_type = F_UNLCK;
        fl.l_whence = SEEK_SET;
        if (fcntl((int)as_number(fd_val), F_SETLK, &fl) != -1) {
            vm_push(vm, BOOL_VAL(1));
            break;
        }
#else
        (void)fd_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1756: { /* glob-expand(pattern) → newline-separated string or nil */
        Value pattern_val = vm_pop(vm);
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
        if (pattern_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[pattern_val.as.ptr]->opaque.ptr;
            if (ps) {
                glob_t g;
                memset(&g, 0, sizeof(g));
                int rc = glob(ps->data, GLOB_NOSORT | GLOB_TILDE, NULL, &g);
                if (rc == 0 && g.gl_pathc > 0) {
                    size_t total = 0;
                    for (size_t i = 0; i < g.gl_pathc; i++)
                        total += strlen(g.gl_pathv[i]) + 1;
                    char* buf = (char*)malloc(total + 1);
                    if (buf) {
                        char* p = buf;
                        for (size_t i = 0; i < g.gl_pathc; i++) {
                            size_t len = strlen(g.gl_pathv[i]);
                            memcpy(p, g.gl_pathv[i], len);
                            p += len;
                            *p++ = '\n';
                        }
                        if (p > buf) p[-1] = 0;
                        else *p = 0;
                        VmString* s = vm_string_from_cstr(&vm->heap.regions, buf);
                        free(buf);
                        globfree(&g);
                        if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
                    } else {
                        globfree(&g);
                    }
                } else {
                    globfree(&g);
                }
            }
        }
#else
        (void)pattern_val;
#endif
        vm_push(vm, NIL_VAL);
        break;
    }

    case 1757: { /* glob-match(pattern, path) → #t or #f */
        Value path_val = vm_pop(vm), pattern_val = vm_pop(vm);
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
        if (pattern_val.type == VAL_STRING && path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[pattern_val.as.ptr]->opaque.ptr;
            VmString* ss = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps && ss && fnmatch(ps->data, ss->data, 0) == 0) {
                vm_push(vm, BOOL_VAL(1));
                break;
            }
        }
#else
        (void)path_val; (void)pattern_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1758: { /* file-mmap(path, offset, length) → bytevector or #f */
        Value len_val = vm_pop(vm), offset_val = vm_pop(vm), path_val = vm_pop(vm);
        int64_t offset = (int64_t)as_number(offset_val);
        int64_t len = (int64_t)as_number(len_val);
        if (path_val.type == VAL_STRING) {
            HeapObject* obj = VM_VALIDATE_HEAP(vm, path_val);
            VmString* ps = (obj && obj->type == HEAP_STRING)
                ? (VmString*)obj->opaque.ptr
                : NULL;
            if (ps) {
                VmBytevector* bv = vm_file_mmap_copy_to_bytevector(vm, ps->data, offset, len);
                if (bv) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_BYTEVECTOR, VAL_BYTEVECTOR, bv); break; }
            }
        }
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1759: { /* file-munmap(bytevector) → void
                  * The standalone VM returns arena-owned bytevectors from
                  * file-mmap after copying mapped bytes, so unmap is a no-op. */
        (void)vm_pop(vm);
        vm_push(vm, (Value){.type = VAL_VOID});
        break;
    }

    case 1760: { /* make-temp-file(prefix, suffix, dir) → path or #f */
        Value dir_val = vm_pop(vm), suffix_val = vm_pop(vm), prefix_val = vm_pop(vm);
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
        VmString* prefix = vm_value_as_string(vm, prefix_val);
        VmString* suffix = vm_value_as_string(vm, suffix_val);
        VmString* dir = vm_value_as_string(vm, dir_val);
        const char* dir_path = (dir && dir->data && dir->byte_len > 0) ? dir->data : getenv("TMPDIR");
        if (!dir_path || !*dir_path) dir_path = "/tmp";
        if (prefix && prefix->data && suffix && suffix->data) {
            const char* sep = (dir_path[strlen(dir_path) - 1] == '/') ? "" : "/";
            struct timeval tv;
            int created = 0;
            gettimeofday(&tv, NULL);
            for (int i = 0; i < 128; i++) {
                char path[4096];
                uint64_t nonce = ((uint64_t)(uint32_t)getpid() << 32) ^
                                 (uint64_t)tv.tv_usec ^ ((uint64_t)i * 0x9e3779b97f4a7c15ULL);
                int n = snprintf(path, sizeof(path), "%s%s%s%016llx%s",
                                 dir_path, sep, prefix->data,
                                 (unsigned long long)nonce, suffix->data);
                if (n <= 0 || n >= (int)sizeof(path)) break;
                int fd = open(path, O_CREAT | O_EXCL | O_RDWR, 0600);
                if (fd >= 0) {
                    close(fd);
                    Value path_value = vm_string_value(vm, path, -1);
                    if (path_value.type == VAL_STRING) {
                        vm_push(vm, path_value);
                        created = 1;
                    } else {
                        (void)unlink(path);
                    }
                    break;
                }
                if (errno != EEXIST) break;
            }
            if (created) break;
        }
#else
        (void)dir_val; (void)suffix_val; (void)prefix_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1761: { /* make-temp-dir(prefix, dir) → path or #f */
        Value dir_val = vm_pop(vm), prefix_val = vm_pop(vm);
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
        VmString* prefix = vm_value_as_string(vm, prefix_val);
        VmString* dir = vm_value_as_string(vm, dir_val);
        const char* dir_path = (dir && dir->data && dir->byte_len > 0) ? dir->data : getenv("TMPDIR");
        if (!dir_path || !*dir_path) dir_path = "/tmp";
        if (prefix && prefix->data) {
            const char* sep = (dir_path[strlen(dir_path) - 1] == '/') ? "" : "/";
            struct timeval tv;
            int created = 0;
            gettimeofday(&tv, NULL);
            for (int i = 0; i < 128; i++) {
                char path[4096];
                uint64_t nonce = ((uint64_t)(uint32_t)getpid() << 32) ^
                                 (uint64_t)tv.tv_usec ^ ((uint64_t)i * 0x9e3779b97f4a7c15ULL);
                int n = snprintf(path, sizeof(path), "%s%s%s%016llx",
                                 dir_path, sep, prefix->data,
                                 (unsigned long long)nonce);
                if (n <= 0 || n >= (int)sizeof(path)) break;
                if (mkdir(path, 0700) == 0) {
                    Value path_value = vm_string_value(vm, path, -1);
                    if (path_value.type == VAL_STRING) {
                        vm_push(vm, path_value);
                        created = 1;
                    } else {
                        (void)rmdir(path);
                    }
                    break;
                }
                if (errno != EEXIST) break;
            }
            if (created) break;
        }
#else
        (void)dir_val; (void)prefix_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Shell Utilities (1770-1779)
     * ══════════════════════════════════════════════════════════════════════ */

    case 1770: { /* shell-quote(str) → single-quoted shell-safe string */
        Value str_val = vm_pop(vm);
        if (str_val.type == VAL_STRING) {
            VmString* ss = (VmString*)vm->heap.objects[str_val.as.ptr]->opaque.ptr;
            if (ss) {
                /* POSIX shell quoting: wrap in single quotes, escape internal ' as '\'' */
                char buf[8192];
                int pos = 0;
                buf[pos++] = '\'';
                for (const char* c = ss->data; *c && pos < 8180; c++) {
                    if (*c == '\'') {
                        buf[pos++] = '\''; buf[pos++] = '\\';
                        buf[pos++] = '\''; buf[pos++] = '\'';
                    } else {
                        buf[pos++] = *c;
                    }
                }
                buf[pos++] = '\'';
                buf[pos] = 0;
                VmString* s = vm_string_from_cstr(&vm->heap.regions, buf);
                if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    case 1771: { /* shell-split(str) → list of strings (basic word splitting) */
        Value str_val = vm_pop(vm);
        if (str_val.type == VAL_STRING) {
            VmString* ss = (VmString*)vm->heap.objects[str_val.as.ptr]->opaque.ptr;
            if (ss) {
                Value result = NIL_VAL;
                char buf[4096];
                strncpy(buf, ss->data, sizeof(buf) - 1); buf[sizeof(buf) - 1] = 0;
                /* Collect words in reverse, then reverse the list */
                char* words[256]; int nwords = 0;
                char* p = buf;
                while (*p && nwords < 256) {
                    while (*p == ' ' || *p == '\t') p++;
                    if (!*p) break;
                    char quote = 0;
                    if (*p == '\'' || *p == '"') { quote = *p; p++; }
                    char* start = p;
                    if (quote) {
                        while (*p && *p != quote) p++;
                        if (*p) *p++ = 0;
                    } else {
                        while (*p && *p != ' ' && *p != '\t') p++;
                        if (*p) *p++ = 0;
                    }
                    words[nwords++] = start;
                }
                /* Build list in order */
                for (int i = nwords - 1; i >= 0; i--) {
                    VmString* ws = vm_string_from_cstr(&vm->heap.regions, words[i]);
                    if (!ws) continue;
                    int32_t sp = heap_alloc(&vm->heap); if (sp < 0) continue;
                    vm->heap.objects[sp]->type = HEAP_STRING;
                    vm->heap.objects[sp]->opaque.ptr = ws;
                    int32_t cp = heap_alloc(&vm->heap); if (cp < 0) continue;
                    vm->heap.objects[cp]->type = HEAP_CONS;
                    vm->heap.objects[cp]->cons.car = (Value){.type = VAL_STRING, .as.ptr = sp};
                    vm->heap.objects[cp]->cons.cdr = result;
                    result = PAIR_VAL(cp);
                }
                vm_push(vm, result);
                break;
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Process Management (1780-1799)
     * ══════════════════════════════════════════════════════════════════════ */

    /* ══════════════════════════════════════════════════════════════════════
     * Knowledge Base Extensions (1800-1809)
     * ══════════════════════════════════════════════════════════════════════ */

    case 1800: { /* kb-count(kb) → integer (number of facts) */
        Value kb_val = vm_pop(vm);
        if (is_heap_type(vm, kb_val, HEAP_KB)) {
            VmKnowledgeBase* kb = (VmKnowledgeBase*)vm->heap.objects[kb_val.as.ptr]->opaque.ptr;
            if (kb) { vm_push(vm, INT_VAL(kb->n_facts)); break; }
        }
        vm_push(vm, INT_VAL(0));
        break;
    }

    case 1801: { /* kb-retract!(kb, fact) -> #t or #f
                  * Remove first structurally matching fact from KB. */
        Value fact_val = vm_pop(vm), kb_val = vm_pop(vm);
        if (is_heap_type(vm, kb_val, HEAP_KB)) {
            VmKnowledgeBase* kb = (VmKnowledgeBase*)vm->heap.objects[kb_val.as.ptr]->opaque.ptr;
            Value target_datum;
            int has_target_datum = vm_kb_extract_fact_datum(vm, fact_val, &target_datum);
            if (kb && has_target_datum) {
                for (int i = 0; i < kb->n_facts; i++) {
                    Value stored_datum;
                    if (vm_kb_stored_fact_datum(vm, kb->facts[i], &stored_datum) &&
                        vm_values_equal_deep(vm, stored_datum, target_datum, 0)) {
                        kb->facts[i] = kb->facts[kb->n_facts - 1];
                        kb->n_facts--;
                        vm_push(vm, BOOL_VAL(1));
                        goto done_1801;
                    }
                }
            }
        }
        vm_push(vm, BOOL_VAL(0));
        done_1801:
        break;
    }

    case 1802: { /* kb-count-predicate(kb, predicate) -> integer */
        Value predicate = vm_pop(vm), kb_val = vm_pop(vm);
        int count = 0;
        if (is_heap_type(vm, kb_val, HEAP_KB)) {
            VmKnowledgeBase* kb = (VmKnowledgeBase*)vm->heap.objects[kb_val.as.ptr]->opaque.ptr;
            if (kb) {
                for (int i = 0; i < kb->n_facts; i++)
                    if (vm_kb_fact_predicate_matches(vm, kb->facts[i], predicate)) count++;
            }
        }
        vm_push(vm, INT_VAL(count));
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Factor Graph Extensions (1810-1819)
     * ══════════════════════════════════════════════════════════════════════ */

    case 1810: { /* fg-marginal(fg, var-idx) -> tensor of belief probabilities */
        Value idx_val = vm_pop(vm), fg_val = vm_pop(vm);
        if (is_heap_type(vm, fg_val, HEAP_FACTOR_GRAPH)) {
            VmFactorGraph* fg = (VmFactorGraph*)vm->heap.objects[fg_val.as.ptr]->opaque.ptr;
            int var = (int)as_number(idx_val);
            if (fg && var >= 0 && var < fg->num_vars) {
                int dim = fg->var_dims[var];
                int64_t shape[1] = { dim };
                VmTensor* t = vm_tensor_zeros(&vm->heap.regions, shape, 1);
                if (t) {
                    double sum = 0.0;
                    for (int i = 0; i < dim; i++) {
                        t->data[i] = exp(fg->beliefs[var][i]);
                        sum += t->data[i];
                    }
                    if (sum > 0.0) for (int i = 0; i < dim; i++) t->data[i] /= sum;
                    VM_PUSH_HEAP_OPAQUE(vm, HEAP_TENSOR, VAL_TENSOR, t);
                    break;
                }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    case 1811: { /* fg-entropy(fg, var-idx) -> scalar entropy H = -sum p*log(p) */
        Value idx_val = vm_pop(vm), fg_val = vm_pop(vm);
        if (is_heap_type(vm, fg_val, HEAP_FACTOR_GRAPH)) {
            VmFactorGraph* fg = (VmFactorGraph*)vm->heap.objects[fg_val.as.ptr]->opaque.ptr;
            int var = (int)as_number(idx_val);
            if (fg && var >= 0 && var < fg->num_vars) {
                int dim = fg->var_dims[var];
                double entropy = 0.0;
                for (int i = 0; i < dim; i++) {
                    double p = exp(fg->beliefs[var][i]);
                    if (p > 1e-15) entropy -= p * log(p);
                }
                vm_push(vm, FLOAT_VAL(entropy));
                break;
            }
        }
        vm_push(vm, FLOAT_VAL(0));
        break;
    }

    case 1812: { /* fg-total-entropy(fg) -> sum of variable marginal entropies */
        Value fg_val = vm_pop(vm);
        if (is_heap_type(vm, fg_val, HEAP_FACTOR_GRAPH)) {
            VmFactorGraph* fg = (VmFactorGraph*)vm->heap.objects[fg_val.as.ptr]->opaque.ptr;
            if (fg) {
                double entropy = 0.0;
                for (int var = 0; var < fg->num_vars; var++) {
                    for (int s = 0; s < fg->var_dims[var]; s++) {
                        double p = exp(fg->beliefs[var][s]);
                        if (p > 1e-15) entropy -= p * log(p);
                    }
                }
                vm_push(vm, FLOAT_VAL(entropy));
                break;
            }
        }
        vm_push(vm, FLOAT_VAL(0));
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Tensor/KB Persistence (1820-1829)
     * Binary format: [magic:4][version:4][ndims:4][shape:ndims*8][data:total*8]
     * ══════════════════════════════════════════════════════════════════════ */

#define TENSOR_FILE_MAGIC 0x45534B54 /* "ESKT" */

    case 1820: { /* tensor-save(path, tensor) → #t or #f */
        Value tensor_val = vm_pop(vm), path_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (path_val.type == VAL_STRING && tensor_val.type == VAL_TENSOR) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            VmTensor* t = (VmTensor*)vm->heap.objects[tensor_val.as.ptr]->opaque.ptr;
            if (ps && t && t->data) {
                FILE* f = fopen(ps->data, "wb");
                if (f) {
                    uint32_t magic = TENSOR_FILE_MAGIC;
                    uint32_t version = 1;
                    uint32_t ndims = (uint32_t)t->n_dims;
                    fwrite(&magic, 4, 1, f);
                    fwrite(&version, 4, 1, f);
                    fwrite(&ndims, 4, 1, f);
                    for (int i = 0; i < t->n_dims; i++) {
                        int64_t dim = t->shape[i];
                        fwrite(&dim, 8, 1, f);
                    }
                    fwrite(t->data, sizeof(double), (size_t)t->total, f);
                    fclose(f);
                    vm_push(vm, BOOL_VAL(1));
                    break;
                }
            }
        }
#else
        (void)tensor_val; (void)path_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1821: { /* tensor-load(path) → tensor or #f */
        Value path_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps) {
                FILE* f = fopen(ps->data, "rb");
                if (f) {
                    uint32_t magic, version, ndims;
                    if (fread(&magic, 4, 1, f) == 1 && magic == TENSOR_FILE_MAGIC &&
                        fread(&version, 4, 1, f) == 1 && version == 1 &&
                        fread(&ndims, 4, 1, f) == 1 && ndims > 0 && ndims <= 8) {
                        int64_t shape[8];
                        int ok = 1;
                        for (uint32_t i = 0; i < ndims; i++) {
                            if (fread(&shape[i], 8, 1, f) != 1) { ok = 0; break; }
                        }
                        if (ok) {
                            int64_t total = 1;
                            for (uint32_t i = 0; i < ndims; i++) total *= shape[i];
                            VmTensor* t = vm_tensor_new(&vm->heap.regions, shape, (int)ndims);
                            if (t && t->data && (int64_t)fread(t->data, sizeof(double), (size_t)total, f) == total) {
                                fclose(f);
                                VM_PUSH_HEAP_OPAQUE(vm, HEAP_TENSOR, VAL_TENSOR, t);
                                break;
                            }
                        }
                    }
                    fclose(f);
                }
            }
        }
#else
        (void)path_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1822: { /* kb-save(path, kb) → #t or #f
                  * Serializes KB: writes fact count + predicate hashes + arities as binary.
                  * For facts with datum (list), writes the list repr.
                  * Format: [magic:4][version:4][n_facts:4][per fact: predicate_hash:8, arity:4, datum_ptr:4] */
        Value kb_val = vm_pop(vm), path_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (path_val.type == VAL_STRING && kb_val.as.ptr >= 0 &&
            vm->heap.objects[kb_val.as.ptr]->type == HEAP_KB) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            VmKnowledgeBase* kb = (VmKnowledgeBase*)vm->heap.objects[kb_val.as.ptr]->opaque.ptr;
            if (ps && kb) {
                FILE* f = fopen(ps->data, "wb");
                if (f) {
                    uint32_t magic = 0x45534B42; /* "ESKB" */
                    uint32_t version = 1;
                    uint32_t nf = (uint32_t)kb->n_facts;
                    fwrite(&magic, 4, 1, f);
                    fwrite(&version, 4, 1, f);
                    fwrite(&nf, 4, 1, f);
                    for (int i = 0; i < kb->n_facts; i++) {
                        VmFact* fact = kb->facts[i];
                        if (!fact) { uint64_t z = 0; fwrite(&z, 8, 1, f); uint32_t za = 0; fwrite(&za, 4, 1, f); continue; }
                        fwrite(&fact->predicate, 8, 1, f);
                        uint32_t ar = (uint32_t)fact->arity;
                        fwrite(&ar, 4, 1, f);
                        /* Write each arg's type byte + data */
                        for (int j = 0; j < fact->arity; j++) {
                            fwrite(&fact->args[j].type, 1, 1, f);
                            fwrite(&fact->args[j].data, 8, 1, f);
                        }
                    }
                    fclose(f);
                    vm_push(vm, BOOL_VAL(1));
                    break;
                }
            }
        }
#else
        (void)kb_val; (void)path_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

#undef TENSOR_FILE_MAGIC

    /* ══════════════════════════════════════════════════════════════════════
     * Image I/O (1850-1859) — native platform/system codec based
     * ══════════════════════════════════════════════════════════════════════ */

    case 1850: { /* image-read(path) → tensor (H,W,C) or #f */
        Value path_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (path_val.type == VAL_STRING) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            if (ps) {
                int w, h, c;
                extern double* eshkol_image_read(const char*, int*, int*, int*);
                double* data = eshkol_image_read(ps->data, &w, &h, &c);
                if (data) {
                    int64_t shape[3] = { h, w, c };
                    int ndims = (c == 1) ? 2 : 3;
                    if (c == 1) { shape[0] = h; shape[1] = w; }
                    VmTensor* t = vm_tensor_from_data(&vm->heap.regions, data, shape, ndims);
                    /* image_io returns global-arena storage; ownership stays
                     * with that arena after the VM copies the pixels. */
                    if (t) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_TENSOR, VAL_TENSOR, t); break; }
                }
            }
        }
#else
        (void)path_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1851: { /* image-write(path, tensor, format) → #t or #f */
        Value fmt_val = vm_pop(vm), tensor_val = vm_pop(vm), path_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (path_val.type == VAL_STRING && tensor_val.type == VAL_TENSOR) {
            VmString* ps = (VmString*)vm->heap.objects[path_val.as.ptr]->opaque.ptr;
            VmTensor* t = (VmTensor*)vm->heap.objects[tensor_val.as.ptr]->opaque.ptr;
            const char* fmt = "png";
            if (fmt_val.type == VAL_STRING) {
                VmString* fs = (VmString*)vm->heap.objects[fmt_val.as.ptr]->opaque.ptr;
                if (fs) fmt = fs->data;
            }
            if (ps && t && t->data && t->n_dims >= 2) {
                int h = (int)t->shape[0], w = (int)t->shape[1];
                int c = (t->n_dims >= 3) ? (int)t->shape[2] : 1;
                extern int eshkol_image_write(const char*, const double*, int, int, int, const char*);
                if (eshkol_image_write(ps->data, t->data, w, h, c, fmt) == 0) {
                    vm_push(vm, BOOL_VAL(1)); break;
                }
            }
        }
#else
        (void)fmt_val; (void)tensor_val; (void)path_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1852: { /* image-to-grayscale(tensor) → tensor (H,W) or #f */
        Value tensor_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (tensor_val.type == VAL_TENSOR) {
            VmTensor* t = (VmTensor*)vm->heap.objects[tensor_val.as.ptr]->opaque.ptr;
            if (t && t->data && t->n_dims >= 2) {
                int h = (int)t->shape[0], w = (int)t->shape[1];
                int c = (t->n_dims >= 3) ? (int)t->shape[2] : 1;
                extern double* eshkol_image_to_grayscale(const double*, int, int, int);
                double* gray = eshkol_image_to_grayscale(t->data, w, h, c);
                if (gray) {
                    int64_t shape[2] = { h, w };
                    VmTensor* gt = vm_tensor_from_data(&vm->heap.regions, gray, shape, 2);
                    /* `gray` is global-arena-owned; never free it directly. */
                    if (gt) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_TENSOR, VAL_TENSOR, gt); break; }
                }
            }
        }
#else
        (void)tensor_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1853: { /* image-resize(tensor, new-h, new-w) → tensor or #f */
        Value nw_val = vm_pop(vm), nh_val = vm_pop(vm), tensor_val = vm_pop(vm);
#ifndef ESHKOL_VM_WASM
        if (tensor_val.type == VAL_TENSOR) {
            VmTensor* t = (VmTensor*)vm->heap.objects[tensor_val.as.ptr]->opaque.ptr;
            int new_h = (int)as_number(nh_val), new_w = (int)as_number(nw_val);
            if (t && t->data && t->n_dims >= 2 && new_h > 0 && new_w > 0) {
                int h = (int)t->shape[0], w = (int)t->shape[1];
                int c = (t->n_dims >= 3) ? (int)t->shape[2] : 1;
                extern double* eshkol_image_resize(const double*, int, int, int, int, int);
                double* resized = eshkol_image_resize(t->data, w, h, c, new_w, new_h);
                if (resized) {
                    int64_t shape[3] = { new_h, new_w, c };
                    int ndims = (c == 1) ? 2 : 3;
                    if (c == 1) { shape[0] = new_h; shape[1] = new_w; }
                    VmTensor* rt = vm_tensor_from_data(&vm->heap.regions, resized, shape, ndims);
                    /* `resized` is global-arena-owned; never free it directly. */
                    if (rt) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_TENSOR, VAL_TENSOR, rt); break; }
                }
            }
        }
#else
        (void)nw_val; (void)nh_val; (void)tensor_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * `quantum-random` family (1860-1862). Routed through eshkol_qrng_* (see
     * the honesty notice above this dispatch block's case labels) so the VM
     * and LLVM AOT/JIT backends agree on both the numbers and the entropy
     * source instead of each running its own generator.
     * ══════════════════════════════════════════════════════════════════════ */
    case 1860: { /* quantum-random() -> double in [0, 1) */
        vm_push(vm, FLOAT_VAL(eshkol_qrng_double()));
        break;
    }
    case 1861: { /* quantum-random-int(bound) -> integer in [0, bound) */
        Value bound_val = vm_pop(vm);
        int64_t bound = (int64_t)as_number(bound_val);
        if (bound <= 1) {
            vm_push(vm, INT_VAL(0));
        } else {
            vm_push(vm, INT_VAL((int64_t)(eshkol_qrng_uint64() % (uint64_t)bound)));
        }
        break;
    }
    case 1862: { /* quantum-random-range(min, max) */
        Value max_val = vm_pop(vm), min_val = vm_pop(vm);
        double lo = as_number(min_val);
        double hi = as_number(max_val);
        if (hi <= lo) {
            vm_push(vm, (min_val.type == VAL_INT && max_val.type == VAL_INT) ? INT_VAL((int64_t)lo) : FLOAT_VAL(lo));
            break;
        }
        if (min_val.type == VAL_INT && max_val.type == VAL_INT) {
            int64_t ilo = min_val.as.i;
            int64_t ihi = max_val.as.i;
            uint64_t span = (uint64_t)(ihi - ilo + 1);
            vm_push(vm, INT_VAL(ilo + (int64_t)(eshkol_qrng_uint64() % span)));
        } else {
            vm_push(vm, FLOAT_VAL(lo + eshkol_qrng_double() * (hi - lo)));
        }
        break;
    }

    case 1840: { /* reverse-gradient(f, point) → tensor of gradients
                  * Uses reverse-mode AD via Wengert tape tracing.
                  * Activates the tape, calls f with traced inputs,
                  * runs backward pass, returns gradient tensor.
                  * Single backward pass → O(1) regardless of input dimension. */
        Value x_val = vm_pop(vm), f_val = vm_pop(vm);

        double point[64];
        int n = 0;
        if (x_val.type == VAL_PAIR) {
            Value cur = x_val;
            while (cur.type == VAL_PAIR && n < 64) {
                point[n++] = as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
            }
        } else if (x_val.type == VAL_TENSOR && x_val.as.ptr >= 0) {
            VmTensor* t = (VmTensor*)vm->heap.objects[x_val.as.ptr]->opaque.ptr;
            if (t && t->data) {
                n = (int)(t->total < 64 ? t->total : 64);
                for (int i = 0; i < n; i++) point[i] = t->data[i];
            }
        } else {
            point[0] = as_number(x_val);
            n = 1;
        }

        if (n == 0) { vm_push(vm, FLOAT_VAL(0)); break; }

        /* Create tape and variable nodes */
        AdTape* tape = ad_tape_new(&vm->heap.regions);
        if (!tape) { vm_push(vm, FLOAT_VAL(0)); break; }
        vm->ad_tape_allocations++;

        int var_nodes[64];
        Value args[64];
        for (int i = 0; i < n; i++) {
            var_nodes[i] = ad_var(tape, point[i]);
            args[i] = FLOAT_VAL(point[i]);
        }

        /* Activate tape tracing on VM */
        void* saved_tape = vm->active_tape;
        vm->active_tape = tape;

        /* Set up ad_node_map for the argument slots.
         * The closure bridge pushes: closure, arg0, arg1, ..., argN-1
         * at stack positions sp, sp+1, ..., sp+N.
         * After frame setup, locals start at fp = sp+N-N = sp.
         * So arg[i] is at stack position (current_sp + 1 + i). */
        int base_sp = vm->sp + 1; /* +1 for closure push */
        for (int i = 0; i < n; i++) {
            if (base_sp + i < STACK_SIZE)
                vm->ad_node_map[base_sp + i] = var_nodes[i];
        }

        /* Call f(x1, x2, ..., xn) — arithmetic will record on tape */
        (void)vm_ad_call_closure(vm, f_val, args, n);

        /* Capture result's tape node (it's at the return value position) */
        /* The closure bridge captures result from stack[sp-1] before restoring sp.
         * At that point, ad_node_map[sp-1] holds the result node. But since sp
         * was already restored by the bridge, we need to find the output node.
         * The last node on the tape IS the output (tape nodes are appended in order). */
        int output_node = tape->len - 1;

        /* Deactivate tape */
        vm->active_tape = saved_tape;
        vm->ad_tape_nodes += (uint64_t)tape->len;

        if (output_node < 0) {
            /* Function didn't produce any tape operations — constant function */
            int64_t shape[1] = { n };
            VmTensor* zt = vm_tensor_zeros(&vm->heap.regions, shape, 1);
            if (zt) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_TENSOR, VAL_TENSOR, zt); }
            else { vm_push(vm, NIL_VAL); }
            break;
        }

        /* Run backward pass */
        ad_backward(tape, output_node);
        vm->ad_reverse_passes++;

        /* Collect gradients from variable nodes */
        if (n == 1) {
            vm_push(vm, FLOAT_VAL(ad_gradient(tape, var_nodes[0])));
        } else {
            double grads[64];
            for (int i = 0; i < n; i++)
                grads[i] = ad_gradient(tape, var_nodes[i]);
            int64_t shape[1] = { n };
            VmTensor* t = vm_tensor_from_data(&vm->heap.regions, grads, shape, 1);
            if (t) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_TENSOR, VAL_TENSOR, t); }
            else { vm_push(vm, NIL_VAL); }
        }
        break;
    }

    case 1830: { /* tensor-from-stack(v0, v1, ..., vN-1, count) → tensor
                  * Internal: compiler emits this for all-numeric #(...) literals.
                  * Stack layout (bottom → top): [val0, val1, ..., valN-1, count].
                  * The count is the top-of-stack and is popped first, so the
                  * element count is exact (no stack-distance heuristic — that
                  * mis-fired on literals like #(2 2 2) where an element value
                  * equalled its distance from TOS). */
        double vals[1024];
        Value count_val = vm_pop(vm);
        int n = (count_val.type == VAL_INT) ? (int)count_val.as.i
                                            : (int)as_number(count_val);
        if (n < 0 || n > 1024) { vm_push(vm, NIL_VAL); break; }
        if (n > 0) {
            for (int i = n - 1; i >= 0; i--)
                vals[i] = as_number(vm_pop(vm));
            int64_t shape[1] = { n };
            VmTensor* t = vm_tensor_from_data(&vm->heap.regions, vals, shape, 1);
            if (t) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_TENSOR, VAL_TENSOR, t); break; }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    case 1780: { /* process-spawn(cmd, args-list, env-alist) → pid or #f
                  * cmd: string, args: list of strings, env: alist of (name . value) or #f
                  * Returns child PID on success, #f on failure */
        Value env_val = vm_pop(vm), args_val = vm_pop(vm), cmd_val = vm_pop(vm);
#if !defined(ESHKOL_VM_WASM)
        if (cmd_val.type == VAL_STRING && is_valid_heap_ptr(vm, cmd_val.as.ptr)) {
            HeapObject* cmd_obj = vm->heap.objects[cmd_val.as.ptr];
            VmString* cs = cmd_obj ? (VmString*)cmd_obj->opaque.ptr : NULL;
            if (cs && cs->data) {
                char* argv_buf[256];
                int argc_local = 0;
                argv_buf[argc_local++] = cs->data;
                Value acur = args_val;
                while (acur.type == VAL_PAIR && argc_local < 255 && is_valid_heap_ptr(vm, acur.as.ptr)) {
                    HeapObject* node = vm->heap.objects[acur.as.ptr];
                    Value elem = node->cons.car;
                    if (elem.type == VAL_STRING && is_valid_heap_ptr(vm, elem.as.ptr)) {
                        HeapObject* elem_obj = vm->heap.objects[elem.as.ptr];
                        VmString* es = elem_obj ? (VmString*)elem_obj->opaque.ptr : NULL;
                        if (es && es->data) argv_buf[argc_local++] = es->data;
                    }
                    acur = node->cons.cdr;
                }
                argv_buf[argc_local] = NULL;

#ifdef _WIN32
                char cmdline[32768];
                if (vm_win_build_process_command_line(cmdline, sizeof(cmdline), argv_buf, argc_local)) {
                    char env_block[32768];
                    char* env_block_ptr = NULL;
                    size_t env_pos = 0;
                    Value ecur = env_val;
                    while (ecur.type == VAL_PAIR && env_pos + 2 < sizeof(env_block) &&
                           is_valid_heap_ptr(vm, ecur.as.ptr)) {
                        HeapObject* list_node = vm->heap.objects[ecur.as.ptr];
                        Value pair = list_node->cons.car;
                        if (pair.type == VAL_PAIR && is_valid_heap_ptr(vm, pair.as.ptr)) {
                            HeapObject* pair_obj = vm->heap.objects[pair.as.ptr];
                            Value key = pair_obj->cons.car;
                            Value val = pair_obj->cons.cdr;
                            if (key.type == VAL_STRING && val.type == VAL_STRING &&
                                is_valid_heap_ptr(vm, key.as.ptr) &&
                                is_valid_heap_ptr(vm, val.as.ptr)) {
                                VmString* ks = (VmString*)vm->heap.objects[key.as.ptr]->opaque.ptr;
                                VmString* vs = (VmString*)vm->heap.objects[val.as.ptr]->opaque.ptr;
                                if (ks && ks->data && vs && vs->data) {
                                    int n = snprintf(env_block + env_pos,
                                                     sizeof(env_block) - env_pos,
                                                     "%s=%s", ks->data, vs->data);
                                    if (n < 0 || (size_t)n + 2 > sizeof(env_block) - env_pos) {
                                        env_pos = 0;
                                        break;
                                    }
                                    env_pos += (size_t)n + 1;
                                    env_block_ptr = env_block;
                                }
                            }
                        }
                        ecur = list_node->cons.cdr;
                    }
                    if (env_block_ptr) env_block[env_pos++] = '\0';

                    STARTUPINFOA si;
                    PROCESS_INFORMATION pi;
                    memset(&si, 0, sizeof(si));
                    memset(&pi, 0, sizeof(pi));
                    si.cb = sizeof(si);
                    if (CreateProcessA(NULL, cmdline, NULL, NULL, FALSE,
                                       CREATE_NEW_PROCESS_GROUP,
                                       env_block_ptr, NULL, &si, &pi)) {
                        CloseHandle(pi.hThread);
                        DWORD pid = pi.dwProcessId;
                        CloseHandle(pi.hProcess);
                        vm_push(vm, INT_VAL((int64_t)pid));
                        break;
                    }
                }
#else
                /* Build environment if provided */
                char* envp_buf[256];
                char env_strs[256][512];
                char** envp = NULL;
                int envc = 0;
                if (env_val.type == VAL_PAIR) {
                    Value ecur = env_val;
                    while (ecur.type == VAL_PAIR && envc < 255 && is_valid_heap_ptr(vm, ecur.as.ptr)) {
                        HeapObject* list_node = vm->heap.objects[ecur.as.ptr];
                        Value pair = list_node->cons.car;
                        if (pair.type == VAL_PAIR && is_valid_heap_ptr(vm, pair.as.ptr)) {
                            HeapObject* pair_obj = vm->heap.objects[pair.as.ptr];
                            Value key = pair_obj->cons.car;
                            Value val = pair_obj->cons.cdr;
                            if (key.type == VAL_STRING && val.type == VAL_STRING &&
                                is_valid_heap_ptr(vm, key.as.ptr) &&
                                is_valid_heap_ptr(vm, val.as.ptr)) {
                                VmString* ks = (VmString*)vm->heap.objects[key.as.ptr]->opaque.ptr;
                                VmString* vs = (VmString*)vm->heap.objects[val.as.ptr]->opaque.ptr;
                                if (ks && vs) {
                                    snprintf(env_strs[envc], 512, "%s=%s", ks->data, vs->data);
                                    envp_buf[envc] = env_strs[envc];
                                    envc++;
                                }
                            }
                        }
                        ecur = list_node->cons.cdr;
                    }
                    envp_buf[envc] = NULL;
                    envp = envp_buf;
                }

                pid_t pid = fork();
                if (pid == 0) {
                    /* Child */
                    (void)setpgid(0, 0);
                    if (envp) execve(argv_buf[0], argv_buf, envp);
                    else execvp(argv_buf[0], argv_buf);
                    _exit(127);
                } else if (pid > 0) {
                    (void)setpgid(pid, pid);
                    vm_push(vm, INT_VAL((int64_t)pid));
                    break;
                }
#endif
            }
        }
#else
        (void)env_val; (void)args_val; (void)cmd_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1781: { /* process-wait(pid) → exit-status integer */
        Value pid_val = vm_pop(vm);
#if defined(_WIN32) && !defined(ESHKOL_VM_WASM)
        DWORD pid = (DWORD)vm_process_pid_from_value(vm, pid_val);
        HANDLE proc = OpenProcess(SYNCHRONIZE | PROCESS_QUERY_INFORMATION, FALSE, pid);
        if (proc) {
            DWORD code = 0;
            DWORD wait_rc = WaitForSingleObject(proc, INFINITE);
            int ok = wait_rc == WAIT_OBJECT_0 && GetExitCodeProcess(proc, &code);
            CloseHandle(proc);
            if (ok) {
                vm_process_forget_pty(vm, (int64_t)pid, 1);
                vm_push(vm, INT_VAL((int64_t)code));
                break;
            }
        }
#elif !defined(ESHKOL_VM_WASM)
        int status = 0;
        pid_t pid = (pid_t)vm_process_pid_from_value(vm, pid_val);
        if (waitpid(pid, &status, 0) >= 0) {
            vm_process_forget_pty(vm, (int64_t)pid, 1);
            vm_push(vm, INT_VAL(WIFEXITED(status) ? WEXITSTATUS(status) : -1));
            break;
        }
#else
        (void)pid_val;
#endif
        vm_push(vm, INT_VAL(-1));
        break;
    }

    case 1782: { /* process-kill(pid, signal) → #t or #f */
        Value sig_val = vm_pop(vm), pid_val = vm_pop(vm);
#if defined(_WIN32) && !defined(ESHKOL_VM_WASM)
        DWORD pid = (DWORD)vm_process_pid_from_value(vm, pid_val);
        UINT exit_code = (UINT)((int)as_number(sig_val) > 0 ? (int)as_number(sig_val) : 1);
        HANDLE proc = OpenProcess(PROCESS_TERMINATE, FALSE, pid);
        if (proc) {
            int ok = TerminateProcess(proc, exit_code) != 0;
            CloseHandle(proc);
            if (ok) { vm_push(vm, BOOL_VAL(1)); break; }
        }
#elif !defined(ESHKOL_VM_WASM)
        pid_t pid = (pid_t)vm_process_pid_from_value(vm, pid_val);
        int sig = (int)as_number(sig_val);
        if (kill(pid, sig) == 0) { vm_push(vm, BOOL_VAL(1)); break; }
#else
        (void)sig_val; (void)pid_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1783: { /* io-poll(fd-list, timeout-ms) → list of ready fds or empty
                  * fd-list: list of integer file descriptors
                  * timeout-ms: integer (-1 = block forever, 0 = non-blocking) */
        Value timeout_val = vm_pop(vm), fds_val = vm_pop(vm);
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
        /* Count fds */
        int nfds = 0;
        Value cur = fds_val;
        while (cur.type == VAL_PAIR && nfds < 256) {
            nfds++;
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
        }
        if (nfds > 0) {
            struct pollfd pfds[256];
            cur = fds_val;
            for (int i = 0; i < nfds; i++) {
                pfds[i].fd = vm_process_fd_from_value(vm, vm->heap.objects[cur.as.ptr]->cons.car);
                pfds[i].events = POLLIN;
                pfds[i].revents = 0;
                cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
            }
            int timeout_ms = (int)as_number(timeout_val);
            int ret = poll(pfds, (nfds_t)nfds, timeout_ms);
            if (ret > 0) {
                Value result = NIL_VAL;
                for (int i = nfds - 1; i >= 0; i--) {
                    if (pfds[i].revents & (POLLIN | POLLHUP | POLLERR)) {
                        int32_t cp = heap_alloc(&vm->heap); if (cp < 0) continue;
                        vm->heap.objects[cp]->type = HEAP_CONS;
                        vm->heap.objects[cp]->cons.car = INT_VAL(pfds[i].fd);
                        vm->heap.objects[cp]->cons.cdr = result;
                        result = PAIR_VAL(cp);
                    }
                }
                vm_push(vm, result);
                break;
            }
        }
#else
        (void)timeout_val; (void)fds_val;
#endif
        vm_push(vm, NIL_VAL);
        break;
    }

    case 1784: { /* process-pid() → current process id */
#if defined(_WIN32) && !defined(ESHKOL_VM_WASM)
        vm_push(vm, INT_VAL((int64_t)GetCurrentProcessId()));
#elif !defined(ESHKOL_VM_WASM)
        vm_push(vm, INT_VAL((int64_t)getpid()));
#else
        vm_push(vm, INT_VAL(0));
#endif
        break;
    }

    case 1785: { /* process-setpgid(pid, pgid) → #t or #f */
        Value pgid_val = vm_pop(vm), pid_val = vm_pop(vm);
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
        pid_t pid = (pid_t)vm_process_pid_from_value(vm, pid_val);
        pid_t pgid = (pid_t)as_number(pgid_val);
        if (pid > 0 && pgid >= 0 && setpgid(pid, pgid) == 0) {
            vm_push(vm, BOOL_VAL(1));
            break;
        }
#else
        (void)pgid_val; (void)pid_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1786: { /* process-kill-tree(pid, signal) → #t or #f
                  * process-spawn creates a new process group for each child.
                  * Killing -pid therefore reaches the child and descendants
                  * that stay in that group; direct kill is the fallback for
                  * externally supplied non-group-leader PIDs. */
        Value sig_val = vm_pop(vm), pid_val = vm_pop(vm);
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
        pid_t pid = (pid_t)vm_process_pid_from_value(vm, pid_val);
        int sig = (int)as_number(sig_val);
        if (pid > 1 && sig > 0) {
            int ok = 0;
            if (getpgid(pid) == pid && kill(-pid, sig) == 0)
                ok = 1;
            if (!ok && kill(pid, sig) == 0)
                ok = 1;
            if (ok) {
                vm_push(vm, BOOL_VAL(1));
                break;
            }
        }
#else
        (void)sig_val; (void)pid_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1787: { /* process-spawn-pty(command) → (pid . master-fd) or #f */
        Value cmd_val = vm_pop(vm);
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
        if (cmd_val.type == VAL_STRING && is_valid_heap_ptr(vm, cmd_val.as.ptr)) {
            VmString* cmd = (VmString*)vm->heap.objects[cmd_val.as.ptr]->opaque.ptr;
            if (cmd && cmd->data) {
                int master_fd = -1;
                pid_t pid = forkpty(&master_fd, NULL, NULL, NULL);
                if (pid == 0) {
                    (void)setpgid(0, 0);
                    execlp("/bin/sh", "sh", "-c", cmd->data, (char*)NULL);
                    _exit(127);
                }
                if (pid > 0 && master_fd >= 0) {
                    (void)setpgid(pid, pid);
                    Value handle = vm_int_pair(vm, (int64_t)pid, (int64_t)master_fd);
                    if (handle.type != VAL_NIL) {
                        vm_process_track_pty(vm, (int64_t)pid, master_fd);
                        vm_push(vm, handle);
                        break;
                    }
                    close(master_fd);
                    (void)kill(pid, SIGTERM);
                }
                if (master_fd >= 0) close(master_fd);
            }
        }
#else
        (void)cmd_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1788: { /* process-read-nonblocking(proc-or-fd, max-bytes) → string or #f */
        Value max_val = vm_pop(vm), proc_val = vm_pop(vm);
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
        int fd = vm_process_fd_from_value(vm, proc_val);
        int max_bytes = (int)as_number(max_val);
        if (fd >= 0 && max_bytes > 0) {
            char buf[8192];
            if (max_bytes > (int)sizeof(buf) - 1) max_bytes = (int)sizeof(buf) - 1;
            int flags = fcntl(fd, F_GETFL, 0);
            if (flags >= 0 && fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0) {
                ssize_t n = read(fd, buf, (size_t)max_bytes);
                (void)fcntl(fd, F_SETFL, flags);
                if (n > 0) {
                    buf[n] = '\0';
                    VmString* s = vm_string_new(&vm->heap.regions, buf, (int64_t)n);
                    if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
                }
            }
        }
#else
        (void)max_val; (void)proc_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1789: { /* fork() → 0 in child, child pid in parent, or #f */
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
        pid_t pid = fork();
        if (pid >= 0) {
            vm_push(vm, INT_VAL((int64_t)pid));
            break;
        }
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1790: { /* unix-socket-connect(path) → fd or #f */
        Value path_val = vm_pop(vm);
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
        VmString* path = vm_value_as_string(vm, path_val);
        if (path && path->data && path->byte_len > 0 &&
            (size_t)path->byte_len < sizeof(((struct sockaddr_un*)0)->sun_path)) {
            int fd = socket(AF_UNIX, SOCK_STREAM, 0);
            if (fd >= 0) {
#ifdef SO_NOSIGPIPE
                int no_sigpipe = 1;
                (void)setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE,
                                 &no_sigpipe, (socklen_t)sizeof(no_sigpipe));
#endif
                struct sockaddr_un addr;
                memset(&addr, 0, sizeof(addr));
                addr.sun_family = AF_UNIX;
                memcpy(addr.sun_path, path->data, (size_t)path->byte_len);
                addr.sun_path[path->byte_len] = '\0';
                if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
                    vm_push(vm, INT_VAL((int64_t)fd));
                    break;
                }
                close(fd);
            }
        }
#else
        (void)path_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1791: { /* socket-send(fd, string-or-bytevector) → bytes-written or #f */
        Value data_val = vm_pop(vm), fd_val = vm_pop(vm);
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
        int fd = vm_process_fd_from_value(vm, fd_val);
        const void* data = NULL;
        size_t len = 0;
        VmString* s = vm_value_as_string(vm, data_val);
        if (s && s->data) {
            data = s->data;
            len = (size_t)s->byte_len;
        } else {
            VmBytevector* bv = vm_value_as_bytevector(vm, data_val);
            if (bv && bv->data) {
                data = bv->data;
                len = (size_t)bv->len;
            }
        }
        if (fd >= 0 && data) {
            int flags = 0;
#ifdef MSG_NOSIGNAL
            flags |= MSG_NOSIGNAL;
#endif
            ssize_t n = send(fd, data, len, flags);
            if (n >= 0) {
                vm_push(vm, INT_VAL((int64_t)n));
                break;
            }
        }
#else
        (void)data_val; (void)fd_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1792: { /* socket-recv(fd, max-bytes) → string or #f */
        Value max_val = vm_pop(vm), fd_val = vm_pop(vm);
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
        int fd = vm_process_fd_from_value(vm, fd_val);
        int max_bytes = (int)as_number(max_val);
        if (fd >= 0 && max_bytes > 0) {
            char buf[8192];
            if (max_bytes > (int)sizeof(buf) - 1) max_bytes = (int)sizeof(buf) - 1;
            int old_flags = fcntl(fd, F_GETFL, 0);
            if (old_flags >= 0 && fcntl(fd, F_SETFL, old_flags | O_NONBLOCK) == 0) {
                ssize_t n = recv(fd, buf, (size_t)max_bytes, 0);
                (void)fcntl(fd, F_SETFL, old_flags);
                if (n > 0) {
                    buf[n] = '\0';
                    VmString* s = vm_string_new(&vm->heap.regions, buf, (int64_t)n);
                    if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
                }
            }
        }
#else
        (void)max_val; (void)fd_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1793: { /* socket-close(fd) → bool */
        Value fd_val = vm_pop(vm);
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
        int fd = vm_process_fd_from_value(vm, fd_val);
        if (fd >= 0 && close(fd) == 0) {
            vm_push(vm, BOOL_VAL(1));
            break;
        }
#else
        (void)fd_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1794: { /* signal-install(signum) → bool */
        Value sig_val = vm_pop(vm);
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
        int signum = (int)as_number(sig_val);
        if (signum > 0) {
            struct sigaction sa;
            memset(&sa, 0, sizeof(sa));
            sa.sa_handler = vm_signal_handler;
            sa.sa_flags = SA_RESTART;
            sigemptyset(&sa.sa_mask);
            if (sigaction(signum, &sa, NULL) == 0) {
                vm_push(vm, BOOL_VAL(1));
                break;
            }
        }
#else
        (void)sig_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1795: { /* signal-check() → signum or 0; consumes pending signal */
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
        int signum = (int)vm_last_signal;
        if (signum != 0) vm_last_signal = 0;
        vm_push(vm, INT_VAL((int64_t)signum));
#else
        vm_push(vm, INT_VAL(0));
#endif
        break;
    }

    case 1796: { /* signal-reset(signum) → bool */
        Value sig_val = vm_pop(vm);
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
        int signum = (int)as_number(sig_val);
        if (signum > 0) {
            struct sigaction sa;
            memset(&sa, 0, sizeof(sa));
            sa.sa_handler = SIG_DFL;
            sigemptyset(&sa.sa_mask);
            if (sigaction(signum, &sa, NULL) == 0) {
                vm_push(vm, BOOL_VAL(1));
                break;
            }
        }
#else
        (void)sig_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1797: { /* signal-ignore(signum) → bool */
        Value sig_val = vm_pop(vm);
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
        int signum = (int)as_number(sig_val);
        if (signum > 0) {
            struct sigaction sa;
            memset(&sa, 0, sizeof(sa));
            sa.sa_handler = SIG_IGN;
            sigemptyset(&sa.sa_mask);
            if (sigaction(signum, &sa, NULL) == 0) {
                vm_push(vm, BOOL_VAL(1));
                break;
            }
        }
#else
        (void)sig_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    case 1798: { /* signal-count() → total handled signals */
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
        vm_push(vm, INT_VAL((int64_t)vm_signal_count));
#else
        vm_push(vm, INT_VAL(0));
#endif
        break;
    }

    case 1799: { /* execv(path, argv) → does not return on success, #f on error */
        Value argv_val = vm_pop(vm), path_val = vm_pop(vm);
#if !defined(ESHKOL_VM_WASM) && !defined(_WIN32)
        VmString* path = vm_value_as_string(vm, path_val);
        if (path && path->data && path->byte_len > 0) {
            char* argv_buf[256];
            int argc = 0;
            Value cur = argv_val;
            while (cur.type == VAL_PAIR && argc < 255 && is_valid_heap_ptr(vm, cur.as.ptr)) {
                HeapObject* node = vm->heap.objects[cur.as.ptr];
                if (!node || node->type != HEAP_CONS) break;
                VmString* arg = vm_value_as_string(vm, node->cons.car);
                if (!arg || !arg->data) {
                    argc = -1;
                    break;
                }
                argv_buf[argc++] = arg->data;
                cur = node->cons.cdr;
            }
            if (argc >= 0 && cur.type == VAL_NIL) {
                if (argc == 0) argv_buf[argc++] = path->data;
                argv_buf[argc] = NULL;
                execv(path->data, argv_buf);
            }
        }
#else
        (void)argv_val; (void)path_val;
#endif
        vm_push(vm, BOOL_VAL(0));
        break;
    }

    default:
        /* Check geometric manifold operations (804-861) */
        if (fid >= 804 && fid <= 861) {
            vm_dispatch_geometric(vm, fid);
            break;
        }
        /* Unknown native ID — warn but don't crash */
        fprintf(stderr, "WARNING: unhandled native call ID %d\n", fid);
        vm_push(vm, NIL_VAL);
        break;
    }
}

/*******************************************************************************
 * VM Execution
 ******************************************************************************/
