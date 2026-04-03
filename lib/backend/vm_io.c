/**
 * @file vm_io.c
 * @brief Port I/O system for the Eshkol bytecode VM.
 *
 * Implements R7RS port operations: file ports, string ports,
 * read/write of characters and lines, EOF detection.
 * Standard ports (stdin/stdout/stderr) are pre-allocated.
 * All allocation via OALR arena (vm_arena.h), no GC.
 *
 * Native call IDs: 580-619
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include "vm_string.c"  /* Includes vm_numeric.h → vm_arena.h; gives us VmString */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#ifndef ESHKOL_VM_NO_DISASM
/* Full POSIX I/O — not available in WASM */
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#endif

/* ── Port Types ── */

typedef enum { VM_PORT_FILE, VM_PORT_STRING } VmPortKind;
typedef enum { VM_PORT_INPUT, VM_PORT_OUTPUT } VmPortDir;

/* Initial and growth factor for string output ports */
#define VM_STRING_PORT_INIT_CAP  256
#define VM_STRING_PORT_GROW_FACTOR 2

typedef struct {
    VmPortKind kind;
    VmPortDir  dir;
    int        is_open;
    union {
        FILE* file;
        struct {
            char* buf;   /* arena-allocated for input, malloc for output (needs realloc) */
            int   len;   /* data length in bytes */
            int   pos;   /* current read/write position */
            int   cap;   /* buffer capacity (output ports only, 0 for input) */
        } string;
    };
} VmPort;

/* ── Standard Ports (static, never freed) ── */

static VmPort vm_stdin_port  = { VM_PORT_FILE, VM_PORT_INPUT,  1, { .file = NULL } };
static VmPort vm_stdout_port = { VM_PORT_FILE, VM_PORT_OUTPUT, 1, { .file = NULL } };
static VmPort vm_stderr_port = { VM_PORT_FILE, VM_PORT_OUTPUT, 1, { .file = NULL } };

static void vm_io_init_std_ports(void) {
    vm_stdin_port.file  = stdin;
    vm_stdout_port.file = stdout;
    vm_stderr_port.file = stderr;
}

/* ── File Ports ── */

/* 580: open-input-file */
static VmPort* vm_port_open_input_file(VmRegionStack* rs, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    VmPort* port = (VmPort*)vm_alloc(rs, sizeof(VmPort));
    if (!port) { fclose(f); return NULL; }
    port->kind = VM_PORT_FILE;
    port->dir = VM_PORT_INPUT;
    port->is_open = 1;
    port->file = f;
    return port;
}

/* 581: open-output-file */
static VmPort* vm_port_open_output_file(VmRegionStack* rs, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) return NULL;

    VmPort* port = (VmPort*)vm_alloc(rs, sizeof(VmPort));
    if (!port) { fclose(f); return NULL; }
    port->kind = VM_PORT_FILE;
    port->dir = VM_PORT_OUTPUT;
    port->is_open = 1;
    port->file = f;
    return port;
}

/* ── String Ports ── */

/* 582: open-input-string */
static VmPort* vm_port_open_input_string(VmRegionStack* rs, const VmString* str) {
    VmPort* port = (VmPort*)vm_alloc(rs, sizeof(VmPort));
    if (!port) return NULL;
    port->kind = VM_PORT_STRING;
    port->dir = VM_PORT_INPUT;
    port->is_open = 1;

    int byte_len = str ? str->byte_len : 0;
    port->string.buf = (char*)vm_alloc(rs, byte_len + 1);
    if (!port->string.buf) return NULL;
    if (byte_len > 0) memcpy(port->string.buf, str->data, byte_len);
    port->string.buf[byte_len] = '\0';
    port->string.len = byte_len;
    port->string.pos = 0;
    port->string.cap = 0; /* input ports don't grow */
    return port;
}

/* 583: open-output-string */
static VmPort* vm_port_open_output_string(VmRegionStack* rs) {
    VmPort* port = (VmPort*)vm_alloc(rs, sizeof(VmPort));
    if (!port) return NULL;
    port->kind = VM_PORT_STRING;
    port->dir = VM_PORT_OUTPUT;
    port->is_open = 1;

    /* Output string ports use malloc (need realloc for growth) */
    port->string.buf = (char*)malloc(VM_STRING_PORT_INIT_CAP);
    if (!port->string.buf) return NULL;
    port->string.buf[0] = '\0';
    port->string.len = 0;
    port->string.pos = 0;
    port->string.cap = VM_STRING_PORT_INIT_CAP;
    return port;
}

/* ── Close ── */

/* 584: close-port */
static void vm_port_close(VmPort* port) {
    if (!port || !port->is_open) return;
    port->is_open = 0;

    if (port->kind == VM_PORT_FILE) {
        /* Don't close stdin/stdout/stderr */
        if (port->file != stdin && port->file != stdout && port->file != stderr) {
            fclose(port->file);
        }
        port->file = NULL;
    } else if (port->kind == VM_PORT_STRING && port->dir == VM_PORT_OUTPUT) {
        /* Output string port buffer was malloc'd */
        free(port->string.buf);
        port->string.buf = NULL;
    }
    /* Input string port buffer is arena-allocated, freed with region */
}

/* ── Grow output string port buffer ── */
static int vm_port_string_ensure(VmPort* port, int need) {
    if (port->string.pos + need < port->string.cap) return 1;
    int new_cap = port->string.cap;
    while (new_cap <= port->string.pos + need) {
        new_cap *= VM_STRING_PORT_GROW_FACTOR;
    }
    char* nb = (char*)realloc(port->string.buf, new_cap);
    if (!nb) return 0;
    port->string.buf = nb;
    port->string.cap = new_cap;
    return 1;
}

/* Forward declaration */
static int vm_port_eof(VmPort* port);

/* ── Read Operations ── */

/* Helper: read one raw byte from port, returns byte or -1 on EOF */
static int vm_port_read_byte(VmPort* port) {
    if (!port || !port->is_open) return -1;

    if (port->kind == VM_PORT_FILE) {
        int c = fgetc(port->file);
        return (c == EOF) ? -1 : c;
    } else { /* STRING input */
        if (port->string.pos >= port->string.len) return -1;
        return (unsigned char)port->string.buf[port->string.pos++];
    }
}

/* Helper: peek one raw byte without advancing */
static int vm_port_peek_byte(VmPort* port) {
    if (!port || !port->is_open) return -1;

    if (port->kind == VM_PORT_FILE) {
        int c = fgetc(port->file);
        if (c == EOF) return -1;
        ungetc(c, port->file);
        return c;
    } else {
        if (port->string.pos >= port->string.len) return -1;
        return (unsigned char)port->string.buf[port->string.pos];
    }
}

/* Helper: unread a byte */
static void vm_port_unread_byte(VmPort* port, int byte) {
    if (!port || !port->is_open || byte < 0) return;

    if (port->kind == VM_PORT_FILE) {
        ungetc(byte, port->file);
    } else {
        if (port->string.pos > 0) port->string.pos--;
    }
}

/* 585: read-char → codepoint (-1 = EOF)
 * Reads 1-4 bytes to decode one full UTF-8 codepoint. */
static int vm_port_read_char(VmPort* port) {
    if (!port || !port->is_open || port->dir != VM_PORT_INPUT) return -1;

    int b0 = vm_port_read_byte(port);
    if (b0 < 0) return -1;

    /* ASCII fast path */
    if (b0 < 0x80) return b0;

    /* Determine expected length from lead byte */
    int expect;
    int cp;
    if ((b0 & 0xE0) == 0xC0) { expect = 1; cp = b0 & 0x1F; }
    else if ((b0 & 0xF0) == 0xE0) { expect = 2; cp = b0 & 0x0F; }
    else if ((b0 & 0xF8) == 0xF0) { expect = 3; cp = b0 & 0x07; }
    else return VM_UNICODE_REPLACEMENT; /* invalid lead byte */

    for (int i = 0; i < expect; i++) {
        int b = vm_port_read_byte(port);
        if (b < 0 || (b & 0xC0) != 0x80) {
            /* Bad continuation — push back if possible */
            if (b >= 0) vm_port_unread_byte(port, b);
            return VM_UNICODE_REPLACEMENT;
        }
        cp = (cp << 6) | (b & 0x3F);
    }

    /* Reject overlong */
    if (expect == 1 && cp < 0x80) return VM_UNICODE_REPLACEMENT;
    if (expect == 2 && cp < 0x800) return VM_UNICODE_REPLACEMENT;
    if (expect == 3 && cp < 0x10000) return VM_UNICODE_REPLACEMENT;

    /* Reject surrogates and out-of-range */
    if (cp >= 0xD800 && cp <= 0xDFFF) return VM_UNICODE_REPLACEMENT;
    if (cp > 0x10FFFF) return VM_UNICODE_REPLACEMENT;

    return cp;
}

/* 586: peek-char → codepoint without advancing
 * Reads bytes, then unreads them. */
static int vm_port_peek_char(VmPort* port) {
    if (!port || !port->is_open || port->dir != VM_PORT_INPUT) return -1;

    /* For string ports, save position and restore */
    if (port->kind == VM_PORT_STRING) {
        int saved_pos = port->string.pos;
        int cp = vm_port_read_char(port);
        port->string.pos = saved_pos;
        return cp;
    }

    /* For file ports, use read + seek back */
    long saved = ftell(port->file);
    int cp = vm_port_read_char(port);
    if (saved >= 0) {
        fseek(port->file, saved, SEEK_SET);
    }
    return cp;
}

/* 587: read-line → VmString* (reads until \n or EOF, strips \n)
 * Returns NULL on EOF with no chars read. */
static VmString* vm_port_read_line(VmRegionStack* rs, VmPort* port) {
    if (!port || !port->is_open || port->dir != VM_PORT_INPUT) return NULL;

    /* Accumulate bytes into temp buffer */
    int cap = 128;
    char* buf = (char*)malloc(cap);
    if (!buf) return NULL;
    int len = 0;

    for (;;) {
        int b = vm_port_read_byte(port);
        if (b < 0) break;       /* EOF */
        if (b == '\n') break;   /* End of line */
        if (b == '\r') {
            /* Handle \r\n */
            int next = vm_port_peek_byte(port);
            if (next == '\n') vm_port_read_byte(port);
            break;
        }

        if (len + 1 >= cap) {
            cap *= 2;
            char* nb = (char*)realloc(buf, cap);
            if (!nb) { free(buf); return NULL; }
            buf = nb;
        }
        buf[len++] = (char)b;
    }

    if (len == 0 && vm_port_eof(port)) {
        free(buf);
        return NULL; /* EOF, no data */
    }

    VmString* result = vm_string_new(rs, buf, len);
    free(buf);
    return result;
}

/* ── Write Operations ── */

/* Helper: write raw bytes to port */
static void vm_port_write_bytes(VmPort* port, const char* data, int len) {
    if (!port || !port->is_open || port->dir != VM_PORT_OUTPUT || len <= 0) return;

    if (port->kind == VM_PORT_FILE) {
        fwrite(data, 1, len, port->file);
    } else { /* STRING output */
        if (!vm_port_string_ensure(port, len + 1)) return;
        memcpy(port->string.buf + port->string.pos, data, len);
        port->string.pos += len;
        if (port->string.pos > port->string.len) {
            port->string.len = port->string.pos;
        }
        port->string.buf[port->string.len] = '\0';
    }
}

/* 588: write-char → write one codepoint */
static void vm_port_write_char(VmPort* port, int cp) {
    char buf[4];
    int len = vm_utf8_encode(cp, buf);
    vm_port_write_bytes(port, buf, len);
}

/* 589: write-string */
static void vm_port_write_string(VmPort* port, const VmString* str) {
    if (!str) return;
    vm_port_write_bytes(port, str->data, str->byte_len);
}

/* 590: write-cstr (convenience for C strings) */
static void vm_port_write_cstr(VmPort* port, const char* cstr) {
    if (!cstr) return;
    vm_port_write_bytes(port, cstr, (int)strlen(cstr));
}

/* 591: get-output-string → VmString* (for string output ports) */
static VmString* vm_port_get_output_string(VmRegionStack* rs, VmPort* port) {
    if (!port || port->kind != VM_PORT_STRING || port->dir != VM_PORT_OUTPUT) return NULL;
    return vm_string_new(rs, port->string.buf, port->string.len);
}

/* ── Status ── */

/* 592: eof? */
static int vm_port_eof(VmPort* port) {
    if (!port || !port->is_open) return 1;

    if (port->kind == VM_PORT_FILE) {
        if (port->dir != VM_PORT_INPUT) return 0;
        int c = fgetc(port->file);
        if (c == EOF) return 1;
        ungetc(c, port->file);
        return 0;
    } else {
        return (port->dir == VM_PORT_INPUT && port->string.pos >= port->string.len);
    }
}

/* 593: port-open? */
static int vm_port_is_open(VmPort* port) {
    return port ? port->is_open : 0;
}

/* 594: input-port? */
static int vm_port_is_input(VmPort* port) {
    return port ? (port->dir == VM_PORT_INPUT) : 0;
}

/* 595: output-port? */
static int vm_port_is_output(VmPort* port) {
    return port ? (port->dir == VM_PORT_OUTPUT) : 0;
}

/* ── File System ── */

/* 596: file-exists? */
static int vm_port_file_exists(const char* path) {
#ifdef ESHKOL_VM_NO_DISASM
    (void)path;
    return 0; /* No filesystem in WASM */
#else
    struct stat st;
    return (stat(path, &st) == 0);
#endif
}

/* 597: delete-file */
static int vm_port_delete_file(const char* path) {
#ifdef ESHKOL_VM_NO_DISASM
    (void)path;
    return 0; /* No filesystem in WASM */
#else
    return (unlink(path) == 0);
#endif
}

/* ── Display / Write (Scheme-style) ── */

/* 598: display — write string without quotes, char without #\ prefix */
static void vm_port_display(VmPort* port, const VmString* str) {
    vm_port_write_string(port, str);
}

/* 599: newline */
static void vm_port_newline(VmPort* port) {
    vm_port_write_char(port, '\n');
}

/* 600: write-u8 (write single byte) */
static void vm_port_write_u8(VmPort* port, int byte) {
    if (!port || !port->is_open || port->dir != VM_PORT_OUTPUT) return;
    char b = (char)(unsigned char)byte;
    vm_port_write_bytes(port, &b, 1);
}

/* 601: read-u8 (read single byte) */
static int vm_port_read_u8(VmPort* port) {
    return vm_port_read_byte(port);
}

/* 602: peek-u8 */
static int vm_port_peek_u8(VmPort* port) {
    return vm_port_peek_byte(port);
}

/* 603: flush-output-port */
static void vm_port_flush(VmPort* port) {
    if (!port || !port->is_open || port->dir != VM_PORT_OUTPUT) return;
    if (port->kind == VM_PORT_FILE && port->file) {
        fflush(port->file);
    }
    /* String ports don't need flushing */
}

/* 604: current-input-port */
static VmPort* vm_port_current_input(void) {
    if (!vm_stdin_port.file) vm_io_init_std_ports();
    return &vm_stdin_port;
}

/* 605: current-output-port */
static VmPort* vm_port_current_output(void) {
    if (!vm_stdout_port.file) vm_io_init_std_ports();
    return &vm_stdout_port;
}

/* 606: current-error-port */
static VmPort* vm_port_current_error(void) {
    if (!vm_stderr_port.file) vm_io_init_std_ports();
    return &vm_stderr_port;
}

/* 607: read-string — read up to k characters, return VmString* */
static VmString* vm_port_read_string(VmRegionStack* rs, VmPort* port, int k) {
    if (!port || !port->is_open || port->dir != VM_PORT_INPUT || k <= 0) return NULL;

    /* Worst case: 4 bytes per codepoint */
    int cap = k * 4;
    char* buf = (char*)malloc(cap + 1);
    if (!buf) return NULL;
    int len = 0;
    int chars_read = 0;

    while (chars_read < k) {
        int cp = vm_port_read_char(port);
        if (cp < 0) break; /* EOF */
        char enc[4];
        int enc_len = vm_utf8_encode(cp, enc);
        if (len + enc_len >= cap) {
            cap *= 2;
            char* nb = (char*)realloc(buf, cap + 1);
            if (!nb) { free(buf); return NULL; }
            buf = nb;
        }
        memcpy(buf + len, enc, enc_len);
        len += enc_len;
        chars_read++;
    }

    if (chars_read == 0) {
        free(buf);
        return NULL; /* EOF */
    }

    VmString* result = vm_string_new(rs, buf, len);
    free(buf);
    return result;
}

/* 608: write-bytevector (write raw bytes) */
static void vm_port_write_bytevector(VmPort* port, const char* data, int len) {
    vm_port_write_bytes(port, data, len);
}

/* 609: open-binary-input-file */
static VmPort* vm_port_open_binary_input_file(VmRegionStack* rs, const char* path) {
    return vm_port_open_input_file(rs, path); /* already binary on most systems */
}

/* 610: open-binary-output-file */
static VmPort* vm_port_open_binary_output_file(VmRegionStack* rs, const char* path) {
    return vm_port_open_output_file(rs, path);
}

/* ── Self-Test ── */

#ifdef VM_IO_TEST
#include <assert.h>

static void test_string_port_write_read(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    /* Write to output string port, then read back */
    VmPort* out = vm_port_open_output_string(&rs);
    assert(out && out->is_open);

    vm_port_write_cstr(out, "hello ");
    vm_port_write_cstr(out, "world");

    VmString* result = vm_port_get_output_string(&rs, out);
    assert(result && strcmp(result->data, "hello world") == 0);
    assert(vm_string_length(result) == 11);

    vm_port_close(out);
    assert(!out->is_open);

    vm_region_stack_destroy(&rs);
    printf("  string_port_write_read: PASS\n");
}

static void test_string_port_chars(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    /* Write individual chars */
    VmPort* out = vm_port_open_output_string(&rs);
    vm_port_write_char(out, 'H');
    vm_port_write_char(out, 0xE9); /* é */
    vm_port_write_char(out, 'l');
    vm_port_write_char(out, 'l');
    vm_port_write_char(out, 'o');

    VmString* result = vm_port_get_output_string(&rs, out);
    assert(result && vm_string_length(result) == 5);
    assert(vm_string_ref(result, 1) == 0xE9);

    vm_port_close(out);
    vm_region_stack_destroy(&rs);
    printf("  string_port_chars: PASS\n");
}

static void test_string_port_read_char(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    VmString* str = vm_string_from_cstr(&rs, "ab\xc3\xa9"); /* "abé" */
    VmPort* in = vm_port_open_input_string(&rs, str);
    assert(in && in->is_open);

    /* read_char */
    assert(vm_port_read_char(in) == 'a');
    assert(vm_port_read_char(in) == 'b');
    assert(vm_port_read_char(in) == 0xE9);
    assert(vm_port_read_char(in) == -1); /* EOF */

    vm_port_close(in);
    vm_region_stack_destroy(&rs);
    printf("  string_port_read_char: PASS\n");
}

static void test_peek_char(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    VmString* str = vm_string_from_cstr(&rs, "abc");
    VmPort* in = vm_port_open_input_string(&rs, str);

    /* peek doesn't advance */
    assert(vm_port_peek_char(in) == 'a');
    assert(vm_port_peek_char(in) == 'a');
    assert(vm_port_read_char(in) == 'a');
    assert(vm_port_peek_char(in) == 'b');
    assert(vm_port_read_char(in) == 'b');

    vm_port_close(in);
    vm_region_stack_destroy(&rs);
    printf("  peek_char: PASS\n");
}

static void test_read_line(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    VmString* str = vm_string_from_cstr(&rs, "line1\nline2\nline3");
    VmPort* in = vm_port_open_input_string(&rs, str);

    VmString* l1 = vm_port_read_line(&rs, in);
    assert(l1 && strcmp(l1->data, "line1") == 0);

    VmString* l2 = vm_port_read_line(&rs, in);
    assert(l2 && strcmp(l2->data, "line2") == 0);

    VmString* l3 = vm_port_read_line(&rs, in);
    assert(l3 && strcmp(l3->data, "line3") == 0);

    /* EOF */
    VmString* l4 = vm_port_read_line(&rs, in);
    assert(l4 == NULL);

    vm_port_close(in);
    vm_region_stack_destroy(&rs);
    printf("  read_line: PASS\n");
}

static void test_read_line_crlf(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    VmString* str = vm_string_from_cstr(&rs, "line1\r\nline2\r\n");
    VmPort* in = vm_port_open_input_string(&rs, str);

    VmString* l1 = vm_port_read_line(&rs, in);
    assert(l1 && strcmp(l1->data, "line1") == 0);

    VmString* l2 = vm_port_read_line(&rs, in);
    assert(l2 && strcmp(l2->data, "line2") == 0);

    vm_port_close(in);
    vm_region_stack_destroy(&rs);
    printf("  read_line_crlf: PASS\n");
}

static void test_eof_detection(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    VmString* str = vm_string_from_cstr(&rs, "ab");
    VmPort* in = vm_port_open_input_string(&rs, str);

    assert(vm_port_eof(in) == 0);
    vm_port_read_char(in); /* a */
    assert(vm_port_eof(in) == 0);
    vm_port_read_char(in); /* b */
    assert(vm_port_eof(in) == 1);

    vm_port_close(in);
    vm_region_stack_destroy(&rs);
    printf("  eof_detection: PASS\n");
}

static void test_file_io(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    const char* path = "/tmp/vm_io_test_file.txt";

    /* Write */
    VmPort* out = vm_port_open_output_file(&rs, path);
    assert(out && out->is_open);
    vm_port_write_cstr(out, "Hello, VM!\n");
    vm_port_write_cstr(out, "Line 2\n");
    vm_port_close(out);

    /* Read back */
    VmPort* in = vm_port_open_input_file(&rs, path);
    assert(in && in->is_open);

    VmString* l1 = vm_port_read_line(&rs, in);
    assert(l1 && strcmp(l1->data, "Hello, VM!") == 0);

    VmString* l2 = vm_port_read_line(&rs, in);
    assert(l2 && strcmp(l2->data, "Line 2") == 0);

    assert(vm_port_eof(in) == 1);
    vm_port_close(in);

    /* file-exists? */
    assert(vm_port_file_exists(path) == 1);

    /* delete-file */
    assert(vm_port_delete_file(path) == 1);
    assert(vm_port_file_exists(path) == 0);

    vm_region_stack_destroy(&rs);
    printf("  file_io: PASS\n");
}

static void test_file_exists(void) {
    assert(vm_port_file_exists("/tmp") == 1);
    assert(vm_port_file_exists("/nonexistent_path_12345") == 0);
    printf("  file_exists: PASS\n");
}

static void test_port_predicates(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    VmPort* out = vm_port_open_output_string(&rs);
    assert(vm_port_is_open(out) == 1);
    assert(vm_port_is_output(out) == 1);
    assert(vm_port_is_input(out) == 0);

    VmString* str = vm_string_from_cstr(&rs, "x");
    VmPort* in = vm_port_open_input_string(&rs, str);
    assert(vm_port_is_input(in) == 1);
    assert(vm_port_is_output(in) == 0);

    vm_port_close(out);
    assert(vm_port_is_open(out) == 0);

    vm_port_close(in);
    vm_region_stack_destroy(&rs);
    printf("  port_predicates: PASS\n");
}

static void test_write_string_obj(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    VmPort* out = vm_port_open_output_string(&rs);
    VmString* s = vm_string_from_cstr(&rs, "test string");
    vm_port_write_string(out, s);

    VmString* result = vm_port_get_output_string(&rs, out);
    assert(result && strcmp(result->data, "test string") == 0);

    vm_port_close(out);
    vm_region_stack_destroy(&rs);
    printf("  write_string_obj: PASS\n");
}

static void test_newline_flush(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    VmPort* out = vm_port_open_output_string(&rs);
    vm_port_write_cstr(out, "hello");
    vm_port_newline(out);
    vm_port_write_cstr(out, "world");

    VmString* result = vm_port_get_output_string(&rs, out);
    assert(result && strcmp(result->data, "hello\nworld") == 0);

    /* Flush on string port is no-op, should not crash */
    vm_port_flush(out);

    vm_port_close(out);
    vm_region_stack_destroy(&rs);
    printf("  newline_flush: PASS\n");
}

static void test_read_string(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    VmString* str = vm_string_from_cstr(&rs, "hello world");
    VmPort* in = vm_port_open_input_string(&rs, str);

    VmString* r = vm_port_read_string(&rs, in, 5);
    assert(r && strcmp(r->data, "hello") == 0);

    VmString* r2 = vm_port_read_string(&rs, in, 100);
    assert(r2 && strcmp(r2->data, " world") == 0);

    /* EOF */
    VmString* r3 = vm_port_read_string(&rs, in, 5);
    assert(r3 == NULL);

    vm_port_close(in);
    vm_region_stack_destroy(&rs);
    printf("  read_string: PASS\n");
}

static void test_binary_io(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    VmPort* out = vm_port_open_output_string(&rs);
    vm_port_write_u8(out, 0x48); /* H */
    vm_port_write_u8(out, 0x69); /* i */

    VmString* result = vm_port_get_output_string(&rs, out);
    assert(result && strcmp(result->data, "Hi") == 0);

    vm_port_close(out);

    /* Read bytes */
    VmString* str = vm_string_from_cstr(&rs, "AB");
    VmPort* in = vm_port_open_input_string(&rs, str);
    assert(vm_port_read_u8(in) == 0x41); /* A */
    assert(vm_port_peek_u8(in) == 0x42); /* B, peek */
    assert(vm_port_read_u8(in) == 0x42); /* B, read */
    assert(vm_port_read_u8(in) == -1);   /* EOF */

    vm_port_close(in);
    vm_region_stack_destroy(&rs);
    printf("  binary_io: PASS\n");
}

static void test_std_ports(void) {
    vm_io_init_std_ports();

    VmPort* in = vm_port_current_input();
    assert(in && in->is_open && vm_port_is_input(in));

    VmPort* out = vm_port_current_output();
    assert(out && out->is_open && vm_port_is_output(out));

    VmPort* err = vm_port_current_error();
    assert(err && err->is_open && vm_port_is_output(err));

    printf("  std_ports: PASS\n");
}

static void test_large_string_port(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    VmPort* out = vm_port_open_output_string(&rs);

    /* Write more than initial capacity (256 bytes) to exercise growth */
    for (int i = 0; i < 100; i++) {
        vm_port_write_cstr(out, "0123456789");
    }

    VmString* result = vm_port_get_output_string(&rs, out);
    assert(result && vm_string_length(result) == 1000);
    assert(result->byte_len == 1000);

    /* Verify content */
    assert(vm_string_ref(result, 0) == '0');
    assert(vm_string_ref(result, 9) == '9');
    assert(vm_string_ref(result, 10) == '0');

    vm_port_close(out);
    vm_region_stack_destroy(&rs);
    printf("  large_string_port: PASS\n");
}

static void test_empty_input(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    VmString* str = vm_string_from_cstr(&rs, "");
    VmPort* in = vm_port_open_input_string(&rs, str);

    assert(vm_port_eof(in) == 1);
    assert(vm_port_read_char(in) == -1);
    assert(vm_port_read_line(&rs, in) == NULL);

    vm_port_close(in);
    vm_region_stack_destroy(&rs);
    printf("  empty_input: PASS\n");
}

int main(void) {
    printf("vm_io self-tests:\n");
    test_string_port_write_read();
    test_string_port_chars();
    test_string_port_read_char();
    test_peek_char();
    test_read_line();
    test_read_line_crlf();
    test_eof_detection();
    test_file_io();
    test_file_exists();
    test_port_predicates();
    test_write_string_obj();
    test_newline_flush();
    test_read_string();
    test_binary_io();
    test_std_ports();
    test_large_string_port();
    test_empty_input();
    printf("vm_io: ALL 17 TESTS PASSED\n");
    return 0;
}
#endif
