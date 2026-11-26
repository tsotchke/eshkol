/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol OS Phase U5: Freestanding zero-LLVM, zero-POSIX, zero-malloc in-OS
 * ESKB v2 bytecode VM shell runtime.
 *
 * External symbol closure: ONLY `eshkol_os_syscall6` from `os_syscall_abi.h`.
 */

#include <eshkol/core/os_vm_shell.h>

#include <stddef.h>
#include <stdint.h>

/* ── ESKB v2 constants (matching lib/backend/eskb_format.h) ── */
#define OS_ESKB_MAGIC            0x45534B42u
#define OS_ESKB_VERSION          2u
#define OS_ESKB_SECTION_CONST    0u
#define OS_ESKB_SECTION_CODE     1u
#define OS_ESKB_SECTION_META     2u
#define OS_ESKB_CONST_NIL        0u
#define OS_ESKB_CONST_INT64      1u
#define OS_ESKB_CONST_F64        2u
#define OS_ESKB_CONST_BOOL       3u
#define OS_ESKB_CONST_STRING     6u

/* ── Bounded in-OS VM ceilings ── */
#define OS_VM_MAX_STACK          1024u
#define OS_VM_MAX_FRAMES         128u
#define OS_VM_MAX_FUNCS          256u
#define OS_VM_MAX_CODE           16384u
#define OS_VM_MAX_CONSTS         2048u
#define OS_VM_MAX_UPVALUES       64u
#define OS_VM_DEFAULT_PAGE_SIZE  4096ULL
#define OS_VM_ARENA_ALIGN        8u

/* ── Core VM opcodes (matching lib/backend/vm_core.c) ── */
enum {
    OS_OP_NOP            = 0,
    OS_OP_CONST          = 1,
    OS_OP_NIL            = 2,
    OS_OP_TRUE           = 3,
    OS_OP_FALSE          = 4,
    OS_OP_POP            = 5,
    OS_OP_DUP            = 6,
    OS_OP_ADD            = 7,
    OS_OP_SUB            = 8,
    OS_OP_MUL            = 9,
    OS_OP_DIV            = 10,
    OS_OP_MOD            = 11,
    OS_OP_NEG            = 12,
    OS_OP_ABS            = 13,
    OS_OP_EQ             = 14,
    OS_OP_LT             = 15,
    OS_OP_GT             = 16,
    OS_OP_LE             = 17,
    OS_OP_GE             = 18,
    OS_OP_NOT            = 19,
    OS_OP_GET_LOCAL      = 20,
    OS_OP_SET_LOCAL      = 21,
    OS_OP_GET_UPVALUE    = 22,
    OS_OP_SET_UPVALUE    = 23,
    OS_OP_CLOSURE        = 24,
    OS_OP_CALL           = 25,
    OS_OP_TAIL_CALL      = 26,
    OS_OP_RETURN         = 27,
    OS_OP_JUMP           = 28,
    OS_OP_JUMP_IF_FALSE  = 29,
    OS_OP_LOOP           = 30,
    OS_OP_CONS           = 31,
    OS_OP_CAR            = 32,
    OS_OP_CDR            = 33,
    OS_OP_NULL_P         = 34,
    OS_OP_PRINT          = 35,
    OS_OP_HALT           = 36,
    OS_OP_NATIVE_CALL    = 37,
    OS_OP_CLOSE_UPVALUE  = 38,
    OS_OP_VEC_CREATE     = 39,
    OS_OP_VEC_REF        = 40,
    OS_OP_VEC_SET        = 41,
    OS_OP_VEC_LEN        = 42,
    OS_OP_STR_REF        = 43,
    OS_OP_STR_LEN        = 44,
    OS_OP_PAIR_P         = 45,
    OS_OP_NUM_P          = 46,
    OS_OP_STR_P          = 47,
    OS_OP_BOOL_P         = 48,
    OS_OP_PROC_P         = 49,
    OS_OP_VEC_P          = 50,
    OS_OP_SET_CAR        = 51,
    OS_OP_SET_CDR        = 52,
    OS_OP_POPN           = 53,
    OS_OP_OPEN_CLOSURE   = 54,
    OS_OP_VOID           = 63,
    OS_OP_GLOBAL_MARK    = 66,
    OS_OP_CLOSURE_LONG   = 67,
    OS_OP_CLOSURE_COUNT  = 68,
    OS_OP_TAIL_CALL_POPN = 70,
    OS_OP_COUNT          = 73
};

/* ── Freestanding byte/string primitives (zero libc dependency) ── */

static void os_vm_zero(void* dst, size_t n) {
    volatile uint8_t* p = (volatile uint8_t*)dst;
    for (size_t i = 0; i < n; i++) {
        p[i] = 0u;
    }
}

static void os_vm_copy(void* dst, const void* src, size_t n) {
    volatile uint8_t* d = (volatile uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;
    for (size_t i = 0; i < n; i++) {
        d[i] = s[i];
    }
}

static size_t os_vm_strlen(const char* s) {
    if (!s) return 0u;
    size_t n = 0u;
    while (s[n] != '\0') n++;
    return n;
}

static int os_vm_streq(const char* a, const char* b) {
    if (!a || !b) return 0;
    size_t i = 0u;
    while (a[i] != '\0' && b[i] != '\0') {
        if (a[i] != b[i]) return 0;
        i++;
    }
    return a[i] == b[i];
}

static uint32_t os_vm_read_u32_le(const uint8_t* p) {
    return ((uint32_t)p[0]) |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t os_vm_read_u64_le(const uint8_t* p) {
    return ((uint64_t)os_vm_read_u32_le(p)) |
           ((uint64_t)os_vm_read_u32_le(p + 4) << 32);
}

static double os_vm_bits_to_f64(uint64_t bits) {
    union {
        uint64_t u;
        double d;
    } conv;
    conv.u = bits;
    return conv.d;
}

static uint32_t os_vm_crc32(const uint8_t* data, size_t len) {
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++) {
        crc ^= (uint32_t)data[i];
        for (int b = 0; b < 8; b++) {
            uint32_t mask = (crc & 1u) ? 0xEDB88320u : 0u;
            crc = (crc >> 1) ^ mask;
        }
    }
    return ~crc;
}

static size_t os_vm_i64_to_dec(int64_t val, char* out, size_t cap) {
    if (!out || cap == 0u) return 0u;
    char rev[32];
    size_t rlen = 0u;
    uint64_t mag = 0u;
    int neg = 0;
    if (val < 0) {
        neg = 1;
        mag = (uint64_t)(-(val + 1)) + 1ULL;
    } else {
        mag = (uint64_t)val;
    }
    if (mag == 0u) {
        rev[rlen++] = '0';
    } else {
        while (mag > 0u && rlen < sizeof(rev)) {
            rev[rlen++] = (char)('0' + (mag % 10ULL));
            mag /= 10ULL;
        }
    }
    size_t pos = 0u;
    if (neg && pos < cap) out[pos++] = '-';
    while (rlen > 0u && pos < cap) {
        out[pos++] = rev[--rlen];
    }
    return pos;
}

/* ── Bounded OALR bump arena inside caller workspace ── */

typedef struct os_vm_arena {
    uint8_t* base;
    size_t capacity;
    size_t used;
    size_t peak;
} os_vm_arena_t;

static void os_vm_arena_init(os_vm_arena_t* a, uint8_t* base, size_t capacity) {
    a->base = base;
    a->capacity = capacity;
    a->used = 0u;
    a->peak = 0u;
}

static void* os_vm_arena_alloc(os_vm_arena_t* a, size_t size) {
    if (!a || !a->base) return NULL;
    if (size == 0u) size = OS_VM_ARENA_ALIGN;
    if (size > ((size_t)-1) - (OS_VM_ARENA_ALIGN - 1u)) return NULL;
    size_t aligned = (size + (OS_VM_ARENA_ALIGN - 1u)) & ~((size_t)(OS_VM_ARENA_ALIGN - 1u));
    if (aligned > a->capacity - a->used) return NULL;
    void* ptr = a->base + a->used;
    a->used += aligned;
    if (a->used > a->peak) a->peak = a->used;
    os_vm_zero(ptr, aligned);
    return ptr;
}

static void os_vm_arena_rewind(os_vm_arena_t* a) {
    if (!a || !a->base) return;
    if (a->used > 0u) {
        os_vm_zero(a->base, a->used);
    }
    a->used = 0u;
}

/* ── ESKB stream reader ── */

typedef struct os_eskb_reader {
    const uint8_t* data;
    size_t len;
    size_t pos;
} os_eskb_reader_t;

static int os_eskb_read_u8(os_eskb_reader_t* r, uint8_t* out) {
    if (r->pos >= r->len) return -1;
    *out = r->data[r->pos++];
    return 0;
}

static int os_eskb_read_bytes(os_eskb_reader_t* r, void* out, size_t n) {
    if (n > r->len - r->pos) return -1;
    os_vm_copy(out, r->data + r->pos, n);
    r->pos += n;
    return 0;
}

static int os_eskb_read_leb(os_eskb_reader_t* r, uint64_t* out) {
    uint64_t acc = 0u;
    unsigned shift = 0u;
    for (unsigned i = 0u; i < 10u; i++) {
        if (r->pos >= r->len) return -1;
        uint8_t byte = r->data[r->pos++];
        acc |= ((uint64_t)(byte & 0x7Fu)) << shift;
        if ((byte & 0x80u) == 0u) {
            *out = acc;
            return 0;
        }
        shift += 7u;
    }
    return -1;
}

static int os_eskb_read_i64(os_eskb_reader_t* r, int64_t* out) {
    uint8_t buf[8];
    if (os_eskb_read_bytes(r, buf, 8u) < 0) return -1;
    *out = (int64_t)os_vm_read_u64_le(buf);
    return 0;
}

static int os_eskb_read_f64(os_eskb_reader_t* r, double* out) {
    uint8_t buf[8];
    if (os_eskb_read_bytes(r, buf, 8u) < 0) return -1;
    *out = os_vm_bits_to_f64(os_vm_read_u64_le(buf));
    return 0;
}

static int os_eskb_read_instr(os_eskb_reader_t* r, uint8_t* op, int32_t* operand) {
    uint64_t zz = 0u;
    if (os_eskb_read_u8(r, op) < 0 || os_eskb_read_leb(r, &zz) < 0) return -1;
    if (zz > 0xFFFFFFFFULL) return -1;
    int64_t dec = (int64_t)(zz >> 1);
    if (zz & 1ULL) dec = -dec - 1;
    if (dec < -2147483648LL || dec > 2147483647LL) return -1;
    *operand = (int32_t)dec;
    return 0;
}

static int os_eskb_is_code_target(uint8_t op) {
    return op == OS_OP_JUMP || op == OS_OP_JUMP_IF_FALSE || op == OS_OP_LOOP;
}

/* ── VM value and bounded OALR heap objects ── */

typedef struct os_vm_val {
    uint32_t kind;
    uint32_t reserved;
    union {
        int64_t i;
        double f;
        int32_t b;
        void* ptr;
    } as;
} os_vm_val_t;

typedef struct os_vm_pair {
    os_vm_val_t car;
    os_vm_val_t cdr;
} os_vm_pair_t;

typedef struct os_vm_closure {
    int32_t func_pc;
    int32_t arity;
    uint32_t n_upvalues;
    uint32_t reserved;
    os_vm_val_t* upvalues;
} os_vm_closure_t;

typedef struct os_vm_string {
    uint32_t len;
    uint32_t reserved;
    char* data;
} os_vm_string_t;

typedef struct os_vm_vector {
    uint32_t len;
    uint32_t reserved;
    os_vm_val_t* items;
} os_vm_vector_t;

typedef struct os_vm_instr {
    uint8_t op;
    uint8_t pad0;
    uint16_t pad1;
    int32_t operand;
} os_vm_instr_t;

typedef struct os_vm_func_desc {
    char* name;
    uint8_t n_params;
    uint8_t pad[3];
    uint32_t n_locals;
    uint32_t n_upvalues;
    uint32_t code_offset;
    uint32_t code_len;
} os_vm_func_desc_t;

typedef struct os_vm_module {
    os_vm_instr_t* code;
    uint32_t code_len;
    os_vm_val_t* constants;
    uint32_t n_constants;
    os_vm_func_desc_t* funcs;
    uint32_t n_funcs;
} os_vm_module_t;

typedef struct os_vm_frame {
    uint32_t return_pc;
    uint32_t return_fp;
} os_vm_frame_t;

static os_vm_val_t os_val_nil(void) {
    os_vm_val_t v;
    v.kind = ESHKOL_OS_VM_VAL_NIL;
    v.reserved = 0u;
    v.as.i = 0;
    return v;
}

static os_vm_val_t os_val_int(int64_t x) {
    os_vm_val_t v;
    v.kind = ESHKOL_OS_VM_VAL_INT;
    v.reserved = 0u;
    v.as.i = x;
    return v;
}

static os_vm_val_t os_val_float(double x) {
    os_vm_val_t v;
    v.kind = ESHKOL_OS_VM_VAL_FLOAT;
    v.reserved = 0u;
    v.as.f = x;
    return v;
}

static os_vm_val_t os_val_bool(int b) {
    os_vm_val_t v;
    v.kind = ESHKOL_OS_VM_VAL_BOOL;
    v.reserved = 0u;
    v.as.b = b ? 1 : 0;
    return v;
}

static os_vm_val_t os_val_void(void) {
    os_vm_val_t v;
    v.kind = ESHKOL_OS_VM_VAL_VOID;
    v.reserved = 0u;
    v.as.i = 0;
    return v;
}

static os_vm_val_t os_val_ptr(uint32_t kind, void* ptr) {
    os_vm_val_t v;
    v.kind = kind;
    v.reserved = 0u;
    v.as.ptr = ptr;
    return v;
}

static int os_val_is_truthy(os_vm_val_t v) {
    if (v.kind == ESHKOL_OS_VM_VAL_BOOL) return v.as.b != 0;
    return 1;
}

static int os_val_is_num(os_vm_val_t v) {
    return v.kind == ESHKOL_OS_VM_VAL_INT || v.kind == ESHKOL_OS_VM_VAL_FLOAT;
}

static double os_val_as_double(os_vm_val_t v) {
    if (v.kind == ESHKOL_OS_VM_VAL_FLOAT) return v.as.f;
    if (v.kind == ESHKOL_OS_VM_VAL_INT) return (double)v.as.i;
    return 0.0;
}

/* ── Capability table helpers ── */

static int os_vm_find_cap_slot(const eshkol_os_vm_shell_t* shell, eshkol_cap_id_t cap_id) {
    if (!shell || cap_id == ESHKOL_CAP_ID_INVALID) return -1;
    for (uint32_t i = 0u; i < ESHKOL_OS_VM_MAX_CAPS; i++) {
        if (shell->caps[i].active && shell->caps[i].cap_id == cap_id) {
            return (int)i;
        }
    }
    return -1;
}

static int os_vm_has_free_cap_slot(const eshkol_os_vm_shell_t* shell) {
    if (!shell) return 0;
    for (uint32_t i = 0u; i < ESHKOL_OS_VM_MAX_CAPS; i++) {
        if (!shell->caps[i].active) return 1;
    }
    return 0;
}

static int os_vm_check_cap_rights(const eshkol_os_vm_shell_t* shell,
                                  eshkol_cap_id_t cap_id,
                                  uint64_t required_rights,
                                  uint64_t* out_rights) {
    int slot = os_vm_find_cap_slot(shell, cap_id);
    if (slot < 0) return 0;
    uint64_t held = shell->caps[slot].rights;
    if ((held & required_rights) != required_rights) return 0;
    if (out_rights) *out_rights = held;
    return 1;
}

/* ── ESKB decoder into bounded OALR workspace ── */

static eshkol_os_vm_status_t os_vm_decode_eskb(os_vm_arena_t* arena,
                                               const uint8_t* data,
                                               size_t size,
                                               os_vm_module_t* mod) {
    if (!data || size < 16u || !mod) return ESHKOL_VM_ERR_BAD_ESKB;
    os_vm_zero(mod, sizeof(*mod));

    uint32_t magic = os_vm_read_u32_le(data);
    uint32_t version = os_vm_read_u32_le(data + 4);
    uint32_t checksum = os_vm_read_u32_le(data + 12);
    if (magic != OS_ESKB_MAGIC || version != OS_ESKB_VERSION) {
        return ESHKOL_VM_ERR_BAD_ESKB;
    }

    const uint8_t* payload = data + 16u;
    size_t payload_len = size - 16u;
    if (os_vm_crc32(payload, payload_len) != checksum) {
        return ESHKOL_VM_ERR_CRC_MISMATCH;
    }

    os_eskb_reader_t r;
    r.data = payload;
    r.len = payload_len;
    r.pos = 0u;

    uint64_t n_sections = 0u;
    if (os_eskb_read_leb(&r, &n_sections) < 0 || n_sections > 16u) {
        return ESHKOL_VM_ERR_BAD_ESKB;
    }

    uint8_t sec_ids[16];
    uint32_t sec_sizes[16];
    size_t sec_offsets[16];
    for (uint64_t i = 0u; i < n_sections; i++) {
        uint64_t sz = 0u;
        if (os_eskb_read_u8(&r, &sec_ids[i]) < 0 || os_eskb_read_leb(&r, &sz) < 0) {
            return ESHKOL_VM_ERR_BAD_ESKB;
        }
        if (sz > 0xFFFFFFFFULL) return ESHKOL_VM_ERR_BAD_ESKB;
        sec_sizes[i] = (uint32_t)sz;
    }

    size_t off = r.pos;
    for (uint64_t i = 0u; i < n_sections; i++) {
        if ((size_t)sec_sizes[i] > payload_len - off) return ESHKOL_VM_ERR_BAD_ESKB;
        sec_offsets[i] = off;
        off += (size_t)sec_sizes[i];
    }
    if (off != payload_len) return ESHKOL_VM_ERR_BAD_ESKB;

    /* Pass 1: parse CONST section, then CODE section */
    for (uint64_t s = 0u; s < n_sections; s++) {
        os_eskb_reader_t sr;
        sr.data = payload + sec_offsets[s];
        sr.len = (size_t)sec_sizes[s];
        sr.pos = 0u;

        if (sec_ids[s] == OS_ESKB_SECTION_CONST) {
            uint64_t nc = 0u;
            if (os_eskb_read_leb(&sr, &nc) < 0 || nc > OS_VM_MAX_CONSTS) {
                return ESHKOL_VM_ERR_BAD_ESKB;
            }
            if (mod->constants != NULL) return ESHKOL_VM_ERR_BAD_ESKB;
            mod->n_constants = (uint32_t)nc;
            mod->constants = (os_vm_val_t*)os_vm_arena_alloc(
                arena, (nc > 0u ? (size_t)nc : 1u) * sizeof(os_vm_val_t));
            if (!mod->constants) return ESHKOL_VM_ERR_OOM;

            for (uint64_t i = 0u; i < nc; i++) {
                uint8_t ctype = 0u;
                if (os_eskb_read_u8(&sr, &ctype) < 0) return ESHKOL_VM_ERR_BAD_ESKB;
                if (ctype == OS_ESKB_CONST_NIL) {
                    mod->constants[i] = os_val_nil();
                } else if (ctype == OS_ESKB_CONST_INT64) {
                    int64_t iv = 0;
                    if (os_eskb_read_i64(&sr, &iv) < 0) return ESHKOL_VM_ERR_BAD_ESKB;
                    mod->constants[i] = os_val_int(iv);
                } else if (ctype == OS_ESKB_CONST_F64) {
                    double fv = 0.0;
                    if (os_eskb_read_f64(&sr, &fv) < 0) return ESHKOL_VM_ERR_BAD_ESKB;
                    mod->constants[i] = os_val_float(fv);
                } else if (ctype == OS_ESKB_CONST_BOOL) {
                    uint8_t bv = 0u;
                    if (os_eskb_read_u8(&sr, &bv) < 0) return ESHKOL_VM_ERR_BAD_ESKB;
                    mod->constants[i] = os_val_bool(bv != 0u);
                } else if (ctype == OS_ESKB_CONST_STRING) {
                    uint64_t slen = 0u;
                    if (os_eskb_read_leb(&sr, &slen) < 0 || slen > sr.len - sr.pos) {
                        return ESHKOL_VM_ERR_BAD_ESKB;
                    }
                    os_vm_string_t* str_obj = (os_vm_string_t*)os_vm_arena_alloc(
                        arena, sizeof(os_vm_string_t));
                    char* sbuf = (char*)os_vm_arena_alloc(arena, (size_t)slen + 1u);
                    if (!str_obj || !sbuf) return ESHKOL_VM_ERR_OOM;
                    if (os_eskb_read_bytes(&sr, sbuf, (size_t)slen) < 0) {
                        return ESHKOL_VM_ERR_BAD_ESKB;
                    }
                    sbuf[slen] = '\0';
                    str_obj->len = (uint32_t)slen;
                    str_obj->data = sbuf;
                    mod->constants[i] = os_val_ptr(ESHKOL_OS_VM_VAL_STRING, str_obj);
                } else {
                    int64_t iv = 0;
                    if (os_eskb_read_i64(&sr, &iv) < 0) return ESHKOL_VM_ERR_BAD_ESKB;
                    mod->constants[i] = os_val_int(iv);
                }
            }
            if (sr.pos != sr.len) return ESHKOL_VM_ERR_BAD_ESKB;
        } else if (sec_ids[s] == OS_ESKB_SECTION_CODE) {
            if (mod->code != NULL || mod->funcs != NULL) return ESHKOL_VM_ERR_BAD_ESKB;

            /* First pass over CODE section to compute total instruction count cleanly */
            os_eskb_reader_t scan = sr;
            uint64_t nf = 0u;
            if (os_eskb_read_leb(&scan, &nf) < 0 || nf == 0u || nf > OS_VM_MAX_FUNCS) {
                return ESHKOL_VM_ERR_BAD_ESKB;
            }
            uint64_t total_code = 0u;
            for (uint64_t fi = 0u; fi < nf; fi++) {
                uint64_t name_len = 0u;
                uint8_t np = 0u;
                uint64_t nl = 0u, nuv = 0u, cl = 0u;
                if (os_eskb_read_leb(&scan, &name_len) < 0 ||
                    name_len == 0u || name_len > 255u ||
                    name_len > scan.len - scan.pos) {
                    return ESHKOL_VM_ERR_BAD_ESKB;
                }
                scan.pos += (size_t)name_len;
                if (os_eskb_read_u8(&scan, &np) < 0 ||
                    os_eskb_read_leb(&scan, &nl) < 0 ||
                    os_eskb_read_leb(&scan, &nuv) < 0 ||
                    os_eskb_read_leb(&scan, &cl) < 0 ||
                    cl == 0u || cl > OS_VM_MAX_CODE) {
                    return ESHKOL_VM_ERR_BAD_ESKB;
                }
                if (total_code > OS_VM_MAX_CODE - cl) return ESHKOL_VM_ERR_BAD_ESKB;
                total_code += cl;
                for (uint64_t i = 0u; i < cl; i++) {
                    uint8_t op = 0u;
                    int32_t operand = 0;
                    if (os_eskb_read_instr(&scan, &op, &operand) < 0) {
                        return ESHKOL_VM_ERR_BAD_ESKB;
                    }
                }
            }
            if (scan.pos != scan.len) return ESHKOL_VM_ERR_BAD_ESKB;

            mod->n_funcs = (uint32_t)nf;
            mod->code_len = (uint32_t)total_code;
            mod->funcs = (os_vm_func_desc_t*)os_vm_arena_alloc(
                arena, (size_t)nf * sizeof(os_vm_func_desc_t));
            mod->code = (os_vm_instr_t*)os_vm_arena_alloc(
                arena, (size_t)total_code * sizeof(os_vm_instr_t));
            if (!mod->funcs || !mod->code) return ESHKOL_VM_ERR_OOM;

            /* Second pass: materialize functions and relocated instructions */
            uint64_t nf_check = 0u;
            if (os_eskb_read_leb(&sr, &nf_check) < 0) return ESHKOL_VM_ERR_BAD_ESKB;
            uint32_t code_cursor = 0u;
            for (uint64_t fi = 0u; fi < nf; fi++) {
                uint64_t name_len = 0u;
                uint8_t np = 0u;
                uint64_t nl = 0u, nuv = 0u, cl = 0u;
                if (os_eskb_read_leb(&sr, &name_len) < 0) return ESHKOL_VM_ERR_BAD_ESKB;
                char* fname = (char*)os_vm_arena_alloc(arena, (size_t)name_len + 1u);
                if (!fname) return ESHKOL_VM_ERR_OOM;
                if (os_eskb_read_bytes(&sr, fname, (size_t)name_len) < 0) {
                    return ESHKOL_VM_ERR_BAD_ESKB;
                }
                fname[name_len] = '\0';
                if (os_eskb_read_u8(&sr, &np) < 0 ||
                    os_eskb_read_leb(&sr, &nl) < 0 ||
                    os_eskb_read_leb(&sr, &nuv) < 0 ||
                    os_eskb_read_leb(&sr, &cl) < 0) {
                    return ESHKOL_VM_ERR_BAD_ESKB;
                }
                if (nuv > OS_VM_MAX_UPVALUES) return ESHKOL_VM_ERR_BAD_ESKB;

                mod->funcs[fi].name = fname;
                mod->funcs[fi].n_params = np;
                mod->funcs[fi].n_locals = (uint32_t)nl;
                mod->funcs[fi].n_upvalues = (uint32_t)nuv;
                mod->funcs[fi].code_offset = code_cursor;
                mod->funcs[fi].code_len = (uint32_t)cl;

                for (uint64_t i = 0u; i < cl; i++) {
                    uint8_t op = 0u;
                    int32_t operand = 0;
                    if (os_eskb_read_instr(&sr, &op, &operand) < 0) {
                        return ESHKOL_VM_ERR_BAD_ESKB;
                    }
                    if (op >= OS_OP_COUNT) return ESHKOL_VM_ERR_BAD_OPCODE;
                    if (os_eskb_is_code_target(op)) {
                        if (operand < 0 || (uint64_t)operand >= cl) {
                            return ESHKOL_VM_ERR_BAD_ESKB;
                        }
                        operand += (int32_t)code_cursor;
                    }
                    /* Fail-closed admission: reject desktop-native OP_NATIVE_CALL IDs */
                    if (op == OS_OP_NATIVE_CALL && operand < ESHKOL_OS_VM_HOST_NATIVE_BASE) {
                        return ESHKOL_VM_ERR_FORBIDDEN_NATIVE;
                    }
                    mod->code[code_cursor + (uint32_t)i].op = op;
                    mod->code[code_cursor + (uint32_t)i].operand = operand;
                }
                code_cursor += (uint32_t)cl;
            }
        }
    }

    if (!mod->code || mod->code_len == 0u || !mod->funcs || mod->n_funcs == 0u) {
        return ESHKOL_VM_ERR_BAD_ESKB;
    }

    /* Validate constant references across decoded bytecode */
    for (uint32_t pc = 0u; pc < mod->code_len; pc++) {
        uint8_t op = mod->code[pc].op;
        int32_t operand = mod->code[pc].operand;
        if (op == OS_OP_CONST) {
            if (operand < 0 || (uint32_t)operand >= mod->n_constants) {
                return ESHKOL_VM_ERR_BAD_ESKB;
            }
        } else if (op == OS_OP_CLOSURE) {
            int32_t cidx = operand & 0xFFFF;
            if (cidx < 0 || (uint32_t)cidx >= mod->n_constants) {
                return ESHKOL_VM_ERR_BAD_ESKB;
            }
        } else if (op == OS_OP_CLOSURE_LONG) {
            if (operand < 0 || (uint32_t)operand >= mod->n_constants ||
                pc + 1u >= mod->code_len ||
                mod->code[pc + 1u].op != OS_OP_CLOSURE_COUNT) {
                return ESHKOL_VM_ERR_BAD_ESKB;
            }
        }
    }

    return ESHKOL_VM_OK;
}

/* ── Capability-gated host-call dispatch ── */

static eshkol_os_vm_status_t os_vm_pop_u64(os_vm_val_t* stack, uint32_t* sp, uint64_t* out) {
    if (*sp == 0u) return ESHKOL_VM_ERR_STACK_UNDERFLOW;
    os_vm_val_t v = stack[--(*sp)];
    if (v.kind != ESHKOL_OS_VM_VAL_INT) return ESHKOL_VM_ERR_TYPE;
    *out = (uint64_t)v.as.i;
    return ESHKOL_VM_OK;
}

static eshkol_os_vm_status_t os_vm_dispatch_host_call(eshkol_os_vm_shell_t* shell,
                                                      int32_t fid,
                                                      os_vm_val_t* stack,
                                                      uint32_t* sp,
                                                      uint32_t max_stack,
                                                      uint64_t* syscalls_dispatched,
                                                      uint64_t* cap_denials) {
    if (fid < ESHKOL_OS_VM_HOST_NATIVE_BASE) {
        return ESHKOL_VM_ERR_FORBIDDEN_NATIVE;
    }

    int64_t sys_res = 0;
    eshkol_cap_id_t checked_cap = ESHKOL_CAP_ID_INVALID;
    uint64_t parent_rights = 0u;
    uint64_t derived_rights = 0u;
    uint64_t page_delta = 0u;
    int is_page_map = 0;
    int is_page_unmap = 0;
    int is_cap_derive = 0;
    int is_cap_revoke_or_drop = 0;

    switch (fid) {
    case ESHKOL_OS_VM_CALL_CAP_DERIVE: {
        uint64_t badge = 0u, rights = 0u, parent = 0u;
        eshkol_os_vm_status_t st;
        if ((st = os_vm_pop_u64(stack, sp, &badge)) != ESHKOL_VM_OK) return st;
        if ((st = os_vm_pop_u64(stack, sp, &rights)) != ESHKOL_VM_OK) return st;
        if ((st = os_vm_pop_u64(stack, sp, &parent)) != ESHKOL_VM_OK) return st;
        checked_cap = parent;
        if (!os_vm_check_cap_rights(shell, parent, ESHKOL_CAP_RIGHT_GRANT, &parent_rights)) {
            (*cap_denials)++;
            return ESHKOL_VM_ERR_CAP_DENIED;
        }
        if (!os_vm_has_free_cap_slot(shell)) {
            return ESHKOL_VM_ERR_OOM;
        }
        derived_rights = parent_rights & rights;
        is_cap_derive = 1;
        (*syscalls_dispatched)++;
        sys_res = eshkol_os_syscall6(ESHKOL_SYS_CAP_DERIVE, parent, derived_rights, badge, 0u, 0u, 0u);
        break;
    }
    case ESHKOL_OS_VM_CALL_CAP_REVOKE: {
        uint64_t cap = 0u;
        eshkol_os_vm_status_t st = os_vm_pop_u64(stack, sp, &cap);
        if (st != ESHKOL_VM_OK) return st;
        checked_cap = cap;
        if (!os_vm_check_cap_rights(shell, cap, ESHKOL_CAP_RIGHT_REVOKE, NULL)) {
            (*cap_denials)++;
            return ESHKOL_VM_ERR_CAP_DENIED;
        }
        is_cap_revoke_or_drop = 1;
        (*syscalls_dispatched)++;
        sys_res = eshkol_os_syscall6(ESHKOL_SYS_CAP_REVOKE, cap, 0u, 0u, 0u, 0u, 0u);
        break;
    }
    case ESHKOL_OS_VM_CALL_CAP_DROP: {
        uint64_t cap = 0u;
        eshkol_os_vm_status_t st = os_vm_pop_u64(stack, sp, &cap);
        if (st != ESHKOL_VM_OK) return st;
        checked_cap = cap;
        if (!os_vm_check_cap_rights(shell, cap, ESHKOL_CAP_RIGHT_NONE, NULL)) {
            (*cap_denials)++;
            return ESHKOL_VM_ERR_CAP_DENIED;
        }
        is_cap_revoke_or_drop = 1;
        (*syscalls_dispatched)++;
        sys_res = eshkol_os_syscall6(ESHKOL_SYS_CAP_DROP, cap, 0u, 0u, 0u, 0u, 0u);
        break;
    }
    case ESHKOL_OS_VM_CALL_IPC_SUBMIT: {
        uint64_t flags = 0u, count = 0u, sqe_addr = 0u, ring_cap = 0u;
        eshkol_os_vm_status_t st;
        if ((st = os_vm_pop_u64(stack, sp, &flags)) != ESHKOL_VM_OK) return st;
        if ((st = os_vm_pop_u64(stack, sp, &count)) != ESHKOL_VM_OK) return st;
        if ((st = os_vm_pop_u64(stack, sp, &sqe_addr)) != ESHKOL_VM_OK) return st;
        if ((st = os_vm_pop_u64(stack, sp, &ring_cap)) != ESHKOL_VM_OK) return st;
        checked_cap = ring_cap;
        if (!os_vm_check_cap_rights(shell, ring_cap, ESHKOL_CAP_RIGHT_SEND, NULL)) {
            (*cap_denials)++;
            return ESHKOL_VM_ERR_CAP_DENIED;
        }
        (*syscalls_dispatched)++;
        sys_res = eshkol_os_syscall6(ESHKOL_SYS_IPC_SUBMIT, ring_cap, sqe_addr, count, flags, 0u, 0u);
        break;
    }
    case ESHKOL_OS_VM_CALL_IPC_COMPLETE: {
        uint64_t flags = 0u, max_count = 0u, cqe_addr = 0u, ring_cap = 0u;
        eshkol_os_vm_status_t st;
        if ((st = os_vm_pop_u64(stack, sp, &flags)) != ESHKOL_VM_OK) return st;
        if ((st = os_vm_pop_u64(stack, sp, &max_count)) != ESHKOL_VM_OK) return st;
        if ((st = os_vm_pop_u64(stack, sp, &cqe_addr)) != ESHKOL_VM_OK) return st;
        if ((st = os_vm_pop_u64(stack, sp, &ring_cap)) != ESHKOL_VM_OK) return st;
        checked_cap = ring_cap;
        if (!os_vm_check_cap_rights(shell, ring_cap, ESHKOL_CAP_RIGHT_RECV, NULL)) {
            (*cap_denials)++;
            return ESHKOL_VM_ERR_CAP_DENIED;
        }
        (*syscalls_dispatched)++;
        sys_res = eshkol_os_syscall6(ESHKOL_SYS_IPC_COMPLETE, ring_cap, cqe_addr, max_count, flags, 0u, 0u);
        break;
    }
    case ESHKOL_OS_VM_CALL_IPC_WAIT: {
        uint64_t timeout_ns = 0u, min_complete = 0u, ring_cap = 0u;
        eshkol_os_vm_status_t st;
        if ((st = os_vm_pop_u64(stack, sp, &timeout_ns)) != ESHKOL_VM_OK) return st;
        if ((st = os_vm_pop_u64(stack, sp, &min_complete)) != ESHKOL_VM_OK) return st;
        if ((st = os_vm_pop_u64(stack, sp, &ring_cap)) != ESHKOL_VM_OK) return st;
        checked_cap = ring_cap;
        if (!os_vm_check_cap_rights(shell, ring_cap, ESHKOL_CAP_RIGHT_RECV, NULL)) {
            (*cap_denials)++;
            return ESHKOL_VM_ERR_CAP_DENIED;
        }
        (*syscalls_dispatched)++;
        sys_res = eshkol_os_syscall6(ESHKOL_SYS_IPC_WAIT, ring_cap, min_complete, timeout_ns, 0u, 0u, 0u);
        break;
    }
    case ESHKOL_OS_VM_CALL_PAGE_MAP: {
        uint64_t flags = 0u, num_pages = 0u, vaddr = 0u, space_cap = 0u;
        eshkol_os_vm_status_t st;
        if ((st = os_vm_pop_u64(stack, sp, &flags)) != ESHKOL_VM_OK) return st;
        if ((st = os_vm_pop_u64(stack, sp, &num_pages)) != ESHKOL_VM_OK) return st;
        if ((st = os_vm_pop_u64(stack, sp, &vaddr)) != ESHKOL_VM_OK) return st;
        if ((st = os_vm_pop_u64(stack, sp, &space_cap)) != ESHKOL_VM_OK) return st;
        checked_cap = space_cap;
        if (!os_vm_check_cap_rights(shell, space_cap, ESHKOL_CAP_RIGHT_WRITE, NULL)) {
            (*cap_denials)++;
            return ESHKOL_VM_ERR_CAP_DENIED;
        }
        page_delta = num_pages;
        is_page_map = 1;
        (*syscalls_dispatched)++;
        sys_res = eshkol_os_syscall6(ESHKOL_SYS_PAGE_MAP, space_cap, vaddr, num_pages, flags, 0u, 0u);
        break;
    }
    case ESHKOL_OS_VM_CALL_PAGE_UNMAP: {
        uint64_t num_pages = 0u, vaddr = 0u, space_cap = 0u;
        eshkol_os_vm_status_t st;
        if ((st = os_vm_pop_u64(stack, sp, &num_pages)) != ESHKOL_VM_OK) return st;
        if ((st = os_vm_pop_u64(stack, sp, &vaddr)) != ESHKOL_VM_OK) return st;
        if ((st = os_vm_pop_u64(stack, sp, &space_cap)) != ESHKOL_VM_OK) return st;
        checked_cap = space_cap;
        if (!os_vm_check_cap_rights(shell, space_cap, ESHKOL_CAP_RIGHT_WRITE, NULL)) {
            (*cap_denials)++;
            return ESHKOL_VM_ERR_CAP_DENIED;
        }
        page_delta = num_pages;
        is_page_unmap = 1;
        (*syscalls_dispatched)++;
        sys_res = eshkol_os_syscall6(ESHKOL_SYS_PAGE_UNMAP, space_cap, vaddr, num_pages, 0u, 0u, 0u);
        break;
    }
    case ESHKOL_OS_VM_CALL_TASK_YIELD: {
        (*syscalls_dispatched)++;
        sys_res = eshkol_os_syscall6(ESHKOL_SYS_TASK_YIELD, 0u, 0u, 0u, 0u, 0u, 0u);
        break;
    }
    case ESHKOL_OS_VM_CALL_TASK_EXIT: {
        uint64_t exit_code = 0u;
        eshkol_os_vm_status_t st = os_vm_pop_u64(stack, sp, &exit_code);
        if (st != ESHKOL_VM_OK) return st;
        (*syscalls_dispatched)++;
        sys_res = eshkol_os_syscall6(ESHKOL_SYS_TASK_EXIT, exit_code, 0u, 0u, 0u, 0u, 0u);
        break;
    }
    case ESHKOL_OS_VM_CALL_CONSOLE_WRITE: {
        uint64_t len = 0u, buf_addr = 0u, console_cap = 0u;
        eshkol_os_vm_status_t st;
        if ((st = os_vm_pop_u64(stack, sp, &len)) != ESHKOL_VM_OK) return st;
        if ((st = os_vm_pop_u64(stack, sp, &buf_addr)) != ESHKOL_VM_OK) return st;
        if ((st = os_vm_pop_u64(stack, sp, &console_cap)) != ESHKOL_VM_OK) return st;
        checked_cap = console_cap;
        if (!os_vm_check_cap_rights(shell, console_cap, ESHKOL_CAP_RIGHT_WRITE, NULL)) {
            (*cap_denials)++;
            return ESHKOL_VM_ERR_CAP_DENIED;
        }
        (*syscalls_dispatched)++;
        sys_res = eshkol_os_syscall6(ESHKOL_SYS_CONSOLE_WRITE, console_cap, buf_addr, len, 0u, 0u, 0u);
        break;
    }
    default:
        return ESHKOL_VM_ERR_FORBIDDEN_NATIVE;
    }

    /* Kernel-side capability enforcement feedback */
    if (sys_res == ESHKOL_OS_ERR_REVOKED) {
        if (checked_cap != ESHKOL_CAP_ID_INVALID) {
            (void)eshkol_os_vm_revoke_cap(shell, checked_cap);
        }
        (*cap_denials)++;
        return ESHKOL_VM_ERR_CAP_DENIED;
    }
    if (sys_res == ESHKOL_OS_ERR_RIGHTS_DENIED || sys_res == ESHKOL_OS_ERR_INVALID_CAP) {
        (*cap_denials)++;
        return ESHKOL_VM_ERR_CAP_DENIED;
    }

    if (is_cap_derive && sys_res > 0) {
        if (eshkol_os_vm_grant_cap(shell, (eshkol_cap_id_t)sys_res, derived_rights) != ESHKOL_VM_OK) {
            (*syscalls_dispatched)++;
            (void)eshkol_os_syscall6(ESHKOL_SYS_CAP_DROP, (eshkol_cap_id_t)sys_res, 0u, 0u, 0u, 0u, 0u);
            return ESHKOL_VM_ERR_OOM;
        }
    } else if (is_cap_revoke_or_drop && sys_res >= 0 && checked_cap != ESHKOL_CAP_ID_INVALID) {
        (void)eshkol_os_vm_revoke_cap(shell, checked_cap);
    } else if (is_page_map && sys_res >= 0) {
        shell->user_mapped_pages += page_delta;
    } else if (is_page_unmap && sys_res >= 0) {
        if (page_delta <= shell->user_mapped_pages) {
            shell->user_mapped_pages -= page_delta;
        } else {
            shell->user_mapped_pages = 0u;
        }
    }

    if (*sp >= max_stack) return ESHKOL_VM_ERR_STACK_OVERFLOW;
    stack[(*sp)++] = os_val_int(sys_res);
    return ESHKOL_VM_OK;
}

/* ── Public API ── */

eshkol_os_vm_status_t eshkol_os_vm_init(eshkol_os_vm_shell_t* shell,
                                        const eshkol_os_vm_config_t* config) {
    if (!shell || !config || !config->workspace || config->workspace_bytes < 4096u) {
        return ESHKOL_VM_ERR_INVALID_ARG;
    }

    os_vm_zero(shell, sizeof(*shell));
    shell->workspace = (uint8_t*)config->workspace;
    shell->workspace_bytes = config->workspace_bytes;
    shell->space_cap = config->space_cap;
    shell->page_size = config->page_size ? config->page_size : OS_VM_DEFAULT_PAGE_SIZE;
    shell->console_cap = config->console_cap;
    shell->fuel_limit = config->fuel_limit ? config->fuel_limit : ESHKOL_OS_VM_DEFAULT_FUEL;

    uint32_t stack_lim = config->max_stack_slots ? config->max_stack_slots : OS_VM_MAX_STACK;
    if (stack_lim > OS_VM_MAX_STACK) stack_lim = OS_VM_MAX_STACK;
    shell->max_stack_slots = stack_lim;

    uint32_t frame_lim = config->max_frames ? config->max_frames : OS_VM_MAX_FRAMES;
    if (frame_lim > OS_VM_MAX_FRAMES) frame_lim = OS_VM_MAX_FRAMES;
    shell->max_frames = frame_lim;

    if (config->max_heap_bytes > 0u && (size_t)config->max_heap_bytes <= config->workspace_bytes) {
        shell->max_heap_bytes = config->max_heap_bytes;
    } else {
        shell->max_heap_bytes = (uint32_t)(config->workspace_bytes > 0xFFFFFFFFu
                                               ? 0xFFFFFFFFu
                                               : config->workspace_bytes);
    }

    if (shell->space_cap != ESHKOL_CAP_ID_INVALID) {
        uint64_t num_pages = (config->workspace_bytes + shell->page_size - 1u) / shell->page_size;
        int64_t map_rc = eshkol_os_syscall6(
            ESHKOL_SYS_PAGE_MAP,
            shell->space_cap,
            (uint64_t)(uintptr_t)shell->workspace,
            num_pages,
            ESHKOL_PAGE_FLAG_READ | ESHKOL_PAGE_FLAG_WRITE | ESHKOL_PAGE_FLAG_ZEROED,
            0u,
            0u);
        shell->total_syscalls++;
        if (map_rc < 0) {
            os_vm_zero(shell, sizeof(*shell));
            return (map_rc == ESHKOL_OS_ERR_RIGHTS_DENIED ||
                    map_rc == ESHKOL_OS_ERR_INVALID_CAP ||
                    map_rc == ESHKOL_OS_ERR_REVOKED)
                       ? ESHKOL_VM_ERR_CAP_DENIED
                       : ESHKOL_VM_ERR_SYSCALL_FAILED;
        }
        shell->workspace_mapped_pages = num_pages;
    }

    os_vm_zero(shell->workspace, shell->workspace_bytes);
    shell->initialized = 1u;

    if (config->console_cap != ESHKOL_CAP_ID_INVALID && config->console_rights != 0u) {
        (void)eshkol_os_vm_grant_cap(shell, config->console_cap, config->console_rights);
    }

    return ESHKOL_VM_OK;
}

eshkol_os_vm_status_t eshkol_os_vm_grant_cap(eshkol_os_vm_shell_t* shell,
                                             eshkol_cap_id_t cap_id,
                                             uint64_t rights) {
    if (!shell || !shell->initialized || cap_id == ESHKOL_CAP_ID_INVALID) {
        return ESHKOL_VM_ERR_INVALID_ARG;
    }
    int existing = os_vm_find_cap_slot(shell, cap_id);
    if (existing >= 0) {
        shell->caps[existing].rights = rights;
        return ESHKOL_VM_OK;
    }
    for (uint32_t i = 0u; i < ESHKOL_OS_VM_MAX_CAPS; i++) {
        if (!shell->caps[i].active) {
            shell->caps[i].cap_id = cap_id;
            shell->caps[i].rights = rights;
            shell->caps[i].active = 1u;
            return ESHKOL_VM_OK;
        }
    }
    return ESHKOL_VM_ERR_OOM;
}

eshkol_os_vm_status_t eshkol_os_vm_revoke_cap(eshkol_os_vm_shell_t* shell,
                                              eshkol_cap_id_t cap_id) {
    if (!shell || !shell->initialized || cap_id == ESHKOL_CAP_ID_INVALID) {
        return ESHKOL_VM_ERR_INVALID_ARG;
    }
    int slot = os_vm_find_cap_slot(shell, cap_id);
    if (slot < 0) return ESHKOL_VM_OK;
    shell->caps[slot].active = 0u;
    shell->caps[slot].rights = 0u;
    shell->caps[slot].cap_id = ESHKOL_CAP_ID_INVALID;
    return ESHKOL_VM_OK;
}

int eshkol_os_vm_has_cap(const eshkol_os_vm_shell_t* shell,
                         eshkol_cap_id_t cap_id,
                         uint64_t required_rights) {
    if (!shell || !shell->initialized) return 0;
    return os_vm_check_cap_rights(shell, cap_id, required_rights, NULL);
}

eshkol_os_vm_status_t eshkol_os_vm_set_fuel_limit(eshkol_os_vm_shell_t* shell,
                                                  uint64_t fuel_limit) {
    if (!shell || !shell->initialized) return ESHKOL_VM_ERR_INVALID_ARG;
    shell->fuel_limit = fuel_limit ? fuel_limit : ESHKOL_OS_VM_DEFAULT_FUEL;
    return ESHKOL_VM_OK;
}

eshkol_os_vm_status_t eshkol_os_vm_eval_eskb(eshkol_os_vm_shell_t* shell,
                                             const void* eskb_data,
                                             size_t eskb_size,
                                             const char* entry_name,
                                             eshkol_os_vm_result_t* out_result) {
    eshkol_os_vm_result_t local_res;
    os_vm_zero(&local_res, sizeof(local_res));

    if (!shell || !shell->initialized || !eskb_data || eskb_size == 0u) {
        local_res.status = ESHKOL_VM_ERR_INVALID_ARG;
        if (out_result) os_vm_copy(out_result, &local_res, sizeof(local_res));
        return local_res.status;
    }

    size_t arena_cap = shell->workspace_bytes;
    if ((size_t)shell->max_heap_bytes < arena_cap) {
        arena_cap = (size_t)shell->max_heap_bytes;
    }

    os_vm_arena_t arena;
    os_vm_arena_init(&arena, shell->workspace, arena_cap);

    os_vm_module_t mod;
    eshkol_os_vm_status_t status = os_vm_decode_eskb(
        &arena, (const uint8_t*)eskb_data, eskb_size, &mod);
    if (status != ESHKOL_VM_OK) {
        local_res.status = status;
        local_res.heap_bytes_peak = arena.peak;
        os_vm_arena_rewind(&arena);
        local_res.heap_bytes_live = arena.used;
        local_res.mapped_pages_live = shell->workspace_mapped_pages + shell->user_mapped_pages;
        if (out_result) os_vm_copy(out_result, &local_res, sizeof(local_res));
        return status;
    }

    /* Locate entry function (default index 0 = main) */
    uint32_t entry_idx = 0u;
    if (entry_name && entry_name[0] != '\0') {
        int found = 0;
        for (uint32_t fi = 0u; fi < mod.n_funcs; fi++) {
            if (os_vm_streq(mod.funcs[fi].name, entry_name)) {
                entry_idx = fi;
                found = 1;
                break;
            }
        }
        if (!found) {
            local_res.status = ESHKOL_VM_ERR_MISSING_ENTRY;
            local_res.heap_bytes_peak = arena.peak;
            os_vm_arena_rewind(&arena);
            local_res.heap_bytes_live = arena.used;
            local_res.mapped_pages_live = shell->workspace_mapped_pages + shell->user_mapped_pages;
            if (out_result) os_vm_copy(out_result, &local_res, sizeof(local_res));
            return ESHKOL_VM_ERR_MISSING_ENTRY;
        }
    }
    if (mod.funcs[entry_idx].n_params != 0u || mod.funcs[entry_idx].n_upvalues != 0u) {
        local_res.status = ESHKOL_VM_ERR_MISSING_ENTRY;
        local_res.heap_bytes_peak = arena.peak;
        os_vm_arena_rewind(&arena);
        local_res.heap_bytes_live = arena.used;
        local_res.mapped_pages_live = shell->workspace_mapped_pages + shell->user_mapped_pages;
        if (out_result) os_vm_copy(out_result, &local_res, sizeof(local_res));
        return ESHKOL_VM_ERR_MISSING_ENTRY;
    }

    /* Allocate bounded stack and call frames on the C stack (zero malloc) */
    os_vm_val_t stack[OS_VM_MAX_STACK];
    os_vm_frame_t frames[OS_VM_MAX_FRAMES];
    uint32_t max_stack = shell->max_stack_slots;
    uint32_t max_frames = shell->max_frames;
    uint32_t sp = 0u;
    uint32_t fp = 0u;
    uint32_t frame_count = 0u;
    uint32_t pc = mod.funcs[entry_idx].code_offset;
    uint64_t executed = 0u;
    uint64_t syscalls_dispatched = 0u;
    uint64_t cap_denials = 0u;
    int halted = 0;

    status = ESHKOL_VM_OK;

    while (!halted) {
        if (executed >= shell->fuel_limit) {
            status = ESHKOL_VM_FUEL_EXHAUSTED;
            local_res.fault_pc = pc;
            break;
        }
        if (pc >= mod.code_len) {
            status = ESHKOL_VM_ERR_BOUNDS;
            local_res.fault_pc = pc;
            break;
        }

        const os_vm_instr_t ins = mod.code[pc++];
        executed++;

        switch (ins.op) {
        case OS_OP_NOP:
        case OS_OP_OPEN_CLOSURE:
        case OS_OP_GLOBAL_MARK:
            break;

        case OS_OP_CONST:
            if (sp >= max_stack) { status = ESHKOL_VM_ERR_STACK_OVERFLOW; break; }
            stack[sp++] = mod.constants[ins.operand];
            break;

        case OS_OP_NIL:
            if (sp >= max_stack) { status = ESHKOL_VM_ERR_STACK_OVERFLOW; break; }
            stack[sp++] = os_val_nil();
            break;

        case OS_OP_TRUE:
            if (sp >= max_stack) { status = ESHKOL_VM_ERR_STACK_OVERFLOW; break; }
            stack[sp++] = os_val_bool(1);
            break;

        case OS_OP_FALSE:
            if (sp >= max_stack) { status = ESHKOL_VM_ERR_STACK_OVERFLOW; break; }
            stack[sp++] = os_val_bool(0);
            break;

        case OS_OP_VOID:
            if (sp >= max_stack) { status = ESHKOL_VM_ERR_STACK_OVERFLOW; break; }
            stack[sp++] = os_val_void();
            break;

        case OS_OP_POP:
            if (sp == 0u) { status = ESHKOL_VM_ERR_STACK_UNDERFLOW; break; }
            sp--;
            break;

        case OS_OP_DUP:
            if (sp == 0u) { status = ESHKOL_VM_ERR_STACK_UNDERFLOW; break; }
            if (sp >= max_stack) { status = ESHKOL_VM_ERR_STACK_OVERFLOW; break; }
            stack[sp] = stack[sp - 1u];
            sp++;
            break;

        case OS_OP_POPN: {
            int32_t n = ins.operand;
            if (n > 0 && sp > (uint32_t)n) {
                os_vm_val_t top = stack[sp - 1u];
                sp -= (uint32_t)n;
                stack[sp - 1u] = top;
            }
            break;
        }

        case OS_OP_ADD:
        case OS_OP_SUB:
        case OS_OP_MUL:
        case OS_OP_DIV: {
            if (sp < 2u) { status = ESHKOL_VM_ERR_STACK_UNDERFLOW; break; }
            os_vm_val_t b = stack[--sp];
            os_vm_val_t a = stack[--sp];
            if (!os_val_is_num(a) || !os_val_is_num(b)) {
                status = ESHKOL_VM_ERR_TYPE;
                break;
            }
            if (a.kind == ESHKOL_OS_VM_VAL_INT && b.kind == ESHKOL_OS_VM_VAL_INT) {
                if (ins.op == OS_OP_ADD) {
                    stack[sp++] = os_val_int(a.as.i + b.as.i);
                } else if (ins.op == OS_OP_SUB) {
                    stack[sp++] = os_val_int(a.as.i - b.as.i);
                } else if (ins.op == OS_OP_MUL) {
                    stack[sp++] = os_val_int(a.as.i * b.as.i);
                } else {
                    if (b.as.i == 0) { status = ESHKOL_VM_ERR_DIV_ZERO; break; }
                    if (a.as.i % b.as.i == 0) {
                        stack[sp++] = os_val_int(a.as.i / b.as.i);
                    } else {
                        stack[sp++] = os_val_float((double)a.as.i / (double)b.as.i);
                    }
                }
            } else {
                double ad = os_val_as_double(a);
                double bd = os_val_as_double(b);
                if (ins.op == OS_OP_ADD) stack[sp++] = os_val_float(ad + bd);
                else if (ins.op == OS_OP_SUB) stack[sp++] = os_val_float(ad - bd);
                else if (ins.op == OS_OP_MUL) stack[sp++] = os_val_float(ad * bd);
                else {
                    if (bd == 0.0) { status = ESHKOL_VM_ERR_DIV_ZERO; break; }
                    stack[sp++] = os_val_float(ad / bd);
                }
            }
            break;
        }

        case OS_OP_MOD: {
            if (sp < 2u) { status = ESHKOL_VM_ERR_STACK_UNDERFLOW; break; }
            os_vm_val_t b = stack[--sp];
            os_vm_val_t a = stack[--sp];
            if (a.kind != ESHKOL_OS_VM_VAL_INT || b.kind != ESHKOL_OS_VM_VAL_INT) {
                status = ESHKOL_VM_ERR_TYPE;
                break;
            }
            if (b.as.i == 0) { status = ESHKOL_VM_ERR_DIV_ZERO; break; }
            int64_t r = a.as.i % b.as.i;
            if (r != 0 && ((r ^ b.as.i) < 0)) r += b.as.i;
            stack[sp++] = os_val_int(r);
            break;
        }

        case OS_OP_NEG: {
            if (sp == 0u) { status = ESHKOL_VM_ERR_STACK_UNDERFLOW; break; }
            os_vm_val_t a = stack[--sp];
            if (a.kind == ESHKOL_OS_VM_VAL_INT) stack[sp++] = os_val_int(-a.as.i);
            else if (a.kind == ESHKOL_OS_VM_VAL_FLOAT) stack[sp++] = os_val_float(-a.as.f);
            else { status = ESHKOL_VM_ERR_TYPE; break; }
            break;
        }

        case OS_OP_ABS: {
            if (sp == 0u) { status = ESHKOL_VM_ERR_STACK_UNDERFLOW; break; }
            os_vm_val_t a = stack[--sp];
            if (a.kind == ESHKOL_OS_VM_VAL_INT) {
                stack[sp++] = os_val_int(a.as.i < 0 ? -a.as.i : a.as.i);
            } else if (a.kind == ESHKOL_OS_VM_VAL_FLOAT) {
                stack[sp++] = os_val_float(a.as.f < 0.0 ? -a.as.f : a.as.f);
            } else {
                status = ESHKOL_VM_ERR_TYPE;
                break;
            }
            break;
        }

        case OS_OP_EQ:
        case OS_OP_LT:
        case OS_OP_GT:
        case OS_OP_LE:
        case OS_OP_GE: {
            if (sp < 2u) { status = ESHKOL_VM_ERR_STACK_UNDERFLOW; break; }
            os_vm_val_t b = stack[--sp];
            os_vm_val_t a = stack[--sp];
            int cmp_res = 0;
            if (a.kind == ESHKOL_OS_VM_VAL_INT && b.kind == ESHKOL_OS_VM_VAL_INT) {
                if (ins.op == OS_OP_EQ) cmp_res = (a.as.i == b.as.i);
                else if (ins.op == OS_OP_LT) cmp_res = (a.as.i < b.as.i);
                else if (ins.op == OS_OP_GT) cmp_res = (a.as.i > b.as.i);
                else if (ins.op == OS_OP_LE) cmp_res = (a.as.i <= b.as.i);
                else cmp_res = (a.as.i >= b.as.i);
            } else if (os_val_is_num(a) && os_val_is_num(b)) {
                double ad = os_val_as_double(a);
                double bd = os_val_as_double(b);
                if (ins.op == OS_OP_EQ) cmp_res = (ad == bd);
                else if (ins.op == OS_OP_LT) cmp_res = (ad < bd);
                else if (ins.op == OS_OP_GT) cmp_res = (ad > bd);
                else if (ins.op == OS_OP_LE) cmp_res = (ad <= bd);
                else cmp_res = (ad >= bd);
            } else if (ins.op == OS_OP_EQ) {
                if (a.kind != b.kind) cmp_res = 0;
                else if (a.kind == ESHKOL_OS_VM_VAL_NIL) cmp_res = 1;
                else if (a.kind == ESHKOL_OS_VM_VAL_BOOL) cmp_res = (a.as.b == b.as.b);
                else cmp_res = (a.as.ptr == b.as.ptr);
            } else {
                status = ESHKOL_VM_ERR_TYPE;
                break;
            }
            stack[sp++] = os_val_bool(cmp_res);
            break;
        }

        case OS_OP_NOT: {
            if (sp == 0u) { status = ESHKOL_VM_ERR_STACK_UNDERFLOW; break; }
            os_vm_val_t a = stack[--sp];
            stack[sp++] = os_val_bool(!os_val_is_truthy(a));
            break;
        }

        case OS_OP_GET_LOCAL: {
            if (ins.operand < 0) { status = ESHKOL_VM_ERR_BOUNDS; break; }
            uint32_t slot = fp + (uint32_t)ins.operand;
            if (slot >= max_stack || sp >= max_stack) {
                status = ESHKOL_VM_ERR_STACK_OVERFLOW;
                break;
            }
            stack[sp++] = stack[slot];
            break;
        }

        case OS_OP_SET_LOCAL: {
            if (ins.operand < 0 || sp == 0u) { status = ESHKOL_VM_ERR_BOUNDS; break; }
            uint32_t slot = fp + (uint32_t)ins.operand;
            if (slot >= max_stack) { status = ESHKOL_VM_ERR_STACK_OVERFLOW; break; }
            stack[slot] = stack[--sp];
            break;
        }

        case OS_OP_GET_UPVALUE: {
            if (fp == 0u || stack[fp - 1u].kind != ESHKOL_OS_VM_VAL_CLOSURE) {
                status = ESHKOL_VM_ERR_TYPE;
                break;
            }
            os_vm_closure_t* cl = (os_vm_closure_t*)stack[fp - 1u].as.ptr;
            if (ins.operand < 0 || (uint32_t)ins.operand >= cl->n_upvalues) {
                status = ESHKOL_VM_ERR_BOUNDS;
                break;
            }
            if (sp >= max_stack) { status = ESHKOL_VM_ERR_STACK_OVERFLOW; break; }
            stack[sp++] = cl->upvalues[ins.operand];
            break;
        }

        case OS_OP_SET_UPVALUE: {
            if (fp == 0u || sp == 0u || stack[fp - 1u].kind != ESHKOL_OS_VM_VAL_CLOSURE) {
                status = ESHKOL_VM_ERR_TYPE;
                break;
            }
            os_vm_closure_t* cl = (os_vm_closure_t*)stack[fp - 1u].as.ptr;
            if (ins.operand < 0 || (uint32_t)ins.operand >= cl->n_upvalues) {
                status = ESHKOL_VM_ERR_BOUNDS;
                break;
            }
            cl->upvalues[ins.operand] = stack[--sp];
            break;
        }

        case OS_OP_CLOSE_UPVALUE: {
            if (sp == 0u || stack[sp - 1u].kind != ESHKOL_OS_VM_VAL_CLOSURE) {
                status = ESHKOL_VM_ERR_TYPE;
                break;
            }
            os_vm_closure_t* cl = (os_vm_closure_t*)stack[sp - 1u].as.ptr;
            if (ins.operand >= 0 && (uint32_t)ins.operand < cl->n_upvalues) {
                cl->upvalues[ins.operand] = stack[sp - 1u];
            }
            break;
        }

        case OS_OP_CLOSURE:
        case OS_OP_CLOSURE_LONG: {
            int32_t cidx = 0;
            uint32_t nuv = 0u;
            if (ins.op == OS_OP_CLOSURE) {
                cidx = ins.operand & 0xFFFF;
                nuv = (uint32_t)((ins.operand >> 16) & 0xFF);
            } else {
                cidx = ins.operand;
                if (pc >= mod.code_len || mod.code[pc].op != OS_OP_CLOSURE_COUNT) {
                    status = ESHKOL_VM_ERR_BAD_ESKB;
                    break;
                }
                nuv = (uint32_t)mod.code[pc++].operand;
            }
            if (cidx < 0 || (uint32_t)cidx >= mod.n_constants || nuv > OS_VM_MAX_UPVALUES) {
                status = ESHKOL_VM_ERR_BAD_ESKB;
                break;
            }
            if (sp < nuv) { status = ESHKOL_VM_ERR_STACK_UNDERFLOW; break; }
            os_vm_val_t fconst = mod.constants[cidx];
            if (fconst.kind != ESHKOL_OS_VM_VAL_INT) {
                status = ESHKOL_VM_ERR_BAD_ESKB;
                break;
            }
            int32_t func_pc = (int32_t)(fconst.as.i & 0xFFFFFFFFLL);
            int32_t arity = ((fconst.as.i >> 40) & 1LL)
                                ? (int32_t)((fconst.as.i >> 32) & 0xFFLL)
                                : -1;
            os_vm_closure_t* cl = (os_vm_closure_t*)os_vm_arena_alloc(
                &arena, sizeof(os_vm_closure_t));
            if (!cl) { status = ESHKOL_VM_ERR_OOM; break; }
            cl->func_pc = func_pc;
            cl->arity = arity;
            cl->n_upvalues = nuv;
            if (nuv > 0u) {
                cl->upvalues = (os_vm_val_t*)os_vm_arena_alloc(
                    &arena, (size_t)nuv * sizeof(os_vm_val_t));
                if (!cl->upvalues) { status = ESHKOL_VM_ERR_OOM; break; }
                for (int32_t i = (int32_t)nuv - 1; i >= 0; i--) {
                    cl->upvalues[i] = stack[--sp];
                }
            }
            if (sp >= max_stack) { status = ESHKOL_VM_ERR_STACK_OVERFLOW; break; }
            stack[sp++] = os_val_ptr(ESHKOL_OS_VM_VAL_CLOSURE, cl);
            break;
        }

        case OS_OP_CALL: {
            if (ins.operand < 0) { status = ESHKOL_VM_ERR_BOUNDS; break; }
            uint32_t argc = (uint32_t)ins.operand;
            if (sp < argc + 1u) { status = ESHKOL_VM_ERR_STACK_UNDERFLOW; break; }
            os_vm_val_t fn_val = stack[sp - 1u - argc];
            if (fn_val.kind != ESHKOL_OS_VM_VAL_CLOSURE) {
                status = ESHKOL_VM_ERR_TYPE;
                break;
            }
            os_vm_closure_t* cl = (os_vm_closure_t*)fn_val.as.ptr;
            if (cl->func_pc < 0 || (uint32_t)cl->func_pc >= mod.code_len) {
                status = ESHKOL_VM_ERR_BOUNDS;
                break;
            }
            if (frame_count >= max_frames) {
                status = ESHKOL_VM_ERR_FRAME_OVERFLOW;
                break;
            }
            frames[frame_count].return_pc = pc;
            frames[frame_count].return_fp = fp;
            frame_count++;
            fp = sp - argc;
            pc = (uint32_t)cl->func_pc;
            break;
        }

        case OS_OP_TAIL_CALL:
        case OS_OP_TAIL_CALL_POPN: {
            uint32_t argc = (ins.op == OS_OP_TAIL_CALL_POPN)
                                ? (uint32_t)(ins.operand & 0xFFFF)
                                : (uint32_t)ins.operand;
            if (sp < argc + 1u) { status = ESHKOL_VM_ERR_STACK_UNDERFLOW; break; }
            os_vm_val_t fn_val = stack[sp - 1u - argc];
            if (fn_val.kind != ESHKOL_OS_VM_VAL_CLOSURE) {
                status = ESHKOL_VM_ERR_TYPE;
                break;
            }
            os_vm_closure_t* cl = (os_vm_closure_t*)fn_val.as.ptr;
            if (cl->func_pc < 0 || (uint32_t)cl->func_pc >= mod.code_len) {
                status = ESHKOL_VM_ERR_BOUNDS;
                break;
            }
            if (fp == 0u) {
                if (1u + argc > max_stack) { status = ESHKOL_VM_ERR_STACK_OVERFLOW; break; }
                for (uint32_t i = 0u; i < argc; i++) {
                    stack[1u + i] = stack[sp - argc + i];
                }
                stack[0] = fn_val;
                fp = 1u;
                sp = 1u + argc;
            } else {
                for (uint32_t i = 0u; i < argc; i++) {
                    stack[fp + i] = stack[sp - argc + i];
                }
                stack[fp - 1u] = fn_val;
                sp = fp + argc;
            }
            pc = (uint32_t)cl->func_pc;
            break;
        }

        case OS_OP_RETURN: {
            if (sp == 0u) { status = ESHKOL_VM_ERR_STACK_UNDERFLOW; break; }
            os_vm_val_t ret_val = stack[--sp];
            if (frame_count == 0u) {
                stack[sp++] = ret_val;
                halted = 1;
                break;
            }
            frame_count--;
            sp = (fp > 0u) ? (fp - 1u) : 0u;
            fp = frames[frame_count].return_fp;
            pc = frames[frame_count].return_pc;
            if (sp >= max_stack) { status = ESHKOL_VM_ERR_STACK_OVERFLOW; break; }
            stack[sp++] = ret_val;
            break;
        }

        case OS_OP_JUMP:
        case OS_OP_LOOP:
            if (ins.operand < 0 || (uint32_t)ins.operand >= mod.code_len) {
                status = ESHKOL_VM_ERR_BOUNDS;
                break;
            }
            pc = (uint32_t)ins.operand;
            break;

        case OS_OP_JUMP_IF_FALSE: {
            if (sp == 0u) { status = ESHKOL_VM_ERR_STACK_UNDERFLOW; break; }
            os_vm_val_t cond = stack[--sp];
            if (!os_val_is_truthy(cond)) {
                if (ins.operand < 0 || (uint32_t)ins.operand >= mod.code_len) {
                    status = ESHKOL_VM_ERR_BOUNDS;
                    break;
                }
                pc = (uint32_t)ins.operand;
            }
            break;
        }

        case OS_OP_CONS: {
            if (sp < 2u) { status = ESHKOL_VM_ERR_STACK_UNDERFLOW; break; }
            os_vm_val_t cdr = stack[--sp];
            os_vm_val_t car = stack[--sp];
            os_vm_pair_t* pair = (os_vm_pair_t*)os_vm_arena_alloc(
                &arena, sizeof(os_vm_pair_t));
            if (!pair) { status = ESHKOL_VM_ERR_OOM; break; }
            pair->car = car;
            pair->cdr = cdr;
            stack[sp++] = os_val_ptr(ESHKOL_OS_VM_VAL_PAIR, pair);
            break;
        }

        case OS_OP_CAR: {
            if (sp == 0u) { status = ESHKOL_VM_ERR_STACK_UNDERFLOW; break; }
            os_vm_val_t p = stack[--sp];
            if (p.kind != ESHKOL_OS_VM_VAL_PAIR || !p.as.ptr) {
                status = ESHKOL_VM_ERR_TYPE;
                break;
            }
            stack[sp++] = ((os_vm_pair_t*)p.as.ptr)->car;
            break;
        }

        case OS_OP_CDR: {
            if (sp == 0u) { status = ESHKOL_VM_ERR_STACK_UNDERFLOW; break; }
            os_vm_val_t p = stack[--sp];
            if (p.kind != ESHKOL_OS_VM_VAL_PAIR || !p.as.ptr) {
                status = ESHKOL_VM_ERR_TYPE;
                break;
            }
            stack[sp++] = ((os_vm_pair_t*)p.as.ptr)->cdr;
            break;
        }

        case OS_OP_SET_CAR: {
            if (sp < 2u) { status = ESHKOL_VM_ERR_STACK_UNDERFLOW; break; }
            os_vm_val_t val = stack[--sp];
            os_vm_val_t p = stack[--sp];
            if (p.kind != ESHKOL_OS_VM_VAL_PAIR || !p.as.ptr) {
                status = ESHKOL_VM_ERR_TYPE;
                break;
            }
            ((os_vm_pair_t*)p.as.ptr)->car = val;
            stack[sp++] = os_val_void();
            break;
        }

        case OS_OP_SET_CDR: {
            if (sp < 2u) { status = ESHKOL_VM_ERR_STACK_UNDERFLOW; break; }
            os_vm_val_t val = stack[--sp];
            os_vm_val_t p = stack[--sp];
            if (p.kind != ESHKOL_OS_VM_VAL_PAIR || !p.as.ptr) {
                status = ESHKOL_VM_ERR_TYPE;
                break;
            }
            ((os_vm_pair_t*)p.as.ptr)->cdr = val;
            stack[sp++] = os_val_void();
            break;
        }

        case OS_OP_VEC_CREATE: {
            int64_t count = ins.operand < 0 ? -(int64_t)ins.operand - 1 : (int64_t)ins.operand;
            if (count < 0 || (uint64_t)count > sp) {
                status = ESHKOL_VM_ERR_STACK_UNDERFLOW;
                break;
            }
            os_vm_vector_t* vec = (os_vm_vector_t*)os_vm_arena_alloc(
                &arena, sizeof(os_vm_vector_t));
            if (!vec) { status = ESHKOL_VM_ERR_OOM; break; }
            vec->len = (uint32_t)count;
            if (count > 0) {
                vec->items = (os_vm_val_t*)os_vm_arena_alloc(
                    &arena, (size_t)count * sizeof(os_vm_val_t));
                if (!vec->items) { status = ESHKOL_VM_ERR_OOM; break; }
                for (int64_t i = count - 1; i >= 0; i--) {
                    vec->items[i] = stack[--sp];
                }
            }
            stack[sp++] = os_val_ptr(ESHKOL_OS_VM_VAL_VECTOR, vec);
            break;
        }

        case OS_OP_VEC_REF: {
            if (sp < 2u) { status = ESHKOL_VM_ERR_STACK_UNDERFLOW; break; }
            os_vm_val_t idx = stack[--sp];
            os_vm_val_t vec_v = stack[--sp];
            if (vec_v.kind != ESHKOL_OS_VM_VAL_VECTOR || !vec_v.as.ptr ||
                idx.kind != ESHKOL_OS_VM_VAL_INT) {
                status = ESHKOL_VM_ERR_TYPE;
                break;
            }
            os_vm_vector_t* vec = (os_vm_vector_t*)vec_v.as.ptr;
            if (idx.as.i < 0 || (uint64_t)idx.as.i >= vec->len) {
                status = ESHKOL_VM_ERR_BOUNDS;
                break;
            }
            stack[sp++] = vec->items[idx.as.i];
            break;
        }

        case OS_OP_VEC_SET: {
            if (sp < 3u) { status = ESHKOL_VM_ERR_STACK_UNDERFLOW; break; }
            os_vm_val_t val = stack[--sp];
            os_vm_val_t idx = stack[--sp];
            os_vm_val_t vec_v = stack[--sp];
            if (vec_v.kind != ESHKOL_OS_VM_VAL_VECTOR || !vec_v.as.ptr ||
                idx.kind != ESHKOL_OS_VM_VAL_INT) {
                status = ESHKOL_VM_ERR_TYPE;
                break;
            }
            os_vm_vector_t* vec = (os_vm_vector_t*)vec_v.as.ptr;
            if (idx.as.i < 0 || (uint64_t)idx.as.i >= vec->len) {
                status = ESHKOL_VM_ERR_BOUNDS;
                break;
            }
            vec->items[idx.as.i] = val;
            stack[sp++] = os_val_void();
            break;
        }

        case OS_OP_VEC_LEN: {
            if (sp == 0u) { status = ESHKOL_VM_ERR_STACK_UNDERFLOW; break; }
            os_vm_val_t vec_v = stack[--sp];
            if (vec_v.kind != ESHKOL_OS_VM_VAL_VECTOR || !vec_v.as.ptr) {
                status = ESHKOL_VM_ERR_TYPE;
                break;
            }
            stack[sp++] = os_val_int((int64_t)((os_vm_vector_t*)vec_v.as.ptr)->len);
            break;
        }

        case OS_OP_STR_REF: {
            if (sp < 2u) { status = ESHKOL_VM_ERR_STACK_UNDERFLOW; break; }
            os_vm_val_t idx = stack[--sp];
            os_vm_val_t str_v = stack[--sp];
            if (str_v.kind != ESHKOL_OS_VM_VAL_STRING || !str_v.as.ptr ||
                idx.kind != ESHKOL_OS_VM_VAL_INT) {
                status = ESHKOL_VM_ERR_TYPE;
                break;
            }
            os_vm_string_t* s = (os_vm_string_t*)str_v.as.ptr;
            if (idx.as.i < 0 || (uint64_t)idx.as.i >= s->len) {
                status = ESHKOL_VM_ERR_BOUNDS;
                break;
            }
            stack[sp++] = os_val_int((int64_t)(unsigned char)s->data[idx.as.i]);
            break;
        }

        case OS_OP_STR_LEN: {
            if (sp == 0u) { status = ESHKOL_VM_ERR_STACK_UNDERFLOW; break; }
            os_vm_val_t str_v = stack[--sp];
            if (str_v.kind != ESHKOL_OS_VM_VAL_STRING || !str_v.as.ptr) {
                status = ESHKOL_VM_ERR_TYPE;
                break;
            }
            stack[sp++] = os_val_int((int64_t)((os_vm_string_t*)str_v.as.ptr)->len);
            break;
        }

        case OS_OP_NULL_P: {
            if (sp == 0u) { status = ESHKOL_VM_ERR_STACK_UNDERFLOW; break; }
            os_vm_val_t v = stack[--sp];
            stack[sp++] = os_val_bool(v.kind == ESHKOL_OS_VM_VAL_NIL);
            break;
        }
        case OS_OP_PAIR_P: {
            if (sp == 0u) { status = ESHKOL_VM_ERR_STACK_UNDERFLOW; break; }
            os_vm_val_t v = stack[--sp];
            stack[sp++] = os_val_bool(v.kind == ESHKOL_OS_VM_VAL_PAIR);
            break;
        }
        case OS_OP_NUM_P: {
            if (sp == 0u) { status = ESHKOL_VM_ERR_STACK_UNDERFLOW; break; }
            os_vm_val_t v = stack[--sp];
            stack[sp++] = os_val_bool(os_val_is_num(v));
            break;
        }
        case OS_OP_STR_P: {
            if (sp == 0u) { status = ESHKOL_VM_ERR_STACK_UNDERFLOW; break; }
            os_vm_val_t v = stack[--sp];
            stack[sp++] = os_val_bool(v.kind == ESHKOL_OS_VM_VAL_STRING);
            break;
        }
        case OS_OP_BOOL_P: {
            if (sp == 0u) { status = ESHKOL_VM_ERR_STACK_UNDERFLOW; break; }
            os_vm_val_t v = stack[--sp];
            stack[sp++] = os_val_bool(v.kind == ESHKOL_OS_VM_VAL_BOOL);
            break;
        }
        case OS_OP_PROC_P: {
            if (sp == 0u) { status = ESHKOL_VM_ERR_STACK_UNDERFLOW; break; }
            os_vm_val_t v = stack[--sp];
            stack[sp++] = os_val_bool(v.kind == ESHKOL_OS_VM_VAL_CLOSURE);
            break;
        }
        case OS_OP_VEC_P: {
            if (sp == 0u) { status = ESHKOL_VM_ERR_STACK_UNDERFLOW; break; }
            os_vm_val_t v = stack[--sp];
            stack[sp++] = os_val_bool(v.kind == ESHKOL_OS_VM_VAL_VECTOR);
            break;
        }

        case OS_OP_PRINT: {
            if (sp == 0u) { status = ESHKOL_VM_ERR_STACK_UNDERFLOW; break; }
            os_vm_val_t v = stack[--sp];
            if (shell->console_cap == ESHKOL_CAP_ID_INVALID ||
                !os_vm_check_cap_rights(shell, shell->console_cap, ESHKOL_CAP_RIGHT_WRITE, NULL)) {
                cap_denials++;
                status = ESHKOL_VM_ERR_CAP_DENIED;
                break;
            }
            char buf[64];
            size_t len = 0u;
            if (v.kind == ESHKOL_OS_VM_VAL_INT) {
                len = os_vm_i64_to_dec(v.as.i, buf, sizeof(buf));
            } else if (v.kind == ESHKOL_OS_VM_VAL_BOOL) {
                buf[0] = '#';
                buf[1] = v.as.b ? 't' : 'f';
                len = 2u;
            } else if (v.kind == ESHKOL_OS_VM_VAL_STRING && v.as.ptr) {
                os_vm_string_t* s = (os_vm_string_t*)v.as.ptr;
                len = s->len < sizeof(buf) ? (size_t)s->len : sizeof(buf);
                os_vm_copy(buf, s->data, len);
            } else if (v.kind == ESHKOL_OS_VM_VAL_NIL) {
                buf[0] = '(';
                buf[1] = ')';
                len = 2u;
            } else {
                const char* tag = "<val>";
                len = os_vm_strlen(tag);
                os_vm_copy(buf, tag, len);
            }
            syscalls_dispatched++;
            int64_t wrc = eshkol_os_syscall6(
                ESHKOL_SYS_CONSOLE_WRITE,
                shell->console_cap,
                (uint64_t)(uintptr_t)buf,
                (uint64_t)len,
                0u, 0u, 0u);
            if (wrc == ESHKOL_OS_ERR_REVOKED) {
                (void)eshkol_os_vm_revoke_cap(shell, shell->console_cap);
                cap_denials++;
                status = ESHKOL_VM_ERR_CAP_DENIED;
                break;
            }
            if (wrc < 0) {
                status = ESHKOL_VM_ERR_SYSCALL_FAILED;
                break;
            }
            break;
        }

        case OS_OP_NATIVE_CALL:
            status = os_vm_dispatch_host_call(
                shell, ins.operand, stack, &sp, max_stack,
                &syscalls_dispatched, &cap_denials);
            break;

        case OS_OP_HALT:
            halted = 1;
            break;

        default:
            status = ESHKOL_VM_ERR_BAD_OPCODE;
            break;
        }

        if (status != ESHKOL_VM_OK) {
            local_res.fault_pc = pc > 0u ? (pc - 1u) : 0u;
            break;
        }
    }

    if (status == ESHKOL_VM_OK && sp > 0u) {
        os_vm_val_t top = stack[sp - 1u];
        local_res.value_kind = top.kind;
        if (top.kind == ESHKOL_OS_VM_VAL_INT) {
            local_res.int_value = top.as.i;
        } else if (top.kind == ESHKOL_OS_VM_VAL_FLOAT) {
            local_res.float_value = top.as.f;
            local_res.int_value = (int64_t)top.as.f;
        } else if (top.kind == ESHKOL_OS_VM_VAL_BOOL) {
            local_res.int_value = top.as.b ? 1 : 0;
        }
    }

    /* Deterministic OALR arena rewind: zero leaked bytes on every exit path */
    local_res.status = status;
    local_res.instructions_executed = executed;
    local_res.heap_bytes_peak = arena.peak;
    os_vm_arena_rewind(&arena);
    local_res.heap_bytes_live = arena.used;
    local_res.syscalls_dispatched = syscalls_dispatched;
    local_res.cap_denials = cap_denials;
    local_res.mapped_pages_live = shell->workspace_mapped_pages + shell->user_mapped_pages;

    shell->total_evals++;
    shell->total_syscalls += syscalls_dispatched;
    shell->total_cap_denials += cap_denials;

    if (out_result) os_vm_copy(out_result, &local_res, sizeof(local_res));
    return status;
}

eshkol_os_vm_status_t eshkol_os_vm_destroy(eshkol_os_vm_shell_t* shell) {
    if (!shell) return ESHKOL_VM_ERR_INVALID_ARG;
    if (!shell->initialized) return ESHKOL_VM_OK;

    eshkol_os_vm_status_t status = ESHKOL_VM_OK;
    if (shell->workspace && shell->workspace_bytes > 0u) {
        os_vm_zero(shell->workspace, shell->workspace_bytes);
    }
    if (shell->space_cap != ESHKOL_CAP_ID_INVALID && shell->workspace_mapped_pages > 0u) {
        int64_t unmap_rc = eshkol_os_syscall6(
            ESHKOL_SYS_PAGE_UNMAP,
            shell->space_cap,
            (uint64_t)(uintptr_t)shell->workspace,
            shell->workspace_mapped_pages,
            0u, 0u, 0u);
        shell->total_syscalls++;
        if (unmap_rc < 0) {
            status = ESHKOL_VM_ERR_SYSCALL_FAILED;
        }
        shell->workspace_mapped_pages = 0u;
    }

    os_vm_zero(shell, sizeof(*shell));
    return status;
}

const char* eshkol_os_vm_status_name(eshkol_os_vm_status_t status) {
    switch (status) {
    case ESHKOL_VM_OK:                   return "ESHKOL_VM_OK";
    case ESHKOL_VM_FUEL_EXHAUSTED:       return "ESHKOL_VM_FUEL_EXHAUSTED";
    case ESHKOL_VM_ERR_INVALID_ARG:      return "ESHKOL_VM_ERR_INVALID_ARG";
    case ESHKOL_VM_ERR_BAD_ESKB:         return "ESHKOL_VM_ERR_BAD_ESKB";
    case ESHKOL_VM_ERR_CRC_MISMATCH:     return "ESHKOL_VM_ERR_CRC_MISMATCH";
    case ESHKOL_VM_ERR_FORBIDDEN_NATIVE: return "ESHKOL_VM_ERR_FORBIDDEN_NATIVE";
    case ESHKOL_VM_ERR_CAP_DENIED:       return "ESHKOL_VM_ERR_CAP_DENIED";
    case ESHKOL_VM_ERR_OOM:              return "ESHKOL_VM_ERR_OOM";
    case ESHKOL_VM_ERR_STACK_OVERFLOW:   return "ESHKOL_VM_ERR_STACK_OVERFLOW";
    case ESHKOL_VM_ERR_STACK_UNDERFLOW:  return "ESHKOL_VM_ERR_STACK_UNDERFLOW";
    case ESHKOL_VM_ERR_FRAME_OVERFLOW:   return "ESHKOL_VM_ERR_FRAME_OVERFLOW";
    case ESHKOL_VM_ERR_TYPE:             return "ESHKOL_VM_ERR_TYPE";
    case ESHKOL_VM_ERR_DIV_ZERO:         return "ESHKOL_VM_ERR_DIV_ZERO";
    case ESHKOL_VM_ERR_BOUNDS:           return "ESHKOL_VM_ERR_BOUNDS";
    case ESHKOL_VM_ERR_BAD_OPCODE:       return "ESHKOL_VM_ERR_BAD_OPCODE";
    case ESHKOL_VM_ERR_MISSING_ENTRY:    return "ESHKOL_VM_ERR_MISSING_ENTRY";
    case ESHKOL_VM_ERR_SYSCALL_FAILED:   return "ESHKOL_VM_ERR_SYSCALL_FAILED";
    }
    return "ESHKOL_VM_UNKNOWN";
}
