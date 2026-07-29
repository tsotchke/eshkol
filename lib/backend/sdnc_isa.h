/**
 * @file sdnc_isa.h
 * @brief Canonical instruction set, state-vector layout and type tags for the
 *        SDNC (Self-Differentiating Neural Computer) weight-matrix layer.
 *
 * This is the single source of truth shared by the two halves of the SDNC
 * pipeline, which sit on opposite sides of a serialization boundary:
 *
 *   - `lib/backend/weight_matrices.c` -- the producer. Analytically constructs
 *     the six-layer computable-transformer weights and exports them as a QLMW
 *     file, and emits the bytecode those weights interpret.
 *   - `lib/backend/qllm_interpreter.c` -- the consumer (`eshkol-qllm-run`).
 *     Loads a QLMW file and executes bytecode by running the transformer
 *     forward pass.
 *
 * Both files previously carried private copies of these tables under identical
 * names (`OpCode`, `Instr`, `S_*`, `TYPE_*`). The copies had drifted: the
 * producer defines `OP_SWAP = 83` with `OP_COUNT = 84`, while the consumer's
 * copy stopped at `OP_COUNT = 83` and therefore decoded a valid SWAP as an
 * out-of-range opcode. Because the drift is on a serialization boundary the
 * compiler could not see it -- a renumbering changes execution silently rather
 * than failing to link. Sharing one header is what makes that class of drift
 * impossible rather than merely unlikely.
 *
 * Deliberately a PRIVATE header under lib/backend/ rather than a public one
 * under inc/. `OpCode` and `Instr` are names several unrelated translation
 * units in this tree also use for their own, different instruction sets
 * (eshkol_compiler.c, vm_core.c, stackvm_codegen.c, weight_compiler.c,
 * eshkol_benchmark.c). No single translation unit sees two of them, so there is
 * no ODR violation today -- but publishing these particular spellings into the
 * public include path would invite one, and would also put an internal ISA into
 * the generated API documentation. Keeping the header beside its only two
 * consumers, next to sibling private headers like eskb_format.h and
 * vm_numeric.h, gives single-sourcing without widening the surface.
 *
 * IMPORTANT -- this is NOT the production bytecode VM's instruction set. The
 * production VM (`lib/backend/vm_core.c`) is a separate 66-opcode ISA whose
 * values 64/65 are `OP_LANGUAGE_COVERAGE`/`OP_LANGUAGE_COVERAGE_CALL`, and it
 * collides irreconcilably with the AD band defined here. The two instruction
 * sets are deliberately distinct and must not be merged. Likewise the
 * `TYPE_*` tags below are the float state-vector encoding used by the weight
 * matrices, not the production VM's `ValType`.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#ifndef ESHKOL_LIB_BACKEND_SDNC_ISA_H
#define ESHKOL_LIB_BACKEND_SDNC_ISA_H

/*******************************************************************************
 * Architecture constants
 *
 * These fix the transformer geometry the weights are constructed for and the
 * QLMW header records. The producer and the consumer must agree exactly or a
 * QLMW file cannot be interpreted.
 ******************************************************************************/

#define D 256                /**< Model width (d_model). */
#define H 16                 /**< Attention heads. */
#define HD 2                 /**< Per-head dimension. */
#define N_LAYERS 6           /**< Transformer layers. */
#define MEM_SIZE 4           /**< Bounded in-state memory cells. */
#define FFN_DIM 2304         /**< Feed-forward inner width. */
#define AD_NODE_FIELDS 8     /**< Fields per reverse-mode AD tape node. */
#define ARENA_CELLS 16       /**< Bounded in-state heap cells. */
#define ARENA_CELL_FIELDS 5  /**< Fields per arena cell. */

/* Opcodes — canonical numbering from eshkol_compiler.c */
typedef enum {
    OP_NOP=0, OP_CONST=1, OP_NIL=2, OP_TRUE=3, OP_FALSE=4, OP_POP=5, OP_DUP=6,
    OP_ADD=7, OP_SUB=8, OP_MUL=9, OP_DIV=10, OP_MOD=11, OP_NEG=12, OP_ABS=13,
    OP_EQ=14, OP_LT=15, OP_GT=16, OP_LE=17, OP_GE=18, OP_NOT=19,
    OP_GET_LOCAL=20, OP_SET_LOCAL=21, OP_GET_UPVALUE=22, OP_SET_UPVALUE=23,
    OP_CLOSURE=24, OP_CALL=25, OP_TAIL_CALL=26, OP_RETURN=27,
    OP_JUMP=28, OP_JUMP_IF_FALSE=29, OP_LOOP=30,
    OP_CONS=31, OP_CAR=32, OP_CDR=33, OP_NULL_P=34,
    OP_PRINT=35, OP_HALT=36, OP_NATIVE_CALL=37,
    OP_CLOSE_UPVALUE=38,
    OP_VEC_CREATE=39, OP_VEC_REF=40, OP_VEC_SET=41, OP_VEC_LEN=42,
    OP_STR_REF=43, OP_STR_LEN=44,
    OP_PAIR_P=45, OP_NUM_P=46, OP_STR_P=47, OP_BOOL_P=48, OP_PROC_P=49, OP_VEC_P=50,
    OP_SET_CAR=51, OP_SET_CDR=52, OP_POPN=53,
    OP_OPEN_CLOSURE=54, OP_CALLCC=55, OP_INVOKE_CC=56,
    OP_PUSH_HANDLER=57, OP_POP_HANDLER=58, OP_GET_EXN=59,
    OP_PACK_REST=60, OP_WIND_PUSH=61, OP_WIND_POP=62, OP_VOID=63,

    /* AD opcodes — bounded tape ops are weight-encoded; libm/precision ops delegate. */
    OP_AD_VAR=64, OP_AD_CONST=65,
    OP_AD_ADD=66, OP_AD_SUB=67, OP_AD_MUL=68,
    OP_AD_NEG=69, OP_AD_ABS=70, OP_AD_RELU=71,
    OP_AD_SIGMOID=72, OP_AD_TANH=73,
    OP_AD_EXP=74, OP_AD_LOG=75, OP_AD_SQRT=76,
    OP_AD_BACKWARD=77, OP_AD_GRAD=78,
    /* AD ops delegated to C (transcendentals / division) */
    OP_AD_DIV=79, OP_AD_POW=80, OP_AD_SIN=81, OP_AD_COS=82,

    /* Base stack op appended after the AD band (base band 0-63 is full).
     * SWAP exchanges the top two stack registers (TOS<->SOS). It is a base
     * stack op, NOT an AD op — no AD range check (<= OP_AD_SQRT) sees it. */
    OP_SWAP=83,

    OP_COUNT=84
} OpCode;

typedef struct { OpCode op; int operand; } Instr;

/* State vector layout (d_model=256) */
enum {
    /* Permanent state (0-15) — persist across steps */
    S_PC=0, S_TOS=1, S_SOS=2, S_R2=3, S_R3=4, S_DEPTH=5,
    S_OUTPUT=6, S_HALT=7,
    S_MEM0=8, S_MEM1=9, S_MEM2=10, S_MEM3=11,
    S_SP=12, S_FP=13, S_HAS_OUT=14, S_CUR_CLOSURE=15,
    S_EXC_DEPTH=S_SP, S_WIND_DEPTH=S_FP,

    /* Intermediate / transient (16-31) — cleared every cycle by Layer 3 */
    S_OPCODE=16, S_OPERAND=17,
    S_PRODUCT=18, S_LOADVAL=19,
    S_STORED0=20, S_STORED1=21, S_STORED2=22, S_STORED3=23,
    S_ZOPER=24, S_ZPC1=25,
    S_CMP_EQ=26, S_CMP_LT=27,
    S_IS_CALL=28, S_IS_RET=29, S_IS_NATIVE=30,
    S_ABS_DELTA=31,

    /* Type tags for TOS/SOS/R2/R3 (32-35) — persist across steps.
     * Type encoding: 0=number, 1=boolean, 2=pair, 3=closure,
     *                4=string, 5=vector, 6=nil, 7=continuation */
    S_TYPE_TOS=32, S_TYPE_SOS=33, S_TYPE_R2=34, S_TYPE_R3=35,

    /* ── Zone B: AD control state (36-47) — persist across steps ── */
    S_AD_TAPE_LEN=36,    /* number of nodes on tape (0..AD_MAX_TAPE) */
    S_AD_CURSOR=37,      /* backward pass cursor (current node index, decrements) */
    S_AD_MODE=38,        /* 0=normal, 1=forward recording, 2=backward pass */
    S_AD_CUR_OP=39,      /* operation type of node at cursor */
    S_AD_CUR_VALUE=40,   /* forward value of node at cursor */
    S_AD_CUR_GRAD=41,    /* gradient of node at cursor */
    S_AD_CUR_LEFT=42,    /* left parent index */
    S_AD_CUR_RIGHT=43,   /* right parent index */
    S_AD_CUR_SAVED=44,   /* auxiliary saved value */
    S_AD_LEFT_VALUE=45,  /* value of left parent (loaded for backward) */
    S_AD_LEFT_GRAD=46,   /* gradient of left parent */
    S_AD_UNARY_ABS_ACTIVE=S_AD_LEFT_GRAD, /* forward scratch alias */
    S_AD_RIGHT_VALUE=47, /* value of right parent */

    /* ── Zone C: AD tape storage (48-111) — 8 nodes x 8 fields ──
     * Node i at dims (48 + i*8) through (48 + i*8 + 7)
     * Fields: [op, value, gradient, left, right, saved, spare0, spare1] */
    S_AD_TAPE_BASE=48,
    /* Access macro: S_AD_TAPE_BASE + node_idx * AD_NODE_FIELDS + field_offset */

    /* ── Zone D: AD transient / precomputed (112-127) ── */
    S_AD_IS_FORWARD=112,     /* indicator: executing AD forward op this cycle */
    S_AD_IS_BACKWARD=113,    /* indicator: in backward pass */
    S_AD_GRAD_ACCUM=114,     /* gradient accumulator */
    S_AD_UNARY_RELU_ACTIVE=S_AD_GRAD_ACCUM, /* forward scratch alias */
    S_AD_PROD_GRAD_LV=115,   /* precomputed: CUR_GRAD * RIGHT_VALUE (left delta for MUL) */
    S_AD_PROD_GRAD_RV=116,   /* precomputed: CUR_GRAD * LEFT_VALUE (right delta for MUL) */
    S_AD_LEFT_GRAD_NEW=117,  /* computed gradient delta for left parent */
    S_AD_RIGHT_GRAD_NEW=118, /* computed gradient delta for right parent */
    S_AD_PROD_LR=119,       /* precomputed: AD_LEFT_VALUE * AD_RIGHT_VALUE */
    S_AD_PROD_GRAD_CV=S_AD_PROD_LR,  /* legacy alias for dim 119 */
    S_AD_PROD_GRAD_SV=120,  /* precomputed: CUR_GRAD * CUR_SAVED (all unary backward) */
    S_AD_SPARE1=120,

    /* Stage-1 VM-as-transformer memory-op transients.
     * These reuse the true spare portion of Zone D. Layer 1 computes
     * saturated one-hot indicators over S_TYPE_TOS; Layer 3 consumes them
     * to execute NULL_P and the six type predicates without IS_NATIVE. */
    S_TYPE_IS_NUM=121,
    S_TYPE_IS_BOOL=122,
    S_TYPE_IS_PAIR=123,
    S_TYPE_IS_PROC=124,
    S_TYPE_IS_STR=125,
    S_TYPE_IS_VEC=126,
    S_TYPE_IS_NIL=127,

    S_AD_SPARE2=121, S_AD_SPARE3=122, S_AD_SPARE4=123,
    S_AD_SPARE5=124, S_AD_SPARE6=125, S_AD_SPARE7=126, S_AD_SPARE8=127,

    /* ── Zone E: bounded arena bank (128-207) ──
     * Cell i stores [kind, car_value, cdr_value, car_type, cdr_type].
     * Stack values hold small cell indices, not host pointers. */
    S_ARENA_BASE=128,
    S_ARENA_NEXT=S_ARENA_BASE + ARENA_CELLS * ARENA_CELL_FIELDS,

    /* Arena operation transients, cleared every cycle. */
    S_ARENA_WRITE_KIND,
    S_ARENA_WRITE_CAR,
    S_ARENA_WRITE_CDR,
    S_ARENA_READ_CAR,
    S_ARENA_READ_CDR,
    S_ARENA_TARGET,
    S_ARENA_NEW_KIND,
    S_ARENA_NEW_CAR,
    S_ARENA_NEW_CDR,
    S_ARENA_NEW_CAR_TYPE,
    S_ARENA_NEW_CDR_TYPE,
    S_ARENA_VEC_WRITE,
    S_ARENA_VEC_BASE,
    S_ARENA_VEC_LEN,
    S_ARENA_VEC_E0,
    S_ARENA_VEC_E1,
    S_ARENA_VEC_E2,
    S_ARENA_VEC_E3,
    S_ARENA_VEC_T0,
    S_ARENA_VEC_T1,
    S_ARENA_VEC_T2,
    S_ARENA_VEC_T3,
    S_ARENA_VEC_HAS_E0,
    S_ARENA_VEC_HAS_E1,
    S_ARENA_VEC_HAS_E2,
    S_ARENA_VEC_HAS_E3,
    S_ARENA_LIST_BASE,
    S_ARENA_LIST_E0,
    S_ARENA_LIST_E1,
    S_ARENA_LIST_E2,
    S_ARENA_LIST_E3,
    S_ARENA_LIST_T0,
    S_ARENA_LIST_T1,
    S_ARENA_LIST_T2,
    S_ARENA_LIST_T3,
    S_ARENA_LIST_CDR0,
    S_ARENA_LIST_CDR1,
    S_ARENA_LIST_CDR2,
    S_ARENA_LIST_CDR3,
    S_ARENA_LIST_CDRT0,
    S_ARENA_LIST_CDRT1,
    S_ARENA_LIST_CDRT2,
    S_ARENA_LIST_CDRT3,
    S_ARENA_LIST_HAS_E0,
    S_ARENA_LIST_HAS_E1,
    S_ARENA_LIST_HAS_E2,
    S_ARENA_LIST_HAS_E3,
    S_ARENA_TRANSIENT_START=S_ARENA_WRITE_KIND,
    S_ARENA_TRANSIENT_END=S_ARENA_LIST_HAS_E3
};

/* Type tag values */
#define TYPE_NUMBER  0.0f
#define TYPE_BOOL    1.0f
#define TYPE_PAIR    2.0f
#define TYPE_CLOSURE 3.0f
#define TYPE_STRING  4.0f
#define TYPE_VECTOR  5.0f
#define TYPE_NIL     6.0f
#define TYPE_CONT    7.0f

#endif /* ESHKOL_LIB_BACKEND_SDNC_ISA_H */
