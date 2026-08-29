/**
 * @file vm_lifecycle.c
 * @brief VM instance lifecycle (vm_create/vm_free) and the hand-assembled
 *        test-program helpers used by vm_tests.c.
 *
 * These are the entry points around the interpreter rather than part of it,
 * so they live beside vm_run.c instead of inside it.
 *
 */

/** @brief Mnemonic names for the first 38 base opcodes, indexed by opcode
 *         value, used by the test-program bytecode disassembler/printer
 *         below (extended opcodes beyond OP_NATIVE_CALL are not covered). */
static const char* opnames[] = {
    "NOP","CONST","NIL","TRUE","FALSE","POP","DUP",
    "ADD","SUB","MUL","DIV","MOD","NEG","ABS",
    "EQ","LT","GT","LE","GE","NOT",
    "GETL","SETL","GETUP","SETUP",
    "CLOSURE","CALL","TCALL","RET",
    "JUMP","JIF","LOOP",
    "CONS","CAR","CDR","NULLP",
    "PRINT","HALT","NATIVE"
};

/** @brief Append one bytecode instruction (@p op, @p operand) to @p vm's
 *         fixed-size (4096-instruction) test-program code buffer. */
static void emit(VM* vm, uint8_t op, int32_t operand) {
    if (vm->code_len >= 4096) return;
    vm->code[vm->code_len++] = (Instr){op, operand};
}

/** @brief Allocate and vm_init() a fresh VM instance with a 4096-instruction
 *         code buffer, for use by hand-assembled test programs (see
 *         vm_tests.c). */
VM* vm_create(void) {
    VM* vm = (VM*)calloc(1, sizeof(VM));
    if (!vm) return NULL;
    vm_init(vm);
    vm->code = (Instr*)calloc(4096, sizeof(Instr));
    if (!vm->code) { free(vm); return NULL; }
    return vm;
}
/** @brief Release all resources owned by @p vm (open regex handles,
 *         dlopen'd libraries, the heap's arena, and the code buffer) and
 *         free @p vm itself. */
void vm_free(VM* vm) {
    vm_regex_free_all(vm);
    vm_dlopen_close_all(vm);
    heap_destroy(&vm->heap);
    free(vm->code);
    free(vm->constants);
    vm->constants = NULL;
    vm->const_cap = 0;
    free(vm);
}
