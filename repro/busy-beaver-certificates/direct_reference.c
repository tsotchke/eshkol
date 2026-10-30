/* Independent raw-transition reference. No macro formulas or proof callbacks. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    const char *code = "1RB1LC_1RC1RB_1RD0LE_1LA1LD_1RZ0LA";
    enum { CELLS = 131072, ORIGIN = CELLS / 2 };
    unsigned char *tape = calloc(CELLS, 1);
    if (!tape) return 2;
    int head = ORIGIN, state = 0;
    uint64_t steps = 0;
    while (state != 5 && steps < 50000000) {
        if (head < 0 || head >= CELLS) { free(tape); return 3; }
        int offset = state * 7 + tape[head] * 3;
        tape[head] = (unsigned char)(code[offset] - '0');
        head += code[offset + 1] == 'R' ? 1 : -1;
        state = code[offset + 2] == 'Z' ? 5 : code[offset + 2] - 'A';
        ++steps;
    }
    if (state != 5 || head < 0 || head >= CELLS) { free(tape); return 3; }
    size_t ones = 0;
    for (size_t i = 0; i < CELLS; ++i) ones += tape[i];
    FILE *out = fopen(argv[1], "wb");
    if (!out) { free(tape); return 2; }
    size_t written = fwrite(tape, 1, CELLS, out);
    int closed = fclose(out);
    printf("%llu %d %d %zu\n", (unsigned long long)steps, head - ORIGIN, state, ones);
    free(tape);
    return written == CELLS && closed == 0 ? 0 : 2;
}
