#include <stdio.h>
#include <unistd.h>

int eshkol_term_read_key_timeout(int timeout_ms);

enum {
    ESHKOL_TERM_READ_TIMEOUT = -1,
    ESHKOL_TERM_READ_EOF = -2,
    ESHKOL_TERM_READ_IO_ERROR = -3
};

static int fail(const char* label, int got, int expected) {
    fprintf(stderr, "%s: expected %d, got %d\n", label, expected, got);
    return 1;
}

int main(void) {
    int saved_stdin = dup(STDIN_FILENO);
    int input[2];
    if (saved_stdin < 0 || pipe(input) != 0) return 1;
    if (dup2(input[0], STDIN_FILENO) < 0) return 1;
    close(input[0]);

    int status = eshkol_term_read_key_timeout(10);
    if (status != ESHKOL_TERM_READ_TIMEOUT)
        return fail("empty input timeout", status, ESHKOL_TERM_READ_TIMEOUT);

    close(input[1]);
    status = eshkol_term_read_key_timeout(10);
    if (status != ESHKOL_TERM_READ_EOF)
        return fail("closed input EOF", status, ESHKOL_TERM_READ_EOF);

    close(STDIN_FILENO);
    status = eshkol_term_read_key_timeout(10);
    if (status != ESHKOL_TERM_READ_IO_ERROR)
        return fail("invalid input descriptor", status, ESHKOL_TERM_READ_IO_ERROR);

    dup2(saved_stdin, STDIN_FILENO);
    close(saved_stdin);
    puts("PASS: terminal read distinguishes timeout, EOF, and I/O error");
    return 0;
}
