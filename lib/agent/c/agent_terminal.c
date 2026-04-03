/*******************************************************************************
 * Terminal Control for Eshkol Agent TUI
 *
 * Raw mode, key reading with escape sequence handling, cursor control,
 * terminal dimensions, and SIGWINCH resize handling.
 *
 * Copyright (c) 2025 Eshkol Project
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <termios.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <errno.h>
#include <poll.h>

/* Forward declarations */
int eshkol_term_read_key_timeout(int timeout_ms);

/* Key code constants (match Eshkol FFI expectations) */
#define KEY_UP        1001
#define KEY_DOWN      1002
#define KEY_LEFT      1003
#define KEY_RIGHT     1004
#define KEY_HOME      1005
#define KEY_END       1006
#define KEY_BACKSPACE 1007
#define KEY_DELETE    1008
#define KEY_TAB       1009
#define KEY_ENTER     1010
#define KEY_ESCAPE    1011
#define KEY_PAGE_UP   1012
#define KEY_PAGE_DOWN 1013
#define KEY_F1        1014
#define KEY_F2        1015
#define KEY_F3        1016
#define KEY_F4        1017

static struct termios g_orig_termios;
static int g_raw_mode = 0;
static volatile int g_resized = 0;
static int g_term_width = 80;
static int g_term_height = 24;

/* SIGWINCH handler */
static void sigwinch_handler(int sig) {
    (void)sig;
    g_resized = 1;
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0) {
        g_term_width = ws.ws_col;
        g_term_height = ws.ws_row;
    }
}

/* Restore terminal on exit */
static void restore_terminal(void) {
    if (g_raw_mode) {
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &g_orig_termios);
        g_raw_mode = 0;
        /* Show cursor */
        write(STDOUT_FILENO, "\033[?25h", 6);
    }
}

/*******************************************************************************
 * Public API
 ******************************************************************************/

int eshkol_term_init(void) {
    /* Save original termios */
    if (tcgetattr(STDIN_FILENO, &g_orig_termios) != 0) return 0;
    atexit(restore_terminal);

    /* Get initial size */
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0) {
        g_term_width = ws.ws_col;
        g_term_height = ws.ws_row;
    }

    /* Install SIGWINCH handler */
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = sigwinch_handler;
    sigaction(SIGWINCH, &sa, NULL);

    return 1;
}

void eshkol_term_shutdown(void) {
    restore_terminal();
}

void eshkol_term_raw_mode(void) {
    if (g_raw_mode) return;
    struct termios raw = g_orig_termios;
    raw.c_iflag &= ~(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
    raw.c_oflag &= ~(OPOST);
    raw.c_cflag |= (CS8);
    raw.c_lflag &= ~(ECHO | ICANON | IEXTEN | ISIG);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 1;  /* 100ms timeout */
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    g_raw_mode = 1;
}

void eshkol_term_cooked_mode(void) {
    if (!g_raw_mode) return;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &g_orig_termios);
    g_raw_mode = 0;
}

int eshkol_term_width(void)  { return g_term_width;  }
int eshkol_term_height(void) { return g_term_height; }

int eshkol_term_resized(void) {
    int r = g_resized;
    g_resized = 0;
    return r;
}

int eshkol_term_read_key(void) {
    return eshkol_term_read_key_timeout(-1);
}

int eshkol_term_read_key_timeout(int timeout_ms) {
    unsigned char c;

    if (timeout_ms >= 0) {
        struct pollfd pfd = { .fd = STDIN_FILENO, .events = POLLIN };
        int ret = poll(&pfd, 1, timeout_ms);
        if (ret <= 0) return -1;
    }

    ssize_t n = read(STDIN_FILENO, &c, 1);
    if (n <= 0) return -1;

    /* Ctrl+A through Ctrl+Z */
    if (c >= 1 && c <= 26) return c;

    /* Tab */
    if (c == 9) return KEY_TAB;

    /* Enter */
    if (c == 10 || c == 13) return KEY_ENTER;

    /* Backspace (127 or 8) */
    if (c == 127 || c == 8) return KEY_BACKSPACE;

    /* Escape sequence */
    if (c == 27) {
        struct pollfd pfd = { .fd = STDIN_FILENO, .events = POLLIN };
        if (poll(&pfd, 1, 50) <= 0) return KEY_ESCAPE;

        unsigned char seq[5];
        n = read(STDIN_FILENO, &seq[0], 1);
        if (n <= 0) return KEY_ESCAPE;

        if (seq[0] == '[') {
            n = read(STDIN_FILENO, &seq[1], 1);
            if (n <= 0) return KEY_ESCAPE;

            switch (seq[1]) {
                case 'A': return KEY_UP;
                case 'B': return KEY_DOWN;
                case 'C': return KEY_RIGHT;
                case 'D': return KEY_LEFT;
                case 'H': return KEY_HOME;
                case 'F': return KEY_END;
                case '1':
                    n = read(STDIN_FILENO, &seq[2], 1);
                    if (n > 0 && seq[2] == '~') return KEY_HOME;
                    if (n > 0 && seq[2] == ';') {
                        read(STDIN_FILENO, &seq[3], 1);
                        read(STDIN_FILENO, &seq[4], 1);
                    }
                    return KEY_HOME;
                case '3':
                    n = read(STDIN_FILENO, &seq[2], 1);
                    if (n > 0 && seq[2] == '~') return KEY_DELETE;
                    return KEY_DELETE;
                case '4':
                    read(STDIN_FILENO, &seq[2], 1);
                    return KEY_END;
                case '5':
                    read(STDIN_FILENO, &seq[2], 1);
                    return KEY_PAGE_UP;
                case '6':
                    read(STDIN_FILENO, &seq[2], 1);
                    return KEY_PAGE_DOWN;
                default:
                    return KEY_ESCAPE;
            }
        } else if (seq[0] == 'O') {
            n = read(STDIN_FILENO, &seq[1], 1);
            if (n <= 0) return KEY_ESCAPE;
            switch (seq[1]) {
                case 'H': return KEY_HOME;
                case 'F': return KEY_END;
                case 'P': return KEY_F1;
                case 'Q': return KEY_F2;
                case 'R': return KEY_F3;
                case 'S': return KEY_F4;
                default: return KEY_ESCAPE;
            }
        }
        return KEY_ESCAPE;
    }

    /* Regular printable character */
    return (int)c;
}

void eshkol_term_clear(void) {
    write(STDOUT_FILENO, "\033[2J\033[H", 7);
}

void eshkol_term_move_to(int row, int col) {
    char buf[32];
    int len = snprintf(buf, sizeof(buf), "\033[%d;%dH", row, col);
    write(STDOUT_FILENO, buf, len);
}

int eshkol_term_cursor_pos(int* row, int* col) {
    char buf[32];
    /* Send DSR (Device Status Report) */
    write(STDOUT_FILENO, "\033[6n", 4);
    /* Read response: ESC [ row ; col R */
    int i = 0;
    while (i < (int)sizeof(buf) - 1) {
        struct pollfd pfd = { .fd = STDIN_FILENO, .events = POLLIN };
        if (poll(&pfd, 1, 100) <= 0) break;
        if (read(STDIN_FILENO, &buf[i], 1) != 1) break;
        if (buf[i] == 'R') break;
        i++;
    }
    buf[i] = '\0';
    if (sscanf(buf, "\033[%d;%d", row, col) != 2) return 0;
    return 1;
}

void eshkol_term_show_cursor(void) {
    write(STDOUT_FILENO, "\033[?25h", 6);
}

void eshkol_term_hide_cursor(void) {
    write(STDOUT_FILENO, "\033[?25l", 6);
}

void eshkol_term_write(const char* str) {
    if (str) write(STDOUT_FILENO, str, strlen(str));
}

void eshkol_term_flush(void) {
    /* stdout is line-buffered or fully-buffered; fsync the fd */
    fflush(stdout);
}

void eshkol_term_set_title(const char* title) {
    if (!title) return;
    char buf[256];
    int len = snprintf(buf, sizeof(buf), "\033]0;%s\007", title);
    write(STDOUT_FILENO, buf, len);
}
