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

#ifndef _WIN32
#include <unistd.h>
#include <termios.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <errno.h>
#include <poll.h>
#else
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

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

#ifndef _WIN32
/* ───────────────────────── POSIX implementation ───────────────────────── */

static struct termios g_orig_termios;
static int g_raw_mode = 0;
static volatile int g_resized = 0;
static int g_term_width = 80;
static int g_term_height = 24;
static int g_cursor_cache_ready = 0;
static int g_cursor_cache_ok = 0;
static int g_cursor_cache_row = 0;
static int g_cursor_cache_col = 0;

/* SIGWINCH handler */
/**
 * @brief SIGWINCH signal handler: flags a pending resize and refreshes the cached terminal dimensions.
 *
 * Queries TIOCGWINSZ directly and updates g_term_width/g_term_height
 * from within the handler.
 *
 * @param sig Unused (the signal number).
 */
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
/**
 * @brief Restores the terminal's original termios settings and shows the cursor, if raw mode is active.
 *
 * Registered via atexit() in eshkol_term_init() so the terminal is
 * always left in a sane state when the process exits.
 */
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

/**
 * @brief Initializes terminal state for the agent TUI.
 *
 * Captures the current termios settings (so they can be restored
 * later), registers restore_terminal() via atexit(), queries the
 * initial terminal dimensions, and installs a SIGWINCH handler to
 * track resizes.
 *
 * @return 1 on success, 0 if the current termios could not be read (e.g. stdin is not a TTY).
 */
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

/**
 * @brief Restores the terminal to its original (cooked) mode.
 *
 * Thin wrapper around restore_terminal(); safe to call even if raw
 * mode was never entered.
 */
void eshkol_term_shutdown(void) {
    restore_terminal();
}

/**
 * @brief Switches the terminal into raw mode for direct key-by-key input.
 *
 * Disables canonical processing, echo, signal generation, and various
 * input/output translations, and configures a 100ms read timeout
 * (VMIN=0, VTIME=1) instead of blocking indefinitely. No-op if raw
 * mode is already active.
 */
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

/**
 * @brief Restores the terminal's original termios settings, leaving raw mode.
 *
 * No-op if the terminal is not currently in raw mode.
 */
void eshkol_term_cooked_mode(void) {
    if (!g_raw_mode) return;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &g_orig_termios);
    g_raw_mode = 0;
}

/** @brief Returns the last known terminal width in columns. */
int eshkol_term_width(void)  { return g_term_width;  }
/** @brief Returns the last known terminal height in rows. */
int eshkol_term_height(void) { return g_term_height; }

/**
 * @brief Checks and clears the pending-resize flag set by the SIGWINCH handler.
 *
 * @return 1 if a resize occurred since the last call, 0 otherwise.
 */
int eshkol_term_resized(void) {
    int r = g_resized;
    g_resized = 0;
    return r;
}

/**
 * @brief Reads a single key from stdin, blocking indefinitely until one is available.
 *
 * Thin wrapper around eshkol_term_read_key_timeout() with an infinite timeout.
 */
int eshkol_term_read_key(void) {
    return eshkol_term_read_key_timeout(-1);
}

/**
 * @brief Reads and decodes a single key press from stdin, with an optional timeout.
 *
 * Recognizes control characters (Ctrl+A..Ctrl+Z), Tab, Enter,
 * Backspace, and multi-byte ANSI escape sequences (arrow keys,
 * Home/End, Delete, Page Up/Down, and F1-F4 in both `ESC [` and
 * `ESC O` forms), decoding them into the KEY_* constants defined
 * above. Continuation bytes of an escape sequence are read with a
 * short 50ms poll so a bare Escape key press (no follow-up bytes) is
 * still reported promptly as KEY_ESCAPE. Any other byte is returned
 * as its raw character code.
 *
 * @param timeout_ms Milliseconds to wait for the first byte; negative
 *   blocks indefinitely. Escape-sequence continuation bytes use their
 *   own fixed short timeouts regardless of this value.
 * @return The key code (a raw byte value or a KEY_* constant), or -1 on timeout or read error.
 */
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

/**
 * @brief Clears the terminal screen and moves the cursor to the home position.
 */
void eshkol_term_clear(void) {
    write(STDOUT_FILENO, "\033[2J\033[H", 7);
}

/**
 * @brief Moves the terminal cursor to the given 1-based row and column.
 *
 * @param row Target row (1-based).
 * @param col Target column (1-based).
 */
void eshkol_term_move_to(int row, int col) {
    char buf[32];
    int len = snprintf(buf, sizeof(buf), "\033[%d;%dH", row, col);
    write(STDOUT_FILENO, buf, len);
}

/**
 * @brief Queries the terminal for the cursor's current row and column via a DSR escape sequence.
 *
 * Temporarily switches stdin to non-canonical, non-echoing mode (if
 * possible), sends the Device Status Report request (`ESC [6n`), and
 * parses the terminal's `ESC [row;colR` reply, polling up to 100ms
 * per byte read. Requires both stdin and stdout to be TTYs. On any
 * failure, @p row and @p col are set to 0.
 *
 * @param row Out-parameter for the 1-based cursor row.
 * @param col Out-parameter for the 1-based cursor column.
 * @return 1 on success, 0 if not a TTY, the write failed, or the reply could not be parsed.
 */
int eshkol_term_cursor_pos(int* row, int* col) {
    if (row) *row = 0;
    if (col) *col = 0;
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

    char buf[32];
    /* Send DSR (Device Status Report) */
    if (write(STDOUT_FILENO, "\033[6n", 4) != 4) {
        if (restore) tcsetattr(STDIN_FILENO, TCSANOW, &orig);
        return 0;
    }

    /* Read response: ESC [ row ; col R */
    int i = 0;
    while (i < (int)sizeof(buf) - 1) {
        struct pollfd pfd = { .fd = STDIN_FILENO, .events = POLLIN };
        if (poll(&pfd, 1, 100) <= 0) break;
        if (read(STDIN_FILENO, &buf[i], 1) != 1) break;
        if (buf[i++] == 'R') break;
    }
    buf[i] = '\0';
    int ok = (sscanf(buf, "\033[%d;%dR", row, col) == 2) ||
             (sscanf(buf, "\033[%d;%d", row, col) == 2);
    if (restore) tcsetattr(STDIN_FILENO, TCSANOW, &orig);
    if (!ok) {
        *row = 0;
        *col = 0;
    }
    return ok;
}

/**
 * @brief Refreshes the cached cursor position by querying the terminal.
 *
 * Updates g_cursor_cache_row/col/ok via eshkol_term_cursor_pos() and marks the cache as ready.
 */
static void eshkol_term_refresh_cursor_cache(void) {
    g_cursor_cache_row = 0;
    g_cursor_cache_col = 0;
    g_cursor_cache_ok = eshkol_term_cursor_pos(&g_cursor_cache_row, &g_cursor_cache_col);
    g_cursor_cache_ready = 1;
}

/**
 * @brief Returns the terminal's current cursor row, always re-querying the terminal.
 *
 * Also refreshes the cursor cache so an immediately following call to
 * eshkol_term_cursor_col() can reuse the result instead of issuing a
 * second DSR round-trip.
 *
 * @return The 1-based cursor row, or 0 if the query failed.
 */
int eshkol_term_cursor_row(void) {
    eshkol_term_refresh_cursor_cache();
    return g_cursor_cache_ok ? g_cursor_cache_row : 0;
}

/**
 * @brief Returns the terminal's current cursor column.
 *
 * Reuses a cached cursor query if one was just populated by
 * eshkol_term_cursor_row(); otherwise queries the terminal directly.
 * The cache is marked stale after this call either way.
 *
 * @return The 1-based cursor column, or 0 if the query failed.
 */
int eshkol_term_cursor_col(void) {
    if (!g_cursor_cache_ready) eshkol_term_refresh_cursor_cache();
    int col = g_cursor_cache_ok ? g_cursor_cache_col : 0;
    g_cursor_cache_ready = 0;
    return col;
}

/** @brief Makes the terminal cursor visible. */
void eshkol_term_show_cursor(void) {
    write(STDOUT_FILENO, "\033[?25h", 6);
}

/** @brief Hides the terminal cursor. */
void eshkol_term_hide_cursor(void) {
    write(STDOUT_FILENO, "\033[?25l", 6);
}

/**
 * @brief Writes a raw string directly to stdout via write(2), bypassing stdio buffering.
 *
 * @param str NUL-terminated string to write; NULL is a no-op.
 */
void eshkol_term_write(const char* str) {
    if (str) write(STDOUT_FILENO, str, strlen(str));
}

/** @brief Flushes stdio's output buffer for stdout. */
void eshkol_term_flush(void) {
    /* stdout is line-buffered or fully-buffered; fsync the fd */
    fflush(stdout);
}

/**
 * @brief Sets the terminal window/tab title via an OSC 0 escape sequence.
 *
 * @param title NUL-terminated title string; NULL is a no-op. Silently
 *   truncated if it would overflow the internal 256-byte buffer.
 */
void eshkol_term_set_title(const char* title) {
    if (!title) return;
    char buf[256];
    int len = snprintf(buf, sizeof(buf), "\033]0;%s\007", title);
    write(STDOUT_FILENO, buf, len);
}

#else /* _WIN32 */
/* ───────────────────────── Windows console implementation ─────────────── */

static HANDLE g_term_input = INVALID_HANDLE_VALUE;
static HANDLE g_term_output = INVALID_HANDLE_VALUE;
static DWORD g_term_input_mode = 0;
static DWORD g_term_output_mode = 0;
static int g_term_modes_saved = 0;
static int g_raw_mode = 0;
static int g_term_width = 80;
static int g_term_height = 24;

static void refresh_dimensions(void) {
    CONSOLE_SCREEN_BUFFER_INFO info;
    if (g_term_output != INVALID_HANDLE_VALUE &&
        GetConsoleScreenBufferInfo(g_term_output, &info)) {
        g_term_width = (int)(info.srWindow.Right - info.srWindow.Left + 1);
        g_term_height = (int)(info.srWindow.Bottom - info.srWindow.Top + 1);
    }
}

static void restore_terminal(void) {
    if (g_term_modes_saved) {
        SetConsoleMode(g_term_input, g_term_input_mode);
        SetConsoleMode(g_term_output, g_term_output_mode);
    }
    g_raw_mode = 0;
    if (g_term_output != INVALID_HANDLE_VALUE) {
        CONSOLE_CURSOR_INFO cursor;
        if (GetConsoleCursorInfo(g_term_output, &cursor)) {
            cursor.bVisible = TRUE;
            SetConsoleCursorInfo(g_term_output, &cursor);
        }
    }
}

int eshkol_term_init(void) {
    g_term_input = GetStdHandle(STD_INPUT_HANDLE);
    g_term_output = GetStdHandle(STD_OUTPUT_HANDLE);
    if (g_term_input == INVALID_HANDLE_VALUE ||
        g_term_output == INVALID_HANDLE_VALUE ||
        !GetConsoleMode(g_term_input, &g_term_input_mode) ||
        !GetConsoleMode(g_term_output, &g_term_output_mode)) {
        return 0;
    }
    g_term_modes_saved = 1;
    /* VT output keeps the public terminal behaviour identical in ConPTY,
       Windows Terminal, and the legacy console host. */
    SetConsoleMode(g_term_output,
                   g_term_output_mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING |
                   DISABLE_NEWLINE_AUTO_RETURN);
    refresh_dimensions();
    atexit(restore_terminal);
    return 1;
}

void eshkol_term_shutdown(void) { restore_terminal(); }

void eshkol_term_raw_mode(void) {
    if (!g_term_modes_saved || g_raw_mode) return;
    DWORD mode = g_term_input_mode;
    mode &= ~(ENABLE_ECHO_INPUT | ENABLE_LINE_INPUT | ENABLE_PROCESSED_INPUT |
              ENABLE_QUICK_EDIT_MODE);
    mode |= ENABLE_EXTENDED_FLAGS | ENABLE_WINDOW_INPUT;
#ifdef ENABLE_VIRTUAL_TERMINAL_INPUT
    mode |= ENABLE_VIRTUAL_TERMINAL_INPUT;
#endif
    if (SetConsoleMode(g_term_input, mode)) g_raw_mode = 1;
}

void eshkol_term_cooked_mode(void) {
    if (g_term_modes_saved && g_raw_mode) {
        SetConsoleMode(g_term_input, g_term_input_mode);
        g_raw_mode = 0;
    }
}

int eshkol_term_width(void) { refresh_dimensions(); return g_term_width; }
int eshkol_term_height(void) { refresh_dimensions(); return g_term_height; }

int eshkol_term_resized(void) {
    int old_width = g_term_width;
    int old_height = g_term_height;
    refresh_dimensions();
    return old_width != g_term_width || old_height != g_term_height;
}

static int translate_key(const KEY_EVENT_RECORD* key) {
    if (!key->bKeyDown) return -1;
    switch (key->wVirtualKeyCode) {
        case VK_UP: return KEY_UP;
        case VK_DOWN: return KEY_DOWN;
        case VK_LEFT: return KEY_LEFT;
        case VK_RIGHT: return KEY_RIGHT;
        case VK_HOME: return KEY_HOME;
        case VK_END: return KEY_END;
        case VK_BACK: return KEY_BACKSPACE;
        case VK_DELETE: return KEY_DELETE;
        case VK_TAB: return KEY_TAB;
        case VK_RETURN: return KEY_ENTER;
        case VK_ESCAPE: return KEY_ESCAPE;
        case VK_PRIOR: return KEY_PAGE_UP;
        case VK_NEXT: return KEY_PAGE_DOWN;
        case VK_F1: return KEY_F1;
        case VK_F2: return KEY_F2;
        case VK_F3: return KEY_F3;
        case VK_F4: return KEY_F4;
        default: break;
    }
    if ((key->dwControlKeyState & (LEFT_CTRL_PRESSED | RIGHT_CTRL_PRESSED)) &&
        key->wVirtualKeyCode >= 'A' && key->wVirtualKeyCode <= 'Z') {
        return (int)(key->wVirtualKeyCode - 'A' + 1);
    }
    if (key->uChar.UnicodeChar != 0) return (int)key->uChar.UnicodeChar;
    return -1;
}

int eshkol_term_read_key(void) { return eshkol_term_read_key_timeout(-1); }

int eshkol_term_read_key_timeout(int timeout_ms) {
    if (g_term_input == INVALID_HANDLE_VALUE) return -1;
    DWORD wait_ms = timeout_ms < 0 ? INFINITE : (DWORD)timeout_ms;
    ULONGLONG deadline = timeout_ms < 0 ? 0 : GetTickCount64() + wait_ms;
    for (;;) {
        DWORD remaining = INFINITE;
        if (timeout_ms >= 0) {
            ULONGLONG now = GetTickCount64();
            if (now >= deadline) return -1;
            remaining = (DWORD)(deadline - now);
        }
        if (WaitForSingleObject(g_term_input, remaining) != WAIT_OBJECT_0) return -1;
        INPUT_RECORD rec;
        DWORD count = 0;
        if (!ReadConsoleInputW(g_term_input, &rec, 1, &count) || count != 1) return -1;
        if (rec.EventType == WINDOW_BUFFER_SIZE_EVENT) {
            refresh_dimensions();
            continue;
        }
        if (rec.EventType == KEY_EVENT) {
            int value = translate_key(&rec.Event.KeyEvent);
            if (value >= 0) return value;
        }
    }
}

void eshkol_term_clear(void) {
    if (g_term_output == INVALID_HANDLE_VALUE) return;
    CONSOLE_SCREEN_BUFFER_INFO info;
    if (!GetConsoleScreenBufferInfo(g_term_output, &info)) return;
    DWORD cells = (DWORD)info.dwSize.X * (DWORD)info.dwSize.Y;
    DWORD written = 0;
    COORD origin = {0, 0};
    FillConsoleOutputCharacterW(g_term_output, L' ', cells, origin, &written);
    FillConsoleOutputAttribute(g_term_output, info.wAttributes, cells, origin, &written);
    SetConsoleCursorPosition(g_term_output, origin);
}

void eshkol_term_move_to(int row, int col) {
    if (g_term_output == INVALID_HANDLE_VALUE) return;
    COORD pos = {(SHORT)(col > 0 ? col - 1 : 0), (SHORT)(row > 0 ? row - 1 : 0)};
    SetConsoleCursorPosition(g_term_output, pos);
}

int eshkol_term_cursor_pos(int* row, int* col) {
    if (row) *row = 0;
    if (col) *col = 0;
    if (!row || !col || g_term_output == INVALID_HANDLE_VALUE) return 0;
    CONSOLE_SCREEN_BUFFER_INFO info;
    if (!GetConsoleScreenBufferInfo(g_term_output, &info)) return 0;
    *row = (int)info.dwCursorPosition.Y + 1;
    *col = (int)info.dwCursorPosition.X + 1;
    return 1;
}

int eshkol_term_cursor_row(void) {
    int row = 0, col = 0;
    return eshkol_term_cursor_pos(&row, &col) ? row : 0;
}

int eshkol_term_cursor_col(void) {
    int row = 0, col = 0;
    return eshkol_term_cursor_pos(&row, &col) ? col : 0;
}

static void set_cursor_visibility(BOOL visible) {
    if (g_term_output == INVALID_HANDLE_VALUE) return;
    CONSOLE_CURSOR_INFO cursor;
    if (GetConsoleCursorInfo(g_term_output, &cursor)) {
        cursor.bVisible = visible;
        SetConsoleCursorInfo(g_term_output, &cursor);
    }
}

void eshkol_term_show_cursor(void) { set_cursor_visibility(TRUE); }
void eshkol_term_hide_cursor(void) { set_cursor_visibility(FALSE); }

void eshkol_term_write(const char* text) {
    if (!text) return;
    DWORD written = 0;
    if (g_term_output != INVALID_HANDLE_VALUE &&
        GetFileType(g_term_output) == FILE_TYPE_CHAR) {
        WriteConsoleA(g_term_output, text, (DWORD)strlen(text), &written, NULL);
    } else if (g_term_output != INVALID_HANDLE_VALUE) {
        WriteFile(g_term_output, text, (DWORD)strlen(text), &written, NULL);
    }
}

void eshkol_term_flush(void) { fflush(stdout); }

void eshkol_term_set_title(const char* title) {
    if (title) SetConsoleTitleA(title);
}

#endif /* _WIN32 */
