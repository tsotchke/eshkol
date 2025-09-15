/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */
#include <eshkol/logger.h>

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <execinfo.h>
#include <cxxabi.h>
#include <unistd.h>
#include <time.h>

#include <mutex>

#define COLOR_PREFIX "\033["
#define BOLD COLOR_PREFIX "1m"
#define RED_COLOR BOLD COLOR_PREFIX "31m"
#define GREEN_COLOR BOLD COLOR_PREFIX "32m"
#define YELLOW_COLOR BOLD COLOR_PREFIX "33m"
#define BLUE_COLOR BOLD COLOR_PREFIX "34m"
#define MAGENTA_COLOR BOLD COLOR_PREFIX "35m"
#define CYAN_COLOR BOLD COLOR_PREFIX "36m"
#define WHITE_COLOR BOLD COLOR_PREFIX "37m"
#define RESET_COLOR COLOR_PREFIX "0m"

static eshkol_logger_t max = ESHKOL_NOTICE;

static const char *log_name[] = {
    "fatal",
    "error",
    "warning",
    "notice",
    "info",
    "debug"
};
static const char *log_color[] = {
    RED_COLOR,
    RED_COLOR,
    YELLOW_COLOR,
    BLUE_COLOR,
    WHITE_COLOR,
    GREEN_COLOR
};

static std::mutex log_mut;

void eshkol_set_logger_level(eshkol_logger_t level)
{
    std::lock_guard<std::mutex> log_lock(log_mut);
    max = level;
}

static void eshkol_vprintf(eshkol_logger_t level, const char *msg, va_list arg)
{
    std::lock_guard<std::mutex> log_lock(log_mut);
    const char *prefix = log_name[level];
    const char *color = log_color[level];

    if (max >= level) {
        time_t t = time(nullptr);
        char time_buf[128] = {};
        bool append_newline = msg[strlen(msg) - 1] != '\n';

        strftime(time_buf, 128, "%x (%X)", localtime(&t));

        printf("%s%10s:%s ", color, prefix, RESET_COLOR);
        vprintf(msg, arg);
        if (append_newline) printf("\n");
    }

    if (level == ESHKOL_FATAL) exit(1);
}

void eshkol_printf(eshkol_logger_t level, const char *msg, ...)
{
    va_list ap;

    va_start(ap, msg);
    eshkol_vprintf(level, msg, ap);
    va_end(ap);
}

void eshkol_stacktrace(eshkol_logger_t level)
{
    std::lock_guard<std::mutex> log_lock(log_mut);
    const uint64_t max_frames = 63;
    const char *prefix = log_name[level];
    const char *color = log_color[level];
    time_t t = time(nullptr);
    char time_buf[128] = {};

    void *addrlist[max_frames+1];
    int addrlen;
    char **symbollist;
    size_t func_name_size = 256;
    char *funcname = nullptr;

    if (level > max) return;

    strftime(time_buf, 128, "%x (%X)", localtime(&t));
    addrlen = backtrace(addrlist, sizeof(addrlist) / sizeof(void*));
    symbollist = backtrace_symbols(addrlist, addrlen);
    funcname = (char*) malloc(func_name_size);

    for (int i = 1; i < addrlen; ++i) {
        char *begin_name = nullptr;
        char *begin_offset = nullptr;
        char *end_offset = nullptr;

        for (char *p = symbollist[i]; *p; ++p) {
            if (*p == '(')
                begin_name = p;
            else if (*p == '+')
                begin_offset = p;
            else if (*p == ')' && begin_offset != nullptr) {
                end_offset = p;
                break;
            }
        }

        if (begin_name != nullptr &&
            begin_offset != nullptr &&
            end_offset != nullptr &&
            begin_name < begin_offset) {
            int status;
            char *ret = nullptr;

            *begin_name++ = '\0';
            *begin_offset++ = '\0';
            *end_offset++ = '\0';

            ret = abi::__cxa_demangle(begin_name, funcname, &func_name_size, &status);

            if (status == 0) {
                funcname = ret;
                printf(
                    "%s%10s:%s  %s (%s+%s)\n",
                    color, prefix, RESET_COLOR,
                    symbollist[i], funcname, begin_offset
                );
            } else {
                printf(
                    "%s%10s:%s  %s (%s+%s)\n",
                    color, prefix, RESET_COLOR,
                    symbollist[i], begin_name, begin_offset
                );
            }
        } else {
            printf("%s%10s:%s  %s\n", color, prefix, RESET_COLOR, symbollist[i]);
        }
    }

    free(funcname);
    free(symbollist);

    if (level == ESHKOL_FATAL) exit(1);
}
