#include "eval.h"
#include "reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void rep(const char *line) {
    const char *p = line;
    char *err = NULL;
    for (;;) {
        while (*p && (*p==' '||*p=='\n'||*p=='\t'||*p=='\r'||*p==';')) {
            if (*p==';') { while (*p && *p!='\n') p++; } else p++;
        }
        if (*p == 0) break;
        const char *rest;
        Value *expr = read_sexpr(p, &rest, &err);
        if (!expr) {
            if (err) { fprintf(stderr, "read error: %s\n", err); free(err); }
            break;
        }
        p = rest;
        Value *r = slermes_eval(expr, g_env);
        v_decref(expr);
        if (r) {
            if (r->type == V_STR) fprintf(stderr, "error: %s\n", r->as.str);
            else { v_print(r); printf("\n"); }
            v_decref(r);
        }
        if (err) { free(err); err = NULL; }
    }
}

int main(int argc, char **argv) {
    eval_init();
    if (argc > 1) {
        /* run file */
        FILE *f = fopen(argv[1], "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        char *buf = malloc(sz + 1); size_t rd = fread(buf, 1, sz, f); (void)rd; buf[sz] = 0; fclose(f);
        /* read all, eval each, last value is returned/displayed by program itself */
        const char *p = buf; char *err = NULL;
        for (;;) {
            while (*p && (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r' ||
                          (*p == ';' && (p[1]==' '||p[1]=='\n'||1)))) {
                if (*p == ';') { while (*p && *p != '\n') p++; }
                else p++;
            }
            if (*p == 0) break;
            const char *rest;
            Value *e = read_sexpr(p, &rest, &err);
            if (!e) {
                if (err) { fprintf(stderr, "read error: %s\n", err); free(err); }
                free(buf); return 1;
            }
            Value *r = slermes_eval(e, g_env);
            v_decref(e);
            if (r && r->type == V_STR) fprintf(stderr, "error: %s\n", r->as.str);
            v_decref(r);
            p = rest;
        }
        free(buf);
        return 0;
    }
    /* REPL */
    char line[4096];
    printf("slermes-eshkol> ");
    fflush(stdout);
    while (fgets(line, sizeof line, stdin)) {
        /* strip trailing newline */
        size_t n = strlen(line); while (n && (line[n-1]=='\n'||line[n-1]=='\r')) line[--n]=0;
        rep(line);
        printf("slermes-eshkol> ");
        fflush(stdout);
    }
    printf("\n");
    return 0;
}
