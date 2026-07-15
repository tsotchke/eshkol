#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int32_t eshkol_compression_available(void);
int32_t eshkol_deflate(const char*, int32_t, char*, int32_t);
int32_t eshkol_inflate_data(const char*, int32_t, char*, int32_t);
int32_t eshkol_gzip(const char*, int32_t, char*, int32_t);
int32_t eshkol_gunzip(const char*, int32_t, char*, int32_t);

int32_t eshkol_ts_available(void);
int64_t eshkol_ts_parser_new(const char*);
void eshkol_ts_parser_free(int64_t);
int64_t eshkol_ts_parse(int64_t, const char*, int32_t);
void eshkol_ts_tree_free(int64_t);
int32_t eshkol_ts_tree_root(int64_t, char*, int32_t);
int32_t eshkol_ts_node_text(int64_t, uint32_t, uint32_t, char*, int32_t);
int64_t eshkol_ts_query_new(const char*, const char*);
void eshkol_ts_query_free(int64_t);
int32_t eshkol_ts_query_matches(int64_t, int64_t, int32_t, char*, int32_t, int32_t*);
int32_t eshkol_ts_tree_sexp(int64_t, char*, int32_t);

int32_t eshkol_yoga_available(void);
int64_t eshkol_yoga_node_create(void);
void eshkol_yoga_node_free(int64_t);
void eshkol_yoga_node_set_float(int64_t, int32_t, double);
void eshkol_yoga_node_set_int(int64_t, int32_t, int32_t);
void eshkol_yoga_node_add_child(int64_t, int64_t, int32_t);
void eshkol_yoga_node_calculate(int64_t, double, double);
double eshkol_yoga_node_get_computed(int64_t, int32_t);

static int g_failed = 0;

static void check(int condition, const char* name) {
    printf("%s: %s\n", name, condition ? "PASS" : "FAIL");
    if (!condition) ++g_failed;
}

static void test_compression(void) {
    static const unsigned char payload[] = {
        0x00, 0x01, 0x7f, 0x80, 0xff, 'E', 's', 'h', 'k', 'o', 'l', 0x00
    };
    unsigned char compressed[256];
    unsigned char restored[256];

    check(eshkol_compression_available() == 1, "compression is production-enabled");

    int32_t compressed_len = eshkol_deflate((const char*)payload,
        (int32_t)sizeof(payload), (char*)compressed, (int32_t)sizeof(compressed));
    int32_t restored_len = compressed_len > 0
        ? eshkol_inflate_data((const char*)compressed, compressed_len,
              (char*)restored, (int32_t)sizeof(restored))
        : -1;
    check(compressed_len > 0 && restored_len == (int32_t)sizeof(payload) &&
              memcmp(payload, restored, sizeof(payload)) == 0,
          "zlib binary round-trip");

    compressed_len = eshkol_gzip((const char*)payload, (int32_t)sizeof(payload),
        (char*)compressed, (int32_t)sizeof(compressed));
    restored_len = compressed_len > 0
        ? eshkol_gunzip((const char*)compressed, compressed_len,
              (char*)restored, (int32_t)sizeof(restored))
        : -1;
    check(compressed_len > 0 && restored_len == (int32_t)sizeof(payload) &&
              memcmp(payload, restored, sizeof(payload)) == 0,
          "gzip binary round-trip");

    compressed_len = eshkol_deflate(NULL, 0, (char*)compressed, (int32_t)sizeof(compressed));
    restored_len = compressed_len > 0
        ? eshkol_inflate_data((const char*)compressed, compressed_len,
              (char*)restored, (int32_t)sizeof(restored))
        : -1;
    check(compressed_len > 0 && restored_len == 0, "empty stream round-trip");
    check(eshkol_gunzip("not-gzip", 8, (char*)restored, (int32_t)sizeof(restored)) == -1,
          "invalid gzip rejected");
}

typedef struct {
    const char* language;
    const char* source;
} language_case_t;

static void test_tree_sitter(void) {
    static const language_case_t cases[] = {
        {"javascript", "function add(a,b){return a+b;}"},
        {"typescript", "const x: number = 1;"},
        {"tsx", "const x = <div>Hello</div>;"},
        {"python", "def f(x):\n  return x + 1\n"},
        {"rust", "fn main() { println!(\"hi\"); }"},
        {"go", "package main\nfunc main() {}\n"},
        {"c", "int main(void) { return 0; }"},
        {"cpp", "template<class T> T id(T x) { return x; }"},
        {"java", "class A { int f() { return 1; } }"},
        {"ruby", "def f(x)\n  x + 1\nend\n"},
        {"bash", "f() { echo ok; }\n"},
    };
    char buffer[4096];

    check(eshkol_ts_available() == 1, "tree-sitter is production-enabled");
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        int64_t parser = eshkol_ts_parser_new(cases[i].language);
        int64_t tree = parser > 0
            ? eshkol_ts_parse(parser, cases[i].source, (int32_t)strlen(cases[i].source))
            : -1;
        int32_t root_len = tree > 0
            ? eshkol_ts_tree_root(tree, buffer, (int32_t)sizeof(buffer))
            : -1;
        int32_t sexp_len = tree > 0
            ? eshkol_ts_tree_sexp(tree, buffer, (int32_t)sizeof(buffer))
            : -1;
        char label[96];
        snprintf(label, sizeof(label), "tree-sitter grammar %s", cases[i].language);
        check(parser > 0 && tree > 0 && root_len > 0 && sexp_len > 0, label);
        if (tree > 0) eshkol_ts_tree_free(tree);
        if (parser > 0) eshkol_ts_parser_free(parser);
    }

    char* source = (char*)malloc(32);
    memcpy(source, "let durable = value;", 21);
    int64_t parser = eshkol_ts_parser_new("javascript");
    int64_t tree = eshkol_ts_parse(parser, source, 20);
    memset(source, 'X', 20);
    free(source);
    int32_t text_len = eshkol_ts_node_text(tree, 4, 11, buffer, (int32_t)sizeof(buffer));
    check(text_len == 7 && memcmp(buffer, "durable", 7) == 0,
          "tree owns source bytes");

    int64_t query = eshkol_ts_query_new("javascript", "(identifier) @identifier");
    int32_t match_count = 0;
    int32_t match_len = eshkol_ts_query_matches(query, tree, 0, buffer,
        (int32_t)sizeof(buffer), &match_count);
    check(query > 0 && match_len > 0 && match_count >= 2,
          "tree-sitter query captures");
    int64_t wrong_query = eshkol_ts_query_new("python", "(identifier) @identifier");
    check(eshkol_ts_query_matches(wrong_query, tree, 0, buffer,
              (int32_t)sizeof(buffer), &match_count) == -1,
          "cross-language query rejected");

    eshkol_ts_query_free(wrong_query);
    eshkol_ts_query_free(query);
    eshkol_ts_tree_free(tree);
    eshkol_ts_parser_free(parser);
    check(eshkol_ts_parser_new("not-a-language") == -1,
          "unknown grammar rejected");
}

static void test_yoga(void) {
    check(eshkol_yoga_available() == 1, "Yoga is production-enabled");
    int64_t root = eshkol_yoga_node_create();
    int64_t left = eshkol_yoga_node_create();
    int64_t right = eshkol_yoga_node_create();
    check(root > 0 && left > 0 && right > 0, "Yoga nodes created");

    eshkol_yoga_node_set_int(root, 0, 2); /* YGFlexDirectionRow */
    eshkol_yoga_node_set_float(root, 9, 5.0);
    eshkol_yoga_node_set_float(left, 0, 30.0);
    eshkol_yoga_node_set_float(right, 6, 1.0);
    eshkol_yoga_node_add_child(root, left, 0);
    eshkol_yoga_node_add_child(root, right, 1);
    eshkol_yoga_node_calculate(root, 100.0, 20.0);
    check(fabs(eshkol_yoga_node_get_computed(left, 2) - 30.0) < 1e-9,
          "Yoga fixed width");
    check(fabs(eshkol_yoga_node_get_computed(right, 2) - 65.0) < 1e-9,
          "Yoga flex width");
    check(fabs(eshkol_yoga_node_get_computed(right, 0) - 35.0) < 1e-9,
          "Yoga gap offset");

    eshkol_yoga_node_free(root);
    int64_t recycled_root = eshkol_yoga_node_create();
    int64_t recycled_left = eshkol_yoga_node_create();
    int64_t recycled_right = eshkol_yoga_node_create();
    check(recycled_root > 0 && recycled_left > 0 && recycled_right > 0,
          "Yoga subtree handles safely recycled");
    eshkol_yoga_node_add_child(recycled_root, recycled_left, 0);
    eshkol_yoga_node_add_child(recycled_left, recycled_root, 0);
    eshkol_yoga_node_calculate(recycled_root, 10.0, 10.0);
    check(isfinite(eshkol_yoga_node_get_computed(recycled_root, 2)),
          "Yoga cycle insertion rejected");
    eshkol_yoga_node_free(recycled_root);
    eshkol_yoga_node_free(recycled_right);
}

int main(void) {
    test_compression();
    test_tree_sitter();
    test_yoga();
    printf("Agent production capabilities: %s (%d failures)\n",
           g_failed == 0 ? "PASS" : "FAIL", g_failed);
    return g_failed == 0 ? 0 : 1;
}
