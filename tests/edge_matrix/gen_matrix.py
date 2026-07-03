#!/usr/bin/env python3
"""gen_matrix.py — deterministic feature-pair composition matrix generator.

Pillar P2 of the adversarial testing campaign
(.swarm/ADVERSARIAL_TESTING_CAMPAIGN.md): every recent compiler bug lived at
a FEATURE COMPOSITION point.  This generator enumerates the language surface
as feature AXES, each with canonical PRODUCER forms (expressions with a
value the generator knows) and CONTEXT forms (wrappers with identity
semantics: the wrapped hole's value is returned unchanged).  For each
ordered axis pair (A, B) it emits probe programs placing every producer of
A inside every context of B, self-checked against the generator-computed
expectation.

Design rules (do not break these when extending):
  * Deterministic: no randomness; iteration order is the declaration order
    below.  Same inputs -> byte-identical corpus.
  * Expectations are computed IN the generator.  A producer's `expected`
    is an Eshkol expression built via a DIFFERENT syntactic route than the
    producer whenever cheap (e.g. producer `'(1 2 3)` vs expected
    `(list 1 2 3)`), so a single broken feature cannot silently make both
    sides wrong in the same way.
  * Contexts must evaluate their hole EXACTLY ONCE (use `let` to avoid
    re-evaluation) so that effectful producers (fresh-counter probes) can
    detect double-evaluation bugs.
  * Every check body is wrapped in a defined function and then called,
    NOT inlined at top level, because the check harness itself must not
    sit on top of known-fragile top-level argument evaluation.  The
    `toplevel` axis probes top level explicitly.
  * Contexts may restrict `accepts` to a set of value types; incompatible
    producer/context combos are skipped and counted.

Usage:
  python3 tests/edge_matrix/gen_matrix.py                 # priority sweep (150 pairs)
  python3 tests/edge_matrix/gen_matrix.py --max-pairs 0   # ALL pairs
  python3 tests/edge_matrix/gen_matrix.py --list-axes
  python3 tests/edge_matrix/gen_matrix.py --emit-features # regenerate FEATURES.md
"""

import argparse
import os
import sys
from collections import OrderedDict

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

# Value types a producer can yield / a context can accept.
NUM = frozenset({"int", "double", "num"})  # num = bignum/rational
SCALAR = frozenset({"int", "double", "num", "bool", "sym", "char"})
ANY = None  # context accepts anything


class Producer:
    def __init__(self, name, expr, expected, vtype, top=None, requires=(),
                 effectful=False, note=""):
        self.name = name          # form name within the axis
        self.expr = expr          # Eshkol expression ({ID} allowed)
        self.expected = expected  # Eshkol expression, different route if possible
        self.vtype = vtype        # one of: int double num bool sym char str list
        self.top = top or []      # top-level prelude lines ({ID} allowed)
        self.requires = tuple(requires)
        self.effectful = effectful
        self.note = note


class Context:
    def __init__(self, name, template, top=None, accepts=ANY, requires=(),
                 note=""):
        self.name = name          # form name within the axis
        self.template = template  # contains {X} hole; {ID} allowed
        self.top = top or []      # top-level prelude lines ({X}/{ID} allowed)
        self.accepts = accepts    # None = any, else frozenset of vtypes
        self.requires = tuple(requires)
        self.note = note


class Axis:
    def __init__(self, name, doc, producers=(), contexts=()):
        self.name = name
        self.doc = doc
        self.producers = list(producers)
        self.contexts = list(contexts)


AXES = OrderedDict()


def axis(name, doc, producers=(), contexts=()):
    AXES[name] = Axis(name, doc, producers, contexts)


# Axes whose composition with ANYTHING is highest-risk (source: bug history —
# set!×capture, quote×let-tail, first-class×predicate, shadow×libc-symbol,
# tail-position miscompiles, variadic ABI).
HIGH_RISK = ("quote", "set_bang", "first_class", "tail_position",
             "shadowing", "variadic", "toplevel")

# ---------------------------------------------------------------------------
# Axis definitions
# ---------------------------------------------------------------------------

axis("quote", "quote / quasiquote / literal data", producers=[
    Producer("sym", "'edge-sym", '(string->symbol "edge-sym")', "sym"),
    Producer("list", "'(1 2 3)", "(list 1 2 3)", "list"),
    Producer("empty", "'()", "(list)", "list"),
    Producer("nested_quote", "(car ''nq)", '(string->symbol "quote")', "sym"),
    Producer("veclit_ref", "(vector-ref #(9 8 7) 2)", "7", "int"),
    Producer("qq_sugar", "`(1 ,(+ 1 1) ,@(list 3 4))", "(list 1 2 3 4)",
             "list"),
    Producer("qq_longform", "(quasiquote (1 (unquote (+ 1 1))))",
             "(list 1 2)", "list",
             note="long-form quasiquote must behave like the ` reader sugar"),
])

axis("set_bang", "set! mutation of locals and captured variables", producers=[
    Producer("local_mut", "(let ((sx 1)) (set! sx 41) (+ sx 1))", "42", "int"),
    Producer("closure_mut", "(let ((sy 0)) ((lambda () (set! sy 7))) sy)",
             "7", "int",
             note="lambda mutating an OUTER variable (bug-RR family)"),
], contexts=[
    Context("store_read", "(let ((sv{ID} #f)) (set! sv{ID} {X}) sv{ID})"),
    Context("lambda_store",
            "(let ((sw{ID} #f)) ((lambda () (set! sw{ID} {X}))) sw{ID})",
            note="hole value escapes via set! from inside a closure"),
    Context("set_twice",
            "(let ((sz{ID} 0)) (set! sz{ID} {X}) (set! sz{ID} sz{ID}) sz{ID})"),
])

axis("first_class", "builtins/lambdas as first-class values", producers=[
    Producer("builtin_var", "(let ((fp null?)) (fp '()))", "#t", "bool"),
    Producer("apply_builtin", "(apply + (list 20 22))", "42", "int"),
    Producer("builtin_from_list", "((car (list +)) 20 22)", "42", "int"),
    Producer("map_builtin", "(map car '((1 2) (3 4)))", "(list 1 3)", "list"),
], contexts=[
    Context("thunk_in_list", "((car (list (lambda () {X}))))"),
    Context("apply_id", "(apply (lambda (fa) fa) (list {X}))"),
    Context("map_hole", "(car (map (lambda (fi) {X}) (list 0)))"),
    Context("fn_arg_call", "((lambda (ff) (ff)) (lambda () {X}))"),
])

axis("tail_position", "expressions in tail position of defining forms",
     producers=[
    Producer("loop_tail_result",
             "(let tpl ((i 0)) (if (= i 100) 'end (tpl (+ i 1))))",
             '(string->symbol "end")', "sym"),
], contexts=[
    Context("define_tail", "(tp{ID})", top=["(define (tp{ID}) {X})"]),
    Context("let_tail_in_define", "(tq{ID})",
            top=["(define (tq{ID}) (let ((tu 1)) {X}))"],
            note="quote×let-body-tail-inside-define was a real bug"),
    Context("if_tail", "(if (> 2 1) {X} 'wrong)"),
    Context("begin_tail", "(begin 'first {X})"),
    Context("cond_else_tail", "(cond ((= 1 2) 'wrong) (else {X}))"),
])

axis("shadowing", "rebinding builtin / libc / parameter names", producers=[
    Producer("shadow_libc_local", "(let ((pow 3)) (+ pow 1))", "4", "int",
             note="user binding colliding with a libc symbol name"),
    Producer("shadow_builtin_rebind",
             "(let ((car cdr)) (car (list 1 2 3)))", "(list 2 3)", "list"),
    Producer("shadow_param", "((lambda (ps) (let ((ps 5)) ps)) 1)", "5",
             "int"),
], contexts=[
    Context("shadow_unused_builtin", "(let ((length 5)) {X})",
            note="`length` must not be used by any producer form"),
    Context("shadow_param_rebind",
            "((lambda (sp{ID}) (let ((sp{ID} {X})) sp{ID})) 'outer)"),
])

axis("variadic", "rest-argument lambdas and defines", producers=[
    Producer("all_rest_car", "((lambda vargs (car vargs)) 9 8 7)", "9", "int"),
    Producer("dotted_rest", "((lambda (va . vr) vr) 1 2 3)", "(list 2 3)",
             "list"),
    Producer("define_variadic", "(vf{ID} 7 8)", "(list 7 8)", "list",
             top=["(define (vf{ID} . vxs) vxs)"]),
], contexts=[
    Context("rest_car", "(car ((lambda vgs vgs) {X}))"),
    Context("dotted_first", "((lambda (vh . vt) vh) {X} 'p1 'p2)"),
    Context("apply_rest", "(apply (lambda (vk . vm) vk) (list {X} 0))"),
])

axis("let_family", "let / let* / letrec / named let", producers=[
    Producer("let_sum", "(let ((la 2) (lb 3)) (+ la lb))", "5", "int"),
    Producer("letstar_chain", "(let* ((lc 2) (ld (* lc lc))) ld)", "4", "int"),
    Producer("letrec_mutual",
             "(letrec ((ev? (lambda (n) (if (= n 0) #t (od? (- n 1)))))"
             " (od? (lambda (n) (if (= n 0) #f (ev? (- n 1))))))"
             " (ev? 10))", "#t", "bool"),
    Producer("named_let_sum",
             "(let nsum ((i 0) (s 0)) (if (= i 5) s (nsum (+ i 1) (+ s i))))",
             "10", "int"),
], contexts=[
    Context("let_bind", "(let ((lx{ID} {X})) lx{ID})"),
    Context("letstar_chain", "(let* ((ly{ID} {X}) (lz{ID} ly{ID})) lz{ID})"),
    Context("letrec_thunk", "(letrec ((lf{ID} (lambda () {X}))) (lf{ID}))"),
    Context("named_let_carry",
            "(let nl{ID} ((i 0) (acc #f))"
            " (if (= i 1) acc (nl{ID} (+ i 1) {X})))"),
])

axis("closures", "capture, sharing, multi-instance, escape", producers=[
    Producer("single_capture",
             "(let ((cn 10)) ((lambda (cx) (+ cx cn)) 5))", "15", "int"),
    Producer("sibling_set",
             "(let ((sn 0))"
             " (let ((inc (lambda () (set! sn (+ sn 1)))) (get (lambda () sn)))"
             " (inc) (inc) (get)))", "2", "int",
             note="two sibling closures sharing one set! variable"),
    Producer("multi_instance",
             "(begin (ca{ID}) (ca{ID}) (+ (ca{ID}) (cb{ID})))", "4", "int",
             top=["(define (mkc{ID}) (let ((mn 0))"
                  " (lambda () (set! mn (+ mn 1)) mn)))",
                  "(define ca{ID} (mkc{ID}))",
                  "(define cb{ID} (mkc{ID}))"],
             note="two instances of the same closure maker stay independent"),
    Producer("counter_once", "(ctr{ID})", "1", "int",
             top=["(define ctr{ID} (let ((qn 0))"
                  " (lambda () (set! qn (+ qn 1)) qn)))"],
             effectful=True,
             note="fresh counter: detects contexts evaluating the hole twice"),
], contexts=[
    Context("escaping_set",
            "(let ((eth{ID} #f)) (set! eth{ID} (lambda () {X})) (eth{ID}))"),
    Context("immediate_lambda", "((lambda () {X}))"),
])

axis("numeric_tower", "int / bignum / rational / double / exactness",
     producers=[
    Producer("int42", "42", "(- 50 8)", "int"),
    Producer("bignum_mul", "(* 99999999999 99999999999)",
             "9999999999800000000001", "num"),
    Producer("expt_2_64", "(expt 2 64)", "(* 4294967296 4294967296)", "num"),
    Producer("rational_third", "(/ 1 3)", "(/ 2 6)", "num"),
    Producer("double_half", "(+ 1.25 1.25)", "2.5", "double"),
    Producer("mixed_exactness", "(+ 1/2 0.25)", "0.75", "double"),
], contexts=[
    Context("plus_zero", "(+ {X} 0)", accepts=NUM),
    Context("times_one", "(* {X} 1)", accepts=NUM),
])

axis("strings", "string construction and slicing", producers=[
    Producer("append", '(string-append "ab" "cd")',
             '(string #\\a #\\b #\\c #\\d)', "str"),
    Producer("substr", '(substring "hello" 1 3)', '"el"', "str"),
    Producer("num_to_str", "(number->string 42)", '"42"', "str"),
    Producer("str_to_num", '(string->number "42")', "42", "int"),
], contexts=[
    Context("append_empty", '(string-append {X} "")',
            accepts=frozenset({"str"})),
    Context("roundtrip_sub",
            "(let ((rs{ID} {X})) (substring rs{ID} 0 (string-length rs{ID})))",
            accepts=frozenset({"str"})),
])

axis("chars", "character literals and conversions", producers=[
    Producer("char_lit", "#\\a", '(string-ref "a" 0)', "char"),
    Producer("upcase", "(char-upcase #\\b)", "#\\B", "char"),
    Producer("int_to_char", "(integer->char 65)", "#\\A", "char"),
], contexts=[
    Context("through_string", "(string-ref (string {X}) 0)",
            accepts=frozenset({"char"})),
])

axis("vectors", "heterogeneous vectors (16-byte tagged elements)", producers=[
    Producer("vec_ref", "(vector-ref (vector 'va 'vb) 1)",
             '(string->symbol "vb")', "sym"),
    Producer("make_set_ref",
             "(let ((mv (make-vector 3 0))) (vector-set! mv 1 9)"
             " (vector-ref mv 1))", "9", "int"),
    Producer("vec_len", "(vector-length (vector 1 2 3))", "3", "int"),
], contexts=[
    Context("singleton_ref", "(vector-ref (vector {X}) 0)"),
    Context("set_then_ref",
            "(let ((vv{ID} (make-vector 1 #f))) (vector-set! vv{ID} 0 {X})"
            " (vector-ref vv{ID} 0))"),
])

axis("hash_tables", "hash-table set/ref", producers=[
    Producer("set_ref",
             "(let ((hh (make-hash-table))) (hash-table-set! hh 'hk 42)"
             " (hash-table-ref hh 'hk))", "42", "int"),
    Producer("str_key",
             '(let ((hs (make-hash-table))) (hash-table-set! hs "k"'
             ' (list 1 2)) (hash-table-ref hs "k"))', "(list 1 2)", "list"),
], contexts=[
    Context("store_lookup",
            "(let ((ht{ID} (make-hash-table))) (hash-table-set! ht{ID} 'ck {X})"
            " (hash-table-ref ht{ID} 'ck))"),
])

axis("tco", "deep tail recursion and loops", producers=[
    Producer("deep_rec", "(dr{ID} 200000)", '(string->symbol "deep-done")',
             "sym",
             top=["(define (dr{ID} n) (if (= n 0) 'deep-done"
                  " (dr{ID} (- n 1))))"]),
    Producer("named_loop_count",
             "(let tls ((i 0) (s 0)) (if (= i 100000) s (tls (+ i 1) (+ s 1))))",
             "100000", "int"),
], contexts=[
    Context("loop_tail_hole",
            "(let tlh{ID} ((i 0)) (if (= i 50) {X} (tlh{ID} (+ i 1))))"),
    Context("rec_fn_base", "(rb{ID} 1000)",
            top=["(define (rb{ID} n) (if (= n 0) {X} (rb{ID} (- n 1))))"]),
])

axis("call_cc", "call/cc escaping and non-escaping", producers=[
    Producer("no_escape", "(call/cc (lambda (k1) (+ 40 2)))", "42", "int"),
    Producer("escape_val", "(+ 1 (call/cc (lambda (k2) (k2 10) 99)))", "11",
             "int"),
], contexts=[
    Context("cc_body", "(call/cc (lambda (ck{ID}) {X}))"),
    Context("cc_escape", "(call/cc (lambda (ce{ID}) (ce{ID} {X}) 'unreached))"),
])

axis("dynamic_wind", "dynamic-wind before/thunk/after", producers=[
    Producer("order",
             "(let ((dwl '())) (dynamic-wind"
             " (lambda () (set! dwl (cons 1 dwl)))"
             " (lambda () (set! dwl (cons 2 dwl)))"
             " (lambda () (set! dwl (cons 3 dwl)))) dwl)",
             "(list 3 2 1)", "list"),
    Producer("value",
             "(dynamic-wind (lambda () 'b) (lambda () 42) (lambda () 'a))",
             "42", "int"),
], contexts=[
    Context("dw_body",
            "(dynamic-wind (lambda () 'dwb) (lambda () {X}) (lambda () 'dwa))"),
])

axis("guard_raise", "guard / raise / error propagation", producers=[
    Producer("catch_sym", "(guard (ge (#t 'caught)) (raise 'boom))",
             '(string->symbol "caught")', "sym"),
    Producer("pass_value", "(guard (gv (#t gv)) (raise 42))", "42", "int"),
    Producer("cond_dispatch",
             "(guard (gw ((symbol? gw) 'was-sym) (#t 'other)) (raise 'zap))",
             '(string->symbol "was-sym")', "sym"),
], contexts=[
    Context("guard_pass", "(guard (gp{ID} (#t 'err)) {X})"),
])

axis("internal_define", "defines inside function bodies (letrec* semantics)",
     producers=[
    Producer("consecutive", "(idp{ID})", "7", "int",
             top=["(define (idp{ID}) (define ia 3) (define ib 4) (+ ia ib))"]),
], contexts=[
    Context("def_then_use", "(idc{ID})",
            top=["(define (idc{ID}) (define it {X}) it)"]),
    Context("def_chain", "(idd{ID})",
            top=["(define (idd{ID}) (define iu {X}) (define iv iu) iv)"]),
])

axis("macros", "define-syntax / syntax-rules", producers=[
    Producer("my_if", "(mi{ID} #t 1 2)", "1", "int",
             top=["(define-syntax mi{ID} (syntax-rules ()"
                  " ((_ c a b) (if c a b))))"]),
    Producer("swap", "(sw{ID} 1 2)", "(list 2 1)", "list",
             top=["(define-syntax sw{ID} (syntax-rules ()"
                  " ((_ a b) (list b a))))"]),
], contexts=[
    Context("identity_macro", "(im{ID} {X})",
            top=["(define-syntax im{ID} (syntax-rules () ((_ e) e)))"]),
    Context("second_macro", "(sm{ID} 'skip {X})",
            top=["(define-syntax sm{ID} (syntax-rules () ((_ a b) b)))"]),
])

axis("delay_force", "promises: delay / force / memoization", producers=[
    Producer("simple", "(force (delay (+ 40 2)))", "42", "int"),
    Producer("memoized", "(let ((mp{ID} (delay (cm{ID}))))"
             " (+ (force mp{ID}) (force mp{ID})))", "2", "int",
             top=["(define cmn{ID} 0)",
                  "(define (cm{ID}) (set! cmn{ID} (+ cmn{ID} 1)) cmn{ID})"],
             note="forcing twice must run the thunk once (memoization)"),
], contexts=[
    Context("force_delay", "(force (delay {X}))"),
    Context("force_twice",
            "(let ((fp{ID} (delay {X}))) (force fp{ID}) (force fp{ID}))",
            note="memoization: hole must be evaluated exactly once"),
])

axis("let_values", "multiple values: values / let-values", producers=[
    Producer("two_values", "(let-values (((va vb) (values 1 2))) (+ va vb))",
             "3", "int"),
    Producer("cwv", "(call-with-values (lambda () (values 20 22))"
             " (lambda (cva cvb) (+ cva cvb)))", "42", "int"),
], contexts=[
    Context("single", "(let-values (((lr{ID}) (values {X}))) lr{ID})"),
    Context("with_extra",
            "(let-values (((ls{ID} ld{ID}) (values {X} 0))) ls{ID})"),
])

axis("match", "pattern matching", producers=[
    Producer("list_pat",
             "(match (list 1 2) ((list ma mb) (+ ma mb)) (_ 'no))", "3",
             "int"),
    Producer("literal_pat", "(match 5 (5 'five) (_ 'no))",
             '(string->symbol "five")', "sym"),
], contexts=[
    Context("result_hole", "(match 1 (1 {X}) (_ 'no))"),
    Context("binder_hole",
            "(match (list {X}) ((list mr{ID}) mr{ID}) (_ 'no))"),
])

axis("hof", "higher-order stdlib: fold / reduce / for-each", producers=[
    Producer("fold_left", "(fold-left + 0 (list 1 2 3))", "6", "int"),
    Producer("reduce", "(reduce + 0 (list 1 2 3))", "6", "int"),
    Producer("for_each_acc",
             "(let ((fs 0)) (for-each (lambda (fx) (set! fs (+ fs fx)))"
             " (list 1 2 3)) fs)", "6", "int"),
], contexts=[
    Context("fold_last", "(fold-left (lambda (facc fx) fx) #f (list {X}))"),
    Context("for_each_capture",
            "(let ((fc{ID} #f)) (for-each (lambda (fe) (set! fc{ID} fe))"
            " (list {X})) fc{ID})"),
])

axis("tensors", "homogeneous f64 tensors incl. #(...) literals", producers=[
    Producer("lit_ref", "(vector-ref #(9.5 8.5) 1)", "8.5", "double"),
    Producer("tensor_op_ref", "(tensor-ref (tensor 1.0 2.5 3.0) 1)", "2.5",
             "double"),
    Producer("lit_int_ref", "(vector-ref #(9 8 7) 0)", "9", "int"),
], contexts=[
    Context("tensor_elem", "(tensor-ref (tensor {X} 0.0) 0)",
            accepts=frozenset({"double"})),
])

axis("ad_basic", "autodiff: derivative / gradient", producers=[
    Producer("deriv_square",
             "(derivative (lambda (adx) (* adx adx)) 3.0)", "6.0", "double"),
    Producer("grad_ref",
             "(vector-ref (gradient (lambda (agv) (* (vector-ref agv 0)"
             " (vector-ref agv 1))) (vector 2.0 3.0)) 0)", "3.0", "double"),
])

axis("regions", "with-region arena scoping", producers=[
    Producer("region_value", "(with-region 'rgp (+ 40 2))", "42", "int"),
], contexts=[
    Context("region_body", "(with-region 'rc{ID} {X})", accepts=SCALAR,
            note="restricted to scalars: heap values may not outlive region"),
])

axis("keyword_args", "#:keyword formals and call sites", producers=[
    Producer("kw_call", "(kwp{ID} 1 #:k 41)", "42", "int",
             top=["(define (kwp{ID} a #:k k) (+ a k))"]),
    Producer("kw_order", "(kwo{ID} #:y 2 #:x 44)", "42", "int",
             top=["(define (kwo{ID} #:x x #:y y) (- x y))"]),
], contexts=[
    Context("kw_pass", "(kwc{ID} #:v {X})",
            top=["(define (kwc{ID} #:v v) v)"]),
])

axis("cond_case", "cond / case / when dispatch", producers=[
    Producer("cond_else", "(cond ((= 1 2) 'a) (else 'b))",
             '(string->symbol "b")', "sym"),
    Producer("case_hit", "(case 2 ((1) 'one) ((2) 'two) (else 'other))",
             '(string->symbol "two")', "sym"),
], contexts=[
    Context("cond_hole", "(cond ((= 1 1) {X}) (else 'no))"),
    Context("case_hole", "(case 1 ((1) {X}) (else 'no))"),
    Context("when_hole", "(when #t {X})"),
])

axis("and_or", "and / or short-circuit values", producers=[
    Producer("and_last", "(and 1 2 3)", "3", "int"),
    Producer("or_first", "(or 42 99)", "42", "int"),
    Producer("or_quote_null", "(or #f '())", "(list)", "list",
             note="bug #229 family: quote in or-argument position"),
    Producer("and_empty", "(and)", "#t", "bool"),
], contexts=[
    Context("and_pass", "(and #t {X})"),
    Context("or_pass", "(or #f {X})"),
    Context("if_or_else", "(if (or #f #f) 'no {X})"),
])

axis("begin_seq", "begin sequencing and ordering", producers=[
    Producer("order",
             "(let ((bl '())) (begin (set! bl (cons 1 bl))"
             " (set! bl (cons 2 bl))) bl)", "(list 2 1)", "list"),
], contexts=[
    Context("begin_tail", "(begin 'bfirst {X})"),
    Context("begin_only", "(begin {X})"),
])

axis("modules_require", "stdlib module functions via (require stdlib)",
     producers=[
    Producer("filter_odd", "(filter odd? (list 1 2 3 4))", "(list 1 3)",
             "list", requires=("stdlib",)),
    Producer("assq_hit", "(assq 'b '((a 1) (b 2)))", "(list 'b 2)", "list"),
], contexts=[
    Context("filter_wrap", "(car (filter (lambda (fw) #t) (list {X})))",
            requires=("stdlib",)),
])

axis("toplevel", "top-level definition and evaluation semantics", producers=[
    Producer("top_def_read", "tlp{ID}", "42", "int",
             top=["(define tlp{ID} 42)"]),
], contexts=[
    Context("top_define", "tlv{ID}", top=["(define tlv{ID} {X})"]),
    Context("top_list_arg", "(car tll{ID})",
            top=["(define tll{ID} (list {X}))"],
            note="known-fragile: top-level constructor args re-evaluated"),
    Context("top_vector_arg", "(vector-ref tlw{ID} 0)",
            top=["(define tlw{ID} (vector {X}))"]),
])

axis("eqv_pred", "eq? / eqv? / equal? semantics", producers=[
    Producer("equal_lists", "(equal? (list 1 2) '(1 2))", "#t", "bool"),
    Producer("eqv_sym", "(eqv? 'a 'a)", "#t", "bool"),
    Producer("eq_null", "(eq? '() '())", "#t", "bool"),
])

axis("parallel_map", "parallel-map over lists", producers=[
    Producer("squares", "(parallel-map (lambda (pm) (* pm pm)) (list 1 2 3))",
             "(list 1 4 9)", "list"),
], contexts=[
    Context("pmap_hole", "(car (parallel-map (lambda (pi) {X}) (list 0)))"),
])

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

CHK_DEF = """\
(define (edge-chk nm actual expected)
  (if (equal? actual expected)
      (begin (display "PASS: ") (display nm) (newline))
      (begin (display "FAIL: ") (display nm)
             (display " expected=") (write expected)
             (display " actual=") (write actual)
             (newline))))
"""


def pair_score(a, b):
    ah, bh = a in HIGH_RISK, b in HIGH_RISK
    if ah and bh:
        return 0
    if bh:
        return 1
    if ah:
        return 2
    return 3


def ordered_pairs():
    names = list(AXES.keys())
    pairs = []
    for a in names:
        if not AXES[a].producers:
            continue
        for b in names:
            if a == b or not AXES[b].contexts:
                continue
            pairs.append((a, b))
    pairs.sort(key=lambda ab: (pair_score(*ab), ab[0], ab[1]))
    return pairs


def compatible(p, c):
    return c.accepts is None or p.vtype in c.accepts


def emit_pair(a, b, idx, uid_counter, outdir):
    """Emit one probe file for ordered pair (a, b). Returns
    (filename, n_checks, n_skipped, uid_counter) or None if no checks."""
    axis_a, axis_b = AXES[a], AXES[b]
    checks = []
    requires = set()
    skipped = 0
    for p in axis_a.producers:
        for c in axis_b.contexts:
            if not compatible(p, c):
                skipped += 1
                continue
            uid = "g%d" % uid_counter
            uid_counter += 1
            name = "%s_%s__in__%s_%s" % (a, p.name, b, c.name)
            top = [t.replace("{ID}", uid) for t in p.top]
            top += [t.replace("{ID}", uid).replace("{X}",
                                                   p.expr.replace("{ID}", uid))
                    for t in c.top]
            expr = c.template.replace("{ID}", uid).replace(
                "{X}", p.expr.replace("{ID}", uid))
            expected = p.expected.replace("{ID}", uid)
            requires.update(p.requires)
            requires.update(c.requires)
            checks.append((name, uid, top, expr, expected))
    if not checks:
        return None, 0, skipped, uid_counter

    fname = "pair%03d_%s__%s.esk" % (idx, a, b)
    lines = []
    lines.append(";; Generated by tests/edge_matrix/gen_matrix.py -- DO NOT EDIT")
    lines.append(";; pair: %s (producers) composed into %s (contexts)" % (a, b))
    lines.append(";; CHECKS: %d" % len(checks))
    for r in sorted(requires):
        lines.append("(require %s)" % r)
    lines.append(CHK_DEF)
    for name, uid, top, expr, expected in checks:
        lines.append(";; check %s" % name)
        lines.extend(top)
        lines.append('(define (edge-check-%s) (edge-chk "%s" %s %s))'
                     % (uid, name, expr, expected))
        lines.append("(edge-check-%s)" % uid)
        lines.append("")
    lines.append('(display "EDGE-MATRIX-DONE") (newline)')
    with open(os.path.join(outdir, fname), "w") as f:
        f.write("\n".join(lines) + "\n")
    return fname, len(checks), skipped, uid_counter


def emit_features(path):
    out = []
    out.append("# Edge-matrix feature axes")
    out.append("")
    out.append("Auto-generated by `gen_matrix.py --emit-features` -- edit the")
    out.append("axis definitions in gen_matrix.py, then regenerate this file.")
    out.append("")
    out.append("Each axis has PRODUCER forms (expression with generator-known"
               " value)")
    out.append("and CONTEXT forms (identity wrappers around a hole `{X}`).")
    out.append("For an ordered pair (A, B), every producer of A is placed in")
    out.append("every type-compatible context of B. High-risk axes (composed")
    out.append("first): %s." % ", ".join(HIGH_RISK))
    out.append("")
    for name, ax in AXES.items():
        risk = " **[high-risk]**" if name in HIGH_RISK else ""
        out.append("## %s%s" % (name, risk))
        out.append("")
        out.append(ax.doc)
        out.append("")
        if ax.producers:
            out.append("Producers:")
            out.append("")
            for p in ax.producers:
                note = " -- %s" % p.note if p.note else ""
                eff = " (effectful)" if p.effectful else ""
                out.append("- `%s`: `%s` = `%s` [%s]%s%s"
                           % (p.name, p.expr, p.expected, p.vtype, eff, note))
            out.append("")
        if ax.contexts:
            out.append("Contexts:")
            out.append("")
            for c in ax.contexts:
                acc = ("any" if c.accepts is None
                       else ",".join(sorted(c.accepts)))
                note = " -- %s" % c.note if c.note else ""
                out.append("- `%s`: `%s` (accepts: %s)%s"
                           % (c.name, c.template, acc, note))
            out.append("")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-pairs", type=int, default=150,
                    help="cap on pairs (priority order); 0 = all pairs")
    ap.add_argument("--outdir", default=os.path.join(here, "generated"))
    ap.add_argument("--list-axes", action="store_true")
    ap.add_argument("--emit-features", action="store_true",
                    help="write FEATURES.md and exit")
    args = ap.parse_args()

    if args.list_axes:
        for name, ax in AXES.items():
            print("%-18s P=%d C=%d %s%s" % (
                name, len(ax.producers), len(ax.contexts),
                "[high-risk] " if name in HIGH_RISK else "", ax.doc))
        return 0

    if args.emit_features:
        path = os.path.join(here, "FEATURES.md")
        emit_features(path)
        print("wrote %s" % path)
        return 0

    pairs = ordered_pairs()
    total_pairs_available = len(pairs)
    if args.max_pairs > 0:
        pairs = pairs[:args.max_pairs]

    os.makedirs(args.outdir, exist_ok=True)
    for old in os.listdir(args.outdir):
        if old.startswith("pair") and old.endswith(".esk"):
            os.remove(os.path.join(args.outdir, old))

    manifest = []
    uid = 0
    n_checks = n_skipped = n_files = 0
    for idx, (a, b) in enumerate(pairs):
        fname, nc, ns, uid = emit_pair(a, b, idx, uid, args.outdir)
        n_skipped += ns
        if fname is None:
            continue
        n_files += 1
        n_checks += nc
        manifest.append("%s\t%s\t%s\t%d" % (fname, a, b, nc))

    with open(os.path.join(args.outdir, "MANIFEST.tsv"), "w") as f:
        f.write("file\tproducer_axis\tcontext_axis\tchecks\n")
        f.write("\n".join(manifest) + "\n")

    print("axes: %d  pairs available: %d  pairs emitted: %d  files: %d"
          % (len(AXES), total_pairs_available, len(pairs), n_files))
    print("checks: %d  skipped (type-incompatible): %d" % (n_checks, n_skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
