#!/usr/bin/env python3
"""gen_language_surface.py — extract the COMPLETE Eshkol language surface.

Ground truth for TOTAL-LANGUAGE exposure-engine coverage. Parses the compiler
and VM source directly (no hand-maintained list to drift) and emits a single
machine-readable manifest describing every construct a program can invoke:

  * BUILTINS      — the native name->(id,arity) dispatch tables registered by
                    the LLVM/native backend (lib/backend/eshkol_compiler.c) and
                    the bytecode VM (lib/backend/eshkol_vm.c). Each entry records
                    which backend(s) register it.
  * SPECIAL FORMS — the surface keywords the parser recognises as syntax rather
                    than a call (lib/frontend/parser.cpp get_operator_type plus
                    the directly-dispatched forms begin/define-library/delay/...),
                    keyed to their eshkol_op_t (inc/eshkol/eshkol.h).
  * PRELUDE       — higher-order library functions defined in the Scheme prelude
                    compiled into every program (map/filter/fold-*/for-each/...).

Every entry is categorised by a risk/kind bucket that drives exposure-engine
prioritisation (numeric, list_pair, vector, string_char, control_flow,
higher_order, io_port, tensor_ad, macro_syntax, ffi, consciousness, vm_system,
predicate, hash, memory_region, misc).

Usage:
  python3 scripts/gen_language_surface.py [--out tests/coverage/language_surface.json]
"""

import argparse
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COMPILER_C = os.path.join(REPO, "lib", "backend", "eshkol_compiler.c")
VM_C = os.path.join(REPO, "lib", "backend", "eshkol_vm.c")
LLVM_CPP = os.path.join(REPO, "lib", "backend", "llvm_codegen.cpp")
PARSER = os.path.join(REPO, "lib", "frontend", "parser.cpp")
ESHKOL_H = os.path.join(REPO, "inc", "eshkol", "eshkol.h")

TRIPLE = re.compile(r'\{\s*"([^"]+)"\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\}')


def _slice_table(path, start_marker, end_marker_after):
    """Return the source text of the BuiltinDef table body."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    i = text.index(start_marker)
    j = text.index(end_marker_after, i)
    return text[i:j]


def extract_builtin_table(path):
    """Extract {name: {id, arity}} from a `static const BuiltinDef BUILTINS[]`."""
    body = _slice_table(path, "BuiltinDef BUILTINS[]", "\n};")
    out = {}
    for m in TRIPLE.finditer(body):
        name, nid, arity = m.group(1), int(m.group(2)), int(m.group(3))
        if name == "NULL":
            continue
        # A name may appear with several ids (overloads / legacy+new). Keep all.
        rec = out.setdefault(name, {"ids": [], "arity": arity})
        if nid not in rec["ids"]:
            rec["ids"].append(nid)
    return out


def extract_llvm_dispatch():
    """Extract call-head builtin names dispatched by the LLVM AOT backend
    (`if (func_name == "name") ...`). This is the AOT intrinsic surface: it
    overlaps the native/VM id-tables and adds AOT-only intrinsics (the R7RS
    IO/mutation forms, the NN/optimizer/linalg surface, atomics, etc.)."""
    with open(LLVM_CPP, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    names = set(re.findall(r'func_name\s*==\s*"([^"]+)"', text))
    # a few dispatch strings are internal markers, not user-callable builtins
    NOISE = {"target-intrinsic", "sum-tag", "sum-value", "features",
             "__arena-used"}
    return {n for n in names if n not in NOISE}


def extract_ast_ops():
    """Extract the eshkol_op_t enum member names."""
    with open(ESHKOL_H, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    m = re.search(r"typedef enum\s*\{(.*?)\}\s*eshkol_op_t;", text, re.S)
    body = m.group(1)
    return [x for x in re.findall(r"\bESHKOL_[A-Z0-9_]+_OP\b", body)]


def extract_special_forms():
    """Extract surface-keyword -> eshkol_op_t from the parser dispatch, plus the
    directly-dispatched syntactic forms not routed through get_operator_type."""
    with open(PARSER, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    forms = {}
    # get_operator_type body: `if (op == "kw") return ESHKOL_X_OP;`
    gm = re.search(r"get_operator_type\(const std::string& op\)\s*\{(.*?)\n\}",
                   text, re.S)
    for kw, op in re.findall(r'op == "([^"]+)"\)\s*return\s+(ESHKOL_[A-Z0-9_]+_OP)',
                             gm.group(1)):
        forms[kw] = op
    # Directly-dispatched syntactic forms (handled before/around get_operator_type).
    extra = {
        "begin": "ESHKOL_SEQUENCE_OP",
        "define-library": "ESHKOL_DEFINE_TYPE_OP",   # library form (parser-transformed)
        "delay": "ESHKOL_CALL_OP",                    # promise form (codegen intrinsic)
        "delay-force": "ESHKOL_CALL_OP",
        "make-promise": "ESHKOL_CALL_OP",
        "force": "ESHKOL_CALL_OP",
        "let-match": "ESHKOL_MATCH_OP",
        "define-values": "ESHKOL_DEFINE_OP",
        "named-let": "ESHKOL_LET_OP",                 # (let name (bindings) body)
    }
    for k, v in extra.items():
        forms.setdefault(k, v)
    return forms


# --- prelude higher-order library functions (compiled into every program) ----
def extract_prelude_defs():
    names = set()
    for path in (COMPILER_C,):
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        # only the scheme_prelude string literal region
        m = re.search(r"scheme_prelude\s*=(.*?);\n", text, re.S)
        region = m.group(1) if m else text
        for name in re.findall(r'\(define \(([a-zA-Z][a-zA-Z0-9!?*/<>=._+-]*)', region):
            names.add(name)
    # canonical R7RS higher-order names that must be reachable
    names.update({"map", "filter", "fold-left", "fold-right", "for-each",
                  "reduce", "compose", "any", "every", "find", "take", "drop"})
    return sorted(names)


# ---------------------------------------------------------------------------
# categoriser — risk/kind bucket per construct (drives prioritisation)
# ---------------------------------------------------------------------------
GEOMETRY_PREFIXES = ("manifold", "riemannian", "geodesic", "spherical",
                     "hyperbolic", "mobius", "se3-", "so3-", "poincare",
                     "curvature", "christoffel", "ricci", "riemann",
                     "quaternion")
GEOMETRY_NAMES = {"hodge-star", "wedge-product", "frechet-mean",
                  "great-circle-distance", "exterior-derivative",
                  "interior-product", "metric-tensor", "pullback",
                  "pullback", "sectional-curvature", "transition-geometry!",
                  "parallel-transport", "geodesic-distance", "slerp",
                  "adaptive-curvature-step", "get-curvature", "set-curvature!",
                  "retraction", "make-euclidean-manifold",
                  "make-product-manifold", "make-riemannian-adam-state"}


STRAGGLERS = {
    "digit-value": "string_char", "matrix-to-string": "string_char",
    "split": "string_char", "procedure-arity": "misc_core",
    "inject-left": "misc_core", "inject-right": "misc_core",
    "%make-lazy-promise": "control_flow",
    "%make-lazy-promise-force": "control_flow",
}


def categorize(name):
    if name in STRAGGLERS:
        return STRAGGLERS[name]
    n = name.lstrip("_")
    # AD-tape / autodiff primitives
    if n.startswith("ad-") or n in {"reverse-gradient"}:
        return "tensor_ad"
    # differential geometry (silent-wrong-prone numeric — own bucket)
    if (n.startswith(GEOMETRY_PREFIXES) or n in GEOMETRY_NAMES
            or n.endswith("-manifold")):
        return "geometry"
    if n in {"batch-matmul", "matmul"}:
        return "tensor_ad"
    if n in {"vref"}:
        return "vector"
    if n in {"current-error-port"}:
        return "io_port"
    if n == "make-hash-table" or n.startswith("hash"):
        return "hash"
    if n.startswith("unix-socket"):
        return "ffi_system"
    # gpu compute
    if n.startswith("gpu-"):
        return "tensor_ad"
    # parallel higher-order
    if n.startswith("parallel-"):
        return "higher_order"
    # neural-net / optimizer / tensor-ml surface
    if (n.endswith("-loss") or n.endswith("-step") or n.endswith("-lr")
            or n.endswith("-pool2d") or n.startswith("dataloader")
            or n.startswith("conv") or n.startswith("dequant")
            or n in {"gelu", "elu", "celu", "selu", "silu", "mish", "softplus",
                     "leaky-relu", "hard-sigmoid", "hard-swish", "relu", "sigmoid",
                     "embedding", "feed-forward", "multi-head-attention",
                     "positional-encoding", "rotary-embedding", "causal-mask",
                     "padding-mask", "dropout", "check-grad-health",
                     "clip-grad-norm!", "zero-grad!", "make-dataloader",
                     "train-test-split", "onnx-export-tensor", "concatenate",
                     "stack", "tile", "squeeze", "unsqueeze", "slice", "pad",
                     "outer", "einsum", "norm", "batch-matmul", "avg-pool2d",
                     "max-pool2d", "fft", "ifft", "det"}
            or n.startswith("xavier") or n.startswith("kaiming")
            or n.startswith("lecun")):
        return "tensor_ad"
    # atomics / low-level memory / FFI pointers
    if (n.startswith("atomic-") or n.startswith("volatile-")
            or n.startswith("ptr") or n.endswith("->ptr")
            or n in {"addr-of", "null-ptr", "memory-fence", "compiler-fence",
                     "usize->ptr", "ptr-add"}):
        return "ffi_system"
    # rng (nondeterministic surface)
    if (n.startswith("prng") or n.startswith("make-prng")
            or n in {"rand", "randint", "randn", "random", "set-random-seed!",
                     "quantum-random", "quantum-random-int", "quantum-random-range"}):
        return "ffi_system"
    # eval / environments (meta)
    if n in {"eval", "interaction-environment", "current-environment",
             "null-environment", "scheme-report-environment"}:
        return "misc_core"
    # extended numeric tower (AOT)
    if n in {"acosh", "asinh", "atanh", "atan2", "cbrt", "ceil", "exp2", "log2",
             "log10", "mod", "floor/", "floor-quotient", "floor-remainder",
             "truncate/", "truncate-quotient", "truncate-remainder", "trunc",
             "square", "/rational", "%", "make-rational", "fabs", "abs"}:
        return "numeric"
    # macro / syntax
    if n in {"quote", "quasiquote", "unquote", "unquote-splicing", "define-syntax",
             "let-syntax", "letrec-syntax", "syntax-error", "syntax-rules"}:
        return "macro_syntax"
    # control flow / special-form control
    if n in {"if", "cond", "case", "when", "unless", "and", "or", "do", "begin",
             "guard", "raise", "call/cc", "call-with-current-continuation",
             "dynamic-wind", "values", "call-with-values", "let-values",
             "let*-values", "delay", "delay-force", "force", "make-promise",
             "match", "let-match", "when", "not", "with-exception-handler",
             "raise-continuable"}:
        return "control_flow"
    # binding forms
    if n in {"define", "lambda", "let", "let*", "letrec", "letrec*", "set!",
             "named-let", "define-values", "define-record-type", "case-lambda",
             "parameterize", "make-parameter", "define-type"}:
        return "binding_form"
    # module / library
    if n in {"import", "require", "provide", "load", "include", "include-ci",
             "define-library", "cond-expand"}:
        return "module"
    # memory / region (OALR)
    if n in {"with-region", "owned", "move", "borrow", "shared", "weak-ref"}:
        return "memory_region"
    # higher-order
    if n in {"map", "filter", "fold-left", "fold-right", "for-each", "reduce",
             "compose", "apply", "any", "every", "find", "sort", "vector-map",
             "vector-for-each", "count", "merge"}:
        return "higher_order"
    # tensor / AD
    if (n.startswith("tensor") or n.startswith("dual") or n.startswith("dnc")
            or n.startswith("sdnc") or n in {
            "make-tensor", "reshape", "transpose", "flatten", "matmul", "conv2d",
            "softmax", "relu", "sigmoid", "dropout", "zeros", "ones", "arange",
            "linspace", "eye", "matrix", "gradient", "jacobian", "hessian",
            "derivative", "derivative-n", "taylor", "divergence", "curl",
            "laplacian", "directional-derivative", "diff", "differentiate", "D",
            "make-dual", "dual?", "mse-loss", "cross-entropy-loss", "batch-norm",
            "layer-norm", "scaled-dot-attention", "tensor-matmul", "tensor-scale",
            "tensor-add", "tensor-mul", "model-save", "model-load"}):
        return "tensor_ad"
    # consciousness engine (logic/inference/workspace)
    if (n.startswith("kb") or n.startswith("fg-") or n.startswith("ws-")
            or n.startswith("make-factor") or n.startswith("make-workspace")
            or n in {"unify", "walk", "make-substitution", "substitution?",
                     "make-fact", "fact?", "make-kb", "kb?", "logic-var?",
                     "factor-graph?", "workspace?", "free-energy",
                     "expected-free-energy", "forall"}):
        return "consciousness"
    # hash tables
    if n.startswith("hash"):
        return "hash"
    # io / port
    if (n.startswith("open-") or n.startswith("read") or n.startswith("write")
            or n.startswith("close") or n.startswith("peek")
            or n.startswith("call-with-input") or n.startswith("call-with-output")
            or n.startswith("call-with-port") or n.startswith("with-input-from")
            or n.startswith("with-output-to") or n.startswith("flush-output")
            or n in {"display", "newline", "eof-object?", "eof-object",
                     "get-output-string", "file-exists?", "flush-output",
                     "current-output-port", "current-input-port", "port?",
                     "input-port?", "output-port?", "with-output-to-string",
                     "call-with-output-string", "display-error", "format"}):
        return "io_port"
    # ffi / system / network / process / regex / time / compression
    if (n.startswith("http") or n.startswith("websocket") or n.startswith("ts-")
            or n.startswith("process") or n.startswith("socket")
            or n.startswith("signal") or n.startswith("dl") or n.startswith("regex")
            or n.startswith("url-") or n.startswith("base64") or n.startswith("semver")
            or n.startswith("term-") or n.startswith("fs-") or n.startswith("db-")
            or n.startswith("yoga") or n.startswith("lru-") or n.startswith("channel")
            or n.startswith("mutex") or n.startswith("condition") or n.startswith("condvar")
            or n.startswith("timer") or n.startswith("interval") or n.startswith("fd-")
            or n.startswith("line-reader") or n.startswith("make-line")
            or n.startswith("json") or n.startswith("compression")
            or n.startswith("file-") or n.startswith("directory")
            or n.startswith("path-") or n.startswith("glob") or n.startswith("image-")
            or n.startswith("symlink") or n.startswith("shell-") or n.startswith("os-")
            or n.startswith("mkdir") or n.startswith("mkstemp") or n.startswith("mkdtemp")
            or n.startswith("thread-pool") or n.startswith("future")
            or n.startswith("quantum-random")
            or n.startswith("current-second") or n.startswith("current-jiffy")
            or n.startswith("current-time") or n.startswith("dequant")
            or n in {"exit", "emergency-exit", "system", "sleep", "time", "trace",
                     "current-seconds", "jiffies-per-second", "poll-fd",
                     "delete-directory", "make-directory", "append-file"}
            or n in {"getpid", "hostname", "username", "sleep-ms", "cpu-count",
                     "current-directory", "set-current-directory!", "home-directory",
                     "command-line", "current-time-ms", "realpath", "delete-file",
                     "force-future", "with-mutex", "make-condition-variable",
                     "make-condvar", "make-timer", "make-interval",
                     "directory-delete-recursive", "directory-entries",
                     "directory-walk", "mkdir-recursive", "utf8->string"}
            or n in {"extern", "extern-var", "fork", "execv", "setenv", "unsetenv",
                     "getenv", "get-environment-variable", "uuid-v4", "sha256-file",
                     "monotonic-time-ms", "executable-path", "temp-directory",
                     "prevent-sleep", "allow-sleep", "make-temp-file", "make-temp-dir",
                     "deflate", "inflate", "gzip", "gunzip", "at-exit", "make-pipe",
                     "make-channel", "make-mutex", "make-event-emitter", "on!", "once!",
                     "off!", "poll", "io-poll", "constant-time-equal?", "make-lru-cache",
                     "current-timestamp", "current-time-ns", "format-iso8601",
                     "parse-iso8601", "format-relative", "local-timezone-offset",
                     "diff-lines", "fuzzy-match", "ansi-strip", "string-display-width"}):
        return "ffi_system"
    # complex / rational (numeric tower non-real)
    if n in {"make-rectangular", "make-polar", "real-part", "imag-part",
             "magnitude", "angle", "conjugate", "complex?", "numerator",
             "denominator", "rationalize", "rational?"}:
        return "numeric"
    # numeric (arithmetic / math / bit)
    if (n in {"+", "-", "*", "/", "add2", "sub2", "mul2", "div2", "expt", "pow",
              "min", "max", "modulo", "remainder", "quotient", "abs", "gcd", "lcm",
              "sin", "cos", "tan", "asin", "acos", "atan", "exp", "log", "sqrt",
              "floor", "ceiling", "round", "truncate", "sinh", "cosh", "tanh",
              "exact", "inexact", "exact->inexact", "inexact->exact",
              "number->string", "string->number", "sign", "arithmetic-shift"}
            or n.startswith("bitwise") or n in {"<", ">", "<=", ">=", "="}):
        return "numeric"
    # predicates
    if n.endswith("?"):
        return "predicate"
    # string / char
    if (n.startswith("string") or n.startswith("char") or n.startswith("substring")
            or n.startswith("symbol->") or n.startswith("->string")
            or n in {"make-string", "list->string", "string->list", "list->symbol",
                     "symbol->string", "string->symbol", "char->integer",
                     "integer->char", "number->string", "string->number",
                     "list->string", "ansi-strip"}):
        return "string_char"
    # vectors / bytevectors
    if n.startswith("vector") or n.startswith("bytevector") or n in {
            "make-vector", "list->vector", "vector->list", "make-bytevector"}:
        return "vector"
    # list / pair
    if (n in {"cons", "car", "cdr", "list", "length", "append", "reverse",
              "member", "memq", "memv", "assoc", "assq", "assv", "list-ref",
              "list-tail", "last-pair", "iota", "list-copy", "map1", "take", "drop",
              "last", "list*", "remove", "remq", "remv", "acons", "split-at",
              "set-car!", "set-cdr!"}
            or re.fullmatch(r"c[ad]{2,4}r", n)):
        return "list_pair"
    # equality / misc core
    if n in {"eq?", "eqv?", "equal?", "not", "void", "error", "type-of",
             "boolean=?", "error-object?", "error-object-message",
             "error-object-irritants"}:
        return "misc_core"
    return "misc"


def build_manifest():
    comp = extract_builtin_table(COMPILER_C)
    vm = extract_builtin_table(VM_C)
    aot = extract_llvm_dispatch()
    ops = extract_ast_ops()
    forms = extract_special_forms()
    prelude = extract_prelude_defs()

    # AOT-dispatched names that are also declared special forms are syntax, not
    # builtins — keep them out of the builtin set (they appear under forms).
    form_names = set(forms)
    aot_builtins = {n for n in aot if n not in form_names}

    names = sorted(set(comp) | set(vm) | aot_builtins)
    builtins = []
    for name in names:
        in_c = name in comp
        in_v = name in vm
        in_a = name in aot_builtins
        ids = sorted(set(comp.get(name, {}).get("ids", []))
                     | set(vm.get(name, {}).get("ids", [])))
        arity = comp.get(name, vm.get(name, {})).get("arity")
        backends = []
        if in_c:
            backends.append("native")
        if in_v:
            backends.append("vm")
        if in_a:
            backends.append("native_llvm")
        builtins.append({
            "name": name,
            "ids": ids,
            "arity": arity,
            "backends": backends,
            "category": categorize(name),
        })

    special = []
    for kw in sorted(forms):
        special.append({
            "name": kw,
            "op": forms[kw],
            "category": categorize(kw),
        })

    prelude_entries = []
    builtin_names = set(names)
    for name in prelude:
        prelude_entries.append({
            "name": name,
            "category": categorize(name),
            "also_builtin": name in builtin_names,
        })

    manifest = {
        "_meta": {
            "description": "Complete Eshkol language surface (ground truth for "
                           "exposure-engine coverage).",
            "sources": {
                "native_builtins": "lib/backend/eshkol_compiler.c BUILTINS[]",
                "vm_builtins": "lib/backend/eshkol_vm.c BUILTINS[]",
                "ast_ops": "inc/eshkol/eshkol.h eshkol_op_t",
                "special_forms": "lib/frontend/parser.cpp get_operator_type + "
                                 "direct dispatch",
                "prelude": "lib/backend/eshkol_compiler.c scheme_prelude",
            },
        },
        "counts": {
            "builtins_total": len(builtins),
            "builtins_in_native_closure_table": sum(1 for b in builtins
                                                    if "native" in b["backends"]),
            "builtins_in_vm_table": sum(1 for b in builtins
                                        if "vm" in b["backends"]),
            "builtins_in_llvm_aot_dispatch": sum(1 for b in builtins
                                                 if "native_llvm" in b["backends"]),
            "builtins_aot_only": sum(1 for b in builtins
                                     if b["backends"] == ["native_llvm"]),
            "special_forms": len(special),
            "ast_ops": len(ops),
            "prelude": len(prelude_entries),
        },
        "builtins_by_category": _count_by_cat(builtins),
        "special_forms_by_category": _count_by_cat(special),
        "ast_ops": ops,
        "builtins": builtins,
        "special_forms": special,
        "prelude": prelude_entries,
    }
    return manifest


def _count_by_cat(entries):
    out = {}
    for e in entries:
        out[e["category"]] = out.get(e["category"], 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO, "tests", "coverage",
                                                  "language_surface.json"))
    args = ap.parse_args()
    manifest = build_manifest()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
        fh.write("\n")
    c = manifest["counts"]
    print("Wrote %s" % args.out)
    print("  builtins: %d (native-closure %d, vm %d, llvm-aot %d, aot-only %d)"
          % (c["builtins_total"], c["builtins_in_native_closure_table"],
             c["builtins_in_vm_table"], c["builtins_in_llvm_aot_dispatch"],
             c["builtins_aot_only"]))
    print("  special forms: %d   ast ops: %d   prelude: %d"
          % (c["special_forms"], c["ast_ops"], c["prelude"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
