/**
 * @file eshkol_module.cpp
 * @brief Python bindings for the Eshkol programming language.
 *
 * Provides:
 *   import eshkol
 *   ctx = eshkol.Context()
 *   result = ctx.eval("(+ 1 2)")          # => 3
 *   result = ctx.eval("(derivative sin 0.5)")  # => cos(0.5)
 *   tensor = ctx.eval("#(1 2 3)")          # => numpy array
 *
 * Requires: pybind11, linked against libeshkol-static.a + LLVM.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <eshkol/eshkol_ffi.h>

#include <string>
#include <stdexcept>

namespace py = pybind11;

/**
 * Convert an Eshkol FFI value to a Python object.
 * Handles: null, int, double, bool, string, pair/list, tensor.
 */
static py::object ffi_value_to_python(eshkol_ffi_context_t* ctx, eshkol_ffi_value_t val) {
    switch (eshkol_ffi_type(val)) {
        case ESHKOL_FFI_TYPE_NULL:
            return py::none();

        case ESHKOL_FFI_TYPE_INT64:
            return py::int_(eshkol_ffi_to_int64(val));

        case ESHKOL_FFI_TYPE_DOUBLE:
            return py::float_(eshkol_ffi_to_double(val));

        case ESHKOL_FFI_TYPE_BOOL:
            return py::bool_(eshkol_ffi_to_bool(val));

        case ESHKOL_FFI_TYPE_HEAP_PTR: {
            /* Check subtype: string, pair, tensor */
            const char* str = eshkol_ffi_to_string(val);
            if (str) {
                return py::str(str);
            }

            if (eshkol_ffi_is_pair(val)) {
                /* Convert list to Python list */
                py::list result;
                eshkol_ffi_value_t current = val;
                while (eshkol_ffi_is_pair(current)) {
                    result.append(ffi_value_to_python(ctx, eshkol_ffi_car(current)));
                    current = eshkol_ffi_cdr(current);
                }
                /* If cdr is not null, it's an improper list — append as tuple */
                if (!eshkol_ffi_is_null(current)) {
                    result.append(ffi_value_to_python(ctx, current));
                }
                return result;
            }

            /* Check for tensor — zero-copy via buffer protocol */
            double* tdata = eshkol_ffi_tensor_data(val);
            if (tdata) {
                int64_t size = eshkol_ffi_tensor_size(val);
                if (size > 0) {
                    /* Zero-copy: create numpy array that views the arena memory.
                     * The capsule prevents Python from freeing the data (arena owns it). */
                    auto capsule = py::capsule(tdata, [](void*) { /* arena-managed, no free */ });
                    return py::array_t<double>(
                        {(py::ssize_t)size},            /* shape */
                        {(py::ssize_t)sizeof(double)},  /* strides */
                        tdata,                           /* data pointer */
                        capsule                          /* prevent dealloc */
                    );
                }
            }

            /* Unknown heap type */
            return py::str("<eshkol object>");
        }

        default:
            return py::str("<eshkol value>");
    }
}

/**
 * Eshkol evaluation context — wraps eshkol_ffi_context_t.
 */
class EshkolContext {
public:
    EshkolContext() {
        ctx_ = eshkol_ffi_init();
        if (!ctx_) {
            throw std::runtime_error("Failed to initialize Eshkol runtime");
        }
    }

    ~EshkolContext() {
        if (ctx_) {
            eshkol_ffi_shutdown(ctx_);
            ctx_ = nullptr;
        }
    }

    /* Evaluate Eshkol source code, return Python value */
    py::object eval(const std::string& source) {
        eshkol_ffi_value_t result;
        int rc = eshkol_ffi_eval(ctx_, source.c_str(), &result);
        if (rc != 0) {
            const char* err = eshkol_ffi_last_error();
            throw std::runtime_error(err ? err : "Evaluation failed");
        }
        return ffi_value_to_python(ctx_, result);
    }

    /* Evaluate and return as double (convenience for numeric work) */
    double eval_double(const std::string& source) {
        double result;
        int rc = eshkol_ffi_eval_double(ctx_, source.c_str(), &result);
        if (rc != 0) {
            const char* err = eshkol_ffi_last_error();
            throw std::runtime_error(err ? err : "Evaluation failed");
        }
        return result;
    }

    /* Evaluate a file */
    void eval_file(const std::string& path) {
        int rc = eshkol_ffi_eval_file(ctx_, path.c_str());
        if (rc != 0) {
            const char* err = eshkol_ffi_last_error();
            throw std::runtime_error(err ? err : "File evaluation failed");
        }
    }

    /* Compute derivative: derivative(f_source, x)
     *
     * Security (#191): func_source is user-controlled Python input.
     * Before v1.2 we snprintf'd it straight into "(derivative %s %g)",
     * meaning a caller could pass "(begin (system \"rm -rf /\") x)"
     * and run arbitrary Eshkol code at FFI invocation. Now we require
     * func_source to be exactly one lambda expression — balanced
     * parens, starts with "(lambda" or "(λ", and contains no unescaped
     * double-quote / string-literal injection points. Anything else
     * raises a ValueError on the Python side rather than reaching the
     * interpreter. */
    double derivative(const std::string& func_source, double x) {
        validate_lambda_source(func_source, "derivative");
        char buf[512];
        snprintf(buf, sizeof(buf), "(derivative %s %g)", func_source.c_str(), x);
        return eval_double(buf);
    }

    /* Compute gradient: gradient(f_source, point_list) */
    py::object gradient(const std::string& func_source, const std::vector<double>& point) {
        validate_lambda_source(func_source, "gradient");
        std::string expr = "(gradient " + func_source + " (list";
        for (double v : point) {
            char num[64];
            snprintf(num, sizeof(num), " %g", v);
            expr += num;
        }
        expr += "))";
        return eval(expr);
    }

    /* Validate a user-supplied Scheme lambda source before embedding it
     * into a larger expression. Must:
     *   1. Be non-empty and start with '(' (a single S-expression form).
     *   2. Start with "(lambda" or "(λ" after skipping leading whitespace.
     *   3. Have balanced parentheses — no suffix garbage.
     *   4. Contain no double-quote (prevents string-literal escape and
     *      subsequent eval-arbitrary-text injection via stringified
     *      primitives).
     * All three fail with py::value_error so the Python caller sees a
     * clear ValueError instead of Eshkol swallowing the payload. */
    static void validate_lambda_source(const std::string& src, const char* caller) {
        std::string s = src;
        /* Strip leading/trailing whitespace. */
        size_t start = s.find_first_not_of(" \t\n\r");
        if (start == std::string::npos) {
            throw py::value_error(std::string(caller) + ": empty func_source");
        }
        size_t end = s.find_last_not_of(" \t\n\r");
        s = s.substr(start, end - start + 1);

        if (s.empty() || s[0] != '(') {
            throw py::value_error(std::string(caller) + ": func_source must be a lambda expression");
        }

        /* Quick head check — skip "(" + optional whitespace. */
        size_t i = 1;
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) i++;
        bool is_lambda = (s.compare(i, 6, "lambda") == 0);
        bool is_lambda_unicode = (s.compare(i, 2, "\xce\xbb") == 0);  /* UTF-8 'λ' */
        if (!is_lambda && !is_lambda_unicode) {
            throw py::value_error(std::string(caller) + ": func_source must start with (lambda …)");
        }

        /* Paren balance + quote check. We treat a '"' as a hard stop
         * because a well-formed lambda for autodiff has no business
         * carrying string literals, and allowing one would re-open the
         * injection hole via (lambda (x) (eval (string->symbol ".."))). */
        int depth = 0;
        for (size_t j = 0; j < s.size(); j++) {
            char c = s[j];
            if (c == '"') {
                throw py::value_error(std::string(caller) + ": func_source cannot contain string literals");
            }
            if (c == '(') depth++;
            else if (c == ')') {
                depth--;
                if (depth < 0) {
                    throw py::value_error(std::string(caller) + ": func_source has unmatched ')'");
                }
                if (depth == 0 && j != s.size() - 1) {
                    /* Paren balanced but there's trailing content — reject. */
                    throw py::value_error(std::string(caller) + ": func_source has trailing garbage after lambda");
                }
            }
        }
        if (depth != 0) {
            throw py::value_error(std::string(caller) + ": func_source has unmatched '('");
        }
    }

private:
    eshkol_ffi_context_t* ctx_;

    /* Non-copyable */
    EshkolContext(const EshkolContext&) = delete;
    EshkolContext& operator=(const EshkolContext&) = delete;
};

/* ── Module definition ── */
PYBIND11_MODULE(eshkol, m) {
    m.doc() = "Eshkol programming language — Python bindings";

    py::class_<EshkolContext>(m, "Context",
        "Eshkol evaluation context. Create one per session.\n"
        "State (definitions, variables) persists across eval() calls.")
        .def(py::init<>(), "Initialize the Eshkol runtime with JIT compilation.")
        .def("eval", &EshkolContext::eval,
             py::arg("source"),
             "Evaluate Eshkol source code and return the result as a Python value.\n"
             "Supports: integers, floats, booleans, strings, lists, tensors (→ numpy).")
        .def("eval_double", &EshkolContext::eval_double,
             py::arg("source"),
             "Evaluate and return result as a Python float.")
        .def("eval_file", &EshkolContext::eval_file,
             py::arg("path"),
             "Load and evaluate an Eshkol source file.")
        .def("derivative", &EshkolContext::derivative,
             py::arg("func"), py::arg("x"),
             "Compute the derivative of a function at point x.\n"
             "Example: ctx.derivative('sin', 0.5)  # => cos(0.5)")
        .def("gradient", &EshkolContext::gradient,
             py::arg("func"), py::arg("point"),
             "Compute the gradient of a function at a point.\n"
             "Example: ctx.gradient('(lambda (x y) (* x y))', [2.0, 3.0])")
    ;
}
