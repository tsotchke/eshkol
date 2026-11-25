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

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

/**
 * Shared ownership handle for an eshkol_ffi_context_t.
 *
 * The real teardown (eshkol_ffi_shutdown, which tears down the process-wide
 * JIT/runtime state the context was keeping resident) only runs once every
 * holder of one of these shared_ptrs has released it. EshkolContext holds
 * one for its own lifetime; every NumPy array exported from that context
 * (see the tensor branch of ffi_value_to_python below) holds a second,
 * independent copy inside its base capsule. This is what closes audit H1:
 * as long as any exported array — or any NumPy view derived from it, since
 * NumPy propagates `base` through slicing/reshaping — is alive, the
 * context cannot be shut down out from under it, even if the Python
 * `Context` object that produced the array has already been deleted.
 */
using EshkolCtxHandle = std::shared_ptr<eshkol_ffi_context_t>;

/**
 * Convert an Eshkol FFI value to a Python object.
 * Handles: null, int, double, bool, string, pair/list, tensor.
 *
 * @param ctx  Shared-ownership handle to the owning context. Threaded
 *             through recursion only so the tensor branch can attach a
 *             strong reference to any NumPy array it hands back — see
 *             EshkolCtxHandle above.
 */
static py::object ffi_value_to_python(const EshkolCtxHandle& ctx, eshkol_ffi_value_t val) {
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

            /* Tensor → numpy N-D array. The shape buffer is sized to
             * the reported ndim rather than a fixed 8 (audit M10 —
             * 9-D tensors were silently truncated, producing numpy
             * arrays with wrong ndim and misinterpreted strides). If
             * ndim is suspicious (≤0 or > 64) we fall back to 1-D.
             *
             * Zero-copy lifetime (audit H1, fixed): the tensor's backing
             * storage is owned by the Eshkol runtime the context keeps
             * resident, not by this array. The array's `base` is a
             * capsule holding a strong reference (EshkolCtxHandle) to
             * that context. NumPy keeps `base` alive for as long as this
             * array — or any view/slice/reshape derived from it — is
             * alive, and the capsule's destructor is the only thing that
             * releases our copy of the shared_ptr. So even if the Python
             * `Context` object is deleted (or every other reference to it
             * dropped) while this array is still around, the underlying
             * eshkol_ffi_shutdown() is deferred until the capsule itself
             * is destroyed — i.e. until this is the last live reference.
             * Callers that need a fully independent, lifetime-decoupled
             * array can still .copy() on the Python side; that is no
             * longer required for correctness, only to avoid pinning the
             * context's resources. */
            double* tdata = eshkol_ffi_tensor_data(val);
            if (tdata) {
                int64_t size = eshkol_ffi_tensor_size(val);
                if (size > 0) {
                    int ndim = eshkol_ffi_tensor_ndims(val);
                    if (ndim <= 0 || ndim > 64) ndim = 1;
                    std::vector<int64_t> shape_buf((size_t)ndim, 0);
                    int n = eshkol_ffi_tensor_shape(val, shape_buf.data(), ndim);
                    if (n <= 0) { shape_buf.resize(1); shape_buf[0] = size; n = 1; }

                    std::vector<py::ssize_t> shape_v(n), strides_v(n);
                    for (int i = 0; i < n; i++) shape_v[i] = (py::ssize_t)shape_buf[i];
                    strides_v[n - 1] = (py::ssize_t)sizeof(double);
                    for (int i = n - 2; i >= 0; i--) {
                        strides_v[i] = strides_v[i + 1] * shape_v[i + 1];
                    }
                    /* Stateless deleter (no captures, so it decays to the
                     * plain `void(*)(void*)` pybind11::capsule expects);
                     * the state lives in the heap-allocated shared_ptr
                     * copy the deleter frees. */
                    auto* owner = new EshkolCtxHandle(ctx);
                    py::capsule owner_capsule(owner, [](void* p) {
                        delete static_cast<EshkolCtxHandle*>(p);
                    });
                    return py::array_t<double>(shape_v, strides_v, tdata, owner_capsule);
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
 *
 * The underlying handle is held via EshkolCtxHandle (shared_ptr with
 * eshkol_ffi_shutdown as its deleter) rather than a raw pointer. This
 * object's own destruction only releases ITS copy of that shared_ptr;
 * the real eshkol_ffi_shutdown() call happens whenever the LAST copy
 * goes away, which may be a NumPy array exported from this context
 * (see the capsule attached in ffi_value_to_python) outliving the
 * EshkolContext object itself. That deferral is audit H1's fix: a
 * Context can be deleted, or fall out of scope, while arrays it
 * produced are still in use, without invalidating them.
 */
class EshkolContext {
public:
    EshkolContext() {
        eshkol_ffi_context_t* raw = eshkol_ffi_init();
        if (!raw) {
            const char* err = eshkol_ffi_last_error();
            throw std::runtime_error(err && err[0] ? err : "Failed to initialize Eshkol runtime");
        }
        ctx_ = EshkolCtxHandle(raw, eshkol_ffi_shutdown);
    }

    /* No custom destructor needed: ctx_'s shared_ptr deleter
     * (eshkol_ffi_shutdown) runs automatically once this is the last
     * reference — see the EshkolCtxHandle comment above. */

    /* Evaluate Eshkol source code, return Python value */
    py::object eval(const std::string& source) {
        eshkol_ffi_value_t result;
        int rc = eshkol_ffi_eval(ctx_.get(), source.c_str(), &result);
        if (rc != 0) {
            const char* err = eshkol_ffi_last_error();
            throw std::runtime_error(err ? err : "Evaluation failed");
        }
        return ffi_value_to_python(ctx_, result);
    }

    /* Evaluate and return as double (convenience for numeric work) */
    double eval_double(const std::string& source) {
        double result;
        int rc = eshkol_ffi_eval_double(ctx_.get(), source.c_str(), &result);
        if (rc != 0) {
            const char* err = eshkol_ffi_last_error();
            throw std::runtime_error(err ? err : "Evaluation failed");
        }
        return result;
    }

    /* Evaluate a file
     *
     * Security (#195): reject paths that are empty, contain NUL
     * bytes (string truncation attack), or exceed 4 KiB (heuristic
     * cap — legitimate paths are nowhere near this). We intentionally
     * do NOT call realpath() here because that would restrict the
     * caller to filesystem paths that already exist, which doesn't
     * match Python semantics for FileNotFoundError reporting. The
     * length + NUL guard is sufficient to close the string-truncation
     * injection surface.
     */
    void eval_file(const std::string& path) {
        if (path.empty()) {
            throw py::value_error("eval_file: path is empty");
        }
        if (path.size() > 4096) {
            throw py::value_error("eval_file: path exceeds 4 KiB");
        }
        if (path.find('\0') != std::string::npos) {
            throw py::value_error("eval_file: path contains NUL byte");
        }
        int rc = eshkol_ffi_eval_file(ctx_.get(), path.c_str());
        if (rc != 0) {
            const char* err = eshkol_ffi_last_error();
            throw std::runtime_error(err ? err : "File evaluation failed");
        }
    }

    /* Convert a NumPy ndarray into an arena-backed Eshkol float64 tensor with
     * strict exactness enforcement (zero silent dtype demotion or precision loss). */
    static eshkol_ffi_value_t numpy_to_ffi_tensor(eshkol_ffi_context_t* raw_ctx,
                                                  const py::object& obj,
                                                  bool allow_widening) {
        if (!py::isinstance<py::array>(obj)) {
            throw py::type_error("expected a numpy.ndarray");
        }
        py::array arr = py::reinterpret_borrow<py::array>(obj);
        py::buffer_info info = arr.request();

        if (info.ndim < 1 || info.ndim > 8) {
            throw py::value_error("tensor ndim must be between 1 and 8");
        }

        std::vector<int64_t> shape(static_cast<size_t>(info.ndim));
        int64_t total = 1;
        for (py::ssize_t i = 0; i < info.ndim; ++i) {
            if (info.shape[i] <= 0) {
                throw py::value_error("tensor dimensions must be strictly positive");
            }
            if (total > std::numeric_limits<int64_t>::max() / static_cast<int64_t>(info.shape[i])) {
                throw py::value_error("tensor element count overflow");
            }
            shape[static_cast<size_t>(i)] = static_cast<int64_t>(info.shape[i]);
            total *= shape[static_cast<size_t>(i)];
        }

        const py::dtype dt = arr.dtype();
        const char kind = dt.kind();
        const py::ssize_t itemsize = dt.itemsize();

        /* Always refuse non-real / object / complex / boolean dtypes. */
        if (kind == 'c') {
            throw py::type_error("complex NumPy dtypes cannot be silently demoted to real float64 tensors");
        }
        if (kind != 'f' && kind != 'i' && kind != 'u') {
            throw py::type_error("unsupported NumPy dtype for Eshkol tensor boundary (expected float64)");
        }

        /* Fast, bit-exact IEEE-754 binary64 path (always permitted). */
        if (kind == 'f' && itemsize == 8) {
            py::array_t<double, py::array::c_style> contig;
            const double* src = nullptr;
            bool is_c_contig = (arr.flags() & py::array::c_style) != 0;
            if (is_c_contig) {
                src = static_cast<const double*>(info.ptr);
            } else {
                contig = py::array_t<double, py::array::c_style>::ensure(arr);
                if (!contig) {
                    throw py::type_error("failed to materialize contiguous float64 view");
                }
                src = static_cast<const double*>(contig.request().ptr);
            }
            eshkol_ffi_value_t v = eshkol_ffi_tensor_from_data(
                raw_ctx, src, shape.data(), static_cast<int>(shape.size()));
            if (eshkol_ffi_is_null(v)) {
                throw std::runtime_error("failed to allocate Eshkol tensor from float64 array");
            }
            return v;
        }

        /* Any non-float64 dtype is refused by default to prevent silent precision
         * mismatches or truncation across the Python/NumPy boundary. */
        if (!allow_widening) {
            throw py::type_error(
                "refusing silent dtype conversion across NumPy->Eshkol boundary: "
                "expected float64 (pass allow_widening=True for lossless widening)");
        }

        /* Explicit lossless widening path: verify exactness for 64-bit integers. */
        std::vector<double> widened(static_cast<size_t>(total));
        if (kind == 'f' && itemsize == 4) {
            auto c = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(arr);
            const float* p = static_cast<const float*>(c.request().ptr);
            for (int64_t i = 0; i < total; ++i) widened[static_cast<size_t>(i)] = static_cast<double>(p[i]);
        } else if (kind == 'f' && itemsize == 2) {
            py::object f64 = arr.attr("astype")(py::dtype::of<double>());
            auto c = py::array_t<double, py::array::c_style>::ensure(f64);
            const double* p = static_cast<const double*>(c.request().ptr);
            std::memcpy(widened.data(), p, static_cast<size_t>(total) * sizeof(double));
        } else if (kind == 'i' && itemsize == 8) {
            auto c = py::array_t<int64_t, py::array::c_style>::ensure(arr);
            const int64_t* p = static_cast<const int64_t*>(c.request().ptr);
            constexpr int64_t kMaxExact = (INT64_C(1) << 53);
            for (int64_t i = 0; i < total; ++i) {
                if (p[i] > kMaxExact || p[i] < -kMaxExact) {
                    throw py::value_error(
                        "refusing lossy int64->float64 demotion: element exceeds exact 53-bit IEEE-754 mantissa");
                }
                widened[static_cast<size_t>(i)] = static_cast<double>(p[i]);
            }
        } else if (kind == 'u' && itemsize == 8) {
            auto c = py::array_t<uint64_t, py::array::c_style>::ensure(arr);
            const uint64_t* p = static_cast<const uint64_t*>(c.request().ptr);
            constexpr uint64_t kMaxExact = (UINT64_C(1) << 53);
            for (int64_t i = 0; i < total; ++i) {
                if (p[i] > kMaxExact) {
                    throw py::value_error(
                        "refusing lossy uint64->float64 demotion: element exceeds exact 53-bit IEEE-754 mantissa");
                }
                widened[static_cast<size_t>(i)] = static_cast<double>(p[i]);
            }
        } else if ((kind == 'i' || kind == 'u') && (itemsize == 1 || itemsize == 2 || itemsize == 4)) {
            py::object f64 = arr.attr("astype")(py::dtype::of<double>());
            auto c = py::array_t<double, py::array::c_style>::ensure(f64);
            const double* p = static_cast<const double*>(c.request().ptr);
            std::memcpy(widened.data(), p, static_cast<size_t>(total) * sizeof(double));
        } else {
            throw py::type_error("unsupported NumPy dtype for lossless widening to float64");
        }

        eshkol_ffi_value_t v = eshkol_ffi_tensor_from_data(
            raw_ctx, widened.data(), shape.data(), static_cast<int>(shape.size()));
        if (eshkol_ffi_is_null(v)) {
            throw std::runtime_error("failed to allocate Eshkol tensor");
        }
        return v;
    }

    /* Round-trip a NumPy array through an Eshkol arena tensor and back as a
     * zero-copy NumPy view, enforcing strict exactness and shape preservation. */
    py::object tensor_roundtrip(const py::object& arr, bool allow_widening = false) {
        eshkol_ffi_value_t v = numpy_to_ffi_tensor(ctx_.get(), arr, allow_widening);
        return ffi_value_to_python(ctx_, v);
    }

    /* Bind a NumPy array directly to a top-level Eshkol variable name without
     * string serialization, preserving exact IEEE-754 bits and N-D shape. */
    void bind_tensor(const std::string& name, const py::object& arr, bool allow_widening = false) {
        eshkol_ffi_value_t v = numpy_to_ffi_tensor(ctx_.get(), arr, allow_widening);
        if (eshkol_ffi_bind_value(ctx_.get(), name.c_str(), v) != 0) {
            const char* err = eshkol_ffi_last_error();
            throw std::runtime_error(err ? err : "Failed to bind tensor");
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
        if (!std::isfinite(x)) {
            throw py::value_error("derivative: evaluation point x must be finite");
        }
        char num[64];
        snprintf(num, sizeof(num), "%.17g", x);
        std::string expr = "(derivative " + func_source + " " + num + ")";
        return eval_double(expr);
    }

    /* Compute gradient: gradient(f_source, point_list) */
    py::object gradient(const std::string& func_source, const std::vector<double>& point) {
        validate_lambda_source(func_source, "gradient");
        std::string expr = "(gradient " + func_source + " (list";
        for (double v : point) {
            if (!std::isfinite(v)) {
                throw py::value_error("gradient: all point coordinates must be finite");
            }
            char num[64];
            snprintf(num, sizeof(num), " %.17g", v);
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
    EshkolCtxHandle ctx_;

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
        .def("bind_tensor", &EshkolContext::bind_tensor,
             py::arg("name"), py::arg("array"), py::arg("allow_widening") = false,
             "Bind a NumPy ndarray directly to a top-level Eshkol variable without stringification.\n"
             "Refuses non-float64 dtypes unless allow_widening=True (and refuses lossy int64 > 2^53).")
        .def("tensor_roundtrip", &EshkolContext::tensor_roundtrip,
             py::arg("array"), py::arg("allow_widening") = false,
             "Round-trip a NumPy ndarray through an Eshkol arena tensor with exact IEEE-754 bit and shape preservation.")
    ;
}
