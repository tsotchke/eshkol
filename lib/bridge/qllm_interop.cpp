#include "eshkol/bridge/qllm_bridge.h"
#include <limits>
#include <cstdlib>
#include <cstring>
#include <mutex>
#if defined(ESHKOL_HAS_QLLM)
#include "../backend/eskb_format.h"
#endif

namespace {
std::mutex bridge_mutex;
bool bridge_ready = false;
#if defined(ESHKOL_HAS_QLLM)
bool cpu_float32_dense(const qllm_tensor_t* tensor) {
    if (!tensor || tensor->options.dtype != QLLM_DTYPE_FLOAT32 ||
        tensor->options.device != QLLM_DEVICE_CPU ||
        tensor->type != QLLM_TENSOR_TYPE_DENSE || !tensor->data ||
        !tensor->shape || !tensor->strides || !tensor->dims) return false;
    size_t count = 1;
    for (size_t i = tensor->dims; i-- > 0;) {
        if (!tensor->shape[i] || tensor->strides[i] != count ||
            count > std::numeric_limits<size_t>::max() / tensor->shape[i]) return false;
        count *= tensor->shape[i];
    }
    return count == tensor->size;
}
#endif
}

extern "C" bool eshkol_qllm_bridge_available(void) {
#if defined(ESHKOL_HAS_QLLM)
    return true;
#else
    return false;
#endif
}

extern "C" qllm_tensor_t* eshkol_to_qllm_tensor(
    const double* values, const size_t* shape, size_t ndim) {
#if defined(ESHKOL_HAS_QLLM)
    if (!values || !shape || !ndim || ndim > SIZE_MAX / sizeof(size_t)) return nullptr;
    size_t count = 1;
    for (size_t i = 0; i < ndim; ++i) {
        if (!shape[i] || count > SIZE_MAX / shape[i]) return nullptr;
        count *= shape[i];
    }
    if (count > SIZE_MAX / sizeof(float)) return nullptr;
    auto options = qllm_tensor_options_default(QLLM_DEVICE_CPU);
    options.dtype = QLLM_DTYPE_FLOAT32;
    auto* tensor = qllm_tensor_create(ndim, shape, &options);
    if (!tensor) return nullptr;
    auto* data = static_cast<float*>(qllm_tensor_get_data(tensor));
    if (!data) { qllm_tensor_destroy(tensor); return nullptr; }
    for (size_t i = 0; i < count; ++i) data[i] = static_cast<float>(values[i]);
    return tensor;
#else
    (void)values; (void)shape; (void)ndim;
    return nullptr;
#endif
}

extern "C" bool qllm_to_eshkol_tensor(
    const qllm_tensor_t* tensor, double* out, size_t* size) {
#if defined(ESHKOL_HAS_QLLM)
    if (!size || !cpu_float32_dense(tensor)) return false;
    const size_t capacity = *size;
    *size = qllm_tensor_get_size(tensor);
    if (!out || capacity < *size) return false;
    const auto* data = static_cast<const float*>(qllm_tensor_get_data_const(tensor));
    for (size_t i = 0; i < *size; ++i) out[i] = data[i];
    return true;
#else
    (void)tensor; (void)out;
    if (size) *size = 0;
    return false;
#endif
}

extern "C" void eshkol_qllm_tensor_destroy(qllm_tensor_t* tensor) {
#if defined(ESHKOL_HAS_QLLM)
    qllm_tensor_destroy(tensor);
#else
    (void)tensor;
#endif
}

#if defined(ESHKOL_HAS_QLLM)
extern "C" qllm_error_code_t eshkol_qllm_tensor_to_eskb(
    const qllm_tensor_t* program, const qllm_eshkol_const_entry_t* constants,
    size_t n_constants, const char* name, uint8_t** out_buf, size_t* out_len) {
    if (out_buf) *out_buf = nullptr;
    if (out_len) *out_len = 0;
    if (!cpu_float32_dense(program)) return QLLM_ERROR_INVALID_PARAM;
    const auto status = qllm_eshkol_tensor_to_eskb_chunk_typed(
        program, constants, n_constants, name, out_buf, out_len);
    if (status != QLLM_SUCCESS) return status;
    uint32_t version = 0;
    if (*out_len >= sizeof(EskbHeader)) std::memcpy(&version, *out_buf + 4, 4);
    if (version != ESKB_VERSION) {
        std::free(*out_buf);
        *out_buf = nullptr;
        *out_len = 0;
        QLLM_SET_ERRORF(QLLM_ERROR_INVALID_STATE_CODE,
            "qLLM emits ESKB version %u; this Eshkol requires version %u",
            version, ESKB_VERSION);
        return QLLM_ERROR_INVALID_STATE_CODE;
    }
    return QLLM_SUCCESS;
}
#endif

extern "C" bool eshkol_qllm_bridge_init(const char* library_path) {
    std::lock_guard<std::mutex> lock(bridge_mutex);
    if (library_path && *library_path) return false;
    if (bridge_ready) return true;
#if defined(ESHKOL_HAS_QLLM)
    if (qllm_eshkol_register_qllm_natives() != QLLM_SUCCESS) return false;
    // qLLM's registrar may return success without a JIT host. Check execution,
    // including the tagged-value C ABI, before publishing session readiness.
    int64_t freed = 0;
    bridge_ready = qllm_eshkol_eval_to_int64(
        "(qllm-tensor-destroy (qllm-tensor-create-zeros 1 1))", &freed)
        == QLLM_SUCCESS && freed == 1;
#endif
    return bridge_ready;
}

extern "C" void eshkol_qllm_bridge_shutdown(void) {
    std::lock_guard<std::mutex> lock(bridge_mutex);
    bridge_ready = false;
}

extern "C" bool eshkol_qllm_bridge_ready(void) {
    std::lock_guard<std::mutex> lock(bridge_mutex);
    return bridge_ready;
}
