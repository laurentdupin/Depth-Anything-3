#include "da3_native.h"

#include "dpt_cpu.h"
#include "encoder_cpu.h"
#include "safetensors.h"

#include <algorithm>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

struct da3_context {
    std::unique_ptr<da3_native::SafeTensors> model;
};

namespace {
thread_local std::string last_error;

da3_status fail(da3_status status, const char* message) {
    last_error = message ? message : "";
    return status;
}

template <typename Function>
da3_status protect(Function&& function) {
    try {
        function();
        last_error.clear();
        return DA3_STATUS_OK;
    } catch (const std::bad_alloc&) {
        return fail(DA3_STATUS_OUT_OF_MEMORY, "out of memory");
    } catch (const std::invalid_argument& error) {
        return fail(DA3_STATUS_INVALID_ARGUMENT, error.what());
    } catch (const std::exception& error) {
        return fail(DA3_STATUS_INTERNAL_ERROR, error.what());
    } catch (...) {
        return fail(DA3_STATUS_INTERNAL_ERROR, "unknown internal error");
    }
}
}

extern "C" {

uint32_t DA3_CALL da3_abi_version(void) {
    return DA3_ABI_VERSION;
}

const char* DA3_CALL da3_version_string(void) {
    return "0.2.0-single-view-cpu";
}

const char* DA3_CALL da3_status_string(da3_status status) {
    switch (status) {
        case DA3_STATUS_OK: return "ok";
        case DA3_STATUS_INVALID_ARGUMENT: return "invalid argument";
        case DA3_STATUS_MODEL_IO: return "model I/O error";
        case DA3_STATUS_MODEL_FORMAT: return "invalid model format";
        case DA3_STATUS_VULKAN_UNAVAILABLE: return "Vulkan unavailable";
        case DA3_STATUS_OUT_OF_MEMORY: return "out of memory";
        case DA3_STATUS_INFERENCE_FAILED: return "inference failed";
        case DA3_STATUS_BUFFER_TOO_SMALL: return "buffer too small";
        case DA3_STATUS_UNSUPPORTED: return "unsupported";
        case DA3_STATUS_INTERNAL_ERROR: return "internal error";
        default: return "unknown status";
    }
}

const char* DA3_CALL da3_last_error(void) {
    return last_error.c_str();
}

da3_status DA3_CALL da3_create(
    const char* model_path,
    da3_context** context) {
    if (!context) {
        return fail(DA3_STATUS_INVALID_ARGUMENT, "context is null");
    }
    *context = nullptr;
    if (!model_path || model_path[0] == '\0') {
        return fail(DA3_STATUS_INVALID_ARGUMENT, "model path is empty");
    }
    return protect([&] {
        auto result = std::make_unique<da3_context>();
        result->model =
            std::make_unique<da3_native::SafeTensors>(model_path);
        *context = result.release();
    });
}

void DA3_CALL da3_destroy(da3_context* context) {
    delete context;
}

da3_status DA3_CALL da3_infer_tensor_f32(
    da3_context* context,
    const float* input,
    int32_t width,
    int32_t height,
    float* depth,
    uint64_t depth_elements) {
    if (!context || !context->model || !input || !depth ||
        width <= 0 || height <= 0 ||
        width % 14 != 0 || height % 14 != 0 ||
        depth_elements < std::uint64_t(width) * height) {
        return fail(
            DA3_STATUS_INVALID_ARGUMENT,
            "invalid single-view tensor inference input");
    }
    return protect([&] {
        std::vector<float> result =
            da3_native::depth_head_single_view_cpu(
                *context->model,
                da3_native::encoder_single_view_cpu(
                    *context->model, input,
                    static_cast<std::uint32_t>(width),
                    static_cast<std::uint32_t>(height)));
        std::copy(result.begin(), result.end(), depth);
    });
}

}
