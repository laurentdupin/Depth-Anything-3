#include "da3_native.h"

#include "dpt_cpu.h"
#include "encoder_cpu.h"
#include "image.h"
#include "safetensors.h"
#if defined(DA3_WITH_VULKAN)
#include "da3_internal.h"
#endif

#include <algorithm>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

struct da3_context {
    std::unique_ptr<da3_native::SafeTensors> model;
#if defined(DA3_WITH_VULKAN)
    std::shared_ptr<da3_native::ExternalGpu> external_gpu;
#endif
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
    return "0.4.0-single-view-image-cpu-vulkan";
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

da3_status DA3_CALL da3_create_vulkan(
    const char* model_path,
    uint32_t device_index,
    da3_context** context) {
    if (!context) {
        return fail(DA3_STATUS_INVALID_ARGUMENT, "context is null");
    }
    *context = nullptr;
    if (!model_path || model_path[0] == '\0') {
        return fail(DA3_STATUS_INVALID_ARGUMENT, "model path is empty");
    }
#if !defined(DA3_WITH_VULKAN)
    (void)device_index;
    return fail(
        DA3_STATUS_VULKAN_UNAVAILABLE,
        "this DLL was built without Vulkan");
#else
    return protect([&] {
        auto result = std::make_unique<da3_context>();
        result->external_gpu =
            da3_native::create_external_gpu(model_path, device_index);
        *context = result.release();
    });
#endif
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
    if (!context || (!context->model
#if defined(DA3_WITH_VULKAN)
            && !context->external_gpu
#endif
        ) || !input || !depth ||
        width <= 0 || height <= 0 ||
        width % 14 != 0 || height % 14 != 0 ||
        depth_elements < std::uint64_t(width) * height) {
        return fail(
            DA3_STATUS_INVALID_ARGUMENT,
            "invalid single-view tensor inference input");
    }
    return protect([&] {
#if defined(DA3_WITH_VULKAN)
        if (context->external_gpu) {
            context->external_gpu->infer(
                input, static_cast<std::uint32_t>(width),
                static_cast<std::uint32_t>(height), depth);
            return;
        }
#endif
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

da3_status DA3_CALL da3_inferbridge_image_shape(
    int32_t image_width,
    int32_t image_height,
    int32_t process_resolution,
    int32_t* depth_width,
    int32_t* depth_height) {
    if (image_width <= 0 || image_height <= 0 ||
        process_resolution <= 0 || !depth_width || !depth_height) {
        return fail(
            DA3_STATUS_INVALID_ARGUMENT,
            "invalid DA3 image shape input");
    }
    return protect([&] {
        const da3_native::ImageShape shape =
            da3_native::inferbridge_image_shape(
                static_cast<std::uint32_t>(image_width),
                static_cast<std::uint32_t>(image_height),
                static_cast<std::uint32_t>(process_resolution));
        *depth_width = static_cast<int32_t>(shape.width);
        *depth_height = static_cast<int32_t>(shape.height);
    });
}

da3_status DA3_CALL da3_infer_bgra8_f32(
    da3_context* context,
    const uint8_t* bgra,
    uint64_t bgra_stride_bytes,
    int32_t image_width,
    int32_t image_height,
    int32_t process_resolution,
    float* depth,
    uint64_t depth_elements) {
    if (!context || (!context->model
#if defined(DA3_WITH_VULKAN)
            && !context->external_gpu
#endif
        ) || !bgra || !depth ||
        image_width <= 0 || image_height <= 0 ||
        process_resolution <= 0 ||
        bgra_stride_bytes <
            static_cast<std::uint64_t>(image_width) * 4) {
        return fail(
            DA3_STATUS_INVALID_ARGUMENT,
            "invalid DA3 BGRA image inference input");
    }
    return protect([&] {
        const da3_native::ImageShape shape =
            da3_native::inferbridge_image_shape(
                static_cast<std::uint32_t>(image_width),
                static_cast<std::uint32_t>(image_height),
                static_cast<std::uint32_t>(process_resolution));
        if (depth_elements <
            std::uint64_t(shape.width) * shape.height) {
            throw std::invalid_argument(
                "DA3 image inference output buffer is too small");
        }
        std::vector<float> prepared =
            da3_native::preprocess_inferbridge_bgra8(
                bgra, static_cast<std::size_t>(bgra_stride_bytes),
                static_cast<std::uint32_t>(image_width),
                static_cast<std::uint32_t>(image_height),
                static_cast<std::uint32_t>(process_resolution));
#if defined(DA3_WITH_VULKAN)
        if (context->external_gpu) {
            context->external_gpu->infer(
                prepared.data(), shape.width, shape.height, depth);
        } else
#endif
        {
            std::vector<float> result =
                da3_native::depth_head_single_view_cpu(
                    *context->model,
                    da3_native::encoder_single_view_cpu(
                        *context->model, prepared.data(),
                        shape.width, shape.height));
            std::copy(result.begin(), result.end(), depth);
        }
        const std::size_t count =
            static_cast<std::size_t>(shape.width) * shape.height;
        const auto bounds = std::minmax_element(depth, depth + count);
        const float minimum = *bounds.first;
        const float span = *bounds.second - minimum;
        if (!(span > 0.0f)) {
            std::fill_n(depth, count, 0.0f);
            return;
        }
        for (std::size_t index = 0; index < count; ++index) {
            depth[index] = (depth[index] - minimum) / span;
        }
    });
}

da3_status DA3_CALL da3_get_transfer_counters(
    da3_transfer_counters* counters) {
    if (counters == nullptr || counters->struct_size < sizeof(*counters))
        return fail(DA3_STATUS_INVALID_ARGUMENT,
                    "invalid DA3 transfer counter descriptor");
    *counters = {};
    counters->struct_size = sizeof(*counters);
    counters->abi_version = DA3_ABI_VERSION;
#if defined(DA3_WITH_VULKAN)
    da3_native::global_transfer_counters(
        counters->tensor_upload_bytes,
        counters->tensor_download_bytes);
#endif
    return DA3_STATUS_OK;
}

}  // extern "C"

#if defined(DA3_WITH_VULKAN)
namespace da3_native {

ExternalGpuCapabilities context_external_capabilities(
    const da3_context* context) {
    return context != nullptr && context->external_gpu
        ? context->external_gpu->capabilities()
        : ExternalGpuCapabilities{};
}

std::shared_ptr<ExternalJob> submit_external_texture(
    da3_context* context,
    const ExternalTextureRequest& request) {
    if (context == nullptr || !context->external_gpu)
        throw std::runtime_error("DA3 context has no Vulkan executor");
    return context->external_gpu->submit_texture(request);
}

void context_transfer_counters(
    const da3_context* context,
    std::uint64_t& upload_bytes,
    std::uint64_t& download_bytes) {
    if (context == nullptr || !context->external_gpu)
        throw std::runtime_error("DA3 context has no Vulkan executor");
    context->external_gpu->transfer_counters(upload_bytes, download_bytes);
}

}  // namespace da3_native
#endif
