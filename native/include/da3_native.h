#ifndef DA3_NATIVE_H
#define DA3_NATIVE_H

#include <stdint.h>

#if defined(_WIN32)
#  if defined(DA3_BUILD_DLL)
#    define DA3_API __declspec(dllexport)
#  else
#    define DA3_API __declspec(dllimport)
#  endif
#  define DA3_CALL __cdecl
#else
#  define DA3_API __attribute__((visibility("default")))
#  define DA3_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define DA3_ABI_VERSION 3u

typedef struct da3_context da3_context;

typedef struct da3_transfer_counters {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t tensor_upload_bytes;
    uint64_t tensor_download_bytes;
} da3_transfer_counters;

typedef enum da3_status {
    DA3_STATUS_OK = 0,
    DA3_STATUS_INVALID_ARGUMENT = 1,
    DA3_STATUS_MODEL_IO = 2,
    DA3_STATUS_MODEL_FORMAT = 3,
    DA3_STATUS_VULKAN_UNAVAILABLE = 4,
    DA3_STATUS_OUT_OF_MEMORY = 5,
    DA3_STATUS_INFERENCE_FAILED = 6,
    DA3_STATUS_BUFFER_TOO_SMALL = 7,
    DA3_STATUS_UNSUPPORTED = 8,
    DA3_STATUS_INTERNAL_ERROR = 9
} da3_status;

DA3_API uint32_t DA3_CALL da3_abi_version(void);
DA3_API const char* DA3_CALL da3_version_string(void);
DA3_API const char* DA3_CALL da3_status_string(da3_status status);
DA3_API const char* DA3_CALL da3_last_error(void);
DA3_API da3_status DA3_CALL da3_create(
    const char* model_safetensors_path_utf8,
    da3_context** context);
/*
 * Creates a real full-graph Vulkan context on the zero-based physical-device
 * index. Failure is reported; this function never falls back to CPU.
 */
DA3_API da3_status DA3_CALL da3_create_vulkan(
    const char* model_safetensors_path_utf8,
    uint32_t device_index,
    da3_context** context);
DA3_API void DA3_CALL da3_destroy(da3_context* context);
/*
 * Executes the lean single-view depth graph used by InferBridge. Input is
 * normalized contiguous RGB CHW FP32; output is contiguous HW depth.
 * Width and height must be positive multiples of 14.
 */
DA3_API da3_status DA3_CALL da3_infer_tensor_f32(
    da3_context* context,
    const float* normalized_rgb_chw,
    int32_t width,
    int32_t height,
    float* depth_hw,
    uint64_t depth_elements);

DA3_API da3_status DA3_CALL da3_inferbridge_image_shape(
    int32_t image_width,
    int32_t image_height,
    int32_t process_resolution,
    int32_t* depth_width,
    int32_t* depth_height);

/*
 * Reproduces the current InferBridge worker's BGRA/BGR byte ordering,
 * upper-bound resizing, patch rounding, ImageNet normalization, and output
 * min/max normalization. Output dimensions are returned by
 * da3_inferbridge_image_shape.
 */
DA3_API da3_status DA3_CALL da3_infer_bgra8_f32(
    da3_context* context,
    const uint8_t* bgra,
    uint64_t bgra_stride_bytes,
    int32_t image_width,
    int32_t image_height,
    int32_t process_resolution,
    float* depth_hw,
    uint64_t depth_elements);

/* Process-wide diagnostics used to prove zero per-frame host staging. */
DA3_API da3_status DA3_CALL da3_get_transfer_counters(
    da3_transfer_counters* counters);

#ifdef __cplusplus
}
#endif

#endif
