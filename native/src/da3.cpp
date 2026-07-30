#include "da3_native.h"

#include <string>

namespace {
thread_local std::string last_error;
}

extern "C" {

uint32_t DA3_CALL da3_abi_version(void) {
    return DA3_ABI_VERSION;
}

const char* DA3_CALL da3_version_string(void) {
    return "0.1.0-model-foundation";
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

}
