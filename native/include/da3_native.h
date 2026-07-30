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

#define DA3_ABI_VERSION 1u

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

#ifdef __cplusplus
}
#endif

#endif
