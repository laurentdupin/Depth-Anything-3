#include "da3_native.h"

#include <assert.h>
#include <string.h>

int main(void) {
    da3_context* context = 0;
    assert(da3_abi_version() == DA3_ABI_VERSION);
    assert(da3_version_string() != 0);
    assert(strcmp(da3_status_string(DA3_STATUS_OK), "ok") == 0);
    assert(da3_last_error() != 0);
    assert(
        da3_create(0, &context) == DA3_STATUS_INVALID_ARGUMENT);
    assert(context == 0);
    assert(
        da3_inferbridge_image_shape(
            0, 0, 0, 0, 0) == DA3_STATUS_INVALID_ARGUMENT);
    assert(
        da3_infer_bgra8_f32(
            0, 0, 0, 0, 0, 0, 0, 0) ==
        DA3_STATUS_INVALID_ARGUMENT);
    da3_destroy(0);
    return 0;
}
