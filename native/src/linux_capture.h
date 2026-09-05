#pragma once
#if defined(__linux__) && !defined(__ANDROID__)
#include <inferbridge/linux_capture_vulkan.h>
struct da3_context;
ibr_linux_capture_capabilities da3_linux_capture_capabilities(da3_context *);
void da3_infer_linux_capture(
    da3_context *, const inferbridge::linux_capture::LinuxDmaBufImage &,
    uint32_t, float *);
#endif
