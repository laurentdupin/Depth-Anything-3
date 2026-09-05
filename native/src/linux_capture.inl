#if defined(__linux__) && !defined(__ANDROID__) && defined(DA3_WITH_VULKAN)
#include "linux_capture.h"
#include <cmath>
#include <inferbridge/linux_capture_harness.h>
ibr_linux_capture_capabilities
da3_linux_capture_capabilities(da3_context *context) {
  return context->external_gpu
             ? context->external_gpu->linux_capture_capabilities()
             : ibr_linux_capture_capabilities{};
}
void da3_infer_linux_capture(
    da3_context *context,
    const inferbridge::linux_capture::LinuxDmaBufImage &source, uint32_t size,
    float *output) {
  if (!context->external_gpu)
    throw std::runtime_error("DA3 Vulkan context unavailable");
  const auto shape =
      da3_native::inferbridge_image_shape(source.width, source.height, size);
  const double scale = double(size) / std::max(source.width, source.height);
  const uint32_t first_width =
      std::max(1, int(std::nearbyint(source.width * scale)));
  const uint32_t first_height =
      std::max(1, int(std::nearbyint(source.height * scale)));
  std::vector<float> depth(uint64_t(shape.width) * shape.height);
  context->external_gpu->infer_linux_capture(source, first_width, first_height,
                                             depth.data());
  const auto bounds = std::minmax_element(depth.begin(), depth.end());
  const float minimum = *bounds.first, span = *bounds.second - minimum;
  for (auto &value : depth)
    value = span > 0 ? 1.f - (value - minimum) / span : 0.f;
  inferbridge::linux_capture::resize_nearest(depth.data(), shape.width,
                                             shape.height, output, source.width,
                                             source.height);
}
#endif
