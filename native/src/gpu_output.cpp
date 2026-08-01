#include "gpu_output.h"

#include "normalize_depth_image_spv.h"
#include "reduce_minmax_spv.h"
#include "resize_depth_spv.h"

#include <stdexcept>

namespace da3_native {

GpuOutput::GpuOutput(VulkanContext& context)
    : context_(context),
      resize_(context.create_pipeline(
          da3_resize_depth_spv, da3_resize_depth_spv_size, 2, 16)),
      reduce_(context.create_pipeline(
          da3_reduce_minmax_spv, da3_reduce_minmax_spv_size, 2, 4)),
      normalize_(context.create_pipeline(
          da3_normalize_depth_image_spv,
          da3_normalize_depth_image_spv_size,
          {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
          {VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
           VK_ACCESS_SHADER_READ_BIT}, 8)) {
    resize_.set_debug_name("da3_resize_depth");
    reduce_.set_debug_name("da3_reduce_minmax");
    normalize_.set_debug_name("da3_normalize_depth_image");
}

void GpuOutput::resize_and_normalize(
    VulkanImage& destination, const VulkanBuffer& source,
    std::uint32_t source_width, std::uint32_t source_height,
    std::uint32_t width, std::uint32_t height) {
    if (source_width == 0u || source_height == 0u || width == 0u ||
        height == 0u || destination.width() != width ||
        destination.height() != height ||
        destination.format() != VK_FORMAT_R32_SFLOAT)
        throw std::invalid_argument("invalid DA3 GPU depth output dimensions");

    VulkanBuffer resized = context_.create_device_buffer(
        static_cast<std::uint64_t>(width) * height * sizeof(float));
    struct ResizeParameters {
        std::uint32_t source_width, source_height, width, height;
    } resize_parameters{source_width, source_height, width, height};
    context_.dispatch(
        resize_, {&source, &resized}, &resize_parameters,
        sizeof(resize_parameters), (width + 7u) / 8u, (height + 7u) / 8u);

    VulkanBuffer range = context_.create_device_buffer(2u * sizeof(float));
    const std::uint32_t count = width * height;
    context_.dispatch(reduce_, {&resized, &range}, &count, sizeof(count), 1u);
    struct OutputParameters { std::uint32_t width, height; } output_parameters{
        width, height};
    context_.dispatch_buffers_to_image(
        normalize_, {&resized, &range}, destination,
        &output_parameters, sizeof(output_parameters),
        (width + 7u) / 8u, (height + 7u) / 8u);
}

}  // namespace da3_native
