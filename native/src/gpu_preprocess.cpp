#include "gpu_preprocess.h"

#include "preprocess_texture_spv.h"
#include "preprocess_finish_spv.h"

#include <stdexcept>

namespace da3_native {

GpuPreprocessor::GpuPreprocessor(VulkanContext& context)
    : context_(context),
      texture_pipeline_(context.create_pipeline(
          da3_preprocess_texture_spv,
          da3_preprocess_texture_spv_size,
          {
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          },
          {
              VK_ACCESS_SHADER_READ_BIT,
              VK_ACCESS_SHADER_WRITE_BIT,
          },
          4 * sizeof(std::uint32_t))),
      finish_pipeline_(context.create_pipeline(
          da3_preprocess_finish_spv,
          da3_preprocess_finish_spv_size,
          {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
          {VK_ACCESS_SHADER_READ_BIT,
           VK_ACCESS_SHADER_WRITE_BIT},
          4 * sizeof(std::uint32_t))) {
    texture_pipeline_.set_debug_name("preprocess_texture");
    finish_pipeline_.set_debug_name("preprocess_finish");
}

void GpuPreprocessor::run_texture(
    VulkanBuffer& destination,
    const VulkanImage& source,
    std::uint32_t intermediate_width,
    std::uint32_t intermediate_height,
    std::uint32_t destination_width,
    std::uint32_t destination_height) {
    if (source.width() == 0 || source.height() == 0 ||
        intermediate_width == 0 || intermediate_height == 0 ||
        destination_width == 0 || destination_height == 0) {
        throw std::invalid_argument(
            "invalid DA3 GPU texture preprocessing dimensions");
    }
    const std::uint64_t destination_bytes =
        static_cast<std::uint64_t>(destination_width) *
        destination_height * 3u * sizeof(float);
    if (destination_bytes > destination.size()) {
        throw std::invalid_argument(
            "DA3 GPU preprocessing buffer is too small");
    }
    VulkanBuffer intermediate = context_.create_device_buffer(
        static_cast<std::uint64_t>(intermediate_width) *
        intermediate_height * sizeof(std::uint32_t));
    struct Parameters {
        std::uint32_t source_width;
        std::uint32_t source_height;
        std::uint32_t destination_width;
        std::uint32_t destination_height;
    } first{
        source.width(), source.height(),
        intermediate_width, intermediate_height};
    context_.dispatch_image_to_buffer(
        texture_pipeline_, source, intermediate,
        &first, sizeof(first),
        (intermediate_width + 7u) / 8u,
        (intermediate_height + 7u) / 8u);
    const Parameters finish{
        intermediate_width, intermediate_height,
        destination_width, destination_height};
    context_.dispatch(
        finish_pipeline_, {&intermediate, &destination},
        &finish, sizeof(finish),
        (destination_width + 7u) / 8u,
        (destination_height + 7u) / 8u);
}

}  // namespace da3_native
