#pragma once

#include "vulkan.h"

#include <cstdint>

namespace da3_native {

// Final relative-depth post-processing. The reduction, resize, and image
// write all remain on the selected Vulkan device; no depth values cross RAM.
class GpuOutput {
public:
    explicit GpuOutput(VulkanContext& context);

    void resize_and_normalize(
        VulkanImage& destination, const VulkanBuffer& source,
        std::uint32_t source_width, std::uint32_t source_height,
        std::uint32_t width, std::uint32_t height);

private:
    VulkanContext& context_;
    VulkanPipeline resize_;
    VulkanPipeline reduce_;
    VulkanPipeline normalize_;
};

}  // namespace da3_native
