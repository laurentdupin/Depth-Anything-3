#pragma once

#include "vulkan.h"

#include <cstdint>

namespace da3_native {

class GpuPreprocessor {
public:
    explicit GpuPreprocessor(VulkanContext& context);
    void run_texture(
        VulkanBuffer& destination,
        const VulkanImage& source,
        std::uint32_t intermediate_width,
        std::uint32_t intermediate_height,
        std::uint32_t destination_width,
        std::uint32_t destination_height);

private:
    VulkanContext& context_;
    VulkanPipeline texture_pipeline_;
    VulkanPipeline finish_pipeline_;
};

}  // namespace da3_native
