#pragma once

#include "gpu_model.h"
#include "operators.h"

#include <cstdint>
#include <vector>

namespace da3_native {

struct GpuEncoderOutput {
    std::uint32_t patch_width = 0;
    std::uint32_t patch_height = 0;
    std::vector<VulkanBuffer> features;
};

GpuEncoderOutput encoder_single_view_gpu(
    VulkanContext& context, GpuModel& model, VulkanOperators& operators,
    const VulkanBuffer& image, std::uint32_t width, std::uint32_t height);

}  // namespace da3_native
