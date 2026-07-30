#pragma once

#include "encoder_gpu.h"

#include <string>

namespace da3_native {

struct GpuFeatureMap {
    VulkanBuffer buffer;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t channels = 0;
};

GpuFeatureMap depth_head_single_view_gpu(
    VulkanContext& context, GpuModel& model, VulkanOperators& operators,
    GpuEncoderOutput&& encoded);

}  // namespace da3_native
