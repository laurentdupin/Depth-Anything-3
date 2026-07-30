#pragma once

#include "safetensors.h"

#include <cstdint>
#include <vector>

namespace da3_native {

struct EncoderOutput {
    std::uint32_t patch_width = 0;
    std::uint32_t patch_height = 0;
    std::vector<std::vector<float>> features;
};

EncoderOutput encoder_single_view_cpu(
    const SafeTensors& model,
    const float* normalized_rgb_chw,
    std::uint32_t width,
    std::uint32_t height);

}  // namespace da3_native
