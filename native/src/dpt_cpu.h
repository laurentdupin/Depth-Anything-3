#pragma once

#include "encoder_cpu.h"
#include "safetensors.h"

#include <vector>

namespace da3_native {

std::vector<float> depth_head_single_view_cpu(
    const SafeTensors& model,
    EncoderOutput&& encoded);

}  // namespace da3_native
