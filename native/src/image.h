#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace da3_native {

struct ImageShape {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

ImageShape inferbridge_image_shape(
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t process_resolution);

std::vector<float> preprocess_inferbridge_bgra8(
    const std::uint8_t* bgra,
    std::size_t stride,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t process_resolution);

}  // namespace da3_native
