#include "image.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace da3_native {
namespace {

struct ByteImage {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<std::uint8_t> values;
};

std::uint8_t rounded_byte(double value) {
    return static_cast<std::uint8_t>(std::clamp(
        static_cast<int>(std::nearbyint(value)), 0, 255));
}

void cubic_coefficients(float x, float coefficients[4]) {
    constexpr float a = -0.75f;
    coefficients[0] =
        ((a * (x + 1.0f) - 5.0f * a) * (x + 1.0f) + 8.0f * a) *
            (x + 1.0f) -
        4.0f * a;
    coefficients[1] =
        ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
    const float opposite = 1.0f - x;
    coefficients[2] =
        ((a + 2.0f) * opposite - (a + 3.0f)) *
            opposite * opposite +
        1.0f;
    coefficients[3] =
        1.0f - coefficients[0] - coefficients[1] - coefficients[2];
}

ByteImage load_bgr_as_rgb(
    const std::uint8_t* bgra,
    std::size_t stride,
    std::uint32_t width,
    std::uint32_t height) {
    ByteImage result{
        width, height,
        std::vector<std::uint8_t>(
            static_cast<std::size_t>(
                std::uint64_t(width) * height * 3))};
    for (std::uint32_t y = 0; y < height; ++y) {
        const std::uint8_t* source =
            bgra + static_cast<std::size_t>(y) * stride;
        std::uint8_t* destination =
            result.values.data() +
            static_cast<std::size_t>(y) * width * 3;
        for (std::uint32_t x = 0; x < width; ++x) {
            // InferBridge passes the first three BGRA bytes as a numpy image.
            // DA3's InputProcessor interprets those bytes as RGB.
            destination[3 * x] = source[4 * x];
            destination[3 * x + 1] = source[4 * x + 1];
            destination[3 * x + 2] = source[4 * x + 2];
        }
    }
    return result;
}

ByteImage resize_cubic(
    const ByteImage& source,
    std::uint32_t width,
    std::uint32_t height) {
    ByteImage result{
        width, height,
        std::vector<std::uint8_t>(
            static_cast<std::size_t>(
                std::uint64_t(width) * height * 3))};
    const double scale_x =
        static_cast<double>(source.width) / width;
    const double scale_y =
        static_cast<double>(source.height) / height;
    for (std::uint32_t y = 0; y < height; ++y) {
        const float coordinate_y =
            static_cast<float>((y + 0.5) * scale_y - 0.5);
        const int base_y =
            static_cast<int>(std::floor(coordinate_y));
        float beta[4];
        cubic_coefficients(coordinate_y - base_y, beta);
        for (std::uint32_t x = 0; x < width; ++x) {
            const float coordinate_x =
                static_cast<float>((x + 0.5) * scale_x - 0.5);
            const int base_x =
                static_cast<int>(std::floor(coordinate_x));
            float alpha[4];
            cubic_coefficients(coordinate_x - base_x, alpha);
            for (std::uint32_t channel = 0; channel < 3; ++channel) {
                double value = 0.0;
                for (int ky = 0; ky < 4; ++ky) {
                    const int sy = std::clamp(
                        base_y - 1 + ky, 0,
                        static_cast<int>(source.height) - 1);
                    double horizontal = 0.0;
                    for (int kx = 0; kx < 4; ++kx) {
                        const int sx = std::clamp(
                            base_x - 1 + kx, 0,
                            static_cast<int>(source.width) - 1);
                        horizontal +=
                            source.values[(
                                std::uint64_t(sy) * source.width + sx) *
                                3 + channel] *
                            alpha[kx];
                    }
                    value += horizontal * beta[ky];
                }
                result.values[(
                    std::uint64_t(y) * width + x) * 3 + channel] =
                    rounded_byte(value);
            }
        }
    }
    return result;
}

ByteImage resize_area(
    const ByteImage& source,
    std::uint32_t width,
    std::uint32_t height) {
    ByteImage result{
        width, height,
        std::vector<std::uint8_t>(
            static_cast<std::size_t>(
                std::uint64_t(width) * height * 3))};
    const double scale_x =
        static_cast<double>(source.width) / width;
    const double scale_y =
        static_cast<double>(source.height) / height;
    const double area = scale_x * scale_y;
    for (std::uint32_t y = 0; y < height; ++y) {
        const double y_begin = y * scale_y;
        const double y_end = (y + 1) * scale_y;
        const int first_y = static_cast<int>(std::floor(y_begin));
        const int last_y = static_cast<int>(std::ceil(y_end));
        for (std::uint32_t x = 0; x < width; ++x) {
            const double x_begin = x * scale_x;
            const double x_end = (x + 1) * scale_x;
            const int first_x = static_cast<int>(std::floor(x_begin));
            const int last_x = static_cast<int>(std::ceil(x_end));
            for (std::uint32_t channel = 0; channel < 3; ++channel) {
                double sum = 0.0;
                for (int sy = first_y; sy < last_y; ++sy) {
                    if (sy < 0 ||
                        sy >= static_cast<int>(source.height)) {
                        continue;
                    }
                    const double wy = std::max(
                        0.0, std::min(y_end, sy + 1.0) -
                            std::max(y_begin, static_cast<double>(sy)));
                    for (int sx = first_x; sx < last_x; ++sx) {
                        if (sx < 0 ||
                            sx >= static_cast<int>(source.width)) {
                            continue;
                        }
                        const double wx = std::max(
                            0.0, std::min(x_end, sx + 1.0) -
                                std::max(
                                    x_begin, static_cast<double>(sx)));
                        sum += source.values[(
                            std::uint64_t(sy) * source.width + sx) *
                            3 + channel] * wx * wy;
                    }
                }
                result.values[(
                    std::uint64_t(y) * width + x) * 3 + channel] =
                    rounded_byte(sum / area);
            }
        }
    }
    return result;
}

ByteImage resize(
    const ByteImage& source,
    std::uint32_t width,
    std::uint32_t height) {
    if (source.width == width && source.height == height) {
        return source;
    }
    const bool upscale =
        width > source.width || height > source.height;
    return upscale
        ? resize_cubic(source, width, height)
        : resize_area(source, width, height);
}

std::uint32_t nearest_patch_multiple(std::uint32_t value) {
    constexpr std::uint32_t patch = 14;
    const std::uint32_t down = value / patch * patch;
    const std::uint32_t up = down + patch;
    return std::max(
        1u, up - value <= value - down ? up : down);
}

}  // namespace

ImageShape inferbridge_image_shape(
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t process_resolution) {
    if (width == 0 || height == 0 || process_resolution == 0) {
        throw std::invalid_argument("invalid DA3 image dimensions");
    }
    const std::uint32_t longest = std::max(width, height);
    const double scale =
        static_cast<double>(process_resolution) / longest;
    const std::uint32_t resized_width = std::max(
        1, static_cast<int>(std::nearbyint(width * scale)));
    const std::uint32_t resized_height = std::max(
        1, static_cast<int>(std::nearbyint(height * scale)));
    return {
        nearest_patch_multiple(resized_width),
        nearest_patch_multiple(resized_height)};
}

std::vector<float> preprocess_inferbridge_bgra8(
    const std::uint8_t* bgra,
    std::size_t stride,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t process_resolution) {
    if (!bgra || stride < std::uint64_t(width) * 4) {
        throw std::invalid_argument("invalid DA3 BGRA image");
    }
    const std::uint32_t longest = std::max(width, height);
    const double scale =
        static_cast<double>(process_resolution) / longest;
    const std::uint32_t first_width = std::max(
        1, static_cast<int>(std::nearbyint(width * scale)));
    const std::uint32_t first_height = std::max(
        1, static_cast<int>(std::nearbyint(height * scale)));
    const ImageShape destination =
        inferbridge_image_shape(width, height, process_resolution);
    ByteImage image =
        load_bgr_as_rgb(bgra, stride, width, height);
    image = resize(image, first_width, first_height);
    image = resize(image, destination.width, destination.height);

    const std::uint64_t plane =
        std::uint64_t(destination.width) * destination.height;
    std::vector<float> output(
        static_cast<std::size_t>(3 * plane));
    constexpr float mean[3] = {0.485f, 0.456f, 0.406f};
    constexpr float deviation[3] = {0.229f, 0.224f, 0.225f};
    for (std::uint64_t pixel = 0; pixel < plane; ++pixel) {
        for (std::uint32_t channel = 0; channel < 3; ++channel) {
            output[static_cast<std::size_t>(
                std::uint64_t(channel) * plane + pixel)] =
                (image.values[static_cast<std::size_t>(
                    pixel * 3 + channel)] / 255.0f - mean[channel]) /
                deviation[channel];
        }
    }
    return output;
}

}  // namespace da3_native
