#include "dpt_cpu.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace da3_native {
namespace {

struct Feature {
    std::uint32_t channels = 0;
    std::uint32_t height = 0;
    std::uint32_t width = 0;
    std::vector<float> data;
};

std::uint64_t count(const Feature& feature) {
    return std::uint64_t(feature.channels) *
        feature.height * feature.width;
}

const TensorView& tensor(
    const SafeTensors& model,
    const std::string& name,
    std::uint32_t rank) {
    const TensorView& result = model.tensor(name);
    if (result.rank != rank) {
        throw std::runtime_error(
            "unexpected DA3 DPT tensor rank: " + name);
    }
    return result;
}

void layer_norm(
    std::vector<float>& values,
    std::uint32_t rows,
    std::uint32_t channels,
    const TensorView& scale,
    const TensorView& bias) {
    if (values.size() != std::uint64_t(rows) * channels ||
        scale.dimensions[0] != channels ||
        bias.dimensions[0] != channels) {
        throw std::runtime_error("DA3 DPT token norm mismatch");
    }
    for (std::uint32_t row = 0; row < rows; ++row) {
        float* value =
            values.data() + std::uint64_t(row) * channels;
        float mean = 0.0f;
        for (std::uint32_t channel = 0; channel < channels; ++channel) {
            mean += value[channel];
        }
        mean /= static_cast<float>(channels);
        float variance = 0.0f;
        for (std::uint32_t channel = 0; channel < channels; ++channel) {
            const float difference = value[channel] - mean;
            variance += difference * difference;
        }
        variance /= static_cast<float>(channels);
        const float inverse =
            1.0f / std::sqrt(variance + 1.0e-5f);
        for (std::uint32_t channel = 0; channel < channels; ++channel) {
            value[channel] =
                (value[channel] - mean) * inverse * scale.data[channel] +
                bias.data[channel];
        }
    }
}

Feature conv(
    const SafeTensors& model,
    const Feature& input,
    const std::string& weight_name,
    const std::string& bias_name,
    std::uint32_t stride,
    std::uint32_t padding) {
    const TensorView& weight = tensor(model, weight_name, 4);
    const TensorView* bias = bias_name.empty()
        ? nullptr
        : &tensor(model, bias_name, 1);
    const std::uint32_t output_channels =
        static_cast<std::uint32_t>(weight.dimensions[0]);
    const std::uint32_t kernel =
        static_cast<std::uint32_t>(weight.dimensions[2]);
    if (weight.dimensions[1] != input.channels ||
        weight.dimensions[3] != kernel ||
        (bias && bias->dimensions[0] != output_channels)) {
        throw std::runtime_error("DA3 DPT convolution mismatch");
    }
    Feature output{
        output_channels,
        (input.height + 2 * padding - kernel) / stride + 1,
        (input.width + 2 * padding - kernel) / stride + 1,
        {},
    };
    output.data.resize(count(output));
    for (std::uint32_t out = 0; out < output.channels; ++out) {
        for (std::uint32_t y = 0; y < output.height; ++y) {
            for (std::uint32_t x = 0; x < output.width; ++x) {
                float value = bias ? bias->data[out] : 0.0f;
                for (std::uint32_t in = 0; in < input.channels; ++in) {
                    for (std::uint32_t ky = 0; ky < kernel; ++ky) {
                        const int sy =
                            static_cast<int>(y * stride + ky) -
                            static_cast<int>(padding);
                        if (sy < 0 || sy >= static_cast<int>(input.height)) {
                            continue;
                        }
                        for (std::uint32_t kx = 0; kx < kernel; ++kx) {
                            const int sx =
                                static_cast<int>(x * stride + kx) -
                                static_cast<int>(padding);
                            if (sx < 0 ||
                                sx >= static_cast<int>(input.width)) {
                                continue;
                            }
                            value += input.data[
                                (std::uint64_t(in) * input.height + sy) *
                                    input.width +
                                sx] *
                                weight.data[
                                    ((std::uint64_t(out) * input.channels +
                                      in) *
                                         kernel +
                                     ky) *
                                        kernel +
                                    kx];
                        }
                    }
                }
                output.data[
                    (std::uint64_t(out) * output.height + y) *
                        output.width +
                    x] = value;
            }
        }
    }
    return output;
}

Feature transpose_conv(
    const SafeTensors& model,
    const Feature& input,
    const std::string& prefix,
    std::uint32_t stride) {
    const TensorView& weight =
        tensor(model, prefix + ".weight", 4);
    const TensorView& bias =
        tensor(model, prefix + ".bias", 1);
    if (weight.dimensions[0] != input.channels ||
        weight.dimensions[2] != stride ||
        weight.dimensions[3] != stride) {
        throw std::runtime_error("DA3 DPT transpose convolution mismatch");
    }
    Feature output{
        static_cast<std::uint32_t>(weight.dimensions[1]),
        input.height * stride,
        input.width * stride,
        {},
    };
    output.data.resize(count(output));
    for (std::uint32_t out = 0; out < output.channels; ++out) {
        for (std::uint32_t position = 0;
             position < output.height * output.width;
             ++position) {
            output.data[
                std::uint64_t(out) * output.height * output.width +
                position] = bias.data[out];
        }
    }
    for (std::uint32_t in = 0; in < input.channels; ++in) {
        for (std::uint32_t y = 0; y < input.height; ++y) {
            for (std::uint32_t x = 0; x < input.width; ++x) {
                const float source = input.data[
                    (std::uint64_t(in) * input.height + y) *
                        input.width +
                    x];
                for (std::uint32_t out = 0;
                     out < output.channels;
                     ++out) {
                    for (std::uint32_t ky = 0; ky < stride; ++ky) {
                        for (std::uint32_t kx = 0; kx < stride; ++kx) {
                            output.data[
                                (std::uint64_t(out) * output.height +
                                 y * stride + ky) *
                                    output.width +
                                x * stride + kx] +=
                                source * weight.data[
                                    ((std::uint64_t(in) *
                                          output.channels +
                                      out) *
                                         stride +
                                     ky) *
                                        stride +
                                    kx];
                        }
                    }
                }
            }
        }
    }
    return output;
}

Feature bilinear(
    const Feature& input,
    std::uint32_t height,
    std::uint32_t width) {
    Feature output{input.channels, height, width, {}};
    output.data.resize(count(output));
    for (std::uint32_t channel = 0;
         channel < input.channels;
         ++channel) {
        for (std::uint32_t y = 0; y < height; ++y) {
            const float source_y = height == 1
                ? 0.0f
                : static_cast<float>(y) *
                    (input.height - 1) / (height - 1);
            const std::uint32_t y0 =
                static_cast<std::uint32_t>(source_y);
            const std::uint32_t y1 =
                std::min(y0 + 1, input.height - 1);
            const float fy = source_y - y0;
            for (std::uint32_t x = 0; x < width; ++x) {
                const float source_x = width == 1
                    ? 0.0f
                    : static_cast<float>(x) *
                        (input.width - 1) / (width - 1);
                const std::uint32_t x0 =
                    static_cast<std::uint32_t>(source_x);
                const std::uint32_t x1 =
                    std::min(x0 + 1, input.width - 1);
                const float fx = source_x - x0;
                const auto at = [&](std::uint32_t sy,
                                    std::uint32_t sx) {
                    return input.data[
                        (std::uint64_t(channel) * input.height + sy) *
                            input.width +
                        sx];
                };
                output.data[
                    (std::uint64_t(channel) * height + y) * width + x] =
                    (at(y0, x0) * (1.0f - fx) + at(y0, x1) * fx) *
                        (1.0f - fy) +
                    (at(y1, x0) * (1.0f - fx) + at(y1, x1) * fx) * fy;
            }
        }
    }
    return output;
}

void relu(Feature& value) {
    for (float& element : value.data) {
        element = std::max(element, 0.0f);
    }
}

void add_uv_position(
    Feature& value,
    std::uint32_t image_width,
    std::uint32_t image_height) {
    if (value.channels % 4 != 0) {
        throw std::runtime_error("DA3 UV embedding channel mismatch");
    }
    const float aspect =
        static_cast<float>(image_width) / image_height;
    const float diagonal = std::sqrt(aspect * aspect + 1.0f);
    const float span_x = aspect / diagonal;
    const float span_y = 1.0f / diagonal;
    const float left =
        -span_x * (value.width - 1) / value.width;
    const float right =
        span_x * (value.width - 1) / value.width;
    const float top =
        -span_y * (value.height - 1) / value.height;
    const float bottom =
        span_y * (value.height - 1) / value.height;
    const std::uint32_t direction_dim = value.channels / 2;
    const std::uint32_t frequency_count = direction_dim / 2;
    for (std::uint32_t y = 0; y < value.height; ++y) {
        const float v = value.height == 1
            ? top
            : top + (bottom - top) * y / (value.height - 1);
        for (std::uint32_t x = 0; x < value.width; ++x) {
            const float u = value.width == 1
                ? left
                : left + (right - left) * x / (value.width - 1);
            const float coordinates[2] = {u, v};
            for (std::uint32_t direction = 0;
                 direction < 2;
                 ++direction) {
                for (std::uint32_t frequency = 0;
                     frequency < frequency_count;
                     ++frequency) {
                    const float omega =
                        1.0f / std::pow(
                            100.0f,
                            static_cast<float>(frequency) /
                                frequency_count);
                    const float angle =
                        coordinates[direction] * omega;
                    const std::uint32_t base =
                        direction * direction_dim;
                    value.data[
                        (std::uint64_t(base + frequency) *
                             value.height +
                         y) *
                            value.width +
                        x] += 0.1f * std::sin(angle);
                    value.data[
                        (std::uint64_t(
                             base + frequency_count + frequency) *
                             value.height +
                         y) *
                            value.width +
                        x] += 0.1f * std::cos(angle);
                }
            }
        }
    }
}

Feature residual(
    const SafeTensors& model,
    const Feature& input,
    const std::string& prefix) {
    Feature result = input;
    relu(result);
    result = conv(
        model, result, prefix + ".conv1.weight",
        prefix + ".conv1.bias", 1, 1);
    relu(result);
    result = conv(
        model, result, prefix + ".conv2.weight",
        prefix + ".conv2.bias", 1, 1);
    for (std::size_t index = 0; index < result.data.size(); ++index) {
        result.data[index] += input.data[index];
    }
    return result;
}

Feature fusion(
    const SafeTensors& model,
    Feature path,
    const Feature* skip,
    const std::string& prefix,
    std::uint32_t height,
    std::uint32_t width) {
    if (skip) {
        Feature processed = residual(
            model, *skip, prefix + ".resConfUnit1");
        for (std::size_t index = 0; index < path.data.size(); ++index) {
            path.data[index] += processed.data[index];
        }
    }
    path = residual(model, path, prefix + ".resConfUnit2");
    path = bilinear(path, height, width);
    return conv(
        model, path, prefix + ".out_conv.weight",
        prefix + ".out_conv.bias", 1, 0);
}

}  // namespace

std::vector<float> depth_head_single_view_cpu(
    const SafeTensors& model,
    EncoderOutput&& encoded) {
    if (encoded.features.size() != 4 ||
        encoded.patch_width == 0 || encoded.patch_height == 0) {
        throw std::invalid_argument("invalid DA3 DPT encoder output");
    }
    const std::uint32_t patches =
        encoded.patch_width * encoded.patch_height;
    const std::uint32_t image_width = encoded.patch_width * 14;
    const std::uint32_t image_height = encoded.patch_height * 14;
    Feature layers[4];
    for (std::uint32_t index = 0; index < 4; ++index) {
        std::vector<float>& tokens = encoded.features[index];
        layer_norm(
            tokens, patches, 768,
            tensor(model, "model.head.norm.weight", 1),
            tensor(model, "model.head.norm.bias", 1));
        Feature spatial{
            768, encoded.patch_height, encoded.patch_width,
            std::vector<float>(std::uint64_t(768) * patches)};
        for (std::uint32_t position = 0;
             position < patches;
             ++position) {
            for (std::uint32_t channel = 0; channel < 768; ++channel) {
                spatial.data[
                    std::uint64_t(channel) * patches + position] =
                    tokens[std::uint64_t(position) * 768 + channel];
            }
        }
        const std::string project =
            "model.head.projects." + std::to_string(index);
        layers[index] = conv(
            model, spatial, project + ".weight",
            project + ".bias", 1, 0);
        add_uv_position(layers[index], image_width, image_height);
        if (index == 0) {
            layers[index] = transpose_conv(
                model, layers[index],
                "model.head.resize_layers.0", 4);
        } else if (index == 1) {
            layers[index] = transpose_conv(
                model, layers[index],
                "model.head.resize_layers.1", 2);
        } else if (index == 3) {
            layers[index] = conv(
                model, layers[index],
                "model.head.resize_layers.3.weight",
                "model.head.resize_layers.3.bias", 2, 1);
        }
    }
    Feature refined[4];
    for (std::uint32_t index = 0; index < 4; ++index) {
        refined[index] = conv(
            model, layers[index],
            "model.head.scratch.layer" + std::to_string(index + 1) +
                "_rn.weight",
            "", 1, 1);
    }
    Feature path = fusion(
        model, std::move(refined[3]), nullptr,
        "model.head.scratch.refinenet4",
        refined[2].height, refined[2].width);
    path = fusion(
        model, std::move(path), &refined[2],
        "model.head.scratch.refinenet3",
        refined[1].height, refined[1].width);
    path = fusion(
        model, std::move(path), &refined[1],
        "model.head.scratch.refinenet2",
        refined[0].height, refined[0].width);
    path = fusion(
        model, std::move(path), &refined[0],
        "model.head.scratch.refinenet1",
        refined[0].height * 2, refined[0].width * 2);
    path = conv(
        model, path,
        "model.head.scratch.output_conv1.weight",
        "model.head.scratch.output_conv1.bias", 1, 1);
    path = bilinear(path, image_height, image_width);
    add_uv_position(path, image_width, image_height);
    path = conv(
        model, path,
        "model.head.scratch.output_conv2.0.weight",
        "model.head.scratch.output_conv2.0.bias", 1, 1);
    relu(path);
    path = conv(
        model, path,
        "model.head.scratch.output_conv2.2.weight",
        "model.head.scratch.output_conv2.2.bias", 1, 0);
    std::vector<float> depth(
        std::uint64_t(image_width) * image_height);
    for (std::size_t index = 0; index < depth.size(); ++index) {
        depth[index] = std::exp(path.data[index]);
    }
    return depth;
}

}  // namespace da3_native
