#include "encoder_cpu.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace da3_native {
namespace {

constexpr std::uint32_t kEmbedding = 384;
constexpr std::uint32_t kHeads = 6;
constexpr std::uint32_t kHead = 64;
constexpr const char* kPrefix = "model.backbone.pretrained.";

const TensorView& tensor(
    const SafeTensors& model,
    const std::string& name,
    std::uint32_t rank) {
    const TensorView& result = model.tensor(name);
    if (result.rank != rank) {
        throw std::runtime_error(
            "unexpected DA3 encoder tensor rank: " + name);
    }
    return result;
}

void linear(
    const std::vector<float>& input,
    std::uint32_t rows,
    std::uint32_t input_channels,
    const TensorView& weight,
    const TensorView& bias,
    std::vector<float>& output) {
    if (weight.rank != 2 || bias.rank != 1 ||
        weight.dimensions[1] != input_channels ||
        bias.dimensions[0] != weight.dimensions[0] ||
        input.size() != std::uint64_t(rows) * input_channels) {
        throw std::runtime_error("DA3 encoder linear shape mismatch");
    }
    const std::uint32_t output_channels =
        static_cast<std::uint32_t>(weight.dimensions[0]);
    output.resize(std::uint64_t(rows) * output_channels);
    for (std::uint32_t row = 0; row < rows; ++row) {
        const float* source =
            input.data() + std::uint64_t(row) * input_channels;
        float* destination =
            output.data() + std::uint64_t(row) * output_channels;
        for (std::uint32_t out = 0; out < output_channels; ++out) {
            const float* kernel =
                weight.data + std::uint64_t(out) * input_channels;
            float value = bias.data[out];
            for (std::uint32_t in = 0; in < input_channels; ++in) {
                value += source[in] * kernel[in];
            }
            destination[out] = value;
        }
    }
}

void layer_norm_rows(
    const std::vector<float>& input,
    std::uint32_t rows,
    std::uint32_t channels,
    const TensorView& scale,
    const TensorView& bias,
    float epsilon,
    std::vector<float>& output) {
    if (input.size() != std::uint64_t(rows) * channels ||
        scale.rank != 1 || bias.rank != 1 ||
        scale.dimensions[0] != channels ||
        bias.dimensions[0] != channels) {
        throw std::runtime_error("DA3 encoder layer norm mismatch");
    }
    output.resize(input.size());
    for (std::uint32_t row = 0; row < rows; ++row) {
        const float* source =
            input.data() + std::uint64_t(row) * channels;
        float* destination =
            output.data() + std::uint64_t(row) * channels;
        float mean = 0.0f;
        for (std::uint32_t channel = 0; channel < channels; ++channel) {
            mean += source[channel];
        }
        mean /= static_cast<float>(channels);
        float variance = 0.0f;
        for (std::uint32_t channel = 0; channel < channels; ++channel) {
            const float difference = source[channel] - mean;
            variance += difference * difference;
        }
        variance /= static_cast<float>(channels);
        const float inverse = 1.0f / std::sqrt(variance + epsilon);
        for (std::uint32_t channel = 0; channel < channels; ++channel) {
            destination[channel] =
                (source[channel] - mean) * inverse * scale.data[channel] +
                bias.data[channel];
        }
    }
}

float cubic(float distance) {
    constexpr float coefficient = -0.75f;
    distance = std::abs(distance);
    if (distance <= 1.0f) {
        return ((coefficient + 2.0f) * distance -
                (coefficient + 3.0f)) *
                distance * distance +
            1.0f;
    }
    if (distance < 2.0f) {
        return ((coefficient * distance -
                 5.0f * coefficient) *
                    distance +
                8.0f * coefficient) *
                distance -
            4.0f * coefficient;
    }
    return 0.0f;
}

void add_position(
    const SafeTensors& model,
    std::vector<float>& state,
    std::uint32_t patch_width,
    std::uint32_t patch_height) {
    const TensorView& position = tensor(
        model, std::string(kPrefix) + "pos_embed", 3);
    const std::uint32_t tokens =
        1 + patch_width * patch_height;
    for (std::uint32_t channel = 0;
         channel < kEmbedding;
         ++channel) {
        state[channel] += position.data[channel];
    }
    for (std::uint32_t y = 0; y < patch_height; ++y) {
        const float source_y =
            (static_cast<float>(y) + 0.5f) * 37.0f /
                (static_cast<float>(patch_height) + 0.1f) -
            0.5f;
        const int base_y = static_cast<int>(std::floor(source_y));
        for (std::uint32_t x = 0; x < patch_width; ++x) {
            const float source_x =
                (static_cast<float>(x) + 0.5f) * 37.0f /
                    (static_cast<float>(patch_width) + 0.1f) -
                0.5f;
            const int base_x = static_cast<int>(std::floor(source_x));
            float* destination = state.data() +
                (1 + std::uint64_t(y) * patch_width + x) * kEmbedding;
            for (std::uint32_t channel = 0;
                 channel < kEmbedding;
                 ++channel) {
                float value = 0.0f;
                for (int oy = -1; oy <= 2; ++oy) {
                    const int sy = std::clamp(base_y + oy, 0, 36);
                    const float wy = cubic(
                        source_y - static_cast<float>(base_y + oy));
                    for (int ox = -1; ox <= 2; ++ox) {
                        const int sx = std::clamp(base_x + ox, 0, 36);
                        const float wx = cubic(
                            source_x - static_cast<float>(base_x + ox));
                        value += wy * wx * position.data[
                            (1 + std::uint64_t(sy) * 37 + sx) *
                                kEmbedding +
                            channel];
                    }
                }
                destination[channel] += value;
            }
        }
    }
    (void)tokens;
}

enum class RopeMode {
    none,
    local,
    global,
};

void normalize_qk(
    std::vector<float>& qkv,
    std::uint32_t tokens,
    const TensorView& q_scale,
    const TensorView& q_bias,
    const TensorView& k_scale,
    const TensorView& k_bias) {
    std::vector<float> row(kHead);
    std::vector<float> normalized;
    for (std::uint32_t token_index = 0;
         token_index < tokens;
         ++token_index) {
        for (std::uint32_t q_or_k = 0; q_or_k < 2; ++q_or_k) {
            for (std::uint32_t head = 0; head < kHeads; ++head) {
                float* source = qkv.data() +
                    std::uint64_t(token_index) * 3 * kEmbedding +
                    q_or_k * kEmbedding + head * kHead;
                std::copy_n(source, kHead, row.data());
                layer_norm_rows(
                    row, 1, kHead,
                    q_or_k == 0 ? q_scale : k_scale,
                    q_or_k == 0 ? q_bias : k_bias,
                    1.0e-5f, normalized);
                std::copy_n(normalized.data(), kHead, source);
            }
        }
    }
}

void rope_half(
    float* values,
    std::uint32_t position) {
    constexpr std::uint32_t half = 32;
    constexpr std::uint32_t quarter = 16;
    float source[half];
    std::copy_n(values, half, source);
    for (std::uint32_t index = 0; index < quarter; ++index) {
        const float exponent =
            static_cast<float>(index * 2) / half;
        const float angle =
            static_cast<float>(position) /
            std::pow(100.0f, exponent);
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);
        values[index] =
            source[index] * cosine - source[quarter + index] * sine;
        values[quarter + index] =
            source[quarter + index] * cosine + source[index] * sine;
    }
}

void apply_rope(
    std::vector<float>& qkv,
    std::uint32_t patch_width,
    std::uint32_t patch_height,
    RopeMode mode) {
    if (mode == RopeMode::none) {
        return;
    }
    const std::uint32_t tokens =
        1 + patch_width * patch_height;
    for (std::uint32_t token_index = 0;
         token_index < tokens;
         ++token_index) {
        std::uint32_t y = 0;
        std::uint32_t x = 0;
        if (token_index != 0) {
            if (mode == RopeMode::global) {
                y = 1;
                x = 1;
            } else {
                const std::uint32_t patch = token_index - 1;
                y = patch / patch_width + 1;
                x = patch % patch_width + 1;
            }
        }
        for (std::uint32_t q_or_k = 0; q_or_k < 2; ++q_or_k) {
            for (std::uint32_t head = 0; head < kHeads; ++head) {
                float* values = qkv.data() +
                    std::uint64_t(token_index) * 3 * kEmbedding +
                    q_or_k * kEmbedding + head * kHead;
                rope_half(values, y);
                rope_half(values + 32, x);
            }
        }
    }
}

void attention(
    const SafeTensors& model,
    const std::string& prefix,
    const std::vector<float>& normalized,
    std::uint32_t patch_width,
    std::uint32_t patch_height,
    bool qk_norm,
    RopeMode rope,
    std::vector<float>& output) {
    const std::uint32_t tokens =
        1 + patch_width * patch_height;
    std::vector<float> qkv;
    linear(
        normalized, tokens, kEmbedding,
        tensor(model, prefix + "qkv.weight", 2),
        tensor(model, prefix + "qkv.bias", 1),
        qkv);
    if (qk_norm) {
        normalize_qk(
            qkv, tokens,
            tensor(model, prefix + "q_norm.weight", 1),
            tensor(model, prefix + "q_norm.bias", 1),
            tensor(model, prefix + "k_norm.weight", 1),
            tensor(model, prefix + "k_norm.bias", 1));
    }
    apply_rope(qkv, patch_width, patch_height, rope);

    std::vector<float> attended(
        std::uint64_t(tokens) * kEmbedding);
    std::vector<float> scores(tokens);
    constexpr float score_scale = 0.125f;
    for (std::uint32_t head = 0; head < kHeads; ++head) {
        for (std::uint32_t query_token = 0;
             query_token < tokens;
             ++query_token) {
            const float* query = qkv.data() +
                std::uint64_t(query_token) * 3 * kEmbedding +
                head * kHead;
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::uint32_t key_token = 0;
                 key_token < tokens;
                 ++key_token) {
                const float* key = qkv.data() +
                    std::uint64_t(key_token) * 3 * kEmbedding +
                    kEmbedding + head * kHead;
                float score = 0.0f;
                for (std::uint32_t channel = 0;
                     channel < kHead;
                     ++channel) {
                    score += query[channel] * key[channel];
                }
                score *= score_scale;
                scores[key_token] = score;
                maximum = std::max(maximum, score);
            }
            float denominator = 0.0f;
            for (float& score : scores) {
                score = std::exp(score - maximum);
                denominator += score;
            }
            float* destination = attended.data() +
                std::uint64_t(query_token) * kEmbedding +
                head * kHead;
            for (std::uint32_t channel = 0;
                 channel < kHead;
                 ++channel) {
                float value = 0.0f;
                for (std::uint32_t source_token = 0;
                     source_token < tokens;
                     ++source_token) {
                    const float* source = qkv.data() +
                        std::uint64_t(source_token) * 3 * kEmbedding +
                        2 * kEmbedding + head * kHead;
                    value += scores[source_token] / denominator *
                        source[channel];
                }
                destination[channel] = value;
            }
        }
    }
    linear(
        attended, tokens, kEmbedding,
        tensor(model, prefix + "proj.weight", 2),
        tensor(model, prefix + "proj.bias", 1),
        output);
}

void mlp(
    const SafeTensors& model,
    const std::string& prefix,
    const std::vector<float>& normalized,
    std::uint32_t rows,
    std::vector<float>& output) {
    std::vector<float> hidden;
    linear(
        normalized, rows, kEmbedding,
        tensor(model, prefix + "fc1.weight", 2),
        tensor(model, prefix + "fc1.bias", 1),
        hidden);
    constexpr float inverse_sqrt_two =
        0.70710678118654752440f;
    for (float& value : hidden) {
        value = 0.5f * value *
            (1.0f + std::erf(value * inverse_sqrt_two));
    }
    linear(
        hidden, rows, kEmbedding * 4,
        tensor(model, prefix + "fc2.weight", 2),
        tensor(model, prefix + "fc2.bias", 1),
        output);
}

}  // namespace

EncoderOutput encoder_single_view_cpu(
    const SafeTensors& model,
    const float* input,
    std::uint32_t width,
    std::uint32_t height) {
    if (!input || width == 0 || height == 0 ||
        width % 14 != 0 || height % 14 != 0) {
        throw std::invalid_argument("invalid DA3 encoder input");
    }
    const std::uint32_t patch_width = width / 14;
    const std::uint32_t patch_height = height / 14;
    const std::uint32_t patches = patch_width * patch_height;
    const std::uint32_t tokens = patches + 1;
    const TensorView& patch_weight = tensor(
        model, std::string(kPrefix) + "patch_embed.proj.weight", 4);
    const TensorView& patch_bias = tensor(
        model, std::string(kPrefix) + "patch_embed.proj.bias", 1);
    const TensorView& cls = tensor(
        model, std::string(kPrefix) + "cls_token", 3);
    std::vector<float> state(std::uint64_t(tokens) * kEmbedding);
    std::copy_n(cls.data, kEmbedding, state.data());
    for (std::uint32_t py = 0; py < patch_height; ++py) {
        for (std::uint32_t px = 0; px < patch_width; ++px) {
            float* destination = state.data() +
                (1 + std::uint64_t(py) * patch_width + px) *
                    kEmbedding;
            for (std::uint32_t out = 0; out < kEmbedding; ++out) {
                float value = patch_bias.data[out];
                for (std::uint32_t channel = 0; channel < 3; ++channel) {
                    for (std::uint32_t ky = 0; ky < 14; ++ky) {
                        for (std::uint32_t kx = 0; kx < 14; ++kx) {
                            value += input[
                                (std::uint64_t(channel) * height +
                                 py * 14 + ky) *
                                    width +
                                px * 14 + kx] *
                                patch_weight.data[
                                    ((std::uint64_t(out) * 3 + channel) *
                                         14 +
                                     ky) *
                                        14 +
                                    kx];
                        }
                    }
                }
                destination[out] = value;
            }
        }
    }
    add_position(model, state, patch_width, patch_height);

    EncoderOutput result;
    result.patch_width = patch_width;
    result.patch_height = patch_height;
    result.features.reserve(4);
    std::vector<float> local = state;
    const std::uint32_t captures[4] = {5, 7, 9, 11};
    std::uint32_t capture = 0;
    for (std::uint32_t block = 0; block < 12; ++block) {
        if (block == 4) {
            const TensorView& camera = tensor(
                model, std::string(kPrefix) + "camera_token", 3);
            std::copy_n(camera.data, kEmbedding, state.data());
        }
        const std::string prefix =
            std::string(kPrefix) + "blocks." +
            std::to_string(block) + ".";
        std::vector<float> normalized;
        layer_norm_rows(
            state, tokens, kEmbedding,
            tensor(model, prefix + "norm1.weight", 1),
            tensor(model, prefix + "norm1.bias", 1),
            1.0e-6f, normalized);
        std::vector<float> attended;
        const RopeMode rope = block < 4
            ? RopeMode::none
            : (block % 2 == 0 ? RopeMode::local : RopeMode::global);
        attention(
            model, prefix + "attn.", normalized,
            patch_width, patch_height,
            block >= 4, rope, attended);
        const TensorView& scale1 =
            tensor(model, prefix + "ls1.gamma", 1);
        for (std::uint32_t row = 0; row < tokens; ++row) {
            for (std::uint32_t channel = 0;
                 channel < kEmbedding;
                 ++channel) {
                state[std::uint64_t(row) * kEmbedding + channel] +=
                    attended[std::uint64_t(row) * kEmbedding + channel] *
                    scale1.data[channel];
            }
        }
        layer_norm_rows(
            state, tokens, kEmbedding,
            tensor(model, prefix + "norm2.weight", 1),
            tensor(model, prefix + "norm2.bias", 1),
            1.0e-6f, normalized);
        std::vector<float> feed;
        mlp(model, prefix + "mlp.", normalized, tokens, feed);
        const TensorView& scale2 =
            tensor(model, prefix + "ls2.gamma", 1);
        for (std::uint32_t row = 0; row < tokens; ++row) {
            for (std::uint32_t channel = 0;
                 channel < kEmbedding;
                 ++channel) {
                state[std::uint64_t(row) * kEmbedding + channel] +=
                    feed[std::uint64_t(row) * kEmbedding + channel] *
                    scale2.data[channel];
            }
        }
        if (block < 4 || block % 2 == 0) {
            local = state;
        }
        if (capture < 4 && block == captures[capture]) {
            std::vector<float> normalized_global;
            layer_norm_rows(
                state, tokens, kEmbedding,
                tensor(model, std::string(kPrefix) + "norm.weight", 1),
                tensor(model, std::string(kPrefix) + "norm.bias", 1),
                1.0e-5f, normalized_global);
            std::vector<float> feature(
                std::uint64_t(patches) * kEmbedding * 2);
            for (std::uint32_t patch = 0; patch < patches; ++patch) {
                std::copy_n(
                    local.data() +
                        std::uint64_t(patch + 1) * kEmbedding,
                    kEmbedding,
                    feature.data() +
                        std::uint64_t(patch) * kEmbedding * 2);
                std::copy_n(
                    normalized_global.data() +
                        std::uint64_t(patch + 1) * kEmbedding,
                    kEmbedding,
                    feature.data() +
                        std::uint64_t(patch) * kEmbedding * 2 +
                        kEmbedding);
            }
            result.features.push_back(std::move(feature));
            ++capture;
        }
    }
    return result;
}

}  // namespace da3_native
