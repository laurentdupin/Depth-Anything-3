#include "encoder_gpu.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace da3_native {
namespace {
constexpr std::uint32_t kEmbedding = 384;
constexpr std::uint32_t kHeads = 6;
constexpr const char* kPrefix = "model.backbone.pretrained.";

const VulkanBuffer& weight(const GpuModel& model, const std::string& name) {
    return model.tensor(name).buffer;
}

std::string block_name(std::uint32_t block, const char* suffix) {
    return std::string(kPrefix) + "blocks." + std::to_string(block) + suffix;
}
}

GpuEncoderOutput encoder_single_view_gpu(
    VulkanContext& context, GpuModel& model, VulkanOperators& operators,
    const VulkanBuffer& image, std::uint32_t width, std::uint32_t height) {
    if (width == 0 || height == 0 || width % 14 != 0 || height % 14 != 0) {
        throw std::invalid_argument("invalid DA3 GPU encoder input");
    }
    const std::uint32_t patch_width = width / 14;
    const std::uint32_t patch_height = height / 14;
    const std::uint32_t patches = patch_width * patch_height;
    const std::uint32_t tokens = patches + 1;
    const std::uint64_t elements = std::uint64_t(tokens) * kEmbedding;
    const VkDeviceSize bytes = elements * sizeof(float);
    VulkanBuffer state = context.create_device_buffer(bytes);
    VulkanBuffer next = context.create_device_buffer(bytes);
    VulkanBuffer normalized = context.create_device_buffer(bytes);
    VulkanBuffer attended = context.create_device_buffer(bytes);
    VulkanBuffer projected = context.create_device_buffer(bytes);
    VulkanBuffer qkv = context.create_device_buffer(bytes * 3);
    VulkanBuffer hidden = context.create_device_buffer(bytes * 4);
    VulkanBuffer local = context.create_device_buffer(bytes);
    VulkanBuffer scores = context.create_device_buffer(
        std::uint64_t(kHeads) * tokens * tokens * sizeof(float));

    context.batch([&] {
        operators.prepare_tokens(
            state, image,
            weight(model, std::string(kPrefix) + "patch_embed.proj.weight"),
            weight(model, std::string(kPrefix) + "patch_embed.proj.bias"),
            weight(model, std::string(kPrefix) + "cls_token"),
            weight(model, std::string(kPrefix) + "pos_embed"),
            width, height, kEmbedding);
    });
    context.copy(local, 0, state, 0, bytes);

    GpuEncoderOutput result;
    result.patch_width = patch_width;
    result.patch_height = patch_height;
    result.features.reserve(4);
    const std::uint32_t captures[4] = {5, 7, 9, 11};
    std::uint32_t capture = 0;
    for (std::uint32_t block = 0; block < 12; ++block) {
        context.batch([&] {
            if (block == 4) {
                operators.replace_token(
                    state,
                    weight(model, std::string(kPrefix) + "camera_token"),
                    kEmbedding);
            }
            operators.layer_norm(
                normalized, state,
                weight(model, block_name(block, ".norm1.weight")),
                weight(model, block_name(block, ".norm1.bias")),
                tokens, kEmbedding, 1.0e-6f);
            operators.linear(
                qkv, normalized,
                weight(model, block_name(block, ".attn.qkv.weight")),
                weight(model, block_name(block, ".attn.qkv.bias")),
                tokens, kEmbedding, kEmbedding * 3, false);
            if (block >= 4) {
                operators.qk_norm_rope(
                    qkv,
                    weight(model, block_name(block, ".attn.q_norm.weight")),
                    weight(model, block_name(block, ".attn.q_norm.bias")),
                    weight(model, block_name(block, ".attn.k_norm.weight")),
                    weight(model, block_name(block, ".attn.k_norm.bias")),
                    tokens, kHeads, patch_width,
                    block % 2 == 0 ? 1u : 2u);
            }
            operators.attention_head64(
                attended, qkv, tokens, kHeads, &scores);
            operators.linear(
                projected, attended,
                weight(model, block_name(block, ".attn.proj.weight")),
                weight(model, block_name(block, ".attn.proj.bias")),
                tokens, kEmbedding, kEmbedding, false);
            operators.add_scaled(
                next, state, projected,
                weight(model, block_name(block, ".ls1.gamma")),
                static_cast<std::uint32_t>(elements), kEmbedding);
            std::swap(state, next);
            operators.layer_norm(
                normalized, state,
                weight(model, block_name(block, ".norm2.weight")),
                weight(model, block_name(block, ".norm2.bias")),
                tokens, kEmbedding, 1.0e-6f);
            operators.linear(
                hidden, normalized,
                weight(model, block_name(block, ".mlp.fc1.weight")),
                weight(model, block_name(block, ".mlp.fc1.bias")),
                tokens, kEmbedding, kEmbedding * 4, true);
            operators.linear(
                projected, hidden,
                weight(model, block_name(block, ".mlp.fc2.weight")),
                weight(model, block_name(block, ".mlp.fc2.bias")),
                tokens, kEmbedding * 4, kEmbedding, false);
            operators.add_scaled(
                next, state, projected,
                weight(model, block_name(block, ".ls2.gamma")),
                static_cast<std::uint32_t>(elements), kEmbedding);
            std::swap(state, next);
        });
        if (block < 4 || block % 2 == 0) {
            context.copy(local, 0, state, 0, bytes);
        }
        if (capture < 4 && block == captures[capture]) {
            VulkanBuffer global = context.create_device_buffer(bytes);
            context.batch([&] {
                operators.layer_norm(
                    global, state,
                    weight(model, std::string(kPrefix) + "norm.weight"),
                    weight(model, std::string(kPrefix) + "norm.bias"),
                    tokens, kEmbedding, 1.0e-5f);
                VulkanBuffer feature = context.create_device_buffer(
                    std::uint64_t(patches) * kEmbedding * 2 * sizeof(float));
                operators.capture_concat(
                    feature, local, global, patches, kEmbedding);
                result.features.push_back(std::move(feature));
                ++capture;
            });
        }
    }
    if (result.features.size() != 4) {
        throw std::runtime_error("DA3 GPU encoder did not capture four features");
    }
    return result;
}

}  // namespace da3_native
