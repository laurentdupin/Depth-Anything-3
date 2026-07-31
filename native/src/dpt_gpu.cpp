#include "dpt_gpu.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace da3_native {
namespace {
constexpr std::uint32_t kFeatures = 64;

std::uint64_t elements(const GpuFeatureMap& value) {
    return std::uint64_t(value.width) * value.height * value.channels;
}

const VulkanBuffer& weight(const GpuModel& model, const std::string& name) {
    return model.tensor(name).buffer;
}

GpuFeatureMap conv(
    VulkanContext& context, GpuModel& model, VulkanOperators& operators,
    GpuFeatureMap&& input, const std::string& weight_name,
    const std::string& bias_name, std::uint32_t output_channels,
    std::uint32_t kernel, std::uint32_t stride,
    std::uint32_t padding, bool has_bias,
    const VulkanBuffer& zero_bias) {
    GpuFeatureMap output{
        context.create_device_buffer(
            std::uint64_t(
                (input.width + 2 * padding - kernel) / stride + 1) *
                ((input.height + 2 * padding - kernel) / stride + 1) *
                output_channels * sizeof(float)),
        (input.width + 2 * padding - kernel) / stride + 1,
        (input.height + 2 * padding - kernel) / stride + 1,
        output_channels,
    };
    operators.conv2d(
        output.buffer, input.buffer, weight(model, weight_name),
        has_bias ? weight(model, bias_name) : zero_bias,
        input.width, input.height, input.channels, output_channels,
        kernel, stride, padding, has_bias);
    return output;
}

GpuFeatureMap residual(
    VulkanContext& context, GpuModel& model, VulkanOperators& operators,
    GpuFeatureMap&& input, const std::string& prefix,
    const VulkanBuffer& zero_bias) {
    GpuFeatureMap activated{
        context.create_device_buffer(elements(input) * sizeof(float)),
        input.width, input.height, input.channels};
    operators.relu(
        activated.buffer, input.buffer,
        static_cast<std::uint32_t>(elements(input)));
    GpuFeatureMap first = conv(
        context, model, operators, std::move(activated),
        prefix + ".conv1.weight", prefix + ".conv1.bias",
        input.channels, 3, 1, 1, true, zero_bias);
    operators.relu(
        first.buffer, first.buffer,
        static_cast<std::uint32_t>(elements(first)));
    GpuFeatureMap second = conv(
        context, model, operators, std::move(first),
        prefix + ".conv2.weight", prefix + ".conv2.bias",
        input.channels, 3, 1, 1, true, zero_bias);
    operators.add(
        second.buffer, second.buffer, input.buffer,
        static_cast<std::uint32_t>(elements(second)));
    return second;
}

GpuFeatureMap fusion(
    VulkanContext& context, GpuModel& model, VulkanOperators& operators,
    GpuFeatureMap&& path, GpuFeatureMap&& skip,
    const std::string& prefix, std::uint32_t output_width,
    std::uint32_t output_height, const VulkanBuffer& zero_bias) {
    if (skip.buffer.handle() != VK_NULL_HANDLE) {
        GpuFeatureMap processed = residual(
            context, model, operators, std::move(skip),
            prefix + ".resConfUnit1", zero_bias);
        operators.add(
            path.buffer, path.buffer, processed.buffer,
            static_cast<std::uint32_t>(elements(path)));
    }
    path = residual(
        context, model, operators, std::move(path),
        prefix + ".resConfUnit2", zero_bias);
    GpuFeatureMap resized{
        context.create_device_buffer(
            std::uint64_t(output_width) * output_height *
            path.channels * sizeof(float)),
        output_width, output_height, path.channels};
    operators.bilinear_align_true(
        resized.buffer, path.buffer, path.width, path.height,
        output_width, output_height, path.channels);
    return conv(
        context, model, operators, std::move(resized),
        prefix + ".out_conv.weight", prefix + ".out_conv.bias",
        kFeatures, 1, 1, 0, true, zero_bias);
}
}

GpuFeatureMap depth_head_single_view_gpu(
    VulkanContext& context, GpuModel& model, VulkanOperators& operators,
    GpuEncoderOutput&& encoded) {
    if (encoded.features.size() != 4 ||
        encoded.patch_width == 0 || encoded.patch_height == 0) {
        throw std::invalid_argument("invalid DA3 GPU DPT input");
    }
    const std::uint32_t patches =
        encoded.patch_width * encoded.patch_height;
    const std::uint32_t image_width = encoded.patch_width * 14;
    const std::uint32_t image_height = encoded.patch_height * 14;
    const VulkanBuffer& zero_bias = model.zero_bias();
    const std::uint32_t project_channels[4] = {48, 96, 192, 384};
    GpuFeatureMap layers[4];
    for (std::uint32_t index = 0; index < 4; ++index) {
        context.batch([&] {
            operators.layer_norm(
                encoded.features[index], encoded.features[index],
                weight(model, "model.head.norm.weight"),
                weight(model, "model.head.norm.bias"),
                patches, 768, 1.0e-5f);
            GpuFeatureMap spatial{
                context.create_device_buffer(
                    std::uint64_t(patches) * 768 * sizeof(float)),
                encoded.patch_width, encoded.patch_height, 768};
            operators.tokens_to_nchw(
                spatial.buffer, encoded.features[index], patches, 768);
            const std::string project =
                "model.head.projects." + std::to_string(index);
            GpuFeatureMap projected = conv(
                context, model, operators, std::move(spatial),
                project + ".weight", project + ".bias",
                project_channels[index], 1, 1, 0, true, zero_bias);
            operators.add_uv(
                projected.buffer, projected.width, projected.height,
                projected.channels, image_width, image_height);
            if (index < 2) {
                const std::uint32_t kernel = index == 0 ? 4 : 2;
                GpuFeatureMap resized{
                    context.create_device_buffer(
                        std::uint64_t(projected.width * kernel) *
                        (projected.height * kernel) * projected.channels *
                        sizeof(float)),
                    projected.width * kernel,
                    projected.height * kernel,
                    projected.channels};
                const std::string resize =
                    "model.head.resize_layers." + std::to_string(index);
                operators.conv_transpose_nonoverlap(
                    resized.buffer, projected.buffer,
                    weight(model, resize + ".weight"),
                    weight(model, resize + ".bias"),
                    projected.width, projected.height,
                    projected.channels, projected.channels, kernel);
                layers[index] = std::move(resized);
            } else if (index == 2) {
                layers[index] = std::move(projected);
            } else {
                layers[index] = conv(
                    context, model, operators, std::move(projected),
                    "model.head.resize_layers.3.weight",
                    "model.head.resize_layers.3.bias",
                    project_channels[3], 3, 2, 1, true, zero_bias);
            }
        });
    }

    GpuFeatureMap refined[4];
    context.batch([&] {
        for (std::uint32_t index = 0; index < 4; ++index) {
            refined[index] = conv(
                context, model, operators, std::move(layers[index]),
                "model.head.scratch.layer" + std::to_string(index + 1) +
                    "_rn.weight",
                "", kFeatures, 3, 1, 1, false, zero_bias);
        }
    });
    GpuFeatureMap path;
    context.batch([&] {
        path = fusion(
            context, model, operators, std::move(refined[3]), {},
            "model.head.scratch.refinenet4",
            refined[2].width, refined[2].height, zero_bias);
    });
    context.batch([&] {
        path = fusion(
            context, model, operators, std::move(path),
            std::move(refined[2]), "model.head.scratch.refinenet3",
            refined[1].width, refined[1].height, zero_bias);
    });
    context.batch([&] {
        path = fusion(
            context, model, operators, std::move(path),
            std::move(refined[1]), "model.head.scratch.refinenet2",
            refined[0].width, refined[0].height, zero_bias);
    });
    context.batch([&] {
        path = fusion(
            context, model, operators, std::move(path),
            std::move(refined[0]), "model.head.scratch.refinenet1",
            refined[0].width * 2, refined[0].height * 2, zero_bias);
    });
    context.batch([&] {
        path = conv(
            context, model, operators, std::move(path),
            "model.head.scratch.output_conv1.weight",
            "model.head.scratch.output_conv1.bias",
            32, 3, 1, 1, true, zero_bias);
        GpuFeatureMap full{
            context.create_device_buffer(
                std::uint64_t(image_width) * image_height * 32 *
                sizeof(float)),
            image_width, image_height, 32};
        operators.bilinear_align_true(
            full.buffer, path.buffer, path.width, path.height,
            full.width, full.height, full.channels);
        operators.add_uv(
            full.buffer, full.width, full.height, full.channels,
            image_width, image_height);
        full = conv(
            context, model, operators, std::move(full),
            "model.head.scratch.output_conv2.0.weight",
            "model.head.scratch.output_conv2.0.bias",
            32, 3, 1, 1, true, zero_bias);
        operators.relu(
            full.buffer, full.buffer,
            static_cast<std::uint32_t>(elements(full)));
        path = conv(
            context, model, operators, std::move(full),
            "model.head.scratch.output_conv2.2.weight",
            "model.head.scratch.output_conv2.2.bias",
            2, 1, 1, 0, true, zero_bias);
        operators.exponential(
            path.buffer, image_width * image_height);
    });
    return path;
}

}  // namespace da3_native
