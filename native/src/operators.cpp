#include "operators.h"
#include "inferbridge/native_harness_environment.h"
#include "inferbridge/native_harness_precision.h"

#include "add_scaled_spv.h"
#include "add_spv.h"
#include "add_position_spv.h"
#include "bilinear_align_true_spv.h"
#include "bilinear_align_true_image_spv.h"
#include "bmm_spv.h"
#include "bmm_score_half_spv.h"
#include "bmm_value_half_spv.h"
#include "conv2d_spv.h"
#include "conv2d_tiled4_spv.h"
#include "conv2d8_spv.h"
#include "conv2d_half_spv.h"
#include "conv2d8_half_spv.h"
#include "conv_transpose_nonoverlap_spv.h"
#include "conv_transpose_nonoverlap_half_spv.h"
#include "gelu_spv.h"
#include "layer_norm_spv.h"
#include "linear_spv.h"
#include "linear16_spv.h"
#include "linear_vec8_spv.h"
#include "linear_half_spv.h"
#include "linear16_half_spv.h"
#include "linear_int8_tiled_spv.h"
#include "quantize_rows_int8_spv.h"
#include "prepare_tokens_spv.h"
#include "position_bicubic_spv.h"
#include "project_tokens_spv.h"
#include "project_tokens_half_spv.h"
#include "relu_spv.h"
#include "softmax_lastdim_spv.h"
#include "softmax_lastdim_half_spv.h"
#include "replace_token_spv.h"
#include "qk_rope_spv.h"
#include "capture_concat_spv.h"
#include "add_uv_spv.h"
#include "exponential_spv.h"
#include "tokens_to_nchw_spv.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace da3_native {
namespace {

std::uint32_t divide_up(std::uint32_t value, std::uint32_t divisor) {
    return (value + divisor - 1) / divisor;
}

void require_bytes(
    const VulkanBuffer& buffer,
    std::uint64_t elements,
    const char* name) {
    if (elements > std::numeric_limits<std::uint64_t>::max() / sizeof(float) ||
        buffer.size() < elements * sizeof(float)) {
        throw std::invalid_argument(
            std::string(name) + " Vulkan buffer is too small");
    }
}

void require_half_elements(
    const VulkanBuffer& buffer,
    std::uint64_t elements,
    const char* name) {
    const std::uint64_t words = (elements + 1) / 2;
    if (words > std::numeric_limits<std::uint64_t>::max() /
            sizeof(std::uint32_t) ||
        buffer.size() < words * sizeof(std::uint32_t)) {
        throw std::invalid_argument(
            std::string(name) + " packed-half Vulkan buffer is too small");
    }
}

}  // namespace

VulkanOperators::VulkanOperators(VulkanContext& context)
    : context_(context),
      linear_(context.create_pipeline(
          da3_linear_spv, da3_linear_spv_size, 4, 12)),
      linear16_(context.create_pipeline(
          da3_linear16_spv, da3_linear16_spv_size, 4, 12)),
      linear_vec8_(context.create_pipeline(
          da3_linear_vec8_spv, da3_linear_vec8_spv_size, 4, 12)),
      quantize_rows_int8_(context.supports_packed_int8_dot() &&
              inferbridge::native::requested_precision() ==
                  inferbridge::native::Precision::int8
          ? context.create_pipeline(
              da3_quantize_rows_int8_spv,
                da3_quantize_rows_int8_spv_size,
              {
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            },
                            {
                                VK_ACCESS_SHADER_READ_BIT,
                                VK_ACCESS_SHADER_WRITE_BIT,
                                VK_ACCESS_SHADER_WRITE_BIT,
                            },
                            4)
          : VulkanPipeline{}),
      linear_int8_tiled_(context.supports_packed_int8_dot() &&
              inferbridge::native::requested_precision() ==
                  inferbridge::native::Precision::int8
          ? context.create_pipeline(
              da3_linear_int8_tiled_spv,
              da3_linear_int8_tiled_spv_size, 6, 28)
          : VulkanPipeline{}),
      linear_half_(context.create_pipeline(
          da3_linear_half_spv, da3_linear_half_spv_size, 4, 12)),
      linear16_half_(context.create_pipeline(
          da3_linear16_half_spv,
          da3_linear16_half_spv_size,
          4,
          12)),
      gelu_(context.create_pipeline(
          da3_gelu_spv, da3_gelu_spv_size, 2, 4)),
      layer_norm_(context.create_pipeline(
          da3_layer_norm_spv, da3_layer_norm_spv_size, 4, 12)),
      add_scaled_(context.create_pipeline(
          da3_add_scaled_spv, da3_add_scaled_spv_size, 4, 8)),
      bmm_(context.create_pipeline(
          da3_bmm_spv, da3_bmm_spv_size, 3, 44)),
      bmm_score_half_(context.create_pipeline(
          da3_bmm_score_half_spv,
          da3_bmm_score_half_spv_size,
          2,
          8)),
      bmm_value_half_(context.create_pipeline(
          da3_bmm_value_half_spv,
          da3_bmm_value_half_spv_size,
          3,
          8)),
      softmax_lastdim_(context.create_pipeline(
          da3_softmax_lastdim_spv,
          da3_softmax_lastdim_spv_size,
          2,
          8)),
      softmax_lastdim_half_(context.create_pipeline(
          da3_softmax_lastdim_half_spv,
          da3_softmax_lastdim_half_spv_size,
          {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
          {
              VK_ACCESS_SHADER_READ_BIT |
              VK_ACCESS_SHADER_WRITE_BIT,
          },
          8)),
      prepare_tokens_(context.create_pipeline(
          da3_prepare_tokens_spv,
          da3_prepare_tokens_spv_size,
          5,
          24)),
      position_bicubic_(context.create_pipeline(
          da3_position_bicubic_spv,
          da3_position_bicubic_spv_size,
          {
              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          0)),
      add_position_(context.create_pipeline(
          da3_add_position_spv,
          da3_add_position_spv_size,
          4,
          20)),
      add_(context.create_pipeline(
          da3_add_spv, da3_add_spv_size, 3, 4)),
      project_tokens_(context.create_pipeline(
          da3_project_tokens_spv,
          da3_project_tokens_spv_size,
          4,
          20)),
      project_tokens_half_(context.create_pipeline(
          da3_project_tokens_half_spv,
          da3_project_tokens_half_spv_size,
          4,
          20)),
      conv2d_(context.create_pipeline(
          da3_conv2d_spv, da3_conv2d_spv_size, 4, 48)),
      conv2d_tiled4_(context.create_pipeline(
          da3_conv2d_tiled4_spv,
          da3_conv2d_tiled4_spv_size,
          4,
          48)),
      conv2d8_(context.create_pipeline(
          da3_conv2d8_spv, da3_conv2d8_spv_size, 4, 48)),
      conv2d_half_(context.create_pipeline(
          da3_conv2d_half_spv, da3_conv2d_half_spv_size, 4, 48)),
      conv2d8_half_(context.create_pipeline(
          da3_conv2d8_half_spv,
          da3_conv2d8_half_spv_size,
          4,
          48)),
      conv_transpose_nonoverlap_(context.create_pipeline(
          da3_conv_transpose_nonoverlap_spv,
          da3_conv_transpose_nonoverlap_spv_size,
          4,
          24)),
      conv_transpose_nonoverlap_half_(context.create_pipeline(
          da3_conv_transpose_nonoverlap_half_spv,
          da3_conv_transpose_nonoverlap_half_spv_size,
          4,
          24)),
      bilinear_align_true_(context.create_pipeline(
          da3_bilinear_align_true_spv,
          da3_bilinear_align_true_spv_size,
          2,
          24)),
      bilinear_align_true_image_(context.create_pipeline(
          da3_bilinear_align_true_image_spv,
          da3_bilinear_align_true_image_spv_size,
          {
              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          },
          16)),
      relu_(context.create_pipeline(
          da3_relu_spv, da3_relu_spv_size, 2, 4)),
      replace_token_(context.create_pipeline(
          da3_replace_token_spv, da3_replace_token_spv_size, 2, 4)),
      qk_rope_(context.create_pipeline(
          da3_qk_rope_spv, da3_qk_rope_spv_size, 5, 16)),
      capture_concat_(context.create_pipeline(
          da3_capture_concat_spv, da3_capture_concat_spv_size, 3, 8)),
      add_uv_(context.create_pipeline(
          da3_add_uv_spv, da3_add_uv_spv_size, 1, 20)),
      exponential_(context.create_pipeline(
          da3_exponential_spv, da3_exponential_spv_size, 1, 4)),
      tokens_to_nchw_(context.create_pipeline(
          da3_tokens_to_nchw_spv, da3_tokens_to_nchw_spv_size, 2, 8)) {
    linear_.set_debug_name("linear");
    linear16_.set_debug_name("linear16");
    linear_vec8_.set_debug_name("linear_vec8");
    linear_half_.set_debug_name("linear_half");
    if (context.supports_packed_int8_dot()) {
        quantize_rows_int8_.set_debug_name("quantize_rows_int8");
        linear_int8_tiled_.set_debug_name("linear_int8_tiled");
    }
    linear16_half_.set_debug_name("linear16_half");
    gelu_.set_debug_name("gelu");
    layer_norm_.set_debug_name("layer_norm");
    add_scaled_.set_debug_name("add_scaled");
    bmm_.set_debug_name("bmm");
    bmm_score_half_.set_debug_name("bmm_score_half");
    bmm_value_half_.set_debug_name("bmm_value_half");
    softmax_lastdim_.set_debug_name("softmax_lastdim");
    softmax_lastdim_half_.set_debug_name(
        "softmax_lastdim_half");
    prepare_tokens_.set_debug_name("prepare_tokens");
    position_bicubic_.set_debug_name("position_bicubic");
    add_position_.set_debug_name("add_position");
    add_.set_debug_name("add");
    project_tokens_.set_debug_name("project_tokens");
    project_tokens_half_.set_debug_name(
        "project_tokens_half");
    conv2d_.set_debug_name("conv2d");
    conv2d_tiled4_.set_debug_name("conv2d_tiled4");
    conv2d8_.set_debug_name("conv2d8");
    conv2d_half_.set_debug_name("conv2d_half");
    conv2d8_half_.set_debug_name("conv2d8_half");
    conv_transpose_nonoverlap_.set_debug_name(
        "conv_transpose_nonoverlap");
    conv_transpose_nonoverlap_half_.set_debug_name(
        "conv_transpose_nonoverlap_half");
    bilinear_align_true_.set_debug_name(
        "bilinear_align_true");
    bilinear_align_true_image_.set_debug_name(
        "bilinear_align_true_image");
    relu_.set_debug_name("relu");
    replace_token_.set_debug_name("replace_token");
    qk_rope_.set_debug_name("qk_rope");
    capture_concat_.set_debug_name("capture_concat");
    add_uv_.set_debug_name("add_uv");
    exponential_.set_debug_name("exponential");
    tokens_to_nchw_.set_debug_name("tokens_to_nchw");
}

void VulkanOperators::linear_int8(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    const VulkanBuffer& packed_weight,
    const VulkanBuffer& weight_scales,
    const VulkanBuffer& bias,
    std::uint32_t rows,
    std::uint32_t input_columns,
    std::uint32_t output_columns,
    bool gelu) {
    if (!context_.supports_packed_int8_dot() || input_columns % 4u != 0u) {
        throw std::runtime_error("accelerated packed INT8 linear is unavailable");
    }
    VulkanBuffer& packed_input = int8_workspace_.packed(
        std::uint64_t(rows) * (input_columns / 4u) * sizeof(std::uint32_t),
        [this](std::uint64_t bytes) {
            return context_.create_device_buffer(bytes);
        });
    VulkanBuffer& input_scales = int8_workspace_.scales(
        std::uint64_t(rows) * sizeof(float),
        [this](std::uint64_t bytes) {
            return context_.create_device_buffer(bytes);
        });
    const std::uint32_t columns = input_columns;
    context_.dispatch(
        quantize_rows_int8_, {&input, &packed_input, &input_scales},
        &columns, sizeof(columns), rows);
    const std::uint32_t parameters[7] = {
        rows, input_columns, output_columns, 0u, output_columns, 0u, 1u};
    context_.dispatch(
        linear_int8_tiled_,
        {&output, &packed_input, &packed_weight, &input_scales,
         &weight_scales, &bias},
        parameters, sizeof(parameters),
        divide_up(output_columns, 64u), divide_up(rows, 56u));
    if (gelu) {
        const std::uint32_t count = rows * output_columns;
        context_.dispatch(
            gelu_, {&output, &output}, &count, sizeof(count),
            divide_up(count, 256u));
    }
}

void VulkanOperators::linear(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    const VulkanBuffer& weight,
    const VulkanBuffer& bias,
    std::uint32_t rows,
    std::uint32_t input_columns,
    std::uint32_t output_columns,
    bool gelu,
    bool block16,
    bool half_weight) {
    if (rows == 0 || input_columns == 0 || output_columns == 0) {
        throw std::invalid_argument("linear dimensions cannot be zero");
    }
    require_bytes(input, std::uint64_t(rows) * input_columns, "input");
    const std::uint64_t weight_elements =
        std::uint64_t(output_columns) * input_columns;
    if (half_weight) {
        require_half_elements(weight, weight_elements, "weight");
    } else {
        require_bytes(weight, weight_elements, "weight");
    }
    require_bytes(bias, output_columns, "bias");
    require_bytes(
        output, std::uint64_t(rows) * output_columns, "output");
    struct Parameters {
        std::uint32_t rows;
        std::uint32_t input_columns;
        std::uint32_t output_columns;
    } parameters{rows, input_columns, output_columns};
    VulkanPipeline& pipeline = half_weight
        ? (block16 ? linear16_half_ : linear_half_)
        : linear_vec8_;
    const std::uint32_t groups_x = half_weight
        ? divide_up(divide_up(output_columns, 4), 8)
        : divide_up(output_columns, 64);
    const std::uint32_t groups_y = half_weight
        ? divide_up(divide_up(rows, 4), 8)
        : divide_up(rows, 56);
    context_.dispatch(
        pipeline,
        {&output, &input, &weight, &bias},
        &parameters,
        sizeof(parameters),
        groups_x,
        groups_y);
    if (gelu) {
        struct GeluParameters {
            std::uint32_t count;
        } gelu_parameters{rows * output_columns};
        context_.dispatch(
            gelu_,
            {&output, &output},
            &gelu_parameters,
            sizeof(gelu_parameters),
            divide_up(gelu_parameters.count, 256));
    }
}

#include <inferbridge/vulkan/operators/vulkan_operators_layer_norm_variant_1.h>

#include <inferbridge/vulkan/operators/vulkan_operators_add_scaled.h>

#include <inferbridge/vulkan/operators/vulkan_operators_attention_head64_variant_1.h>

#include <inferbridge/vulkan/operators/vulkan_operators_prepare_tokens_variant_1.h>

#include <inferbridge/vulkan/operators/vulkan_operators_project_tokens_variant_1.h>

void VulkanOperators::conv2d(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    const VulkanBuffer& weight,
    const VulkanBuffer& bias,
    std::uint32_t input_width,
    std::uint32_t input_height,
    std::uint32_t input_channels,
    std::uint32_t output_channels,
    std::uint32_t kernel,
    std::uint32_t stride,
    std::uint32_t padding,
    bool has_bias,
    bool block8,
    bool half_weight,
    std::uint32_t batches) {
    if (input_width == 0 || input_height == 0 || input_channels == 0 ||
        output_channels == 0 || kernel == 0 || stride == 0 ||
        input_width + 2 * padding < kernel ||
        input_height + 2 * padding < kernel ||
        batches == 0) {
        throw std::invalid_argument("invalid convolution dimensions");
    }
    const std::uint32_t output_width =
        (input_width + 2 * padding - kernel) / stride + 1;
    const std::uint32_t output_height =
        (input_height + 2 * padding - kernel) / stride + 1;
    require_bytes(
        input,
        std::uint64_t(batches) * input_width *
            input_height * input_channels,
        "convolution input");
    const std::uint64_t weight_elements =
        std::uint64_t(output_channels) * input_channels * kernel * kernel;
    if (half_weight) {
        require_half_elements(
            weight, weight_elements, "convolution weight");
    } else {
        require_bytes(weight, weight_elements, "convolution weight");
    }
    require_bytes(bias, has_bias ? output_channels : 1, "convolution bias");
    require_bytes(
        output,
        std::uint64_t(batches) * output_width *
            output_height * output_channels,
        "convolution output");
    struct Parameters {
        std::uint32_t input_width;
        std::uint32_t input_height;
        std::uint32_t input_channels;
        std::uint32_t output_width;
        std::uint32_t output_height;
        std::uint32_t output_channels;
        std::uint32_t kernel;
        std::uint32_t stride;
        std::int32_t padding;
        std::uint32_t has_bias;
        std::uint32_t batches;
        std::uint32_t output_channel_blocks;
    };
    const std::uint32_t output_channel_blocks =
        divide_up(output_channels, block8 ? 8 : 4);
    const Parameters parameters{
        input_width, input_height, input_channels,
        output_width, output_height, output_channels,
        kernel, stride, static_cast<std::int32_t>(padding),
        has_bias ? 1u : 0u,
        batches, output_channel_blocks,
    };
    const bool tiled4 = !half_weight && !block8 &&
        !inferbridge::native_harness::environment_flag_enabled(
            "INFERBRIDGE_DISABLE_TILED_CONVOLUTION") &&
        kernel == 3 && stride == 1 && padding == 1 &&
        output_width == input_width && output_height == input_height;
    context_.dispatch(
        tiled4
            ? conv2d_tiled4_
            : half_weight
                ? (block8 ? conv2d8_half_ : conv2d_half_)
                : (block8 ? conv2d8_ : conv2d_),
        {&output, &input, &weight, &bias},
        &parameters,
        sizeof(parameters),
        divide_up(output_width, 8),
        divide_up(output_height, 8),
        output_channel_blocks * batches);
}

#include <inferbridge/vulkan/operators/vulkan_operators_conv_transpose_nonoverlap_variant_2.h>

#include <inferbridge/vulkan/operators/vulkan_operators_bilinear_align_true_variant_1.h>

#include <inferbridge/vulkan/operators/vulkan_operators_bilinear_align_true_image.h>

#include <inferbridge/vulkan/operators/vulkan_operators_relu.h>

#include <inferbridge/vulkan/operators/vulkan_operators_add.h>

#include <inferbridge/vulkan/operators/vulkan_operators_replace_token.h>

#include <inferbridge/vulkan/operators/vulkan_operators_qk_norm_rope.h>

#include <inferbridge/vulkan/operators/vulkan_operators_capture_concat.h>

#include <inferbridge/vulkan/operators/vulkan_operators_add_uv.h>

#include <inferbridge/vulkan/operators/vulkan_operators_exponential.h>

#include <inferbridge/vulkan/operators/vulkan_operators_tokens_to_nchw.h>

}  // namespace da3_native
