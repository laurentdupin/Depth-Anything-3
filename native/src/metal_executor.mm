#include "metal_executor.h"
#include "safetensors.h"

#include <inferbridge/native_harness_precision.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace da3_native {
namespace {

MPSShape* shape(std::initializer_list<NSInteger> values) {
    NSMutableArray<NSNumber*>* result =
        [NSMutableArray arrayWithCapacity:values.size()];
    for (NSInteger value : values) [result addObject:@(value)];
    return result;
}

MPSShape* shape(const TensorView& tensor) {
    NSMutableArray<NSNumber*>* result =
        [NSMutableArray arrayWithCapacity:tensor.rank];
    for (std::uint32_t i = 0; i < tensor.rank; ++i)
        [result addObject:@(tensor.dimensions[i])];
    return result;
}

float cubic(float distance) {
    constexpr float a = -0.75f;
    distance = std::abs(distance);
    if (distance <= 1.0f)
        return ((a + 2.0f) * distance - (a + 3.0f)) *
            distance * distance + 1.0f;
    if (distance < 2.0f)
        return ((a * distance - 5.0f * a) * distance + 8.0f * a) *
            distance - 4.0f * a;
    return 0.0f;
}

std::vector<float> position_embedding(
    const TensorView& source, int patch_height, int patch_width) {
    constexpr int embedding = 384;
    std::vector<float> result(
        static_cast<std::size_t>(patch_height * patch_width + 1) * embedding);
    std::copy_n(source.data, embedding, result.data());
    for (int y = 0; y < patch_height; ++y) {
        const float sy = (y + 0.5f) * 37.0f / (patch_height + 0.1f) - 0.5f;
        const int by = static_cast<int>(std::floor(sy));
        for (int x = 0; x < patch_width; ++x) {
            const float sx = (x + 0.5f) * 37.0f / (patch_width + 0.1f) - 0.5f;
            const int bx = static_cast<int>(std::floor(sx));
            float* destination = result.data() +
                static_cast<std::size_t>(1 + y * patch_width + x) * embedding;
            for (int channel = 0; channel < embedding; ++channel) {
                float value = 0.0f;
                for (int oy = -1; oy <= 2; ++oy) {
                    const int py = std::clamp(by + oy, 0, 36);
                    const float wy = cubic(sy - (by + oy));
                    for (int ox = -1; ox <= 2; ++ox) {
                        const int px = std::clamp(bx + ox, 0, 36);
                        value += wy * cubic(sx - (bx + ox)) * source.data[
                            static_cast<std::size_t>(1 + py * 37 + px) *
                                embedding + channel];
                    }
                }
                destination[channel] = value;
            }
        }
    }
    return result;
}

std::vector<float> uv_embedding(
    int channels, int height, int width, int image_height, int image_width) {
    std::vector<float> result(
        static_cast<std::size_t>(channels) * height * width, 0.0f);
    const float aspect = static_cast<float>(image_width) / image_height;
    const float diagonal = std::sqrt(aspect * aspect + 1.0f);
    const float span_x = aspect / diagonal;
    const float span_y = 1.0f / diagonal;
    const float left = -span_x * (width - 1.0f) / width;
    const float right = span_x * (width - 1.0f) / width;
    const float top = -span_y * (height - 1.0f) / height;
    const float bottom = span_y * (height - 1.0f) / height;
    const int direction_dim = channels / 2;
    const int frequencies = direction_dim / 2;
    for (int y = 0; y < height; ++y) {
        const float v = height == 1 ? top : top + (bottom - top) * y / (height - 1);
        for (int x = 0; x < width; ++x) {
            const float u = width == 1 ? left : left + (right - left) * x / (width - 1);
            const float coordinates[2] = {u, v};
            for (int direction = 0; direction < 2; ++direction) {
                for (int frequency = 0; frequency < frequencies; ++frequency) {
                    const float omega = 1.0f / std::pow(
                        100.0f, static_cast<float>(frequency) / frequencies);
                    const float angle = coordinates[direction] * omega;
                    const int base = direction * direction_dim;
                    result[(static_cast<std::size_t>(base + frequency) * height + y) * width + x] =
                        0.1f * std::sin(angle);
                    result[(static_cast<std::size_t>(base + frequencies + frequency) * height + y) * width + x] =
                        0.1f * std::cos(angle);
                }
            }
        }
    }
    return result;
}

enum class RopeMode { none, local, global };

class GraphBuilder {
public:
    GraphBuilder(const SafeTensors& model, int width, int height, bool fp16)
        : model_(model), width_(width), height_(height),
          patch_width_(width / 14), patch_height_(height / 14),
          tokens_(patch_width_ * patch_height_ + 1), fp16_(fp16),
          graph_([MPSGraph new]) {}

    MPSGraph* graph() const { return graph_; }
    MPSGraphTensor* input() const { return input_; }
    MPSGraphTensor* output() const { return output_; }

    void build() {
        input_ = [graph_ placeholderWithShape:shape({1, 3, height_, width_})
            dataType:MPSDataTypeFloat32 name:@"normalized_rgb_chw"];
        MPSGraphTensor* state = fp16_ ? [graph_ castTensor:input_
            toType:MPSDataTypeFloat16 name:nil] : input_;
        state = conv(state, prefix() + "patch_embed.proj", 14, 0);
        state = [graph_ reshapeTensor:state
            withShape:shape({1, 384, patch_height_ * patch_width_}) name:nil];
        state = [graph_ transposeTensor:state dimension:1 withDimension:2 name:nil];
        state = [graph_ concatTensors:@[constant(prefix() + "cls_token"), state]
            dimension:1 name:nil];
        const auto positions = position_embedding(
            model_.tensor(prefix() + "pos_embed"), patch_height_, patch_width_);
        state = add(state, owned(positions, shape({1, tokens_, 384})));

        MPSGraphTensor* local = state;
        std::array<MPSGraphTensor*, 4> captures{};
        const int capture_blocks[4] = {5, 7, 9, 11};
        int capture = 0;
        for (int block = 0; block < 12; ++block) {
            if (block == 4) {
                state = [graph_ concatTensors:@[
                    slice(constant(prefix() + "camera_token"), 1, 0, 1),
                    slice(state, 1, 1, tokens_ - 1)] dimension:1 name:nil];
            }
            const std::string p = prefix() + "blocks." + std::to_string(block) + ".";
            MPSGraphTensor* normalized = layer_norm(state, p + "norm1", 1.0e-6f);
            const RopeMode rope = block < 4 ? RopeMode::none :
                (block % 2 == 0 ? RopeMode::local : RopeMode::global);
            MPSGraphTensor* attended = attention(normalized, p + "attn.", block >= 4, rope);
            state = add(state, multiply(attended, constant(p + "ls1.gamma")));
            normalized = layer_norm(state, p + "norm2", 1.0e-6f);
            MPSGraphTensor* hidden = gelu(linear(normalized, p + "mlp.fc1"));
            hidden = linear(hidden, p + "mlp.fc2");
            state = add(state, multiply(hidden, constant(p + "ls2.gamma")));
            if (block < 4 || block % 2 == 0) local = state;
            if (capture < 4 && block == capture_blocks[capture]) {
                MPSGraphTensor* global = layer_norm(
                    state, prefix() + "norm", 1.0e-5f);
                captures[capture++] = [graph_ concatTensors:@[
                    slice(local, 1, 1, tokens_ - 1),
                    slice(global, 1, 1, tokens_ - 1)] dimension:2 name:nil];
            }
        }
        output_ = dpt(captures);
        if (fp16_) output_ = [graph_ castTensor:output_
            toType:MPSDataTypeFloat32 name:@"depth_float32"];
    }

private:
    std::string prefix() const { return "model.backbone.pretrained."; }

    MPSGraphTensor* constant(const std::string& name) {
        const TensorView& tensor = model_.tensor(name);
        NSData* data = [NSData dataWithBytesNoCopy:const_cast<float*>(tensor.data)
            length:tensor.elements * sizeof(float) freeWhenDone:NO];
        MPSGraphTensor* value = [graph_ constantWithData:data
            shape:shape(tensor) dataType:MPSDataTypeFloat32];
        return fp16_ ? [graph_ castTensor:value toType:MPSDataTypeFloat16 name:nil] : value;
    }

    MPSGraphTensor* owned(const std::vector<float>& values, MPSShape* dimensions) {
        NSData* data = [NSData dataWithBytes:values.data()
            length:values.size() * sizeof(float)];
        MPSGraphTensor* value = [graph_ constantWithData:data shape:dimensions
            dataType:MPSDataTypeFloat32];
        return fp16_ ? [graph_ castTensor:value toType:MPSDataTypeFloat16 name:nil] : value;
    }

    MPSGraphTensor* scalar(float value) {
        MPSGraphTensor* result = [graph_ constantWithScalar:value
            dataType:MPSDataTypeFloat32];
        return fp16_ ? [graph_ castTensor:result toType:MPSDataTypeFloat16 name:nil] : result;
    }

    MPSGraphTensor* add(MPSGraphTensor* a, MPSGraphTensor* b) {
        return [graph_ additionWithPrimaryTensor:a secondaryTensor:b name:nil];
    }
    MPSGraphTensor* subtract(MPSGraphTensor* a, MPSGraphTensor* b) {
        return [graph_ subtractionWithPrimaryTensor:a secondaryTensor:b name:nil];
    }
    MPSGraphTensor* multiply(MPSGraphTensor* a, MPSGraphTensor* b) {
        return [graph_ multiplicationWithPrimaryTensor:a secondaryTensor:b name:nil];
    }
    MPSGraphTensor* slice(MPSGraphTensor* value, NSUInteger dimension,
                          NSInteger start, NSInteger length) {
        return [graph_ sliceTensor:value dimension:dimension start:start
            length:length name:nil];
    }

    MPSGraphTensor* linear(MPSGraphTensor* value, const std::string& p) {
        MPSGraphTensor* weight = [graph_ transposeTensor:constant(p + ".weight")
            dimension:0 withDimension:1 name:nil];
        MPSGraphTensor* result = [graph_ matrixMultiplicationWithPrimaryTensor:value
            secondaryTensor:weight name:nil];
        if (model_.contains(p + ".bias")) result = add(result, constant(p + ".bias"));
        return result;
    }

    MPSGraphTensor* layer_norm(MPSGraphTensor* value, const std::string& p,
                               float epsilon) {
        NSArray<NSNumber*>* axes = @[@(-1)];
        MPSGraphTensor* mean = [graph_ meanOfTensor:value axes:axes name:nil];
        MPSGraphTensor* variance = [graph_ varianceOfTensor:value
            meanTensor:mean axes:axes name:nil];
        return [graph_ normalizationWithTensor:value meanTensor:mean
            varianceTensor:variance gammaTensor:constant(p + ".weight")
            betaTensor:constant(p + ".bias") epsilon:epsilon name:nil];
    }

    MPSGraphTensor* gelu(MPSGraphTensor* value) {
        MPSGraphTensor* error = [graph_ erfWithTensor:
            multiply(value, scalar(0.7071067811865475f)) name:nil];
        return multiply(multiply(value, scalar(0.5f)), add(error, scalar(1.0f)));
    }

    MPSGraphTensor* rotate_half(
        MPSGraphTensor* value, int offset, bool y_axis, RopeMode mode) {
        MPSGraphTensor* first = slice(value, 4, offset, 16);
        MPSGraphTensor* second = slice(value, 4, offset + 16, 16);
        std::vector<float> cosines(static_cast<std::size_t>(tokens_) * 16);
        std::vector<float> sines(cosines.size());
        for (int token = 0; token < tokens_; ++token) {
            int position = 0;
            if (token != 0) {
                if (mode == RopeMode::global) position = 1;
                else {
                    const int patch = token - 1;
                    position = y_axis ? patch / patch_width_ + 1 : patch % patch_width_ + 1;
                }
            }
            for (int index = 0; index < 16; ++index) {
                const float angle = position / std::pow(100.0f, index * 2.0f / 32.0f);
                cosines[static_cast<std::size_t>(token) * 16 + index] = std::cos(angle);
                sines[static_cast<std::size_t>(token) * 16 + index] = std::sin(angle);
            }
        }
        MPSGraphTensor* cosine = owned(cosines, shape({1, tokens_, 1, 1, 16}));
        MPSGraphTensor* sine = owned(sines, shape({1, tokens_, 1, 1, 16}));
        return [graph_ concatTensors:@[
            subtract(multiply(first, cosine), multiply(second, sine)),
            add(multiply(second, cosine), multiply(first, sine))]
            dimension:4 name:nil];
    }

    MPSGraphTensor* rope(MPSGraphTensor* value, RopeMode mode) {
        return [graph_ concatTensors:@[
            rotate_half(value, 0, true, mode),
            rotate_half(value, 32, false, mode)] dimension:4 name:nil];
    }

    MPSGraphTensor* attention(MPSGraphTensor* value, const std::string& p,
                              bool normalize_qk, RopeMode mode) {
        MPSGraphTensor* qkv = [graph_ reshapeTensor:linear(value, p + "qkv")
            withShape:shape({1, tokens_, 3, 6, 64}) name:nil];
        MPSGraphTensor* query = slice(qkv, 2, 0, 1);
        MPSGraphTensor* key = slice(qkv, 2, 1, 1);
        MPSGraphTensor* values = slice(qkv, 2, 2, 1);
        if (normalize_qk) {
            query = layer_norm(query, p + "q_norm", 1.0e-5f);
            key = layer_norm(key, p + "k_norm", 1.0e-5f);
        }
        if (mode != RopeMode::none) {
            query = rope(query, mode);
            key = rope(key, mode);
        }
        query = [graph_ reshapeTensor:query withShape:shape({1, tokens_, 6, 64}) name:nil];
        key = [graph_ reshapeTensor:key withShape:shape({1, tokens_, 6, 64}) name:nil];
        values = [graph_ reshapeTensor:values withShape:shape({1, tokens_, 6, 64}) name:nil];
        query = [graph_ transposeTensor:query permutation:@[@0, @2, @1, @3] name:nil];
        key = [graph_ transposeTensor:key permutation:@[@0, @2, @3, @1] name:nil];
        values = [graph_ transposeTensor:values permutation:@[@0, @2, @1, @3] name:nil];
        MPSGraphTensor* scores = [graph_ matrixMultiplicationWithPrimaryTensor:
            multiply(query, scalar(0.125f)) secondaryTensor:key name:nil];
        scores = [graph_ softMaxWithTensor:scores axis:-1 name:nil];
        MPSGraphTensor* result = [graph_ matrixMultiplicationWithPrimaryTensor:scores
            secondaryTensor:values name:nil];
        result = [graph_ transposeTensor:result permutation:@[@0, @2, @1, @3] name:nil];
        result = [graph_ reshapeTensor:result withShape:shape({1, tokens_, 384}) name:nil];
        return linear(result, p + "proj");
    }

    MPSGraphConvolution2DOpDescriptor* conv_descriptor(int stride, int padding) {
        return [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:stride
            strideInY:stride dilationRateInX:1 dilationRateInY:1 groups:1
            paddingLeft:padding paddingRight:padding paddingTop:padding
            paddingBottom:padding paddingStyle:MPSGraphPaddingStyleExplicit
            dataLayout:MPSGraphTensorNamedDataLayoutNCHW
            weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
    }

    MPSGraphTensor* conv(MPSGraphTensor* value, const std::string& p,
                         int stride, int padding, bool bias = true) {
        MPSGraphTensor* result = [graph_ convolution2DWithSourceTensor:value
            weightsTensor:constant(p + ".weight") descriptor:conv_descriptor(stride, padding)
            name:nil];
        if (bias && model_.contains(p + ".bias")) {
            const auto& b = model_.tensor(p + ".bias");
            result = add(result, [graph_ reshapeTensor:constant(p + ".bias")
                withShape:shape({1, static_cast<NSInteger>(b.elements), 1, 1}) name:nil]);
        }
        return result;
    }

    MPSGraphTensor* transpose_conv(MPSGraphTensor* value, const std::string& p,
                                    int channels, int source_h, int source_w, int stride) {
        MPSGraphTensor* result = [graph_ convolutionTranspose2DWithSourceTensor:value
            weightsTensor:constant(p + ".weight")
            outputShape:shape({1, channels, source_h * stride, source_w * stride})
            descriptor:conv_descriptor(stride, 0) name:nil];
        return add(result, [graph_ reshapeTensor:constant(p + ".bias")
            withShape:shape({1, channels, 1, 1}) name:nil]);
    }

    MPSGraphTensor* resize(MPSGraphTensor* value, int height, int width) {
        return [graph_ resizeTensor:value size:shape({height, width})
            mode:MPSGraphResizeBilinear centerResult:NO alignCorners:YES
            layout:MPSGraphTensorNamedDataLayoutNCHW name:nil];
    }

    MPSGraphTensor* residual(MPSGraphTensor* value, const std::string& p) {
        MPSGraphTensor* result = [graph_ reLUWithTensor:value name:nil];
        result = conv(result, p + ".conv1", 1, 1);
        result = [graph_ reLUWithTensor:result name:nil];
        return add(value, conv(result, p + ".conv2", 1, 1));
    }

    MPSGraphTensor* fusion(MPSGraphTensor* path, MPSGraphTensor* skip,
                           const std::string& p, int height, int width) {
        if (skip != nil) path = add(path, residual(skip, p + ".resConfUnit1"));
        path = residual(path, p + ".resConfUnit2");
        return conv(resize(path, height, width), p + ".out_conv", 1, 0);
    }

    MPSGraphTensor* add_uv(MPSGraphTensor* value, int channels, int height,
                           int width) {
        return add(value, owned(uv_embedding(channels, height, width,
            height_, width_), shape({1, channels, height, width})));
    }

    MPSGraphTensor* dpt(const std::array<MPSGraphTensor*, 4>& features) {
        std::array<MPSGraphTensor*, 4> layers{};
        const int heights[4] = {patch_height_ * 4, patch_height_ * 2,
            patch_height_, (patch_height_ + 1) / 2};
        const int widths[4] = {patch_width_ * 4, patch_width_ * 2,
            patch_width_, (patch_width_ + 1) / 2};
        for (int i = 0; i < 4; ++i) {
            MPSGraphTensor* value = layer_norm(features[i], "model.head.norm", 1.0e-5f);
            value = [graph_ transposeTensor:value dimension:1 withDimension:2 name:nil];
            value = [graph_ reshapeTensor:value
                withShape:shape({1, 768, patch_height_, patch_width_}) name:nil];
            const std::string project =
                "model.head.projects." + std::to_string(i);
            const int project_channels = static_cast<int>(
                model_.tensor(project + ".weight").dimensions[0]);
            value = conv(value, project, 1, 0);
            value = add_uv(value, project_channels, patch_height_, patch_width_);
            if (i < 2) {
                const std::string resize_layer =
                    "model.head.resize_layers." + std::to_string(i);
                const int output_channels = static_cast<int>(
                    model_.tensor(resize_layer + ".weight").dimensions[1]);
                value = transpose_conv(value, resize_layer, output_channels,
                    patch_height_, patch_width_, i == 0 ? 4 : 2);
            }
            else if (i == 3) value = conv(value, "model.head.resize_layers.3", 2, 1);
            layers[i] = conv(value, "model.head.scratch.layer" +
                std::to_string(i + 1) + "_rn", 1, 1, false);
        }
        MPSGraphTensor* path = fusion(layers[3], nil,
            "model.head.scratch.refinenet4", heights[2], widths[2]);
        path = fusion(path, layers[2], "model.head.scratch.refinenet3", heights[1], widths[1]);
        path = fusion(path, layers[1], "model.head.scratch.refinenet2", heights[0], widths[0]);
        path = fusion(path, layers[0], "model.head.scratch.refinenet1", heights[0] * 2, widths[0] * 2);
        path = conv(path, "model.head.scratch.output_conv1", 1, 1);
        path = resize(path, height_, width_);
        path = add_uv(path, static_cast<int>(model_.tensor(
            "model.head.scratch.output_conv1.bias").elements), height_, width_);
        path = conv(path, "model.head.scratch.output_conv2.0", 1, 1);
        path = [graph_ reLUWithTensor:path name:nil];
        path = conv(path, "model.head.scratch.output_conv2.2", 1, 0);
        return [graph_ exponentWithTensor:path name:nil];
    }

    const SafeTensors& model_;
    int width_, height_, patch_width_, patch_height_, tokens_;
    bool fp16_;
    MPSGraph* graph_ = nil;
    MPSGraphTensor* input_ = nil;
    MPSGraphTensor* output_ = nil;
};

struct Plan {
    MPSGraph* graph = nil;
    MPSGraphTensor* input = nil;
    MPSGraphExecutable* executable = nil;
};

}  // namespace

class MetalExecutor::Impl {
public:
    explicit Impl(const std::string& path) : model_(path) {
        device_ = MTLCreateSystemDefaultDevice();
        queue_ = [device_ newCommandQueue];
        graph_device_ = [MPSGraphDevice deviceWithMTLDevice:device_];
        if (device_ == nil || queue_ == nil || graph_device_ == nil)
            throw std::runtime_error("Metal is unavailable for Depth Anything V3");
        const auto precision = inferbridge::native::requested_precision();
        if (precision == inferbridge::native::Precision::int8)
            throw std::invalid_argument("Depth Anything V3 Metal does not support INT8");
        fp16_ = precision == inferbridge::native::Precision::fp16 ||
            precision == inferbridge::native::Precision::automatic;
    }

    void infer(const float* input, std::uint32_t width, std::uint32_t height,
               float* depth, std::uint64_t depth_elements) {
        if (!input || !depth || !width || !height || width % 14 || height % 14 ||
            depth_elements < static_cast<std::uint64_t>(width) * height)
            throw std::invalid_argument("invalid Metal DA3 tensor shape");
        std::lock_guard<std::mutex> lock(mutex_);
        @autoreleasepool {
            Plan& plan = get_plan(width, height);
            id<MTLBuffer> buffer = [device_ newBufferWithBytes:input
                length:static_cast<NSUInteger>(width) * height * 3u * sizeof(float)
                options:MTLResourceStorageModeShared];
            MPSGraphTensorData* data = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:buffer shape:shape({1, 3,
                    static_cast<NSInteger>(height), static_cast<NSInteger>(width)})
                dataType:MPSDataTypeFloat32];
            MPSGraphExecutableExecutionDescriptor* descriptor =
                [MPSGraphExecutableExecutionDescriptor new];
            descriptor.waitUntilCompleted = YES;
            NSArray<MPSGraphTensorData*>* results = [plan.executable
                runWithMTLCommandQueue:queue_ inputsArray:@[data]
                resultsArray:nil executionDescriptor:descriptor];
            if (results.count != 1u) throw std::runtime_error("DA3 Metal graph returned no output");
            [results[0].mpsndarray readBytes:depth strideBytes:nil];
        }
    }

private:
    Plan& get_plan(int width, int height) {
        const std::uint64_t key = (static_cast<std::uint64_t>(width) << 32u) |
            static_cast<std::uint32_t>(height);
        auto found = plans_.find(key);
        if (found != plans_.end()) return found->second;
        GraphBuilder builder(model_, width, height, fp16_);
        builder.build();
        MPSGraphShapedType* type = [[MPSGraphShapedType alloc]
            initWithShape:shape({1, 3, height, width}) dataType:MPSDataTypeFloat32];
        MPSGraphCompilationDescriptor* descriptor = [MPSGraphCompilationDescriptor new];
        descriptor.optimizationLevel = MPSGraphOptimizationLevel1;
        descriptor.waitForCompilationCompletion = YES;
        MPSGraphExecutable* executable = [builder.graph() compileWithDevice:graph_device_
            feeds:@{builder.input(): type} targetTensors:@[builder.output()]
            targetOperations:nil compilationDescriptor:descriptor];
        if (executable == nil) throw std::runtime_error("failed to compile DA3 Metal graph");
        executable.options = MPSGraphOptionsSynchronizeResults;
        return plans_.emplace(key, Plan{builder.graph(), builder.input(), executable}).first->second;
    }

    SafeTensors model_;
    bool fp16_ = false;
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    MPSGraphDevice* graph_device_ = nil;
    std::unordered_map<std::uint64_t, Plan> plans_;
    std::mutex mutex_;
};

MetalExecutor::MetalExecutor(const std::string& path)
    : impl_(std::make_unique<Impl>(path)) {}
MetalExecutor::~MetalExecutor() = default;
void MetalExecutor::infer(const float* input, std::uint32_t width,
                          std::uint32_t height, float* depth,
                          std::uint64_t depth_elements) {
    impl_->infer(input, width, height, depth, depth_elements);
}

}  // namespace da3_native
