
#include "external_gpu.h"

#include "dpt_gpu.h"
#include "encoder_gpu.h"
#include "gpu_model.h"
#include "gpu_preprocess.h"
#include "gpu_output.h"
#include "image.h"
#include "operators.h"
#include "safetensors.h"
#include "vulkan.h"
#include "inferbridge/native_harness_resource_lifetime.h"
#include "inferbridge/native_harness_resource_cache.h"

#include <array>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <utility>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#endif

namespace da3_native {
namespace {

#if defined(_WIN32)
using Microsoft::WRL::ComPtr;
constexpr std::uint32_t kMaxInFlightJobs = 3u;

void check_hresult(HRESULT result, const char* operation) {
    if (FAILED(result)) {
        throw std::runtime_error(
            std::string(operation) + " failed with HRESULT " +
            std::to_string(static_cast<long>(result)));
    }
}
ComPtr<ID3D12Device> matching_d3d12_device(std::uint64_t luid) {
    if (luid == 0u) return {};
    ComPtr<IDXGIFactory6> factory;
    check_hresult(
        CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)),
        "CreateDXGIFactory2");
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        const HRESULT enumerated = factory->EnumAdapters1(index, &adapter);
        if (enumerated == DXGI_ERROR_NOT_FOUND) break;
        check_hresult(enumerated, "EnumAdapters1");
        DXGI_ADAPTER_DESC1 description{};
        check_hresult(adapter->GetDesc1(&description), "GetDesc1");
        std::uint64_t candidate = 0u;
        static_assert(sizeof(candidate) == sizeof(description.AdapterLuid));
        std::memcpy(&candidate, &description.AdapterLuid, sizeof(candidate));
        if (candidate != luid) continue;
        ComPtr<ID3D12Device> device;
        check_hresult(
            D3D12CreateDevice(
                adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                IID_PPV_ARGS(&device)),
            "D3D12CreateDevice");
        return device;
    }
    return {};
}

void validate_texture(ID3D12Device* device, std::uintptr_t handle,
    std::uint32_t width, std::uint32_t height, DXGI_FORMAT format,
    const char* operation) {
    ComPtr<ID3D12Resource> resource;
    check_hresult(device->OpenSharedHandle(
        reinterpret_cast<HANDLE>(handle), IID_PPV_ARGS(&resource)), operation);
    const auto d = resource->GetDesc();
    if (d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
        d.Width != width || d.Height != height || d.DepthOrArraySize != 1u ||
        d.MipLevels != 1u || d.SampleDesc.Count != 1u || d.Format != format)
        throw std::invalid_argument("DA3 shared texture descriptor mismatch");
}
class ExternalJobImpl final : public ExternalJob {
public:
    ExternalJobImpl(std::shared_ptr<ExternalGpu> owner,
        VulkanSubmission submission,
        inferbridge::native_harness::ResourceLifetimeDomainPtr lifetime)
        : owner_(std::move(owner)), submission_(std::move(submission)),
          lifetime_(std::move(lifetime)) {}
    ~ExternalJobImpl() override {
        inferbridge::native_harness::wait_then_retire(
            lifetime_, submission_, [] {});
    }
    ExternalJobState state() const override {
        if(cancelled_.load())return ExternalJobState::cancelled;
        return submission_.ready()?ExternalJobState::complete:ExternalJobState::running;
    }
    void cancel() override { cancelled_.store(true); }
private:
    std::shared_ptr<ExternalGpu> owner_;
    VulkanSubmission submission_;
    inferbridge::native_harness::ResourceLifetimeDomainPtr lifetime_;
    std::atomic<bool> cancelled_{false};
};
#endif

class ExternalGpuImpl final : public ExternalGpu {
public:
    ExternalGpuImpl(const std::string& model_path, std::uint32_t device_index)
        : model_(model_path), context_(device_index),
          gpu_model_(model_, context_), operators_(context_),
          preprocessor_(context_), output_(context_)
#if defined(_WIN32)
          , d3d12_device_(matching_d3d12_device(context_.adapter_luid()))
#endif
          {}

    void infer(
        const float* input, std::uint32_t width, std::uint32_t height,
        float* output) override {
        if (input == nullptr || output == nullptr || width == 0u ||
            height == 0u || width % 14u != 0u || height % 14u != 0u)
            throw std::invalid_argument("invalid DA3 Vulkan tensor shape");
        VulkanBuffer image = context_.create_device_buffer(
            static_cast<std::uint64_t>(width) * height * 3u * sizeof(float));
        context_.upload(
            image, input,
            static_cast<std::size_t>(width) * height * 3u * sizeof(float));
        GpuFeatureMap depth = depth_head_single_view_gpu(
            context_, gpu_model_, operators_,
            encoder_single_view_gpu(
                context_, gpu_model_, operators_, image, width, height));
        context_.download(
            depth.buffer, output,
            static_cast<std::size_t>(width) * height * sizeof(float));
    }

    ExternalGpuCapabilities capabilities() const override {
#if defined(_WIN32)
        const VulkanExternalCapabilities& external =
            context_.external_capabilities();
        const bool available = d3d12_device_ != nullptr &&
            external.timeline_semaphore &&
            external.d3d12_resource_import &&
            external.d3d12_fence_import &&
            external.d3d12_bgra8_sampled_image_import &&
            external.d3d12_r32_storage_image_import;
        return {available, available ? context_.adapter_luid() : 0u,
                available ? kMaxInFlightJobs : 0u};
#else
        return {};
#endif
    }

    std::shared_ptr<ExternalJob> submit_texture(
        const ExternalTextureRequest& request) override {
#if !defined(_WIN32)
        (void)request;
        throw std::runtime_error(
            "DA3 D3D12 texture submission is only available on Windows");
#else
        if (!capabilities().available)
            throw std::runtime_error(
                "complete DA3 D3D12/Vulkan texture interop is unavailable");
        if (!request.shared_texture_handle || !request.wait_fence_handle ||
            !request.output_texture_handle || !request.signal_fence_handle ||
            !request.width || !request.height || !request.process_resolution ||
            request.output_width != request.width ||
            request.output_height != request.height)
            throw std::invalid_argument("invalid DA3 GPU texture request");
        const ImageShape shape = inferbridge_image_shape(
            request.width, request.height, request.process_resolution);
        const std::uint32_t longest = std::max(request.width, request.height);
        const double scale =
            static_cast<double>(request.process_resolution) / longest;
        const std::uint32_t intermediate_width = std::max(
            1, static_cast<int>(std::nearbyint(request.width * scale)));
        const std::uint32_t intermediate_height = std::max(
            1, static_cast<int>(std::nearbyint(request.height * scale)));
        try {
            auto lifetime_guard = lifetime_->acquire();
            const auto input_usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
            const auto output_usage = VK_IMAGE_USAGE_STORAGE_BIT |
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            VulkanImage& input = input_cache_.get_or_create({
                inferbridge::native_harness::stable_resource_identity(
                    request.shared_texture_handle,
                    request.shared_texture_identity),
                request.width, request.height, VK_FORMAT_B8G8R8A8_UNORM,
                input_usage}, [&] {
                    validate_texture(d3d12_device_.Get(),
                        request.shared_texture_handle, request.width,
                        request.height, DXGI_FORMAT_B8G8R8A8_UNORM,
                        "OpenSharedHandle(DA3 input)");
                    return context_.import_d3d12_image(
                        reinterpret_cast<void*>(request.shared_texture_handle),
                        request.width, request.height,
                        VK_FORMAT_B8G8R8A8_UNORM, input_usage);
                });
            VulkanImage& output = output_cache_.get_or_create({
                inferbridge::native_harness::stable_resource_identity(
                    request.output_texture_handle,
                    request.output_texture_identity),
                request.output_width, request.output_height,
                VK_FORMAT_R32_SFLOAT, output_usage}, [&] {
                    validate_texture(d3d12_device_.Get(),
                        request.output_texture_handle, request.output_width,
                        request.output_height, DXGI_FORMAT_R32_FLOAT,
                        "OpenSharedHandle(DA3 output)");
                    return context_.import_d3d12_image(
                        reinterpret_cast<void*>(request.output_texture_handle),
                        request.output_width, request.output_height,
                        VK_FORMAT_R32_SFLOAT, output_usage);
                });
            VulkanSemaphore wait = context_.import_d3d12_fence(
                reinterpret_cast<void*>(request.wait_fence_handle),
                request.wait_fence_value);
            VulkanSemaphore signal = context_.import_d3d12_fence(
                reinterpret_cast<void*>(request.signal_fence_handle),
                request.signal_fence_value);
            VulkanSubmission submission = context_.segmented_batch_async(
                std::move(wait), std::move(signal), [&] {
                    VulkanBuffer image = context_.create_device_buffer(
                        static_cast<std::uint64_t>(shape.width) *
                        shape.height * 3u * sizeof(float));
                    context_.acquire_external_image(
                        input, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_ACCESS_SHADER_READ_BIT);
                    context_.acquire_external_image(
                        output, VK_IMAGE_LAYOUT_GENERAL,
                        VK_ACCESS_SHADER_WRITE_BIT);
                    preprocessor_.run_texture(
                        image, input,
                        intermediate_width, intermediate_height,
                        shape.width, shape.height);
                    GpuFeatureMap depth = depth_head_single_view_gpu(
                        context_, gpu_model_, operators_,
                        encoder_single_view_gpu(
                            context_, gpu_model_, operators_, image,
                            shape.width, shape.height));
                    output_.resize_and_normalize(
                        output, depth.buffer, depth.width, depth.height,
                        request.width, request.height);
                    context_.release_external_image(
                        input, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_ACCESS_SHADER_READ_BIT);
                    context_.release_external_image(
                        output, VK_IMAGE_LAYOUT_GENERAL,
                        VK_ACCESS_SHADER_WRITE_BIT);
                });
            return std::make_shared<ExternalJobImpl>(
                shared_from_this(), std::move(submission), lifetime_);
        } catch (...) { throw; }
#endif
    }

    void transfer_counters(
        std::uint64_t& upload_bytes,
        std::uint64_t& download_bytes) const override {
        context_.transfer_counters(upload_bytes, download_bytes);
    }

private:
    SafeTensors model_;
    VulkanContext context_;
    GpuModel gpu_model_;
    VulkanOperators operators_;
    GpuPreprocessor preprocessor_;
    GpuOutput output_;
#if defined(_WIN32)
    ComPtr<ID3D12Device> d3d12_device_;
    inferbridge::native_harness::StableResourceCache<VulkanImage>
        input_cache_;
    inferbridge::native_harness::StableResourceCache<VulkanImage>
        output_cache_;
    inferbridge::native_harness::ResourceLifetimeDomainPtr lifetime_ =
        inferbridge::native_harness::make_resource_lifetime_domain();
#endif
};

}  // namespace

std::shared_ptr<ExternalGpu> create_external_gpu(
    const std::string& model_path,
    std::uint32_t device_index) {
    return std::make_shared<ExternalGpuImpl>(model_path, device_index);
}

ExternalGpuCapabilities probe_external_gpu(std::uint32_t device_index) {
#if defined(_WIN32)
    VulkanContext context(device_index);
    const auto device = matching_d3d12_device(context.adapter_luid());
    const VulkanExternalCapabilities& external = context.external_capabilities();
    const bool available = device != nullptr &&
        external.d3d12_resource_import && external.d3d12_fence_import &&
        external.d3d12_bgra8_sampled_image_import &&
        external.d3d12_r32_storage_image_import;
    return {available, available ? context.adapter_luid() : 0u,
            available ? kMaxInFlightJobs : 0u};
#else
    (void)device_index;
    return {};
#endif
}

}  // namespace da3_native
