#include "external_gpu.h"

#include "dpt_gpu.h"
#include "encoder_gpu.h"
#include "gpu_model.h"
#include "gpu_preprocess.h"
#include "image.h"
#include "operators.h"
#include "safetensors.h"
#include "vulkan.h"

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
constexpr std::uint32_t kGpuSlotCount = 3u;

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

struct SharedOutput {
    ComPtr<ID3D12Resource> resource;
    ComPtr<ID3D12Fence> fence;
    HANDLE resource_handle = nullptr;
    HANDLE fence_handle = nullptr;

    SharedOutput() = default;
    SharedOutput(const SharedOutput&) = delete;
    SharedOutput& operator=(const SharedOutput&) = delete;
    SharedOutput(SharedOutput&& other) noexcept
        : resource(std::move(other.resource)),
          fence(std::move(other.fence)),
          resource_handle(std::exchange(other.resource_handle, nullptr)),
          fence_handle(std::exchange(other.fence_handle, nullptr)) {}
    SharedOutput& operator=(SharedOutput&& other) noexcept {
        if (this != &other) {
            if (resource_handle != nullptr) CloseHandle(resource_handle);
            if (fence_handle != nullptr) CloseHandle(fence_handle);
            resource = std::move(other.resource);
            fence = std::move(other.fence);
            resource_handle = std::exchange(other.resource_handle, nullptr);
            fence_handle = std::exchange(other.fence_handle, nullptr);
        }
        return *this;
    }
    ~SharedOutput() {
        if (resource_handle != nullptr) CloseHandle(resource_handle);
        if (fence_handle != nullptr) CloseHandle(fence_handle);
    }
};

struct GpuSlot {
    std::atomic<bool> occupied{false};
    SharedOutput shared;
    std::uint32_t width = 0u;
    std::uint32_t height = 0u;
    std::uint64_t fence_value = 0u;
};

SharedOutput create_shared_output(
    ID3D12Device* device,
    std::uint32_t width,
    std::uint32_t height) {
    const D3D12_HEAP_PROPERTIES heap{
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN,
        1, 1};
    const D3D12_RESOURCE_DESC description{
        D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        0, width, height, 1, 1,
        DXGI_FORMAT_R32_FLOAT,
        {1, 0},
        D3D12_TEXTURE_LAYOUT_UNKNOWN,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
    SharedOutput result;
    check_hresult(
        device->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_SHARED, &description,
            D3D12_RESOURCE_STATE_COMMON, nullptr,
            IID_PPV_ARGS(&result.resource)),
        "CreateCommittedResource(DA3 output texture)");
    check_hresult(
        device->CreateFence(
            0u, D3D12_FENCE_FLAG_SHARED,
            IID_PPV_ARGS(&result.fence)),
        "CreateFence(DA3 output)");
    check_hresult(
        device->CreateSharedHandle(
            result.resource.Get(), nullptr, GENERIC_ALL,
            nullptr, &result.resource_handle),
        "CreateSharedHandle(DA3 output texture)");
    check_hresult(
        device->CreateSharedHandle(
            result.fence.Get(), nullptr, GENERIC_ALL,
            nullptr, &result.fence_handle),
        "CreateSharedHandle(DA3 output fence)");
    return result;
}

void validate_input_texture(
    ID3D12Device* device,
    std::uintptr_t handle,
    std::uint32_t width,
    std::uint32_t height) {
    ComPtr<ID3D12Resource> resource;
    check_hresult(
        device->OpenSharedHandle(
            reinterpret_cast<HANDLE>(handle),
            IID_PPV_ARGS(&resource)),
        "OpenSharedHandle(DA3 input texture)");
    const D3D12_RESOURCE_DESC description = resource->GetDesc();
    if (description.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
        description.Width != width || description.Height != height ||
        description.DepthOrArraySize != 1u || description.MipLevels != 1u ||
        description.SampleDesc.Count != 1u ||
        description.Format != DXGI_FORMAT_B8G8R8A8_UNORM) {
        throw std::invalid_argument(
            "shared DA3 input is not the declared BGRA8 texture");
    }
}

std::shared_ptr<GpuSlot> acquire_slot(
    const std::array<std::shared_ptr<GpuSlot>, kGpuSlotCount>& slots,
    std::atomic<std::uint32_t>& next) {
    const std::uint32_t first =
        next.fetch_add(1u, std::memory_order_relaxed) % kGpuSlotCount;
    for (std::uint32_t offset = 0u; offset < kGpuSlotCount; ++offset) {
        const auto& slot = slots[(first + offset) % kGpuSlotCount];
        bool expected = false;
        if (slot->occupied.compare_exchange_strong(
                expected, true, std::memory_order_acq_rel)) return slot;
    }
    throw GpuSlotsExhausted();
}

VulkanImage prepare_output(
    GpuSlot& slot,
    ID3D12Device* device,
    VulkanContext& context,
    std::uint32_t width,
    std::uint32_t height) {
    if (slot.shared.resource == nullptr ||
        slot.width != width || slot.height != height) {
        slot.shared = SharedOutput{};
        slot.width = 0u;
        slot.height = 0u;
        slot.fence_value = 0u;
        slot.shared = create_shared_output(device, width, height);
        slot.width = width;
        slot.height = height;
    }
    return context.import_d3d12_image(
        slot.shared.resource_handle, width, height,
        VK_FORMAT_R32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
}
#endif

class ExternalGpuImpl;

#if defined(_WIN32)
class ExternalJobImpl final : public ExternalJob {
public:
    ExternalJobImpl(
        std::shared_ptr<ExternalGpu> owner,
        std::shared_ptr<GpuSlot> slot,
        VulkanImage input,
        VulkanImage output,
        VulkanSubmission submission,
        const ExternalTextureRequest& request,
        std::uint64_t fence_value)
        : owner_(std::move(owner)), slot_(std::move(slot)),
          input_(std::move(input)), output_(std::move(output)),
          submission_(std::move(submission)), request_(request),
          fence_value_(fence_value) {}

    ~ExternalJobImpl() override {
        try { submission_.wait(); } catch (...) {}
        submission_ = VulkanSubmission{};
        output_ = VulkanImage{};
        input_ = VulkanImage{};
        slot_->occupied.store(false, std::memory_order_release);
    }

    ExternalJobState state() const override {
        if (cancelled_.load(std::memory_order_acquire))
            return ExternalJobState::cancelled;
        return submission_.ready()
            ? ExternalJobState::complete
            : ExternalJobState::running;
    }
    void cancel() override {
        cancelled_.store(true, std::memory_order_release);
    }
    ExternalTextureOutput output() const override {
        if (cancelled_.load(std::memory_order_acquire))
            throw std::runtime_error("DA3 GPU job was cancelled");
        return {
            reinterpret_cast<std::uintptr_t>(slot_->shared.resource_handle),
            request_.width, request_.height,
            reinterpret_cast<std::uintptr_t>(slot_->shared.fence_handle),
            fence_value_, request_.source_frame_id, request_.timestamp_ns};
    }

private:
    std::shared_ptr<ExternalGpu> owner_;
    std::shared_ptr<GpuSlot> slot_;
    VulkanImage input_;
    VulkanImage output_;
    VulkanSubmission submission_;
    ExternalTextureRequest request_{};
    std::uint64_t fence_value_ = 0u;
    std::atomic<bool> cancelled_{false};
};
#endif

class ExternalGpuImpl final : public ExternalGpu {
public:
    ExternalGpuImpl(const std::string& model_path, std::uint32_t device_index)
        : model_(model_path), context_(device_index),
          gpu_model_(model_, context_), operators_(context_),
          preprocessor_(context_)
#if defined(_WIN32)
          , d3d12_device_(matching_d3d12_device(context_.adapter_luid())),
          slots_{std::make_shared<GpuSlot>(), std::make_shared<GpuSlot>(),
                 std::make_shared<GpuSlot>()}
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
                available ? kGpuSlotCount : 0u};
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
        if (request.shared_texture_handle == 0u ||
            request.wait_fence_handle == 0u || request.width == 0u ||
            request.height == 0u || request.process_resolution == 0u)
            throw std::invalid_argument("invalid DA3 GPU texture request");
        validate_input_texture(
            d3d12_device_.Get(), request.shared_texture_handle,
            request.width, request.height);
        const ImageShape shape = inferbridge_image_shape(
            request.width, request.height, request.process_resolution);
        const std::uint32_t longest = std::max(request.width, request.height);
        const double scale =
            static_cast<double>(request.process_resolution) / longest;
        const std::uint32_t intermediate_width = std::max(
            1, static_cast<int>(std::nearbyint(request.width * scale)));
        const std::uint32_t intermediate_height = std::max(
            1, static_cast<int>(std::nearbyint(request.height * scale)));
        auto slot = acquire_slot(slots_, next_slot_);
        try {
            std::lock_guard<std::mutex> lock(record_mutex_);
            VulkanImage output = prepare_output(
                *slot, d3d12_device_.Get(), context_,
                request.width, request.height);
            const std::uint64_t signal_value = ++slot->fence_value;
            VulkanImage input = context_.import_d3d12_image(
                reinterpret_cast<void*>(request.shared_texture_handle),
                request.width, request.height,
                VK_FORMAT_B8G8R8A8_UNORM,
                VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
            VulkanSemaphore wait = context_.import_d3d12_fence(
                reinterpret_cast<void*>(request.wait_fence_handle),
                request.wait_fence_value);
            VulkanSemaphore signal = context_.import_d3d12_fence(
                slot->shared.fence_handle, signal_value);
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
                    operators_.bilinear_align_true_image(
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
                shared_from_this(), slot, std::move(input),
                std::move(output), std::move(submission),
                request, signal_value);
        } catch (...) {
            slot->occupied.store(false, std::memory_order_release);
            throw;
        }
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
#if defined(_WIN32)
    ComPtr<ID3D12Device> d3d12_device_;
    std::array<std::shared_ptr<GpuSlot>, kGpuSlotCount> slots_;
    std::atomic<std::uint32_t> next_slot_{0u};
    std::mutex record_mutex_;
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
            available ? kGpuSlotCount : 0u};
#else
    (void)device_index;
    return {};
#endif
}

}  // namespace da3_native
