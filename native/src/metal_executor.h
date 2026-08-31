#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "external_gpu.h"

namespace da3_native {

class MetalExecutor {
public:
    explicit MetalExecutor(const std::string& model_path);
    ~MetalExecutor();
    MetalExecutor(const MetalExecutor&) = delete;
    MetalExecutor& operator=(const MetalExecutor&) = delete;

    void infer(const float* input, std::uint32_t width,
               std::uint32_t height, float* depth,
               std::uint64_t depth_elements);
    std::shared_ptr<ExternalJob> submit_texture(
        const ExternalTextureRequest& request);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace da3_native
