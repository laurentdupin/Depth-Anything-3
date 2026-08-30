#pragma once

#include <cstdint>
#include <memory>
#include <string>

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

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace da3_native
