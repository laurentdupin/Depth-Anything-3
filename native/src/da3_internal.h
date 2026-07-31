#pragma once

#include "external_gpu.h"

#include <cstdint>
#include <memory>

struct da3_context;

namespace da3_native {

void global_transfer_counters(
    std::uint64_t& upload_bytes,
    std::uint64_t& download_bytes);

ExternalGpuCapabilities context_external_capabilities(
    const da3_context* context);
std::shared_ptr<ExternalJob> submit_external_texture(
    da3_context* context,
    const ExternalTextureRequest& request);
void context_transfer_counters(
    const da3_context* context,
    std::uint64_t& upload_bytes,
    std::uint64_t& download_bytes);

}  // namespace da3_native
