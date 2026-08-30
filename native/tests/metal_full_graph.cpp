#include "da3_native.h"

#include <inferbridge/native_harness_precision.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

bool compare(
    const char* path, inferbridge::native::Precision precision,
    const std::vector<float>& input, const std::vector<float>& reference) {
    da3_context* metal = nullptr;
    {
        const inferbridge::native::ScopedPrecisionRequest scope(precision);
        if (da3_create_metal(path, &metal) != DA3_STATUS_OK) {
            std::fprintf(stderr, "Metal load failed: %s\n", da3_last_error());
            return false;
        }
    }
    std::vector<float> output(reference.size());
    const da3_status status = da3_infer_tensor_f32(
        metal, input.data(), 56, 56, output.data(), output.size());
    da3_destroy(metal);
    if (status != DA3_STATUS_OK) {
        std::fprintf(stderr, "Metal inference failed: %s\n", da3_last_error());
        return false;
    }
    double error = 0.0;
    double magnitude = 0.0;
    float maximum = 0.0f;
    for (std::size_t i = 0; i < output.size(); ++i) {
        if (!std::isfinite(output[i])) return false;
        const float difference = std::abs(output[i] - reference[i]);
        error += difference;
        magnitude += std::abs(reference[i]);
        maximum = std::max(maximum, difference);
    }
    const double relative = error / std::max(magnitude, 1.0e-30);
    std::printf("precision=%s relative_l1=%.9g maximum_absolute=%.9g\n",
        precision == inferbridge::native::Precision::fp16 ? "fp16" : "fp32",
        relative, maximum);
    return relative <=
        (precision == inferbridge::native::Precision::fp16 ? 0.03 : 0.002);
}

int main() {
    const char* path = std::getenv("DA3_MODEL");
    if (!path || !*path) return 77;
    constexpr std::size_t pixels = 56u * 56u;
    std::vector<float> input(pixels * 3u);
    for (std::size_t i = 0; i < input.size(); ++i)
        input[i] = (static_cast<float>((i * 1664525u + 1013904223u) & 1023u) /
            1023.0f - 0.5f) * 2.0f;
    da3_context* cpu = nullptr;
    if (da3_create(path, &cpu) != DA3_STATUS_OK) return 1;
    std::vector<float> reference(pixels);
    const da3_status status = da3_infer_tensor_f32(
        cpu, input.data(), 56, 56, reference.data(), reference.size());
    da3_destroy(cpu);
    if (status != DA3_STATUS_OK) return 1;
    return compare(path, inferbridge::native::Precision::fp32, input, reference) &&
        compare(path, inferbridge::native::Precision::fp16, input, reference) ? 0 : 1;
}
