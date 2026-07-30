#include "da3_native.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

std::vector<float> read_floats(const char* path, std::uint64_t count) {
    std::vector<float> values(static_cast<std::size_t>(count));
    std::ifstream input(path, std::ios::binary);
    input.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(count * sizeof(float)));
    if (!input || input.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("invalid raw tensor file");
    }
    return values;
}

int main(int argc, char** argv) {
    if (argc != 7 && argc != 8) {
        std::cerr << "usage: da3_gpu_full_graph_probe model device size "
                     "input.bin depth.bin tolerance [iterations]\n";
        return 2;
    }
    try {
        const std::uint32_t device = std::stoul(argv[2]);
        const std::uint32_t size = std::stoul(argv[3]);
        const float tolerance = std::stof(argv[6]);
        const std::uint32_t iterations =
            argc == 8 ? std::stoul(argv[7]) : 1;
        const auto input =
            read_floats(argv[4], std::uint64_t(3) * size * size);
        const auto reference =
            read_floats(argv[5], std::uint64_t(size) * size);
        da3_context* context = nullptr;
        if (da3_create_vulkan(argv[1], device, &context) != DA3_STATUS_OK) {
            throw std::runtime_error(da3_last_error());
        }
        std::vector<float> output(std::size_t(size) * size);
        std::vector<double> samples;
        samples.reserve(iterations);
        for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
            const auto start = std::chrono::steady_clock::now();
            const da3_status status = da3_infer_tensor_f32(
                context, input.data(), size, size,
                output.data(), output.size());
            samples.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count());
            if (status != DA3_STATUS_OK) {
                std::string error = da3_last_error();
                da3_destroy(context);
                throw std::runtime_error(error);
            }
        }
        da3_destroy(context);
        std::sort(samples.begin(), samples.end());
        double error_sum = 0.0;
        double reference_sum = 0.0;
        float maximum = 0.0f;
        for (std::size_t index = 0; index < output.size(); ++index) {
            const float difference =
                std::abs(output[index] - reference[index]);
            error_sum += difference;
            reference_sum += std::abs(reference[index]);
            maximum = std::max(maximum, difference);
        }
        const double relative =
            error_sum / std::max(reference_sum, 1.0e-30);
        std::cout << "median_ms=" << samples[samples.size() / 2]
                  << "\nmaximum_absolute=" << maximum
                  << "\nrelative_l1=" << relative << "\n";
        return relative <= tolerance ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
