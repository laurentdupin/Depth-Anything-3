#include "da3_native.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {
std::vector<float> read_floats(
    const std::string& path,
    std::uint64_t count) {
    std::vector<float> values(static_cast<std::size_t>(count));
    std::ifstream input(path, std::ios::binary);
    input.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(count * sizeof(float)));
    if (!input || input.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("invalid raw tensor file: " + path);
    }
    return values;
}
}

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "usage: da3_full_graph_probe model.safetensors size "
                     "input.bin depth.bin tolerance\n";
        return 2;
    }
    try {
        const std::uint32_t size =
            static_cast<std::uint32_t>(std::stoul(argv[2]));
        const float tolerance = std::stof(argv[5]);
        const std::vector<float> input = read_floats(
            argv[3], std::uint64_t(3) * size * size);
        const std::vector<float> reference = read_floats(
            argv[4], std::uint64_t(size) * size);
        da3_context* context = nullptr;
        const da3_status create_status =
            da3_create(argv[1], &context);
        if (create_status != DA3_STATUS_OK) {
            throw std::runtime_error(
                std::string("da3_create: ") + da3_last_error());
        }
        std::vector<float> output(
            static_cast<std::size_t>(size) * size);
        const auto start = std::chrono::steady_clock::now();
        const da3_status inference_status = da3_infer_tensor_f32(
            context, input.data(),
            static_cast<std::int32_t>(size),
            static_cast<std::int32_t>(size),
            output.data(), output.size());
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        if (inference_status != DA3_STATUS_OK) {
            const std::string error = da3_last_error();
            da3_destroy(context);
            throw std::runtime_error(
                std::string("da3_infer_tensor_f32: ") + error);
        }
        da3_destroy(context);
        double absolute_sum = 0.0;
        double reference_sum = 0.0;
        float maximum = 0.0f;
        for (std::size_t index = 0; index < output.size(); ++index) {
            const float difference =
                std::abs(output[index] - reference[index]);
            maximum = std::max(maximum, difference);
            absolute_sum += difference;
            reference_sum += std::abs(reference[index]);
        }
        const double relative =
            absolute_sum / std::max(reference_sum, 1.0e-30);
        std::cout << "seconds=" << seconds
                  << "\nmaximum_absolute=" << maximum
                  << "\nrelative_l1=" << relative << "\n";
        return relative <= tolerance ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
