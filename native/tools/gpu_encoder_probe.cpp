#include "encoder_gpu.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

std::vector<float> read_values(const std::string& path, std::uint64_t count) {
    std::vector<float> values(static_cast<std::size_t>(count));
    std::ifstream stream(path, std::ios::binary);
    stream.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!stream || stream.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("invalid raw tensor file: " + path);
    }
    return values;
}

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "usage: da3_gpu_encoder_probe model device size input "
                     "reference-prefix tolerance\n";
        return 2;
    }
    try {
        const std::uint32_t device = std::stoul(argv[2]);
        const std::uint32_t size = std::stoul(argv[3]);
        const float tolerance = std::stof(argv[6]);
        auto input = read_values(argv[4], std::uint64_t(3) * size * size);
        da3_native::SafeTensors model(argv[1]);
        da3_native::VulkanContext context(device);
        da3_native::GpuModel gpu_model(model, context);
        da3_native::VulkanOperators operators(context);
        auto image = context.create_device_buffer(input.size() * sizeof(float));
        context.upload(image, input.data(), input.size() * sizeof(float));
        auto output = da3_native::encoder_single_view_gpu(
            context, gpu_model, operators, image, size, size);
        const std::uint64_t count =
            std::uint64_t(output.patch_width) * output.patch_height * 768;
        double worst = 0.0;
        for (std::uint32_t level = 0; level < 4; ++level) {
            std::vector<float> actual(static_cast<std::size_t>(count));
            context.download(
                output.features[level], actual.data(),
                actual.size() * sizeof(float));
            auto reference = read_values(
                std::string(argv[5]) + ".feature" +
                std::to_string(level) + ".bin", count);
            double difference = 0.0, magnitude = 0.0;
            float maximum = 0.0f;
            for (std::size_t index = 0; index < actual.size(); ++index) {
                const float error =
                    std::abs(actual[index] - reference[index]);
                difference += error;
                magnitude += std::abs(reference[index]);
                maximum = std::max(maximum, error);
            }
            const double relative = difference / magnitude;
            worst = std::max(worst, relative);
            std::cout << "feature" << level << "_relative_l1="
                      << relative << "\nfeature" << level
                      << "_maximum_absolute=" << maximum << "\n";
        }
        return worst <= tolerance ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
