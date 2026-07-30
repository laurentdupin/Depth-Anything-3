#include "safetensors.h"

#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: da3_model_probe model.safetensors\n";
        return 2;
    }
    try {
        da3_native::SafeTensors model(argv[1]);
        const da3_native::TensorView& patch = model.tensor(
            "model.backbone.pretrained.patch_embed.proj.weight");
        const bool expected =
            model.tensor_count() == 437 &&
            model.contains(
                "model.head.scratch.output_conv2_aux.3.2.weight") &&
            patch.rank == 4 &&
            patch.dimensions[0] == 384 &&
            patch.dimensions[1] == 3 &&
            patch.dimensions[2] == 14 &&
            patch.dimensions[3] == 14;
        std::cout << "tensors=" << model.tensor_count()
                  << "\npatch_shape="
                  << patch.dimensions[0] << "x"
                  << patch.dimensions[1] << "x"
                  << patch.dimensions[2] << "x"
                  << patch.dimensions[3] << "\n";
        return expected ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
