cmake_minimum_required(VERSION 3.20)
project(depth_anything_3_native VERSION 0.1.0 LANGUAGES C CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../../native_support/InferBridgeNativeHarnessSupport.cmake")
link_libraries(InferBridge::NativeHarnessSupport)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    set(
        CMAKE_MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

option(DA3_WITH_VULKAN
    "Build the dependency-free Vulkan executor" ON)

if(DA3_WITH_VULKAN)
    find_package(Vulkan REQUIRED COMPONENTS glslc)
    set(DA3_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
    file(MAKE_DIRECTORY "${DA3_GENERATED_DIR}")

    function(da3_compile_shader NAME)
        set(SOURCE
            "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/${NAME}.glsl")
        set(SPIRV "${DA3_GENERATED_DIR}/${NAME}.spv")
        set(HEADER "${DA3_GENERATED_DIR}/${NAME}_spv.h")
        add_custom_command(
            OUTPUT "${SPIRV}"
            COMMAND Vulkan::glslc -fshader-stage=compute
                --target-env=vulkan1.3 --target-spv=spv1.6 -Werror
                "-I${CMAKE_CURRENT_SOURCE_DIR}/src/shaders"
                -o "${SPIRV}" "${SOURCE}"
            DEPENDS
                "${SOURCE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/indexing.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/conv2d_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/linear_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/project_tokens_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/conv_transpose_nonoverlap_common.glsl"
            VERBATIM)
        add_custom_command(
            OUTPUT "${HEADER}"
            COMMAND "${CMAKE_COMMAND}"
                "-DINPUT=${SPIRV}"
                "-DOUTPUT=${HEADER}"
                "-DVARIABLE=da3_${NAME}_spv"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
            DEPENDS
                "${SPIRV}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
            VERBATIM)
        set(DA3_SHADER_HEADERS
            ${DA3_SHADER_HEADERS} "${HEADER}" PARENT_SCOPE)
    endfunction()

    foreach(SHADER
            add_scaled add add_position bilinear_align_true
            bilinear_align_true_image bmm bmm_score_half
            bmm_value_half conv2d conv2d8 conv2d_half conv2d8_half
            conv_transpose_nonoverlap conv_transpose_nonoverlap_half
            elementwise gelu layer_norm linear linear16 linear_vec8 linear_half
            linear16_half prepare_tokens position_bicubic project_tokens
            project_tokens_half relu softmax_lastdim
            softmax_lastdim_half replace_token qk_rope capture_concat
            add_uv exponential tokens_to_nchw preprocess_texture
            preprocess_finish resize_depth reduce_minmax
            normalize_depth_image)
        da3_compile_shader(${SHADER})
    endforeach()
endif()

add_library(depth_anything_3_native SHARED
    src/da3.cpp
    src/image.cpp
    src/inferbridge_harness.cpp
    src/safetensors.cpp
    src/encoder_cpu.cpp
    src/dpt_cpu.cpp)
if(DA3_WITH_VULKAN)
    target_sources(depth_anything_3_native PRIVATE
        src/vulkan.cpp
        src/operators.cpp
        src/gpu_model.cpp
        src/gpu_preprocess.cpp
        src/gpu_output.cpp
        src/external_gpu.cpp
        src/encoder_gpu.cpp
        src/dpt_gpu.cpp
        ${DA3_SHADER_HEADERS})
    target_include_directories(
        depth_anything_3_native PRIVATE "${DA3_GENERATED_DIR}")
    target_link_libraries(
        depth_anything_3_native PRIVATE Vulkan::Vulkan)
    if(WIN32)
        target_link_libraries(
            depth_anything_3_native PRIVATE d3d12 dxgi)
    endif()
    target_compile_definitions(
        depth_anything_3_native PRIVATE DA3_WITH_VULKAN=1)
endif()
target_include_directories(
    depth_anything_3_native PUBLIC include PRIVATE src)
target_compile_definitions(
    depth_anything_3_native PRIVATE
        DA3_BUILD_DLL
        INFERBRIDGE_HARNESS_BUILDING_LIBRARY)
set_target_properties(depth_anything_3_native PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)
if(MSVC)
    target_compile_options(
        depth_anything_3_native PRIVATE /W4 /permissive-)
endif()

add_executable(da3_model_probe
    tools/model_probe.cpp
    src/safetensors.cpp)
target_include_directories(da3_model_probe PRIVATE include src)
if(MSVC)
    target_compile_options(da3_model_probe PRIVATE /W4 /permissive-)
endif()

add_executable(da3_encoder_probe
    tools/encoder_probe.cpp
    src/safetensors.cpp
    src/encoder_cpu.cpp)
target_include_directories(da3_encoder_probe PRIVATE include src)
if(MSVC)
    target_compile_options(da3_encoder_probe PRIVATE /W4 /permissive-)
endif()

add_executable(da3_full_graph_probe
    tools/full_graph_probe.cpp)
target_link_libraries(
    da3_full_graph_probe PRIVATE depth_anything_3_native)
target_include_directories(da3_full_graph_probe PRIVATE include)
if(MSVC)
    target_compile_options(
        da3_full_graph_probe PRIVATE /W4 /permissive-)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(da3_c_abi_smoke tests/c_abi_smoke.c)
    target_link_libraries(
        da3_c_abi_smoke PRIVATE depth_anything_3_native)
    target_include_directories(da3_c_abi_smoke PRIVATE include)
    add_test(NAME da3_c_abi_smoke COMMAND da3_c_abi_smoke)
    add_executable(da3_harness_abi_smoke tests/harness_abi_smoke.c)
    target_link_libraries(
        da3_harness_abi_smoke PRIVATE depth_anything_3_native)
    target_include_directories(da3_harness_abi_smoke PRIVATE include)
    add_test(NAME da3_harness_abi_smoke COMMAND da3_harness_abi_smoke)
    add_executable(da3_harness_full_graph tests/harness_full_graph.cpp)
    target_link_libraries(
        da3_harness_full_graph PRIVATE depth_anything_3_native)
    target_include_directories(da3_harness_full_graph PRIVATE include)
    add_test(NAME da3_harness_full_graph COMMAND da3_harness_full_graph)
    set_tests_properties(
        da3_harness_full_graph PROPERTIES SKIP_RETURN_CODE 77)
    if(DA3_WITH_VULKAN)
        if(WIN32)
            add_executable(
                da3_d3d12_full_graph tests/d3d12_full_graph.cpp)
            target_link_libraries(
                da3_d3d12_full_graph PRIVATE
                    depth_anything_3_native d3d12 dxgi)
            target_include_directories(
                da3_d3d12_full_graph PRIVATE include)
            add_test(
                NAME da3_d3d12_full_graph
                COMMAND da3_d3d12_full_graph)
            set_tests_properties(
                da3_d3d12_full_graph PROPERTIES SKIP_RETURN_CODE 77)
        endif()
        add_executable(
            da3_gpu_full_graph_probe tools/gpu_full_graph_probe.cpp)
        target_link_libraries(
            da3_gpu_full_graph_probe PRIVATE depth_anything_3_native)
        target_include_directories(
            da3_gpu_full_graph_probe PRIVATE include)
        add_executable(da3_gpu_encoder_probe
            tools/gpu_encoder_probe.cpp
            src/encoder_gpu.cpp src/vulkan.cpp src/operators.cpp
            src/gpu_model.cpp src/safetensors.cpp
            ${DA3_SHADER_HEADERS})
        target_include_directories(da3_gpu_encoder_probe PRIVATE
            include src "${DA3_GENERATED_DIR}")
        target_link_libraries(
            da3_gpu_encoder_probe PRIVATE Vulkan::Vulkan)
    endif()
endif()
