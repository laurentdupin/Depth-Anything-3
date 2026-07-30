cmake_minimum_required(VERSION 3.20)
project(depth_anything_3_native VERSION 0.1.0 LANGUAGES C CXX)

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

add_library(depth_anything_3_native SHARED src/da3.cpp)
target_include_directories(depth_anything_3_native PUBLIC include)
target_compile_definitions(
    depth_anything_3_native PRIVATE DA3_BUILD_DLL)
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

include(CTest)
if(BUILD_TESTING)
    add_executable(da3_c_abi_smoke tests/c_abi_smoke.c)
    target_link_libraries(
        da3_c_abi_smoke PRIVATE depth_anything_3_native)
    target_include_directories(da3_c_abi_smoke PRIVATE include)
    add_test(NAME da3_c_abi_smoke COMMAND da3_c_abi_smoke)
endif()
