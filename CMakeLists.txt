cmake_minimum_required(VERSION 3.18)
project(MLLibrary LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Generates compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(include)

# ── Library sources ───────────────────────────────────────────────────────────
add_library(mllib
    src/tensor.cpp
    src/tensor_iterator.cpp
    src/autograd.cpp
    src/ops.cpp
    src/layers.cpp
    src/models.cpp
    src/optimizers.cpp
    src/augment.cpp
    src/metrics.cpp
    src/visualize.cpp
    src/backend/tensor_cpu.cpp
    src/backend/tensor_cuda.cu
    src/backend/cuda_pool.cpp
)

target_include_directories(mllib PUBLIC include)
set_target_properties(mllib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Link CUDA runtime
find_package(CUDAToolkit REQUIRED)
target_link_libraries(mllib PUBLIC CUDA::cudart CUDA::curand CUDA::cublas)

# ── Main executable ───────────────────────────────────────────────────────────
add_executable(main_lr main_lr.cpp)
target_link_libraries(main_lr PRIVATE mllib)

# ── MNIST main ────────────────────────────────────────────────────────────────
add_executable(main_nn main_nn.cpp)
target_link_libraries(main_nn PRIVATE mllib)

# ── CNN MNIST main ───────────────────────────────────────────────────────────
add_executable(main_cnn main_cnn.cpp)
target_link_libraries(main_cnn PRIVATE mllib)

# ── Fashion MNIST with MaxPool ────────────────────────────────────────────────
add_executable(main_cnn2 main_cnn2.cpp)
target_link_libraries(main_cnn2 PRIVATE mllib)

# ── CIFAR-10 VGG-B with BatchNorm ────────────────────────────────────────────
add_executable(main_cnn3 main_cnn3.cpp)
target_link_libraries(main_cnn3 PRIVATE mllib)

# ── CIFAR-10 ResNet-18 ────────────────────────────────────────────────────────
add_executable(main_resnet main_resnet.cpp)
target_link_libraries(main_resnet PRIVATE mllib)

# ── GPT char-level language model ────────────────────────────────────────────
add_executable(main_gpt main_gpt.cpp)
target_link_libraries(main_gpt PRIVATE mllib)

# ── Python bindings ──────────────────────────────────────────────────────────
include(FetchContent)
FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.13.6)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(gpt_lib src/python/gpt_trainer.cpp)
target_link_libraries(gpt_lib PRIVATE mllib)

# ── Tests ─────────────────────────────────────────────────────────────────────
enable_testing()


add_executable(test_tensor test/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE mllib)
add_test(NAME tensor_cpu  COMMAND test_tensor --cpu WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test)
add_test(NAME tensor_cuda COMMAND test_tensor --gpu WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test)

add_executable(test_layer test/test_layer.cpp)
target_link_libraries(test_layer PRIVATE mllib)
add_test(NAME layer_cpu  COMMAND test_layer --cpu WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test)
add_test(NAME layer_cuda COMMAND test_layer --gpu WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test)

add_executable(test_reduce test/test_reduce.cpp)
target_link_libraries(test_reduce PRIVATE mllib)
add_test(NAME reduce COMMAND test_reduce WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
