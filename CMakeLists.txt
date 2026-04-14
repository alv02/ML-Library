cmake_minimum_required(VERSION 3.18)
project(MLLibrary LANGUAGES CXX CUDA)

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
    src/models.cpp
    src/optimizers.cpp
    src/backend/tensor_cpu.cpp
    src/backend/tensor_cuda.cu
)

target_include_directories(mllib PUBLIC include)

# Link CUDA runtime
find_package(CUDAToolkit REQUIRED)
target_link_libraries(mllib PUBLIC CUDA::cudart CUDA::curand)

# ── Main executable ───────────────────────────────────────────────────────────
add_executable(main main.cpp)
target_link_libraries(main PRIVATE mllib)

# ── MNIST main ────────────────────────────────────────────────────────────────
add_executable(main_mnist main_mnist.cpp)
target_link_libraries(main_mnist PRIVATE mllib)

# ── Tests ─────────────────────────────────────────────────────────────────────
enable_testing()

add_executable(test_broadcast test/broadcast.cpp)
target_link_libraries(test_broadcast PRIVATE mllib)
add_test(NAME broadcast COMMAND test_broadcast WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test)

add_executable(test_cuda test/test_cuda.cpp)
target_link_libraries(test_cuda PRIVATE mllib)
add_test(NAME cuda COMMAND test_cuda WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test)
