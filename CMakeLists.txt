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
    src/metrics.cpp
    src/visualize.cpp
    src/backend/tensor_cpu.cpp
    src/backend/tensor_cuda.cu
    src/backend/cuda_mem_arena.cpp
)

target_include_directories(mllib PUBLIC include)

# Link CUDA runtime
find_package(CUDAToolkit REQUIRED)
target_link_libraries(mllib PUBLIC CUDA::cudart CUDA::curand)

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

# ── Tests ─────────────────────────────────────────────────────────────────────
enable_testing()

add_executable(test_broadcast test/broadcast.cpp)
target_link_libraries(test_broadcast PRIVATE mllib)
add_test(NAME broadcast COMMAND test_broadcast WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test)

add_executable(test_tensor test/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE mllib)
add_test(NAME tensor_cpu  COMMAND test_tensor --cpu WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test)
add_test(NAME tensor_cuda COMMAND test_tensor --gpu WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test)
