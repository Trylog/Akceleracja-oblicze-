cmake_minimum_required(VERSION 3.29)
project(CUDA_test CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(CUDA_test main.cu)

set_target_properties(CUDA_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
