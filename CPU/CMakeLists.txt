cmake_minimum_required(VERSION 3.10)

project(MatrixOperationsProject)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add AVX, AVX2, and FMA flags for both Debug and Release builds
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mavx2 -mfma")

# Include header files (optional if .h files are in the same directory)
include_directories(${CMAKE_SOURCE_DIR})

# Add the executable
add_executable(
        App
        main.cpp
)
