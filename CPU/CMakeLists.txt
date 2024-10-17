cmake_minimum_required(VERSION 3.10)

project(MatrixOperationsProject)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include header files (this is optional if the .h file is in the same directory)
include_directories(${CMAKE_SOURCE_DIR})

# Add the executable
add_executable(
        App
        main.cpp
        matrixMultiplication.cpp
)
