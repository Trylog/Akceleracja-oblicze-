# Akceleracja-obliczen-CUDA
This repository contains solutions for "Acceleration of Calculations in Data Processing" university project whose purpose was to compare efficiency of different solutions using either a CPU or GPU for multiplying matrices problem.

# CPU
Implemented versions of algorithm:
- Single Thread
- Single Thread SIMD
- Naive Multi Thread
- Thread pool with batching
- Thread pool with batching and queue
- Thread pool with batching and queue SIMD

# GPU
It was developed using CUDA and cuBLAS library
## CUDA
- basic version
- version with different division of tasks into thread blocks
## cuBLAS
- basic version
- version using unified memory
