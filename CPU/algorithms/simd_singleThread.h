#pragma once

#include <immintrin.h> // AVX
#include <vector>
#include <cstdint>
#include <functional>

#include "_helperFunctions.h"
#include "_simdMultiplicationFuntions.h"


template <typename T>
std::vector<std::vector<T>> AVX_singleThread(const std::vector<std::vector<T>>& a,
                                             const std::vector<std::vector<T>>& b) {
    const std::vector<std::vector<T>>& bT = transposeMatrix(b); // Ensure this function is defined
    int rowsA = a.size();
    int columnsA = a[0].size();
    int rowsBT = bT.size();

    std::vector<std::vector<T>> resultMatrix(rowsA, std::vector<T>(rowsBT, 0));

    std::function<void(const std::vector<std::vector<T>>&,
                       const std::vector<std::vector<T>>&,
                       std::vector<std::vector<T>>&,
                       int, int, int, int)> multiplyFunc;

    if constexpr (std::is_same_v<T, float>) {
        multiplyFunc = multiplySingleResultFloat;
    } else if constexpr (std::is_same_v<T, double>) {
        multiplyFunc = multiplySingleResultDouble;
    } else if constexpr (std::is_same_v<T, int8_t>) {
        multiplyFunc = multiplySingleResultInt8;
    } else if constexpr (std::is_same_v<T, int>) {
        multiplyFunc = multiplySingleResultInt;
    }

    const int simdSizeBytes = 32;
    const int numOfVariablesPerSimd = simdSizeBytes / sizeof(T);
    int numOfElementsThatFitSimd = columnsA / numOfVariablesPerSimd;
    numOfElementsThatFitSimd *= numOfVariablesPerSimd;

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < rowsBT; ++j) {
            multiplyFunc(a, bT, resultMatrix, i, j, columnsA, numOfElementsThatFitSimd);
        }
    }

    return resultMatrix;
}
