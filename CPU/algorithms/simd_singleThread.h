#pragma once

#include <immintrin.h> // AVX
#include <vector>
#include <cstdint>
#include <functional>

#include "avxAlignedVector.h"
#include "algorithms/_helperFunctions.h"
#include "algorithms/_simdMultiplicationFunctions.h"


template <typename T>
AvxAlignedMatrix<T> AVX_singleThread(const AvxAlignedMatrix<T> &a,
                                     const AvxAlignedMatrix<T> &b) {
    const AvxAlignedMatrix<T> &bT =  transposeMatrix(b);
    int rowsA = a.size();
    int columnsA = a[0].size();
    int rowsBT = bT.size();

    AvxAlignedMatrix<T> resultMatrix = createAvxAlignedMatrix<T>(rowsA, rowsBT);

    std::function<void(const AvxAlignedMatrix<T>&,
                       const AvxAlignedMatrix<T>&,
                       AvxAlignedMatrix<T>&,
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
