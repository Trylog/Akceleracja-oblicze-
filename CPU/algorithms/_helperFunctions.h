#pragma once
#include <vector>

#include "avxAlignedVector.h"


template<typename T>
void multiplySingleColumn(AvxAlignedMatrix<T> &result,
                          const AvxAlignedMatrix<T> &a, const AvxAlignedMatrix<T> &b,
                          int aIndex, int bIndex, int numOfElements, bool wasTransposed) {
    T sum = T{}; // T{} is uniform initialization. Return 0 for numeric types

    for (int i = 0; i < numOfElements; ++i) {
        if (wasTransposed) {
            sum += a[aIndex][i] * b[bIndex][i];
        }
        else{
            sum += a[aIndex][i] * b[i][bIndex];
        }
    }

    result[aIndex][bIndex] = sum;
}


template<typename T>
AvxAlignedMatrix<T> transposeMatrix(const AvxAlignedMatrix<T>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    auto transposed = createAvxAlignedMatrix<T>(cols, rows);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }

    return transposed;
}