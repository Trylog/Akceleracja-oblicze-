#pragma once
#include <vector>


template<typename T>
void multiplySingleColumn(std::vector<std::vector<T> > &result,
                          const std::vector<std::vector<T> > &a, const std::vector<std::vector<T> > &b,
                          int aIndex, int bIndex, int numOfElements, bool wasTransposed = false) {
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
std::vector<std::vector<T>> transposeMatrix(const std::vector<std::vector<T>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    std::vector<std::vector<T>> transposed(cols, std::vector<T>(rows));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }

    return transposed;
}